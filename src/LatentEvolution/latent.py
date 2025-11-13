"""
Latent space voltage model with optional YAML config, torch.compile support,
and reproducible training with run logging.
"""

from pathlib import Path
from typing import Callable, Iterator
from uuid import uuid4
from datetime import datetime
import random
import sys
import subprocess
import re

import torch
import torch.nn as nn
import yaml
import tyro
import numpy as np
from pydantic import BaseModel, Field, field_validator, ConfigDict

from LatentEvolution.load_flyvis import SimulationResults, FlyVisSim, DataSplit
from LatentEvolution.gpu_stats import GPUMonitor
from LatentEvolution.diagnostics import post_training_diagnostics
from LatentEvolution.eed_model import (
    MLP,
    MLPParams,
    Evolver,
    EvolverParams,
    EncoderParams,
    DecoderParams,
    StimulusEncoderParams,
)


# -------------------------------------------------------------------
# Pydantic Config Classes
# -------------------------------------------------------------------


class TrainingConfig(BaseModel):
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    optimizer: str = Field("Adam", description="Optimizer name from torch.optim")
    train_step: str = Field("train_step", description="Compiled train step function")
    simulation_config: str
    column_to_model: str = "CALCIUM"
    use_tf32_matmul: bool = Field(
        False, description="Enable fast tf32 multiplication on certain NVIDIA GPUs"
    )
    seed: int = 42
    data_split: DataSplit
    data_passes_per_epoch: int = 1
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @field_validator("optimizer")
    @classmethod
    def validate_optimizer(cls, v: str) -> str:
        if not hasattr(torch.optim, v):
            raise ValueError(f"Unknown optimizer '{v}' in torch.optim")
        return v


class ModelParams(BaseModel):
    latent_dims: int
    num_neurons: int
    use_batch_norm: bool = True
    encoder_params: EncoderParams
    decoder_params: DecoderParams
    evolver_params: EvolverParams
    stimulus_encoder_params: StimulusEncoderParams
    training: TrainingConfig

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    def flatten(self, sep: str = ".") -> dict[str, int | float | str | bool]:
        """
        Flatten the ModelParams into a single-level dictionary.

        Args:
            sep: Separator to use for nested keys (default: ".")

        Returns:
            A flat dictionary with nested keys joined by the separator.

        Example:
            >>> params.flatten()
            {'latent_dims': 10, 'encoder_params.num_hidden_units': 64, ...}
        """
        def _flatten_dict(
            d: dict[str, int | float | str | bool | dict],
            parent_key: str = "",
        ) -> dict[str, int | float | str | bool]:
            items: list[tuple[str, int | float | str | bool]] = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(_flatten_dict(v, new_key).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        return _flatten_dict(self.model_dump())


# -------------------------------------------------------------------
# PyTorch Models
# -------------------------------------------------------------------


class LatentModel(nn.Module):
    def __init__(self, params: ModelParams):
        super().__init__()
        self.encoder = MLP(
            MLPParams(
                num_input_dims=params.num_neurons,
                num_hidden_layers=params.encoder_params.num_hidden_layers,
                num_hidden_units=params.encoder_params.num_hidden_units,
                num_output_dims=params.latent_dims,
                use_batch_norm=params.use_batch_norm,
            )
        )
        self.decoder = MLP(
            MLPParams(
                num_input_dims=params.latent_dims,
                num_hidden_layers=params.decoder_params.num_hidden_layers,
                num_hidden_units=params.decoder_params.num_hidden_units,
                num_output_dims=params.num_neurons,
                use_batch_norm=params.use_batch_norm,
            )
        )
        self.stimulus_encoder = MLP(
            MLPParams(
                num_input_dims=params.stimulus_encoder_params.num_input_dims,
                num_hidden_units=params.stimulus_encoder_params.num_hidden_units,
                num_hidden_layers=params.stimulus_encoder_params.num_hidden_layers,
                num_output_dims=params.stimulus_encoder_params.num_output_dims,
                use_batch_norm=False,
            )
        )
        self.evolver = Evolver(
            latent_dims=params.latent_dims,
            stim_dims=params.stimulus_encoder_params.num_output_dims,
            evolver_params=params.evolver_params,
            use_batch_norm=params.use_batch_norm,
        )

    def forward(self, x_t, stim_t):
        proj_t = self.encoder(x_t)
        proj_stim_t = self.stimulus_encoder(stim_t)
        proj_t_next = self.evolver(torch.concatenate([proj_t, proj_stim_t], dim=1))
        x_t_next = self.decoder(proj_t_next[:, :-proj_stim_t.shape[1]])
        return x_t_next


# -------------------------------------------------------------------
# Data + Batching
# -------------------------------------------------------------------

def make_batches_random(
    data: torch.Tensor, stim: torch.Tensor, batch_size: int, time_units: int
) -> Iterator[tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]]:
    total_time = data.shape[0]
    if total_time <= time_units:
        raise ValueError(
            f"Not enough time points ({total_time}) for time_units={time_units}"
        )
    while True:
        start_indices = torch.randint(
            low=0, high=total_time - time_units, size=(batch_size,), device=data.device
        )
        x_t = data[start_indices]
        stim_t = stim[start_indices]

        x_t_plus = data[start_indices + time_units]
        yield (x_t, stim_t), x_t_plus


# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------


def get_git_commit_hash() -> str:
    """Get the short git commit hash, with -dirty suffix if working tree has changes."""
    try:
        # Get short commit hash
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()

        # Check if working tree is dirty
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()

        if status:
            commit_hash += "-dirty"

        return commit_hash
    except (subprocess.CalledProcessError, FileNotFoundError):
        # If git is not available or not a git repo, return a placeholder
        return "unknown"


def seed_everything(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Cross-platform device selection."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("Using Apple MPS backend for training.")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("Using CPU for training.")
        return torch.device("cpu")


def train_step_nocompile(model: LatentModel, x_t, stim_t, x_t_plus, cfg: ModelParams):
    # regularization loss
    reg_loss = 0.0
    for p in model.encoder.parameters():
        reg_loss += torch.abs(p).mean()*cfg.encoder_params.l1_reg_loss
    for p in model.decoder.parameters():
        reg_loss += torch.abs(p).mean()*cfg.decoder_params.l1_reg_loss
    for p in model.evolver.parameters():
        reg_loss += torch.abs(p).mean()*cfg.evolver_params.l1_reg_loss

    # evolution loss
    output = model(x_t, stim_t)
    evolve_loss = torch.nn.functional.mse_loss(output, x_t_plus)

    # reconstruction loss
    recon = model.decoder(model.encoder(x_t))
    recon_loss = torch.nn.functional.mse_loss(recon, x_t)

    loss = evolve_loss + recon_loss + reg_loss
    return (loss, recon_loss, evolve_loss, reg_loss)

train_step = torch.compile(train_step_nocompile, fullgraph=True, mode="reduce-overhead")

# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------


def train(cfg: ModelParams, run_dir: Path):
    """Configurable training loop with train/val/test evaluation."""
    # --- Reproducibility ---
    seed_everything(cfg.training.seed)

    # --- Get git commit hash ---
    commit_hash = get_git_commit_hash()


    log_path = run_dir / "stdout.log"
    err_path = run_dir / "stderr.log"
    with open(log_path, "w", buffering=1) as log_file, open(err_path, "w", buffering=1) as err_log:
        sys.stdout = log_file
        sys.stderr = err_log

        print(f"Run directory: {run_dir.resolve()}")

        # Save full config
        config_path = run_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(cfg.model_dump(), f, sort_keys=False, indent=2)
        print(f"Saved config to {config_path}")

        # --- Device setup ---
        device = get_device()
        if cfg.training.use_tf32_matmul and device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            print("TF32 matmul precision: enabled ('high')")

        # --- Model, optimizer ---
        model = LatentModel(cfg).to(device)
        model.train()
        OptimizerClass = getattr(torch.optim, cfg.training.optimizer)
        optimizer = OptimizerClass(model.parameters(), lr=cfg.training.learning_rate)
        train_step_fn: Callable = globals()[cfg.training.train_step]

        # --- Load data ---
        data_path = f"graphs_data/fly/{cfg.training.simulation_config}/x_list_0.npy"
        sim_data = SimulationResults.load(data_path)
        data = torch.from_numpy(sim_data[FlyVisSim[cfg.training.column_to_model]]).to(device)
        split = cfg.training.data_split

        total_time_points = data.shape[0]
        assert split.train_end <= total_time_points
        assert split.validation_end <= total_time_points
        assert split.test_end <= total_time_points

        # Extract subsets
        train_data = data[split.train_start : split.train_end]
        val_data = data[split.validation_start : split.validation_end]
        test_data = data[split.test_start : split.test_end]

        # Load stimulus
        stim = torch.from_numpy(sim_data[FlyVisSim.STIMULUS]).to(device)

        # HACK: first 1736 entries are the stimulus
        train_stim = stim[split.train_start : split.train_end, :cfg.stimulus_encoder_params.num_input_dims]
        val_stim = stim[split.validation_start : split.validation_end, :cfg.stimulus_encoder_params.num_input_dims]
        test_stim = stim[split.test_start : split.test_end, :cfg.stimulus_encoder_params.num_input_dims]

        print(
            f"Data split: train {train_data.shape}, "
            f"val {val_data.shape}, test {test_data.shape}"
        )

        metrics = {
            "val_loss_constant_model": torch.nn.functional.mse_loss(
                val_data[: -cfg.evolver_params.time_units], val_data[cfg.evolver_params.time_units:]
            ).item(),
            "train_loss_constant_model": torch.nn.functional.mse_loss(
                train_data[: -cfg.evolver_params.time_units], train_data[cfg.evolver_params.time_units:]
            ).item(),
            "test_loss_constant_model": torch.nn.functional.mse_loss(
                test_data[: -cfg.evolver_params.time_units], test_data[cfg.evolver_params.time_units:]
            ).item(),
        }
        print(f"Constant model loss: {metrics}")

        # --- Batching setup ---
        num_time_points = train_data.shape[0]
        # one pass over the data is < 1s, so avoid the overhead of an epoch by artificially
        # resampling data points.
        batches_per_epoch = (
            max(1, num_time_points // cfg.training.batch_size) * cfg.training.data_passes_per_epoch
        )
        batch_iter = make_batches_random(
            train_data, train_stim, cfg.training.batch_size, cfg.evolver_params.time_units
        )

        # --- Log file ---
        log_path = run_dir / "training_log.csv"
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,val_loss\n")

        # --- Initialize GPU monitoring ---
        gpu_monitor = GPUMonitor()
        if gpu_monitor.enabled:
            print(f"GPU monitoring enabled for: {gpu_monitor.gpu_name}")
        else:
            print("GPU monitoring not available (no NVIDIA GPU detected)")

        training_start = datetime.now()
        epoch_durations = []

        # --- Epoch loop ---
        for epoch in range(cfg.training.epochs):
            epoch_start = datetime.now()
            gpu_monitor.sample_epoch_start()
            running_loss = 0.0

            # ---- Training phase ----
            for _ in range(batches_per_epoch):
                optimizer.zero_grad()
                (x_t, stim_t), x_t_plus = next(batch_iter)
                (loss, _recon_loss, _evolve_loss, _reg_loss) = train_step_fn(model, x_t, stim_t, x_t_plus, cfg)
                loss.backward()
                optimizer.step()
                running_loss += loss.detach().item()

            mean_train_loss = running_loss / batches_per_epoch

            # ---- Validation phase ----
            model.eval()
            with torch.no_grad():
                val_x_t = val_data[: -cfg.evolver_params.time_units]
                val_stim_t = val_stim[: -cfg.evolver_params.time_units]
                val_x_t_plus = val_data[cfg.evolver_params.time_units :]
                val_loss = torch.nn.functional.mse_loss(
                    model(val_x_t, val_stim_t), val_x_t_plus
                ).item()
            model.train()

            epoch_end = datetime.now()
            gpu_monitor.sample_epoch_end()
            epoch_duration = (epoch_end - epoch_start).total_seconds()
            epoch_durations.append(epoch_duration)
            total_elapsed = (epoch_end - training_start).total_seconds()

            print(
                f"Epoch {epoch+1}/{cfg.training.epochs} | "
                f"Train Loss: {mean_train_loss:.4e} | Val Loss: {val_loss:.4e} | "
                f"Duration: {epoch_duration:.2f}s (Total: {total_elapsed:.1f}s)"
            )

            # Log to CSV
            with open(log_path, "a") as f:
                f.write(f"{epoch+1},{mean_train_loss:.6f},{val_loss:.6f}\n")

        # --- Final test evaluation ---
        model.eval()
        with torch.no_grad():
            test_x_t = test_data[: -cfg.evolver_params.time_units]
            test_stim_t = test_stim[: -cfg.evolver_params.time_units]
            test_x_t_plus = test_data[cfg.evolver_params.time_units :]
            test_loss = torch.nn.functional.mse_loss(
                model(test_x_t, test_stim_t), test_x_t_plus
            ).item()

        print(f"Final Test Loss: {test_loss:.4e}")

        # Calculate training statistics
        training_end = datetime.now()
        total_training_duration = (training_end - training_start).total_seconds()
        avg_epoch_duration = sum(epoch_durations) / len(epoch_durations) if epoch_durations else 0.0

        # Collect GPU metrics
        gpu_metrics = gpu_monitor.get_metrics()

        metrics.update(
            {
                    "final_train_loss": mean_train_loss,
                    "final_val_loss": val_loss,
                    "final_test_loss": test_loss,
                    "commit_hash": commit_hash,
                    "training_duration_seconds": round(total_training_duration, 2),
                    "avg_epoch_duration_seconds": round(avg_epoch_duration, 2),
                }
        )
        metrics.update(gpu_metrics)

        # --- Save final model ---
        model_path = run_dir / "model_final.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Saved final model to {model_path}")

        # --- Run post-training diagnostics ---
        post_run_metrics = post_training_diagnostics(
            run_dir=run_dir,
            val_data=val_data,
            neuron_data=sim_data.neuron_data,
            val_stim=val_stim,
            model=model,
            config=cfg,
        )
        metrics.update(post_run_metrics)

        # Save final metrics
        metrics_path = run_dir / "final_metrics.yaml"
        with open(metrics_path, "w") as f:
            yaml.dump(
                metrics,
                f,
                sort_keys=False,
                indent=2,
            )
        print(f"Saved metrics to {metrics_path}")


# -------------------------------------------------------------------
# CLI Entry Point
# -------------------------------------------------------------------

if __name__ == "__main__":
    msg = "First argument should be an expt_code that turns into a directory. \
expt_code should match `[A-Za-z0-9_]+`. To view available overrides run \
`latent.py dummy_code --help`."
    if len(sys.argv) == 1:
        print(msg)
        sys.exit(1)

    # Extract expt_code from command line
    expt_code = sys.argv[1]

    if not re.match("[A-Za-z0-9_]+", expt_code):
        print(msg)
        sys.exit(1)

    # Create argument list for tyro (excluding expt_code)
    tyro_args = sys.argv[2:]

    commit_hash = get_git_commit_hash()

    # Make run dir with hierarchical structure: runs / expt_code / run_id
    run_id = datetime.now().strftime("%Y%m%d") + "_" + commit_hash + "_" + str(uuid4())[:8]
    run_dir = Path("runs") / expt_code / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Log command line in run dir for tracking
    with open(run_dir / "command_line.txt", "w") as out:
        out.write("\n".join(sys.argv))

    # Load default YAML
    default_path = (
        Path(__file__).resolve().parent / "latent_default.yaml"
    )
    with open(default_path, "r") as f:
        data = yaml.safe_load(f)
    default_cfg = ModelParams(**data)

    # Parse CLI overrides with Tyro, passing filtered args explicitly
    cfg = tyro.cli(ModelParams, default=default_cfg, args=tyro_args)

    train(cfg, run_dir)

    # add a completion flag
    with open(run_dir / "complete", "w"):
        pass


