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

import torch
import torch.nn as nn
import yaml
import tyro
import numpy as np
from pydantic import BaseModel, Field, field_validator, ConfigDict

from LatentEvolution.load_flyvis import SimulationResults, FlyVisSim


# -------------------------------------------------------------------
# Pydantic Config Classes
# -------------------------------------------------------------------


class MLPParams(BaseModel):
    num_input_dims: int
    num_output_dims: int
    num_hidden_layers: int
    num_hidden_units: int
    activation: str = Field("ReLU", description="Activation function from torch.nn")

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @field_validator("activation")
    @classmethod
    def validate_activation(cls, v: str) -> str:
        if not hasattr(nn, v):
            raise ValueError(f"Unknown activation '{v}' in torch.nn")
        return v


class EvolverParams(BaseModel):
    time_units: int
    num_hidden_units: int
    num_hidden_layers: int
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class EncoderParams(BaseModel):
    num_hidden_units: int
    num_hidden_layers: int
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class DecoderParams(BaseModel):
    num_hidden_units: int
    num_hidden_layers: int
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class DataSplit(BaseModel):
    """Split the time series into train/validation/test."""

    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    test_start: int
    test_end: int

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @field_validator("*")
    @classmethod
    def check_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Indices in data_split must be non-negative.")
        return v

    @field_validator("train_end")
    @classmethod
    def check_order(cls, v, info):
        # very basic ordering sanity check
        d = info.data
        if "train_start" in d and v <= d["train_start"]:
            raise ValueError("train_end must be greater than train_start.")
        return v


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
    encoder_params: EncoderParams
    decoder_params: DecoderParams
    evolver_params: EvolverParams
    training: TrainingConfig

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# -------------------------------------------------------------------
# PyTorch Models
# -------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, params: MLPParams):
        super().__init__()
        self.layers = nn.ModuleList()
        input_dims = params.num_input_dims

        for _ in range(params.num_hidden_layers):
            self.layers.append(nn.Linear(input_dims, params.num_hidden_units))
            self.layers.append(nn.BatchNorm1d(params.num_hidden_units))
            self.layers.append(getattr(nn, params.activation)())
            input_dims = params.num_hidden_units

        if params.num_hidden_layers:
            self.layers.append(
                nn.Linear(params.num_hidden_units, params.num_output_dims)
            )
        else:
            self.layers.append(nn.Linear(params.num_input_dims, params.num_output_dims))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Evolver(nn.Module):
    def __init__(self, params: ModelParams):
        super().__init__()
        self.time_units = params.evolver_params.time_units
        self.evolver = MLP(
            MLPParams(
                num_input_dims=params.latent_dims,
                num_hidden_layers=params.evolver_params.num_hidden_layers,
                num_hidden_units=params.evolver_params.num_hidden_units,
                num_output_dims=params.latent_dims,
            )
        )

    def forward(self, x):
        for _ in range(self.time_units):
            x = x + self.evolver(x)
        return x


class LatentModel(nn.Module):
    def __init__(self, params: ModelParams):
        super().__init__()
        self.encoder = MLP(
            MLPParams(
                num_input_dims=params.num_neurons,
                num_hidden_layers=params.encoder_params.num_hidden_layers,
                num_hidden_units=params.encoder_params.num_hidden_units,
                num_output_dims=params.latent_dims,
            )
        )
        self.decoder = MLP(
            MLPParams(
                num_input_dims=params.latent_dims,
                num_hidden_layers=params.decoder_params.num_hidden_layers,
                num_hidden_units=params.decoder_params.num_hidden_units,
                num_output_dims=params.num_neurons,
            )
        )
        self.evolver = Evolver(params)

    def forward(self, x_t):
        proj_t = self.encoder(x_t)
        proj_t_next = self.evolver(proj_t)
        x_t_next = self.decoder(proj_t_next)
        return x_t_next


# -------------------------------------------------------------------
# Data + Batching
# -------------------------------------------------------------------


def load_column_data(path: str, column: FlyVisSim) -> torch.Tensor:
    sim = SimulationResults.load(path)
    return torch.from_numpy(sim[column]).float()  # (T, N)


def make_batches_random(
    data: torch.Tensor, batch_size: int, time_units: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
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
        x_t_plus = data[start_indices + time_units]
        yield x_t, x_t_plus


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


torch._dynamo.config.compiled_autograd = True


@torch.compile(fullgraph=True, mode="max-autotune")
def train_step(model, x_t, x_t_plus):
    # evolution loss
    output = model(x_t)
    evolve_loss = torch.nn.functional.mse_loss(output, x_t_plus)

    # reconstruction loss
    recon = model.decoder(model.encoder(x_t))
    recon_loss = torch.nn.functional.mse_loss(recon, x_t)

    loss = evolve_loss + recon_loss
    return (loss, recon_loss, evolve_loss)


# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------


def train(cfg: ModelParams):
    """Configurable training loop with train/val/test evaluation."""
    # --- Reproducibility ---
    seed_everything(cfg.training.seed)

    # --- Get git commit hash ---
    commit_hash = get_git_commit_hash()

    # --- Run directory creation ---
    run_id = datetime.now().strftime("%Y%m%d") + "_" + commit_hash + "_" + str(uuid4())[:8]
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "stdout.log"

    with open(log_path, "w", buffering=1) as log_file:  # line-buffered
        sys.stdout = log_file
        sys.stderr = log_file  # capture errors too

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
        data = load_column_data(
            data_path, FlyVisSim[cfg.training.column_to_model]
        ).to(device)
        split = cfg.training.data_split

        total_time_points = data.shape[0]
        assert split.train_end <= total_time_points
        assert split.validation_end <= total_time_points
        assert split.test_end <= total_time_points

        # Extract subsets
        train_data = data[split.train_start : split.train_end]
        val_data = data[split.validation_start : split.validation_end]
        test_data = data[split.test_start : split.test_end]

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
        batches_per_epoch = max(1, num_time_points // cfg.training.batch_size)
        batch_iter = make_batches_random(
            train_data, cfg.training.batch_size, cfg.evolver_params.time_units
        )

        # --- Log file ---
        log_path = run_dir / "training_log.csv"
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,val_loss\n")

        training_start = datetime.now()

        # --- Epoch loop ---
        for epoch in range(cfg.training.epochs):
            epoch_start = datetime.now()
            running_loss = 0.0

            # ---- Training phase ----
            for _ in range(batches_per_epoch):
                optimizer.zero_grad()
                x_t, x_t_plus = next(batch_iter)
                (loss, _recon_loss, _evolve_loss) = train_step_fn(model, x_t, x_t_plus)
                loss.backward()
                optimizer.step()
                running_loss += loss.detach().item()

            mean_train_loss = running_loss / batches_per_epoch

            # ---- Validation phase ----
            model.eval()
            with torch.no_grad():
                val_x_t = val_data[: -cfg.evolver_params.time_units]
                val_x_t_plus = val_data[cfg.evolver_params.time_units :]
                val_loss = torch.nn.functional.mse_loss(
                    model(val_x_t), val_x_t_plus
                ).item()
            model.train()

            epoch_end = datetime.now()
            epoch_duration = (epoch_end - epoch_start).total_seconds()
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
            test_x_t_plus = test_data[cfg.evolver_params.time_units :]
            test_loss = torch.nn.functional.mse_loss(
                model(test_x_t), test_x_t_plus
            ).item()

        print(f"Final Test Loss: {test_loss:.4e}")

        metrics.update(
            {
                    "final_train_loss": mean_train_loss,
                    "final_val_loss": val_loss,
                    "final_test_loss": test_loss,
                    "commit_hash": commit_hash,
                }
        )
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

        # --- Save final model ---
        model_path = run_dir / "model_final.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Saved final model to {model_path}")


# -------------------------------------------------------------------
# CLI Entry Point
# -------------------------------------------------------------------


if __name__ == "__main__":
    # Load default YAML
    default_path = (
        Path(__file__).resolve().parent / "latent_default.yaml"
    )
    with open(default_path, "r") as f:
        data = yaml.safe_load(f)
    default_cfg = ModelParams(**data)

    # Parse CLI overrides with Tyro
    cfg = tyro.cli(ModelParams, default=default_cfg)

    train(cfg)
