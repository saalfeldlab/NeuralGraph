"""
Latent space voltage model with optional YAML config, torch.compile support,
and reproducible training with run logging.
"""

from pathlib import Path
from typing import Callable, Iterator
from datetime import datetime
from dataclasses import dataclass
import random
import sys
import re

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import yaml
import tyro
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from LatentEvolution.load_flyvis import SimulationResults, FlyVisSim, DataSplit
from LatentEvolution.gpu_stats import GPUMonitor
from LatentEvolution.diagnostics import run_validation_diagnostics, PlotMode
from LatentEvolution.hparam_paths import create_run_directory, get_git_commit_hash
from LatentEvolution.eed_model import (
    MLP,
    MLPWithSkips,
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


class ProfileConfig(BaseModel):
    """Configuration for PyTorch profiler to generate Chrome traces."""
    wait: int = Field(
        1, description="Number of epochs to skip before starting profiler warmup"
    )
    warmup: int = Field(
        1, description="Number of epochs for profiler warmup"
    )
    active: int = Field(
        1, description="Number of epochs to actively profile"
    )
    repeat: int = Field(
        0, description="Number of times to repeat the profiling cycle"
    )
    record_shapes: bool = Field(
        True, description="Record tensor shapes in the trace"
    )
    profile_memory: bool = Field(
        True, description="Profile memory usage"
    )
    with_stack: bool = Field(
        False, description="Record source code stack traces (increases overhead)"
    )
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class TrainingConfig(BaseModel):
    time_units: int = Field(
        1,
        description="Observation interval: activity data available every N steps. Evolver unrolled N times during training.",
        json_schema_extra={"short_name": "tu"}
    )
    epochs: int = Field(10, json_schema_extra={"short_name": "ep"})
    batch_size: int = Field(32, json_schema_extra={"short_name": "bs"})
    learning_rate: float = Field(1e-3, json_schema_extra={"short_name": "lr"})
    optimizer: str = Field("Adam", description="Optimizer name from torch.optim", json_schema_extra={"short_name": "opt"})
    train_step: str = Field("train_step", description="Compiled train step function")
    simulation_config: str
    column_to_model: str = "CALCIUM"
    use_tf32_matmul: bool = Field(
        False, description="Enable fast tf32 multiplication on certain NVIDIA GPUs"
    )
    seed: int = Field(42, json_schema_extra={"short_name": "seed"})
    data_split: DataSplit
    data_passes_per_epoch: int = 1
    diagnostics_freq_epochs: int = Field(
        0, description="Run validation diagnostics every N epochs (0 = only at end of training)"
    )
    save_checkpoint_every_n_epochs: int = Field(
        10, description="Save model checkpoint every N epochs (0 = disabled)"
    )
    save_best_checkpoint: bool = Field(
        True, description="Save checkpoint when validation loss improves"
    )
    loss_function: str = Field(
        "mse_loss", description="Loss function name from torch.nn.functional (e.g., 'mse_loss', 'huber_loss', 'l1_loss')"
    )
    lp_norm_weight: float = Field(
        0.0, description="Weight for LP norm penalty on latent activations (for outlier control)", json_schema_extra={"short_name": "lp_w"}
    )
    lp_norm_p: int = Field(
        8, description="P value for LP norm penalty (higher values penalize outliers more)", json_schema_extra={"short_name": "lp_p"}
    )
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @field_validator("optimizer")
    @classmethod
    def validate_optimizer(cls, v: str) -> str:
        if not hasattr(torch.optim, v):
            raise ValueError(f"Unknown optimizer '{v}' in torch.optim")
        return v

    @field_validator("loss_function")
    @classmethod
    def validate_loss_function(cls, v: str) -> str:
        if not hasattr(torch.nn.functional, v):
            raise ValueError(f"Unknown loss function '{v}' in torch.nn.functional")
        return v


class CrossValidationConfig(BaseModel):
    """Configuration for cross-dataset validation."""
    simulation_config: str
    name: str | None = None  # Optional human-readable name
    data_split: DataSplit | None = None # data split

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ModelParams(BaseModel):
    latent_dims: int = Field(..., json_schema_extra={"short_name": "ld"})
    num_neurons: int
    use_batch_norm: bool = True
    activation: str = Field("ReLU", description="Activation function from torch.nn")
    encoder_params: EncoderParams
    decoder_params: DecoderParams
    evolver_params: EvolverParams
    stimulus_encoder_params: StimulusEncoderParams
    training: TrainingConfig
    profiling: ProfileConfig | None = Field(
        None, description="Optional profiler configuration to generate Chrome traces for performance analysis"
    )
    cross_validation_configs: list[CrossValidationConfig] = Field(
        default_factory=lambda: [CrossValidationConfig(simulation_config="fly_N9_62_0")],
        description="List of datasets to validate on after training"
    )

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @field_validator("activation")
    @classmethod
    def validate_activation(cls, v: str) -> str:
        if not hasattr(nn, v):
            raise ValueError(f"Unknown activation '{v}' in torch.nn")
        return v

    @model_validator(mode='after')
    def validate_encoder_decoder_symmetry(self):
        """Ensure encoder and decoder have symmetric MLP parameters."""
        if self.encoder_params.num_hidden_units != self.decoder_params.num_hidden_units:
            raise ValueError(
                f"Encoder and decoder must have the same num_hidden_units. "
                f"Got encoder={self.encoder_params.num_hidden_units}, decoder={self.decoder_params.num_hidden_units}"
            )
        if self.encoder_params.num_hidden_layers != self.decoder_params.num_hidden_layers:
            raise ValueError(
                f"Encoder and decoder must have the same num_hidden_layers. "
                f"Got encoder={self.encoder_params.num_hidden_layers}, decoder={self.decoder_params.num_hidden_layers}"
            )
        if self.encoder_params.use_input_skips != self.decoder_params.use_input_skips:
            raise ValueError(
                f"Encoder and decoder must have the same use_input_skips setting. "
                f"Got encoder={self.encoder_params.use_input_skips}, decoder={self.decoder_params.use_input_skips}"
            )
        return self

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

        # Encoder: use MLPWithSkips if flag is set
        encoder_cls = MLPWithSkips if params.encoder_params.use_input_skips else MLP
        self.encoder = encoder_cls(
            MLPParams(
                num_input_dims=params.num_neurons,
                num_hidden_layers=params.encoder_params.num_hidden_layers,
                num_hidden_units=params.encoder_params.num_hidden_units,
                num_output_dims=params.latent_dims,
                use_batch_norm=params.use_batch_norm,
                activation=params.activation,
            )
        )

        # Decoder: use MLPWithSkips if flag is set
        decoder_cls = MLPWithSkips if params.decoder_params.use_input_skips else MLP
        self.decoder = decoder_cls(
            MLPParams(
                num_input_dims=params.latent_dims,
                num_hidden_layers=params.decoder_params.num_hidden_layers,
                num_hidden_units=params.decoder_params.num_hidden_units,
                num_output_dims=params.num_neurons,
                use_batch_norm=params.use_batch_norm,
                activation=params.activation,
            )
        )

        # Stimulus encoder: use MLPWithSkips if flag is set
        stimulus_encoder_cls = MLPWithSkips if params.stimulus_encoder_params.use_input_skips else MLP
        self.stimulus_encoder = stimulus_encoder_cls(
            MLPParams(
                num_input_dims=params.stimulus_encoder_params.num_input_dims,
                num_hidden_units=params.stimulus_encoder_params.num_hidden_units,
                num_hidden_layers=params.stimulus_encoder_params.num_hidden_layers,
                num_output_dims=params.stimulus_encoder_params.num_output_dims,
                use_batch_norm=False,
                activation=params.activation,
            )
        )

        self.evolver = Evolver(
            latent_dims=params.latent_dims,
            stim_dims=params.stimulus_encoder_params.num_output_dims,
            evolver_params=params.evolver_params,
            use_batch_norm=params.use_batch_norm,
            activation=params.activation,
        )

    def forward(self, x_t, stim_t):
        proj_t = self.encoder(x_t)
        proj_stim_t = self.stimulus_encoder(stim_t)
        proj_t_next = self.evolver(proj_t, proj_stim_t)
        x_t_next = self.decoder(proj_t_next)
        return x_t_next


# -------------------------------------------------------------------
# Data + Batching
# -------------------------------------------------------------------

def make_batches_random(
    data: torch.Tensor, stim: torch.Tensor, batch_size: int, time_units: int
) -> Iterator[torch.Tensor]:
    """Randomly sample `batch_size` starting points."""
    total_time = data.shape[0]
    if total_time <= time_units:
        raise ValueError(
            f"Not enough time points ({total_time}) for time_units={time_units}"
        )
    while True:
        start_indices = torch.randint(
            low=0, high=total_time - time_units, size=(batch_size,), device=data.device
        )
        yield start_indices


# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------


@dataclass
class LossComponents:
    """Accumulator for tracking loss components."""
    total: float = 0.0
    recon: float = 0.0
    evolve: float = 0.0
    reg: float = 0.0
    lp_norm: float = 0.0
    count: int = 0

    def accumulate(self, *losses):
        """Add losses from one batch (total, recon, evolve, reg, lp_norm)."""
        self.total += losses[0].detach().item()
        self.recon += losses[1].detach().item()
        self.evolve += losses[2].detach().item()
        self.reg += losses[3].detach().item()
        self.lp_norm += losses[4].detach().item()
        self.count += 1

    def mean(self) -> 'LossComponents':
        """Return a new LossComponents with mean values."""
        return LossComponents(
            total=self.total / self.count,
            recon=self.recon / self.count,
            evolve=self.evolve / self.count,
            reg=self.reg / self.count,
            lp_norm=self.lp_norm / self.count,
            count=self.count,
        )


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


def train_step_nocompile(
        model: LatentModel,
        train_data: torch.Tensor,
        train_stim: torch.Tensor,
        batch_indices: torch.Tensor,
        cfg: ModelParams
    ):

    # Get loss function from config
    loss_fn = getattr(torch.nn.functional, cfg.training.loss_function)

    # regularization loss
    reg_loss = torch.tensor(0.0, device=train_data.device)
    if cfg.encoder_params.l1_reg_loss > 0.:
        for p in model.encoder.parameters():
            reg_loss += torch.abs(p).mean()*cfg.encoder_params.l1_reg_loss
    if cfg.decoder_params.l1_reg_loss > 0.:
        for p in model.decoder.parameters():
            reg_loss += torch.abs(p).mean()*cfg.decoder_params.l1_reg_loss
    if cfg.evolver_params.l1_reg_loss > 0.:
        for p in model.evolver.parameters():
            reg_loss += torch.abs(p).mean()*cfg.evolver_params.l1_reg_loss


    # b = batch size
    # N = neurons
    # L = latent dim for activity
    # Ls = latent dim for stimulus
    dt = cfg.training.time_units
    x_t = train_data[batch_indices] # b x N
    proj_t = model.encoder(x_t) # b x L

    stim_indices = torch.unsqueeze(batch_indices, dim=0) + torch.unsqueeze(torch.arange(dt, device=train_data.device), dim=1)
    # dt x b x 1736
    stim_t = train_stim[stim_indices, :]
    dim_stim = train_stim.shape[1]
    dim_stim_latent = cfg.stimulus_encoder_params.num_output_dims
    # dt x b x Ls
    proj_stim_t = model.stimulus_encoder(stim_t.reshape((-1, dim_stim))).reshape((dt, -1, dim_stim_latent))

    # prediction target
    x_t_plus = train_data[batch_indices + dt]

    # reconstruction loss
    recon_t = model.decoder(proj_t)
    recon_loss = loss_fn(recon_t, x_t)

    # evolution loss
    # evolve proj_t by dt
    for i in range(dt):
        proj_t = model.evolver(proj_t, proj_stim_t[i])
    pred_t_plus = model.decoder(proj_t)
    evolve_loss = loss_fn(pred_t_plus, x_t_plus)


    # LP norm penalty on prediction errors (for outlier control)
    lp_norm_loss = torch.tensor(0.0, device=x_t.device)
    if cfg.training.lp_norm_weight > 0.:
        lp_norm_evolve = torch.norm(pred_t_plus - x_t_plus, p=cfg.training.lp_norm_p, dim=1).mean()
        lp_norm_recon = torch.norm(recon_t - x_t, p=cfg.training.lp_norm_p, dim=1).mean()
        lp_norm_loss = cfg.training.lp_norm_weight * (lp_norm_evolve + lp_norm_recon)

    loss = evolve_loss + recon_loss + reg_loss + lp_norm_loss
    return (loss, recon_loss, evolve_loss, reg_loss, lp_norm_loss)

train_step = torch.compile(train_step_nocompile, fullgraph=True, mode="reduce-overhead")

# -------------------------------------------------------------------
# Data Loading and Evaluation
# -------------------------------------------------------------------


def load_dataset(
    simulation_config: str,
    column_to_model: str,
    data_split: DataSplit,
    num_input_dims: int,
    device: torch.device
):
    """
    Load and split a dataset.

    Args:
        simulation_config: Name of simulation config (e.g., "fly_N9_62_1")
        column_to_model: Column name to model (e.g., "VOLTAGE", "CALCIUM")
        data_split: DataSplit object with train/val/test time ranges
        num_input_dims: Number of stimulus input dimensions to keep
        device: PyTorch device to load data onto

    Returns:
        Tuple of (train_data, val_data, test_data, train_stim, val_stim, test_stim, neuron_data)
    """
    data_path = f"graphs_data/fly/{simulation_config}/x_list_0.npy"
    sim_data = SimulationResults.load(data_path)

    # Extract subsets using split_column method
    train_data_np, val_data_np, test_data_np = sim_data.split_column(
        FlyVisSim[column_to_model], data_split
    )
    train_data = torch.from_numpy(train_data_np).to(device)
    val_data = torch.from_numpy(val_data_np).to(device)
    test_data = torch.from_numpy(test_data_np).to(device)

    # Load stimulus (keep only first num_input_dims features)
    train_stim_np, val_stim_np, test_stim_np = sim_data.split_column(
        FlyVisSim.STIMULUS, data_split, keep_first_n_limit=num_input_dims
    )
    train_stim = torch.from_numpy(train_stim_np).to(device)
    val_stim = torch.from_numpy(val_stim_np).to(device)
    test_stim = torch.from_numpy(test_stim_np).to(device)

    return train_data, val_data, test_data, train_stim, val_stim, test_stim, sim_data.neuron_data


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

        # --- TensorBoard setup ---
        writer = SummaryWriter(log_dir=run_dir)
        print(f"TensorBoard logs will be saved to {run_dir}")

        # Log model graph to TensorBoard
        dummy_x_t = torch.randn(cfg.training.batch_size, cfg.num_neurons).to(device)
        dummy_stim_t = torch.randn(cfg.training.batch_size, cfg.stimulus_encoder_params.num_input_dims).to(device)
        writer.add_graph(model, (dummy_x_t, dummy_stim_t))
        print("Logged model graph to TensorBoard")

        # --- Load data ---
        train_data, val_data, test_data, train_stim, val_stim, test_stim, neuron_data = load_dataset(
            simulation_config=cfg.training.simulation_config,
            column_to_model=cfg.training.column_to_model,
            data_split=cfg.training.data_split,
            num_input_dims=cfg.stimulus_encoder_params.num_input_dims,
            device=device,
        )

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

        # Log baseline metrics to TensorBoard
        writer.add_scalar("Baseline/train_loss_constant_model", metrics["train_loss_constant_model"], 0)
        writer.add_scalar("Baseline/val_loss_constant_model", metrics["val_loss_constant_model"], 0)
        writer.add_scalar("Baseline/test_loss_constant_model", metrics["test_loss_constant_model"], 0)

        # --- Batching setup ---
        num_time_points = train_data.shape[0]
        # one pass over the data is < 1s, so avoid the overhead of an epoch by artificially
        # resampling data points.
        batches_per_epoch = (
            max(1, num_time_points // cfg.training.batch_size) * cfg.training.data_passes_per_epoch
        )
        batch_indices_iter = make_batches_random(
            train_data, train_stim, cfg.training.batch_size, cfg.evolver_params.time_units
        )

        # --- Initialize GPU monitoring ---
        gpu_monitor = GPUMonitor()
        if gpu_monitor.enabled:
            print(f"GPU monitoring enabled for: {gpu_monitor.gpu_name}")
        else:
            print("GPU monitoring not available (no NVIDIA GPU detected)")

        training_start = datetime.now()
        epoch_durations = []

        # --- Checkpoint setup ---
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        best_val_loss = float('inf')

        # --- Profiler setup ---
        profiler = None
        if cfg.profiling is not None:
            assert device.type != "mps", (
                "PyTorch profiler is not supported on MPS (Apple Silicon). "
                "Please disable profiling or use CUDA/CPU for profiling."
            )

            print(f"PyTorch profiler enabled with config: {cfg.profiling.model_dump()}")
            profile_dir = run_dir / "profiler_traces"
            profile_dir.mkdir(exist_ok=True)

            profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(
                    wait=cfg.profiling.wait,
                    warmup=cfg.profiling.warmup,
                    active=cfg.profiling.active,
                    repeat=cfg.profiling.repeat
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_dir)),
                record_shapes=cfg.profiling.record_shapes,
                profile_memory=cfg.profiling.profile_memory,
                with_stack=cfg.profiling.with_stack,
                with_flops=True,
            )
            profiler.__enter__()
            print(f"Profiler traces will be saved to {profile_dir}")
        else:
            print("PyTorch profiler disabled")

        # --- Epoch loop ---
        for epoch in range(cfg.training.epochs):
            epoch_start = datetime.now()
            gpu_monitor.sample_epoch_start()
            losses = LossComponents()

            # ---- Training phase ----
            for _ in range(batches_per_epoch):
                optimizer.zero_grad()
                batch_indices = next(batch_indices_iter)
                loss_tuple = train_step_fn(model, train_data, train_stim, batch_indices, cfg)
                loss_tuple[0].backward()
                optimizer.step()
                losses.accumulate(*loss_tuple)

            mean_losses = losses.mean()

            # ---- Validation phase ----
            model.eval()
            with torch.no_grad():
                start_indices = torch.arange(val_data.shape[0] - cfg.evolver_params.time_units, device=device)
                loss_tuple = train_step_fn(model, val_data, val_stim, start_indices, cfg)
                val_loss = loss_tuple[0].item()

            model.train()

            epoch_end = datetime.now()
            gpu_monitor.sample_epoch_end()
            epoch_duration = (epoch_end - epoch_start).total_seconds()
            epoch_durations.append(epoch_duration)
            total_elapsed = (epoch_end - training_start).total_seconds()

            print(
                f"Epoch {epoch+1}/{cfg.training.epochs} | "
                f"Train Loss: {mean_losses.total:.4e} | Val Loss: {val_loss:.4e} | "
                f"Duration: {epoch_duration:.2f}s (Total: {total_elapsed:.1f}s)"
            )

            # Log to TensorBoard
            writer.add_scalar("Loss/train", mean_losses.total, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Loss/train_recon", mean_losses.recon, epoch)
            writer.add_scalar("Loss/train_evolve", mean_losses.evolve, epoch)
            writer.add_scalar("Loss/train_reg", mean_losses.reg, epoch)
            writer.add_scalar("Loss/train_lp_norm", mean_losses.lp_norm, epoch)
            writer.add_scalar("Time/epoch_duration", epoch_duration, epoch)
            writer.add_scalar("Time/total_elapsed", total_elapsed, epoch)

            # Run periodic diagnostics
            if cfg.training.diagnostics_freq_epochs > 0 and (epoch + 1) % cfg.training.diagnostics_freq_epochs == 0:
                model.eval()
                diagnostics_start = datetime.now()
                diagnostic_metrics, diagnostic_figures = run_validation_diagnostics(
                    run_dir=run_dir,
                    val_data=val_data,
                    neuron_data=neuron_data,
                    val_stim=val_stim,
                    model=model,
                    config=cfg,
                    plot_mode = PlotMode.TRAINING,
                )
                diagnostics_duration = (datetime.now() - diagnostics_start).total_seconds()

                # Log diagnostic metrics to TensorBoard
                for metric_name, metric_value in diagnostic_metrics.items():
                    writer.add_scalar(f"Diagnostics/{metric_name}", metric_value, epoch)

                # Log diagnostic figures to TensorBoard
                for fig_name, fig in diagnostic_figures.items():
                    writer.add_figure(f"Diagnostics/{fig_name}", fig, epoch)

                # Log diagnostics duration
                writer.add_scalar("Time/diagnostics_duration", diagnostics_duration, epoch)
                print(f"  Diagnostics completed in {diagnostics_duration:.2f}s")

                model.train()

            # --- Checkpointing ---
            # Save best checkpoint
            if cfg.training.save_best_checkpoint and val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = checkpoint_dir / "checkpoint_best.pt"
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  → Saved best checkpoint (val_loss: {val_loss:.4e})")

            # Save periodic checkpoint
            if cfg.training.save_checkpoint_every_n_epochs > 0 and (epoch + 1) % cfg.training.save_checkpoint_every_n_epochs == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1:04d}.pt"
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  → Saved periodic checkpoint at epoch {epoch+1}")

            # Step profiler if enabled
            if profiler is not None:
                profiler.step()

        # --- Cleanup profiler ---
        if profiler is not None:
            profiler.__exit__(None, None, None)
            print("Profiler traces saved successfully")

        # --- Final test evaluation ---
        model.eval()
        with torch.no_grad():
            start_indices = torch.arange(test_data.shape[0] - cfg.evolver_params.time_units, device=device)
            loss_tuple = train_step_fn(model, test_data, test_stim, start_indices, cfg)
            test_loss = loss_tuple[0].item()

        print(f"Final Test Loss: {test_loss:.4e}")

        # Log final test loss to TensorBoard
        writer.add_scalar("Loss/test", test_loss, cfg.training.epochs)

        # Calculate training statistics
        training_end = datetime.now()
        total_training_duration = (training_end - training_start).total_seconds()
        avg_epoch_duration = sum(epoch_durations) / len(epoch_durations) if epoch_durations else 0.0

        # Collect GPU metrics
        gpu_metrics = gpu_monitor.get_metrics()

        metrics.update(
            {
                    "final_train_loss": mean_losses.total,
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

        # --- Run post-training analysis ---
        print("\n=== Running post-training analysis ===")

        # Run diagnostics on main validation dataset
        run_validation_diagnostics(
            run_dir=run_dir,
            val_data=val_data,
            neuron_data=neuron_data,
            val_stim=val_stim,
            model=model,
            config=cfg,
            plot_mode = PlotMode.POST_RUN,
        )
        print(f"Saved main validation figures to {run_dir}")

        # Run cross-validation diagnostics
        if cfg.cross_validation_configs:
            print("\n=== Running Cross-Dataset Validation ===")

            for cv_config in cfg.cross_validation_configs:
                cv_name = cv_config.name or cv_config.simulation_config
                print(f"\nEvaluating on {cv_name} ({cv_config.simulation_config})...")

                # default to the same data split as training, unless specified
                data_split = cv_config.data_split or cfg.training.data_split

                # Load cross-validation dataset (only need validation split)
                _, cv_val_data, _, _, cv_val_stim, _, cv_neuron_data = load_dataset(
                    simulation_config=cv_config.simulation_config,
                    column_to_model=cfg.training.column_to_model,
                    data_split=data_split,
                    num_input_dims=cfg.stimulus_encoder_params.num_input_dims,
                    device=device,
                )

                # Run diagnostics on cross-validation dataset
                cv_out_dir = run_dir / "cross_validation" / cv_name
                cv_metrics, cv_figures = run_validation_diagnostics(
                    run_dir=cv_out_dir,
                    val_data=cv_val_data,
                    neuron_data=cv_neuron_data,
                    val_stim=cv_val_stim,
                    model=model,
                    config=cfg,
                    plot_mode = PlotMode.POST_RUN
                )
                print(f"Saved cross-validation figures to {cv_out_dir}")

                # Log cross-validation scalar metrics to TensorBoard
                for metric_name, metric_value in cv_metrics.items():
                    writer.add_scalar(f"CrossVal/{cv_name}/{metric_name}", metric_value, 0)
                print(f"Logged {len(cv_metrics)} cross-validation scalar metrics to TensorBoard")

                # Log figures to TensorBoard
                for fig_name, fig in cv_figures.items():
                    writer.add_figure(f"CrossVal/{cv_name}/{fig_name}", fig, 0)
                print("Logged MSE figures to TensorBoard")

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

        # Log hyperparameters to TensorBoard for comparison across runs
        # omit logging of profiling/cross-validation info
        hparams = {
            k: v for k, v in cfg.flatten().items() if not (
                k.startswith("profiling") or k.startswith("cross_validation_configs")
            )
        }
        metric_dict = {
            "hparam/final_train_loss": metrics["final_train_loss"],
            "hparam/final_val_loss": metrics["final_val_loss"],
            "hparam/final_test_loss": metrics["final_test_loss"],
        }
        writer.add_hparams(hparams, metric_dict)
        print("Logged hyperparameters to TensorBoard")

        # Close TensorBoard writer
        writer.close()
        print("TensorBoard logging completed")


# -------------------------------------------------------------------
# CLI Entry Point
# -------------------------------------------------------------------

if __name__ == "__main__":
    msg = """Usage: python latent.py <expt_code> <default_config> [overrides...]

Arguments:
  expt_code       Experiment code (must match [A-Za-z0-9_]+)
  default_config  Base config file (e.g., latent_1step.yaml, latent_5step.yaml)

Example:
  python latent.py my_experiment latent_1step.yaml --training.epochs 100

To view available overrides:
  python latent.py dummy latent_1step.yaml --help"""

    if len(sys.argv) < 3:
        print(msg)
        sys.exit(1)

    # Extract positional arguments
    expt_code = sys.argv[1]
    default_yaml = sys.argv[2]

    if not re.match("[A-Za-z0-9_]+", expt_code):
        print(f"Error: expt_code must match [A-Za-z0-9_]+, got: {expt_code}")
        sys.exit(1)

    # Resolve default path (relative to this file's directory)
    default_path = Path(__file__).resolve().parent / default_yaml
    if not default_path.exists():
        print(f"Error: Default config file not found: {default_path}")
        sys.exit(1)

    # Create argument list for tyro (excluding expt_code and default_config)
    tyro_args = sys.argv[3:]

    commit_hash = get_git_commit_hash()

    # Make run dir with hierarchical structure: runs / expt_code_date_hash / hparam1 / hparam2 / ... / uuid
    run_dir = create_run_directory(
        expt_code=expt_code,
        tyro_args=tyro_args,
        model_class=ModelParams,
        commit_hash=commit_hash,
    )

    # Log command line in run dir for tracking
    with open(run_dir / "command_line.txt", "w") as out:
        out.write("\n".join(sys.argv))

    # Load default YAML
    with open(default_path, "r") as f:
        data = yaml.safe_load(f)
    default_cfg = ModelParams(**data)

    # Parse CLI overrides with Tyro, passing filtered args explicitly
    cfg = tyro.cli(ModelParams, default=default_cfg, args=tyro_args)

    train(cfg, run_dir)

    # add a completion flag
    with open(run_dir / "complete", "w"):
        pass


