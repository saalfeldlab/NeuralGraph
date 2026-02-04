"""
stimulus-only null model: predicts neural activity from stimulus context alone.

tests whether stimulus context is sufficient to predict activity,
without encoder/decoder/latent space. serves as a baseline for the
full EED model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable
from datetime import datetime
from enum import Enum, auto
import gc
import sys
import re
import time
import signal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import yaml
import tyro
from pydantic import BaseModel, Field, ConfigDict, field_validator

from LatentEvolution.load_flyvis import (
    NeuronData,
    FlyVisSim,
    load_column_slice,
)
from LatentEvolution.training_config import (
    TrainingConfig,
    CrossValidationConfig,
    ProfileConfig,
)
from LatentEvolution.stimulus_ae_model import (
    StimulusEncoderParams,
    pretrain_stimulus_ae,
)
from LatentEvolution.mlp import MLP, MLPWithSkips, MLPParams
from LatentEvolution.gpu_stats import GPUMonitor
from LatentEvolution.diagnostics import (
    MultiStartRolloutResults,
    compute_linear_interpolation_baseline,
    plot_multi_start_rollout_figures,
    plot_short_rollout_mse,
    PlotMode,
)
from LatentEvolution.hparam_paths import create_run_directory, get_git_commit_hash
from LatentEvolution.training_utils import (
    LossAccumulator,
    seed_everything,
    get_device,
)
from LatentEvolution.chunk_streaming import (
    calculate_chunk_params,
    ChunkLatencyStats,
)
from LatentEvolution.latent import load_dataset, load_val_only


# -------------------------------------------------------------------
# config classes
# -------------------------------------------------------------------


class PredictorParams(BaseModel):
    num_hidden_layers: int = 2
    num_hidden_units: int = 512
    l1_reg_loss: float = 0.0
    use_input_skips: bool = Field(False, description="if true, use MLPWithSkips instead of standard MLP")
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class StimulusOnlyModelParams(BaseModel):
    num_neurons: int
    activation: str = Field("ReLU", description="activation function from torch.nn")
    stimulus_encoder_params: StimulusEncoderParams
    predictor_params: PredictorParams = Field(default_factory=PredictorParams)
    training: TrainingConfig
    profiling: ProfileConfig | None = Field(
        None, description="optional profiler configuration"
    )
    cross_validation_configs: list[CrossValidationConfig] = Field(
        default_factory=lambda: [CrossValidationConfig(simulation_config="fly_N9_62_0")],
        description="list of datasets to validate on after training"
    )

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @field_validator("activation")
    @classmethod
    def validate_activation(cls, v: str) -> str:
        if not hasattr(nn, v):
            raise ValueError(f"unknown activation '{v}' in torch.nn")
        return v

    def flatten(self, sep: str = ".") -> dict[str, int | float | str | bool]:
        """flatten params into a single-level dictionary."""
        def _flatten_dict(d, parent_key=""):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(_flatten_dict(v, new_key).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        return _flatten_dict(self.model_dump())


# -------------------------------------------------------------------
# loss types
# -------------------------------------------------------------------


class LossType(Enum):
    TOTAL = auto()
    PREDICT = auto()
    REG = auto()


# -------------------------------------------------------------------
# model
# -------------------------------------------------------------------


class StimulusOnlyModel(nn.Module):
    def __init__(self, params: StimulusOnlyModelParams):
        super().__init__()
        self.time_units = params.training.time_units

        # stimulus encoder (same as latent.py)
        stim_cls = MLPWithSkips if params.stimulus_encoder_params.use_input_skips else MLP
        self.stimulus_encoder = stim_cls(
            MLPParams(
                num_input_dims=params.stimulus_encoder_params.num_input_dims,
                num_hidden_units=params.stimulus_encoder_params.num_hidden_units,
                num_hidden_layers=params.stimulus_encoder_params.num_hidden_layers,
                num_output_dims=params.stimulus_encoder_params.num_output_dims,
                use_batch_norm=False,
                activation=params.activation,
            )
        )

        # predictor MLP: takes flattened stimulus context -> neural activity
        stim_output_dims = params.stimulus_encoder_params.num_output_dims
        predictor_input_dims = params.training.time_units * stim_output_dims
        pred_cls = MLPWithSkips if params.predictor_params.use_input_skips else MLP
        self.predictor = pred_cls(
            MLPParams(
                num_input_dims=predictor_input_dims,
                num_hidden_layers=params.predictor_params.num_hidden_layers,
                num_hidden_units=params.predictor_params.num_hidden_units,
                num_output_dims=params.num_neurons,
                use_batch_norm=False,
                activation=params.activation,
            )
        )

    def forward(self, stim_context: torch.Tensor) -> torch.Tensor:
        """
        predict neural activity from stimulus context.

        args:
            stim_context: (batch, time_units, stim_input_dims)

        returns:
            x_hat: (batch, num_neurons)
        """
        batch, tu, stim_dim = stim_context.shape
        # encode each frame
        z_s = self.stimulus_encoder(stim_context.reshape(batch * tu, stim_dim))
        stim_latent_dim = z_s.shape[1]
        z_s = z_s.reshape(batch, tu * stim_latent_dim)
        # predict
        x_hat = self.predictor(z_s)
        return x_hat


# -------------------------------------------------------------------
# training step
# -------------------------------------------------------------------


def train_step_nocompile(
    model: StimulusOnlyModel,
    train_data: torch.Tensor,
    train_stim: torch.Tensor,
    batch_indices: torch.Tensor,
    cfg: StimulusOnlyModelParams,
) -> dict[LossType, torch.Tensor]:
    """single training step for stimulus-only model."""
    device = train_data.device
    tu = cfg.training.time_units
    loss_fn = getattr(torch.nn.functional, cfg.training.loss_function)

    # gather stimulus context: stim[t-tu:t] for each t in batch_indices
    # batch_indices are valid starting points where t >= tu
    # stim_context[i] = stim[batch_indices[i]-tu : batch_indices[i]]
    offsets = torch.arange(tu, device=device).unsqueeze(0)  # (1, tu)
    context_starts = (batch_indices - tu).unsqueeze(1)  # (batch, 1)
    context_indices = context_starts + offsets  # (batch, tu)

    stim_context = train_stim[context_indices]  # (batch, tu, stim_dim)

    # target: activity at time t
    target = train_data[batch_indices]  # (batch, num_neurons)

    # forward
    x_hat = model(stim_context)
    predict_loss = loss_fn(x_hat, target)

    # regularization
    reg_loss = torch.tensor(0.0, device=device)
    if cfg.predictor_params.l1_reg_loss > 0.0:
        for p in model.predictor.parameters():
            reg_loss += torch.abs(p).mean() * cfg.predictor_params.l1_reg_loss

    total_loss = predict_loss + reg_loss

    return {
        LossType.TOTAL: total_loss,
        LossType.PREDICT: predict_loss,
        LossType.REG: reg_loss,
    }


train_step = torch.compile(train_step_nocompile, fullgraph=True, mode="reduce-overhead")


# -------------------------------------------------------------------
# diagnostics
# -------------------------------------------------------------------


def make_evolve_fn(
    model: StimulusOnlyModel,
    time_units: int,
) -> Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]:
    """create evolve_fn for stimulus-only rollouts.

    the evolve_fn predicts x_hat(t) from stimulus context at each step,
    ignoring initial_state (since the model is stimulus-only).
    """
    def evolve_fn(
        initial_state: torch.Tensor,
        stimulus_segment: torch.Tensor,
        n_steps: int,
    ) -> torch.Tensor:
        """
        predict neural activity for n_steps from stimulus alone.

        non-autoregressive: all steps batched in a single forward pass.

        args:
            initial_state: (neurons,) - ignored by this model
            stimulus_segment: (tu - 1 + n_steps, stim_dim) - includes prior context
            n_steps: number of steps to predict

        returns:
            predicted: (n_steps, neurons)
        """
        tu = time_units
        device = stimulus_segment.device

        # stimulus_segment has tu-1 context frames prepended, so total length
        # is tu - 1 + n_steps. build sliding windows of size tu.
        # for step t (0-indexed), context = stimulus_segment[t : t + tu]
        indices = torch.arange(n_steps, device=device).unsqueeze(1) + torch.arange(tu, device=device).unsqueeze(0)
        stim_context = stimulus_segment[indices]  # (n_steps, tu, stim_dim)

        return model(stim_context)

    return evolve_fn


def compute_stimulus_only_rollout_mse(
    evolve_fn: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
    val_data: torch.Tensor,
    val_stim: torch.Tensor,
    time_units: int,
    n_steps: int = 2000,
    n_starts: int = 10,
) -> MultiStartRolloutResults:
    """compute multi-start rollout mse for the stimulus-only model.

    constrains start indices to >= time_units so the evolve_fn always
    receives real stimulus context (no zero-padding needed).

    args:
        evolve_fn: callable(initial_state, stimulus_segment, n_steps) -> (n_steps, neurons).
            stimulus_segment has shape (time_units - 1 + n_steps, stim_dim).
        val_data: validation data (T, N)
        val_stim: validation stimulus (T, S)
        time_units: number of prior stimulus frames needed for context
        n_steps: number of rollout steps
        n_starts: number of random starting points
    """
    tu = time_units
    min_start = tu
    max_start = val_data.shape[0] - n_steps - 1
    if max_start - min_start < n_starts:
        raise ValueError(
            f"not enough data for {n_starts} starts with {n_steps} steps "
            f"and min_start={min_start}. available range: {max_start - min_start}, "
            f"have {val_data.shape[0]} time points"
        )

    rng = np.random.default_rng(seed=0)
    start_indices = rng.choice(np.arange(min_start, max_start), size=n_starts, replace=False)

    n_neurons = val_data.shape[1]
    mse_array = np.zeros((n_starts, n_steps, n_neurons))
    null_model_mse_by_time = np.zeros(n_steps)

    best_mse = float("inf")
    worst_mse = -float("inf")
    best_segment_data = None
    worst_segment_data = None

    for i, start_idx in enumerate(start_indices):
        initial_state = val_data[start_idx]
        # include tu-1 prior frames of stimulus for context
        stimulus_segment = val_stim[start_idx - (tu - 1):start_idx + n_steps]
        real_segment = val_data[start_idx + 1:start_idx + n_steps + 1]

        # null model mse
        null_squared_error = torch.pow(real_segment - initial_state, 2)
        null_model_mse_by_time += null_squared_error.mean(dim=1).detach().cpu().numpy()

        predicted_segment = evolve_fn(initial_state, stimulus_segment, n_steps)

        squared_error = torch.pow(predicted_segment - real_segment, 2).detach().cpu().numpy()
        mse_array[i] = squared_error
        mean_mse = np.nanmean(squared_error)

        real_np = real_segment.detach().cpu().numpy()
        pred_np = predicted_segment.detach().cpu().numpy()

        if best_segment_data is None:
            best_segment_data = (start_idx, real_np, pred_np)
            worst_segment_data = (start_idx, real_np, pred_np)
            best_mse = mean_mse
            worst_mse = mean_mse
        else:
            if mean_mse > worst_mse:
                worst_segment_data = (start_idx, real_np, pred_np)
                worst_mse = mean_mse
            if mean_mse < best_mse:
                best_segment_data = (start_idx, real_np, pred_np)
                best_mse = mean_mse

        del predicted_segment, real_segment, squared_error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    null_model_mse_by_time /= n_starts

    return MultiStartRolloutResults(
        mse_array=mse_array,
        start_indices=start_indices.tolist(),
        best_segment=best_segment_data,
        worst_segment=worst_segment_data,
        null_model_mse_by_time=null_model_mse_by_time,
    )


def run_diagnostics(
    run_dir: Path,
    val_data: torch.Tensor,
    neuron_data: NeuronData,
    val_stim: torch.Tensor,
    model: StimulusOnlyModel,
    config: StimulusOnlyModelParams,
    plot_mode: PlotMode,
) -> tuple[dict[str, float | int], dict[str, plt.Figure]]:
    """run validation diagnostics for stimulus-only model."""
    overall_start = time.time()
    metrics = {}
    figures = {}

    run_dir.mkdir(parents=True, exist_ok=True)

    tu = config.training.time_units
    ems = config.training.evolve_multiple_steps

    evolve_fn = make_evolve_fn(model, tu)

    with torch.no_grad():
        results = compute_stimulus_only_rollout_mse(
            evolve_fn, val_data, val_stim, time_units=tu, n_steps=2000, n_starts=20,
        )

    new_figs, new_metrics = plot_multi_start_rollout_figures(
        results, neuron_data, "stimulus_only", plot_mode, tu, ems,
    )
    metrics.update(new_metrics)
    figures.update(new_figs)

    # short rollout mse plot (fixed 250-step window, comparable across runs)
    short_rollout_steps = 250
    print("computing short rollout mse...")

    constant_baseline_accumulator = np.zeros(short_rollout_steps)
    for start_idx in results.start_indices:
        x_0 = val_data[start_idx]
        ground_truth = val_data[start_idx + 1:start_idx + 1 + short_rollout_steps]
        constant_pred = x_0.unsqueeze(0).expand(ground_truth.shape[0], -1)
        constant_se = torch.pow(constant_pred - ground_truth, 2).mean(dim=1)
        constant_baseline_accumulator += constant_se.detach().cpu().numpy()
    constant_baseline = constant_baseline_accumulator / len(results.start_indices)

    linear_interp_baseline = None
    if tu > 1:
        baseline_ems = -(-short_rollout_steps // tu)  # ceil division
        linear_interp_baseline = compute_linear_interpolation_baseline(
            val_data, results.start_indices, results.mse_array.shape[1],
            tu, baseline_ems,
        )

    fig = plot_short_rollout_mse(
        results.mse_array, constant_baseline,
        tu, ems, "stimulus_only",
        n_steps=short_rollout_steps,
        linear_interp_baseline=linear_interp_baseline,
        column_to_model=config.training.column_to_model,
    )
    figures["short_rollout_mse_stimulus_only"] = fig

    if plot_mode.save_figures:
        for key, fig in figures.items():
            fig.savefig(run_dir / f"{key}.jpg", dpi=100)

    elapsed = time.time() - overall_start
    print(f"validation diagnostics complete ({elapsed:.1f}s)")
    return metrics, figures


# -------------------------------------------------------------------
# training loop
# -------------------------------------------------------------------


def train(cfg: StimulusOnlyModelParams, run_dir: Path):
    """training loop for stimulus-only null model."""
    seed_everything(cfg.training.seed)

    # signal handling for graceful termination
    terminate_flag = {"value": False}

    def handle_sigusr2(signum, frame):
        terminate_flag["value"] = True
        print("\nSIGUSR2 received - will terminate after current epoch")

    signal.signal(signal.SIGUSR2, handle_sigusr2)

    commit_hash = get_git_commit_hash()

    log_path = run_dir / "stdout.log"
    err_path = run_dir / "stderr.log"
    with open(log_path, "w", buffering=1) as log_file, open(err_path, "w", buffering=1) as err_log:
        sys.stdout = log_file
        sys.stderr = err_log

        print(f"run directory: {run_dir.resolve()}")

        # save config
        config_path = run_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(cfg.model_dump(mode='json'), f, sort_keys=False, indent=2)
        print(f"saved config to {config_path}")

        # device setup
        device = get_device()
        if cfg.training.use_tf32_matmul and device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            print("TF32 matmul precision: enabled ('high')")

        # model + optimizer
        model = StimulusOnlyModel(cfg).to(device)
        print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")
        model.train()
        OptimizerClass = getattr(torch.optim, cfg.training.optimizer)
        optimizer = OptimizerClass(model.parameters(), lr=cfg.training.learning_rate)

        # tensorboard
        writer = SummaryWriter(log_dir=run_dir)
        print(f"tensorboard --logdir={run_dir} --samples_per_plugin=images=1000")

        # load data
        chunk_loader, val_data, val_stim, neuron_data, train_total_timesteps = load_dataset(
            simulation_config=cfg.training.simulation_config,
            column_to_model=cfg.training.column_to_model,
            data_split=cfg.training.data_split,
            num_input_dims=cfg.stimulus_encoder_params.num_input_dims,
            device=device,
            chunk_size=65536,
            time_units=cfg.training.time_units,
            training_data_path=cfg.training.training_data_path,
            gpu_prefetch=2,
        )

        print(f"chunked streaming: train {train_total_timesteps} timesteps (chunked), val {val_data.shape}")

        tu = cfg.training.time_units

        # baselines
        total_steps = tu * cfg.training.evolve_multiple_steps
        metrics = {
            "val_loss_constant_model": torch.nn.functional.mse_loss(
                val_data[:-total_steps], val_data[total_steps:]
            ).item(),
        }
        print(f"constant model baseline (val): {metrics['val_loss_constant_model']:.4e}")
        writer.add_scalar("Baseline/val_loss_constant_model", metrics["val_loss_constant_model"], 0)

        # cross-validation datasets
        cv_datasets: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for cv_config in cfg.cross_validation_configs:
            cv_name = cv_config.name or cv_config.simulation_config
            data_split = cv_config.data_split or cfg.training.data_split
            cv_val_data, cv_val_stim = load_val_only(
                simulation_config=cv_config.simulation_config,
                column_to_model=cfg.training.column_to_model,
                data_split=data_split,
                num_input_dims=cfg.stimulus_encoder_params.num_input_dims,
                device=device,
            )
            cv_datasets[cv_name] = (cv_val_data, cv_val_stim)
            print(f"loaded cross-validation dataset: {cv_name} (val shape: {cv_val_data.shape})")

        # chunking parameters
        chunk_size = 65536
        chunks_per_epoch, batches_per_chunk, batches_per_epoch = calculate_chunk_params(
            total_timesteps=train_total_timesteps,
            chunk_size=chunk_size,
            batch_size=cfg.training.batch_size,
            data_passes_per_epoch=cfg.training.data_passes_per_epoch,
        )
        print(f"chunking: {chunks_per_epoch} chunks/epoch, {batches_per_chunk} batches/chunk, {batches_per_epoch} total batches/epoch")

        # stimulus autoencoder pretraining
        if cfg.training.pretrain_stimulus_ae:
            print("\n=== stimulus autoencoder pretraining ===")
            if cfg.training.training_data_path is not None:
                stim_data_path = cfg.training.training_data_path
            else:
                stim_data_path = f"graphs_data/fly/{cfg.training.simulation_config}/x_list_0"

            stim_np = load_column_slice(
                stim_data_path,
                FlyVisSim.STIMULUS.value,
                cfg.training.data_split.train_start,
                cfg.training.data_split.train_end,
                neuron_limit=cfg.stimulus_encoder_params.num_input_dims,
            )

            stim_ae = pretrain_stimulus_ae(
                stim_np=stim_np,
                encoder_params=cfg.stimulus_encoder_params,
                train_cfg=cfg.training.stimulus_ae,
                activation=cfg.activation,
                device=device,
                run_dir=run_dir,
                writer=writer,
            )
            del stim_np
            gc.collect()

            # copy pretrained encoder weights and freeze
            model.stimulus_encoder.load_state_dict(stim_ae.encoder.state_dict())
            model.stimulus_encoder.requires_grad_(False)
            print("=== stimulus encoder frozen ===\n")

        # gpu monitoring
        gpu_monitor = GPUMonitor()
        if gpu_monitor.enabled:
            print(f"GPU monitoring enabled for: {gpu_monitor.gpu_name}")
        else:
            print("GPU monitoring not available (no NVIDIA GPU detected)")

        training_start = datetime.now()
        epoch_durations = []

        # checkpoint setup
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        # latency tracking
        latency_stats = ChunkLatencyStats()

        # epoch loop
        for epoch in range(cfg.training.epochs):
            epoch_start = datetime.now()
            gpu_monitor.sample_epoch_start()
            losses = LossAccumulator(LossType)

            chunk_loader.start_epoch(num_chunks=chunks_per_epoch)

            for _ in range(chunks_per_epoch):
                get_start = time.time()
                chunk_start, chunk_data, chunk_stim = chunk_loader.get_next_chunk()
                latency_stats.record_chunk_get(time.time() - get_start)

                if chunk_data is None:
                    break

                for batch_in_chunk in range(batches_per_chunk):
                    optimizer.zero_grad()

                    # sample random time indices where t >= tu within the chunk
                    chunk_len = chunk_data.shape[0]
                    batch_indices = torch.randint(
                        low=tu, high=chunk_len, size=(cfg.training.batch_size,),
                        device=device,
                    )

                    forward_start = time.time()
                    loss_dict = train_step_nocompile(
                        model, chunk_data, chunk_stim, batch_indices, cfg,
                    )
                    forward_time = time.time() - forward_start

                    backward_start = time.time()
                    loss_dict[LossType.TOTAL].backward()
                    backward_time = time.time() - backward_start

                    step_start = time.time()
                    if cfg.training.grad_clip_max_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_max_norm)
                    optimizer.step()
                    step_time = time.time() - step_start

                    losses.accumulate(loss_dict)

                    if batch_in_chunk % 10 == 0:
                        latency_stats.record_batch_times(forward_time, backward_time, step_time)

            mean_losses = losses.mean()

            epoch_end = datetime.now()
            gpu_monitor.sample_epoch_end()
            epoch_duration = (epoch_end - epoch_start).total_seconds()
            epoch_durations.append(epoch_duration)
            total_elapsed = (epoch_end - training_start).total_seconds()

            print(
                f"epoch {epoch+1}/{cfg.training.epochs} | "
                f"train loss: {mean_losses[LossType.TOTAL]:.4e} | "
                f"duration: {epoch_duration:.2f}s (total: {total_elapsed:.1f}s)"
            )

            for loss_type, loss_value in mean_losses.items():
                writer.add_scalar(f"Loss/train_{loss_type.name.lower()}", loss_value, epoch)
            writer.add_scalar("Time/epoch_duration", epoch_duration, epoch)
            writer.add_scalar("Time/total_elapsed", total_elapsed, epoch)

            # latency stats
            latency_summary = latency_stats.get_summary()
            writer.add_scalar("Latency/chunk_get_mean_ms", latency_summary["chunk_get_mean_ms"], epoch)
            writer.add_scalar("Latency/chunk_get_max_ms", latency_summary["chunk_get_max_ms"], epoch)
            writer.add_scalar("Latency/batch_forward_mean_ms", latency_summary["batch_forward_mean_ms"], epoch)
            writer.add_scalar("Latency/batch_backward_mean_ms", latency_summary["batch_backward_mean_ms"], epoch)
            writer.add_scalar("Latency/batch_step_mean_ms", latency_summary["batch_step_mean_ms"], epoch)

            # periodic diagnostics
            if cfg.training.diagnostics_freq_epochs > 0 and (epoch + 1) % cfg.training.diagnostics_freq_epochs == 0:
                model.eval()
                diagnostics_start = datetime.now()
                diagnostic_metrics, diagnostic_figures = run_diagnostics(
                    run_dir=run_dir,
                    val_data=val_data,
                    neuron_data=neuron_data,
                    val_stim=val_stim,
                    model=model,
                    config=cfg,
                    plot_mode=PlotMode.TRAINING,
                )
                diagnostics_duration = (datetime.now() - diagnostics_start).total_seconds()

                for metric_name, metric_value in diagnostic_metrics.items():
                    writer.add_scalar(f"Val/{metric_name}", metric_value, epoch)
                for fig_name, fig in diagnostic_figures.items():
                    writer.add_figure(f"Val/{fig_name}", fig, epoch)
                writer.add_scalar("Time/diagnostics_duration", diagnostics_duration, epoch)
                print(f"  diagnostics completed in {diagnostics_duration:.2f}s")

                # cross-validation diagnostics
                for cv_name, (cv_val_data, cv_val_stim) in cv_datasets.items():
                    cv_start = datetime.now()
                    cv_metrics, cv_figures = run_diagnostics(
                        run_dir=run_dir / "cross_validation" / cv_name,
                        val_data=cv_val_data,
                        neuron_data=neuron_data,
                        val_stim=cv_val_stim,
                        model=model,
                        config=cfg,
                        plot_mode=PlotMode.TRAINING,
                    )
                    cv_duration = (datetime.now() - cv_start).total_seconds()
                    for metric_name, metric_value in cv_metrics.items():
                        writer.add_scalar(f"CrossVal/{cv_name}/{metric_name}", metric_value, epoch)
                    for fig_name, fig in cv_figures.items():
                        writer.add_figure(f"CrossVal/{cv_name}/{fig_name}", fig, epoch)
                    print(f"  cross-validation ({cv_name}) completed in {cv_duration:.2f}s")

                model.train()

            # checkpointing
            if cfg.training.save_checkpoint_every_n_epochs > 0 and (epoch + 1) % cfg.training.save_checkpoint_every_n_epochs == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1:04d}.pt"
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  -> saved periodic checkpoint at epoch {epoch+1}")

            latest_checkpoint_path = checkpoint_dir / "checkpoint_latest.pt"
            torch.save(model.state_dict(), latest_checkpoint_path)

            # graceful termination
            if terminate_flag["value"]:
                print(f"\n=== graceful termination at epoch {epoch+1} ===")
                break

        # training complete
        training_end = datetime.now()
        total_training_duration = (training_end - training_start).total_seconds()
        avg_epoch_duration = sum(epoch_durations) / len(epoch_durations) if epoch_durations else 0.0

        gpu_metrics = gpu_monitor.get_metrics()

        metrics.update({
            "final_train_loss": mean_losses[LossType.TOTAL],
            "commit_hash": commit_hash,
            "training_duration_seconds": round(total_training_duration, 2),
            "avg_epoch_duration_seconds": round(avg_epoch_duration, 2),
        })
        metrics.update(gpu_metrics)

        # save final model
        model_path = run_dir / "model_final.pt"
        torch.save(model.state_dict(), model_path)
        print(f"saved final model to {model_path}")

        # post-training analysis
        print("\n=== running post-training analysis ===")

        model.eval()
        run_diagnostics(
            run_dir=run_dir,
            val_data=val_data,
            neuron_data=neuron_data,
            val_stim=val_stim,
            model=model,
            config=cfg,
            plot_mode=PlotMode.POST_RUN,
        )
        print(f"saved main validation figures to {run_dir}")

        # cross-validation
        if cv_datasets:
            print("\n=== running cross-dataset validation ===")
            for cv_name, (cv_val_data, cv_val_stim) in cv_datasets.items():
                print(f"\nevaluating on {cv_name}...")
                cv_out_dir = run_dir / "cross_validation" / cv_name
                cv_metrics, cv_figures = run_diagnostics(
                    run_dir=cv_out_dir,
                    val_data=cv_val_data,
                    neuron_data=neuron_data,
                    val_stim=cv_val_stim,
                    model=model,
                    config=cfg,
                    plot_mode=PlotMode.POST_RUN,
                )
                print(f"saved cross-validation figures to {cv_out_dir}")
                for metric_name, metric_value in cv_metrics.items():
                    writer.add_scalar(f"CrossVal/{cv_name}/{metric_name}", metric_value, cfg.training.epochs)
                for fig_name, fig in cv_figures.items():
                    writer.add_figure(f"CrossVal/{cv_name}/{fig_name}", fig, cfg.training.epochs)

        # save metrics
        metrics_path = run_dir / "final_metrics.yaml"
        with open(metrics_path, "w") as f:
            yaml.dump(metrics, f, sort_keys=False, indent=2)
        print(f"saved metrics to {metrics_path}")

        # tensorboard hparams
        hparams = {
            k: v for k, v in cfg.flatten().items() if not (
                k.startswith("profiling") or k.startswith("cross_validation_configs")
            ) and isinstance(v, (int, float, str, bool))
        }
        metric_dict = {
            "hparam/final_train_loss": metrics["final_train_loss"],
        }
        writer.add_hparams(hparams, metric_dict)
        print("logged hyperparameters to tensorboard")

        # latency stats
        print("\n=== final chunked streaming latency stats ===")
        latency_stats.print_summary()

        chunk_loader.cleanup()
        print("chunk loader cleanup complete")

        writer.close()
        print("tensorboard logging completed")

        return terminate_flag["value"]


# -------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------

if __name__ == "__main__":
    msg = """usage: python stimulus_only.py <expt_code> <default_config> [overrides...]

arguments:
  expt_code       experiment code (must match [A-Za-z0-9_]+)
  default_config  base config file (e.g., stimulus_only_20step.yaml)

example:
  python stimulus_only.py my_experiment stimulus_only_20step.yaml --training.epochs 100

to view available overrides:
  python stimulus_only.py dummy stimulus_only_20step.yaml --help"""

    if len(sys.argv) < 3:
        print(msg)
        sys.exit(1)

    expt_code = sys.argv[1]
    default_yaml = sys.argv[2]

    if not re.match("[A-Za-z0-9_]+", expt_code):
        print(f"error: expt_code must match [A-Za-z0-9_]+, got: {expt_code}")
        sys.exit(1)

    default_path = Path(__file__).resolve().parent / default_yaml
    if not default_path.exists():
        print(f"error: default config file not found: {default_path}")
        sys.exit(1)

    tyro_args = sys.argv[3:]

    commit_hash = get_git_commit_hash()

    run_dir = create_run_directory(
        expt_code=expt_code,
        tyro_args=tyro_args,
        model_class=StimulusOnlyModelParams,
        commit_hash=commit_hash,
    )

    with open(run_dir / "command_line.txt", "w") as out:
        out.write("\n".join(sys.argv))

    with open(default_path, "r") as f:
        data = yaml.safe_load(f)
    default_cfg = StimulusOnlyModelParams(**data)

    cfg = tyro.cli(StimulusOnlyModelParams, default=default_cfg, args=tyro_args)

    was_terminated = train(cfg, run_dir)

    flag_file = "terminated" if was_terminated else "complete"
    with open(run_dir / flag_file, "w"):
        pass
