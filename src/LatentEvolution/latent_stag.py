"""
Latent staggered model: decode-only architecture with learned initial latents.

This model removes the encoder and instead learns z0 for each time window.
Requires staggered acquisition mode where different neurons are observed at
different phases within each time_units cycle.
"""

from pathlib import Path
from datetime import datetime
from enum import Enum, auto
import sys
import re
import signal

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import yaml
import tyro
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from LatentEvolution.training_config import TrainingConfig, CrossValidationConfig
from LatentEvolution.hparam_paths import create_run_directory, get_git_commit_hash
from LatentEvolution.eed_model import (
    MLP,
    MLPWithSkips,
    MLPParams,
    Evolver,
    EncoderParams,
    DecoderParams,
    EvolverParams,
    StimulusEncoderParams,
)
from LatentEvolution.training_utils import (
    LossAccumulator,
    seed_everything,
    get_device,
)
from LatentEvolution.chunk_streaming import calculate_chunk_params
from LatentEvolution.acquisition import (
    compute_neuron_phases,
    sample_batch_indices,
    StaggeredRandomMode,
)
from LatentEvolution.latent import load_dataset, load_val_only
from LatentEvolution.pipeline_chunk_loader import PipelineProfiler
from LatentEvolution.diagnostics_stag import run_validation_diagnostics


# -------------------------------------------------------------------
# Config Classes
# -------------------------------------------------------------------


class StagModelParams(BaseModel):
    """model parameters for staggered latent model with encoder warmup."""
    latent_dims: int = Field(..., json_schema_extra={"short_name": "ld"})
    num_neurons: int
    use_batch_norm: bool = True
    activation: str = Field("ReLU", description="activation function from torch.nn")
    encoder_params: EncoderParams
    decoder_params: DecoderParams
    evolver_params: EvolverParams
    stimulus_encoder_params: StimulusEncoderParams
    training: TrainingConfig
    cross_validation_configs: list[CrossValidationConfig] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @field_validator("activation")
    @classmethod
    def validate_activation(cls, v: str) -> str:
        if not hasattr(nn, v):
            raise ValueError(f"unknown activation '{v}' in torch.nn")
        return v

    @model_validator(mode='after')
    def validate_staggered_acquisition(self):
        """ensure acquisition mode is staggered_random."""
        if not isinstance(self.training.acquisition_mode, StaggeredRandomMode):
            raise ValueError(
                f"latent_stag requires staggered_random acquisition mode. "
                f"got: {self.training.acquisition_mode.mode}"
            )
        return self

    @model_validator(mode='after')
    def validate_encoder_decoder_symmetry(self):
        """ensure encoder and decoder have symmetric mlp parameters."""
        if self.encoder_params.num_hidden_units != self.decoder_params.num_hidden_units:
            raise ValueError(
                f"encoder and decoder must have the same num_hidden_units. "
                f"got encoder={self.encoder_params.num_hidden_units}, decoder={self.decoder_params.num_hidden_units}"
            )
        if self.encoder_params.num_hidden_layers != self.decoder_params.num_hidden_layers:
            raise ValueError(
                f"encoder and decoder must have the same num_hidden_layers. "
                f"got encoder={self.encoder_params.num_hidden_layers}, decoder={self.decoder_params.num_hidden_layers}"
            )
        if self.encoder_params.use_input_skips != self.decoder_params.use_input_skips:
            raise ValueError(
                f"encoder and decoder must have the same use_input_skips setting. "
                f"got encoder={self.encoder_params.use_input_skips}, decoder={self.decoder_params.use_input_skips}"
            )
        return self

    def flatten(self, sep: str = ".") -> dict[str, int | float | str | bool]:
        """flatten the params into a single-level dictionary."""
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
# Loss Types
# -------------------------------------------------------------------


class LossType(Enum):
    """loss component types for staggered model."""
    TOTAL = auto()
    RECON = auto()  # reconstruction loss (warmup phase)
    EVOLVE = auto()
    REG = auto()
    TV_LOSS = auto()
    Z0_CONSISTENCY = auto()  # consistency between evolved latent and z0_bank
    ENCODER_CONSISTENCY = auto()  # consistency z ≈ encoder(decoder(z))


# -------------------------------------------------------------------
# PyTorch Models
# -------------------------------------------------------------------


class LatentStagModel(nn.Module):
    """model with encoder (for warmup), decoder, evolver and stimulus encoder.

    the encoder is used during warmup to train encoder/decoder for reconstruction,
    then frozen. after warmup, the encoder is used to initialize the z0 bank.
    """

    def __init__(self, params: StagModelParams):
        super().__init__()

        # encoder (used for warmup only, then frozen)
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

        # decoder
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

        # stimulus encoder
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

        # evolver
        self.evolver = Evolver(
            latent_dims=params.latent_dims,
            stim_dims=params.stimulus_encoder_params.num_output_dims,
            evolver_params=params.evolver_params,
            use_batch_norm=params.use_batch_norm,
            activation=params.activation,
        )


# -------------------------------------------------------------------
# Training Step
# -------------------------------------------------------------------


@torch.compile(fullgraph=True, mode="reduce-overhead")
def train_step_warmup(
    model: LatentStagModel,
    train_data: torch.Tensor,
    observation_indices: torch.Tensor,
    cfg: StagModelParams,
) -> dict[LossType, torch.Tensor]:
    """
    warmup training step for encoder/decoder reconstruction.

    trains encoder and decoder to reconstruct observations. the staggered
    observation_indices are treated as if they were time-aligned (we just
    encode and decode without evolving).

    args:
        model: the staggered model (encoder, decoder, evolver, stim encoder)
        train_data: chunk data (chunk_timesteps, num_neurons)
        observation_indices: (batch_size, num_neurons) observation times
        cfg: model configuration

    returns:
        dict of loss components
    """
    device = train_data.device
    loss_fn = getattr(torch.nn.functional, cfg.training.loss_function)

    # regularization loss (encoder + decoder only)
    reg_loss = torch.tensor(0.0, device=device)
    if cfg.encoder_params.l1_reg_loss > 0.:
        for p in model.encoder.parameters():
            reg_loss += torch.abs(p).mean() * cfg.encoder_params.l1_reg_loss
    if cfg.decoder_params.l1_reg_loss > 0.:
        for p in model.decoder.parameters():
            reg_loss += torch.abs(p).mean() * cfg.decoder_params.l1_reg_loss

    # reconstruction loss
    # observation_indices: (b, N) - time index for each neuron in each batch sample
    # use advanced indexing to extract the right (time, neuron) pairs
    batch_size, num_neurons = observation_indices.shape
    neuron_indices = torch.arange(num_neurons, device=device).unsqueeze(0).expand(batch_size, num_neurons)
    x_t = train_data[observation_indices, neuron_indices]  # (b, N)
    proj_t = model.encoder(x_t)
    recon_t = model.decoder(proj_t)
    recon_loss = loss_fn(recon_t, x_t)

    # total loss
    total_loss = recon_loss + reg_loss

    return {
        LossType.TOTAL: total_loss,
        LossType.RECON: recon_loss,
        LossType.REG: reg_loss,
    }


def train_step_nocompile(
    model: LatentStagModel,
    z0_bank: nn.Embedding,
    z0_indices: torch.Tensor,
    train_data: torch.Tensor,
    train_stim: torch.Tensor,
    observation_indices: torch.Tensor,
    cfg: StagModelParams,
) -> dict[LossType, torch.Tensor]:
    """
    training step for staggered model.

    args:
        model: the staggered model (decoder, evolver, stim encoder)
        z0_bank: embedding layer storing z0 for each time window
        z0_indices: window indices for this batch (batch_size,)
        train_data: chunk data (chunk_timesteps, num_neurons)
        train_stim: chunk stimulus (chunk_timesteps, stim_dims)
        observation_indices: (batch_size, num_neurons) observation times
        cfg: model configuration

    returns:
        dict of loss components
    """
    device = train_data.device
    loss_fn = getattr(torch.nn.functional, cfg.training.loss_function)

    losses: dict[LossType, torch.Tensor] = {}

    # regularization loss
    reg_loss = torch.tensor(0.0, device=device)
    if cfg.decoder_params.l1_reg_loss > 0.:
        for p in model.decoder.parameters():
            reg_loss += torch.abs(p).mean() * cfg.decoder_params.l1_reg_loss
    if cfg.evolver_params.l1_reg_loss > 0.:
        for p in model.evolver.parameters():
            reg_loss += torch.abs(p).mean() * cfg.evolver_params.l1_reg_loss
    losses[LossType.REG] = reg_loss

    # total variation regularization
    tv_loss = torch.tensor(0.0, device=device)

    dt = cfg.training.time_units
    num_multiples = cfg.training.evolve_multiple_steps
    total_steps = dt * num_multiples

    batch_size, num_neurons = observation_indices.shape
    neuron_indices = torch.arange(num_neurons, device=device).unsqueeze(0).expand(batch_size, num_neurons)

    # stimulus for the full window
    batch_start_times = observation_indices.min(dim=1).values  # (batch_size,)
    stim_indices = batch_start_times.unsqueeze(0) + torch.arange(total_steps, device=device).unsqueeze(1)
    stim_t = train_stim[stim_indices, :]  # (total_steps, batch_size, stim_dims)
    dim_stim = train_stim.shape[1]
    dim_stim_latent = cfg.stimulus_encoder_params.num_output_dims

    # encode all stimulus: (total_steps, batch_size, stim_latent_dims)
    proj_stim_t = model.stimulus_encoder(stim_t.reshape((-1, dim_stim))).reshape((total_steps, -1, dim_stim_latent))

    # evolve from z0, compute loss at tu boundaries
    z0 = z0_bank(z0_indices)  # (batch_size, latent_dims)
    proj_t = z0
    evolve_loss = torch.tensor(0.0, device=device)
    z0_consistency_loss = torch.tensor(0.0, device=device)
    encoder_consistency_loss = torch.tensor(0.0, device=device)
    z0c_weight = cfg.training.z0_consistency_loss
    enc_weight = cfg.training.encoder_consistency_loss

    if cfg.evolver_params.tv_reg_loss > 0.:
        for t in range(total_steps):
            delta_z = model.evolver.evolver(torch.cat([proj_t, proj_stim_t[t]], dim=1))
            tv_loss += torch.abs(delta_z).mean() * cfg.evolver_params.tv_reg_loss
            proj_t = proj_t + delta_z

            # at tu boundaries, compute loss
            if (t + 1) % dt == 0:
                m = (t + 1) // dt
                x_pred = model.decoder(proj_t)  # prediction at tu boundary
                # ground truth at staggered observation times
                target_indices = observation_indices + m * dt
                x_target = train_data[target_indices, neuron_indices]
                evolve_loss = evolve_loss + loss_fn(x_pred, x_target)

                # z0 consistency: evolved latent should match z0_bank at this window
                if z0c_weight > 0. and m < num_multiples:
                    z0_target = z0_bank(z0_indices + m)
                    z0_consistency_loss = z0_consistency_loss + loss_fn(proj_t, z0_target)

                # encoder consistency: z ≈ encoder(decoder(z))
                if enc_weight > 0.:
                    z_recon = model.encoder(x_pred)
                    encoder_consistency_loss = encoder_consistency_loss + loss_fn(proj_t, z_recon)
    else:
        for t in range(total_steps):
            proj_t = model.evolver(proj_t, proj_stim_t[t])

            # at tu boundaries, compute loss
            if (t + 1) % dt == 0:
                m = (t + 1) // dt
                x_pred = model.decoder(proj_t)  # prediction at tu boundary
                # ground truth at staggered observation times
                target_indices = observation_indices + m * dt
                x_target = train_data[target_indices, neuron_indices]
                evolve_loss = evolve_loss + loss_fn(x_pred, x_target)

                # z0 consistency: evolved latent should match z0_bank at this window
                if z0c_weight > 0. and m < num_multiples:
                    z0_target = z0_bank(z0_indices + m)
                    z0_consistency_loss = z0_consistency_loss + loss_fn(proj_t, z0_target)

                # encoder consistency: z ≈ encoder(decoder(z))
                if enc_weight > 0.:
                    z_recon = model.encoder(x_pred)
                    encoder_consistency_loss = encoder_consistency_loss + loss_fn(proj_t, z_recon)

    losses[LossType.EVOLVE] = evolve_loss
    losses[LossType.Z0_CONSISTENCY] = z0_consistency_loss
    losses[LossType.ENCODER_CONSISTENCY] = encoder_consistency_loss
    losses[LossType.TV_LOSS] = tv_loss
    losses[LossType.TOTAL] = (
        reg_loss + tv_loss + evolve_loss
        + z0c_weight * z0_consistency_loss
        + enc_weight * encoder_consistency_loss
    )

    return losses


train_step = torch.compile(train_step_nocompile, fullgraph=True, mode="reduce-overhead")


# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------


def train(cfg: StagModelParams, run_dir: Path):
    """training loop for staggered latent model."""
    seed_everything(cfg.training.seed)

    # --- Signal handling for graceful termination ---
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
            yaml.dump(cfg.model_dump(), f, sort_keys=False, indent=2)
        print(f"saved config to {config_path}")

        # device setup
        device = get_device()
        if cfg.training.use_tf32_matmul and device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            print("tf32 matmul precision: enabled ('high')")

        # model
        model = LatentStagModel(cfg).to(device)
        print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")
        model.train()

        # pipeline profiler for chrome tracing (only profile first N epochs to limit file size)
        profile_first_n_epochs = 5
        profiler = PipelineProfiler()
        profiler.start()

        # load data (reuse from latent.py)
        dt = cfg.training.time_units
        chunk_loader, val_data, val_stim, _neuron_data, train_total_timesteps = load_dataset(
            simulation_config=cfg.training.simulation_config,
            column_to_model=cfg.training.column_to_model,
            data_split=cfg.training.data_split,
            num_input_dims=cfg.stimulus_encoder_params.num_input_dims,
            device=device,
            chunk_size=65536,
            time_units=dt,
            training_data_path=cfg.training.training_data_path,
            gpu_prefetch=2,  # double buffer for cpu->gpu transfer overlap
            profiler=profiler,
        )
        print(f"training data: {train_total_timesteps} timesteps (pipeline chunked streaming)")

        # z0 bank
        num_z0_windows = train_total_timesteps // dt
        print(f"z0 bank: {num_z0_windows} windows (one per {dt} timesteps)")

        z0_bank = nn.Embedding(num_z0_windows, cfg.latent_dims).to(device)
        nn.init.normal_(z0_bank.weight, mean=0.0, std=0.01)
        print(f"z0 bank parameters: {z0_bank.weight.numel():,}")

        # optimizer
        OptimizerClass = getattr(torch.optim, cfg.training.optimizer)
        optimizer = OptimizerClass(
            [
                {'params': model.parameters()},
                {'params': z0_bank.parameters()},
            ],
            lr=cfg.training.learning_rate
        )

        # tensorboard
        writer = SummaryWriter(log_dir=run_dir)
        print(f"tensorboard --logdir={run_dir}")

        # chunking
        chunk_size = 65536
        chunks_per_epoch, batches_per_chunk, batches_per_epoch = calculate_chunk_params(
            total_timesteps=train_total_timesteps,
            chunk_size=chunk_size,
            batch_size=cfg.training.batch_size,
            data_passes_per_epoch=cfg.training.data_passes_per_epoch,
        )
        print(f"chunking: {chunks_per_epoch} chunks/epoch, {batches_per_chunk} batches/chunk, {batches_per_epoch} total batches/epoch")

        # neuron phases (required for staggered acquisition)
        neuron_phases = compute_neuron_phases(
            num_neurons=cfg.num_neurons,
            time_units=dt,
            acquisition_mode=cfg.training.acquisition_mode,
            device=device,
        )
        assert neuron_phases is not None, "staggered_random acquisition mode requires neuron phases"
        print(f"acquisition mode: staggered_random, phases for {cfg.num_neurons} neurons")

        # --- load cross-validation datasets ---
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

        train_step_fn = globals()[cfg.training.train_step]
        total_steps = dt * cfg.training.evolve_multiple_steps

        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        # --- warmup phase: train encoder/decoder for reconstruction ---
        warmup_epochs = cfg.training.reconstruction_warmup_epochs
        if warmup_epochs > 0:
            print(f"\n=== encoder/decoder warmup: {warmup_epochs} epochs ===")

            # optimizer for warmup (encoder + decoder only)
            warmup_optimizer = OptimizerClass(
                [
                    {'params': model.encoder.parameters()},
                    {'params': model.decoder.parameters()},
                ],
                lr=cfg.training.learning_rate
            )

            for warmup_epoch in range(warmup_epochs):
                warmup_epoch_start = datetime.now()
                warmup_losses = LossAccumulator(LossType)

                chunk_loader.start_epoch(num_chunks=chunks_per_epoch)

                for _ in range(chunks_per_epoch):
                    chunk_start, chunk_data, chunk_stim = chunk_loader.get_next_chunk()
                    if chunk_data is None:
                        break

                    for _ in range(batches_per_chunk):
                        warmup_optimizer.zero_grad()

                        observation_indices = sample_batch_indices(
                            chunk_size=chunk_data.shape[0],
                            total_steps=total_steps,
                            time_units=dt,
                            batch_size=cfg.training.batch_size,
                            num_neurons=cfg.num_neurons,
                            neuron_phases=neuron_phases,
                            device=device,
                        )

                        loss_dict = train_step_warmup(
                            model, chunk_data, observation_indices, cfg
                        )
                        loss_dict[LossType.TOTAL].backward()
                        warmup_optimizer.step()
                        warmup_losses.accumulate(loss_dict)

                warmup_mean = warmup_losses.mean()
                warmup_duration = (datetime.now() - warmup_epoch_start).total_seconds()
                print(
                    f"warmup {warmup_epoch+1}/{warmup_epochs} | "
                    f"recon loss: {warmup_mean[LossType.RECON]:.4e} | "
                    f"duration: {warmup_duration:.2f}s"
                )

                # log to tensorboard
                for loss_type, loss_value in warmup_mean.items():
                    writer.add_scalar(f"Warmup/{loss_type.name.lower()}", loss_value, warmup_epoch)
                writer.add_scalar("Warmup/epoch_duration", warmup_duration, warmup_epoch)

            # --- initialize z0 bank from trained encoder ---
            print("\n=== initializing z0 bank from trained encoder ===")
            model.eval()

            chunk_loader.start_epoch(num_chunks=chunks_per_epoch)
            z0_init_count = 0

            with torch.no_grad():
                for _ in range(chunks_per_epoch):
                    chunk_start, chunk_data, chunk_stim = chunk_loader.get_next_chunk()
                    if chunk_data is None or chunk_start is None:
                        break

                    chunk_len = chunk_data.shape[0]
                    # encode at each tu boundary within this chunk
                    for offset in range(0, chunk_len - total_steps, dt):
                        global_time = chunk_start + offset
                        window_idx = global_time // dt

                        if window_idx >= num_z0_windows:
                            continue

                        # get observation at this tu boundary using neuron phases
                        obs_indices = neuron_phases + offset  # (num_neurons,)
                        obs_indices = obs_indices.unsqueeze(0)  # (1, num_neurons)
                        neuron_idx = torch.arange(cfg.num_neurons, device=device).unsqueeze(0)
                        x_t = chunk_data[obs_indices, neuron_idx]  # (1, num_neurons)

                        z0 = model.encoder(x_t)  # (1, latent_dims)
                        z0_bank.weight.data[window_idx] = z0.squeeze(0)
                        z0_init_count += 1

            print(f"initialized {z0_init_count} z0 entries from encoder")
            model.train()
            print("=== warmup complete ===\n")

        training_start = datetime.now()
        epoch_durations = []

        # epoch loop
        for epoch in range(cfg.training.epochs):
          # stop profiler after first N epochs and save trace immediately
          if epoch == profile_first_n_epochs and profiler.is_enabled():
              profiler.stop()
              trace_path = run_dir / "pipeline_trace.json"
              profiler.save(trace_path)
              print(f"profiler stopped after epoch {epoch}, saved to {trace_path}")
              profiler.print_stats()

          with profiler.event("epoch", "training", thread="main", epoch=epoch):
            epoch_start = datetime.now()
            losses = LossAccumulator(LossType)

            chunk_loader.start_epoch(num_chunks=chunks_per_epoch)

            for chunk_idx in range(chunks_per_epoch):
              with profiler.event("chunk", "pipeline", thread="main", chunk=chunk_idx):
                chunk_start, chunk_data, chunk_stim = chunk_loader.get_next_chunk()
                if chunk_data is None or chunk_start is None or chunk_stim is None:
                    break

                with profiler.event("train", "compute", thread="main"):
                    for _ in range(batches_per_chunk):
                        optimizer.zero_grad()

                        observation_indices = sample_batch_indices(
                            chunk_size=chunk_data.shape[0],
                            total_steps=total_steps,
                            time_units=dt,
                            batch_size=cfg.training.batch_size,
                            num_neurons=cfg.num_neurons,
                            neuron_phases=neuron_phases,
                            device=device,
                        )

                        # z0 indices for this batch
                        batch_start_times = observation_indices.min(dim=1).values
                        global_start_times = chunk_start + batch_start_times
                        z0_indices = global_start_times // dt

                        loss_dict = train_step_fn(
                            model, z0_bank, z0_indices, chunk_data, chunk_stim, observation_indices, cfg
                        )
                        loss_dict[LossType.TOTAL].backward()

                        if cfg.training.grad_clip_max_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                list(model.parameters()) + list(z0_bank.parameters()),
                                cfg.training.grad_clip_max_norm
                            )
                        optimizer.step()
                        losses.accumulate(loss_dict)

            mean_losses = losses.mean()
            epoch_end = datetime.now()
            epoch_duration = (epoch_end - epoch_start).total_seconds()
            epoch_durations.append(epoch_duration)
            total_elapsed = (epoch_end - training_start).total_seconds()

            print(
                f"epoch {epoch+1}/{cfg.training.epochs} | "
                f"loss: {mean_losses[LossType.TOTAL]:.4e} | "
                f"evolve: {mean_losses[LossType.EVOLVE]:.4e} | "
                f"duration: {epoch_duration:.2f}s"
            )

            for loss_type, loss_value in mean_losses.items():
                writer.add_scalar(f"Loss/train_{loss_type.name.lower()}", loss_value, epoch)
            writer.add_scalar("Time/epoch_duration", epoch_duration, epoch)
            writer.add_scalar("Time/total_elapsed", total_elapsed, epoch)

            # run validation diagnostics
            if cfg.training.diagnostics_freq_epochs > 0 and (epoch + 1) % cfg.training.diagnostics_freq_epochs == 0:
                model.eval()

                # main validation dataset
                diag_start = datetime.now()
                val_metrics, val_figures = run_validation_diagnostics(
                    val_data=val_data,
                    val_stim=val_stim,
                    model=model,
                    cfg=cfg,
                    epoch=epoch,
                )
                diag_duration = (datetime.now() - diag_start).total_seconds()

                for metric_name, metric_value in val_metrics.items():
                    writer.add_scalar(f"Val/{metric_name}", metric_value, epoch)
                for fig_name, fig in val_figures.items():
                    writer.add_figure(f"Val/{fig_name}", fig, epoch)
                    plt.close(fig)

                writer.add_scalar("Time/diagnostics_duration", diag_duration, epoch)
                print(f"  validation diagnostics: {diag_duration:.1f}s")

                # cross-validation datasets
                for cv_name, (cv_val_data, cv_val_stim) in cv_datasets.items():
                    cv_start = datetime.now()
                    cv_metrics, cv_figures = run_validation_diagnostics(
                        val_data=cv_val_data,
                        val_stim=cv_val_stim,
                        model=model,
                        cfg=cfg,
                        epoch=epoch,
                    )
                    cv_duration = (datetime.now() - cv_start).total_seconds()

                    for metric_name, metric_value in cv_metrics.items():
                        writer.add_scalar(f"CrossVal/{cv_name}/{metric_name}", metric_value, epoch)
                    for fig_name, fig in cv_figures.items():
                        writer.add_figure(f"CrossVal/{cv_name}/{fig_name}", fig, epoch)
                        plt.close(fig)

                    print(f"  cv/{cv_name} diagnostics: {cv_duration:.1f}s")

                model.train()

            if cfg.training.save_checkpoint_every_n_epochs > 0 and (epoch + 1) % cfg.training.save_checkpoint_every_n_epochs == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1:04d}.pt"
                torch.save({
                    'model': model.state_dict(),
                    'z0_bank': z0_bank.state_dict(),
                }, checkpoint_path)
                print(f"  -> saved checkpoint at epoch {epoch+1}")

            # save latest checkpoint (overwrite each epoch)
            latest_checkpoint_path = checkpoint_dir / "checkpoint_latest.pt"
            torch.save({
                'model': model.state_dict(),
                'z0_bank': z0_bank.state_dict(),
            }, latest_checkpoint_path)

            # check for graceful termination signal
            if terminate_flag["value"]:
                print(f"\n=== graceful termination at epoch {epoch+1} ===")
                break

        # training complete
        training_end = datetime.now()
        total_training_duration = (training_end - training_start).total_seconds()
        avg_epoch_duration = sum(epoch_durations) / len(epoch_durations) if epoch_durations else 0.0

        model_path = run_dir / "model_final.pt"
        torch.save({
            'model': model.state_dict(),
            'z0_bank': z0_bank.state_dict(),
        }, model_path)
        print(f"saved final model to {model_path}")

        metrics = {
            "final_train_loss": float(mean_losses[LossType.TOTAL]),
            "commit_hash": commit_hash,
            "training_duration_seconds": round(total_training_duration, 2),
            "avg_epoch_duration_seconds": round(avg_epoch_duration, 2),
            "num_z0_windows": num_z0_windows,
        }
        metrics_path = run_dir / "final_metrics.yaml"
        with open(metrics_path, "w") as f:
            yaml.dump(metrics, f, sort_keys=False, indent=2)
        print(f"saved metrics to {metrics_path}")

        chunk_loader.cleanup()

        # save profiler if still running (epochs < profile_first_n_epochs)
        if profiler.is_enabled():
            profiler.stop()
            trace_path = run_dir / "pipeline_trace.json"
            profiler.save(trace_path)
            print(f"saved pipeline trace to {trace_path}")
            profiler.print_stats()

        writer.close()
        print("training complete")

        return terminate_flag["value"]


# -------------------------------------------------------------------
# Inference / Rollout
# -------------------------------------------------------------------


def optimize_z0(
    model: LatentStagModel,
    target_data: torch.Tensor,
    stimulus: torch.Tensor,
    latent_dims: int,
    n_steps: int = 100,
    lr: float = 1e-2,
) -> torch.Tensor:
    """
    optimize z0 to fit a short window of data with frozen model.

    args:
        model: the staggered model (must be in eval mode, params frozen)
        target_data: ground truth data (window_len, num_neurons)
        stimulus: stimulus for the window (window_len, stim_dims)
        latent_dims: dimension of latent space
        n_steps: number of optimization steps
        lr: learning rate for z0 optimization

    returns:
        optimized z0 tensor (1, latent_dims)
    """
    device = target_data.device
    window_len = target_data.shape[0]

    # learnable z0
    z0 = torch.randn(1, latent_dims, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([z0], lr=lr)

    # pre-encode stimulus (detach so it's not part of the backward graph)
    with torch.no_grad():
        stim_latent = model.stimulus_encoder(stimulus)  # (window_len, stim_latent_dim)

    for _ in range(n_steps):
        optimizer.zero_grad()

        # evolve from z0
        z = z0
        predictions = []
        for t in range(window_len):
            z = model.evolver(z, stim_latent[t:t+1])
            pred = model.decoder(z)
            predictions.append(pred)

        pred_trace = torch.cat(predictions, dim=0)  # (window_len, num_neurons)
        loss = torch.nn.functional.mse_loss(pred_trace, target_data)
        loss.backward()
        optimizer.step()

    return z0.detach()


def evolve_n_steps_latent(
    model: LatentStagModel,
    fit_target_data: torch.Tensor,
    fit_stimulus: torch.Tensor,
    rollout_stimulus: torch.Tensor,
    latent_dims: int,
    fit_steps: int = 100,
    fit_lr: float = 1e-2,
) -> torch.Tensor:
    """
    evolve the staggered model by n time steps in latent space.

    first optimizes z0 to fit a window of data (e.g., ems * time_units steps),
    then rolls out from that z0.

    args:
        model: the staggered model (decoder, evolver, stim encoder)
        fit_target_data: target data for fitting z0 (fit_window, num_neurons)
            this is x[t+1], x[t+2], ..., x[t+fit_window]
        fit_stimulus: stimulus for fitting window (fit_window, stim_dims)
            this is stim[t], stim[t+1], ..., stim[t+fit_window-1]
        rollout_stimulus: stimulus for rollout (n_steps, stim_dims)
            starts after fit window
        latent_dims: dimension of latent space
        fit_steps: number of optimization steps for z0
        fit_lr: learning rate for z0 optimization

    returns:
        predicted_trace: tensor of shape (n_steps, neurons) with predicted states
    """
    # fit z0 using the fit window
    z0 = optimize_z0(
        model=model,
        target_data=fit_target_data,
        stimulus=fit_stimulus,
        latent_dims=latent_dims,
        n_steps=fit_steps,
        lr=fit_lr,
    )

    # evolve through fit window to get to the rollout start point
    fit_window = fit_target_data.shape[0]
    fit_stim_latent = model.stimulus_encoder(fit_stimulus)
    z = z0
    for t in range(fit_window):
        z = model.evolver(z, fit_stim_latent[t:t+1])

    # now roll out from z (which is at the end of fit window)
    n_steps = rollout_stimulus.shape[0]
    rollout_stim_latent = model.stimulus_encoder(rollout_stimulus)

    latent_trace = []
    for t in range(n_steps):
        z = model.evolver(z, rollout_stim_latent[t:t+1])
        latent_trace.append(z.squeeze(0))

    latent_trace = torch.stack(latent_trace, dim=0)  # (n_steps, latent_dims)
    predicted_trace = model.decoder(latent_trace)  # (n_steps, neurons)

    return predicted_trace


# -------------------------------------------------------------------
# CLI Entry Point
# -------------------------------------------------------------------


if __name__ == "__main__":
    msg = """Usage: python latent_stag.py <expt_code> <default_config> [overrides...]

Arguments:
  expt_code       Experiment code (must match [A-Za-z0-9_]+)
  default_config  Base config file (e.g., latent_stag_20step.yaml)

Example:
  python latent_stag.py my_experiment latent_stag_20step.yaml --training.epochs 100"""

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
        print(f"error: config file not found: {default_path}")
        sys.exit(1)

    tyro_args = sys.argv[3:]
    commit_hash = get_git_commit_hash()

    run_dir = create_run_directory(
        expt_code=expt_code,
        tyro_args=tyro_args,
        model_class=StagModelParams,
        commit_hash=commit_hash,
    )

    with open(run_dir / "command_line.txt", "w") as out:
        out.write("\n".join(sys.argv))

    with open(default_path, "r") as f:
        data = yaml.safe_load(f)
    default_cfg = StagModelParams(**data)

    cfg = tyro.cli(StagModelParams, default=default_cfg, args=tyro_args)

    was_terminated = train(cfg, run_dir)

    # add a completion/termination flag
    flag_file = "terminated" if was_terminated else "complete"
    with open(run_dir / flag_file, "w"):
        pass
