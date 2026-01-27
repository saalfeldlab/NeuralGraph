"""
Latent space voltage model with optional YAML config, torch.compile support,
and reproducible training with run logging.
"""

from pathlib import Path
from typing import Callable, Iterator
from datetime import datetime
from enum import Enum, auto
import sys
import re
import time
import signal

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import yaml
import tyro
from LatentEvolution.load_flyvis import (
    NeuronData,
    FlyVisSim,
    load_connectome_graph,
    load_column_slice,
    load_metadata,
)
from LatentEvolution.training_config import DataSplit
from LatentEvolution.stimulus_utils import downsample_stimulus
from LatentEvolution.pipeline_chunk_loader import PipelineChunkLoader
from LatentEvolution.gpu_stats import GPUMonitor
from LatentEvolution.diagnostics import run_validation_diagnostics, PlotMode
from LatentEvolution.hparam_paths import create_run_directory, get_git_commit_hash
from LatentEvolution.eed_model import (
    MLP,
    MLPWithSkips,
    MLPParams,
    Evolver,
    ModelParams,
)
from LatentEvolution.training_utils import (
    LossAccumulator,
    seed_everything,
    get_device,
)
from LatentEvolution.chunk_streaming import (
    calculate_chunk_params,
    ChunkLatencyStats,
    create_zarr_loader,
)
from LatentEvolution.acquisition import (
    compute_neuron_phases,
    sample_batch_indices,
)


# -------------------------------------------------------------------
# Data Loading
# -------------------------------------------------------------------


def load_dataset(
    simulation_config: str,
    column_to_model: str,
    data_split: DataSplit,
    num_input_dims: int,
    device: torch.device,
    chunk_size: int = 65536,
    time_units: int = 1,
    training_data_path: str | None = None,
    gpu_prefetch: int = 1,
):
    """
    load dataset from zarr with chunked streaming for training data.

    training data is streamed in chunks via PipelineChunkLoader to reduce GPU memory.
    validation data is loaded directly to GPU (small enough to fit).

    args:
        simulation_config: name of simulation config (e.g., "fly_N9_62_1")
        column_to_model: column name to model (e.g., "VOLTAGE", "CALCIUM")
        data_split: DataSplit object with train/val time ranges
        num_input_dims: number of stimulus input dimensions to keep
        device: pytorch device to load data onto
        chunk_size: chunk size for streaming (default: 65536 = 64K)
        time_units: alignment constraint for chunk starts (default: 1)
        training_data_path: absolute path to data directory (overrides simulation_config)
        gpu_prefetch: number of chunks to buffer on gpu (1=no overlap, 2=double buffer
            for overlapping cpu->gpu transfer with training)

    returns:
        tuple of (chunk_loader, val_data, val_stim, neuron_data, train_total_timesteps)
    """
    if training_data_path is not None:
        data_path = training_data_path
    else:
        data_path = f"graphs_data/fly/{simulation_config}/x_list_0"
    column_idx = FlyVisSim[column_to_model].value

    # load val data directly to GPU (small enough to fit)
    val_data = torch.from_numpy(
        load_column_slice(data_path, column_idx, data_split.validation_start, data_split.validation_end)
    ).to(device)

    val_stim = torch.from_numpy(
        load_column_slice(data_path, FlyVisSim.STIMULUS.value, data_split.validation_start, data_split.validation_end, neuron_limit=num_input_dims)
    ).to(device)

    # load neuron metadata
    metadata = load_metadata(data_path)
    neuron_data = NeuronData.from_metadata(metadata)

    # create chunk loader for training data (streams from disk -> GPU)
    train_total_timesteps = data_split.train_end - data_split.train_start

    # create zarr loading function
    zarr_load_fn = create_zarr_loader(
        data_path=data_path,
        column_idx=column_idx,
        stim_column_idx=FlyVisSim.STIMULUS.value,
        num_stim_dims=num_input_dims,
    )

    # wrap to offset by train_start
    def offset_load_fn(start: int, end: int):
        return zarr_load_fn(data_split.train_start + start, data_split.train_start + end)

    # create chunk loader with 3-stage pipeline
    chunk_loader = PipelineChunkLoader(
        load_fn=offset_load_fn,
        total_timesteps=train_total_timesteps,
        chunk_size=chunk_size,
        device=device,
        prefetch=6,  # buffer 6 chunks in cpu_queue
        seed=None,  # will be set per epoch in training loop
        time_units=time_units,
        gpu_prefetch=gpu_prefetch,  # buffer chunks in gpu_queue for transfer/training overlap
    )

    return chunk_loader, val_data, val_stim, neuron_data, train_total_timesteps


def load_val_only(
    simulation_config: str,
    column_to_model: str,
    data_split: DataSplit,
    num_input_dims: int,
    device: torch.device
):
    """
    load only validation data for cross-validation (memory efficient).

    streams data directly from zarr to device memory.

    args:
        simulation_config: name of simulation config (e.g., "fly_N9_62_1")
        column_to_model: column name to model (e.g., "VOLTAGE", "CALCIUM")
        data_split: DataSplit object with train/val time ranges
        num_input_dims: number of stimulus input dimensions to keep
        device: pytorch device to load data onto

    returns:
        tuple of (val_data, val_stim)
    """
    data_path = f"graphs_data/fly/{simulation_config}/x_list_0"
    column_idx = FlyVisSim[column_to_model].value

    # load val data column slice directly to device
    val_data = torch.from_numpy(
        load_column_slice(data_path, column_idx, data_split.validation_start, data_split.validation_end)
    ).to(device)

    # load val stimulus slice directly to device
    val_stim = torch.from_numpy(
        load_column_slice(data_path, FlyVisSim.STIMULUS.value, data_split.validation_start, data_split.validation_end, neuron_limit=num_input_dims)
    ).to(device)

    return val_data, val_stim


# -------------------------------------------------------------------
# Loss Types
# -------------------------------------------------------------------


class LossType(Enum):
    """loss component types for EED model.

    train steps return dict[LossType, Tensor] for semantic access.

    benchmarking results (gpu, 256 batch, realistic encoder/decoder ops):
    - dict with enum keys: +0.5% to +1.4% overhead vs tuple
    - namedtuple: +0.8% to +2.0% overhead vs tuple
    - compilation speedup: 1.87-1.93x faster (all return types)
    - conclusion: negligible overhead, semantic access worth it
    """
    TOTAL = auto()
    RECON = auto()
    EVOLVE = auto()
    REG = auto()
    AUG_LOSS = auto()
    TV_LOSS = auto()


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
    data: torch.Tensor,
    stim: torch.Tensor,
    wmat_indices: torch.Tensor,
    wmat_indptr: torch.Tensor,
    cfg: ModelParams,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Randomly sample `batch_size` starting points and augmentation indices.

    Returns:
        Tuple of (batch_indices, selected_neurons, needed_indices)
        - batch_indices: starting time indices for the batch
        - selected_neurons: neurons selected for augmentation loss
        - needed_indices: neurons that can impact selected_neurons (based on connectome)
    """
    batch_size = cfg.training.batch_size
    time_units = cfg.training.time_units
    total_steps = time_units * cfg.training.evolve_multiple_steps
    num_neurons = data.shape[1]
    num_neurons_to_zero = cfg.training.unconnected_to_zero.num_neurons

    total_time = data.shape[0]
    if total_time <= total_steps:
        raise ValueError(
            f"Not enough time points ({total_time}) for total_steps={total_steps}"
        )
    while True:
        start_indices = torch.randint(
            low=0, high=total_time - total_steps, size=(batch_size,), device=data.device
        )

        if num_neurons_to_zero > 0:
            selected_neurons = torch.randint(
                low=0, high=num_neurons, size=(num_neurons_to_zero,), device=data.device
            )
            # based on the connectome, which neurons can actually impact the value
            # of the `selected_neurons`
            needed_indices = torch.concatenate(
                [wmat_indices[wmat_indptr[i]:wmat_indptr[i+1]] for i in selected_neurons]
            )
            # also keep the actual selected neurons
            needed_indices = torch.unique(torch.concatenate([needed_indices, selected_neurons]))
        else:
            selected_neurons = torch.empty(0, dtype=torch.long, device=data.device)
            needed_indices = torch.empty(0, dtype=torch.long, device=data.device)

        yield start_indices, selected_neurons, needed_indices


def train_step_reconstruction_only_nocompile(
        model: LatentModel,
        train_data: torch.Tensor,
        _train_stim: torch.Tensor,
        observation_indices: torch.Tensor,
        _selected_neurons: torch.Tensor,
        _needed_indices: torch.Tensor,
        cfg: ModelParams
    ) -> dict[LossType, torch.Tensor]:
    """Train encoder/decoder only with reconstruction loss."""
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

    # return only computed losses (no evolve, aug, tv during warmup)
    return {
        LossType.TOTAL: total_loss,
        LossType.RECON: recon_loss,
        LossType.REG: reg_loss,
    }


train_step_reconstruction_only = torch.compile(
    train_step_reconstruction_only_nocompile, fullgraph=True, mode="max-autotune"
)


def train_step_nocompile(
        model: LatentModel,
        train_data: torch.Tensor,
        train_stim: torch.Tensor,
        observation_indices: torch.Tensor,
        selected_neurons: torch.Tensor,
        needed_indices: torch.Tensor,
        cfg: ModelParams
    ) -> dict[LossType, torch.Tensor]:

    device=train_data.device

    # initialize loss dict
    losses: dict[LossType, torch.Tensor] = {}

    # Get loss function from config
    loss_fn = getattr(torch.nn.functional, cfg.training.loss_function)

    # regularization loss
    reg_loss = torch.tensor(0.0, device=device)
    if cfg.encoder_params.l1_reg_loss > 0.:
        for p in model.encoder.parameters():
            reg_loss += torch.abs(p).mean()*cfg.encoder_params.l1_reg_loss
    if cfg.decoder_params.l1_reg_loss > 0.:
        for p in model.decoder.parameters():
            reg_loss += torch.abs(p).mean()*cfg.decoder_params.l1_reg_loss
    if cfg.evolver_params.l1_reg_loss > 0.:
        for p in model.evolver.parameters():
            reg_loss += torch.abs(p).mean()*cfg.evolver_params.l1_reg_loss
    losses[LossType.REG] = reg_loss

    # total variation regularization on evolver updates
    tv_loss = torch.tensor(0.0, device=device)


    # b = batch size
    # N = neurons
    # L = latent dim for activity
    # Ls = latent dim for stimulus
    dt = cfg.training.time_units
    num_multiples = cfg.training.evolve_multiple_steps
    total_steps = dt * num_multiples

    # observation_indices: (b, N) - observation time for each neuron in each batch sample
    # x_t: (b, N) - neural activity at observation times
    # use advanced indexing to extract the right (time, neuron) pairs
    batch_size, num_neurons = observation_indices.shape
    neuron_indices = torch.arange(num_neurons, device=device).unsqueeze(0).expand(batch_size, num_neurons)
    x_t = train_data[observation_indices, neuron_indices]  # (b, N)
    proj_t = model.encoder(x_t)  # (b, L)

    # for stimulus: use minimum observation time in each batch (cycle start time)
    # this works for both time_aligned (all same) and staggered_random (min is cycle start)
    batch_start_times = observation_indices.min(dim=1).values  # (b,)
    stim_indices = batch_start_times.unsqueeze(0) + torch.arange(total_steps, device=device).unsqueeze(1)
    # stim_indices: (total_steps, b)
    stim_t = train_stim[stim_indices, :]
    dim_stim = train_stim.shape[1]
    dim_stim_latent = cfg.stimulus_encoder_params.num_output_dims
    # total_steps x b x Ls
    proj_stim_t_all = model.stimulus_encoder(stim_t.reshape((-1, dim_stim))).reshape((total_steps, -1, dim_stim_latent))

    # downsample stimulus based on frequency mode
    proj_stim_t = downsample_stimulus(
        proj_stim_t_all,
        tu=dt,
        num_multiples=num_multiples,
        stimulus_frequency=cfg.training.stimulus_frequency,
    )

    # reconstruction loss
    recon_t = model.decoder(proj_t)
    losses[LossType.RECON] = loss_fn(recon_t, x_t)

    # Evolve by 1 time step. This is a special case since we may opt to apply
    # a connectome constraint via data augmentation.
    if cfg.evolver_params.tv_reg_loss > 0.:
        # compute delta_z explicitly for TV norm
        delta_z = model.evolver.evolver(torch.cat([proj_t, proj_stim_t[0]], dim=1))
        tv_loss += torch.abs(delta_z).mean() * cfg.evolver_params.tv_reg_loss
        proj_t = proj_t + delta_z
    else:
        proj_t = model.evolver(proj_t, proj_stim_t[0])
    # apply connectome loss after evolving by 1 time step
    aug_loss = torch.tensor(0.0, device=device)
    if (
            cfg.training.unconnected_to_zero.num_neurons > 0 and
            cfg.training.unconnected_to_zero.loss_coeff > 0.
    ):
        # unconnected_to_zero strategy
        pred_t_plus_1 = model.decoder(proj_t)

        x_t_aug = torch.zeros_like(x_t)
        x_t_aug[:, needed_indices] = x_t[:, needed_indices]
        proj_t_aug = model.evolver(model.encoder(x_t_aug), proj_stim_t[0])
        pred_t_plus_1_aug = model.decoder(proj_t_aug)
        aug_loss += cfg.training.unconnected_to_zero.loss_coeff * loss_fn(pred_t_plus_1_aug[:, selected_neurons], pred_t_plus_1[:, selected_neurons])
    losses[LossType.AUG_LOSS] = aug_loss

    # evolve for remaining dt-1 time steps (first window)
    evolve_loss = torch.tensor(0.0, device=device)
    if cfg.evolver_params.tv_reg_loss > 0.:
        for i in range(1, dt):
            delta_z = model.evolver.evolver(torch.cat([proj_t, proj_stim_t[i]], dim=1))
            tv_loss += torch.abs(delta_z).mean() * cfg.evolver_params.tv_reg_loss
            proj_t = proj_t + delta_z
    else:
        for i in range(1, dt):
            proj_t = model.evolver(proj_t, proj_stim_t[i])

    # loss at first multiple (dt)
    pred_t_plus_dt = model.decoder(proj_t)
    # target at t+dt: observation_indices + dt
    target_indices_dt = observation_indices + dt
    x_t_plus_dt = train_data[target_indices_dt, neuron_indices]  # (b, N)
    evolve_loss = evolve_loss + loss_fn(pred_t_plus_dt, x_t_plus_dt)

    # additional multiples (2, 3, ..., num_multiples)
    if cfg.evolver_params.tv_reg_loss > 0.:
        for m in range(2, num_multiples + 1):
            # evolve dt more steps
            start_idx = (m - 1) * dt
            for i in range(dt):
                delta_z = model.evolver.evolver(torch.cat([proj_t, proj_stim_t[start_idx + i]], dim=1))
                tv_loss += torch.abs(delta_z).mean() * cfg.evolver_params.tv_reg_loss
                proj_t = proj_t + delta_z
            # loss at this multiple
            pred = model.decoder(proj_t)
            # target at t + m*dt
            target_indices_m = observation_indices + m * dt
            x_target = train_data[target_indices_m, neuron_indices]  # (b, N)
            evolve_loss = evolve_loss + loss_fn(pred, x_target)
    else:
        for m in range(2, num_multiples + 1):
            # evolve dt more steps
            start_idx = (m - 1) * dt
            for i in range(dt):
                proj_t = model.evolver(proj_t, proj_stim_t[start_idx + i])
            # loss at this multiple
            pred = model.decoder(proj_t)
            # target at t + m*dt
            target_indices_m = observation_indices + m * dt
            x_target = train_data[target_indices_m, neuron_indices]  # (b, N)
            evolve_loss = evolve_loss + loss_fn(pred, x_target)

    losses[LossType.EVOLVE] = evolve_loss
    losses[LossType.TV_LOSS] = tv_loss

    # compute total loss
    losses[LossType.TOTAL] = sum(losses.values())

    return losses

train_step = torch.compile(train_step_nocompile, fullgraph=True, mode="reduce-overhead")

# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------


def train(cfg: ModelParams, run_dir: Path):
    """Configurable training loop with train/val/test evaluation."""
    # --- Reproducibility ---
    seed_everything(cfg.training.seed)

    # --- Signal handling for graceful termination ---
    terminate_flag = {"value": False}

    def handle_sigusr2(signum, frame):
        terminate_flag["value"] = True
        print("\nSIGUSR2 received - will terminate after current epoch")

    signal.signal(signal.SIGUSR2, handle_sigusr2)

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
        print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")
        model.train()
        OptimizerClass = getattr(torch.optim, cfg.training.optimizer)
        optimizer = OptimizerClass(model.parameters(), lr=cfg.training.learning_rate)
        train_step_fn: Callable = globals()[cfg.training.train_step]

        # --- TensorBoard setup ---
        writer = SummaryWriter(log_dir=run_dir)
        print(f"tensorboard --logdir={run_dir} --samples_per_plugin=images=1000")

        # Log model graph to TensorBoard
        dummy_x_t = torch.randn(cfg.training.batch_size, cfg.num_neurons).to(device)
        dummy_stim_t = torch.randn(cfg.training.batch_size, cfg.stimulus_encoder_params.num_input_dims).to(device)
        writer.add_graph(model, (dummy_x_t, dummy_stim_t))
        print("Logged model graph to TensorBoard")

        # --- Load data ---
        chunk_loader, val_data, val_stim, neuron_data, train_total_timesteps = load_dataset(
            simulation_config=cfg.training.simulation_config,
            column_to_model=cfg.training.column_to_model,
            data_split=cfg.training.data_split,
            num_input_dims=cfg.stimulus_encoder_params.num_input_dims,
            device=device,
            chunk_size=65536,  # 64K timesteps per chunk
            time_units=cfg.training.time_units,
            training_data_path=cfg.training.training_data_path,
        )

        print(f"chunked streaming: train {train_total_timesteps} timesteps (chunked), val {val_data.shape}")

        # Load connectome weights
        wmat = load_connectome_graph(f"graphs_data/fly/{cfg.training.simulation_config}").to(device)
        wmat_indices = wmat.col_indices()
        wmat_indptr = wmat.crow_indices()

        total_steps = cfg.training.time_units * cfg.training.evolve_multiple_steps
        metrics = {
            "val_loss_constant_model": torch.nn.functional.mse_loss(
                val_data[: -total_steps], val_data[total_steps:]
            ).item(),
        }
        print(f"constant model baseline (val): {metrics['val_loss_constant_model']:.4e}")

        # log baseline metrics to tensorboard
        writer.add_scalar("Baseline/val_loss_constant_model", metrics["val_loss_constant_model"], 0)

        # --- Load cross-validation datasets ---
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
            print(f"Loaded cross-validation dataset: {cv_name} (val shape: {cv_val_data.shape})")

        # --- Calculate chunking parameters ---
        chunk_size = 65536
        chunks_per_epoch, batches_per_chunk, batches_per_epoch = calculate_chunk_params(
            total_timesteps=train_total_timesteps,
            chunk_size=chunk_size,
            batch_size=cfg.training.batch_size,
            data_passes_per_epoch=cfg.training.data_passes_per_epoch,
        )
        print(f"chunking: {chunks_per_epoch} chunks/epoch, {batches_per_chunk} batches/chunk, {batches_per_epoch} total batches/epoch")

        # --- Compute neuron phases for acquisition mode ---
        neuron_phases = compute_neuron_phases(
            num_neurons=cfg.num_neurons,
            time_units=cfg.training.time_units,
            acquisition_mode=cfg.training.acquisition_mode,
            device=device,
        )
        if neuron_phases is not None:
            print(f"acquisition mode: {cfg.training.acquisition_mode.mode}, computed phases for {cfg.num_neurons} neurons")
        else:
            print("acquisition mode: all_time_points (no phase restrictions)")

        # --- Reconstruction warmup loop ---
        recon_warmup_epochs = cfg.training.reconstruction_warmup_epochs
        if recon_warmup_epochs > 0:
            print(f"\n=== reconstruction warmup: {recon_warmup_epochs} epochs ===")
            model.evolver.requires_grad_(False)

            for warmup_epoch in range(recon_warmup_epochs):
                warmup_epoch_start = datetime.now()
                warmup_losses = LossAccumulator(LossType)

                # start loading chunks for this warmup epoch
                chunk_loader.start_epoch(num_chunks=chunks_per_epoch)

                for _ in range(chunks_per_epoch):
                    chunk_start, chunk_data, chunk_stim = chunk_loader.get_next_chunk()
                    if chunk_data is None:
                        break

                    # train on batches within this chunk
                    for _ in range(batches_per_chunk):
                        optimizer.zero_grad()

                        # sample batch observation indices using acquisition mode
                        observation_indices = sample_batch_indices(
                            chunk_size=chunk_data.shape[0],
                            total_steps=total_steps,
                            time_units=cfg.training.time_units,
                            batch_size=cfg.training.batch_size,
                            num_neurons=cfg.num_neurons,
                            neuron_phases=neuron_phases,
                            device=device,
                        )

                        # no augmentation during warmup
                        selected_neurons = torch.empty(0, dtype=torch.long, device=device)
                        needed_indices = torch.empty(0, dtype=torch.long, device=device)

                        # use nocompile version for warmup
                        losses = train_step_reconstruction_only_nocompile(
                            model, chunk_data, chunk_stim, observation_indices,
                            selected_neurons, needed_indices, cfg
                        )
                        losses[LossType.TOTAL].backward()
                        optimizer.step()
                        warmup_losses.accumulate(losses)

                warmup_epoch_duration = (datetime.now() - warmup_epoch_start).total_seconds()
                mean_warmup = warmup_losses.mean()
                print(f"warmup {warmup_epoch+1}/{recon_warmup_epochs} | recon loss: {mean_warmup[LossType.RECON]:.4e} | duration: {warmup_epoch_duration:.2f}s")

                # log to tensorboard (only computed losses during warmup)
                for loss_type, loss_value in mean_warmup.items():
                    writer.add_scalar(f"ReconWarmup/{loss_type.name.lower()}", loss_value, warmup_epoch)
                writer.add_scalar("ReconWarmup/epoch_duration", warmup_epoch_duration, warmup_epoch)

            model.evolver.requires_grad_(True)
            print("=== reconstruction warmup complete ===\n")

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

        # --- Initialize latency tracking ---
        latency_stats = ChunkLatencyStats()

        # --- Early stopping tracking ---
        early_stop_history = []  # track max intervening mse over epochs
        best_early_stop_metric = float('inf')

        # --- Epoch loop ---
        for epoch in range(cfg.training.epochs):
            epoch_start = datetime.now()
            gpu_monitor.sample_epoch_start()
            losses = LossAccumulator(LossType)

            # start loading chunks for this epoch
            chunk_loader.start_epoch(num_chunks=chunks_per_epoch)

            # ---- Training phase (chunked iteration) ----
            for _ in range(chunks_per_epoch):
                # get next chunk (blocks until ready, overlaps with previous training)
                get_start = time.time()
                chunk_start, chunk_data, chunk_stim = chunk_loader.get_next_chunk()
                latency_stats.record_chunk_get(time.time() - get_start)

                if chunk_data is None:
                    break  # end of epoch

                # train on batches within this chunk
                for batch_in_chunk in range(batches_per_chunk):
                    optimizer.zero_grad()

                    # sample batch observation indices using acquisition mode
                    observation_indices = sample_batch_indices(
                        chunk_size=chunk_data.shape[0],
                        total_steps=total_steps,
                        time_units=cfg.training.time_units,
                        batch_size=cfg.training.batch_size,
                        num_neurons=cfg.num_neurons,
                        neuron_phases=neuron_phases,
                        device=device,
                    )

                    # handle unconnected_to_zero augmentation
                    if cfg.training.unconnected_to_zero.num_neurons > 0:
                        num_neurons_to_zero = cfg.training.unconnected_to_zero.num_neurons
                        selected_neurons = torch.randint(
                            0, cfg.num_neurons, size=(num_neurons_to_zero,), device=device
                        )
                        # find neurons that can impact selected_neurons via connectome
                        needed_indices = torch.concatenate([
                            wmat_indices[wmat_indptr[i]:wmat_indptr[i+1]]
                            for i in selected_neurons
                        ])
                        needed_indices = torch.unique(
                            torch.concatenate([needed_indices, selected_neurons])
                        )
                    else:
                        selected_neurons = torch.empty(0, dtype=torch.long, device=device)
                        needed_indices = torch.empty(0, dtype=torch.long, device=device)

                    # training step (timing for latency tracking)
                    forward_start = time.time()
                    loss_dict = train_step_fn(
                        model, chunk_data, chunk_stim, observation_indices, selected_neurons, needed_indices, cfg
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

                    # sample timing every 10 batches
                    if batch_in_chunk % 10 == 0:
                        latency_stats.record_batch_times(forward_time, backward_time, step_time)

            mean_losses = losses.mean()

            epoch_end = datetime.now()
            gpu_monitor.sample_epoch_end()
            epoch_duration = (epoch_end - epoch_start).total_seconds()
            epoch_durations.append(epoch_duration)
            total_elapsed = (epoch_end - training_start).total_seconds()

            print(
                f"Epoch {epoch+1}/{cfg.training.epochs} | "
                f"Train Loss: {mean_losses[LossType.TOTAL]:.4e} | "
                f"Duration: {epoch_duration:.2f}s (Total: {total_elapsed:.1f}s)"
            )

            # log to tensorboard (programmatically iterate over all losses)
            for loss_type, loss_value in mean_losses.items():
                writer.add_scalar(f"Loss/train_{loss_type.name.lower()}", loss_value, epoch)
            writer.add_scalar("Time/epoch_duration", epoch_duration, epoch)
            writer.add_scalar("Time/total_elapsed", total_elapsed, epoch)

            # log latency stats
            latency_summary = latency_stats.get_summary()
            writer.add_scalar("Latency/chunk_get_mean_ms", latency_summary["chunk_get_mean_ms"], epoch)
            writer.add_scalar("Latency/chunk_get_max_ms", latency_summary["chunk_get_max_ms"], epoch)
            writer.add_scalar("Latency/batch_forward_mean_ms", latency_summary["batch_forward_mean_ms"], epoch)
            writer.add_scalar("Latency/batch_backward_mean_ms", latency_summary["batch_backward_mean_ms"], epoch)
            writer.add_scalar("Latency/batch_step_mean_ms", latency_summary["batch_step_mean_ms"], epoch)

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
                    writer.add_scalar(f"Val/{metric_name}", metric_value, epoch)

                # Log diagnostic figures to TensorBoard
                for fig_name, fig in diagnostic_figures.items():
                    writer.add_figure(f"Val/{fig_name}", fig, epoch)

                # Log diagnostics duration
                writer.add_scalar("Time/diagnostics_duration", diagnostics_duration, epoch)
                print(f"  Diagnostics completed in {diagnostics_duration:.2f}s")

                # Run cross-validation diagnostics
                for cv_name, (cv_val_data, cv_val_stim) in cv_datasets.items():
                    cv_start = datetime.now()
                    cv_metrics, cv_figures = run_validation_diagnostics(
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
                    print(f"  Cross-validation ({cv_name}) completed in {cv_duration:.2f}s")

                # --- Early stopping check ---
                if cfg.training.early_stop_intervening_mse and cfg.training.time_units > 1:
                    # get max intervening mse metric and first divergence
                    max_intervening_key = None
                    first_divergence_key = None
                    for key in diagnostic_metrics:
                        if "max_intervening_0_to_tu" in key:
                            max_intervening_key = key
                        if "first_divergence_step" in key:
                            first_divergence_key = key

                    if max_intervening_key and first_divergence_key:
                        max_intervening_mse = diagnostic_metrics[max_intervening_key]
                        first_divergence = diagnostic_metrics[first_divergence_key]

                        # track history
                        early_stop_history.append(max_intervening_mse)
                        if max_intervening_mse < best_early_stop_metric:
                            best_early_stop_metric = max_intervening_mse

                        writer.add_scalar("EarlyStop/max_intervening_mse", max_intervening_mse, epoch)
                        writer.add_scalar("EarlyStop/best_max_intervening_mse", best_early_stop_metric, epoch)
                        writer.add_scalar("EarlyStop/first_divergence", first_divergence, epoch)

                        # check early stopping criterion
                        if (
                            first_divergence >= cfg.training.early_stop_min_divergence
                            and len(early_stop_history) >= cfg.training.early_stop_patience_epochs
                        ):
                            # check if metric hasn't improved by 10% in last patience epochs
                            patience_window = early_stop_history[-cfg.training.early_stop_patience_epochs:]
                            best_in_window = min(patience_window)
                            current_metric = patience_window[-1]

                            # 10% improvement threshold
                            improvement_threshold = best_in_window * 0.9

                            if current_metric >= improvement_threshold:
                                print(f"\n=== Early stopping triggered at epoch {epoch+1} ===")
                                print(f"  first divergence: {first_divergence}")
                                print(f"  max intervening mse: {max_intervening_mse:.4e}")
                                print(f"  best in last {cfg.training.early_stop_patience_epochs} epochs: {best_in_window:.4e}")
                                print(f"  no 10% improvement in {cfg.training.early_stop_patience_epochs} epochs")
                                print("=== stopping training early ===\n")

                                # save early stop checkpoint
                                early_stop_path = checkpoint_dir / "checkpoint_early_stop.pt"
                                torch.save(model.state_dict(), early_stop_path)
                                print(f"  → Saved early stop checkpoint at epoch {epoch+1}")

                                # break out of epoch loop
                                break

                model.train()

            # --- Checkpointing ---
            # Save periodic checkpoint
            if cfg.training.save_checkpoint_every_n_epochs > 0 and (epoch + 1) % cfg.training.save_checkpoint_every_n_epochs == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1:04d}.pt"
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  → Saved periodic checkpoint at epoch {epoch+1}")

            # Save latest checkpoint (overwrite each epoch)
            latest_checkpoint_path = checkpoint_dir / "checkpoint_latest.pt"
            torch.save(model.state_dict(), latest_checkpoint_path)

            # Step profiler if enabled
            if profiler is not None:
                profiler.step()

            # Check for graceful termination signal
            if terminate_flag["value"]:
                print(f"\n=== graceful termination at epoch {epoch+1} ===")
                break

        # --- Cleanup profiler ---
        if profiler is not None:
            profiler.__exit__(None, None, None)
            print("Profiler traces saved successfully")

        # Calculate training statistics
        training_end = datetime.now()
        total_training_duration = (training_end - training_start).total_seconds()
        avg_epoch_duration = sum(epoch_durations) / len(epoch_durations) if epoch_durations else 0.0

        # Collect GPU metrics
        gpu_metrics = gpu_monitor.get_metrics()

        metrics.update(
            {
                    "final_train_loss": mean_losses[LossType.TOTAL],
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

        # Run cross-validation diagnostics (using pre-loaded datasets)
        if cv_datasets:
            print("\n=== Running Cross-Dataset Validation ===")

            for cv_name, (cv_val_data, cv_val_stim) in cv_datasets.items():
                print(f"\nEvaluating on {cv_name}...")

                cv_out_dir = run_dir / "cross_validation" / cv_name
                cv_metrics, cv_figures = run_validation_diagnostics(
                    run_dir=cv_out_dir,
                    val_data=cv_val_data,
                    neuron_data=neuron_data,
                    val_stim=cv_val_stim,
                    model=model,
                    config=cfg,
                    plot_mode=PlotMode.POST_RUN
                )
                print(f"Saved cross-validation figures to {cv_out_dir}")

                # Log cross-validation scalar metrics to TensorBoard
                for metric_name, metric_value in cv_metrics.items():
                    writer.add_scalar(f"CrossVal/{cv_name}/{metric_name}", metric_value, cfg.training.epochs)
                print(f"Logged {len(cv_metrics)} cross-validation scalar metrics to TensorBoard")

                # Log figures to TensorBoard
                for fig_name, fig in cv_figures.items():
                    writer.add_figure(f"CrossVal/{cv_name}/{fig_name}", fig, cfg.training.epochs)
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
        # omit logging of profiling/cross-validation info, and filter to scalar types only
        hparams = {
            k: v for k, v in cfg.flatten().items() if not (
                k.startswith("profiling") or k.startswith("cross_validation_configs")
            ) and isinstance(v, (int, float, str, bool))
        }
        metric_dict = {
            "hparam/final_train_loss": metrics["final_train_loss"],
        }
        writer.add_hparams(hparams, metric_dict)
        print("Logged hyperparameters to TensorBoard")

        # print final latency statistics
        print("\n=== final chunked streaming latency stats ===")
        latency_stats.print_summary()

        # cleanup chunk loader
        chunk_loader.cleanup()
        print("chunk loader cleanup complete")

        # close tensorboard writer
        writer.close()
        print("tensorboard logging completed")

        return terminate_flag["value"]


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

    was_terminated = train(cfg, run_dir)

    # add a completion/termination flag
    flag_file = "terminated" if was_terminated else "complete"
    with open(run_dir / flag_file, "w"):
        pass


