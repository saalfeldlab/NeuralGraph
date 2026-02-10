"""
latent zapbench: per-condition pre-interpolated data on GPU.

data flow:
  startup: load sparse -> interpolate on GPU -> store on CPU (headroom)
           -> after all conditions loaded, transfer all to GPU
  training: sample on GPU -> forward/backward

usage:
  python latent_zapbench.py <expt_code> [--overrides]

  example:
    python latent_zapbench.py my_expt --train.epochs 50 --train.learning_rate 1e-4
"""

import logging
import os
import queue
import random
import signal
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
import yaml
from torch.utils.tensorboard import SummaryWriter

# configure logging with HH:MM:SS timestamp
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def seed_everything(seed: int = 42):
    """seed all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

from LatentEvolution.zapbench import (
    load_sparse_activity,
    interpolate_sparse_compiled,  # compiled version needed for memory efficiency
)
from LatentEvolution.zapbench_config import (
    DataConfig,
    DataSplit,
    ModelConfig,
    TrainConfig,
)
from LatentEvolution.zapbench_model import EEDModel
from LatentEvolution.hparam_paths import create_run_directory, get_git_commit_hash

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# combined config for CLI
# ---------------------------------------------------------------------------


class ZapbenchConfig(BaseModel):
    """combined config for zapbench training."""
    data: DataConfig = Field(default_factory=DataConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    model: ModelConfig = Field(default_factory=lambda: ModelConfig(num_neurons=0))  # set by data loader
    split: DataSplit = Field(default_factory=DataSplit)

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# observation mask
# ---------------------------------------------------------------------------


def build_obs_mask(
    obs_times: torch.Tensor,
    num_bins: int,
    device: torch.device,
) -> torch.Tensor:
    """build observation mask from sparse observation times.

    args:
        obs_times: (N, K) sorted bin indices per neuron
        num_bins: total number of time bins (T)
        device: target device for computation

    returns:
        obs_mask: (T, N) bool tensor, True where neuron was observed
    """
    N, K = obs_times.shape

    # create neuron indices: (N, K) where each row is [n, n, n, ...]
    neuron_indices = torch.arange(N, device=device).unsqueeze(1).expand(N, K)

    # scatter True into mask at observation positions
    obs_mask = torch.zeros(num_bins, N, dtype=torch.bool, device=device)
    obs_times_dev = obs_times.to(device)
    obs_mask[obs_times_dev, neuron_indices] = True

    return obs_mask


# ---------------------------------------------------------------------------
# per-condition data (for val/test on CPU)
# ---------------------------------------------------------------------------


@dataclass
class ConditionData:
    """per-condition data on CPU for validation/test.

    kept separate by condition for diagnostics (no concatenation).
    """
    activity: torch.Tensor   # (T, N) float16 on CPU
    obs_mask: torch.Tensor   # (T, N) bool on CPU
    valid_start: int         # first valid sampling index (after first obs)
    valid_end: int           # last valid sampling index (before last obs - fitting_window)
    name: str                # condition name
    num_neurons: int


# ---------------------------------------------------------------------------
# concatenated training data (for training on GPU)
# ---------------------------------------------------------------------------


@dataclass
class ConcatTrainingData:
    """concatenated training data for all conditions on GPU.

    activity stored as float16 to save memory, converted to float32 when sampled.
    obs_mask stored as bool, used to compute loss only on real observations.
    all conditions are concatenated along the time axis into a single tensor.
    valid_starts and valid_ends store global indices into this tensor.
    """
    activity: torch.Tensor      # (total_T, N) float16 on GPU
    obs_mask: torch.Tensor      # (total_T, N) bool on GPU
    valid_starts: torch.Tensor  # (num_conds,) global valid start indices
    valid_ends: torch.Tensor    # (num_conds,) global valid end indices
    weights: torch.Tensor       # (num_conds,) sampling weights per condition
    num_neurons: int
    num_conds: int

    @property
    def total_weight(self) -> int:
        """total number of valid start positions across all conditions."""
        return int(self.weights.sum().item())

    def sample_batch(
        self, batch_size: int, time_offsets: torch.Tensor, rng: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """sample batch on GPU - fully vectorized, no CPU sync.

        args:
            batch_size: number of samples to draw.
            time_offsets: precomputed arange(fitting_window) on GPU.
            rng: random number generator on GPU.

        returns:
            batch: (B, fitting_window, N) float32 GPU tensor.
            mask: (B, fitting_window, N) bool GPU tensor, True at real observations.
        """
        device = self.activity.device

        # sample which condition each batch element comes from: (B,)
        cond_indices = torch.multinomial(
            self.weights, batch_size, replacement=True, generator=rng,
        )

        # look up valid range for each sample's condition: (B,)
        valid_starts = self.valid_starts[cond_indices]
        valid_ends = self.valid_ends[cond_indices]

        # sample start positions using uniform scaling
        # start = valid_start + floor(u * (valid_end - valid_start))
        u = torch.rand(batch_size, device=device, generator=rng)
        starts = valid_starts + (u * (valid_ends - valid_starts).float()).long()

        # compute gather indices: (B, T)
        gather_indices = starts[:, None] + time_offsets

        # gather activity (float16 -> float32) and mask (bool)
        batch = self.activity[gather_indices].float()  # (B, T, N) float32
        mask = self.obs_mask[gather_indices]           # (B, T, N) bool

        return batch, mask


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------


def _interpolate_and_mask(
    obs_times: torch.Tensor,
    obs_vals: torch.Tensor,
    counts: torch.Tensor,
    num_bins: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """interpolate sparse data and build mask on GPU, return on CPU.

    args:
        obs_times: (N, K) sparse observation times on CPU
        obs_vals: (N, K) sparse values on CPU
        counts: (N,) counts per neuron on CPU
        num_bins: number of output bins
        device: GPU device for computation

    returns:
        activity: (T, N) float16 on CPU
        obs_mask: (T, N) bool on CPU
    """
    # transfer to GPU
    obs_times_gpu = obs_times.to(device)
    obs_vals_gpu = obs_vals.to(device)
    counts_gpu = counts.to(device)

    # interpolate on GPU
    activity_gpu = interpolate_sparse_compiled(
        obs_times_gpu, obs_vals_gpu, counts_gpu, num_bins,
    )

    # build observation mask on GPU
    obs_mask_gpu = build_obs_mask(obs_times, num_bins, device)

    # move to CPU as float16
    activity_cpu = activity_gpu.half().cpu()
    obs_mask_cpu = obs_mask_gpu.cpu()

    # free GPU memory
    del obs_times_gpu, obs_vals_gpu, counts_gpu, activity_gpu, obs_mask_gpu
    torch.cuda.empty_cache()

    return activity_cpu, obs_mask_cpu


def load_all_data(
    traces_path: str,
    ephys_path: str,
    bin_size_ms: float,
    split: DataSplit,
    fitting_window: int,
    device: torch.device,
) -> tuple[ConcatTrainingData, list[ConditionData], list[ConditionData]]:
    """load all data: train on GPU, val/test on CPU per-condition.

    loads each condition once from zarr (efficient I/O), splits into
    train/val/test, concatenates train for GPU, keeps val/test separate on CPU.

    args:
        traces_path: path to traces zarr array.
        ephys_path: path to ephys.zarr directory.
        bin_size_ms: bin width in milliseconds.
        split: data split configuration.
        fitting_window: number of time steps for training window.
        device: GPU device for interpolation and training data.

    returns:
        train_data: ConcatTrainingData on GPU
        val_data: list of ConditionData on CPU (one per condition with val split)
        test_data: list of ConditionData on CPU (one per condition with test split)
    """
    from LatentEvolution.zapbench_config import CONDITIONS

    # accumulators for train (will be concatenated)
    train_activities: list[torch.Tensor] = []
    train_masks: list[torch.Tensor] = []
    train_valid_starts: list[int] = []
    train_valid_ends: list[int] = []
    train_weights: list[int] = []
    train_offset = 0

    # accumulators for val/test (kept separate per condition)
    val_data: list[ConditionData] = []
    test_data: list[ConditionData] = []

    num_neurons = None

    for cond_split in split.conditions:
        cond = next(c for c in CONDITIONS if c.name == cond_split.condition_name)
        cond_name = cond.name
        padded_start = cond.offset[0] + cond.padding
        padded_end = cond.offset[1] - cond.padding

        # load entire condition from zarr (one I/O per condition)
        log.info(f"  loading {cond_name}...")
        obs_times, obs_vals, _counts, _num_bins = load_sparse_activity(
            traces_path, ephys_path, bin_size_ms,
            time_slice=slice(padded_start, padded_end),
        )
        num_neurons = obs_vals.shape[0]

        # process each split (train/val/test) from the loaded data
        for split_name, split_range in [
            ("train", cond_split.train),
            ("val", cond_split.val),
            ("test", cond_split.test),
        ]:
            if split_range is None:
                continue

            frame_start, frame_end = split_range

            # slice the sparse data for this split
            split_obs_times = obs_times[:, frame_start:frame_end].contiguous()
            split_obs_vals = obs_vals[:, frame_start:frame_end].contiguous()
            split_counts = torch.full((num_neurons,), frame_end - frame_start, dtype=torch.long)

            # recompute bin indices relative to this split's start
            # (subtract minimum to start bins at 0)
            min_bin = split_obs_times.min().item()
            split_obs_times = split_obs_times - min_bin
            split_num_bins = int(split_obs_times.max().item()) + 1

            # interpolate and build mask
            activity, obs_mask = _interpolate_and_mask(
                split_obs_times, split_obs_vals, split_counts, split_num_bins, device,
            )

            # compute valid sampling range
            first_obs = int(split_obs_times[:, 0].max().item()) + 1
            last_obs = int(split_obs_times[:, -1].min().item())
            valid_start = first_obs
            valid_end = max(0, last_obs - fitting_window + 1)

            T = activity.shape[0]
            log.info(f"    {split_name}: T={T}, valid=[{valid_start}, {valid_end})")

            if split_name == "train":
                # accumulate for concatenation
                train_activities.append(activity)
                train_masks.append(obs_mask)
                train_valid_starts.append(train_offset + valid_start)
                train_valid_ends.append(train_offset + valid_end)
                train_weights.append(max(0, valid_end - valid_start))
                train_offset += T
            else:
                # store per-condition on CPU
                cond_data = ConditionData(
                    activity=activity,
                    obs_mask=obs_mask,
                    valid_start=valid_start,
                    valid_end=valid_end,
                    name=cond_name,
                    num_neurons=num_neurons,
                )
                if split_name == "val":
                    val_data.append(cond_data)
                else:
                    test_data.append(cond_data)

    # concatenate train data and transfer to GPU
    log.info("  concatenating train data and transferring to GPU...")
    all_train_activity = torch.cat(train_activities, dim=0)
    all_train_mask = torch.cat(train_masks, dim=0)
    del train_activities, train_masks

    # compute stats
    activity_gb = all_train_activity.numel() * all_train_activity.element_size() / 1e9
    mask_gb = all_train_mask.numel() * all_train_mask.element_size() / 1e9
    obs_density = all_train_mask.sum().item() / all_train_mask.numel()
    log.info(f"  train: {activity_gb:.2f} GB (float16), mask: {mask_gb:.2f} GB (bool)")
    log.info(f"  observation density: {obs_density:.2%}")

    # transfer to GPU
    train_activity_gpu = all_train_activity.to(device)
    del all_train_activity
    train_mask_gpu = all_train_mask.to(device)
    del all_train_mask

    assert num_neurons is not None
    train_data = ConcatTrainingData(
        activity=train_activity_gpu,
        obs_mask=train_mask_gpu,
        valid_starts=torch.tensor(train_valid_starts, device=device, dtype=torch.long),
        valid_ends=torch.tensor(train_valid_ends, device=device, dtype=torch.long),
        weights=torch.tensor(train_weights, device=device, dtype=torch.float),
        num_neurons=num_neurons,
        num_conds=len([c for c in split.conditions if c.train is not None]),
    )

    # log val/test stats
    val_total = sum(c.activity.shape[0] for c in val_data)
    test_total = sum(c.activity.shape[0] for c in test_data)
    log.info(f"  val: {len(val_data)} conditions, {val_total} total bins (CPU)")
    log.info(f"  test: {len(test_data)} conditions, {test_total} total bins (CPU)")

    return train_data, val_data, test_data


# ---------------------------------------------------------------------------
# train step (similar to latent_stag_interp.py)
# ---------------------------------------------------------------------------

@torch.compile(mode="reduce-overhead", fullgraph=True)
def train_step(
    model: EEDModel,
    batch: torch.Tensor,    # (B, T, N)
    obs_mask: torch.Tensor, # (B, T, N) bool
) -> torch.Tensor:
    """training step: encode, evolve, decode, compute masked loss.

    follows latent_stag_interp.py pattern:
    - encode initial state
    - loop through timesteps: decode, compute loss only on observed values, evolve

    args:
        model: EED model
        batch: (B, T, N) neural activity sequence
        obs_mask: (B, T, N) bool, True where values are real observations

    returns:
        total loss (scalar)
    """
    device = batch.device
    T = batch.shape[1]

    # encode initial state
    z = model.encode(batch[:, 0, :])  # (B, L)

    loss = torch.tensor(0.0, device=device)
    for t in range(T):
        # decode current latent
        x_pred = model.decode(z)

        # masked MSE: only count observed values
        error_sq = (x_pred - batch[:, t, :]) ** 2  # (B, N)
        mask_t = obs_mask[:, t, :]                  # (B, N)
        # sum over observed values, normalize by count
        loss = loss + (error_sq * mask_t).sum() / mask_t.sum().clamp(min=1)

        # evolve to next latent (no stimulus)
        z = model.evolve(z)

    return loss


# ---------------------------------------------------------------------------
# CPU validation (runs in background thread)
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """result from background validation."""
    epoch: int
    mean_mse: float
    mean_mae: float
    per_condition_mse: dict  # {name: (rollout_len,) mse array}
    per_condition_mae: dict  # {name: (rollout_len,) mae array}


def run_validation_cpu(
    model_cfg: "ModelConfig",
    state_dict: dict,
    val_conditions: list[ConditionData],
    result_queue: "queue.Queue[ValidationResult]",
    epoch: int,
    max_rollout_steps: int | None = None,
) -> None:
    """run validation on CPU, put result in queue.

    args:
        model_cfg: model configuration for creating CPU model.
        state_dict: model weights (already on CPU).
        val_conditions: list of ConditionData on CPU.
        result_queue: queue to put results.
        epoch: which epoch this validation corresponds to.
        max_rollout_steps: cap rollout length (None = rollout to end).
    """
    with torch.profiler.record_function(f"validation_epoch_{epoch}"):

        # create model on CPU, load weights
        with torch.profiler.record_function("validation_model_setup"):
            model = EEDModel(model_cfg).cpu()
            model.load_state_dict(state_dict)
            model.eval()

        results_mse: dict[str, np.ndarray] = {}
        results_mae: dict[str, np.ndarray] = {}

        with torch.no_grad():
            for cond in val_conditions:
                with torch.profiler.record_function(f"validation_{cond.name}"):
                    T = cond.activity.shape[0]
                    start = cond.valid_start
                    end = T if max_rollout_steps is None else min(T, start + max_rollout_steps)
                    rollout_len = end - start

                    if rollout_len <= 0:
                        continue

                    # get ground truth and mask (float16 -> float32)
                    gt = cond.activity[start:end].float()  # (rollout_len, N)
                    mask = cond.obs_mask[start:end]         # (rollout_len, N)

                    # encode initial state
                    z = model.encode(gt[0:1])  # (1, L)

                    mses = []
                    maes = []
                    for t in range(rollout_len):
                        x_pred = model.decode(z)  # (1, N)

                        # masked MSE and MAE
                        err = x_pred[0] - gt[t]
                        mask_t = mask[t]
                        count = mask_t.sum().clamp(min=1)
                        mse_t = ((err ** 2) * mask_t).sum() / count
                        mae_t = (err.abs() * mask_t).sum() / count
                        mses.append(mse_t.item())
                        maes.append(mae_t.item())

                        z = model.evolve(z)

                    results_mse[cond.name] = np.array(mses)
                    results_mae[cond.name] = np.array(maes)

        mean_mse = float(np.mean([m.mean() for m in results_mse.values()])) if results_mse else 0.0
        mean_mae = float(np.mean([m.mean() for m in results_mae.values()])) if results_mae else 0.0

        result_queue.put(ValidationResult(
            epoch=epoch,
            mean_mse=mean_mse,
            mean_mae=mean_mae,
            per_condition_mse=results_mse,
            per_condition_mae=results_mae,
        ))


def log_validation_result(result: ValidationResult, writer: SummaryWriter | None = None) -> None:
    """log validation result as a table and to tensorboard."""
    log.info(f"validation (epoch {result.epoch}):")
    log.info(f"  {'condition':<12} {'steps':>6} {'mean_mse':>10} {'mean_mae':>10}")
    log.info(f"  {'-'*12} {'-'*6} {'-'*10} {'-'*10}")
    for name in sorted(result.per_condition_mse.keys()):
        mses = result.per_condition_mse[name]
        maes = result.per_condition_mae[name]
        log.info(f"  {name:<12} {len(mses):>6} {mses.mean():>10.4f} {maes.mean():>10.4f}")
        # tensorboard logging per condition
        if writer is not None:
            writer.add_scalar(f"val/{name}/mean_mse", mses.mean(), result.epoch)
            writer.add_scalar(f"val/{name}/final_mse", mses[-1], result.epoch)
            writer.add_scalar(f"val/{name}/mean_mae", maes.mean(), result.epoch)
            writer.add_scalar(f"val/{name}/final_mae", maes[-1], result.epoch)
    log.info(f"  {'-'*12} {'-'*6} {'-'*10} {'-'*10}")
    log.info(f"  {'MEAN':<12} {'':>6} {result.mean_mse:>10.4f} {result.mean_mae:>10.4f}")
    # tensorboard: overall mean
    if writer is not None:
        writer.add_scalar("val/mean_mse", result.mean_mse, result.epoch)
        writer.add_scalar("val/mean_mae", result.mean_mae, result.epoch)
    # flush to ensure output is visible immediately
    for handler in logging.root.handlers:
        handler.flush()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def train(cfg: ZapbenchConfig, run_dir: Path) -> tuple[bool, ValidationResult | None]:
    """training loop with per-condition sparse data.

    args:
        cfg: combined config for data/train/model/split.
        run_dir: directory to save outputs (model, metrics, etc.).

    returns:
        (was_terminated, final_val_result) tuple.
    """
    import time

    # redirect stdout/stderr to log files
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    stdout_file = open(stdout_path, "w", buffering=1)
    stderr_file = open(stderr_path, "w", buffering=1)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = stdout_file
    sys.stderr = stderr_file

    # reconfigure logging to write to the log file
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    try:
        return _train_impl(cfg, run_dir)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        stdout_file.close()
        stderr_file.close()


def _train_impl(cfg: ZapbenchConfig, run_dir: Path) -> tuple[bool, ValidationResult | None]:
    """actual training implementation."""
    import time

    # limit CPU threads (for validation and any CPU ops)
    num_threads = int(os.environ.get("LSB_DJOB_NUMPROC", "12"))
    torch.set_num_threads(num_threads)
    torch._inductor.config.compile_threads = num_threads  # type: ignore[attr-defined]
    log.info(f"CPU threads: {num_threads}")

    # signal handling for graceful termination
    terminate_flag = {"value": False}

    def handle_sigusr2(_signum, _frame):
        terminate_flag["value"] = True
        log.info("SIGUSR2 received - will terminate after current epoch")

    signal.signal(signal.SIGUSR2, handle_sigusr2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device: {device}")

    # enable TF32 for faster matmul/conv on Ampere+ GPUs
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        log.info("tf32 precision: enabled")

    seed_everything(cfg.train.seed)

    # load all data: train on GPU, val/test on CPU
    log.info("loading data...")
    train_data, val_data, _test_data = load_all_data(
        cfg.data.traces_path, cfg.data.ephys_path, cfg.data.bin_size_ms,
        cfg.split, cfg.train.fitting_window, device,
    )
    log.info(f"GPU allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB, reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # update model config with actual num_neurons from data
    model_cfg = ModelConfig(
        num_neurons=train_data.num_neurons,
        latent_dims=cfg.model.latent_dims,
        encoder_decoder=cfg.model.encoder_decoder,
        evolver=cfg.model.evolver,
    )

    # batches per epoch (0 = 1 full pass over data)
    full_pass_batches = train_data.total_weight // cfg.train.batch_size
    log.info(f"total training samples: {train_data.total_weight}")
    log.info(f"batches_per_epoch for 1x coverage: {full_pass_batches}")
    batches_per_epoch = full_pass_batches if cfg.train.batches_per_epoch == 0 else cfg.train.batches_per_epoch
    # RNG and time offsets for batch sampling (on GPU)
    rng = torch.Generator(device=device)
    time_offsets = torch.arange(cfg.train.fitting_window, device=device)

    batch_shape = (cfg.train.batch_size, cfg.train.fitting_window, train_data.num_neurons)
    batch_mb = batch_shape[0] * batch_shape[1] * batch_shape[2] * 4 / 1e6
    log.info(f"batch shape: {batch_shape}, {batch_mb:.1f} MB per batch")

    # model
    model = EEDModel(model_cfg).to(device)
    log.info(f"model: {sum(p.numel() for p in model.parameters()):,} parameters")
    log.info(f"GPU after model: allocated={torch.cuda.memory_allocated() / 1e9:.2f} GB, reserved={torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
    log.info(f"GPU after optimizer: allocated={torch.cuda.memory_allocated() / 1e9:.2f} GB, reserved={torch.cuda.memory_reserved() / 1e9:.2f} GB")

    log.info(f"training: {cfg.train.epochs} epochs, {batches_per_epoch} batches/epoch")

    # tensorboard
    writer = SummaryWriter(log_dir=run_dir)
    log.info(f"tensorboard --logdir={run_dir}")

    # validation setup
    val_queue: queue.Queue[ValidationResult] = queue.Queue()
    val_thread: threading.Thread | None = None
    final_val_result: ValidationResult | None = None

    def start_validation(epoch: int) -> threading.Thread:
        """copy weights to CPU and start validation in background."""
        state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        t = threading.Thread(
            target=run_validation_cpu,
            args=(model_cfg, state_dict, val_data, val_queue, epoch),
        )
        t.start()
        return t

    def check_validation() -> ValidationResult | None:
        """check if validation finished and log results."""
        nonlocal val_thread
        if val_thread is not None and not val_thread.is_alive():
            result = val_queue.get_nowait()
            log_validation_result(result, writer)
            val_thread = None
            return result
        return None

    # start validation for epoch 0 (untrained model baseline)
    val_thread = start_validation(epoch=0)

    # chrome profiler (only record 5 steps during epoch 1)
    # epoch 0 = steps 0-N, epoch 1 = steps N+1-2N, etc.
    prof_wait = batches_per_epoch + 2  # skip epoch 0, then 2 warmup steps in epoch 1
    profile_path = run_dir / "zapbench_profile.json"
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=prof_wait, warmup=1, active=5, repeat=1),
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for epoch in range(cfg.train.epochs):
            model.train()
            epoch_loss = torch.tensor(0.0, device=device)
            epoch_start = time.time()

            for _batch_idx in range(batches_per_epoch):
                # sample batch and mask on GPU - fully vectorized, no CPU sync
                with torch.profiler.record_function("sample"):
                    batch, mask = train_data.sample_batch(cfg.train.batch_size, time_offsets, rng)

                optimizer.zero_grad()

                with torch.profiler.record_function("forward"):
                    loss = train_step(model, batch, mask)

                with torch.profiler.record_function("backward"):
                    loss.backward()

                with torch.profiler.record_function("optimizer_step"):
                    optimizer.step()

                epoch_loss = epoch_loss + loss.detach()  # stay on GPU, no sync
                prof.step()  # advance profiler schedule

            # sync only at epoch end
            avg_loss = epoch_loss.item() / batches_per_epoch
            epoch_time = time.time() - epoch_start
            log.info(f"epoch {epoch}: loss={avg_loss:.4f}, time={epoch_time:.1f}s")
            writer.add_scalar("train/loss", avg_loss, epoch)
            if epoch == 0:
                log.info(f"GPU after epoch 0: allocated={torch.cuda.memory_allocated() / 1e9:.2f} GB, reserved={torch.cuda.memory_reserved() / 1e9:.2f} GB")

            # check if validation finished, log results
            result = check_validation()
            if result is not None:
                final_val_result = result

            # start validation for next epoch if not busy
            if val_thread is None:
                val_thread = start_validation(epoch + 1)

            # check for graceful termination
            if terminate_flag["value"]:
                log.info(f"=== graceful termination at epoch {epoch + 1} ===")
                break

    # determine final epoch number (after all training)
    final_epoch = epoch + 1  # epoch is 0-indexed, so +1 for "after N epochs"

    # wait for any in-progress validation to complete
    last_validated_epoch = -1
    if val_thread is not None:
        log.info("waiting for in-progress validation...")
        val_thread.join()
        result = val_queue.get()
        log_validation_result(result, writer)
        final_val_result = result
        last_validated_epoch = result.epoch

    # run validation on final model if not already done
    if last_validated_epoch < final_epoch:
        log.info(f"running final validation (epoch {final_epoch})...")
        val_thread = start_validation(final_epoch)
        val_thread.join()
        result = val_queue.get()
        log_validation_result(result, writer)
        final_val_result = result

    # save profile
    prof.export_chrome_trace(str(profile_path))
    log.info(f"profile saved to {profile_path}")

    # print key timings
    log.info("=" * 60)
    log.info("KEY TIMINGS")
    log.info("=" * 60)
    key_averages = prof.key_averages()
    key_names = [
        "sample", "forward", "backward", "optimizer_step",
        "cudaStreamSynchronize", "aten::copy_", "aten::index",
    ]
    for event in key_averages:
        if event.key in key_names or any(k in event.key for k in key_names):
            log.info(f"  {event.key:45s}: {event.cpu_time_total/1000:8.1f} ms total, "
                     f"{event.cpu_time_total/1000/max(1,event.count):6.1f} ms avg, n={event.count}")
    log.info("=" * 60)

    # save final model
    model_path = run_dir / "model_final.pt"
    torch.save(model.state_dict(), model_path)
    log.info(f"saved final model to {model_path}")

    # save final metrics
    if final_val_result is not None:
        per_condition = {}
        for name in final_val_result.per_condition_mse:
            mses = final_val_result.per_condition_mse[name]
            maes = final_val_result.per_condition_mae[name]
            per_condition[name] = {
                "mean_mse": float(mses.mean()),
                "final_mse": float(mses[-1]),
                "mean_mae": float(maes.mean()),
                "final_mae": float(maes[-1]),
                "rollout_steps": len(mses),
            }
        metrics = {
            "final_epoch": final_epoch,
            "mean_mse": final_val_result.mean_mse,
            "mean_mae": final_val_result.mean_mae,
            "per_condition": per_condition,
            "was_terminated": terminate_flag["value"],
        }
        metrics_path = run_dir / "final_metrics.yaml"
        with open(metrics_path, "w") as f:
            yaml.dump(metrics, f, sort_keys=False, indent=2)
        log.info(f"saved final metrics to {metrics_path}")

    writer.close()
    log.info("done")
    return terminate_flag["value"], final_val_result


def main():
    """CLI entry point."""
    import re

    # parse expt_code from first argument
    if len(sys.argv) < 2:
        print("usage: python latent_zapbench.py <expt_code> [--overrides]")
        print("  example: python latent_zapbench.py my_expt --train.epochs 50")
        sys.exit(1)

    expt_code = sys.argv[1]
    if not re.match(r"^[A-Za-z0-9_]+$", expt_code):
        print(f"error: expt_code must match [A-Za-z0-9_]+, got: {expt_code}")
        sys.exit(1)

    # remaining args are tyro overrides
    tyro_args = sys.argv[2:]

    # get git commit hash
    commit_hash = get_git_commit_hash()

    # create run directory
    run_dir = create_run_directory(
        expt_code=expt_code,
        tyro_args=tyro_args,
        model_class=ZapbenchConfig,
        commit_hash=commit_hash,
    )

    # log command line
    with open(run_dir / "command_line.txt", "w") as f:
        f.write("\n".join(sys.argv))

    # parse config with tyro
    cfg = tyro.cli(ZapbenchConfig, args=tyro_args)

    # save config
    config_path = run_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg.model_dump(mode="json"), f, sort_keys=False, indent=2)

    log.info(f"run directory: {run_dir.resolve()}")
    log.info(f"config saved to {config_path}")

    # run training
    was_terminated, _ = train(cfg, run_dir)

    # add completion/termination flag
    flag_file = "terminated" if was_terminated else "complete"
    with open(run_dir / flag_file, "w"):
        pass


if __name__ == "__main__":
    main()
