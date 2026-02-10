"""
latent zapbench: per-condition pre-interpolated data on GPU.

data flow:
  startup: load sparse -> interpolate on GPU -> store on CPU (headroom)
           -> after all conditions loaded, transfer all to GPU
  training: sample on GPU -> forward/backward

# TODO: add CPU validation running in parallel with GPU training
#
# design:
#   - validation runs on CPU in background thread while GPU trains next epoch
#   - PyTorch releases GIL during tensor ops, so true parallelism is possible
#   - limit CPU threads with torch.set_num_threads(8)
#
# data loading (at startup):
#   - load validation data same as training: sparse -> interpolate on GPU -> CPU
#   - keep per-condition on CPU (don't concatenate like training)
#   - store as list of (activity_cpu, valid_start, valid_end) per condition
#   - validation doesn't need weighted sampling, just sequential rollouts
#
# validation data structure:
#   @dataclass
#   class ValidationCondition:
#       activity: torch.Tensor  # (T, N) on CPU
#       valid_start: int
#       valid_end: int
#       name: str
#
# validation function:
#   def run_validation_cpu(
#       state_dict: dict,           # model weights copied to CPU
#       val_conditions: list[ValidationCondition],
#       fitting_window: int,
#       rollout_steps: int,         # how far to roll out (e.g., 100, 500, 2000)
#       result_queue: queue.Queue,
#   ):
#       torch.set_num_threads(8)
#
#       # create model on CPU, load weights
#       model = EEDModel(cfg).cpu()
#       model.load_state_dict(state_dict)
#       model.eval()
#
#       all_mses = []
#       with torch.no_grad():
#           for cond in val_conditions:
#               # pick rollout start points (e.g., every 100 steps in valid range)
#               starts = range(cond.valid_start, cond.valid_end, fitting_window)
#
#               for start in starts:
#                   # ground truth window
#                   gt = cond.activity[start:start + rollout_steps]  # (rollout_steps, N)
#
#                   # encode initial state
#                   z = model.encode(gt[0:1])  # (1, L)
#
#                   # rollout
#                   preds = []
#                   for t in range(rollout_steps):
#                       x_pred = model.decode(z)  # (1, N)
#                       preds.append(x_pred)
#                       z = model.evolve(z)
#
#                   preds = torch.cat(preds, dim=0)  # (rollout_steps, N)
#                   mse = ((preds - gt) ** 2).mean(dim=1)  # (rollout_steps,) per-step MSE
#                   all_mses.append(mse)
#
#       # aggregate: mean MSE per rollout step across all conditions/starts
#       all_mses = torch.stack(all_mses)  # (num_rollouts, rollout_steps)
#       mean_mse_by_step = all_mses.mean(dim=0)  # (rollout_steps,)
#
#       result_queue.put({
#           'mean_mse_by_step': mean_mse_by_step,
#           'total_mse': mean_mse_by_step.mean().item(),
#       })
#
# integration in training loop:
#   val_queue = queue.Queue()
#   val_thread = None
#
#   for epoch in range(epochs):
#       # check if previous validation finished
#       if val_thread is not None and not val_thread.is_alive():
#           metrics = val_queue.get_nowait()
#           log.info(f"validation mse: {metrics['total_mse']:.4f}")
#           # log to tensorboard: mse by rollout step
#           val_thread = None
#
#       # train epoch on GPU
#       ...
#
#       # start validation in background after epoch
#       if epoch % val_every == 0 and val_thread is None:
#           state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
#           val_thread = threading.Thread(
#               target=run_validation_cpu,
#               args=(state_dict, val_conditions, fitting_window, rollout_steps, val_queue),
#           )
#           val_thread.start()
#
#   # wait for final validation
#   if val_thread is not None:
#       val_thread.join()
#       metrics = val_queue.get()
"""

import logging
import queue
import random
import signal
import threading
from dataclasses import dataclass

import numpy as np
import torch

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
    per_condition: dict  # {name: (rollout_len,) mse array}


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

        results: dict[str, np.ndarray] = {}

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
                    for t in range(rollout_len):
                        x_pred = model.decode(z)  # (1, N)

                        # masked MSE
                        err_sq = (x_pred[0] - gt[t]) ** 2
                        mask_t = mask[t]
                        mse_t = (err_sq * mask_t).sum() / mask_t.sum().clamp(min=1)
                        mses.append(mse_t.item())

                        z = model.evolve(z)

                    results[cond.name] = np.array(mses)

        mean_mse = float(np.mean([m.mean() for m in results.values()])) if results else 0.0

        result_queue.put(ValidationResult(
            epoch=epoch,
            mean_mse=mean_mse,
            per_condition=results,
        ))


def log_validation_result(result: ValidationResult) -> None:
    """log validation result as a table."""
    log.info(f"validation (epoch {result.epoch}):")
    log.info(f"  {'condition':<12} {'steps':>6} {'mean_mse':>10} {'final_mse':>10}")
    log.info(f"  {'-'*12} {'-'*6} {'-'*10} {'-'*10}")
    for name, mses in sorted(result.per_condition.items()):
        log.info(f"  {name:<12} {len(mses):>6} {mses.mean():>10.4f} {mses[-1]:>10.4f}")
    log.info(f"  {'-'*12} {'-'*6} {'-'*10} {'-'*10}")
    log.info(f"  {'MEAN':<12} {'':>6} {result.mean_mse:>10.4f}")
    # flush to ensure output is visible immediately
    for handler in logging.root.handlers:
        handler.flush()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    """training loop with per-condition sparse data."""
    import time

    # limit CPU threads (for validation and any CPU ops)
    import os
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

    # config
    data_cfg = DataConfig()
    train_cfg = TrainConfig()

    seed_everything(train_cfg.seed)

    # load all data: train on GPU, val/test on CPU
    log.info("loading data...")
    split = DataSplit()
    train_data, val_data, _test_data = load_all_data(
        data_cfg.traces_path, data_cfg.ephys_path, data_cfg.bin_size_ms,
        split, train_cfg.fitting_window, device,
    )
    log.info(f"GPU allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB, reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # batches per epoch for 1x coverage
    batches_per_epoch = train_data.total_weight // train_cfg.batch_size
    log.info(f"total training samples: {train_data.total_weight}")
    log.info(f"batches_per_epoch for 1x coverage: {batches_per_epoch}")

    # batches_per_epoch = 10  # uncomment for quick testing
    # RNG and time offsets for batch sampling (on GPU)
    rng = torch.Generator(device=device)
    time_offsets = torch.arange(train_cfg.fitting_window, device=device)

    batch_shape = (train_cfg.batch_size, train_cfg.fitting_window, train_data.num_neurons)
    batch_mb = batch_shape[0] * batch_shape[1] * batch_shape[2] * 4 / 1e6
    log.info(f"batch shape: {batch_shape}, {batch_mb:.1f} MB per batch")

    # model
    model_cfg = ModelConfig(num_neurons=train_data.num_neurons)
    model = EEDModel(model_cfg).to(device)
    log.info(f"model: {sum(p.numel() for p in model.parameters()):,} parameters")
    log.info(f"GPU after model: allocated={torch.cuda.memory_allocated() / 1e9:.2f} GB, reserved={torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
    log.info(f"GPU after optimizer: allocated={torch.cuda.memory_allocated() / 1e9:.2f} GB, reserved={torch.cuda.memory_reserved() / 1e9:.2f} GB")

    log.info(f"training: {train_cfg.epochs} epochs, {batches_per_epoch} batches/epoch")

    # validation setup
    val_queue: queue.Queue[ValidationResult] = queue.Queue()
    val_thread: threading.Thread | None = None

    def start_validation(epoch: int) -> threading.Thread:
        """copy weights to CPU and start validation in background."""
        state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        t = threading.Thread(
            target=run_validation_cpu,
            args=(model_cfg, state_dict, val_data, val_queue, epoch),
        )
        t.start()
        return t

    def check_validation() -> None:
        """check if validation finished and log results."""
        nonlocal val_thread
        if val_thread is not None and not val_thread.is_alive():
            result = val_queue.get_nowait()
            log_validation_result(result)
            val_thread = None

    # start validation for epoch 0 (untrained model baseline)
    val_thread = start_validation(epoch=0)

    # chrome profiler (only record 5 steps during epoch 1)
    # epoch 0 = steps 0-N, epoch 1 = steps N+1-2N, etc.
    prof_wait = batches_per_epoch + 2  # skip epoch 0, then 2 warmup steps in epoch 1
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=prof_wait, warmup=1, active=5, repeat=1),
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for epoch in range(train_cfg.epochs):
            model.train()
            epoch_loss = torch.tensor(0.0, device=device)
            epoch_start = time.time()

            for _batch_idx in range(batches_per_epoch):
                # sample batch and mask on GPU - fully vectorized, no CPU sync
                with torch.profiler.record_function("sample"):
                    batch, mask = train_data.sample_batch(train_cfg.batch_size, time_offsets, rng)

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
            if epoch == 0:
                log.info(f"GPU after epoch 0: allocated={torch.cuda.memory_allocated() / 1e9:.2f} GB, reserved={torch.cuda.memory_reserved() / 1e9:.2f} GB")

            # check if validation finished, log results
            check_validation()

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
        log_validation_result(result)
        last_validated_epoch = result.epoch

    # run validation on final model if not already done
    if last_validated_epoch < final_epoch:
        log.info(f"running final validation (epoch {final_epoch})...")
        val_thread = start_validation(final_epoch)
        val_thread.join()
        result = val_queue.get()
        log_validation_result(result)

    # save profile
    prof.export_chrome_trace("zapbench_profile.json")
    log.info("profile saved to zapbench_profile.json")

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
    log.info("done")


if __name__ == "__main__":
    main()
