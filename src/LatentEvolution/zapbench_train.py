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
import signal
import sys
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
import yaml
from torch.utils.tensorboard import SummaryWriter

from LatentEvolution.hparam_paths import create_run_directory, get_git_commit_hash
from LatentEvolution.training_utils import LossAccumulator, seed_everything


class LossType(Enum):
    """loss component types for zapbench EED model."""
    TOTAL = "total"
    RECON = "recon"
    EVOLVE = "evolve"
    EVOLVE_L1_REG = "evolve_l1_reg"


from LatentEvolution.zapbench_data import (
    load_sparse_activity,
    interpolate_sparse_compiled,
)
from LatentEvolution.zapbench_config import (
    DataConfig,
    DataSplit,
    ModelConfig,
    TrainConfig,
)
from LatentEvolution.zapbench_eed import EEDModel

# configure logging with HH:MM:SS timestamp
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

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


def build_frame_index(
    obs_times: torch.Tensor,
    num_bins: int,
    device: torch.device,
) -> torch.Tensor:
    """build frame index tensor from sparse observation times.

    args:
        obs_times: (N, K) sorted bin indices per neuron, K = num original frames
        num_bins: total number of time bins (T)
        device: target device for computation

    returns:
        frame_index: (T, N) int16 tensor
            frame_index[bin, n] = k if frame k observed at (bin, n), else -1
            use (frame_index >= 0) as observation mask
    """
    N, K = obs_times.shape

    # create indices
    neuron_indices = torch.arange(N, device=device).unsqueeze(1).expand(N, K)
    frame_indices = torch.arange(K, device=device, dtype=torch.int16).unsqueeze(0).expand(N, K)

    # scatter frame indices into dense tensor (-1 = no observation)
    frame_index = torch.full((num_bins, N), -1, dtype=torch.int16, device=device)
    obs_times_dev = obs_times.to(device)
    frame_index[obs_times_dev, neuron_indices] = frame_indices

    return frame_index


# ---------------------------------------------------------------------------
# per-condition data (for val/test on CPU)
# ---------------------------------------------------------------------------


@dataclass
class ConditionData:
    """per-condition data on CPU for validation/test.

    kept separate by condition for diagnostics (no concatenation).
    """
    activity: torch.Tensor    # (T, N) float16 on CPU
    frame_index: torch.Tensor # (T, N) int16 on CPU, -1 = no obs, >=0 = frame index
    valid_start: int          # first valid sampling index (after first obs)
    valid_end: int            # last valid sampling index (before last obs - fitting_window)
    name: str                 # condition name
    num_neurons: int
    num_frames: int           # K, number of original frames


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
    activity: torch.Tensor        # (total_T, N) float16 on GPU
    obs_mask: torch.Tensor        # (total_T, N) bool on GPU
    valid_starts: torch.Tensor    # (num_conds,) global valid start indices
    valid_ends: torch.Tensor      # (num_conds,) global valid end indices
    weights: torch.Tensor         # (num_conds,) sampling weights per condition
    change_threshold: torch.Tensor  # (N,) per-neuron significant change threshold
    num_neurons: int
    num_conds: int

    @property
    def total_weight(self) -> int:
        """total number of valid start positions across all conditions."""
        return int(self.weights.sum().item())

    def sample_batch(
        self, batch_size: int, time_offsets: torch.Tensor, rng: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """sample batch on GPU - fully vectorized, no CPU sync.

        args:
            batch_size: number of samples to draw.
            time_offsets: precomputed arange(fitting_window) on GPU.
            rng: random number generator on GPU.

        returns:
            batch: (B, fitting_window, N) float32 GPU tensor.
            mask: (B, fitting_window, N) bool GPU tensor, True at real observations.
            sig_mask: (B, fitting_window, N) bool GPU tensor, True at significant changes.
        """
        device = self.activity.device
        T = len(time_offsets)

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

        # compute sig_mask: significant changes between consecutive real observations
        sig_mask = self._compute_sig_mask(batch, mask, T)

        return batch, mask, sig_mask

    def _compute_sig_mask(
        self, batch: torch.Tensor, mask: torch.Tensor, T: int,
    ) -> torch.Tensor:
        """compute significant change mask from batch data.

        for each observation, checks if change from previous real observation
        of the same neuron exceeds the threshold.

        args:
            batch: (B, T, N) activity values
            mask: (B, T, N) bool, True at real observations
            T: number of timesteps

        returns:
            sig_mask: (B, T, N) bool, True at significant changes
        """
        B, _, N = batch.shape
        device = batch.device

        # get observation indices sorted by (batch, neuron, time)
        obs_b, obs_t, obs_n = torch.where(mask)
        obs_vals = batch[obs_b, obs_t, obs_n]

        # sort by (batch, neuron, time) to group observations
        sort_key = obs_b * (N * T) + obs_n * T + obs_t
        sort_idx = torch.argsort(sort_key)

        obs_b_sorted = obs_b[sort_idx]
        obs_n_sorted = obs_n[sort_idx]
        obs_vals_sorted = obs_vals[sort_idx]

        # find consecutive observations of same (batch, neuron)
        same_bn = (obs_b_sorted[1:] == obs_b_sorted[:-1]) & (obs_n_sorted[1:] == obs_n_sorted[:-1])

        # compute changes and check against threshold
        changes = torch.abs(obs_vals_sorted[1:] - obs_vals_sorted[:-1])
        threshold_per_obs = self.change_threshold[obs_n_sorted[1:]]
        is_sig = same_bn & (changes > threshold_per_obs)

        # build sig_flags: first observation of each (batch, neuron) is never significant
        sig_flags = torch.zeros(len(obs_vals), dtype=torch.bool, device=device)
        sig_flags[1:] = is_sig

        # unsort to original order
        unsort_idx = torch.argsort(sort_idx)
        sig_flags_unsorted = sig_flags[unsort_idx]

        # scatter back to dense mask
        sig_mask = torch.zeros_like(mask)
        sig_mask[obs_b, obs_t, obs_n] = sig_flags_unsorted

        return sig_mask


# ---------------------------------------------------------------------------
# significant change threshold computation
# ---------------------------------------------------------------------------


def compute_change_thresholds(
    activity: torch.Tensor,      # (total_T, N) float16
    obs_mask: torch.Tensor,      # (total_T, N) bool
    num_sigma: float = 2.0,
) -> torch.Tensor:
    """compute per-neuron threshold for 'significant' activity change.

    for each neuron, computes mean + num_sigma * std of absolute changes
    between consecutive observed timepoints.

    args:
        activity: (total_T, N) float16 tensor
        obs_mask: (total_T, N) bool tensor
        num_sigma: number of standard deviations above mean (default 2.0)

    returns:
        change_threshold: (N,) float32 tensor
    """
    T, N = activity.shape
    device = activity.device

    # get observation indices sorted by neuron (transpose trick)
    obs_n, obs_t = torch.where(obs_mask.T)
    obs_vals = activity[obs_t, obs_n].float()

    # compute changes between consecutive same-neuron observations
    same_neuron = obs_n[1:] == obs_n[:-1]
    all_changes = torch.abs(obs_vals[1:] - obs_vals[:-1])
    valid_changes = all_changes[same_neuron]
    change_neurons = obs_n[1:][same_neuron]

    # compute sum, sum of squares, and count per neuron using scatter_add
    change_sum = torch.zeros(N, device=device)
    change_sq_sum = torch.zeros(N, device=device)
    change_counts = torch.zeros(N, device=device)

    change_sum.scatter_add_(0, change_neurons, valid_changes)
    change_sq_sum.scatter_add_(0, change_neurons, valid_changes ** 2)
    change_counts.scatter_add_(0, change_neurons, torch.ones_like(valid_changes))

    # compute mean and std
    mean = change_sum / change_counts.clamp(min=1)
    var = (change_sq_sum / change_counts.clamp(min=1)) - mean ** 2
    std = var.clamp(min=0).sqrt()

    # threshold = mean + num_sigma * std
    thresholds = mean + num_sigma * std

    # neurons with < 2 changes get inf threshold
    thresholds[change_counts < 2] = float('inf')

    return thresholds


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------


def _interpolate_and_frame_index(
    obs_times: torch.Tensor,
    obs_vals: torch.Tensor,
    counts: torch.Tensor,
    num_bins: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """interpolate sparse data and build frame index on GPU, return on CPU.

    args:
        obs_times: (N, K) sparse observation times on CPU
        obs_vals: (N, K) sparse values on CPU
        counts: (N,) counts per neuron on CPU
        num_bins: number of output bins
        device: GPU device for computation

    returns:
        activity: (T, N) float16 on CPU
        frame_index: (T, N) int16 on CPU, -1 = no obs, >=0 = frame index
        num_frames: K, number of original frames
    """
    N, K = obs_times.shape

    # transfer to GPU
    obs_times_gpu = obs_times.to(device)
    obs_vals_gpu = obs_vals.to(device)
    counts_gpu = counts.to(device)

    # interpolate on GPU
    activity_gpu = interpolate_sparse_compiled(
        obs_times_gpu, obs_vals_gpu, counts_gpu, num_bins,
    )

    # build frame index on GPU
    frame_index_gpu = build_frame_index(obs_times, num_bins, device)

    # move to CPU
    activity_cpu = activity_gpu.half().cpu()
    frame_index_cpu = frame_index_gpu.cpu()

    # free GPU memory
    del obs_times_gpu, obs_vals_gpu, counts_gpu, activity_gpu, frame_index_gpu
    torch.cuda.empty_cache()

    return activity_cpu, frame_index_cpu, K


def load_all_data(
    traces_path: str,
    ephys_path: str,
    bin_size_ms: float,
    split: DataSplit,
    fitting_window: int,
    device: torch.device,
    skip_train: bool = False,
) -> tuple[ConcatTrainingData | None, list[ConditionData], list[ConditionData]]:
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
        skip_train: if True, skip loading train data (for evaluation only).

    returns:
        train_data: ConcatTrainingData on GPU (None if skip_train=True)
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
            if skip_train and split_name == "train":
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

            # interpolate and build frame index
            activity, frame_index, num_frames = _interpolate_and_frame_index(
                split_obs_times, split_obs_vals, split_counts, split_num_bins, device,
            )

            # compute valid sampling range
            first_obs = int(split_obs_times[:, 0].max().item()) + 1
            last_obs = int(split_obs_times[:, -1].min().item())
            valid_start = first_obs
            valid_end = max(0, last_obs - fitting_window + 1)

            T = activity.shape[0]
            log.info(f"    {split_name}: T={T}, frames={num_frames}, valid=[{valid_start}, {valid_end})")

            if split_name == "train":
                # accumulate for concatenation (derive obs_mask from frame_index)
                obs_mask = frame_index >= 0
                train_activities.append(activity)
                train_masks.append(obs_mask)
                train_valid_starts.append(train_offset + valid_start)
                train_valid_ends.append(train_offset + valid_end)
                train_weights.append(max(0, valid_end - valid_start))
                train_offset += T
            else:
                # store per-condition on CPU with frame_index for per-frame metrics
                cond_data = ConditionData(
                    activity=activity,
                    frame_index=frame_index,
                    valid_start=valid_start,
                    valid_end=valid_end,
                    name=cond_name,
                    num_neurons=num_neurons,
                    num_frames=num_frames,
                )
                if split_name == "val":
                    val_data.append(cond_data)
                else:
                    test_data.append(cond_data)

    # skip train data processing if requested
    if skip_train:
        val_total = sum(c.activity.shape[0] for c in val_data)
        test_total = sum(c.activity.shape[0] for c in test_data)
        log.info(f"  val: {len(val_data)} conditions, {val_total} total bins (CPU)")
        log.info(f"  test: {len(test_data)} conditions, {test_total} total bins (CPU)")
        return None, val_data, test_data

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

    # compute per-neuron change thresholds for significant-event masking
    log.info("  computing per-neuron change thresholds...")
    change_threshold = compute_change_thresholds(all_train_activity, all_train_mask)
    num_valid = (change_threshold < float('inf')).sum().item()
    log.info(f"  neurons with valid threshold: {num_valid:,} / {change_threshold.shape[0]:,}")

    # transfer to GPU
    train_activity_gpu = all_train_activity.to(device)
    del all_train_activity
    train_mask_gpu = all_train_mask.to(device)
    del all_train_mask
    change_threshold_gpu = change_threshold.to(device)

    assert num_neurons is not None
    train_data = ConcatTrainingData(
        activity=train_activity_gpu,
        obs_mask=train_mask_gpu,
        valid_starts=torch.tensor(train_valid_starts, device=device, dtype=torch.long),
        valid_ends=torch.tensor(train_valid_ends, device=device, dtype=torch.long),
        weights=torch.tensor(train_weights, device=device, dtype=torch.float),
        change_threshold=change_threshold_gpu,
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
    sig_mask: torch.Tensor, # (B, T, N) bool, significant changes
    evolve_l1_reg_weight: float,
) -> dict[LossType, torch.Tensor]:
    """training step: encode, evolve, decode, compute masked loss.

    computes loss components:
    - RECON: autoencoder reconstruction loss at each timestep (encode -> decode)
    - EVOLVE: evolution loss only at timesteps with significant activity changes
    - EVOLVE_L1_REG: L1 regularization on evolver delta_z (pushes toward identity)

    args:
        model: EED model
        batch: (B, T, N) neural activity sequence
        obs_mask: (B, T, N) bool, True where values are real observations
        sig_mask: (B, T, N) bool, True where significant change from prev real obs
        evolve_l1_reg_weight: weight for L1 regularization on evolver updates

    returns:
        dict mapping LossType to scalar tensor
    """
    device = batch.device
    T = batch.shape[1]

    # === RECONSTRUCTION LOSS ===
    # encode full input (real + interpolated), constrain decoder only on real obs
    recon_loss = torch.tensor(0.0, device=device)
    for t in range(T):
        mask_t = obs_mask[:, t, :]
        z_t = model.encode(batch[:, t, :])
        x_recon = model.decode(z_t)
        error_sq = (x_recon - batch[:, t, :]) ** 2
        # use clamp(min=1) to avoid div by zero when no observations
        recon_loss = recon_loss + (error_sq * mask_t).sum() / mask_t.sum().clamp(min=1)

    # === EVOLUTION LOSS (significant changes only) + L1 REG ===
    z = model.encode(batch[:, 0, :])
    evolve_loss = torch.tensor(0.0, device=device)
    l1_reg = torch.tensor(0.0, device=device)

    for t in range(T):
        x_pred = model.decode(z)

        # accumulate evolution loss only at significant change points
        mask_t = sig_mask[:, t, :]
        error_sq = (x_pred - batch[:, t, :]) ** 2
        evolve_loss = evolve_loss + (error_sq * mask_t).sum() / mask_t.sum().clamp(min=1)

        # evolve and accumulate L1 reg on delta_z
        z_next = model.evolve(z)
        delta_z = z_next - z
        l1_reg = l1_reg + torch.abs(delta_z).mean()
        z = z_next

    total_loss = recon_loss + evolve_loss + evolve_l1_reg_weight * l1_reg

    return {
        LossType.TOTAL: total_loss,
        LossType.RECON: recon_loss,
        LossType.EVOLVE: evolve_loss,
        LossType.EVOLVE_L1_REG: l1_reg,
    }


# ---------------------------------------------------------------------------
# CPU validation (runs in background thread)
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """result from background validation."""
    epoch: int
    mean_mse: float
    mean_mae: float
    per_condition_mse: dict  # {name: (num_frames,) mse array}
    per_condition_mae: dict  # {name: (num_frames,) mae array}
    baseline_mse: dict  # {name: (num_frames,) baseline mse} - predict mean of first 4 frames
    baseline_mae: dict  # {name: (num_frames,) baseline mae}


def run_validation_cpu(
    model_cfg: "ModelConfig",
    state_dict: dict,
    val_conditions: list[ConditionData],
    result_queue: "queue.Queue[ValidationResult]",
    epoch: int,
    max_rollout_frames: int | None = None,
) -> None:
    """run validation on CPU, put result in queue.

    args:
        model_cfg: model configuration for creating CPU model.
        state_dict: model weights (already on CPU).
        val_conditions: list of ConditionData on CPU.
        result_queue: queue to put results.
        epoch: which epoch this validation corresponds to.
        max_rollout_frames: cap rollout length (None = rollout to end).
    """
    with torch.profiler.record_function(f"validation_epoch_{epoch}"):

        # create model on CPU, load weights
        with torch.profiler.record_function("validation_model_setup"):
            model = EEDModel(model_cfg).cpu()
            model.load_state_dict(state_dict)
            model.eval()

        results_mse: dict[str, np.ndarray] = {}
        results_mae: dict[str, np.ndarray] = {}
        baseline_mse: dict[str, np.ndarray] = {}
        baseline_mae: dict[str, np.ndarray] = {}

        with torch.no_grad():
            for cond in val_conditions:
                with torch.profiler.record_function(f"validation_{cond.name}"):
                    T = cond.activity.shape[0]
                    start = cond.valid_start
                    end = T  # process all bins (frame limiting done later)
                    rollout_len = end - start

                    if rollout_len <= 0:
                        continue

                    # get ground truth and frame index (float16 -> float32)
                    gt = cond.activity[start:end].float()  # (rollout_len, N)
                    frame_idx = cond.frame_index[start:end]  # (rollout_len, N) int16
                    obs_mask = frame_idx >= 0  # (rollout_len, N) bool

                    # encode initial state
                    z = model.encode(gt[0:1])  # (1, L)

                    # collect predictions for all bins (pre-allocated)
                    preds = torch.empty(rollout_len, cond.num_neurons)
                    for t in range(rollout_len):
                        preds[t] = model.decode(z)[0]
                        z = model.evolve(z)

                    # compute errors at observed points
                    errors_sq = (preds - gt) ** 2  # (rollout_len, N)
                    errors_abs = (preds - gt).abs()  # (rollout_len, N)

                    # accumulate per-frame MSE/MAE
                    # frame_idx[bin, n] = frame index for that observation
                    # limit to max_rollout_steps frames (not bins!)
                    num_frames = cond.num_frames if max_rollout_frames is None else min(cond.num_frames, max_rollout_frames)
                    frame_mse_sum = torch.zeros(num_frames)
                    frame_mae_sum = torch.zeros(num_frames)
                    frame_counts = torch.zeros(num_frames)

                    # get indices of observed points
                    bin_indices, neuron_indices = torch.where(obs_mask)
                    frame_indices_abs = frame_idx[bin_indices, neuron_indices].long()

                    # shift frame indices to be relative to rollout start
                    # (frame 0 observations happened before valid_start, so first frame we see is > 0)
                    min_frame = frame_indices_abs.min().item() if len(frame_indices_abs) > 0 else 0
                    frame_indices = frame_indices_abs - min_frame

                    # filter to first num_frames (relative)
                    valid_mask = frame_indices < num_frames
                    bin_indices = bin_indices[valid_mask]
                    neuron_indices = neuron_indices[valid_mask]
                    frame_indices = frame_indices[valid_mask]

                    # gather errors at observed points
                    obs_errors_sq = errors_sq[bin_indices, neuron_indices]
                    obs_errors_abs = errors_abs[bin_indices, neuron_indices]

                    # scatter_add into per-frame accumulators
                    frame_mse_sum.scatter_add_(0, frame_indices, obs_errors_sq)
                    frame_mae_sum.scatter_add_(0, frame_indices, obs_errors_abs)
                    frame_counts.scatter_add_(0, frame_indices, torch.ones_like(obs_errors_sq))

                    # compute mean per frame
                    frame_mse = frame_mse_sum / frame_counts.clamp(min=1)
                    frame_mae = frame_mae_sum / frame_counts.clamp(min=1)

                    results_mse[cond.name] = frame_mse.numpy()
                    results_mae[cond.name] = frame_mae.numpy()

                    # baseline: predict mean of first 4 frames per neuron (relative to rollout start)
                    # collect observed values from first 4 frames
                    baseline_frames = 4
                    # use relative frame indices (frame_idx - min_frame)
                    frame_idx_rel = frame_idx - min_frame
                    first_frames_mask = (frame_idx_rel >= 0) & (frame_idx_rel < baseline_frames)
                    first_frames_obs = obs_mask & first_frames_mask
                    # compute mean per neuron from observed values in first 4 frames
                    neuron_sum = torch.zeros(cond.num_neurons)
                    neuron_count = torch.zeros(cond.num_neurons)
                    obs_bin, obs_neuron = torch.where(first_frames_obs)
                    neuron_sum.scatter_add_(0, obs_neuron, gt[obs_bin, obs_neuron])
                    neuron_count.scatter_add_(0, obs_neuron, torch.ones(len(obs_neuron)))
                    mean_per_neuron = neuron_sum / neuron_count.clamp(min=1)  # (N,)

                    # compute baseline errors (constant prediction = mean_per_neuron)
                    baseline_errors_sq = (mean_per_neuron.unsqueeze(0) - gt) ** 2
                    baseline_errors_abs = (mean_per_neuron.unsqueeze(0) - gt).abs()

                    # accumulate baseline per-frame MSE/MAE
                    bl_mse_sum = torch.zeros(num_frames)
                    bl_mae_sum = torch.zeros(num_frames)
                    bl_obs_errors_sq = baseline_errors_sq[bin_indices, neuron_indices]
                    bl_obs_errors_abs = baseline_errors_abs[bin_indices, neuron_indices]
                    bl_mse_sum.scatter_add_(0, frame_indices, bl_obs_errors_sq)
                    bl_mae_sum.scatter_add_(0, frame_indices, bl_obs_errors_abs)

                    baseline_mse[cond.name] = (bl_mse_sum / frame_counts.clamp(min=1)).numpy()
                    baseline_mae[cond.name] = (bl_mae_sum / frame_counts.clamp(min=1)).numpy()

        mean_mse = float(np.mean([m.mean() for m in results_mse.values()])) if results_mse else 0.0
        mean_mae = float(np.mean([m.mean() for m in results_mae.values()])) if results_mae else 0.0

        result_queue.put(ValidationResult(
            epoch=epoch,
            mean_mse=mean_mse,
            mean_mae=mean_mae,
            per_condition_mse=results_mse,
            per_condition_mae=results_mae,
            baseline_mse=baseline_mse,
            baseline_mae=baseline_mae,
        ))


def plot_per_frame_metrics(
    result: ValidationResult,
    fitting_window: int,
    prefix: str = "val",
) -> dict[str, plt.Figure]:
    """plot MSE and MAE vs frame index for each condition.

    args:
        result: validation result with per-condition per-frame metrics.
        fitting_window: draw a vertical line at this frame index (training horizon).
        prefix: title prefix ("val" or "test").

    returns:
        dict of {name: figure} for tensorboard logging.
    """
    figures = {}

    def compute_mean_over_conditions(per_condition: dict[str, np.ndarray]) -> np.ndarray:
        """compute mean over conditions at each frame, handling different lengths."""
        max_len = max(len(v) for v in per_condition.values())
        # pad with nan, then nanmean
        padded = np.full((len(per_condition), max_len), np.nan)
        for i, v in enumerate(per_condition.values()):
            padded[i, :len(v)] = v
        return np.nanmean(padded, axis=0)

    # MSE plot
    mse_fig, mse_ax = plt.subplots(figsize=(10, 6))
    for name in sorted(result.per_condition_mse.keys()):
        mses = result.per_condition_mse[name]
        mse_ax.plot(np.arange(1, len(mses) + 1), mses, label=name, alpha=0.8)
    # mean over conditions
    mse_mean = compute_mean_over_conditions(result.per_condition_mse)
    mse_ax.plot(np.arange(1, len(mse_mean) + 1), mse_mean, "k--", label="mean", linewidth=2)
    # baseline mean (predict mean of first 4 frames)
    bl_mse_mean = compute_mean_over_conditions(result.baseline_mse)
    mse_ax.plot(np.arange(1, len(bl_mse_mean) + 1), bl_mse_mean, "r--", label="baseline", linewidth=2)
    mse_ax.set_xlabel("frame index")
    mse_ax.set_ylabel("MSE")
    mse_ax.set_yscale("log")
    mse_ax.set_xlim(1, 32)
    mse_ax.set_ylim(1e-4, 0.1)
    mse_ax.legend(loc="upper left", fontsize=8)
    mse_ax.set_title(f"{prefix} MSE vs frame (epoch {result.epoch})")
    mse_ax.grid(True, alpha=0.3)
    mse_ax.axvline(x=fitting_window, color="k", linestyle=":", alpha=0.5)
    mse_fig.tight_layout()
    figures["mse_vs_frame"] = mse_fig

    # MAE plot
    mae_fig, mae_ax = plt.subplots(figsize=(10, 6))
    for name in sorted(result.per_condition_mae.keys()):
        maes = result.per_condition_mae[name]
        mae_ax.plot(np.arange(1, len(maes) + 1), maes, label=name, alpha=0.8)
    # mean over conditions
    mae_mean = compute_mean_over_conditions(result.per_condition_mae)
    mae_ax.plot(np.arange(1, len(mae_mean) + 1), mae_mean, "k--", label="mean", linewidth=2)
    # baseline mean
    bl_mae_mean = compute_mean_over_conditions(result.baseline_mae)
    mae_ax.plot(np.arange(1, len(bl_mae_mean) + 1), bl_mae_mean, "r--", label="baseline", linewidth=2)
    mae_ax.set_xlabel("frame index")
    mae_ax.set_ylabel("MAE")
    mae_ax.set_yscale("log")
    mae_ax.set_xlim(1, 32)
    mae_ax.set_ylim(1e-3, 1.0)
    mae_ax.legend(loc="upper left", fontsize=8)
    mae_ax.set_title(f"{prefix} MAE vs frame (epoch {result.epoch})")
    mae_ax.grid(True, alpha=0.3)
    mae_ax.axvline(x=fitting_window, color="k", linestyle=":", alpha=0.5)
    mae_fig.tight_layout()
    figures["mae_vs_frame"] = mae_fig

    return figures


def print_results_table(
    result: ValidationResult,
    name: str,
    include_baseline: bool = False,
) -> None:
    """print results tables showing metrics at key time steps.

    args:
        result: validation result with per-condition per-frame metrics.
        name: table title (e.g., "Validation", "Test").
        include_baseline: if True, also print baseline tables.
    """
    # key steps to display (0-indexed internally, 1-indexed for display)
    key_steps = [1, 2, 4, 8, 16, 32]

    def make_table(per_condition: dict[str, np.ndarray], metric_name: str):
        """print a table for one metric."""
        # header
        header = f"{'condition':<12}"
        for step in key_steps:
            header += f" {f'step{step}':>8}"
        log.info(header)
        log.info("-" * len(header))

        # rows for each condition
        means_per_step = {step: [] for step in key_steps}
        for cond_name in sorted(per_condition.keys()):
            values = per_condition[cond_name]
            row = f"{cond_name:<12}"
            for step in key_steps:
                idx = step - 1  # 0-indexed
                if idx < len(values):
                    val = values[idx]
                    row += f" {val:>8.4f}"
                    means_per_step[step].append(val)
                else:
                    row += f" {'n/a':>8}"
            log.info(row)

        # mean row
        log.info("-" * len(header))
        mean_row = f"{'MEAN':<12}"
        for step in key_steps:
            if means_per_step[step]:
                mean_row += f" {np.mean(means_per_step[step]):>8.4f}"
            else:
                mean_row += f" {'n/a':>8}"
        log.info(mean_row)

    log.info(f"\n{name} - MAE by Frame:")
    make_table(result.per_condition_mae, "MAE")

    if include_baseline:
        log.info(f"\n{name} - Baseline MAE by Frame:")
        make_table(result.baseline_mae, "Baseline MAE")

    # skip mse to avoid noisy stdout
    # log.info(f"\n{name} - MSE by Frame:")
    # make_table(result.per_condition_mse, "MSE")

    # if include_baseline:
    #     log.info(f"\n{name} - Baseline MSE by Frame:")
    #     make_table(result.baseline_mse, "Baseline MSE")


def log_validation_result(
    result: ValidationResult,
    fitting_window: int,
    writer: SummaryWriter | None = None,
    prefix: str = "val",
) -> None:
    """log validation result as a table and to tensorboard.

    args:
        result: validation result to log.
        fitting_window: training horizon (for vertical line on plots).
        writer: tensorboard writer (optional).
        prefix: tensorboard metric prefix ("val" or "test").
    """
    # print key-step table (no baseline during training)
    print_results_table(result, f"{prefix} (epoch {result.epoch})", include_baseline=False)

    # tensorboard logging per condition
    for name in sorted(result.per_condition_mse.keys()):
        mses = result.per_condition_mse[name]
        maes = result.per_condition_mae[name]
        if writer is not None:
            writer.add_scalar(f"{prefix}/{name}/mean_mse", mses.mean(), result.epoch)
            writer.add_scalar(f"{prefix}/{name}/final_mse", mses[-1], result.epoch)
            writer.add_scalar(f"{prefix}/{name}/mean_mae", maes.mean(), result.epoch)
            writer.add_scalar(f"{prefix}/{name}/final_mae", maes[-1], result.epoch)

    # tensorboard: overall mean and figures
    if writer is not None:
        writer.add_scalar(f"{prefix}/mean_mse", result.mean_mse, result.epoch)
        writer.add_scalar(f"{prefix}/mean_mae", result.mean_mae, result.epoch)
        # add per-frame plots
        figures = plot_per_frame_metrics(result, fitting_window, prefix)
        for fig_name, fig in figures.items():
            writer.add_figure(f"{prefix}/{fig_name}", fig, result.epoch)
            plt.close(fig)

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

    # limit CPU threads (for validation and any CPU ops), leave 1 for main thread
    num_threads = max(1, int(os.environ.get("LSB_DJOB_NUMPROC", "12")) - 1)
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
    train_data, val_data, test_data = load_all_data(
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
    eval_queue: queue.Queue[ValidationResult] = queue.Queue()
    val_thread: threading.Thread | None = None
    final_val_result: ValidationResult | None = None

    def start_eval(data: list[ConditionData], epoch: int, max_rollout: int = 32) -> threading.Thread:
        """copy weights to CPU and start evaluation in background."""
        state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        t = threading.Thread(
            target=run_validation_cpu,
            args=(model_cfg, state_dict, data, eval_queue, epoch, max_rollout),
        )
        t.start()
        return t

    def check_validation() -> ValidationResult | None:
        """check if validation finished and log results."""
        nonlocal val_thread
        if val_thread is not None and not val_thread.is_alive():
            result = eval_queue.get_nowait()
            log_validation_result(result, cfg.train.fitting_window, writer, prefix="val")
            val_thread = None
            return result
        return None

    # start validation for epoch 0 (untrained model baseline)
    val_thread = start_eval(val_data, epoch=0)

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
            epoch_losses = LossAccumulator(LossType)
            epoch_start = time.time()

            for _batch_idx in range(batches_per_epoch):
                # sample batch, mask, and sig_mask on GPU
                with torch.profiler.record_function("sample"):
                    batch, mask, sig_mask = train_data.sample_batch(cfg.train.batch_size, time_offsets, rng)

                optimizer.zero_grad()

                with torch.profiler.record_function("forward"):
                    evolve_mask = sig_mask if cfg.train.evolve_significant_only else mask
                    loss_dict = train_step(model, batch, mask, evolve_mask, cfg.train.evolve_l1_reg_weight)

                with torch.profiler.record_function("backward"):
                    loss_dict[LossType.TOTAL].backward()

                with torch.profiler.record_function("optimizer_step"):
                    optimizer.step()

                epoch_losses.accumulate(loss_dict)
                prof.step()  # advance profiler schedule

            # sync only at epoch end
            mean_losses = epoch_losses.mean()
            epoch_time = time.time() - epoch_start
            log.info(f"epoch {epoch}: total={mean_losses[LossType.TOTAL]:.4e} "
                     f"recon={mean_losses[LossType.RECON]:.4e} "
                     f"evolve={mean_losses[LossType.EVOLVE]:.4e} "
                     f"l1_reg={mean_losses[LossType.EVOLVE_L1_REG]:.4e}, time={epoch_time:.1f}s")
            for loss_type in LossType:
                writer.add_scalar(f"train/{loss_type.value}", mean_losses[loss_type], epoch)
            if epoch == 0:
                log.info(f"GPU after epoch 0: allocated={torch.cuda.memory_allocated() / 1e9:.2f} GB, reserved={torch.cuda.memory_reserved() / 1e9:.2f} GB")

            # check if validation finished, log results
            result = check_validation()
            if result is not None:
                final_val_result = result

            # start validation for next epoch if not busy
            if val_thread is None:
                val_thread = start_eval(val_data, epoch + 1)

            # check for graceful termination
            if terminate_flag["value"]:
                log.info(f"=== graceful termination at epoch {epoch + 1} ===")
                break

    # save profile immediately after profiler context ends
    if prof.profiler is not None:
        prof.export_chrome_trace(str(profile_path))
        log.info(f"profile saved to {profile_path}")
        log.info("=" * 60)
        log.info("KEY TIMINGS")
        log.info("=" * 60)
        key_averages = prof.key_averages()
        # exact match keys
        exact_keys = ["sample", "forward", "backward", "optimizer_step", "validation_model_setup"]
        # prefix match keys (for dynamic names like validation_epoch_0, validation_gain)
        prefix_keys = ["validation_epoch_", "validation_"]
        for event in key_averages:
            is_exact = event.key in exact_keys
            is_prefix = any(event.key.startswith(p) for p in prefix_keys)
            if is_exact or is_prefix:
                log.info(f"  {event.key:45s}: {event.cpu_time_total/1000:8.1f} ms total, "
                         f"{event.cpu_time_total/1000/max(1,event.count):6.1f} ms avg, n={event.count}")
        log.info("=" * 60)

    # determine final epoch number (after all training)
    final_epoch = epoch + 1  # epoch is 0-indexed, so +1 for "after N epochs"

    # wait for any in-progress validation to complete
    last_validated_epoch = -1
    if val_thread is not None:
        log.info("waiting for in-progress validation...")
        val_thread.join()
        result = eval_queue.get()
        log_validation_result(result, cfg.train.fitting_window, writer)
        final_val_result = result
        last_validated_epoch = result.epoch

    # run validation on final model if not already done
    if last_validated_epoch < final_epoch:
        log.info(f"running final validation (epoch {final_epoch})...")
        val_thread = start_eval(val_data, final_epoch)
        val_thread.join()
        result = eval_queue.get()
        log_validation_result(result, cfg.train.fitting_window, writer)
        final_val_result = result

    # save final model
    model_path = run_dir / "model_final.pt"
    torch.save(model.state_dict(), model_path)
    log.info(f"saved final model to {model_path}")

    # run test evaluation (same code path as validation)
    log.info("running test evaluation...")
    test_thread = start_eval(test_data, final_epoch)
    test_thread.join()
    test_result = eval_queue.get()
    log_validation_result(test_result, cfg.train.fitting_window, writer, prefix="test")

    # print final results with baselines
    log.info("\n" + "=" * 70)
    log.info("FINAL RESULTS (with baseline comparison)")
    log.info("=" * 70)
    if final_val_result is not None:
        print_results_table(final_val_result, "Validation", include_baseline=True)
    print_results_table(test_result, "Test", include_baseline=True)

    # helper to extract per-condition metrics
    def extract_condition_metrics(result: ValidationResult) -> dict:
        per_cond = {}
        for name in result.per_condition_mse:
            mses = result.per_condition_mse[name]
            maes = result.per_condition_mae[name]
            per_cond[name] = {
                "mean_mse": float(mses.mean()),
                "final_mse": float(mses[-1]),
                "mean_mae": float(maes.mean()),
                "final_mae": float(maes[-1]),
                "rollout_steps": len(mses),
            }
        return per_cond

    # save final metrics
    metrics = {
        "final_epoch": final_epoch,
        "was_terminated": terminate_flag["value"],
        "val": {
            "mean_mse": final_val_result.mean_mse if final_val_result else None,
            "mean_mae": final_val_result.mean_mae if final_val_result else None,
            "per_condition": extract_condition_metrics(final_val_result) if final_val_result else {},
        },
        "test": {
            "mean_mse": test_result.mean_mse,
            "mean_mae": test_result.mean_mae,
            "per_condition": extract_condition_metrics(test_result),
        },
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

    # print to stdout (visible even when logging redirected to file)
    print(f"run directory: {run_dir.resolve()}", flush=True)
    print(f"stdout.log: {run_dir.resolve() / 'stdout.log'}", flush=True)

    # run training
    was_terminated, _ = train(cfg, run_dir)

    # add completion/termination flag
    flag_file = "terminated" if was_terminated else "complete"
    with open(run_dir / flag_file, "w"):
        pass


if __name__ == "__main__":
    main()
