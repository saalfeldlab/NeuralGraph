"""
latent zapbench: per-condition pre-interpolated data on GPU.

data flow:
  startup: load sparse -> interpolate on GPU -> store on CPU (headroom)
           -> after all conditions loaded, transfer all to GPU
  training: sample on GPU -> forward/backward
"""

import logging
import random
import signal
from dataclasses import dataclass
from typing import Literal

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
    interpolate_sparse_compiled,
)
from LatentEvolution.zapbench_config import (
    ConditionName,
    DataConfig,
    DataSplit,
    ModelConfig,
    TrainConfig,
)
from LatentEvolution.zapbench_model import EEDModel


# ---------------------------------------------------------------------------
# per-condition data structure
# ---------------------------------------------------------------------------


@dataclass
class ConditionData:
    """pre-interpolated dense data for one condition, stored on GPU."""
    activity: torch.Tensor  # (T, N) float32, interpolated activity on GPU
    valid_start: int        # min valid start for sampling
    valid_end: int          # max valid start (exclusive)

    @property
    def weight(self) -> int:
        """sampling weight = number of valid start positions."""
        return max(0, self.valid_end - self.valid_start)


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------


def load_all_conditions(
    traces_path: str,
    ephys_path: str,
    bin_size_ms: float,
    split: DataSplit,
    split_type: Literal["train", "val", "test"],
    fitting_window: int,
    device: torch.device,
) -> dict[ConditionName, ConditionData]:
    """load and pre-interpolate data for all conditions.

    for each condition:
    1. load sparse data from zarr
    2. transfer to GPU and interpolate (needs headroom)
    3. move dense result to CPU temporarily
    4. after all conditions loaded, transfer all to GPU

    args:
        traces_path: path to traces zarr array.
        ephys_path: path to ephys.zarr directory.
        bin_size_ms: bin width in milliseconds.
        split: data split configuration.
        split_type: which split to load ("train", "val", "test").
        fitting_window: number of time steps to predict into future.
        device: target device (GPU) for interpolation and storage.

    returns:
        dict mapping condition name to ConditionData (dense tensors on GPU).
    """
    # first pass: interpolate each condition on GPU, store on CPU
    cpu_data: dict[ConditionName, tuple[torch.Tensor, int, int]] = {}
    ranges = split.get_ranges(split_type)

    for cond_name, abs_start, abs_end in ranges:
        # load sparse data to CPU
        obs_times, obs_vals, counts, num_bins = load_sparse_activity(
            traces_path, ephys_path, bin_size_ms,
            time_slice=slice(abs_start, abs_end),
        )

        # transfer to GPU for fast interpolation
        obs_times_gpu = obs_times.to(device)
        obs_vals_gpu = obs_vals.to(device)
        counts_gpu = counts.to(device)

        # interpolate on GPU: (T, N)
        activity_gpu = interpolate_sparse_compiled(
            obs_times_gpu, obs_vals_gpu, counts_gpu, num_bins,
        )

        # move to CPU temporarily (frees GPU for next condition's interpolation)
        activity_cpu = activity_gpu.cpu()

        # free GPU memory
        del obs_times_gpu, obs_vals_gpu, counts_gpu, activity_gpu
        torch.cuda.empty_cache()

        # compute valid sampling range
        first_obs = obs_times[:, 0].max().item() + 1
        last_obs = obs_times[:, -1].min().item()
        valid_start = int(first_obs)
        valid_end = int(last_obs) - fitting_window + 1

        cpu_data[cond_name] = (activity_cpu, valid_start, valid_end)
        log.info(f"  {cond_name}: {activity_cpu.shape} interpolated, valid=[{valid_start}, {valid_end})")

    # second pass: transfer all to GPU
    log.info("  transferring all conditions to GPU...")
    condition_data: dict[ConditionName, ConditionData] = {}
    for cond_name, (activity_cpu, valid_start, valid_end) in cpu_data.items():
        activity_gpu = activity_cpu.to(device)
        condition_data[cond_name] = ConditionData(
            activity=activity_gpu,
            valid_start=valid_start,
            valid_end=valid_end,
        )

    torch.cuda.empty_cache()
    return condition_data


# ---------------------------------------------------------------------------
# batch sampling
# ---------------------------------------------------------------------------


class BatchSampler:
    """vectorized batch sampler on GPU with precomputed constants."""

    def __init__(
        self,
        condition_data: dict[ConditionName, ConditionData],
        batch_size: int,
        fitting_window: int,
        device: torch.device,
    ):
        """initialize sampler with precomputed constants on GPU.

        args:
            condition_data: dict mapping condition name to ConditionData (on GPU).
            batch_size: number of samples per batch.
            fitting_window: number of time steps per sample.
            device: GPU device for sampling.
        """
        self.condition_data = condition_data
        self.batch_size = batch_size
        self.fitting_window = fitting_window
        self.device = device

        # precompute constants on GPU
        conds = list(condition_data.keys())
        self.num_conds = len(conds)
        self.cond_data_list = [condition_data[c] for c in conds]  # list for O(1) access
        self.weights = torch.tensor(
            [d.weight for d in self.cond_data_list],
            dtype=torch.float,
            device=device,
        )
        self.time_offsets = torch.arange(fitting_window, device=device)

        # get num_neurons from first condition
        self.num_neurons = self.cond_data_list[0].activity.shape[1]

    def sample(self, rng: torch.Generator) -> torch.Tensor:
        """sample batch on GPU using vectorized gather.

        uses sort + bincount to avoid per-condition CPU syncs.

        args:
            rng: random number generator on GPU.

        returns:
            batch: (B, T, N) GPU tensor.
        """
        # allocate output on GPU
        batch = torch.empty(
            self.batch_size, self.fitting_window, self.num_neurons,
            dtype=torch.float32, device=self.device,
        )

        # sample which condition each batch element comes from
        cond_indices = torch.multinomial(
            self.weights, self.batch_size, replacement=True, generator=rng,
        )

        # sort to group by condition - sort_perm maps sorted index -> original index
        _, sort_perm = cond_indices.sort()

        # get unique conditions and counts - sorted=True so offsets are cumulative
        unique_conds, counts = torch.unique(cond_indices, return_counts=True, sorted=True)

        # one sync to get both to CPU
        unique_conds_list = unique_conds.tolist()
        counts_list = counts.tolist()

        # process only conditions that appear (no zero-count check needed)
        offset = 0
        for c_idx, count in zip(unique_conds_list, counts_list):
            # slice into sorted permutation to get original batch indices
            batch_indices = sort_perm[offset:offset + count]

            data = self.cond_data_list[c_idx]

            # vectorized: sample all starts at once (on GPU)
            starts = torch.randint(
                data.valid_start, data.valid_end, (count,),
                device=self.device, generator=rng,
            )

            # vectorized: compute all gather indices at once
            gather_indices = starts[:, None] + self.time_offsets  # (count, T)

            # vectorized: one gather for all samples (GPU memory access)
            samples = data.activity[gather_indices]  # (count, T, N)

            # vectorized: one write
            batch[batch_indices] = samples

            offset += count

        return batch


# ---------------------------------------------------------------------------
# train step (similar to latent_stag_interp.py)
# ---------------------------------------------------------------------------

@torch.compile(mode="reduce-overhead", fullgraph=True)
def train_step(
    model: EEDModel,
    batch: torch.Tensor,  # (B, T, N)
) -> torch.Tensor:
    """training step: encode, evolve, decode, compute loss.

    follows latent_stag_interp.py pattern:
    - encode initial state
    - loop through timesteps: decode, compute loss, evolve

    args:
        model: EED model
        batch: (B, T, N) neural activity sequence

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

        # compute loss against ground truth
        loss = loss + torch.nn.functional.mse_loss(x_pred, batch[:, t, :])

        # evolve to next latent (no stimulus)
        z = model.evolve(z)

    return loss


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    """training loop with per-condition sparse data."""
    import time

    # signal handling for graceful termination
    terminate_flag = {"value": False}

    def handle_sigusr2(signum, frame):
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

    # load training data
    log.info("loading training data...")
    split = DataSplit()
    train_data = load_all_conditions(
        data_cfg.traces_path, data_cfg.ephys_path, data_cfg.bin_size_ms,
        split, "train", train_cfg.fitting_window, device,
    )
    total_weight = 0
    total_bytes = 0
    num_neurons = None
    for cond, data in train_data.items():
        total_weight += data.weight
        total_bytes += data.activity.numel() * data.activity.element_size()
        num_neurons = data.activity.shape[1]
    log.info(f"total GPU memory for data: {total_bytes / 1e9:.2f} GB")

    # batches per epoch for 1x coverage
    batches_per_epoch = total_weight // train_cfg.batch_size
    log.info(f"total training samples: {total_weight}")
    log.info(f"batches_per_epoch for 1x coverage: {batches_per_epoch}")

    # batches_per_epoch = 36

    # RNG for batch sampling (on GPU)
    rng = torch.Generator(device=device)

    # batch sampler on GPU with precomputed constants
    assert num_neurons is not None
    sampler = BatchSampler(train_data, train_cfg.batch_size, train_cfg.fitting_window, device)
    batch_shape = (train_cfg.batch_size, train_cfg.fitting_window, num_neurons)
    batch_mb = batch_shape[0] * batch_shape[1] * batch_shape[2] * 4 / 1e6
    log.info(f"batch shape: {batch_shape}, {batch_mb:.1f} MB per batch")

    # model
    model_cfg = ModelConfig(num_neurons=num_neurons)
    model = EEDModel(model_cfg).to(device)
    log.info(f"model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)

    log.info(f"training: {train_cfg.epochs} epochs, {batches_per_epoch} batches/epoch")


    # chrome profiler (only record 5 steps during epoch 1)
    # epoch 0 = steps 0-35, epoch 1 = steps 36-71
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=38, warmup=1, active=5, repeat=1),
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for epoch in range(train_cfg.epochs):
            model.train()
            epoch_loss = torch.tensor(0.0, device=device)
            epoch_start = time.time()

            for _ in range(batches_per_epoch):
                # sample batch on GPU (no CPU->GPU transfer needed)
                with torch.profiler.record_function("sample"):
                    batch = sampler.sample(rng)

                optimizer.zero_grad()

                with torch.profiler.record_function("forward"):
                    loss = train_step(model, batch)

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

            # check for graceful termination
            if terminate_flag["value"]:
                log.info(f"=== graceful termination at epoch {epoch + 1} ===")
                break

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
