"""
latent zapbench: per-condition pre-interpolated data on CPU with prefetching.

data flow:
  startup: load sparse -> interpolate on GPU -> move dense to CPU
  prefetch thread: sample -> pinned buffer -> GPU transfer -> filled_queue
  main thread: filled_queue -> forward/backward -> return buffer to pool
"""

import random
import signal
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch


def seed_everything(seed: int = 42):
    """seed all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

from LatentEvolution.batch_prefetcher import BatchPrefetcher
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
    """pre-interpolated dense data for one condition, stored on CPU."""
    activity: torch.Tensor  # (T, N) float32, interpolated activity on CPU
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
    2. transfer to GPU and interpolate (fast)
    3. move dense result to CPU (saves GPU memory)

    args:
        traces_path: path to traces zarr array.
        ephys_path: path to ephys.zarr directory.
        bin_size_ms: bin width in milliseconds.
        split: data split configuration.
        split_type: which split to load ("train", "val", "test").
        fitting_window: number of time steps to predict into future.
        device: target device (GPU) for interpolation.

    returns:
        dict mapping condition name to ConditionData (dense tensors on CPU).
    """
    condition_data: dict[ConditionName, ConditionData] = {}
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

        # move to CPU to save GPU memory
        activity_cpu = activity_gpu.cpu()

        # free GPU memory
        del obs_times_gpu, obs_vals_gpu, counts_gpu, activity_gpu
        torch.cuda.empty_cache()

        # compute valid sampling range
        # valid_start: first bin where all neurons have data (after first obs)
        # valid_end: last valid start so window fits before end
        first_obs = obs_times[:, 0].max().item() + 1
        last_obs = obs_times[:, -1].min().item()
        valid_start = int(first_obs)
        valid_end = int(last_obs) - fitting_window + 1

        condition_data[cond_name] = ConditionData(
            activity=activity_cpu,
            valid_start=valid_start,
            valid_end=valid_end,
        )
        print(f"  {cond_name}: {activity_cpu.shape} -> CPU, valid=[{valid_start}, {valid_end})")

    return condition_data


# ---------------------------------------------------------------------------
# batch sampling
# ---------------------------------------------------------------------------


def sample_into_buffer(
    condition_data: dict[ConditionName, ConditionData],
    pinned_buffer: torch.Tensor,
    rng: torch.Generator,
) -> None:
    """sample batch from pre-interpolated data into pinned buffer.

    samples conditions proportionally to their valid start ranges, then
    samples random starts and copies into pinned buffer.

    args:
        condition_data: dict mapping condition name to ConditionData.
        pinned_buffer: (B, T, N) pinned CPU tensor to fill.
        rng: random number generator for reproducibility.
    """
    conds = list(condition_data.keys())
    batch_size = pinned_buffer.shape[0]
    fitting_window = pinned_buffer.shape[1]

    # sample conditions proportionally to valid start ranges
    weights = torch.tensor(
        [condition_data[c].weight for c in conds],
        dtype=torch.float,
    )
    cond_indices = torch.multinomial(
        weights, batch_size, replacement=True, generator=rng,
    )

    # copy samples into pinned buffer
    for b in range(batch_size):
        cond = conds[cond_indices[b].item()]
        data = condition_data[cond]

        # sample start position
        start = torch.randint(
            data.valid_start, data.valid_end, (1,), generator=rng,
        ).item()

        # copy into pinned buffer: (fitting_window, N)
        pinned_buffer[b] = data.activity[start:start + fitting_window, :]


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
        print("\nSIGUSR2 received - will terminate after current epoch")

    signal.signal(signal.SIGUSR2, handle_sigusr2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # enable TF32 for faster matmul/conv on Ampere+ GPUs
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("tf32 precision: enabled")

    # config
    data_cfg = DataConfig()
    train_cfg = TrainConfig()

    seed_everything(train_cfg.seed)

    # load training data
    print("loading training data...")
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
    print(f"total CPU memory: {total_bytes / 1e9:.2f} GB")

    # batches per epoch for 1x coverage
    batches_per_epoch = total_weight // train_cfg.batch_size
    print(f"\ntotal training samples: {total_weight}")
    print(f"batches_per_epoch for 1x coverage: {batches_per_epoch}")

    batches_per_epoch = 36

    # batch prefetcher with pinned buffer pool
    assert num_neurons is not None
    batch_shape = (train_cfg.batch_size, train_cfg.fitting_window, num_neurons)
    num_buffers = 3
    buffer_mb = batch_shape[0] * batch_shape[1] * batch_shape[2] * 4 / 1e6
    print(f"pinned buffers: {num_buffers} x {batch_shape}, {buffer_mb:.1f} MB each")

    # RNG for batch sampling (must be created before prefetcher for reproducibility)
    rng = torch.Generator(device="cpu")

    prefetcher = BatchPrefetcher(
        sample_fn=lambda buf: sample_into_buffer(train_data, buf, rng),
        transfer_fn=lambda buf: buf.to(device, non_blocking=True),
        batch_shape=batch_shape,
        num_buffers=num_buffers,
    )

    # model
    model_cfg = ModelConfig(num_neurons=num_neurons)
    model = EEDModel(model_cfg).to(device)
    print(f"\nmodel: {sum(p.numel() for p in model.parameters()):,} parameters")

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)

    print(f"\ntraining: {train_cfg.epochs} epochs, {batches_per_epoch} batches/epoch")

    # start prefetching for all epochs
    total_batches = train_cfg.epochs * batches_per_epoch
    prefetcher.start(total_batches)

    # chrome profiler (only record 5 steps to keep file small)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=2, warmup=1, active=5, repeat=1),
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for epoch in range(train_cfg.epochs):
            model.train()
            epoch_loss = torch.tensor(0.0, device=device)
            epoch_start = time.time()

            for _ in range(batches_per_epoch):
                batch = prefetcher.get()
                if batch is None:
                    break

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
            print(f"epoch {epoch}: loss={avg_loss:.4f}, time={epoch_time:.1f}s")

            # check for graceful termination
            if terminate_flag["value"]:
                print(f"\n=== graceful termination at epoch {epoch + 1} ===")
                prefetcher.stop()
                break

    prefetcher.stop()

    # save profile
    prof.export_chrome_trace("zapbench_profile.json")
    print("\nprofile saved to zapbench_profile.json")
    print("done")


if __name__ == "__main__":
    main()
