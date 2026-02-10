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
# concatenated training data
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


def load_concat_training_data(
    traces_path: str,
    ephys_path: str,
    bin_size_ms: float,
    split: DataSplit,
    split_type: Literal["train", "val", "test"],
    fitting_window: int,
    device: torch.device,
) -> ConcatTrainingData:
    """load and concatenate all conditions into a single GPU tensor.

    for each condition:
    1. load sparse data from zarr to CPU
    2. transfer to GPU and interpolate (needs headroom)
    3. build observation mask on GPU
    4. move activity (as float16) and mask to CPU
    5. concatenate on CPU, then transfer to GPU

    args:
        traces_path: path to traces zarr array.
        ephys_path: path to ephys.zarr directory.
        bin_size_ms: bin width in milliseconds.
        split: data split configuration.
        split_type: which split to load ("train", "val", "test").
        fitting_window: number of time steps to predict into future.
        device: target device (GPU) for interpolation and storage.

    returns:
        ConcatTrainingData with all conditions concatenated on GPU.
    """
    activities_cpu: list[torch.Tensor] = []
    masks_cpu: list[torch.Tensor] = []
    valid_starts: list[int] = []
    valid_ends: list[int] = []
    weights: list[int] = []

    ranges = split.get_ranges(split_type)
    offset = 0
    num_neurons = None

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

        # build observation mask on GPU: (T, N) bool
        obs_mask_gpu = build_obs_mask(obs_times, num_bins, device)

        # move to CPU as float16 (frees GPU for next condition)
        activity_cpu = activity_gpu.half().cpu()
        obs_mask_cpu = obs_mask_gpu.cpu()

        # free GPU memory
        del obs_times_gpu, obs_vals_gpu, counts_gpu, activity_gpu, obs_mask_gpu
        torch.cuda.empty_cache()

        # compute valid sampling range in global coordinates
        first_obs = int(obs_times[:, 0].max().item()) + 1
        last_obs = int(obs_times[:, -1].min().item())
        local_valid_start = first_obs
        local_valid_end = last_obs - fitting_window + 1
        weight = max(0, local_valid_end - local_valid_start)

        valid_starts.append(offset + local_valid_start)
        valid_ends.append(offset + local_valid_end)
        weights.append(weight)

        T = activity_cpu.shape[0]
        num_neurons = activity_cpu.shape[1]
        log.info(f"  {cond_name}: T={T}, valid=[{offset + local_valid_start}, {offset + local_valid_end}), weight={weight}")

        activities_cpu.append(activity_cpu)
        masks_cpu.append(obs_mask_cpu)
        offset += T

    # concatenate on CPU, then transfer to GPU (single copy)
    log.info("  concatenating on CPU and transferring to GPU...")
    all_activity_cpu = torch.cat(activities_cpu, dim=0)
    all_mask_cpu = torch.cat(masks_cpu, dim=0)
    del activities_cpu, masks_cpu

    # compute stats on CPU (GPU sum() allocates int64 intermediate = 8x memory)
    activity_gb = all_activity_cpu.numel() * all_activity_cpu.element_size() / 1e9
    mask_gb = all_mask_cpu.numel() * all_mask_cpu.element_size() / 1e9
    obs_density = all_mask_cpu.sum().item() / all_mask_cpu.numel()
    log.info(f"  activity: {activity_gb:.2f} GB (float16), mask: {mask_gb:.2f} GB (bool)")
    log.info(f"  observation density: {obs_density:.2%}")

    # transfer to GPU
    all_activity = all_activity_cpu.to(device)
    del all_activity_cpu
    all_mask = all_mask_cpu.to(device)
    del all_mask_cpu

    assert num_neurons is not None
    return ConcatTrainingData(
        activity=all_activity,
        obs_mask=all_mask,
        valid_starts=torch.tensor(valid_starts, device=device, dtype=torch.long),
        valid_ends=torch.tensor(valid_ends, device=device, dtype=torch.long),
        weights=torch.tensor(weights, device=device, dtype=torch.float),
        num_neurons=num_neurons,
        num_conds=len(ranges),
    )


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
    train_data = load_concat_training_data(
        data_cfg.traces_path, data_cfg.ephys_path, data_cfg.bin_size_ms,
        split, "train", train_cfg.fitting_window, device,
    )

    # # free CUDA graphs from compiled interpolation (~10 GB)
    # doing this really slows down compilation of the train step, so don't do it
    # torch._dynamo.reset()
    # torch.cuda.empty_cache()

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
