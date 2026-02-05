"""benchmark zapbench gpu interpolation.

cpu reference (381, 71721) @ 2.6% observed, 10 frames, bin_size=24ms:
  load_staggered_activity: 585.6ms
  interpolate (cpu eager): 1751.6ms
  interpolate (cpu compiled): 1165.7ms

gpu dense vs sparse (15234, 71721) @ 2.6% observed, 400 frames, bin_size=24ms:
  load:     2098.6ms (dense) vs 1322.7ms (sparse) -> 1.6x
  transfer: 1043.8ms (dense) vs   47.5ms (sparse) -> 22.0x
  interp:     99.3ms (dense) vs   30.2ms (sparse) -> 3.3x
  total:    3241.7ms (dense) vs 1400.3ms (sparse) -> 2.3x
  peak mem: 30219MB  (dense) vs  8994MB  (sparse) -> 3.4x
  max diff: 1.19e-07
"""

import time

import torch

from LatentEvolution.zapbench import (
    load_sparse_activity,
    interpolate_sparse_compiled,
)

TRACES_PATH = "gs://zapbench-release/volumes/20240930/traces"
ACQ_PATH = "/groups/saalfeld/home/kumarv4/repos/zapbench/output.zarr/cell_acquisition_ms"

BIN_SIZE_MS = 24.0
FRAME_PERIOD_MS = 914.0
NUM_FRAMES = 3000
NUM_TRIALS = 3


def _bench(fn, num_trials: int = NUM_TRIALS) -> tuple:
    """run fn num_trials times and return the mean elapsed time in ms."""
    times = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    mean_ms = 1000.0 * sum(times) / len(times)
    return mean_ms, result


def _gpu_mem_mb() -> tuple[float, float]:
    """return (allocated, reserved) gpu memory in MB."""
    allocated = torch.cuda.memory_allocated() / (1024 * 1024)
    reserved = torch.cuda.memory_reserved() / (1024 * 1024)
    return allocated, reserved


def _print_gpu_mem(label: str):
    """print current gpu memory usage."""
    alloc, reserved = _gpu_mem_mb()
    print(f"  gpu mem [{label}]: {alloc:.1f}MB allocated, {reserved:.1f}MB reserved")


LOAD_KWARGS = dict(
    traces_path=TRACES_PATH,
    acq_path=ACQ_PATH,
    bin_size_ms=BIN_SIZE_MS,
    frame_period_ms=FRAME_PERIOD_MS,
    time_slice=slice(0, NUM_FRAMES),
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "this benchmark requires a gpu"
    print(f"device: {device}")
    print(f"trials: {NUM_TRIALS}")

    torch.cuda.reset_peak_memory_stats()
    _print_gpu_mem("baseline")

    # ===================== sparse path =====================
    print("\n--- sparse (load -> sparse cpu -> gpu -> searchsorted) ---")

    t0 = time.perf_counter()
    obs_times, obs_vals, counts, num_bins = load_sparse_activity(**LOAD_KWARGS)
    t_load = (time.perf_counter() - t0) * 1000.0
    print(f"load_sparse_activity: {t_load:.1f}ms")
    print(f"  obs_times shape: {obs_times.shape} (K={obs_times.shape[1]})")
    print(f"  num_bins: {num_bins}")

    # transfer sparse to gpu
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    obs_times_gpu = obs_times.to(device)
    obs_vals_gpu = obs_vals.to(device)
    counts_gpu = counts.to(device)
    torch.cuda.synchronize()
    t_transfer = (time.perf_counter() - t0) * 1000.0
    print(f"cpu -> gpu transfer: {t_transfer:.1f}ms")
    _print_gpu_mem("after transfer")

    # warmup
    _ = interpolate_sparse_compiled(obs_times_gpu, obs_vals_gpu, counts_gpu, num_bins)
    torch.cuda.synchronize()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t_interp, _ = _bench(
        lambda: interpolate_sparse_compiled(
            obs_times_gpu, obs_vals_gpu, counts_gpu, num_bins,
        ),
    )
    print(f"interpolate (gpu compiled): {t_interp:.1f}ms")
    peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    _print_gpu_mem("after interp")
    print(f"  gpu mem [peak]: {peak:.1f}MB")

    print(f"\ntotal: {t_load + t_transfer + t_interp:.1f}ms")


if __name__ == "__main__":
    main()
