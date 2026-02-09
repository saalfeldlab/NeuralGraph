"""load zapbench calcium data and interpolate staggered observations.

sparse representation: zarr -> (N, K) sparse -> searchsorted interpolation.
at 2.6% observed density: 22x faster transfer, 3.3x faster interp, 3.4x less
peak gpu memory.

data format (ephys.zarr):
- traces: (T, N) calcium fluorescence values
- cell_ephys_index: (T, N) sample indices at sampling_frequency_hz (e.g. 6kHz)
- sampling_frequency_hz: stored in zarr.json attributes
"""

import json
from pathlib import Path
from typing import Literal

import numpy as np
import tensorstore as ts
import torch
from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# condition metadata and data split config
# ---------------------------------------------------------------------------

ConditionName = Literal[
    "gain", "dots", "flash", "taxis", "turning",
    "position", "open_loop", "rotation", "dark",
]


class Condition(BaseModel):
    """static condition metadata."""
    name: ConditionName
    offset: tuple[int, int]  # [start, end) along T dimension
    padding: int             # timesteps excluded at start/end

    model_config = ConfigDict(extra="forbid")


# constant list derived from zapbench CONDITION_OFFSETS
CONDITIONS: list[Condition] = [
    Condition(name="gain", offset=(0, 649), padding=1),
    Condition(name="dots", offset=(649, 2422), padding=1),
    Condition(name="flash", offset=(2422, 3078), padding=1),
    Condition(name="taxis", offset=(3078, 3735), padding=1),
    Condition(name="turning", offset=(3735, 5047), padding=1),
    Condition(name="position", offset=(5047, 5638), padding=1),
    Condition(name="open_loop", offset=(5638, 6623), padding=1),
    Condition(name="rotation", offset=(6623, 7279), padding=1),
    Condition(name="dark", offset=(7279, 7879), padding=1),
]


class ConditionSplit(BaseModel):
    """split for one condition (references by name)."""
    condition_name: ConditionName
    train: tuple[int, int] | None = None  # [start, end) relative to padded start
    val: tuple[int, int] | None = None
    test: tuple[int, int] | None = None

    model_config = ConfigDict(extra="forbid")


class DataSplit(BaseModel):
    """zapbench data split across all conditions."""
    conditions: list[ConditionSplit]

    model_config = ConfigDict(extra="forbid")


# default split computed from zapbench constants:
# - train conditions: 70% train, 10% val, 20% test
# - holdout (taxis): test only
# padded_len = (end - start) - 2 * padding
DEFAULT_SPLIT = DataSplit(conditions=[
    # gain: padded_len=647, train=453, val=65, test=129
    ConditionSplit(condition_name="gain", train=(0, 453), val=(453, 518), test=(518, 647)),
    # dots: padded_len=1771, train=1240, val=177, test=354
    ConditionSplit(condition_name="dots", train=(0, 1240), val=(1240, 1417), test=(1417, 1771)),
    # flash: padded_len=654, train=458, val=65, test=131
    ConditionSplit(condition_name="flash", train=(0, 458), val=(458, 523), test=(523, 654)),
    # taxis (holdout): padded_len=655, test only
    ConditionSplit(condition_name="taxis", train=None, val=None, test=(0, 655)),
    # turning: padded_len=1310, train=917, val=131, test=262
    ConditionSplit(condition_name="turning", train=(0, 917), val=(917, 1048), test=(1048, 1310)),
    # position: padded_len=589, train=412, val=59, test=118
    ConditionSplit(condition_name="position", train=(0, 412), val=(412, 471), test=(471, 589)),
    # open_loop: padded_len=983, train=688, val=98, test=197
    ConditionSplit(condition_name="open_loop", train=(0, 688), val=(688, 786), test=(786, 983)),
    # rotation: padded_len=654, train=458, val=65, test=131
    ConditionSplit(condition_name="rotation", train=(0, 458), val=(458, 523), test=(523, 654)),
    # dark: padded_len=598, train=418, val=60, test=120
    ConditionSplit(condition_name="dark", train=(0, 418), val=(418, 478), test=(478, 598)),
])


# ---------------------------------------------------------------------------
# zarr i/o helpers
# ---------------------------------------------------------------------------

def _make_kvstore(path: str) -> dict:
    """build tensorstore kvstore spec from a local or gs:// path."""
    if path.startswith("gs://"):
        without_scheme = path[len("gs://"):]
        bucket, _, prefix = without_scheme.partition("/")
        return {"driver": "gcs", "bucket": bucket, "path": prefix}
    return {"driver": "file", "path": path}


def _open_zarr(path: str) -> ts.TensorStore:
    """open a zarr array via tensorstore (local, NFS, or GCS).

    tries zarr v2 first, falls back to zarr v3 if v2 metadata is missing.
    """
    kvstore = _make_kvstore(path)
    try:
        return ts.open({"driver": "zarr", "kvstore": kvstore}).result()
    except ValueError:
        return ts.open({"driver": "zarr3", "kvstore": kvstore}).result()


def _read_sampling_frequency(ephys_zarr_path: str) -> float:
    """read sampling_frequency_hz from ephys.zarr/zarr.json attributes.

    args:
        ephys_zarr_path: path to ephys.zarr directory (parent of cell_ephys_index)

    returns:
        sampling frequency in Hz (e.g. 6000.0)
    """
    zarr_json_path = Path(ephys_zarr_path) / "zarr.json"
    with open(zarr_json_path) as f:
        metadata = json.load(f)
    return float(metadata["attributes"]["sampling_frequency_hz"])


def _load_zarr_arrays(
    traces_path: str,
    ephys_path: str,
    time_slice: slice | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """read traces and cell_ephys_index from zarr, return (T, N) arrays.

    if time_slice exceeds array bounds, it is clamped to valid range.

    args:
        traces_path: path to traces zarr array (T, N)
        ephys_path: path to ephys.zarr directory containing cell_ephys_index
        time_slice: optional slice along T axis

    returns:
        traces_np: (T, N) float array of calcium traces
        acq_ms: (T, N) float array of acquisition times in milliseconds
        T: number of timepoints loaded
    """
    traces_store = _open_zarr(traces_path)

    # cell_ephys_index is stored as (T, N) - no transpose needed
    cell_index_path = str(Path(ephys_path) / "cell_ephys_index")
    cell_index_store = _open_zarr(cell_index_path)

    # read sampling frequency from zarr.json
    sampling_freq_hz = _read_sampling_frequency(ephys_path)

    # get array dimensions
    total_frames = traces_store.shape[0]

    # clamp time_slice to valid bounds
    if time_slice is not None:
        start = time_slice.start if time_slice.start is not None else 0
        stop = time_slice.stop if time_slice.stop is not None else total_frames
        start = max(0, min(start, total_frames))
        stop = max(start, min(stop, total_frames))
        time_slice = slice(start, stop)

    # load arrays - cell_ephys_index is already (T, N)
    if time_slice is not None:
        traces_np = traces_store[time_slice, :].read().result()
        cell_index_np = cell_index_store[time_slice, :].read().result()
    else:
        traces_np = traces_store.read().result()
        cell_index_np = cell_index_store.read().result()

    T, N = traces_np.shape
    assert cell_index_np.shape == (T, N), (
        f"traces shape {traces_np.shape} != cell_ephys_index shape {cell_index_np.shape}"
    )

    # convert sample indices to milliseconds
    acq_ms = np.asarray(cell_index_np, dtype=np.float64) * 1000.0 / sampling_freq_hz

    return np.asarray(traces_np), acq_ms, T


def _compute_bin_indices(
    acq_ms: np.ndarray,
    bin_size_ms: float,
) -> tuple[np.ndarray, int]:
    """compute bin indices from acquisition times in milliseconds.

    args:
        acq_ms: (T, N) array of acquisition times in milliseconds
        bin_size_ms: bin width in milliseconds

    returns:
        bin_indices: (T, N) int64 array of bin indices, clamped to [0, num_bins-1].
        num_bins: total number of output time bins.
    """
    # compute bin indices from absolute times
    bin_indices = np.floor(acq_ms / bin_size_ms).astype(np.int64)

    # shift to start at 0 (relative to first observation in this slice)
    min_bin = bin_indices.min()
    bin_indices = bin_indices - min_bin

    num_bins = int(bin_indices.max() + 1)
    return bin_indices, num_bins


# ---------------------------------------------------------------------------
# sparse path: zarr -> (N, K) sparse -> searchsorted interpolation
# ---------------------------------------------------------------------------

def load_sparse_activity(
    traces_path: str,
    ephys_path: str,
    bin_size_ms: float,
    time_slice: slice | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """load traces and cell_ephys_index from zarr directly into sparse representation.

    skips the intermediate dense (num_bins, N) matrix entirely. each
    raw observation (frame, neuron) maps to a bin index; we pack these
    per-neuron into (N, K_max) padded arrays sorted by bin time.

    since each frame contributes exactly one observation per neuron,
    K_max == T (number of loaded frames) and every neuron has the same
    count. padding is only needed if frames are subsetted unevenly,
    which doesn't happen in practice.

    args:
        traces_path: path to zarr array, shape (T, N).
        ephys_path: path to ephys.zarr directory containing cell_ephys_index.
        bin_size_ms: bin width in milliseconds.
        time_slice: optional slice along T axis.

    returns:
        obs_times: (N, K_max) long cpu tensor, sorted bin indices.
        obs_vals: (N, K_max) float32 cpu tensor, trace values.
        counts: (N,) long cpu tensor, observations per neuron.
        num_bins: total number of output time bins.
    """
    traces_np, acq_ms, T = _load_zarr_arrays(traces_path, ephys_path, time_slice)
    N = traces_np.shape[1]
    bin_indices, num_bins = _compute_bin_indices(acq_ms, bin_size_ms)

    # bin_indices is (T, N) — each frame gives one obs per neuron.
    # transpose to (N, T) for per-neuron layout, already sorted by
    # frame order which is monotonically increasing in time.
    obs_times = torch.from_numpy(
        np.ascontiguousarray(bin_indices.T, dtype=np.int64),
    )  # (N, T)
    obs_vals = torch.from_numpy(
        np.ascontiguousarray(traces_np.T, dtype=np.float32),
    )  # (N, T)

    # every neuron has exactly T observations
    counts = torch.full((N,), T, dtype=torch.long, device="cpu")

    return obs_times, obs_vals, counts, num_bins


def load_and_interpolate(
    traces_path: str,
    ephys_path: str,
    bin_size_ms: float,
    time_slice: slice | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """load zarr data and produce interpolated dense activity on device.

    goes zarr -> sparse (N, K) on cpu -> transfer to device ->
    searchsorted interpolation -> (num_bins, N) dense result.
    never materializes the full dense staggered matrix on cpu or gpu.

    args:
        traces_path: path to zarr array, shape (T, N).
        ephys_path: path to ephys.zarr directory containing cell_ephys_index.
        bin_size_ms: bin width in milliseconds.
        time_slice: optional slice along T axis.
        device: target device for interpolation and output.

    returns:
        (num_bins, N) float32 tensor on device, fully interpolated.
    """
    obs_times, obs_vals, counts, num_bins = load_sparse_activity(
        traces_path, ephys_path, bin_size_ms, time_slice,
    )
    obs_times = obs_times.to(device)
    obs_vals = obs_vals.to(device)
    counts = counts.to(device)
    return interpolate_sparse_compiled(obs_times, obs_vals, counts, num_bins)


def interpolate_sparse(
    obs_times: torch.Tensor,
    obs_vals: torch.Tensor,
    counts: torch.Tensor,
    T: int,
) -> torch.Tensor:
    """interpolate from padded sparse representation using searchsorted.

    for each target time t in [0, T) and each neuron n, finds the
    bracketing observations via binary search and linearly interpolates.
    at boundaries (before first / after last observation), the nearest
    observation is used (constant extrapolation).

    args:
        obs_times: (N, K_max) long tensor, sorted observation times.
            padded entries must be T+1.
        obs_vals: (N, K_max) float tensor, values at observation times.
        counts: (N,) long tensor, number of real observations per neuron.
        T: number of output time bins.

    returns:
        (T, N) float tensor with all time bins filled by interpolation.
    """
    device = obs_times.device
    N, K_max = obs_times.shape

    # target times: (N, T) for batched searchsorted
    target = torch.arange(T, device=device, dtype=torch.long)
    target_expanded = target.unsqueeze(0).expand(N, T).contiguous()  # (N, T)

    # searchsorted(right=False): index of first obs_time >= target.
    # padding at T+1 is beyond all targets, so it's never selected
    # unless the target is past the last real observation.
    idx_hi = torch.searchsorted(obs_times, target_expanded)  # (N, T)
    idx_lo = idx_hi - 1

    # clamp to valid observation range [0, count-1]
    max_idx = (counts.unsqueeze(1) - 1).clamp(min=0)  # (N, 1)
    idx_lo = torch.clamp(idx_lo, min=0)
    idx_lo = torch.min(idx_lo, max_idx)
    idx_hi = torch.clamp(idx_hi, min=0)
    idx_hi = torch.min(idx_hi, max_idx)

    # gather times and values at lo/hi
    t_lo = obs_times.gather(1, idx_lo)  # (N, T)
    t_hi = obs_times.gather(1, idx_hi)  # (N, T)
    v_lo = obs_vals.gather(1, idx_lo)   # (N, T)
    v_hi = obs_vals.gather(1, idx_hi)   # (N, T)

    # interpolation weight
    span = (t_hi - t_lo).float()
    offset = (target_expanded - t_lo).float()
    w = torch.where(span > 0, offset / span, torch.zeros_like(span))

    result = (1.0 - w) * v_lo + w * v_hi  # (N, T)
    return result.T  # (T, N)


interpolate_sparse_compiled = torch.compile(
    interpolate_sparse, mode="reduce-overhead", fullgraph=True,
)
