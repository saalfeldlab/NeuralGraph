"""zapbench stimulus encoding at bin resolution.

encodes raw ephys stimulus signals (condition, stimParam3, stimParam4, visual_velocity)
into a 26D feature vector at configurable time resolution. the encoding matches the
paper's stimuli_features when evaluated at frame resolution.

26D encoding (0-indexed dimensions):
- dims 0-1: gain (value, indicator)
- dims 2-3: dots (value, indicator)
- dims 4-5: flash (value, indicator)
- dims 6-8: taxis (left, right, indicator)
- dims 9-12: turning (velocity, sin, cos, indicator)
- dims 13-17: position (grating_type x3, delay, indicator)
- dim 18: open_loop (indicator only)
- dims 19-20: rotation (direction, indicator)
- dim 21: dark (indicator only)
- dims 22-25: specimen (all zeros for single specimen)
"""

import json
from pathlib import Path

import numpy as np
import tensorstore as ts
import torch


# ---------------------------------------------------------------------------
# zarr i/o helpers (reused from zapbench_data.py pattern)
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
    """read sampling_frequency_hz from ephys.zarr/zarr.json attributes."""
    zarr_json_path = Path(ephys_zarr_path) / "zarr.json"
    with open(zarr_json_path) as f:
        metadata = json.load(f)
    return float(metadata["attributes"]["sampling_frequency_hz"])


# ---------------------------------------------------------------------------
# raw stimulus loading
# ---------------------------------------------------------------------------


def load_raw_stimulus(
    ephys_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """load raw stimulus arrays from ephys.zarr.

    args:
        ephys_path: path to ephys.zarr directory

    returns:
        condition: (S,) int8 - condition index 0-8
        stimParam3: (S,) float32
        stimParam4: (S,) int8
        visual_velocity: (S,) float32
        imaging_sample_index: (T, 72) int64 - frame x z-slice sample indices
    """
    condition_path = str(Path(ephys_path) / "condition")
    sp3_path = str(Path(ephys_path) / "stimParam3")
    sp4_path = str(Path(ephys_path) / "stimParam4")
    vv_path = str(Path(ephys_path) / "visual_velocity")
    isi_path = str(Path(ephys_path) / "imaging_sample_index")

    condition = np.asarray(_open_zarr(condition_path).read().result())
    stimParam3 = np.asarray(_open_zarr(sp3_path).read().result())
    stimParam4 = np.asarray(_open_zarr(sp4_path).read().result())
    visual_velocity = np.asarray(_open_zarr(vv_path).read().result())
    imaging_sample_index = np.asarray(_open_zarr(isi_path).read().result())

    return condition, stimParam3, stimParam4, visual_velocity, imaging_sample_index


# ---------------------------------------------------------------------------
# core encoding function
# ---------------------------------------------------------------------------


def encode_stimulus_at_samples(
    condition: np.ndarray,
    stimParam3: np.ndarray,
    stimParam4: np.ndarray,
    sample_indices: np.ndarray,
    prev_sample_indices: np.ndarray | None = None,
) -> np.ndarray:
    """encode stimulus to 26D at specified sample indices.

    args:
        condition: (S,) int8 - raw condition array at 6kHz
        stimParam3: (S,) float32 - raw stimParam3 array at 6kHz
        stimParam4: (S,) int8 - raw stimParam4 array at 6kHz
        sample_indices: (T,) int64 - sample indices to encode at
        prev_sample_indices: (T,) int64 - previous sample indices for rotation
            derivative. if None, uses sample_indices - 1.

    returns:
        (T, 26) float32 array of stimulus encoding
    """
    T = len(sample_indices)
    encoding = np.zeros((T, 26), dtype=np.float32)

    # get values at sample times
    cond = condition[sample_indices]
    sp3 = stimParam3[sample_indices]
    sp4 = stimParam4[sample_indices]

    # GAIN (cond 0): dims 0-1
    mask = (cond == 0)
    encoding[mask, 0] = np.where(sp3[mask] == 1, -1.0, 1.0)
    encoding[mask, 1] = 1.0

    # DOTS (cond 1): dims 2-3
    mask = (cond == 1)
    encoding[mask, 2] = np.where((sp3[mask] == 90) & (sp4[mask] == 1), 1.0, -1.0)
    encoding[mask, 3] = 1.0

    # FLASH (cond 2): dims 4-5
    mask = (cond == 2)
    encoding[mask, 4] = np.where(sp3[mask] == 0, -1.0, 1.0)
    encoding[mask, 5] = 1.0

    # TAXIS (cond 3): dims 6-8
    mask = (cond == 3)
    encoding[mask, 6] = np.where(sp3[mask] == 0, -1.0, 1.0)
    encoding[mask, 7] = np.where(sp4[mask] == 0, -1.0, 1.0)
    encoding[mask, 8] = 1.0

    # TURNING (cond 4): dims 9-12
    mask = (cond == 4)
    encoding[mask, 9] = sp3[mask]  # velocity
    # direction sin/cos from stimParam4
    sp4_turning = sp4[mask]
    angles = np.zeros_like(sp4_turning, dtype=np.float32)
    angles[sp4_turning == -1] = 180
    angles[sp4_turning == 90] = 90
    angles[sp4_turning == -90] = 270
    # sp4=0 or sp4=1 -> 0 degrees (already 0)
    encoding[mask, 10] = np.sin(np.radians(angles))
    encoding[mask, 11] = np.cos(np.radians(angles))
    encoding[mask, 12] = 1.0

    # POSITION (cond 5): dims 13-17
    mask = (cond == 5)
    encoding[mask, 13] = (sp3[mask] == -1).astype(np.float32)
    encoding[mask, 14] = (sp3[mask] == 0).astype(np.float32)
    encoding[mask, 15] = (sp3[mask] == 1).astype(np.float32)
    encoding[mask, 16] = sp4[mask] / 10.0  # delay: 0,1,3,6,9 -> 0.0,0.1,0.3,0.6,0.9
    encoding[mask, 17] = 1.0

    # OPEN_LOOP (cond 6): dim 18
    mask = (cond == 6)
    encoding[mask, 18] = 1.0

    # ROTATION (cond 7): dims 19-20
    mask = (cond == 7)
    if prev_sample_indices is not None:
        sp3_prev = stimParam3[prev_sample_indices]
        delta = sp3[mask] - sp3_prev[mask]
        # handle wraparound (0-90 range)
        delta = np.where(delta > 45, delta - 90, delta)
        delta = np.where(delta < -45, delta + 90, delta)
        encoding[mask, 19] = np.where(delta > 0, 1.0, np.where(delta < 0, -1.0, 0.0))
    encoding[mask, 20] = 1.0

    # DARK (cond 8): dim 21
    mask = (cond == 8)
    encoding[mask, 21] = 1.0

    # dims 22-25 (specimen): all zeros (already initialized)

    return encoding


# ---------------------------------------------------------------------------
# frame-resolution encoding (for verification)
# ---------------------------------------------------------------------------


def load_stimulus_encoding_at_frames(
    ephys_path: str,
    time_slice: slice | None = None,
) -> np.ndarray:
    """load stimulus encoding at frame resolution (for testing).

    uses imaging_sample_index[:, 0] to get frame start times.
    returns (num_frames, 26) to compare against official encoding.

    args:
        ephys_path: path to ephys.zarr directory
        time_slice: optional slice along frame axis

    returns:
        (num_frames, 26) float32 array
    """
    condition, stimParam3, stimParam4, _, imaging_sample_index = load_raw_stimulus(
        ephys_path
    )

    # use first z-slice sample index as frame time
    sample_indices = imaging_sample_index[:, 0].astype(np.int64)
    if time_slice is not None:
        sample_indices = sample_indices[time_slice]

    # for rotation derivative, use previous frame's sample
    prev_indices = np.roll(sample_indices, 1)
    prev_indices[0] = sample_indices[0]  # no prev for first frame

    return encode_stimulus_at_samples(
        condition, stimParam3, stimParam4, sample_indices, prev_indices
    )


# ---------------------------------------------------------------------------
# bin-resolution encoding (production use)
# ---------------------------------------------------------------------------


def load_stimulus_encoding(
    ephys_path: str,
    bin_size_ms: float,
    time_slice: slice | None = None,
) -> torch.Tensor:
    """load stimulus encoding at specified bin resolution.

    args:
        ephys_path: path to ephys.zarr directory
        bin_size_ms: bin width in milliseconds (40.0 for production)
        time_slice: optional slice along frame axis to determine sample range

    returns:
        (num_bins, 26) float32 tensor
    """
    condition, stimParam3, stimParam4, _, imaging_sample_index = load_raw_stimulus(
        ephys_path
    )

    # read sampling frequency
    sampling_freq_hz = _read_sampling_frequency(ephys_path)
    samples_per_bin = int(bin_size_ms * sampling_freq_hz / 1000)

    # determine sample range from imaging_sample_index
    if time_slice is not None:
        start_idx = time_slice.start if time_slice.start is not None else 0
        stop_idx = time_slice.stop if time_slice.stop is not None else len(imaging_sample_index)
        sample_start = int(imaging_sample_index[start_idx, 0])
        sample_end = int(imaging_sample_index[stop_idx - 1, -1])
    else:
        sample_start = int(imaging_sample_index[0, 0])
        sample_end = int(imaging_sample_index[-1, -1])

    # compute bin sample indices (sample at bin start)
    num_bins = (sample_end - sample_start) // samples_per_bin
    sample_indices = sample_start + np.arange(num_bins, dtype=np.int64) * samples_per_bin
    prev_sample_indices = sample_indices - samples_per_bin
    prev_sample_indices = np.maximum(prev_sample_indices, 0)  # clamp

    encoding = encode_stimulus_at_samples(
        condition, stimParam3, stimParam4, sample_indices, prev_sample_indices
    )
    return torch.from_numpy(encoding)
