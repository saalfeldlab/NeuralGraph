"""verification script for stimulus encoding - run manually, not via make test.

compares our encoding against the official zapbench stimuli_features when
computed at frame resolution. also verifies shape and indicator exclusivity
at production bin resolution.

usage:
    python -m LatentEvolution.zapbench_stimulus_verify

note: named _verify.py (not _test.py) to avoid discovery by `make test`.
this script requires data access and takes significant time to run.

transition frames: the official encoding uses a different convention at frames
where stimulus parameters change (either condition boundaries or within-condition
parameter changes). our encoding samples at frame start (Z=0), while the official
appears to use the previous frame's value at transitions. since training excludes
these frames anyway (padding=1 in CONDITIONS), we verify only stable frames.
"""

import numpy as np

from LatentEvolution.zapbench_stimulus import (
    _open_zarr,
    load_raw_stimulus,
    load_stimulus_encoding,
    load_stimulus_encoding_at_frames,
)


# paths (local mount of zapbench data)
EPHYS_PATH = "/groups/saalfeld/home/kumarv4/repos/zapbench/ephys.zarr"
# official encoding from zapbench release
OFFICIAL_STIM_PATH = "gs://zapbench-release/volumes/20240930/stimuli_features"


def load_official_encoding() -> np.ndarray:
    """load official stimuli_features from GCS."""
    store = _open_zarr(OFFICIAL_STIM_PATH)
    return np.asarray(store.read().result())


def find_transition_frames(ephys_path: str) -> set[int]:
    """find frames where stimulus parameters change (to exclude from comparison).

    transition frames include:
    - frames where any parameter changes from prev frame end to current frame start
    - frames where any parameter changes within the frame (start != end)
    - the frame before any such transition
    """
    condition, sp3, sp4, _, isi = load_raw_stimulus(ephys_path)

    sample_indices_start = isi[:, 0].astype(np.int64)
    sample_indices_end = isi[:, -1].astype(np.int64)

    sp3_start = sp3[sample_indices_start]
    sp3_end = sp3[sample_indices_end]
    sp4_start = sp4[sample_indices_start]
    sp4_end = sp4[sample_indices_end]
    cond_start = condition[sample_indices_start]
    cond_end = condition[sample_indices_end]

    transition_frames: set[int] = {0}  # first frame has no previous
    for f in range(1, len(sample_indices_start)):
        # within-frame change
        if (sp3_start[f] != sp3_end[f] or
            sp4_start[f] != sp4_end[f] or
            cond_start[f] != cond_end[f]):
            transition_frames.add(f)
        # between-frame change (prev end to current start)
        if (sp3_start[f] != sp3_end[f - 1] or
            sp4_start[f] != sp4_end[f - 1] or
            cond_start[f] != cond_end[f - 1]):
            transition_frames.add(f)
            transition_frames.add(f - 1)  # also exclude previous frame

    return transition_frames


def verify_frame_resolution_matches_official() -> None:
    """encoding at frame resolution should match official for stable frames."""
    print("loading official encoding...")
    official = load_official_encoding()
    print(f"official shape: {official.shape}")

    print("computing our encoding at frame resolution...")
    our_encoding = load_stimulus_encoding_at_frames(EPHYS_PATH)
    print(f"our shape: {our_encoding.shape}")

    # find and exclude transition frames
    transition_frames = find_transition_frames(EPHYS_PATH)
    print(f"excluding {len(transition_frames)} transition frames")

    stable_mask = np.array(
        [f not in transition_frames for f in range(len(our_encoding))]
    )

    # compare stable frames only
    diff = np.abs(our_encoding[stable_mask] - official[stable_mask])
    max_diff = diff.max()
    num_mismatches = (diff > 1e-5).sum()

    print(f"max difference (stable frames): {max_diff}")
    print(f"mismatches (>1e-5): {num_mismatches} / {stable_mask.sum() * 26}")

    if num_mismatches > 0:
        # show first few mismatches
        idx = np.where(diff > 1e-5)
        stable_indices = np.where(stable_mask)[0]
        print("first mismatches:")
        for i in range(min(10, len(idx[0]))):
            frame = stable_indices[idx[0][i]]
            dim = idx[1][i]
            print(
                f"  frame {frame}, dim {dim}: "
                f"ours={our_encoding[frame, dim]:.4f}, "
                f"official={official[frame, dim]:.4f}"
            )

    assert num_mismatches == 0, f"encoding mismatch: {num_mismatches} values differ"
    print("PASSED: encoding matches official for stable frames")


def verify_shape_at_40ms() -> None:
    """check shape at 40ms bin resolution."""
    print("\nverifying shape at 40ms bin resolution...")
    encoding = load_stimulus_encoding(EPHYS_PATH, bin_size_ms=40.0)
    print(f"shape at 40ms: {encoding.shape}")
    assert encoding.shape[1] == 26, f"expected 26 dims, got {encoding.shape[1]}"
    # expected bins: ~43M samples / (6000 * 0.040) = ~180k bins
    # actual range depends on imaging_sample_index span
    print("PASSED: shape correct")


def verify_indicator_exclusivity() -> None:
    """exactly one condition indicator should be 1 at each timestep."""
    print("\nverifying indicator exclusivity...")
    encoding = load_stimulus_encoding(EPHYS_PATH, bin_size_ms=40.0)

    # indicator dimensions (0-indexed)
    indicator_dims = [1, 3, 5, 8, 12, 17, 18, 20, 21]
    indicators = encoding[:, indicator_dims].numpy()
    sums = indicators.sum(axis=1)

    num_bad = (np.abs(sums - 1.0) > 1e-5).sum()
    print(f"indicator sum != 1: {num_bad} / {len(sums)}")

    if num_bad > 0:
        bad_idx = np.where(np.abs(sums - 1.0) > 1e-5)[0]
        print(f"first bad indices: {bad_idx[:10]}")
        print(f"sums at those indices: {sums[bad_idx[:10]]}")

    assert num_bad == 0, f"indicator exclusivity violated: {num_bad} timesteps"
    print("PASSED: indicator exclusivity")


def main() -> None:
    """run all verifications."""
    verify_frame_resolution_matches_official()
    verify_shape_at_40ms()
    verify_indicator_exclusivity()
    print("\n=== ALL VERIFICATIONS PASSED ===")


if __name__ == "__main__":
    main()
