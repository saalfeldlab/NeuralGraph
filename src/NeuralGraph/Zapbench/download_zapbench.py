#!/usr/bin/env python3
"""
Export ZAPBench to NumPy arrays for PyTorch/PyG:
  - traces.npy       (ΔF/F activity, [T, N])
  - conditions.npy   (stimulus/condition id per timestep, [T], int32 in [0..8])
  - positions.npy    (3D centroids from segmentation labels, [K, 3], float64)
  - label_ids.npy    (label ids corresponding to rows of positions.npy, [K], int32)

Usage (default GCS bucket paths):
  python zapbench_export_numpy.py --outdir ./zapbench_numpy

To force a local copy (e.g., after `gsutil -m cp -r`), pass local paths:
  python zapbench_export_numpy.py --traces ./traces --segmentation ./segmentation --outdir ./zapbench_numpy
"""

import os
import argparse
import numpy as np
import tensorstore as ts

# ZAPBench public bucket (ICLR 2025 release)
BUCKET_ROOT = "gs://zapbench-release/volumes/20240930"
DEFAULT_TRACES = f"{BUCKET_ROOT}/traces"
DEFAULT_SEG = f"{BUCKET_ROOT}/segmentation"

def open_ts(gspath: str):
    """Open a Zarr store with TensorStore, trying v3 then v2."""
    try:
        return ts.open({'open': True, 'driver': 'zarr3', 'kvstore': gspath}).result()
    except Exception:
        return ts.open({'open': True, 'driver': 'zarr', 'kvstore': gspath}).result()

def export_traces_and_conditions(traces_store: str, outdir: str):
    """Read ΔF/F traces [T, N] and create conditions.npy via official bounds."""
    print(f"[TRACES] Opening: {traces_store}")
    ds_traces = open_ts(traces_store)
    print(f"[TRACES] Schema: {ds_traces.schema}")
    traces = ds_traces.read().result()   # shape [T, N]
    T, N = traces.shape
    print(f"[TRACES] Loaded matrix: T={T}, N={N}")

    # Build conditions via zapbench helper
    from zapbench import constants, data_utils
    cond = np.full((T,), fill_value=-1, dtype=np.int32)
    for cid, cname in enumerate(constants.CONDITION_NAMES):
        t0, t1 = data_utils.get_condition_bounds(cid)
        cond[t0:t1] = cid
        print(f"[COND] {cname:<10}: [{t0}, {t1}) -> {t1 - t0} frames")
    if (cond < 0).any():
        missing = int((cond < 0).sum())
        print(f"[WARN] {missing} timesteps were not assigned a condition id (left as -1).")

    # Save
    np.save(os.path.join(outdir, "traces.npy"), traces)
    np.save(os.path.join(outdir, "conditions.npy"), cond)
    print(f"[SAVE] traces.npy, conditions.npy -> {outdir}")

def export_positions_from_seg(seg_store: str, outdir: str, z_stride: int = 1):
    """
    Compute per-label centroids from a 3D segmentation volume using two passes:
      1) find max label id (streamed over z)
      2) accumulate count and sums of x,y,z with np.bincount in chunks

    z_stride>1 can reduce IO (sample every z-th slice) at the cost of slight bias.
    """
    print(f"[SEG] Opening: {seg_store}")
    ds_seg = open_ts(seg_store)
    schema = ds_seg.schema
    print(f"[SEG] Schema: {schema}")

    # Infer shape as [X, Y, Z]
    # (TensorStore domain returns inclusive_min/exclusive_max per dim)
    dom = schema.domain
    dims = []
    for i in range(dom.rank):
        d = dom[i]
        dims.append(d.exclusive_max - d.inclusive_min)
    if len(dims) != 3:
        raise RuntimeError(f"Expected 3D segmentation, got shape {dims}")
    X, Y, Z = dims
    print(f"[SEG] Volume dims: X={X}, Y={Y}, Z={Z}")

    # -------- PASS 1: find max label id (to size accumulators) --------
    max_label = 0
    for z in range(0, Z, z_stride):
        seg_z = ds_seg[:, :, z].read().result()
        if seg_z.size == 0:
            continue
        max_label = max(max_label, int(seg_z.max()))
        if (z % 10) == 0:
            print(f"[PASS1] z={z}/{Z}  current max label={max_label}")
    if max_label == 0:
        print("[SEG] No positive labels found; skipping positions.")
        return
    L = max_label + 1
    print(f"[PASS1] Final max label: {max_label} -> allocating accumulators of size {L}")

    # Prepare accumulators (float64 to keep precision for large sums)
    count = np.zeros(L, dtype=np.float64)
    sum_x = np.zeros(L, dtype=np.float64)
    sum_y = np.zeros(L, dtype=np.float64)
    sum_z = np.zeros(L, dtype=np.float64)

    # Precompute per-slice coordinate grids (flattened) for efficiency
    xs = np.broadcast_to(np.arange(X)[:, None], (X, Y)).ravel().astype(np.float64)
    ys = np.broadcast_to(np.arange(Y)[None, :], (X, Y)).ravel().astype(np.float64)

    # -------- PASS 2: accumulate counts and coordinate sums --------
    for z in range(0, Z, z_stride):
        seg_z = ds_seg[:, :, z].read().result()
        lab = seg_z.ravel().astype(np.int64)
        m = lab > 0
        if not m.any():
            continue
        idx = lab[m]
        # Update counts and coordinate sums via bincount
        cnt = np.bincount(idx, minlength=L).astype(np.float64)
        sx  = np.bincount(idx, weights=xs[m], minlength=L)
        sy  = np.bincount(idx, weights=ys[m], minlength=L)
        sz  = np.bincount(idx, weights=(np.float64(z) * np.ones_like(idx, dtype=np.float64)), minlength=L)
        count += cnt
        sum_x += sx
        sum_y += sy
        sum_z += sz
        if (z % 10) == 0:
            nz = int(cnt[1:].sum())
            print(f"[PASS2] z={z}/{Z}  nonzero voxels this slice: ~{nz}")

    # Build centroid table for labels that actually occurred (count>0)
    has = count > 0
    label_ids = np.nonzero(has)[0].astype(np.int32)       # includes label 0 if present; remove it
    keep = label_ids > 0
    label_ids = label_ids[keep]

    cx = (sum_x[has] / np.maximum(count[has], 1))[keep]
    cy = (sum_y[has] / np.maximum(count[has], 1))[keep]
    cz = (sum_z[has] / np.maximum(count[has], 1))[keep]

    positions = np.stack([cx, cy, cz], axis=1)  # [K, 3], float64

    # Save
    np.save(os.path.join(outdir, "positions.npy"), positions)
    np.save(os.path.join(outdir, "label_ids.npy"), label_ids)
    print(f"[SAVE] positions.npy, label_ids.npy -> {outdir}")
    print("[NOTE] positions are indexed by segmentation label id via label_ids.npy; "
          "reorder to match traces columns if needed.")

def main():
    ap = argparse.ArgumentParser(description="Export ZAPBench as NumPy arrays (traces, conditions, positions).")
    ap.add_argument("--traces", default=DEFAULT_TRACES,
                    help="Path to traces store (gs:// or local zarr dir).")
    ap.add_argument("--segmentation", default=DEFAULT_SEG,
                    help="Path to segmentation store (gs:// or local zarr dir).")
    ap.add_argument("--outdir", default="./zapbench_numpy", help="Output directory for .npy files.")
    ap.add_argument("--no-positions", action="store_true", help="Skip positions export.")
    ap.add_argument("--z-stride", type=int, default=1, help="Stride over z-slices for centroid estimation (>=1).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # traces.npy + conditions.npy
    export_traces_and_conditions(args.traces, args.outdir)

    # positions.npy (+ label_ids.npy)
    if not args.no_positions:
        export_positions_from_seg(args.segmentation, args.outdir, z_stride=max(1, args.z_stride))
    else:
        print("[SKIP] positions export skipped (--no-positions)")

    print("\nDone.")

if __name__ == "__main__":
    main()

