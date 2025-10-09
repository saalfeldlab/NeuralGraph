#!/usr/bin/env python3
# Read ONE ZapBench volume (by frame index) without downloading the whole dataset,
# and view it in napari with proper Z/XY scaling.

import argparse
import numpy as np
import tensorstore as ts

def open_ts_any(path: str):
    """Open either a local Zarr path or a gs:// Zarr; try zarr3, fall back to zarr2."""
    cfg = {'open': True, 'kvstore': path} if path.startswith('gs://') else \
          {'open': True, 'kvstore': {'driver': 'file', 'path': path}}
    try:
        return ts.open({**cfg, 'driver': 'zarr3'}).result()
    except Exception:
        return ts.open({**cfg, 'driver': 'zarr'}).result()

def get_frame_as_zyx(ds, frame_idx: int, axis_order: str):
    """
    Returns a numpy array shaped (Z, Y, X) for the requested frame.
    axis_order is a string like 'TZYX', 'ZYXT', or 'ZYX' (no time axis).
    Only the requested frame is read from storage.
    """
    axis_order = axis_order.upper().replace(',', '').replace(' ', '')
    shape = ds.domain.shape
    rank = len(shape)
    if len(axis_order) != rank:
        raise ValueError(f"axis_order length {len(axis_order)} must match rank {rank} (dataset shape {shape})")

    # Build index tuple that selects exactly one time frame (or pass-through if no T axis)
    index = []
    for ax in axis_order:
        if ax == 'T':
            index.append(int(frame_idx))
        else:
            index.append(slice(None))
    index = tuple(index)

    # Read JUST that indexed region into memory
    vol = ds[index].read().result()

    # Determine spatial axis permutation → Z,Y,X
    spatial_axes = [a for a in axis_order if a != 'T']
    if sorted(spatial_axes) != ['X','Y','Z']:
        raise ValueError(f"Spatial axes must be some permutation of Z,Y,X, got {spatial_axes}")
    perm = [spatial_axes.index(ax) for ax in ['Z','Y','X']]
    vol_zyx = np.transpose(vol, axes=perm)
    return vol_zyx

def main():
    ap = argparse.ArgumentParser(description="View one ZapBench raw frame in napari (no full download).")
    ap.add_argument("--store", required=False, help="gs://.../raw or local ./zapbench_raw.zarr")
    ap.add_argument("--frame", type=int, default=0, help="Time frame index to read (default 0)")
    ap.add_argument("--axis_order", default="TZYX", help="Axis order, e.g. TZYX, ZYXT, or ZYX")
    ap.add_argument("--dz_um", type=float, required=False, help="Voxel size along Z (µm)")
    ap.add_argument("--dy_um", type=float, required=False, help="Voxel size along Y (µm)")
    ap.add_argument("--dx_um", type=float, required=False, help="Voxel size along X (µm)")
    ap.add_argument("--contrast", default="1,99.9", help="Percentile contrast limits, e.g. 1,99.9")
    ap.add_argument("--rendering", default="mip", choices=["mip","translucent","additive","iso"],
                    help="3D rendering mode (napari).")
    args = ap.parse_args()


    args.store = "gs://zapbench-release/volumes/20240930/raw"
    args.frame = 3736
    args.axis_order = "XYZT"
    args.rendering = "mip"
    args.contrast = "1,99.9"


    ds = open_ts_any(args.store)
    print(f"[INFO] opened {args.store}  shape={ds.domain.shape}  dtype={ds.dtype}")
    vol_zyx = get_frame_as_zyx(ds, args.frame, args.axis_order)
    print(f"[INFO] volume ZYX shape: {vol_zyx.shape}, dtype: {vol_zyx.dtype}")


    p_lo, p_hi = [float(x) for x in args.contrast.split(',')]
    lo, hi = np.percentile(vol_zyx, [p_lo, p_hi])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(vol_zyx.min()), float(vol_zyx.max()*2)


    args.dz_um = 4.0
    args.dy_um = 0.406
    args.dx_um = 0.406

    import napari
    viewer = napari.Viewer()
    viewer.add_image(
        vol_zyx,
        name=f"frame_{args.frame}",
        scale=(args.dz_um, args.dy_um, args.dx_um),  # anisotropic voxel scale (Z, Y, X) in µm
        contrast_limits=(float(lo), float(hi)),
        rendering=args.rendering,
    )
    viewer.dims.ndisplay = 3
    napari.run()

if __name__ == "__main__":
    main()


# python zapbench_napari.py --store gs://zapbench-release/volumes/20240930/raw --frame 420 --axis_order TZYX --dz_um 2.0 --dy_um 0.8 --dx_um 0.8
