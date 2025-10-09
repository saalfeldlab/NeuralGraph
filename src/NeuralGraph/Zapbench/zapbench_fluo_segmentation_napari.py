#!/usr/bin/env python3
# View ZapBench segmentation (XYZ) and optional raw overlay (XYZT) in napari.

import argparse
import numpy as np
import tensorstore as ts


# ---------- I/O helpers ----------

def open_ts_any(path: str):
    """Open either a gs:// or local Zarr; try zarr3, fall back to zarr2."""
    cfg = {'open': True, 'kvstore': path} if path.startswith('gs://') else \
          {'open': True, 'kvstore': {'driver': 'file', 'path': path}}
    try:
        return ts.open({**cfg, 'driver': 'zarr3'}).result()
    except Exception:
        return ts.open({**cfg, 'driver': 'zarr'}).result()


def get_frame_as_zyx(ds, frame_idx: int, axis_order: str):
    """
    Read one time frame from a 4D array with an axis string that includes T once.
    Returns a numpy array shaped (Z, Y, X).

    Examples:
      axis_order="XYZT" (ZapBench raw default) or "TZYX", "ZXYT", etc.
    """
    ax = axis_order.upper().replace(',', '').replace(' ', '')
    if ax.count('T') != 1:
        raise ValueError("axis_order must contain exactly one 'T' for time.")

    # Build index tuple — slice all spatial, pick the requested T
    index = []
    for c in ax:
        index.append(int(frame_idx) if c == 'T' else slice(None))
    vol = ds[tuple(index)].read().result()  # 3D array in whatever XYZ permutation

    # Map whatever spatial order to Z,Y,X
    spatial = [c for c in ax if c != 'T']         # 3 chars in some order of X,Y,Z
    if sorted(spatial) != ['X', 'Y', 'Z']:
        raise ValueError(f"Spatial axes must be a permutation of X,Y,Z, got {spatial}")
    perm = [spatial.index('Z'), spatial.index('Y'), spatial.index('X')]
    return np.transpose(vol, axes=perm)


def get_volume_as_zyx(ds, axis_order: str):
    """
    Read a 3D volume and return as (Z, Y, X).

    Example for ZapBench segmentation: axis_order="XYZ" (ds shape (2048, 1328, 72))
    """
    ax = axis_order.upper().replace(',', '').replace(' ', '')
    if len(ax) != 3 or sorted(ax) != ['X', 'Y', 'Z']:
        raise ValueError("axis_order for segmentation must be a 3-letter permutation of X,Y,Z.")

    vol = ds[...].read().result()
    perm = [ax.index('Z'), ax.index('Y'), ax.index('X')]
    return np.transpose(vol, axes=perm)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="View ZapBench segmentation (XYZ) + optional raw overlay (XYZT).")
    # Segmentation (labels)
    ap.add_argument("--seg_store", default="gs://zapbench-release/volumes/20240930/segmentation", help="Segmentation Zarr (XYZ).")
    ap.add_argument("--seg_axis", default="XYZ", help="Axis order of segmentation (default XYZ).")

    # Raw overlay (image)
    ap.add_argument("--raw_store", default="gs://zapbench-release/volumes/20240930/aligned",
                    help="Raw Zarr (XYZT). Leave as default or override.")
    ap.add_argument("--raw_axis", default="XYZT", help="Axis order of raw (default XYZT).")
    ap.add_argument("--frame", type=int, default=3736, help="Raw frame index to overlay.")

    # Voxel sizes (µm)
    ap.add_argument("--dz_um", type=float, default=4.0, help="Voxel size Z in µm.")
    ap.add_argument("--dy_um", type=float, default=0.406, help="Voxel size Y in µm.")
    ap.add_argument("--dx_um", type=float, default=0.406, help="Voxel size X in µm.")

    # Rendering / contrast for raw
    ap.add_argument("--rendering", default="mip", choices=["mip", "translucent", "additive", "iso"],
                    help="3D rendering mode for raw.")
    ap.add_argument("--contrast", default="1,99.9", help="Raw percentile contrast, e.g. '1,99.9'.")

    # Labels display
    ap.add_argument("--labels_opacity", type=float, default=0.7, help="Opacity of labels layer.")
    ap.add_argument("--show_boundaries", action="store_true", help="Show label contours.")

    args = ap.parse_args()

    # --- Segmentation ---
    ds_seg = open_ts_any(args.seg_store)
    print(f"[SEG] opened {args.seg_store}  shape={ds_seg.domain.shape}  dtype={ds_seg.dtype}")
    seg_zyx = get_volume_as_zyx(ds_seg, args.seg_axis).astype(np.int64)  # napari labels: any int type
    print(f"[SEG] volume ZYX shape: {seg_zyx.shape}, dtype: {seg_zyx.dtype}, min/max: {seg_zyx.min()} / {seg_zyx.max()}")

    # --- Raw overlay ---
    ds_raw = open_ts_any(args.raw_store)
    print(f"[RAW] opened {args.raw_store}  shape={ds_raw.domain.shape}  dtype={ds_raw.dtype}")
    vol_zyx = get_frame_as_zyx(ds_raw, args.frame, args.raw_axis).astype(np.float32)
    print(f"[RAW] volume ZYX shape: {vol_zyx.shape}, dtype: {vol_zyx.dtype}")

    # Contrast (your fallback: hi = max*2 if percentile fails)
    p_lo, p_hi = [float(x) for x in args.contrast.split(',')]
    lo, hi = np.percentile(vol_zyx, [p_lo, p_hi])
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (hi <= lo):
        lo = float(vol_zyx.min())
        hi = float(vol_zyx.max() * 2)

    # --- napari ---
    import napari
    viewer = napari.Viewer()

    # Raw image first (underlay)
    viewer.add_image(
        vol_zyx,
        name=f"raw_{args.frame}",
        scale=(args.dz_um, args.dy_um, args.dx_um),
        contrast_limits=(float(lo), float(hi)),
        rendering=args.rendering,
        opacity=0.9,
        blending='translucent',
    )

    # Segmentation labels (overlay)
    viewer.add_labels(
        seg_zyx,
        name="segmentation",
        scale=(args.dz_um, args.dy_um, args.dx_um),
        opacity=args.labels_opacity,
        blending='translucent',
    )
    if args.show_boundaries:
        layer = viewer.layers["segmentation"]
        layer.show_contours = True
        layer.contour = 1  # 1-voxel contour

    viewer.dims.ndisplay = 3
    napari.run()


if __name__ == "__main__":
    main()
