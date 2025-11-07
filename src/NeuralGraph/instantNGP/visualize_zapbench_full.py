#!/usr/bin/env python3
"""
ZapBench 3D Neural Reconstruction Visualization - Full Volume

This script loads a trained instantNGP model and visualizes the reconstructed full volume for a given frame using napari.
Only the reconstructed volume is shown in napari (no original/crop overlays).
"""

import argparse
import numpy as np
import os
import sys
import torch
import tifffile
from types import SimpleNamespace

try:
    import napari
except ImportError:
    napari = None

try:
    import tinycudann as tcnn
except ImportError:
    print("Requires tiny-cuda-nn")
    sys.exit()

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from NeuralGraph.utils import open_gcs_zarr

def create_config():
    config = SimpleNamespace()
    config.zarr = SimpleNamespace()
    config.zarr.store_fluo = "gs://zapbench-release/volumes/20240930/aligned"
    config.zarr.dz_um = 4.0
    config.zarr.dy_um = 0.406
    config.zarr.dx_um = 0.406
    return config

def read_volume(frame_idx):
    """Load single frame from Zarr"""
    config = create_config()
    ds = open_gcs_zarr(config.zarr.store_fluo)
    vol_xyz = ds[..., frame_idx].read().result()
    volume = vol_xyz.transpose(2, 0, 1).astype(np.float32)
    if volume.max() > 1.0:
        volume = volume / volume.max()
    return volume

def reconstruct_full_4d(model, device, depth, height, width, time_coord=0.5, upsample_factors=None):
    """Reconstruct full volume from 4D model at specific time"""
    if upsample_factors:
        z_up, xy_up = upsample_factors
        depth_out = depth * z_up
        height_out = height * xy_up
        width_out = width * xy_up
    else:
        depth_out, height_out, width_out = depth, height, width

    # Generate coordinates
    z_coords = torch.linspace(0.5/depth, 1-0.5/depth, depth)
    if upsample_factors:
        z_coords = torch.nn.functional.interpolate(
            z_coords.unsqueeze(0).unsqueeze(0), size=depth_out, mode='linear', align_corners=False
        ).squeeze()
    y_coords = torch.linspace(0.5/height, 1-0.5/height, height)
    if upsample_factors:
        y_coords = torch.nn.functional.interpolate(
            y_coords.unsqueeze(0).unsqueeze(0), size=height_out, mode='linear', align_corners=False
        ).squeeze()
    x_coords = torch.linspace(0.5/width, 1-0.5/width, width)
    if upsample_factors:
        x_coords = torch.nn.functional.interpolate(
            x_coords.unsqueeze(0).unsqueeze(0), size=width_out, mode='linear', align_corners=False
        ).squeeze()

    zv, yv, xv = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
    tv = torch.full_like(zv, time_coord)
    coords = torch.stack([zv.flatten(), yv.flatten(), xv.flatten(), tv.flatten()]).t().to(device)

    # Reconstruct in batches
    batch_size = 2**18
    reconstructed = torch.zeros(depth_out * height_out * width_out, device=device)

    with torch.no_grad():
        for i in range(0, coords.shape[0], batch_size):
            batch = coords[i:i+batch_size]
            reconstructed[i:i+batch_size] = model(batch).squeeze()

    return reconstructed.reshape(depth_out, height_out, width_out).clamp(0, 1)

def launch_napari_full(recon_vol):
    """Launch Napari with only the reconstructed volume"""
    if napari is None:
        print("Napari not available")
        return
    import imageio
    config = create_config()
    scale = [config.zarr.dz_um, config.zarr.dy_um, config.zarr.dx_um]
    viewer = napari.Viewer()
    layer = viewer.add_image(
        recon_vol.cpu().numpy(),
        name='Reconstructed',
        scale=scale,
        colormap='green',
        contrast_limits=[0, 0.5]
    )
    viewer.dims.ndisplay = 3
    # Set camera settings
    viewer.camera.center = (142.0, 415.541, 269.381)
    viewer.camera.angles = (112.593, -12.926, 81.064)
    viewer.camera.zoom = 1.5414443744227209
    viewer.camera.perspective = 0.0

    # Rotate and capture snapshots
    n_frames = 480  # More frames for slower effect
    angle_step = 10 / n_frames  # Very little rotation
    snapshots = []
    import time
    for i in range(n_frames):
        # Camera trajectory: minimal rotation, strong zoom
        z_angle = 112.593 + i * angle_step  # minimal rotation
        y_angle = -12.926  # fixed Y
        zoom = 1.0 + 2.0 * np.sin(2 * np.pi * i / n_frames)  # much more zoom, little dezoom
        viewer.camera.angles = (
            z_angle,
            y_angle,
            81.064
        )
        viewer.camera.zoom = zoom
        viewer.window._qt_window.repaint()  # Force redraw
        time.sleep(0.07)  # Smooth video
        screenshot = viewer.screenshot(canvas_only=True)
        snapshots.append(screenshot)

    # Save snapshots as MP4
    output_dir = "instantngp_outputs"
    os.makedirs(output_dir, exist_ok=True)
    mp4_path = os.path.join(output_dir, "recons_rotation.mp4")
    imageio.mimsave(mp4_path, snapshots, fps=12)  # Slower playback
    print(f"Saved rotation MP4 to {mp4_path}")

    napari.run()

def main():
    parser = argparse.ArgumentParser(description="Visualize ZapBench full volume reconstruction (only reconstructed volume)")
    parser.add_argument("model_path", nargs="?", default="instantngp_outputs/zapbench_trained_model.pth")
    parser.add_argument("--frame", type=int, default=3739)
    parser.add_argument("--time_coord", type=float, default=0.5, help="Time coordinate: 0.0, 0.5, or 1.0")
    parser.add_argument("--save_only", action="store_true")
    parser.add_argument("--upsample_xy", type=int, default=1)
    parser.add_argument("--upsample_z", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    checkpoint = torch.load(args.model_path, map_location=device)
    config = checkpoint['config']
    depth, height, width = checkpoint['volume_shape']

    model = tcnn.NetworkWithInputEncoding(
        n_input_dims=4, n_output_dims=1,
        encoding_config=config["encoding"],
        network_config=config["network"]
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Reconstruct full volume
    print(f"Reconstructing full volume at t={args.time_coord}...")
    upsample_factors = (args.upsample_z, args.upsample_xy) if (args.upsample_z > 1 or args.upsample_xy > 1) else None
    recon_vol = reconstruct_full_4d(model, device, depth, height, width, time_coord=args.time_coord, upsample_factors=upsample_factors)
    print(f"Reconstructed shape: {recon_vol.shape}")

    # Save TIFF
    output_dir = "instantngp_outputs"
    os.makedirs(output_dir, exist_ok=True)
    tifffile.imwrite(f"{output_dir}/full_recon_f{args.frame}_t{args.time_coord:.1f}.tif",
                     (recon_vol.cpu().numpy() * 65535).astype(np.uint16))
    print(f"Saved to {output_dir}/full_recon_f{args.frame}_t{args.time_coord:.1f}.tif")

    if not args.save_only:
        launch_napari_full(recon_vol)

    tcnn.free_temporary_memory()

if __name__ == "__main__":
    main()
