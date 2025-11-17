#!/usr/bin/env python3
"""ZapBench 3D Neural Reconstruction Visualization - Updated for 3-frame temporal model"""

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

def reconstruct_crop_4d(model, device, crop_coords, depth, height, width, 
                        crop_depth=None, time_coord=0.5, upsample_factors=None):
    """Reconstruct crop from 4D model at specific time"""
    x_start, y_start, crop_w, crop_h = crop_coords
    crop_d = crop_depth or depth // 4
    
    if upsample_factors:
        z_up, xy_up = upsample_factors
        crop_d_out = crop_d * z_up
        crop_h_out = crop_h * xy_up
        crop_w_out = crop_w * xy_up
    else:
        crop_d_out, crop_h_out, crop_w_out = crop_d, crop_h, crop_w
    
    # Generate coordinates
    z_coords = torch.linspace(0.5/depth, 1-0.5/depth, depth)[depth//2 - crop_d//2 : depth//2 + crop_d//2]
    z_coords = torch.nn.functional.interpolate(
        z_coords.unsqueeze(0).unsqueeze(0), size=crop_d_out, mode='linear', align_corners=False
    ).squeeze()
    
    y_start_norm = (y_start + 0.5) / height
    y_end_norm = (y_start + crop_h - 0.5) / height
    y_coords = torch.linspace(y_start_norm, y_end_norm, crop_h_out)
    
    x_start_norm = (x_start + 0.5) / width
    x_end_norm = (x_start + crop_w - 0.5) / width
    x_coords = torch.linspace(x_start_norm, x_end_norm, crop_w_out)
    
    zv, yv, xv = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
    tv = torch.full_like(zv, time_coord)
    coords = torch.stack([zv.flatten(), yv.flatten(), xv.flatten(), tv.flatten()]).t().to(device)
    
    # Reconstruct in batches
    batch_size = 2**18
    reconstructed = torch.zeros(crop_d_out * crop_h_out * crop_w_out, device=device)
    
    with torch.no_grad():
        for i in range(0, coords.shape[0], batch_size):
            batch = coords[i:i+batch_size]
            reconstructed[i:i+batch_size] = model(batch).squeeze()
    
    return reconstructed.reshape(crop_d_out, crop_h_out, crop_w_out).clamp(0, 1)

def extract_crop_from_volume(volume, crop_coords, crop_depth=None):
    """Extract crop from original volume"""
    x_start, y_start, crop_w, crop_h = crop_coords
    depth, height, width = volume.shape
    crop_d = crop_depth or depth // 4
    z_center = depth // 2
    z_start = z_center - crop_d // 2
    z_end = z_start + crop_d
    return volume[z_start:z_end, y_start:y_start+crop_h, x_start:x_start+crop_w]

def calculate_psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return (20 * torch.log10(max_val / torch.sqrt(mse))).item()

def launch_napari(orig_crop, recon_crop, up_crop=None):
    """Launch Napari with crops - upsample orig/recon to match upsampled dimensions"""
    if napari is None:
        print("Napari not available")
        return
    
    viewer = napari.Viewer()
    scale = [4.0/10, 0.406, 0.406]  # Z scaled for visualization
    
    # If upsampled exists, align all volumes to same size
    if up_crop is not None:
        import torch.nn.functional as F
        
        # Upsample original with nearest neighbor (no interpolation)
        orig_up_nearest = F.interpolate(
            orig_crop.unsqueeze(0).unsqueeze(0).float(),
            size=up_crop.shape, mode='nearest'
        ).squeeze(0).squeeze(0)
        
        # Upsample original with trilinear (interpolated)
        orig_up_interp = F.interpolate(
            orig_crop.unsqueeze(0).unsqueeze(0).float(),
            size=up_crop.shape, mode='trilinear', align_corners=False
        ).squeeze(0).squeeze(0)
        
        # Upsample reconstructed with nearest neighbor
        recon_up_nearest = F.interpolate(
            recon_crop.unsqueeze(0).unsqueeze(0).float(),
            size=up_crop.shape, mode='nearest'
        ).squeeze(0).squeeze(0)
        
        # Add all layers at same resolution
        viewer.add_image(orig_up_nearest.cpu().numpy(), name='Original (nearest)', 
                        scale=scale, colormap='viridis', contrast_limits=[0,1])
        viewer.add_image(orig_up_interp.cpu().numpy(), name='Original (interpolated)',
                        scale=scale, colormap='viridis', contrast_limits=[0,1], visible=False)
        viewer.add_image(recon_up_nearest.cpu().numpy(), name='Reconstructed (nearest)',
                        scale=scale, colormap='viridis', contrast_limits=[0,1])
        viewer.add_image(up_crop.cpu().numpy(), name='Reconstructed (upsampled)',
                        scale=scale, colormap='viridis', contrast_limits=[0,1])
    else:
        # No upsampling - just show original and reconstructed
        viewer.add_image(orig_crop.cpu().numpy(), name='Original', scale=scale, 
                         colormap='viridis', contrast_limits=[0,1])
        viewer.add_image(recon_crop.cpu().numpy(), name='Reconstructed', scale=scale,
                         colormap='viridis', contrast_limits=[0,1])
    
    viewer.dims.ndisplay = 3
    napari.run()

def main():
    parser = argparse.ArgumentParser(description="Visualize ZapBench reconstruction (default: original + recon + upsampled)")
    parser.add_argument("model_path", nargs="?", default="instantngp_outputs/zapbench_trained_model.pth")
    parser.add_argument("--frame", type=int, default=3739)
    parser.add_argument("--crop_x", type=int, default=782)
    parser.add_argument("--crop_y", type=int, default=1049)
    parser.add_argument("--crop_width", type=int, default=140)
    parser.add_argument("--crop_height", type=int, default=140)
    parser.add_argument("--time_coord", type=float, default=0.5, help="Time coordinate: 0.0, 0.5, or 1.0")
    parser.add_argument("--save_only", action="store_true")
    parser.add_argument("--no_upsample", action="store_true", help="Skip upsampled reconstruction")
    parser.add_argument("--upsample_xy", type=int, default=2)
    parser.add_argument("--upsample_z", type=int, default=8)
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
    
    crop_coords = (args.crop_x, args.crop_y, args.crop_width, args.crop_height)
    
    # Load original volume and extract crop
    print(f"Loading frame {args.frame}...")
    volume = read_volume(args.frame)
    orig_crop = torch.from_numpy(extract_crop_from_volume(volume, crop_coords)).to(device)
    
    # Reconstruct at specified time
    print(f"Reconstructing at t={args.time_coord}...")
    recon_crop = reconstruct_crop_4d(model, device, crop_coords, depth, height, width,
                                     time_coord=args.time_coord)
    
    psnr = calculate_psnr(recon_crop, orig_crop)
    print(f"PSNR: {psnr:.2f} dB")
    
    # Upsample by default (unless --no_upsample flag)
    up_crop = None
    if not args.no_upsample:
        print(f"Upsampling {args.upsample_xy}x XY, {args.upsample_z}x Z...")
        up_crop = reconstruct_crop_4d(model, device, crop_coords, depth, height, width,
                                      time_coord=args.time_coord,
                                      upsample_factors=(args.upsample_z, args.upsample_xy))
        print(f"Upsampled shape: {up_crop.shape}")
    
    # Save TIFFs
    output_dir = "instantngp_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    tifffile.imwrite(f"{output_dir}/crop_orig_f{args.frame}.tif", 
                     (orig_crop.cpu().numpy() * 65535).astype(np.uint16))
    tifffile.imwrite(f"{output_dir}/crop_recon_f{args.frame}_t{args.time_coord:.1f}.tif",
                     (recon_crop.cpu().numpy() * 65535).astype(np.uint16))
    if up_crop is not None:
        tifffile.imwrite(f"{output_dir}/crop_up_f{args.frame}_t{args.time_coord:.1f}.tif",
                        (up_crop.cpu().numpy() * 65535).astype(np.uint16))
    
    print(f"Saved to {output_dir}/")
    
    if not args.save_only:
        launch_napari(orig_crop, recon_crop, up_crop)
    
    tcnn.free_temporary_memory()

if __name__ == "__main__":
    main()
