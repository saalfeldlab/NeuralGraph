#!/usr/bin/env python3

"""
ZapBench 3D Neural Reconstruction Visualization

This script provides interactive 3D visualization of InstantNGP neural field reconstruction
compared to original ZapBench calcium imaging data using Napari.

Usage:
    python visualize_zapbench_crop.py                    # Launch with Napari (default parameters)
    python visualize_zapbench_crop.py --save_only        # Save TIFF files only
    python visualize_zapbench_crop.py --upsample         # Generate upsampled reconstruction (2x XY, 8x Z)
    python visualize_zapbench_crop.py --help            # Show all options

Features:
- Loads trained 4D neural field model
- Extracts 3D crop using ImageJ coordinates: makeRectangle(782, 1049, 140, 140)  
- Creates side-by-side comparison (original | spacer | reconstruction)
- Interactive Napari 3D visualization with layer toggles
- Saves individual and combined TIFF volumes
"""

# Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice, this list of
#       conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
#       to endorse or promote products derived from this software without specific prior written
#       permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# @file   visualize_zapbench_crop.py
# @author Based on instantngp_zapbench_zarr.py
# @brief  Visual inspection of ZapBench temporal reconstruction with ImageJ crop coordinates

import argparse
import numpy as np
import os
import sys
import torch
import tifffile
from tqdm import trange
from types import SimpleNamespace

try:
    import napari
except ImportError:
    print("Warning: napari not available. Install with: pip install napari[all]")
    napari = None

try:
	import tinycudann as tcnn
except ImportError:
	print("This sample requires the tiny-cuda-nn extension for PyTorch.")
	sys.exit()

# Add the parent directory to the path to import NeuralGraph utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from NeuralGraph.utils import open_gcs_zarr

def create_config():
    """Create configuration object with ZapBench Zarr parameters"""
    config = SimpleNamespace()
    config.zarr = SimpleNamespace()
    config.zarr.store_fluo = "gs://zapbench-release/volumes/20240930/aligned"
    config.zarr.store_seg = "gs://zapbench-release/volumes/20240930/segmentation"
    config.zarr.dz_um = 4.0
    config.zarr.dy_um = 0.406
    config.zarr.dx_um = 0.406
    return config

def read_temporal_volume(center_frame_idx):
    """Read 3 temporal frames from ZapBench Zarr store for 4D training"""
    print(f"loading ZapBench temporal sequence centered on frame: {center_frame_idx}")
    
    # Create config and open Zarr dataset
    config = create_config()
    ds = open_gcs_zarr(config.zarr.store_fluo)
    
    # Load 3 frames: 1 before, center, 1 after
    frames = []
    frame_indices = [center_frame_idx - 1, center_frame_idx, center_frame_idx + 1]
    time_coords = [-1.0, 0.0, 1.0]
    
    for i, (frame_idx, t_coord) in enumerate(zip(frame_indices, time_coords)):
        print(f"  downloading frame {frame_idx} (t={t_coord})...")
        vol_xyz = ds[..., frame_idx].read().result()
        
        # Transpose to (Z,Y,X) format: (1328, 2048, 72) -> (72, 1328, 2048)
        volume = vol_xyz.transpose(2, 0, 1).astype(np.float32)
        
        # Normalize to [0,1]
        if volume.max() > 1.0:
            volume = volume / volume.max()
            
        frames.append((volume, t_coord))
        
        if i == 0:  # Print details for first frame
            print(f"  original shape: {vol_xyz.shape}, dtype: {vol_xyz.dtype}")
            print(f"  transposed to ZYX: {volume.shape}")
            print(f"  normalized value range: [{volume.min():.6f}, {volume.max():.6f}]")
    
    print(f"loaded {len(frames)} temporal frames for visualization")
    return frames

def reconstruct_crop_3d_4d(model, device, crop_coords, depth, height, width, crop_depth=None, time_coord=0.0, upsample_factors=None):
    """Reconstruct a 3D cropped region from the 4D model with optional upsampling
    
    Args:
        model: Trained 4D model
        device: PyTorch device
        crop_coords: (x_start, y_start, crop_width, crop_height) from ImageJ coordinates
        depth: Full volume depth
        height: Full volume height  
        width: Full volume width
        crop_depth: Number of Z slices in crop (default: depth//4)
        time_coord: Time coordinate for reconstruction (default 0.0 for center frame)
        upsample_factors: (z_factor, xy_factor) for upsampling (default: None for original resolution)
    """
    x_start, y_start, crop_width, crop_height = crop_coords
    
    if crop_depth is None:
        crop_depth = depth // 4  # Use quarter of the full depth for crop
    
    # Center the Z crop around middle
    middle_z = depth // 2
    z_start = max(0, middle_z - crop_depth // 2)
    z_end = min(depth, z_start + crop_depth)
    actual_crop_depth = z_end - z_start
    
    # Apply upsampling if requested
    if upsample_factors is not None:
        z_factor, xy_factor = upsample_factors
        upsampled_depth = actual_crop_depth * z_factor
        upsampled_height = crop_height * xy_factor
        upsampled_width = crop_width * xy_factor

        print(f"reconstructing upsampled 3D crop: {upsampled_depth}×{upsampled_height}×{upsampled_width} (Z×Y×X)")
        print(f"  original crop region: x={x_start}:{x_start+crop_width}, y={y_start}:{y_start+crop_height}, z={z_start}:{z_end}")
        print(f"  upsampling factors: {z_factor}x Z, {xy_factor}x XY, t={time_coord}")

        # Create high-resolution coordinate grids
        # The upsampled Z/Y/X grid should cover the same physical range as the original crop
        z_step = (z_end - z_start) / upsampled_depth
        y_step = crop_height / upsampled_height
        x_step = crop_width / upsampled_width

        z_indices = z_start + (torch.arange(upsampled_depth, device=device) + 0.5) * z_step
        y_indices = y_start + (torch.arange(upsampled_height, device=device) + 0.5) * y_step
        x_indices = x_start + (torch.arange(upsampled_width, device=device) + 0.5) * x_step

        # Normalize to [0,1] coordinates
        z_coords = z_indices / depth
        y_coords = y_indices / height
        x_coords = x_indices / width
        final_depth, final_height, final_width = upsampled_depth, upsampled_height, upsampled_width
    else:
        print(f"reconstructing 3D crop: x={x_start}:{x_start+crop_width}, y={y_start}:{y_start+crop_height}, z={z_start}:{z_end}, t={time_coord}")

        # Create coordinate grids for the 3D crop at original resolution
        z_indices = torch.arange(z_start, z_end, device=device, dtype=torch.float32)
        y_indices = torch.arange(y_start, y_start + crop_height, device=device, dtype=torch.float32)
        x_indices = torch.arange(x_start, x_start + crop_width, device=device, dtype=torch.float32)

        # Normalize to [0,1] coordinates
        z_coords = (z_indices + 0.5) / depth
        y_coords = (y_indices + 0.5) / height
        x_coords = (x_indices + 0.5) / width
        final_depth, final_height, final_width = actual_crop_depth, crop_height, crop_width
    
    # Create 3D meshgrid for the crop
    zv, yv, xv = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
    t_vals = torch.full_like(zv, time_coord, device=device)
    
    # Add Z-dependent time correction for light sheet acquisition (1 second across 72 Z slices)
    # Each Z slice is acquired ~13.9ms apart (1000ms / 72 slices)
    z_time_offset = (zv - 0.5) * (1.0 / 2.0)  # Map Z=[0,1] to time=[-0.5,+0.5] seconds
    t_vals_corrected = t_vals + z_time_offset
    
    # Create 4D coordinates (z, y, x, t) with Z-dependent time correction
    coords = torch.stack([zv.flatten(), yv.flatten(), xv.flatten(), t_vals_corrected.flatten()], dim=1)
    
    # Reconstruct in batches
    batch_size = 16384  # 16K samples per batch
    total_voxels = final_depth * final_height * final_width
    reconstructed = torch.zeros(total_voxels, device=device, dtype=torch.float32)
    
    print(f"reconstructing {total_voxels:,} voxels in batches of {batch_size:,}")
    
    with torch.no_grad():
        for start_idx in trange(0, len(coords), batch_size, desc="reconstructing crop", ncols=100):
            end_idx = min(start_idx + batch_size, len(coords))
            batch_coords = coords[start_idx:end_idx]
            batch_output = model(batch_coords).squeeze().float()
            reconstructed[start_idx:end_idx] = batch_output.clamp(0.0, 1.0)
    
    # Reshape to 3D crop (Z, Y, X)
    crop_reconstructed = reconstructed.view(final_depth, final_height, final_width)
    return crop_reconstructed

def extract_crop_3d_from_volume(volume, crop_coords, crop_depth=None):
    """Extract 3D crop from original volume data
    
    Args:
        volume: Original volume tensor (Z, Y, X)
        crop_coords: (x_start, y_start, crop_width, crop_height) 
        crop_depth: Number of Z slices in crop (default: volume.shape[0]//4)
    """
    x_start, y_start, crop_width, crop_height = crop_coords
    depth = volume.shape[0]
    if crop_depth is None:
        crop_depth = depth // 4  # Use quarter of the full depth for crop

    # Center the Z crop around the middle
    z_start = (depth - crop_depth) // 2
    z_end = z_start + crop_depth
    crop = volume[z_start:z_end, y_start:y_start+crop_height, x_start:x_start+crop_width]
    return crop
def save_comparison_crops_3d(original_crop, reconstructed_crop, frame_idx, time_coord, output_dir, crop_coords):
    """Save 3D original and reconstructed crops as TIFF files and create side-by-side comparison"""
    
    x_start, y_start, crop_width, crop_height = crop_coords
    
    # Convert to numpy and scale to uint16
    orig_np = (original_crop.cpu().numpy() * 65535).astype(np.uint16)
    recon_np = (reconstructed_crop.cpu().numpy() * 65535).astype(np.uint16)
    
    # Save original 3D crop
    orig_path = os.path.join(output_dir, f"crop3d_original_frame_{frame_idx}_t{time_coord:.1f}_x{x_start}y{y_start}_{crop_width}x{crop_height}.tif")
    tifffile.imwrite(orig_path, orig_np, 
                     metadata={
                         'axes': 'ZYX',
                         'source': 'ZapBench original',
                         'crop_coords': f'x={x_start}, y={y_start}, w={crop_width}, h={crop_height}'
                     })
    
    # Save reconstructed 3D crop  
    recon_path = os.path.join(output_dir, f"crop3d_reconstructed_frame_{frame_idx}_t{time_coord:.1f}_x{x_start}y{y_start}_{crop_width}x{crop_height}.tif")
    tifffile.imwrite(recon_path, recon_np,
                     metadata={
                         'axes': 'ZYX', 
                         'source': 'ZapBench InstantNGP reconstruction',
                         'crop_coords': f'x={x_start}, y={y_start}, w={crop_width}, h={crop_height}'
                     })
    
    print("saved 3D crops:")
    print(f"  original: {orig_path}")
    print(f"  reconstructed: {recon_path}")
    return orig_path, recon_path

def calculate_psnr(pred, target, max_val=1.0):
    """Calculate Peak Signal-to-Noise Ratio in dB"""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()

class ZapBenchVolume(torch.nn.Module):
    def __init__(self, filename, device):
        super(ZapBenchVolume, self).__init__()
        # Load temporal frames for 4D training
        self.temporal_frames = read_temporal_volume(filename)
        
        # Store all frames and their time coordinates
        self.volumes = []
        self.time_coords = []
        for volume, t_coord in self.temporal_frames:
            self.volumes.append(torch.from_numpy(volume).float().to(device))
            self.time_coords.append(t_coord)
        
        self.shape = self.volumes[0].shape  # Expected: (72, 1328, 2048) - Z×Y×X
        print(f"ZapBench temporal volumes loaded: {len(self.volumes)} frames, shape: {self.shape} (Z×Y×X)")
        print(f"Time coordinates: {self.time_coords}")
        
        # Pre-compute scaling tensor for coordinate normalization (spatial only)
        depth, height, width = self.shape
        self.scale_tensor = torch.tensor([depth-1, height-1, width-1], device=device, dtype=torch.float32)
        
        # ZapBench-specific properties
        self.physical_spacing = (4.0, 0.406, 0.406)  # μm per voxel (dz, dy, dx)
        self.total_voxels = depth * height * width

def launch_napari_comparison(combined_volume, original_crop, reconstructed_crop, upsampled_crop, crop_coords, frame_idx, time_coord, upsample_factors=None):
    # Create napari viewer (without title to avoid potential issues)
    viewer = napari.Viewer()

    # Physical spacing for ZapBench (dz, dy, dx in micrometers)
    dz_um = 4.0 / 10.0  # Scale Z by dividing by 10
    dy_um = 0.406
    dx_um = 0.406

    # Prepare upsampled original crop with interpolation if upsampled_crop and upsample_factors are provided
    orig_interp_np = None
    if upsampled_crop is not None and upsample_factors is not None:
        import torch.nn.functional as F
        upsampled_shape = upsampled_crop.shape
        # Interpolate original crop to upsampled shape
        orig_interp = F.interpolate(original_crop.unsqueeze(0).unsqueeze(0).float(),
                                   size=upsampled_shape,
                                   mode='trilinear',
                                   align_corners=False).squeeze(0).squeeze(0)
        orig_interp_np = orig_interp.cpu().numpy() if torch.is_tensor(orig_interp) else orig_interp
    # If upsampled_crop is present, upsample original and reconstructed crops for direct comparison
    if upsampled_crop is not None and upsample_factors is not None:
        import torch.nn.functional as F
        z_factor, xy_factor = upsample_factors
        # Only upsample if shapes differ
        if hasattr(original_crop, 'shape') and original_crop.shape != upsampled_crop.shape:
            original_crop = F.interpolate(original_crop.unsqueeze(0).unsqueeze(0).float(),
                                          size=upsampled_crop.shape,
                                          mode='nearest').squeeze(0).squeeze(0)
        if hasattr(reconstructed_crop, 'shape') and reconstructed_crop.shape != upsampled_crop.shape:
            reconstructed_crop = F.interpolate(reconstructed_crop.unsqueeze(0).unsqueeze(0).float(),
                                              size=upsampled_crop.shape,
                                              mode='nearest').squeeze(0).squeeze(0)
    # Sanity check: print shapes before launching Napari
    print("\nSanity check: volume shapes before Napari viewer")
    print(f"  Original crop shape: {getattr(original_crop, 'shape', type(original_crop))}") 
    print(f"  Reconstructed crop shape: {getattr(reconstructed_crop, 'shape', type(reconstructed_crop))}") 
    if upsampled_crop is not None:
        print(f"  Upsampled crop shape: {getattr(upsampled_crop, 'shape', type(upsampled_crop))}")
    """Launch Napari with side-by-side comparison and individual volumes"""
    
    if napari is None:
        print("Napari not available. Please install with: pip install napari[all]")
        return
        
    x_start, y_start, crop_width, crop_height = crop_coords
    
    # Physical spacing for ZapBench (dz, dy, dx in micrometers)
    dz_um = 4.0 / 10.0  # Scale Z by dividing by 10
    dy_um = 0.406
    dx_um = 0.406
    
    # Create napari viewer (without title to avoid potential issues)
    viewer = napari.Viewer()
    
    # Convert tensors to numpy arrays if needed
    orig_np = original_crop.cpu().numpy() if torch.is_tensor(original_crop) else original_crop
    recon_np = reconstructed_crop.cpu().numpy() if torch.is_tensor(reconstructed_crop) else reconstructed_crop
    
    # Add original crop with physical scaling
    viewer.add_image(orig_np,
                    name='Original ZapBench', 
                    scale=[dz_um, dy_um, dx_um],
                    colormap='viridis',
                    contrast_limits=[0, 1],
                    visible=True)  # Start visible
    
    # Add reconstructed crop with physical scaling
    viewer.add_image(recon_np,
                    name='Neural Reconstruction',
                    scale=[dz_um, dy_um, dx_um],
                    colormap='viridis', 
                    contrast_limits=[0, 1],
                    visible=True)  # Start visible

    # Add upsampled original crop with interpolation (viridis colormap)
    if orig_interp_np is not None:
        viewer.add_image(orig_interp_np,
                        name='Original Upsampled (interp)',
                        scale=[dz_um, dy_um, dx_um],
                        colormap='viridis',
                        contrast_limits=[0, 1],
                        visible=True)
    
    # Add upsampled reconstruction if available
    if upsampled_crop is not None and upsample_factors is not None:
        z_factor, xy_factor = upsample_factors
        upsampled_np = upsampled_crop.cpu().numpy() if torch.is_tensor(upsampled_crop) else upsampled_crop
        
        # Calculate upsampled physical spacing
        upsampled_scale = [dz_um, dy_um, dx_um]  # Set the same scale for all layers
        
        viewer.add_image(upsampled_np,
                        name=f'Upsampled ({xy_factor}x XY, {z_factor}x Z)',
                        scale=upsampled_scale,
                        colormap='viridis',  # Use viridis for upsampled
                        contrast_limits=[0, 1],
                        visible=True)  # Start visible
        
        print(f"   🔬 Upsampled: {upsampled_np.shape} at {upsampled_scale[0]:.3f}×{upsampled_scale[1]:.3f}×{upsampled_scale[2]:.3f} μm/voxel")
    
    # Set 3D display mode
    viewer.dims.ndisplay = 3
    
    print("🎯 Napari launched successfully!")
    print(f"   📏 Crop: x={x_start}, y={y_start}, size={crop_width}×{crop_height}")  
    print(f"   📐 Volume shape: {orig_np.shape} (Z×Y×X)")
    print(f"   🔬 Physical scale: {dz_um}×{dy_um}×{dx_um} μm/voxel")
    print("   🔄 Toggle layer visibility to compare original, reconstructed, and upsampled volumes")
    if upsampled_crop is not None:
        print("   🚀 Upsampled reconstruction available - demonstrates neural field continuity!")
    
    # Show the viewer - this should work now
    napari.run()

def get_args():
    parser = argparse.ArgumentParser(description="Visual inspection of ZapBench temporal reconstruction with Napari")
    
    parser.add_argument("model_path", nargs="?", default="instantngp_outputs/zapbench_trained_model.pth", help="Path to saved model (.pth file)")
    parser.add_argument("--frame", type=int, default=3739, help="ZapBench frame index for comparison (default: 3739)")
    parser.add_argument("--crop_x", type=int, default=782, help="ImageJ crop X start coordinate (default: 782)")
    parser.add_argument("--crop_y", type=int, default=1049, help="ImageJ crop Y start coordinate (default: 1049)")
    parser.add_argument("--crop_width", type=int, default=140, help="ImageJ crop width (default: 140)")
    parser.add_argument("--crop_height", type=int, default=140, help="ImageJ crop height (default: 140)")
    parser.add_argument("--crop_depth", type=int, default=None, help="Number of Z slices in crop (default: depth//4)")
    parser.add_argument("--time_coords", nargs='+', type=float, default=[0.0], help="Time coordinates to visualize (default: [0.0])")
    parser.add_argument("--save_only", action="store_true", help="Save files only, don't launch Napari")
    parser.add_argument("--upsample", action="store_true", help="Generate upsampled reconstruction (2x XY, 8x Z)")
    parser.add_argument("--upsample_xy", type=int, default=2, help="XY upsampling factor (default: 2)")
    parser.add_argument("--upsample_z", type=int, default=8, help="Z upsampling factor (default: 8)")
    
    args = parser.parse_args()
    return args

def main():
    napari_layers = []
    """Main visualization function"""
    print("="*70)
    print("🔬 ZapBench Neural Reconstruction 3D Visualization")
    print("="*70)
    print("📍 ImageJ crop: makeRectangle(782, 1049, 140, 140)")
    print("🖥️  Interactive Napari 3D viewer")
    print("   - Toggle layer visibility to compare original, reconstructed, and upsampled volumes")
    print("="*70)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    args = get_args()
    
    # Construct full path to model file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.model_path):
        model_path = os.path.join(script_dir, args.model_path)
    else:
        model_path = args.model_path
    
    # Load the saved model
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        print("Make sure you have trained a model first using instantngp_zapbench_zarr.py")
        sys.exit(1)
        
    print(f"loading trained model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model configuration
    config = checkpoint['config']
    volume_shape = checkpoint['volume_shape']
    frame_idx = args.frame
    
    print("model info:")
    print(f"  dataset: {checkpoint.get('dataset', 'unknown')}")
    print(f"  original frame: {checkpoint.get('frame', 'unknown')}")
    print(f"  volume shape: {volume_shape}")
    print(f"  final PSNR: {checkpoint.get('final_psnr', 'unknown'):.2f} dB")
    print(f"  total iterations: {checkpoint.get('total_iterations', 'unknown')}")
    
    # Create model with same architecture
    model = tcnn.NetworkWithInputEncoding(
        n_input_dims=4, 
        n_output_dims=1, 
        encoding_config=config["encoding"], 
        network_config=config["network"]
    ).to(device)
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("model loaded successfully")
    
    # Load original volume data for comparison
    print(f"loading original ZapBench data for frame {frame_idx}...")
    volume_data = ZapBenchVolume(frame_idx, device)
    depth, height, width = volume_data.shape
    
    # ImageJ crop coordinates (x, y, width, height)
    crop_coords = (args.crop_x, args.crop_y, args.crop_width, args.crop_height)
    middle_z = depth // 2
    
    print(f"ImageJ crop coordinates: makeRectangle({args.crop_x}, {args.crop_y}, {args.crop_width}, {args.crop_height})")
    print(f"middle Z slice: {middle_z} (of {depth})")
    
    # Create output directory
    output_dir = os.path.join(script_dir, "visualization_outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each requested time coordinate
    combined_volumes = []
    
    for time_coord in args.time_coords:
        print(f"\nprocessing time coordinate: {time_coord}")
        
        # Find the closest available frame for comparison
        closest_frame_idx = min(range(len(volume_data.time_coords)), 
                               key=lambda i: abs(volume_data.time_coords[i] - time_coord))
        closest_time = volume_data.time_coords[closest_frame_idx]
        
        print(f"using original frame with time {closest_time} (closest to requested {time_coord})")
        
        # Extract 3D crop from original volume
        original_volume = volume_data.volumes[closest_frame_idx]
        original_crop = extract_crop_3d_from_volume(original_volume, crop_coords, args.crop_depth)
        
        # Reconstruct 3D crop from model (original resolution)
        reconstructed_crop = reconstruct_crop_3d_4d(model, device, crop_coords, depth, height, width, 
                                                   args.crop_depth, time_coord)
        
        # Create upsampled version if requested
        reconstructed_crop_upsampled = None
        if args.upsample:
            print(f"\n🔬 Generating upsampled reconstruction ({args.upsample_xy}x XY, {args.upsample_z}x Z)...")
            upsample_factors = (args.upsample_z, args.upsample_xy)
            reconstructed_crop_upsampled = reconstruct_crop_3d_4d(model, device, crop_coords, depth, height, width, 
                                                                 args.crop_depth, time_coord, upsample_factors)
        
        # Calculate PSNR for the 3D crop
        crop_psnr = calculate_psnr(reconstructed_crop, original_crop)
        
        print("3D crop PSNR: {crop_psnr:.2f} dB")
        print("3D crop statistics:")
        print(f"  original    - shape: {original_crop.shape}, min: {original_crop.min():.6f}, max: {original_crop.max():.6f}, mean: {original_crop.mean():.6f}")
        print(f"  reconstructed - shape: {reconstructed_crop.shape}, min: {reconstructed_crop.min():.6f}, max: {reconstructed_crop.max():.6f}, mean: {reconstructed_crop.mean():.6f}")
        
        # Save 3D comparison crops and create side-by-side volume
        orig_path, recon_path = save_comparison_crops_3d(
            original_crop, reconstructed_crop, frame_idx, time_coord, output_dir, crop_coords)
        
        # Save upsampled reconstruction if generated
        if reconstructed_crop_upsampled is not None:
            print("upsampled crop statistics:")
            print(f"  upsampled   - shape: {reconstructed_crop_upsampled.shape}, min: {reconstructed_crop_upsampled.min():.6f}, max: {reconstructed_crop_upsampled.max():.6f}, mean: {reconstructed_crop_upsampled.mean():.6f}")
            print("  note: PSNR not calculated for upsampled version (different resolution than original)")
            
            # Save upsampled TIFF
            upsampled_np = (reconstructed_crop_upsampled.cpu().numpy() * 65535).astype(np.uint16)
            upsampled_path = os.path.join(output_dir, f"crop3d_upsampled_{args.upsample_xy}x{args.upsample_z}_frame_{frame_idx}_t{time_coord:.1f}_x{crop_coords[0]}y{crop_coords[1]}_{crop_coords[2]}x{crop_coords[3]}.tif")
            
            # Calculate physical spacing for upsampled volume
            dz_um, dy_um, dx_um = 4.0, 0.406, 0.406  # Original ZapBench spacing
            upsampled_spacing = (dz_um / args.upsample_z, dy_um / args.upsample_xy, dx_um / args.upsample_xy)
            
            tifffile.imwrite(upsampled_path, upsampled_np,
                           metadata={
                               'axes': 'ZYX',
                               'source': 'ZapBench InstantNGP upsampled reconstruction',
                               'spacing': upsampled_spacing,
                               'upsample_factors': f'{args.upsample_z}x Z, {args.upsample_xy}x XY'
                           })
            print(f"saved upsampled reconstruction: {upsampled_path}")
        
    # Store for Napari visualization (include upsampled if available)
    napari_layers.append((original_crop, reconstructed_crop, reconstructed_crop_upsampled, time_coord))
    
    print(f"\nvisualization completed. Output files saved to: {output_dir}")
    
    # Launch Napari with the comparison (use first time coordinate)
    if not args.save_only and napari_layers:
        print("\n🎨 Launching Napari for interactive 3D visualization...")
        print("   - Use mouse to rotate, zoom, and slice through the 3D volume")
        print("   - Toggle layer visibility to compare original, reconstructed, and upsampled volumes")
        if args.upsample:
            print(f"   - Upsampled layer: {args.upsample_xy}x XY, {args.upsample_z}x Z resolution")
            print("   - Compare upsampled vs original to see neural field continuity!")

        original_crop, reconstructed_crop, upsampled_crop, time_coord = napari_layers[0]
        upsample_factors = (args.upsample_z, args.upsample_xy) if args.upsample else None
        launch_napari_comparison(None, original_crop, reconstructed_crop, upsampled_crop,
                               crop_coords, frame_idx, time_coord, upsample_factors)
        print("\n💾 3D TIFF files saved successfully!")
        print("   Open the combined comparison file in ImageJ or Napari:")
        print(f"   → {output_dir}/crop3d_comparison_frame_{frame_idx}_t*.tif")
        if args.upsample:
            print(f"   → {output_dir}/crop3d_upsampled_{args.upsample_xy}x{args.upsample_z}_frame_{frame_idx}_t*.tif")
    
    tcnn.free_temporary_memory()
    print("="*70)
    print("✅ Visualization completed successfully!")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Visualization interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()