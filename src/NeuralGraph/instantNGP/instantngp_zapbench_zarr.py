#!/usr/bin/env python3

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

# @file   instantngp_zapbench.py
# @author Based on Thomas Müller's mlp_learning_an_image_pytorch.py and instantngp_kidney.py
# @brief  3D volume reconstruction for ZapBench dataset (72x1328x2048)

import argparse
import json
import numpy as np
import os
import sys
import torch
import time
import tifffile
from tqdm import trange
from types import SimpleNamespace

try:
	import tinycudann as tcnn
except ImportError:
	print("This sample requires the tiny-cuda-nn extension for PyTorch.")
	print("You can install it by running:")
	print("============================================================")
	print("tiny-cuda-nn$ cd bindings/torch")
	print("tiny-cuda-nn/bindings/torch$ python setup.py install")
	print("============================================================")
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

def read_volume(frame_idx):
    """Read 3D volume from ZapBench Zarr store and normalize to [0,1] - EXACTLY LIKE ORIGINAL"""
    print(f"loading ZapBench volume from Google Cloud Zarr store, frame: {frame_idx}")
    
    # Create config and open Zarr dataset
    config = create_config()
    ds = open_gcs_zarr(config.zarr.store_fluo)
    
    # Load the specified frame
    print(f"  downloading frame {frame_idx}...")
    vol_xyz = ds[..., frame_idx].read().result()
    print(f"  original shape: {vol_xyz.shape}, dtype: {vol_xyz.dtype}")
    print(f"  original value range: [{vol_xyz.min():.0f}, {vol_xyz.max():.0f}]")
    
    # Transpose to (Z,Y,X) format: (1328, 2048, 72) -> (72, 1328, 2048)
    volume = vol_xyz.transpose(2, 0, 1).astype(np.float32)
    print(f"  transposed to ZYX: {volume.shape}")
    
    # Normalize to [0,1] - EXACTLY LIKE ORIGINAL TIFF PROCESSING
    if volume.max() > 1.0:
        volume = volume / volume.max()
    
    print(f"normalized value range: [{volume.min():.6f}, {volume.max():.6f}]")
    print(f"ZapBench physical dimensions: {volume.shape[0]*4.0:.1f}μm × {volume.shape[1]*0.406:.1f}μm × {volume.shape[2]*0.406:.1f}μm (Z×Y×X)")
    
    return volume

def write_volume_tiff(filename, volume_array, physical_spacing=(4.0, 0.406, 0.406)):
    """Write 3D volume as TIFF file with ZapBench metadata"""
    # Ensure proper scaling for TIFF output
    volume_array = np.clip(volume_array, 0, 1)
    # Convert back to uint16 for saving with full dynamic range
    volume_uint16 = (volume_array * 65535).astype(np.uint16)
    
    # Add ZapBench-specific metadata
    tifffile.imwrite(filename, volume_uint16, 
                     imagej=True,
                     metadata={
                         'axes': 'ZYX',
                         'unit': 'micrometer',
                         'spacing': physical_spacing,  # dz, dy, dx in micrometers
                         'source': 'ZapBench dataset',
                         'method': 'InstantNGP reconstruction'
                     })

def calculate_psnr(pred, target, max_val=1.0):
    """Calculate Peak Signal-to-Noise Ratio in dB"""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')  # Perfect reconstruction
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()

def print_hash_table_analysis(config, volume_shape=None):
    """Print detailed hash table breakdown per level for ZapBench 3D volumes"""
    encoding_config = config["encoding"]
    n_levels = encoding_config["n_levels"]
    n_features_per_level = encoding_config["n_features_per_level"]
    log2_hashmap_size = encoding_config["log2_hashmap_size"]
    base_resolution = encoding_config["base_resolution"]
    per_level_scale = encoding_config["per_level_scale"]
    
    hashmap_size = 2 ** log2_hashmap_size
    
    print("\n")
    print("="*90)
    print(f"{'level':<6} {'resolution':<12} {'pixels':<12} {'grid size':<12} {'hash entries':<12} {'features':<12}")
    print("-" * 90)
    
    total_parameters = 0
    
    for level in range(n_levels):
        resolution = int(base_resolution * (per_level_scale ** level))
        pixels = resolution ** 2  # 2D pixels for display
        grid_size = resolution ** 3  # 3D grid
        hash_entries = min(grid_size, hashmap_size)
        level_features = hash_entries * n_features_per_level
        total_parameters += level_features
        
        print(f"{level:<6} {resolution:<12} {pixels:<12.2e} {grid_size:<12.2e} {hash_entries:<12.2e} {level_features:<12.2e}")
    
    print("-" * 90)
    print(f"total encoding parameters: {total_parameters:.2e}")
    print(f"highest resolution voxels: {resolution**3:.2e} ({resolution}³)")
    
    # Add network parameters estimate
    network_config = config["network"]
    n_neurons = network_config["n_neurons"]
    n_hidden_layers = network_config["n_hidden_layers"]
    
    # Input layer: (n_levels * n_features_per_level) -> n_neurons
    network_params = (n_levels * n_features_per_level * n_neurons) + n_neurons
    
    # Hidden layers: n_neurons -> n_neurons
    for _ in range(n_hidden_layers):
        network_params += (n_neurons * n_neurons) + n_neurons
    
    # Output layer: n_neurons -> 1
    network_params += n_neurons + 1
    
    print(f"network parameters: {network_params:.2e}")
    print(f"total estimated parameters: {total_parameters + network_params:.2e}")
    # Add original volume voxels in scientific notation if volume_shape provided
    if volume_shape is not None:
        depth, height, width = volume_shape
        original_voxels = depth * height * width
        print(f"original volume voxels: {original_voxels:.2e} ({depth}×{height}×{width})")
    print("="*90 + "\n")

def reconstruct_full_volume_4d(model, xyzt, depth, height, width, device, batch_size_vol=None):
    """Reconstruct full 3D volume from 4D model in memory-efficient batches optimized for ZapBench"""
    if batch_size_vol is None:
        # For ZapBench (72×1328×2048), process in slices to manage memory
        batch_size_vol = height * width  # One Z-slice at a time
    
    total_voxels = depth * height * width
    reconstructed = torch.zeros(depth, height, width, device=device, dtype=torch.float32)
    
    with torch.no_grad():
        for start_idx in range(0, total_voxels, batch_size_vol):
            end_idx = min(start_idx + batch_size_vol, total_voxels)
            batch_coords = xyzt[start_idx:end_idx]
            
            # Get model predictions for this batch and convert to float32
            batch_output = model(batch_coords).squeeze().float()
            
            flat_indices = torch.arange(start_idx, end_idx, device=device)
            z_indices = flat_indices // (height * width)
            y_indices = (flat_indices % (height * width)) // width
            x_indices = flat_indices % width
            
            # Place results back into 3D volume
            reconstructed[z_indices, y_indices, x_indices] = batch_output.clamp(0.0, 1.0)
    
    return reconstructed

def reconstruct_full_volume(model, xyz, depth, height, width, device, batch_size_vol=None):
    """Reconstruct full 3D volume from model in memory-efficient batches optimized for ZapBench"""
    if batch_size_vol is None:
        # For ZapBench (72×1328×2048), process in slices to manage memory
        batch_size_vol = height * width  # One Z-slice at a time
    
    total_voxels = depth * height * width
    reconstructed = torch.zeros(depth, height, width, device=device, dtype=torch.float32)
    
    with torch.no_grad():
        for start_idx in range(0, total_voxels, batch_size_vol):
            end_idx = min(start_idx + batch_size_vol, total_voxels)
            batch_coords = xyz[start_idx:end_idx]
            
            # Get model predictions for this batch and convert to float32
            batch_output = model(batch_coords).squeeze().float()
            
            flat_indices = torch.arange(start_idx, end_idx, device=device)
            z_indices = flat_indices // (height * width)
            y_indices = (flat_indices % (height * width)) // width
            x_indices = flat_indices % width
            
            # Place results back into 3D volume
            reconstructed[z_indices, y_indices, x_indices] = batch_output.clamp(0.0, 1.0)
    
    return reconstructed

def reconstruct_upsampled_volume(model, depth, height, width, device, z_factor=8, xy_factor=2, batch_size_vol=None):
    """Reconstruct volume with upsampled dimensions (depth * z_factor, height * xy_factor, width * xy_factor)"""
    new_depth = depth * z_factor
    new_height = height * xy_factor
    new_width = width * xy_factor
    
    if batch_size_vol is None:
        # Much smaller batch size for memory efficiency
        batch_size_vol = 65536  # ~64K voxels per batch
    
    print(f"reconstructing upsampled volume: {new_depth}×{new_height}×{new_width} voxels in batches of {batch_size_vol:,}")
    
    # Pre-compute coordinate arrays (keep on CPU initially)
    z_coords = torch.linspace(0.5/new_depth, 1-0.5/new_depth, new_depth)
    y_coords = torch.linspace(0.5/new_height, 1-0.5/new_height, new_height)  
    x_coords = torch.linspace(0.5/new_width, 1-0.5/new_width, new_width)
    
    total_voxels = new_depth * new_height * new_width
    reconstructed = torch.zeros(new_depth, new_height, new_width, device=device, dtype=torch.float32)
    
    with torch.no_grad():
        for start_idx in range(0, total_voxels, batch_size_vol):
            end_idx = min(start_idx + batch_size_vol, total_voxels)
            batch_size = end_idx - start_idx
            
            # Generate coordinates for this batch on-the-fly to save memory
            flat_indices = torch.arange(start_idx, end_idx, device='cpu')
            z_indices_cpu = flat_indices // (new_height * new_width)
            y_indices_cpu = (flat_indices % (new_height * new_width)) // new_width
            x_indices_cpu = flat_indices % new_width
            
            # Create coordinate batch
            z_batch = z_coords[z_indices_cpu]
            y_batch = y_coords[y_indices_cpu]  
            x_batch = x_coords[x_indices_cpu]
            batch_coords = torch.stack([z_batch, y_batch, x_batch], dim=1).to(device)
            
            # Get model predictions for this batch and convert to float32
            batch_output = model(batch_coords).squeeze().float()
            
            # Move indices to GPU for placement
            z_indices = z_indices_cpu.to(device)
            y_indices = y_indices_cpu.to(device)
            x_indices = x_indices_cpu.to(device)
            
            # Place results back into 3D volume
            reconstructed[z_indices, y_indices, x_indices] = batch_output.clamp(0.0, 1.0)
    
    return reconstructed

class ZapBenchVolume(torch.nn.Module):
    def __init__(self, filename, device):
        super(ZapBenchVolume, self).__init__()
        self.data = read_volume(filename)
        self.shape = self.data.shape  # Expected: (72, 1328, 2048) - Z×Y×X
        
        print(f"ZapBench volume loaded: {self.shape} (Z×Y×X)")
        
        self.data = torch.from_numpy(self.data).float().to(device)
        
        # Pre-compute scaling tensor for coordinate normalization
        depth, height, width = self.shape
        self.scale_tensor = torch.tensor([depth-1, height-1, width-1], device=device, dtype=torch.float32)
        
        # ZapBench-specific properties
        self.physical_spacing = (4.0, 0.406, 0.406)  # μm per voxel (dz, dy, dx)
        self.total_voxels = depth * height * width
        print(f"Total voxels: {self.total_voxels:,}")

    def forward(self, coords):
        """Trilinear interpolation for ZapBench volume - 4D input but ignores time dimension"""
        with torch.no_grad():
            # coords is Nx4 with spatial values in [0,1] and time coordinate
            # For single frame, we ignore the time coordinate and use spatial only
            if coords.shape[1] == 4:
                coords = coords[:, :3]  # Extract only spatial coordinates (z,y,x)
            
            # Scale to volume dimensions
            depth, height, width = self.shape
            scaled_coords = coords * self.scale_tensor
            
            # Get integer indices and interpolation weights
            indices = scaled_coords.long()
            weights = scaled_coords - indices.float()

            # Clamp indices to volume bounds
            z0 = indices[:, 0].clamp(min=0, max=depth-1)
            y0 = indices[:, 1].clamp(min=0, max=height-1)
            x0 = indices[:, 2].clamp(min=0, max=width-1)
            z1 = (z0 + 1).clamp(max=depth-1)
            y1 = (y0 + 1).clamp(max=height-1)
            x1 = (x0 + 1).clamp(max=width-1)

            # Get interpolation weights
            wz = weights[:, 0:1]
            wy = weights[:, 1:2]
            wx = weights[:, 2:3]

            # Trilinear interpolation
            c000 = self.data[z0, y0, x0].unsqueeze(1)
            c001 = self.data[z0, y0, x1].unsqueeze(1)
            c010 = self.data[z0, y1, x0].unsqueeze(1)
            c011 = self.data[z0, y1, x1].unsqueeze(1)
            c100 = self.data[z1, y0, x0].unsqueeze(1)
            c101 = self.data[z1, y0, x1].unsqueeze(1)
            c110 = self.data[z1, y1, x0].unsqueeze(1)
            c111 = self.data[z1, y1, x1].unsqueeze(1)

            # Interpolate along x
            c00 = c000 * (1 - wx) + c001 * wx
            c01 = c010 * (1 - wx) + c011 * wx
            c10 = c100 * (1 - wx) + c101 * wx
            c11 = c110 * (1 - wx) + c111 * wx

            # Interpolate along y
            c0 = c00 * (1 - wy) + c01 * wy
            c1 = c10 * (1 - wy) + c11 * wy

            # Interpolate along z
            result = c0 * (1 - wz) + c1 * wz

            return result

def get_args():
    parser = argparse.ArgumentParser(description="3D ZapBench volume reconstruction using InstantNGP from Zarr store.")
    
    parser.add_argument("volume", nargs="?", default="3739", help="ZapBench frame index to reconstruct (e.g., 3739)")
    parser.add_argument("config", nargs="?", default="config_hash_zarr.json", help="JSON config for tiny-cuda-nn")
    parser.add_argument("n_steps", nargs="?", type=int, default=10000000, help="Number of training steps")
    parser.add_argument("result_filename", nargs="?", default="", help="Output volume filename")
    parser.add_argument("--extract_features", action="store_true", help="Run downstream feature extraction after training")
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    
    args = get_args()

    # Get script directory and construct paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, args.config)
    frame_idx = int(args.volume)  # Convert frame index to integer

    with open(config_path) as config_file:
        config = json.load(config_file)

    # Print config filename in green
    print(f"\033[92mUsing config: {args.config}\033[0m")
    print(f"\033[94mLoading ZapBench frame: {frame_idx}\033[0m")

    # Load ZapBench 3D volume from Zarr store
    volume = ZapBenchVolume(frame_idx, device)
    depth, height, width = volume.shape

    # Create 4D neural network (4D input -> 1D output for scalar field with time)
    model = tcnn.NetworkWithInputEncoding(
        n_input_dims=4, 
        n_output_dims=1, 
        encoding_config=config["encoding"], 
        network_config=config["network"]
    ).to(device)
    
    print(model)
    print("using modern tiny-cuda-nn for 4D spatiotemporal volume reconstruction.")

    # Print detailed hash table analysis for ZapBench
    print_hash_table_analysis(config, volume.shape)

    # Use learning rate from config file
    learning_rate = config["optimizer"]["learning_rate"]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print(f"using learning rate: {learning_rate}")

    # Create 4D coordinate grid with time dimension
    n_voxels = depth * height * width
    
    # Create coordinate meshgrid with proper ZapBench sampling
    z_coords = torch.linspace(0.5/depth, 1-0.5/depth, depth, device=device)
    y_coords = torch.linspace(0.5/height, 1-0.5/height, height, device=device)  
    x_coords = torch.linspace(0.5/width, 1-0.5/width, width, device=device)
    
    zv, yv, xv = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
    
    # Add time dimension: single frame at t=0.5 (middle of -3 to +4 range)
    t_val = 0.5  # Normalized time coordinate for single frame
    tv = torch.full_like(zv, t_val, device=device)
    
    # Create 4D coordinates (z, y, x, t)
    xyzt = torch.stack((zv.flatten(), yv.flatten(), xv.flatten(), tv.flatten())).t()
    print(f"Created 4D coordinate grid: {xyzt.shape} points with time coordinate t={t_val}")


    middle_slice = depth // 2

    # batch size for 4d (smaller due to memory constraints)
    batch_size = 2**18  # 262,144 - Adjusted for 4D volume
    print(f"using batch size: {batch_size:,} samples")

    # Skip JIT tracing for temporal debugging - use regular volume sampling
    print("Using regular volume sampling (JIT tracing disabled for temporal debugging)")
    traced_volume = volume

    print("\ntraining for 10000 iterations...")
    
    start_time = time.perf_counter()
    max_iterations = 10000
    
    for i in trange(max_iterations, desc="training", ncols=150):
        # Generate random 4D coordinates (z, y, x, t) where t=0.5 for single frame
        batch_3d = torch.rand([batch_size, 3], device=device, dtype=torch.float32)
        batch_t = torch.full([batch_size, 1], 0.5, device=device, dtype=torch.float32)
        batch = torch.cat([batch_3d, batch_t], dim=1)
        
        # Get targets using 4D coordinates (but volume ignores time)
        targets = traced_volume(batch)
        output = model(batch)

        relative_l2_error = (output - targets)**2 / (targets**2 + 0.01)
        loss = relative_l2_error.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % max(1, max_iterations//10) == 0:
            with torch.no_grad():
                # Reconstruct volume for evaluation
                pred_volume = reconstruct_full_volume_4d(model, xyzt, depth, height, width, device)
                pred_slice = pred_volume[middle_slice]
                target_slice = volume.data[middle_slice]
                psnr_db = calculate_psnr(pred_slice, target_slice)
            
            print(f"iteration {i:5d}: loss = {loss.item():.6f}, psnr = {psnr_db:.2f} db")

    total_wall_time = time.perf_counter() - start_time
    
    # Reconstruct original volume for PSNR calculation
    print("reconstructing final volume for PSNR calculation...")
    final_volume = reconstruct_full_volume_4d(model, xyzt, depth, height, width, device)
    
    # Calculate final PSNR on original resolution
    final_slice = final_volume[middle_slice]
    target_slice = volume.data[middle_slice]
    final_psnr = calculate_psnr(final_slice, target_slice)
    
    # Change output directory to instantngp_outputs to match request
    output_dir = os.path.join(script_dir, "instantngp_outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original resolution volume (72 × 2048 × 1328)
    original_volume_path = os.path.join(output_dir, f"zapbench_final_volume_original_frame_{frame_idx}.tif")
    write_volume_tiff(original_volume_path, final_volume.cpu().numpy(), volume.physical_spacing)
    print(f"original resolution volume saved: {original_volume_path}")
    
    # Skip upsampled reconstruction for faster validation
    print("skipping upsampled volume reconstruction (validation mode)")
    upsampled_volume_path = "skipped_for_validation"
    upsampled_spacing = (volume.physical_spacing[0] / 8, volume.physical_spacing[1] / 2, volume.physical_spacing[2] / 2)
    
    # save final full volume if requested
    if args.result_filename:
        result_path = os.path.join(script_dir, args.result_filename)
        write_volume_tiff(result_path, final_volume.cpu().numpy(), volume.physical_spacing)
        print(f"saved final reconstructed volume: {result_path}")

    print("\ntraining completed")
    print("================================================================")
    print(f"wall time: {total_wall_time:.3f}s")
    print(f"total iterations: {max_iterations}")
    print(f"final psnr (middle slice): {final_psnr:.2f} db")
    print(f"original volume saved: {original_volume_path}")
    print(f"original dimensions: {depth}×{height}×{width} voxels")
    print("upsampled volume: skipped for validation")
    print("================================================================")

    # Save the trained model
    model_path = os.path.join(output_dir, "zapbench_trained_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'volume_shape': (depth, height, width),
        'physical_spacing': volume.physical_spacing,
        'final_psnr': final_psnr,
        'dataset': 'ZapBench',
        'frame': 3736,
        'total_iterations': max_iterations
    }, model_path)
    print(f"Model saved to: {model_path}")

    tcnn.free_temporary_memory()