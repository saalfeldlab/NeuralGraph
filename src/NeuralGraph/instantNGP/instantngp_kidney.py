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

# @file   instantngp_kidney.py
# @author Based on Thomas Müller's mlp_learning_an_image_pytorch.py
# @brief  3D volume reconstruction for kidney.tif (1200x1200x30)

import argparse
import json
import numpy as np
import os
import sys
import torch
import time
import tifffile

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

def read_volume(filename):
    """Read 3D volume from TIFF file and normalize to [0,1]"""
    print(f"reading volume: {filename}")
    volume = tifffile.imread(filename)
    print(f"original volume shape: {volume.shape}, dtype: {volume.dtype}")
    print(f"original value range: [{volume.min():.6f}, {volume.max():.6f}]")
    
    # Convert to float32 and normalize to [0,1]
    volume = volume.astype(np.float32)
    if volume.max() > 1.0:
        volume = volume / volume.max()
    
    print(f"normalized value range: [{volume.min():.6f}, {volume.max():.6f}]")
    return volume

def write_volume_tiff(filename, volume_array):
    """Write 3D volume as TIFF file"""
    # Ensure proper scaling for TIFF output
    volume_array = np.clip(volume_array, 0, 1)
    # Convert back to original data type for saving
    volume_uint16 = (volume_array * 65535).astype(np.uint16)
    tifffile.imwrite(filename, volume_uint16)

def calculate_psnr(pred, target, max_val=1.0):
    """Calculate Peak Signal-to-Noise Ratio in dB"""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')  # Perfect reconstruction
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()

def print_hash_table_analysis(config):
    """Print detailed hash table breakdown per level"""
    encoding_config = config["encoding"]
    n_levels = encoding_config["n_levels"]
    n_features_per_level = encoding_config["n_features_per_level"]
    log2_hashmap_size = encoding_config["log2_hashmap_size"]
    base_resolution = encoding_config["base_resolution"]
    per_level_scale = encoding_config["per_level_scale"]
    
    hashmap_size = 2 ** log2_hashmap_size
    
    print("\n" + "="*90)
    print("HASH GRID ENCODING ANALYSIS")
    print("="*90)
    print(f"{'Level':<6} {'Resolution':<12} {'Pixels':<12} {'Grid Size':<12} {'Hash Entries':<12} {'Features':<12}")
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
    print(f"Total encoding parameters: {total_parameters:.2e}")
    print(f"Highest resolution voxels: {resolution**3:.2e} ({resolution}³)")
    
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
    
    print(f"Network parameters: {network_params:.2e}")
    print(f"Total estimated parameters: {total_parameters + network_params:.2e}")
    print("="*90 + "\n")

def reconstruct_full_volume(model, xyz, depth, height, width, device, batch_size_vol=None):
    """Reconstruct full 3D volume from model in memory-efficient batches"""
    if batch_size_vol is None:
        batch_size_vol = height * width  # Process one slice at a time by default
    
    total_voxels = depth * height * width
    # Use float32 to match model output after conversion
    reconstructed = torch.zeros(depth, height, width, device=device, dtype=torch.float32)

    
    with torch.no_grad():
        for start_idx in range(0, total_voxels, batch_size_vol):
            end_idx = min(start_idx + batch_size_vol, total_voxels)
            batch_coords = xyz[start_idx:end_idx]
            
            # Get model predictions for this batch and convert to float32
            batch_output = model(batch_coords).squeeze().float()
            
            # Convert flat indices back to 3D indices
            # TODO (Cedric): `batch_size_actual` is not used anywhere. Bug or intentional?
            _batch_size_actual = end_idx - start_idx
            flat_indices = torch.arange(start_idx, end_idx, device=device)
            z_indices = flat_indices // (height * width)
            y_indices = (flat_indices % (height * width)) // width
            x_indices = flat_indices % width
            
            # Place results back into 3D volume
            reconstructed[z_indices, y_indices, x_indices] = batch_output.clamp(0.0, 1.0)
    
    return reconstructed

class Volume(torch.nn.Module):
    def __init__(self, filename, device):
        super(Volume, self).__init__()
        self.data = read_volume(filename)
        self.shape = self.data.shape  # (depth, height, width) or (height, width, depth)
        
        # Ensure consistent shape: (depth, height, width)
        if len(self.shape) == 3:
            # If shape is (1200, 1200, 30), reorder to (30, 1200, 1200)
            if self.shape[2] < self.shape[0]:
                self.data = np.transpose(self.data, (2, 0, 1))
                self.shape = self.data.shape
                print(f"Reordered volume to: {self.shape} (depth, height, width)")
        
        self.data = torch.from_numpy(self.data).float().to(device)
        
        # Pre-compute scaling tensor to avoid TracerWarning
        depth, height, width = self.shape
        self.scale_tensor = torch.tensor([depth-1, height-1, width-1], device=device, dtype=torch.float32)

    def forward(self, coords):
        """Trilinear interpolation for 3D volume sampling"""
        with torch.no_grad():
            # coords is Nx3 with values in [0,1]
            # Scale to volume dimensions using pre-computed tensor
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
    parser = argparse.ArgumentParser(description="3D Volume reconstruction using InstantNGP.")
    
    parser.add_argument("volume", nargs="?", default="kidney_512.tif", help="3D volume file to reconstruct")
    parser.add_argument("config", nargs="?", default="config_hash_3d.json", help="JSON config for tiny-cuda-nn")
    parser.add_argument("n_steps", nargs="?", type=int, default=10000000, help="Number of training steps")
    parser.add_argument("result_filename", nargs="?", default="", help="Output volume filename")

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    
    args = get_args()

    # Get script directory and construct paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, args.config)
    volume_path = os.path.join(script_dir, args.volume)

    with open(config_path) as config_file:
        config = json.load(config_file)

    # Load 3D volume
    volume = Volume(volume_path, device)
    depth, height, width = volume.shape

    # Create 3D neural network (3D input -> 1D output for scalar field)
    model = tcnn.NetworkWithInputEncoding(
        n_input_dims=3, 
        n_output_dims=1, 
        encoding_config=config["encoding"], 
        network_config=config["network"]
    ).to(device)
    
    print(model)
    print("using modern tiny-cuda-nn for 3D volume reconstruction.")

    # Print detailed hash table analysis
    print_hash_table_analysis(config)

    # Use learning rate from config file
    learning_rate = config["optimizer"]["learning_rate"]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print(f"using learning rate: {learning_rate}")

    # Create 3D coordinate grid
    n_voxels = depth * height * width
    print(f"volume dimensions: {depth}x{height}x{width} = {n_voxels:,} voxels")

    # Create coordinate meshgrid
    z_coords = torch.linspace(0.5/depth, 1-0.5/depth, depth, device=device)
    y_coords = torch.linspace(0.5/height, 1-0.5/height, height, device=device)  
    x_coords = torch.linspace(0.5/width, 1-0.5/width, width, device=device)
    
    zv, yv, xv = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
    xyz = torch.stack((zv.flatten(), yv.flatten(), xv.flatten())).t()
    
    print(f"coordinate grid shape: {xyz.shape}")

    # Clean and create output directory
    import shutil
    output_dir = os.path.join(script_dir, "instantngp_outputs")
    if os.path.exists(output_dir):
        # Count files being removed for user feedback
        existing_files = os.listdir(output_dir)
        tif_files = [f for f in existing_files if f.endswith('.tif')]
        png_files = [f for f in existing_files if f.endswith('.png')]
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)

    middle_slice = depth // 2

    # batch size for 3d (smaller due to memory constraints)
    batch_size = 2**20  # 1,048,576 - Adjusted for 3D volume
    print(f"using batch size: {batch_size:,} samples")

    try:
        batch = torch.rand([batch_size, 3], device=device, dtype=torch.float32)
        traced_volume = torch.jit.trace(volume, batch)
        print("volume tracing successful.")
    except:
        print("WARNING: PyTorch JIT trace failed. Using regular volume sampling.")
        traced_volume = volume

    print("\nphase 1: calibration - measuring iterations per 10s")
    
    # calibration phase - 100 seconds for more data points
    calibration_start = time.perf_counter()
    calibration_iterations = []
    calibration_times = []
    i = 0
    
    while time.perf_counter() - calibration_start < 100.0:
        batch = torch.rand([batch_size, 3], device=device, dtype=torch.float32)
        targets = traced_volume(batch)
        output = model(batch)

        relative_l2_error = (output - targets)**2 / (targets**2 + 0.01)
        loss = relative_l2_error.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        current_time = time.perf_counter()
        elapsed_time = current_time - calibration_start
        
        # Record every 10s
        if len(calibration_times) == 0 or elapsed_time >= (len(calibration_times) * 10):
            calibration_iterations.append(i)
            calibration_times.append(elapsed_time)
            print(f"calibration: {elapsed_time:.3f}s = iteration {i}  loss={loss.item():.6f}")
        
        i += 1
    
    print("\nphase 2: training with iteration-based saving")
    
    # Reset model for actual training
    model = tcnn.NetworkWithInputEncoding(
        n_input_dims=3, 
        n_output_dims=1, 
        encoding_config=config["encoding"], 
        network_config=config["network"]
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    start_time = time.perf_counter()
    total_training_time = 0.0
    save_counter = 0
    i = 0
    
    # Save initial state (t=0) - full volume and slice
    initial_volume = reconstruct_full_volume(model, xyz, depth, height, width, device)
    
    # Save full volume as TIFF
    volume_path = os.path.join(output_dir, "time_000_000ms_volume.tif")
    write_volume_tiff(volume_path, initial_volume.cpu().numpy())
    
    # calculate initial psnr on middle slice for monitoring
    initial_slice = initial_volume[middle_slice]
    target_slice = volume.data[middle_slice]
    initial_psnr = calculate_psnr(initial_slice, target_slice)
    
    print(f"initial psnr (middle slice): {initial_psnr:.2f}db")
    save_counter += 1
    
    # Training loop
    max_iterations = calibration_iterations[-1] if calibration_iterations else 1000
    
    while i <= max_iterations:
        train_start = time.perf_counter()
        
        batch = torch.rand([batch_size, 3], device=device, dtype=torch.float32)
        targets = traced_volume(batch)
        output = model(batch)

        relative_l2_error = (output - targets)**2 / (targets**2 + 0.01)
        loss = relative_l2_error.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_training_time += time.perf_counter() - train_start
        
        # save at calibration points
        if save_counter < len(calibration_iterations) and i == calibration_iterations[save_counter]:
            expected_ms = save_counter * 10000
        
            # reconstruct and save full volume
            pred_volume = reconstruct_full_volume(model, xyz, depth, height, width, device)
            
            # save full volume as tiff
            volume_path = os.path.join(output_dir, f"time_{save_counter:03d}_{expected_ms:04d}ms_volume.tif")
            write_volume_tiff(volume_path, pred_volume.cpu().numpy())
            
            # calculate psnr on middle slice for monitoring
            pred_slice = pred_volume[middle_slice]
            target_slice = volume.data[middle_slice]
            psnr_db = calculate_psnr(pred_slice, target_slice)

            print(f"checkpoint {save_counter}: psnr = {psnr_db:.2f} db")
            
            save_counter += 1
        
        i += 1

    total_wall_time = time.perf_counter() - start_time
    
    # calculate final psnr using the final reconstruction
    print("reconstructing final volume for psnr calculation...")
    final_volume = reconstruct_full_volume(model, xyz, depth, height, width, device)
    final_slice = final_volume[middle_slice]
    target_slice = volume.data[middle_slice]
    final_psnr = calculate_psnr(final_slice, target_slice)
    
    # save final full volume if requested
    if args.result_filename:
        result_path = os.path.join(script_dir, args.result_filename)
        write_volume_tiff(result_path, final_volume.cpu().numpy())
        print(f"saved final reconstructed volume: {result_path}")

    print("================================================================")
    print("training completed")
    print("================================================================")
    print(f"wall time: {total_wall_time:.3f}s")
    print(f"pure training time: {total_training_time:.3f}s")
    print(f"total iterations: {i}")
    print(f"final psnr (middle slice): {final_psnr:.2f} db")
    print(f"training efficiency: {total_training_time/total_wall_time*100:.1f}% (rest is i/o overhead)")
    print(f"volumes saved: {save_counter} (every 10s from 0ms to {int((save_counter-1)*10000)}ms)")
    print(f"output directory: {output_dir}")
    print("================================================================")

    tcnn.free_temporary_memory()