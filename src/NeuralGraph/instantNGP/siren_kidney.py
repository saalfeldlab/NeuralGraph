#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import time
from PIL import Image as PILImage
import os
import tifffile

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                           1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                           np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, 
                 outermost_linear=False, first_omega_0=30, hidden_omega_0=30):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                            np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords)
        return output

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

def calculate_psnr(img1, img2, max_val=1.0):
    """Calculate PSNR between two images/volumes"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))

def reconstruct_full_volume(model, coords, depth, height, width, device, batch_size_vol=None):
    """Reconstruct full 3D volume from model in memory-efficient batches"""
    if batch_size_vol is None:
        batch_size_vol = height * width  # Process one slice at a time by default
    
    total_voxels = depth * height * width
    reconstructed = torch.zeros(depth, height, width, device=device, dtype=torch.float32)
    
    with torch.no_grad():
        for start_idx in range(0, total_voxels, batch_size_vol):
            end_idx = min(start_idx + batch_size_vol, total_voxels)
            batch_coords = coords[start_idx:end_idx]
            
            # Get model predictions for this batch
            batch_output = model(batch_coords).squeeze()
            
            # Convert flat indices back to 3D indices
            flat_indices = torch.arange(start_idx, end_idx, device=device)
            z_indices = flat_indices // (height * width)
            y_indices = (flat_indices % (height * width)) // width
            x_indices = flat_indices % width
            
            # Place results back into 3D volume
            reconstructed[z_indices, y_indices, x_indices] = batch_output.clamp(0.0, 1.0)
    
    return reconstructed

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    
    # Get script directory for file operations
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load 3D volume
    volume_path = os.path.join(script_dir, "kidney.tif")
    original_volume = read_volume(volume_path)
    
    # Ensure consistent shape: (depth, height, width)
    if len(original_volume.shape) == 3:
        # If shape is (1200, 1200, 30), reorder to (30, 1200, 1200)
        if original_volume.shape[2] < original_volume.shape[0]:
            original_volume = np.transpose(original_volume, (2, 0, 1))
            print(f"reordered volume to: {original_volume.shape} (depth, height, width)")
    
    depth, height, width = original_volume.shape
    print(f"processing volume dimensions: {depth}x{height}x{width}")
    
    # Create 3D coordinate grid
    z_coords, y_coords, x_coords = np.mgrid[0:depth, 0:height, 0:width]
    z_coords = z_coords.astype(np.float32) / (depth - 1)   # Normalize to [0, 1]
    y_coords = y_coords.astype(np.float32) / (height - 1)  # Normalize to [0, 1]
    x_coords = x_coords.astype(np.float32) / (width - 1)   # Normalize to [0, 1]
    coords = np.stack([z_coords.ravel(), y_coords.ravel(), x_coords.ravel()], axis=1)
    coords = torch.from_numpy(coords).to(device)
    print(f"coordinate grid shape: {coords.shape}")
    
    target_voxels = torch.from_numpy(original_volume.reshape(-1, 1)).to(device)
    print(f"target voxels shape: {target_voxels.shape}")

    # Create SIREN model optimized for 3D volumes
    model = Siren(
        in_features=3,        # 3D input (x, y, z)
        out_features=1,       # 1D output (scalar field)
        hidden_features=256,  # Larger network for 3D complexity
        hidden_layers=5,      # More layers for 3D features
        outermost_linear=True,
        first_omega_0=30,     # Standard first layer frequency
        hidden_omega_0=30     # Standard hidden layer frequency
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Lower LR for 3D stability
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {num_params:,}")

    # Clean and create output directory
    import shutil
    output_dir = os.path.join(script_dir, "siren_kidney_outputs")
    if os.path.exists(output_dir):
        # Count files being removed for user feedback
        existing_files = os.listdir(output_dir)
        tif_files = [f for f in existing_files if f.endswith('.tif')]
        png_files = [f for f in existing_files if f.endswith('.png')]
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)

    middle_slice = depth // 2
    
    print("\nphase 1: calibration - measuring iterations per 2.5s without i/o")
    
    # Calibration phase - pure training for 100 seconds (SIREN is much slower than InstantNGP)
    calibration_start = time.perf_counter()
    calibration_iterations = []
    calibration_times = []
    i = 0
    
    # Training parameters for calibration
    batch_size = 100000  # Smaller batches for 3D volumes
    n_voxels = coords.shape[0]
    
    while time.perf_counter() - calibration_start < 100.0:
        optimizer.zero_grad()

        # Process in batches to avoid OOM
        total_loss = 0.0
        n_batches = (n_voxels + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_voxels)
            
            batch_coords = coords[start_idx:end_idx]
            batch_targets = target_voxels[start_idx:end_idx]
            
            # Forward pass
            batch_predicted = model(batch_coords)
            
            # Compute loss for this batch
            batch_loss = torch.nn.functional.mse_loss(batch_predicted, batch_targets)
            
            # Scale loss by batch proportion and accumulate
            batch_weight = (end_idx - start_idx) / n_voxels
            scaled_loss = batch_loss * batch_weight
            scaled_loss.backward()
            
            total_loss += batch_loss.item() * batch_weight

        # Step optimizer after all batches
        optimizer.step()
        
        current_time = time.perf_counter()
        elapsed_time = current_time - calibration_start
        
        # Record every 2.5s
        if len(calibration_times) == 0 or elapsed_time >= (len(calibration_times) * 2.5):
            calibration_iterations.append(i)
            calibration_times.append(elapsed_time)
            print(f"calibration: {elapsed_time:.3f}s = iteration {i}")
        
        i += 1
    
    print("\nphase 2: training with iteration-based saving")
    
    # Reset model for actual training
    model = Siren(
        in_features=3,
        out_features=1,
        hidden_features=256,
        hidden_layers=5,
        outermost_linear=True,
        first_omega_0=30,
        hidden_omega_0=30
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    start_time = time.perf_counter()
    total_training_time = 0.0
    save_counter = 0
    i = 0
    
    # Save initial state (t=0) - full volume only
    initial_volume = reconstruct_full_volume(model, coords, depth, height, width, device)
    
    # Save full volume as TIFF
    volume_path = os.path.join(output_dir, "time_000_000ms_volume.tif")
    write_volume_tiff(volume_path, initial_volume.cpu().numpy())
    
    # calculate psnr on middle slice for monitoring
    initial_slice = initial_volume[middle_slice]
    target_slice = original_volume[middle_slice]
    initial_psnr = calculate_psnr(initial_slice.cpu().numpy(), target_slice)
    
    print(f"initial psnr (middle slice): {initial_psnr:.2f}db")
    save_counter += 1
    
    # Training loop using calibrated iteration points
    while i < calibration_iterations[-1]:
        iter_start = time.perf_counter()
        
        optimizer.zero_grad()

        # Process in batches to avoid OOM
        total_loss = 0.0
        n_batches = (n_voxels + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_voxels)
            
            batch_coords = coords[start_idx:end_idx]
            batch_targets = target_voxels[start_idx:end_idx]
            
            # Forward pass
            batch_predicted = model(batch_coords)
            
            # Compute loss for this batch
            batch_loss = torch.nn.functional.mse_loss(batch_predicted, batch_targets)
            
            # Scale loss by batch proportion and accumulate
            batch_weight = (end_idx - start_idx) / n_voxels
            scaled_loss = batch_loss * batch_weight
            scaled_loss.backward()
            
            total_loss += batch_loss.item() * batch_weight

        # Step optimizer after all batches
        optimizer.step()
        
        iter_end = time.perf_counter()
        total_training_time += (iter_end - iter_start)
        
        # Check if this iteration corresponds to a save point
        if i in calibration_iterations:
            save_idx = calibration_iterations.index(i)
            expected_time_ms = (save_idx + 1) * 2500  # +1 because we start from 2.5s
            
            print(f"saving checkpoint at {expected_time_ms}ms...")
            
            # reconstruct and save full volume
            pred_volume = reconstruct_full_volume(model, coords, depth, height, width, device)
            
            # save full volume as tiff
            volume_path = os.path.join(output_dir, f"time_{save_idx + 1:03d}_{expected_time_ms:04d}ms_volume.tif")
            write_volume_tiff(volume_path, pred_volume.cpu().numpy())
            
            # calculate psnr on middle slice for monitoring
            pred_slice = pred_volume[middle_slice]
            target_slice = original_volume[middle_slice]
            psnr_db = calculate_psnr(pred_slice.cpu().numpy(), target_slice)
    
        i += 1

    total_wall_time = time.perf_counter() - start_time
    
    # calculate final psnr using the final reconstruction
    print("reconstructing final volume for psnr calculation...")
    final_volume = reconstruct_full_volume(model, coords, depth, height, width, device)
    final_slice = final_volume[middle_slice].cpu().numpy()
    target_slice = original_volume[middle_slice]
    final_psnr = calculate_psnr(final_slice, target_slice)
    
    # Get all saved images and calculate PSNR progression
    image_files = sorted([f for f in os.listdir(output_dir) if f.endswith('_volume.tif')])
    psnr_values = []
    time_values = []
    
    # Calculate PSNR for initial and final states
    psnr_values = [initial_psnr, final_psnr]
    time_values = [0.0, total_training_time]
    
    print("\n================================================================")
    print("training completed")
    print("================================================================")
    print(f"wall time: {total_wall_time:.3f}s")
    print(f"pure training time: {total_training_time:.3f}s")
    print(f"total iterations: {i}")
    print(f"initial psnr: {psnr_values[0]:.2f} db (t=0)")
    print(f"final psnr: {psnr_values[-1]:.2f} db")
    print(f"psnr gain: {psnr_values[-1] - psnr_values[0]:.2f} db")
    print(f"training efficiency: {total_training_time/total_wall_time*100:.1f}% (rest is i/o overhead)")
    print(f"volumes saved: {len(image_files)} (every 2.5s from 0ms to {int((len(image_files)-1)*2500)}ms)")
    print(f"output directory: {output_dir}")
    print("================================================================")