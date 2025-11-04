#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import time
from PIL import Image as PILImage
import os

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

def read_image(filename):
    """Read image and convert to numpy array with values in [0,1]"""
    img = PILImage.open(filename).convert('RGB')
    return np.array(img).astype(np.float32) / 255.0

def write_image(filename, img_array):
    """Write numpy array to image file with explicit RGB mode"""
    img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8)
    img = PILImage.fromarray(img_array, mode='RGB')
    # Change extension to PNG if it's JPG
    if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
        filename = filename.rsplit('.', 1)[0] + '.png'
    img.save(filename, format='PNG')

def calculate_psnr(img1, img2, max_val=1.0):
    """Calculate PSNR between two images"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    
    # Get script directory for file operations
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load image
    image_path = os.path.join(script_dir, "girl_with_a_pearl_earring.jpg")

    original_img = read_image(image_path)
    height, width, channels = original_img.shape
    
    # Create coordinate grid
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    x_coords = x_coords.astype(np.float32) / (width - 1)  # Normalize to [0, 1]
    y_coords = y_coords.astype(np.float32) / (height - 1)  # Normalize to [0, 1]
    coords = np.stack([x_coords.ravel(), y_coords.ravel()], axis=1)
    coords = torch.from_numpy(coords).to(device)
    print(f"coordinate grid shape: {coords.shape}")
    
    target_pixels = torch.from_numpy(original_img.reshape(-1, channels)).to(device)
    print(f"target pixels shape: {target_pixels.shape}")
    
    # Create optimized SIREN model
    model = Siren(
        in_features=2, 
        out_features=3, 
        hidden_features=256,  # Optimized
        hidden_layers=4,      # Optimized 
        outermost_linear=True,
        first_omega_0=60,     # Optimized
        hidden_omega_0=30     # Optimized
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)  # Optimized learning rate
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {num_params:,}")
    
    # Clear and create output directory
    output_dir = os.path.join(script_dir, "siren_outputs")
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    print("\nphase 1: calibration - measuring iterations per 250ms without I/O\n")
    
    # Calibration phase - pure training for 40 seconds to get 40 save points (every 250ms)
    calibration_start = time.perf_counter()
    calibration_iterations = []
    calibration_times = []
    i = 0
    
    # Training parameters for calibration
    batch_size = 50000  # Process in smaller batches to fit GPU memory
    n_pixels = coords.shape[0]
    
    while time.perf_counter() - calibration_start < 40.0:
        optimizer.zero_grad()

        # Process in batches to avoid OOM
        total_loss = 0.0
        n_batches = (n_pixels + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_pixels)
            
            batch_coords = coords[start_idx:end_idx]
            batch_targets = target_pixels[start_idx:end_idx]
            
            # Forward pass
            batch_predicted = model(batch_coords)
            
            # Compute loss for this batch
            batch_loss = torch.nn.functional.mse_loss(batch_predicted, batch_targets)
            
            # Scale loss by batch proportion and accumulate
            batch_weight = (end_idx - start_idx) / n_pixels
            scaled_loss = batch_loss * batch_weight
            scaled_loss.backward()
            
            total_loss += batch_loss.item() * batch_weight

        # Step optimizer after all batches
        optimizer.step()
        
        current_time = time.perf_counter()
        elapsed_time = current_time - calibration_start
        
        # Record every 250ms
        if len(calibration_times) == 0 or elapsed_time >= (len(calibration_times) * 0.25):
            calibration_iterations.append(i)
            calibration_times.append(elapsed_time)
            print(f"calibration: {elapsed_time:.3f}s = iteration {i}")
        
        i += 1
    
    
    print("\nphase 2: training with iteration-based saving")
    
    # Reset model for actual training
    model = Siren(
        in_features=2, 
        out_features=3, 
        hidden_features=256,  # Optimized
        hidden_layers=4,      # Optimized 
        outermost_linear=True,
        first_omega_0=60,     # Optimized
        hidden_omega_0=30     # Optimized
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)  # Optimized learning rate
    
    start_time = time.perf_counter()
    total_training_time = 0.0
    save_counter = 0
    i = 0
    
    # Save initial state (t=0)
    path = os.path.join(output_dir, "time_000_000ms.png")
    with torch.no_grad():
        predicted_image = model(coords).reshape(height, width, channels)
        predicted_image = torch.clamp(predicted_image, 0, 1)
        write_image(path, predicted_image.cpu().numpy())
    save_counter += 1
    
    # Training loop using calibrated iteration points
    while i < calibration_iterations[-1]:  # Train to same number as calibration
        iter_start = time.perf_counter()
        
        optimizer.zero_grad()

        # Process in batches to avoid OOM
        total_loss = 0.0
        n_batches = (n_pixels + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_pixels)
            
            batch_coords = coords[start_idx:end_idx]
            batch_targets = target_pixels[start_idx:end_idx]
            
            # Forward pass
            batch_predicted = model(batch_coords)
            
            # Compute loss for this batch
            batch_loss = torch.nn.functional.mse_loss(batch_predicted, batch_targets)
            
            # Scale loss by batch proportion and accumulate
            batch_weight = (end_idx - start_idx) / n_pixels
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
            expected_time_ms = (save_idx + 1) * 250  # +1 because we start from 250ms
            
            path = os.path.join(output_dir, f"time_{save_idx + 1:03d}_{expected_time_ms:03d}ms.png")
            
            with torch.no_grad():
                # Reconstruct full image in batches
                predicted_full = torch.zeros_like(target_pixels)
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, n_pixels)
                    batch_coords = coords[start_idx:end_idx]
                    predicted_full[start_idx:end_idx] = model(batch_coords)
                
                predicted_image = predicted_full.reshape(height, width, channels)
                predicted_image = torch.clamp(predicted_image, 0, 1)
                write_image(path, predicted_image.cpu().numpy())
        
        i += 1

    total_wall_time = time.perf_counter() - start_time
    
    # Get all saved images and calculate PSNR
    image_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])
    psnr_values = []
    time_values = []
    
    for img_file in image_files:
        # Extract time from filename (e.g., "time_001_250ms.png" or "time_000_000ms.png")
        parts = img_file.split('_')
        if len(parts) >= 3 and parts[2].endswith('ms.png'):
            time_ms = int(parts[2].replace('ms.png', ''))
            time_s = time_ms / 1000.0
        else:
            continue  # Skip files that don't match expected format
        
        # Load reconstructed image
        img_path = os.path.join(output_dir, img_file)
        reconstructed_img = read_image(img_path)
        
        # Calculate PSNR
        psnr = calculate_psnr(original_img, reconstructed_img)
        
        psnr_values.append(psnr)
        time_values.append(time_s)
    
    print("\n================================================================")
    print("training completed")
    print("================================================================")
    print(f"wall time: {total_wall_time:.3f}s")
    print(f"pure training time: {total_training_time:.3f}s")
    print(f"total iterations: {i}")
    print(f"initial PSNR: {psnr_values[0]:.2f} dB (t=0)")
    print(f"final PSNR: {psnr_values[-1]:.2f} dB")
    print(f"PSNR gain: {psnr_values[-1] - psnr_values[0]:.2f} dB")
    print(f"training efficiency: {total_training_time/total_wall_time*100:.1f}% (rest is I/O overhead)")
    print(f"images saved: {len(image_files)} (every 250ms from 0ms to {int((len(image_files)-1)*250)}ms)")
    print(f"output directory: {output_dir}")
    print("================================================================")
