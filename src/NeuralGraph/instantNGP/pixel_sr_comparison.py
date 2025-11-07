import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Force Agg backend for headless environments
import matplotlib.pyplot as plt
import tifffile as tiff
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os
from tqdm import tqdm, trange
import tinycudann as tcnn

# Check for CUDA
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this implementation")

# Set constants
DEVICE = torch.device("cuda:0")
DTYPE = torch.float32  # Use full precision for better quality

def generate_shifted_downsampled_images(hr_image, downscale_factor, num_shifts):
    """
    Generate a set of shifted and downsampled images from a high-resolution image.
    Returns both the images and the exact shift values.
    """
    h, w = hr_image.shape
    
    # Create storage for low-res images and shifts
    lr_images = []
    shifts = []
    
    # Generate subpixel shifts in a grid pattern for even coverage
    grid_size = int(np.ceil(np.sqrt(num_shifts)))
    step = min(1.0, 1.5 / grid_size)
    
    shift_idx = 0
    for i in trange(grid_size, desc="Generating shifted images", ncols=80):
        for j in range(grid_size):
            if shift_idx >= num_shifts:
                break
                
            # Subpixel shift in range [0, 1) for both x and y
            shift_y = (i * step) % 1.0
            shift_x = (j * step) % 1.0
            shifts.append([shift_y, shift_x])
            
            # Apply shift and downsample
            grid_y, grid_x = torch.meshgrid(
                torch.arange(0, h, downscale_factor) + shift_y,
                torch.arange(0, w, downscale_factor) + shift_x,
                indexing='ij'
            )
            
            # Ensure we stay within image boundaries
            grid_y = torch.clamp(grid_y, 0, h-1.001)
            grid_x = torch.clamp(grid_x, 0, w-1.001)
            
            # Interpolate values at shifted positions
            y0, x0 = grid_y.floor().long(), grid_x.floor().long()
            y1, x1 = y0 + 1, x0 + 1
            
            # Ensure we don't go out of bounds
            y1 = torch.clamp(y1, 0, h-1)
            x1 = torch.clamp(x1, 0, w-1)
            
            wy = grid_y - y0.float()
            wx = grid_x - x0.float()
            
            # Bilinear interpolation
            val = (hr_image[y0, x0] * (1-wy) * (1-wx) + 
                  hr_image[y0, x1] * (1-wy) * wx + 
                  hr_image[y1, x0] * wy * (1-wx) + 
                  hr_image[y1, x1] * wy * wx)
            
            lr_images.append(val)
            shift_idx += 1
    
    return lr_images, shifts[:num_shifts]

def create_improved_instantngp_config():
    """Create improved network configuration for InstantNGP with higher capacity"""
    return {
        "encoding": {
            "otype": "HashGrid",
            "n_levels": 24,              # Increased from 16 to 24 for more detail
            "n_features_per_level": 4,    # Increased from 2 to 4 for more features
            "log2_hashmap_size": 22,      # Increased from 19 to 22 for larger hash table
            "base_resolution": 16,
            "per_level_scale": 1.382      # Tuned for better frequency coverage
        },
        "network": {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 128,             # Increased from 64 to 128
            "n_hidden_layers": 3,         # Increased from 2 to 3
            "precision": "full"
        }
    }

class NeuralSRRefinementNet(torch.nn.Module):
    """Neural network for refining SR result - the CNN approach"""
    def __init__(self, channels=64):
        super(NeuralSRRefinementNet, self).__init__()
        # Input refinement
        self.conv_in = torch.nn.Conv2d(1, channels, kernel_size=3, padding=1)
        
        # Residual blocks
        self.res_blocks = torch.nn.ModuleList([
            ResidualBlock(channels) for _ in range(5)
        ])
        
        # Global skip connection - important to preserve original details
        self.global_skip = torch.nn.Sequential(
            torch.nn.Conv2d(1, channels, kernel_size=1),
            torch.nn.Conv2d(channels, channels, kernel_size=1),
        )
        
        # Output projection
        self.conv_out = torch.nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        
        # Edge enhancement
        self.edge_enhance = torch.nn.Sequential(
            torch.nn.Conv2d(1, channels, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        # Store input for skip connection
        residual = x
        
        # Feature extraction
        feat = torch.nn.functional.relu(self.conv_in(x))
        
        # Residual blocks
        for res_block in self.res_blocks:
            feat = res_block(feat)
        
        # Global skip connection
        global_res = self.global_skip(residual)
        
        # Add global residual
        feat = feat + global_res
        
        # Output projection
        out = self.conv_out(feat)
        
        # Edge enhancement
        edge = self.edge_enhance(residual)
        
        # Add edge details to output
        return out + edge * 0.1 + residual


class ResidualBlock(torch.nn.Module):
    """Residual block with two convolutions and a skip connection"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual
        return self.relu(out)


def neural_sr_refinement(input_lr_images, shifts, downscale_factor, original_hr=None, device=DEVICE, num_epochs=500):
    """CNN-based neural SR approach"""
    # First apply classical SR to create initial high-res image
    hr_h, hr_w = input_lr_images[0].shape[0] * downscale_factor, input_lr_images[0].shape[1] * downscale_factor
    
    # Create an initial interpolated image from the first LR image
    initial_hr = torch.nn.functional.interpolate(
        input_lr_images[0].unsqueeze(0).unsqueeze(0),
        size=(hr_h, hr_w),
        mode='bicubic'
    ).squeeze()
    
    # Move to device
    initial_hr = initial_hr.to(device)
    
    # Add batch and channel dimensions
    input_sr = initial_hr.unsqueeze(0).unsqueeze(0)
    
    # Create model
    model = NeuralSRRefinementNet().to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop with progress bar
    if original_hr is not None:
        original_hr = original_hr.to(device)
        target_hr = original_hr.unsqueeze(0).unsqueeze(0)
        target_lr = torch.nn.functional.interpolate(
            target_hr, scale_factor=1/downscale_factor, mode='bicubic'
        )
        
        for epoch in trange(num_epochs, desc="Training neural refinement", ncols=100):
            # Forward pass
            refined = model(input_sr)
            
            # Create a downsampled version of our refined output
            refined_lr = torch.nn.functional.interpolate(
                refined, scale_factor=1/downscale_factor, mode='bicubic'
            )
            
            # Calculate losses:
            # 1. Supervised loss (if we have original HR)
            l_supervised = torch.nn.functional.l1_loss(refined, target_hr)
            
            # 2. Cycle consistency loss
            l_cycle = torch.nn.functional.l1_loss(refined_lr, target_lr)
            
            # 3. Edge preservation loss - using gradients
            refined_grad_x = torch.abs(refined[:,:,1:,:] - refined[:,:,:-1,:])
            refined_grad_y = torch.abs(refined[:,:,:,1:] - refined[:,:,:,:-1])
            target_grad_x = torch.abs(target_hr[:,:,1:,:] - target_hr[:,:,:-1,:])
            target_grad_y = torch.abs(target_hr[:,:,:,1:] - target_hr[:,:,:,:-1])
            
            l_edge = torch.nn.functional.l1_loss(refined_grad_x, target_grad_x) + \
                    torch.nn.functional.l1_loss(refined_grad_y, target_grad_y)
            
            # Combined loss with weights
            loss = l_supervised + 0.1 * l_cycle + 0.5 * l_edge
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    else:
        # If we don't have the ground truth, just do self-supervised training
        for epoch in trange(num_epochs, desc="Training neural refinement", ncols=100):
            # Forward pass
            refined = model(input_sr)
            
            # Create a downsampled version of our refined output
            refined_lr = torch.nn.functional.interpolate(
                refined, scale_factor=1/downscale_factor, mode='bicubic'
            )
            
            # Calculate losses:
            # Unsupervised loss based on self-consistency
            l_self = torch.nn.functional.l1_loss(
                torch.nn.functional.interpolate(refined_lr, scale_factor=downscale_factor, mode='bicubic'),
                refined
            )
            
            # Total variation loss for smoothness
            l_tv = torch.mean(torch.abs(refined[:,:,:-1,:] - refined[:,:,1:,:])) + \
                   torch.mean(torch.abs(refined[:,:,:,:-1] - refined[:,:,:,1:]))
            
            # Combined loss with weights
            loss = l_self + 0.01 * l_tv
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Final inference
    with torch.no_grad():
        refined = model(input_sr)
    
    return refined.squeeze().cpu()

class ImprovedNeuralFieldSR:
    """Improved neural field-based super-resolution using enhanced architecture"""
    def __init__(self, lr_images, shifts, downscale_factor, hr_shape):
        self.lr_images = [img.to(DEVICE).to(DTYPE) for img in lr_images]
        self.shifts = shifts
        self.downscale_factor = downscale_factor
        self.hr_h, self.hr_w = hr_shape
        
        # Initialize a bias field for correcting systematic errors
        self.use_bias_field = True
        
        # Create coordinate grids
        self.create_grids()
        
        # Set up neural fields
        self.setup_neural_fields()
    
    def create_grids(self):
        """Create normalized coordinate grids with improved sampling"""
        # Normalized coordinates for HR grid [0, 1]
        y = torch.linspace(0, 1, self.hr_h, device=DEVICE).to(DTYPE)
        x = torch.linspace(0, 1, self.hr_w, device=DEVICE).to(DTYPE)
        yv, xv = torch.meshgrid(y, x, indexing='ij')
        self.coords_hr = torch.stack([xv.flatten(), yv.flatten()], dim=1)
        
        # Create batch coordinate set with time for all LR images
        # Here, time is just the index of the LR image
        self.coords_lr_all = []
        self.values_lr_all = []
        
        for idx, (lr_img, shift) in enumerate(zip(self.lr_images, self.shifts)):
            # Get shape of current LR image
            lr_h, lr_w = lr_img.shape
            
            # Create coordinate grid for this LR image with improved handling of boundaries
            y = torch.arange(0, lr_h, device=DEVICE).float().to(DTYPE)
            x = torch.arange(0, lr_w, device=DEVICE).float().to(DTYPE)
            yv, xv = torch.meshgrid(y, x, indexing='ij')
            
            # Scale to [0, 1] and apply shift with improved accuracy
            # Convert pixel coordinates to normalized coordinates
            norm_y = (yv.flatten() * self.downscale_factor + shift[0] * self.downscale_factor) / (self.hr_h - 1)
            norm_x = (xv.flatten() * self.downscale_factor + shift[1] * self.downscale_factor) / (self.hr_w - 1)
            
            # Clamp to ensure valid range
            norm_y = torch.clamp(norm_y, 0.0, 1.0)
            norm_x = torch.clamp(norm_x, 0.0, 1.0)
            
            # Normalize time to [0, 1]
            t = torch.full_like(norm_x, idx / max(1, len(self.lr_images) - 1))
            
            # Stack coordinates with time
            coords = torch.stack([norm_x, norm_y, t], dim=1)
            
            # Get values
            values = lr_img.flatten()
            
            self.coords_lr_all.append(coords)
            self.values_lr_all.append(values)
            
        # Concatenate all
        self.coords_lr_all = torch.cat(self.coords_lr_all, dim=0)
        self.values_lr_all = torch.cat(self.values_lr_all, dim=0)
        
        # Add stratified samples for improved coverage
        self.add_stratified_samples()
    
    def add_stratified_samples(self, samples_per_bin=16):
        """Add stratified samples to improve training stability"""
        # Create stratified sampling grid
        bins_h = min(32, self.hr_h // 8)
        bins_w = min(32, self.hr_w // 8)
        
        # Create bins
        for t_idx in range(len(self.lr_images)):
            t_normalized = t_idx / max(1, len(self.lr_images) - 1)
            
            for h_idx in range(bins_h):
                for w_idx in range(bins_w):
                    # Get bin boundaries
                    y_min = h_idx / bins_h
                    y_max = (h_idx + 1) / bins_h
                    x_min = w_idx / bins_w
                    x_max = (w_idx + 1) / bins_w
                    
                    # Generate random samples in bin
                    y_samples = torch.rand(samples_per_bin, device=DEVICE).to(DTYPE) * (y_max - y_min) + y_min
                    x_samples = torch.rand(samples_per_bin, device=DEVICE).to(DTYPE) * (x_max - x_min) + x_min
                    t_samples = torch.full_like(x_samples, t_normalized)
                    
                    # Stack coordinates
                    bin_coords = torch.stack([x_samples, y_samples, t_samples], dim=1)
                    
                    # For each coordinate, find nearest LR pixel
                    # This is an approximation - in a real implementation, you'd use proper interpolation
                    # But for now, we'll just use nearest LR pixel value
                    lr_h, lr_w = self.lr_images[t_idx].shape
                    
                    # Convert normalized coordinates to pixel coordinates
                    px = torch.clamp((x_samples * (self.hr_w - 1) - self.shifts[t_idx][1] * self.downscale_factor) / self.downscale_factor, 0, lr_w - 1).long()
                    py = torch.clamp((y_samples * (self.hr_h - 1) - self.shifts[t_idx][0] * self.downscale_factor) / self.downscale_factor, 0, lr_h - 1).long()
                    
                    # Get values
                    bin_values = self.lr_images[t_idx][py, px]
                    
                    # Add to training data
                    self.coords_lr_all = torch.cat([self.coords_lr_all, bin_coords], dim=0)
                    self.values_lr_all = torch.cat([self.values_lr_all, bin_values], dim=0)
    
    def setup_neural_fields(self):
        """Set up enhanced neural fields for scene and motion"""
        self.config = create_improved_instantngp_config()
        
        # Scene network (2D input -> 1D output)
        self.scene_net = tcnn.NetworkWithInputEncoding(
            n_input_dims=2,         # x, y
            n_output_dims=1,        # intensity
            encoding_config=self.config["encoding"],
            network_config=self.config["network"]
        ).to(DEVICE)
        
        # Motion network (3D input -> 2D output)
        self.motion_net = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,         # x, y, t
            n_output_dims=2,        # motion field (dx, dy)
            encoding_config=self.config["encoding"],
            network_config=self.config["network"]
        ).to(DEVICE)
        
        # Optional bias field for systematic error correction
        if self.use_bias_field:
            self.bias_net = tcnn.NetworkWithInputEncoding(
                n_input_dims=3,     # x, y, t
                n_output_dims=1,    # bias correction
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": 4,
                    "n_features_per_level": 2,
                    "log2_hashmap_size": 14,
                    "base_resolution": 8,
                    "per_level_scale": 2.0
                },
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 32,
                    "n_hidden_layers": 1,
                    "precision": "full"
                }
            ).to(DEVICE)
        
        # Create optimizers with improved learning rates and scheduling
        self.scene_optimizer = torch.optim.Adam(self.scene_net.parameters(), lr=1e-3)
        self.motion_optimizer = torch.optim.Adam(self.motion_net.parameters(), lr=5e-4)
        
        if self.use_bias_field:
            self.bias_optimizer = torch.optim.Adam(self.bias_net.parameters(), lr=1e-3)
        
        # Learning rate schedulers
        self.scene_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.scene_optimizer, mode='min', factor=0.5, patience=100, verbose=True
        )
        self.motion_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.motion_optimizer, mode='min', factor=0.5, patience=100, verbose=True
        )
    
    def sample_training_points(self, batch_size):
        """Sample random points from all LR images with improved sampling strategy"""
        # Random indices with importance sampling
        indices = torch.randint(0, len(self.values_lr_all), (batch_size,), device=DEVICE)
        
        # Get coordinates and values
        coords = self.coords_lr_all[indices]
        values = self.values_lr_all[indices]
        
        return coords, values.unsqueeze(1)
    
    def train_step(self, batch_size, step, total_steps):
        """Perform a single training step with improved loss function"""
        # Sample training points
        coords, targets = self.sample_training_points(batch_size)
        
        # Get motion at these coordinates and time
        motion = self.motion_net(coords)
        
        # Apply motion to get corrected coordinates
        corrected_coords = coords[:, :2] + motion
        
        # Clamp coordinates to [0, 1]
        corrected_coords = torch.clamp(corrected_coords, 0.0, 1.0)
        
        # Evaluate scene at corrected coordinates
        scene_values = self.scene_net(corrected_coords)
        
        # Apply bias correction if enabled
        if self.use_bias_field:
            bias = self.bias_net(coords)
            predictions = scene_values + bias * 0.1  # Scale bias effect
        else:
            predictions = scene_values
        
        # Calculate data fidelity loss
        data_loss = torch.nn.functional.mse_loss(predictions, targets)
        
        # Add regularization terms with annealing weights
        progress = step / total_steps
        
        # Motion smoothness regularization - encourage smooth motion fields
        motion_smoothness_weight = 0.01 * (1.0 - progress)  # Decrease weight over time
        
        # Compute motion gradients
        if motion_smoothness_weight > 0:
            # Group by time for meaningful spatial derivatives
            t_values = coords[:, 2]
            unique_t = torch.unique(t_values)
            
            motion_smoothness_loss = 0.0
            
            for t in unique_t:
                t_mask = t_values == t
                if t_mask.sum() > 100:  # Only if we have enough points at this time
                    t_coords = coords[t_mask, :2]  # Spatial coordinates
                    t_motion = motion[t_mask]      # Motion vectors
                    
                    # Compute approximate derivatives by finding nearby points
                    sorted_indices = torch.argsort(t_coords[:, 1])  # Sort by y
                    sorted_coords = t_coords[sorted_indices]
                    sorted_motion = t_motion[sorted_indices]
                    
                    # Compute y-derivatives (approximation)
                    dy_motion = torch.abs(sorted_motion[1:] - sorted_motion[:-1])
                    motion_smoothness_loss += dy_motion.mean()
                    
                    # Repeat for x-direction
                    sorted_indices = torch.argsort(t_coords[:, 0])  # Sort by x
                    sorted_coords = t_coords[sorted_indices]
                    sorted_motion = t_motion[sorted_indices]
                    
                    # Compute x-derivatives (approximation)
                    dx_motion = torch.abs(sorted_motion[1:] - sorted_motion[:-1])
                    motion_smoothness_loss += dx_motion.mean()
            
            # Normalize by number of unique times
            if len(unique_t) > 0:
                motion_smoothness_loss /= len(unique_t)
        else:
            motion_smoothness_loss = 0.0
        
        # Scene smoothness regularization - encourage smooth scene
        scene_smoothness_weight = 0.005 * (1.0 - progress)  # Decrease weight over time
        
        if scene_smoothness_weight > 0 and step % 10 == 0:  # Every 10 steps to save computation
            # Sample random points in scene for computing gradients
            random_coords = torch.rand(1000, 2, device=DEVICE).to(DTYPE)
            random_coords.requires_grad = True
            
            # Forward pass
            random_values = self.scene_net(random_coords)
            
            # Compute gradients
            grad_outputs = torch.ones_like(random_values)
            gradients = torch.autograd.grad(
                outputs=random_values,
                inputs=random_coords,
                grad_outputs=grad_outputs,
                create_graph=True
            )[0]
            
            # Penalize high gradients
            scene_smoothness_loss = gradients.pow(2).mean()
        else:
            scene_smoothness_loss = 0.0
        
        # Total variation loss for scene values
        tv_weight = 0.001 * (1.0 - progress)
        if tv_weight > 0 and step % 10 == 0:
            # Sample a grid of points
            grid_size = 32
            y = torch.linspace(0, 1, grid_size, device=DEVICE).to(DTYPE)
            x = torch.linspace(0, 1, grid_size, device=DEVICE).to(DTYPE)
            yv, xv = torch.meshgrid(y, x, indexing='ij')
            grid_coords = torch.stack([xv.flatten(), yv.flatten()], dim=1)
            
            # Forward pass
            grid_values = self.scene_net(grid_coords).reshape(grid_size, grid_size)
            
            # Calculate total variation
            tv_loss = torch.mean(torch.abs(grid_values[:, :-1] - grid_values[:, 1:])) + \
                     torch.mean(torch.abs(grid_values[:-1, :] - grid_values[1:, :]))
        else:
            tv_loss = 0.0
        
        # Combine losses
        total_loss = data_loss + \
                     motion_smoothness_weight * motion_smoothness_loss + \
                     scene_smoothness_weight * scene_smoothness_loss + \
                     tv_weight * tv_loss
        
        # Backward pass
        self.scene_optimizer.zero_grad()
        self.motion_optimizer.zero_grad()
        if self.use_bias_field:
            self.bias_optimizer.zero_grad()
        
        total_loss.backward()
        
        # Apply gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.scene_net.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.motion_net.parameters(), 1.0)
        if self.use_bias_field:
            torch.nn.utils.clip_grad_norm_(self.bias_net.parameters(), 1.0)
        
        # Optimizer step
        self.scene_optimizer.step()
        self.motion_optimizer.step()
        if self.use_bias_field:
            self.bias_optimizer.step()
        
        return {
            'total': total_loss.item(),
            'data': data_loss.item(),
            'motion_smoothness': motion_smoothness_loss if isinstance(motion_smoothness_loss, float) else motion_smoothness_loss.item(),
            'scene_smoothness': scene_smoothness_loss if isinstance(scene_smoothness_loss, float) else scene_smoothness_loss.item(),
            'tv': tv_loss if isinstance(tv_loss, float) else tv_loss.item()
        }
    
    def train(self, num_steps=10000, batch_size=65536, progress_interval=500):
        """Train the neural field model with improved training loop"""
        loss_history = {'total': [], 'data': [], 'motion_smoothness': [], 'scene_smoothness': [], 'tv': []}
        
        # Dynamic batch size adjustment
        current_batch_size = batch_size
        
        # For early stopping
        best_loss = float('inf')
        patience = 10  # Increased patience
        patience_counter = 0
        
        for step in trange(num_steps, desc="Training neural SR model", ncols=100):
            try:
                # Train step
                losses = self.train_step(current_batch_size, step, num_steps)
                
                # Record losses
                for k, v in losses.items():
                    loss_history[k].append(v)
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and current_batch_size > 1000:
                    # Reduce batch size and try again
                    current_batch_size = current_batch_size // 2
                    torch.cuda.empty_cache()
                    print(f"\nReducing batch size to {current_batch_size} due to OOM error")
                    continue
                else:
                    # Re-raise if not an OOM error
                    raise
            
            # Show progress and update schedulers
            if step % progress_interval == 0 or step == num_steps - 1:
                # Report current loss
                tqdm.write(f"Step {step}, Loss: {losses['total']:.6f} (Data: {losses['data']:.6f})")
                
                # Update LR schedulers
                self.scene_scheduler.step(losses['data'])
                self.motion_scheduler.step(losses['data'])
                
                # Check for early stopping
                if losses['data'] < best_loss:
                    best_loss = losses['data']
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at step {step}")
                    break
        
        return loss_history
    
    def reconstruct(self, time_idx=0, super_resolution_factor=1):
        """Reconstruct the HR image at a specific time index with optional super-resolution"""
        with torch.no_grad():
            # Create grid with optional super-resolution
            if super_resolution_factor > 1:
                # Create higher resolution grid
                sr_h, sr_w = self.hr_h * super_resolution_factor, self.hr_w * super_resolution_factor
                y = torch.linspace(0, 1, sr_h, device=DEVICE).to(DTYPE)
                x = torch.linspace(0, 1, sr_w, device=DEVICE).to(DTYPE)
                yv, xv = torch.meshgrid(y, x, indexing='ij')
                coords_sr = torch.stack([xv.flatten(), yv.flatten()], dim=1)
                
                # Add time dimension
                time = torch.full((coords_sr.shape[0], 1), time_idx / max(1, len(self.lr_images) - 1), 
                                device=DEVICE).to(DTYPE)
                coords_with_time = torch.cat([coords_sr, time], dim=1)
                
                # Process in chunks to avoid OOM
                chunk_size = 65536
                outputs = []
                
                for i in range(0, coords_sr.shape[0], chunk_size):
                    # Get chunk
                    coords_chunk = coords_with_time[i:i+chunk_size]
                    
                    # Get motion at these coordinates and time
                    motion = self.motion_net(coords_chunk)
                    
                    # Apply motion to get corrected coordinates
                    corrected_coords = coords_chunk[:, :2] + motion
                    
                    # Clamp coordinates to [0, 1]
                    corrected_coords = torch.clamp(corrected_coords, 0.0, 1.0)
                    
                    # Evaluate scene at corrected coordinates
                    scene_values = self.scene_net(corrected_coords)
                    
                    # Apply bias correction if enabled
                    if self.use_bias_field:
                        bias = self.bias_net(coords_chunk)
                        outputs.append(scene_values + bias * 0.1)
                    else:
                        outputs.append(scene_values)
                
                # Concatenate results
                full_output = torch.cat(outputs, dim=0)
                
                # Reshape to image
                image = full_output.reshape(sr_h, sr_w).float().cpu()
            else:
                # Standard resolution reconstruction
                # Add time dimension to coordinates
                time = torch.full((self.coords_hr.shape[0], 1), time_idx / max(1, len(self.lr_images) - 1), 
                                device=DEVICE).to(DTYPE)
                coords_with_time = torch.cat([self.coords_hr, time], dim=1)
                
                # Process in chunks to avoid OOM
                chunk_size = 65536
                outputs = []
                
                for i in range(0, self.coords_hr.shape[0], chunk_size):
                    # Get chunk
                    coords_chunk = coords_with_time[i:i+chunk_size]
                    
                    # Get motion at these coordinates and time
                    motion = self.motion_net(coords_chunk)
                    
                    # Apply motion to get corrected coordinates
                    corrected_coords = coords_chunk[:, :2] + motion
                    
                    # Clamp coordinates to [0, 1]
                    corrected_coords = torch.clamp(corrected_coords, 0.0, 1.0)
                    
                    # Evaluate scene at corrected coordinates
                    scene_values = self.scene_net(corrected_coords)
                    
                    # Apply bias correction if enabled
                    if self.use_bias_field:
                        bias = self.bias_net(coords_chunk)
                        outputs.append(scene_values + bias * 0.1)
                    else:
                        outputs.append(scene_values)
                
                # Concatenate results
                full_output = torch.cat(outputs, dim=0)
                
                # Reshape to image
                image = full_output.reshape(self.hr_h, self.hr_w).float().cpu()
            
        return image

def main():
    # Create output directory
    output_dir = 'neural_sr_comparison'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the test image
    script_dir = os.path.dirname(os.path.abspath(__file__))
    hr_image_path = os.path.join(script_dir, 'pics_boat.tif')
    
    try:
        hr_image = tiff.imread(hr_image_path)
    except FileNotFoundError:
        print(f"Warning: Could not find {hr_image_path}")
        print("Using synthetic test image instead")
        # Create synthetic test image with fine details
        size = 512
        x = np.linspace(-4, 4, size)
        y = np.linspace(-4, 4, size)
        xx, yy = np.meshgrid(x, y)
        # Create pattern with varying frequencies
        pattern = np.sin(xx**2 + yy**2) + np.sin(5*xx) * np.cos(5*yy)
        hr_image = (pattern - pattern.min()) / (pattern.max() - pattern.min())
    
    hr_image = torch.tensor(hr_image, dtype=torch.float32)
    
    # Normalize to [0, 1]
    hr_image = hr_image / hr_image.max()
    
    # Parameters
    downscale_factor = 4
    num_shifts = 16
    
    print(f"High-resolution image shape: {hr_image.shape}")
    print(f"Downscale factor: {downscale_factor}")
    print(f"Number of shifted images: {num_shifts}")
    
    # Generate low-resolution shifted images
    print("Generating shifted low-resolution images...")
    lr_images, shifts = generate_shifted_downsampled_images(hr_image, downscale_factor, num_shifts)
    
    # Take a single downsampled image for comparison
    single_lr = lr_images[0]
    
    # CNN-based Neural SR approach
    print("Running CNN-based Neural SR approach...")
    cnn_sr_result = neural_sr_refinement(lr_images, shifts, downscale_factor, hr_image)
    
    # Apply improved neural field SR
    print("Training improved neural field SR model...")
    neural_sr = ImprovedNeuralFieldSR(lr_images, shifts, downscale_factor, hr_image.shape)
    loss_history = neural_sr.train(num_steps=5000, batch_size=32768)
    
    # Get final SR image at standard resolution
    print("Generating output images...")
    nstm_sr_result = neural_sr.reconstruct()
    
    # Calculate metrics
    data_range = 1.0  # Images normalized to [0,1]
    psnr_cnn = psnr(hr_image.numpy(), cnn_sr_result.numpy())
    psnr_nstm = psnr(hr_image.numpy(), nstm_sr_result.numpy())
    
    ssim_cnn = ssim(hr_image.numpy(), cnn_sr_result.numpy(), data_range=data_range)
    ssim_nstm = ssim(hr_image.numpy(), nstm_sr_result.numpy(), data_range=data_range)
    
    # Print metrics
    print("\nMetrics vs. ground truth:")
    print(f"CNN-based Neural SR - PSNR: {psnr_cnn:.2f} dB, SSIM: {ssim_cnn:.4f}")
    print(f"Improved NSTM SR - PSNR: {psnr_nstm:.2f} dB, SSIM: {ssim_nstm:.4f}")
    
    # Create the comparison plot
    plt.figure(figsize=(16, 16))
    
    # Original HR
    plt.subplot(2, 2, 1)
    plt.imshow(hr_image, cmap='gray')
    plt.title('Original High-Resolution Image')
    plt.axis('off')
    
    # Low-res
    plt.subplot(2, 2, 2)
    plt.imshow(single_lr, cmap='gray')
    plt.title('Single Low-Resolution Image')
    plt.axis('off')
    
    # CNN-based Neural SR
    plt.subplot(2, 2, 3)
    plt.imshow(cnn_sr_result, cmap='gray')
    plt.title(f'CNN-based Neural SR\nPSNR: {psnr_cnn:.2f} dB, SSIM: {ssim_cnn:.4f}')
    plt.axis('off')
    
    # NSTM-based Neural SR
    plt.subplot(2, 2, 4)
    plt.imshow(nstm_sr_result, cmap='gray')
    plt.title(f'Improved NSTM SR\nPSNR: {psnr_nstm:.2f} dB, SSIM: {ssim_nstm:.4f}')
    plt.axis('off')
    
    plt.tight_layout()
    # Save the comparison plot
    plt.savefig(os.path.join(output_dir, 'neural_sr_comparison.png'))
    plt.close()
    
    print(f"Comparison plot saved to {output_dir}/neural_sr_comparison.png")

if __name__ == "__main__":
    main()