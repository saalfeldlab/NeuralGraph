import numpy as np
import torch
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
DTYPE = torch.float16  # Use half precision for consistency with tcnn

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

def classical_pixel_sr(lr_images, shifts, downscale_factor):
    """
    Perform classical pixel super-resolution on a set of low-resolution images.
    """
    # Determine the high-resolution size
    lr_h, lr_w = lr_images[0].shape
    hr_h, hr_w = lr_h * downscale_factor, lr_w * downscale_factor
    
    # Initialize high-res grid with zeros
    hr_image = torch.zeros((hr_h, hr_w), dtype=torch.float32)
    contribution_count = torch.zeros((hr_h, hr_w), dtype=torch.float32)
    
    # For each shifted low-res image
    for idx, (lr_img, shift) in enumerate(tqdm(zip(lr_images, shifts), total=len(lr_images), 
                                              desc="Processing images", ncols=80)):
        shift_y, shift_x = shift
        
        # Calculate the starting position in the high-res grid
        start_y = int(round(shift_y * downscale_factor))
        start_x = int(round(shift_x * downscale_factor))
        
        # Place the values in the high-res grid with the right stride
        for i in range(lr_h):
            for j in range(lr_w):
                hr_y = (start_y + i * downscale_factor) % hr_h
                hr_x = (start_x + j * downscale_factor) % hr_w
                
                hr_image[hr_y, hr_x] += lr_img[i, j]
                contribution_count[hr_y, hr_x] += 1
    
    # Average values where we have multiple contributions
    valid_mask = contribution_count > 0
    hr_image[valid_mask] /= contribution_count[valid_mask]
    
    # Simple interpolation for missing pixels
    if not torch.all(valid_mask):
        missing_mask = ~valid_mask
        
        # Use a simple iterative diffusion to fill missing values
        for iter_idx in trange(20, desc="Filling missing values", ncols=80):
            # Create a new tensor with current HR image
            new_hr = hr_image.clone()
            
            # For each missing pixel, average surrounding valid pixels
            kernel = torch.ones((3, 3), dtype=torch.float32)
            kernel[1, 1] = 0  # Don't include the center pixel
            
            # Apply convolution to get weighted sum of neighbors
            from torch.nn.functional import conv2d
            
            # Add padding and reshape for conv2d
            padded = torch.nn.functional.pad(hr_image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate')
            neighborhood_sum = conv2d(padded, kernel.unsqueeze(0).unsqueeze(0), padding=0)
            
            # Also count valid neighbors with another convolution
            valid_padded = torch.nn.functional.pad(valid_mask.float().unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='constant')
            valid_count = conv2d(valid_padded, kernel.unsqueeze(0).unsqueeze(0), padding=0)
            
            # Compute average of valid neighbors where we have some
            has_valid_neighbors = valid_count.squeeze() > 0
            missing_with_neighbors = missing_mask & has_valid_neighbors.bool()
            
            # Update missing pixels with average of valid neighbors
            if torch.any(missing_with_neighbors):
                new_hr[missing_with_neighbors] = (neighborhood_sum.squeeze()[missing_with_neighbors] / 
                                                  valid_count.squeeze()[missing_with_neighbors])
            
            # Update image and valid mask
            hr_image = new_hr
            newly_valid = missing_with_neighbors
            valid_mask = valid_mask | newly_valid
            missing_mask = ~valid_mask
            
            # Stop if we've filled all missing pixels
            if not torch.any(missing_mask):
                break
    
    return hr_image

def create_instantngp_config(precision="half"):
    """Create network configuration for InstantNGP"""
    return {
        "encoding": {
            "otype": "HashGrid",
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hashmap_size": 19,
            "base_resolution": 16,
            "per_level_scale": 1.5
        },
        "network": {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 64,
            "n_hidden_layers": 2,
            "precision": precision
        }
    }

class DualNeuralFieldSR:
    """Neural field-based super-resolution using dual networks like NSTM"""
    def __init__(self, lr_images, shifts, downscale_factor, hr_shape):
        self.lr_images = [img.to(DEVICE).to(DTYPE) for img in lr_images]
        self.shifts = shifts
        self.downscale_factor = downscale_factor
        self.hr_h, self.hr_w = hr_shape
        
        # Create coordinate grids
        self.create_grids()
        
        # Set up neural fields
        self.setup_neural_fields()
    
    def create_grids(self):
        """Create normalized coordinate grids"""
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
            
            # Create coordinate grid for this LR image
            y = torch.arange(0, lr_h, device=DEVICE).float().to(DTYPE)
            x = torch.arange(0, lr_w, device=DEVICE).float().to(DTYPE)
            yv, xv = torch.meshgrid(y, x, indexing='ij')
            
            # Scale to [0, 1] and apply shift
            norm_y = (yv.flatten() * self.downscale_factor + shift[0] * self.downscale_factor) / (self.hr_h - 1)
            norm_x = (xv.flatten() * self.downscale_factor + shift[1] * self.downscale_factor) / (self.hr_w - 1)
            
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
    
    def setup_neural_fields(self):
        """Set up dual neural fields for scene and motion"""
        self.config = create_instantngp_config()
        
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
        
        # Create optimizers
        self.scene_optimizer = torch.optim.Adam(self.scene_net.parameters(), lr=5e-4)
        self.motion_optimizer = torch.optim.Adam(self.motion_net.parameters(), lr=5e-4)
    
    def sample_training_points(self, batch_size):
        """Sample random points from all LR images"""
        # Random indices
        indices = torch.randint(0, len(self.values_lr_all), (batch_size,), device=DEVICE)
        
        # Get coordinates and values
        coords = self.coords_lr_all[indices]
        values = self.values_lr_all[indices]
        
        return coords, values.unsqueeze(1)
    
    def train_step(self, batch_size):
        """Perform a single training step"""
        # Sample training points
        coords, targets = self.sample_training_points(batch_size)
        
        # Get motion at these coordinates and time
        motion = self.motion_net(coords)
        
        # Apply motion to get corrected coordinates
        corrected_coords = coords[:, :2] + motion
        
        # Clamp coordinates to [0, 1]
        corrected_coords = torch.clamp(corrected_coords, 0.0, 1.0)
        
        # Evaluate scene at corrected coordinates
        predictions = self.scene_net(corrected_coords)
        
        # Calculate loss
        loss = torch.nn.functional.mse_loss(predictions, targets)
        
        # Backward pass for scene net
        self.scene_optimizer.zero_grad()
        self.motion_optimizer.zero_grad()
        loss.backward()
        self.scene_optimizer.step()
        self.motion_optimizer.step()
        
        return loss.item()
    
    def train(self, num_steps=5000, batch_size=65536, progress_interval=500):
        """Train the neural field model"""
        loss_history = []
        progress_images = []
        progress_motion = []
        
        # Dynamic batch size adjustment
        current_batch_size = batch_size
        
        # For early stopping
        best_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for step in trange(num_steps, desc="Training neural SR model", ncols=100):
            try:
                # Train step
                loss = self.train_step(current_batch_size)
                loss_history.append(loss)
                
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
            
            # Show progress
            if step % progress_interval == 0 or step == num_steps - 1:
                # Reconstruct image
                sr_image = self.reconstruct()
                progress_images.append(sr_image.cpu().numpy())
                
                # Visualize motion field
                motion_vis = self.visualize_motion()
                progress_motion.append(motion_vis)
                
                # Report current loss
                tqdm.write(f"Step {step}, Loss: {loss:.6f}, Batch size: {current_batch_size}")
                
                # Check for early stopping
                if loss < best_loss:
                    best_loss = loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at step {step}")
                    break
        
        return loss_history, progress_images, progress_motion
    
    def reconstruct(self, time_idx=0):
        """Reconstruct the HR image at a specific time index"""
        with torch.no_grad():
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
                outputs.append(self.scene_net(corrected_coords))
            
            # Concatenate results
            full_output = torch.cat(outputs, dim=0)
            
            # Reshape to image
            image = full_output.reshape(self.hr_h, self.hr_w).float().cpu()
            
        return image
    
    def extract_static_scene(self):
        """Extract the static scene without any motion"""
        with torch.no_grad():
            # Process in chunks to avoid OOM
            chunk_size = 65536
            outputs = []
            
            for i in range(0, self.coords_hr.shape[0], chunk_size):
                # Get chunk
                coords_chunk = self.coords_hr[i:i+chunk_size]
                
                # Evaluate scene at original coordinates
                outputs.append(self.scene_net(coords_chunk))
            
            # Concatenate results
            full_output = torch.cat(outputs, dim=0)
            
            # Reshape to image
            image = full_output.reshape(self.hr_h, self.hr_w).float().cpu()
            
        return image
    
    def visualize_motion(self, time_idx=0, scale=10.0):
        """Visualize the motion field at a specific time index"""
        with torch.no_grad():
            # Create a lower resolution grid for visualization
            vis_h, vis_w = min(40, self.hr_h // 10), min(40, self.hr_w // 10)
            y = torch.linspace(0, 1, vis_h, device=DEVICE).to(DTYPE)
            x = torch.linspace(0, 1, vis_w, device=DEVICE).to(DTYPE)
            yv, xv = torch.meshgrid(y, x, indexing='ij')
            vis_coords = torch.stack([xv.flatten(), yv.flatten()], dim=1)
            
            # Add time dimension
            time = torch.full((vis_coords.shape[0], 1), time_idx / max(1, len(self.lr_images) - 1), 
                            device=DEVICE).to(DTYPE)
            vis_coords_with_time = torch.cat([vis_coords, time], dim=1)
            
            # Get motion at these coordinates
            motion = self.motion_net(vis_coords_with_time)
            
            # Reshape to grid
            motion_x = motion[:, 0].reshape(vis_h, vis_w).float().cpu().numpy()
            motion_y = motion[:, 1].reshape(vis_h, vis_w).float().cpu().numpy()
            
            # Create figure
            plt.figure(figsize=(8, 8))
            plt.title(f"Motion Field (t={time_idx})")
            
            # Create meshgrid for visualization
            Y, X = np.mgrid[0:vis_h, 0:vis_w]
            
            # Plot motion field
            plt.quiver(X, Y, motion_x * scale, motion_y * scale, 
                     color='red', scale=1.0, scale_units='xy', width=0.002)
            
            # Display scene as background
            scene = self.extract_static_scene()
            plt.imshow(scene.numpy(), cmap='gray', alpha=0.5)
            
            # Save figure to buffer
            fig = plt.gcf()
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            
        return img

def main():
    # Create output directory
    output_dir = 'sr_nstm_outputs'
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
    single_lr_upscaled = torch.nn.functional.interpolate(
        single_lr.unsqueeze(0).unsqueeze(0), 
        scale_factor=downscale_factor, 
        mode='bicubic'
    ).squeeze()
    
    # Apply classical pixel super-resolution
    print("Running classical pixel super-resolution...")
    classical_sr_result = classical_pixel_sr(lr_images, shifts, downscale_factor)
    
    # Apply neural field SR
    print("Training neural field SR model...")
    neural_sr = DualNeuralFieldSR(lr_images, shifts, downscale_factor, hr_image.shape)
    loss_history, progress_images, progress_motion = neural_sr.train(num_steps=2000, batch_size=16384)
    
    # Get final SR image and static scene
    sr_result = neural_sr.reconstruct()
    static_scene = neural_sr.extract_static_scene()
    
    # Calculate metrics
    data_range = 1.0  # Images normalized to [0,1]
    psnr_bicubic = psnr(hr_image.numpy(), single_lr_upscaled.numpy())
    psnr_classical = psnr(hr_image.numpy(), classical_sr_result.numpy())
    psnr_neural = psnr(hr_image.numpy(), sr_result.numpy())
    psnr_static = psnr(hr_image.numpy(), static_scene.numpy())
    
    ssim_bicubic = ssim(hr_image.numpy(), single_lr_upscaled.numpy(), data_range=data_range)
    ssim_classical = ssim(hr_image.numpy(), classical_sr_result.numpy(), data_range=data_range)
    ssim_neural = ssim(hr_image.numpy(), sr_result.numpy(), data_range=data_range)
    ssim_static = ssim(hr_image.numpy(), static_scene.numpy(), data_range=data_range)
    
    # Print metrics
    print("\nMetrics vs. ground truth:")
    print(f"Bicubic upscaling - PSNR: {psnr_bicubic:.2f} dB, SSIM: {ssim_bicubic:.4f}")
    print(f"Classical SR - PSNR: {psnr_classical:.2f} dB, SSIM: {ssim_classical:.4f}")
    print(f"Neural SR (with motion) - PSNR: {psnr_neural:.2f} dB, SSIM: {ssim_neural:.4f}")
    print(f"Neural SR (static scene) - PSNR: {psnr_static:.2f} dB, SSIM: {ssim_static:.4f}")
    
    # Plot loss history
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Step')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.savefig(f'{output_dir}/loss_history.png')
    plt.close()
    
    # Visualize results
    plt.figure(figsize=(15, 20))
    
    # Original and LR
    plt.subplot(3, 2, 1)
    plt.imshow(hr_image, cmap='gray')
    plt.title('Original High-Resolution Image')
    plt.axis('off')
    
    plt.subplot(3, 2, 2)
    plt.imshow(single_lr, cmap='gray')
    plt.title('Single Low-Resolution Image')
    plt.axis('off')
    
    # Classical and neural SR
    plt.subplot(3, 2, 3)
    plt.imshow(classical_sr_result, cmap='gray')
    plt.title(f'Classical Pixel SR\nPSNR: {psnr_classical:.2f} dB, SSIM: {ssim_classical:.4f}')
    plt.axis('off')
    
    plt.subplot(3, 2, 4)
    plt.imshow(sr_result, cmap='gray')
    plt.title(f'Neural Field SR (with Motion)\nPSNR: {psnr_neural:.2f} dB, SSIM: {ssim_neural:.4f}')
    plt.axis('off')
    
    # Static scene and motion field
    plt.subplot(3, 2, 5)
    plt.imshow(static_scene, cmap='gray')
    plt.title(f'Neural Field (Static Scene Only)\nPSNR: {psnr_static:.2f} dB, SSIM: {ssim_static:.4f}')
    plt.axis('off')
    
    plt.subplot(3, 2, 6)
    plt.imshow(progress_motion[-1])
    plt.title('Motion Field Visualization')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sr_comparison.png')
    plt.close()
    
    # Save output images
    tiff.imwrite(f'{output_dir}/neural_sr_result.tif', sr_result.numpy())
    tiff.imwrite(f'{output_dir}/neural_sr_static.tif', static_scene.numpy())
    
    # Create animation of training progress
    try:
        import imageio.v3 as imageio
        
        # Save SR progress
        progress_frames = []
        for img in progress_images:
            # Normalize to 0-255
            img_norm = np.clip(img, 0, 1) * 255
            progress_frames.append(img_norm.astype(np.uint8))
            
        if progress_frames:
            imageio.imwrite(f'{output_dir}/sr_progress.gif', progress_frames, loop=0, fps=5)
        
        # Save motion field progress
        if progress_motion:
            imageio.imwrite(f'{output_dir}/motion_progress.gif', progress_motion, loop=0, fps=5)
            
    except ImportError:
        print("Could not create animations - imageio not available")
    
    print(f"All results saved to '{output_dir}'")

if __name__ == "__main__":
    main()
