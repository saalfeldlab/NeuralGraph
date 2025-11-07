import numpy as np
import torch
import matplotlib.pyplot as plt
import tifffile as tiff
import os
from tqdm import trange
import tinycudann as tcnn
from scipy.ndimage import map_coordinates
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Check for CUDA
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this code")

# Set constants
DEVICE = torch.device("cuda:0")
DTYPE = torch.float16  # Critical: Use half precision consistently

# Create output directory
output_dir = 'NSTM_outputs'
os.makedirs(output_dir, exist_ok=True)

def generate_motion_frames(image, num_frames=16, motion_intensity=0.015):
    """Generate simple motion sequence using SciPy"""
    h, w = image.shape
    images = []
    
    for t in range(num_frames):
        # Create coordinate grids
        y_grid, x_grid = np.meshgrid(
            np.linspace(0, 1, h),
            np.linspace(0, 1, w),
            indexing='ij'
        )
        
        # Create displacement fields with time-varying frequency
        t_norm = t / num_frames
        freq_x = 2 + t_norm * 2
        freq_y = 3 + t_norm * 1
        
        dx = motion_intensity * np.sin(freq_x * np.pi * y_grid) * np.cos(freq_y * np.pi * x_grid)
        dy = motion_intensity * np.cos(freq_x * np.pi * y_grid) * np.sin(freq_y * np.pi * x_grid)
        
        # Create source coordinates
        coords_y, coords_x = np.meshgrid(
            np.arange(h),
            np.arange(w),
            indexing='ij'
        )
        
        # Apply displacement
        sample_y = coords_y - dy * h
        sample_x = coords_x - dx * w
        
        # Ensure coordinates are within bounds
        sample_y = np.clip(sample_y, 0, h - 1)
        sample_x = np.clip(sample_x, 0, w - 1)
        
        # Warp the image
        warped = map_coordinates(image, [sample_y, sample_x], order=1, mode='reflect')
        
        images.append(warped)
    
    return images

def create_network_config(precision="half"):
    """Create a network configuration"""
    return {
        "encoding": {
            "otype": "HashGrid",
            "n_levels": 8,
            "n_features_per_level": 2,
            "log2_hashmap_size": 16,
            "base_resolution": 16,
            "per_level_scale": 1.5
        },
        "network": {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 32,
            "n_hidden_layers": 2,
            "precision": precision
        }
    }

class NeuralSpaceTimeModel:
    """Simple class to hold both networks and provide convenience methods"""
    def __init__(self, motion_net, scene_net, res):
        self.motion_net = motion_net
        self.scene_net = scene_net
        self.res = res
        
        # Create coordinate grid
        y = torch.linspace(0, 1, res, device=DEVICE, dtype=DTYPE)
        x = torch.linspace(0, 1, res, device=DEVICE, dtype=DTYPE)
        yv, xv = torch.meshgrid(y, x, indexing='ij')
        self.coords_2d = torch.stack([xv.flatten(), yv.flatten()], dim=1)
    
    def get_image_at_time(self, t):
        """Get image at time t"""
        with torch.no_grad():
            # Add time dimension
            t_tensor = torch.full_like(self.coords_2d[:, 0:1], t)
            coords_3d = torch.cat([self.coords_2d, t_tensor], dim=1)
            
            # Get motion
            motion = self.motion_net(coords_3d)
            
            # Apply motion to coordinates
            corrected_coords = self.coords_2d + motion
            corrected_coords = torch.clamp(corrected_coords, 0, 1)
            
            # Get image values
            values = self.scene_net(corrected_coords)
            
            # Reshape to image
            image = values.reshape(self.res, self.res).cpu().numpy()
            
        return image
    
    def get_motion_field_at_time(self, t, scale=1.0):
        """Get motion field at time t"""
        with torch.no_grad():
            # Add time dimension
            t_tensor = torch.full_like(self.coords_2d[:, 0:1], t)
            coords_3d = torch.cat([self.coords_2d, t_tensor], dim=1)
            
            # Get motion
            motion = self.motion_net(coords_3d)
            
            # Reshape to grid
            motion_x = motion[:, 0].reshape(self.res, self.res).cpu().numpy() * scale
            motion_y = motion[:, 1].reshape(self.res, self.res).cpu().numpy() * scale
            
        return motion_x, motion_y
    
    def get_fixed_scene(self):
        """Extract the fixed scene without motion"""
        with torch.no_grad():
            # Forward pass through scene network with original coordinates
            values = self.scene_net(self.coords_2d)
            
            # Reshape to image
            fixed_scene = values.reshape(self.res, self.res).cpu().numpy()
            
        return fixed_scene

def train_model(num_training_steps=3000):
    # For video: store fixed scene frames at intervals
    video_frames = []
    video_interval = max(1, num_training_steps // 100)  # Save 100 frames over training
    """Train the neural space-time model and visualize results"""
    # Load image
    print("Loading image...")
    hr_image = tiff.imread('pics_boat.tif')
    hr_image = hr_image.astype(np.float32) / hr_image.max()
    
    # Generate motion frames with reduced intensity
    print("Generating motion frames...")
    num_frames = 16
    motion_intensity = 0.015
    motion_images = generate_motion_frames(hr_image, num_frames, motion_intensity)
    
    # Visualize motion frames in a grid
    grid_size = int(np.ceil(np.sqrt(num_frames)))
    plt.figure(figsize=(16, 16))
    for i in range(num_frames):
        plt.subplot(grid_size, grid_size, i+1)
        plt.imshow(motion_images[i], cmap='gray')
        plt.title(f'Frame {i}')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/motion_frames.png')
    plt.close()
    
    # Save individual frames
    for i, img in enumerate(motion_images):
        tiff.imwrite(f'{output_dir}/motion_frame_{i:02d}.tif', img.astype(np.float32))
    
    # Use full resolution
    res = hr_image.shape[0]
    scaled_images = motion_images
    scaled_original = hr_image

    # Convert to tensors
    motion_tensors = [torch.tensor(img, dtype=DTYPE).to(DEVICE) for img in scaled_images]
    
    # Create networks
    print("Creating networks...")
    config = create_network_config(precision="half")
    
    motion_net = tcnn.NetworkWithInputEncoding(
        n_input_dims=3,  # x, y, t
        n_output_dims=2,  # δx, δy
        encoding_config=config["encoding"],
        network_config=config["network"]
    ).to(DEVICE)
    
    scene_net = tcnn.NetworkWithInputEncoding(
        n_input_dims=2,  # x, y
        n_output_dims=1,  # Intensity
        encoding_config=config["encoding"],
        network_config=config["network"]
    ).to(DEVICE)
    
    # Create model wrapper
    model = NeuralSpaceTimeModel(motion_net, scene_net, res)
    
    # Create optimizer
    params = list(motion_net.parameters()) + list(scene_net.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)  # Using the modified learning rate
    
    # Create coordinate grid
    y = torch.linspace(0, 1, res, device=DEVICE, dtype=DTYPE)
    x = torch.linspace(0, 1, res, device=DEVICE, dtype=DTYPE)
    yv, xv = torch.meshgrid(y, x, indexing='ij')
    coords_2d = torch.stack([xv.flatten(), yv.flatten()], dim=1)
    
    # Learning rate schedule - reduce after 50% of training
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=[num_training_steps // 2, num_training_steps * 3 // 4], 
        gamma=0.1
    )
    
    # Training loop with progress bar
    print(f"Training for {num_training_steps} steps...")
    loss_history = []
    batch_size = min(4096, res*res)
    
    for step in trange(num_training_steps):
        # Select random frame
        t_idx = np.random.randint(0, num_frames)
        t_normalized = t_idx / (num_frames - 1)
        
        # Select random batch of pixels
        indices = torch.randperm(res*res, device=DEVICE)[:batch_size]
        batch_coords = coords_2d[indices]
        
        # Target values
        target = motion_tensors[t_idx].reshape(-1, 1)[indices]
        
        # Create 3D coordinates
        t_tensor = torch.full_like(batch_coords[:, 0:1], t_normalized)
        coords_3d = torch.cat([batch_coords, t_tensor], dim=1)
        
        # Forward pass - motion network
        motion = motion_net(coords_3d)
        
        # Apply motion
        corrected_coords = batch_coords + motion
        corrected_coords = torch.clamp(corrected_coords, 0, 1)
        
        # Forward pass - scene network
        values = scene_net(corrected_coords)
        
        # Loss
        loss = torch.nn.functional.mse_loss(values, target)
        
        # Add regularization for smoother motion
        motion_smoothness = torch.mean(torch.abs(motion))
        total_loss = loss + 0.01 * motion_smoothness
        
        # Optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Update learning rate
        scheduler.step()
        
        # Record loss
        loss_history.append(loss.item())
        
        # Save checkpoints and visualize
        if (step+1) % 300 == 0 or step == 0 or step == num_training_steps-1:  # Show progress every 300 steps
            print(f"Step {step+1}, Loss: {loss.item():.6f}")
            # ...existing visualization code...
            with torch.no_grad():
                fixed_scene = model.get_fixed_scene()
            # Save frame for video at regular intervals
            if (step+1) % video_interval == 0 or step == 0 or step == num_training_steps-1:
                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(fixed_scene, cmap='gray', vmin=0, vmax=1)
                ax.set_title(f'Fixed Scene\nStep {step+1}')
                ax.axis('off')
                fig.tight_layout()
                fig.canvas.draw()
        fig.canvas.draw()
        rgba_buffer = np.asarray(fig.canvas.buffer_rgba())
        frame = rgba_buffer[..., :3]
        # Compute PSNR for this frame (assuming scaled_original and scaled_images are available)
        if 'scaled_original' in locals() and 'scaled_images' in locals():
            # Use the current frame index to get the corresponding image
            frame_idx = len(video_frames)
            if frame_idx < len(scaled_images):
                frame_psnr = psnr(scaled_original, scaled_images[frame_idx])
                print(f"Frame {frame_idx}: PSNR = {frame_psnr:.2f} dB")
        video_frames.append(frame)
        plt.close(fig)
    # After training, save video
    import imageio.v2 as imageio
    video_path = f'{output_dir}/fixed_scene_progress.mp4'
    if video_frames:
        print(f"Saving video to: {video_path}")
        ext = video_path.lower().split('.')[-1]
        # Increase FPS for faster playback
        if ext in ('mp4', 'gif'):
            # Set FPS so video duration is 10 seconds
            target_duration = 10.0
            fps = max(1, int(len(video_frames) / target_duration))
            imageio.mimsave(video_path, video_frames, fps=fps)
        else:
            imageio.mimsave(video_path, video_frames)
        print(f"Fixed scene progress video saved: {video_path}")
    else:
        print("No video frames were generated for fixed scene progress.")
    
    # Plot loss history
    plt.figure(figsize=(8, 6))
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/loss_history.png')
    plt.close()
    
    print("Training complete!")
    return model, scaled_original, scaled_images, res

def create_motion_field_visualization(img, motion_x, motion_y, res, step_size=8):
    """Create motion field visualization with clear arrows using OpenCV"""
    # Create RGB image from grayscale
    img_rgb = np.stack([img, img, img], axis=2)
    img_uint8 = (np.clip(img_rgb, 0, 1) * 255).astype(np.uint8)
    
    # Scale factor to make arrows visible
    arrow_scale = 100.0  # Further increased scale for even longer arrows
    
    # Create coordinate grid with larger step_size for fewer arrows
    step_size = max(1, res // 10)  # Fewer arrows, e.g., 10x10 grid
    y_coords = np.arange(0, res, step_size)
    x_coords = np.arange(0, res, step_size)
    
    # Draw arrows for motion vectors
    for y in y_coords:
        for x in x_coords:
            # Get motion at this point (scale for visibility)
            dx = int(motion_x[y, x] * arrow_scale)
            dy = int(motion_y[y, x] * arrow_scale)
            # Calculate end point with boundary checking
            end_x = np.clip(x + dx, 0, res-1)
            end_y = np.clip(y + dy, 0, res-1)
            # Draw arrow with thicker line and more visible color (green)
            cv2.arrowedLine(
                img_uint8,
                (x, y),                 # Start point
                (end_x, end_y),         # End point
                (0, 255, 0),            # Color (bright green)
                3,                      # Thickness
                tipLength=0.4,          # Arrow tip size
                line_type=cv2.LINE_AA   # Anti-aliased line
            )
    
    return img_uint8

def create_quad_panel_video(model, original_image, motion_frames, res, num_frames=90):
    """Create a four-panel comparison video including fixed scene"""
    print("Generating quad-panel comparison video...")
    
    # Create temporary directory for frames
    temp_dir = f"{output_dir}/temp_frames"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Prepare for interpolation panel
    
    # Find original frames at evenly spaced time points
    original_indices = [min(int(round(i * (len(motion_frames) - 1) / (num_frames - 1))), len(motion_frames) - 1) 
                        for i in range(num_frames)]
    
    # Generate frames for each time point
    for i in trange(num_frames, desc="Creating video frames"):
        # Get normalized time
        t = i / (num_frames - 1)

        # Get original frame
        orig_idx = original_indices[i]
        orig_frame = motion_frames[orig_idx]
        if orig_frame.shape != (res, res):
            from skimage.transform import resize
            orig_frame = resize(orig_frame, (res, res), order=1, anti_aliasing=True)

        # Get neural field reconstruction
        recon_frame = model.get_image_at_time(t)

        # Get motion field for visualization
        motion_x, motion_y = model.get_motion_field_at_time(t, scale=5.0)

        # Create motion field visualization
        motion_vis = create_motion_field_visualization(recon_frame, motion_x, motion_y, res, step_size=8)

        # Interpolation between two nearest frames
        t_idx_float = t * (len(motion_frames) - 1)
        t_idx0 = int(np.floor(t_idx_float))
        t_idx1 = min(t_idx0 + 1, len(motion_frames) - 1)
        alpha = t_idx_float - t_idx0
        interp_frame = (1 - alpha) * motion_frames[t_idx0] + alpha * motion_frames[t_idx1]
        interp_uint8 = (np.clip(interp_frame, 0, 1) * 255).astype(np.uint8)
        interp_rgb = np.stack([interp_uint8, interp_uint8, interp_uint8], axis=2)

        # Convert everything to uint8 RGB
        orig_uint8 = (np.clip(orig_frame, 0, 1) * 255).astype(np.uint8)
        recon_uint8 = (np.clip(recon_frame, 0, 1) * 255).astype(np.uint8)
        orig_rgb = np.stack([orig_uint8, orig_uint8, orig_uint8], axis=2)
        recon_rgb = np.stack([recon_uint8, recon_uint8, recon_uint8], axis=2)

        # Add small text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1

        # Position in bottom left
        margin = 5
        y_pos = res - margin

        # Add text to each panel
        cv2.putText(orig_rgb, f"Original t={t:.2f}", (margin, y_pos), font, font_scale, (0, 0, 0), thickness+1)
        cv2.putText(orig_rgb, f"Original t={t:.2f}", (margin, y_pos), font, font_scale, (255, 255, 255), thickness)

        cv2.putText(recon_rgb, f"Recon t={t:.2f}", (margin, y_pos), font, font_scale, (0, 0, 0), thickness+1)
        cv2.putText(recon_rgb, f"Recon t={t:.2f}", (margin, y_pos), font, font_scale, (255, 255, 255), thickness)

        cv2.putText(motion_vis, "Motion Field", (margin, y_pos), font, font_scale, (0, 0, 0), thickness+1)
        cv2.putText(motion_vis, "Motion Field", (margin, y_pos), font, font_scale, (255, 255, 255), thickness)

        cv2.putText(interp_rgb, "Linear Interp", (margin, y_pos), font, font_scale, (0, 0, 0), thickness+1)
        cv2.putText(interp_rgb, "Linear Interp", (margin, y_pos), font, font_scale, (255, 255, 255), thickness)

        # Create 2x2 grid layout
        top_row = np.hstack([orig_rgb, recon_rgb])
        bottom_row = np.hstack([motion_vis, interp_rgb])
        combined = np.vstack([top_row, bottom_row])

        # Save frame
        cv2.imwrite(f"{temp_dir}/frame_{i:04d}.png", combined)

    # Create video with ffmpeg
    video_path = f'{output_dir}/neural_field_motion.mp4'
    fps = 30

    print("Creating video from frames...")
    import subprocess
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-framerate", str(fps), 
        "-pattern_type", "glob", "-i", f"{temp_dir}/frame_*.png",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "slow", "-crf", "22",
        video_path
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Video saved to {video_path}")
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"Error creating video with ffmpeg: {e}")
        print("Falling back to OpenCV video writer...")

        # Fallback to OpenCV
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, fps, (res*2, res*2))

        for i in range(num_frames):
            frame_path = f"{temp_dir}/frame_{i:04d}.png"
            frame = cv2.imread(frame_path)
            video.write(frame)

        video.release()
        print(f"Video saved to {video_path}")

    # Clean up temporary files
    print("Cleaning up temporary files...")
    import shutil
    shutil.rmtree(temp_dir)

    # Also remove any other PNG files in output directory
    for file in os.listdir(output_dir):
        if file.endswith('.png'):
            os.remove(os.path.join(output_dir, file))

    return video_path

def main():
    """Main function"""
    try:
        # Train the model
        model, original_image, motion_frames, res = train_model(num_training_steps=3000)

        # Save fixed scene to TIFF
        fixed_scene = model.get_fixed_scene()
        tiff.imwrite(f'{output_dir}/fixed_scene.tif', fixed_scene)

        # Create quad-panel video with motion field visualization and fixed scene
        video_path = create_quad_panel_video(model, original_image, motion_frames, res, num_frames=90)
        print("All processing completed successfully!")
        print(f"Video saved to: {video_path}")
        print(f"Fixed scene saved to: {output_dir}/fixed_scene.tif")
        print(f"Fixed scene progress video saved to: {output_dir}/fixed_scene_progress.mp4")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()