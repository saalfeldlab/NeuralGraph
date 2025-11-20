import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import rc
from NeuralGraph.utils import to_numpy
from tifffile import imread, imwrite
from tqdm import trange, tqdm
import os
from scipy.ndimage import map_coordinates
<<<<<<< HEAD
=======
try:
    import tinycudann as tcnn
except ImportError:
    tcnn = None
>>>>>>> 4be5279 (calcium generation)
from skimage.metrics import structural_similarity as ssim
import cv2
import subprocess
import shutil
from NeuralGraph.models.Siren_Network import Siren
from PIL import Image as PILImage

# from fa2_modified import ForceAtlas2
# import h5py as h5
# import zarr
# import xarray as xr

try:
    import tinycudann as tcnn
except ImportError:
    tcnn = None
    print("Warning: tinycudann not installed. Falling back to slow mode.")

def compute_gt_deformation_field(frame_idx, num_frames, res, motion_intensity=0.015):
    """Compute ground truth deformation field for visualization

    Returns:
        dx, dy: Deformation fields of shape (res, res) in normalized coordinates [0, 1]
    """
    # Create coordinate grids in normalized space [0, 1]
    y_grid, x_grid = np.meshgrid(
        np.linspace(0, 1, res),
        np.linspace(0, 1, res),
        indexing='ij'
    )

    # Create displacement fields with time-varying frequency
    t_norm = frame_idx / num_frames
    freq_x = 2 + t_norm * 2
    freq_y = 3 + t_norm * 1

    # Compute normalized deformation (same as in apply_sinusoidal_warp)
    dx = motion_intensity * np.sin(freq_x * np.pi * y_grid) * np.cos(freq_y * np.pi * x_grid)
    dy = motion_intensity * np.cos(freq_x * np.pi * y_grid) * np.sin(freq_y * np.pi * x_grid)

    return dx, dy


def apply_sinusoidal_warp(image, frame_idx, num_frames, motion_intensity=0.015):
    """Apply sinusoidal warping to an image, similar to pixel_NSTM.py"""
    h, w = image.shape

    # Create coordinate grids
    y_grid, x_grid = np.meshgrid(
        np.linspace(0, 1, h),
        np.linspace(0, 1, w),
        indexing='ij'
    )

    # Create displacement fields with time-varying frequency
    t_norm = frame_idx / num_frames
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

    return warped


def create_network_config():
    """Create network configuration for NSTM"""
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
            "n_hidden_layers": 2
        }
    }


class NeuralSpaceTimeModel:
    """Wrapper class for motion and scene networks"""
    def __init__(self, motion_net, scene_net, res, device):
        self.motion_net = motion_net
        self.scene_net = scene_net
        self.res = res
        self.device = device

        # Create coordinate grid (use float16 for tinycudann)
        y = torch.linspace(0, 1, res, device=device, dtype=torch.float16)
        x = torch.linspace(0, 1, res, device=device, dtype=torch.float16)
        yv, xv = torch.meshgrid(y, x, indexing='ij')
        self.coords_2d = torch.stack([xv.flatten(), yv.flatten()], dim=1)

    def get_fixed_scene(self):
        """Extract the fixed scene without motion"""
        with torch.no_grad():
            values = self.scene_net(self.coords_2d)
            fixed_scene = values.reshape(self.res, self.res).cpu().numpy()
        return fixed_scene


def pretrain_siren_image(activity_dir, device, output_dir, num_training_steps=5000,
                          nnr_f_xy_period=1.0, nnr_f_T_period=10):
    """Pre-train SIREN network on grayscale image from activity folder

    Args:
        activity_dir: Path to activity folder containing frame images
        device: torch device
        output_dir: Directory to save outputs
        num_training_steps: Number of training steps
        nnr_f_xy_period: Period for spatial coordinates (x,y will be scaled by 2π/period)
        nnr_f_T_period: Period for temporal coordinate (t will be scaled by 2π/period)

    Returns:
        siren_net: Pre-trained SIREN network mapping (x,y,t) -> grayscale
                   Coordinates are scaled by their respective periods
    """
    import glob

    print("pre-training SIREN network on first 10 activity frames...")

    # Find frames in activity folder (sorted by filename)
    frame_files = sorted(glob.glob(f"{activity_dir}/frame_*.tif"))
    if len(frame_files) == 0:
        raise FileNotFoundError(f"No frames found in {activity_dir}")

    # Load first 10 frames
    n_train_frames = 10
    frame_files = frame_files[:n_train_frames]
    print(f"loading {len(frame_files)} frames...")

    # Load all frames
    from tifffile import imread
    images = []
    for frame_path in frame_files:
        img_array = imread(frame_path)
        if img_array is None:
            raise ValueError(f"Failed to load image from {frame_path}")

        # Ensure float32 and normalize to [0, 1] if needed
        img_array = img_array.astype(np.float32)
        if img_array.max() > 1.0:
            img_array = img_array / 255.0
        images.append(img_array)

    height, width = images[0].shape
    print(f"image shape: ({height}, {width}), n_frames: {len(images)}")
    print(f"value range: [{min(img.min() for img in images):.4f}, {max(img.max() for img in images):.4f}]")
    print(f"coordinate periods: xy_period={nnr_f_xy_period}, T_period={nnr_f_T_period}")

    # Create coordinate grid with periodic scaling
    # Coordinates are scaled by 2π/period to match the periodic nature of SIREN
    all_coords = []
    all_targets = []

    for t_idx, img in enumerate(images):
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        # Normalize to [0, 1] then scale by 2π/period
        x_coords = x_coords.astype(np.float32) / (width - 1) * (2 * np.pi / nnr_f_xy_period)
        y_coords = y_coords.astype(np.float32) / (height - 1) * (2 * np.pi / nnr_f_xy_period)

        # Time normalized to [0, 1] then scaled by 2π/period
        t_value = t_idx / (len(images) - 1) if len(images) > 1 else 0.0
        t_value = t_value * (2 * np.pi / nnr_f_T_period)
        t_coords = np.full_like(x_coords, t_value, dtype=np.float32)

        frame_coords = np.stack([x_coords.ravel(), y_coords.ravel(), t_coords.ravel()], axis=1)
        all_coords.append(frame_coords)
        all_targets.append(img.ravel())

    # Concatenate all frames
    coords = np.concatenate(all_coords, axis=0)
    target_pixels = np.concatenate(all_targets, axis=0)

    coords = torch.from_numpy(coords).to(device)
    target_pixels = torch.from_numpy(target_pixels).unsqueeze(1).to(device)

    print(f"coordinate grid shape: {coords.shape} (x, y, t)")
    print(f"target pixels shape: {target_pixels.shape}")
    print(f"spatial range: [0, {2*np.pi/nnr_f_xy_period:.4f}], temporal range: [0, {2*np.pi/nnr_f_T_period:.4f}]")
    print(f"training on {len(images)} frames")

    # Create SIREN network (x, y, t) -> grayscale
    siren_net = Siren(
        in_features=3,  # x, y, t
        out_features=1,  # Grayscale output
        hidden_features=256,
        hidden_layers=4,
        outermost_linear=True,
        first_omega_0=60,
        hidden_omega_0=30
    ).to(device)

    num_params = sum(p.numel() for p in siren_net.parameters())
    print(f"model parameters: {num_params:,}")

    optimizer = torch.optim.Adam(siren_net.parameters(), lr=3e-4)

    # Training parameters
    batch_size = 50000  # Process in batches for memory efficiency
    n_pixels = coords.shape[0]
    n_batches = (n_pixels + batch_size - 1) // batch_size

    print(f"training SIREN for {num_training_steps} steps...")
    print(f"batch_size: {batch_size} pixels, n_batches: {n_batches}")

    os.makedirs(output_dir, exist_ok=True)
    loss_history = []

    pbar = trange(num_training_steps, ncols=100)
    for step in pbar:
        optimizer.zero_grad()
        total_loss = 0.0

        # Process in batches
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_pixels)

            batch_coords = coords[start_idx:end_idx]
            batch_targets = target_activities[start_idx:end_idx]

            # Forward pass
            batch_predicted = siren_net(batch_coords)

            # Compute loss for this batch
            batch_loss = torch.nn.functional.mse_loss(batch_predicted, batch_targets)

            # Scale loss by batch proportion and accumulate
            batch_weight = (end_idx - start_idx) / n_pixels
            scaled_loss = batch_loss * batch_weight
            scaled_loss.backward()

            total_loss += batch_loss.item() * batch_weight

        # Step optimizer after all batches
        optimizer.step()

        loss_history.append(total_loss)

        if step % 100 == 0:
            pbar.set_postfix({'loss': f'{total_loss:.6f}'})

    print("SIREN pre-training complete!")

    # Clear GPU cache before reconstruction
    torch.cuda.empty_cache()

    # Reconstruct all frames
    print("reconstructing frames...")
    siren_net.eval()

    reconstructed_frames = []
    psnr_values = []

    with torch.no_grad():
        # Create spatial grid (same for all frames) with periodic scaling
        y_grid, x_grid = np.mgrid[0:height, 0:width]
        x_grid = x_grid.astype(np.float32) / (width - 1) * (2 * np.pi / nnr_f_xy_period)
        y_grid = y_grid.astype(np.float32) / (height - 1) * (2 * np.pi / nnr_f_xy_period)

        for t_idx in range(len(images)):
            # Create time value for this frame with periodic scaling
            t_value = t_idx / (len(images) - 1) if len(images) > 1 else 0.0
            t_value = t_value * (2 * np.pi / nnr_f_T_period)
            t_grid = np.full_like(x_grid, t_value, dtype=np.float32)

            # Create coordinates for this frame
            frame_coords = np.stack([x_grid.ravel(), y_grid.ravel(), t_grid.ravel()], axis=1)
            frame_coords = torch.from_numpy(frame_coords).to(device)

            # Reconstruct in batches
            predicted_pixels = []
            n_pixels_per_frame = height * width
            n_batches_frame = (n_pixels_per_frame + batch_size - 1) // batch_size

            for batch_idx in range(n_batches_frame):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_pixels_per_frame)
                batch_coords = frame_coords[start_idx:end_idx]
                predicted_pixels.append(siren_net(batch_coords))

            predicted_frame = torch.cat(predicted_pixels, dim=0).reshape(height, width).cpu().numpy()
            predicted_frame = np.clip(predicted_frame, 0, 1)
            reconstructed_frames.append(predicted_frame)

            # Calculate PSNR for this frame
            mse = np.mean((images[t_idx] - predicted_frame) ** 2)
            psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
            psnr_values.append(psnr)

    mean_psnr = np.mean(psnr_values)

    # Save first frame as PNG
    reconstructed_uint8 = (reconstructed_frames[0] * 255).astype(np.uint8)
    reconstructed_pil = PILImage.fromarray(reconstructed_uint8, mode='L')
    reconstructed_pil.save(f'{output_dir}/siren_reconstructed_frame0.png')

    # Create video from reconstructed frames
    print("creating mp4 video...")
    video_temp_dir = f"{output_dir}/video_frames"
    os.makedirs(video_temp_dir, exist_ok=True)

    # Save all frames as PNG for video
    for i, frame in enumerate(reconstructed_frames):
        frame_uint8 = (frame * 255).astype(np.uint8)
        frame_pil = PILImage.fromarray(frame_uint8, mode='L')
        frame_pil.save(f"{video_temp_dir}/frame_{i:04d}.png")

    # Create mp4 using ffmpeg
    video_path = f"{output_dir}/siren_reconstructed.mp4"
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-framerate", "2",  # 2 fps for 10 frames
        "-i", f"{video_temp_dir}/frame_%04d.png",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-vf", "scale=512:512",
        video_path
    ]
    subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # Clean up temp frames
    shutil.rmtree(video_temp_dir)

    # Plot loss history
    plt.figure(figsize=(10, 4))
    plt.plot(loss_history)
    plt.title('SIREN pre-training loss', fontsize=12)
    plt.xlabel('step')
    plt.ylabel('MSE loss')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/pretrain_loss.png', dpi=150)
    plt.close()

    print(f"final loss: {loss_history[-1]:.6f}")
    print(f"mean PSNR: {mean_psnr:.2f} dB")
    print(f"PSNR range: [{min(psnr_values):.2f}, {max(psnr_values):.2f}] dB")




    siren_net.train()
    return siren_net


def pretrain_siren_discrete(x_list, device, output_dir, num_training_steps=5000,
                          nnr_f_xy_period=1.0, nnr_f_T_period=10, n_train_frames=1,
                          n_augmentations=10):
    """Pre-train SIREN network on discrete neuron data with random spatial offsets

    Args:
        x_list: List of neuron data arrays (n_frames, n_neurons, features)
                x_list[frame][neuron, 1:3] = positions (x, y)
                x_list[frame][neuron, 6] = activity value
        device: torch device
        output_dir: Directory to save outputs
        num_training_steps: Number of training steps
        nnr_f_xy_period: Period for spatial coordinates (x,y will be scaled by 2π/period)
        nnr_f_T_period: Period for temporal coordinate (t will be scaled by 2π/period)
        n_train_frames: Number of frames to train on (default=1, first frame only)
        n_augmentations: Number of random spatial offsets to sample per frame (default=10)

    Returns:
        siren_net: Pre-trained SIREN network mapping (x,y,t) -> activity
                   Coordinates are scaled by their respective periods
    """
    print(f"pre-training SIREN network on discrete neurons from first {n_train_frames} frame(s)...")
    print(f"using {n_augmentations} random spatial offsets per frame")

    # Offset parameters - use very small std to cluster augmentations tightly around true positions
    # This allows SIREN to memorize the exact neuron positions
    offset_std = 0.005  # Standard deviation for Gaussian sampling (very small for tight clustering)
    print(f"Gaussian offset std: {offset_std:.6f} (in original coordinates)")

    # Extract data from first n_train_frames
    all_coords = []
    all_targets = []

    for t_idx in range(n_train_frames):
        # Get frame data
        frame_data = x_list[0][t_idx]  # x_list[run][frame]

        # Extract positions (columns 1:3) and activity (column 6)
        positions = frame_data[:, 1:3]  # (n_neurons, 2) - x, y coordinates
        activities = frame_data[:, 6]    # (n_neurons,) - activity values

        n_neurons = positions.shape[0]

        # Print statistics for first frame
        if t_idx == 0:
            print(f"n_neurons: {n_neurons}")
            print(f"position range: x=[{positions[:, 0].min():.4f}, {positions[:, 0].max():.4f}], y=[{positions[:, 1].min():.4f}, {positions[:, 1].max():.4f}]")
            print(f"activity range: [{activities.min():.4f}, {activities.max():.4f}]")

        # Generate multiple augmented versions with random offsets
        for aug_idx in range(n_augmentations):
            # Sample random offset from Gaussian distribution
            offset = np.random.randn(2).astype(np.float32) * offset_std  # (2,) - dx, dy

            # Apply offset to all neurons
            positions_offset = positions + offset

            # Normalize positions: divide by xy_period then scale by 2π
            # This matches the normalization in train_nstm: pos_frame / xy_period
            positions_normalized = positions_offset.astype(np.float32) * (2 * np.pi / nnr_f_xy_period)

            # Time value for this frame
            t_value = t_idx / (n_train_frames - 1) if n_train_frames > 1 else 0.0
            t_value = t_value * (2 * np.pi / nnr_f_T_period)

            # Create (x, y, t) coordinates for all neurons in this frame
            t_coords = np.full((n_neurons, 1), t_value, dtype=np.float32)
            frame_coords = np.concatenate([positions_normalized, t_coords], axis=1)

            all_coords.append(frame_coords)
            all_targets.append(activities)

            if t_idx == 0 and aug_idx == 0:
                print(f"normalized position range (first augmentation): x=[{positions_normalized[:, 0].min():.4f}, {positions_normalized[:, 0].max():.4f}], y=[{positions_normalized[:, 1].min():.4f}, {positions_normalized[:, 1].max():.4f}]")
                print(f"time value: {t_value:.4f}")
                print(f"first offset: dx={offset[0]:.4f}, dy={offset[1]:.4f}")

    # Concatenate all frames
    coords = np.concatenate(all_coords, axis=0)
    target_activities = np.concatenate(all_targets, axis=0)

    coords = torch.from_numpy(coords).to(device)
    target_activities = torch.from_numpy(target_activities).unsqueeze(1).to(device)

    print("\nTraining data:")
    print(f"  coordinate grid shape: {coords.shape} (x, y, t)")
    print(f"  target activities shape: {target_activities.shape}")
    print(f"  coordinate periods: xy_period={nnr_f_xy_period}, T_period={nnr_f_T_period}")
    print(f"  spatial range: [0, {2*np.pi/nnr_f_xy_period:.4f}]")
    print(f"  temporal range: [0, {2*np.pi/nnr_f_T_period:.4f}]")
    print(f"  training on {n_train_frames} frame(s) × {n_augmentations} augmentations × {n_neurons} neurons = {coords.shape[0]} total data points")

    # Create SIREN network (x, y, t) -> activity
    siren_net = Siren(
        in_features=3,  # x, y, t
        out_features=1,  # Activity output
        hidden_features=256,
        hidden_layers=4,
        outermost_linear=True,
        first_omega_0=60,
        hidden_omega_0=30
    ).to(device)

    num_params = sum(p.numel() for p in siren_net.parameters())
    print(f"model parameters: {num_params:,}")

    optimizer = torch.optim.Adam(siren_net.parameters(), lr=1e-5)

    # Training parameters
    batch_size = 50000  # Process in batches for memory efficiency
    n_pixels = coords.shape[0]
    n_batches = (n_pixels + batch_size - 1) // batch_size

    print(f"training SIREN for {num_training_steps} steps...")
    print(f"batch_size: {batch_size} pixels, n_batches: {n_batches}")

    os.makedirs(output_dir, exist_ok=True)
    loss_history = []

    pbar = trange(num_training_steps, ncols=100)
    for step in pbar:
        optimizer.zero_grad()
        total_loss = 0.0

        # Process in batches
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_pixels)

            batch_coords = coords[start_idx:end_idx]
            batch_targets = target_activities[start_idx:end_idx]

            # Forward pass
            batch_predicted = siren_net(batch_coords)

            # Compute loss for this batch
            batch_loss = torch.nn.functional.mse_loss(batch_predicted, batch_targets)

            # Scale loss by batch proportion and accumulate
            batch_weight = (end_idx - start_idx) / n_pixels
            scaled_loss = batch_loss * batch_weight
            scaled_loss.backward()

            total_loss += batch_loss.item() * batch_weight

        # Step optimizer after all batches
        optimizer.step()

        loss_history.append(total_loss)

        if step % 100 == 0:
            pbar.set_postfix({'loss': f'{total_loss:.6f}'})

    print("SIREN pre-training complete!")

    # Clear GPU cache before reconstruction
    torch.cuda.empty_cache()

    # Evaluate reconstruction on training data
    print("evaluating reconstruction...")
    siren_net.eval()


    # Plot loss history
    plt.figure(figsize=(10, 4))
    plt.plot(loss_history)
    plt.title('SIREN discrete pre-training loss', fontsize=12)
    plt.xlabel('step')
    plt.ylabel('MSE loss')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/pretrain_discrete_loss.png', dpi=150)
    plt.close()

    print(f"final loss: {loss_history[-1]:.6f}")

    # Test 1: Compare ground truth vs SIREN predictions for all neurons (on training frame only)
    n_test_neurons = n_neurons

    # Test all neurons
    test_neuron_indices = np.arange(n_neurons)

    # Extract ground truth and predictions for test neurons on frame 0 only
    gt_values = []
    pred_values = []

    frame_data = x_list[0][0]  # Frame 0 only

    for neuron_idx in test_neuron_indices:
        # Ground truth position and activity
        pos = frame_data[neuron_idx, 1:3].astype(np.float32)
        activity_gt = frame_data[neuron_idx, 6]

        # Normalize position and add time (t=0)
        pos_normalized = pos * (2 * np.pi / nnr_f_xy_period)
        t_normalized = 0.0  # Frame 0

        # Query SIREN
        coords_query = torch.tensor([[pos_normalized[0], pos_normalized[1], t_normalized]],
                                   dtype=torch.float32, device=device)
        with torch.no_grad():
            activity_pred = siren_net(coords_query).cpu().item()

        gt_values.append(activity_gt)
        pred_values.append(activity_pred)

    # Calculate overall statistics and plot comparison
    gt_array = np.array(gt_values)
    pred_array = np.array(pred_values)

    # Calculate R² only
    r2_overall = 1 - (np.sum((gt_array - pred_array) ** 2) / np.sum((gt_array - gt_array.mean()) ** 2))

    # Plot scatter comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.scatter(gt_array, pred_array, c='white', s=100, alpha=0.8, edgecolors='green', linewidths=2, label='SIREN vs GT')

    # Add diagonal reference line
    min_val = min(gt_array.min(), pred_array.min())
    max_val = max(gt_array.max(), pred_array.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'g--', alpha=0.5, linewidth=2, label='Perfect Fit')

    # Add R² text
    ax.text(0.02, 0.98, f'R²={r2_overall:.4f}', transform=ax.transAxes,
           fontsize=14, verticalalignment='top', horizontalalignment='left',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
           color='white' if r2_overall > 0.9 else ('orange' if r2_overall > 0.7 else 'red'))

    ax.set_xlabel('Ground Truth Activity', fontsize=12)
    ax.set_ylabel('SIREN Predicted Activity', fontsize=12)
    ax.set_title('SIREN Prediction vs Ground Truth at Neuron Positions (Frame 0)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/siren_neuron_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"R² = {r2_overall:.4f} ({n_test_neurons} neurons)")

    # Test 2: Generate 512×512 image for frame 0 only

    # Create coordinate grid for full image
    y_grid, x_grid = np.mgrid[0:512, 0:512]
    x_grid_norm = (x_grid.astype(np.float32) / 511.0 - 0.5)  # Normalize to [-0.5, 0.5]
    y_grid_norm = (y_grid.astype(np.float32) / 511.0 - 0.5)
    x_grid_norm = x_grid_norm * (2 * np.pi / nnr_f_xy_period)
    y_grid_norm = y_grid_norm * (2 * np.pi / nnr_f_xy_period)

    # Generate single frame (t=0)
    batch_size = 50000
    t_normalized = 0.0  # Frame 0
    t_grid = np.full_like(x_grid_norm, t_normalized, dtype=np.float32)

    # Create coordinates (512*512, 3)
    coords_frame = np.stack([x_grid_norm.ravel(), y_grid_norm.ravel(), t_grid.ravel()], axis=1)
    coords_tensor = torch.from_numpy(coords_frame).to(device)

    # Query SIREN in batches
    predicted_pixels = []
    n_pixels = coords_tensor.shape[0]
    n_batches = (n_pixels + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_pixels)
            batch_coords = coords_tensor[start_idx:end_idx]
            predicted_pixels.append(siren_net(batch_coords))

    # Reshape to image
    frame = torch.cat(predicted_pixels, dim=0).reshape(512, 512).cpu().numpy()

    # Save image
    vmin, vmax = frame.min(), frame.max()
    frame_norm = (frame - vmin) / (vmax - vmin + 1e-8)
    frame_uint8 = (frame_norm * 255).astype(np.uint8)
    frame_pil = PILImage.fromarray(frame_uint8, mode='L')
    image_path = f"{output_dir}/siren_generated_image.png"
    frame_pil.save(image_path)

    siren_net.train()
    return siren_net


def train_siren(x_list, device, output_dir, num_training_steps=10000,
                                  nnr_f_T_period=10, n_train_frames=10, n_neurons=100):
    """Pre-train SIREN network on discrete neuron time series (t -> [activity_1, ..., activity_n])

    Args:
        x_list: List of neuron data arrays (n_frames, n_neurons, features)
                x_list[frame][neuron, 6] = activity value
        device: torch device
        output_dir: Directory to save outputs
        num_training_steps: Number of training steps
        nnr_f_T_period: Period for temporal coordinate (t will be scaled by 2π/period)
        n_train_frames: Number of frames to train on
        n_neurons: Number of neurons (output dimension)

    Returns:
        siren_net: Pre-trained SIREN network mapping t -> [activity_1, ..., activity_n]
    """
    print(f"pre-training SIREN time network on {n_neurons} neurons across {n_train_frames} frames...")

    # Extract activity data from all training frames
    all_t_coords = []
    all_activities = []

    for t_idx in range(n_train_frames):
        frame_data = x_list[0][t_idx]
        activities = frame_data[:, 6]  # (n_neurons,) - activity values for all neurons

        # Normalize time coordinate
        t_value = t_idx / (n_train_frames - 1) if n_train_frames > 1 else 0.0
        t_normalized = t_value * (2 * np.pi / nnr_f_T_period)

        all_t_coords.append(t_normalized)
        all_activities.append(activities)

    # Print statistics
    all_activities_array = np.array(all_activities)
    print(f"n_neurons: {n_neurons}")
    print(f"n_frames: {n_train_frames}")
    print(f"time range: [0, {all_t_coords[-1]:.4f}] (normalized)")
    print(f"activity range: [{all_activities_array.min():.4f}, {all_activities_array.max():.4f}]")

    # Convert to tensors
    t_coords_tensor = torch.tensor(all_t_coords, dtype=torch.float32, device=device).reshape(-1, 1)  # (n_frames, 1)
    activities_tensor = torch.tensor(all_activities_array, dtype=torch.float32, device=device)  # (n_frames, n_neurons)

    # Create SIREN network: input = t (1D), output = n_neurons activities
    siren_net = Siren(
        in_features=1,  # Just time
        out_features=n_neurons,  # One output per neuron
        hidden_features=256,
        hidden_layers=4,
        outermost_linear=True,
        first_omega_0=60,
        hidden_omega_0=30
    ).to(device)

    optimizer = torch.optim.Adam(siren_net.parameters(), lr=1e-4)

    # Training loop
    loss_history = []
    siren_net.train()

    pbar = tqdm(range(num_training_steps), desc="training SIREN time network", ncols=100)
    for step in pbar:
        optimizer.zero_grad()

        # Forward pass: input is time, output is all neuron activities
        predicted_activities = siren_net(t_coords_tensor)  # (n_frames, n_neurons)

        # Loss: MSE between predicted and true activities
        loss = torch.nn.functional.mse_loss(predicted_activities, activities_tensor)

        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if step % 1000 == 0:
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    # Evaluation
    print("evaluating reconstruction...")
    siren_net.eval()

    with torch.no_grad():
        # Plot loss history
        plt.figure(figsize=(10, 4))
        plt.plot(loss_history)
        plt.title('SIREN time pre-training loss', fontsize=12)
        plt.xlabel('step')
        plt.ylabel('MSE loss')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/pretrain_time_loss.png', dpi=150)
        plt.close()

        print(f"final loss: {loss_history[-1]:.6f}")

        # Test: scatter plot of all predictions vs ground truth
        predicted_all = siren_net(t_coords_tensor).cpu().numpy()  # (n_frames, n_neurons)
        ground_truth_all = activities_tensor.cpu().numpy()  # (n_frames, n_neurons)

        # Flatten to get all (frame, neuron) pairs
        pred_flat = predicted_all.flatten()
        gt_flat = ground_truth_all.flatten()

        # Calculate R²
        r2 = 1 - (np.sum((gt_flat - pred_flat) ** 2) / np.sum((gt_flat - gt_flat.mean()) ** 2))

        # Plot scatter comparison
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        ax.scatter(gt_flat, pred_flat, c='white', s=50, alpha=0.6, edgecolors='green', linewidths=1, label='SIREN vs GT')

        # Add diagonal reference line
        min_val = min(gt_flat.min(), pred_flat.min())
        max_val = max(gt_flat.max(), pred_flat.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'g--', alpha=0.5, linewidth=2, label='Perfect Fit')

        # Add R² text
        ax.text(0.02, 0.98, f'R²={r2:.4f}', transform=ax.transAxes,
               fontsize=14, verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
               color='white' if r2 > 0.9 else ('orange' if r2 > 0.7 else 'red'))

        ax.set_xlabel('Ground Truth Activity', fontsize=12)
        ax.set_ylabel('SIREN Predicted Activity', fontsize=12)
        ax.set_title(f'SIREN Time Model: {n_neurons} neurons × {n_train_frames} frames', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/siren_time_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"R² = {r2:.4f} ({n_neurons * n_train_frames} data points)")

    siren_net.train()
    return siren_net


def get_activity_at_coords(coords_2d, t_normalized, pretrained_activity_net, neuron_positions,
                           nnr_f_T_period, device, dot_size=32, image_size=512,
                           affine_scale=None, affine_bias=None):
    """Get activity values at arbitrary 2D coordinates using Gaussian splatting

    Uses the same Gaussian splatting method as render_activity_image to ensure consistency
    between ground truth rendering and NSTM training. Applies affine transform to map SIREN
    output range to activity image range.

    Args:
        coords_2d: (N, 2) tensor of 2D coordinates in [0, 1]
        t_normalized: scalar, normalized time value in [0, 1]
        pretrained_activity_net: SIREN network mapping t -> [activity_1, ..., activity_n]
        neuron_positions: (n_neurons, 2) tensor of neuron positions in [-0.5, 0.5]
        nnr_f_T_period: Temporal period for SIREN
        device: torch device
        dot_size: Diameter of Gaussian dots in pixels (default 32)
        image_size: Image resolution for pixel conversion (default 512)
        affine_scale: Learnable scale parameter for affine transform (default None)
        affine_bias: Learnable bias parameter for affine transform (default None)

    Returns:
        (N,) tensor of activity values at the queried coordinates
    """
    # Step 1: Query pretrained SIREN to get activities for all neurons at this time
    t_siren = t_normalized * (2 * np.pi / nnr_f_T_period)
    t_tensor = torch.tensor([[t_siren]], dtype=torch.float32, device=device)

    with torch.no_grad():
        neuron_activities = pretrained_activity_net(t_tensor)[0]  # (n_neurons,)

    # Apply affine transform to SIREN outputs: y = scale * x + bias
    # This maps SIREN range (e.g., [-20, 20]) to activity image range (e.g., [0, 120])
    if affine_scale is not None and affine_bias is not None:
        neuron_activities = affine_scale * neuron_activities + affine_bias

    # Step 2: Convert coordinates to pixel space for Gaussian splatting
    # coords_2d in [0, 1] -> pixel coords in [0, image_size-1]
    coords_pixels = coords_2d * (image_size - 1)  # (N, 2)

    # neuron_positions in [-0.5, 0.5] -> pixel coords in [0, image_size-1]
    neuron_positions_pixels = (neuron_positions + 0.5) * (image_size - 1)  # (n_neurons, 2)

    # Step 3: Gaussian splatting - compute Gaussian weights from each neuron to each query point
    # Gaussian sigma from dot size (diameter = 2.355 * sigma for Gaussian)
    sigma = dot_size / 2.355

    # Compute squared distances: (N, 1, 2) - (1, n_neurons, 2) = (N, n_neurons, 2)
    diffs = coords_pixels.unsqueeze(1) - neuron_positions_pixels.unsqueeze(0)
    dist_sq = torch.sum(diffs ** 2, dim=2)  # (N, n_neurons)

    # Gaussian kernel: exp(-dist^2 / (2*sigma^2))
    gaussian_weights = torch.exp(-dist_sq / (2 * sigma ** 2))  # (N, n_neurons)

    # Step 4: Sum weighted activities (Gaussian splatting)
    # Each query point receives contribution from all neurons weighted by Gaussian
    splatted_activities = torch.sum(gaussian_weights * neuron_activities.unsqueeze(0), dim=1)  # (N,)

    return splatted_activities


def render_siren_activity_video(pretrained_activity_net, neuron_positions, affine_scale, affine_bias,
                                  nnr_f_T_period, n_frames, res, device, output_dir, fps=30):
    """Render a video of SIREN activity with learned affine transform using Gaussian splatting

    Args:
        pretrained_activity_net: SIREN network mapping t -> [activity_1, ..., activity_n]
        neuron_positions: (n_neurons, 2) tensor of neuron positions in [-0.5, 0.5]
        affine_scale: Learned scale parameter
        affine_bias: Learned bias parameter
        nnr_f_T_period: Temporal period for SIREN
        n_frames: Number of frames to render
        res: Image resolution (e.g., 512)
        device: torch device
        output_dir: Directory to save video
        fps: Frames per second for video (default 30)
    """
    import cv2
    import os

    # Create temporary directory for frames
    temp_dir = f"{output_dir}/siren_activity_frames"
    os.makedirs(temp_dir, exist_ok=True)

    # Create full 2D coordinate grid
    y_coords = torch.linspace(0, 1, res, device=device, dtype=torch.float32)
    x_coords = torch.linspace(0, 1, res, device=device, dtype=torch.float32)
    yv, xv = torch.meshgrid(y_coords, x_coords, indexing='ij')
    coords_2d = torch.stack([xv.flatten(), yv.flatten()], dim=1)  # (res*res, 2)

    print(f"rendering {n_frames} frames...")
    for frame_idx in tqdm(range(n_frames), ncols=100):
        t_normalized = frame_idx / (n_frames - 1)

        # Get activity at all pixel coordinates using learned affine transform
        with torch.no_grad():
            activity_img = get_activity_at_coords(
                coords_2d,
                t_normalized,
                pretrained_activity_net,
                neuron_positions,
                nnr_f_T_period,
                device,
                dot_size=32,
                image_size=res,
                affine_scale=affine_scale,
                affine_bias=affine_bias
            ).reshape(res, res).cpu().numpy()

        # Normalize to [0, 1] for visualization
        activity_min = activity_img.min()
        activity_max = activity_img.max()
        if activity_max > activity_min:
            activity_norm = (activity_img - activity_min) / (activity_max - activity_min)
        else:
            activity_norm = np.zeros_like(activity_img)

        # Convert to 8-bit RGB
        activity_uint8 = (np.clip(activity_norm, 0, 1) * 255).astype(np.uint8)
        activity_rgb = np.stack([activity_uint8, activity_uint8, activity_uint8], axis=2)

        # Add text overlay with frame info
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Frame {frame_idx}/{n_frames-1} | t={t_normalized:.3f}"
        cv2.putText(activity_rgb, text, (10, 30), font, 0.7, (255, 255, 255), 2)

        # Add scale/bias info
        text2 = f"scale={affine_scale.item():.2f}, bias={affine_bias.item():.1f}"
        cv2.putText(activity_rgb, text2, (10, 60), font, 0.6, (255, 255, 255), 2)

        # Save frame
        cv2.imwrite(f"{temp_dir}/frame_{frame_idx:06d}.png", activity_rgb)

    # Create video using ffmpeg
    video_path = f"{output_dir}/siren_activity_affine.mp4"
    ffmpeg_cmd = (
        f"ffmpeg -y -framerate {fps} -i {temp_dir}/frame_%06d.png "
        f"-c:v libx264 -pix_fmt yuv420p -crf 18 {video_path}"
    )

    import subprocess
    subprocess.run(ffmpeg_cmd, shell=True, check=True, capture_output=True)

    # Clean up temporary frames
    import shutil
    shutil.rmtree(temp_dir)

    print(f"SIREN activity video saved to: {video_path}")


def train_nstm(motion_frames_dir, activity_dir, n_frames, res, device, output_dir, num_training_steps=3000,
               siren_config=None, pretrained_activity_net=None, x_list=None,
               use_siren=True, siren_lr=1e-4, nstm_lr=5e-4, siren_loss_weight=1.0):
    """Train Neural Space-Time Model with fixed_scene + activity decomposition

    Args:
        siren_config: Dict with keys: hidden_dim_nnr_f, n_layers_nnr_f, omega_f,
                      nnr_f_xy_period, nnr_f_T_period, outermost_linear_nnr_f
        x_list: List of neuron data arrays for SIREN supervision (n_frames, n_neurons, features)
        use_siren: Boolean to use SIREN network for activity (True) or grid_sample (False)
        siren_lr: Learning rate for SIREN network
        nstm_lr: Learning rate for deformation and fixed_scene networks
        siren_loss_weight: Weight for SIREN supervision loss on discrete neurons
    """
    # Load motion frames
    print("loading motion frames...")
    motion_images = []
    for i in range(n_frames):
        img = imread(f"{motion_frames_dir}/frame_{i:06d}.tif")
        motion_images.append(img)

    # Load activity images (only if not using pretrained SIREN)
    activity_images = []
    if pretrained_activity_net is None:
        print("loading original activity images...")
        for i in range(n_frames):
            img = imread(f"{activity_dir}/frame_{i:06d}.tif")
            activity_images.append(img)
        print(f"loaded {n_frames} activity images")
    else:
        print("using pretrained SIREN for activity (no activity images loaded)")

    # Compute normalization statistics from motion images
    all_pixels = np.concatenate([img.flatten() for img in motion_images])
    data_min = all_pixels.min()
    data_max = all_pixels.max()
    print(f"input data range: [{data_min:.2f}, {data_max:.2f}]")

    # Normalize motion images to [0, 1] and convert to tensors
    motion_tensors = [torch.tensor((img - data_min) / (data_max - data_min), dtype=torch.float32).to(device) for img in motion_images]

    # Normalize activity images to [0, 1] and convert to tensors (if loaded)
    if activity_images:
        activity_tensors = [torch.tensor((img - data_min) / (data_max - data_min), dtype=torch.float32).to(device) for img in activity_images]
        print(f"loaded {n_frames} frames into gpu memory")
    else:
        activity_tensors = None
        print(f"loaded {n_frames} motion frames into gpu memory")

    # Create networks
    print("creating networks...")
    config = create_network_config()

    # Deformation network: (x, y, t) -> (δx, δy)
    deformation_net = tcnn.NetworkWithInputEncoding(
        n_input_dims=3,  # x, y, t
        n_output_dims=2,  # δx, δy
        encoding_config=config["encoding"],
        network_config=config["network"]
    ).to(device)

    # Fixed scene network: (x, y) -> raw mask value
    fixed_scene_net = tcnn.NetworkWithInputEncoding(
        n_input_dims=2,  # x, y
        n_output_dims=1,  # mask value
        encoding_config=config["encoding"],
        network_config=config["network"]
    ).to(device)

    # Note: Activity is handled by pretrained_activity_net via get_activity_at_coords()
    # No separate activity network needed in train_nstm

    # Create coordinate grid (use float16 for tinycudann, float32 for coords)
    y_coords = torch.linspace(0, 1, res, device=device, dtype=torch.float32)
    x_coords = torch.linspace(0, 1, res, device=device, dtype=torch.float32)
    yv, xv = torch.meshgrid(y_coords, x_coords, indexing='ij')
    coords_2d = torch.stack([xv.flatten(), yv.flatten()], dim=1)

    # Convert to float16 for tinycudann networks
    coords_2d_f16 = coords_2d.to(torch.float16)

    # Create learnable affine transform parameters for SIREN activity mapping
    # Maps SIREN output range (e.g., [-20, 20]) to activity image range (e.g., [0, 120])
    # Initialize with reasonable defaults: scale=3.0, bias=60.0 for [-20,20] -> [0,120]
    affine_scale = torch.nn.Parameter(torch.tensor(3.0, device=device, dtype=torch.float32))
    affine_bias = torch.nn.Parameter(torch.tensor(60.0, device=device, dtype=torch.float32))

    # Create optimizer for NSTM networks only (deformation + fixed_scene + affine params)
    nstm_params = (list(deformation_net.parameters()) +
                   list(fixed_scene_net.parameters()) +
                   [affine_scale, affine_bias])
    optimizer_nstm = torch.optim.Adam(nstm_params, lr=nstm_lr)

    # Learning rate schedule for NSTM
    scheduler_nstm = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_nstm,
        milestones=[num_training_steps // 2, num_training_steps * 3 // 4],
        gamma=0.1
    )

    # Extract neuron positions for activity sampling (if using pretrained SIREN)
    neuron_positions = None
    if pretrained_activity_net is not None and x_list is not None:
        print("extracting neuron positions for activity sampling...")
        frame_data = x_list[0][0]
        neuron_positions = frame_data[:, 1:3].astype(np.float32)  # (n_neurons, 2) in [-0.5, 0.5]
        neuron_positions = torch.tensor(neuron_positions, dtype=torch.float32, device=device)
        print(f"using {neuron_positions.shape[0]} neuron positions")

    # Training loop
    print(f"training for {num_training_steps} steps...")
    loss_history = []
    regularization_history = {'deformation': []}
    batch_size = min(16384, res*res)

    pbar = trange(num_training_steps, ncols=150)
    for step in pbar:
        # Select random frame
        t_idx = np.random.randint(0, n_frames)
        t_normalized = t_idx / (n_frames - 1)

        # Select random batch of pixels
        indices = torch.randperm(res*res, device=device)[:batch_size]
        batch_coords = coords_2d[indices]

        # Target values from warped motion frames
        target = motion_tensors[t_idx].reshape(-1, 1)[indices]

        # Create 3D coordinates for deformation network (convert to float16 for tcnn)
        t_tensor = torch.full_like(batch_coords[:, 0:1], t_normalized)
        coords_3d = torch.cat([batch_coords, t_tensor], dim=1).to(torch.float16)

        # Forward pass - deformation network (backward warp)
        deformation = deformation_net(coords_3d).to(torch.float32)

        # Compute source coordinates (backward warp)
        source_coords = batch_coords - deformation  # Note: MINUS for backward warp
        source_coords = torch.clamp(source_coords, 0, 1)

        # Sample fixed scene mask at source coordinates (convert to float16 for tcnn)
        fixed_scene_mask = torch.sigmoid(fixed_scene_net(source_coords.to(torch.float16))).to(torch.float32)

        # Sample activity at source coordinates
        if pretrained_activity_net is not None and neuron_positions is not None:
            # Use pretrained temporal SIREN to get activity at 2D coordinates
            sampled_activity = get_activity_at_coords(
                source_coords,
                t_normalized,
                pretrained_activity_net,
                neuron_positions,
                siren_config['nnr_f_T_period'],
                device,
                affine_scale=affine_scale,
                affine_bias=affine_bias
            ).unsqueeze(1)  # (batch_size, 1)
        else:
            # Fallback: use grid_sample on ground-truth activity images
            source_coords_normalized = source_coords * 2 - 1
            source_coords_grid = source_coords_normalized.reshape(1, 1, batch_size, 2)
            activity_frame = activity_tensors[t_idx].reshape(1, 1, res, res)
            sampled_activity = torch.nn.functional.grid_sample(
                activity_frame,
                source_coords_grid,
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            ).reshape(-1, 1)

        # Reconstruction: fixed_scene × activity
        reconstructed = fixed_scene_mask * sampled_activity

        # Main reconstruction loss
        recon_loss = torch.nn.functional.mse_loss(reconstructed, target)

        # Regularization: Deformation smoothness
        deformation_smoothness = torch.mean(torch.abs(deformation))

        # Combined loss with regularization
        total_loss = recon_loss + 0.01 * deformation_smoothness

        # Optimize NSTM networks (deformation + fixed_scene only)
        optimizer_nstm.zero_grad()
        total_loss.backward()
        optimizer_nstm.step()

        # Update learning rate
        scheduler_nstm.step()

        # Record losses
        loss_history.append(recon_loss.item())
        regularization_history['deformation'].append(deformation_smoothness.item())

        # Update progress bar
        postfix_dict = {
            'recon': f'{recon_loss.item():.6f}',
            'def': f'{deformation_smoothness.item():.4f}'
        }
        # Add affine parameters if using pretrained SIREN
        if pretrained_activity_net is not None:
            postfix_dict['scale'] = f'{affine_scale.item():.2f}'
            postfix_dict['bias'] = f'{affine_bias.item():.1f}'
        pbar.set_postfix(postfix_dict)

    print("training complete!")

    # Print learned affine parameters if using pretrained SIREN
    if pretrained_activity_net is not None:
        print(f"learned affine transform: scale={affine_scale.item():.3f}, bias={affine_bias.item():.3f}")

    # Extract learned fixed_scene mask
    print("extracting learned fixed_scene mask...")
    with torch.no_grad():
        fixed_scene_mask = torch.sigmoid(fixed_scene_net(coords_2d))
        fixed_scene_mask_img = fixed_scene_mask.reshape(res, res).cpu().numpy()

    # Compute evaluation metrics (only if activity images were loaded)
    if activity_images:
        # Compute average of original activity images
        activity_average = np.mean(activity_images, axis=0)

        # Create ground-truth mask from activity average (threshold at mean)
        threshold = activity_average.mean()
        ground_truth_mask = (activity_average > threshold).astype(np.float32)
        n_gt_pixels = ground_truth_mask.sum()

        # Normalize fixed_scene mask for comparison
        fixed_scene_mask_norm = (fixed_scene_mask_img - fixed_scene_mask_img.min()) / (fixed_scene_mask_img.max() - fixed_scene_mask_img.min() + 1e-8)

        # Threshold fixed_scene to match ground truth coverage (top N pixels)
        flat_fixed_scene = fixed_scene_mask_norm.flatten()
        n_pixels_to_select = int(n_gt_pixels)
        sorted_indices = np.argsort(flat_fixed_scene)[::-1]  # Sort descending
        fixed_scene_binary_flat = np.zeros_like(flat_fixed_scene, dtype=np.float32)
        fixed_scene_binary_flat[sorted_indices[:n_pixels_to_select]] = 1.0
        fixed_scene_binary = fixed_scene_binary_flat.reshape(fixed_scene_mask_norm.shape)

        # Compute DICE score and IoU between learned fixed_scene and ground-truth mask
        intersection = np.sum(fixed_scene_binary * ground_truth_mask)
        union = np.sum(fixed_scene_binary) + np.sum(ground_truth_mask)
        dice_score = 2 * intersection / (union + 1e-8)
        iou_score = intersection / (union - intersection + 1e-8)

        # Compute median of motion frames (warped frames) as baseline
        motion_median = np.median(motion_images, axis=0)

        # Reconstruct fixed scene: fixed_scene × activity_average
        fixed_scene_denorm = fixed_scene_mask_img * activity_average

        # Compute RMSE between fixed scene and activity average
        rmse_activity = np.sqrt(np.mean((fixed_scene_denorm - activity_average) ** 2))

        # Compute SSIM between fixed scene and activity average
        # Normalize both to [0, 1] for SSIM computation
        fixed_scene_norm = (fixed_scene_denorm - fixed_scene_denorm.min()) / (fixed_scene_denorm.max() - fixed_scene_denorm.min() + 1e-8)
        activity_avg_norm = (activity_average - activity_average.min()) / (activity_average.max() - activity_average.min() + 1e-8)
        ssim_activity = ssim(activity_avg_norm, fixed_scene_norm, data_range=1.0)

        # Compute RMSE between motion median and activity average (baseline)
        rmse_baseline = np.sqrt(np.mean((motion_median - activity_average) ** 2))

        # Compute SSIM between motion median and activity average (baseline)
        motion_median_norm = (motion_median - motion_median.min()) / (motion_median.max() - motion_median.min() + 1e-8)
        ssim_baseline = ssim(activity_avg_norm, motion_median_norm, data_range=1.0)

        # Print simplified metrics (5 lines)
        print(f"fixed_scene: range=[{fixed_scene_mask_img.min():.3f}, {fixed_scene_mask_img.max():.3f}] | dice={dice_score:.3f} | iou={iou_score:.3f}")
        print(f"reconstruction: rmse={rmse_activity:.4f} | ssim={ssim_activity:.4f}")
        print(f"baseline:       rmse={rmse_baseline:.4f} | ssim={ssim_baseline:.4f}")
        print(f"improvement:    rmse={((rmse_baseline - rmse_activity) / rmse_baseline * 100):+.1f}% | ssim={((ssim_activity - ssim_baseline) / (1.0 - ssim_baseline) * 100):+.1f}%")
    else:
        # Using pretrained SIREN - skip metrics that require activity images
        print(f"fixed_scene: range=[{fixed_scene_mask_img.min():.3f}, {fixed_scene_mask_img.max():.3f}]")
        print("(skipping DICE/IoU/SSIM metrics - using pretrained SIREN without activity images)")

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)

    # Save learned fixed scene mask
    imwrite(f'{output_dir}/fixed_scene_learned.tif', fixed_scene_mask_img.astype(np.float32))

    # Save comparison figures (only if activity images were loaded)
    if activity_images:
        imwrite(f'{output_dir}/fixed_scene_binary.tif', fixed_scene_binary.astype(np.float32))

        # Create fixed scene comparison figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        axes[0, 0].imshow(fixed_scene_mask_norm, cmap='viridis')
        axes[0, 0].set_title('Learned Fixed Scene (Normalized)')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(ground_truth_mask, cmap='gray')
        axes[0, 1].set_title('Ground Truth Mask')
        axes[0, 1].axis('off')

        axes[1, 0].imshow(fixed_scene_binary, cmap='gray')
        axes[1, 0].set_title(f'Learned Fixed Scene (Binary)\nDICE: {dice_score:.3f}, IoU: {iou_score:.3f}')
        axes[1, 0].axis('off')

        # Difference map
        diff_map = np.abs(fixed_scene_binary - ground_truth_mask)
        axes[1, 1].imshow(diff_map, cmap='hot')
        axes[1, 1].set_title('Difference (Error Map)')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/fixed_scene_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

    # Create fixed scene distribution histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of raw learned fixed scene values
    axes[0].hist(fixed_scene_mask_img.flatten(), bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Fixed Scene Value', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Learned Fixed Scene Distribution (Raw)', fontsize=14)
    axes[0].axvline(fixed_scene_mask_img.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {fixed_scene_mask_img.mean():.3f}')
    axes[0].axvline(np.median(fixed_scene_mask_img), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(fixed_scene_mask_img):.3f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Histogram of normalized fixed_scene values
    fixed_scene_mask_norm_hist = (fixed_scene_mask_img - fixed_scene_mask_img.min()) / (fixed_scene_mask_img.max() - fixed_scene_mask_img.min() + 1e-8)
    axes[1].hist(fixed_scene_mask_norm_hist.flatten(), bins=100, color='darkgreen', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Normalized Fixed Scene Value', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Learned Fixed Scene Distribution (Normalized)', fontsize=14)
    axes[1].axvline(fixed_scene_mask_norm_hist.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {fixed_scene_mask_norm_hist.mean():.3f}')
    axes[1].axvline(np.median(fixed_scene_mask_norm_hist), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(fixed_scene_mask_norm_hist):.3f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fixed_scene_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save metrics to file
    near_zero_count = np.sum(np.abs(fixed_scene_mask_img) < 0.01)
    sparsity = near_zero_count / fixed_scene_mask_img.size * 100

    # Plot and save loss history with regularization terms
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Main reconstruction loss
    axes[0].plot(loss_history)
    axes[0].set_title('NSTM Reconstruction Loss', fontsize=12)
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('MSE Loss')
    axes[0].grid(True, alpha=0.3)

    # Deformation regularization
    axes[1].plot(regularization_history['deformation'], color='orange')
    axes[1].set_title('Deformation Smoothness', fontsize=12)
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('L1 Deformation')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/loss_history.png', dpi=150)
    plt.close()

    # Generate SIREN activity video with learned affine transform
    if pretrained_activity_net is not None and neuron_positions is not None:
        print("generating SIREN activity video with learned affine transform...")
        render_siren_activity_video(
            pretrained_activity_net,
            neuron_positions,
            affine_scale,
            affine_bias,
            siren_config['nnr_f_T_period'],
            n_frames,
            res,
            device,
            output_dir
        )

    return deformation_net, fixed_scene_net, pretrained_activity_net, loss_history


def create_motion_field_visualization(base_image, motion_x, motion_y, res, step_size=16, black_background=False):
    """Create visualization of motion field with arrows"""
    if black_background:
        # Black background
        vis = np.zeros((res, res, 3), dtype=np.uint8)
    else:
        # Convert base image to RGB
        base_uint8 = (np.clip(base_image, 0, 1) * 255).astype(np.uint8)
        vis = np.stack([base_uint8, base_uint8, base_uint8], axis=2).copy()

    # Draw arrows
    for y in range(0, res, step_size):
        for x in range(0, res, step_size):
            dx = motion_x[y, x] * res * 5  # Scale for visibility (reduced from 10 to 5)
            dy = motion_y[y, x] * res * 5  # Scale for visibility (reduced from 10 to 5)

            if abs(dx) > 0.5 or abs(dy) > 0.5:  # Only draw significant motion
                pt1 = (int(x), int(y))
                pt2 = (int(x + dx), int(y + dy))
                cv2.arrowedLine(vis, pt1, pt2, (0, 255, 0), 1, tipLength=0.3)

    return vis


def create_quad_panel_video(deformation_net, fixed_scene_net, activity_images, motion_images,
                            data_min, data_max, res, device, output_dir, num_frames=90, boat_fixed_scene=None):
    """Create an 8-panel comparison video (4 columns x 2 rows)

    Top row (Training Data):
    - Col 1: Activity
    - Col 2: Activity × Boat fixed_scene
    - Col 3: Ground truth motion field (arrows on black)
    - Col 4: Target (warped motion frames)

    Bottom row (Learned):
    - Col 1: Learned fixed_scene
    - Col 2: Fixed Scene × Activity
    - Col 3: Learned motion field (arrows on black)
    - Col 4: NSTM Reconstruction
    """
    # Create temporary directory for frames
    temp_dir = f"{output_dir}/temp_frames"
    os.makedirs(temp_dir, exist_ok=True)

    # Create coordinate grid
    y_coords = torch.linspace(0, 1, res, device=device, dtype=torch.float16)
    x_coords = torch.linspace(0, 1, res, device=device, dtype=torch.float16)
    yv, xv = torch.meshgrid(y_coords, x_coords, indexing='ij')
    coords_2d = torch.stack([xv.flatten(), yv.flatten()], dim=1)

    # Extract fixed_scene mask once
    with torch.no_grad():
        fixed_scene_mask = torch.sigmoid(fixed_scene_net(coords_2d))
        fixed_scene_mask_img = fixed_scene_mask.reshape(res, res).cpu().numpy()

    # Normalize fixed_scene for visualization
    fixed_scene_norm = (fixed_scene_mask_img - fixed_scene_mask_img.min()) / (fixed_scene_mask_img.max() - fixed_scene_mask_img.min() + 1e-8)

    # Prepare boat fixed_scene for visualization (if provided)
    if boat_fixed_scene is not None:
        boat_norm = (boat_fixed_scene - boat_fixed_scene.min()) / (boat_fixed_scene.max() - boat_fixed_scene.min() + 1e-8)
        boat_uint8 = (np.clip(boat_norm, 0, 1) * 255).astype(np.uint8)
        boat_rgb = np.stack([boat_uint8, boat_uint8, boat_uint8], axis=2)
    else:
        boat_rgb = np.zeros((res, res, 3), dtype=np.uint8)

    # Find original frames at evenly spaced time points
    n_activity_frames = len(activity_images)
    original_indices = [min(int(round(i * (n_activity_frames - 1) / (num_frames - 1))), n_activity_frames - 1)
                        for i in range(num_frames)]

    # Generate frames for each time point
    for i in trange(num_frames, desc="Creating video frames", ncols=100):
        # Get normalized time
        t = i / (num_frames - 1)
        t_idx = original_indices[i]

        # === TOP ROW: Training Data ===

        # Top-1: Activity
        activity_frame = activity_images[t_idx]
        activity_norm = (activity_frame - data_min) / (data_max - data_min)
        activity_uint8 = (np.clip(activity_norm, 0, 1) * 255).astype(np.uint8)
        activity_rgb = np.stack([activity_uint8, activity_uint8, activity_uint8], axis=2)

        # Top-2: Activity × Boat fixed_scene (before warping)
        if boat_fixed_scene is not None:
            activity_times_boat = activity_frame * boat_fixed_scene
            activity_boat_norm = (activity_times_boat - activity_times_boat.min()) / (activity_times_boat.max() - activity_times_boat.min() + 1e-8)
            activity_boat_uint8 = (np.clip(activity_boat_norm, 0, 1) * 255).astype(np.uint8)
            activity_boat_rgb = np.stack([activity_boat_uint8, activity_boat_uint8, activity_boat_uint8], axis=2)
        else:
            activity_boat_rgb = activity_rgb.copy()

        # Top-3: Ground truth motion field (arrows on black background)
        # Compute GT deformation at current frame
        gt_dx, gt_dy = compute_gt_deformation_field(t_idx, len(activity_images), res, motion_intensity=0.015)
        gt_motion_vis = create_motion_field_visualization(None, gt_dx, gt_dy, res, step_size=16, black_background=True)

        # Top-4: Target (warped motion frame)
        motion_frame = motion_images[t_idx]
        motion_norm = (motion_frame - data_min) / (data_max - data_min)
        motion_uint8 = (np.clip(motion_norm, 0, 1) * 255).astype(np.uint8)
        target_rgb = np.stack([motion_uint8, motion_uint8, motion_uint8], axis=2)

        # === BOTTOM ROW: Learned Components ===

        # Compute learned components
        with torch.no_grad():
            # Create 3D coordinates for deformation
            t_tensor = torch.full((coords_2d.shape[0], 1), t, device=device, dtype=torch.float16)
            coords_3d = torch.cat([coords_2d, t_tensor], dim=1)

            # Get deformation
            deformation = deformation_net(coords_3d)

            # Backward warp
            source_coords = coords_2d - deformation
            source_coords = torch.clamp(source_coords, 0, 1)

            # Sample fixed_scene at source coords
            fixed_scene_at_source = torch.sigmoid(fixed_scene_net(source_coords))

            # Sample activity at source coords
            activity_tensor = torch.tensor((activity_frame - data_min) / (data_max - data_min),
                                          dtype=torch.float16, device=device).reshape(1, 1, res, res)
            source_coords_normalized = source_coords * 2 - 1
            source_coords_grid = source_coords_normalized.reshape(1, 1, -1, 2)

            sampled_activity = torch.nn.functional.grid_sample(
                activity_tensor, source_coords_grid,
                mode='bilinear', padding_mode='border', align_corners=True
            ).reshape(-1, 1)

            # Reconstruction
            recon = (fixed_scene_at_source * sampled_activity).reshape(res, res).cpu().numpy()
            sampled_activity_img = sampled_activity.reshape(res, res).cpu().numpy()

        # Bottom-1: Learned fixed_scene
        fixed_scene_uint8 = (np.clip(fixed_scene_norm, 0, 1) * 255).astype(np.uint8)
        learned_fixed_scene_rgb = np.stack([fixed_scene_uint8, fixed_scene_uint8, fixed_scene_uint8], axis=2)

        # Bottom-2: Fixed Scene × Activity
        fixed_scene_times_activity = fixed_scene_mask_img * activity_frame
        fixed_scene_act_norm = (fixed_scene_times_activity - fixed_scene_times_activity.min()) / (fixed_scene_times_activity.max() - fixed_scene_times_activity.min() + 1e-8)
        fixed_scene_act_uint8 = (np.clip(fixed_scene_act_norm, 0, 1) * 255).astype(np.uint8)
        fixed_scene_activity_rgb = np.stack([fixed_scene_act_uint8, fixed_scene_act_uint8, fixed_scene_act_uint8], axis=2)

        # Bottom-3: Learned motion field (arrows on black background)
        deformation_2d = deformation.reshape(res, res, 2).cpu().numpy()
        motion_x = deformation_2d[:, :, 0]
        motion_y = deformation_2d[:, :, 1]
        learned_motion_vis = create_motion_field_visualization(fixed_scene_norm, motion_x, motion_y, res, step_size=16, black_background=True)

        # Bottom-4: NSTM Reconstruction
        recon_denorm = recon * (data_max - data_min) + data_min
        recon_norm = (recon_denorm - data_min) / (data_max - data_min)
        recon_uint8 = (np.clip(recon_norm, 0, 1) * 255).astype(np.uint8)
        recon_rgb = np.stack([recon_uint8, recon_uint8, recon_uint8], axis=2)

        # Add text labels - top-left position with larger font
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8  # Increased from 0.4
        thickness = 2  # Increased from 1
        margin = 10
        y_pos = margin + 25  # Top position (changed from bottom)

        # Top row labels - all white text
        cv2.putText(activity_rgb, "Activity", (margin, y_pos), font, font_scale, (255, 255, 255), thickness)

        cv2.putText(activity_boat_rgb, "Activity x Boat", (margin, y_pos), font, font_scale, (255, 255, 255), thickness)

        cv2.putText(gt_motion_vis, "GT Motion", (margin, y_pos), font, font_scale, (255, 255, 255), thickness)

        cv2.putText(target_rgb, "Target", (margin, y_pos), font, font_scale, (255, 255, 255), thickness)

        # Bottom row labels - all white text
        cv2.putText(learned_fixed_scene_rgb, "Learned Fixed Scene", (margin, y_pos), font, font_scale, (255, 255, 255), thickness)

        cv2.putText(fixed_scene_activity_rgb, "Fixed Scene x Activity", (margin, y_pos), font, font_scale, (255, 255, 255), thickness)

        cv2.putText(learned_motion_vis, "Learned Motion", (margin, y_pos), font, font_scale, (255, 255, 255), thickness)

        cv2.putText(recon_rgb, "Reconstruction", (margin, y_pos), font, font_scale, (255, 255, 255), thickness)

        # Create 4x2 grid layout
        top_row = np.hstack([activity_rgb, activity_boat_rgb, gt_motion_vis, target_rgb])
        bottom_row = np.hstack([learned_fixed_scene_rgb, fixed_scene_activity_rgb, learned_motion_vis, recon_rgb])
        combined = np.vstack([top_row, bottom_row])

        # Save frame
        cv2.imwrite(f"{temp_dir}/frame_{i:04d}.png", combined)

    # Create video with ffmpeg
    video_path = f'{output_dir}/nstm.mp4'
    fps = 30

    print("creating video from frames...")
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-pattern_type", "glob", "-i", f"{temp_dir}/frame_*.png",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "slow", "-crf", "22",
        video_path
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"video saved to {video_path}")
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"error creating video with ffmpeg: {e}")
        print("falling back to opencv video writer...")

        # Fallback to OpenCV
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, fps, (res*4, res*2))  # 4 columns x 2 rows

        for j in range(num_frames):
            frame_path = f"{temp_dir}/frame_{j:04d}.png"
            frame = cv2.imread(frame_path)
            video.write(frame)

        video.release()
        print(f"video saved to {video_path}")

    # Clean up temporary files
    print("cleaning up temporary files...")
    shutil.rmtree(temp_dir)

    return video_path


def data_instant_NGP(config=None, style=None, device=None):


    simulation_config = config.simulation
    training_config = config.training
    model_config = config.graph_model


    dataset_name = config.dataset
    n_frames = 256
    delta_t = simulation_config.delta_t
    dimension = simulation_config.dimension


    # Create motion_frames and activity directories
    motion_frames_dir = f'/groups/saalfeld/home/allierc/Py/NeuralGraph/log/{dataset_name}/motion_frames'
    activity_dir = f'/groups/saalfeld/home/allierc/Py/NeuralGraph/log/{dataset_name}/activity'
    os.makedirs(motion_frames_dir, exist_ok=True)
    os.makedirs(activity_dir, exist_ok=True)

    print(f"generating {motion_frames_dir} motion frames with fixed_scene modulation and sinusoidal warping...")

    # Clear existing motion frames and activity frames
    files = glob.glob(f'{motion_frames_dir}/*')
    for f in files:
        os.remove(f)
    files = glob.glob(f'{activity_dir}/*')
    for f in files:
        os.remove(f)

    # Load boat fixed_scene image
    import os as os_module
    current_dir = os_module.path.dirname(os_module.path.abspath(__file__))
    boat_fixed_scene_path = os_module.path.join(current_dir, 'pics_boat_512.tif')
    print(f"loading boat fixed_scene from {boat_fixed_scene_path}")
    boat_fixed_scene = imread(boat_fixed_scene_path).astype(np.float32)
    print(f"boat fixed_scene shape: {boat_fixed_scene.shape}, range: [{boat_fixed_scene.min():.4f}, {boat_fixed_scene.max():.4f}]")


    if "latex" in style:
        plt.rcParams["text.usetex"] = True
        rc("font", **{"family": "serif", "serif": ["Palatino"]})
    if "black" in style:
        plt.style.use("dark_background")
    matplotlib.rcParams["savefig.pad_inches"] = 0

    id_fig = 0
    run = 0
    n_runs = 1

    x_list = []
    y_list = []
    for run in trange(0,n_runs, ncols=100):
        x = np.load(f'graphs_data/{dataset_name}/x_list_{run}.npy')
        y = np.load(f'graphs_data/{dataset_name}/y_list_{run}.npy')
        x_list.append(x)
        y_list.append(y)
    
    run = 0
    x = x_list[0][n_frames - 10]
    n_neurons = x.shape[0]
    config.simulation.n_neurons =n_neurons
    type_list = torch.tensor(x[:, 1 + 2 * dimension:2 + 2 * dimension], device=device)
    X1 = torch.tensor(x[:, 1 : 1 + dimension], device=device)

    # COMMENTED OUT FOR FASTER TESTING - UNCOMMENT TO REGENERATE DATA
    for it in trange(0, n_frames, ncols=100):

        x = torch.tensor(x_list[run][it], dtype=torch.float32, device=device)

        num = f"{id_fig:06}"
        id_fig += 1


        plt.figure(figsize=(10, 10))
        plt.axis("off")
        # Create figure and render to get pixel data
        fig = plt.figure(figsize=(512/80, 512/80), dpi=80)  # 512x512 pixels
        plt.scatter(
            to_numpy(X1[:, 0]),
            to_numpy(X1[:, 1]),
            s=700,
            c=to_numpy(x[:, 6]),
            cmap="viridis",
            vmin=0,
            vmax=20,
        )
        plt.axis("off")
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()

        # Render to canvas and extract grayscale data
        fig.canvas.draw()
        img_rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_rgba = img_rgba.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img_rgba = img_rgba[:, :, :3]  # Convert RGBA to RGB

        # Convert RGB to grayscale
        img_gray = np.dot(img_rgba[...,:3], [0.2989, 0.5870, 0.1140])

        # Resize to exactly 512x512 if needed
        from scipy.ndimage import zoom
        if img_gray.shape != (512, 512):
            zoom_factors = (512 / img_gray.shape[0], 512 / img_gray.shape[1])
            img_gray = zoom(img_gray, zoom_factors, order=1)

        # Save activity image: dots at FIXED initial position with changing activity (x[:,6])
        # Activity changes over time, so save each frame
        img_activity_32bit = img_gray.astype(np.float32)
        imwrite(
            f"{activity_dir}/frame_{num}.tif",
            img_activity_32bit,
            photometric='minisblack',
            dtype=np.float32
        )

        # For motion frames: multiply activity by boat fixed_scene, then warp
        # Element-wise multiplication: activity × boat_fixed_scene
        img_with_fixed_scene = img_gray * boat_fixed_scene

        # Apply sinusoidal warping to the combined image
        # The warped result contains both activity and fixed_scene information
        img_warped = apply_sinusoidal_warp(img_with_fixed_scene, it, n_frames, motion_intensity=0.015)

        # Convert to 32-bit float (single channel)
        img_32bit = img_warped.astype(np.float32)

        # Save as 32-bit single channel TIF to motion_frames directory
        imwrite(
            f"{motion_frames_dir}/frame_{num}.tif",
            img_32bit,
            photometric='minisblack',  # grayscale
            dtype=np.float32
        )
        plt.close()

    print(f"generated {n_frames} warped motion frames in {motion_frames_dir}/")
    print("frame format: 512x512, 32-bit float, single channel tif")
    print("applied sinusoidal warping with motion_intensity=0.015")

    # Train NSTM on the generated frames
    nstm_output_dir = f'/groups/saalfeld/home/allierc/Py/NeuralGraph/log/{dataset_name}/NSTM_outputs'

    # Prepare SIREN config if available in model_config
    siren_config = None
    if hasattr(model_config, 'hidden_dim_nnr_f'):
        siren_config = {
            'hidden_dim_nnr_f': model_config.hidden_dim_nnr_f,
            'n_layers_nnr_f': model_config.n_layers_nnr_f,
            'omega_f': model_config.omega_f,
            'nnr_f_xy_period': model_config.nnr_f_xy_period,
            'nnr_f_T_period': model_config.nnr_f_T_period,
            'outermost_linear_nnr_f': model_config.outermost_linear_nnr_f,
            'output_size_nnr_f': model_config.output_size_nnr_f
        }
        print(f"using siren activity network with config: {siren_config}")
    else:
        print("no siren config found, using grid_sample for activity")

    # Hardcoded parameters
    use_siren = False  # Set to False to use grid_sample instead of SIREN
    siren_loss_weight = 1.0  # Weight for SIREN supervision loss on discrete neurons
    nstm_lr = 1e-4  # Learning rate for deformation and fixed_scene networks

    # Get SIREN learning rate from config if available
    siren_lr = 1e-4  # Default
    if hasattr(training_config, 'learning_rate_NNR_f'):
        siren_lr = training_config.learning_rate_NNR_f

    # Pre-train SIREN on discrete neuron time series (t -> activities)
    pretrained_activity_net = None
    if siren_config is not None and use_siren:
        print("pre-training siren time network on discrete neuron data...")
        pretrained_activity_net = train_siren(
            x_list=x_list,
            device=device,
            output_dir=nstm_output_dir,
            num_training_steps=10000,
            nnr_f_T_period=siren_config['nnr_f_T_period'],
            n_train_frames=256,
            n_neurons=siren_config['output_size_nnr_f']
        )

    deformation_net, fixed_scene_net, activity_net, loss_history = train_nstm(
        motion_frames_dir=motion_frames_dir,
        activity_dir=activity_dir,
        n_frames=n_frames,
        res=512,
        device=device,
        output_dir=nstm_output_dir,
        num_training_steps=10000,  # Increased from 5000 (4x more)
        siren_config=siren_config,
        pretrained_activity_net=pretrained_activity_net,
        x_list=x_list,
        use_siren=use_siren,
        siren_lr=siren_lr,
        nstm_lr=nstm_lr,
        siren_loss_weight=siren_loss_weight
    )

    # Load motion and activity images for video creation
    motion_images_list = []
    activity_images_list = []
    for i in range(n_frames):
        motion_images_list.append(imread(f"{motion_frames_dir}/frame_{i:06d}.tif"))
        activity_images_list.append(imread(f"{activity_dir}/frame_{i:06d}.tif"))

    # Compute data range for normalization
    all_pixels = np.concatenate([img.flatten() for img in motion_images_list])
    data_min = all_pixels.min()
    data_max = all_pixels.max()

    # Create 4-panel comparison video
    video_path = create_quad_panel_video(
        deformation_net=deformation_net,
        fixed_scene_net=fixed_scene_net,
        activity_images=activity_images_list,
        motion_images=motion_images_list,
        data_min=data_min,
        data_max=data_max,
        res=512,
        device=device,
        output_dir=nstm_output_dir,
        num_frames=256,  # Use all frames
        boat_fixed_scene=boat_fixed_scene  # Pass the boat fixed_scene for visualization
    )
    print(f"8-panel video (4x2) saved to: {video_path}")
    print("training completed.")

