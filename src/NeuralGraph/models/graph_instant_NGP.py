import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch_geometric.data as data
from matplotlib import rc
from NeuralGraph.data_loaders import load_wormvae_data, load_zebrafish_data
from NeuralGraph.generators.davis import AugmentedDavis
from NeuralGraph.generators.utils import (
    choose_model,
    init_neurons,
    init_mesh,
    generate_compressed_video_mp4,
    init_connectivity,
    get_equidistant_points,
)
from NeuralGraph.utils import to_numpy, CustomColorMap, check_and_clear_memory, get_datavis_root_dir
from tifffile import imread, imwrite
from tqdm import tqdm, trange
import os
from scipy.ndimage import map_coordinates
import tinycudann as tcnn
from skimage.metrics import structural_similarity as ssim
import cv2
import subprocess
import shutil

# from fa2_modified import ForceAtlas2
# import h5py as h5
# import zarr
# import xarray as xr
import torch_geometric as pyg


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


def train_nstm(motion_frames_dir, activity_dir, n_frames, res, device, output_dir, num_training_steps=3000):
    """Train Neural Space-Time Model with anatomy + activity decomposition"""
    print(f"\n{'='*60}")
    print("Starting NSTM Training with Anatomy Network")
    print(f"{'='*60}")

    # Load motion frames
    print("Loading motion frames...")
    motion_images = []
    for i in range(n_frames):
        img = imread(f"{motion_frames_dir}/frame_{i:06d}.tif")
        motion_images.append(img)

    # Load original activity images
    print("Loading original activity images...")
    activity_images = []
    for i in range(n_frames):
        img = imread(f"{activity_dir}/frame_{i:06d}.tif")
        activity_images.append(img)

    # Compute normalization statistics
    all_pixels = np.concatenate([img.flatten() for img in motion_images])
    data_min = all_pixels.min()
    data_max = all_pixels.max()
    print(f"Input data range: [{data_min:.2f}, {data_max:.2f}]")

    # Normalize to [0, 1] and convert to tensors (use float16 for tinycudann compatibility)
    # Keep both motion (warped) and activity (ground-truth) in GPU memory
    motion_tensors = [torch.tensor((img - data_min) / (data_max - data_min), dtype=torch.float16).to(device) for img in motion_images]
    activity_tensors = [torch.tensor((img - data_min) / (data_max - data_min), dtype=torch.float16).to(device) for img in activity_images]

    print(f"Loaded {n_frames} frames into GPU memory")

    # Create networks
    print("Creating networks...")
    config = create_network_config()

    # Deformation network: (x, y, t) -> (δx, δy)
    deformation_net = tcnn.NetworkWithInputEncoding(
        n_input_dims=3,  # x, y, t
        n_output_dims=2,  # δx, δy
        encoding_config=config["encoding"],
        network_config=config["network"]
    ).to(device)

    # Anatomy network: (x, y) -> raw mask value
    anatomy_net = tcnn.NetworkWithInputEncoding(
        n_input_dims=2,  # x, y
        n_output_dims=1,  # mask value
        encoding_config=config["encoding"],
        network_config=config["network"]
    ).to(device)

    # Create coordinate grid (use float16 for tinycudann)
    y_coords = torch.linspace(0, 1, res, device=device, dtype=torch.float16)
    x_coords = torch.linspace(0, 1, res, device=device, dtype=torch.float16)
    yv, xv = torch.meshgrid(y_coords, x_coords, indexing='ij')
    coords_2d = torch.stack([xv.flatten(), yv.flatten()], dim=1)

    # Create optimizer
    params = list(deformation_net.parameters()) + list(anatomy_net.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)

    # Learning rate schedule
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[num_training_steps // 2, num_training_steps * 3 // 4],
        gamma=0.1
    )

    # Training loop
    print(f"Training for {num_training_steps} steps...")
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

        # Create 3D coordinates for deformation network
        t_tensor = torch.full_like(batch_coords[:, 0:1], t_normalized)
        coords_3d = torch.cat([batch_coords, t_tensor], dim=1)

        # Forward pass - deformation network (backward warp)
        deformation = deformation_net(coords_3d)

        # Compute source coordinates (backward warp)
        source_coords = batch_coords - deformation  # Note: MINUS for backward warp
        source_coords = torch.clamp(source_coords, 0, 1)

        # Sample anatomy mask at source coordinates
        anatomy_mask = anatomy_net(source_coords)

        # Sample ground-truth activity at source coordinates using grid_sample
        # Convert source_coords from [0,1] to [-1,1] for grid_sample
        source_coords_normalized = source_coords * 2 - 1
        source_coords_grid = source_coords_normalized.reshape(1, 1, batch_size, 2)

        # Reshape activity tensor for grid_sample: [1, 1, H, W]
        activity_frame = activity_tensors[t_idx].reshape(1, 1, res, res)

        # Sample using bilinear interpolation
        sampled_activity = torch.nn.functional.grid_sample(
            activity_frame,
            source_coords_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        sampled_activity = sampled_activity.reshape(-1, 1).to(torch.float16)

        # Reconstruction: anatomy × activity
        reconstructed = anatomy_mask * sampled_activity

        # Main reconstruction loss
        loss = torch.nn.functional.mse_loss(reconstructed, target)

        # Regularization: Deformation smoothness
        deformation_smoothness = torch.mean(torch.abs(deformation))

        # Combined loss with regularization
        total_loss = loss + 0.01 * deformation_smoothness

        # Optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Update learning rate
        scheduler.step()

        # Record losses
        loss_history.append(loss.item())
        regularization_history['deformation'].append(deformation_smoothness.item())

        # Update progress bar with losses
        pbar.set_postfix({
            'loss': f'{loss.item():.6f}',
            'def': f'{deformation_smoothness.item():.4f}'
        })

    print("Training complete!")

    # Extract learned anatomy mask
    print("Extracting learned anatomy mask...")
    with torch.no_grad():
        anatomy_mask = anatomy_net(coords_2d)
        anatomy_mask_img = anatomy_mask.reshape(res, res).cpu().numpy()

    print(f"Learned anatomy range: [{anatomy_mask_img.min():.4f}, {anatomy_mask_img.max():.4f}]")

    # Compute average of original activity images
    print("\nComputing average of original activity images...")
    activity_average = np.mean(activity_images, axis=0)

    # Create ground-truth mask from activity average (threshold at mean)
    threshold = activity_average.mean()
    ground_truth_mask = (activity_average > threshold).astype(np.float32)
    n_gt_pixels = ground_truth_mask.sum()
    print(f"Ground-truth mask: {n_gt_pixels}/{ground_truth_mask.size} pixels above threshold")

    # Normalize anatomy mask for comparison
    anatomy_mask_norm = (anatomy_mask_img - anatomy_mask_img.min()) / (anatomy_mask_img.max() - anatomy_mask_img.min() + 1e-8)

    print(f"\n🔍 DEBUG: Using TOP-N thresholding (fixed evaluation method)")
    print(f"   Anatomy norm stats: min={anatomy_mask_norm.min():.4f}, max={anatomy_mask_norm.max():.4f}, mean={anatomy_mask_norm.mean():.4f}")

    # Threshold anatomy to match ground truth coverage (top N pixels)
    flat_anatomy = anatomy_mask_norm.flatten()
    n_pixels_to_select = int(n_gt_pixels)
    sorted_indices = np.argsort(flat_anatomy)[::-1]  # Sort descending
    anatomy_binary_flat = np.zeros_like(flat_anatomy, dtype=np.float32)
    anatomy_binary_flat[sorted_indices[:n_pixels_to_select]] = 1.0
    anatomy_binary = anatomy_binary_flat.reshape(anatomy_mask_norm.shape)

    print(f"   Selecting top {n_pixels_to_select} pixels from {anatomy_binary.size} total")
    print(f"Learned anatomy binary: {int(anatomy_binary.sum())}/{anatomy_binary.size} pixels (matched to GT coverage)")

    # Compute DICE score and IoU between learned anatomy and ground-truth mask
    intersection = np.sum(anatomy_binary * ground_truth_mask)
    union = np.sum(anatomy_binary) + np.sum(ground_truth_mask)
    dice_score = 2 * intersection / (union + 1e-8)
    iou_score = intersection / (union - intersection + 1e-8)

    print(f"\n{'='*60}")
    print("Anatomy Mask Evaluation")
    print(f"{'='*60}")
    print(f"DICE Score: {dice_score:.4f}")
    print(f"IoU Score: {iou_score:.4f}")
    print(f"{'='*60}\n")

    # Compute median of motion frames (warped frames) as baseline
    print("Computing median of motion frames...")
    motion_median = np.median(motion_images, axis=0)

    # Reconstruct fixed scene: anatomy × activity_average
    fixed_scene_denorm = anatomy_mask_img * activity_average
    print(f"Fixed scene range: [{fixed_scene_denorm.min():.2f}, {fixed_scene_denorm.max():.2f}]")

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

    print(f"{'='*60}")
    print("NSTM Reconstruction Metrics")
    print(f"{'='*60}")
    print(f"Fixed Scene vs Activity Average:")
    print(f"  RMSE: {rmse_activity:.4f}")
    print(f"  SSIM: {ssim_activity:.4f}")
    print(f"\nBaseline (Motion Median vs Activity Average):")
    print(f"  RMSE: {rmse_baseline:.4f}")
    print(f"  SSIM: {ssim_baseline:.4f}")
    print(f"\nImprovement over baseline:")
    print(f"  RMSE: {((rmse_baseline - rmse_activity) / rmse_baseline * 100):.2f}%")
    print(f"  SSIM: {((ssim_activity - ssim_baseline) / (1.0 - ssim_baseline) * 100):.2f}%")
    print(f"{'='*60}\n")

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)

    # Save only anatomy masks (not all intermediate images)
    imwrite(f'{output_dir}/anatomy_learned.tif', anatomy_mask_img.astype(np.float32))
    imwrite(f'{output_dir}/anatomy_binary.tif', anatomy_binary.astype(np.float32))
    print(f"Saved anatomy masks")

    # Create anatomy comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].imshow(anatomy_mask_norm, cmap='viridis')
    axes[0, 0].set_title('Learned Anatomy (Normalized)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(ground_truth_mask, cmap='gray')
    axes[0, 1].set_title('Ground Truth Mask')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(anatomy_binary, cmap='gray')
    axes[1, 0].set_title(f'Learned Anatomy (Binary)\nDICE: {dice_score:.3f}, IoU: {iou_score:.3f}')
    axes[1, 0].axis('off')

    # Difference map
    diff_map = np.abs(anatomy_binary - ground_truth_mask)
    axes[1, 1].imshow(diff_map, cmap='hot')
    axes[1, 1].set_title('Difference (Error Map)')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/anatomy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved anatomy comparison to {output_dir}/anatomy_comparison.png")

    # Create anatomy distribution histogram with metrics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of raw learned anatomy values
    axes[0].hist(anatomy_mask_img.flatten(), bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Anatomy Value', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Learned Anatomy Distribution (Raw)', fontsize=14)
    axes[0].axvline(anatomy_mask_img.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {anatomy_mask_img.mean():.3f}')
    axes[0].axvline(np.median(anatomy_mask_img), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(anatomy_mask_img):.3f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Histogram of normalized anatomy values
    axes[1].hist(anatomy_mask_norm.flatten(), bins=100, color='darkgreen', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Normalized Anatomy Value', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Learned Anatomy Distribution (Normalized)', fontsize=14)
    axes[1].axvline(anatomy_mask_norm.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {anatomy_mask_norm.mean():.3f}')
    axes[1].axvline(np.median(anatomy_mask_norm), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(anatomy_mask_norm):.3f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/anatomy_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved anatomy distribution histogram to {output_dir}/anatomy_distribution.png")

    # Compute and print distribution metrics
    from scipy import stats
    print(f"\n{'='*60}")
    print("Anatomy Distribution Metrics")
    print(f"{'='*60}")
    print(f"Raw Anatomy Values:")
    print(f"  Min: {anatomy_mask_img.min():.4f}")
    print(f"  Max: {anatomy_mask_img.max():.4f}")
    print(f"  Mean: {anatomy_mask_img.mean():.4f}")
    print(f"  Median: {np.median(anatomy_mask_img):.4f}")
    print(f"  Std: {anatomy_mask_img.std():.4f}")
    print(f"  Skewness: {stats.skew(anatomy_mask_img.flatten()):.4f}")
    print(f"  Kurtosis: {stats.kurtosis(anatomy_mask_img.flatten()):.4f}")

    # Percentiles
    p25, p50, p75, p95, p99 = np.percentile(anatomy_mask_img.flatten(), [25, 50, 75, 95, 99])
    print(f"\nPercentiles:")
    print(f"  25th: {p25:.4f}")
    print(f"  50th: {p50:.4f}")
    print(f"  75th: {p75:.4f}")
    print(f"  95th: {p95:.4f}")
    print(f"  99th: {p99:.4f}")

    # Sparsity analysis
    near_zero_count = np.sum(np.abs(anatomy_mask_img) < 0.01)
    sparsity = near_zero_count / anatomy_mask_img.size * 100
    print(f"\nSparsity Analysis:")
    print(f"  Near-zero values (|x| < 0.01): {near_zero_count}/{anatomy_mask_img.size} ({sparsity:.2f}%)")
    print(f"  Non-zero values: {anatomy_mask_img.size - near_zero_count}/{anatomy_mask_img.size} ({100-sparsity:.2f}%)")
    print(f"{'='*60}\n")

    # Save metrics to file
    with open(f'{output_dir}/metrics.txt', 'w') as f:
        f.write("NSTM Evaluation Metrics\n")
        f.write("="*60 + "\n\n")

        f.write("Anatomy Mask Evaluation:\n")
        f.write(f"  DICE Score: {dice_score:.4f}\n")
        f.write(f"  IoU Score: {iou_score:.4f}\n\n")

        f.write("Fixed Scene vs Activity Average:\n")
        f.write(f"  RMSE: {rmse_activity:.4f}\n")
        f.write(f"  SSIM: {ssim_activity:.4f}\n\n")

        f.write("Baseline (Motion Median vs Activity Average):\n")
        f.write(f"  RMSE: {rmse_baseline:.4f}\n")
        f.write(f"  SSIM: {ssim_baseline:.4f}\n\n")

        f.write("Improvement over baseline:\n")
        f.write(f"  RMSE: {((rmse_baseline - rmse_activity) / rmse_baseline * 100):.2f}%\n")
        f.write(f"  SSIM: {((ssim_activity - ssim_baseline) / (1.0 - ssim_baseline) * 100):.2f}%\n\n")

        f.write("Anatomy Distribution:\n")
        f.write(f"  Min: {anatomy_mask_img.min():.4f}\n")
        f.write(f"  Max: {anatomy_mask_img.max():.4f}\n")
        f.write(f"  Mean: {anatomy_mask_img.mean():.4f}\n")
        f.write(f"  Median: {np.median(anatomy_mask_img):.4f}\n")
        f.write(f"  Std Dev: {anatomy_mask_img.std():.4f}\n")
        f.write(f"  Skewness: {stats.skew(anatomy_mask_img.flatten()):.4f}\n")
        f.write(f"  Kurtosis: {stats.kurtosis(anatomy_mask_img.flatten()):.4f}\n")
        f.write(f"  25th Percentile: {np.percentile(anatomy_mask_img, 25):.4f}\n")
        f.write(f"  75th Percentile: {np.percentile(anatomy_mask_img, 75):.4f}\n")
        f.write(f"  Sparsity (|x| < 0.01): {sparsity:.2f}%\n")
        f.write("="*60 + "\n")
    print(f"Saved metrics to {output_dir}/metrics.txt")

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
    print(f"Saved loss plot to {output_dir}/loss_history.png")

    print(f"{'='*60}\n")

    return deformation_net, anatomy_net, loss_history


def create_motion_field_visualization(base_image, motion_x, motion_y, res, step_size=16):
    """Create visualization of motion field with arrows"""
    # Convert base image to RGB
    base_uint8 = (np.clip(base_image, 0, 1) * 255).astype(np.uint8)
    vis = np.stack([base_uint8, base_uint8, base_uint8], axis=2).copy()

    # Draw arrows
    for y in range(0, res, step_size):
        for x in range(0, res, step_size):
            dx = motion_x[y, x] * res * 20  # Scale for visibility
            dy = motion_y[y, x] * res * 20

            if abs(dx) > 0.5 or abs(dy) > 0.5:  # Only draw significant motion
                pt1 = (int(x), int(y))
                pt2 = (int(x + dx), int(y + dy))
                cv2.arrowedLine(vis, pt1, pt2, (0, 255, 0), 1, tipLength=0.3)

    return vis


def create_quad_panel_video(deformation_net, anatomy_net, activity_images, motion_images,
                            data_min, data_max, res, device, output_dir, num_frames=90):
    """Create a four-panel comparison video

    Panels:
    - Top left: Original activity frame
    - Top right: Warped motion frame
    - Bottom left: NSTM reconstruction (anatomy × activity with deformation)
    - Bottom right: Anatomy + deformation visualization
    """
    print("\n" + "="*60)
    print("Creating 4-panel comparison video...")
    print("="*60)

    # Create temporary directory for frames
    temp_dir = f"{output_dir}/temp_frames"
    os.makedirs(temp_dir, exist_ok=True)

    # Create coordinate grid
    y_coords = torch.linspace(0, 1, res, device=device, dtype=torch.float16)
    x_coords = torch.linspace(0, 1, res, device=device, dtype=torch.float16)
    yv, xv = torch.meshgrid(y_coords, x_coords, indexing='ij')
    coords_2d = torch.stack([xv.flatten(), yv.flatten()], dim=1)

    # Extract anatomy mask once
    with torch.no_grad():
        anatomy_mask = anatomy_net(coords_2d)
        anatomy_mask_img = anatomy_mask.reshape(res, res).cpu().numpy()

    # Normalize anatomy for visualization
    anatomy_norm = (anatomy_mask_img - anatomy_mask_img.min()) / (anatomy_mask_img.max() - anatomy_mask_img.min() + 1e-8)

    # Find original frames at evenly spaced time points
    n_activity_frames = len(activity_images)
    original_indices = [min(int(round(i * (n_activity_frames - 1) / (num_frames - 1))), n_activity_frames - 1)
                        for i in range(num_frames)]

    # Generate frames for each time point
    for i in trange(num_frames, desc="Creating video frames"):
        # Get normalized time
        t = i / (num_frames - 1)
        t_idx = original_indices[i]

        # Panel 1: Original activity frame
        activity_frame = activity_images[t_idx]
        activity_norm = (activity_frame - data_min) / (data_max - data_min)
        activity_uint8 = (np.clip(activity_norm, 0, 1) * 255).astype(np.uint8)
        activity_rgb = np.stack([activity_uint8, activity_uint8, activity_uint8], axis=2)

        # Panel 2: Warped motion frame
        motion_frame = motion_images[t_idx]
        motion_norm = (motion_frame - data_min) / (data_max - data_min)
        motion_uint8 = (np.clip(motion_norm, 0, 1) * 255).astype(np.uint8)
        motion_rgb = np.stack([motion_uint8, motion_uint8, motion_uint8], axis=2)

        # Panel 3: NSTM reconstruction (anatomy × activity with deformation)
        with torch.no_grad():
            # Create 3D coordinates for deformation
            t_tensor = torch.full((coords_2d.shape[0], 1), t, device=device, dtype=torch.float16)
            coords_3d = torch.cat([coords_2d, t_tensor], dim=1)

            # Get deformation
            deformation = deformation_net(coords_3d)

            # Backward warp
            source_coords = coords_2d - deformation
            source_coords = torch.clamp(source_coords, 0, 1)

            # Sample anatomy at source coords
            anatomy_at_source = anatomy_net(source_coords)

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
            recon = (anatomy_at_source * sampled_activity).reshape(res, res).cpu().numpy()

        # Denormalize and convert to uint8
        recon_denorm = recon * (data_max - data_min) + data_min
        recon_norm = (recon_denorm - data_min) / (data_max - data_min)
        recon_uint8 = (np.clip(recon_norm, 0, 1) * 255).astype(np.uint8)
        recon_rgb = np.stack([recon_uint8, recon_uint8, recon_uint8], axis=2)

        # Panel 4: Anatomy + deformation visualization
        with torch.no_grad():
            deformation_2d = deformation.reshape(res, res, 2).cpu().numpy()
            motion_x = deformation_2d[:, :, 0]
            motion_y = deformation_2d[:, :, 1]

        anatomy_deform_vis = create_motion_field_visualization(anatomy_norm, motion_x, motion_y, res, step_size=16)

        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        margin = 10
        y_pos = res - margin

        cv2.putText(activity_rgb, f"Activity t={t:.2f}", (margin, y_pos), font, font_scale, (0, 0, 0), thickness+1)
        cv2.putText(activity_rgb, f"Activity t={t:.2f}", (margin, y_pos), font, font_scale, (255, 255, 255), thickness)

        cv2.putText(motion_rgb, f"Warped t={t:.2f}", (margin, y_pos), font, font_scale, (0, 0, 0), thickness+1)
        cv2.putText(motion_rgb, f"Warped t={t:.2f}", (margin, y_pos), font, font_scale, (255, 255, 255), thickness)

        cv2.putText(recon_rgb, f"NSTM Recon t={t:.2f}", (margin, y_pos), font, font_scale, (0, 0, 0), thickness+1)
        cv2.putText(recon_rgb, f"NSTM Recon t={t:.2f}", (margin, y_pos), font, font_scale, (255, 255, 255), thickness)

        cv2.putText(anatomy_deform_vis, "Anatomy+Deformation", (margin, y_pos), font, font_scale, (0, 0, 0), thickness+1)
        cv2.putText(anatomy_deform_vis, "Anatomy+Deformation", (margin, y_pos), font, font_scale, (255, 255, 255), thickness)

        # Create 2x2 grid layout
        top_row = np.hstack([activity_rgb, motion_rgb])
        bottom_row = np.hstack([recon_rgb, anatomy_deform_vis])
        combined = np.vstack([top_row, bottom_row])

        # Save frame
        cv2.imwrite(f"{temp_dir}/frame_{i:04d}.png", combined)

    # Create video with ffmpeg
    video_path = f'{output_dir}/nstm_4panel.mp4'
    fps = 30

    print("Creating video from frames...")
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

        for j in range(num_frames):
            frame_path = f"{temp_dir}/frame_{j:04d}.png"
            frame = cv2.imread(frame_path)
            video.write(frame)

        video.release()
        print(f"Video saved to {video_path}")

    # Clean up temporary files
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)

    print(f"{'='*60}\n")
    return video_path


def data_instant_NGP(config=None, style=None, device=None):


    simulation_config = config.simulation
    training_config = config.training
    model_config = config.graph_model


    dataset_name = config.dataset
    n_frames = 128
    delta_t = simulation_config.delta_t
    dimension = simulation_config.dimension


    # Create motion_frames and activity directories
    motion_frames_dir = f'./graphs_data/{dataset_name}/motion_frames'
    activity_dir = f'./graphs_data/{dataset_name}/activity'
    os.makedirs(motion_frames_dir, exist_ok=True)
    os.makedirs(activity_dir, exist_ok=True)

    # Clear existing motion frames and activity frames
    files = glob.glob(f'{motion_frames_dir}/*')
    for f in files:
        os.remove(f)
    files = glob.glob(f'{activity_dir}/*')
    for f in files:
        os.remove(f)


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

        # Save original (unwarped) activity image
        img_activity_32bit = img_gray.astype(np.float32)
        imwrite(
            f"{activity_dir}/frame_{num}.tif",
            img_activity_32bit,
            photometric='minisblack',
            dtype=np.float32
        )

        # Apply sinusoidal warping to the grayscale image
        img_warped = apply_sinusoidal_warp(img_gray, it, n_frames, motion_intensity=0.015)

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

    print(f"\nGenerated {n_frames} warped motion frames in {motion_frames_dir}/")
    print(f"Frame format: 512x512, 32-bit float, single channel TIF")
    print(f"Applied sinusoidal warping with motion_intensity=0.015")

    # Train NSTM on the generated frames
    nstm_output_dir = f'./graphs_data/{dataset_name}/NSTM_outputs'
    deformation_net, anatomy_net, loss_history = train_nstm(
        motion_frames_dir=motion_frames_dir,
        activity_dir=activity_dir,
        n_frames=n_frames,
        res=512,
        device=device,
        output_dir=nstm_output_dir,
        num_training_steps=5000
    )

    # Load motion and activity images for video creation
    print("\nLoading images for video creation...")
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
        anatomy_net=anatomy_net,
        activity_images=activity_images_list,
        motion_images=motion_images_list,
        data_min=data_min,
        data_max=data_max,
        res=512,
        device=device,
        output_dir=nstm_output_dir,
        num_frames=128  # Use all frames
    )
    print(f"4-panel video saved to: {video_path}")

