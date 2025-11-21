"""
NGP_trainer.py - Sequential Pipeline for Neural Space-Time Model Training

This script implements a 4-stage pipeline for training instant-NGP based
Neural Space-Time Model (NSTM) for activity decomposition:

Stage 1: Train SIREN on discrete neuron activities (t -> [activity_1, ..., activity_n])
Stage 2: Generate activity images from SIREN using Gaussian splatting
Stage 3: Generate warped motion frames (activity × boat + sinusoidal warp)
Stage 4: Train NSTM (deformation + fixed_scene) on warped frames

Each stage produces visualizations and MP4 videos for validation.
"""

import os
import numpy as np
import torch
import cv2
import subprocess
import shutil
from tqdm import tqdm
from tifffile import imread, imwrite
import matplotlib
import matplotlib.pyplot as plt
from NeuralGraph.utils import to_numpy

# Functions train_siren, train_nstm, and apply_sinusoidal_warp are defined below


def apply_sinusoidal_warp(image, frame_idx, num_frames, motion_intensity=0.015):
    """Apply sinusoidal warping to an image, similar to pixel_NSTM.py"""
    from scipy.ndimage import map_coordinates

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
    from NeuralGraph.models.Siren_Network import Siren

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


def train_nstm(motion_frames_dir, activity_dir, n_frames, res, device, output_dir, num_training_steps=3000,
               siren_config=None, pretrained_activity_net=None, x_list=None,
               use_siren=True, siren_lr=1e-4, nstm_lr=5e-4, siren_loss_weight=1.0):
    """Train Neural Space-Time Model with fixed_scene + activity decomposition

    Note: This is a placeholder. The full implementation is very long.
    """
    raise NotImplementedError("train_nstm needs full implementation from original code")


def create_video(frames_list, output_path, fps=30, layout='1panel', panel_titles=None):
    """
    Universal video creation function for 1-panel, 2-panel, or 8-panel layouts

    Args:
        frames_list: List of numpy arrays or dict with keys for different panels
                     - For 1panel: single list of (H, W) or (H, W, 3) arrays
                     - For 2panel: dict like {'left': [...], 'right': [...]}
                     - For 8panel: dict like {'row0_col0': [...], 'row0_col1': [...], ...}
        output_path: Path to save MP4 file
        fps: Frames per second
        layout: '1panel', '2panel', or '8panel'
        panel_titles: List of titles for each panel (optional)
    """
    temp_dir = f"{os.path.dirname(output_path)}/temp_video_frames"
    os.makedirs(temp_dir, exist_ok=True)

    if layout == '1panel':
        n_frames = len(frames_list)

        # Compute global min/max across all frames for consistent normalization
        all_values = np.concatenate([frame.flatten() for frame in frames_list])
        global_min = all_values.min()
        global_max = all_values.max()

        for i, frame in enumerate(tqdm(frames_list, desc="creating video", ncols=100)):
            # Convert to RGB uint8 with global normalization
            frame_rgb = _to_rgb_uint8(frame, global_min, global_max)

            # Add title if provided
            if panel_titles and len(panel_titles) > 0:
                cv2.putText(frame_rgb, panel_titles[0], (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            cv2.imwrite(f"{temp_dir}/frame_{i:06d}.png", frame_rgb)

    elif layout == '2panel':
        # Assume frames_list is dict with 'left' and 'right' keys
        left_frames = frames_list['left']
        right_frames = frames_list['right']
        n_frames = len(left_frames)

        # Compute global min/max for left and right separately
        all_left = np.concatenate([frame.flatten() for frame in left_frames])
        all_right = np.concatenate([frame.flatten() for frame in right_frames])
        left_min, left_max = all_left.min(), all_left.max()
        right_min, right_max = all_right.min(), all_right.max()

        for i in tqdm(range(n_frames), desc="creating video", ncols=100):
            left = left_frames[i]
            right = right_frames[i]

            # Convert to RGB uint8 with global normalization per panel
            left_rgb = _to_rgb_uint8(left, left_min, left_max)
            right_rgb = _to_rgb_uint8(right, right_min, right_max)

            # Add titles if provided
            if panel_titles:
                cv2.putText(left_rgb, panel_titles[0], (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(right_rgb, panel_titles[1], (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            # Horizontal concatenation
            combined = np.hstack([left_rgb, right_rgb])
            cv2.imwrite(f"{temp_dir}/frame_{i:06d}.png", combined)

    elif layout == '8panel':
        # Assume frames_list is dict with grid structure
        # Expected keys: 'row0_col0', 'row0_col1', ..., 'row1_col3'
        n_frames = len(frames_list['row0_col0'])

        for i in tqdm(range(n_frames), desc="creating video", ncols=100):
            # Top row (4 panels)
            top_panels = []
            for col in range(4):
                key = f'row0_col{col}'
                panel = _to_rgb_uint8(frames_list[key][i])
                if panel_titles and col < len(panel_titles):
                    cv2.putText(panel, panel_titles[col], (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                top_panels.append(panel)

            # Bottom row (4 panels)
            bottom_panels = []
            for col in range(4):
                key = f'row1_col{col}'
                panel = _to_rgb_uint8(frames_list[key][i])
                if panel_titles and (4 + col) < len(panel_titles):
                    cv2.putText(panel, panel_titles[4 + col], (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                bottom_panels.append(panel)

            # Concatenate
            top_row = np.hstack(top_panels)
            bottom_row = np.hstack(bottom_panels)
            combined = np.vstack([top_row, bottom_row])

            cv2.imwrite(f"{temp_dir}/frame_{i:06d}.png", combined)

    else:
        raise ValueError(f"Unknown layout: {layout}")

    # Create video with ffmpeg
    ffmpeg_cmd = (
        f"ffmpeg -y -framerate {fps} -i {temp_dir}/frame_%06d.png "
        f"-c:v libx264 -pix_fmt yuv420p -crf 18 {output_path}"
    )
    subprocess.run(ffmpeg_cmd, shell=True, check=True, capture_output=True)

    # Clean up
    shutil.rmtree(temp_dir)


def _to_rgb_uint8(frame, global_min=None, global_max=None):
    """Convert frame to RGB uint8 format

    Args:
        frame: Input frame (grayscale or RGB)
        global_min: Global minimum value for normalization (if None, use frame min)
        global_max: Global maximum value for normalization (if None, use frame max)
    """
    # Ensure RGB
    if len(frame.shape) == 2:
        frame_rgb = np.stack([frame, frame, frame], axis=2)
    else:
        frame_rgb = frame

    # Ensure uint8
    if frame_rgb.dtype != np.uint8:
        # Use global min/max if provided, otherwise per-frame normalization
        frame_min = global_min if global_min is not None else frame_rgb.min()
        frame_max = global_max if global_max is not None else frame_rgb.max()

        if frame_max > frame_min:
            frame_norm = (frame_rgb - frame_min) / (frame_max - frame_min)
        else:
            frame_norm = np.zeros_like(frame_rgb)
        frame_rgb = (np.clip(frame_norm, 0, 1) * 255).astype(np.uint8)

    return frame_rgb


def stage1_train_siren(x_list, device, output_dir, config):
    """
    Stage 1: Train SIREN on discrete neuron activities

    Args:
        x_list: List of neuron data arrays (n_frames, n_neurons, features)
        device: torch device
        output_dir: Directory for outputs
        config: Configuration dict with SIREN parameters

    Returns:
        siren_net: Trained SIREN network
    """
    print("stage 1: training siren")

    siren_net = train_siren(
        x_list=x_list,
        device=device,
        output_dir=output_dir,
        num_training_steps=config.get('num_training_steps', 10000),
        nnr_f_T_period=config.get('nnr_f_T_period', 10),
        n_train_frames=config.get('n_train_frames', 256),
        n_neurons=config.get('n_neurons', 100)
    )

    return siren_net


def stage2_generate_activity_images(x_list, neuron_positions, n_frames, res, device,
                                     output_dir, activity_dir, activity_original_dir=None,
                                     threshold_activity=False, run=0):
    """
    Stage 2: Generate activity images using matplotlib scatter (matching original workflow)

    Args:
        x_list: List of neuron data arrays (n_frames, n_neurons, features)
        neuron_positions: (n_neurons, 2) array of positions in [-0.5, 0.5]
        n_frames: Number of frames to generate
        res: Image resolution
        device: torch device
        output_dir: Directory for outputs
        activity_dir: Directory to save activity images
        activity_original_dir: Directory to save original (pre-threshold) activity images
        threshold_activity: If True, threshold activity to binary values (0 or 50)
        run: Run index (default 0)

    Returns:
        activity_images: List of generated activity images
    """
    print("stage 2: generating activity images")
    if threshold_activity:
        print("activity will be thresholded to binary (0 or 50)")

    os.makedirs(activity_dir, exist_ok=True)
    if activity_original_dir is not None:
        os.makedirs(activity_original_dir, exist_ok=True)

    activity_images = []
    activity_images_original = [] if threshold_activity else None

    # Convert neuron positions to tensor for plotting
    X1 = torch.tensor(neuron_positions, dtype=torch.float32, device=device)
    for frame_idx in tqdm(range(n_frames), ncols=100):
        x = torch.tensor(x_list[run][frame_idx], dtype=torch.float32, device=device)

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
        img_activity_32bit = img_gray.astype(np.float32)

        # Save original before thresholding if requested
        if activity_original_dir is not None:
            imwrite(
                f"{activity_original_dir}/frame_{frame_idx:06d}.tif",
                img_activity_32bit,
                photometric='minisblack',
                dtype=np.float32
            )
            # Keep original for video display
            activity_images_original.append(img_activity_32bit.copy())

        # Apply thresholding if requested
        if threshold_activity:
            # Threshold: values > mean become 50, values <= mean become 0
            threshold_value = img_activity_32bit.mean()
            img_activity_32bit = np.where(img_activity_32bit > threshold_value, 50.0, 0.0).astype(np.float32)

        activity_images.append(img_activity_32bit)

        # Save as TIFF (float32, matching original)
        imwrite(
            f"{activity_dir}/frame_{frame_idx:06d}.tif",
            img_activity_32bit,
            photometric='minisblack',
            dtype=np.float32
        )

        plt.close(fig)

    video_path = f"{output_dir}/stage2_activity.mp4"
    create_video(
        frames_list=activity_images,
        output_path=video_path,
        fps=30,
        layout='1panel',
        panel_titles=['activity']
    )

    return activity_images, activity_images_original


def stage3_generate_warped_motion_frames(activity_images, boat_fixed_scene, n_frames,
                                         output_dir, motion_frames_dir, target_downsample_factor=1,
                                         motion_intensity=0.015):
    """
    Stage 3: Generate warped motion frames (activity × boat + sinusoidal warp)

    Args:
        activity_images: List of activity images
        boat_fixed_scene: Boat anatomy/fixed scene image
        n_frames: Number of frames
        output_dir: Directory for outputs
        motion_frames_dir: Directory to save motion frames
        target_downsample_factor: Downsample targets for super-resolution test
        motion_intensity: Sinusoidal warp intensity

    Returns:
        motion_images: List of warped motion frames
    """
    print("stage 3: generating motion frames")
    if target_downsample_factor > 1:
        print(f"targets will be downsampled by {target_downsample_factor}x for super-resolution test")

    os.makedirs(motion_frames_dir, exist_ok=True)

    motion_images = []
    from scipy.ndimage import zoom

    for frame_idx in tqdm(range(n_frames), ncols=100):
        activity_frame = activity_images[frame_idx]

        # Element-wise multiplication: activity × boat_fixed_scene
        img_with_fixed_scene = activity_frame * boat_fixed_scene

        # Apply sinusoidal warping
        img_warped = apply_sinusoidal_warp(img_with_fixed_scene, frame_idx, n_frames,
                                           motion_intensity=motion_intensity)

        # Downsample target if requested (for super-resolution test)
        if target_downsample_factor > 1:
            res = img_warped.shape[0]
            downsampled_size = res // target_downsample_factor
            # Downsample using nearest neighbor
            zoom_factor_down = downsampled_size / res
            img_downsampled = zoom(img_warped, zoom_factor_down, order=0)
            # Upsample back to original resolution
            zoom_factor_up = res / downsampled_size
            img_warped = zoom(img_downsampled, zoom_factor_up, order=0)

        motion_images.append(img_warped)

        # Save as TIFF
        imwrite(f"{motion_frames_dir}/frame_{frame_idx:06d}.tif", img_warped.astype(np.float32))

    print(f"activity: [{np.min([img.min() for img in activity_images]):.2f}, {np.max([img.max() for img in activity_images]):.2f}]")
    print(f"boat: [{boat_fixed_scene.min():.2f}, {boat_fixed_scene.max():.2f}]")
    print(f"motion: [{np.min([img.min() for img in motion_images]):.2f}, {np.max([img.max() for img in motion_images]):.2f}]")

    # Create 1-panel MP4 video (warped target only)
    video_path = f"{output_dir}/stage3_motion.mp4"
    create_video(
        frames_list=motion_images,
        output_path=video_path,
        fps=30,
        layout='1panel',
        panel_titles=['motion']
    )

    return motion_images


def stage4_train_nstm(motion_frames_dir, activity_dir, n_frames, res, device, output_dir,
                      siren_config, pretrained_activity_net, x_list, boat_fixed_scene=None,
                      boat_downsample_factor=1, activity_images_original=None):
    """
    Stage 4: Train NSTM (deformation + fixed_scene) on warped frames

    Args:
        motion_frames_dir: Directory with warped motion frames
        activity_dir: Directory with activity images
        n_frames: Number of frames
        res: Image resolution
        device: torch device
        output_dir: Directory for outputs
        siren_config: SIREN configuration dict
        pretrained_activity_net: Trained SIREN network (optional)
        x_list: Neuron data for SIREN supervision (optional)
        boat_fixed_scene: Boat fixed scene image (optional)
        boat_downsample_factor: Downsampling factor for boat image (optional)

    Returns:
        Trained networks and loss history
    """
    print("stage 4: training nstm")

    deformation_net, fixed_scene_net, activity_net, loss_history = train_nstm(
        motion_frames_dir=motion_frames_dir,
        activity_dir=activity_dir,
        n_frames=n_frames,
        res=res,
        device=device,
        output_dir=output_dir,
        num_training_steps=5000,
        siren_config=siren_config,
        pretrained_activity_net=pretrained_activity_net,
        x_list=x_list,
        use_siren=False,  # Use ground truth activity images, not SIREN
        siren_lr=1e-4,
        nstm_lr=5e-4,
        siren_loss_weight=1.0
    )

    # Create 8-panel video after training
    from NeuralGraph.models.graph_instant_NGP import create_quad_panel_video
    from tifffile import imread

    # Load all frames for video creation
    motion_images_list = []
    activity_images_list = []
    for i in range(n_frames):
        motion_images_list.append(imread(f"{motion_frames_dir}/frame_{i:06d}.tif"))
        activity_images_list.append(imread(f"{activity_dir}/frame_{i:06d}.tif"))

    # Compute data range for normalization
    all_pixels = np.concatenate([img.flatten() for img in motion_images_list])
    data_min = all_pixels.min()
    data_max = all_pixels.max()

    # Load original activity images if thresholding was used
    activity_images_original_list = None
    if activity_images_original is not None:
        activity_images_original_list = activity_images_original  # Use all frames

    # Create 8-panel comparison video
    video_path = create_quad_panel_video(
        deformation_net=deformation_net,
        fixed_scene_net=fixed_scene_net,
        activity_images=activity_images_list,
        motion_images=motion_images_list,
        data_min=data_min,
        data_max=data_max,
        res=res,
        device=device,
        output_dir=output_dir,
        num_frames=n_frames,
        boat_fixed_scene=boat_fixed_scene,
        boat_downsample_factor=boat_downsample_factor,
        activity_images_original=activity_images_original_list
    )
    print(f"stage4_nstm.mp4")

    return deformation_net, fixed_scene_net, activity_net, loss_history


def data_train_NGP(config=None, device=None):
    """
    Main pipeline: Sequential execution of 4 stages

    Args:
        config: NeuralGraphConfig object (optional)
        device: torch device (optional)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths
    if config is not None:
        dataset_name = config.dataset
    else:
        dataset_name = "signal_N11_2_1"
    base_dir = f"/groups/saalfeld/home/allierc/Py/NeuralGraph/log/{dataset_name}"
    output_dir = f"{base_dir}/nstm_output"
    activity_dir = f"{base_dir}/activity"
    activity_original_dir = f"{base_dir}/activity_original"  # For saving original before thresholding
    motion_frames_dir = f"{base_dir}/motion_frames"


    # Parameters
    n_frames = 512
    res = 512
    target_downsample_factor = 4  # Downsample motion frame targets for super-resolution test (1 = no downsampling, 4 = quarter resolution, etc.)
    motion_intensity = 0.05  # Sinusoidal warp intensity (higher = more motion, better sub-pixel sampling for super-resolution)
    threshold_activity = False  # If True, threshold activity to binary values (0 or 50)



    os.makedirs(output_dir, exist_ok=True)
    if threshold_activity:
        os.makedirs(activity_original_dir, exist_ok=True)



    # Load neuron data
    x_list = []
    y_list = []
    n_runs = 1
    for run in range(n_runs):
        x = np.load(f'graphs_data/{dataset_name}/x_list_{run}.npy')
        y = np.load(f'graphs_data/{dataset_name}/y_list_{run}.npy')
        x_list.append(x)
        y_list.append(y)

    run = 0
    x = x_list[0][n_frames - 10]
    n_neurons = x.shape[0]

    # SIREN config
    siren_config = {
        'num_training_steps': 10000,
        'nnr_f_T_period': 10,
        'n_train_frames': n_frames,
        'n_neurons': n_neurons
    }

    # Extract neuron positions
    neuron_positions = x_list[0][0][:, 1:3]

    # Load boat fixed scene
    import os as os_module
    from scipy.ndimage import zoom
    current_dir = os_module.path.dirname(os_module.path.abspath(__file__))
    boat_fixed_scene_path = os_module.path.join(current_dir, 'pics_boat_512.tif')

    if os.path.exists(boat_fixed_scene_path):
        boat_fixed_scene = imread(boat_fixed_scene_path).astype(np.float32)
        print(f"boat: loaded high-res")
    else:
        boat_fixed_scene = np.ones((res, res), dtype=np.float32)
        print(f"boat: using default")

    print(f"boat: [{boat_fixed_scene.min():.2f}, {boat_fixed_scene.max():.2f}]")

    # Set matplotlib style for black background
    plt.style.use("dark_background")
    matplotlib.rcParams["savefig.pad_inches"] = 0

    # Clear existing files
    import glob
    for f in glob.glob(f'{motion_frames_dir}/*'):
        os.remove(f)
    for f in glob.glob(f'{activity_dir}/*'):
        os.remove(f)

    # Stage 1: Train SIREN
    siren_net = stage1_train_siren(x_list, device, output_dir, siren_config)

    # Stage 2: Generate Activity Images
    activity_images, activity_images_original = stage2_generate_activity_images(
        x_list=x_list,
        neuron_positions=neuron_positions,
        n_frames=n_frames,
        res=res,
        device=device,
        output_dir=output_dir,
        activity_dir=activity_dir,
        activity_original_dir=activity_original_dir if threshold_activity else None,
        threshold_activity=threshold_activity,
        run=0
    )

    # Stage 3: Generate Warped Motion Frames
    # IMPORTANT: Use original (non-thresholded) activity for target generation
    # This creates a mismatch when training uses thresholded activity
    motion_input_activity = activity_images_original if activity_images_original is not None else activity_images
    motion_images = stage3_generate_warped_motion_frames(
        activity_images=motion_input_activity,
        boat_fixed_scene=boat_fixed_scene,
        n_frames=n_frames,
        output_dir=output_dir,
        motion_frames_dir=motion_frames_dir,
        target_downsample_factor=target_downsample_factor,
        motion_intensity=motion_intensity
    )

    # Stage 4: Train NSTM
    deformation_net, fixed_scene_net, activity_net, loss_history = stage4_train_nstm(
        motion_frames_dir=motion_frames_dir,
        activity_dir=activity_dir,
        n_frames=n_frames,
        res=res,
        device=device,
        output_dir=output_dir,
        siren_config=siren_config,
        pretrained_activity_net=None,
        x_list=None,
        boat_fixed_scene=boat_fixed_scene,
        boat_downsample_factor=1,  # No longer downsampling boat - it's high-res
        activity_images_original=activity_images_original
    )

if __name__ == "__main__":
    data_train_NGP()
