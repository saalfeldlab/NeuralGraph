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
import sys
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

# Import from graph_instant_NGP
from src.NeuralGraph.models.graph_instant_NGP import (
    train_siren,
    train_nstm,
    apply_sinusoidal_warp
)


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
        for i, frame in enumerate(tqdm(frames_list, desc="Creating 1-panel video")):
            # Ensure RGB
            if len(frame.shape) == 2:
                frame_rgb = np.stack([frame, frame, frame], axis=2)
            else:
                frame_rgb = frame

            # Ensure uint8
            if frame_rgb.dtype != np.uint8:
                frame_rgb = (np.clip(frame_rgb, 0, 1) * 255).astype(np.uint8)

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

        for i in tqdm(range(n_frames), desc="Creating 2-panel video"):
            left = left_frames[i]
            right = right_frames[i]

            # Ensure RGB and uint8
            left_rgb = _to_rgb_uint8(left)
            right_rgb = _to_rgb_uint8(right)

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

        for i in tqdm(range(n_frames), desc="Creating 8-panel video"):
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
    print(f"Video saved to: {output_path}")


def _to_rgb_uint8(frame):
    """Convert frame to RGB uint8 format"""
    # Ensure RGB
    if len(frame.shape) == 2:
        frame_rgb = np.stack([frame, frame, frame], axis=2)
    else:
        frame_rgb = frame

    # Ensure uint8
    if frame_rgb.dtype != np.uint8:
        # Normalize to [0, 1] then to [0, 255]
        frame_min = frame_rgb.min()
        frame_max = frame_rgb.max()
        if frame_max > frame_min:
            frame_norm = (frame_rgb - frame_min) / (frame_max - frame_min)
        else:
            frame_norm = np.zeros_like(frame_rgb)
        frame_rgb = (frame_norm * 255).astype(np.uint8)

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
    print("=" * 80)
    print("STAGE 1: Training SIREN on discrete neuron activities")
    print("=" * 80)

    siren_net = train_siren(
        x_list=x_list,
        device=device,
        output_dir=output_dir,
        num_training_steps=config.get('num_training_steps', 10000),
        nnr_f_T_period=config.get('nnr_f_T_period', 10),
        n_train_frames=config.get('n_train_frames', 256),
        n_neurons=config.get('n_neurons', 100)
    )

    print(f"✓ SIREN training complete. Check scatter plot in: {output_dir}")
    return siren_net


def stage2_generate_activity_images(x_list, neuron_positions, n_frames, res, device,
                                     output_dir, activity_dir, run=0):
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
        run: Run index (default 0)

    Returns:
        activity_images: List of generated activity images
    """
    print("=" * 80)
    print("STAGE 2: Generating activity images using matplotlib scatter")
    print("=" * 80)

    os.makedirs(activity_dir, exist_ok=True)

    activity_images = []

    # Convert neuron positions to tensor for plotting
    X1 = torch.tensor(neuron_positions, dtype=torch.float32, device=device)

    print(f"Rendering {n_frames} activity images...")
    for frame_idx in tqdm(range(n_frames)):
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
        activity_images.append(img_activity_32bit)

        # Save as TIFF (float32, matching original)
        imwrite(
            f"{activity_dir}/frame_{frame_idx:06d}.tif",
            img_activity_32bit,
            photometric='minisblack',
            dtype=np.float32
        )

        plt.close(fig)

    print(f"✓ Activity images saved to: {activity_dir}")

    # Create 1-panel MP4 video
    print("Creating activity video...")
    video_path = f"{output_dir}/stage2_activity.mp4"
    create_video(
        frames_list=activity_images,
        output_path=video_path,
        fps=30,
        layout='1panel',
        panel_titles=['SIREN Activity']
    )

    return activity_images


def stage3_generate_warped_motion_frames(activity_images, boat_fixed_scene, n_frames,
                                         output_dir, motion_frames_dir):
    """
    Stage 3: Generate warped motion frames (activity × boat + sinusoidal warp)

    Args:
        activity_images: List of activity images
        boat_fixed_scene: Boat anatomy/fixed scene image
        n_frames: Number of frames
        output_dir: Directory for outputs
        motion_frames_dir: Directory to save motion frames

    Returns:
        motion_images: List of warped motion frames
    """
    print("=" * 80)
    print("STAGE 3: Generating warped motion frames")
    print("=" * 80)

    os.makedirs(motion_frames_dir, exist_ok=True)

    motion_images = []

    print(f"Applying boat mask and warping {n_frames} frames...")
    for frame_idx in tqdm(range(n_frames)):
        activity_frame = activity_images[frame_idx]

        # Element-wise multiplication: activity × boat_fixed_scene
        img_with_fixed_scene = activity_frame * boat_fixed_scene

        # Apply sinusoidal warping
        img_warped = apply_sinusoidal_warp(img_with_fixed_scene, frame_idx, n_frames,
                                           motion_intensity=0.015)

        motion_images.append(img_warped)

        # Save as TIFF
        imwrite(f"{motion_frames_dir}/frame_{frame_idx:06d}.tif", img_warped.astype(np.float32))

    print(f"✓ Motion frames saved to: {motion_frames_dir}")

    # Create 2-panel MP4 video (activity | target)
    print("Creating 2-panel video (activity | warped target)...")
    video_path = f"{output_dir}/stage3_activity_and_warped.mp4"
    create_video(
        frames_list={'left': activity_images, 'right': motion_images},
        output_path=video_path,
        fps=30,
        layout='2panel',
        panel_titles=['Activity', 'Warped Target']
    )

    return motion_images


def stage4_train_nstm(motion_frames_dir, activity_dir, n_frames, res, device, output_dir,
                      siren_config, pretrained_activity_net, x_list):
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

    Returns:
        Trained networks and loss history
    """
    print("=" * 80)
    print("STAGE 4: Training NSTM (deformation + fixed_scene)")
    print("=" * 80)

    deformation_net, fixed_scene_net, activity_net, loss_history = train_nstm(
        motion_frames_dir=motion_frames_dir,
        activity_dir=activity_dir,
        n_frames=n_frames,
        res=res,
        device=device,
        output_dir=output_dir,
        num_training_steps=3000,
        siren_config=siren_config,
        pretrained_activity_net=pretrained_activity_net,
        x_list=x_list,
        use_siren=False,  # Use ground truth activity images, not SIREN
        siren_lr=1e-4,
        nstm_lr=5e-4,
        siren_loss_weight=1.0
    )

    print(f"✓ NSTM training complete. Check outputs in: {output_dir}")
    print(f"✓ 8-panel video created during training")

    return deformation_net, fixed_scene_net, activity_net, loss_history


def main(config=None, device=None):
    """
    Main pipeline: Sequential execution of 4 stages

    Args:
        config: NeuralGraphConfig object (optional)
        device: torch device (optional)
    """
    print("\n" + "=" * 80)
    print("NGP TRAINER - 4-Stage Sequential Pipeline")
    print("=" * 80 + "\n")

    # ========== Configuration ==========
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Paths
    if config is not None:
        dataset_name = config.dataset  # Use config.dataset, not config.config_file
        config_file = config.config_file
    else:
        dataset_name = "signal_N11_2_1"
        config_file = "signal_N11_2_1_1"
    base_dir = f"/groups/saalfeld/home/allierc/Py/NeuralGraph/log/{config_file}"
    output_dir = f"{base_dir}/nstm_output"
    activity_dir = f"{base_dir}/activity"
    motion_frames_dir = f"{base_dir}/motion_frames"

    os.makedirs(output_dir, exist_ok=True)

    # Parameters
    n_frames = 256
    res = 512

    # Load neuron data (discrete activities)
    print("Loading neuron data...")
    x_list = []
    y_list = []
    n_runs = 1
    for run in range(n_runs):
        x = np.load(f'graphs_data/{dataset_name}/x_list_{run}.npy')
        y = np.load(f'graphs_data/{dataset_name}/y_list_{run}.npy')
        x_list.append(x)
        y_list.append(y)

    # Get number of neurons from data
    run = 0
    x = x_list[0][n_frames - 10]
    n_neurons = x.shape[0]
    print(f"Loaded data: {n_neurons} neurons, {n_frames} frames")

    # SIREN config (after we know n_neurons)
    siren_config = {
        'num_training_steps': 10000,
        'nnr_f_T_period': 10,
        'n_train_frames': n_frames,
        'n_neurons': n_neurons
    }

    # Extract neuron positions (use first frame)
    neuron_positions = x_list[0][0][:, 1:3]  # (n_neurons, 2)

    # Load boat fixed scene
    print("Loading boat fixed scene...")
    boat_fixed_scene_path = "/workspace/src/NeuralGraph/models/pics_boat_512.tif"
    if os.path.exists(boat_fixed_scene_path):
        boat_fixed_scene = imread(boat_fixed_scene_path).astype(np.float32)
        # Normalize to [0, 1] to match sigmoid output of fixed_scene_net
        boat_min = boat_fixed_scene.min()
        boat_max = boat_fixed_scene.max()
        if boat_max > boat_min:
            boat_fixed_scene = (boat_fixed_scene - boat_min) / (boat_max - boat_min)
        print(f"Boat fixed scene normalized to [0, 1]")
    else:
        print(f"Warning: Boat image not found at {boat_fixed_scene_path}, using uniform mask")
        boat_fixed_scene = np.ones((res, res), dtype=np.float32)

    print(f"Boat fixed scene shape: {boat_fixed_scene.shape}, range: [{boat_fixed_scene.min():.4f}, {boat_fixed_scene.max():.4f}]\n")

    # Set matplotlib style for black background
    plt.style.use("dark_background")
    matplotlib.rcParams["savefig.pad_inches"] = 0

    # Clear existing files in activity and motion_frames directories
    print("Clearing existing activity and motion frames...")
    import glob
    for f in glob.glob(f'{motion_frames_dir}/*'):
        os.remove(f)
    for f in glob.glob(f'{activity_dir}/*'):
        os.remove(f)

    # ========== Stage 1: Train SIREN ==========
    siren_net = stage1_train_siren(x_list, device, output_dir, siren_config)

    # ========== Stage 2: Generate Activity Images ==========
    activity_images = stage2_generate_activity_images(
        x_list=x_list,
        neuron_positions=neuron_positions,
        n_frames=n_frames,
        res=res,
        device=device,
        output_dir=output_dir,
        activity_dir=activity_dir,
        run=0
    )

    # ========== Stage 3: Generate Warped Motion Frames ==========
    motion_images = stage3_generate_warped_motion_frames(
        activity_images=activity_images,
        boat_fixed_scene=boat_fixed_scene,
        n_frames=n_frames,
        output_dir=output_dir,
        motion_frames_dir=motion_frames_dir
    )

    # ========== Stage 4: Train NSTM ==========
    deformation_net, fixed_scene_net, activity_net, loss_history = stage4_train_nstm(
        motion_frames_dir=motion_frames_dir,
        activity_dir=activity_dir,
        n_frames=n_frames,
        res=res,
        device=device,
        output_dir=output_dir,
        siren_config=siren_config,
        pretrained_activity_net=None,  # Don't use SIREN in NSTM training
        x_list=None
    )

    # ========== Pipeline Complete ==========
    print("\n" + "=" * 80)
    print("✓ PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\nOutputs saved to: {output_dir}")
    print("\nGenerated videos:")
    print(f"  - Stage 1: SIREN scatter plot (PNG)")
    print(f"  - Stage 2: {output_dir}/stage2_activity.mp4 (1-panel)")
    print(f"  - Stage 3: {output_dir}/stage3_activity_and_warped.mp4 (2-panel)")
    print(f"  - Stage 4: {output_dir}/quad_panel_video.mp4 (8-panel)")
    print(f"\nActivity images: {activity_dir}/")
    print(f"Motion frames: {motion_frames_dir}/")
    print()


if __name__ == "__main__":
    main()
