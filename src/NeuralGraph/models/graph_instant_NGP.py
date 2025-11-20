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
from skimage.metrics import structural_similarity as ssim
import cv2
import subprocess
import shutil
from NeuralGraph.models.Siren_Network import Siren

# from fa2_modified import ForceAtlas2
# import h5py as h5
# import zarr
# import xarray as xr
import torch_geometric as pyg

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


def pretrain_siren_on_particles(x_list, n_frames, res, device, output_dir, siren_config, num_training_steps=5000):
    """Pre-train SIREN activity network on discrete neuron data

    Args:
        x_list: List of arrays with shape (n_frames, n_neurons, features)
                where x[:, :, :2] are positions and x[:, :, 6] is activity
        n_frames: Number of frames to use
        res: Resolution for visualization (512)
        device: torch device
        output_dir: Directory to save outputs
        siren_config: SIREN configuration dict
        num_training_steps: Number of training steps

    Returns:
        activity_net: Pre-trained SIREN network
    """
    print("pre-training siren activity network on discrete neuron data...")

    # Extract neuron positions and activities
    x_data = x_list[0][:n_frames]  # Use first run
    n_neurons = x_data.shape[1]

    # Extract positions (x, y) and activity
    positions = x_data[:, :, :2]  # (n_frames, n_neurons, 2)
    activities = x_data[:, :, 6:7]  # (n_frames, n_neurons, 1)

    print(f"neuron data: {n_frames} frames, {n_neurons} neurons")
    print(f"position range: x=[{positions[:, :, 0].min():.3f}, {positions[:, :, 0].max():.3f}], y=[{positions[:, :, 1].min():.3f}, {positions[:, :, 1].max():.3f}]")
    print(f"activity range: [{activities.min():.3f}, {activities.max():.3f}]")

    # Normalize positions to [0, 1]
    pos_min = positions.min(axis=(0, 1))
    pos_max = positions.max(axis=(0, 1))
    positions_norm = (positions - pos_min) / (pos_max - pos_min + 1e-8)

    # Normalize activities to [0, 1]
    act_min = activities.min()
    act_max = activities.max()
    activities_norm = (activities - act_min) / (act_max - act_min + 1e-8)

    # Convert to torch tensors
    positions_torch = torch.tensor(positions_norm, dtype=torch.float32, device=device)
    activities_torch = torch.tensor(activities_norm, dtype=torch.float32, device=device)

    # Create SIREN network
    activity_net = Siren(
        in_features=3,  # x, y, t
        hidden_features=siren_config['hidden_dim_nnr_f'],
        hidden_layers=siren_config['n_layers_nnr_f'],
        out_features=1,  # scalar activity
        outermost_linear=siren_config['outermost_linear_nnr_f'],
        first_omega_0=siren_config['omega_f'],
        hidden_omega_0=siren_config['omega_f']
    ).to(device)

    # Period normalization factors
    xy_period = siren_config['nnr_f_xy_period']
    t_period = siren_config['nnr_f_T_period']

    # Optimizer
    optimizer = torch.optim.Adam(activity_net.parameters(), lr=1e-4)

    # Training loop
    print(f"training siren on neurons for {num_training_steps} steps...")
    loss_history = []

    pbar = trange(num_training_steps, ncols=100)
    for step in pbar:
        # Sample random frame
        t_idx = np.random.randint(0, n_frames)
        t_normalized = t_idx / (n_frames - 1)

        # Get neuron data for this frame
        pos_frame = positions_torch[t_idx]  # (n_neurons, 2)
        act_frame = activities_torch[t_idx]  # (n_neurons, 1)

        # Create 3D coordinates with period normalization
        t_tensor = torch.full((n_neurons, 1), t_normalized / t_period, device=device, dtype=torch.float32)
        coords_3d = torch.cat([pos_frame / xy_period, t_tensor], dim=1)

        # Forward pass
        predicted_activity = activity_net(coords_3d)

        # Loss
        loss = torch.nn.functional.mse_loss(predicted_activity, act_frame)

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if step % 100 == 0:
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    print("siren pre-training complete!")

    # Generate visualization video
    print("generating siren visualization video...")
    video_dir = f"{output_dir}/siren_pretrain"
    os.makedirs(video_dir, exist_ok=True)

    # Create coordinate grid for visualization
    y_coords = torch.linspace(0, 1, res, device=device, dtype=torch.float32)
    x_coords = torch.linspace(0, 1, res, device=device, dtype=torch.float32)
    yv, xv = torch.meshgrid(y_coords, x_coords, indexing='ij')
    coords_2d = torch.stack([xv.flatten(), yv.flatten()], dim=1)

    # Generate frames
    temp_dir = f"{video_dir}/frames"
    os.makedirs(temp_dir, exist_ok=True)

    activity_net.eval()
    with torch.no_grad():
        for frame_idx in trange(n_frames, desc="Rendering frames", ncols=100):
            t_norm = frame_idx / (n_frames - 1)

            # Create 3D coordinates
            t_tensor = torch.full((res*res, 1), t_norm / t_period, device=device, dtype=torch.float32)
            coords_3d = torch.cat([coords_2d / xy_period, t_tensor], dim=1)

            # Evaluate SIREN
            activity_pred = activity_net(coords_3d)
            activity_img = activity_pred.reshape(res, res).cpu().numpy()

            # Normalize to [0, 1] for visualization
            activity_vis = (activity_img - activity_img.min()) / (activity_img.max() - activity_img.min() + 1e-8)
            activity_uint8 = (np.clip(activity_vis, 0, 1) * 255).astype(np.uint8)

            # Save frame
            cv2.imwrite(f"{temp_dir}/frame_{frame_idx:06d}.png", activity_uint8)

    # Create video with ffmpeg
    video_path = f'{video_dir}/siren_pretrained.mp4'
    fps = 30

    print("creating video from frames...")
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", f"{temp_dir}/frame_%06d.png",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "slow", "-crf", "22",
        video_path
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"siren visualization video saved to {video_path}")
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"error creating video with ffmpeg: {e}")

    # Clean up frames
    shutil.rmtree(temp_dir)

    # Plot loss history
    plt.figure(figsize=(10, 4))
    plt.plot(loss_history)
    plt.title('SIREN Pre-training Loss', fontsize=12)
    plt.xlabel('Step')
    plt.ylabel('MSE Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{video_dir}/pretrain_loss.png', dpi=150)
    plt.close()

    activity_net.train()
    return activity_net


def train_nstm(motion_frames_dir, activity_dir, n_frames, res, device, output_dir, num_training_steps=3000,
               siren_config=None, pretrained_activity_net=None, x_list=None,
               use_siren=True, siren_lr=1e-4, nstm_lr=5e-4, siren_loss_weight=1.0):
    """Train Neural Space-Time Model with anatomy + activity decomposition

    Args:
        siren_config: Dict with keys: hidden_dim_nnr_f, n_layers_nnr_f, omega_f,
                      nnr_f_xy_period, nnr_f_T_period, outermost_linear_nnr_f
        x_list: List of neuron data arrays for SIREN supervision (n_frames, n_neurons, features)
        use_siren: Boolean to use SIREN network for activity (True) or grid_sample (False)
        siren_lr: Learning rate for SIREN network
        nstm_lr: Learning rate for deformation and anatomy networks
        siren_loss_weight: Weight for SIREN supervision loss on discrete neurons
    """
    # Load motion frames
    print("loading motion frames...")
    motion_images = []
    for i in range(n_frames):
        img = imread(f"{motion_frames_dir}/frame_{i:06d}.tif")
        motion_images.append(img)

    # Load original activity images
    print("loading original activity images...")
    activity_images = []
    for i in range(n_frames):
        img = imread(f"{activity_dir}/frame_{i:06d}.tif")
        activity_images.append(img)

    # Compute normalization statistics
    all_pixels = np.concatenate([img.flatten() for img in motion_images])
    data_min = all_pixels.min()
    data_max = all_pixels.max()
    print(f"input data range: [{data_min:.2f}, {data_max:.2f}]")

    # Normalize to [0, 1] and convert to tensors (use float32 for SIREN compatibility)
    # Keep both motion (warped) and activity (ground-truth) in GPU memory
    motion_tensors = [torch.tensor((img - data_min) / (data_max - data_min), dtype=torch.float32).to(device) for img in motion_images]
    activity_tensors = [torch.tensor((img - data_min) / (data_max - data_min), dtype=torch.float32).to(device) for img in activity_images]

    print(f"loaded {n_frames} frames into gpu memory")

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

    # Anatomy network: (x, y) -> raw mask value
    anatomy_net = tcnn.NetworkWithInputEncoding(
        n_input_dims=2,  # x, y
        n_output_dims=1,  # mask value
        encoding_config=config["encoding"],
        network_config=config["network"]
    ).to(device)

    # Activity network: SIREN (x, y, t) -> activity value or grid_sample
    if siren_config is not None and use_siren:
        if pretrained_activity_net is not None:
            print("using pre-trained siren activity network...")
            activity_net = pretrained_activity_net
        else:
            print("creating siren activity network...")
            activity_net = Siren(
                in_features=3,  # x, y, t
                hidden_features=siren_config['hidden_dim_nnr_f'],
                hidden_layers=siren_config['n_layers_nnr_f'],
                out_features=1,  # scalar activity
                outermost_linear=siren_config['outermost_linear_nnr_f'],
                first_omega_0=siren_config['omega_f'],
                hidden_omega_0=siren_config['omega_f']
            ).to(device)

        # Period normalization factors
        xy_period = siren_config['nnr_f_xy_period']
        t_period = siren_config['nnr_f_T_period']
        print(f"siren activity network: xy_period={xy_period}, t_period={t_period}, omega={siren_config['omega_f']}")
    else:
        activity_net = None
        xy_period = 1.0
        t_period = 1.0
        if not use_siren:
            print("using grid_sample for activity (use_siren=False)")

    # Create coordinate grid (use float16 for tinycudann, float32 for coords)
    y_coords = torch.linspace(0, 1, res, device=device, dtype=torch.float32)
    x_coords = torch.linspace(0, 1, res, device=device, dtype=torch.float32)
    yv, xv = torch.meshgrid(y_coords, x_coords, indexing='ij')
    coords_2d = torch.stack([xv.flatten(), yv.flatten()], dim=1)

    # Convert to float16 for tinycudann networks
    coords_2d_f16 = coords_2d.to(torch.float16)

    # Create separate optimizers with different learning rates
    nstm_params = list(deformation_net.parameters()) + list(anatomy_net.parameters())
    optimizer_nstm = torch.optim.Adam(nstm_params, lr=nstm_lr)

    if activity_net is not None:
        optimizer_siren = torch.optim.Adam(activity_net.parameters(), lr=siren_lr)
    else:
        optimizer_siren = None

    # Learning rate schedules
    scheduler_nstm = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_nstm,
        milestones=[num_training_steps // 2, num_training_steps * 3 // 4],
        gamma=0.1
    )

    if optimizer_siren is not None:
        scheduler_siren = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_siren,
            milestones=[num_training_steps // 2, num_training_steps * 3 // 4],
            gamma=0.1
        )

    # Prepare neuron data for SIREN supervision (if available)
    neuron_data = None
    if activity_net is not None and x_list is not None:
        print("preparing neuron data for siren supervision...")
        x_data = x_list[0][:n_frames]  # Use first run
        n_neurons = x_data.shape[1]

        # Extract positions (x, y) and activity
        positions = x_data[:, :, :2]  # (n_frames, n_neurons, 2)
        activities = x_data[:, :, 6:7]  # (n_frames, n_neurons, 1)

        # Normalize positions to [0, 1]
        pos_min = positions.min(axis=(0, 1))
        pos_max = positions.max(axis=(0, 1))
        positions_norm = (positions - pos_min) / (pos_max - pos_min + 1e-8)

        # Normalize activities to [0, 1]
        act_min = activities.min()
        act_max = activities.max()
        activities_norm = (activities - act_min) / (act_max - act_min + 1e-8)

        # Convert to torch tensors
        neuron_data = {
            'positions': torch.tensor(positions_norm, dtype=torch.float32, device=device),
            'activities': torch.tensor(activities_norm, dtype=torch.float32, device=device),
            'n_neurons': n_neurons
        }
        print(f"neuron data: {n_frames} frames, {n_neurons} neurons")

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

        # Sample anatomy mask at source coordinates (convert to float16 for tcnn)
        anatomy_mask = torch.sigmoid(anatomy_net(source_coords.to(torch.float16))).to(torch.float32)

        # Sample activity at source coordinates
        if activity_net is not None:
            # Use SIREN network with period-based normalization
            # Create 3D coordinates: (x/xy_period, y/xy_period, t/t_period)
            source_coords_3d = torch.cat([
                source_coords / xy_period,
                torch.full_like(source_coords[:, 0:1], t_normalized / t_period)
            ], dim=1)
            sampled_activity = activity_net(source_coords_3d)
        else:
            # Fallback: use grid_sample on ground-truth activity
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

        # Reconstruction: anatomy × activity
        reconstructed = anatomy_mask * sampled_activity

        # Main reconstruction loss
        recon_loss = torch.nn.functional.mse_loss(reconstructed, target)

        # Activity supervision loss using discrete neuron data (if using SIREN)
        siren_loss = torch.tensor(0.0, device=device)
        if activity_net is not None and neuron_data is not None:
            # Get neuron positions and activities for this frame
            pos_frame = neuron_data['positions'][t_idx]  # (n_neurons, 2)
            act_frame = neuron_data['activities'][t_idx]  # (n_neurons, 1)

            # Create 3D coordinates with period normalization for neurons
            t_tensor = torch.full((neuron_data['n_neurons'], 1), t_normalized / t_period,
                                 device=device, dtype=torch.float32)
            neuron_coords_3d = torch.cat([pos_frame / xy_period, t_tensor], dim=1)

            # Forward pass through SIREN
            predicted_activity = activity_net(neuron_coords_3d)

            # Compute loss on discrete neurons
            siren_loss = torch.nn.functional.mse_loss(predicted_activity, act_frame)

        # Regularization: Deformation smoothness
        deformation_smoothness = torch.mean(torch.abs(deformation))

        # Combined loss with regularization
        total_loss_nstm = recon_loss + 0.01 * deformation_smoothness

        # Optimize NSTM networks (deformation + anatomy)
        optimizer_nstm.zero_grad()
        if optimizer_siren is not None and neuron_data is not None:
            # Combined loss for backward pass
            total_loss = total_loss_nstm + siren_loss_weight * siren_loss
            total_loss.backward()
            optimizer_nstm.step()
            optimizer_siren.step()
        else:
            total_loss_nstm.backward()
            optimizer_nstm.step()

        # Update learning rates
        scheduler_nstm.step()
        if optimizer_siren is not None:
            scheduler_siren.step()

        # Record losses
        loss_history.append(recon_loss.item())
        regularization_history['deformation'].append(deformation_smoothness.item())

        # Update progress bar with losses
        if activity_net is not None and neuron_data is not None:
            pbar.set_postfix({
                'recon': f'{recon_loss.item():.6f}',
                'def': f'{deformation_smoothness.item():.4f}',
                'siren': f'{siren_loss.item():.6f}'
            })
        else:
            pbar.set_postfix({
                'recon': f'{recon_loss.item():.6f}',
                'def': f'{deformation_smoothness.item():.4f}'
            })

    print("training complete!")

    # Extract learned anatomy mask
    print("extracting learned anatomy mask...")
    with torch.no_grad():
        anatomy_mask = torch.sigmoid(anatomy_net(coords_2d))
        anatomy_mask_img = anatomy_mask.reshape(res, res).cpu().numpy()

    # Compute average of original activity images
    activity_average = np.mean(activity_images, axis=0)

    # Create ground-truth mask from activity average (threshold at mean)
    threshold = activity_average.mean()
    ground_truth_mask = (activity_average > threshold).astype(np.float32)
    n_gt_pixels = ground_truth_mask.sum()

    # Normalize anatomy mask for comparison
    anatomy_mask_norm = (anatomy_mask_img - anatomy_mask_img.min()) / (anatomy_mask_img.max() - anatomy_mask_img.min() + 1e-8)

    # Threshold anatomy to match ground truth coverage (top N pixels)
    flat_anatomy = anatomy_mask_norm.flatten()
    n_pixels_to_select = int(n_gt_pixels)
    sorted_indices = np.argsort(flat_anatomy)[::-1]  # Sort descending
    anatomy_binary_flat = np.zeros_like(flat_anatomy, dtype=np.float32)
    anatomy_binary_flat[sorted_indices[:n_pixels_to_select]] = 1.0
    anatomy_binary = anatomy_binary_flat.reshape(anatomy_mask_norm.shape)

    # Compute DICE score and IoU between learned anatomy and ground-truth mask
    intersection = np.sum(anatomy_binary * ground_truth_mask)
    union = np.sum(anatomy_binary) + np.sum(ground_truth_mask)
    dice_score = 2 * intersection / (union + 1e-8)
    iou_score = intersection / (union - intersection + 1e-8)

    # Compute median of motion frames (warped frames) as baseline
    motion_median = np.median(motion_images, axis=0)

    # Reconstruct fixed scene: anatomy × activity_average
    fixed_scene_denorm = anatomy_mask_img * activity_average

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
    print(f"anatomy: range=[{anatomy_mask_img.min():.3f}, {anatomy_mask_img.max():.3f}] | dice={dice_score:.3f} | iou={iou_score:.3f}")
    print(f"reconstruction: rmse={rmse_activity:.4f} | ssim={ssim_activity:.4f}")
    print(f"baseline:       rmse={rmse_baseline:.4f} | ssim={ssim_baseline:.4f}")
    print(f"improvement:    rmse={((rmse_baseline - rmse_activity) / rmse_baseline * 100):+.1f}% | ssim={((ssim_activity - ssim_baseline) / (1.0 - ssim_baseline) * 100):+.1f}%")

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)

    # Save only anatomy masks (not all intermediate images)
    imwrite(f'{output_dir}/anatomy_learned.tif', anatomy_mask_img.astype(np.float32))
    imwrite(f'{output_dir}/anatomy_binary.tif', anatomy_binary.astype(np.float32))

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

    # Save metrics to file
    from scipy import stats
    near_zero_count = np.sum(np.abs(anatomy_mask_img) < 0.01)
    sparsity = near_zero_count / anatomy_mask_img.size * 100

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

    return deformation_net, anatomy_net, activity_net, loss_history


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


def create_quad_panel_video(deformation_net, anatomy_net, activity_images, motion_images,
                            data_min, data_max, res, device, output_dir, num_frames=90, boat_anatomy=None):
    """Create an 8-panel comparison video (4 columns x 2 rows)

    Top row (Training Data):
    - Col 1: Activity
    - Col 2: Activity × Boat anatomy
    - Col 3: Ground truth motion field (arrows on black)
    - Col 4: Target (warped motion frames)

    Bottom row (Learned):
    - Col 1: Learned anatomy
    - Col 2: Anatomy × Activity
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

    # Extract anatomy mask once
    with torch.no_grad():
        anatomy_mask = torch.sigmoid(anatomy_net(coords_2d))
        anatomy_mask_img = anatomy_mask.reshape(res, res).cpu().numpy()

    # Normalize anatomy for visualization
    anatomy_norm = (anatomy_mask_img - anatomy_mask_img.min()) / (anatomy_mask_img.max() - anatomy_mask_img.min() + 1e-8)

    # Prepare boat anatomy for visualization (if provided)
    if boat_anatomy is not None:
        boat_norm = (boat_anatomy - boat_anatomy.min()) / (boat_anatomy.max() - boat_anatomy.min() + 1e-8)
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

        # Top-2: Activity × Boat anatomy (before warping)
        if boat_anatomy is not None:
            activity_times_boat = activity_frame * boat_anatomy
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

            # Sample anatomy at source coords
            anatomy_at_source = torch.sigmoid(anatomy_net(source_coords))

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
            sampled_activity_img = sampled_activity.reshape(res, res).cpu().numpy()

        # Bottom-1: Learned anatomy
        anatomy_uint8 = (np.clip(anatomy_norm, 0, 1) * 255).astype(np.uint8)
        learned_anatomy_rgb = np.stack([anatomy_uint8, anatomy_uint8, anatomy_uint8], axis=2)

        # Bottom-2: Anatomy × Activity
        anatomy_times_activity = anatomy_mask_img * activity_frame
        anat_act_norm = (anatomy_times_activity - anatomy_times_activity.min()) / (anatomy_times_activity.max() - anatomy_times_activity.min() + 1e-8)
        anat_act_uint8 = (np.clip(anat_act_norm, 0, 1) * 255).astype(np.uint8)
        anatomy_activity_rgb = np.stack([anat_act_uint8, anat_act_uint8, anat_act_uint8], axis=2)

        # Bottom-3: Learned motion field (arrows on black background)
        deformation_2d = deformation.reshape(res, res, 2).cpu().numpy()
        motion_x = deformation_2d[:, :, 0]
        motion_y = deformation_2d[:, :, 1]
        learned_motion_vis = create_motion_field_visualization(anatomy_norm, motion_x, motion_y, res, step_size=16, black_background=True)

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
        cv2.putText(learned_anatomy_rgb, "Learned Fixed Scene", (margin, y_pos), font, font_scale, (255, 255, 255), thickness)

        cv2.putText(anatomy_activity_rgb, "Fixed Scene x Activity", (margin, y_pos), font, font_scale, (255, 255, 255), thickness)

        cv2.putText(learned_motion_vis, "Learned Motion", (margin, y_pos), font, font_scale, (255, 255, 255), thickness)

        cv2.putText(recon_rgb, "Reconstruction", (margin, y_pos), font, font_scale, (255, 255, 255), thickness)

        # Create 4x2 grid layout
        top_row = np.hstack([activity_rgb, activity_boat_rgb, gt_motion_vis, target_rgb])
        bottom_row = np.hstack([learned_anatomy_rgb, anatomy_activity_rgb, learned_motion_vis, recon_rgb])
        combined = np.vstack([top_row, bottom_row])

        # Save frame
        cv2.imwrite(f"{temp_dir}/frame_{i:04d}.png", combined)

    # Create video with ffmpeg
    video_path = f'{output_dir}/nstm_8panel.mp4'
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

    # Load boat anatomy image
    import os as os_module
    current_dir = os_module.path.dirname(os_module.path.abspath(__file__))
    boat_anatomy_path = os_module.path.join(current_dir, 'pics_boat_512.tif')
    print(f"loading boat anatomy from {boat_anatomy_path}")
    boat_anatomy = imread(boat_anatomy_path).astype(np.float32)
    print(f"boat anatomy shape: {boat_anatomy.shape}, range: [{boat_anatomy.min():.4f}, {boat_anatomy.max():.4f}]")


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

        # Save activity image: dots at FIXED initial position with changing activity (x[:,6])
        # Activity changes over time, so save each frame
        img_activity_32bit = img_gray.astype(np.float32)
        imwrite(
            f"{activity_dir}/frame_{num}.tif",
            img_activity_32bit,
            photometric='minisblack',
            dtype=np.float32
        )

        # For motion frames: multiply activity by boat anatomy, then warp
        # Element-wise multiplication: activity × boat_anatomy
        img_with_anatomy = img_gray * boat_anatomy

        # Apply sinusoidal warping to the combined image
        # The warped result contains both activity and anatomy information
        img_warped = apply_sinusoidal_warp(img_with_anatomy, it, n_frames, motion_intensity=0.015)

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
    print(f"frame format: 512x512, 32-bit float, single channel tif")
    print(f"applied sinusoidal warping with motion_intensity=0.015")

    # Train NSTM on the generated frames
    nstm_output_dir = f'./graphs_data/{dataset_name}/NSTM_outputs'

    # Prepare SIREN config if available in model_config
    siren_config = None
    if hasattr(model_config, 'hidden_dim_nnr_f'):
        siren_config = {
            'hidden_dim_nnr_f': model_config.hidden_dim_nnr_f,
            'n_layers_nnr_f': model_config.n_layers_nnr_f,
            'omega_f': model_config.omega_f,
            'nnr_f_xy_period': model_config.nnr_f_xy_period,
            'nnr_f_T_period': model_config.nnr_f_T_period,
            'outermost_linear_nnr_f': model_config.outermost_linear_nnr_f
        }
        print(f"using siren activity network with config: {siren_config}")
    else:
        print("no siren config found, using grid_sample for activity")

    # Hardcoded parameters
    use_siren = True  # Set to False to use grid_sample instead of SIREN
    siren_loss_weight = 1.0  # Weight for SIREN supervision loss on discrete neurons
    nstm_lr = 5e-4  # Learning rate for deformation and anatomy networks

    # Get SIREN learning rate from config if available
    siren_lr = 1e-4  # Default
    if hasattr(training_config, 'learning_rate_NNR_f'):
        siren_lr = training_config.learning_rate_NNR_f

    # Pre-train SIREN on neuron data if config available and use_siren=True
    pretrained_activity_net = None
    if siren_config is not None and use_siren:
        print("pre-training siren network on discrete neuron data...")
        pretrained_activity_net = pretrain_siren_on_particles(
            x_list=x_list,
            n_frames=n_frames,
            res=512,
            device=device,
            output_dir=nstm_output_dir,
            siren_config=siren_config,
            num_training_steps=5000
        )

    deformation_net, anatomy_net, activity_net, loss_history = train_nstm(
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
        anatomy_net=anatomy_net,
        activity_images=activity_images_list,
        motion_images=motion_images_list,
        data_min=data_min,
        data_max=data_max,
        res=512,
        device=device,
        output_dir=nstm_output_dir,
        num_frames=256,  # Use all frames
        boat_anatomy=boat_anatomy  # Pass the boat anatomy for visualization
    )
    print(f"8-panel video (4x2) saved to: {video_path}")
    print("training completed.")

