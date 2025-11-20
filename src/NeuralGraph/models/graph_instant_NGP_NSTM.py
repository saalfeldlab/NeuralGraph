import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import rc
from NeuralGraph.utils import to_numpy
from tifffile import imread, imwrite
from tqdm import trange
import os
from scipy.ndimage import map_coordinates
import tinycudann as tcnn
from skimage.metrics import structural_similarity as ssim

# from fa2_modified import ForceAtlas2
# import h5py as h5
# import zarr
# import xarray as xr


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
    """Train Neural Space-Time Model on motion frames"""
    print(f"\n{'='*60}")
    print("Starting NSTM Training")
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
    motion_tensors = [torch.tensor((img - data_min) / (data_max - data_min), dtype=torch.float16).to(device) for img in motion_images]

    # Create networks
    print("Creating networks...")
    config = create_network_config()

    motion_net = tcnn.NetworkWithInputEncoding(
        n_input_dims=3,  # x, y, t
        n_output_dims=2,  # δx, δy
        encoding_config=config["encoding"],
        network_config=config["network"]
    ).to(device)

    scene_net = tcnn.NetworkWithInputEncoding(
        n_input_dims=2,  # x, y
        n_output_dims=1,  # Intensity
        encoding_config=config["encoding"],
        network_config=config["network"]
    ).to(device)

    # Create model wrapper
    model = NeuralSpaceTimeModel(motion_net, scene_net, res, device)

    # Create optimizer
    params = list(motion_net.parameters()) + list(scene_net.parameters())
    optimizer = torch.optim.Adam(params, lr=5e-3)

    # Learning rate schedule
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[num_training_steps // 2, num_training_steps * 3 // 4],
        gamma=0.1
    )

    # Training loop
    print(f"Training for {num_training_steps} steps...")
    loss_history = []
    batch_size = min(16384, res*res)

    pbar = trange(num_training_steps, ncols=150)
    for step in pbar:
        # Select random frame
        t_idx = np.random.randint(0, n_frames)
        t_normalized = t_idx / (n_frames - 1)

        # Select random batch of pixels
        indices = torch.randperm(res*res, device=device)[:batch_size]
        batch_coords = model.coords_2d[indices]

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

        # Update progress bar with loss
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    print("Training complete!")

    # Extract fixed scene
    print("Extracting fixed scene...")
    fixed_scene = model.get_fixed_scene()

    # Denormalize back to original data range
    fixed_scene_denorm = fixed_scene * (data_max - data_min) + data_min
    print(f"Fixed scene range after denormalization: [{fixed_scene_denorm.min():.2f}, {fixed_scene_denorm.max():.2f}]")

    # Compute average of original activity images
    print("\nComputing average of original activity images...")
    activity_average = np.mean(activity_images, axis=0)

    # Compute median of motion frames (warped frames) as baseline
    print("Computing median of motion frames...")
    motion_median = np.median(motion_images, axis=0)

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

    print(f"\n{'='*60}")
    print("NSTM Evaluation Metrics")
    print(f"{'='*60}")
    print("Fixed Scene vs Activity Average:")
    print(f"  RMSE: {rmse_activity:.4f}")
    print(f"  SSIM: {ssim_activity:.4f}")
    print("\nBaseline (Motion Median vs Activity Average):")
    print(f"  RMSE: {rmse_baseline:.4f}")
    print(f"  SSIM: {ssim_baseline:.4f}")
    print("\nImprovement over baseline:")
    print(f"  RMSE: {((rmse_baseline - rmse_activity) / rmse_baseline * 100):.2f}%")
    print(f"  SSIM: {((ssim_activity - ssim_baseline) / (1.0 - ssim_baseline) * 100):.2f}%")
    print(f"{'='*60}\n")

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)

    # Save denormalized fixed scene
    imwrite(f'{output_dir}/fixed_scene.tif', fixed_scene_denorm.astype(np.float32))
    print(f"Saved fixed scene to {output_dir}/fixed_scene.tif")

    # Save activity average
    imwrite(f'{output_dir}/activity_average.tif', activity_average.astype(np.float32))
    print(f"Saved activity average to {output_dir}/activity_average.tif")

    # Save motion median
    imwrite(f'{output_dir}/motion_median.tif', motion_median.astype(np.float32))
    print(f"Saved motion median to {output_dir}/motion_median.tif")

    # Save metrics to file
    with open(f'{output_dir}/metrics.txt', 'w') as f:
        f.write("NSTM Evaluation Metrics\n")
        f.write("="*60 + "\n")
        f.write("Fixed Scene vs Activity Average:\n")
        f.write(f"  RMSE: {rmse_activity:.4f}\n")
        f.write(f"  SSIM: {ssim_activity:.4f}\n")
        f.write("\nBaseline (Motion Median vs Activity Average):\n")
        f.write(f"  RMSE: {rmse_baseline:.4f}\n")
        f.write(f"  SSIM: {ssim_baseline:.4f}\n")
        f.write("\nImprovement over baseline:\n")
        f.write(f"  RMSE: {((rmse_baseline - rmse_activity) / rmse_baseline * 100):.2f}%\n")
        f.write(f"  SSIM: {((ssim_activity - ssim_baseline) / (1.0 - ssim_baseline) * 100):.2f}%\n")
        f.write("="*60 + "\n")
    print(f"Saved metrics to {output_dir}/metrics.txt")

    # Plot and save loss history
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title('NSTM Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/loss_history.png', dpi=150)
    plt.close()
    print(f"Saved loss plot to {output_dir}/loss_history.png")

    print(f"{'='*60}\n")

    return model, loss_history


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
    print("Frame format: 512x512, 32-bit float, single channel TIF")
    print("Applied sinusoidal warping with motion_intensity=0.015")

    # Train NSTM on the generated frames
    nstm_output_dir = f'./graphs_data/{dataset_name}/NSTM_outputs'
    model, loss_history = train_nstm(
        motion_frames_dir=motion_frames_dir,
        activity_dir=activity_dir,
        n_frames=n_frames,
        res=512,
        device=device,
        output_dir=nstm_output_dir,
        num_training_steps=10000
    )


    # generate_compressed_video_mp4(output_dir=f"./graphs_data/{dataset_name}", run=run,
    #                                 output_name=output_name, framerate=20)

    # files = glob.glob(f'./graphs_data/{dataset_name}/Fig/*')
    # for f in files:
    #     os.remove(f)


    # if (n_neurons <= 1000):
    #     print('plot activity ...')
    #     activity = x_list[:, :, 6:7]
    #     activity = activity.squeeze()
    #     activity = activity.T
    #     activity = activity - 10 * np.arange(n_neurons)[:, None] + 200
    #     plt.figure(figsize=(10, 20))
    #     plt.plot(activity.T, linewidth=2)

    #     for i in range(0, n_neurons, 5):
    #         plt.text(-100, activity[i, 0], str(i), fontsize=24, va='center', ha='right')

    #     ax = plt.gca()
    #     ax.text(-1500, activity.mean(), 'neuron index', fontsize=32, va='center', ha='center', rotation=90)
    #     plt.xlabel("time", fontsize=32)
    #     plt.xticks(fontsize=24)
    #     ax.spines['left'].set_visible(False)
    #     ax.spines['top'].set_visible(False)
    #     ax.yaxis.set_ticks_position('right')
    #     ax.set_yticks([0, 20, 40])
    #     ax.set_yticklabels(['0', '20', '40'], fontsize=20)
    #     ax.text(n_frames * 1.2, 24, 'voltage', fontsize=24, va='center', ha='left', rotation=90)
    #     plt.tight_layout()
    #     plt.savefig(f"graphs_data/{dataset_name}/activity_1000.png", dpi=300)
    #     plt.close()




