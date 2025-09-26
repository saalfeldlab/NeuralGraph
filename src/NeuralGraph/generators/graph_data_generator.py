import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import torch
from NeuralGraph.generators.utils import *
from NeuralGraph.models.utils import *
from NeuralGraph.data_loaders import *

from GNN_Main import *
from NeuralGraph.utils import set_size
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import tifffile
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
import os

# from fa2_modified import ForceAtlas2
# import h5py as h5
# import zarr
# import xarray as xr
import pandas as pd
import tables
import torch_geometric.utils as pyg_utils
from scipy.ndimage import zoom
import re
import imageio
from NeuralGraph.generators.utils import *
from NeuralGraph.generators.davis import *
# import taichi as ti
import random
import json

def data_generate(
    config,
    visualize=True,
    run_vizualized=0,
    style="color",
    erase=False,
    step=5,
    alpha=0.2,
    ratio=1,
    scenario="none",
    best_model=None,
    device=None,
    bSave=True,
):

    has_signal = "PDE_N" in config.graph_model.signal_model_name
    has_fly = "fly" in config.dataset

    dataset_name = config.dataset

    print("")
    print(f"\033[94mdataset_name: {dataset_name}\033[0m")

    if (os.path.isfile(f"./graphs_data/{dataset_name}/x_list_0.npy")) | (
        os.path.isfile(f"./graphs_data/{dataset_name}/x_list_0.pt")
    ):
        print("watch out: data already generated")
        # return

    if config.data_folder_name != "none":
        generate_from_data(config=config, device=device, visualize=visualize, style=style, step=step)
    elif has_fly:
        data_generate_fly_voltage(
            config,
            visualize=visualize,
            run_vizualized=run_vizualized,
            style=style,
            erase=erase,
            step=step,
            device=device,
            bSave=bSave,
        )
    elif has_signal:
        data_generate_synaptic(
            config,
            visualize=visualize,
            run_vizualized=run_vizualized,
            style=style,
            erase=erase,
            step=step,
            device=device,
            bSave=bSave,
        )

    plt.style.use("default")


def generate_from_data(config, device, visualize=True, step=None, cmap=None, style=None):
    data_folder_name = config.data_folder_name

    if "wormvae" in data_folder_name:
        load_wormvae_data(config, device, visualize, step)
    elif "NeuroPAL" in data_folder_name:
        load_neuropal_data(config, device, visualize, step)
    elif 'Zapbench' in data_folder_name:
        load_zebrafish_data(config, device, visualize, step, cmap, style)
    else:
        raise ValueError(f"Unknown data folder name {data_folder_name}")

def mseq_bits(p=8, taps=(8,6,5,4), seed=1, length=None):
    """
    Simple LFSR-based m-sequence generator that returns a numpy array of ±1.
    Default p=8 -> period 2**8 - 1 = 255.
    """
    if length is None:
        length = 2**p - 1
    state = (1 << p) - 1 if seed is None else (seed % (1 << p)) or 1
    bits = []
    for _ in range(length):
        bits.append(1 if (state & 1) else -1)
        fb = 0
        for t in taps:
            fb ^= (state >> (t-1)) & 1
        state = (state >> 1) | (fb << (p-1))
    return np.array(bits, dtype=np.int8)

def assign_columns_from_uv(u_coords, v_coords, n_cols, random_state=0):
    """Cluster photoreceptors into n_cols tiles via k-means on (u,v)."""
    try:
        from sklearn.cluster import KMeans
    except Exception as e:
        raise RuntimeError("scikit-learn is required for 'tile_mseq' visual_input_type") from e
    X = np.stack([u_coords, v_coords], axis=1)
    km = KMeans(n_clusters=n_cols, n_init=10, random_state=random_state)
    labels = km.fit_predict(X)
    return labels

def compute_column_labels(u_coords, v_coords, n_columns, seed=0):
    labels = assign_columns_from_uv(u_coords, v_coords, n_columns, random_state=seed)
    centers = np.zeros((n_columns, 2), dtype=np.float32)
    counts = np.zeros(n_columns, dtype=np.int32)
    for i, lab in enumerate(labels):
        centers[lab, 0] += u_coords[i]
        centers[lab, 1] += v_coords[i]
        counts[lab] += 1
    counts[counts == 0] = 1
    centers /= counts[:, None]
    return labels, centers

def build_neighbor_graph(centers, k=6):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=min(k+1, len(centers)), algorithm="auto").fit(centers)
    dists, idxs = nbrs.kneighbors(centers)
    adj = [set() for _ in range(len(centers))]
    for i in range(len(centers)):
        for j in idxs[i,1:]:
            adj[i].add(int(j))
            adj[int(j)].add(i)
    return adj

def greedy_blue_mask(adj, n_cols, target_density=0.5, rng=None):
    if rng is None:
        rng = np.random.RandomState(0)
    order = rng.permutation(n_cols)
    chosen = np.zeros(n_cols, dtype=bool)
    blocked = np.zeros(n_cols, dtype=bool)
    target = int(target_density * n_cols)
    for i in order:
        if not blocked[i]:
            chosen[i] = True
            for j in adj[i]:
                blocked[j] = True
        if chosen.sum() >= target:
            break
    if chosen.sum() < target:
        remain = np.where(~chosen)[0]
        rng.shuffle(remain)
        for i in remain:
            conflict = any(chosen[j] for j in adj[i])
            if not conflict:
                chosen[i] = True
            if chosen.sum() >= target:
                break
    return chosen

def apply_pairwise_knobs_torch(code_pm1: torch.Tensor,
                                corr_strength: float,
                                flip_prob: float,
                                seed: int) -> torch.Tensor:
    """
    code_pm1: shape [n_tiles], values in approximately {-1, +1}
    corr_strength: 0..1; blends in a global shared ±1 component (↑ pairwise corr)
    flip_prob: 0..1; per-tile random sign flips (decorrelates)
    seed: for reproducibility (we also add tile_idx later to vary per frame)
    """
    out = code_pm1.clone()

    # Torch RNG on correct device
    gen = torch.Generator(device=out.device)
    gen.manual_seed(int(seed) & 0x7FFFFFFF)

    # (1) Optional global shared component
    if corr_strength > 0.0:
        g = torch.randint(0, 2, (1,), generator=gen, device=out.device, dtype=torch.int64)
        g = g.float().mul_(2.0).add_(-1.0)  # {0,1} -> {-1,+1}
        out.mul_(1.0 - float(corr_strength)).add_(float(corr_strength) * g)

    # (2) Optional per-tile random flips
    if flip_prob > 0.0:
        flips = torch.rand(out.shape, generator=gen, device=out.device) < float(flip_prob)
        out[flips] = -out[flips]

    return out

def data_generate_fly_voltage(config, visualize=True, run_vizualized=0, style="color", erase=False, step=5, device=None,
                              bSave=True):
    if "black" in style:
        plt.style.use("dark_background")

    simulation_config = config.simulation
    training_config = config.training
    model_config = config.graph_model

    torch.random.fork_rng(devices=device)
    if simulation_config.seed != 42:
        torch.random.manual_seed(simulation_config.seed)

    print(
        f"generating data ... {model_config.particle_model_name} {model_config.mesh_model_name} seed: {simulation_config.seed}")

    dataset_name = config.dataset
    n_neurons = simulation_config.n_neurons
    n_neuron_types = simulation_config.n_neuron_types
    n_input_neurons = simulation_config.n_input_neurons
    delta_t = simulation_config.delta_t
    n_frames = simulation_config.n_frames

    ensemble_id = simulation_config.ensemble_id
    model_id = simulation_config.model_id

    measurement_noise_level = training_config.measurement_noise_level
    noise_model_level = training_config.noise_model_level

    n_extra_null_edges = simulation_config.n_extra_null_edges

    noise_visual_input = simulation_config.noise_visual_input
    only_noise_visual_input = simulation_config.only_noise_visual_input
    visual_input_type = simulation_config.visual_input_type

    calcium_type = simulation_config.calcium_type  # "none", "leaky"
    calcium_activation = simulation_config.calcium_activation  # "softplus", "relu", "tanh", "identity"
    calcium_tau = simulation_config.calcium_tau  # time constant for calcium dynamics
    calcium_alpha = simulation_config.calcium_alpha
    calcium_beta = simulation_config.calcium_beta

    run = 0

    os.makedirs("./graphs_data/fly", exist_ok=True)
    folder = f"./graphs_data/{dataset_name}/"
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f"./graphs_data/{dataset_name}/Fig/", exist_ok=True)
    files = glob.glob(f'./graphs_data/{dataset_name}/Fig/*')
    for f in files:
        os.remove(f)

    plt.style.use("dark_background")
    extent = 8

    # Import only what's needed for mixed functionality
    from flyvis.datasets.sintel import AugmentedSintel
    from flyvis import NetworkView, Network
    from flyvis.utils.config_utils import get_default_config, CONFIG_PATH
    from NeuralGraph.generators.PDE_N9 import PDE_N9, get_photoreceptor_positions_from_net, \
        group_by_direction_and_function

    # Initialize datasets
    if "DAVIS" in visual_input_type or "mixed" in visual_input_type:
        datavis_root = "/groups/saalfeld/home/allierc/signaling/DATAVIS/JPEGImages/480p"
        davis_config = {
            "root_dir": datavis_root,
            "n_frames": 50,
            "max_frames": 80,
            "flip_axes": [0, 1],
            "n_rotations": [0, 90, 180, 270],
            "temporal_split": True,
            "dt": delta_t,
            "interpolate": True,
            "boxfilter": dict(extent=extent, kernel_size=13),
            "vertical_splits": 1,
            "center_crop_fraction": 0.6,
            "augment": False,
            "unittest": False
        }
        davis_dataset = AugmentedDavis(**davis_config)
    else:
        davis_dataset = None

    if "DAVIS" in visual_input_type:
        stimulus_dataset = davis_dataset
    else:
        sintel_config = {
            "n_frames": 19,
            "flip_axes": [0, 1],
            "n_rotations": [0, 1, 2, 3, 4, 5],
            "temporal_split": True,
            "dt": delta_t,
            "interpolate": True,
            "boxfilter": dict(extent=extent, kernel_size=13),
            "vertical_splits": 3,
            "center_crop_fraction": 0.7
        }
        stimulus_dataset = AugmentedSintel(**sintel_config)

    # Initialize network
    config_net = get_default_config(overrides=[], path=f"{CONFIG_PATH}/network/network.yaml")
    config_net.connectome.extent = extent
    net = Network(**config_net)
    nnv = NetworkView(f"flow/{ensemble_id}/{model_id}")
    trained_net = nnv.init_network(checkpoint=0)
    net.load_state_dict(trained_net.state_dict())
    torch.set_grad_enabled(False)

    params = net._param_api()
    p = {"tau_i": params.nodes.time_const, "V_i_rest": params.nodes.bias,
         "w": params.edges.syn_strength * params.edges.syn_count * params.edges.sign}
    edge_index = torch.stack(
        [torch.tensor(net.connectome.edges.source_index[:]), torch.tensor(net.connectome.edges.target_index[:])],
        dim=0).to(device)

    if n_extra_null_edges > 0:
        print(f"adding {n_extra_null_edges} extra null edges...")
        existing_edges = set(zip(edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()))
        import random
        extra_edges = []
        max_attempts = n_extra_null_edges * 10
        attempts = 0
        while len(extra_edges) < n_extra_null_edges and attempts < max_attempts:
            source = random.randint(0, n_neurons - 1)
            target = random.randint(0, n_neurons - 1)
            if (source, target) not in existing_edges and source != target:
                extra_edges.append([source, target])
            attempts += 1
        if extra_edges:
            extra_edge_index = torch.tensor(extra_edges, dtype=torch.long, device=device).t()
            edge_index = torch.cat([edge_index, extra_edge_index], dim=1)
            p["w"] = torch.cat([p["w"], torch.zeros(len(extra_edges), device=device)])

    pde = PDE_N9(p=p, f=torch.nn.functional.relu, params=simulation_config.params,
                 model_type=model_config.signal_model_name, n_neuron_types=n_neuron_types, device=device)

    if bSave:
        # torch.save(mask, f"./graphs_data/{dataset_name}/mask.pt")
        # torch.save(connectivity, f"./graphs_data/{dataset_name}/connectivity.pt")
        torch.save(p["w"], f"./graphs_data/{dataset_name}/weights.pt")
        torch.save(edge_index, f"graphs_data/{dataset_name}/edge_index.pt")
        torch.save(p["tau_i"], f"./graphs_data/{dataset_name}/taus.pt")
        torch.save(p["V_i_rest"], f"./graphs_data/{dataset_name}/V_i_rest.pt")


    x_coords, y_coords, u_coords, v_coords = get_photoreceptor_positions_from_net(net)

    node_types = np.array(net.connectome.nodes["type"])
    node_types_str = [t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in node_types]
    grouped_types = np.array([group_by_direction_and_function(t) for t in node_types_str])
    unique_types, node_types_int = np.unique(node_types, return_inverse=True)

    X1 = torch.tensor(np.stack((x_coords, y_coords), axis=1), dtype=torch.float32, device=device)

    from NeuralGraph.generators.utils import get_equidistant_points
    xc, yc = get_equidistant_points(n_points=n_neurons - x_coords.shape[0])
    pos = torch.tensor(np.stack((xc, yc), axis=1), dtype=torch.float32, device=device) / 2
    X1 = torch.cat((X1, pos[torch.randperm(pos.size(0))]), dim=0)

    state = net.steady_state(t_pre=2.0, dt=delta_t, batch_size=1)
    initial_state = state.nodes.activity.squeeze()
    n_neurons = len(initial_state)

    sequences = stimulus_dataset[0]["lum"]
    frame = sequences[0][None, None]
    net.stimulus.add_input(frame)

    x = torch.zeros(n_neurons, 9, dtype=torch.float32, device=device)
    x[:, 1:3] = X1
    x[:, 0] = torch.arange(n_neurons, dtype=torch.float32)
    x[:, 3] = initial_state
    x[:, 4] = net.stimulus().squeeze()
    x[:, 5] = torch.tensor(grouped_types, dtype=torch.float32, device=device)
    x[:, 6] = torch.tensor(node_types_int, dtype=torch.float32, device=device)
    x[:, 7] = torch.rand(n_neurons, dtype=torch.float32, device=device)
    x[:, 8] = calcium_alpha * x[:, 7] + calcium_beta

    # Mixed sequence setup
    if "mixed" in visual_input_type:
        mixed_types = ["sintel", "davis", "blank", "noise"]
        mixed_cycle_lengths = [60, 60, 30, 60]  # Different lengths for each type
        mixed_current_type = 0
        mixed_frame_count = 0
        current_cycle_length = mixed_cycle_lengths[mixed_current_type]
        if not davis_dataset:
            sintel_config_mixed = {
                "n_frames": 19,
                "flip_axes": [0, 1],
                "n_rotations": [0, 1, 2, 3, 4, 5],
                "temporal_split": True,
                "dt": delta_t,
                "interpolate": True,
                "boxfilter": dict(extent=extent, kernel_size=13),
                "vertical_splits": 3,
                "center_crop_fraction": 0.7
            }
            davis_dataset = AugmentedSintel(**sintel_config_mixed)
        sintel_iter = iter(stimulus_dataset)
        davis_iter = iter(davis_dataset)
        current_sintel_seq = None
        current_davis_seq = None
        sintel_frame_idx = 0
        davis_frame_idx = 0

    target_frames = n_frames
    dataset_length = len(stimulus_dataset)
    frames_per_sequence = 35
    total_frames_per_pass = dataset_length * frames_per_sequence
    num_passes_needed = (target_frames // total_frames_per_pass) + 1

    y_list = []
    x_list = []
    it = simulation_config.start_frame
    id_fig = 0

    tile_labels = None
    tile_codes_torch = None
    tile_period = None
    tile_idx = 0
    tile_contrast = simulation_config.tile_contrast
    n_columns = n_input_neurons // 8
    tile_seed = simulation_config.seed

    with torch.no_grad():
        for pass_num in range(num_passes_needed):
            for data_idx, data in enumerate(tqdm(stimulus_dataset, desc="Processing stimulus data")):
                if simulation_config.simulation_initial_state:
                    x[:, 3] = initial_state
                    if only_noise_visual_input > 0:
                        x[:n_input_neurons, 4:5] = torch.clamp(torch.relu(
                            0.5 + torch.rand((n_input_neurons, 1), dtype=torch.float32,
                                             device=device) * only_noise_visual_input / 2), 0, 1)

                sequences = data["lum"]

                # Sample flash parameters for each subsequence if flash stimulus is requested
                if "flash" in visual_input_type:
                    # Sample flash duration from specific values: 1, 2, 5, 10, 20 frames
                    flash_duration_options = [1, 2, 5] #, 10, 20]
                    flash_cycle_frames = flash_duration_options[
                        torch.randint(0, len(flash_duration_options), (1,), device=device).item()
                    ]

                    flash_intensity = torch.abs(torch.rand(n_input_neurons, device=device) * 0.5 + 0.5)

                if "mixed" in visual_input_type:
                    if mixed_frame_count >= current_cycle_length:
                        mixed_current_type = (mixed_current_type + 1) % 4
                        mixed_frame_count = 0
                        current_cycle_length = mixed_cycle_lengths[mixed_current_type]
                    current_type = mixed_types[mixed_current_type]

                    if current_type == "sintel":
                        if current_sintel_seq is None or sintel_frame_idx >= current_sintel_seq["lum"].shape[0]:
                            try:
                                current_sintel_seq = next(sintel_iter)
                                sintel_frame_idx = 0
                            except StopIteration:
                                sintel_iter = iter(stimulus_dataset)
                                current_sintel_seq = next(sintel_iter)
                                sintel_frame_idx = 0
                        sequences = current_sintel_seq["lum"]
                        start_frame = sintel_frame_idx
                    elif current_type == "davis":
                        if current_davis_seq is None or davis_frame_idx >= current_davis_seq["lum"].shape[0]:
                            try:
                                current_davis_seq = next(davis_iter)
                                davis_frame_idx = 0
                            except StopIteration:
                                davis_iter = iter(davis_dataset)
                                current_davis_seq = next(davis_iter)
                                davis_frame_idx = 0
                        sequences = current_davis_seq["lum"]
                        start_frame = davis_frame_idx
                    else:
                        start_frame = 0

                # Determine sequence length based on stimulus type
                if "flash" in visual_input_type:
                    sequence_length = 60  # Fixed 60 frames for flash sequences
                else:
                    sequence_length = sequences.shape[0]

                for frame_id in range(sequence_length):
                    if "flash" in visual_input_type:
                        # Generate repeating flash stimulus
                        current_flash_frame = frame_id % (flash_cycle_frames * 2)  # Create on/off cycle
                        x[:, 4] = 0
                        if current_flash_frame < flash_cycle_frames:
                            x[:n_input_neurons, 4] = flash_intensity
                    elif "mixed" in visual_input_type:
                        current_type = mixed_types[mixed_current_type]

                        if current_type == "blank":
                            x[:, 4] = 0
                        elif current_type == "noise":
                            x[:n_input_neurons, 4:5] = torch.relu(
                                0.5 + torch.rand((n_input_neurons, 1), dtype=torch.float32, device=device) * 0.5)
                        else:
                            actual_frame_id = (start_frame + frame_id) % sequences.shape[0]
                            frame = sequences[actual_frame_id][None, None]
                            net.stimulus.add_input(frame)
                            x[:, 4] = net.stimulus().squeeze()
                            if current_type == "sintel":
                                sintel_frame_idx += 1
                            elif current_type == "davis":
                                davis_frame_idx += 1
                        mixed_frame_count += 1
                    elif "tile_mseq" in visual_input_type:
                        if tile_codes_torch is None:
                            # 1) Cluster photoreceptors into columns based on (u,v)
                            tile_labels_np = assign_columns_from_uv(
                                u_coords, v_coords, n_columns, random_state=tile_seed
                            )  # shape: (n_input_neurons,)
    
                            # 2) Build per-column m-sequences (±1) with random phase per column
                            base = mseq_bits(p=8, seed=tile_seed).astype(np.float32)  # ±1, shape (255,)
                            rng = np.random.RandomState(tile_seed)
                            phases = rng.randint(0, base.shape[0], size=n_columns)
                            tile_codes_np = np.stack([np.roll(base, ph) for ph in phases], axis=0)  # (n_columns, 255), ±1
    
                            # 3) Convert to torch on the right device/dtype; keep as ±1 (no [0,1] mapping here)
                            tile_codes_torch = torch.from_numpy(tile_codes_np).to(x.device,
                                                                                  dtype=x.dtype)  # (n_columns, 255), ±1
                            tile_labels = torch.from_numpy(tile_labels_np).to(x.device,
                                                                              dtype=torch.long)  # (n_input_neurons,)
                            tile_period = tile_codes_torch.shape[1]
                            tile_idx = 0
    
                        # 4) Baseline for all neurons (mean luminance), then write per-column values to PRs
                        x[:, 4] = 0.5
                        col_vals_pm1 = tile_codes_torch[:, tile_idx % tile_period]  # (n_columns,), ±1 before knobs
    
                        # Apply the two simple knobs per frame on ±1 codes
                        col_vals_pm1 = apply_pairwise_knobs_torch(
                            code_pm1=col_vals_pm1,
                            corr_strength=float(simulation_config.tile_corr_strength),
                            flip_prob=float(simulation_config.tile_flip_prob),
                            seed=int(simulation_config.seed) + int(tile_idx)
                        )
    
                        # Map to [0,1] with your contrast convention and broadcast via labels
                        col_vals_01 = 0.5 + (tile_contrast * 0.5) * col_vals_pm1
                        x[:n_input_neurons, 4] = col_vals_01[tile_labels]
    
                        tile_idx += 1
                    elif "tile_blue_noise" in visual_input_type:
                        if tile_codes_torch is None:
                            # Label columns and build neighborhood graph
                            tile_labels_np, col_centers = compute_column_labels(u_coords, v_coords, n_columns, seed=tile_seed)
                            try:
                                adj = build_neighbor_graph(col_centers, k=6)
                            except Exception:
                                from scipy.spatial.distance import pdist, squareform
                                D = squareform(pdist(col_centers))
                                nn = np.partition(D + np.eye(D.shape[0]) * 1e9, 1, axis=1)[:, 1]
                                radius = 1.3 * np.median(nn)
                                adj = [set(np.where((D[i] > 0) & (D[i] <= radius))[0].tolist()) for i in
                                       range(len(col_centers))]
        
                            tile_labels = torch.from_numpy(tile_labels_np).to(x.device, dtype=torch.long)
                            tile_period = 257
                            tile_idx = 0
        
                            # Pre-generate ±1 codes (keep ±1; no [0,1] mapping here)
                            tile_codes_torch = torch.empty((n_columns, tile_period), dtype=x.dtype, device=x.device)
                            rng = np.random.RandomState(tile_seed)
                            for t in range(tile_period):
                                mask = greedy_blue_mask(adj, n_columns, target_density=0.5, rng=rng)  # boolean mask
                                vals = np.where(mask, 1.0, -1.0).astype(np.float32)  # ±1
                                # NOTE: do not apply flip prob here; we do it uniformly via the helper per frame below
                                tile_codes_torch[:, t] = torch.from_numpy(vals).to(x.device, dtype=x.dtype)
        
                        # Baseline luminance
                        x[:, 4] = 0.5
                        col_vals_pm1 = tile_codes_torch[:, tile_idx % tile_period]  # (n_columns,), ±1 before knobs
        
                        # Apply the two simple knobs per frame on ±1 codes
                        col_vals_pm1 = apply_pairwise_knobs_torch(
                            code_pm1=col_vals_pm1,
                            corr_strength=float(simulation_config.tile_corr_strength),
                            flip_prob=float(simulation_config.tile_flip_prob),
                            seed=int(simulation_config.seed) + int(tile_idx)
                        )
        
                        # Map to [0,1] with contrast and broadcast via labels
                        col_vals_01 = 0.5 + (tile_contrast * 0.5) * col_vals_pm1
                        x[:n_input_neurons, 4] = col_vals_01[tile_labels]
        
                        tile_idx += 1
                    else:
                        frame = sequences[frame_id][None, None]
                        net.stimulus.add_input(frame)
                        if (only_noise_visual_input > 0):
                            if (visual_input_type == "") | (it == 0) | ("50/50" in visual_input_type):
                                x[:n_input_neurons, 4:5] = torch.relu(
                                    0.5 + torch.rand((n_input_neurons, 1), dtype=torch.float32,
                                                     device=device) * only_noise_visual_input / 2)
                        else:
                            if 'blank' in visual_input_type:
                                if (data_idx % simulation_config.blank_freq > 0):
                                    x[:, 4] = net.stimulus().squeeze()
                                else:
                                    x[:, 4] = 0
                            else:
                                x[:, 4] = net.stimulus().squeeze()
                            if noise_visual_input > 0:
                                x[:n_input_neurons, 4:5] = x[:n_input_neurons, 4:5] + torch.randn((n_input_neurons, 1),
                                                                                                  dtype=torch.float32,
                                                                                                  device=device) * noise_visual_input

                    dataset = pyg.data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index)

                    y = pde(dataset, has_field=False)

                    x_list.append(to_numpy(x.clone().detach()))

                    if noise_model_level > 0:
                        x[:, 3:4] = x[:, 3:4] + delta_t * y + torch.randn((n_neurons, 1), dtype=torch.float32,
                                                                          device=device) * noise_model_level
                    else:
                        x[:, 3:4] = x[:, 3:4] + delta_t * y

                    if calcium_type == "leaky":
                        # Voltage-driven activation
                        if calcium_activation == "softplus":
                            u = torch.nn.functional.softplus(x[:, 3:4])
                        elif calcium_activation == "relu":
                            u = torch.nn.functional.relu(x[:, 3:4])
                        elif calcium_activation == "tanh":
                            u = torch.tanh(x[:, 3:4])
                        elif calcium_activation == "identity":
                            u = x[:, 3:4].clone()

                        x[:, 7:8] = x[:, 7:8] + (delta_t / calcium_tau) * (-x[:, 7:8] + u)
                        x[:, 7:8] = torch.clamp(x[:, 7:8], min=0.0)
                        x[:, 8:9] = calcium_alpha * x[:, 7:8] + calcium_beta

                        y = (x[:, 7:8] - torch.tensor(x_list[-1][:, 7:8], dtype=torch.float32,device=device)) / delta_t

                    y_list.append(to_numpy(y.clone().detach()))

                    if (visualize & (run == run_vizualized) & (it>0) & (it % step == 0) & (it <= 400 * step)):
                        if "latex" in style:
                            plt.rcParams["text.usetex"] = True
                            rc("font", **{"family": "serif", "serif": ["Palatino"]})

                        matplotlib.rcParams["savefig.pad_inches"] = 0
                        num = f"{id_fig:06}"
                        id_fig += 1

                        neuron_types = to_numpy(x[:, 6]).astype(int)
                        index_to_name = {0: 'Am', 1: 'C2', 2: 'C3', 3: 'CT1(Lo1)', 4: 'CT1(M10)', 5: 'L1', 6: 'L2',
                                         7: 'L3', 8: 'L4', 9: 'L5', 10: 'Lawf1', 11: 'Lawf2', 12: 'Mi1', 13: 'Mi10',
                                         14: 'Mi11', 15: 'Mi12', 16: 'Mi13', 17: 'Mi14', 18: 'Mi15', 19: 'Mi2',
                                         20: 'Mi3', 21: 'Mi4', 22: 'Mi9', 23: 'R1', 24: 'R2', 25: 'R3', 26: 'R4',
                                         27: 'R5', 28: 'R6', 29: 'R7', 30: 'R8', 31: 'T1', 32: 'T2', 33: 'T2a',
                                         34: 'T3', 35: 'T4a', 36: 'T4b', 37: 'T4c', 38: 'T4d', 39: 'T5a', 40: 'T5b',
                                         41: 'T5c', 42: 'T5d', 43: 'Tm1', 44: 'Tm16', 45: 'Tm2', 46: 'Tm20', 47: 'Tm28',
                                         48: 'Tm3', 49: 'Tm30', 50: 'Tm4', 51: 'Tm5Y', 52: 'Tm5a', 53: 'Tm5b',
                                         54: 'Tm5c', 55: 'Tm9', 56: 'TmY10', 57: 'TmY13', 58: 'TmY14', 59: 'TmY15',
                                         60: 'TmY18', 61: 'TmY3', 62: 'TmY4', 63: 'TmY5a', 64: 'TmY9'}

                        anatomical_order = [None, 23, 24, 25, 26, 27, 28, 29, 30, 5, 6, 7, 8, 9, 10, 11, 12, 19, 20, 21,
                                            22, 13, 14, 15, 16, 17, 18, 43, 45, 48, 50, 44, 46, 47, 49, 51, 52, 53, 54,
                                            55, 61, 62, 63, 56, 57, 58, 59, 60, 64, 1, 2, 4, 3, 31, 32, 33, 34, 35, 36,
                                            37, 38, 39, 40, 41, 42, 0]

                        if calcium_type != "none":

                            n_rows = 16  # 8 for voltage, 8 for calcium
                            n_cols = 9
                            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18.04, 32.46), facecolor='black')
                            plt.subplots_adjust(hspace=1.2)
                            axes_flat = axes.flatten()

                            neuron_types = to_numpy(x[:, 6]).astype(int)
                            all_voltages = to_numpy(x[:, 3])
                            all_calcium = to_numpy(x[:, 7])

                            for panel_idx in range(66,72):
                                axes_flat[panel_idx].set_visible(False)
                                axes_flat[panel_idx].set_visible(False)

                            # Add row labels
                            # fig.text(0.5, 0.95, 'Voltage', ha='center', va='center', fontsize=22, color='white')
                            # fig.text(0.5, 0.48, 'Calcium', ha='center', va='center', fontsize=22, color='white')

                            panel_idx = 0
                            for type_idx in anatomical_order:
                                # --- top row: voltage ---
                                ax_v = axes_flat[panel_idx]
                                if type_idx is None:
                                    ax_v.scatter(to_numpy(X1[:n_input_neurons, 0]), to_numpy(X1[:n_input_neurons, 1]),
                                                 s=64, c=to_numpy(x[:n_input_neurons, 4]), cmap="viridis",
                                                 vmin=0, vmax=1.05, marker='h', alpha=1.0, linewidths=0,
                                                 edgecolors='black')
                                    ax_v.set_title('Stimuli', fontsize=18, color='white')
                                else:
                                    mask = neuron_types == type_idx
                                    if np.sum(mask) > 0:
                                        voltages = all_voltages[mask]
                                        positions_x = to_numpy(X1[:np.sum(mask), 0])
                                        positions_y = to_numpy(X1[:np.sum(mask), 1])
                                        ax_v.scatter(positions_x, positions_y, s=72, c=voltages,
                                                     cmap='viridis', vmin=-2, vmax=2, marker='h', alpha=1,
                                                     linewidths=0, edgecolors='black')
                                    ax_v.set_title(index_to_name.get(type_idx, f"Type_{type_idx}"), fontsize=18,
                                                    color='white')  # increased fontsize

                                ax_v.set_facecolor('black')
                                ax_v.set_xticks([])
                                ax_v.set_yticks([])
                                ax_v.set_aspect('equal')
                                for spine in ax_v.spines.values():
                                    spine.set_visible(False)

                                # --- bottom row: calcium ---
                                ax_ca = axes_flat[panel_idx + n_cols * 8]
                                if type_idx is None:
                                    ax_ca.scatter(to_numpy(X1[:n_input_neurons, 0]), to_numpy(X1[:n_input_neurons, 1]),
                                                  s=64, c=to_numpy(x[:n_input_neurons, 4]), cmap="viridis",
                                                  vmin=0, vmax=1.05, marker='h', alpha=1.0, linewidths=0,
                                                  edgecolors='black')
                                    ax_ca.set_title('Stimuli', fontsize=18, color='white')
                                else:
                                    mask = neuron_types == type_idx
                                    if np.sum(mask) > 0:
                                        calcium_values = all_calcium[mask]
                                        positions_x = to_numpy(X1[:np.sum(mask), 0])
                                        positions_y = to_numpy(X1[:np.sum(mask), 1])
                                        ax_ca.scatter(positions_x, positions_y, s=72, c=calcium_values,
                                                      cmap='plasma', vmin=0, vmax=2, marker='h',
                                                      alpha=1, linewidths=0, edgecolors='black')  # green LUT
                                    else:
                                        ax_ca.text(0.5, 0.5, f'No neurons', transform=ax_ca.transAxes, ha='center',
                                                   va='center', color='red', fontsize=10)
                                    ax_ca.set_title(index_to_name.get(type_idx, f"Type_{type_idx}"), fontsize=18,
                                                    color='white')  # increased fontsize

                                ax_ca.set_facecolor('black')
                                ax_ca.set_xticks([])
                                ax_ca.set_yticks([])
                                ax_ca.set_aspect('equal')
                                for spine in ax_ca.spines.values():
                                    spine.set_visible(False)

                                panel_idx += 1

                            for i in range(panel_idx + n_cols * 8, len(axes_flat)):
                                axes_flat[i].set_visible(False)

                            plt.tight_layout()
                            plt.subplots_adjust(top=0.92, bottom=0.05)
                            plt.savefig(f"graphs_data/{dataset_name}/Fig/Fig_{run}_{num}.png", dpi=80)
                            plt.close()

                        else:

                            fig, axes = plt.subplots(8, 9, figsize=(18.04, 16.23), facecolor='black')
                            plt.subplots_adjust(top=1.2, bottom=0.05, hspace=1.2)
                            axes_flat = axes.flatten()

                            panel_idx = 0
                            for type_idx in anatomical_order:
                                if panel_idx >= len(axes_flat):
                                    break
                                ax = axes_flat[panel_idx]

                                if type_idx is None:
                                    stimulus_scatter = ax.scatter(to_numpy(X1[:n_input_neurons, 0]),
                                                                  to_numpy(X1[:n_input_neurons, 1]), s=64,
                                                                  c=to_numpy(x[:n_input_neurons, 4]), cmap="viridis",
                                                                  vmin=0, vmax=1.05, marker='h', alpha=1.0, linewidths=0.0,
                                                                  edgecolors='black')
                                    ax.set_title('stimuli', fontsize=18, color='white', pad=8, y=0.95)
                                else:
                                    type_mask = neuron_types == type_idx
                                    type_count = np.sum(type_mask)
                                    type_name = index_to_name.get(type_idx, f'Type_{type_idx}')
                                    if type_count > 0:
                                        type_voltages = to_numpy(x[type_mask, 3])
                                        hex_positions_x = to_numpy(X1[:type_count, 0])
                                        hex_positions_y = to_numpy(X1[:type_count, 1])
                                        neural_scatter = ax.scatter(hex_positions_x, hex_positions_y, s=72, c=type_voltages,
                                                                    cmap='viridis', vmin=-2, vmax=2, marker='h', alpha=1,
                                                                    linewidths=0.0, edgecolors='black')
                                        if type_name.startswith('R'):
                                            title_color = 'yellow'
                                        elif type_name.startswith(('L', 'Lawf')):
                                            title_color = 'cyan'
                                        elif type_name.startswith(('Mi', 'Tm', 'TmY')):
                                            title_color = 'orange'
                                        elif type_name.startswith('T'):
                                            title_color = 'red'
                                        elif type_name.startswith('C'):
                                            title_color = 'magenta'
                                        else:
                                            title_color = 'white'
                                        ax.set_title(f'{type_name}', fontsize=18, color='white', pad=8, y=0.95)
                                    else:
                                        ax.text(0.5, 0.5, f'No {type_name}\nNeurons', transform=ax.transAxes, ha='center',
                                                va='center', color='red', fontsize=8)
                                        ax.set_title(f'{type_name}\n(0)', fontsize=10, color='gray', pad=8, y=0.95)

                                ax.set_facecolor('black')
                                ax.set_xticks([])
                                ax.set_yticks([])
                                ax.set_aspect('equal')
                                for spine in ax.spines.values():
                                    spine.set_visible(False)
                                panel_idx += 1

                            for i in range(panel_idx, len(axes_flat)):
                                axes_flat[i].set_visible(False)

                            plt.tight_layout()
                            plt.subplots_adjust(top=0.95, bottom=0.05)
                            plt.savefig(f"graphs_data/{dataset_name}/Fig/Fig_{run}_{num}.png", dpi=80)
                            plt.close()

                    it = it + 1
                    if it >= target_frames:
                        break
                if it >= target_frames:
                    break
            if it >= target_frames:
                break

    print(f"generated {len(x_list)} frames total")

    if visualize & (run == run_vizualized):
        print('generating lossless video ...')

        config_indices = dataset_name.split('fly_N9_')[1] if 'fly_N9_' in dataset_name else 'no_id'
        src = f"./graphs_data/{dataset_name}/Fig/Fig_0_000000.png"
        dst = f"./graphs_data/{dataset_name}/input_{config_indices}.png"
        with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
            fdst.write(fsrc.read())

        generate_lossless_video_ffv1(output_dir=f"./graphs_data/{dataset_name}", run=run, config_indices=config_indices,framerate=20)
        generate_lossless_video_libx264(output_dir=f"./graphs_data/{dataset_name}", run=run,
                                        config_indices=config_indices,framerate=20)
        generate_compressed_video_mp4(output_dir=f"./graphs_data/{dataset_name}", run=run,
                                      config_indices=config_indices,framerate=20)

        files = glob.glob(f'./graphs_data/{dataset_name}/Fig/*')
        for f in files:
            os.remove(f)

    print('Ising analysis ...')
    x_list = np.array(x_list)
    y_list = np.array(y_list)

    if bSave:
        print('plot activity ...')

        activity = torch.tensor(x_list[:, :, 3:4], device=device).squeeze().t()  # voltage
        input_visual = torch.tensor(x_list[:, :, 4:5], device=device).squeeze().t()

        if calcium_type != "none":
            calcium_activity = torch.tensor(x_list[:, :, 7:8], device=device).squeeze().t()

            plt.figure(figsize=(16, 12))

            # --- Top row: visual input ---
            plt.subplot(3, 1, 1)
            plt.title("Input to visual neurons", fontsize=24)
            input_neurons = [731, 1042, 329, 1110, 1176, 1526, 1350, 90, 813, 1695]
            for i in input_neurons:
                plt.plot(to_numpy(input_visual[i, :]), linewidth=1)
            plt.ylabel("$x_i$", fontsize=20)
            plt.xlim([0, n_frames // 300])
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)

            # --- Middle row: voltage ---
            plt.subplot(3, 1, 2)
            plt.title("Voltage activity of neurons (x10)", fontsize=24)
            neurons_to_plot = [2602, 3175, 12915, 10391, 13120, 9939, 12463, 3758, 10341, 4293]
            for i in neurons_to_plot:
                plt.plot(to_numpy(activity[i, :]), linewidth=1)
            plt.ylabel("$V_i$", fontsize=20)
            plt.xlim([0, n_frames // 300])
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)

            # --- Bottom row: calcium ---
            plt.subplot(3, 1, 3)
            plt.title("Calcium activity of neurons", fontsize=24)
            for i in neurons_to_plot:
                plt.plot(to_numpy(calcium_activity[i, :]), linewidth=1)
            plt.xlabel("time", fontsize=20)
            plt.ylabel("$[Ca]_i$", fontsize=20)
            plt.xlim([0, n_frames // 300])
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)

            plt.tight_layout()
            plt.savefig(f"graphs_data/{dataset_name}/activity_voltage_calcium.tif", dpi=300)
            plt.close()
        else:
            # Original 2-row figure (input + voltage)
            plt.figure(figsize=(16, 8))

            # Left subplot: visual input
            plt.subplot(1, 2, 1)
            plt.title("Input to visual neurons", fontsize=24)
            input_neurons = [731, 1042, 329, 1110, 1176, 1526, 1350, 90, 813, 1695]
            for i in input_neurons:
                plt.plot(to_numpy(input_visual[i, :]), linewidth=1)
            plt.xlabel("time", fontsize=24)
            plt.ylabel("$x_i$", fontsize=24)
            plt.xlim([0, n_frames // 300])
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)

            # Right subplot: voltage
            plt.subplot(1, 2, 2)
            plt.title("Voltage activity of neurons (x10)", fontsize=24)
            neurons_to_plot = [2602, 3175, 12915, 10391, 13120, 9939, 12463, 3758, 10341, 4293]
            for i in neurons_to_plot:
                plt.plot(to_numpy(activity[i, :]), linewidth=1)
            plt.xlabel("time", fontsize=24)
            plt.ylabel("$x_i$", fontsize=24)
            plt.xlim([0, n_frames // 300])
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)

            plt.tight_layout()
            plt.savefig(f"graphs_data/{dataset_name}/activity_voltage.tif", dpi=300)
            plt.close()

        print('save data ...')

        if measurement_noise_level > 0:
            np.save(f"graphs_data/{dataset_name}/raw_x_list_{run}.npy", x_list)
            np.save(f"graphs_data/{dataset_name}/raw_y_list_{run}.npy", y_list)
            for k in range(x_list.shape[0]):
                x_list[k, :, 3] = x_list[k, :, 3] + np.random.normal(0, measurement_noise_level, x_list.shape[1])
            for k in range(1, x_list.shape[0] - 1):
                y_list[k] = (x_list[k + 1, :, 3:4] - x_list[k, :, 3:4]) / delta_t
            np.save(f"graphs_data/{dataset_name}/x_list_{run}.npy", x_list)
            np.save(f"graphs_data/{dataset_name}/y_list_{run}.npy", y_list)
            print("data + noise saved ...")
        else:
            np.save(f"graphs_data/{dataset_name}/x_list_{run}.npy", x_list)
            np.save(f"graphs_data/{dataset_name}/y_list_{run}.npy", y_list)
            print("data saved ...")

def data_generate_synaptic(
    config,
    visualize=True,
    run_vizualized=0,
    style="color",
    erase=False,
    step=5,
    alpha=0.2,
    ratio=1,
    scenario="none",
    device=None,
    bSave=True,
):
    simulation_config = config.simulation
    training_config = config.training
    model_config = config.graph_model

    torch.random.fork_rng(devices=device)
    torch.random.manual_seed(simulation_config.seed)

    print(
        f"generating data ... {model_config.particle_model_name} {model_config.mesh_model_name}"
    )

    n_neuron_types = simulation_config.n_neuron_types
    n_neurons = simulation_config.n_neurons

    delta_t = simulation_config.delta_t
    n_frames = simulation_config.n_frames
    has_particle_dropout = training_config.particle_dropout > 0
    dataset_name = config.dataset
    excitation = simulation_config.excitation
    noise_model_level = training_config.noise_model_level
    measurement_noise_level = training_config.measurement_noise_level
    cmap = CustomColorMap(config=config)

    field_type = model_config.field_type
    if field_type != "":
        n_nodes = simulation_config.n_nodes
        n_nodes_per_axis = int(np.sqrt(n_nodes))

    torch.random.fork_rng(devices=device)
    torch.random.manual_seed(simulation_config.seed)

    if config.data_folder_name != "none":
        print(f"generating from data ...")
        generate_from_data(
            config=config, device=device, visualize=visualize, folder=folder, step=step
        )
        return

    folder = f"./graphs_data/{dataset_name}/"
    if erase:
        files = glob.glob(f"{folder}/*")
        for f in files:
            if (
                (not ("X1.pt" in f))
                & (not ("Signal" in f))
                & (not ("Viz" in f))
                & (not ("Exc" in f))
                & (f[-3:] != "Fig")
                & (f[-14:] != "generated_data")
                & (f != "p.pt")
                & (f != "cycle_length.pt")
                & (f != "model_config.json")
                & (f != "generation_code.py")
            ):
                os.remove(f)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f"./graphs_data/{dataset_name}/Fig/", exist_ok=True)
    files = glob.glob(f"./graphs_data/{dataset_name}/Fig/*")
    for f in files:
        os.remove(f)
    os.makedirs(f"./graphs_data/{dataset_name}/Viz/", exist_ok=True)
    files = glob.glob(f"./graphs_data/{dataset_name}/Viz/*")
    for f in files:
        os.remove(f)
    os.makedirs(f"./graphs_data/{dataset_name}/Exc/", exist_ok=True)
    files = glob.glob(f"./graphs_data/{dataset_name}/Exc/*")
    for f in files:
        os.remove(f)
    os.makedirs(f"./graphs_data/{dataset_name}/Signal/", exist_ok=True)
    files = glob.glob(f"./graphs_data/{dataset_name}/Signal/*")
    for f in files:
        os.remove(f)

    particle_dropout_mask = np.arange(n_neurons)
    if has_particle_dropout:
        draw = np.random.permutation(np.arange(n_neurons))
        cut = int(n_neurons * (1 - training_config.particle_dropout))
        particle_dropout_mask = draw[0:cut]
        inv_particle_dropout_mask = draw[cut:]
        x_removed_list = []

    if (("modulation" in model_config.field_type) | ("visual" in model_config.field_type)
    ) & ("PDE_N6" not in model_config.signal_model_name):
        im = imread(f"graphs_data/{simulation_config.node_value_map}")
    if "permutation" in field_type:
        permutation_indices = torch.randperm(n_neurons)
        inverse_permutation_indices = torch.argsort(permutation_indices)
        torch.save(
            permutation_indices, f"./graphs_data/{dataset_name}/permutation_indices.pt"
        )
        torch.save(
            inverse_permutation_indices,
            f"./graphs_data/{dataset_name}/inverse_permutation_indices.pt",
        )
    if "excitation_single" in field_type:
        parts = field_type.split("_")
        period = int(parts[-2])
        amplitude = float(parts[-1])

    if "black" in style:
        plt.style.use("dark_background")

    for run in range(config.training.n_runs):

        id_fig = 0

        X = torch.zeros((n_neurons, n_frames + 1), device=device)

        x_list = []
        y_list = []

        # initialize particle and graph states
        X1, V1, T1, H1, A1, N1 = init_neurons(
            config=config, scenario=scenario, ratio=ratio, device=device
        )

        A1 = torch.ones((n_neurons, 1), dtype=torch.float32, device=device)
        U1 = torch.rand_like(H1, device=device)
        U1[:, 1] = 0

        if simulation_config.shuffle_neuron_types:
            if run == 0:
                index = torch.randperm(n_neurons)
                T1 = T1[index]
                first_T1 = T1.clone().detach()
            else:
                T1 = first_T1.clone().detach()

        if run == 0:
            edge_index, connectivity, mask = init_connectivity(
                simulation_config.connectivity_file,
                simulation_config.connectivity_distribution,
                simulation_config.connectivity_filling_factor,
                T1,
                n_neurons,
                n_neuron_types,
                dataset_name,
                device,
            )

            model, bc_pos, bc_dpos = choose_model(
                config=config, W=connectivity, device=device
            )

            torch.save(edge_index, f"./graphs_data/{dataset_name}/edge_index.pt")
            torch.save(mask, f"./graphs_data/{dataset_name}/mask.pt")
            torch.save(connectivity, f"./graphs_data/{dataset_name}/connectivity.pt")

        if run == run_vizualized:
            if "black" in style:
                plt.style.use("dark_background")
            plt.figure(figsize=(10, 10))
            for n in range(n_neuron_types):
                pos = torch.argwhere(T1.squeeze() == n)
                plt.scatter(
                    to_numpy(X1[pos, 0]),
                    to_numpy(X1[pos, 1]),
                    s=100,
                    color=cmap.color(n),
                )
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(f"graphs_data/{dataset_name}/type_distribution.tif", dpi=130)
            plt.close()

        if "modulation" in field_type:
            if run == 0:
                X1_mesh, V1_mesh, T1_mesh, H1_mesh, A1_mesh, N1_mesh, mesh_data = (
                    init_mesh(config, device=device)
                )
                X1 = X1_mesh

        elif "visual" in field_type:
            if run == 0:
                X1_mesh, V1_mesh, T1_mesh, H1_mesh, A1_mesh, N1_mesh, mesh_data = (
                    init_mesh(config, device=device)
                )
                x, y = get_equidistant_points(n_points=1024)
                X1 = (
                    torch.tensor(
                        np.stack((x, y), axis=1), dtype=torch.float32, device=device
                    )
                    / 2
                )
                X1[:, 1] = X1[:, 1] + 1.5
                X1[:, 0] = X1[:, 0] + 0.5
                X1 = torch.cat((X1_mesh, X1[0 : n_neurons - n_nodes]), 0)

        x = torch.concatenate(
            (
                N1.clone().detach(),
                X1.clone().detach(),
                V1.clone().detach(),
                T1.clone().detach(),
                H1.clone().detach(),
                A1.clone().detach(),
            ),
            1,
        )
        check_and_clear_memory(
            device=device,
            iteration_number=0,
            every_n_iterations=1,
            memory_percentage_threshold=0.6,
        )

        time.sleep(0.5)
        for it in trange(simulation_config.start_frame, n_frames + 1):
            # calculate type change
            with torch.no_grad():
                if ("modulation" in field_type) & (it >= 0):
                    im_ = im[int(it / n_frames * 256)].squeeze()
                    im_ = np.rot90(im_, 3)
                    im_ = np.reshape(im_, (n_nodes_per_axis * n_nodes_per_axis))
                    if "permutation" in model_config.field_type:
                        im_ = im_[permutation_indices]
                    A1[:, 0:1] = torch.tensor(
                        im_[:, None], dtype=torch.float32, device=device
                    )
                if ("visual" in field_type) & (it >= 0):
                    im_ = im[int(it / n_frames * 256)].squeeze()
                    im_ = np.rot90(im_, 3)
                    im_ = np.reshape(im_, (n_nodes_per_axis * n_nodes_per_axis))
                    A1[:n_nodes, 0:1] = torch.tensor(
                        im_[:, None], dtype=torch.float32, device=device
                    )
                    A1[n_nodes:n_neurons, 0:1] = 1

                    # plt.scatter(to_numpy(X1_mesh[:, 1]), to_numpy(X1_mesh[:, 0]), s=40, c=to_numpy(A1), cmap='grey', vmin=0,vmax=1)
                if "excitation_single" in field_type:
                    parts = field_type.split("_")
                    period = int(parts[-2])
                    amplitude = float(parts[-1])
                    if (it - 100) % period == 0:
                        H1[0, 0] = H1[0, 0] + torch.tensor(
                            amplitude, dtype=torch.float32, device=device
                        )

                x = torch.concatenate(
                    (
                        N1.clone().detach(),
                        X1.clone().detach(),
                        V1.clone().detach(),
                        T1.clone().detach(),
                        H1.clone().detach(),
                        A1.clone().detach(),
                        U1.clone().detach(),
                    ),
                    1,
                )
                X[:, it] = H1[:, 0].clone().detach()
                dataset = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index)

                # model prediction
                if ("modulation" in field_type) & (it >= 0):
                    y = model(dataset, has_field=True)
                elif ("visual" in field_type) & (it >= 0):
                    y = model(dataset, has_field=True)
                elif "PDE_N3" in model_config.signal_model_name:
                    y = model(dataset, has_field=False, alpha=it / n_frames)
                elif "PDE_N6" in model_config.signal_model_name:
                    (y,p,) = model(dataset, has_field=False)
                elif "PDE_N7" in model_config.signal_model_name:
                    (y,p,) = model(dataset, has_field=False)
                else:
                    y = model(dataset, has_field=False)

            # append list
            if (it >= 0) & bSave:
                x_list.append(to_numpy(x))
                y_list.append(to_numpy(y))

            # Particle update
            if (config.graph_model.signal_model_name == "PDE_N6") | (
                config.graph_model.signal_model_name == "PDE_N7"
            ):
                H1[:, 1] = y.squeeze()
                H1[:, 0] = H1[:, 0] + H1[:, 1] * delta_t
                if noise_model_level > 0:
                    H1[:, 0] = (
                        H1[:, 0]
                        + torch.randn(n_neurons, device=device) * noise_model_level
                    )
                H1[:, 3] = p.squeeze()
                H1[:, 2] = torch.relu(H1[:, 2] + H1[:, 3] * delta_t)

            else:
                H1[:, 1] = y.squeeze()
                H1[:, 0] = H1[:, 0] + H1[:, 1] * delta_t
                if noise_model_level > 0:
                    H1[:, 0] = (
                        H1[:, 0]
                        + torch.randn(n_neurons, device=device) * noise_model_level
                    )

            # print(f"Total allocated memory: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB")
            # print(f"Total reserved memory:  {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")

            # output plots
            if visualize & (run == run_vizualized) & (it % step == 0) & (it >= 0):
                if "latex" in style:
                    plt.rcParams["text.usetex"] = True
                    rc("font", **{"family": "serif", "serif": ["Palatino"]})

                matplotlib.rcParams["savefig.pad_inches"] = 0
                num = f"{id_fig:06}"
                id_fig += 1

                if "visual" in field_type:
                    fig = plt.figure(figsize=(8, 8))
                    plt.axis("off")
                    plt.subplot(211)
                    plt.axis("off")
                    plt.title("$b_i$", fontsize=24)
                    plt.scatter(
                        to_numpy(X1[0:1024, 1]) * 0.95,
                        to_numpy(X1[0:1024, 0]) * 0.95,
                        s=15,
                        c=to_numpy(A1[0:1024, 0]),
                        cmap="viridis",
                        vmin=0,
                        vmax=2,
                    )
                    plt.scatter(
                        to_numpy(X1[1024:, 1]) * 0.95 + 0.2,
                        to_numpy(X1[1024:, 0]) * 0.95,
                        s=15,
                        c=to_numpy(A1[1024:, 0]),
                        cmap="viridis",
                        vmin=-4,
                        vmax=4,
                    )
                    # cbar = plt.colorbar()
                    # cbar.ax.yaxis.set_tick_params(labelsize=8)
                    plt.xticks([])
                    plt.yticks([])
                    plt.subplot(212)
                    plt.axis("off")
                    plt.title("$x_i$", fontsize=24)
                    plt.scatter(
                        to_numpy(X1[0:1024, 1]),
                        to_numpy(X1[0:1024, 0]),
                        s=15,
                        c=to_numpy(H1[0:1024, 0]),
                        cmap="viridis",
                        vmin=-10,
                        vmax=10,
                    )
                    plt.scatter(
                        to_numpy(X1[1024:, 1]) + 0.2,
                        to_numpy(X1[1024:, 0]),
                        s=15,
                        c=to_numpy(H1[1024:, 0]),
                        cmap="viridis",
                        vmin=-10,
                        vmax=10,
                    )
                    # cbar = plt.colorbar()
                    # cbar.ax.yaxis.set_tick_params(labelsize=8)
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(f"graphs_data/{dataset_name}/Fig/Fig_{run}_{num}.png", dpi=80)
                    plt.close()

                elif "modulation" in field_type:
                    fig = plt.figure(figsize=(12, 12))
                    plt.subplot(221)
                    plt.scatter(
                        to_numpy(X1[:, 1]),
                        to_numpy(X1[:, 0]),
                        s=100,
                        c=to_numpy(A1[:, 0]),
                        cmap="viridis",
                        vmin=0,
                        vmax=2,
                    )
                    plt.subplot(222)
                    plt.scatter(
                        to_numpy(X1[:, 1]),
                        to_numpy(X1[:, 0]),
                        s=100,
                        c=to_numpy(H1[:, 0]),
                        cmap="viridis",
                        vmin=-5,
                        vmax=5,
                    )
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(
                        f"graphs_data/{dataset_name}/Fig/Fig_{run}_{num}.png", dpi=80
                    )
                    plt.close()

                else:
                    if ("PDE_N6" in model_config.signal_model_name) | (
                        "PDE_N7" in model_config.signal_model_name
                    ):
                        plt.figure(figsize=(12, 5.6))
                        plt.axis("off")
                        plt.axis("off")
                        plt.subplot(121)
                        plt.title("activity $x_i$", fontsize=24)
                        plt.scatter(
                            to_numpy(X1[:, 0]),
                            to_numpy(X1[:, 1]),
                            s=200,
                            c=to_numpy(x[:, 6]),
                            cmap="viridis",
                            vmin=-5,
                            vmax=5,
                            edgecolors="k",
                            alpha=1,
                        )
                        cbar = plt.colorbar()
                        cbar.ax.yaxis.set_tick_params(labelsize=12)
                        plt.xticks([])
                        plt.yticks([])
                        plt.subplot(122)
                        plt.title("short term plasticity $y_i$", fontsize=24)
                        plt.scatter(
                            to_numpy(X1[:, 0]),
                            to_numpy(X1[:, 1]),
                            s=200,
                            c=to_numpy(x[:, 8]),
                            cmap="grey",
                            vmin=0,
                            vmax=1,
                            edgecolors="k",
                            alpha=1,
                        )
                        cbar = plt.colorbar()
                        cbar.ax.yaxis.set_tick_params(labelsize=12)
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(
                            f"graphs_data/{dataset_name}/Fig/Fig_{run}_{num}.tif",
                            dpi=170,
                        )
                        plt.close()

                    else:
                        plt.figure(figsize=(10, 10))
                        # plt.scatter(to_numpy(X1[:, 0]), to_numpy(X1[:, 1]), s=10, c=to_numpy(x[:, 6]),
                        #             cmap='viridis', vmin=-10, vmax=10, edgecolors='k', alpha=1)
                        plt.axis("off")
                        plt.scatter(
                            to_numpy(X1[:, 0]),
                            to_numpy(X1[:, 1]),
                            s=100,
                            c=to_numpy(x[:, 6]),
                            cmap="viridis",
                            vmin=-40,
                            vmax=40,
                        )
                        # cbar = plt.colorbar()
                        # cbar.ax.yaxis.set_tick_params(labelsize=8)
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(
                            f"graphs_data/{dataset_name}/Fig/Fig_{run}_{num}.tif",
                            dpi=170,
                        )
                        plt.close()

                        im_ = imread(
                            f"graphs_data/{dataset_name}/Fig/Fig_{run}_{num}.tif"
                        )
                        plt.figure(figsize=(10, 10))
                        plt.imshow(im_)
                        plt.xticks([])
                        plt.yticks([])
                        plt.subplot(3, 3, 1)
                        plt.imshow(im_[800:1000, 800:1000, :])
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(
                            f"graphs_data/{dataset_name}/Fig/Fig_{run}_{num}.png",
                            dpi=80,
                        )
                        plt.close()

        print(f"generated {len(x_list)} frames total")

        if visualize & (run == run_vizualized):
            print('generating lossless video ...')

            config_indices = dataset_name.split('signal_')[1] if 'signal_' in dataset_name else 'no_id'
            src = f"./graphs_data/{dataset_name}/Fig/Fig_0_000000.png"
            dst = f"./graphs_data/{dataset_name}/input_{config_indices}.png"
            with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
                fdst.write(fsrc.read())

            generate_lossless_video_ffv1(output_dir=f"./graphs_data/{dataset_name}", run=run,
                                         config_indices=config_indices, framerate=20)
            generate_lossless_video_libx264(output_dir=f"./graphs_data/{dataset_name}", run=run,
                                            config_indices=config_indices, framerate=20)
            generate_compressed_video_mp4(output_dir=f"./graphs_data/{dataset_name}", run=run,
                                          config_indices=config_indices, framerate=20)

            files = glob.glob(f'./graphs_data/{dataset_name}/Fig/*')
            for f in files:
                os.remove(f)

            print('Ising analysis ...')
            x_list = np.array(x_list)
            y_list = np.array(y_list)

        if bSave:
            x_list = np.array(x_list)
            y_list = np.array(y_list)
            # torch.save(x_list, f'graphs_data/{dataset_name}/x_list_{run}.pt')
            np.save(f"graphs_data/{dataset_name}/x_list_{run}.npy", x_list)
            if has_particle_dropout:
                torch.save(
                    x_removed_list,
                    f"graphs_data/{dataset_name}/x_removed_list_{run}.pt",
                )
                np.save(
                    f"graphs_data/{dataset_name}/particle_dropout_mask.npy",
                    particle_dropout_mask,
                )
                np.save(
                    f"graphs_data/{dataset_name}/inv_particle_dropout_mask.npy",
                    inv_particle_dropout_mask,
                )
            # torch.save(y_list, f'graphs_data/{dataset_name}/y_list_{run}.pt')
            np.save(f"graphs_data/{dataset_name}/y_list_{run}.npy", y_list)
            torch.save(model.p, f"graphs_data/{dataset_name}/model_p.pt")
            print("data saved ...")

        if measurement_noise_level > 0:
            np.save(f"graphs_data/{dataset_name}/raw_x_list_{run}.npy", x_list)
            np.save(f"graphs_data/{dataset_name}/raw_y_list_{run}.npy", y_list)

            for k in range(x_list.shape[0]):
                x_list[k, :, 6] = x_list[k, :, 6] + np.random.normal(
                    0, measurement_noise_level, x_list.shape[1]
                )
            for k in range(1, x_list.shape[0] - 1):
                y_list[k] = (x_list[k + 1, :, 6:7] - x_list[k, :, 6:7]) / delta_t

            np.save(f"graphs_data/{dataset_name}/x_list_{run}.npy", x_list)
            np.save(f"graphs_data/{dataset_name}/y_list_{run}.npy", y_list)

        activity = torch.tensor(x_list[:, :, 6:7], device=device)
        activity = activity.squeeze()
        activity = activity.t()
        plt.figure(figsize=(15, 10))
        ax = sns.heatmap(
            to_numpy(activity), center=0, cmap="viridis", cbar_kws={"fraction": 0.046}
        )
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=32)
        ax.invert_yaxis()
        plt.ylabel("neurons", fontsize=64)
        plt.xlabel("time", fontsize=64)
        plt.xticks([10000, 99000], [10000, 100000], fontsize=48)
        plt.yticks([0, 999], [1, 1000], fontsize=48)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"graphs_data/{dataset_name}/kinograph.png", dpi=300)
        plt.close()

        plt.figure(figsize=(15, 10))
        if n_neurons > 2:
            n = np.random.permutation(n_neurons)
            NN = 25
        else:
            n = np.arange(n_neurons)
            NN = 2
        for i in range(NN):
            plt.plot(to_numpy(activity[n[i].astype(int), :]), linewidth=2)
        plt.xlabel("time", fontsize=64)
        plt.ylabel("$v_{i}$", fontsize=64)
        # plt.xticks([10000, 99000], [10000, 100000], fontsize=48)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.tight_layout()
        plt.savefig(f"graphs_data/{dataset_name}/activity.png", dpi=300)
        plt.xlim([0, 1000])
        plt.tight_layout()
        plt.savefig(f"graphs_data/{dataset_name}/activity_1000.png", dpi=300)
        plt.close()

        # torch.cuda.memory_allocated(device)
        # gc.collect()
        # torch.cuda.empty_cache()
        # print(f"Total allocated memory: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB")
        # print(f"Total reserved memory:  {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")


