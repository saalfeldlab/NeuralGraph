import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric as pyg
from matplotlib import rc
from NeuralGraph.generators.davis import AugmentedDavis
from NeuralGraph.generators.graph_data_generator import assign_columns_from_uv as _assign_columns_from_uv, mseq_bits as _mseq_bits
from NeuralGraph.generators.utils import generate_compressed_video_mp4
from NeuralGraph.data_loaders import (
    data_generate_synaptic,
    data_generate_particle,
    load_solar_system,
    load_RGB_grid_data,
    load_wormvae_data,
)
from NeuralGraph.utils import to_numpy, get_datavis_root_dir
from NeuralGraph.sparsify import sparse_ising_fit_fast
from tqdm import tqdm
import os


# === End helpers ===


# from fa2_modified import ForceAtlas2
# import h5py as h5
# import zarr
# import xarray as xr
# import taichi as ti

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
        generate_from_data(config=config, device=device, visualize=visualize)
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
    else:
        data_generate_particle(
            config,
            visualize=visualize,
            run_vizualized=run_vizualized,
            style=style,
            erase=erase,
            step=step,
            alpha=0.2,
            ratio=ratio,
            scenario=scenario,
            device=device,
            bSave=bSave,
        )

    plt.style.use("default")


def generate_from_data(config, device, visualize=True, step=None, cmap=None):
    data_folder_name = config.data_folder_name
    image_data = config.image_data

    if data_folder_name == "graphs_data/solar_system":
        load_solar_system(config, device, visualize, step)
    elif "RGB" in config.graph_model.particle_model_name:
        load_RGB_grid_data(config, device, visualize, step)
    elif "LG-ODE" in data_folder_name:
        raise NotImplementedError("load_LG_ODE not yet implemented")
    elif "WaterDropSmall" in data_folder_name:
        raise NotImplementedError("load_WaterDropSmall not yet implemented")
    elif "WaterRamps" in data_folder_name:
        raise NotImplementedError("load_Goole_data not yet implemented")
    elif "MultiMaterial" in data_folder_name:
        raise NotImplementedError("load_Goole_data not yet implemented")
    elif "Kato" in data_folder_name:
        raise NotImplementedError("load_worm_Kato_data not yet implemented")
    elif "wormvae" in data_folder_name:
        load_wormvae_data(config, device, visualize, step)
    elif "NeuroPAL" in data_folder_name:
        raise NotImplementedError("load_neuropal_data not yet implemented")
    elif "U2OS" in data_folder_name:
        raise NotImplementedError("load_2Dfluo_data_on_mesh not yet implemented")
    elif "cardio" in data_folder_name:
        raise NotImplementedError("load_2Dgrid_data not yet implemented")
    elif image_data.file_type != "none":
        if image_data.file_type == "3D fluo Cellpose":
            raise NotImplementedError("load_3Dfluo_data_with_Cellpose not yet implemented")
        if image_data.file_type == "2D fluo Cellpose":
            raise NotImplementedError("load_2Dfluo_data_with_Cellpose not yet implemented")
    else:
        raise ValueError(f"Unknown data folder name {data_folder_name}")



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
    noise_model_level = simulation_config.noise_model_level

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
        datavis_root = os.path.join(get_datavis_root_dir(), "JPEGImages/480p")
        assert os.path.exists(datavis_root)
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



                if "tile_mseq" in visual_input_type:
                    # Build (once) per-column sequences and mapping from photoreceptors to columns
                    if it == simulation_config.start_frame:
                        # default: 217 columns unless specified
                        n_columns = getattr(simulation_config, "n_columns", 217)
                        # group photoreceptors into columns using k-means on (u,v)
                        global __tile_labels, __tile_codes, __tile_idx, __tile_contrast, __tile_period
                        __tile_labels = _assign_columns_from_uv(u_coords, v_coords, n_columns, random_state=simulation_config.seed)
                        # per-column m-seq (length 255) with random phase
                        base = _mseq_bits(p=8, seed=simulation_config.seed)
                        phases = np.random.RandomState(simulation_config.seed).randint(0, len(base), size=n_columns)
                        __tile_codes = np.stack([np.roll(base, ph) for ph in phases], axis=0)  # (n_columns, 255)
                        __tile_idx = 0
                        __tile_period = __tile_codes.shape[1]
                        # mid contrast by default; sweep externally by changing this param
                        __tile_contrast = float(getattr(simulation_config, "tile_contrast", 0.2))
                    # write per-frame luminance to x[:,4] (photoreceptors only)
                    col_vals = __tile_codes[:, __tile_idx % __tile_period]  # ±1
                    # project to [0,1]: 0.5 + c*0.5*bit
                    col_vals = 0.5 + (__tile_contrast * 0.5) * col_vals.astype(np.float32)
                    # broadcast to photoreceptors based on labels
                    x[:n_input_neurons, 4] = torch.from_numpy(col_vals[__tile_labels]).to(x.device, dtype=x.dtype)
                    __tile_idx += 1
                # --- End "tile_mseq" ---


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
                                        ax_ca.text(0.5, 0.5, 'No neurons', transform=ax_ca.transAxes, ha='center',
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
                                    ax.scatter(to_numpy(X1[:n_input_neurons, 0]),
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
                                        ax.scatter(hex_positions_x, hex_positions_y, s=72, c=type_voltages,
                                                                    cmap='viridis', vmin=-2, vmax=2, marker='h', alpha=1,
                                                                    linewidths=0.0, edgecolors='black')
                                        if type_name.startswith('R'):
                                            pass
                                        elif type_name.startswith(('L', 'Lawf')):
                                            pass
                                        elif type_name.startswith(('Mi', 'Tm', 'TmY')):
                                            pass
                                        elif type_name.startswith('T'):
                                            pass
                                        elif type_name.startswith('C'):
                                            pass
                                        else:
                                            pass
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

        output_name = dataset_name.split('fly_N9_')[1] if 'fly_N9_' in dataset_name else 'no_id'
        src = f"./graphs_data/{dataset_name}/Fig/Fig_0_000000.png"
        dst = f"./graphs_data/{dataset_name}/input_{output_name}.png"
        with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
            fdst.write(fsrc.read())


        generate_compressed_video_mp4(output_dir=f"./graphs_data/{dataset_name}", run=run,
                                      output_name=output_name,framerate=20)

        files = glob.glob(f'./graphs_data/{dataset_name}/Fig/*')
        for f in files:
            os.remove(f)

    print('Ising analysis ...')
    x_list = np.array(x_list)
    y_list = np.array(y_list)

    energy_stride = 1
    s, h, J, E = sparse_ising_fit_fast(x=x_list, voltage_col=3, top_k=50, block_size=2000,
                                       energy_stride=energy_stride)

    fig = plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, (1,2))
    plt.plot(np.arange(0, len(E) * energy_stride, energy_stride), E, lw=0.5)
    plt.xlabel("Frame", fontsize=18)
    plt.ylabel("Energy", fontsize=18)
    plt.title("Ising Energy Over Frames", fontsize=18)
    plt.xlim(0, 600)
    plt.xticks(np.arange(0, 601, 100), fontsize=12)  # Only show every 100 frames
    plt.yticks(fontsize=12)

    plt.subplot(2, 2, 3)
    plt.hist(E, bins=200, density=True)
    plt.xlabel("Energy", fontsize=18)
    plt.ylabel("Density", fontsize=18)
    plt.title("Energy Distribution (full range)", fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.subplot(2, 2, 4)
    plt.hist(E, bins=200, density=True)
    plt.xlabel("Energy", fontsize=18)
    plt.ylabel("Density", fontsize=18)
    plt.xlim([-1000, 1000])
    plt.title("Energy Distribution (fixed range)", fontsize=18)
    plt.xticks(np.arange(-1000, 1001, 500), fontsize=12)  # -1000, -500, 0, 500, 1000
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.savefig(f"graphs_data/{dataset_name}/E_panels.png", dpi=150)
    plt.close(fig)

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



