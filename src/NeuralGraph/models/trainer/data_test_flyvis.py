"""Extracted from graph_trainer.py"""
from ._imports import *

def data_test_flyvis(config, visualize=True, style="color", verbose=False, best_model=None, step=5, test_mode='', new_params = None, device=None):


    if "black" in style:
        plt.style.use("dark_background")
        mc = 'w'
    else:
        plt.style.use("default")
        mc = 'k'

    simulation_config = config.simulation
    training_config = config.training
    model_config = config.graph_model

    log_dir = 'log/' + config.config_file

    torch.random.fork_rng(devices=device)
    if simulation_config.seed is not None:
        torch.random.manual_seed(simulation_config.seed)
        np.random.seed(simulation_config.seed)

    print(
        f"testing... {model_config.particle_model_name} {model_config.mesh_model_name} seed: {simulation_config.seed}")

    dataset_name = config.dataset

    training_selected_neurons = training_config.training_selected_neurons
    if training_selected_neurons:
        n_neurons = 13741
        n_neuron_types = 1736
    else:
        n_neurons = simulation_config.n_neurons
        n_neuron_types = simulation_config.n_neuron_types
    n_input_neurons = simulation_config.n_input_neurons
    delta_t = simulation_config.delta_t
    n_frames = simulation_config.n_frames
    field_type = model_config.field_type
    signal_model_name = model_config.signal_model_name


    ensemble_id = simulation_config.ensemble_id
    model_id = simulation_config.model_id

    measurement_noise_level = training_config.measurement_noise_level
    noise_model_level = training_config.noise_model_level
    warm_up_length = 100

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

    pde = PDE_N9(p=p, f=torch.nn.functional.relu, params=simulation_config.params, model_type=model_config.signal_model_name, n_neuron_types=n_neuron_types, device=device)
    pde_modified = PDE_N9(p=copy.deepcopy(p), f=torch.nn.functional.relu, params=simulation_config.params, model_type=model_config.signal_model_name, n_neuron_types=n_neuron_types, device=device)


    if 'RNN' in signal_model_name:
        model = Signal_Propagation_RNN(aggr_type=model_config.aggr_type, config=config, device=device)
    elif 'LSTM' in signal_model_name:
        model = Signal_Propagation_LSTM(aggr_type=model_config.aggr_type, config=config, device=device)
    elif 'MLP_ODE' in signal_model_name:
        model = Signal_Propagation_MLP_ODE(aggr_type=model_config.aggr_type, config=config, device=device)
    elif 'MLP' in signal_model_name:
        model = Signal_Propagation_MLP(aggr_type=model_config.aggr_type, config=config, device=device)
    else:
        model = Signal_Propagation_FlyVis(aggr_type=model_config.aggr_type, config=config, device=device)


    if best_model == 'best':
        files = glob.glob(f"{log_dir}/models/*")
        files.sort(key=sort_key)
        filename = files[-1]
        filename = filename.split('/')[-1]
        filename = filename.split('graphs')[-1][1:-3]
        best_model = filename
        print(f'best model: {best_model}')
    netname = f"{log_dir}/models/best_model_with_0_graphs_{best_model}.pt"
    print(f'load {netname} ...')
    state_dict = torch.load(netname, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])

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

    if training_selected_neurons:
        selected_neuron_ids =  training_config.selected_neuron_ids
        x_selected = torch.zeros(len(selected_neuron_ids), 9, dtype=torch.float32, device=device)
        selected_neuron_ids = np.array(selected_neuron_ids).astype(int)
        print(f'testing single neuron id {selected_neuron_ids} ...')
        x_selected[:, 1:3] = X1[selected_neuron_ids,:]
        x_selected[:, 0] = torch.arange(1, dtype=torch.float32)
        x_selected[:, 3] = initial_state[selected_neuron_ids]
        x_selected[:, 4] = net.stimulus().squeeze()[selected_neuron_ids]
        x_selected[:, 5] = torch.tensor(grouped_types[selected_neuron_ids], dtype=torch.float32, device=device)
        x_selected[:, 6] = torch.tensor(node_types_int[selected_neuron_ids], dtype=torch.float32, device=device)
        x_selected[:, 7] = torch.rand(1, dtype=torch.float32, device=device)
        x_selected[:, 8] = calcium_alpha * x_selected[0, 7] + calcium_beta

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

    if 'full' in test_mode:
        target_frames = 90000
        step = 25000
    else:
        target_frames = 2000
        step = 10
    print(f'plot activity frames \033[92m0-{target_frames}...\033[0m')

    dataset_length = len(stimulus_dataset)
    frames_per_sequence = 35
    total_frames_per_pass = dataset_length * frames_per_sequence
    num_passes_needed = (target_frames // total_frames_per_pass) + 1

    y_list = []
    x_list = []
    x_generated_list = []
    x_generated_modified_list = []
    rmserr_list = []

    x_generated = x.clone()
    x_generated_modified = x.clone()

    # Initialize RNN hidden state
    if 'RNN' in signal_model_name:
        h_state = None
    if 'LSTM' in signal_model_name:
        h_state = None
        c_state = None

    it = simulation_config.start_frame
    id_fig = 0

    tile_labels = None
    tile_codes_torch = None
    tile_period = None
    tile_idx = 0
    tile_contrast = simulation_config.tile_contrast
    n_columns = n_input_neurons // 8
    tile_seed = simulation_config.seed

    edges = torch.load(f'./graphs_data/{dataset_name}/edge_index.pt', map_location=device)
    mask = torch.arange(edges.shape[1])

    if ('test_ablation' in test_mode) & (not('MLP' in signal_model_name)) & (not('RNN' in signal_model_name)) & (not('LSTM' in signal_model_name)):
        #  test_mode="test_ablation_100"
        ablation_ratio = int(test_mode.split('_')[-1]) / 100
        if ablation_ratio > 0:
            print(f'\033[93mtest ablation ratio {ablation_ratio} \033[0m')
        n_ablation = int(edges.shape[1] * ablation_ratio)
        index_ablation = np.random.choice(np.arange(edges.shape[1]), n_ablation, replace=False)

        with torch.no_grad():
            pde.p['w'][index_ablation] = 0
            pde_modified.p['w'][index_ablation] = 0
            model.W[index_ablation] = 0

    if 'test_modified' in test_mode:
        noise_W = float(test_mode.split('_')[-1])
        if noise_W > 0:
            print(f'\033[93mtest modified W with noise level {noise_W} \033[0m')
            noise_p_W = torch.randn_like(pde.p['w']) * noise_W # + torch.ones_like(pde.p['w'])
            pde_modified.p['w'] = pde.p['w'].clone() + noise_p_W

        plot_weight_comparison(pde.p['w'], pde_modified.p['w'], f"./{log_dir}/results/weight_comparison_{noise_W}.png")


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


    # MAIN LOOP #####################################

    with torch.no_grad():
        for pass_num in range(num_passes_needed):
            for data_idx, data in enumerate(tqdm(stimulus_dataset, desc="processing stimulus data", ncols=150)):

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

                    x_generated[:,4] = x[:,4]
                    dataset = pyg.data.Data(x=x_generated, pos=x_generated[:, 1:3], edge_index=edge_index)
                    y_generated = pde(dataset, has_field=False)

                    x_generated_modified[:,4] = x[:,4]
                    dataset_modified = pyg.data.Data(x=x_generated_modified, pos=x_generated_modified[:, 1:3], edge_index=edge_index)
                    y_generated_modified = pde_modified(dataset_modified, has_field=False)

                    if 'visual' in field_type:
                        visual_input = model.forward_visual(x, it)
                        x[:model.n_input_neurons, 4:5] = visual_input
                        x[model.n_input_neurons:, 4:5] = 0

                    # Prediction step
                    if training_selected_neurons:
                        x_selected[:,4] = x[:,4][selected_neuron_ids].clone().detach()
                        if 'RNN' in signal_model_name:
                            y, h_state = model(x_selected, h=h_state, return_all=True)
                        elif 'LSTM' in signal_model_name:
                            y, h_state, c_state = model(x_selected, h=h_state, c=c_state, return_all=True)
                        elif 'MLP_ODE' in signal_model_name:
                            v = x_selected[:, 3:4]
                            I = x_selected[:, 4:5]
                            y = model.rollout_step(v, I, dt=delta_t, method='rk4') - v  # Return as delta
                        elif 'MLP' in signal_model_name:
                            y = model(x_selected, data_id=None, mask=None, return_all=False)

                    else:
                        if 'RNN' in signal_model_name:
                            y, h_state = model(x, h=h_state, return_all=True)
                        elif 'LSTM' in signal_model_name:
                            y, h_state, c_state = model(x, h=h_state, c=c_state, return_all=True)
                        elif 'MLP_ODE' in signal_model_name:
                            v = x[:, 3:4]
                            I = x[:n_input_neurons, 4:5]
                            y = model.rollout_step(v, I, dt=delta_t, method='rk4') - v  # Return as delta
                        elif 'MLP' in signal_model_name:
                            y = model(x, data_id=None, mask=None, return_all=False)
                        else:
                            dataset = pyg.data.Data(x=x, pos=x, edge_index=edge_index)
                            data_id = torch.zeros((x.shape[0], 1), dtype=torch.int, device=device)
                            y = model(dataset, data_id, mask, False)

                    # Save states
                    x_generated_list.append(to_numpy(x_generated.clone().detach()))
                    x_generated_modified_list.append(to_numpy(x_generated_modified.clone().detach()))

                    if training_selected_neurons:
                        x_list.append(to_numpy(x_selected.clone().detach()))
                    else:
                        x_list.append(to_numpy(x.clone().detach()))

                    # Integration step
                    if noise_model_level > 0:
                        x_generated[:, 3:4] = x_generated[:, 3:4] + delta_t * y_generated + torch.randn((n_neurons, 1), dtype=torch.float32, device=device) * noise_model_level
                        x_generated_modified[:, 3:4] = x_generated_modified[:, 3:4] + delta_t * y_generated_modified + torch.randn((n_neurons, 1), dtype=torch.float32, device=device) * noise_model_level
                    else:
                        x_generated[:, 3:4] = x_generated[:, 3:4] + delta_t * y_generated
                        x_generated_modified[:, 3:4] = x_generated_modified[:, 3:4] + delta_t * y_generated_modified

                    if training_selected_neurons:
                        if 'MLP_ODE' in signal_model_name:
                            x_selected[:, 3:4] = x_selected[:, 3:4] + y  # y already contains full update
                        else:
                            x_selected[:, 3:4] = x_selected[:, 3:4] + delta_t * y
                        if (it <= warm_up_length) and ('RNN' in signal_model_name or 'LSTM' in signal_model_name):
                            x_selected[:, 3:4] = x_generated[selected_neuron_ids, 3:4].clone()
                    else:
                        if 'MLP_ODE' in signal_model_name:
                            x[:, 3:4] = x[:, 3:4] + y  # y already contains full update
                        else:
                            x[:, 3:4] = x[:, 3:4] + delta_t * y
                        if (it <= warm_up_length) and ('RNN' in signal_model_name):
                            x[:, 3:4] = x_generated[:, 3:4].clone()

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

                    if (it>0) & (it<100) & (it % step == 0) & visualize & (not training_selected_neurons):
                        if "latex" in style:
                            plt.rcParams["text.usetex"] = True
                            rc("font", **{"family": "serif", "serif": ["Palatino"]})

                        mpl.rcParams["savefig.pad_inches"] = 0
                        num = f"{id_fig:06}"
                        id_fig += 1

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
                            plt.savefig(f"./{log_dir}/tmp_recons/Fig_{run}_{num}.png", dpi=80)
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
                            plt.savefig(f"./{log_dir}/tmp_recons/Fig_{run}_{num}.png", dpi=80)
                            plt.close()

                    it = it + 1
                    if it >= target_frames:
                        break
                if it >= target_frames:
                    break
            if it >= target_frames:
                break
    print(f"generated {len(x_list)} frames total")

    if (False):
        print('generating lossless video ...')

        config_indices = dataset_name.split('fly_N9_')[1] if 'fly_N9_' in dataset_name else 'no_id'
        src = f"./{log_dir}/tmp_recons/Fig_0_000000.png"
        dst = f"./{log_dir}/results/input_{config_indices}.png"
        with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
            fdst.write(fsrc.read())

        # generate_lossless_video_ffv1(output_dir=f"./{log_dir}/results", run=run, config_indices=config_indices,framerate=20)
        # generate_lossless_video_libx264(output_dir=f"./{log_dir}/results", run=run,
        #                                 config_indices=config_indices,framerate=20)
        generate_compressed_video_mp4(output_dir=f"./{log_dir}/results", run=run,
                                        config_indices=config_indices,framerate=20)

        # files = glob.glob(f'./{log_dir}/tmp_recons/*')
        # for f in files:
        #     os.remove(f)

    x_list = np.array(x_list)
    x_generated_list = np.array(x_generated_list)
    x_generated_modified_list = np.array(x_generated_modified_list)
    y_list = np.array(y_list)


    if calcium_type != "none":
        # Use calcium (index 7)
        activity_true = x_generated_list[:, :, 7].squeeze().T  # (n_neurons, n_frames)
        activity_pred = x_list[:, :, 7].squeeze().T
        activity_label = "calcium"
        y_lim = [0, 3]
    else:
        # Use voltage (index 3)
        activity_true = x_generated_list[:, :, 3].squeeze().T
        visual_input_true = x_generated_list[:, :, 4].squeeze().T
        activity_true_modified = x_generated_modified_list[:, :, 3].squeeze().T
        activity_pred = x_list[:, :, 3].squeeze().T
        activity_label = "voltage"
        y_lim = [-3, 3]


    start_frame = 0
    end_frame = target_frames


    if training_selected_neurons:
        print(f"evaluating on selected neurons only: {selected_neuron_ids}")
        x_generated_list = x_generated_list[:, selected_neuron_ids, :]
        x_generated_modified_list = x_generated_modified_list[:, selected_neuron_ids, :]
        neuron_types = neuron_types[selected_neuron_ids]

        plt.style.use('default')

        fig, ax = plt.subplots(1, 1, figsize=(12, 18))

        true_slice = activity_true[selected_neuron_ids, start_frame:end_frame]
        visual_input_slice = visual_input_true[selected_neuron_ids, start_frame:end_frame]
        pred_slice = activity_pred[start_frame:end_frame]
        if len(selected_neuron_ids)==1:
            pred_slice = pred_slice[None,:]
        step_v = 2.5
        lw = 4

        # Plot ground truth (green, thick)
        for i in trange(len(selected_neuron_ids),ncols=50):
            baseline = np.mean(true_slice[i])
            ax.plot(true_slice[i] - baseline + i * step_v, linewidth=lw, c='green', alpha=0.5,
                    label='ground truth' if i == 0 else None)
            if visual_input_slice[i].mean() > 0:
                ax.plot(visual_input_slice[i] - baseline + i * step_v, linewidth=2, c='blue', alpha=0.5,
                        label='visual input' if i == 0 else None)
            ax.plot(pred_slice[i] - baseline + i * step_v, linewidth=2, c='black',
                    label='prediction' if i == 0 else None)

        # Add neuron type labels on left
        for i in range(len(selected_neuron_ids)):
            type_idx = int(to_numpy(x[selected_neuron_ids[i], 6]).item())
            ax.text(-50, i * step_v, f'{index_to_name[type_idx]}',
                    fontsize=18, va='bottom', ha='right')

            if len(selected_neuron_ids) <= 20:
                ax.text(-50, i * step_v - 0.3, f'{selected_neuron_ids[i]}',
                        fontsize=12, va='top', ha='right', color='black')

        ax.set_ylim([-step_v, len(selected_neuron_ids) * step_v])
        ax.set_yticks([])
        ax.set_xlabel('frame', fontsize=20)


        # Remove unnecessary spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # Add legend
        ax.legend(loc='upper right', fontsize=14)

        for i in range(len(selected_neuron_ids)):
            # Calculate metrics for this neuron
            true_i = true_slice[i]
            pred_i = pred_slice[i]

            # Remove NaNs for correlation
            valid_mask = ~(np.isnan(true_i) | np.isnan(pred_i))
            if valid_mask.sum() > 1:
                rmse = np.sqrt(np.mean((true_i[valid_mask] - pred_i[valid_mask])**2))
                pearson_r, _ = pearsonr(true_i[valid_mask], pred_i[valid_mask])
            else:
                rmse = np.nan
                pearson_r = np.nan

            # Add text on the right side
            ax.text(end_frame - start_frame + 10, i * step_v,
                    f'RMSE: {rmse:.3f}\nr: {pearson_r:.3f}',
                    fontsize=10, va='center', ha='left',
                    color='black')

        # Adjust x-axis limit to make room for metrics
        ax.set_xlim([0, end_frame - start_frame + 100])

        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/rollout_traces_selected_neurons.png", dpi=80, bbox_inches='tight')

        plt.close()

        rmse_sel, pearson_sel, _ = compute_trace_metrics(true_slice, pred_slice, "Selected Neurons")
        np.save(f"./{log_dir}/results/rmse_selected_neurons.npy", rmse_sel)
        np.save(f"./{log_dir}/results/pearson_selected_neurons.npy", pearson_sel)


        return

    rmse_all, pearson_all, _ = compute_trace_metrics(activity_true, activity_pred, "All Neurons")
    np.save(f"./{log_dir}/results/rmse_per_neuron.npy", rmse_all)
    np.save(f"./{log_dir}/results/pearson_per_neuron.npy", pearson_all)


    if 'full' in test_mode:

        print('computing overall activity range statistics...')
        activity_range_per_neuron = np.max(activity_true, axis=1) - np.min(activity_true, axis=1)

        activity_mean = np.mean(activity_true)
        activity_std = np.std(activity_true)
        activity_min = np.min(activity_true)
        activity_max = np.max(activity_true)
        activity_range_mean = np.mean(activity_range_per_neuron)
        activity_range_std = np.std(activity_range_per_neuron)

        print(f"overall activity statistics:")
        print(f"  mean: {activity_mean:.6f}")
        print(f"  std: {activity_std:.6f}")
        print(f"  min: {activity_min:.6f}")
        print(f"  max: {activity_max:.6f}")
        print(f"  global range: {activity_max - activity_min:.6f}")
        print(f"  per-neuron range - mean: {activity_range_mean:.6f}, std: {activity_range_std:.6f}")



    # Add at the end of data_test_flyvis, after the 5x4 panel plot

    print('plot rollout traces for selected neuron types...')

    # Define selected neuron types and their indices
    selected_types = [55, 50, 43, 39, 35, 31, 23, 19, 12, 5]  # L1, Mi1, Mi2, R1, T1, T4a, T5a, Tm1, Tm4, Tm9
    neuron_indices = []
    for stype in selected_types:
        indices = np.where(neuron_types == stype)[0]
        if len(indices) > 0:
            neuron_indices.append(indices[0])

    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(12, 18))

    true_slice = activity_true[neuron_indices, start_frame:end_frame]
    pred_slice = activity_pred[neuron_indices, start_frame:end_frame]
    step_v = 2.5
    lw = 4

    # Plot ground truth (green, thick)
    for i in range(len(neuron_indices)):
        baseline = np.mean(true_slice[i])
        ax.plot(true_slice[i] - baseline + i * step_v, linewidth=lw, c='green', alpha=0.5,
                label='ground truth' if i == 0 else None)
        ax.plot(pred_slice[i] - baseline + i * step_v, linewidth=2, c='black',
                label='prediction' if i == 0 else None)

    # Add neuron type labels on left
    for i in range(len(neuron_indices)):
        type_idx = selected_types[i]
        ax.text(-50, i * step_v, f'{index_to_name[type_idx]}',
                fontsize=18, va='bottom', ha='right')
        ax.text(-50, i * step_v - 0.3, f'{neuron_indices[i]}',
                fontsize=12, va='top', ha='right', color='black')

        # Calculate and add metrics on right
        true_i = true_slice[i]
        pred_i = pred_slice[i]

        # Remove NaNs for correlation
        valid_mask = ~(np.isnan(true_i) | np.isnan(pred_i))
        if valid_mask.sum() > 1:
            rmse = np.sqrt(np.mean((true_i[valid_mask] - pred_i[valid_mask])**2))
            pearson_r, _ = pearsonr(true_i[valid_mask], pred_i[valid_mask])
        else:
            rmse = np.nan
            pearson_r = np.nan

        # Add metrics text on the right
        ax.text(end_frame - start_frame + 10, i * step_v,
                f'RMSE: {rmse:.3f}\nr: {pearson_r:.3f}',
                fontsize=10, va='center', ha='left',
                color='black')

    ax.set_ylim([-step_v, len(neuron_indices) * step_v])
    ax.set_yticks([])
    ax.set_xticks([0, (end_frame - start_frame) // 2, end_frame - start_frame])
    ax.set_xticklabels([start_frame, end_frame//2, end_frame], fontsize=16)
    ax.set_xlabel('frame', fontsize=20)

    # Adjust x-axis limit to make room for metrics
    ax.set_xlim([-50, end_frame - start_frame + 100])

    # Remove unnecessary spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Add legend
    ax.legend(loc='upper right', fontsize=14)

    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/rollout_traces_selected_types.png", dpi=300, bbox_inches='tight')
    plt.close()

    if ('test_ablation' in test_mode) or ('test_inactivity' in test_mode):
        np.save(f"./{log_dir}/results/activity_modified.npy", activity_true_modified)
        np.save(f"./{log_dir}/results/activity_modified_pred.npy", activity_pred)
    else:
        np.save(f"./{log_dir}/results/activity_true.npy", activity_true)
        np.save(f"./{log_dir}/results/activity_pred.npy", activity_pred)

