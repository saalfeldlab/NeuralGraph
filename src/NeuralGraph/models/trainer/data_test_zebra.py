"""Extracted from graph_trainer.py"""
from ._imports import *

def data_test_zebra(config, visualize, style, verbose, best_model, step, test_mode, device):

    dataset_name = config.dataset
    simulation_config = config.simulation
    model_config = config.graph_model
    training_config = config.training

    n_neuron_types = simulation_config.n_neuron_types
    n_neurons = simulation_config.n_neurons
    n_nodes = simulation_config.n_nodes
    n_runs = training_config.n_runs
    n_frames = simulation_config.n_frames
    delta_t = simulation_config.delta_t
    time_window = training_config.time_window
    time_step = training_config.time_step
    plot_batch_size = config.plotting.plot_batch_size

    cmap = CustomColorMap(config=config)
    dimension = simulation_config.dimension

    has_missing_activity = training_config.has_missing_activity
    has_excitation = ('excitation' in model_config.update_type)
    baseline_value = simulation_config.baseline_value

    torch.random.fork_rng(devices=device)
    torch.random.manual_seed(simulation_config.seed)

    ones = torch.ones((n_neurons, 1), dtype=torch.float32, device=device)

    run = 0

    if 'latex' in style:
        print('latex style...')
        plt.rcParams['text.usetex'] = True
        rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    if 'black' in style:
        plt.style.use('dark_background')
        mc = 'w'
    else:
        plt.style.use('default')
        mc = 'k'


    log_dir = 'log/' + config.config_file
    os.makedirs(f"./{log_dir}/results/Fig/", exist_ok=True)
    files = glob.glob(f"./{log_dir}/results/Fig/*")
    for f in files:
        os.remove(f)

    if best_model == 'best':
        files = glob.glob(f"{log_dir}/models/*")
        files.sort(key=sort_key)
        filename = files[-1]
        filename = filename.split('/')[-1]
        filename = filename.split('graphs')[-1][1:-3]
        best_model = filename
        print(f'best model: {best_model}')
    net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs_{best_model}.pt"

    x_list = []
    y_list = []

    print('load data...')
    if os.path.exists(f'graphs_data/{dataset_name}/x_list_{run}.pt'):
        x = torch.load(f'graphs_data/{dataset_name}/x_list_{run}.pt', map_location=device)
        y = torch.load(f'graphs_data/{dataset_name}/y_list_{run}.pt', map_location=device)
        x_list.append(x)
        y_list.append(y)
    else:
        x = np.load(f'graphs_data/{dataset_name}/x_list_{run}.npy')
        x = torch.tensor(x, dtype=torch.float32, device=device)
        y = np.load(f'graphs_data/{dataset_name}/y_list_{run}.npy')
        y = torch.tensor(y, dtype=torch.float32, device=device)
        x_list.append(x)
        y_list.append(y)
        x = x_list[0][0].clone().detach()
        n_neurons = int(x.shape[0])
        config.simulation.n_neurons = n_neurons
        n_frames = len(x_list[0])
        index_particles = get_index_particles(x, n_neuron_types, dimension)
        if n_neuron_types > 1000:
            index_particles = []
            for n in range(3):
                index = np.arange(n_neurons * n // 3, n_neurons * (n + 1) // 3)
                index_particles.append(index)
                n_neuron_types = 3
    ynorm = torch.load(f'{log_dir}/ynorm.pt', map_location=device, weights_only=True)
    vnorm = torch.load(f'{log_dir}/vnorm.pt', map_location=device, weights_only=True)


    model = Signal_Propagation_Zebra(aggr_type=model_config.aggr_type, config=config, device=device)
    state_dict = torch.load(net, map_location=device, weights_only=True)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    print('recons...')

    generated_x_list = []


    # tail_list = np.array(to_numpy(x_list[0][:,0,11:14]))
    # print (tail_list.shape)

    it_idx = 0
    for it in trange(0, min(n_frames,7800), 1):
        x = torch.tensor(x_list[run][it], dtype=torch.float32, device=device)
        with torch.no_grad():
            in_features = torch.cat((x[:,1:4]/model.NNR_f_xy_period, it/model.NNR_f_T_period * ones), 1)
            neural_field_list = []
            for start in range(0, in_features.shape[0], plot_batch_size):
                end = min(start + plot_batch_size, in_features.shape[0])
                batch = in_features[start:end]
                neural_field_list.append(model.NNR_f(batch)**2)
            neural_field = torch.cat(neural_field_list, dim=0)
            generated_x_list.append(to_numpy(neural_field.clone().detach()))
            if (it % step == 0) & (visualize == True):
                # plot field comparison
                output_path = f"./{log_dir}/results/Fig/Fig_{run}_{it_idx:06d}.png"

                if 'continuous_slice' in style:
                    plot_field_comparison_continuous_slices(
                        x, model, it, n_frames, ones,
                        f"./{log_dir}/results/Fig/Fig_{run}_{it_idx:06d}.png",
                        voxel_size=0.001,          # your setting
                        vmin=0.0, vmax=0.75,       # your setting
                        # optional mask controls
                        mask_points_per_neuron=32,
                        mask_jitter_sigma=0.005,
                        slice_half_thickness=0.008,
                        mask_min_count=1,
                        rng_seed=1234,
                        dpi=300
                    )
                elif 'discrete_slice' in style:
                    plot_field_comparison_discrete_slices(
                        x, model, it, n_frames, ones,
                        f"./{log_dir}/results/Fig/Fig_{run}_{it_idx:06d}.png",
                        z_slices=(0.10, 0.15),
                        y_slices=(0.20, 0.30),
                        slice_half_thickness=0.004,  # adjust band thickness
                        dot_size=2.0,                # your dot size here
                        vmin=0.0, vmax=0.75,
                        dpi=300,
                        flip_top_y=True,
                        flip_bottom_x=False,         # set True if you want to mirror X on bottom
                        flip_bottom_z=True          # set True to mirror Z (vertical) on bottom
                    )
                elif 'grid' in style:
                    plot_field_discrete_xy_slices_grid(
                        x, model, it, n_frames, ones,
                        f"./{log_dir}/results/Fig/Fig_{run}_{it_idx:06d}_xy_slices_discrete.png",
                        z_start=0.02,
                        z_step=0.008,
                        n_cols=5, n_rows=4,
                        slice_half_thickness=0.002,  # adjust band thickness as you like
                        dot_size=1.0,
                        vmin=0.0, vmax=0.75,
                        dpi=300
                    )
                else:
                    plot_field_comparison(x, model, it, n_frames, ones, output_path, 50, plot_batch_size)


                it_idx += 1

    generated_x_list = np.array(generated_x_list)
    print(f"generated {len(generated_x_list)} frames total")
    print(f"saving ./{log_dir}/results/recons_field.npy")
    np.save(f"./{log_dir}/results/recons_field.npy", generated_x_list)

    if (visualize == True):
        print('save video...')
        files = glob.glob(f'./{log_dir}/results/Fig/*')
        src = files[0]
        dst = f"./{log_dir}/results/input_zebra.png"
        with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
            fdst.write(fsrc.read())
        generate_compressed_video_mp4(
            output_dir=f"./{log_dir}/results",
            run=0,
            config_indices="zebra",
            framerate=40
        )
        print(f"video saved to {log_dir}/results/")

        # for f in files:
        #     os.remove(f)

    # generated_x_list = np.load(f"./{log_dir}/results/recons_field.npy")

    reconstructed = generated_x_list
    true = to_numpy(x_list[0][0:min(n_frames,7800),:,6:7])

    # comparison between x_list and generated_x_list

    condition_names = {
        0: "gain",
        1: "dots",
        2: "flash",
        3: "taxis",
        4: "turning",
        5: "position",
        6: "open loop",
        7: "rotation",
        8: "dark",
        -1: "none"
    }

    # per-frame MAE across neurons
    mae_per_frame = np.mean(np.abs(reconstructed - true), axis=1)  # [700]
    conditions = to_numpy(x_list[0][:,0,7:8]).squeeze()

    mean_mae, sem_mae, labels, counts = [], [], [], []
    for cond_id, cond_name in condition_names.items():
        if cond_id == -1:
            continue
        mask = conditions == cond_id
        if mask.sum() == 0:
            continue
        vals = mae_per_frame[mask]
        mean_mae.append(vals.mean())
        sem_mae.append(vals.std() / np.sqrt(len(vals)))
        labels.append(cond_name)
        counts.append(mask.sum())

    # grand average
    valid_mask = conditions != -1
    vals_all = mae_per_frame[valid_mask]
    mean_mae.append(vals_all.mean())
    sem_mae.append(vals_all.std() / np.sqrt(len(vals_all)))
    labels.append("grand average")
    counts.append(valid_mask.sum())

    # plot
    plt.figure(figsize=(10,5))
    bars = plt.bar(
        labels, mean_mae, yerr=sem_mae, capsize=5, color="skyblue",
        error_kw=dict(ecolor="white", elinewidth=2, capsize=5, capthick=2)
    )

    # add frame counts above bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2, height + 0.001,
            f"{count}", ha="center", va="bottom", fontsize=12, color="white"
        )

    plt.ylabel("MAE", fontsize=16)
    plt.title("reconstruction Error per Condition (±SEM)", fontsize=16)
    plt.xticks(rotation=45, ha="right", fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/recons_error_per_condition.png", dpi=150)
    plt.close()

    print(f"grand average MAE: {mean_mae[-1]:.4f} ± {sem_mae[-1]:.4f} (N={counts[-1]})")


    reconstructed = reconstructed.squeeze()
    true = true.squeeze()

