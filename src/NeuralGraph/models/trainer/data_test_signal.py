"""Extracted from graph_trainer.py"""
from ._imports import *

def data_test_signal(config=None, config_file=None, visualize=False, style='color frame', verbose=True, best_model=20, step=15, ratio=1, run=1, test_mode='', sample_embedding=False, particle_of_interest=1, new_params = None, device=[]):
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

    cmap = CustomColorMap(config=config)
    dimension = simulation_config.dimension

    has_missing_activity = training_config.has_missing_activity
    has_excitation = ('excitation' in model_config.update_type)
    baseline_value = simulation_config.baseline_value
    x_inference_list = []  # Initialize for inference test mode

    torch.random.fork_rng(devices=device)
    torch.random.manual_seed(simulation_config.seed)

    if 'latex' in style:
        print('latex style...')
        # plt.rcParams['text.usetex'] = True
        # rc('font', **{'family': 'serif', 'serif': ['Palatino']})
        # mpl.rcParams.update({
        #     "text.usetex": True,                    # use LaTeX for all text
        #     "font.family": "serif",                 # tell mpl to prefer serifs
        #     "text.latex.preamble": r"""
        #         \usepackage[T1]{fontenc}
        #         \usepackage[sc]{mathpazo} % Palatino text + math
        #         \linespread{1.05}         % optional: Palatino needs a bit more leading
        #     """,
        # })
        plt.rcParams['text.usetex'] = True
        rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    if 'black' in style:
        plt.style.use('dark_background')
        mc = 'w'
    else:
        plt.style.use('default')
        mc = 'k'


    field_type = model_config.field_type
    if field_type != '':
        n_nodes = simulation_config.n_nodes
        n_nodes_per_axis = int(np.sqrt(n_nodes))

    log_dir = 'log/' + config.config_file
    files = glob.glob(f"./{log_dir}/tmp_recons/*")
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

    n_sub_population = n_neurons // n_neuron_types
    first_cell_id_particles = []
    for n in range(n_neuron_types):
        index = np.arange(n_neurons * n // n_neuron_types, n_neurons * (n + 1) // n_neuron_types)
        first_cell_id_particles.append(index)

    print(f'load data...')

    x_list = []
    y_list = []

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
    if vnorm == 0:
        vnorm = ynorm


    connectivity = torch.load(f'./graphs_data/{dataset_name}/connectivity.pt', map_location=device)
    if training_config.with_connectivity_mask:
        model_mask = (connectivity > 0) * 1.0
        adj_t = model_mask.float() * 1
        adj_t = adj_t.t()
        edge_index = adj_t.nonzero().t().contiguous()
    else:
        edge_index = torch.load(f'./graphs_data/{dataset_name}/edge_index.pt', map_location=device)

    if ('modulation' in model_config.field_type) | ('visual' in model_config.field_type):
        print('load b_i movie ...')
        im = imread(f"graphs_data/{simulation_config.node_value_map}")
        A1 = torch.zeros((n_neurons, 1), device=device)

        # neuron_index = torch.randint(0, n_neurons, (6,))
        neuron_gt_list = []
        neuron_pred_list = []
        modulation_gt_list = []
        modulation_pred_list = []
        node_gt_list = []
        node_pred_list = []

        if os.path.exists(f'./graphs_data/{dataset_name}/X1.pt') > 0:
            X1_first = torch.load(f'./graphs_data/{dataset_name}/X1.pt', map_location=device)
            X_msg = torch.load(f'./graphs_data/{dataset_name}/X_msg.pt', map_location=device)
        else:
            xc, yc = get_equidistant_points(n_points=n_neurons)
            X1_first = torch.tensor(np.stack((xc, yc), axis=1), dtype=torch.float32, device=device) / 2
            perm = torch.randperm(X1_first.size(0))
            X1_first = X1_first[perm]
            torch.save(X1_first, f'./graphs_data/{dataset_name}/X1_first.pt')
            xc, yc = get_equidistant_points(n_points=n_neurons ** 2)
            X_msg = torch.tensor(np.stack((xc, yc), axis=1), dtype=torch.float32, device=device) / 2
            perm = torch.randperm(X_msg.size(0))
            X_msg = X_msg[perm]
            torch.save(X_msg, f'./graphs_data/{dataset_name}/X_msg.pt')

    model_generator, bc_pos, bc_dpos = choose_model(config=config, W=connectivity, device=device)

    model, bc_pos, bc_dpos = choose_training_model(config, device)
    model.ynorm = ynorm
    model.vnorm = vnorm
    model.particle_of_interest = particle_of_interest
    if training_config.with_connectivity_mask:
        model.mask = (connectivity > 0) * 1.0
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    state_dict = torch.load(net, map_location=device, weights_only=True)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()


    if ('short_term_plasticity' in field_type) | ('modulation_permutation' in field_type):
        model_f = Siren(in_features=model_config.input_size_nnr, out_features=model_config.output_size_nnr,
                        hidden_features=model_config.hidden_dim_nnr,
                        hidden_layers=model_config.n_layers_nnr, first_omega_0=model_config.omega,
                        hidden_omega_0=model_config.omega,
                        outermost_linear=model_config.outermost_linear_nnr)
        net = f'{log_dir}/models/best_model_f_with_1_graphs_{best_model}.pt'
        state_dict = torch.load(net, map_location=device)
        model_f.load_state_dict(state_dict['model_state_dict'])
        model_f.to(device=device)
        model_f.eval()
    if ('modulation' in model_config.field_type) | ('visual' in model_config.field_type):
        model_f = Siren_Network(image_width=n_nodes_per_axis, in_features=model_config.input_size_nnr,
                                out_features=model_config.output_size_nnr,
                                hidden_features=model_config.hidden_dim_nnr,
                                hidden_layers=model_config.n_layers_nnr, outermost_linear=True,
                                device=device,
                                first_omega_0=model_config.omega, hidden_omega_0=model_config.omega)
        net = f'{log_dir}/models/best_model_f_with_1_graphs_{best_model}.pt'
        state_dict = torch.load(net, map_location=device)
        model_f.load_state_dict(state_dict['model_state_dict'])
        model_f.to(device=device)
        model_f.eval()
    if has_missing_activity:
        model_missing_activity = nn.ModuleList([
            Siren(in_features=model_config.input_size_nnr, out_features=model_config.output_size_nnr,
                    hidden_features=model_config.hidden_dim_nnr,
                    hidden_layers=model_config.n_layers_nnr, first_omega_0=model_config.omega,
                    hidden_omega_0=model_config.omega,
                    outermost_linear=model_config.outermost_linear_nnr)
            for n in range(n_runs)
        ])
        model_missing_activity.to(device=device)
        net = f'{log_dir}/models/best_model_missing_activity_with_{n_runs - 1}_graphs_{best_model}.pt'
        state_dict = torch.load(net, map_location=device)
        model_missing_activity.load_state_dict(state_dict['model_state_dict'])
        model_missing_activity.to(device=device)
        model_missing_activity.eval()

    rmserr_list = []
    pred_err_list = []
    geomloss_list = []
    angle_list = []
    time.sleep(1)

    if time_window > 0:
        start_it = time_window
        stop_it = n_frames - 1
    else:
        start_it = 0
        stop_it = n_frames - 1

    start_it = 12

    x = x_list[0][start_it].clone().detach()
    x_generated = x_list[0][start_it].clone().detach()

    if 'test_ablation' in test_mode:
        #  test_mode="test_ablation_100"
        ablation_ratio = int(test_mode.split('_')[-1]) / 100
        if ablation_ratio > 0:
            print(f'\033[93mtest ablation ratio {ablation_ratio} \033[0m')
        n_ablation = int(n_neurons * ablation_ratio)
        index_ablation = np.random.choice(np.arange(n_neurons), n_ablation, replace=False)
        with torch.no_grad():
            model.W[index_ablation, :] = 0
            model_generator.W[index_ablation, :] = 0
    else:
        ablation_ratio = 0

    if 'test_inactivity' in test_mode:
        #  test_mode="test_inactivity_100"
        inactivity_ratio = int(test_mode.split('_')[-1]) / 100
        if inactivity_ratio > 0:
            print(f'\033[93mtest inactivity ratio {inactivity_ratio} \033[0m')
        n_inactivity = int(n_neurons * inactivity_ratio)
        index_inactivity = np.random.choice(np.arange(n_neurons), n_inactivity, replace=False)

        x[index_inactivity, 6] = 0
        x_generated[index_inactivity, 6] = 0

        with torch.no_grad():
            model.W[index_inactivity, :] = 0
            model.W[:, index_inactivity] = 0
            model_generator.W[index_inactivity, :] = 0
            model_generator.W[:, index_inactivity] = 0
    else:
        inactivity_ratio = 0

    if 'test_permutation' in test_mode:
        permutation_ratio = int(test_mode.split('_')[-1]) / 100
        if permutation_ratio > 0:
            print(f'\033[93mtest permutation ratio {permutation_ratio} \033[0m')
        n_permutation = int(n_neurons * permutation_ratio)
        index_permutation = np.random.choice(np.arange(n_neurons), n_permutation, replace=False)
        rnd_perm = torch.randperm(n_permutation)

        x_permuted = x[index_permutation, 5].clone().detach()
        x_generated_permuted = x_generated[index_permutation, 5].clone().detach()

        x[index_permutation, 5] = x_permuted[rnd_perm]
        x_generated[index_permutation, 5] = x_generated_permuted[rnd_perm]

        a_permuted = model.a[index_permutation].clone().detach()
        with torch.no_grad():
            model.a[index_permutation] = a_permuted[rnd_perm]
    else:
        permutation_ratio = 0

    if new_params is not None:
        print('set new parameters for testing ...')


        plt.figure(figsize=(10, 10))
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif']
        ax = sns.heatmap(to_numpy(connectivity), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046},
                         vmin=-0.1, vmax=0.1)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=32)
        plt.xticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
        plt.yticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
        plt.xticks(rotation=0)
        plt.subplot(2, 2, 1)
        ax = sns.heatmap(to_numpy(connectivity[0:20, 0:20]), cbar=False, center=0, square=True, cmap='bwr', vmin=-0.1, vmax=0.1)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/results/true connectivity.png', dpi=300)
        plt.close()

        edge_index_, connectivity, mask = init_connectivity(
                simulation_config.connectivity_file,
                simulation_config.connectivity_distribution,
                new_params[0],
                None,
                n_neurons,
                n_neuron_types,
                dataset_name,
                device,
            )


        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(to_numpy(connectivity), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046},
                         vmin=-0.1, vmax=0.1)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=32)
        plt.xticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
        plt.yticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
        plt.xticks(rotation=0)
        plt.subplot(2, 2, 1)
        ax = sns.heatmap(to_numpy(connectivity[0:20, 0:20]), cbar=False, center=0, square=True, cmap='bwr', vmin=-0.1, vmax=0.1)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/results/new connectivity.png', dpi=300)
        plt.close()

        second_correction = np.load(f'{log_dir}/second_correction.npy')
        print(f'second_correction: {second_correction}')

        with torch.no_grad():
            model_generator.W = torch.nn.Parameter(torch.tensor(connectivity, device=device))
            model.W = torch.nn.Parameter(model_generator.W.clone() * torch.tensor(second_correction, device=device))

        cell_types = to_numpy(x[:, 5]).astype(int)
        type_counts = [np.sum(cell_types == n) for n in range(n_neuron_types)]
        plt.figure(figsize=(10, 10))
        bars = plt.bar(range(n_neuron_types), type_counts,
                    color=[cmap.color(n) for n in range(n_neuron_types)])

        plt.xlabel('neuron type', fontsize=48)
        plt.ylabel('count', fontsize=48)
        plt.xticks(range(n_neuron_types),fontsize=24)
        plt.yticks(fontsize=24)
        plt.grid(axis='y', alpha=0.3)
        for i, count in enumerate(type_counts):
            plt.text(i, count + max(type_counts)*0.01, str(count),
                    ha='center', va='bottom', fontsize=32)
        plt.ylim(0, 1000)
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/results/neuron_type_histogram.png', dpi=300)
        plt.close()


        first_cell_id_neurons = []
        for n in range(n_neuron_types):
            index = np.arange(n_neurons * n // n_neuron_types, n_neurons * (n + 1) // n_neuron_types)
            first_cell_id_neurons.append(index)
        id=0
        for n in range(n_neuron_types):
            print(f'neuron type {n}, first cell id {id}')
            x[id:id+int(new_params[n+1]*n_neurons/100), 5] = n
            x_generated[id:id+int(new_params[n+1]*n_neurons/100), 5] = n
            id = id + int(new_params[n+1]*n_neurons/100)
        print(f'last cell id {id}, total number of neurons {n_neurons}')

        first_embedding = model.a.clone().detach()
        model_a_ = nn.Parameter(torch.tensor(np.ones((int(n_neurons), model.embedding_dim)), device=device, requires_grad=False,dtype=torch.float32))
        for n in range(n_neurons):
            t = to_numpy(x[n, 5]).astype(int)
            index = first_cell_id_neurons[t][np.random.randint(len(first_cell_id_neurons[t]))]
            with torch.no_grad():
                model_a_[n] = first_embedding[index].clone().detach()
        model.a = nn.Parameter(
            torch.tensor(np.ones((int(n_neurons), model.embedding_dim)), device=device,
                         requires_grad=False, dtype=torch.float32))
        with torch.no_grad():
            for n in range(model.a.shape[0]):
                model.a[n] = model_a_[n]

                cell_types = to_numpy(x[:, 5]).astype(int)
        type_counts = [np.sum(cell_types == n) for n in range(n_neuron_types)]
        plt.figure(figsize=(10, 10))
        bars = plt.bar(range(n_neuron_types), type_counts,
                    color=[cmap.color(n) for n in range(n_neuron_types)])

        plt.xlabel('neuron type', fontsize=48)
        plt.ylabel('count', fontsize=48)
        plt.xticks(range(n_neuron_types),fontsize=24)
        plt.yticks(fontsize=24)
        plt.grid(axis='y', alpha=0.3)
        for i, count in enumerate(type_counts):
            plt.text(i, count + max(type_counts)*0.01, str(count),
                    ha='center', va='bottom', fontsize=32)
        plt.ylim(0, 1000)
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/results/new_neuron_type_histogram.png', dpi=300)
        plt.close()

    n_neurons = x.shape[0]
    neuron_gt_list = []
    neuron_pred_list = []
    neuron_generated_list = []

    for it in trange(start_it,start_it+4000):  # start_it + min(9600+start_it,stop_it-time_step)): #  start_it+200): # min(9600+start_it,stop_it-time_step)):

        if it < n_frames - 4:
            x0 = x_list[0][it].clone().detach()
            x0_next = x_list[0][(it + time_step)].clone().detach()
            y0 = y_list[0][it].clone().detach()
        if has_excitation:
            x[:, 10: 10 + model_config.excitation_dim] = x0[:, 10: 10 + model_config.excitation_dim]

        # error calculations
        x0[:, 6] = torch.where(torch.isnan(x0[:, 6]), baseline_value, x0[:, 6])
        x[:, 6]  = torch.where(torch.isnan(x[:, 6]),  baseline_value, x[:, 6])
        x_generated[:, 6] = torch.where(torch.isnan(x_generated[:, 6]), baseline_value, x_generated[:, 6])


        if 'ablation' in test_mode:
            rmserr = torch.sqrt(torch.mean((x_generated[:n_neurons, 6] - x0[:, 6]) ** 2))
        else:
            rmserr = torch.sqrt(torch.mean((x[:n_neurons, 6] - x0[:, 6]) ** 2))
        neuron_gt_list.append(x0[:, 6:7])
        neuron_pred_list.append(x[:n_neurons, 6:7].clone().detach())
        neuron_generated_list.append(x_generated[:n_neurons, 6:7].clone().detach())

        if ('short_term_plasticity' in field_type) | ('modulation' in field_type):
            modulation_gt_list.append(x0[:, 8:9])
            modulation_pred_list.append(x[:, 8:9].clone().detach())
        rmserr_list.append(rmserr.item())

        if config.training.shared_embedding:
            data_id = torch.ones((n_neurons, 1), dtype=torch.int, device=device)
        else:
            data_id = torch.ones((n_neurons, 1), dtype=torch.int, device=device) * run

        # update calculations
        if 'visual' in field_type:
            x[:n_nodes, 8:9] = model_f(time=it / n_frames) ** 2
            x[n_nodes:n_neurons, 8:9] = 1
        elif 'learnable_short_term_plasticity' in field_type:
            alpha = (it % model.embedding_step) / model.embedding_step
            x[:, 8] = alpha * model.b[:, it // model.embedding_step + 1] ** 2 + (1 - alpha) * model.b[:,
                                                                                                it // model.embedding_step] ** 2
        elif ('short_term_plasticity' in field_type) | ('modulation_permutation' in field_type):
            t = torch.zeros((1, 1, 1), dtype=torch.float32, device=device)
            t[:, 0, :] = torch.tensor(it / n_frames, dtype=torch.float32, device=device)
            x[:, 8] = model_f(t).squeeze() ** 2
        elif 'modulation' in field_type:
            x[:, 8:9] = model_f(time=it / n_frames) ** 2

        if has_missing_activity:
            t = torch.tensor([it / n_frames], dtype=torch.float32, device=device)
            missing_activity = baseline_value + model_missing_activity[run](t).squeeze()
            ids_missing = torch.argwhere(x[:, 6] == baseline_value)
            x[ids_missing, 6] = missing_activity[ids_missing]

        with torch.no_grad():
            dataset = pyg_Data(x=x, pos=x[:, 1:3], edge_index=edge_index)
            pred = model(dataset, data_id=data_id)
            y = pred
            dataset = pyg_Data(x=x_generated, pos=x[:, 1:3], edge_index=edge_index)
            pred_generator = model_generator(dataset, data_id=data_id)

        # signal update
        x[:n_neurons, 6:7] = x[:n_neurons, 6:7] + y[:n_neurons] * delta_t
        x_generated[:n_neurons, 6:7] = x_generated[:n_neurons, 6:7] + pred_generator[:n_neurons] * delta_t

        if 'test_inactivity' in test_mode:
            x[index_inactivity, 6:7] = 0
            x_generated[index_inactivity, 6:7] = 0

        # if 'CElegans' in dataset_name:
        #     x[:n_neurons, 6:7] = torch.clamp(x[:n_neurons, 6:7], min=0, max=10)

        # vizualization
        if 'plot_data' in test_mode:
            x = x_list[0][it].clone().detach()

        if (it % step == 0) & (it >= 0) & visualize:

            num = f"{it:06}"

            fig, ax = fig_init(formatx='%.1f', formaty='%.1f')
            ax.tick_params(axis='both', which='major', pad=15)

            if ('visual' in field_type):
                if 'plot_data' in test_mode:
                    plt.close()

                    im_ = im[int(it / n_frames * 256)].squeeze()
                    im_ = np.rot90(im_, 3)
                    im_ = np.reshape(im_, (n_nodes_per_axis * n_nodes_per_axis))
                    if ('modulation' in field_type):
                        A1[:, 0:1] = torch.tensor(im_[:, None], dtype=torch.float32, device=device)
                    if ('visual' in field_type):
                        A1[:n_nodes, 0:1] = torch.tensor(im_[:, None], dtype=torch.float32, device=device)
                        A1[n_nodes:n_neurons, 0:1] = 1

                fig = plt.figure(figsize=(8, 12))
                plt.subplot(211)
                plt.title(r'$b_i$', fontsize=48)
                plt.scatter(to_numpy(x0[:, 2]), to_numpy(x0[:, 1]), s=8, c=to_numpy(A1[:, 0]), cmap='viridis', vmin=0,
                            vmax=2)
                plt.xticks([])
                plt.yticks([])
                plt.subplot(212)
                plt.title(r'$x_i$', fontsize=48)
                plt.scatter(to_numpy(x0[:, 2]), to_numpy(x0[:, 1]), s=8, c=to_numpy(x[:, 6:7]), cmap='viridis',
                            vmin=-10,
                            vmax=10)
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_recons/Fig_{config_file}_{num}.tif", dpi=80)
                plt.close()

            else:

                plt.close()
                mpl.rcParams['savefig.pad_inches'] = 0

                black_to_green = LinearSegmentedColormap.from_list('black_green', ['black', 'green'])
                black_to_yellow = LinearSegmentedColormap.from_list('black_yellow', ['black', 'yellow'])

                plt.figure(figsize=(10, 10))
                plt.scatter(to_numpy(x[:, 1]), to_numpy(x[:, 2]), s=700, c=to_numpy(x[:, 6]), alpha=1, edgecolors='none', vmin =2 , vmax=8, cmap=black_to_green)
                plt.axis('off')
                plt.xticks([])
                plt.yticks([])
                plt.xlim([-0.6, 0.6])
                plt.ylim([-0.6, 0.6])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_recons/Nodes_{config_file}_{num}.tif", dpi=80)
                plt.close()


                if ('short_term_plasticity' in field_type) | ('modulation' in field_type):

                    modulation_gt_list_ = torch.cat(modulation_gt_list, 0)
                    modulation_pred_list_ = torch.cat(modulation_pred_list, 0)
                    modulation_gt_list_ = torch.reshape(modulation_gt_list_,
                                                        (modulation_gt_list_.shape[0] // n_neurons, n_neurons))
                    modulation_pred_list_ = torch.reshape(modulation_pred_list_,
                                                          (modulation_pred_list_.shape[0] // n_neurons,
                                                           n_neurons))

                    plt.figure(figsize=(20, 10))
                    if 'latex' in style:
                        plt.rcParams['text.usetex'] = True
                        rc('font', **{'family': 'serif', 'serif': ['Palatino']})

                    ax = plt.subplot(122)
                    plt.scatter(to_numpy(modulation_gt_list_[-1, :]), to_numpy(modulation_pred_list_[-1, :]), s=10,
                                c=mc)
                    plt.xticks(fontsize=24)
                    plt.yticks(fontsize=24)
                    x_data = to_numpy(modulation_gt_list_[-1, :])
                    y_data = to_numpy(modulation_pred_list_[-1, :])
                    lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                    residuals = y_data - linear_model(x_data, *lin_fit)
                    ss_res = np.sum(residuals ** 2)
                    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot)
                    plt.xlabel(r'true modulation', fontsize=48)
                    plt.ylabel(r'learned modulation', fontsize=48)
                    # plt.text(0.05, 0.9 * lin_fit[0], f'$R^2$: {np.round(r_squared, 3)}', fontsize=34)
                    # plt.text(0.05, 0.8 * lin_fit[0], f'slope: {np.round(lin_fit[0], 2)}', fontsize=34)
                    ax = plt.subplot(121)
                    plt.plot(modulation_gt_list_[:, n[0]].detach().cpu().numpy(), c='k', linewidth=8, label='true',
                             alpha=0.25)
                    plt.plot(modulation_pred_list_[:, n[0]].detach().cpu().numpy() / lin_fit[0], linewidth=4, c='k',
                             label='learned')
                    plt.legend(fontsize=24)
                    plt.plot(modulation_gt_list_[:, n[1:10]].detach().cpu().numpy(), c='k', linewidth=8, alpha=0.25)
                    plt.plot(modulation_pred_list_[:, n[1:10]].detach().cpu().numpy() / lin_fit[0], linewidth=4)
                    plt.xlim([0, 1400])
                    plt.xlabel(r'time-points', fontsize=48)
                    plt.ylabel(r'modulation', fontsize=48)
                    plt.xticks(fontsize=24)
                    plt.yticks(fontsize=24)
                    plt.ylim([0, 2])
                    # plt.text(40, 26, f'time: {it}', fontsize=34)

        if (it % 200 == 0) & (it > 0) & (it <=4000):

            if n_neurons <= 100:
                n = np.arange(0,n_neurons,2)
            elif 'CElegans' in dataset_name:
                n = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
            else:
                n = [20, 30, 100, 150, 260, 270, 520, 620, 720, 820]

            neuron_gt_list_ = torch.cat(neuron_gt_list, 0)
            neuron_pred_list_ = torch.cat(neuron_pred_list, 0)
            neuron_generated_list_ = torch.cat(neuron_generated_list, 0)
            neuron_gt_list_ = torch.reshape(neuron_gt_list_, (neuron_gt_list_.shape[0] // n_neurons, n_neurons))
            neuron_pred_list_ = torch.reshape(neuron_pred_list_, (neuron_pred_list_.shape[0] // n_neurons, n_neurons))
            neuron_generated_list_ = torch.reshape(neuron_generated_list_, (neuron_generated_list_.shape[0] // n_neurons, n_neurons))

            mpl.rcParams.update({
                "text.usetex": True,
                "font.family": "serif",
                "font.size": 12,           # Base font size
                "axes.labelsize": 14,      # Axis labels
                "legend.fontsize": 12,     # Legend
                "xtick.labelsize": 10,     # Tick labels
                "ytick.labelsize": 10,
                "text.latex.preamble": r"""
                    \usepackage[T1]{fontenc}
                    \usepackage[sc]{mathpazo}
                    \linespread{1.05}
                """,
            })

            plt.figure(figsize=(20, 20))

            ax = plt.subplot(121)
            # Plot ground truth with distinct gray color, visible in legend
            for i in range(len(n)):
                color = 'gray' if i == 0 else None  # Only label first
                if ablation_ratio > 0:
                    label = f'true ablation {ablation_ratio}' if i == 0 else None
                elif inactivity_ratio > 0:
                    label = f'true inactivity {inactivity_ratio}' if i == 0 else None
                elif permutation_ratio > 0:
                    label = f'true permutation {permutation_ratio}' if i == 0 else None
                else:
                    label = 'true' if i == 0 else None
                plt.plot(neuron_generated_list_[:, n[i]].detach().cpu().numpy() + i * 25,
                        c='gray', linewidth=8, alpha=0.5, label=label)

            # Plot predictions with colored lines
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
            for i in range(len(n)):
                label = 'learned' if i == 0 else None
                plt.plot(neuron_pred_list_[:, n[i]].detach().cpu().numpy() + i * 25,
                        linewidth=3, c=colors[i%10], label=label)

            plt.xlim([0, len(neuron_gt_list_)])

            # Auto ylim from ground truth range (ignore predictions if exploded)
            y_gt = np.concatenate([neuron_generated_list_[:, n[i]].detach().cpu().numpy() + i*25 for i in range(len(n))])
            y_pred = np.concatenate([neuron_pred_list_[:, n[i]].detach().cpu().numpy() + i*25 for i in range(len(n))])

            if np.abs(y_pred).max() > 10 * np.abs(y_gt).max():  # Explosion
                ylim = [y_gt.min() - 10, y_gt.max() + 10]
            else:
                y_all = np.concatenate([y_gt, y_pred])
                margin = (y_all.max() - y_all.min()) * 0.05
                ylim = [y_all.min() - margin, y_all.max() + margin]

            plt.ylim(ylim)
            plt.xlabel('time-points', fontsize=48)
            plt.ylabel('neurons', fontsize=48)
            plt.xticks(fontsize=24)
            plt.yticks([])

            ax = plt.subplot(222)
            x_data = to_numpy(neuron_generated_list_[-1, :])
            y_data = to_numpy(neuron_pred_list_[-1, :])

            # print(f"x: [{x_data.min():.2e}, {x_data.max():.2e}]")
            # print(f"y: [{y_data.min():.2e}, {y_data.max():.2e}], std={y_data.std():.2e}")

            # Detect severe collapse (constant predictions) or explosion
            severe_collapse = y_data.std() < 0.1 * x_data.std()
            explosion = np.abs(y_data).max() > 1e10

            if not (severe_collapse or explosion):
                # Normal/mild collapse: fit and show all data
                mask = (np.abs(x_data - np.median(x_data)) < 3*np.std(x_data)) & \
                    (np.abs(y_data - np.median(y_data)) < 3*np.std(y_data))
                if mask.sum() > 10:
                    lin_fit, _ = curve_fit(linear_model, x_data[mask], y_data[mask])
                    slope, intercept = lin_fit
                    r2 = 1 - np.sum((y_data - linear_model(x_data, *lin_fit))**2) / np.sum((y_data - np.mean(y_data))**2)
                else:
                    slope, intercept, r2 = 0, 0, 0

                # Auto limits from combined data
                all_data = np.concatenate([x_data, y_data])
                margin = (all_data.max() - all_data.min()) * 0.1
                lim = [all_data.min() - margin, all_data.max() + margin]

                plt.scatter(x_data, y_data, s=100, c=mc, alpha=0.8, edgecolors='k', linewidths=0.5)
                if mask.sum() > 10:
                    x_line = np.array(lim)
                    plt.plot(x_line, linear_model(x_line, slope, intercept), 'r--', linewidth=2)
                plt.plot(lim, lim, 'k--', alpha=0.3, linewidth=1)
                plt.text(0.05, 0.95, f'$R^2$: {r2:.3f}\nslope: {slope:.3f}',
                        transform=plt.gca().transAxes, fontsize=34, va='top')
            else:
                # Severe collapse/explosion
                lim = [x_data.min()*1.1, 0]
                plt.scatter(x_data, np.clip(y_data, lim[0], lim[1]), s=100, c=mc, alpha=0.8, edgecolors='k', linewidths=0.5)
                plt.text(0.5, 0.5, 'collapsed' if severe_collapse else 'explosion',
                        ha='center', fontsize=48, color='red', alpha=0.3, transform=plt.gca().transAxes)

            plt.xlim(lim)
            plt.ylim(lim)
            plt.xlabel('true $x_i$', fontsize=48)
            plt.ylabel('learned $x_i$', fontsize=48)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)

            ax = plt.subplot(224)
            corrs = []
            for i in range(n_neurons):
                r, _ = scipy.stats.pearsonr(
                    to_numpy(neuron_generated_list_[:, i]),
                    to_numpy(neuron_pred_list_[:, i])
                )
                corrs.append(r)
            plt.hist(corrs, bins=30, edgecolor='black')
            plt.xlim([0, 1])
            plt.xlabel('Pearson $r$', fontsize=48)
            plt.ylabel('count', fontsize=48)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.text(0.05, plt.ylim()[1]*0.9, f'mean: {np.mean(corrs):.3f}', fontsize=34)

            plt.tight_layout()

            if ablation_ratio>0:
                filename = f'comparison_vi_{it}_ablation_{ablation_ratio}.png'
            elif inactivity_ratio>0:
                filename = f'comparison_vi_{it}_inactivity_{inactivity_ratio}.png'
            elif permutation_ratio>0:
                filename = f'comparison_vi_{it}_permutation_{permutation_ratio}.png'
            else:
                filename = f'comparison_vi_{it}.png'

            plt.savefig(f'./{log_dir}/results/{filename}', dpi=80)
            plt.close()
            # print(f'saved figure ./log/{log_dir}/results/{filename}')


    if 'inference' in test_mode:
        torch.save(x_inference_list, f"./{log_dir}/x_inference_list_{run}.pt")

    # print('average rollout RMSE {:.3e}+/-{:.3e}'.format(np.mean(rmserr_list), np.std(rmserr_list)))

    if 'PDE_N' in model_config.signal_model_name:

        torch.save(neuron_gt_list, f"./{log_dir}/neuron_gt_list.pt")
        torch.save(neuron_pred_list, f"./{log_dir}/neuron_pred_list.pt")

    else:
        if False:
            # geomloss_list == []:
            geomloss_list = [0, 0]
            r = [np.mean(rmserr_list), np.std(rmserr_list), np.mean(geomloss_list), np.std(geomloss_list)]
            print('average rollout Sinkhorn div. {:.3e}+/-{:.3e}'.format(np.mean(geomloss_list), np.std(geomloss_list)))
            np.save(f"./{log_dir}/rmserr_geomloss_{config_file}.npy", r)

        if False:
            rmserr_list = np.array(rmserr_list)
            fig, ax = fig_init(formatx='%.1f', formaty='%.1f')
            x_ = np.arange(len(rmserr_list))
            y_ = rmserr_list
            plt.scatter(x_, y_, c=mc)
            plt.xticks(fontsize=48)
            plt.yticks(fontsize=48)
            plt.xlabel(r'$Epochs$', fontsize=78)
            plt.ylabel(r'$RMSE$', fontsize=78)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/rmserr_{config_file}_plot.tif", dpi=170.7)

        if False:
            x0_next = x_list[0][it].clone().detach()
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(1, 1, 1)
            temp1 = torch.cat((x, x0_next), 0)
            temp2 = torch.tensor(np.arange(n_neurons), device=device)
            temp3 = torch.tensor(np.arange(n_neurons) + n_neurons, device=device)
            temp4 = torch.concatenate((temp2[:, None], temp3[:, None]), 1)
            temp4 = torch.t(temp4)
            distance4 = torch.sqrt(torch.sum((x[:, 1:3] - x0_next[:, 1:3]) ** 2, 1))
            p = torch.argwhere(distance4 < 0.3)

            temp1_ = temp1[:, [2, 1]].clone().detach()
            pos = dict(enumerate(np.array((temp1_).detach().cpu()), 0))
            dataset = pyg_Data(x=temp1_, edge_index=torch.squeeze(temp4[:, p]))
            vis = to_networkx(dataset, remove_self_loops=True, to_undirected=True)
            nx.draw_networkx(vis, pos=pos, node_size=0, linewidths=0, with_labels=False, ax=ax, edge_color='r', width=4)
            for n in range(n_neuron_types):
                plt.scatter(x[index_particles[n], 2].detach().cpu().numpy(),
                            x[index_particles[n], 1].detach().cpu().numpy(), s=100, color=cmap.color(n))
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.xlabel(r'$x$', fontsize=78)
            plt.ylabel(r'$y$', fontsize=78)
            formatx = '%.1f'
            formaty = '%.1f'
            ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True, axis='both', which='major', pad=15)
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            ax.xaxis.set_major_formatter(FormatStrFormatter(formatx))
            ax.yaxis.set_major_formatter(FormatStrFormatter(formaty))
            plt.xticks(fontsize=48.0)
            plt.yticks(fontsize=48.0)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/rmserr_{config_file}_{it}.tif", dpi=170.7)
            plt.close()

            fig = plt.figure(figsize=(12, 12))
            for n in range(n_neuron_types):
                plt.scatter(x0_next[index_particles[n], 2].detach().cpu().numpy(),
                            x0_next[index_particles[n], 1].detach().cpu().numpy(), s=50, color=cmap.color(n))
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.xlabel(r'$x$', fontsize=78)
            plt.ylabel(r'$y$', fontsize=78)
            formatx = '%.2f'
            formaty = '%.2f'
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            ax.xaxis.set_major_formatter(FormatStrFormatter(formatx))
            ax.yaxis.set_major_formatter(FormatStrFormatter(formaty))
            plt.xticks(fontsize=48.0)
            plt.yticks(fontsize=48.0)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/GT_{config_file}_{it}.tif", dpi=170.7)
            plt.close()

    if len(angle_list) > 0:
        angle = torch.stack(angle_list)
        fig = plt.figure(figsize=(12, 12))
        plt.hist(to_numpy(angle), bins=1000, color='w')
        plt.xlabel('angle', fontsize=48)
        plt.ylabel('count', fontsize=48)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.xlim([-90, 90])
        plt.savefig(f"./{log_dir}/results/angle.tif", dpi=170.7)
        plt.close

