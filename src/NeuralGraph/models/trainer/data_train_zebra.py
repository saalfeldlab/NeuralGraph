"""Extracted from graph_trainer.py"""
from ._imports import *

def data_train_zebra(config, erase, best_model, device):
    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    dimension = simulation_config.dimension
    n_epochs = train_config.n_epochs
    n_neurons = simulation_config.n_neurons
    n_input_neurons = simulation_config.n_input_neurons
    n_neuron_types = simulation_config.n_neuron_types
    calcium_type = simulation_config.calcium_type
    delta_t = simulation_config.delta_t

    dataset_name = config.dataset
    n_runs = train_config.n_runs
    n_frames = simulation_config.n_frames

    data_augmentation_loop = train_config.data_augmentation_loop
    recurrent_training = train_config.recurrent_training
    recurrent_loop = train_config.recurrent_loop
    batch_size = train_config.batch_size
    batch_ratio = train_config.batch_ratio
    training_NNR_start_epoch = train_config.training_NNR_start_epoch
    time_window = train_config.time_window
    plot_batch_size = config.plotting.plot_batch_size

    field_type = model_config.field_type
    time_step = train_config.time_step

    coeff_W_sign = train_config.coeff_W_sign
    coeff_update_msg_diff = train_config.coeff_update_msg_diff
    coeff_update_u_diff = train_config.coeff_update_u_diff
    coeff_edge_norm = train_config.coeff_edge_norm
    coeff_update_msg_sign = train_config.coeff_update_msg_sign
    coeff_edge_weight_L2 = train_config.coeff_edge_weight_L2
    coeff_phi_weight_L2 = train_config.coeff_phi_weight_L2
    coeff_NNR_f = train_config.coeff_NNR_f

    replace_with_cluster = 'replace' in train_config.sparsity
    sparsity_freq = train_config.sparsity_freq

    if config.training.seed != 42:
        torch.random.fork_rng(devices=device)
        torch.random.manual_seed(config.training.seed)

    cmap = CustomColorMap(config=config)
    plt.style.use('dark_background')

    coeff_loop = torch.tensor(train_config.coeff_loop, device = device)

    log_dir, logger = create_log_dir(config, erase)
    print(f'loading graph files N: {n_runs} ...')
    logger.info(f'Graph files N: {n_runs}')

    x_list = []
    y_list = []
    for run in trange(0,n_runs):
        x = np.load(f'graphs_data/{dataset_name}/x_list_{run}.npy')
        y = np.load(f'graphs_data/{dataset_name}/y_list_{run}.npy')
        x_list.append(x)
        y_list.append(y)
    print(f'dataset: {len(x_list)} run, {len(x_list[0])} frames')

    if x_list[0].shape[0] < n_frames:
        n_frames = x_list[0].shape[0]
    print(f'number of frames: {n_frames}')
    config.simulation.n_frames = n_frames

    x = x_list[0][n_frames - 10]

    activity = torch.tensor(x_list[0][:, :, 3:4], device=device)
    activity = activity.squeeze()
    distrib = activity.flatten()
    valid_distrib = distrib[~torch.isnan(distrib)]
    if len(valid_distrib) > 0:
        xnorm = torch.round(1.5 * torch.std(valid_distrib))
    else:
        print('no valid distribution found, setting xnorm to 1.0')
        xnorm = torch.tensor(1.0, device=device)
    torch.save(xnorm, os.path.join(log_dir, 'xnorm.pt'))
    print(f'xnorm: {to_numpy(xnorm)}')
    logger.info(f'xnorm: {to_numpy(xnorm)}')

    n_neurons = x.shape[0]
    print(f'N neurons: {n_neurons}')
    logger.info(f'N neurons: {n_neurons}')
    config.simulation.n_neurons =n_neurons
    type_list = torch.tensor(x[:, 2 + 2 * dimension:3 + 2 * dimension], device=device)
    vnorm = torch.tensor(1.0, device=device)
    ynorm = torch.tensor(1.0, device=device)
    torch.save(vnorm, os.path.join(log_dir, 'vnorm.pt'))
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))
    time.sleep(0.5)
    print(f'vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')
    logger.info(f'vnorm ynorm: {to_numpy(vnorm)} {to_numpy(ynorm)}')

    print('create models ...')
    model = Signal_Propagation_Zebra(aggr_type=model_config.aggr_type, config=config, device=device)

    start_epoch = 0
    list_loss = []
    if (best_model != None) & (best_model != ''):
        net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs_{best_model}.pt"
        print(f'load {net} ...')
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        start_epoch = int(best_model.split('_')[0])
        print(f'best_model: {best_model}  start_epoch: {start_epoch}')
        logger.info(f'best_model: {best_model}  start_epoch: {start_epoch}')
        list_loss = torch.load(f"{log_dir}/loss.pt")
    elif  train_config.pretrained_model !='':
        net = train_config.pretrained_model
        print(f'load pretrained {net} ...')
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        logger.info(f'pretrained: {net}')

    lr = train_config.learning_rate_start
    if train_config.learning_rate_update_start == 0:
        lr_update = train_config.learning_rate_start
    else:
        lr_update = train_config.learning_rate_update_start
    learning_rate_NNR_f = train_config.learning_rate_NNR_f
    learning_rate_NNR_f_start = train_config.learning_rate_NNR_f_start
    if learning_rate_NNR_f_start == 0:
        learning_rate_NNR_f_start = learning_rate_NNR_f
    lr_embedding = train_config.learning_rate_embedding_start
    lr_W = train_config.learning_rate_W_start

    print(
        f'learning rates: lr_W {lr_W}, lr {lr}, lr_update {lr_update}, lr_embedding {lr_embedding}, learning_rate_NNR_f {learning_rate_NNR_f}')
    logger.info(
        f'learning rates: lr_W {lr_W}, lr {lr}, lr_update {lr_update}, lr_embedding {lr_embedding}, learning_rate_NNR {learning_rate_NNR_f}')

    optimizer, n_total_params = set_trainable_parameters(model=model, lr_embedding=lr_embedding, lr=lr,
                                                         lr_update=lr_update, lr_W=lr_W, learning_rate_NNR=None, learning_rate_NNR_f=learning_rate_NNR_f_start)
    model.train()

    net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs.pt"
    print(f'network: {net}')
    print(f'initial batch_size: {batch_size}')
    logger.info(f'network: {net}')
    logger.info(f'N epochs: {n_epochs}')
    logger.info(f'initial batch_size: {batch_size}')

    # connectivity = torch.load(f'./graphs_data/{dataset_name}/connectivity.pt', map_location=device)

    if coeff_W_sign > 0:
        index_weight = []
        for i in range(n_neurons):
            index_weight.append(torch.argwhere(model.mask[:, i] > 0).squeeze())

    coeff_W_L1 = train_config.coeff_W_L1
    coeff_edge_diff = train_config.coeff_edge_diff
    coeff_update_diff = train_config.coeff_update_diff
    logger.info(f'coeff_W_L1: {coeff_W_L1} coeff_edge_diff: {coeff_edge_diff} coeff_update_diff: {coeff_update_diff}')
    print(f'coeff_W_L1: {coeff_W_L1} coeff_edge_diff: {coeff_edge_diff} coeff_update_diff: {coeff_update_diff}')

    print("start training ...")

    check_and_clear_memory(device=device, iteration_number=0, every_n_iterations=1, memory_percentage_threshold=0.6)
    # torch.autograd.set_detect_anomaly(True)

    list_loss_regul = []
    time.sleep(0.2)

    ones = torch.ones((n_neurons, 1), dtype=torch.float32, device=device)

    Niter = int(n_frames * data_augmentation_loop // batch_size / batch_ratio)
    plot_frequency = int(Niter // 5)
    print(f'{Niter} iterations per epoch')
    logger.info(f'{Niter} iterations per epoch')
    print(f'plot every {plot_frequency} iterations')

    print("start training ...")

    check_and_clear_memory(device=device, iteration_number=0, every_n_iterations=1, memory_percentage_threshold=0.6)
    # torch.autograd.set_detect_anomaly(True)

    time.sleep(0.2)

    list_loss = []

    N_slices = 72
    z_min = x_list[0][0,:,3:4].min()
    z_max = x_list[0][0,:,3:4].max()
    z_thickness = 0.28 / N_slices # guard against z == z_max landing on index 72
    z_thickness = torch.tensor(z_thickness, dtype=torch.float32, device=device)
    delta_t_step = torch.tensor(delta_t/N_slices, dtype=torch.float32, device=device)

    print(f'z: {z_min:0.5f} - {z_max:0.5f}    z_thickness {to_numpy(z_thickness):0.5f}    delta_t_step: {to_numpy(delta_t_step):0.5f} / {delta_t}')


    for epoch in range(start_epoch, n_epochs + 1):

        total_loss = torch.tensor(0.0, dtype=torch.float32, device=device)

        if epoch == 2:
            optimizer, n_total_params = set_trainable_parameters(model=model, lr_embedding=lr_embedding, lr=lr,
                                                                lr_update=lr_update, lr_W=lr_W, learning_rate_NNR=None, learning_rate_NNR_f=learning_rate_NNR_f)
            model.train()

        for N in trange(Niter):

            optimizer.zero_grad()

            loss = 0

            dataset_batch = []
            ids_batch = []
            ids_index = 0
            edges = []

            for batch in range(batch_size):

                k = np.random.randint(n_frames - 1)

                ids = np.arange(n_neurons)

                if batch_ratio < 1:
                    ids_ = np.random.permutation(ids.shape[0])[:int(ids.shape[0] * batch_ratio)]
                    ids = np.sort(ids_)
                    # edges = edges_all.clone().detach()
                    # mask = torch.isin(edges[1, :], torch.tensor(ids, device=device))
                    # edges = edges[:, mask]


                x = torch.tensor(x_list[run][k, :, 0:7], dtype=torch.float32, device=device).clone().detach()
                y = torch.tensor(x_list[run][k, :, 6:7], dtype=torch.float32, device=device).clone().detach()

                dataset = pyg_Data(x=x, edge_index=edges)
                dataset_batch.append(dataset)

                k_t = torch.ones((x.shape[0], 1), dtype=torch.float32, device=device) * k * delta_t
                k_t = k_t + torch.floor(x[:,3:4] / z_thickness) * delta_t_step # correction for light sheet acquisition

                if batch == 0:

                    data_id = torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * run
                    y_batch = y
                    k_batch = k_t
                    ids_batch = ids

                else:

                    data_id = torch.cat((data_id, torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * run), dim=0)
                    y_batch = torch.cat((y_batch, y), dim=0)
                    k_batch = torch.cat((k_batch, k_t), dim=0)
                    ids_batch = np.concatenate((ids_batch, ids + ids_index), axis=0)

                ids_index += x.shape[0]

            ids_batch_t = torch.as_tensor(ids_batch, device=device, dtype=torch.long)
            batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)

            for batch in batch_loader:
                pred, field_f, field_f_laplacians = model(batch, data_id=data_id, k=k_batch, ids=ids_batch_t)

            loss = loss + (field_f[:,None] - y_batch[ids_batch]).norm(2)
            if coeff_NNR_f > 0:
                loss = loss + (field_f_laplacians).norm(2)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (N % plot_frequency == 0):
                x = torch.tensor(x_list[run][20], dtype=torch.float32, device=device)
                with torch.no_grad():
                    plot_field_comparison(x, model, 20, n_frames, ones, f"./{log_dir}/tmp_training/field/field_{epoch}_{N}.png", 100, plot_batch_size)

                torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))

        print("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / n_neurons))
        logger.info("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / n_neurons))
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}.pt'))

        list_loss.append(to_numpy(total_loss) / n_neurons)
        # torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))

        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(1, 2, 1)
        plt.plot(list_loss, color='w', linewidth=1)
        plt.xlim([0, n_epochs])
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylabel('loss', fontsize=12)
        plt.xlabel('epochs', fontsize=12)
        ax = fig.add_subplot(1, 2, 2)
        field_files = glob.glob(f"./{log_dir}/tmp_training/field/*.png")
        last_file = max(field_files, key=os.path.getctime)  # or use os.path.getmtime for modification time
        filename = os.path.basename(last_file)
        filename = filename.replace('.png', '')
        parts = filename.split('_')
        if len(parts) >= 3:
            last_epoch = parts[1]
            last_N = parts[2]
        else:
            last_epoch, last_N = parts[-2], parts[-1]
        img = imageio.imread(f"./{log_dir}/tmp_training/field/field_{last_epoch}_{last_N}.png")
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/epoch_{epoch}.tif")
        plt.close()

