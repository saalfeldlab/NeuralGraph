"""Extracted from graph_trainer.py"""
from ._imports import *

def data_train_flyvis_calcium(config, erase, best_model, device):
    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    signal_model_name = model_config.signal_model_name

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

    field_type = model_config.field_type
    time_step = train_config.time_step

    coeff_W_sign = train_config.coeff_W_sign
    coeff_update_msg_diff = train_config.coeff_update_msg_diff
    coeff_update_u_diff = train_config.coeff_update_u_diff
    coeff_edge_norm = train_config.coeff_edge_norm
    coeff_update_msg_sign = train_config.coeff_update_msg_sign
    coeff_edge_weight_L2 = train_config.coeff_edge_weight_L2
    coeff_phi_weight_L2 = train_config.coeff_phi_weight_L2
    coeff_loop = torch.tensor(train_config.coeff_loop, device = device)

    # VAE parameters
    latent_dim = model_config.latent_dim
    latent_update_steps = model_config.latent_update_steps
    stochastic_latent = model_config.stochastic_latent

    lr_encoder = train_config.learning_rate_encoder
    lr_latent_update = train_config.learning_rate_latent_update
    lr_decoder = train_config.learning_rate_decoder


    replace_with_cluster = 'replace' in train_config.sparsity
    sparsity_freq = train_config.sparsity_freq

    n_edges = simulation_config.n_edges
    n_extra_null_edges = simulation_config.n_extra_null_edges

    if config.training.seed != 42:
        torch.random.fork_rng(devices=device)
        torch.random.manual_seed(config.training.seed)

    torch.set_grad_enabled(True)

    cmap = CustomColorMap(config=config)

    if 'visual' in field_type:
        has_visual_field = True
        print('train with visual field NNR')
    else:
        has_visual_field = False

    log_dir, logger = create_log_dir(config, erase)

    x_list = []
    y_list = []
    for run in trange(0,n_runs):
        x = np.load(f'graphs_data/{dataset_name}/x_list_{run}.npy')
        y = np.load(f'graphs_data/{dataset_name}/y_list_{run}.npy')
        x_list.append(x)
        y_list.append(y)
    print(f'dataset: {len(x_list)} run, {len(x_list[0])} frames')
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
    ynorm = torch.tensor(1.0, device=device)
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))
    time.sleep(0.5)
    print(f'ynorm: {to_numpy(ynorm)}')
    logger.info(f'ynorm: {to_numpy(ynorm)}')

    print('create models ...')
    model = Calcium_Latent_Dynamics(config=config, device=device)

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



    optimizer, n_total_params = set_trainable_parameters_vae( model=model, lr_encoder=lr_encoder, lr_latent_update=lr_latent_update, lr_decoder=lr_decoder)         # optional WD for all groups
    model.train()

    print(
        f'n_total_params: {n_total_params}, learning rates: lr_encoder {lr_encoder}, lr_latent_update {lr_latent_update}, lr_decoder {lr_decoder}')
    logger.info(
        f'n_total_params: {n_total_params}, learning rates: lr_encoder {lr_encoder}, lr_latent_update {lr_latent_update}, lr_decoder {lr_decoder}')


    net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs.pt"
    print(f'network: {net}')
    print(f'initial batch_size: {batch_size}')
    logger.info(f'network: {net}')
    logger.info(f'N epochs: {n_epochs}')
    logger.info(f'initial batch_size: {batch_size}')

    # connectivity = torch.load(f'./graphs_data/{dataset_name}/connectivity.pt', map_location=device)
    gt_weights = torch.load(f'./graphs_data/{dataset_name}/weights.pt', map_location=device)
    edges = torch.load(f'./graphs_data/{dataset_name}/edge_index.pt', map_location=device)
    edges_all = edges.clone().detach()
    print(f'{edges.shape[1]} edges')

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

    for epoch in range(start_epoch, n_epochs + 1):

        if batch_ratio < 1:
            Niter = int(n_frames * data_augmentation_loop // batch_size / batch_ratio * 0.2)
        else:
            Niter = int(n_frames * data_augmentation_loop // batch_size * 0.2 // max(recurrent_loop, 1))

        plot_frequency = int(Niter // 20)
        print(f'{Niter} iterations per epoch')
        logger.info(f'{Niter} iterations per epoch')
        print(f'plot every {plot_frequency} iterations')

        total_loss = 0
        total_loss_regul = 0
        k = 0

        # anneal loss_noise_level, decrease with epoch
        loss_noise_level = train_config.loss_noise_level * (0.95 ** epoch)


        for N in trange(Niter):

            optimizer.zero_grad()

            loss = 0
            run = np.random.randint(n_runs)

            x_batch = list([])
            y_batch = list([])
            ids_batch = list([])

            loss = torch.zeros(1, device=device)

            for batch in range(batch_size):

                k = np.random.randint(n_frames - 4 - time_step - time_window) + time_window
                k = k - k % latent_update_steps     # we sample data every latent_update_steps
                x = torch.tensor(x_list[run][k,:,7:8], dtype=torch.float32, device=device)
                y = torch.tensor(x_list[run][k+1,:,7:8], device=device)   # auto-encoder_loss
                ids = np.arange(n_neurons)
                if not (torch.isnan(x).any()):
                    x_batch.append(x)
                    if batch_ratio < 1:
                        ids_ = np.random.permutation(ids.shape[0])[:int(ids.shape[0] * batch_ratio)]
                        ids = np.sort(ids_)
                    ids_batch.append(ids)
                    if loss_noise_level>0:
                        y = y + torch.randn(y.shape, device=device) * loss_noise_level
                    y_batch.append(y)

            x_batch = torch.stack(x_batch, dim=0).squeeze(-1)
            y_batch = torch.stack(y_batch, dim=0).squeeze(-1)
            ids_batch = np.concatenate(ids_batch, axis=0)

            pred, mu, logvar  = model(x_batch)

            loss = loss + (pred - y_batch).norm(2)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if ((N % plot_frequency == 0) | (N == 0)):

                torch.save(
                    {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                    os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))

            # check_and_clear_memory(device=device, iteration_number=N, every_n_iterations=Niter // 50, memory_percentage_threshold=0.6)

        print("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / n_neurons))
        logger.info("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / n_neurons))
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}.pt'))

        list_loss.append((total_loss-total_loss_regul) / n_neurons)

        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))

        fig = plt.figure(figsize=(15, 10))

        ax = fig.add_subplot(2, 3, 1)
        plt.plot(list_loss, color='k', linewidth=1)
        plt.xlim([0, n_epochs])
        plt.ylabel('loss', fontsize=12)
        plt.xlabel('epochs', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/epoch_{epoch}.tif")
        plt.close()

