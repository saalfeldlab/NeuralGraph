"""Extracted from graph_trainer.py"""
from ._imports import *

def data_train_signal(config, erase, best_model, device):

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    n_epochs = train_config.n_epochs
    n_runs = train_config.n_runs
    n_neuron_types = simulation_config.n_neuron_types

    dataset_name = config.dataset
    n_frames = simulation_config.n_frames
    dimension = simulation_config.dimension
    data_augmentation_loop = train_config.data_augmentation_loop
    recurrent_loop = train_config.recurrent_loop
    recurrent_parameters = train_config.recurrent_parameters.copy()
    target_batch_size = train_config.batch_size
    delta_t = simulation_config.delta_t
    if train_config.small_init_batch_size:
        get_batch_size = increasing_batch_size(target_batch_size)
    else:
        get_batch_size = constant_batch_size(target_batch_size)
    batch_size = get_batch_size(0)
    batch_ratio = train_config.batch_ratio
    replace_with_cluster = 'replace' in train_config.sparsity
    sparsity_freq = train_config.sparsity_freq


    field_type = model_config.field_type

    embedding_cluster = EmbeddingCluster(config)

    time_step = train_config.time_step
    has_missing_activity = train_config.has_missing_activity
    multi_connectivity = config.training.multi_connectivity
    baseline_value = simulation_config.baseline_value
    cmap = CustomColorMap(config=config)

    if config.training.seed != 42:
        torch.random.fork_rng(devices=device)
        torch.random.manual_seed(config.training.seed)
        np.random.seed(config.training.seed)

    if field_type != '':
        n_nodes = simulation_config.n_nodes
        has_neural_field = True
    else:
        n_nodes = simulation_config.n_neurons
        has_neural_field = False

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
    x = x_list[0][n_frames - 10]
    n_neurons = x.shape[0]
    config.simulation.n_neurons =n_neurons
    type_list = torch.tensor(x[:, 1 + 2 * dimension:2 + 2 * dimension], device=device)

    activity = torch.tensor(x_list[0][:, :, 6], device=device)
    distrib = activity.flatten()
    distrib = distrib[~torch.isnan(distrib)]
    if len(distrib) > 0:
        xnorm = torch.round(1.5 * torch.std(distrib))
    else:
        print('no valid distribution found, setting xnorm to 1.0')
        xnorm = torch.tensor(1.0, device=device)
    vnorm = torch.tensor(1.0, device=device)
    ynorm = torch.tensor(1.0, device=device)
    torch.save(xnorm, os.path.join(log_dir, 'xnorm.pt'))
    torch.save(vnorm, os.path.join(log_dir, 'vnorm.pt'))
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))
    time.sleep(0.5)

    print(f'N neurons: {n_neurons}, xnorm: {to_numpy(xnorm)}, vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')
    logger.info(f'N neurons: {n_neurons}, xnorm: {to_numpy(xnorm)}, vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')


    print('create models ...')
    model, bc_pos, bc_dpos = choose_training_model(model_config=config, device=device)
    model.train()

    if has_missing_activity:
        assert batch_ratio == 1, f"batch_ratio must be 1, got {batch_ratio}"
        model_missing_activity = nn.ModuleList([
            Siren(in_features=model_config.input_size_nnr, out_features=model_config.output_size_nnr,
                  hidden_features=model_config.hidden_dim_nnr,
                  hidden_layers=model_config.n_layers_nnr, first_omega_0=model_config.omega,
                  hidden_omega_0=model_config.omega,
                  outermost_linear=model_config.outermost_linear_nnr)
            for n in range(n_runs)
        ])
        model_missing_activity.to(device=device)
        optimizer_missing_activity = torch.optim.Adam(lr=train_config.learning_rate_missing_activity,
                                                      params=model_missing_activity.parameters())
        model_missing_activity.train()
    if has_neural_field:
        modulation = None
        if ('short_term_plasticity' in field_type) | ('modulation' in field_type):
            model_f = nn.ModuleList([
                Siren(in_features=model_config.input_size_nnr, out_features=model_config.output_size_nnr,
                      hidden_features=model_config.hidden_dim_nnr,
                      hidden_layers=model_config.n_layers_nnr, first_omega_0=model_config.omega,
                      hidden_omega_0=model_config.omega,
                      outermost_linear=model_config.outermost_linear_nnr)
                for n in range(n_runs)
            ])
            if ('short_term_plasticity' in field_type):
                modulation = torch.tensor(x_list[0], device=device)
                modulation = modulation[:, :, 8:9].squeeze()
                modulation = modulation.t()
                modulation = modulation.clone().detach()
                d_modulation = (modulation[:, 1:] - modulation[:, :-1]) / delta_t
                modulation_norm = torch.tensor(1.0E-2, device=device)
        elif 'visual' in field_type:
            n_nodes_per_axis = int(np.sqrt(n_nodes))
            model_f = Siren_Network(image_width=n_nodes_per_axis, in_features=model_config.input_size_nnr,
                                    out_features=model_config.output_size_nnr,
                                    hidden_features=model_config.hidden_dim_nnr,
                                    hidden_layers=model_config.n_layers_nnr, outermost_linear=True, device=device,
                                    first_omega_0=model_config.omega, hidden_omega_0=model_config.omega)
        model_f.to(device=device)
        optimizer_f = torch.optim.Adam(lr=train_config.learning_rate_NNR, params=model_f.parameters())
        model_f.train()

    if (best_model != None) & (best_model != '') & (best_model != 'None'):
        net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs_{best_model}.pt"
        print(f'load {net} ...')
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        start_epoch = int(best_model.split('_')[0])
        print(f'best_model: {best_model}  start_epoch: {start_epoch}')
        logger.info(f'best_model: {best_model}  start_epoch: {start_epoch}')
        if has_neural_field:
            net = f'{log_dir}/models/best_model_f_with_{n_runs - 1}_graphs_{best_model}.pt'
            state_dict = torch.load(net, map_location=device)
            model_f.load_state_dict(state_dict['model_state_dict'])
        list_loss = torch.load(os.path.join(log_dir, 'loss.pt'))
    else:
        start_epoch = 0
        list_loss = []

    print('set optimizer ...')
    lr = train_config.learning_rate_start
    if train_config.learning_rate_update_start == 0:
        lr_update = train_config.learning_rate_start
    else:
        lr_update = train_config.learning_rate_update_start
    lr_embedding = train_config.learning_rate_embedding_start
    lr_W = train_config.learning_rate_W_start
    lr_modulation = train_config.learning_rate_modulation_start
    optimizer, n_total_params = set_trainable_parameters(model=model, lr_embedding=lr_embedding, lr=lr,
                                                        lr_update=lr_update, lr_W=lr_W, lr_modulation=lr_modulation)

    print(f'learning rates: lr_W {lr_W}, lr {lr}, lr_update {lr_update}, lr_embedding {lr_embedding}, lr_modulation {lr_modulation}')
    logger.info(f'learning rates: lr_W {lr_W}, lr {lr}, lr_update {lr_update}, lr_embedding {lr_embedding}, lr_modulation {lr_modulation}')


    net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs.pt"
    print(f'network: {net}')
    print(f'initial batch_size: {batch_size}')
    logger.info(f'network: {net}  N epochs: {n_epochs}  initial batch_size: {batch_size}')


    print('training setup ...')
    connectivity = torch.load(f'./graphs_data/{dataset_name}/connectivity.pt', map_location=device)

    if train_config.with_connectivity_mask:
        model.mask = (connectivity > 0) * 1.0
        adj_t = model.mask.float() * 1
        adj_t = adj_t.t()
        edges = adj_t.nonzero().t().contiguous()
        edges_all = edges.clone().detach()

        with torch.no_grad():
            if multi_connectivity:
                for run_ in range(n_runs):
                    model.W[run_].copy_(model.W[run_] * model.mask)
            else:
                model.W.copy_(model.W * model.mask)
    else:
        edges = torch.load(f'./graphs_data/{dataset_name}/edge_index.pt', map_location=device)
        edges_all = edges.clone().detach()

    if train_config.coeff_W_sign > 0:
        index_weight = []
        for i in range(n_neurons):
            index_weight.append(torch.argwhere(model.mask[:, i] > 0).squeeze())

    print(f'{edges.shape[1]} edges')

    if 'PDE_N3' in model_config.signal_model_name:          # PDE_N3 is special, embedding changes over time
        ind_a = torch.tensor(np.arange(1, n_neurons * 100), device=device)
        pos = torch.argwhere(ind_a % 100 != 99).squeeze()
        ind_a = ind_a[pos]

    coeff_lin_phi_zero = train_config.coeff_lin_phi_zero
    coeff_lin_modulation = train_config.coeff_lin_modulation
    coeff_model_b = train_config.coeff_model_b
    coeff_W_sign = train_config.coeff_W_sign
    coeff_update_msg_diff = train_config.coeff_update_msg_diff
    coeff_update_u_diff = train_config.coeff_update_u_diff
    coeff_edge_norm = train_config.coeff_edge_norm
    coeff_update_msg_sign = train_config.coeff_update_msg_sign
    coeff_W_L1 = train_config.coeff_W_L1
    coeff_edge_diff = train_config.coeff_edge_diff
    coeff_update_diff = train_config.coeff_update_diff

    print(f'coeff_W_L1: {coeff_W_L1} coeff_edge_diff: {coeff_edge_diff} coeff_update_diff: {coeff_update_diff}')
    logger.info(f'coeff_W_L1: {coeff_W_L1} coeff_edge_diff: {coeff_edge_diff} coeff_update_diff: {coeff_update_diff}')

    print("start training ...")

    check_and_clear_memory(device=device, iteration_number=0, every_n_iterations=1, memory_percentage_threshold=0.6)
    # torch.autograd.set_detect_anomaly(True)

    list_loss_regul = []
    time.sleep(1.0)

    for epoch in range(start_epoch, n_epochs + 1):

        if (epoch == train_config.epoch_reset) | ((epoch > 0) & (epoch % train_config.epoch_reset_freq == 0)):
            with torch.no_grad():
                model.W.copy_(model.W * 0)
                model.a.copy_(model.a * 0)
            logger.info(f'reset W model.a at epoch : {epoch}')
            print(f'reset W model.a at epoch : {epoch}')
        if epoch == train_config.n_epochs_init:
            coeff_edge_diff = coeff_update_diff / 100
            coeff_update_diff = coeff_update_diff / 100
            logger.info(f'coeff_W_L1: {coeff_W_L1} coeff_edge_diff: {coeff_edge_diff} coeff_update_diff: {coeff_update_diff}')
            print(f'coeff_W_L1: {coeff_W_L1} coeff_edge_diff: {coeff_edge_diff} coeff_update_diff: {coeff_update_diff}')

        batch_size = get_batch_size(epoch)
        logger.info(f'batch_size: {batch_size}')

        if batch_ratio < 1:
            Niter = int(n_frames * data_augmentation_loop // batch_size / batch_ratio * 0.2)
        else:
            Niter = int(n_frames * data_augmentation_loop // batch_size * 0.2 // max(recurrent_loop, 1))

        plot_frequency = int(Niter // 5)
        print(f'{Niter} iterations per epoch, {plot_frequency} iterations per plot')
        logger.info(f'{Niter} iterations per epoch')

        total_loss = 0
        total_loss_regul = 0
        k = 0

        time.sleep(1.0)
        for N in trange(Niter):

            if has_missing_activity:
                optimizer_missing_activity.zero_grad()
            if has_neural_field:
                optimizer_f.zero_grad()
            optimizer.zero_grad()

            dataset_batch = []
            ids_batch = []
            ids_index = 0

            loss = 0
            run = np.random.randint(n_runs-1)

            for batch in range(batch_size):

                k = np.random.randint(n_frames - 4 - time_step)

                x = torch.tensor(x_list[run][k], dtype=torch.float32, device=device)
                ids = torch.argwhere(x[:, 6] != baseline_value)
                ids = to_numpy(ids.squeeze())

                if not (torch.isnan(x).any()):
                    if has_missing_activity:
                        t = torch.tensor([k / n_frames], dtype=torch.float32, device=device)
                        missing_activity = baseline_value + model_missing_activity[run](t).squeeze()
                        if (train_config.coeff_missing_activity>0):
                            loss_missing_activity = (missing_activity[ids] - x[ids, 6].clone().detach()).norm(2)
                            loss = loss + loss_missing_activity * train_config.coeff_missing_activity
                        ids_missing = torch.argwhere(x[:, 6] == baseline_value)
                        x[ids_missing,6] = missing_activity[ids_missing]
                    if has_neural_field:
                        if 'visual' in field_type:
                            x[:n_nodes, 8:9] = model_f(time=k / n_frames) ** 2
                            x[n_nodes:n_neurons, 8:9] = 1
                        elif 'learnable_short_term_plasticity' in field_type:
                            alpha = (k % model.embedding_step) / model.embedding_step
                            x[:, 8] = alpha * model.b[:, k // model.embedding_step + 1] ** 2 + (1 - alpha) * model.b[:,
                                                                                                             k // model.embedding_step] ** 2
                            loss = loss + (model.b[:, 1:] - model.b[:, :-1]).norm(2) * coeff_model_b
                        elif ('short_term_plasticity' in field_type) | ('modulation' in field_type):
                            t = torch.tensor([k / n_frames], dtype=torch.float32, device=device)
                            if 'derivative' in field_type:
                                m = model_f[run](t) ** 2
                                x[:, 8] = m
                                m_next = model_f[run](t + 1.0E-3).squeeze() ** 2
                                grad = (m_next - m) / 1.0E-3
                                in_modulation = torch.cat((x[:, 6:7].clone().detach(), m[:, None]), dim=1)
                                pred_modulation = model.lin_modulation(in_modulation)
                                loss += (grad - pred_modulation.squeeze()).norm(2) * coeff_lin_modulation
                            else:
                                x[:, 8] = model_f[run](t) ** 2
                    else:
                        x[:, 8:9] = torch.ones_like(x[:, 0:1])

                    if multi_connectivity:
                        model_W = model.W[run]
                    else:
                        model_W = model.W

                    # regularisation lin_phi(0)=0
                    if coeff_lin_phi_zero > 0:
                        in_features = get_in_features_update(rr=None, model=model, device=device)
                        func_phi = model.lin_phi(in_features[ids].float())
                        loss = loss + func_phi.norm(2) * coeff_lin_phi_zero
                    # regularisation sparsity on Wij
                    if coeff_W_L1 > 0:
                        loss = loss + model_W[:n_neurons, :n_neurons].norm(1) * coeff_W_L1
                    # regularisation lin_edge
                    in_features, in_features_next = get_in_features_lin_edge(x, model, model_config, xnorm, n_neurons,device)
                    if coeff_edge_diff > 0:
                        if model_config.lin_edge_positive:
                            msg0 = model.lin_edge(in_features[ids].clone().detach()) ** 2
                            msg1 = model.lin_edge(in_features_next[ids].clone().detach()) ** 2
                        else:
                            msg0 = model.lin_edge(in_features[ids].clone().detach())
                            msg1 = model.lin_edge(in_features_next[ids].clone().detach())
                        loss = loss + torch.relu(msg0 - msg1).norm(2) * coeff_edge_diff      # lin_edge monotonically increasing  over voltage for all embedding values
                    if coeff_edge_norm > 0:
                        in_features[:,0] = 2 * xnorm
                        if model_config.lin_edge_positive:
                            msg = model.lin_edge(in_features[ids].clone().detach()) ** 2
                        else:
                            msg = model.lin_edge(in_features[ids].clone().detach())
                        loss = loss + (msg-1).norm(2) * coeff_edge_norm                 # normalization lin_edge(xnorm) = 1 for all embedding values
                    # regularisation sign Wij
                    if (coeff_W_sign > 0) and (N%4 == 0):
                        W_sign = torch.tanh(5 * model_W)
                        loss_contribs = []
                        for i in range(n_neurons):
                            indices = index_weight[int(i)]
                            if indices.numel() > 0:
                                values = W_sign[indices,i]
                                std = torch.std(values, unbiased=False)
                                loss_contribs.append(std)
                        if loss_contribs:
                            loss = loss + torch.stack(loss_contribs).norm(2) * coeff_W_sign
                    # miscalleneous regularisations
                    if (model.update_type == 'generic') & (coeff_update_diff > 0):
                        in_feature_update = torch.cat((torch.zeros((n_neurons, 1), device=device),
                                                       model.a[:n_neurons], msg0,
                                                       torch.ones((n_neurons, 1), device=device)), dim=1)
                        in_feature_update = in_feature_update[ids]
                        in_feature_update_next = torch.cat((torch.zeros((n_neurons, 1), device=device),
                                                            model.a[:n_neurons], msg1,
                                                            torch.ones((n_neurons, 1), device=device)), dim=1)
                        in_feature_update_next = in_feature_update_next[ids]
                        if 'positive' in train_config.diff_update_regul:
                            loss = loss + torch.relu(model.lin_phi(in_feature_update) - model.lin_phi(in_feature_update_next)).norm(2) * coeff_update_diff
                        if 'TV' in train_config.diff_update_regul:
                            in_feature_update_next_bis = torch.cat((torch.zeros((n_neurons, 1), device=device),
                                                                    model.a[:n_neurons], msg1,
                                                                    torch.ones((n_neurons, 1), device=device) * 1.1),
                                                                   dim=1)
                            in_feature_update_next_bis = in_feature_update_next_bis[ids]
                            loss = loss + (model.lin_phi(in_feature_update) - model.lin_phi(in_feature_update_next_bis)).norm(2) * coeff_update_diff
                        if 'second_derivative' in train_config.diff_update_regul:
                            in_feature_update_prev = torch.cat((torch.zeros((n_neurons, 1), device=device),
                                                                model.a[:n_neurons], msg1,
                                                                torch.ones((n_neurons, 1), device=device)), dim=1)
                            in_feature_update_prev = in_feature_update_prev[ids]
                            loss = loss + (model.lin_phi(in_feature_update_prev) + model.lin_phi(
                                in_feature_update_next) - 2 * model.lin_phi(in_feature_update)).norm(
                                2) * coeff_update_diff

                    if batch_ratio < 1:
                        ids_ = np.random.permutation(ids.shape[0])[:int(ids.shape[0] * batch_ratio)]
                        ids = np.sort(ids_)
                        edges = edges_all.clone().detach()
                        mask = torch.isin(edges[1, :], torch.tensor(ids, device=device))
                        edges = edges[:, mask]

                    if recurrent_loop > 1:
                        y = torch.tensor(y_list[run][k + recurrent_loop], device=device) / ynorm
                    elif time_step == 1:
                        y = torch.tensor(y_list[run][k], device=device) / ynorm
                    elif time_step > 1:
                        y = torch.tensor(x_list[run][k + time_step, :, 6:7], device=device).clone().detach()

                    if not (torch.isnan(y).any()):

                        dataset = pyg_Data(x=x, edge_index=edges)
                        dataset_batch.append(dataset)

                        if len(dataset_batch) == 1:
                            data_id = torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * run
                            x_batch = x[:, 6:7]
                            y_batch = y
                            k_batch = torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * k
                            ids_batch = ids
                        else:
                            data_id = torch.cat((data_id, torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * run), dim=0)
                            x_batch = torch.cat((x_batch, x[:, 6:7]), dim=0)
                            y_batch = torch.cat((y_batch, y), dim=0)
                            k_batch = torch.cat(
                                (k_batch, torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * k), dim=0)
                            ids_batch = np.concatenate((ids_batch, ids + ids_index), axis=0)

                        ids_index += x.shape[0]

            if not (dataset_batch == []):

                total_loss_regul += loss.item()

                batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
                for batch in batch_loader:
                    if (coeff_update_msg_diff > 0) | (coeff_update_u_diff > 0) | (coeff_update_msg_sign>0):
                        pred, in_features = model(batch, data_id=data_id, k=k_batch, return_all=True)
                        if coeff_update_msg_diff > 0 : # Penalized when pred_u_next > pred (output increases with voltage)
                            pred_msg = model.lin_phi(in_features)
                            in_features_msg_next = in_features.clone().detach()
                            in_features_msg_next[:, model_config.embedding_dim+1] = in_features_msg_next[:, model_config.embedding_dim+1] * 1.05
                            pred_msg_next = model.lin_phi(in_features_msg_next.clone().detach())
                            loss = loss + torch.relu(pred_msg[ids_batch]-pred_msg_next[ids_batch]).norm(2) * coeff_update_msg_diff
                        if coeff_update_u_diff > 0: #  Penalizes when pred > pred_msg_next (output decreases with message)
                            pred_u =  model.lin_phi(in_features)
                            in_features_u_next = in_features.clone().detach()
                            in_features_u_next[:, 0] = in_features_u_next[:, 0] * 1.05  # Perturb voltage (first column)
                            pred_u_next = model.lin_phi(in_features_u_next.clone().detach())
                            loss = loss + torch.relu(pred_u_next[ids_batch] - pred_u[ids_batch]).norm(2) * coeff_update_u_diff
                        if coeff_update_msg_sign > 0: # Penalizes when pred_msg not of same sign as msg
                            in_features_modified = in_features.clone().detach()
                            in_features_modified[:, 0] = 0
                            pred_msg = model.lin_phi(in_features_modified)
                            msg = in_features[:,model_config.embedding_dim+1].clone().detach()
                            loss = loss + (torch.tanh(pred_msg / 0.001) - torch.tanh(msg / 0.001)).norm(2) * coeff_update_msg_sign
                    # Enable gradients for direct derivative computation
                    # in_features.requires_grad_(True)
                    # pred = model.lin_phi(in_features)
                    # grad_u = torch.autograd.grad(pred.sum(), in_features, retain_graph=True)[0][:, 0]
                    # grad_msg = torch.autograd.grad(pred.sum(), in_features)[0][:, model_config.embedding_dim]
                    # loss += torch.relu(grad_u[ids_batch]).norm(2) * coeff_update_u_diff
                    # loss += torch.relu(-grad_msg[ids_batch]).norm(2) * coeff_update_msg_diff
                    else:
                        pred = model(batch, data_id=data_id, k=k_batch)

                if time_step == 1:
                    loss = loss + (pred[ids_batch] - y_batch[ids_batch]).norm(2)

                elif time_step > 1:
                    loss = loss + (x_batch[ids_batch] + pred[ids_batch] * delta_t * time_step - y_batch[ids_batch]).norm(2)

                if 'PDE_N3' in model_config.signal_model_name:
                    loss = loss + train_config.coeff_model_a * (model.a[ind_a + 1] - model.a[ind_a]).norm(2)

                loss.backward()
                optimizer.step()
                if has_missing_activity:
                    optimizer_missing_activity.step()
                if has_neural_field:
                    optimizer_f.step()

                total_loss += loss.item()

                if ((N % plot_frequency == 0) | (N == 0)):
                    plot_training_signal(config, model, x, connectivity, log_dir, epoch, N, n_neurons, type_list, cmap,
                                         device)
                    if time_step > 1:
                        fig = plt.figure(figsize=(10, 10))
                        plt.scatter(to_numpy(y_batch), to_numpy(x_batch + pred * delta_t * time_step), s=10, color='k')
                        plt.scatter(to_numpy(y_batch), to_numpy(x_batch), s=1, color='b', alpha=0.5)
                        plt.plot(to_numpy(y_batch), to_numpy(y_batch), color='g')

                        x_data = y_batch
                        y_data = x_batch
                        err0 = torch.sqrt((y_data - x_data).norm(2))

                        y_data = (x_batch + pred * delta_t * time_step)
                        err = torch.sqrt((y_data - x_data).norm(2))

                        plt.text(0.05, 0.95, f'data: {run}   frame: {k}',
                                 transform=plt.gca().transAxes, fontsize=12,
                                 verticalalignment='top')
                        plt.text(0.05, 0.9, f'err: {err.item():0.4f}  err0: {err0.item():0.4f}',
                                 transform=plt.gca().transAxes, fontsize=12,
                                 verticalalignment='top')

                        x_data = to_numpy(x_data.squeeze())
                        y_data = to_numpy(y_data.squeeze())
                        lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)

                        residuals = y_data - linear_model(x_data, *lin_fit)
                        ss_res = np.sum(residuals ** 2)
                        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot)
                        plt.text(0.05, 0.85, f'R2: {r_squared:0.4f}  slope: {np.round(lin_fit[0], 4)}',
                                 transform=plt.gca().transAxes, fontsize=12,
                                 verticalalignment='top')
                        plt.tight_layout()
                        plt.savefig(f'{log_dir}/tmp_training/prediction/pred_{epoch}_{N}.tif')
                        plt.close()

                    if has_neural_field:
                        with torch.no_grad():
                            plot_training_signal_field(x, n_nodes, recurrent_loop, k, time_step,
                                                       x_list, run, model, field_type, model_f,
                                                       edges, y_list, ynorm, delta_t, n_frames, log_dir, epoch, N,
                                                       recurrent_parameters, modulation, device)
                        torch.save({'model_state_dict': model_f.state_dict(),
                                    'optimizer_state_dict': optimizer_f.state_dict()},
                                   os.path.join(log_dir, 'models',
                                                f'best_model_f_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))

                    if has_missing_activity:
                        with torch.no_grad():
                            plot_training_signal_missing_activity(n_frames, k, x_list, baseline_value,
                                                                  model_missing_activity, log_dir, epoch, N, device)
                        torch.save({'model_state_dict': model_missing_activity.state_dict(),
                                    'optimizer_state_dict': optimizer_missing_activity.state_dict()},
                                   os.path.join(log_dir, 'models',
                                                f'best_model_missing_activity_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))
                    torch.save(
                        {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                        os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))

            # check_and_clear_memory(device=device, iteration_number=N, every_n_iterations=Niter // 50, memory_percentage_threshold=0.6)

        print("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / n_neurons))
        logger.info("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / n_neurons))
        logger.info(f'recurrent_parameters: {recurrent_parameters[0]:.2f}')
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}.pt'))
        if has_neural_field:
            torch.save({'model_state_dict': model_f.state_dict(),
                        'optimizer_state_dict': optimizer_f.state_dict()},
                       os.path.join(log_dir, 'models', f'best_model_f_with_{n_runs - 1}_graphs_{epoch}.pt'))

        list_loss.append((total_loss-total_loss_regul) / n_neurons)

        list_loss_regul.append(total_loss_regul / n_neurons)

        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))

        fig = plt.figure(figsize=(15, 10))

        # Plot 1: Loss
        ax = fig.add_subplot(2, 3, 1)
        plt.plot(list_loss, color='k', linewidth=1)
        plt.xlim([0, n_epochs])
        plt.ylabel('loss', fontsize=24)
        plt.xlabel('epochs', fontsize=24)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Find the last saved file to get epoch and N
        embedding_files = glob.glob(f"./{log_dir}/tmp_training/embedding/*.tif")
        if embedding_files:
            last_file = max(embedding_files, key=os.path.getctime)  # or use os.path.getmtime for modification time
            filename = os.path.basename(last_file)
            last_epoch, last_N = filename.replace('.tif', '').split('_')

            # Load and display last saved figures
            from tifffile import imread

            # Plot 2: Last embedding
            ax = fig.add_subplot(2, 3, 2)
            img = imread(f"./{log_dir}/tmp_training/embedding/{last_epoch}_{last_N}.tif")
            plt.imshow(img)
            plt.axis('off')

            # Plot 3: Last weight comparison
            ax = fig.add_subplot(2, 3, 3)
            img = imread(f"./{log_dir}/tmp_training/matrix/comparison_{last_epoch}_{last_N}.tif")
            plt.imshow(img)
            plt.axis('off')

            # Plot 4: Last phi function
            ax = fig.add_subplot(2, 3, 4)
            img = imread(f"./{log_dir}/tmp_training/function/lin_phi/func_{last_epoch}_{last_N}.tif")
            plt.imshow(img)
            plt.axis('off')

            # Plot 5: Last edge function
            ax = fig.add_subplot(2, 3, 5)
            img = imread(f"./{log_dir}/tmp_training/function/lin_edge/func_{last_epoch}_{last_N}.tif")
            plt.imshow(img)
            plt.axis('off')


        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/epoch_{epoch}.tif")
        plt.close()

        if replace_with_cluster:

            if (epoch % sparsity_freq == sparsity_freq - 1) & (epoch < n_epochs - sparsity_freq):

                embedding = to_numpy(model.a.squeeze())
                model_MLP = model.lin_phi
                update_type = model.update_type

                func_list, proj_interaction = analyze_edge_function(rr=torch.linspace(config.plotting.xlim[0], config.plotting.xlim[1], 1000, device=device),
                                                                     vizualize=True, config=config,
                                                                     model_MLP=model_MLP, model=model,
                                                                     n_nodes=0,
                                                                     n_neurons=n_neurons, ynorm=ynorm,
                                                                     type_list=to_numpy(x[:, 1 + 2 * dimension]),
                                                                     cmap=cmap, update_type=update_type, device=device)

                labels, n_clusters, new_labels = sparsify_cluster(train_config.cluster_method, proj_interaction, embedding, train_config.cluster_distance_threshold, type_list, n_neuron_types, embedding_cluster)

                model_a_ = model.a.clone().detach()
                for n in range(n_clusters):
                    pos = np.argwhere(labels == n).squeeze().astype(int)
                    pos = np.array(pos)
                    if pos.size > 0:
                        median_center = model_a_[pos, :]
                        median_center = torch.median(median_center, dim=0).values
                        model_a_[pos, :] = median_center

                # Constrain embedding domain
                with torch.no_grad():
                    model.a.copy_(model_a_)
                print(f'regul_embedding: replaced')
                logger.info(f'regul_embedding: replaced')

                # Constrain function domain
                if train_config.sparsity == 'replace_embedding_function':

                    logger.info(f'replace_embedding_function')
                    y_func_list = func_list * 0

                    ax = fig.add_subplot(2, 5, 9)
                    for n in np.unique(new_labels):
                        pos = np.argwhere(new_labels == n)
                        pos = pos.squeeze()
                        if pos.size > 0:
                            target_func = torch.median(func_list[pos, :], dim=0).values.squeeze()
                            y_func_list[pos] = target_func
                        plt.plot(to_numpy(target_func) * to_numpy(ynorm), linewidth=2, alpha=1)
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()

                    lr_embedding = 1E-12
                    optimizer, n_total_params = set_trainable_parameters(model=model, lr_embedding=lr_embedding, lr=lr,
                                                                         lr_update=lr_update, lr_W=lr_W,
                                                                         lr_modulation=lr_modulation)
                    for sub_epochs in trange(20):
                        rr = torch.tensor(np.linspace(-5, 5, 1000)).to(device)
                        pred = []
                        optimizer.zero_grad()
                        for n in range(n_neurons):
                            embedding_ = model.a[n, :].clone().detach() * torch.ones((1000, model_config.embedding_dim),
                                                                                     device=device)
                            in_features = get_in_features_update(rr=rr[:, None], model=model, device=device)
                            pred.append(model.lin_phi(in_features.float()))
                        pred = torch.stack(pred)
                        loss = (pred[:, :, 0] - y_func_list.clone().detach()).norm(2)
                        logger.info(f'    loss: {np.round(loss.item() / n_neurons, 3)}')
                        loss.backward()
                        optimizer.step()
                if train_config.fix_cluster_embedding:
                    lr = 1E-12
                    lr_embedding = 1E-12
                    optimizer, n_total_params = set_trainable_parameters(model=model, lr_embedding=lr_embedding, lr=lr,
                                                                         lr_update=lr_update, lr_W=lr_W,
                                                                         lr_modulation=lr_modulation)
                    logger.info(
                        f'learning rates: lr_W {lr_W}, lr {lr}, lr_embedding {lr_embedding}, lr_modulation {lr_modulation}')
            else:
                lr = train_config.learning_rate_start
                lr_embedding = train_config.learning_rate_embedding_start
                optimizer, n_total_params = set_trainable_parameters(model=model, lr_embedding=lr_embedding, lr=lr,
                                                                     lr_update=lr_update, lr_W=lr_W,
                                                                     lr_modulation=lr_modulation)
                logger.info( f'learning rates: lr_W {lr_W}, lr {lr}, lr_embedding {lr_embedding}, lr_modulation {lr_modulation}')

