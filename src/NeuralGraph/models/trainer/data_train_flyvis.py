"""Extracted from graph_trainer.py"""
from ._imports import *

def data_train_flyvis(config, erase, best_model, device):
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
    small_init_batch_size = train_config.small_init_batch_size
    batch_ratio = train_config.batch_ratio
    training_NNR_start_epoch = train_config.training_NNR_start_epoch
    time_window = train_config.time_window
    training_selected_neurons = train_config.training_selected_neurons

    prediction_time_step = train_config.time_step
    prediction = model_config.prediction

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
    if coeff_loop.numel() < train_config.recurrent_loop:
        coeff_loop = torch.linspace(coeff_loop[0], coeff_loop[-1], train_config.recurrent_loop, device=device)

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
    if 'test' in field_type:
        test_neural_field = True
        print('train with test field NNR')
    else:
        test_neural_field = False

    log_dir, logger = create_log_dir(config, erase)

    x_list = []
    y_list = []
    for run in trange(0,n_runs, ncols=50):
        x = np.load(f'graphs_data/{dataset_name}/x_list_{run}.npy')
        y = np.load(f'graphs_data/{dataset_name}/y_list_{run}.npy')

        if training_selected_neurons:
            selected_neuron_ids = np.array(train_config.selected_neuron_ids).astype(int)
            x = x[:, selected_neuron_ids, :]
            y = y[:, selected_neuron_ids, :]

        x_list.append(x)
        y_list.append(y)

    print(f'dataset: {len(x_list)} run, {len(x_list[0])} frames')
    x = x_list[0][n_frames - 10]

    activity = torch.tensor(x_list[0][:, :, 3:4], device=device)
    activity = activity.squeeze()
    distrib = activity.flatten()
    valid_distrib = distrib[~torch.isnan(distrib)]
    if len(valid_distrib) > 0:
        xnorm = 1.5 * torch.std(valid_distrib)
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
    if time_window >0:
        model = Signal_Propagation_Temporal(aggr_type=model_config.aggr_type, config=config, device=device)
    elif 'MLP_ODE' in signal_model_name:
        model = Signal_Propagation_MLP_ODE(aggr_type=model_config.aggr_type, config=config, device=device)
    elif 'MLP' in signal_model_name:
        model = Signal_Propagation_MLP(aggr_type=model_config.aggr_type, config=config, device=device)
    else:
        model = Signal_Propagation_FlyVis(aggr_type=model_config.aggr_type, config=config, device=device)

    # Count parameters
    n_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'total parameters: {n_total_params:,}')
    logger.info(f'total parameters: {n_total_params:,}')

    start_epoch = 0
    list_loss = []
    if (best_model != None) & (best_model != '') & (best_model != '') & (best_model != 'None'):
        net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs_{best_model}.pt"
        print(f'load {net} ...')
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        start_epoch = int(best_model.split('_')[0])
        print(f'best_model: {best_model}  start_epoch: {start_epoch}')
        logger.info(f'best_model: {best_model}  start_epoch: {start_epoch}')
        # list_loss = torch.load(f"{log_dir}/loss.pt")
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
    lr_embedding = train_config.learning_rate_embedding_start
    lr_W = train_config.learning_rate_W_start
    lr_modulation = train_config.learning_rate_modulation_start
    learning_rate_NNR = train_config.learning_rate_NNR
    learning_rate_NNR_f = train_config.learning_rate_NNR_f

    print(
        f'learning rates: lr_W {lr_W}, lr {lr}, lr_update {lr_update}, lr_embedding {lr_embedding}, learning_rate_NNR {learning_rate_NNR}')
    logger.info(
        f'learning rates: lr_W {lr_W}, lr {lr}, lr_update {lr_update}, lr_embedding {lr_embedding}, learning_rate_NNR {learning_rate_NNR}')

    optimizer, n_total_params = set_trainable_parameters(model=model, lr_embedding=lr_embedding, lr=lr,
                                                         lr_update=lr_update, lr_W=lr_W, learning_rate_NNR=learning_rate_NNR, learning_rate_NNR_f = learning_rate_NNR_f)
    model.train()

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


    # res = analyze_type_neighbors(
    #     type_name="Mi1",
    #     edges_all=edges_all,
    #     type_list=type_list,          # (N,1) or (N,)
    #     n_hops=10,
    #     direction='in',
    #     verbose=True
    # )
    # # for hop in res["per_hop"]:
    # #     print('hop ', hop["hop"], ':' , hop["n_new"], hop["type_counts"])
    # hop_counts = [h["n_new"] for h in res["per_hop"]]          # number of neurons at each hop
    # total_excl_target = sum(hop_counts)                        # unique neurons discovered (excluding target)
    # total_incl_target = 1 + total_excl_target                  # including the target neuron
    # cumulative_by_hop = np.cumsum(hop_counts).tolist()
    # print("per-hop:", hop_counts)
    # print("cumulative:", cumulative_by_hop)
    # print("total excl target:", total_excl_target)


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
            Niter = int(n_frames * data_augmentation_loop // batch_size * 0.2)

        plot_frequency = int(Niter // 10)
        print(f'{Niter} iterations per epoch')
        logger.info(f'{Niter} iterations per epoch')
        print(f'plot every {plot_frequency} iterations')

        total_loss = 0
        total_loss_regul = 0
        k = 0

        # anneal loss_noise_level, decrease with epoch
        loss_noise_level = train_config.loss_noise_level * (0.95 ** epoch)
        # anneal weight_L1, increase with epoch
        coeff_edge_weight_L1= train_config.coeff_edge_weight_L1 * (1 - np.exp(-train_config.coeff_edge_weight_L1_rate**epoch))
        coeff_phi_weight_L1 = train_config.coeff_phi_weight_L1 * (1 - np.exp(-train_config.coeff_phi_weight_L1_rate*epoch))
        coeff_W_L1 = train_config.coeff_W_L1 * (1 - np.exp(-train_config.coeff_W_L1_rate * epoch))

        for N in trange(Niter,ncols=150):

            optimizer.zero_grad()

            dataset_batch = []
            ids_batch = []
            mask_batch = []
            k_batch = []
            visual_input_batch = []
            ids_index = 0
            mask_index = 0

            loss = 0
            run = np.random.randint(n_runs)

            if batch_ratio < 1:
                # Sample core neurons
                n_core = int(n_neurons * batch_ratio)
                core_ids = np.sort(np.random.choice(n_neurons, n_core, replace=False))

                # Determine which neurons we need based on training mode
                if recurrent_training and recurrent_loop > 0:
                    # For recurrent: need n-hop neighborhood (compute ONCE)


                    verbose = (N == 0) & (epoch==0)  # Print only on first iteration
                    # or: verbose = (N % 100 == 0)  # Print every 100 iterations
                    # or: verbose = False  # Never print

                    required_ids = get_n_hop_neighborhood_with_stats(
                        core_ids, edges_all, recurrent_loop, verbose=verbose
                    )

                else:
                    # For non-recurrent: just core neurons
                    required_ids = core_ids

                # Get edges ONCE for all timesteps
                mask = torch.isin(edges_all[1, :], torch.tensor(required_ids, device=device))
                edges = edges_all[:, mask]
                mask = torch.arange(edges_all.shape[1], device=device)[mask]

                # Store core_ids for loss computation
                ids = core_ids

            else:
                # use all neurons
                edges = edges_all.clone().detach()
                mask = torch.arange(edges_all.shape[1])
                ids = np.arange(n_neurons)
                core_ids = ids  # All neurons are "core" when batch_ratio = 1


            for batch in range(batch_size):

                if prediction == 'next_activity':
                    k = np.random.randint(n_frames - 4 - time_step)
                    k = k - k%time_step
                else:
                    k = np.random.randint(n_frames - 4 - time_step - time_window) + time_window

                x = torch.tensor(x_list[run][k], dtype=torch.float32, device=device)
                ids = np.arange(n_neurons)

                if time_window > 0:
                    x_temporal = x_list[run][k - time_window + 1: k + 1, :, 3:4].transpose(1, 0, 2).squeeze(-1)
                    x = torch.cat((x, torch.tensor(x_temporal.reshape(n_neurons, time_window), dtype=torch.float32, device=device)), dim=1)

                if has_visual_field:
                    visual_input = model.forward_visual(x,k)
                    x[:model.n_input_neurons, 4:5] = visual_input
                    x[model.n_input_neurons:, 4:5] = 0

                loss = torch.zeros(1, device=device)

                if not (torch.isnan(x).any()):
                    # regularisation sparsity on Wij
                    if coeff_W_L1>0:
                        loss = loss + model.W.norm(1) * coeff_W_L1
                    # regularisation sparsity on weights of model.lin_edge
                    if (coeff_edge_weight_L1+coeff_edge_weight_L2)>0:
                        for param in model.lin_edge.parameters():
                            loss = loss + param.norm(1) * coeff_edge_weight_L1 + param.norm(2) * coeff_edge_weight_L2
                    if (coeff_phi_weight_L1+coeff_phi_weight_L2)>0:
                        for param in model.lin_phi.parameters():
                            loss = loss + param.norm(2) * coeff_phi_weight_L1 + param.norm(2) * coeff_phi_weight_L2

                    # regularisation lin_edge
                    in_features, in_features_next = get_in_features_lin_edge(x, model, model_config, xnorm, n_neurons,device)
                    if coeff_edge_diff > 0:
                        if model_config.lin_edge_positive:
                            msg0 = model.lin_edge(in_features[ids].clone()) ** 2
                            msg1 = model.lin_edge(in_features_next[ids].clone()) ** 2
                        else:
                            msg0 = model.lin_edge(in_features[ids].clone())
                            msg1 = model.lin_edge(in_features_next[ids].clone())
                        loss = loss + torch.relu(msg0 - msg1).norm(2) * coeff_edge_diff      # lin_edge monotonically increasing  over voltage for all embedding values
                    if coeff_edge_norm > 0:
                        in_features[:,0] = 2 * xnorm
                        if model_config.lin_edge_positive:
                            msg = model.lin_edge(in_features[ids].clone()) ** 2
                        else:
                            msg = model.lin_edge(in_features[ids].clone())
                        loss = loss + (msg - 2 * xnorm).norm(2) * coeff_edge_norm

                    # # regularisation sign Wij
                    # if (coeff_W_sign > 0) and (N%4 == 0):
                    #     W_sign = torch.tanh(5 * model_W)
                    #     loss_contribs = []
                    #     for i in range(n_neurons):
                    #         indices = index_weight[int(i)]
                    #         if indices.numel() > 0:
                    #             values = W_sign[indices,i]
                    #             std = torch.std(values, unbiased=False)
                    #             loss_contribs.append(std)
                    #     if loss_contribs:
                    #         loss = loss + torch.stack(loss_contribs).norm(2) * coeff_W_sign

                    if prediction == 'next_activity':
                        y = torch.tensor(x_list[run][k+time_step,:,3:4], dtype=torch.float32, device=device).detach()
                    else:
                        y = torch.tensor(y_list[run][k], device=device) / ynorm

                    if test_neural_field:
                        y = torch.tensor(x_list[run][k, :n_input_neurons, 4:5], device=device)
                    if loss_noise_level>0:
                        y = y + torch.randn(y.shape, device=device) * loss_noise_level

                    if not (torch.isnan(y).any()):

                        dataset = pyg_Data(x=x, edge_index=edges)
                        dataset_batch.append(dataset)

                        if len(dataset_batch) == 1:
                            data_id = torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * run
                            x_batch = x[:, 3:5]
                            y_batch = y
                            ids_batch = ids
                            mask_batch = mask
                            k_batch = torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * k
                            if test_neural_field:
                                visual_input_batch = visual_input
                        else:
                            data_id = torch.cat((data_id, torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * run), dim=0)
                            x_batch = torch.cat((x_batch, x[:, 3:5]), dim=0)
                            y_batch = torch.cat((y_batch, y), dim=0)
                            ids_batch = np.concatenate((ids_batch, ids + ids_index), axis=0)
                            mask_batch = torch.cat((mask_batch, mask + mask_index), dim=0)
                            k_batch = torch.cat((k_batch, torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * k), dim=0)
                            if test_neural_field:
                                visual_input_batch = torch.cat((visual_input_batch, visual_input), dim=0)

                        ids_index += x.shape[0]
                        mask_index += edges_all.shape[1]

            if not (dataset_batch == []):

                total_loss_regul += loss.item()


                if test_neural_field:
                    loss = loss + (visual_input_batch - y_batch).norm(2)
                elif 'MLP_ODE' in signal_model_name:
                    batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
                    for batch in batch_loader:
                        pred = model(batch.x, data_id=data_id, mask=mask_batch, return_all=False)
                    loss = loss + (pred[ids_batch] - y_batch[ids_batch]).norm(2)
                elif 'MLP' in signal_model_name:
                    batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
                    for batch in batch_loader:
                        pred = model(batch.x, data_id=data_id, mask=mask_batch, return_all=False)
                    loss = loss + (pred[ids_batch] - y_batch[ids_batch]).norm(2)

                elif prediction == 'next_activity':

                    for n_loop in range(time_step):
                        batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
                        for batch in batch_loader:
                            pred = model(batch, data_id=data_id, mask=mask_batch, return_all=False)
                        for batch in range(batch_size):
                            dataset_batch[batch].x[:, 3:4] += delta_t * pred[batch*n_neurons:(batch+1)*n_neurons]

                    pred_x = []
                    for batch in range(batch_size):
                        pred_x_ = dataset_batch[batch].x[:, 3:4]
                        pred_x.append(pred_x_)
                    pred_x = torch.cat(pred_x, dim=0)

                    loss = loss + (pred_x[ids_batch] - y_batch[ids_batch]).norm(2)



                else:
                    batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
                    for batch in batch_loader:
                        if (coeff_update_msg_diff > 0) | (coeff_update_u_diff > 0) | (coeff_update_msg_sign>0):
                            pred, in_features,msg = model(batch, data_id=data_id, mask=mask_batch, return_all=True)
                            if coeff_update_msg_diff > 0 :      # Enforces that increasing the message input should increase the output (monotonic increasing)
                                pred_msg = model.lin_phi(in_features.clone().detach())
                                in_features_msg_next = in_features.clone().detach()
                                in_features_msg_next[:, model_config.embedding_dim+1] = in_features_msg_next[:, model_config.embedding_dim+1] * 1.05
                                pred_msg_next = model.lin_phi(in_features_msg_next.clone().detach())
                                loss = loss + torch.relu(pred_msg[ids_batch]-pred_msg_next[ids_batch]).norm(2) * coeff_update_msg_diff
                            if coeff_update_u_diff > 0:
                                pred_u = model.lin_phi(in_features.clone().detach())
                                in_features_u_next = in_features.clone().detach()
                                in_features_u_next[:, 0] = in_features_u_next[:, 0] * 1.05  # Perturb voltage (first column)
                                pred_u_next = model.lin_phi(in_features_u_next.clone().detach())
                                loss = loss + torch.relu(pred_u_next[ids_batch] - pred_u[ids_batch]).norm(2) * coeff_update_u_diff
                            if coeff_update_msg_sign > 0: # Penalizes when pred_msg not of same sign as msg
                                in_features_modified = in_features.clone().detach()
                                in_features_modified[:, 0] = 0
                                pred_msg = model.lin_phi(in_features_modified)
                                msg = in_features[:,model_config.embedding_dim+1].clone().detach()
                                loss = loss + (torch.tanh(pred_msg / 0.1) - torch.tanh(msg / 0.1)).norm(2) * coeff_update_msg_sign
                        else:
                            pred, in_features, msg = model(batch, data_id=data_id, mask=mask_batch, return_all=True)
                    loss = loss + (pred[ids_batch] - y_batch[ids_batch]).norm(2)

                if recurrent_training:

                    for n_loop in range(recurrent_loop):

                        for batch in range(batch_size):
                            k = k_batch[batch * n_neurons] + n_loop + 1

                            # Update only required neurons (not all)
                            if batch_ratio < 1:
                                update_indices = required_ids
                            else:
                                update_indices = np.arange(n_neurons)

                            # Apply state update
                            dataset_batch[batch].x[update_indices, 3:4] += (
                                delta_t * pred[batch*n_neurons:(batch+1)*n_neurons][update_indices] * ynorm
                            )

                            # Update external input
                            dataset_batch[batch].x[update_indices, 4:5] = torch.tensor(
                                x_list[run][k.item(), update_indices, 4:5], device=device
                            )

                        # Forward pass with SAME edges/mask
                        batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
                        for batch in batch_loader:
                            pred = model(batch, data_id=data_id, mask=mask_batch, return_all=False)

                        # Loss computation on core neurons only
                        y_batch_list = []
                        for batch in range(batch_size):
                            k = k_batch[batch * n_neurons] + n_loop + 1
                            y = torch.tensor(y_list[run][k.item(), core_ids], device=device) / ynorm
                            y_batch_list.append(y)

                        y_batch = torch.cat(y_batch_list, dim=0)

                        # Indices for loss (core neurons across all batches)
                        ids_batch = np.concatenate([
                            core_ids + batch * n_neurons for batch in range(batch_size)
                        ])

                        # Loss only on core neurons
                        loss = loss + (pred[ids_batch] - y_batch[ids_batch]).norm(2) / coeff_loop[n_loop]



                loss.backward()

                optimizer.step()


                total_loss += loss.item()

                if ((N % plot_frequency == 0) & (N > 0)):
                    if has_visual_field:
                        with torch.no_grad():
                            plt.style.use('dark_background')

                            # Static XY locations (take from first frame of this run)
                            X1 = to_numpy(x_list[run][0][:n_input_neurons, 1:3])

                            # group-based selection of 10 traces
                            groups = 217
                            group_size = n_input_neurons // groups  # expect 8
                            assert groups * group_size == n_input_neurons, "Unexpected packing of input neurons"
                            picked_groups = np.linspace(0, groups - 1, 10, dtype=int)
                            member_in_group = group_size // 2
                            trace_ids = (picked_groups * group_size + member_in_group).astype(int)

                            # MP4 writer setup
                            fps = 10
                            metadata = dict(title='Field Evolution', artist='Matplotlib', comment='NN Reconstruction over time')
                            writer = FFMpegWriter(fps=fps, metadata=metadata)
                            fig = plt.figure(figsize=(12, 4))

                            out_dir = f"./{log_dir}/tmp_training/field"
                            os.makedirs(out_dir, exist_ok=True)
                            out_path = f"{out_dir}/field_movie_{epoch}_{N}.mp4"
                            if os.path.exists(out_path):
                                os.remove(out_path)

                            # rolling buffers
                            win = 200
                            offset = 1.25
                            hist_t = deque(maxlen=win)
                            hist_gt = {i: deque(maxlen=win) for i in trace_ids}
                            hist_pred = {i: deque(maxlen=win) for i in trace_ids}

                            step_video = 2

                            with writer.saving(fig, out_path, dpi=200):
                                error_list = []

                                for k in trange(0, 800, step_video):
                                    # inputs and predictions
                                    x = torch.tensor(x_list[run][k], dtype=torch.float32, device=device)
                                    pred = to_numpy(model.forward_visual(x, k))
                                    pred_vec = np.asarray(pred).squeeze()  # (n_input_neurons,)

                                    if k==0:
                                        vmin = pred_vec.min()
                                        vmax = pred_vec.max()

                                    gt_field = x_list[0][k, :n_input_neurons, 4:5]
                                    gt_vec = to_numpy(gt_field).squeeze()  # (n_input_neurons,)

                                    # update rolling traces
                                    hist_t.append(k)
                                    for i in trace_ids:
                                        hist_gt[i].append(gt_vec[i])
                                        hist_pred[i].append(pred_vec[i])

                                    # draw three panels
                                    fig.clf()

                                    # Traces

                                    rmse_frame = float(np.sqrt(((pred_vec - gt_vec) ** 2).mean()))
                                    mae_frame  = float(np.mean(np.abs(pred_vec - gt_vec)))
                                    running_rmse = float(np.mean(error_list + [rmse_frame])) if len(error_list) else rmse_frame


                                    ax3 = fig.add_subplot(1, 3, 3)
                                    ax3.set_axis_off()
                                    ax3.set_facecolor("black")

                                    t = np.arange(len(hist_t))
                                    for j, i in enumerate(trace_ids):
                                        y0 = j * offset

                                        # correction = 1/(vmax+1E-8)   #
                                        correction = np.mean(np.array(hist_gt[i]) / (np.array(hist_pred[i])+1E-16))

                                        ax3.plot(t, np.array(hist_gt[i])   + y0, color='lime',  lw=1.6, alpha=0.95)
                                        ax3.plot(t, np.array(hist_pred[i])*correction + y0, color='white', lw=1.2, alpha=0.95)

                                    ax3.set_xlim(max(0, len(t) - win), len(t))
                                    ax3.set_ylim(-offset * 0.5, offset * (len(trace_ids) + 0.5))
                                    ax3.text(
                                        0.02, 0.98,
                                        f"frame: {k}   RMSE: {rmse_frame:.3f}   avg RMSE: {running_rmse:.3f}",
                                        transform=ax3.transAxes,
                                        va='top', ha='left',
                                        fontsize=10, color='white')

                                                                        # GT field

                                    ax1 = fig.add_subplot(1, 3, 1)
                                    ax1.scatter(X1[:, 0], X1[:, 1], s=256, c=gt_vec, cmap="viridis", marker='h', vmin=0, vmax=1)
                                    ax1.set_axis_off()
                                    ax1.set_title('ground truth', fontsize=12)

                                    # Predicted field
                                    ax2 = fig.add_subplot(1, 3, 2)
                                    ax2.scatter(X1[:, 0], X1[:, 1], s=256, c=pred_vec, cmap="viridis", marker='h', vmin=0, vmax=1/correction)
                                    ax2.set_axis_off()
                                    ax2.set_title('prediction', fontsize=12)

                                    plt.tight_layout()
                                    writer.grab_frame()

                                    # RMSE for this frame
                                    error_list.append(np.sqrt(((pred_vec * correction - gt_vec) ** 2).mean()))
                    if (not(test_neural_field)) & (not('MLP' in signal_model_name)):
                        plot_training_flyvis(x_list, model, config, epoch, N, log_dir, device, cmap, type_list, gt_weights, n_neurons=n_neurons, n_neuron_types=n_neuron_types)
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
        list_loss_regul.append(total_loss_regul / n_neurons)

        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))

        fig = plt.figure(figsize=(15, 10))

        # Plot 1: Loss
        ax = fig.add_subplot(2, 3, 1)
        plt.plot(list_loss, color='k', linewidth=1)
        plt.xlim([0, n_epochs])
        plt.ylabel('loss', fontsize=12)
        plt.xlabel('epochs', fontsize=12)

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
            plt.title('Embedding', fontsize=12)

            # Plot 3: Last weight comparison
            ax = fig.add_subplot(2, 3, 3)
            img = imread(f"./{log_dir}/tmp_training/matrix/comparison_{last_epoch}_{last_N}.tif")
            plt.imshow(img)
            plt.axis('off')
            plt.title('Weight Comparison', fontsize=12)

            # Plot 4: Last edge function
            ax = fig.add_subplot(2, 3, 4)
            img = imread(f"./{log_dir}/tmp_training/function/lin_edge/func_{last_epoch}_{last_N}.tif")
            plt.imshow(img)
            plt.axis('off')
            plt.title('Edge Function', fontsize=12)

            # Plot 5: Last phi function
            ax = fig.add_subplot(2, 3, 5)
            img = imread(f"./{log_dir}/tmp_training/function/lin_phi/func_{last_epoch}_{last_N}.tif")
            plt.imshow(img)
            plt.axis('off')
            plt.title('Phi Function', fontsize=12)

        if replace_with_cluster:

            if (epoch % sparsity_freq == sparsity_freq - 1) & (epoch < n_epochs - sparsity_freq):
                print('replace embedding with clusters ...')
                eps = train_config.cluster_distance_threshold
                results = clustering_evaluation(to_numpy(model.a), type_list, eps=eps)
                print(f"eps={eps}: {results['n_clusters_found']} clusters, "
                      f"accuracy={results['accuracy']:.3f}")

                labels = results['cluster_labels']

                for n in np.unique(labels):
                    # if n == -1:
                    #     continue
                    indices = np.where(labels == n)[0]
                    if len(indices) > 1:
                        with torch.no_grad():
                            model.a[indices, :] = torch.mean(model.a[indices, :], dim=0, keepdim=True)

                ax = fig.add_subplot(2, 3, 6)
                for n in range(n_neuron_types):
                    pos = torch.argwhere(type_list == n)
                    plt.scatter(to_numpy(model.a[pos, 0]), to_numpy(model.a[pos, 1]), s=5, color=cmap.color(n),
                                alpha=0.7, edgecolors='none')
                plt.xlabel('embedding 0', fontsize=18)
                plt.ylabel('embedding 1', fontsize=18)
                plt.xticks([])
                plt.yticks([])
                plt.text(0.5, 0.9, f"eps={eps}: {results['n_clusters_found']} clusters, accuracy={results['accuracy']:.3f}")

                if train_config.fix_cluster_embedding:
                    lr_embedding = 1.0E-10
                    # the embedding is fixed for 1 epoch

            else:
                lr = train_config.learning_rate_start
                lr_embedding = train_config.learning_rate_embedding_start
                lr_W = train_config.learning_rate_W_start
                learning_rate_NNR = train_config.learning_rate_NNR

            logger.info(f'learning rates: lr_W {lr_W}, lr {lr}, lr_update {lr_update}, lr_embedding {lr_embedding}, learning_rate_NNR {learning_rate_NNR}')
            optimizer, n_total_params = set_trainable_parameters(model=model, lr_embedding=lr_embedding, lr=lr, lr_update=lr_update, lr_W=lr_W,
                                                                 learning_rate_NNR=learning_rate_NNR)

        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/epoch_{epoch}.tif")
        plt.close()

