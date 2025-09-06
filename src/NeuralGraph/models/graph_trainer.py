import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import random

from GNN_Main import *
from NeuralGraph.models.utils import *
from NeuralGraph.utils import *
from NeuralGraph.models.Siren_Network import *
from NeuralGraph.models.Signal_Propagation_FlyVis import *
from NeuralGraph.sparsify import EmbeddingCluster, sparsify_cluster, clustering_evaluation
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import curve_fit

from NeuralGraph.fitting_models import linear_model
from torch_geometric.utils import dense_to_sparse
import torch.optim as optim
import seaborn as sns
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from NeuralGraph.denoise_data import *
from scipy.spatial import KDTree
from sklearn import neighbors, metrics
from scipy.ndimage import median_filter
from tifffile import imwrite, imread
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree
from NeuralGraph.generators.utils import *
from scipy.special import logsumexp

def data_train(config=None, erase=False, best_model=None, device=None):
    # plt.rcParams['text.usetex'] = True
    # rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    # matplotlib.rcParams['savefig.pad_inches'] = 0

    seed = config.training.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # torch.autograd.set_detect_anomaly(True)

    dataset_name = config.dataset
    print('')
    print(f"\033[94mdataset_name: {dataset_name}\033[0m")

    if 'fly' in config.dataset:
        data_train_flyvis(config, erase, best_model, device)
    else:
        data_train_synaptic2(config, erase, best_model, device)



def data_train_synaptic2(config, erase, best_model, device):
    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    dimension = simulation_config.dimension
    n_epochs = train_config.n_epochs
    n_neuron_types = simulation_config.n_neuron_types

    dataset_name = config.dataset
    n_frames = simulation_config.n_frames
    data_augmentation_loop = train_config.data_augmentation_loop
    recursive_loop = train_config.recursive_loop
    target_batch_size = train_config.batch_size
    delta_t = simulation_config.delta_t
    if train_config.small_init_batch_size:
        get_batch_size = increasing_batch_size(target_batch_size)
    else:
        get_batch_size = constant_batch_size(target_batch_size)
    batch_ratio = train_config.batch_ratio
    batch_size = get_batch_size(0)
    embedding_cluster = EmbeddingCluster(config)
    cmap = CustomColorMap(config=config)
    n_runs = train_config.n_runs
    field_type = model_config.field_type
    coeff_lin_modulation = train_config.coeff_lin_modulation
    coeff_model_b = train_config.coeff_model_b
    coeff_W_sign = train_config.coeff_W_sign
    coeff_update_msg_diff = train_config.coeff_update_msg_diff
    coeff_update_u_diff = train_config.coeff_update_u_diff
    coeff_edge_norm = train_config.coeff_edge_norm
    coeff_update_msg_sign = train_config.coeff_update_msg_sign

    time_step = train_config.time_step
    has_missing_activity = train_config.has_missing_activity
    multi_connectivity = config.training.multi_connectivity
    baseline_value = simulation_config.baseline_value

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

    print(f'has_neural_field: {has_neural_field}, has_missing_activity: {has_missing_activity}')

    replace_with_cluster = 'replace' in train_config.sparsity
    sparsity_freq = train_config.sparsity_freq
    recursive_parameters = train_config.recursive_parameters.copy()

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

    activity = torch.tensor(x_list[0][:, :, 6:7], device=device)
    activity = activity.squeeze()
    distrib = activity.flatten()

    # pred_kinograph = y_list[0]
    # fig = plt.figure(figsize=(10, 10))
    # plt.imshow(np.transpose(pred_kinograph), aspect='auto',vmin =-3, vmax=3, cmap='viridis')
    # plt.tight_layout()
    # plt.savefig(f"./{log_dir}/tmp_training/pred_kinograph.tif", dpi=170)
    # plt.close()

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
    type_list = torch.tensor(x[:, 1 + 2 * dimension:2 + 2 * dimension], device=device)
    vnorm = torch.tensor(1.0, device=device)
    ynorm = torch.tensor(1.0, device=device)
    torch.save(vnorm, os.path.join(log_dir, 'vnorm.pt'))
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))
    time.sleep(0.5)
    print(f'vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')
    logger.info(f'vnorm ynorm: {to_numpy(vnorm)} {to_numpy(ynorm)}')

    if model_config.embedding_init != '':
        print('compute init embedding ...')
        for j in trange(n_frames):
            if j == 0:
                time_series = np.array(x_list[0][j][:, 6:7])
            else:
                time_series = np.concatenate((time_series, x_list[0][j][:, 6:7]), axis=1)
        time_series = np.array(time_series)

        match model_config.embedding_init:
            case 'umap':
                trans = umap.UMAP(n_neighbors=50, n_components=2, transform_queue_size=0,
                                  random_state=config.training.seed).fit(time_series)
                projections = trans.transform(time_series)
            case 'pca':
                pca = PCA(n_components=2)
                projections = pca.fit_transform(time_series)
            case 'svd':
                svd = TruncatedSVD(n_components=2)
                projections = svd.fit_transform(time_series)
            case 'tsne':
                tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
                projections = tsne.fit_transform(time_series)

        fig = plt.figure(figsize=(8, 8))
        for n in range(n_neuron_types):
            pos = torch.argwhere(type_list == n).squeeze()
            plt.scatter(projections[to_numpy(pos), 0], projections[to_numpy(pos), 1], s=10, color=cmap.color(n))
        plt.xlabel('Embedding 0', fontsize=12)
        plt.ylabel('Embedding 1', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/Embedding_init.tif")
        plt.close()
    else:
        projections = None

    print('create models ...')
    model, bc_pos, bc_dpos = choose_training_model(model_config=config, device=device, projections=projections)
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

    print(
        f'learning rates: lr_W {lr_W}, lr {lr}, lr_update {lr_update}, lr_embedding {lr_embedding}, lr_modulation {lr_modulation}')
    logger.info(
        f'learning rates: lr_W {lr_W}, lr {lr}, lr_update {lr_update}, lr_embedding {lr_embedding}, lr_modulation {lr_modulation}')

    optimizer, n_total_params = set_trainable_parameters(model=model, lr_embedding=lr_embedding, lr=lr,
                                                         lr_update=lr_update, lr_W=lr_W, lr_modulation=lr_modulation)

    net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs.pt"
    print(f'network: {net}')
    print(f'initial batch_size: {batch_size}')
    logger.info(f'network: {net}')
    logger.info(f'N epochs: {n_epochs}')
    logger.info(f'initial batch_size: {batch_size}')


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

        # pos = torch.argwhere(edges[1,:]==0)
        # neurons_sender_to_0 = edges[0,pos]
        # model.mask = (connectivity > 0) * 1.0
        # adj_t = model.mask.float() * 1
        # adj_t = adj_t.t() #[ post, pre] -> [pre, post]
        # edges = adj_t.nonzero().T.contiguous()   #[(pre, post), n_elements]
        # edges_all = edges.clone().detach()

    else:
        edges = torch.load(f'./graphs_data/{dataset_name}/edge_index.pt', map_location=device)
        edges_all = edges.clone().detach()

    if coeff_W_sign > 0:
        index_weight = []
        for i in range(n_neurons):
            index_weight.append(torch.argwhere(model.mask[:, i] > 0).squeeze())

    print(f'{edges.shape[1]} edges')

    if 'PDE_N3' in model_config.signal_model_name:          # PDE_N3 is special, embedding changes over time
        ind_a = torch.tensor(np.arange(1, n_neurons * 100), device=device)
        pos = torch.argwhere(ind_a % 100 != 99).squeeze()
        ind_a = ind_a[pos]

    coeff_W_L1 = train_config.coeff_W_L1
    coeff_edge_diff = train_config.coeff_edge_diff
    coeff_update_diff = train_config.coeff_update_diff
    logger.info(f'coeff_W_L1: {coeff_W_L1} coeff_edge_diff: {coeff_edge_diff} coeff_update_diff: {coeff_update_diff}')
    print(f'coeff_W_L1: {coeff_W_L1} coeff_edge_diff: {coeff_edge_diff} coeff_update_diff: {coeff_update_diff}')

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
            Niter = int(n_frames * data_augmentation_loop // batch_size * 0.2 // max(recursive_loop, 1))

        plot_frequency = int(Niter // 200)
        print(f'{Niter} iterations per epoch')
        logger.info(f'{Niter} iterations per epoch')
        print(f'plot every {plot_frequency} iterations')

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
            run = np.random.randint(n_runs)

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
                    in_features = get_in_features_update(rr=None, model=model, device=device)
                    func_phi = model.lin_phi(in_features[ids].float())
                    loss = loss + func_phi.norm(2)
                    # regularisation sparsity on Wij
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
                                                                model.a[:n_neurons], msg_1,
                                                                torch.ones((n_neurons, 1), device=device)), dim=1)
                            in_feature_update_prev = in_feature_update_prev[ids]
                            loss = loss + (model.lin_phi(in_feature_update_prev) + model.lin_phi(
                                in_feature_update_next) - 2 * model.lin_phi(in_feature_update)).norm(
                                2) * coeff_update_diff

                    if batch_ratio < 1:
                        ids_ = np.random.permutation(ids.shape[0])[:int(ids.shape[0] * batch_ratio)]
                        ids = np.sort(ids)
                        edges = edges_all.clone().detach()
                        mask = torch.isin(edges[1, :], torch.tensor(ids, device=device))
                        edges = edges[:, mask]

                    if recursive_loop > 1:
                        y = torch.tensor(y_list[run][k + recursive_loop], device=device) / ynorm
                    elif time_step == 1:
                        y = torch.tensor(y_list[run][k], device=device) / ynorm
                    elif time_step > 1:
                        y = torch.tensor(x_list[run][k + time_step, :, 6:7], device=device).clone().detach()

                    if not (torch.isnan(y).any()):

                        dataset = data.Data(x=x, edge_index=edges)
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
                            plot_training_signal_field(x, n_nodes, recursive_loop, k, time_step,
                                                       x_list, run, model, field_type, model_f,
                                                       edges, y_list, ynorm, delta_t, n_frames, log_dir, epoch, N,
                                                       recursive_parameters, modulation, device)
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
        logger.info(f'recursive_parameters: {recursive_parameters[0]:.2f}')
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

        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/epoch_{epoch}.tif")
        plt.close()

        if replace_with_cluster:

            if (epoch % sparsity_freq == sparsity_freq - 1) & (epoch < n_epochs - sparsity_freq):

                embedding = to_numpy(model.a.squeeze())
                model_MLP = model.lin_phi
                update_type = model.update_type

                func_list, proj_interaction_ = analyze_edge_function(rr=torch.linspace(config.plotting.xlim[0], config.plotting.xlim[1], 1000, device=device),
                                                                     vizualize=True, config=config,
                                                                     model_MLP=model_MLP, model=model,
                                                                     n_nodes=0,
                                                                     n_neurons=n_neurons, ynorm=ynorm,
                                                                     type_list=to_numpy(x[:, 1 + 2 * dimension]),
                                                                     cmap=cmap, update_type=update_type, device=device)



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

            if (epoch == 20) & (train_config.coeff_anneal_L1 > 0):
                coeff_W_L1 = train_config.coeff_anneal_L1
                logger.info(f'coeff_W_L1: {coeff_W_L1}')


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
    recursive_training = train_config.recursive_training
    recursive_loop = train_config.recursive_loop
    batch_size = train_config.batch_size
    batch_ratio = train_config.batch_ratio
    training_NNR_start_epoch = train_config.training_NNR_start_epoch

    field_type = model_config.field_type
    time_step = train_config.time_step

    coeff_W_sign = train_config.coeff_W_sign
    coeff_update_msg_diff = train_config.coeff_update_msg_diff
    coeff_update_u_diff = train_config.coeff_update_u_diff
    coeff_edge_norm = train_config.coeff_edge_norm
    coeff_update_msg_sign = train_config.coeff_update_msg_sign
    coeff_edge_weight_L2 = train_config.coeff_edge_weight_L2
    coeff_phi_weight_L2 = train_config.coeff_phi_weight_L2

    pre_trained_W = train_config.pre_trained_W

    replace_with_cluster = 'replace' in train_config.sparsity
    sparsity_freq = train_config.sparsity_freq

    n_edges = simulation_config.n_edges
    n_extra_null_edges = simulation_config.n_extra_null_edges

    if config.training.seed != 42:
        torch.random.fork_rng(devices=device)
        torch.random.manual_seed(config.training.seed)

    cmap = CustomColorMap(config=config)

    if 'visual' in field_type:
        has_visual_field = True
        print('train with visual field NNR')
    else:
        has_visual_field = False

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
    model = Signal_Propagation_FlyVis(aggr_type=model_config.aggr_type, config=config, device=device)

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

    if pre_trained_W != '':
        print(f'load pre-trained W: {pre_trained_W}')
        logger.info(f'load pre-trained W: {pre_trained_W}')
        W_ = np.load(pre_trained_W)
        with torch.no_grad():
            model.W = nn.Parameter(torch.tensor(W_[:,None], dtype=torch.float32, device=device), requires_grad=False)

    lr = train_config.learning_rate_start
    if train_config.learning_rate_update_start == 0:
        lr_update = train_config.learning_rate_start
    else:
        lr_update = train_config.learning_rate_update_start
    lr_embedding = train_config.learning_rate_embedding_start
    lr_W = train_config.learning_rate_W_start
    lr_modulation = train_config.learning_rate_modulation_start
    learning_rate_NNR = train_config.learning_rate_NNR

    print(
        f'learning rates: lr_W {lr_W}, lr {lr}, lr_update {lr_update}, lr_embedding {lr_embedding}, learning_rate_NNR {learning_rate_NNR}')
    logger.info(
        f'learning rates: lr_W {lr_W}, lr {lr}, lr_update {lr_update}, lr_embedding {lr_embedding}, learning_rate_NNR {learning_rate_NNR}')

    optimizer, n_total_params = set_trainable_parameters(model=model, lr_embedding=lr_embedding, lr=lr,
                                                         lr_update=lr_update, lr_W=lr_W, learning_rate_NNR=learning_rate_NNR)
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
            Niter = int(n_frames * data_augmentation_loop // batch_size * 0.2 // max(recursive_loop, 1))

        if (pre_trained_W != '') & (epoch == 1):
            model.W.requires_grad = True
            optimizer, n_total_params = set_trainable_parameters(model=model, lr_embedding=lr_embedding, lr=lr,
                                                                 lr_update=lr_update, lr_W=lr_W,
                                                                 learning_rate_NNR=learning_rate_NNR)

        plot_frequency = int(Niter // 20)
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

        for N in trange(Niter):

            optimizer.zero_grad()

            dataset_batch = []
            ids_batch = []
            mask_batch = []
            k_batch = []
            ids_index = 0
            mask_index = 0

            loss = 0
            run = np.random.randint(n_runs)

            for batch in range(batch_size):

                k = np.random.randint(n_frames - 4 - time_step)
                x = torch.tensor(x_list[run][k], dtype=torch.float32, device=device)
                ids = np.arange(n_neurons)

                if (has_visual_field) & (epoch >= training_NNR_start_epoch):
                    if model_config.input_size_nnr == 1:
                        x[:n_input_neurons, 4] = model.visual_NNR(torch.tensor([k / n_frames], dtype=torch.float32, device=device)) ** 2

                    # if model_config.input_size_nnr == 1:
                    #     in_features = torch.tensor([k / n_frames], dtype=torch.float32, device=device) ** 2
                    #     x[:n_input_neurons, 4] = model.visual_NNR(in_features) ** 2
                    # else:
                    #     t = torch.tensor([k / n_frames], dtype=torch.float32, device=device)
                    #     in_features = torch.cat((x[:n_input_neurons, 1:3], t.unsqueeze(0).repeat(n_input_neurons, 1)), dim=1)
                    #     x[:n_input_neurons, 4:5] = model.visual_NNR(in_features) ** 2

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

                    if batch_ratio < 1:
                        ids_ = np.random.permutation(ids.shape[0])[:int(ids.shape[0] * batch_ratio)]
                        ids = np.sort(ids_)
                        edges = edges_all.clone().detach()
                        mask = torch.isin(edges[1, :], torch.tensor(ids, device=device))
                        edges = edges[:, mask]
                        mask = torch.arange(edges_all.shape[1],device=device)[mask]
                    else:
                        edges = edges_all.clone().detach()
                        mask = torch.arange(edges_all.shape[1])

                    y = torch.tensor(y_list[run][k], device=device) / ynorm
                    if loss_noise_level>0:
                        y = y + torch.randn(y.shape, device=device) * loss_noise_level

                    if not (torch.isnan(y).any()):

                        dataset = data.Data(x=x, edge_index=edges)
                        dataset_batch.append(dataset)

                        if len(dataset_batch) == 1:
                            data_id = torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * run
                            x_batch = x[:, 3:4]
                            y_batch = y
                            ids_batch = ids
                            mask_batch = mask
                            k_batch = torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * k
                        else:
                            data_id = torch.cat((data_id, torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * run), dim=0)
                            x_batch = torch.cat((x_batch, x[:, 4:5]), dim=0)
                            y_batch = torch.cat((y_batch, y), dim=0)
                            ids_batch = np.concatenate((ids_batch, ids + ids_index), axis=0)
                            mask_batch = torch.cat((mask_batch, mask + mask_index), dim=0)
                            k_batch = torch.cat((k_batch, torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * k), dim=0)

                        ids_index += x.shape[0]
                        mask_index += edges_all.shape[1]

            if not (dataset_batch == []):

                total_loss_regul += loss.item()

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

                if recursive_training:
                    for n_loop in range(recursive_loop):
                        for batch in range(batch_size):
                            k = k_batch[batch * x.shape[0]] + n_loop + 1
                            dataset_batch[batch].x[:, 3:4] = dataset_batch[batch].x[:, 3:4] + delta_t * pred[0+batch*x.shape[0]:x.shape[0]+batch*x.shape[0]] * ynorm
                            dataset_batch[batch].x[:, 4:5] = torch.tensor(x_list[run][k.item(),:,4:5], device=device)
                        batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
                        for batch in batch_loader:
                            pred = model(batch, data_id=data_id, mask=mask_batch, return_all=False)
                        for batch in range(batch_size):
                            k = k_batch[batch*x.shape[0]] + n_loop + 1
                            y = torch.tensor(y_list[run][k.item()], device=device) / ynorm
                            if batch == 0:
                                    y_batch = y
                            else:
                                    y_batch = torch.cat((y_batch, y), dim=0)
                        loss = loss + (pred[ids_batch] - y_batch[ids_batch]).norm(2) / coeff_loop[batch]

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                if ((N % plot_frequency == 0) | (N == 0)):

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


def data_test(config=None, config_file=None, visualize=False, style='color frame', verbose=True, best_model=20, step=15,
              ratio=1, run=1, test_mode='', sample_embedding=False, particle_of_interest=1, device=[]):
    dataset_name = config.dataset
    simulation_config = config.simulation
    model_config = config.graph_model
    training_config = config.training

    has_adjacency_matrix = (simulation_config.connectivity_file != '')
    has_mesh = (config.graph_model.mesh_model_name != '')
    only_mesh = (config.graph_model.particle_model_name == '') & has_mesh
    n_ghosts = config.training.n_ghosts
    has_ghost = config.training.n_ghosts > 0
    has_bounding_box = 'PDE_F' in model_config.particle_model_name
    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_neuron_types = simulation_config.n_neuron_types
    n_neurons = simulation_config.n_neurons
    n_nodes = simulation_config.n_nodes
    n_runs = training_config.n_runs
    n_frames = simulation_config.n_frames
    delta_t = simulation_config.delta_t
    time_window = training_config.time_window
    time_step = training_config.time_step
    sub_sampling = simulation_config.sub_sampling
    bounce_coeff = simulation_config.bounce_coeff
    cmap = CustomColorMap(config=config)
    dimension = simulation_config.dimension
    has_particle_field = ('PDE_ParticleField' in config.graph_model.particle_model_name)
    has_mesh_field = (model_config.field_type != '') & ('RD_Mesh' in model_config.mesh_model_name)
    has_field = (model_config.field_type != '') & (has_mesh_field == False) & (has_particle_field == False)
    has_missing_activity = training_config.has_missing_activity
    has_excitation = ('excitation' in model_config.update_type)
    baseline_value = simulation_config.baseline_value
    omega = model_config.omega

    do_tracking = training_config.do_tracking
    has_state = (config.simulation.state_type != 'discrete')
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

    print(f'load data run {run} ...')
    if only_mesh:
        vnorm = torch.tensor(1.0, device=device)
        ynorm = torch.tensor(1.0, device=device)
        hnorm = torch.load(f'{log_dir}/hnorm.pt', map_location=device).to(device)
        x_mesh_list = []
        y_mesh_list = []
        time.sleep(0.5)
        x_mesh = torch.load(f'graphs_data/{dataset_name}/x_mesh_list_{run}.pt', map_location=device)
        x_mesh_list.append(x_mesh)
        h = torch.load(f'graphs_data/{dataset_name}/y_mesh_list_{run}.pt', map_location=device)
        y_mesh_list.append(h)
        x_list = x_mesh_list
        y_list = y_mesh_list
    elif has_particle_field:
        x_list = []
        y_list = []
        x_mesh_list = []
        x_mesh = torch.load(f'graphs_data/{dataset_name}/x_mesh_list_{run}.pt', map_location=device)
        x_mesh_list.append(x_mesh)
        hnorm = torch.load(f'{log_dir}/hnorm.pt', map_location=device).to(device)
        x_list.append(torch.load(f'graphs_data/{dataset_name}/x_list_{run}.pt', map_location=device))
        y_list.append(torch.load(f'graphs_data/{dataset_name}/y_list_{run}.pt', map_location=device))
        ynorm = torch.load(f'{log_dir}/ynorm.pt', map_location=device, weights_only=True).to(device)
        vnorm = torch.load(f'{log_dir}/vnorm.pt', map_location=device, weights_only=True).to(device)
        x = x_list[0][0].clone().detach()
        n_neurons = x.shape[0]
        config.simulation.n_neurons = n_neurons
        index_particles = get_index_particles(x, n_neuron_types, dimension)
        x_mesh = x_mesh_list[0][0].clone().detach()
    else:
        x_list = []
        y_list = []
        if (model_config.particle_model_name == 'PDE_R'):
            x = torch.load(f'graphs_data/{dataset_name}/x_list_{run}.pt', map_location=device)
            x_list.append(x)
        else:
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
                if ('PDE_MLPs' not in model_config.particle_model_name) & (
                        'PDE_F' not in model_config.particle_model_name) & (
                        'PDE_M' not in model_config.particle_model_name):
                    n_neurons = int(x.shape[0] / ratio)
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

    if do_tracking | has_state:
        for k in range(len(x_list[0])):
            type = x_list[0][k][:, 2 * dimension + 1]
            if k == 0:
                type_list = type
            else:
                type_list = torch.concatenate((type_list, type))
        n_neurons_max = len(type_list) + 1
        config.simulation.n_neurons_max = n_neurons_max
    if ratio > 1:
        new_nparticles = int(n_neurons * ratio)
        model.a = nn.Parameter(
            torch.tensor(np.ones((n_runs, int(new_nparticles), 2)), device=device, dtype=torch.float32,
                         requires_grad=False))
        n_neurons = new_nparticles
        index_particles = get_index_particles(x, n_neuron_types, dimension)
    if sample_embedding:
        model_a_ = nn.Parameter(
            torch.tensor(np.ones((int(n_neurons), model.embedding_dim)), device=device, requires_grad=False,
                         dtype=torch.float32))
        for n in range(n_neurons):
            t = to_numpy(x[n, 5]).astype(int)
            index = first_cell_id_particles[t][np.random.randint(n_sub_population)]
            with torch.no_grad():
                model_a_[n] = first_embedding[index].clone().detach()
        model.a = nn.Parameter(
            torch.tensor(np.ones((model.n_dataset, int(n_neurons), model.embedding_dim)), device=device,
                         requires_grad=False, dtype=torch.float32))
        with torch.no_grad():
            for n in range(model.a.shape[0]):
                model.a[n] = model_a_
    if has_ghost & ('PDE_N' not in model_config.signal_model_name):
        model_ghost = Ghost_Particles(config, n_neurons, vnorm, device)
        net = f"{log_dir}/models/best_ghost_particles_with_{n_runs - 1}_graphs_20.pt"
        state_dict = torch.load(net, map_location=device)
        model_ghost.load_state_dict(state_dict['model_state_dict'])
        model_ghost.eval()
        x_removed_list = torch.load(f'graphs_data/{dataset_name}/x_removed_list_0.pt', map_location=device)
        mask_ghost = np.concatenate((np.ones(n_neurons), np.zeros(config.training.n_ghosts)))
        mask_ghost = np.argwhere(mask_ghost == 1)
        mask_ghost = mask_ghost[:, 0].astype(int)
    if has_mesh:
        n_nodes_per_axis = int(np.sqrt(n_nodes))

        hnorm = torch.load(f'{log_dir}/hnorm.pt', map_location=device).to(device)

        mesh_data = torch.load(f'graphs_data/{dataset_name}/mesh_data_{run}.pt', map_location=device)
        mask_mesh = mesh_data['mask']
        edge_index_mesh = mesh_data['edge_index']
        edge_weight_mesh = mesh_data['edge_weight']

        ids = to_numpy(torch.argwhere(mesh_data['mask'] == True)[:, 0].squeeze())
        mask = torch.isin(edge_index_mesh[1, :], torch.tensor(ids, device=device))
        edge_index_mesh = edge_index_mesh[:, mask]
        edge_weight_mesh = edge_weight_mesh[mask]

        node_gt_list = []
        node_pred_list = []

        if 'WaveMeshSmooth' in model_config.mesh_model_name:
            with torch.no_grad():
                distance = torch.sum((mesh_data['mesh_pos'][:, None, :] - mesh_data['mesh_pos'][None, :, :]) ** 2,
                                     dim=2)
                adj_t = ((distance < max_radius ** 2) & (distance >= min_radius ** 2)).float() * 1
                edge_index_mesh = adj_t.nonzero().t().contiguous()

        # plt.scatter(x_, y_, s=2, c=to_numpy(mask_mesh))
    if has_adjacency_matrix:
        if model_config.signal_model_name == 'PDE_N':
            mat = scipy.io.loadmat(simulation_config.connectivity_file)
            adjacency = torch.tensor(mat['A'], device=device)
            adj_t = adjacency > 0
            edge_index = adj_t.nonzero().t().contiguous()
            edge_attr_adjacency = adjacency[adj_t]
        else:
            adjacency = torch.load(f'./graphs_data/{dataset_name}/adjacency.pt', map_location=device)
            adjacency_ = adjacency.t().clone().detach()
            adj_t = torch.abs(adjacency_) > 0
            edge_index = adj_t.nonzero().t().contiguous()
    if 'PDE_N' in model_config.signal_model_name:
        has_adjacency_matrix = True
        adjacency = torch.load(f'./graphs_data/{dataset_name}/adjacency.pt', map_location=device)
        if training_config.with_connectivity_mask:
            model_mask = (adjacency > 0) * 1.0
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

    if verbose:
        print(f'test data ... {model_config.particle_model_name} {model_config.mesh_model_name}')
        print('log_dir: {}'.format(log_dir))
        print(f'network: {net}')
        print(table)
        print(f"total trainable Params: {total_params}")

    if 'test_simulation' in 'test_mode':
        if has_mesh:
            mesh_model, bc_pos, bc_dpos = choose_mesh_model(config, device)
        else:
            model, bc_pos, bc_dpos = choose_model(config, device=device)
    elif has_mesh:
        mesh_model, bc_pos, bc_dpos = choose_training_model(config, device)
        state_dict = torch.load(net, map_location=device)
        mesh_model.load_state_dict(state_dict['model_state_dict'])
        mesh_model.eval()
        if has_mesh_field:
            model_f = Siren_Network(image_width=n_nodes_per_axis, in_features=model_config.input_size_nnr,
                                    out_features=model_config.output_size_nnr,
                                    hidden_features=model_config.hidden_dim_nnr,
                                    hidden_layers=model_config.n_layers_nnr, outermost_linear=True, device=device,
                                    first_omega_0=omega, hidden_omega_0=omega)
            net = f'{log_dir}/models/best_model_f_with_1_graphs_{best_model}.pt'
            state_dict = torch.load(net, map_location=device)
            model_f.load_state_dict(state_dict['model_state_dict'])
            model_f.to(device=device)
            model_f.eval()
    else:
        model, bc_pos, bc_dpos = choose_training_model(config, device)
        model.ynorm = ynorm
        model.vnorm = vnorm
        model.particle_of_interest = particle_of_interest
        if training_config.with_connectivity_mask:
            model.mask = (adjacency > 0) * 1.0
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
        mesh_model = None
        if 'PDE_K' in model_config.particle_model_name:
            model.connection_matrix = torch.load(f'graphs_data/{dataset_name}/connection_matrix_list.pt',
                                                 map_location=device)
            timeit = np.load(f'graphs_data/{dataset_name}/times_train_springs_example.npy',
                             allow_pickle=True)
            timeit = timeit[run][0]
            time_id = 0
        if 'PDE_N' in model_config.signal_model_name:
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

        if has_particle_field:
            model_f_p = model
            image_width = int(np.sqrt(n_nodes))
            if 'siren_with_time' in model_config.field_type:
                model_f = Siren_Network(image_width=n_nodes_per_axis, in_features=model_config.input_size_nnr,
                                        out_features=model_config.output_size_nnr,
                                        hidden_features=model_config.hidden_dim_nnr,
                                        hidden_layers=model_config.n_layers_nnr, outermost_linear=True, device=device,
                                        first_omega_0=model_config.omega, hidden_omega_0=model_config.omega)
                net = f'{log_dir}/models/best_model_f_with_1_graphs_{best_model}.pt'
                state_dict = torch.load(net, map_location=device)
                model_f.load_state_dict(state_dict['model_state_dict'])
                model_f.to(device=device)
                model_f.eval()
                table = PrettyTable(["Modules", "Parameters"])
                total_params = 0
                for name, parameter in model_f.named_parameters():
                    if not parameter.requires_grad:
                        continue
                    param = parameter.numel()
                    table.add_row([name, param])
                    total_params += param
                if verbose:
                    print(table)
                    print(f"Total Trainable Params: {total_params}")
            else:
                t = model.field[run].reshape(image_width, image_width)
                t = torch.rot90(t)
                t = torch.flipud(t)
                t = t.reshape(image_width * image_width, 1)
                with torch.no_grad():
                    model.a = a_.clone().detach()
                    model.field[run] = t.clone().detach()
        elif has_field:
            image_width = int(np.sqrt(n_nodes))
            model_f = Siren_Network(image_width=n_nodes_per_axis, in_features=model_config.input_size_nnr,
                                    out_features=model_config.output_size_nnr,
                                    hidden_features=model_config.hidden_dim_nnr,
                                    hidden_layers=model_config.n_layers_nnr, outermost_linear=True, device=device,
                                    first_omega_0=model_config.omega, hidden_omega_0=model_config.omega)
            net = f'{log_dir}/models/best_model_f_with_1_graphs_{best_model}.pt'
            state_dict = torch.load(net, map_location=device)
            model_f.load_state_dict(state_dict['model_state_dict'])
            model_f.to(device=device)
            model_f.eval()
            table = PrettyTable(["Modules", "Parameters"])
            total_params = 0
            for name, parameter in model_f.named_parameters():
                if not parameter.requires_grad:
                    continue
                param = parameter.numel()
                table.add_row([name, param])
                total_params += param
            if verbose:
                print(table)
                print(f"Total Trainable Params: {total_params}")


    rmserr_list = []
    pred_err_list = []
    gloss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
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
    n_neurons = x.shape[0]
    x_inference_list = []


    for it in trange(start_it,start_it+800):  # start_it + min(9600+start_it,stop_it-time_step)): #  start_it+200): # min(9600+start_it,stop_it-time_step)):

        check_and_clear_memory(device=device, iteration_number=it, every_n_iterations=25,
                               memory_percentage_threshold=0.6)
        # print(f"Total allocated memory: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB")
        # print(f"Total reserved memory:  {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")

        # with torch.no_grad():
        if it < n_frames - 4:
            x0 = x_list[0][it].clone().detach()
            x0_next = x_list[0][(it + time_step)].clone().detach()
            if not (model_config.particle_model_name == 'PDE_R'):
                y0 = y_list[0][it].clone().detach()
        if has_mesh:
            x[:, 1:5] = x0[:, 1:5].clone().detach()
            if has_mesh_field:
                field = model_f(time=it / n_frames) ** 2
                x[:, 9:10] = field
            dataset_mesh = data.Data(x=x, edge_index=edge_index_mesh, edge_attr=edge_weight_mesh, device=device)
        if do_tracking:
            x = x0.clone().detach()
        if has_excitation:
            x[:, 10: 10 + model_config.excitation_dim] = x0[:, 10: 10 + model_config.excitation_dim]

        # error calculations
        if 'PDE_N' in model_config.signal_model_name:
            nan_mask = torch.isnan(x0[:, 6])
            x0[nan_mask, 6] = baseline_value
            nan_mask = torch.isnan(x[:, 6])
            x[nan_mask, 6] = baseline_value
            rmserr = torch.sqrt(torch.mean(torch.sum(bc_dpos(x[:n_neurons, 6:7] - x0[:, 6:7]) ** 2, axis=1)))
            neuron_gt_list.append(x0[:, 6:7])
            neuron_pred_list.append(x[:n_neurons, 6:7].clone().detach())
            if ('short_term_plasticity' in field_type) | ('modulation' in field_type):
                modulation_gt_list.append(x0[:, 8:9])
                modulation_pred_list.append(x[:, 8:9].clone().detach())
        elif 'WaveMesh' in model_config.mesh_model_name:
            rmserr = torch.sqrt(torch.mean((x[mask_mesh.squeeze(), 6:7] - x0[mask_mesh.squeeze(), 6:7]) ** 2))
        elif 'RD_Mesh' in model_config.mesh_model_name:
            rmserr = torch.sqrt(
                torch.mean(torch.sum((x[mask_mesh.squeeze(), 6:9] - x0[mask_mesh.squeeze(), 6:9]) ** 2, axis=1)))
            node_gt_list.append(x0[:, 6:9])
            node_pred_list.append(x[:n_neurons, 6:9].clone().detach())
        elif has_bounding_box:
            rmserr = torch.sqrt(
                torch.mean(torch.sum(bc_dpos(x[:, 1:dimension + 1] - x0[:, 1:dimension + 1]) ** 2, axis=1)))
        else:
            if (do_tracking) | (x.shape[0] != x0.shape[0]):
                rmserr = torch.zeros(1, device=device)
            else:
                rmserr = torch.sqrt(
                    torch.mean(torch.sum(bc_dpos(x[:, 1:dimension + 1] - x0[:, 1:dimension + 1]) ** 2, axis=1)))
            if x.shape[0] > 5000:
                geomloss = gloss(x[0:5000, 1:3], x0[0:5000, 1:3])
            else:
                geomloss = gloss(x[:, 1:3], x0[:, 1:3])
            geomloss_list.append(geomloss.item())
        rmserr_list.append(rmserr.item())

        if config.training.shared_embedding:
            data_id = torch.ones((n_neurons, 1), dtype=torch.int, device=device)
        else:
            data_id = torch.ones((n_neurons, 1), dtype=torch.int, device=device) * run

        # update calculations
        if model_config.mesh_model_name == 'DiffMesh':
            with torch.no_grad():
                pred = mesh_model(dataset_mesh, data_id=0, )
            x[:, 6:7] += pred * hnorm * delta_t
        elif model_config.mesh_model_name == 'WaveMeshSmooth':
            pred = mesh_model(dataset_mesh, data_id=data_id, training=False, phi=torch.zeros(1, device=device))
            with torch.no_grad():
                x[mask_mesh.squeeze(), 7:8] += pred[mask_mesh.squeeze()] * hnorm * delta_t
                x[mask_mesh.squeeze(), 6:7] += x[mask_mesh.squeeze(), 7:8] * delta_t
        elif model_config.mesh_model_name == 'WaveMesh':
            with torch.no_grad():
                pred = mesh_model(dataset_mesh, data_id=data_id)
            x[mask_mesh.squeeze(), 7:8] += pred[mask_mesh.squeeze()] * hnorm * delta_t
            x[mask_mesh.squeeze(), 6:7] += x[mask_mesh.squeeze(), 7:8] * delta_t
        elif 'RD_Mesh' in model_config.mesh_model_name:
            with torch.no_grad():
                if 'test_simulation' in test_mode:
                    y = y0 / hnorm
                    pred = y
                else:
                    pred = mesh_model(dataset_mesh, data_id=data_id, training=False, has_field=has_mesh_field)
                if has_mesh_field:
                    field = model_f(time=it / n_frames) ** 2
                    x[:, 9:10] = field
                    if 'replace_blue' in field_type:
                        x[:, 8:9] = field

                if model_config.prediction == '2nd_derivative':
                    x[mask_mesh.squeeze(), 9:12] += pred[mask_mesh.squeeze()] * hnorm * delta_t
                    x[mask_mesh.squeeze(), 6:9] += x[mask_mesh.squeeze(), 9:12] * delta_t
                else:
                    x[mask_mesh.squeeze(), 6:9] += pred[mask_mesh.squeeze()] * hnorm * delta_t
                x[:, 6:9] = torch.clamp(x[:, 6:9], 0, 1.1)
        elif has_particle_field:
            match model_config.field_type:
                case 'tensor':
                    x_mesh[:, 6:7] = model.field[run]
                case 'siren':
                    x_mesh[:, 6:7] = model_f() ** 2
                case 'siren_with_time':
                    x_mesh[:, 6:7] = model_f(time=it / n_frames) ** 2
            x_particle_field = torch.concatenate((x_mesh, x), dim=0)

            distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
            adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
            edge_index = adj_t.nonzero().t().contiguous()
            dataset_p_p = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index, field=[])

            distance = torch.sum(
                bc_dpos(x_particle_field[:, None, 1:dimension + 1] - x_particle_field[None, :, 1:dimension + 1]) ** 2,
                dim=2)
            adj_t = ((distance < (max_radius / 2) ** 2) & (distance > min_radius ** 2)).float() * 1
            edge_index = adj_t.nonzero().t().contiguous()
            pos = torch.argwhere((edge_index[1, :] >= n_nodes) & (edge_index[0, :] < n_nodes))
            pos = to_numpy(pos[:, 0])
            edge_index = edge_index[:, pos]
            dataset_f_p = data.Data(x=x_particle_field, pos=x_particle_field[:, 1:3], edge_index=edge_index,
                                    field=x_particle_field[:, 6:7])

            with torch.no_grad():
                y0 = model(dataset_p_p, data_id=1, training=False, phi=torch.zeros(1, device=device),
                           has_field=False)
                y1 = model_f_p(dataset_f_p, data_id=1, training=False, phi=torch.zeros(1, device=device),
                               has_field=True)[n_nodes:]
                y = y0 + y1

            if model_config.prediction == '2nd_derivative':
                y = y * ynorm * delta_t
                x[:, 3:5] = x[:, 3:5] + y  # speed update
            else:
                y = y * vnorm
                x[:, 3:5] = y

            x[:, 1:3] = bc_pos(x[:, 1:3] + x[:, 3:5] * delta_t)
        elif 'PDE_N' in model_config.signal_model_name:
            if 'visual' in field_type:
                x[:n_nodes, 8:9] = model_f(time=it / n_frames) ** 2
                x[n_nodes:n_neurons, 8:9] = 1
            elif 'learnable_short_term_plasticity' in field_type:
                alpha = (k % model.embedding_step) / model.embedding_step
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

            nan_mask = torch.isnan(x[:, 6])
            x[nan_mask, 6] = baseline_value

            dataset = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index)
            if 'test_simulation' in test_mode:
                y = y0 / ynorm
            else:
                with torch.no_grad():
                    pred = model(dataset, data_id=data_id)
                    y = pred
            # signal update
            x[:n_neurons, 6:7] = x[:n_neurons, 6:7] + y[:n_neurons] * delta_t
            # if 'CElegans' in dataset_name:
            #     x[:n_neurons, 6:7] = torch.clamp(x[:n_neurons, 6:7], min=0, max=10)
        else:
            with torch.no_grad():
                if has_ghost:
                    x_ = x
                    x_ghost = model_ghost.get_pos(dataset_id=run, frame=it, bc_pos=bc_pos)
                    x_ = torch.cat((x_, x_ghost), 0)
                # compute connectivity and prediction

                distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
                adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
                edge_index = adj_t.nonzero().t().contiguous()

                if has_field:
                    field = model_f(time=it / n_frames) ** 2
                    x[:, 6:7] = field

                if time_window > 0:
                    xt = []
                    for t in range(time_window):
                        x_ = x_list[0][it - t].clone().detach()
                        xt.append(x_[:, :])
                    dataset = data.Data(x=xt, edge_index=edge_index)
                else:
                    dataset = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index)

                if 'test_simulation' in test_mode:
                    y = y0 / ynorm
                    pred = y
                else:
                    pred = model(dataset, data_id=data_id, training=False, has_field=has_field, k=it)
                    y = pred

                if has_ghost:
                    y = y[mask_ghost]

                if sub_sampling > 1:
                    # predict position, does not work with rotation_augmentation
                    if time_step == 1:
                        x_next = bc_pos(y[:, 0:dimension])
                    elif time_step == 2:
                        x_next = bc_pos(y[:, dimension:2 * dimension])
                    x[:, dimension + 1:2 * dimension + 1] = (x_next - x[:, 1:dimension + 1]) / delta_t
                    x[:, 1:dimension + 1] = x_next
                    loss = (x[:, 1:dimension + 1] - x0_next[:, 1:dimension + 1]).norm(2)
                    pred_err_list.append(to_numpy(torch.sqrt(loss)))
                elif do_tracking:
                    x_pos_next = x0_next[:, 1:dimension + 1].clone().detach()
                    if pred.shape[1] != dimension:
                        pred = torch.cat((pred, torch.zeros(pred.shape[0], 1, device=pred.device)), dim=1)
                    if model_config.prediction == '2nd_derivative':
                        x_pos_pred = (x[:, 1:dimension + 1] + delta_t * time_step * (
                                    x[:, dimension + 1:2 * dimension + 1] + delta_t * time_step * pred * ynorm))
                    else:
                        x_pos_pred = (x[:, 1:dimension + 1] + delta_t * time_step * pred * ynorm)
                    distance = torch.sum(bc_dpos(x_pos_pred[:, None, :] - x_pos_next[None, :, :]) ** 2, dim=2)
                    result = distance.min(dim=1)
                    min_value = result.values
                    indices = result.indices
                    loss = torch.std(torch.sqrt(min_value))
                    pred_err_list.append(to_numpy(torch.sqrt(loss)))
                    if 'inference' in test_mode:
                        x[:, dimension + 1:2 * dimension + 1] = pred.clone().detach() / (delta_t * time_step)

                else:
                    pred_err_list.append(to_numpy(torch.sqrt(loss)))
                    if model_config.prediction == '2nd_derivative':
                        y = y * ynorm * delta_t
                        x[:n_neurons, dimension + 1:2 * dimension + 1] = x[:n_neurons, dimension + 1:2 * dimension + 1] + y[:n_neurons]  # speed update
                    else:
                        y = y * vnorm
                        if 'PDE_N' in model_config.signal_model_name:
                            x[:n_neurons, 6:7] += y[:n_neurons] * delta_t  # signal update
                        else:
                            x[:n_neurons, dimension + 1:2 * dimension + 1] = y[:n_neurons]
                    x[:, 1:dimension + 1] = bc_pos(
                        x[:, 1:dimension + 1] + x[:, dimension + 1:2 * dimension + 1] * delta_t)  # position update

                    # matplotlib.use("Qt5Agg")
                    # fig = plt.figure()
                    # plt.scatter(to_numpy(y0), to_numpy(ynorm*pred[:n_neurons, 0:dimension]))
                    # plt.xlabel('y0')
                    # plt.ylabel('pred')
                    # plt.savefig(f'pred_{it}.png')
                    # plt.close()

                if 'bounce_all_v0' in test_mode:
                    gap = 0.005
                    bouncing_pos = torch.argwhere((x[:, 1] <= 0.1 - gap) | (x[:, 1] >= 0.9 + gap)).squeeze()
                    if bouncing_pos.numel() > 0:
                        x[bouncing_pos, 3] = - 0.7 * x[bouncing_pos, 3]
                        x[bouncing_pos, 1] += x[bouncing_pos, 3] * delta_t
                    bouncing_pos = torch.argwhere((x[:, 2] <= 0.1 - gap) | (x[:, 2] >= 0.9 + gap)).squeeze()
                    if bouncing_pos.numel() > 0:
                        x[bouncing_pos, 4] = - 0.7 * x[bouncing_pos, 4]
                        x[bouncing_pos, 2] += x[bouncing_pos, 4] * delta_t
                if 'bounce_all' in test_mode:
                    gap = 0.005
                    bouncing_pos = torch.argwhere((x[:, 1] <= 0.1 + gap) | (x[:, 1] >= 0.9 - gap)).squeeze()
                    if bouncing_pos.numel() > 0:
                        x[bouncing_pos, 3] = - 0.7 * bounce_coeff * x[bouncing_pos, 3]
                        x[bouncing_pos, 1] += x[bouncing_pos, 3] * delta_t * 10
                    bouncing_pos = torch.argwhere((x[:, 2] <= 0.1 + gap) | (x[:, 2] >= 0.9 - gap)).squeeze()
                    if bouncing_pos.numel() > 0:
                        x[bouncing_pos, 4] = - 0.7 * bounce_coeff * x[bouncing_pos, 4]
                        x[bouncing_pos, 2] += x[bouncing_pos, 4] * delta_t * 10
                if 'fixed' in test_mode:
                    fixed_pos = torch.argwhere(x[:, 5] == 0)
                    x[fixed_pos.squeeze(), 1:2 * dimension + 1] = x_list[0][it + 1, fixed_pos.squeeze(),
                                                                  1:2 * dimension + 1].clone().detach()
                if 'inference' in test_mode:
                    x_inference_list.append(x)

                if (time_window > 1) & ('plot_data' not in test_mode):
                    moving_pos = torch.argwhere(x[:, 5] != 0)
                    x_list[0][it + 1, moving_pos.squeeze(), 1:2 * dimension + 1] = x[moving_pos.squeeze(),
                                                                                   1:2 * dimension + 1].clone().detach()

        # vizualization
        if 'plot_data' in test_mode:
            x = x_list[0][it].clone().detach()

        if (it % step == 0) & (it >= 0) & visualize:

            # print(f'acceleration {torch.max(y[:, 0])}')
            # print(f'speed {torch.max(x[:, 3])}')

            num = f"{it:06}"

            if 'latex' in style:
                plt.rcParams['text.usetex'] = True
                rc('font', **{'family': 'serif', 'serif': ['Palatino']})
            if 'black' in style:
                plt.style.use('dark_background')
                mc = 'w'
            else:
                plt.style.use('default')
                mc = 'k'

            fig, ax = fig_init(formatx='%.1f', formaty='%.1f')
            ax.tick_params(axis='both', which='major', pad=15)

            if has_mesh:
                pts = x[:, 1:3].detach().cpu().numpy()
                tri = Delaunay(pts)
                colors = torch.sum(x[tri.simplices, 6], dim=1) / 3.0
                if model_config.mesh_model_name == 'DiffMesh':
                    plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                  facecolors=colors.detach().cpu().numpy(), vmin=0, vmax=1000)
                if 'WaveMesh' in model_config.mesh_model_name:
                    plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                  facecolors=colors.detach().cpu().numpy(), vmin=-1000, vmax=1000)
                    fmt = lambda x, pos: '{:.1f}'.format((x) / 100, pos)
                    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
                    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
                if 'RD_Gray_Scott_Mesh' in model_config.mesh_model_name:
                    fig = plt.figure(figsize=(12, 6))
                    ax = fig.add_subplot(1, 2, 1)
                    colors = torch.sum(x[tri.simplices, 6], dim=1) / 3.0
                    plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                  facecolors=colors.detach().cpu().numpy(), vmin=0, vmax=1)
                    plt.xticks([])
                    plt.yticks([])
                    plt.axis('off')
                    ax = fig.add_subplot(1, 2, 2)
                    colors = torch.sum(x[tri.simplices, 7], dim=1) / 3.0
                    plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                  facecolors=colors.detach().cpu().numpy(), vmin=0, vmax=1)
                    plt.xticks([])
                    plt.yticks([])
                    plt.axis('off')
                if 'RD_Mesh' in model_config.mesh_model_name:
                    H1_IM = torch.reshape(x[:, 6:9], (n_nodes_per_axis, n_nodes_per_axis, 3))
                    H1_IM = torch.clip(H1_IM, 0, 1)

                    imwrite(f"./{log_dir}/tmp_recons/H1_IM_{config_file}_{num}.tif",
                            (to_numpy(H1_IM) * 255).astype(np.uint8))

                    plt.imshow(to_numpy(H1_IM))
                    plt.axis('off')
                    plt.xticks([])
                    plt.yticks([])
                    # plt.xticks([])
                    # plt.yticks([])
                    # plt.axis('off')

                    # plt.figure()
                    # plt.scatter(to_numpy(x[:, 2]), to_numpy(x[:, 1]), s=8, c=to_numpy(pred[:, 0] * delta_t * hnorm), cmap='viridis', vmin=0, vmax=5)
            elif ('visual' in field_type) & ('PDE_N' in model_config.signal_model_name):
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
            elif 'PDE_N' in model_config.signal_model_name:
                plt.close()
                matplotlib.rcParams['savefig.pad_inches'] = 0

                black_to_green = LinearSegmentedColormap.from_list('black_green', ['black', 'green'])
                black_to_yellow = LinearSegmentedColormap.from_list('black_yellow', ['black', 'yellow'])

                plt.figure(figsize=(10, 10))
                plt.scatter(to_numpy(x[:, 1]), to_numpy(x[:, 2]), s=700, c=to_numpy(x[:, 6]), alpha=1, edgecolors='none', vmin =2 , vmax=8, cmap=black_to_green)

                if 'excitation' in model.update_type:
                    plt.scatter(-0.45, 0.5, s=700, c=to_numpy(x[0, 10]) + 0.25, cmap=black_to_yellow, vmin=0, vmax=1)
                    plt.scatter(-0.4, 0.5, s=700, c=to_numpy(x[0, 11]) + 0.25, cmap=black_to_yellow, vmin=0, vmax=1)
                    plt.scatter(-0.35, 0.5, s=700, c=to_numpy(x[0, 12]) + 0.25, cmap=black_to_yellow, vmin=0, vmax=1)

                plt.axis('off')
                plt.xticks([])
                plt.yticks([])
                plt.xlim([-0.6, 0.6])
                plt.ylim([-0.6, 0.6])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_recons/Nodes_{config_file}_{num}.tif", dpi=80)
                plt.close()

                # im = imread(f"./{log_dir}/tmp_recons/Nodes_{config_file}_{num}.tif")
                # plt.figure(figsize=(10, 10))
                # plt.imshow(im)
                # plt.axis('off')
                # plt.xticks([])
                # plt.yticks([])
                # plt.subplot(3, 3, 1)
                # plt.imshow(im[800:1000, 800:1000, :])
                # plt.xticks([])
                # plt.yticks([])
                # plt.tight_layout()
                # plt.savefig(f"./{log_dir}/tmp_recons/Nodes_{config_file}_{num}.tif", dpi=80)
                # plt.close()

            elif 'PDE_K' in model_config.particle_model_name:

                plt.close()
                fig = plt.figure(figsize=(12, 12))
                plt.scatter(x[:, 2].detach().cpu().numpy(),
                            x[:, 1].detach().cpu().numpy(), s=20, color='r')
                if it < n_frames - 1:
                    x0_ = x_list[0][it + 1].clone().detach()
                    plt.scatter(x0_[:, 2].detach().cpu().numpy(),
                                x0_[:, 1].detach().cpu().numpy(), s=40, color='w', alpha=1, edgecolors='None')

                plt.xlim([-3, 3])
                plt.ylim([-3, 3])
            elif do_tracking:

                # plt.scatter(to_numpy(x[:, 2]), to_numpy(x[:, 1]), s=1, c='w', alpha=0.25)
                plt.scatter(to_numpy(x0[:, 2]), to_numpy(x0[:, 1]), s=10, c='w', alpha=0.5)
                plt.scatter(to_numpy(x_pos_pred[:, 1]), to_numpy(x_pos_pred[:, 0]), s=10, c='r')
                x1 = x_list[0][it + time_step].clone().detach()
                plt.scatter(to_numpy(x1[:, 2]), to_numpy(x1[:, 1]), s=10, c='g')

                plt.xticks([])
                plt.yticks([])

                if 'zoom' in style:
                    for m in range(x.shape[0]):
                        plt.arrow(x=to_numpy(x0[m, 2]), y=to_numpy(x0[m, 1]),
                                  dx=to_numpy(x[m, dimension + 2]) * delta_t,
                                  dy=to_numpy(x[m, dimension + 1]) * delta_t, head_width=0.004,
                                  length_includes_head=True, color='g')
                    plt.xlim([300, 400])
                    plt.ylim([300, 400])
                else:
                    plt.xlim([0, 700])
                    plt.ylim([0, 700])
                plt.tight_layout()
            else:
                s_p = 10
                index_particles = get_index_particles(x, n_neuron_types, dimension)
                for n in range(n_neuron_types):
                    if 'bw' in style:
                        plt.scatter(x[index_particles[n], 2].detach().cpu().numpy(),
                                    x[index_particles[n], 1].detach().cpu().numpy(), s=s_p, color='w')
                    else:
                        plt.scatter(x[index_particles[n], 2].detach().cpu().numpy(),
                                    x[index_particles[n], 1].detach().cpu().numpy(), s=s_p, color=cmap.color(n))
                plt.xlim([0, 1])
                plt.ylim([0, 1])

                if ('field' in style) & has_field:
                    if 'zoom' in style:
                        plt.scatter(to_numpy(x[:, 2]), to_numpy(x[:, 1]), s=s_p * 50, c=to_numpy(x[:, 6]) * 20,
                                    alpha=0.5, cmap='viridis', vmin=0, vmax=1.0)
                    else:
                        plt.scatter(to_numpy(x[:, 2]), to_numpy(x[:, 1]), s=s_p * 2, c=to_numpy(x[:, 6]) * 20,
                                    alpha=0.5, cmap='viridis', vmin=0, vmax=1.0)

                if particle_of_interest > 1:

                    xc = to_numpy(x[particle_of_interest, 2])
                    yc = to_numpy(x[particle_of_interest, 1])
                    pos = torch.argwhere(edge_index[1, :] == particle_of_interest)
                    pos = pos[:, 0]
                    if 'zoom' in style:
                        plt.scatter(to_numpy(x[edge_index[0, pos], 2]), to_numpy(x[edge_index[0, pos], 1]), s=s_p * 10,
                                    color=mc, alpha=1.0)
                    else:
                        plt.scatter(to_numpy(x[edge_index[0, pos], 2]), to_numpy(x[edge_index[0, pos], 1]), s=s_p * 1,
                                    color=mc, alpha=1.0)

                    # for k in range(pos.shape[0]):
                    #     plt.arrow(x[edge_index[1,pos[k]], 2].detach().cpu().numpy(),
                    #                 x[edge_index[1,pos[k]], 1].detach().cpu().numpy(),  dx=to_numpy(model.msg[k,1]) * delta_t/20,
                    #               dy=to_numpy(model.msg[k,0]) * delta_t/20, head_width=0.004, length_includes_head=True, color=mc,alpha=0.25)

                    plt.arrow(x=to_numpy(x[particle_of_interest, 2]), y=to_numpy(x[particle_of_interest, 1]),
                              dx=to_numpy(x[particle_of_interest, 4]) * delta_t * 100,
                              dy=to_numpy(x[particle_of_interest, 3]) * delta_t * 100, head_width=0.004,
                              length_includes_head=True, color='b')
                    if model_config.prediction == '2nd_derivative':
                        plt.arrow(x=to_numpy(x[particle_of_interest, 2]), y=to_numpy(x[particle_of_interest, 1]),
                                  dx=to_numpy(y0[particle_of_interest, 1]) * delta_t ** 2 * 100,
                                  dy=to_numpy(y0[particle_of_interest, 0]) * delta_t ** 2 * 100, head_width=0.004,
                                  length_includes_head=True, color='g')
                        plt.arrow(x=to_numpy(x[particle_of_interest, 2]), y=to_numpy(x[particle_of_interest, 1]),
                                  dx=to_numpy(y[particle_of_interest, 1]) * delta_t * 100,
                                  dy=to_numpy(y[particle_of_interest, 0]) * delta_t * 100, head_width=0.004,
                                  length_includes_head=True, color='r')

                if 'zoom' in style:
                    plt.xlim([xc - 0.1, xc + 0.1])
                    plt.ylim([yc - 0.1, yc + 0.1])
                    plt.xticks([])
                    plt.yticks([])

            if 'latex' in style:
                plt.xlabel(r'$x$', fontsize=78)
                plt.ylabel(r'$y$', fontsize=78)
                plt.xticks(fontsize=48.0)
                plt.yticks(fontsize=48.0)
            if 'frame' in style:
                plt.xlabel('x', fontsize=48)
                plt.ylabel('y', fontsize=48)
                plt.xticks(fontsize=48.0)
                plt.yticks(fontsize=48.0)
                plt.text(0, 1.1, f'   ', ha='left', va='top', transform=ax.transAxes, fontsize=48)
                ax.tick_params(axis='both', which='major', pad=15)
                # cbar = plt.colorbar(shrink=0.5)
                # cbar.ax.tick_params(labelsize=32)
            if 'arrow' in style:
                for m in range(x.shape[0]):
                    if x[m, 4] != 0:
                        if 'speed' in style:
                            # plt.arrow(x=to_numpy(x[m, 2]), y=to_numpy(x[m, 1]), dx=to_numpy(y0[m, 1]) * delta_t * 50, dy=to_numpy(y0[m, 0]) * delta_t * 50, head_width=0.004, length_includes_head=False, color='g')
                            # plt.arrow(x=to_numpy(x[m, 2]), y=to_numpy(x[m, 1]), dx=to_numpy(x[m, 4]) * delta_t * 50, dy=to_numpy(x[m, 3]) * delta_t * 50, head_width=0.004, length_includes_head=False, color='w')
                            # angle = compute_signed_angle(x[m, 3:5], y0[m, 0:2])
                            # angle_list.append(angle)
                            plt.arrow(x=to_numpy(x[m, 2]), y=to_numpy(x[m, 1]), dx=to_numpy(x[m, 4]) * delta_t * 2,
                                      dy=to_numpy(x[m, 3]) * delta_t * 2, head_width=0.004, length_includes_head=True,
                                      color='g')
                        if 'acc_true' in style:
                            plt.arrow(x=to_numpy(x[m, 2]), y=to_numpy(x[m, 1]), dx=to_numpy(y0[m, 1]) / 5E3,
                                      dy=to_numpy(y0[m, 0]) / 5E3, head_width=0.004, length_includes_head=True,
                                      color='r')
                        if 'acc_learned' in style:
                            plt.arrow(x=to_numpy(x[m, 2]), y=to_numpy(x[m, 1]),
                                      dx=to_numpy(pred[m, 1] * ynorm.squeeze()) / 5E3,
                                      dy=to_numpy(pred[m, 0] * ynorm.squeeze()) / 5E3, head_width=0.004,
                                      length_includes_head=True, color='r')
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                # plt.xlim([0.4,0.6])
                # plt.ylim([0.4,0.6])
            if 'name' in style:
                plt.title(f"{os.path.basename(log_dir)}", fontsize=24)

            if 'no_ticks' in style:
                plt.xticks([])
                plt.yticks([])
            if 'PDE_G' in model_config.particle_model_name:
                plt.xlim([-2, 2])
                plt.ylim([-2, 2])
            if 'PDE_GS' in model_config.particle_model_name:

                object_list = ['sun', 'mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune',
                               'pluto', 'io',
                               'europa', 'ganymede', 'callisto', 'mimas', 'enceladus', 'tethys', 'dione', 'rhea',
                               'titan', 'hyperion', 'moon',
                               'phobos', 'deimos', 'charon']

                masses = torch.tensor(
                    [1.989e30, 3.30e23, 4.87e24, 5.97e24, 6.42e23, 1.90e27, 5.68e26, 8.68e25, 1.02e26, 1.31e22,
                     8.93e22, 4.80e22, 1.48e23, 1.08e23, 3.75e19, 1.08e20,
                     6.18e20, 1.10e21, 2.31e21, 1.35e23, 5.62e18, 7.35e22, 1.07e16, 1.48e15, 1.52e21],
                    device=device)

                pos = x[:, 1:dimension + 1]  # - x[0,1:dimension + 1]
                distance = torch.sqrt(torch.sum(bc_dpos(pos[:, None, :] - pos[None, 0, :]) ** 2, dim=2))
                unit_vector = pos / distance

                if it == 0:
                    log_coeff = torch.log(distance[1:])
                    log_coeff_min = torch.min(log_coeff)
                    log_coeff_max = torch.max(log_coeff)
                    log_coeff_edge_diff = log_coeff_max - log_coeff_min
                    d_log = [log_coeff_min, log_coeff_max, log_coeff_edge_diff]

                    log_coeff = torch.log(masses)
                    log_coeff_min = torch.min(log_coeff)
                    log_coeff_max = torch.max(log_coeff)
                    log_coeff_edge_diff = log_coeff_max - log_coeff_min
                    m_log = [log_coeff_min, log_coeff_max, log_coeff_edge_diff]

                    m_ = torch.log(masses) / m_log[2]

                distance_ = (torch.log(distance) - d_log[0]) / d_log[2]
                pos = distance_ * unit_vector
                pos = to_numpy(pos)
                pos[0] = 0

                for n in range(25):
                    plt.scatter(pos[n, 1], pos[n, 0], s=200 * to_numpy(m_[n] ** 3), color=cmap.color(n))
                    # plt.text(pos[n,1], pos[n, 0], object_list[n], fontsize=8)
                plt.xlim([-1.2, 1.2])
                plt.ylim([-1.2, 1.2])
                plt.xticks([])
                plt.yticks([])
            if 'PDE_K' in model_config.particle_model_name:
                plt.xlim([-3, 3])
                plt.ylim([-3, 3])
                plt.xticks(fontsize=24)
                plt.yticks(fontsize=24)

            # save figure
            if not ('PDE_N' in model_config.signal_model_name):
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_recons/Fig_{config_file}_{run}_{num}.tif", dpi=100)
                plt.close()

            if ('RD_Mesh' in model_config.mesh_model_name) & (it % 40 == 0) & (it > 0):

                node_gt_list_ = torch.cat(node_gt_list, 0)
                node_pred_list_ = torch.cat(node_pred_list, 0)
                node_gt_list_ = torch.reshape(node_gt_list_, (node_gt_list_.shape[0] // n_neurons, n_neurons, 3))
                node_pred_list_ = torch.reshape(node_pred_list_,
                                                (node_pred_list_.shape[0] // n_neurons, n_neurons, 3))

                plt.figure(figsize=(10, 10))
                n_list = []
                for k in range(0, n_neurons, n_neurons // 20):
                    if torch.max(node_gt_list_[:, k, 0].squeeze()) > 0.5:
                        plt.plot(to_numpy(node_gt_list_[:, k, 0].squeeze()))
                        n_list.append(k)

                if n_nodes == 4096:
                    n = [612, 714, 1428, 1632, 1836, 2142, 2346, 3162, 3264, 3672]
                elif n_nodes == 16384:
                    n = [2454, 3272, 4908, 5317, 7362, 7771, 9407, 11452, 12270, 14724]
                elif n_nodes == 65536:
                    n = [13104, 14742, 18018, 22932, 26208, 31122, 36036, 39312, 42588, 49140, 50778, 58968]
                elif n_nodes == 10000:
                    n = [2250, 2500, 3500, 4750, 5000, 5750, 6250, 9500]

                plt.figure(figsize=(20, 10))
                ax = plt.subplot(121)
                plt.plot(to_numpy(node_gt_list_[:, n[0], 0]), c='r', linewidth=4, label='true', alpha=0.5)
                plt.plot(to_numpy(node_pred_list_[:, n[0], 0]), linewidth=2, c='r', label='learned')
                plt.legend(fontsize=24)

                plt.plot(to_numpy(node_gt_list_[:, n[1:5], 0]), c='r', linewidth=4, alpha=0.5)
                plt.plot(to_numpy(node_pred_list_[:, n[1:5], 0]), c='r', linewidth=2)
                plt.ylim([0, 1])
                plt.xlim([0, 200])

                ax = plt.subplot(122)
                plt.scatter(to_numpy(node_gt_list_[-1, :]), to_numpy(node_pred_list_[-1, :]), s=1, c=mc)
                plt.xlim([0, 1])
                plt.ylim([0, 1])

                x_data = to_numpy(node_gt_list_[-1, :, 0])
                y_data = to_numpy(node_pred_list_[-1, :, 0])
                lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                residuals = y_data - linear_model(x_data, *lin_fit)
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                plt.xlabel(r'true $x_i$', fontsize=48)
                plt.ylabel(r'learned $x_i$', fontsize=48)
                plt.text(0.05, 0.95, f'$R^2$: {np.round(r_squared, 3)}', fontsize=34)
                plt.text(0.05, 0.85, f'slope: {np.round(lin_fit[0], 2)}', fontsize=34)
                plt.tight_layout()
                plt.savefig(f'./{log_dir}/results/comparison_xi_{it}.png', dpi=80)
                plt.close()

            if ('PDE_N' in model_config.signal_model_name) & (it % 200 == 0) & (it > 0):
                if 'CElegans' in dataset_name:
                    n = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
                else:
                    n = [20, 30, 100, 150, 260, 270, 520, 620, 720, 820]

                neuron_gt_list_ = torch.cat(neuron_gt_list, 0)
                neuron_pred_list_ = torch.cat(neuron_pred_list, 0)
                neuron_gt_list_ = torch.reshape(neuron_gt_list_,
                                                (neuron_gt_list_.shape[0] // n_neurons, n_neurons))
                neuron_pred_list_ = torch.reshape(neuron_pred_list_,
                                                  (neuron_pred_list_.shape[0] // n_neurons, n_neurons))

                plt.figure(figsize=(20, 10))
                if 'latex' in style:
                    plt.rcParams['text.usetex'] = True
                    rc('font', **{'family': 'serif', 'serif': ['Palatino']})
                ax = plt.subplot(121)
                plt.plot(neuron_gt_list_[:, n[0]].detach().cpu().numpy(), c=mc, linewidth=8, label='true',
                         alpha=0.25)
                plt.plot(neuron_pred_list_[:, n[0]].detach().cpu().numpy(), linewidth=4, c='k',
                         label='learned')
                plt.legend(fontsize=24)
                plt.plot(neuron_gt_list_[:, n[1:10]].detach().cpu().numpy(), c=mc, linewidth=8, alpha=0.25)
                plt.plot(neuron_pred_list_[:, n[1:10]].detach().cpu().numpy(), linewidth=4)
                plt.xlim([0, n_frames])
                plt.xlabel(r'time-points', fontsize=48)
                plt.ylabel(r'$x_i$', fontsize=48)
                plt.xticks(fontsize=24)
                plt.yticks(fontsize=24)
                plt.ylim([0, 10])
                # plt.ylim([-30, 30])
                # plt.text(40, 26, f'time: {it}', fontsize=34)
                ax = plt.subplot(122)
                plt.scatter(to_numpy(neuron_gt_list_[-1, :]), to_numpy(neuron_pred_list_[-1, :]), s=10, c=mc)
                plt.xlim([-30, 30])
                plt.ylim([-30, 30])
                plt.xticks(fontsize=24)
                plt.yticks(fontsize=24)
                x_data = to_numpy(neuron_gt_list_[-1, :])
                y_data = to_numpy(neuron_pred_list_[-1, :])
                lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                residuals = y_data - linear_model(x_data, *lin_fit)
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                plt.xlabel(r'true $x_i$', fontsize=48)
                plt.ylabel(r'learned $x_i$', fontsize=48)
                plt.text(-28.5, 26, f'$R^2$: {np.round(r_squared, 3)}', fontsize=34)
                plt.text(-28.5, 22, f'slope: {np.round(lin_fit[0], 2)}', fontsize=34)
                plt.tight_layout()
                plt.savefig(f'./{log_dir}/results/comparison_xi_{it}.png', dpi=80)
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
                    plt.tight_layout()
                    plt.tight_layout()
                    plt.savefig(f'./{log_dir}/results/comparison_modulation_{it}.png', dpi=80)
                    plt.close()

            if ('feature' in style) & ('PDE_MLPs_A' in config.graph_model.particle_model_name):
                if 'PDE_MLPs_A_bis' in model.model:
                    fig = plt.figure(figsize=(22, 5))
                else:
                    fig = plt.figure(figsize=(22, 6))
                for k in range(model.new_features.shape[1]):
                    ax = fig.add_subplot(1, model.new_features.shape[1], k + 1)
                    plt.scatter(to_numpy(x[:, 2]), to_numpy(x[:, 1]), c=to_numpy(model.new_features[:, k]), s=5,
                                cmap='viridis')
                    ax.set_title(f'new_features {k}')
                    # cbar = plt.colorbar()
                    # cbar.ax.tick_params(labelsize=6)
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_recons/Features_{config_file}_{run}_{num}.tif", dpi=100)
                plt.close()

            if 'boundary' in style:
                fig, ax = fig_init(formatx='%.1f', formaty='%.1f')
                t = torch.min(x[:, 7:], -1).values
                plt.scatter(to_numpy(x[:, 2]), to_numpy(x[:, 1]), s=25, c=to_numpy(t), vmin=-1, vmax=1)
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_recons/Boundary_{config_file}_{num}.tif", dpi=80)
                plt.close()



    if 'inference' in test_mode:
        torch.save(x_inference_list, f"./{log_dir}/x_inference_list_{run}.pt")

    print('prediction error {:.3e}+/-{:.3e}'.format(np.mean(pred_err_list), np.std(pred_err_list)))
    print('average rollout RMSE {:.3e}+/-{:.3e}'.format(np.mean(rmserr_list), np.std(rmserr_list)))

    # if has_mesh:
    #     h = x_mesh_list[0][0][:, 6:7]
    #     for k in range(n_frames):
    #         h = torch.cat((h, x_mesh_list[0][k][:, 6:7]), 0)
    #     h = to_numpy(h)
    #     print(h.shape)
    #     print('average u {:.3e}+/-{:.3e}'.format(np.mean(h), np.std(h)))

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
            dataset = data.Data(x=temp1_, edge_index=torch.squeeze(temp4[:, p]))
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

