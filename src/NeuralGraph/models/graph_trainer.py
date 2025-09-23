import os
import time

import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
import torch
import torch.nn.functional as F
import random

from GNN_Main import *
from NeuralGraph.models.utils import *
from NeuralGraph.utils import *
from NeuralGraph.models.Siren_Network import *
from NeuralGraph.models.Signal_Propagation_FlyVis import *
from NeuralGraph.models.Signal_Propagation_Zebra import *
from NeuralGraph.models.Signal_Propagation_Temporal import *
from NeuralGraph.sparsify import EmbeddingCluster, sparsify_cluster, clustering_evaluation
from NeuralGraph.generators.davis import *
from NeuralGraph.fitting_models import linear_model
from NeuralGraph.models.utils_zebra import *

from sklearn.neighbors import NearestNeighbors
from scipy.optimize import curve_fit

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
from NeuralGraph.generators.utils import generate_compressed_video_mp4

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
    print(f"\033[92mdataset_name: {dataset_name}\033[0m")

    if 'fly' in config.dataset:
        data_train_flyvis(config, erase, best_model, device)
    elif 'zebra' in config.dataset:
        data_train_zebra(config, erase, best_model, device)
    else:
        data_train_signal(config, erase, best_model, device)


def data_train_signal(config, erase, best_model, device):
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
    time_window = train_config.time_window

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
                        ids = np.sort(ids_)
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
    if time_window >0:
        model = Signal_Propagation_Temporal(aggr_type=model_config.aggr_type, config=config, device=device)
    else:   
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

                k = np.random.randint(n_frames - 4 - time_step - time_window) + time_window
                x = torch.tensor(x_list[run][k], dtype=torch.float32, device=device)
                ids = np.arange(n_neurons)


                if time_window > 0:
                    x_temporal = x_list[run][k - time_window + 1: k + 1, :, 3:4].transpose(1, 0, 2).squeeze(-1)
                    x = torch.cat((x, torch.tensor(x_temporal.reshape(n_neurons, time_window), dtype=torch.float32, device=device)), dim=1)


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
    recursive_training = train_config.recursive_training
    recursive_loop = train_config.recursive_loop
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

    pre_trained_W = train_config.pre_trained_W

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
    learning_rate_NNR_f = train_config.learning_rate_NNR_f

    print(
        f'learning rates: lr_W {lr_W}, lr {lr}, lr_update {lr_update}, lr_embedding {lr_embedding}, learning_rate_NNR_f {learning_rate_NNR_f}')
    logger.info(
        f'learning rates: lr_W {lr_W}, lr {lr}, lr_update {lr_update}, lr_embedding {lr_embedding}, learning_rate_NNR {learning_rate_NNR_f}')

    optimizer, n_total_params = set_trainable_parameters(model=model, lr_embedding=lr_embedding, lr=lr,
                                                         lr_update=lr_update, lr_W=lr_W, learning_rate_NNR=None, learning_rate_NNR_f=learning_rate_NNR_f)
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

    for epoch in range(start_epoch, n_epochs + 1):

        total_loss = 0

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
                y = torch.tensor(x_list[run][k, :, 6:7], device=device).clone().detach()

                dataset = data.Data(x=x, edge_index=edges)
                dataset_batch.append(dataset)

                if batch == 0:

                    data_id = torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * run
                    y_batch = y
                    k_batch = torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * k
                    ids_batch = ids

                else:

                    data_id = torch.cat((data_id, torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * run), dim=0)
                    y_batch = torch.cat((y_batch, y), dim=0)
                    k_batch = torch.cat((k_batch, torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * k), dim=0)
                    ids_batch = np.concatenate((ids_batch, ids + ids_index), axis=0)

                ids_index += x.shape[0]


            # field = model.NNR_f(in_features)**2

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

        list_loss.append(total_loss / n_neurons)
        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))

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






                                                    
                                            



def data_test(config=None, config_file=None, visualize=False, style='color frame', verbose=True, best_model=20, step=15,
              ratio=1, run=1, test_mode='', sample_embedding=False, particle_of_interest=1, new_params = None, device=[]):

    dataset_name = config.dataset
    print(f"\033[92mdataset_name: {dataset_name}\033[0m")

    if 'fly' in config.dataset:
        data_test_flyvis(config, visualize, style, verbose, best_model, step, test_mode, new_params, device)
    elif 'zebra' in config.dataset:
        data_test_zebra(config, visualize, style, verbose, best_model, step, test_mode, device)
    else:
        data_test_signal(config, config_file, visualize, style, verbose, best_model, step, ratio, run, test_mode, sample_embedding, particle_of_interest, new_params, device)


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

    for it in trange(start_it,start_it+800):  # start_it + min(9600+start_it,stop_it-time_step)): #  start_it+200): # min(9600+start_it,stop_it-time_step)):

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

        with torch.no_grad():
            dataset = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index)
            pred = model(dataset, data_id=data_id)
            y = pred
            dataset = data.Data(x=x_generated, pos=x[:, 1:3], edge_index=edge_index)
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
                matplotlib.rcParams['savefig.pad_inches'] = 0

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

            if (it % 200 == 0) & (it > 0):
                if 'CElegans' in dataset_name:
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


                plt.figure(figsize=(20, 10))

                ax = plt.subplot(121)
                # Plot ground truth with distinct gray color, visible in legend
                for i in range(10):
                    color = 'gray' if i == 0 else None  # Only label first
                    if ablation_ratio > 0:
                        label = f'true ablation {ablation_ratio}' if i == 0 else None
                    elif inactivity_ratio > 0:
                        label = f'true inactivity {inactivity_ratio}' if i == 0 else None
                    elif permutation_ratio > 0:
                        label = f'true permutation {permutation_ratio}' if i == 0 else None
                    else:
                        label = 'true' if i == 0 else None
                    plt.plot(neuron_generated_list_[:, n[i]].detach().cpu().numpy(), 
                            c='gray', linewidth=8, alpha=0.5, label=label)
                # Plot predictions with colored lines
                colors = plt.cm.tab10(np.linspace(0, 1, 10))
                for i in range(10):
                    label = 'learned' if i == 0 else None
                    plt.plot(neuron_pred_list_[:, n[i]].detach().cpu().numpy(), 
                            linewidth=3, c=colors[i], label=label)
                plt.legend(fontsize=24)
                plt.xlim([0, 800])
                plt.xlabel('time-points', fontsize=48)
                plt.ylabel('$x_i$', fontsize=48)
                plt.xticks(fontsize=24)
                plt.yticks(fontsize=24)
                plt.ylim([-30, 30])

                ax = plt.subplot(122)
                plt.scatter(to_numpy(neuron_generated_list_[-1, :]), 
                        to_numpy(neuron_pred_list_[-1, :]), s=10, c=mc)
                plt.xlim([-30, 30])
                plt.ylim([-30, 30])
                plt.xticks(fontsize=24)
                plt.yticks(fontsize=24)

                x_data = to_numpy(neuron_generated_list_[-1, :])
                y_data = to_numpy(neuron_pred_list_[-1, :])
                lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                residuals = y_data - linear_model(x_data, *lin_fit)
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)

                plt.xlabel('true $x_i$', fontsize=48)
                plt.ylabel('learned $x_i$', fontsize=48)
                plt.text(-28.5, 26, f'$R^2$: {np.round(r_squared, 3)}', fontsize=34)
                plt.text(-28.5, 22, f'slope: {np.round(lin_fit[0], 2)}', fontsize=34)


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
                print(f'saved figure ./log/{log_dir}/results/{filename}')

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


    if 'inference' in test_mode:
        torch.save(x_inference_list, f"./{log_dir}/x_inference_list_{run}.pt")

    print('average rollout RMSE {:.3e}+/-{:.3e}'.format(np.mean(rmserr_list), np.std(rmserr_list)))

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

    pde = PDE_N9(p=p, f=torch.nn.functional.relu, params=simulation_config.params, model_type=model_config.signal_model_name, n_neuron_types=n_neuron_types, device=device)
    
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

    target_frames = 200
    step = 10
    dataset_length = len(stimulus_dataset)
    frames_per_sequence = 35
    total_frames_per_pass = dataset_length * frames_per_sequence
    num_passes_needed = (target_frames // total_frames_per_pass) + 1

    y_list = []
    x_list = []
    x_generated_list = []
    rmserr_list = []

    x_generated = x.clone()

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

    if 'test_ablation' in test_mode:
        #  test_mode="test_ablation_100"
        ablation_ratio = int(test_mode.split('_')[-1]) / 100
        print(f'\033[93mtest ablation ratio {ablation_ratio} \033[0m')
        n_ablation = int(edges.shape[1] * ablation_ratio)
        index_ablation = np.random.choice(np.arange(edges.shape[1]), n_ablation, replace=False)

        with torch.no_grad():
            pde.p['w'][index_ablation] = 0
            model.W[index_ablation] = 0


    with torch.no_grad():
        for pass_num in range(num_passes_needed):
            for data_idx, data in enumerate(tqdm(stimulus_dataset, desc="processing stimulus data")):

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


                    dataset = pyg.data.Data(x=x, pos=x, edge_index=edge_index)
                    data_id = torch.zeros((x.shape[0], 1), dtype=torch.int, device=device)
                    y = model(dataset, data_id, mask, False)

                    x_generated_list.append(to_numpy(x_generated.clone().detach()))
                    x_list.append(to_numpy(x.clone().detach()))

                    x_generated[:, 3:4] = x_generated[:, 3:4] + delta_t * y_generated
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

                    if (it>0) & (it % step == 0):
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


    print('generating lossless video ...')

    config_indices = dataset_name.split('fly_N9_')[1] if 'fly_N9_' in dataset_name else 'no_id'
    src = f"./{log_dir}/tmp_recons/Fig_0_000000.png"
    dst = f"./{log_dir}/results/input_{config_indices}.png"
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
        fdst.write(fsrc.read())

    # generate_lossless_video_ffv1(output_dir=f"./{log_dir}/results", run=run, config_indices=config_indices,framerate=20)
    # generate_lossless_video_libx264(output_dir=f"./{log_dir}/results", run=run,
    #                                 config_indices=config_indices,framerate=20)
    # generate_compressed_video_mp4(output_dir=f"./{log_dir}/results", run=run,
    #                                 config_indices=config_indices,framerate=20)

    # files = glob.glob(f'./{log_dir}/tmp_recons/*')
    # for f in files:
    #     os.remove(f)

    x_list = np.array(x_list)
    x_generated_list = np.array(x_generated_list)
    y_list = np.array(y_list)

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
        plt.savefig(f"./{log_dir}/results/activity_voltage_calcium.tif", dpi=300)
        plt.close()
    else:
        # Original 2-row figure (input + voltage)
        plt.figure(figsize=(16, 8))

        # Left subplot: visual input
        plt.subplot(1, 2, 1)
        plt.title("input to visual neurons", fontsize=24)
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
        plt.title("voltage activity of neurons (x10)", fontsize=24)
        neurons_to_plot = [2602, 3175, 12915, 10391, 13120, 9939, 12463, 3758, 10341, 4293]
        for i in neurons_to_plot:
            plt.plot(to_numpy(activity[i, :]), linewidth=1)
        plt.xlabel("time", fontsize=24)
        plt.ylabel("$v_i$", fontsize=24)
        plt.xlim([0, target_frames])
        plt.ylim([-3, 3])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/activity_voltage.tif", dpi=300)
        plt.close()

    # Add this code after the existing activity plots (around line 800)

    # 8x8 panel plot for each neuron type - comparing ground truth vs predicted
    print('plot 8x8 panel for neuron types (ground truth vs predicted)...')

    fig, axes = plt.subplots(8, 8, figsize=(36, 24))
    axes_flat = axes.flatten()

    panel_order = [23, 24, 25, 26, 27, 28, 29, 30, 5, 6, 7, 8, 9, 10, 11, 12, 19, 20, 21,
                22, 13, 14, 15, 16, 17, 18, 43, 45, 48, 50, 44, 46, 47, 49, 51, 52, 53, 54,
                55, 61, 62, 63, 56, 57, 58, 59, 60, 64, 1, 2, 4, 3, 31, 32, 33, 34, 35, 36,
                37, 38, 39, 40, 41, 42, 0]

    # Convert lists to numpy arrays for easier indexing
    x_generated_np = np.array(x_generated_list)  # Ground truth
    x_list_np = np.array(x_list)  # Predicted

    # Get neuron types from data
    neuron_types = x_list[-1, :, 6].astype(int)

    # Extract activity data
    if calcium_type != "none":
        # Use calcium (index 7)
        activity_true = x_generated_list[:, :, 7].squeeze().T  # (n_neurons, n_frames)
        activity_pred = x_list[:, :, 7].squeeze().T
        activity_label = "Calcium"
        y_lim = [0, 3]
    else:
        # Use voltage (index 3)
        activity_true = x_generated_list[:, :, 3].squeeze().T  # (n_neurons, n_frames)
        activity_pred = x_list[:, :, 3].squeeze().T
        activity_label = "Voltage"
        y_lim = [-3, 3]

    # Set random seed for reproducible neuron selection
    np.random.seed(42)

    for idx, type_idx in enumerate(panel_order):
        if idx >= 64:  # Only 64 panels in 8x8
            break
        
        ax = axes_flat[idx]
        
        # Find all neurons of this type
        type_mask = neuron_types == type_idx
        neuron_indices = np.where(type_mask)[0]
        
        if len(neuron_indices) > 0:
            # Randomly select one neuron of this type for clarity
            selected_neuron = np.random.choice(neuron_indices, 1)[0]
            
            # Get data for this neuron
            true_data = activity_true[selected_neuron, :]
            pred_data = activity_pred[selected_neuron, :]
            
            # Calculate dynamic y-limits with padding
            y_min = min(np.min(true_data), np.min(pred_data))
            y_max = max(np.max(true_data), np.max(pred_data))
            y_range = y_max - y_min
            y_padding = y_range * 0.1  # 10% padding
            
            # Plot ground truth (green line, thicker)
            ax.plot(true_data, linewidth=4, color='green', alpha=0.9)
            
            # Plot predicted (colored line)
            ax.plot(pred_data, linewidth=1, color=mc, alpha=1.0)
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean((true_data - pred_data)**2))
            
            # Add type name in top left corner
            type_name = index_to_name.get(type_idx, f'Type_{type_idx}')
            ax.text(0.05, 0.95, type_name, transform=ax.transAxes, 
                    ha='left', va='top', fontsize=14, color=mc)
            
            # Add RMSE below type name
            ax.text(0.05, 0.85, f'RMSE: {rmse:.3f}', transform=ax.transAxes, 
                    ha='left', va='top', fontsize=12, color=mc)
            
            # Set axis limits with dynamic range
            ax.set_xlim([0, min(target_frames, activity_pred.shape[1])])
            ax.set_ylim([y_min - y_padding, y_max + y_padding])
            
            # Add y-axis ticks for all panels
            ax.set_yticks([y_min, (y_min+y_max)/2, y_max])
            ax.set_yticklabels([f'{y_min:.2f}', f'{(y_min+y_max)/2:.2f}', f'{y_max:.2f}'], fontsize=14)
            
        else:
            # No neurons of this type
            type_name = index_to_name.get(type_idx, f"Type_{type_idx}")
            ax.text(0.5, 0.5, f'{type_name}\n(n=0)', 
                    transform=ax.transAxes, ha='center', va='center', color='gray', fontsize=8)
            # Add y-axis ticks even for empty panels for consistency
            ax.set_yticks([])
        
        # Remove x-ticks for cleaner look
        ax.set_xticks([])
        
        if idx == 56:  # Bottom left corner - add axis labels with much larger font
            ax.set_xlabel('Time (frames)', fontsize=16)
            ax.set_ylabel(f'{activity_label}', fontsize=16)
            n_ticks = min(target_frames, activity_pred.shape[1])
            ax.set_xticks([0, n_ticks//2, n_ticks])
            ax.set_xticklabels(['0', f'{n_ticks//2}', f'{n_ticks}'], fontsize=14)
            # Use dynamic y-range for this panel too
            true_data_panel = activity_true[selected_neuron, :]
            pred_data_panel = activity_pred[selected_neuron, :]
            y_min_panel = min(np.min(true_data_panel), np.min(pred_data_panel))
            y_max_panel = max(np.max(true_data_panel), np.max(pred_data_panel))
            ax.set_yticks([y_min_panel, (y_min_panel+y_max_panel)/2, y_max_panel])
            ax.set_yticklabels([f'{y_min_panel:.2f}', f'{(y_min_panel+y_max_panel)/2:.2f}', f'{y_max_panel:.2f}'], fontsize=14)

    # Hide remaining panels if any
    for idx in range(64, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/activity_8x8_panel_comparison.png", dpi=150)
    plt.close()


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

    os.makedirs(f"./{log_dir}/results/Fig/", exist_ok=True)
    generated_x_list = []
    it_idx = 0
    for it in trange(0, min(n_frames,7800), 1):
        x = torch.tensor(x_list[run][it], dtype=torch.float32, device=device)
        with torch.no_grad():
            in_features = torch.cat((x[:,1:4], it/n_frames * ones), 1)
            neural_field_list = []
            for start in range(0, in_features.shape[0], plot_batch_size):
                end = min(start + plot_batch_size, in_features.shape[0])
                batch = in_features[start:end]
                neural_field_list.append(model.NNR_f(batch)**2)
            neural_field = torch.cat(neural_field_list, dim=0)
            generated_x_list.append(to_numpy(neural_field.clone().detach()))
            if it % step == 0:
                # plot field comparison
                plot_field_comparison(x, model, it, n_frames, ones, f"./{log_dir}/results/Fig/Fig_{run}_{it_idx:06d}.png", 50, plot_batch_size)
                it_idx += 1

    generated_x_list = np.array(generated_x_list)    
    print(f"generated {len(generated_x_list)} frames total")
    print(f"saving ./{log_dir}/results/recons_field.npy")
    np.save(f"./{log_dir}/results/recons_field.npy", generated_x_list)

    print('save video...')
    src = f"./{log_dir}/results/Fig/Fig_0_000000.png"
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
    files = glob.glob(f'./{log_dir}/results/Fig/*')
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

    print(f"Grand average MAE: {mean_mae[-1]:.4f} ± {sem_mae[-1]:.4f} (N={counts[-1]})")



