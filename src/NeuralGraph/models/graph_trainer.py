import os
import time
import glob
import warnings
import logging

# Suppress matplotlib/PDF warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*Glyph.*')
warnings.filterwarnings('ignore', message='.*Missing.*')

# Suppress fontTools logging (PDF font subsetting messages)
logging.getLogger('fontTools').setLevel(logging.ERROR)
logging.getLogger('fontTools.subset').setLevel(logging.ERROR)

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.animation import FFMpegWriter
import matplotlib as mpl
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
import tifffile
import numpy as np
from torch_geometric.utils import dense_to_sparse

from NeuralGraph.models.utils import (
    choose_training_model,
    choose_inr_model,
    increasing_batch_size,
    constant_batch_size,
    set_trainable_parameters,
    get_in_features_update,
    analyze_edge_function,
    plot_training_signal,
    plot_training_signal_field,
    plot_training_signal_missing_activity,
    plot_training_flyvis,
    plot_weight_comparison,
    get_index_particles,
    analyze_data_svd,
)
from NeuralGraph.utils import (
    to_numpy,
    CustomColorMap,
    create_log_dir,
    check_and_clear_memory,
    sort_key,
    fig_init,
    get_equidistant_points,
    open_gcs_zarr,
    compute_trace_metrics,
    get_datavis_root_dir,
    LossRegularizer,
)
from NeuralGraph.models.Siren_Network import Siren, Siren_Network
from NeuralGraph.models.LowRank_INR import LowRankINR
from NeuralGraph.models.Signal_Propagation_FlyVis import Signal_Propagation_FlyVis
from NeuralGraph.models.Signal_Propagation_MLP import Signal_Propagation_MLP
from NeuralGraph.models.Signal_Propagation_MLP_ODE import Signal_Propagation_MLP_ODE
from NeuralGraph.models.Signal_Propagation_Zebra import Signal_Propagation_Zebra
from NeuralGraph.models.Neural_ode_wrapper_FlyVis import (
    integrate_neural_ode_FlyVis, neural_ode_loss_FlyVis,
    debug_check_gradients, DEBUG_ODE
)
from NeuralGraph.models.Neural_ode_wrapper_Signal import integrate_neural_ode_Signal, neural_ode_loss_Signal
from NeuralGraph.models.Signal_Propagation_Temporal import Signal_Propagation_Temporal
from NeuralGraph.models.Signal_Propagation_RNN import Signal_Propagation_RNN
from NeuralGraph.models.Signal_Propagation_LSTM import Signal_Propagation_LSTM
from NeuralGraph.models.utils_zebra import (
    plot_field_comparison,
    plot_field_comparison_continuous_slices,
    plot_field_comparison_discrete_slices,
    plot_field_discrete_xy_slices_grid,
)

from NeuralGraph.models.HashEncoding_Network import HashEncodingMLP, create_hash_encoding_mlp, TCNN_AVAILABLE

from NeuralGraph.sparsify import EmbeddingCluster, sparsify_cluster, clustering_evaluation
from NeuralGraph.fitting_models import linear_model

from scipy.optimize import curve_fit

from torch_geometric.data import Data as pyg_Data
from torch_geometric.loader import DataLoader
import torch_geometric as pyg
import seaborn as sns
# denoise_data import not needed - removed star import
from tifffile import imread
from matplotlib.colors import LinearSegmentedColormap
from NeuralGraph.generators.utils import choose_model, plot_signal_loss, generate_compressed_video_mp4, init_connectivity
from NeuralGraph.generators.graph_data_generator import (
    apply_pairwise_knobs_torch,
    assign_columns_from_uv,
    build_neighbor_graph,
    compute_column_labels,
    greedy_blue_mask,
    mseq_bits,
)
from NeuralGraph.generators.davis import AugmentedDavis
import pandas as pd
import napari
from collections import deque
from tqdm import tqdm, trange
from prettytable import PrettyTable
import imageio


def data_train(config=None, erase=False, best_model=None, style=None, device=None):
    # plt.rcParams['text.usetex'] = False  # LaTeX disabled - use mathtext instead
    # rc('font', **{'family': 'serif', 'serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif', 'serif']})
    # matplotlib.rcParams['savefig.pad_inches'] = 0

    seed = config.training.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # torch.autograd.set_detect_anomaly(True)

    dataset_name = config.dataset
    print(f"\033[94mdataset_name: {dataset_name}\033[0m")
    print(f"\033[92m{config.description}\033[0m")

    if 'fly' in config.dataset:
        if 'RNN' in config.graph_model.signal_model_name or 'LSTM' in config.graph_model.signal_model_name:
            data_train_flyvis_RNN(config, erase, best_model, device)
        else:
            data_train_flyvis(config, erase, best_model, device)
    elif 'zebra_fluo' in config.dataset:
        data_train_zebra_fluo(config, erase, best_model, device)
    elif 'zebra' in config.dataset:
        data_train_zebra(config, erase, best_model, device)
    else:
        data_train_signal(config, erase, best_model, style, device)

    print("training completed.")




def data_train_signal(config, erase, best_model, style, device):

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
    recurrent_training = train_config.recurrent_training
    noise_recurrent_level = train_config.noise_recurrent_level
    recurrent_parameters = train_config.recurrent_parameters.copy()
    neural_ODE_training = train_config.neural_ODE_training
    ode_method = train_config.ode_method
    ode_rtol = train_config.ode_rtol
    ode_atol = train_config.ode_atol
    ode_adjoint = train_config.ode_adjoint
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

    # external input configuration (hierarchy: visual > signal > none)
    external_input_type = simulation_config.external_input_type
    external_input_mode = simulation_config.external_input_mode
    signal_input_type = simulation_config.signal_input_type
    learn_external_input = train_config.learn_external_input
    inr_type = model_config.inr_type

    embedding_cluster = EmbeddingCluster(config)

    time_step = train_config.time_step
    has_missing_activity = train_config.has_missing_activity
    multi_connectivity = config.training.multi_connectivity
    baseline_value = simulation_config.baseline_value
    cmap = CustomColorMap(config=config)

    if "black" in style:
        plt.style.use("dark_background")
        mc = 'white'
    else:
        plt.style.use("default")
        mc = 'black'

    if config.training.seed != 42:
        torch.random.fork_rng(devices=device)
        torch.random.manual_seed(config.training.seed)
        np.random.seed(config.training.seed)

    log_dir, logger = create_log_dir(config, erase)
    print(f'loading data...')

    x_list = []
    y_list = []
    for run in trange(0,n_runs, ncols=80):
        x = np.load(f'graphs_data/{dataset_name}/x_list_{run}.npy')
        y = np.load(f'graphs_data/{dataset_name}/y_list_{run}.npy')
        x_list.append(x)
        y_list.append(y)

    run = 0
    x = x_list[0][n_frames - 10]
    n_neurons = x.shape[0]
    config.simulation.n_neurons =n_neurons
    type_list = torch.tensor(x[:, 6:7], device=device)  # neuron_type is at column 6

    activity = torch.tensor(x_list[0][:, :, 3], device=device)  # signal state is at column 3
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

    # SVD analysis of activity and external_input (skip if already exists)
    # svd_plot_path = os.path.join(log_dir, 'results', 'svd_analysis.png')
    # if not os.path.exists(svd_plot_path):
    #     analyze_data_svd(x_list[0], log_dir, config=config, logger=logger)
    # else:
    #     print(f'svd analysis already exists: {svd_plot_path}')

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
    # external input model (model_f) for learning external_input reconstruction
    model_f = choose_inr_model(config=config, n_neurons=n_neurons, n_frames=n_frames, x_list=x_list, device=device)
    optimizer_f = None
    if model_f is not None:
        # Separate omega parameters from other parameters for different learning rates
        omega_params = [(name, p) for name, p in model_f.named_parameters() if 'omega' in name]
        other_params = [p for name, p in model_f.named_parameters() if 'omega' not in name]
        if omega_params:
            print(f"model_f omega parameters found: {[name for name, p in omega_params]}")
            optimizer_f = torch.optim.Adam([
                {'params': other_params, 'lr': train_config.learning_rate_NNR_f},
                {'params': [p for name, p in omega_params], 'lr': train_config.learning_rate_omega_f}
            ])
        else:
            print("model_f: no omega parameters found (omega_f_learning=False or non-SIREN model)")
            optimizer_f = torch.optim.Adam(lr=train_config.learning_rate_NNR_f, params=model_f.parameters())
        model_f.train()
        # Print initial omega values if learnable
        if hasattr(model_f, 'get_omegas'):
            omegas = model_f.get_omegas()
            if omegas:
                print(f"model_f initial omegas: {omegas}")
                print(f"model_f omega LR: {train_config.learning_rate_omega_f}, L2 coeff: {train_config.coeff_omega_f_L2}")

    if (best_model != None) & (best_model != '') & (best_model != 'None'):
        net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs_{best_model}.pt"
        print(f'load {net} ...')
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        start_epoch = int(best_model.split('_')[0])
        print(f'best_model: {best_model}  start_epoch: {start_epoch}')
        logger.info(f'best_model: {best_model}  start_epoch: {start_epoch}')
        if model_f is not None:
            net = f'{log_dir}/models/best_model_f_with_{n_runs - 1}_graphs_{best_model}.pt'
            state_dict = torch.load(net, map_location=device)
            model_f.load_state_dict(state_dict['model_state_dict'])
        list_loss = torch.load(os.path.join(log_dir, 'loss.pt'))
    else:
        start_epoch = 0
        list_loss = []

    loss_components = {'loss': []}  # regularizer handles other components

    print('set optimizer ...')
    lr = train_config.learning_rate_start
    lr_update = lr
    if train_config.init_training_single_type:
        lr_embedding = 1.0E-16
    else:
        lr_embedding = train_config.learning_rate_embedding_start
    lr_W = train_config.learning_rate_W_start
    lr_modulation = train_config.learning_rate_modulation_start

    learning_rate_NNR = train_config.learning_rate_NNR
    learning_rate_NNR_f = train_config.learning_rate_NNR_f

    optimizer, n_total_params = set_trainable_parameters(model=model, lr_embedding=lr_embedding, lr=lr, lr_update=lr_update, lr_W=lr_W, learning_rate_NNR=learning_rate_NNR, learning_rate_NNR_f = learning_rate_NNR_f)
    model.train()

    print(f'learning rates: lr_W {lr_W}, lr {lr}, lr_update {lr_update}, lr_embedding {lr_embedding}, lr_modulation {lr_modulation}')
    logger.info(f'learning rates: lr_W {lr_W}, lr {lr}, lr_update {lr_update}, lr_embedding {lr_embedding}, lr_modulation {lr_modulation}')

    net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs.pt"
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

    print(f'n neurons: {n_neurons}, edges:{edges.shape[1]}, xnorm: {to_numpy(xnorm)}, vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')
    logger.info(f'n neurons: {n_neurons}, edges:{edges.shape[1]}, xnorm: {to_numpy(xnorm)}, vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')


     # PDE_N3 is special, embedding changes over time
    if 'PDE_N3' in model_config.signal_model_name:
        ind_a = torch.tensor(np.arange(1, n_neurons * 100), device=device)
        pos = torch.argwhere(ind_a % 100 != 99).squeeze()
        ind_a = ind_a[pos]

    # initialize regularizer
    regularizer = LossRegularizer(
        train_config=train_config,
        model_config=model_config,
        activity_column=3,  # signal uses column 6
        plot_frequency=1,   # will be updated per epoch
        n_neurons=n_neurons,
        trainer_type='signal'
    )

    print("start training ...")

    check_and_clear_memory(device=device, iteration_number=0, every_n_iterations=1, memory_percentage_threshold=0.6)
    # torch.autograd.set_detect_anomaly(True)

    list_loss_regul = []
    time.sleep(1.0)

    for epoch in range(start_epoch, n_epochs + 1):

        if (epoch == train_config.epoch_reset):
            with torch.no_grad():
                model.W.copy_(model.W * 0)
                model.a.copy_(model.a * 0)
            logger.info(f'reset W model.a at epoch : {epoch}')
            print(f'reset W model.a at epoch : {epoch}')
        if (epoch == 1) & (train_config.init_training_single_type):
            lr_embedding = train_config.learning_rate_embedding_start
            optimizer, n_total_params = set_trainable_parameters(model=model, lr_embedding=lr_embedding, lr=lr, lr_update=lr_update, lr_W=lr_W, learning_rate_NNR=learning_rate_NNR, learning_rate_NNR_f = learning_rate_NNR_f)
            model.train()


        batch_size = get_batch_size(epoch)
        logger.info(f'batch_size: {batch_size}')

        if batch_ratio < 1:
            Niter = int(n_frames * data_augmentation_loop // batch_size / batch_ratio * 0.2)
        else:
            Niter = int(n_frames * data_augmentation_loop // batch_size * 0.2 )

        plot_frequency = int(Niter // 4)
        if epoch ==0:
            print(f'{Niter} iterations per epoch, {plot_frequency} iterations per plot')
            logger.info(f'{Niter} iterations per epoch')

        regularizer.set_epoch(epoch, plot_frequency)

        total_loss = 0
        total_loss_regul = 0
        run = 0


        time.sleep(1.0)
        for N in trange(Niter, ncols=150):

            if has_missing_activity:
                optimizer_missing_activity.zero_grad()
            if model_f is not None:
                optimizer_f.zero_grad()
            optimizer.zero_grad()

            dataset_batch = []
            ids_batch = []
            ids_index = 0

            loss = torch.zeros(1, device=device)

            regularizer.reset_iteration()

            for batch in range(batch_size):

                k = np.random.randint(n_frames - 4 - time_step)

                if recurrent_training or neural_ODE_training:
                    k = k - k % time_step

                x = torch.tensor(x_list[run][k], dtype=torch.float32, device=device)

                ids = np.arange(n_neurons)

                if not (torch.isnan(x).any()):

                    # special case regularizations (kept outside LossRegularizer)
                    if has_missing_activity:
                        t = torch.tensor([k / n_frames], dtype=torch.float32, device=device)
                        missing_activity = baseline_value + model_missing_activity[run](t).squeeze()
                        if (train_config.coeff_missing_activity>0):
                            loss_missing_activity = (missing_activity[ids] - x[ids, 3].clone().detach()).norm(2)
                            regul_term = loss_missing_activity * train_config.coeff_missing_activity
                            loss = loss + regul_term
                        ids_missing = torch.argwhere(x[:, 3] == baseline_value)
                        x[ids_missing,3] = missing_activity[ids_missing]
                    # external input reconstruction (when learn_external_input=True)
                    if model_f is not None:
                        nnr_f_T_period = model_config.nnr_f_T_period
                        if external_input_type == 'visual':
                            n_input_neurons = simulation_config.n_input_neurons
                            x[:n_input_neurons, 4:5] = model_f(time=k / n_frames) ** 2
                            x[n_input_neurons:n_neurons, 4:5] = 1
                        elif external_input_type == 'signal':
                            t_norm = torch.tensor([[k / nnr_f_T_period]], dtype=torch.float32, device=device)
                            if inr_type == 'siren_t':
                                x[:, 4] = model_f(t_norm).squeeze()
                            elif inr_type == 'lowrank':
                                t_idx = torch.tensor([k], dtype=torch.long, device=device)
                                x[:, 4] = model_f(t_idx).squeeze()
                            elif inr_type == 'ngp':
                                x[:, 4] = model_f(t_norm).squeeze()
                            # siren_id and siren_x would need position/id info - not implemented in training loop yet

                    regul_loss = regularizer.compute(
                        model=model,
                        x=x,
                        in_features=None,
                        ids=ids,
                        ids_batch=None,
                        edges=edges,
                        device=device,
                        xnorm=xnorm,
                        index_weight=index_weight if train_config.coeff_W_sign > 0 else None
                    )

                    loss = loss + regul_loss

                    if batch_ratio < 1:
                        ids_ = np.random.permutation(ids.shape[0])[:int(ids.shape[0] * batch_ratio)]
                        ids = np.sort(ids_)
                        edges = edges_all.clone().detach()
                        mask = torch.isin(edges[1, :], torch.tensor(ids, device=device))
                        edges = edges[:, mask]

                    if recurrent_training or neural_ODE_training:
                        y = torch.tensor(x_list[run][k + time_step, :, 3:4], dtype=torch.float32, device=device).detach()
                    else:
                        y = torch.tensor(y_list[run][k], device=device) / ynorm


                    if not (torch.isnan(y).any()):

                        dataset = pyg_Data(x=x, edge_index=edges)
                        dataset_batch.append(dataset)

                        if len(dataset_batch) == 1:
                            data_id = torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * run
                            x_batch = x[:, 3:4]
                            y_batch = y
                            k_batch = torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * k
                            ids_batch = ids
                        else:
                            data_id = torch.cat((data_id, torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * run), dim=0)
                            x_batch = torch.cat((x_batch, x[:, 3:4]), dim=0)
                            y_batch = torch.cat((y_batch, y), dim=0)
                            k_batch = torch.cat(
                                (k_batch, torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * k), dim=0)
                            ids_batch = np.concatenate((ids_batch, ids + ids_index), axis=0)

                        ids_index += x.shape[0]

            if not (dataset_batch == []):

                batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
                for batch in batch_loader:
                    if regularizer.needs_update_regul():
                        pred, in_features = model(batch, data_id=data_id, k=k_batch, return_all=True)
                        update_regul_loss = regularizer.compute_update_regul(
                            model=model,
                            in_features=in_features,
                            ids_batch=ids_batch,
                            device=device,
                            x=x,
                            xnorm=xnorm,
                            ids=ids
                        )
                        loss = loss + update_regul_loss
                    else:
                        pred = model(batch, data_id=data_id, k=k_batch)

                if neural_ODE_training:
                    ode_loss, pred_x = neural_ode_loss_Signal(
                        model=model,
                        dataset_batch=dataset_batch,
                        x_list=x_list,
                        run=run,
                        k_batch=k_batch,
                        time_step=time_step,
                        batch_size=batch_size,
                        n_neurons=n_neurons,
                        ids_batch=ids_batch,
                        delta_t=delta_t,
                        device=device,
                        data_id=data_id,
                        y_batch=y_batch,
                        noise_level=noise_recurrent_level,
                        ode_method=ode_method,
                        rtol=ode_rtol,
                        atol=ode_atol,
                        adjoint=ode_adjoint
                    )
                    loss = loss + ode_loss

                elif recurrent_training:

                    pred_x = x_batch + delta_t * pred + noise_recurrent_level * torch.randn_like(pred)

                    if time_step > 1:
                        for step in range(1, time_step):
                            dataset_batch_new = []
                            for b in range(batch_size):
                                start_idx = b * n_neurons
                                end_idx = (b + 1) * n_neurons
                                dataset_batch[b].x[:, 3:4] = pred_x[start_idx:end_idx].reshape(-1, 1)

                                # update external_input for next time step during rollout
                                k_current = k_batch[start_idx, 0].item() + step
                                if model_f is not None:
                                    nnr_f_T_period = model_config.nnr_f_T_period
                                    if external_input_type == 'visual':
                                        n_input_neurons = simulation_config.n_input_neurons
                                        dataset_batch[b].x[:n_input_neurons, 4:5] = model_f(time=k_current / n_frames) ** 2
                                        dataset_batch[b].x[n_input_neurons:n_neurons, 4:5] = 1
                                    elif external_input_type == 'signal':
                                        t_norm = torch.tensor([[k_current / nnr_f_T_period]], dtype=torch.float32, device=device)
                                        if inr_type == 'siren_t':
                                            dataset_batch[b].x[:, 4] = model_f(t_norm).squeeze()
                                        elif inr_type == 'lowrank':
                                            t_idx = torch.tensor([k_current], dtype=torch.long, device=device)
                                            dataset_batch[b].x[:, 4] = model_f(t_idx).squeeze()
                                        elif inr_type == 'ngp':
                                            dataset_batch[b].x[:, 4] = model_f(t_norm).squeeze()
                                else:
                                    # use ground truth external_input from x_list
                                    x_next = torch.tensor(x_list[run][k_current], dtype=torch.float32, device=device)
                                    dataset_batch[b].x[:, 4:5] = x_next[:, 4:5]

                                dataset_batch_new.append(dataset_batch[b])
                            batch_loader_recur = DataLoader(dataset_batch_new, batch_size=batch_size, shuffle=False)
                            for batch in batch_loader_recur:
                                pred = model(batch, data_id=data_id, k=k_batch + step)
                            pred_x = pred_x + delta_t * pred + noise_recurrent_level * torch.randn_like(pred)

                    loss = loss + ((pred_x[ids_batch] - y_batch[ids_batch]) / (delta_t * time_step)).norm(2)

                else:

                    loss = loss + (pred[ids_batch] - y_batch[ids_batch]).norm(2)



                 # PDE_N3 is special, embedding changes over time
                
                if 'PDE_N3' in model_config.signal_model_name:
                    loss = loss + train_config.coeff_model_a * (model.a[ind_a + 1] - model.a[ind_a]).norm(2)

                # omega L2 regularization for learnable omega in SIREN (encourages smaller omega)
                if model_f is not None and train_config.coeff_omega_f_L2 > 0:
                    if hasattr(model_f, 'get_omega_L2_loss'):
                        omega_L2_loss = model_f.get_omega_L2_loss()
                        loss = loss + train_config.coeff_omega_f_L2 * omega_L2_loss

                loss.backward()
                optimizer.step()
                regularizer.finalize_iteration()

                if has_missing_activity:
                    optimizer_missing_activity.step()
                if model_f is not None:
                    optimizer_f.step()

                regul_total_this_iter = regularizer.get_iteration_total()
                total_loss += loss.item()
                total_loss_regul += regul_total_this_iter

                if regularizer.should_record():
                    # store in dictionary lists
                    current_loss = loss.item()
                    loss_components['loss'].append((current_loss - regul_total_this_iter) / n_neurons)

                    plot_training_signal(config, model, x, connectivity, log_dir, epoch, N, n_neurons, type_list, cmap, mc, device)

                    # merge loss_components with regularizer history for plotting
                    plot_dict = {**regularizer.get_history(), **loss_components}
                    plot_signal_loss(plot_dict, log_dir, epoch=epoch, Niter=N, debug=False,
                                   current_loss=current_loss / n_neurons, current_regul=regul_total_this_iter / n_neurons,
                                   total_loss=total_loss, total_loss_regul=total_loss_regul)

                    if model_f is not None:
                        torch.save({'model_state_dict': model_f.state_dict(),
                                    'optimizer_state_dict': optimizer_f.state_dict()},
                                   os.path.join(log_dir, 'models',
                                                f'best_model_f_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))
                        # plot external_input learned vs ground truth (like train_INR)
                        with torch.no_grad():
                            external_input_gt = x_list[0][:, :, 4]  # (n_frames, n_neurons)
                            nnr_f_T_period = model_config.nnr_f_T_period
                            if inr_type == 'siren_t':
                                time_input = torch.arange(0, n_frames, dtype=torch.float32, device=device).unsqueeze(1) / nnr_f_T_period
                                pred_all = model_f(time_input).cpu().numpy()
                            elif inr_type == 'lowrank':
                                pred_all = model_f().cpu().numpy()
                            elif inr_type == 'ngp':
                                time_input = torch.arange(0, n_frames, dtype=torch.float32, device=device).unsqueeze(1) / nnr_f_T_period
                                pred_all = model_f(time_input).cpu().numpy()
                            else:
                                pred_all = None
                            if pred_all is not None:
                                gt_np = external_input_gt[:n_frames]  # ensure same length as pred
                                pred_np = pred_all
                                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                                fig.patch.set_facecolor('black')
                                ax.set_facecolor('black')
                                ax.set_axis_off()
                                n_traces = 10
                                trace_ids = np.linspace(0, n_neurons - 1, n_traces, dtype=int)
                                offset = np.abs(gt_np).max() * 1.5
                                t = np.arange(gt_np.shape[0])
                                for j, n_idx in enumerate(trace_ids):
                                    y0 = j * offset
                                    ax.plot(t, gt_np[:, n_idx] + y0, color='darkgreen', lw=2.0, alpha=0.95)
                                    ax.plot(t, pred_np[:, n_idx] + y0, color='white', lw=0.5, alpha=0.95)
                                ax.set_xlim(0, min(20000, gt_np.shape[0]))
                                ax.set_ylim(-offset * 0.5, offset * (n_traces + 0.5))
                                mse = ((pred_np - gt_np) ** 2).mean()
                                omega_str = ''
                                if hasattr(model_f, 'get_omegas'):
                                    omegas = model_f.get_omegas()
                                    if omegas:
                                        omega_str = f'  ω: {omegas[0]:.1f}'
                                ax.text(0.02, 0.98, f'MSE: {mse:.6f}{omega_str}', transform=ax.transAxes, va='top', ha='left', fontsize=12, color='white')
                                out_dir = os.path.join(log_dir, 'tmp_training', 'external_input')
                                os.makedirs(out_dir, exist_ok=True)
                                plt.tight_layout()
                                plt.savefig(f"{out_dir}/inr_{epoch}_{N}.png", dpi=150)
                                plt.close()

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


        epoch_total_loss = total_loss / n_neurons
        epoch_regul_loss = total_loss_regul / n_neurons
        epoch_pred_loss = (total_loss - total_loss_regul) / n_neurons

        print("epoch {}. loss: {:.6f} (pred: {:.6f}, regul: {:.6f})".format(
            epoch, epoch_total_loss, epoch_pred_loss, epoch_regul_loss))
        logger.info("Epoch {}. Loss: {:.6f} (pred: {:.6f}, regul: {:.6f})".format(
            epoch, epoch_total_loss, epoch_pred_loss, epoch_regul_loss))
        logger.info(f'recurrent_parameters: {recurrent_parameters[0]:.2f}')
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}.pt'))
        if model_f is not None:
            torch.save({'model_state_dict': model_f.state_dict(),
                        'optimizer_state_dict': optimizer_f.state_dict()},
                       os.path.join(log_dir, 'models', f'best_model_f_with_{n_runs - 1}_graphs_{epoch}.pt'))
            # Print omega values at end of each epoch
            if hasattr(model_f, 'get_omegas'):
                omegas = model_f.get_omegas()
                if omegas:
                    print(f"  model_f omegas after epoch {epoch}: {omegas}")

        list_loss.append(epoch_pred_loss)
        list_loss_regul.append(epoch_regul_loss)

        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))

        fig = plt.figure(figsize=(15, 10))

        fig.add_subplot(2, 3, 1)
        plt.plot(list_loss, color=mc, linewidth=1)
        plt.xlim([0, n_epochs])
        plt.ylabel('loss', fontsize=24)
        plt.xlabel('epochs', fontsize=24)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        embedding_files = glob.glob(f"./{log_dir}/tmp_training/embedding/*.tif")
        if embedding_files:
            last_file = max(embedding_files, key=os.path.getctime)  # or use os.path.getmtime for modification time
            filename = os.path.basename(last_file)
            last_epoch, last_N = filename.replace('.tif', '').split('_')

            # Load and display last saved figures
            from tifffile import imread

            def safe_load_and_display(ax, filepath):
                """Load and display image if file exists, otherwise leave panel empty."""
                if os.path.exists(filepath):
                    img = imread(filepath)
                    ax.imshow(img)
                ax.axis('off')

            # Plot 2: Last embedding
            ax = fig.add_subplot(2, 3, 2)
            safe_load_and_display(ax, f"./{log_dir}/tmp_training/embedding/{last_epoch}_{last_N}.tif")

            # Plot 3: Last weight comparison
            ax = fig.add_subplot(2, 3, 3)
            safe_load_and_display(ax, f"./{log_dir}/tmp_training/matrix/comparison_{last_epoch}_{last_N}.tif")

            # Plot 4: Last phi function
            ax = fig.add_subplot(2, 3, 4)
            safe_load_and_display(ax, f"./{log_dir}/tmp_training/function/MLP0/func_{last_epoch}_{last_N}.tif")

            # Plot 5: Last edge function
            ax = fig.add_subplot(2, 3, 5)
            safe_load_and_display(ax, f"./{log_dir}/tmp_training/function/MLP1/func_{last_epoch}_{last_N}.tif")

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
                                                                     type_list=to_numpy(x[:, 6]),  # neuron_type is at column 6
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
                print('regul_embedding: replaced')
                logger.info('regul_embedding: replaced')

                # Constrain function domain
                if train_config.sparsity == 'replace_embedding_function':

                    logger.info('replace_embedding_function')
                    y_func_list = func_list * 0

                    fig.add_subplot(2, 5, 9)
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
                    for sub_epochs in trange(20, ncols=100):
                        rr = torch.linspace(config.plotting.xlim[0], config.plotting.xlim[1], 1000, device=device)
                        pred = []
                        optimizer.zero_grad()
                        for n in range(n_neurons):
                            embedding = model.a[n, :].clone().detach() * torch.ones((1000, model_config.embedding_dim),
                                                                                     device=device)
                            in_features = get_in_features_update(rr=rr[:, None], model=model, embedding=embedding, device=device)
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
    delta_t = simulation_config.delta_t

    dataset_name = config.dataset
    n_runs = train_config.n_runs
    n_frames = simulation_config.n_frames

    data_augmentation_loop = train_config.data_augmentation_loop
    recurrent_training = train_config.recurrent_training
    noise_recurrent_level = train_config.noise_recurrent_level
    neural_ODE_training = train_config.neural_ODE_training
    ode_method = train_config.ode_method
    ode_rtol = train_config.ode_rtol
    ode_atol = train_config.ode_atol
    ode_adjoint = train_config.ode_adjoint
    batch_size = train_config.batch_size
    time_window = train_config.time_window
    training_selected_neurons = train_config.training_selected_neurons

    field_type = model_config.field_type
    time_step = train_config.time_step

    replace_with_cluster = 'replace' in train_config.sparsity
    sparsity_freq = train_config.sparsity_freq


    if config.training.seed != 42:
        torch.random.fork_rng(devices=device)
        torch.random.manual_seed(config.training.seed)

    cmap = CustomColorMap(config=config)

    if 'visual' in field_type:
        has_visual_field = True
        if 'instantNGP' in field_type:
            print('train with visual field instantNGP')
        else:
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
    print(f'n neurons: {n_neurons}')
    logger.info(f'N neurons: {n_neurons}')
    config.simulation.n_neurons =n_neurons
    type_list = torch.tensor(x[:, 6:7], device=device)  # neuron_type is at column 6
    ynorm = torch.tensor(1.0, device=device)
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))
    time.sleep(0.5)
    print(f'ynorm: {to_numpy(ynorm)}')
    logger.info(f'ynorm: {to_numpy(ynorm)}')

    # SVD analysis of activity and visual stimuli (skip if already exists)
    svd_plot_path = os.path.join(log_dir, 'results', 'svd_analysis.png')
    if not os.path.exists(svd_plot_path):
        analyze_data_svd(x_list[0], log_dir, config=config, logger=logger, is_flyvis=True)
    else:
        print(f'svd analysis already exists: {svd_plot_path}')

    print('create models ...')
    if time_window >0:
        model = Signal_Propagation_Temporal(aggr_type=model_config.aggr_type, config=config, device=device)
    elif 'MLP_ODE' in signal_model_name:
        model = Signal_Propagation_MLP_ODE(aggr_type=model_config.aggr_type, config=config, device=device)
    elif 'MLP' in signal_model_name:
        model = Signal_Propagation_MLP(aggr_type=model_config.aggr_type, config=config, device=device)
    else:
        model = Signal_Propagation_FlyVis(aggr_type=model_config.aggr_type, config=config, device=device)


    model = model.to(device)
    n_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'total parameters: {n_total_params:,}')

    start_epoch = 0
    list_loss = []
    if (best_model != None) & (best_model != '') & (best_model != '') & (best_model != 'None'):
        net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs_{best_model}.pt"
        print(f'loading state_dict from {net} ...')
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        start_epoch = int(best_model.split('_')[0])
        print(f'state_dict loaded: best_model={best_model}, start_epoch={start_epoch}')
    elif  train_config.pretrained_model !='':
        net = train_config.pretrained_model
        print(f'loading pretrained state_dict from {net} ...')
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        print('pretrained state_dict loaded')
        logger.info(f'pretrained: {net}')
    else:
        print('no state_dict loaded - using freshly initialized model')

    lr = train_config.learning_rate_start
    if train_config.learning_rate_update_start == 0:
        lr_update = train_config.learning_rate_start
    else:
        lr_update = train_config.learning_rate_update_start
    lr_embedding = train_config.learning_rate_embedding_start
    lr_W = train_config.learning_rate_W_start
    learning_rate_NNR = train_config.learning_rate_NNR
    learning_rate_NNR_f = train_config.learning_rate_NNR_f

    print(f'learning rates: lr_W {lr_W}, lr {lr}, lr_update {lr_update}, lr_embedding {lr_embedding}, learning_rate_NNR {learning_rate_NNR}')

    optimizer, n_total_params = set_trainable_parameters(model=model, lr_embedding=lr_embedding, lr=lr,
                                                         lr_update=lr_update, lr_W=lr_W, learning_rate_NNR=learning_rate_NNR, learning_rate_NNR_f = learning_rate_NNR_f)
    model.train()

    net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs.pt"
    print(f'network: {net}')
    print(f'initial batch_size: {batch_size}')

    # connectivity = torch.load(f'./graphs_data/{dataset_name}/connectivity.pt', map_location=device)
    gt_weights = torch.load(f'./graphs_data/{dataset_name}/weights.pt', map_location=device)
    edges = torch.load(f'./graphs_data/{dataset_name}/edge_index.pt', map_location=device)
    print(f'{edges.shape[1]} edges')

    ids = np.arange(n_neurons)

    if train_config.coeff_W_sign > 0:
        index_weight = []
        for i in range(n_neurons):
            # get source neurons that connect to neuron i
            mask = edges[1] == i
            index_weight.append(edges[0][mask])

    logger.info(f'coeff_W_L1: {train_config.coeff_W_L1} coeff_edge_diff: {train_config.coeff_edge_diff} coeff_update_diff: {train_config.coeff_update_diff}')
    print(f'coeff_W_L1: {train_config.coeff_W_L1} coeff_edge_diff: {train_config.coeff_edge_diff} coeff_update_diff: {train_config.coeff_update_diff}')

    print("start training ...")

    check_and_clear_memory(device=device, iteration_number=0, every_n_iterations=1, memory_percentage_threshold=0.6)
    # torch.autograd.set_detect_anomaly(True)

    list_loss_regul = []

    regularizer = LossRegularizer(
        train_config=train_config,
        model_config=model_config,
        activity_column=3,  # flyvis uses column 3 for activity
        plot_frequency=1,   # will be updated per epoch
        n_neurons=n_neurons,
        trainer_type='flyvis'
    )

    loss_components = {'loss': []}

    time.sleep(0.2)

    for epoch in range(start_epoch, n_epochs + 1):

        Niter = int(n_frames * data_augmentation_loop // batch_size * 0.2)
        plot_frequency = int(Niter // 20)
        print(f'{Niter} iterations per epoch, plot every {plot_frequency} iterations')

        total_loss = 0
        total_loss_regul = 0
        k = 0

        loss_noise_level = train_config.loss_noise_level * (0.95 ** epoch)
        regularizer.set_epoch(epoch, plot_frequency)

        for N in trange(Niter,ncols=150):

            optimizer.zero_grad()

            dataset_batch = []
            ids_batch = []
            k_batch = []
            visual_input_batch = []
            ids_index = 0

            loss = 0
            run = np.random.randint(n_runs)

            for batch in range(batch_size):

                k = np.random.randint(n_frames - 4 - time_step - time_window) + time_window

                if recurrent_training or neural_ODE_training:
                    k = k - k % time_step

                x = torch.tensor(x_list[run][k], dtype=torch.float32, device=device)

                if time_window > 0:
                    x_temporal = x_list[run][k - time_window + 1: k + 1, :, 3:4].transpose(1, 0, 2).squeeze(-1)
                    x = torch.cat((x, torch.tensor(x_temporal.reshape(n_neurons, time_window), dtype=torch.float32, device=device)), dim=1)

                if has_visual_field:
                    visual_input = model.forward_visual(x,k)
                    x[:model.n_input_neurons, 4:5] = visual_input
                    x[model.n_input_neurons:, 4:5] = 0

                loss = torch.zeros(1, device=device)
                regularizer.reset_iteration()

                if not (torch.isnan(x).any()):
                    regul_loss = regularizer.compute(
                        model=model,
                        x=x,
                        in_features=None,  
                        ids=ids,
                        ids_batch=None,  
                        edges=edges,
                        device=device,
                        xnorm=xnorm
                    )
                    loss = loss + regul_loss

                    if recurrent_training or neural_ODE_training:
                        y = torch.tensor(x_list[run][k + time_step,:,3:4], dtype=torch.float32, device=device).detach()       # loss on next activity
                    elif test_neural_field:
                        y = torch.tensor(x_list[run][k, :n_input_neurons, 4:5], device=device)  # loss on current excitation
                    else:
                        y = torch.tensor(y_list[run][k], device=device) / ynorm     # loss on activity derivative


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
                            k_batch = torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * k
                            if test_neural_field:
                                visual_input_batch = visual_input
                        else:
                            data_id = torch.cat((data_id, torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * run), dim=0)
                            x_batch = torch.cat((x_batch, x[:, 3:5]), dim=0)
                            y_batch = torch.cat((y_batch, y), dim=0)
                            ids_batch = np.concatenate((ids_batch, ids + ids_index), axis=0)
                            k_batch = torch.cat((k_batch, torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * k), dim=0)
                            if test_neural_field:
                                visual_input_batch = torch.cat((visual_input_batch, visual_input), dim=0)

                        ids_index += x.shape[0]


            if not (dataset_batch == []):

                total_loss_regul += loss.item()


                if test_neural_field:
                    loss = loss + (visual_input_batch - y_batch).norm(2)



                elif 'MLP_ODE' in signal_model_name:
                    batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
                    for batch in batch_loader:
                        pred = model(batch.x, data_id=data_id, return_all=False)

                    loss = loss + (pred[ids_batch] - y_batch[ids_batch]).norm(2)



                elif 'MLP' in signal_model_name:
                    batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
                    for batch in batch_loader:
                        pred = model(batch.x, data_id=data_id, return_all=False)

                    loss = loss + (pred[ids_batch] - y_batch[ids_batch]).norm(2)


                else:


                    batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
                    for batch in batch_loader:
                        pred, in_features, msg = model(batch, data_id=data_id, return_all=True)

                    update_regul = regularizer.compute_update_regul(model, in_features, ids_batch, device)
                    loss = loss + update_regul


                    if neural_ODE_training:

                        ode_state_clamp = getattr(train_config, 'ode_state_clamp', 10.0)
                        ode_stab_lambda = getattr(train_config, 'ode_stab_lambda', 0.0)
                        ode_loss, pred_x = neural_ode_loss_FlyVis(
                            model=model,
                            dataset_batch=dataset_batch,
                            x_list=x_list,
                            run=run,
                            k_batch=k_batch,
                            time_step=time_step,
                            batch_size=batch_size,
                            n_neurons=n_neurons,
                            ids_batch=ids_batch,
                            delta_t=delta_t,
                            device=device,
                            data_id=data_id,
                            has_visual_field=has_visual_field,
                            y_batch=y_batch,
                            noise_level=noise_recurrent_level,
                            ode_method=ode_method,
                            rtol=ode_rtol,
                            atol=ode_atol,
                            adjoint=ode_adjoint,
                            iteration=N,
                            state_clamp=ode_state_clamp,
                            stab_lambda=ode_stab_lambda
                        )
                        loss = loss + ode_loss


                    elif recurrent_training:

                        pred_x = x_batch[:, 0:1] + delta_t * pred + noise_recurrent_level * torch.randn_like(pred)

                        if time_step > 1:
                            for step in range(time_step - 1):
                                dataset_batch_new = []
                                neurons_per_sample = dataset_batch[0].x.shape[0]

                                for b in range(batch_size):
                                    start_idx = b * neurons_per_sample
                                    end_idx = (b + 1) * neurons_per_sample
                                    dataset_batch[b].x[:, 3:4] = pred_x[start_idx:end_idx].reshape(-1, 1)

                                    k_current = k_batch[start_idx, 0].item() + step + 1  

                                    if has_visual_field:
                                        visual_input_next = model.forward_visual(dataset_batch[b].x, k_current)
                                        dataset_batch[b].x[:model.n_input_neurons, 4:5] = visual_input_next
                                        dataset_batch[b].x[model.n_input_neurons:, 4:5] = 0
                                    else:
                                        x_next = torch.tensor(x_list[run][k_current], dtype=torch.float32, device=device)
                                        dataset_batch[b].x[:, 4:5] = x_next[:, 4:5]

                                    dataset_batch_new.append(dataset_batch[b])

                                batch_loader = DataLoader(dataset_batch_new, batch_size=batch_size, shuffle=False)
                                for batch in batch_loader:
                                    pred, in_features, msg = model(batch, data_id=data_id, return_all=True)

                                pred_x = pred_x + delta_t * pred + noise_recurrent_level * torch.randn_like(pred)

                        loss = loss + ((pred_x[ids_batch] - y_batch[ids_batch]) / (delta_t * time_step)).norm(2)



                    else:



                        loss = loss + (pred[ids_batch] - y_batch[ids_batch]).norm(2)


                loss.backward()


                # debug gradient check for neural ODE training
                if DEBUG_ODE and neural_ODE_training and (N % 500 == 0):
                    debug_check_gradients(model, loss, N)

                optimizer.step()

                total_loss += loss.item()
                total_loss_regul += regularizer.get_iteration_total()

                # finalize iteration to record history
                regularizer.finalize_iteration()

                if (N < 10000) & (N % 2000 == 0) & hasattr(model, 'W') :

                    plt.style.use('dark_background')

                    row_start = 1736
                    row_end = 1736 + 217 * 2  # 2160   L1 L2
                    col_start = 0
                    col_end = 217 * 2  # 424
                    learned_in_region = torch.zeros((n_neurons, n_neurons), dtype=torch.float32, device=edges.device)
                    learned_in_region[edges[1], edges[0]] = model.W.squeeze()
                    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                    ax1 = sns.heatmap(to_numpy(learned_in_region[row_start:row_end, col_start:col_end]), center=0, square=True, cmap='bwr',
                                        cbar=False, ax=ax)
                    ax.set_title('learned connectivity', fontsize=24)
                    ax.set_xlabel('columns [0:434] (R1 R2)', fontsize=18)
                    ax.set_ylabel('rows [1736:2160] (L1 L2)', fontsize=18)
                    plt.tight_layout()
                    plt.savefig(f'{log_dir}/results/connectivity_comparison_R_to_L_{N:04d}.png', dpi=150, bbox_inches='tight')
                    plt.close()

                if regularizer.should_record():
                    # get history from regularizer and add loss component
                    current_loss = loss.item()
                    regul_total_this_iter = regularizer.get_iteration_total()
                    loss_components['loss'].append((current_loss - regul_total_this_iter) / n_neurons)

                    # merge loss_components with regularizer history for plotting
                    plot_dict = {**regularizer.get_history(), 'loss': loss_components['loss']}

                    # pass per-neuron normalized values to debug (to match dictionary values)
                    plot_signal_loss(plot_dict, log_dir, epoch=epoch, Niter=N, debug=False,
                                   current_loss=current_loss / n_neurons, current_regul=regul_total_this_iter / n_neurons,
                                   total_loss=total_loss, total_loss_regul=total_loss_regul)


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

                            out_dir = f"./{log_dir}/tmp_training/external_input"
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
                                        pred_vec.min()
                                        pred_vec.max()

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
                                    float(np.mean(np.abs(pred_vec - gt_vec)))
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
                        plot_training_flyvis(x_list, model, config, epoch, N, log_dir, device, cmap, type_list, gt_weights, edges, n_neurons=n_neurons, n_neuron_types=n_neuron_types)
                    torch.save(
                        {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                        os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))

            # check_and_clear_memory(device=device, iteration_number=N, every_n_iterations=Niter // 50, memory_percentage_threshold=0.6)




        # Calculate epoch-level losses
        epoch_total_loss = total_loss / n_neurons
        epoch_regul_loss = total_loss_regul / n_neurons
        epoch_pred_loss = (total_loss - total_loss_regul) / n_neurons

        print("epoch {}. loss: {:.6f} (pred: {:.6f}, regul: {:.6f})".format(
            epoch, epoch_total_loss, epoch_pred_loss, epoch_regul_loss))
        logger.info("Epoch {}. Loss: {:.6f} (pred: {:.6f}, regul: {:.6f})".format(
            epoch, epoch_total_loss, epoch_pred_loss, epoch_regul_loss))
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}.pt'))

        list_loss.append(epoch_pred_loss)
        list_loss_regul.append(epoch_regul_loss)

        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))

        fig = plt.figure(figsize=(15, 10))

        # Plot 1: Loss
        fig.add_subplot(2, 3, 1)
        plt.plot(list_loss, color=mc, linewidth=1)
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
            fig.add_subplot(2, 3, 2)
            img = imread(f"./{log_dir}/tmp_training/embedding/{last_epoch}_{last_N}.tif")
            plt.imshow(img)
            plt.axis('off')
            plt.title('Embedding', fontsize=12)

            # Plot 3: Last weight comparison
            fig.add_subplot(2, 3, 3)
            img = imread(f"./{log_dir}/tmp_training/matrix/comparison_{last_epoch}_{last_N}.tif")
            plt.imshow(img)
            plt.axis('off')
            plt.title('Weight Comparison', fontsize=12)

            # Plot 4: Last edge function
            fig.add_subplot(2, 3, 4)
            img = imread(f"./{log_dir}/tmp_training/function/MLP1/func_{last_epoch}_{last_N}.tif")
            plt.imshow(img)
            plt.axis('off')
            plt.title('Edge Function', fontsize=12)

            # Plot 5: Last phi function
            fig.add_subplot(2, 3, 5)
            img = imread(f"./{log_dir}/tmp_training/function/MLP0/func_{last_epoch}_{last_N}.tif")
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

                fig.add_subplot(2, 3, 6)
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



def data_train_flyvis_RNN(config, erase, best_model, device):
    """RNN training with sequential processing through time"""

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    signal_model_name = model_config.signal_model_name
    n_epochs = train_config.n_epochs
    dataset_name = config.dataset
    n_runs = train_config.n_runs
    n_frames = simulation_config.n_frames
    data_augmentation_loop = train_config.data_augmentation_loop
    training_selected_neurons = train_config.training_selected_neurons

    warm_up_length = train_config.warm_up_length  # e.g., 10
    sequence_length = train_config.sequence_length  # e.g., 32
    total_length = warm_up_length + sequence_length

    seed = config.training.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    log_dir, logger = create_log_dir(config, erase)

    print(f"Loading data from {dataset_name}...")
    x_list = []
    y_list = []
    for run in trange(0, n_runs, ncols=50):
        x = np.load(f'graphs_data/{dataset_name}/x_list_{run}.npy')
        y = np.load(f'graphs_data/{dataset_name}/y_list_{run}.npy')

        if training_selected_neurons:
            selected_neuron_ids = np.array(train_config.selected_neuron_ids).astype(int)
            x = x[:, selected_neuron_ids, :]
            y = y[:, selected_neuron_ids, :]

        x_list.append(x)
        y_list.append(y)

    print(f'dataset: {len(x_list)} runs, {len(x_list[0])} frames')

    # Normalization
    activity = torch.tensor(x_list[0][:, :, 3:4], device=device)
    activity = activity.squeeze()
    distrib = activity.flatten()
    valid_distrib = distrib[~torch.isnan(distrib)]

    if len(valid_distrib) > 0:
        xnorm = 1.5 * torch.std(valid_distrib)
    else:
        xnorm = torch.tensor(1.0, device=device)

    ynorm = torch.tensor(1.0, device=device)
    torch.save(xnorm, os.path.join(log_dir, 'xnorm.pt'))
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))

    print(f'xnorm: {xnorm.item():.3f}')
    print(f'ynorm: {ynorm.item():.3f}')
    logger.info(f'xnorm: {xnorm.item():.3f}')
    logger.info(f'ynorm: {ynorm.item():.3f}')

    # Create model
    if 'LSTM' in signal_model_name:
        model = Signal_Propagation_LSTM(aggr_type=model_config.aggr_type, config=config, device=device)
        use_lstm = True
    else:  # GRU/RNN
        model = Signal_Propagation_RNN(aggr_type=model_config.aggr_type, config=config, device=device)
        use_lstm = False

    # Count parameters
    n_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'total parameters: {n_total_params:,}')
    logger.info(f'Total parameters: {n_total_params:,}')

    # Optimizer
    lr = train_config.learning_rate_start
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    print(f'learning rate: {lr}')
    logger.info(f'learning rate: {lr}')

    print("starting RNN training...")
    logger.info("Starting RNN training...")

    list_loss = []

    for epoch in range(n_epochs):

        # Number of sequences per epoch
        n_sequences = (n_frames - total_length) // 10 * data_augmentation_loop
        plot_frequency = int(n_sequences // 10) # Sample ~10% of possible sequences
        if epoch == 0:
            print(f'{n_sequences} sequences per epoch, plot every {plot_frequency} sequences')
            logger.info(f'{n_sequences} sequences per epoch, plot every {plot_frequency} sequences')

        total_loss = 0
        model.train()

        for seq_idx in trange(n_sequences, ncols=150, desc=f"Epoch {epoch}"):

            optimizer.zero_grad()

            # Sample random sequence
            run = np.random.randint(n_runs)
            k_start = np.random.randint(0, n_frames - total_length)

            # Initialize hidden state to None (GRU will initialize to zeros)
            h = None
            c = None if use_lstm else None

            # Warm-up phase
            with torch.no_grad():
                for t in range(k_start, k_start + warm_up_length):
                    x = torch.tensor(x_list[run][t], dtype=torch.float32, device=device)
                    if use_lstm:
                        _, h, c = model(x, h=h, c=c, return_all=True)
                    else:
                        _, h = model(x, h=h, return_all=True)

            # Prediction phase (compute loss)
            loss = 0
            for t in range(k_start + warm_up_length, k_start + total_length):
                x = torch.tensor(x_list[run][t], dtype=torch.float32, device=device)
                y_true = torch.tensor(y_list[run][t], dtype=torch.float32, device=device)

                # Forward pass
                if use_lstm:
                    y_pred, h, c = model(x, h=h, c=c, return_all=True)
                else:
                    y_pred, h = model(x, h=h, return_all=True)

                # Accumulate loss
                loss += (y_pred - y_true).norm(2)

                # # Truncated BPTT: detach hidden state
                # h = h.detach()

            # Normalize by sequence length
            loss = loss / sequence_length

            # Backward and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            if (seq_idx % plot_frequency == 0) and (seq_idx > 0):
                # Save intermediate model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(log_dir, 'models', f'best_model_with_{n_runs-1}_graphs_{epoch}_{seq_idx}.pt'))

        # Epoch statistics
        avg_loss = total_loss / n_sequences
        print(f"Epoch {epoch}. Loss: {avg_loss:.6f}")
        logger.info(f"Epoch {epoch}. Loss: {avg_loss:.6f}")

        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(log_dir, 'models', f'best_model_with_{n_runs-1}_graphs_{epoch}.pt'))

        list_loss.append(avg_loss)
        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))

        # Learning rate decay
        if (epoch + 1) % 10 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            print(f"Learning rate decreased to {param_group['lr']}")
            logger.info(f"Learning rate decreased to {param_group['lr']}")



def data_train_zebra(config, erase, best_model, device):
    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    dimension = simulation_config.dimension
    n_epochs = train_config.n_epochs
    n_neurons = simulation_config.n_neurons
    delta_t = simulation_config.delta_t

    dataset_name = config.dataset
    n_runs = train_config.n_runs
    n_frames = simulation_config.n_frames

    data_augmentation_loop = train_config.data_augmentation_loop
    batch_size = train_config.batch_size
    batch_ratio = train_config.batch_ratio
    plot_batch_size = config.plotting.plot_batch_size


    coeff_W_sign = train_config.coeff_W_sign
    coeff_NNR_f = train_config.coeff_NNR_f


    if config.training.seed != 42:
        torch.random.fork_rng(devices=device)
        torch.random.manual_seed(config.training.seed)

    CustomColorMap(config=config)
    plt.style.use('dark_background')
    mc = 'w'

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
    print(f'n neurons: {n_neurons}')
    logger.info(f'N neurons: {n_neurons}')
    config.simulation.n_neurons =n_neurons
    torch.tensor(x[:, 2 + 2 * dimension:3 + 2 * dimension], device=device)
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

    # if coeff_W_sign > 0:
    #     index_weight = []
    #     for i in range(n_neurons):
    #         # Get source neurons that connect to neuron i
    #         mask = edges[1] == i
    #         index_weight.append(edges[0][mask])

    coeff_W_L1 = train_config.coeff_W_L1
    coeff_edge_diff = train_config.coeff_edge_diff
    coeff_update_diff = train_config.coeff_update_diff
    logger.info(f'coeff_W_L1: {coeff_W_L1} coeff_edge_diff: {coeff_edge_diff} coeff_update_diff: {coeff_update_diff}')
    print(f'coeff_W_L1: {coeff_W_L1} coeff_edge_diff: {coeff_edge_diff} coeff_update_diff: {coeff_update_diff}')

    print("start training ...")

    check_and_clear_memory(device=device, iteration_number=0, every_n_iterations=1, memory_percentage_threshold=0.6)
    # torch.autograd.set_detect_anomaly(True)

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
                    plot_field_comparison(x, model, 20, n_frames, ones, f"./{log_dir}/tmp_training/external_input/field_{epoch}_{N}.png", 100, plot_batch_size)

                torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))

        print("epoch {}. loss: {:.6f}".format(epoch, total_loss / n_neurons))
        logger.info("epoch {}. loss: {:.6f}".format(epoch, total_loss / n_neurons))
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}.pt'))

        list_loss.append(to_numpy(total_loss) / n_neurons)
        # torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))

        fig = plt.figure(figsize=(20, 10))
        fig.add_subplot(1, 2, 1)
        plt.plot(list_loss, color=mc, linewidth=1)
        plt.xlim([0, n_epochs])
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylabel('loss', fontsize=12)
        plt.xlabel('epochs', fontsize=12)
        fig.add_subplot(1, 2, 2)
        field_files = glob.glob(f"./{log_dir}/tmp_training/external_input/*.png")
        last_file = max(field_files, key=os.path.getctime)  # or use os.path.getmtime for modification time
        filename = os.path.basename(last_file)
        filename = filename.replace('.png', '')
        parts = filename.split('_')
        if len(parts) >= 3:
            last_epoch = parts[1]
            last_N = parts[2]
        else:
            last_epoch, last_N = parts[-2], parts[-1]
        img = imageio.imread(f"./{log_dir}/tmp_training/external_input/field_{last_epoch}_{last_N}.png")
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/epoch_{epoch}.tif")
        plt.close()



def data_train_zebra_fluo(config, erase, best_model, device):
    train_config = config.training



    batch_ratio = train_config.batch_ratio


    if config.training.seed != 42:
        torch.random.fork_rng(devices=device)
        torch.random.manual_seed(config.training.seed)

    CustomColorMap(config=config)
    plt.style.use('dark_background')

    log_dir, logger = create_log_dir(config, erase)

    dy_um = config.zarr.dy_um
    dx_um = config.zarr.dx_um
    dz_um = config.zarr.dz_um  # light sheet step

    STORE = config.zarr.store_fluo
    FRAME = 3736

    ds = open_gcs_zarr(STORE)
    vol_xyz = ds[..., FRAME].read().result()

    ground_truth = torch.tensor(vol_xyz, device=device, dtype=torch.float32) / 512
    ground_truth = ground_truth.permute(1,0,2)


    print("saving vol_xyz as TIFF...")
    print(f"  shape: {vol_xyz.shape}")
    print(f"  dtype: {vol_xyz.dtype}")
    print(f"  value range: [{vol_xyz.min():.4f}, {vol_xyz.max():.4f}]")

    vol_norm = vol_xyz / 1600

    # Transpose to put Z dimension first for ImageJ: (1328, 2048, 72) -> (72, 1328, 2048)
    vol_norm = vol_norm.transpose(2, 0, 1)  # Move Z from last to first dimension
    print(f"  transposed shape for ImageJ: {vol_norm.shape} (Z×Y×X)")

    # Convert to uint16 for TIFF
    vol_uint16 = (vol_norm * 65535).astype(np.uint16)

    tifffile.imwrite(
        'zapbench.tif',
        vol_uint16,
        imagej=True,
        metadata={
            'axes': 'ZYX',
            'unit': 'micrometer',
            'spacing': [4.0, 0.406, 0.406]  # dz, dy, dx
        },
        description=f"ZapBench volume, frame {FRAME}, shape: {vol_xyz.shape}"
    )

    print(f"saved zapbench.tif - shape: {vol_uint16.shape}, size: {vol_uint16.nbytes/(1024*1024):.1f} MB")

    # down sample
    factor = 4
    gt = ground_truth.unsqueeze(0).unsqueeze(0)
    gt_down = F.interpolate(
        gt,
        size=(gt.shape[2] // factor, gt.shape[3] // factor, gt.shape[4]),
        mode='trilinear',
        align_corners=False
    )
    ground_truth = gt_down.squeeze(0).squeeze(0)
    nx, ny, nz = ground_truth.shape
    print(f'\nshape: {ground_truth.shape}')

    side_length = max(nx, ny)
    iy, ix, iz = torch.meshgrid(
        torch.arange(ny, device=device),
        torch.arange(nx, device=device),
        torch.arange(nz, device=device),
        indexing='ij'  # valid values: 'ij' or 'xy'
    )
    model_input = torch.stack([iy.reshape(-1), ix.reshape(-1), iz.reshape(-1)], dim=1).to(dtype=torch.float32)  # [nx*ny*nz, 3]
    model_input[:,0] = model_input[:,0].float() / (side_length - 1)
    model_input[:,1] = model_input[:,1].float() / (side_length - 1)
    model_input[:,2] = model_input[:,2].float() / nz
    model_input = model_input.cuda()

    ground_truth = ground_truth.reshape([nx*ny*nz, 1]).cuda()

    img_siren = Siren(in_features=3, out_features=1, hidden_features=1024, hidden_layers=3, outermost_linear=True, first_omega_0=512., hidden_omega_0=512.)
    img_siren.cuda()
    total_steps = 3000  # Since the whole image is our dataset, this just means 500 gradient descent steps.
    steps_til_summary = 1000
    optim = torch.optim.Adam(lr=1e-5, params=img_siren.parameters())

    batch_ratio = 0.01

    loss_list = []
    for step in trange(total_steps+1):

        sample_ids = np.random.choice(model_input.shape[0], int(model_input.shape[0]*batch_ratio), replace=False)
        model_input_batch = model_input[sample_ids]
        ground_truth_batch = ground_truth[sample_ids]
        loss = ((img_siren(model_input_batch) - ground_truth_batch) ** 2).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()

        loss_list.append(loss.item())

    if (step % steps_til_summary == 0) and (step>0):
        print("step %d, total loss %0.6f" % (step, loss))

        z_idx = 20

        z_vals = torch.unique(model_input[:, 2], sorted=True)
        z_val  = z_vals[z_idx]
        plane_mask = torch.isclose(model_input[:, 2], z_val, atol=1e-6, rtol=0.0)
        plane_in = model_input[plane_mask]

        with torch.no_grad():
            model_output = img_siren(plane_in)

        gt = to_numpy(ground_truth).reshape(nx, ny, nz)   # both expected (nx, ny, nz)
        gt_xy = gt[:, :, z_idx].astype(np.float32)
        pd_xy = to_numpy(model_output).reshape(nx, ny)

        vmin, vmax = np.percentile(gt_xy, [1, 99.9]);
        vmin = float(vmin) if np.isfinite(vmin) else float(min(gt_xy.min(), pd_xy.min()))
        vmax = float(vmax) if np.isfinite(vmax) and vmax>vmin else float(max(gt_xy.max(), pd_xy.max()))
        rmse = float(np.sqrt(np.mean((pd_xy - gt_xy) ** 2)))
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        ax[0].imshow(gt_xy, cmap="gray", vmin=vmin, vmax=vmax); ax[0].set_title(f"GT z={z_idx}"); ax[0].axis("off")
        im1 = ax[1].imshow(pd_xy, cmap="gray", vmin=vmin, vmax=vmax); ax[1].set_title(f"Pred z={z_idx}  RMSE={rmse:.4f}"); ax[1].axis("off")
        fig.colorbar(im1, ax=ax.ravel().tolist(), fraction=0.046, pad=0.04, label="intensity")
        plt.show()

    viewer = napari.Viewer()
    viewer.add_image(gt, name='ground_truth', scale=[dy_um*factor, dx_um*factor, dz_um])
    viewer.add_image(pd, name='pred_img', scale=[dy_um*factor, dx_um*factor, dz_um])
    viewer.dims.ndisplay = 3
    napari.run()



def data_train_INR(config=None, device=None, total_steps=5000, erase=False):
    """
    Train nnr_f (SIREN/INR network) on external_input data from x_list.

    This pre-trains the implicit neural representation (INR) network before
    joint learning with GNN. The INR learns to map time -> external_input
    for all neurons.

    INR types:
        siren_t: input=t, output=n_neurons (works for n_neurons < 100)
        siren_id: input=(t, id/n_neurons), output=1 (scales better for large n_neurons)
        siren_x: input=(t, x, y), output=1 (uses neuron positions)
        ngp: instantNGP hash encoding

    Args:
        config: NeuralGraphConfig object
        device: torch device
        total_steps: number of training steps (default: 5000)
        erase: whether to erase existing log files (default: False)

    Returns:
        nnr_f: trained SIREN model
        loss_list: list of training losses
    """

    # create log directory
    log_dir, logger = create_log_dir(config, erase)
    output_folder = os.path.join(log_dir, 'tmp_training', 'external_input')
    os.makedirs(output_folder, exist_ok=True)

    dataset_name = config.dataset
    data_folder = f"graphs_data/{dataset_name}/"
    print(f"loading data from: {data_folder}")

    # load x_list data
    x_list = np.load(f"{data_folder}x_list_0.npy")
    print(f"x_list shape: {x_list.shape}")  # (n_frames, n_neurons, n_features)

    n_frames, n_neurons, n_features = x_list.shape
    print(f"n_frames: {n_frames}, n_neurons: {n_neurons}, n_features: {n_features}")

    # extract external_input from x_list (column 4)
    external_input = x_list[:, :, 4]  # shape: (n_frames, n_neurons)

    # SVD analysis
    U, S, Vt = np.linalg.svd(external_input, full_matrices=False)

    # effective rank
    cumvar = np.cumsum(S**2) / np.sum(S**2)
    rank_90 = np.searchsorted(cumvar, 0.90) + 1
    rank_99 = np.searchsorted(cumvar, 0.99) + 1

    print(f"effective rank (90% var): {rank_90}")
    print(f"effective rank (99% var): {rank_99}")
    print(f"compression: {n_frames * n_neurons / (rank_99 * (n_frames + n_neurons)):.1f}x")
    print()

    # extract neuron positions from x_list (columns 1, 2) - use first frame as reference
    neuron_positions = x_list[0, :, 1:3]  # shape: (n_neurons, 2)


    # extract neuron ids from x_list (column 0)
    neuron_ids = x_list[0, :, 0]  # shape: (n_neurons,)

    # get nnr_f config parameters
    model_config = config.graph_model
    hidden_dim_nnr_f = getattr(model_config, 'hidden_dim_nnr_f', 1024)
    n_layers_nnr_f = getattr(model_config, 'n_layers_nnr_f', 3)
    outermost_linear_nnr_f = getattr(model_config, 'outermost_linear_nnr_f', True)
    omega_f = getattr(model_config, 'omega_f', 1024)
    nnr_f_T_period = getattr(model_config, 'nnr_f_T_period', 10000)

    # get training config parameters
    training_config = config.training
    batch_size = getattr(training_config, 'batch_size', 8)
    learning_rate = getattr(training_config, 'learning_rate_NNR_f', 1e-6)

    # get simulation config for calculation check
    sim_config = config.simulation
    delta_t = getattr(sim_config, 'delta_t', 0.01)
    oscillation_frequency = getattr(sim_config, 'oscillation_frequency', 0.1)

    # calculation check
    total_sim_time = n_frames * delta_t
    period_time_units = 1.0 / oscillation_frequency if oscillation_frequency > 0 else float('inf')
    period_frames = period_time_units / delta_t if oscillation_frequency > 0 else float('inf')
    total_cycles = total_sim_time / period_time_units if oscillation_frequency > 0 else 0
    normalized_time_max = n_frames / nnr_f_T_period
    cycles_in_normalized_range = total_cycles * normalized_time_max
    recommended_omega = 2 * np.pi * cycles_in_normalized_range

    # get INR type from config
    inr_type = getattr(model_config, 'inr_type', 'siren_t')

    print(f"siren calculation check:")
    print(f"  total simulation time: {n_frames} × {delta_t} = {total_sim_time:.1f} time units")
    print(f"  period: 1/{oscillation_frequency} = {period_time_units:.1f} time units = {period_frames:.0f} frames")
    print(f"  total cycles: {total_cycles:.0f}")
    print(f"  normalized input range: [0, {n_frames}/{nnr_f_T_period}] = [0, {normalized_time_max:.2f}]")
    print(f"  cycles in normalized range: {total_cycles:.0f} × {normalized_time_max:.2f} = {cycles_in_normalized_range:.1f}")
    print(f"  recommended omega_f: 2π × {cycles_in_normalized_range:.1f} ≈ {recommended_omega:.0f}")
    print(f"  omega_f (config): {omega_f}")
    if omega_f > 5 * recommended_omega:
        print(f"  ⚠️  omega_f is {omega_f/recommended_omega:.1f}× recommended — may cause slow convergence")

    # data dimensions to learn
    data_dims = n_frames * n_neurons
    print(f"\ndata to learn: {n_frames:,} frames × {n_neurons:,} neurons = {data_dims:,.0f} values")

    # determine input/output dimensions based on inr_type
    if inr_type == 'siren_t':
        input_size_nnr_f = 1
        output_size_nnr_f = n_neurons
    elif inr_type == 'siren_id':
        input_size_nnr_f = 2  # (t, id)
        output_size_nnr_f = 1
    elif inr_type == 'siren_x':
        input_size_nnr_f = 3  # (t, x, y)
        output_size_nnr_f = 1
    elif inr_type == 'ngp':
        input_size_nnr_f = getattr(model_config, 'input_size_nnr_f', 1)
        output_size_nnr_f = getattr(model_config, 'output_size_nnr_f', n_neurons)
    elif inr_type == 'lowrank':
        # lowrank doesn't use input/output sizes in the same way
        pass
    else:
        raise ValueError(f"unknown inr_type: {inr_type}")

    # create INR model based on type
    if inr_type == 'ngp':

        # get NGP config parameters
        ngp_n_levels = getattr(model_config, 'ngp_n_levels', 24)
        ngp_n_features_per_level = getattr(model_config, 'ngp_n_features_per_level', 2)
        ngp_log2_hashmap_size = getattr(model_config, 'ngp_log2_hashmap_size', 22)
        ngp_base_resolution = getattr(model_config, 'ngp_base_resolution', 16)
        ngp_per_level_scale = getattr(model_config, 'ngp_per_level_scale', 1.4)
        ngp_n_neurons = getattr(model_config, 'ngp_n_neurons', 128)
        ngp_n_hidden_layers = getattr(model_config, 'ngp_n_hidden_layers', 4)

        nnr_f = HashEncodingMLP(
            n_input_dims=input_size_nnr_f,
            n_output_dims=output_size_nnr_f,
            n_levels=ngp_n_levels,
            n_features_per_level=ngp_n_features_per_level,
            log2_hashmap_size=ngp_log2_hashmap_size,
            base_resolution=ngp_base_resolution,
            per_level_scale=ngp_per_level_scale,
            n_neurons=ngp_n_neurons,
            n_hidden_layers=ngp_n_hidden_layers,
            output_activation='none'
        )
        nnr_f = nnr_f.to(device)

        # count parameters
        encoding_params = sum(p.numel() for p in nnr_f.encoding.parameters())
        mlp_params = sum(p.numel() for p in nnr_f.mlp.parameters())
        total_params = encoding_params + mlp_params

        print(f"\nusing HashEncodingMLP (instantNGP):")
        print(f"  hash encoding: {ngp_n_levels} levels × {ngp_n_features_per_level} features")
        print(f"  hash table: 2^{ngp_log2_hashmap_size} = {2**ngp_log2_hashmap_size:,} entries")
        print(f"  mlp: {ngp_n_neurons} × {ngp_n_hidden_layers} hidden → {output_size_nnr_f}")
        print(f"  parameters: {total_params:,} (encoding: {encoding_params:,}, mlp: {mlp_params:,})")
        print(f"  compression ratio: {data_dims / total_params:.2f}x")

    elif inr_type in ['siren_t', 'siren_id', 'siren_x']:
        # create SIREN model for nnr_f
        omega_f_learning = getattr(model_config, 'omega_f_learning', False)
        nnr_f = Siren(
            in_features=input_size_nnr_f,
            hidden_features=hidden_dim_nnr_f,
            hidden_layers=n_layers_nnr_f,
            out_features=output_size_nnr_f,
            outermost_linear=outermost_linear_nnr_f,
            first_omega_0=omega_f,
            hidden_omega_0=omega_f,
            learnable_omega=omega_f_learning
        )
        nnr_f = nnr_f.to(device)

        # count parameters
        total_params = sum(p.numel() for p in nnr_f.parameters())

        print(f"\nusing SIREN ({inr_type}):")
        print(f"  architecture: {input_size_nnr_f} → {hidden_dim_nnr_f} × {n_layers_nnr_f} hidden → {output_size_nnr_f}")
        print(f"  omega_f: {omega_f} (learnable: {omega_f_learning})")
        if omega_f_learning and hasattr(nnr_f, 'get_omegas'):
            print(f"  initial omegas: {nnr_f.get_omegas()}")
        print(f"  parameters: {total_params:,}")
        print(f"  compression ratio: {data_dims / total_params:.2f}x")

    elif inr_type == 'lowrank':
        # get lowrank config parameters
        lowrank_rank = getattr(model_config, 'lowrank_rank', 64)
        lowrank_svd_init = getattr(model_config, 'lowrank_svd_init', True)

        # create LowRankINR model
        init_data = external_input if lowrank_svd_init else None
        nnr_f = LowRankINR(
            n_frames=n_frames,
            n_neurons=n_neurons,
            rank=lowrank_rank,
            init_data=init_data
        )
        nnr_f = nnr_f.to(device)

        # count parameters
        total_params = sum(p.numel() for p in nnr_f.parameters())

        print(f"\nusing LowRankINR:")
        print(f"  rank: {lowrank_rank}")
        print(f"  U: ({n_frames}, {lowrank_rank}), V: ({lowrank_rank}, {n_neurons})")
        print(f"  parameters: {total_params:,}")
        print(f"  compression ratio: {data_dims / total_params:.2f}x")
        print(f"  SVD init: {lowrank_svd_init}")

    print(f"\ntraining: batch_size={batch_size}, learning_rate={learning_rate}")

    # prepare training data
    ground_truth = torch.tensor(external_input, dtype=torch.float32, device=device)  # (n_frames, n_neurons)

    # prepare inputs based on inr_type
    if inr_type == 'siren_t':
        # input: normalized time, output: all neurons
        time_input = torch.arange(0, n_frames, dtype=torch.float32, device=device).unsqueeze(1) / nnr_f_T_period

    elif inr_type == 'siren_id':
        # input: (t, id), output: 1
        # normalize id by n_neurons
        neuron_ids_norm = torch.tensor(neuron_ids / n_neurons, dtype=torch.float32, device=device)  # (n_neurons,)

    elif inr_type == 'siren_x':
        # input: (t, x, y), output: 1
        # positions are already normalized
        neuron_pos = torch.tensor(neuron_positions, dtype=torch.float32, device=device)  # (n_neurons, 2)

    steps_til_summary = 5000

    # Separate omega parameters from other parameters for different learning rates
    omega_f_learning = getattr(model_config, 'omega_f_learning', False)
    learning_rate_omega_f = getattr(training_config, 'learning_rate_omega_f', learning_rate)
    omega_params = [p for name, p in nnr_f.named_parameters() if 'omega' in name]
    other_params = [p for name, p in nnr_f.named_parameters() if 'omega' not in name]
    if omega_params and omega_f_learning:
        optim = torch.optim.Adam([
            {'params': other_params, 'lr': learning_rate},
            {'params': omega_params, 'lr': learning_rate_omega_f}
        ])
        print(f"using separate learning rates: network={learning_rate}, omega={learning_rate_omega_f}")
    else:
        optim = torch.optim.Adam(lr=learning_rate, params=nnr_f.parameters())

    print(f"training nnr_f for {total_steps} steps...")

    loss_list = []
    pbar = trange(total_steps + 1, ncols=150)
    for step in pbar:

        if inr_type == 'siren_t':
            # sample batch_size time frames
            sample_ids = np.random.choice(n_frames, batch_size, replace=False)
            time_batch = time_input[sample_ids]  # (batch_size, 1)
            gt_batch = ground_truth[sample_ids]  # (batch_size, n_neurons)
            pred = nnr_f(time_batch)  # (batch_size, n_neurons)

        elif inr_type == 'siren_id':
            # sample batch_size time frames, predict all neurons for each frame
            sample_ids = np.random.choice(n_frames, batch_size, replace=False)
            t_norm = torch.tensor(sample_ids / nnr_f_T_period, dtype=torch.float32, device=device)  # (batch_size,)
            # expand to all neurons: (batch_size, n_neurons, 2)
            t_expanded = t_norm[:, None, None].expand(batch_size, n_neurons, 1)
            id_expanded = neuron_ids_norm[None, :, None].expand(batch_size, n_neurons, 1)
            input_batch = torch.cat([t_expanded, id_expanded], dim=2)  # (batch_size, n_neurons, 2)
            input_batch = input_batch.reshape(batch_size * n_neurons, 2)  # (batch_size * n_neurons, 2)
            gt_batch = ground_truth[sample_ids].reshape(batch_size * n_neurons)  # (batch_size * n_neurons,)
            pred = nnr_f(input_batch).squeeze()  # (batch_size * n_neurons,)

        elif inr_type == 'siren_x':
            # sample batch_size time frames, predict all neurons for each frame
            sample_ids = np.random.choice(n_frames, batch_size, replace=False)
            t_norm = torch.tensor(sample_ids / nnr_f_T_period, dtype=torch.float32, device=device)  # (batch_size,)
            # expand to all neurons: (batch_size, n_neurons, 3)
            t_expanded = t_norm[:, None, None].expand(batch_size, n_neurons, 1)
            pos_expanded = neuron_pos[None, :, :].expand(batch_size, n_neurons, 2)
            input_batch = torch.cat([t_expanded, pos_expanded], dim=2)  # (batch_size, n_neurons, 3)
            input_batch = input_batch.reshape(batch_size * n_neurons, 3)  # (batch_size * n_neurons, 3)
            gt_batch = ground_truth[sample_ids].reshape(batch_size * n_neurons)  # (batch_size * n_neurons,)
            pred = nnr_f(input_batch).squeeze()  # (batch_size * n_neurons,)

        elif inr_type == 'ngp':
            # sample batch_size time frames (same as siren_t)
            sample_ids = np.random.choice(n_frames, batch_size, replace=False)
            time_batch = torch.tensor(sample_ids / nnr_f_T_period, dtype=torch.float32, device=device).unsqueeze(1)
            gt_batch = ground_truth[sample_ids]
            pred = nnr_f(time_batch)

        elif inr_type == 'lowrank':
            # sample batch_size time frames
            sample_ids = np.random.choice(n_frames, batch_size, replace=False)
            t_indices = torch.tensor(sample_ids, dtype=torch.long, device=device)
            gt_batch = ground_truth[sample_ids]  # (batch_size, n_neurons)
            pred = nnr_f(t_indices)  # (batch_size, n_neurons)

        # compute loss
        if inr_type == 'ngp':
            # relative L2 error - convert targets to match output dtype (tcnn uses float16)
            relative_l2_error = (pred - gt_batch.to(pred.dtype)) ** 2 / (pred.detach() ** 2 + 0.01)
            loss = relative_l2_error.mean()
        else:
            # standard MSE for SIREN
            loss = ((pred - gt_batch) ** 2).mean()

        # omega L2 regularization for learnable omega in SIREN (encourages smaller omega)
        coeff_omega_f_L2 = getattr(training_config, 'coeff_omega_f_L2', 0.0)
        if omega_f_learning and coeff_omega_f_L2 > 0 and hasattr(nnr_f, 'get_omega_L2_loss'):
            omega_L2_loss = nnr_f.get_omega_L2_loss()
            loss = loss + coeff_omega_f_L2 * omega_L2_loss

        optim.zero_grad()
        loss.backward()
        optim.step()

        loss_list.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.6f}")

        if step % steps_til_summary == 0:
            with torch.no_grad():
                # compute predictions for all frames
                if inr_type == 'siren_t':
                    pred_all = nnr_f(time_input)  # (n_frames, n_neurons)

                elif inr_type == 'siren_id':
                    # predict all (t, id) combinations
                    pred_list = []
                    for t_idx in range(n_frames):
                        t_val = torch.full((n_neurons, 1), t_idx / nnr_f_T_period, device=device)
                        input_t = torch.cat([t_val, neuron_ids_norm[:, None]], dim=1)  # (n_neurons, 2)
                        pred_t = nnr_f(input_t).squeeze()  # (n_neurons,)
                        pred_list.append(pred_t)
                    pred_all = torch.stack(pred_list, dim=0)  # (n_frames, n_neurons)

                elif inr_type == 'siren_x':
                    # predict all (t, x, y) combinations
                    pred_list = []
                    for t_idx in range(n_frames):
                        t_val = torch.full((n_neurons, 1), t_idx / nnr_f_T_period, device=device)
                        input_t = torch.cat([t_val, neuron_pos], dim=1)  # (n_neurons, 3)
                        pred_t = nnr_f(input_t).squeeze()  # (n_neurons,)
                        pred_list.append(pred_t)
                    pred_all = torch.stack(pred_list, dim=0)  # (n_frames, n_neurons)

                elif inr_type == 'ngp':
                    time_all = torch.arange(0, n_frames, dtype=torch.float32, device=device).unsqueeze(1) / nnr_f_T_period
                    pred_all = nnr_f(time_all)

                elif inr_type == 'lowrank':
                    pred_all = nnr_f()  # returns full (n_frames, n_neurons) matrix

                gt_np = ground_truth.cpu().numpy()
                pred_np = pred_all.cpu().numpy()

                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                fig.patch.set_facecolor('black')

                # loss plot
                axes[0].set_facecolor('black')
                axes[0].plot(loss_list, color='white', lw=0.1)
                axes[0].set_xlabel('step', color='white', fontsize=12)
                loss_label = 'Relative L2 Loss' if inr_type == 'ngp' else 'MSE Loss'
                axes[0].set_ylabel(loss_label, color='white', fontsize=12)
                axes[0].set_yscale('log')
                axes[0].tick_params(colors='white', labelsize=11)
                for spine in axes[0].spines.values():
                    spine.set_color('white')

                # traces plot (10 neurons, darkgreen=GT, white=pred)
                axes[1].set_facecolor('black')
                axes[1].set_axis_off()
                n_traces = 10
                trace_ids = np.linspace(0, n_neurons - 1, n_traces, dtype=int)
                offset = np.abs(gt_np).max() * 1.5
                t = np.arange(n_frames)

                for j, n_idx in enumerate(trace_ids):
                    y0 = j * offset
                    axes[1].plot(t, gt_np[:, n_idx] + y0, color='darkgreen', lw=2.0, alpha=0.95)
                    axes[1].plot(t, pred_np[:, n_idx] + y0, color='white', lw=0.5, alpha=0.95)

                axes[1].set_xlim(0, min(20000, n_frames))
                axes[1].set_ylim(-offset * 0.5, offset * (n_traces + 0.5))
                mse = ((pred_np - gt_np) ** 2).mean()
                omega_str = ''
                if hasattr(nnr_f, 'get_omegas'):
                    omegas = nnr_f.get_omegas()
                    if omegas:
                        omega_str = f'  ω: {omegas[0]:.1f}'
                axes[1].text(0.02, 0.98, f'MSE: {mse:.6f}{omega_str}',
                            transform=axes[1].transAxes, va='top', ha='left',
                            fontsize=12, color='white')

                plt.tight_layout()
                plt.savefig(f"{output_folder}/{inr_type}_{step}.png", dpi=150)
                plt.close()

    # save trained model
    # save_path = f"{output_folder}/nnr_f_{inr_type}_pretrained.pt"
    # torch.save(nnr_f.state_dict(), save_path)
    # print(f"\nsaved pretrained nnr_f to: {save_path}")

    # compute final MSE
    with torch.no_grad():
        if inr_type == 'siren_t':
            pred_all = nnr_f(time_input)
        elif inr_type == 'siren_id':
            pred_list = []
            for t_idx in range(n_frames):
                t_val = torch.full((n_neurons, 1), t_idx / nnr_f_T_period, device=device)
                input_t = torch.cat([t_val, neuron_ids_norm[:, None]], dim=1)
                pred_t = nnr_f(input_t).squeeze()
                pred_list.append(pred_t)
            pred_all = torch.stack(pred_list, dim=0)
        elif inr_type == 'siren_x':
            pred_list = []
            for t_idx in range(n_frames):
                t_val = torch.full((n_neurons, 1), t_idx / nnr_f_T_period, device=device)
                input_t = torch.cat([t_val, neuron_pos], dim=1)
                pred_t = nnr_f(input_t).squeeze()
                pred_list.append(pred_t)
            pred_all = torch.stack(pred_list, dim=0)
        elif inr_type == 'ngp':
            time_all = torch.arange(0, n_frames, dtype=torch.float32, device=device).unsqueeze(1) / nnr_f_T_period
            pred_all = nnr_f(time_all)

        final_mse = ((pred_all - ground_truth) ** 2).mean().item()
        print(f"final MSE: {final_mse:.6f}")
        if hasattr(nnr_f, 'get_omegas'):
            print(f"final omegas: {nnr_f.get_omegas()}")

    return nnr_f, loss_list







def data_test(config=None, config_file=None, visualize=False, style='color frame', verbose=True, best_model=20, step=15, n_rollout_frames=600,
              ratio=1, run=0, test_mode='', sample_embedding=False, particle_of_interest=1, new_params = None, device=[],
              rollout_without_noise: bool = False, log_file=None):

    dataset_name = config.dataset
    print(f"\033[92mdataset_name: {dataset_name}\033[0m")
    print(f"\033[92m{config.description}\033[0m")

    if test_mode == "":
        test_mode = "test_ablation_0"

    if 'fly' in config.dataset:
        data_test_flyvis(
            config,
            visualize,
            style,
            verbose,
            best_model,
            step,
            n_rollout_frames,
            test_mode,
            new_params,
            device,
            rollout_without_noise=rollout_without_noise,
        )

    elif 'zebra' in config.dataset:
        data_test_zebra(config, visualize, style, verbose, best_model, step, test_mode, device)

    else:
        data_test_signal(config, config_file, visualize, style, verbose, best_model, step, n_rollout_frames,ratio, run, test_mode, sample_embedding, particle_of_interest, new_params, device, log_file)



def data_test_signal(config=None, config_file=None, visualize=False, style='color frame', verbose=True, best_model=20, step=15, n_rollout_frames=600, ratio=1, run=0, test_mode='', sample_embedding=False, particle_of_interest=1, new_params = None, device=[], log_file=None):
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
    neural_ODE_training = training_config.neural_ODE_training
    ode_method = training_config.ode_method
    ode_rtol = training_config.ode_rtol
    ode_atol = training_config.ode_atol

    cmap = CustomColorMap(config=config)
    dimension = simulation_config.dimension

    has_missing_activity = training_config.has_missing_activity
    has_excitation = ('excitation' in model_config.update_type)
    baseline_value = simulation_config.baseline_value

    torch.random.fork_rng(devices=device)
    torch.random.manual_seed(simulation_config.seed)

    if 'latex' in style:
        print('latex style...')
        plt.rcParams['text.usetex'] = False  # LaTeX disabled - use mathtext instead
        rc('font', **{'family': 'serif', 'serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif', 'serif']})
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
    os.makedirs(f'./{log_dir}/results/Fig', exist_ok=True)
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

    print('load data...')

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

    edge_index_generated = edge_index.clone().detach()


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
    geomloss_list = []
    angle_list = []
    time.sleep(1)

    if time_window > 0:
        start_it = time_window
        n_frames - 1
    else:
        start_it = 0
        n_frames - 1

    start_it = 0

    x = x_list[0][start_it].clone().detach()
    x_generated = x_list[0][start_it].clone().detach()

    if 'test_ablation' in test_mode:
        #  test_mode="test_ablation_0 by default
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
                simulation_config.connectivity_type,
                simulation_config.connectivity_filling_factor,
                new_params[0],
                n_neurons,
                n_neuron_types,
                dataset_name,
                device,
                connectivity_rank=simulation_config.connectivity_rank,
                Dale_law=simulation_config.Dale_law,
                Dale_law_factor=simulation_config.Dale_law_factor,
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

        cell_types = to_numpy(x[:, 6]).astype(int)
        type_counts = [np.sum(cell_types == n) for n in range(n_neuron_types)]
        plt.figure(figsize=(10, 10))
        plt.bar(range(n_neuron_types), type_counts,
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
            x[id:id+int(new_params[n+1]*n_neurons/100), 6] = n
            x_generated[id:id+int(new_params[n+1]*n_neurons/100), 6] = n
            id = id + int(new_params[n+1]*n_neurons/100)
        print(f'last cell id {id}, total number of neurons {n_neurons}')

        first_embedding = model.a.clone().detach()
        model_a_ = nn.Parameter(torch.tensor(np.ones((int(n_neurons), model.embedding_dim)), device=device, requires_grad=False,dtype=torch.float32))
        for n in range(n_neurons):
            t = to_numpy(x[n, 6]).astype(int)
            index = first_cell_id_neurons[t][np.random.randint(len(first_cell_id_neurons[t]))]
            with torch.no_grad():
                model_a_[n] = first_embedding[index].clone().detach()
        model.a = nn.Parameter(
            torch.tensor(np.ones((int(n_neurons), model.embedding_dim)), device=device,
                         requires_grad=False, dtype=torch.float32))
        with torch.no_grad():
            for n in range(model.a.shape[0]):
                model.a[n] = model_a_[n]

                cell_types = to_numpy(x[:, 6]).astype(int)
        type_counts = [np.sum(cell_types == n) for n in range(n_neuron_types)]
        plt.figure(figsize=(10, 10))
        plt.bar(range(n_neuron_types), type_counts,
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

    # n_neurons = x.shape[0]
    neuron_gt_list = []
    neuron_pred_list = []
    neuron_generated_list = []

    x_inference_list = []
    x_generated_list = []

    R2_list = []
    it_list = []
    id_fig = 0


    n_test_frames = n_rollout_frames
    it_step = step

    print('rollout inference...')

    for it in trange(start_it, start_it + n_test_frames, ncols=100):

        if it < n_frames - 4:
            x0 = x_list[0][it].clone().detach()
            x0_next = x_list[0][(it + time_step)].clone().detach()
            y_list[0][it].clone().detach()
        if has_excitation:
            x[:, 10: 10 + model_config.excitation_dim] = x0[:, 10: 10 + model_config.excitation_dim]

        x0[:, 3] = torch.where(torch.isnan(x0[:, 3]), baseline_value, x0[:, 3])
        x[:, 3]  = torch.where(torch.isnan(x[:, 3]),  baseline_value, x[:, 3])
        x_generated[:, 3] = torch.where(torch.isnan(x_generated[:, 3]), baseline_value, x_generated[:, 3])


        x_inference_list.append(x[:, 3:4].clone().detach())
        x_generated_list.append(x_generated[:, 3:4].clone().detach())

        if ablation_ratio > 0:
            rmserr = torch.sqrt(torch.mean((x_generated[:n_neurons, 3] - x0[:, 3]) ** 2))
        else:
            rmserr = torch.sqrt(torch.mean((x[:n_neurons, 3] - x0[:, 3]) ** 2))

        neuron_gt_list.append(x0[:, 3:4])
        neuron_pred_list.append(x[:n_neurons, 3:4].clone().detach())
        neuron_generated_list.append(x_generated[:n_neurons, 3:4].clone().detach())

        if ('short_term_plasticity' in field_type) | ('modulation' in field_type):
            modulation_gt_list.append(x0[:, 4:5])
            modulation_pred_list.append(x[:, 4:5].clone().detach())
        rmserr_list.append(rmserr.item())

        if config.training.shared_embedding:
            data_id = torch.ones((n_neurons, 1), dtype=torch.int, device=device)
        else:
            data_id = torch.ones((n_neurons, 1), dtype=torch.int, device=device) * run

        # update calculations
        if 'visual' in field_type:
            x[:n_nodes, 4:5] = model_f(time=it / n_frames) ** 2
            x[n_nodes:n_neurons, 4:5] = 1
        elif 'learnable_short_term_plasticity' in field_type:
            alpha = (it % model.embedding_step) / model.embedding_step
            x[:, 4] = alpha * model.b[:, it // model.embedding_step + 1] ** 2 + (1 - alpha) * model.b[:,
                                                                                                it // model.embedding_step] ** 2
        elif ('short_term_plasticity' in field_type) | ('modulation_permutation' in field_type):
            t = torch.zeros((1, 1, 1), dtype=torch.float32, device=device)
            t[:, 0, :] = torch.tensor(it / n_frames, dtype=torch.float32, device=device)
            x[:, 4] = model_f(t).squeeze() ** 2
        elif 'modulation' in field_type:
            x[:, 4:5] = model_f(time=it / n_frames) ** 2

        if has_missing_activity:
            t = torch.tensor([it / n_frames], dtype=torch.float32, device=device)
            missing_activity = baseline_value + model_missing_activity[run](t).squeeze()
            ids_missing = torch.argwhere(x[:, 3] == baseline_value)
            x[ids_missing, 3] = missing_activity[ids_missing]

        with torch.no_grad():
            dataset = pyg_Data(x=x, pos=x[:, 1:3], edge_index=edge_index)
            if neural_ODE_training:
                # Use Neural ODE integration with time_step=1
                u0 = x[:, 3].flatten()
                u_final, _ = integrate_neural_ode_Signal(
                    model=model,
                    u0=u0,
                    data_template=dataset,
                    data_id=data_id,
                    time_steps=1,
                    delta_t=delta_t,
                    neurons_per_sample=n_neurons,
                    batch_size=1,
                    x_list=None,
                    run=0,
                    device=device,
                    k_batch=torch.tensor([it], device=device),
                    ode_method=ode_method,
                    rtol=ode_rtol,
                    atol=ode_atol,
                    adjoint=False,
                    noise_level=0.0
                )
                y = (u_final.view(-1, 1) - x[:, 3:4]) / delta_t
            else:
                pred = model(dataset, data_id=data_id, k=it)
                y = pred
            dataset = pyg_Data(x=x_generated, pos=x[:, 1:3], edge_index=edge_index_generated)
            if "PDE_N3" in model_config.signal_model_name:
                pred_generator = model_generator(dataset, data_id=data_id, alpha=it/n_frames)
            else:
                pred_generator = model_generator(dataset, data_id=data_id)

        # signal update
        x[:n_neurons, 3:4] = x[:n_neurons, 3:4] + y[:n_neurons] * delta_t
        x_generated[:n_neurons, 3:4] = x_generated[:n_neurons, 3:4] + pred_generator[:n_neurons] * delta_t

        if 'test_inactivity' in test_mode:
            x[index_inactivity, 3:4] = 0
            x_generated[index_inactivity, 3:4] = 0

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
                plt.scatter(to_numpy(x0[:, 2]), to_numpy(x0[:, 1]), s=8, c=to_numpy(x[:, 3:4]), cmap='viridis',
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
                LinearSegmentedColormap.from_list('black_yellow', ['black', 'yellow'])

                plt.figure(figsize=(10, 10))
                plt.scatter(to_numpy(x[:, 1]), to_numpy(x[:, 2]), s=700, c=to_numpy(x[:, 3]), alpha=1, edgecolors='none', vmin =2 , vmax=8, cmap=black_to_green)
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
                        plt.rcParams['text.usetex'] = False
                        plt.rcParams['font.family'] = 'sans-serif'

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
                    1 - (ss_res / ss_tot)
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

        if (it % it_step == 0) & (it > 0) & (it <=n_test_frames):

            num = f"{id_fig:06}"
            id_fig += 1

            if n_neurons <= 101:
                n = np.arange(0, n_neurons, 4)
            elif 'CElegans' in dataset_name:
                n = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
            else:
                n = [20, 30, 100, 150, 260, 270, 520, 620, 720, 780]

            neuron_gt_list_ = torch.cat(neuron_gt_list, 0)
            neuron_pred_list_ = torch.cat(neuron_pred_list, 0)
            neuron_generated_list_ = torch.cat(neuron_generated_list, 0)
            neuron_gt_list_ = torch.reshape(neuron_gt_list_, (neuron_gt_list_.shape[0] // n_neurons, n_neurons))
            neuron_pred_list_ = torch.reshape(neuron_pred_list_, (neuron_pred_list_.shape[0] // n_neurons, n_neurons))
            neuron_generated_list_ = torch.reshape(neuron_generated_list_, (neuron_generated_list_.shape[0] // n_neurons, n_neurons))

            mpl.rcParams.update({
                "text.usetex": False,
                "font.family": "sans-serif",
                "font.size": 12,
                "axes.labelsize": 14,
                "legend.fontsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
            })

            plt.figure(figsize=(20, 10))

            ax = plt.subplot(121)
            # Plot ground truth with distinct gray color, visible in legend
            for i in range(len(n)):
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

                if 'test_generated' in test_mode:
                    plt.plot(neuron_generated_list_[:, n[i]].detach().cpu().numpy() + i * 25,
                            linewidth=3, c=colors[i%10], label=label)
                else:
                    plt.plot(neuron_pred_list_[:, n[i]].detach().cpu().numpy() + i * 25,
                            linewidth=3, c=colors[i%10], label=label)

            plt.xlim([0, len(neuron_gt_list_)])

            # Auto ylim from ground truth range (ignore predictions if exploded)
            y_gt = np.concatenate([neuron_gt_list_[:, n[i]].detach().cpu().numpy() + i*25 for i in range(len(n))])
            y_pred = np.concatenate([neuron_pred_list_[:, n[i]].detach().cpu().numpy() + i*25 for i in range(len(n))])

            if np.abs(y_pred).max() > 10 * np.abs(y_gt).max():  # Explosion
                ylim = [y_gt.min() - 10, y_gt.max() + 10]
            else:
                y_all = np.concatenate([y_gt, y_pred])
                margin = (y_all.max() - y_all.min()) * 0.05
                ylim = [y_all.min() - margin, y_all.max() + margin]

            plt.xlim([0, n_test_frames])
            plt.ylim(ylim)
            plt.xlabel('frame', fontsize=48)
            if 'PDE_N11' in config.graph_model.signal_model_name:
                plt.ylabel('$h_i$', fontsize=48)
            else:
                plt.ylabel('$x_i$', fontsize=48)
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

                plt.scatter(x_data, y_data, s=20, c=mc, alpha=0.8, edgecolors='none', linewidths=0.5)
                if mask.sum() > 10:
                    x_line = np.array(lim)
                    plt.plot(x_line, linear_model(x_line, slope, intercept), 'r--', linewidth=2)
                plt.plot(lim, lim, 'k--', alpha=0.3, linewidth=1)
                plt.text(0.05, 0.95, f'$R^2$: {r2:.3f}\nslope: {slope:.3f}',
                        transform=plt.gca().transAxes, fontsize=24, va='top')
            else:
                # Severe collapse/explosion
                lim = [x_data.min()*1.1, 0]
                plt.scatter(x_data, np.clip(y_data, lim[0], lim[1]), s=100, c=mc, alpha=0.8, edgecolors='k', linewidths=0.5)
                plt.text(0.5, 0.5, 'collapsed' if severe_collapse else 'explosion',
                        ha='center', fontsize=48, color='red', alpha=0.3, transform=plt.gca().transAxes)
                r2 = 0

            # plt.xlim([-20,20])
            # plt.ylim([-20,20])
            if 'PDE_N11' in config.graph_model.signal_model_name:
                plt.xlabel('true $h_i$', fontsize=48)
                plt.ylabel('learned $h_i$', fontsize=48)
            else:
                plt.xlabel('true $x_i$', fontsize=48)
                plt.ylabel('learned $x_i$', fontsize=48)
            plt.xticks([])
            plt.yticks([])


            R2_list.append(r2)
            it_list.append(it)


            ax = plt.subplot(224)
            plt.scatter(it_list, R2_list, s=20, c=mc)
            plt.xlim([0, n_test_frames])
            plt.ylim([0, 1])
            plt.axhline(1, color='green', linestyle='--', linewidth=2)
            plt.xlabel('frame', fontsize=48)
            plt.ylabel('$R^2$', fontsize=48)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.tight_layout()

            if ablation_ratio>0:
                filename = f'comparison_vi_{it}_ablation_{ablation_ratio}.png'
            elif inactivity_ratio>0:
                filename = f'comparison_vi_{it}_inactivity_{inactivity_ratio}.png'
            elif permutation_ratio>0:
                filename = f'comparison_vi_{it}_permutation_{permutation_ratio}.png'
            else:
                filename = f'comparison_vi_{it}.png'

            plt.savefig(f'./{log_dir}/results/Fig/Fig_{run}_{num}.png', dpi=80)
            plt.close()
            # print(f'saved figure ./log/{log_dir}/results/{filename}')





    dataset_name_ = dataset_name.split('/')[-1]
    generate_compressed_video_mp4(output_dir=f"./{log_dir}/results/", run=run, output_name=dataset_name_, framerate=20)

    # Copy the last PNG file before erasing Fig folder
    files = glob.glob(f'./{log_dir}/results/Fig/*.png')
    if files:
        files.sort()  # Sort to get the last file
        last_file = files[-1]
        dataset_name_ = dataset_name.split('/')[-1]
        dst_file = f"./{log_dir}/results/{dataset_name_}.png"
        import shutil
        shutil.copy(last_file, dst_file)
        print(f"saved last frame: {dst_file}")

    files = glob.glob(f'./{log_dir}/results/Fig/*')
    for f in files:
        os.remove(f)


    x_inference_list = torch.cat(x_inference_list, 1)
    x_generated_list = torch.cat(x_generated_list, 1)



    print('plot prediction ...')
    # Single panel plot: green=GT, white=prediction, R² on the right
    # Limit to max 50 traces evenly spaced across neurons
    n_traces = min(50, n_neurons)
    trace_ids = np.linspace(0, n_neurons - 1, n_traces, dtype=int)

    # Stack ground truth and prediction lists
    neuron_gt_stacked = torch.cat(neuron_gt_list, dim=0).reshape(-1, n_neurons).T  # [n_neurons, n_frames]
    neuron_pred_stacked = torch.cat(neuron_pred_list, dim=0).reshape(-1, n_neurons).T  # [n_neurons, n_frames]

    activity_gt = to_numpy(neuron_gt_stacked)  # ground truth
    activity_pred = to_numpy(neuron_pred_stacked)  # prediction
    n_frames_plot = activity_gt.shape[1]

    # Compute per-neuron R² for selected traces
    r2_per_neuron = []
    for idx in trace_ids:
        gt_trace = activity_gt[idx]
        pred_trace = activity_pred[idx]
        ss_res = np.sum((gt_trace - pred_trace) ** 2)
        ss_tot = np.sum((gt_trace - np.mean(gt_trace)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        r2_per_neuron.append(r2)

    fig, ax = plt.subplots(1, 1, figsize=(20, 12))

    # Compute offset based on data range
    offset = np.abs(activity_gt[trace_ids]).max() * 1.5
    if offset == 0:
        offset = 1.0

    for j, n_idx in enumerate(trace_ids):
        y0 = j * offset
        baseline = np.mean(activity_gt[n_idx])
        # Ground truth in green (thicker line)
        ax.plot(activity_gt[n_idx] - baseline + y0, color='green', lw=4.0, alpha=0.9)
        # Prediction in white (or mc for style consistency)
        ax.plot(activity_pred[n_idx] - baseline + y0, color=mc, lw=0.8, alpha=0.9)

        # Neuron index on the left
        ax.text(-n_frames_plot * 0.02, y0, str(n_idx), fontsize=10, va='center', ha='right')

        # R² on the right with color coding
        r2_val = r2_per_neuron[j]
        r2_color = 'red' if r2_val < 0.5 else ('orange' if r2_val < 0.8 else mc)
        ax.text(n_frames_plot * 1.02, y0, f'R²:{r2_val:.2f}', fontsize=9, va='center', ha='left', color=r2_color)

    ax.set_xlim([-n_frames_plot * 0.05, n_frames_plot * 1.1])
    ax.set_ylim([-offset, n_traces * offset])
    ax.set_xlabel('frame', fontsize=24)
    ax.set_ylabel('neuron', fontsize=24)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_bounds(0, n_frames_plot)
    ax.set_yticks([])

    # Add overall R² in title
    mean_r2 = np.mean(r2_per_neuron)
    ax.set_title(f'Activity traces (n={n_traces} of {n_neurons} neurons) | mean R²={mean_r2:.3f}', fontsize=20)

    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/{dataset_name_}_prediction.pdf", dpi=300)
    plt.close()



    if 'PDE_N' in model_config.signal_model_name:
        torch.save(neuron_gt_list, f"./{log_dir}/neuron_gt_list.pt")
        torch.save(neuron_pred_list, f"./{log_dir}/neuron_pred_list.pt")

    # Comprehensive R² analysis
    if len(R2_list) > 0:
        R2_array = np.array(R2_list)
        it_array = np.array(it_list)

        # Basic statistics
        r2_mean = np.mean(R2_array)
        r2_std = np.std(R2_array)
        r2_min = np.min(R2_array)
        r2_max = np.max(R2_array)
        r2_median = np.median(R2_array)

        # High R² analysis (R² > 0.9)
        high_r2_mask = R2_array > 0.9
        n_frames_high_r2 = np.sum(high_r2_mask)
        pct_frames_high_r2 = 100.0 * n_frames_high_r2 / len(R2_array)

        # Find longest consecutive run of R² > 0.9
        high_r2_runs = []
        current_run_start = None
        current_run_length = 0

        for i, (r2_val, frame_idx) in enumerate(zip(R2_array, it_array)):
            if r2_val > 0.9:
                if current_run_start is None:
                    current_run_start = frame_idx
                    current_run_length = 1
                else:
                    current_run_length += 1
            else:
                if current_run_start is not None:
                    high_r2_runs.append((current_run_start, current_run_length))
                    current_run_start = None
                    current_run_length = 0

        # Don't forget the last run if it extends to the end
        if current_run_start is not None:
            high_r2_runs.append((current_run_start, current_run_length))

        if high_r2_runs:
            longest_run_start, longest_run_length = max(high_r2_runs, key=lambda x: x[1])
        else:
            longest_run_start, longest_run_length = 0, 0
        print(f"mean R2: \033[92m{r2_mean:.4f}\033[0m +/- {r2_std:.4f}")
        print(f"range: [{r2_min:.4f}, {r2_max:.4f}]")
        if log_file:
            log_file.write(f"test_R2: {r2_mean:.4f}\n")

        # Compute Pearson correlation per neuron across time
        from scipy.stats import pearsonr
        neuron_gt_array = torch.stack(neuron_gt_list, dim=0).squeeze(-1)  # [n_frames, n_neurons]
        neuron_pred_array = torch.stack(neuron_pred_list, dim=0).squeeze(-1)  # [n_frames, n_neurons]
        neuron_gt_np = to_numpy(neuron_gt_array)
        neuron_pred_np = to_numpy(neuron_pred_array)

        pearson_list = []
        for i in range(neuron_gt_np.shape[1]):
            gt_trace = neuron_gt_np[:, i]
            pred_trace = neuron_pred_np[:, i]
            valid = ~(np.isnan(gt_trace) | np.isnan(pred_trace))
            if valid.sum() > 1 and np.std(gt_trace[valid]) > 1e-8 and np.std(pred_trace[valid]) > 1e-8:
                pearson_list.append(pearsonr(gt_trace[valid], pred_trace[valid])[0])
            else:
                pearson_list.append(np.nan)
        pearson_array = np.array(pearson_list)
        print(f"Pearson r: \033[92m{np.nanmean(pearson_array):.4f}\033[0m +/- {np.nanstd(pearson_array):.4f} [{np.nanmin(pearson_array):.4f}, {np.nanmax(pearson_array):.4f}]")
        if log_file:
            log_file.write(f"test_pearson: {np.nanmean(pearson_array):.4f}\n")



def data_test_flyvis(
        config,
        visualize=True,
        style="color",
        verbose=False,
        best_model=None,
        step=5,
        n_rollout_frames=600,
        test_mode='',
        new_params=None,
        device=None,
        rollout_without_noise: bool = False,
):


    if "black" in style:
        plt.style.use("dark_background")
        mc = 'white'
    else:
        plt.style.use("default")
        mc = 'black'

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
    field_type = model_config.field_type
    signal_model_name = model_config.signal_model_name

    neural_ODE_training = training_config.neural_ODE_training
    ode_method = training_config.ode_method
    ode_rtol = training_config.ode_rtol
    ode_atol = training_config.ode_atol
    ode_adjoint = training_config.ode_adjoint
    time_step = training_config.time_step

    ensemble_id = simulation_config.ensemble_id
    model_id = simulation_config.model_id

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
    X1 = torch.cat((X1, pos[torch.randperm(pos.size(0), device=device)]), dim=0)

    state = net.steady_state(t_pre=2.0, dt=delta_t, batch_size=1)
    initial_state = state.nodes.activity.squeeze()
    n_neurons = len(initial_state)

    sequences = stimulus_dataset[0]["lum"]
    frame = sequences[0][None, None]
    net.stimulus.add_input(frame)

    x = torch.zeros(n_neurons, 9, dtype=torch.float32, device=device)
    x[:, 1:3] = X1
    x[:, 0] = torch.arange(n_neurons, dtype=torch.float32, device=device)
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
        x_selected[:, 0] = torch.arange(1, dtype=torch.float32, device=device)
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

    target_frames = n_rollout_frames

    if 'full' in test_mode:
        target_frames = simulation_config.n_frames
        step = 25000
    else:
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
            print(f'\033[93mtest modified W with noise level {noise_W}\033[0m')
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


    # Main loop #####################################

    with torch.no_grad():
        for pass_num in range(num_passes_needed):
            for data_idx, data in enumerate(tqdm(stimulus_dataset, desc="processing stimulus data", ncols=100)):

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
                            y = model(x_selected, data_id=None, return_all=False)

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
                            y = model(x, data_id=None, return_all=False)
                        elif neural_ODE_training:
                            dataset = pyg.data.Data(x=x, pos=x, edge_index=edge_index)
                            data_id = torch.zeros((x.shape[0], 1), dtype=torch.int, device=device)
                            v0 = x[:, 3].flatten()
                            v_final, _ = integrate_neural_ode_FlyVis(
                                model=model,
                                v0=v0,
                                data_template=dataset,
                                data_id=data_id,
                                time_steps=1,
                                delta_t=delta_t,
                                neurons_per_sample=n_neurons,
                                batch_size=1,
                                has_visual_field='visual' in field_type,
                                x_list=None,
                                run=0,
                                device=device,
                                k_batch=torch.tensor([it], device=device),
                                ode_method=ode_method,
                                rtol=ode_rtol,
                                atol=ode_atol,
                                adjoint=False,
                                noise_level=0.0
                            )
                            y = (v_final.view(-1, 1) - x[:, 3:4]) / delta_t
                        else:
                            dataset = pyg.data.Data(x=x, pos=x, edge_index=edge_index)
                            data_id = torch.zeros((x.shape[0], 1), dtype=torch.int, device=device)
                            y = model(dataset, data_id=data_id, return_all=False)

                    # Save states
                    x_generated_list.append(to_numpy(x_generated.clone().detach()))
                    x_generated_modified_list.append(to_numpy(x_generated_modified.clone().detach()))

                    if training_selected_neurons:
                        x_list.append(to_numpy(x_selected.clone().detach()))
                    else:
                        x_list.append(to_numpy(x.clone().detach()))

                    # Integration step
                    # Optionally disable process noise at test time, even if model was trained with noise
                    effective_noise_level = 0.0 if rollout_without_noise else noise_model_level
                    if effective_noise_level != noise_model_level:
                        print(f"Effective noise level: {effective_noise_level} (rollout_without_noise: {rollout_without_noise})")
                    if effective_noise_level > 0:
                        x_generated[:, 3:4] = x_generated[:, 3:4] + delta_t * y_generated + torch.randn(
                            (n_neurons, 1), dtype=torch.float32, device=device
                        ) * effective_noise_level
                        x_generated_modified[:, 3:4] = x_generated_modified[:, 3:4] + delta_t * y_generated_modified + torch.randn(
                            (n_neurons, 1), dtype=torch.float32, device=device
                        ) * effective_noise_level
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
                            plt.rcParams["text.usetex"] = False  # Disabled due to font issues
                            rc("font", **{"family": "serif", "serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif", "serif"]})

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



    if visualize:
        print('generating lossless video ...')

        output_name = dataset_name.split('fly_N9_')[1] if 'fly_N9_' in dataset_name else 'no_id'
        src = f"./{log_dir}/tmp_recons/Fig_0_000000.png"
        dst = f"./{log_dir}/results/input_{output_name}.png"
        with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
            fdst.write(fsrc.read())

        generate_compressed_video_mp4(output_dir=f"./{log_dir}/results", run=run,
                                        output_name=output_name,framerate=20)

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
    else:
        # Use voltage (index 3)
        activity_true = x_generated_list[:, :, 3].squeeze().T
        visual_input_true = x_generated_list[:, :, 4].squeeze().T
        activity_true_modified = x_generated_modified_list[:, :, 3].squeeze().T
        activity_pred = x_list[:, :, 3].squeeze().T


    start_frame = 0
    end_frame = target_frames


    if training_selected_neurons:           # MLP, RNN and ODE are trained on limted number of neurons

        print(f"evaluating on selected neurons only: {selected_neuron_ids}")
        x_generated_list = x_generated_list[:, selected_neuron_ids, :]
        x_generated_modified_list = x_generated_modified_list[:, selected_neuron_ids, :]
        neuron_types = neuron_types[selected_neuron_ids]

        true_slice = activity_true[selected_neuron_ids, start_frame:end_frame]
        visual_input_slice = visual_input_true[selected_neuron_ids, start_frame:end_frame]
        pred_slice = activity_pred[start_frame:end_frame]

        rmse_all, pearson_all, feve_all, r2_all = compute_trace_metrics(true_slice, pred_slice, "selected neurons")

        # Log rollout metrics to file
        rollout_log_path = f"./{log_dir}/results_rollout.log"
        with open(rollout_log_path, 'w') as f:
            f.write("Rollout Metrics for Selected Neurons\n")
            f.write("="*60 + "\n")
            f.write(f"RMSE: {np.mean(rmse_all):.4f} ± {np.std(rmse_all):.4f} [{np.min(rmse_all):.4f}, {np.max(rmse_all):.4f}]\n")
            f.write(f"Pearson r: {np.nanmean(pearson_all):.3f} ± {np.nanstd(pearson_all):.3f} [{np.nanmin(pearson_all):.3f}, {np.nanmax(pearson_all):.3f}]\n")
            f.write(f"R²: {np.nanmean(r2_all):.3f} ± {np.nanstd(r2_all):.3f} [{np.nanmin(r2_all):.3f}, {np.nanmax(r2_all):.3f}]\n")
            f.write(f"FEVE: {np.mean(feve_all):.3f} ± {np.std(feve_all):.3f} [{np.min(feve_all):.3f}, {np.max(feve_all):.3f}]\n")
            f.write(f"\nNumber of neurons evaluated: {len(selected_neuron_ids)}\n")

        if len(selected_neuron_ids)==1:
            pred_slice = pred_slice[None,:]

        filename_ = dataset_name.split('fly_N9_')[1] if 'fly_N9_' in dataset_name else 'no_id'

        # Determine which figures to create
        if len(selected_neuron_ids) > 50:
            # Create sample: take the last 10 neurons from selected_neuron_ids
            sample_indices = list(range(len(selected_neuron_ids) - 10, len(selected_neuron_ids)))

            figure_configs = [
                ("all", list(range(len(selected_neuron_ids)))),
                ("sample", sample_indices)
            ]
        else:
            figure_configs = [("", list(range(len(selected_neuron_ids))))]

        for fig_suffix, neuron_plot_indices in figure_configs:
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))

            step_v = 2.5
            lw = 6

            # Adjust fontsize based on number of neurons being plotted
            name_fontsize = 10 if len(neuron_plot_indices) > 50 else 18

            # Plot ground truth (green, thick)
            for plot_idx, i in enumerate(trange(len(neuron_plot_indices), ncols=100, desc=f"plotting {fig_suffix}")):
                neuron_idx = neuron_plot_indices[i]
                baseline = np.mean(true_slice[neuron_idx])
                ax.plot(true_slice[neuron_idx] - baseline + plot_idx * step_v, linewidth=lw, c='green', alpha=0.9,
                        label='ground truth' if plot_idx == 0 else None)
                # Plot visual input only for neuron_id = 0
                if ((selected_neuron_ids[neuron_idx] == 0) | (len(neuron_plot_indices) < 50)) and visual_input_slice[neuron_idx].mean() > 0:
                    ax.plot(visual_input_slice[neuron_idx] - baseline + plot_idx * step_v, linewidth=1, c='yellow', alpha=0.9,
                            linestyle='--', label='visual input')
                ax.plot(pred_slice[neuron_idx] - baseline + plot_idx * step_v, linewidth=1, c=mc,
                        label='prediction' if plot_idx == 0 else None)

            for plot_idx, i in enumerate(neuron_plot_indices):
                type_idx = int(to_numpy(x[selected_neuron_ids[i], 6]).item())
                # Color code R²: red if <0.5, orange if <0.8, white otherwise
                r2_color = 'red' if r2_all[i] < 0.5 else ('orange' if r2_all[i] < 0.8 else 'white')
                ax.text(-50, plot_idx * step_v, f'{index_to_name[type_idx]}', fontsize=name_fontsize, va='bottom', ha='right', color=r2_color)
                ax.text(end_frame - start_frame + 20, plot_idx * step_v, f'$R^2$: {r2_all[i]:.2f}', fontsize=10, va='center', ha='left', color=r2_color)
                if len(neuron_plot_indices) <= 20:
                    ax.text(-50, plot_idx * step_v - 0.3, f'{selected_neuron_ids[i]}',
                            fontsize=12, va='top', ha='right', color='black')

            ax.set_ylim([-step_v, len(neuron_plot_indices) * (step_v + 0.25 + 0.15 * (len(neuron_plot_indices)//50))])
            ax.set_yticks([])
            ax.set_xlabel('frame', fontsize=20)
            ax.set_xticks([0, (end_frame - start_frame) // 2, end_frame - start_frame])
            ax.set_xticklabels([start_frame, end_frame//2, end_frame], fontsize=16)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)

            ax.legend(loc='upper right', fontsize=14)
            ax.set_xlim([0, end_frame - start_frame + 100])

            plt.tight_layout()
            save_suffix = f"_{fig_suffix}" if fig_suffix else ""
            plt.savefig(f"./{log_dir}/results/rollout_{filename_}_{simulation_config.visual_input_type}{save_suffix}.png", dpi=300, bbox_inches='tight')
            plt.close()

    else:

        rmse_all, pearson_all, feve_all, r2_all = compute_trace_metrics(activity_true, activity_pred, "all neurons")

        # Log rollout metrics to file
        rollout_log_path = f"./{log_dir}/results_rollout.log"
        with open(rollout_log_path, 'w') as f:
            f.write("Rollout Metrics for All Neurons\n")
            f.write("="*60 + "\n")
            f.write(f"RMSE: {np.mean(rmse_all):.4f} ± {np.std(rmse_all):.4f} [{np.min(rmse_all):.4f}, {np.max(rmse_all):.4f}]\n")
            f.write(f"Pearson r: {np.nanmean(pearson_all):.3f} ± {np.nanstd(pearson_all):.3f} [{np.nanmin(pearson_all):.3f}, {np.nanmax(pearson_all):.3f}]\n")
            f.write(f"R²: {np.nanmean(r2_all):.3f} ± {np.nanstd(r2_all):.3f} [{np.nanmin(r2_all):.3f}, {np.nanmax(r2_all):.3f}]\n")
            f.write(f"FEVE: {np.mean(feve_all):.3f} ± {np.std(feve_all):.3f} [{np.min(feve_all):.3f}, {np.max(feve_all):.3f}]\n")
            f.write(f"\nNumber of neurons evaluated: {len(activity_true)}\n")
            f.write(f"Frames evaluated: {start_frame} to {end_frame}\n")

        filename_ = dataset_name.split('fly_N9_')[1] if 'fly_N9_' in dataset_name else 'no_id'

        # Create two figures with different neuron type selections
        for fig_name, selected_types in [
            ("selected", [55, 15, 43, 39, 35, 31, 23, 19, 12, 5]),  # L1, Mi12, Mi2, R1, T1, T4a, T5a, Tm1, Tm4, Tm9
            ("all", np.arange(0, n_neuron_types))
        ]:
            neuron_indices = []
            for stype in selected_types:
                indices = np.where(neuron_types == stype)[0]
                if len(indices) > 0:
                    neuron_indices.append(indices[0])

            fig, ax = plt.subplots(1, 1, figsize=(15, 10))

            true_slice = activity_true[neuron_indices, start_frame:end_frame]
            visual_input_slice = visual_input_true[neuron_indices, start_frame:end_frame]
            pred_slice = activity_pred[neuron_indices, start_frame:end_frame]
            step_v = 2.5
            lw = 6

            # Adjust fontsize based on number of neurons
            name_fontsize = 10 if len(selected_types) > 50 else 18

            for i in range(len(neuron_indices)):
                baseline = np.mean(true_slice[i])
                ax.plot(true_slice[i] - baseline + i * step_v, linewidth=lw, c='green', alpha=0.9,
                        label='ground truth' if i == 0 else None)
                # Plot visual input for neuron 0 OR when fewer than 50 neurons
                if ((neuron_indices[i] == 0) | (len(neuron_indices) < 50)) and visual_input_slice[i].mean() > 0:
                    ax.plot(visual_input_slice[i] - baseline + i * step_v, linewidth=1, c='yellow', alpha=0.9,
                            linestyle='--', label='visual input')
                ax.plot(pred_slice[i] - baseline + i * step_v, linewidth=1, label='prediction' if i == 0 else None, c=mc)


            for i in range(len(neuron_indices)):
                type_idx = selected_types[i]
                # Color code R²: red if <0.5, orange if <0.8, white otherwise
                r2_color = 'red' if r2_all[neuron_indices[i]] < 0.5 else ('orange' if r2_all[neuron_indices[i]] < 0.8 else 'white')
                ax.text(-50, i * step_v, f'{index_to_name[type_idx]}', fontsize=name_fontsize, va='bottom', ha='right', color=r2_color)
                ax.text(end_frame - start_frame + 20, i * step_v, f'$R^2$: {r2_all[neuron_indices[i]]:.2f}', fontsize=10, va='center', ha='left', color=r2_color)

            ax.set_ylim([-step_v, len(neuron_indices) * (step_v + 0.25 + 0.15 * (len(neuron_indices)//50))])
            ax.set_yticks([])
            ax.set_xticks([0, (end_frame - start_frame) // 2, end_frame - start_frame])
            ax.set_xticklabels([start_frame, end_frame//2, end_frame], fontsize=16)
            ax.set_xlabel('frame', fontsize=20)
            ax.set_xlim([-50, end_frame - start_frame + 100])

            print([-50, end_frame - start_frame + 100])

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)

            ax.legend(loc='upper right', fontsize=14)

            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/rollout_{filename_}_{simulation_config.visual_input_type}_{fig_name}.png", dpi=300, bbox_inches='tight')
            plt.close()

        if ('test_ablation' in test_mode) or ('test_inactivity' in test_mode):
            np.save(f"./{log_dir}/results/activity_modified.npy", activity_true_modified)
            np.save(f"./{log_dir}/results/activity_modified_pred.npy", activity_pred)
        else:
            np.save(f"./{log_dir}/results/activity_true.npy", activity_true)
            np.save(f"./{log_dir}/results/activity_pred.npy", activity_pred)



def data_test_zebra(config, visualize, style, verbose, best_model, step, test_mode, device):

    dataset_name = config.dataset
    simulation_config = config.simulation
    model_config = config.graph_model
    training_config = config.training

    n_neuron_types = simulation_config.n_neuron_types
    n_neurons = simulation_config.n_neurons
    n_runs = training_config.n_runs
    n_frames = simulation_config.n_frames
    plot_batch_size = config.plotting.plot_batch_size

    CustomColorMap(config=config)
    dimension = simulation_config.dimension


    torch.random.fork_rng(devices=device)
    torch.random.manual_seed(simulation_config.seed)

    ones = torch.ones((n_neurons, 1), dtype=torch.float32, device=device)

    run = 0

    if 'latex' in style:
        print('latex style...')
        plt.rcParams['text.usetex'] = False  # LaTeX disabled - use mathtext instead
        rc('font', **{'family': 'serif', 'serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif', 'serif']})
    if 'black' in style:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')


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
    torch.load(f'{log_dir}/ynorm.pt', map_location=device, weights_only=True)
    torch.load(f'{log_dir}/vnorm.pt', map_location=device, weights_only=True)


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
            output_name="zebra",
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

    print(f"grand average MAE: {mean_mae[-1]:.4f} +/- {sem_mae[-1]:.4f} (N={counts[-1]})")


    reconstructed = reconstructed.squeeze()
    true = true.squeeze()


