import os
import umap
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter
from matplotlib.animation import FFMpegWriter
from torch_geometric.loader import DataLoader
import torch_geometric.data as data
import imageio.v2 as imageio
from matplotlib import rc
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture
from shutil import copyfile
from collections import defaultdict
import scipy
import logging
import re
import matplotlib

# os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2023/bin/x86_64-linux'

# from data_loaders import *


from NeuralGraph.fitting_models import linear_model
from NeuralGraph.sparsify import EmbeddingCluster, sparsify_cluster, clustering_gmm
from NeuralGraph.models.utils import (
    choose_training_model,
    get_in_features,
    get_in_features_update,
    get_index_particles,
    analyze_odor_responses_by_neuron,
    plot_odor_heatmaps,
)
from NeuralGraph.models.plot_utils import (
    analyze_mlp_edge_lines,
    analyze_mlp_edge_lines_weighted_with_max,
    analyze_mlp_phi_synaptic,
    find_top_responding_pairs,
    run_neural_architecture_pipeline,
)
from NeuralGraph.utils import (
    to_numpy,
    CustomColorMap,
    sort_key,
    fig_init,
    get_equidistant_points,
    map_matrix,
    create_log_dir,
    find_suffix_pairs_with_index,
    add_pre_folder
)
from NeuralGraph.models.Siren_Network import Siren, Siren_Network
from NeuralGraph.models.Signal_Propagation_FlyVis import Signal_Propagation_FlyVis
from NeuralGraph.models.Signal_Propagation_Zebra import Signal_Propagation_Zebra
from NeuralGraph.models.Calcium_Latent_Dynamics import Calcium_Latent_Dynamics
from NeuralGraph.models.graph_trainer import data_test
from NeuralGraph.generators.utils import choose_model
from NeuralGraph.config import NeuralGraphConfig

from NeuralGraph.models.Ising_analysis import analyze_ising_model

from scipy import stats
from io import StringIO
import sys
import warnings
import seaborn as sns
import glob
import numpy as np
import pickle
import json
from tqdm import tqdm, trange
import time
from sklearn import metrics
from tifffile import imread


import matplotlib.ticker as ticker
import shutil

# Optional dependency
# try:
#     from pysr import PySRRegressor

# except ImportError:
#     PySRRegressor = None


def get_training_files(log_dir, n_runs):
    files = glob.glob(f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs_*.pt")
    files.sort(key=sort_key)
    flag = True
    file_id = 0
    while (flag):
        if sort_key(files[file_id]) > 0:
            flag = False
            file_id = file_id - 1
        file_id += 1

    files = files[file_id:]

    # Filter based on the Y value (number after "graphs")
    files_with_0 = [file for file in files if int(file.split('_')[-2]) == 0]
    files_without_0 = [file for file in files if int(file.split('_')[-2]) != 0]

    # Generate 50 evenly spaced indices for each list

    # indices_with_0 = np.linspace(0, len(files_with_0) - 1, dtype=int)
    indices_with_0 = np.arange(0, len(files_with_0) - 1, dtype=int)
    indices_without_0 = np.linspace(0, len(files_without_0) - 1, 50, dtype=int)

    # Select the files using the generated indices
    selected_files_with_0 = [files_with_0[i] for i in indices_with_0]
    if len(files_without_0) > 0:
        selected_files_without_0 = [files_without_0[i] for i in indices_without_0]
        selected_files = selected_files_with_0 + selected_files_without_0
    else:
        selected_files = selected_files_with_0

    return selected_files, np.arange(0, len(selected_files), 1)

    # len_files = len(files)
    # print(len_files, len_files//10, len_files//500, len_files//10, len_files, len_files//50)
    # file_id_list0 = np.arange(0, len_files//10, len_files//500)
    # file_id_list1 = np.arange(len_files//10, len_files, len_files//50)
    # file_id_list = np.concatenate((file_id_list0, file_id_list1))
    # # file_id_list = np.arange(0, len(files), (len(files) / 100)).astype(int)
    # return files, file_id_list


def load_training_data(dataset_name, n_runs, log_dir, device):
    x_list = []
    y_list = []
    print('load data ...')
    time.sleep(0.5)
    for run in trange(n_runs, ncols=90):
        # check if path exists
        if os.path.exists(f'graphs_data/{dataset_name}/x_list_{run}.pt'):
            x = torch.load(f'graphs_data/{dataset_name}/x_list_{run}.pt', map_location=device)
            y = torch.load(f'graphs_data/{dataset_name}/y_list_{run}.pt', map_location=device)
        else:
            x = np.load(f'graphs_data/{dataset_name}/x_list_{run}.npy')
            x = torch.tensor(x, dtype=torch.float32, device=device)
            y = np.load(f'graphs_data/{dataset_name}/y_list_{run}.npy')
            y = torch.tensor(y, dtype=torch.float32, device=device)

        x_list.append(x)
        y_list.append(y)
    vnorm = torch.load(os.path.join(log_dir, 'vnorm.pt'), map_location=device).squeeze()
    ynorm = torch.load(os.path.join(log_dir, 'ynorm.pt'), map_location=device).squeeze()
    print("vnorm:{:.2e},  ynorm:{:.2e}".format(to_numpy(vnorm), to_numpy(ynorm)))
    x = []
    y = []

    return x_list, y_list, vnorm, ynorm


def plot_confusion_matrix(index, true_labels, new_labels, n_neuron_types, epoch, it, fig, ax, style):
    # print(f'plot confusion matrix epoch:{epoch} it: {it}')
    plt.text(-0.25, 1.1, f'{index}', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    confusion_matrix = metrics.confusion_matrix(true_labels, new_labels)  # , normalize='true')
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    if n_neuron_types > 8:
        cm_display.plot(ax=fig.gca(), cmap='Blues', include_values=False, colorbar=False)
    else:
        cm_display.plot(ax=fig.gca(), cmap='Blues', include_values=True, values_format='d', colorbar=False)
    accuracy = metrics.accuracy_score(true_labels, new_labels)
    plt.title(f'accuracy: {np.round(accuracy, 2)}', fontsize=12)
    # print(f'accuracy: {np.round(accuracy,3)}')
    plt.xticks(fontsize=10.0)
    plt.yticks(fontsize=10.0)
    plt.xlabel(r'Predicted label', fontsize=12)
    plt.ylabel(r'True label', fontsize=12)

    return accuracy




def plot_signal(config, epoch_list, log_dir, logger, cc, style, extended, device):

    dataset_name = config.dataset

    train_config = config.training
    model_config = config.graph_model

    n_frames = config.simulation.n_frames
    n_runs = config.training.n_runs
    n_neuron_types = config.simulation.n_neuron_types
    delta_t = config.simulation.delta_t
    p = config.simulation.params
    omega = model_config.omega
    cmap = CustomColorMap(config=config)
    dimension = config.simulation.dimension
    max_radius = config.simulation.max_radius
    embedding_cluster = EmbeddingCluster(config)
    field_type = model_config.field_type
    if field_type != '':
        n_nodes = config.simulation.n_nodes
        n_nodes_per_axis = int(np.sqrt(n_nodes))
        has_field = True
    else:
        has_field = False
    n_ghosts = int(train_config.n_ghosts)
    has_ghost = n_ghosts > 0

    x_list = []
    y_list = []
    for run in trange(1,2, ncols=90):
        if os.path.exists(f'graphs_data/{dataset_name}/x_list_{run}.pt'):
            x = torch.load(f'graphs_data/{dataset_name}/x_list_{run}.pt', map_location=device)
            y = torch.load(f'graphs_data/{dataset_name}/y_list_{run}.pt', map_location=device)
            x = to_numpy(torch.stack(x))
            y = to_numpy(torch.stack(y))

        else:
            x = np.load(f'graphs_data/{dataset_name}/x_list_{run}.npy')
            y = np.load(f'graphs_data/{dataset_name}/y_list_{run}.npy')
            if os.path.exists(f'graphs_data/{dataset_name}/raw_x_list_{run}.npy'):
                raw_x = np.load(f'graphs_data/{dataset_name}/raw_x_list_{run}.npy')
        x_list.append(x)
        y_list.append(y)
    vnorm = torch.load(os.path.join(log_dir, 'vnorm.pt'))
    ynorm = torch.load(os.path.join(log_dir, 'ynorm.pt'))
    if os.path.exists(os.path.join(log_dir, 'xnorm.pt')):
        xnorm = torch.load(os.path.join(log_dir, 'xnorm.pt'))
    else:
        xnorm = torch.tensor([5], device=device)
    print(f'xnorm: {to_numpy(xnorm)}, vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')

    print('update variables ...')
    x = x_list[0][n_frames - 5]
    n_neurons = x.shape[0]
    print(f'N neurons: {n_neurons}')
    logger.info(f'N neurons: {n_neurons}')
    config.simulation.n_neurons = n_neurons
    type_list = torch.tensor(x[:, 1 + 2 * dimension:2 + 2 * dimension], device=device)

    activity = torch.tensor(x_list[0][:, :, 6:7],device=device)
    activity = activity.squeeze()
    to_numpy(activity.flatten())
    activity = activity.t()

    if os.path.exists(f'graphs_data/{dataset_name}/raw_x_list_{run}.npy'):
        raw_activity = torch.tensor(raw_x[:, :, 6:7], device=device)
        raw_activity = raw_activity.squeeze()
        raw_activity = raw_activity.t()

    xc, yc = get_equidistant_points(n_points=n_neurons)
    X1_first = torch.tensor(np.stack((xc, yc), axis=1), dtype=torch.float32, device=device) / 2
    perm = torch.randperm(X1_first.size(0))
    X1_first = X1_first[perm]
    torch.save(X1_first, f'./graphs_data/{dataset_name}/X1.pt')
    xc, yc = get_equidistant_points(n_points=n_neurons ** 2)
    X_msg = torch.tensor(np.stack((xc, yc), axis=1), dtype=torch.float32, device=device) / 2
    perm = torch.randperm(X_msg.size(0))
    X_msg = X_msg[perm]
    torch.save(X_msg, f'./graphs_data/{dataset_name}/X_msg.pt')

    if 'black' in style:
        mc = 'w'
    else:
        mc = 'k'

    if has_ghost:
        model_missing_activity = nn.ModuleList([
            Siren(in_features=model_config.input_size_nnr, out_features=model_config.output_size_nnr,
                  hidden_features=model_config.hidden_dim_nnr,
                  hidden_layers=model_config.n_layers_nnr, first_omega_0=omega, hidden_omega_0=omega,
                  outermost_linear=model_config.outermost_linear_nnr)
            for n in range(n_runs)
        ])
        model_missing_activity.to(device=device)
        model_missing_activity.eval()

    if has_field:
        if ('short_term_plasticity' in field_type) | ('modulation_permutation' in field_type):
            model_f = Siren(in_features=model_config.input_size_nnr, out_features=model_config.output_size_nnr,
                            hidden_features=model_config.hidden_dim_nnr,
                            hidden_layers=model_config.n_layers_nnr, first_omega_0=omega, hidden_omega_0=omega,
                            outermost_linear=model_config.outermost_linear_nnr)
        else:
            model_f = Siren_Network(image_width=n_nodes_per_axis, in_features=model_config.input_size_nnr, out_features=model_config.output_size_nnr, hidden_features=model_config.hidden_dim_nnr,
                                            hidden_layers=model_config.n_layers_nnr, outermost_linear=True, device=device, first_omega_0=omega, hidden_omega_0=omega)
        model_f.to(device=device)
        model_f.train()

        modulation = torch.tensor(x_list[0], device=device)
        modulation = modulation[:, :, 8:9].squeeze()
        modulation = modulation.t()
        modulation = modulation.clone().detach()
        (modulation[:, 1:] - modulation[:, :-1]) / delta_t

    if epoch_list[0] == 'all':
        files = glob.glob(f"{log_dir}/models/*.pt")
        files.sort(key=os.path.getmtime)

        model, bc_pos, bc_dpos = choose_training_model(config, device)

        # plt.rcParams['text.usetex'] = False
        # plt.rc('font', family='sans-serif')
        # plt.rc('text', usetex=False)
        # matplotlib.rcParams['savefig.pad_inches'] = 0

        files, file_id_list = get_training_files(log_dir, n_runs)

        r_squared_list = []
        slope_list = []
        it = -1
        with torch.no_grad():
            for file_id_ in trange(0,len(file_id_list), ncols=90):
                it = it + 1
                num = str(it).zfill(4)
                file_id = file_id_list[file_id_]
                epoch = files[file_id].split('graphs')[1][1:-3]
                net = f"{log_dir}/models/best_model_with_{n_runs-1}_graphs_{epoch}.pt"

                state_dict = torch.load(net, map_location=device)
                model.load_state_dict(state_dict['model_state_dict'])
                # if train_config.with_connectivity_mask:
                #     inv_mask = torch.load(f'./graphs_data/{dataset_name}/inv_mask.pt', map_location=device)
                #     with torch.no_grad():
                #         model.W.copy_(model.W * inv_mask)
                model.eval()

                if has_field:
                    net = f'{log_dir}/models/best_model_f_with_{n_runs-1}_graphs_{epoch}.pt'
                    state_dict = torch.load(net, map_location=device)
                    model_f.load_state_dict(state_dict['model_state_dict'])

                amax = torch.max(model.a, dim=0).values
                amin = torch.min(model.a, dim=0).values
                model_a = (model.a - amin) / (amax - amin)

                fig, ax = fig_init()
                for n in range(n_neuron_types-1,-1,-1):
                    pos = torch.argwhere(type_list == n).squeeze()
                    plt.scatter(to_numpy(model_a[pos, 0]), to_numpy(model_a[pos, 1]), s=50, color=cmap.color(n), alpha=1.0, edgecolors='none')   # cmap.color(n)
                plt.xlabel(r'$\mathbf{a}_0$', fontsize=68)
                plt.ylabel(r'$\mathbf{a}_1$', fontsize=68)
                plt.xlim([-0.1, 1.1])
                plt.ylim([-0.1, 1.1])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/embedding_{num}.png", dpi=80)
                plt.close()

                if os.path.exists(f'{log_dir}/correction.pt'):
                    correction = torch.load(f'{log_dir}/correction.pt',map_location=device)
                    second_correction = np.load(f'{log_dir}/second_correction.npy')
                else:
                    correction = torch.tensor(1.0, device=device)
                    second_correction = 1.0


                i, j = torch.triu_indices(n_neurons, n_neurons, requires_grad=False, device=device)
                A = model.W.clone().detach() / correction
                A[i, i] = 0
                A = A.t()

                fig, ax = fig_init()
                ax = sns.heatmap(to_numpy(A)/second_correction, center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046}, vmin=-0.1,vmax=0.1)
                cbar = ax.collections[0].colorbar
                cbar.ax.tick_params(labelsize=48)
                plt.xticks([0, n_neurons - 1], [1, n_neurons], fontsize=24)
                plt.yticks([0, n_neurons - 1], [1, n_neurons], fontsize=24)
                plt.subplot(2, 2, 1)
                ax = sns.heatmap(to_numpy(A[0:20, 0:20])/second_correction, cbar=False, center=0, square=True, cmap='bwr', vmin=-0.1, vmax=0.1)
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/W_{num}.png", dpi=80)
                plt.close()


                rr = torch.linspace(-xnorm.squeeze(), xnorm.squeeze(), 1000).to(device)
                if model_config.signal_model_name == 'PDE_N5':
                    fig, ax = fig_init()
                    plt.axis('off')
                    for k in range(n_neuron_types):
                        ax = fig.add_subplot(2, 2, k + 1)
                        for spine in ax.spines.values():
                            spine.set_edgecolor(cmap.color(k))  # Set the color of the outline
                            spine.set_linewidth(3)
                        if k==0:
                            plt.ylabel(r'learned $\mathrm{MLP_1}( a_i, a_j, v_j)$', fontsize=32)
                        for n in range(n_neuron_types):
                            for m in range(250):
                                pos0 = to_numpy(torch.argwhere(type_list == k).squeeze())
                                pos1 = to_numpy(torch.argwhere(type_list == n).squeeze())
                                n0 = np.random.randint(len(pos0))
                                n0 = pos0[n0, 0]
                                n1 = np.random.randint(len(pos1))
                                n1 = pos1[n1, 0]
                                embedding0 = model.a[n0, :] * torch.ones((1000, config.graph_model.embedding_dim),
                                                                         device=device)
                                embedding1 = model.a[n1, :] * torch.ones((1000, config.graph_model.embedding_dim),
                                                                         device=device)
                                in_features = torch.cat((rr[:, None], embedding0, embedding1), dim=1)
                                func = model.lin_edge(in_features.float()) * correction
                                plt.plot(to_numpy(rr), to_numpy(func), 2, color=cmap.color(n), linewidth=3, alpha=0.25)
                        plt.ylim([-1.6, 1.6])
                        plt.xlim([-5, 5])
                        plt.xticks([])
                        plt.yticks([])
                    plt.xlabel(r'$x_j$', fontsize=68)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/all/MLP1_{num}.png", dpi=80)
                    plt.close()
                elif (model_config.signal_model_name == 'PDE_N4'):
                    fig, ax = fig_init()
                    for k in range(n_neuron_types):
                        for m in range(250):
                            pos0 = to_numpy(torch.argwhere(type_list == k).squeeze())
                            n0 = np.random.randint(len(pos0))
                            n0 = pos0[n0, 0]
                            embedding0 = model.a[n0, :] * torch.ones((1000, config.graph_model.embedding_dim),
                                                                     device=device)
                            # in_features = torch.cat((rr[:, None], embedding0), dim=1)
                            in_features = get_in_features(rr, embedding0, model_config.signal_model_name, max_radius)
                            if config.graph_model.lin_edge_positive:
                                func = model.lin_edge(in_features.float()) ** 2 * correction
                            else:
                                func = model.lin_edge(in_features.float()) * correction
                            plt.plot(to_numpy(rr), to_numpy(func), color=cmap.color(k), linewidth=2, alpha=0.25)
                    plt.xlabel(r'$x_j$', fontsize=68)
                    plt.ylabel(r'learned $\mathrm{MLP_1}(\mathbf{a}_j, v_j)$', fontsize=68)
                    if config.graph_model.lin_edge_positive:
                        plt.ylim([-0.2, 1.2])
                    else:
                        plt.ylim([-1.6, 1.6])
                    plt.xlim([-to_numpy(xnorm)//2, to_numpy(xnorm)//2])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/all/MLP1_{num}.png", dpi=80)
                    plt.close()
                elif (model_config.signal_model_name == 'PDE_N8'):
                    rr = torch.linspace(0, 10, 1000).to(device)
                    fig, ax = fig_init()
                    for idx, k in enumerate(np.linspace(4, 10, 13)):  # Corrected step size to generate 13 evenly spaced values

                        for n in range(0,n_neurons,4):
                            embedding_i = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                            embedding_j = model.a[np.random.randint(n_neurons), :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                            if model.embedding_trial:
                                in_features = torch.cat((torch.ones_like(rr[:, None]) * k, rr[:, None], embedding_i, embedding_j, model.b[0].repeat(1000, 1)), dim=1)
                            else:
                                in_features = torch.cat((rr[:, None], torch.ones_like(rr[:, None])*k, embedding_i, embedding_j), dim=1)
                            with torch.no_grad():
                                func = model.lin_edge(in_features.float())
                            if config.graph_model.lin_edge_positive:
                                func = func ** 2
                            plt.plot(to_numpy(rr-k), to_numpy(func), 2, color=cmap.color(idx), linewidth=2, alpha=0.25)
                    plt.xlabel(r'$x_i-x_j$', fontsize=68)
                    # plt.ylabel(r'learned $\psi^*(\mathbf{a}_i, v_i)$', fontsize=68)
                    plt.ylabel(r'$\mathrm{MLP_1}(\mathbf{a}_i, a_j, v_i, v_j)$', fontsize=68)
                    plt.ylim([0,4])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/all/MLP1_{num}.png", dpi=80)
                    plt.close()
                else:
                    fig, ax = fig_init()
                    in_features = rr[:, None]
                    with torch.no_grad():
                        func = model.lin_edge(in_features.float()) * correction
                    if config.graph_model.lin_edge_positive:
                        func = func ** 2
                    plt.plot(to_numpy(rr), to_numpy(func), color=mc, linewidth=8, label=r'learned')
                    plt.xlabel(r'$x_j$', fontsize=68)
                    # plt.ylabel(r'learned $\psi^*(\mathbf{a}_i, v_i)$', fontsize=68)
                    plt.ylabel(r'learned $\mathrm{MLP_1}(\mathbf{a}_j, v_j)$', fontsize=68)
                    plt.ylim([-1.5, 1.5])
                    plt.xlim([-5,5])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/all/MLP1_{num}.png", dpi=80)
                    plt.close()

                fig, ax = fig_init()
                func_list = []
                for n in range(n_neurons):
                    embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                    # in_features = torch.cat((rr[:, None], embedding_), dim=1)
                    in_features = get_in_features_update(rr[:, None], model, embedding_, device)
                    with torch.no_grad():
                        func = model.lin_phi(in_features.float())
                    func = func[:, 0]
                    plt.plot(to_numpy(rr), to_numpy(func) * to_numpy(ynorm), color=cmap.color(to_numpy(type_list[n]).astype(int)), linewidth=2, alpha=0.25) #
                plt.ylim([-4,4])
                plt.xlabel(r'$v_i$', fontsize=68)
                # plt.ylabel(r'learned $\phi^*(\mathbf{a}_i, v_i)$', fontsize=68)
                plt.ylabel(r'learned $\mathrm{MLP_0}(\mathbf{a}_i, v_i)$', fontsize=68)

                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/MLP0_{num}.png", dpi=80)
                plt.close()

                connectivity = torch.load(f'./graphs_data/{dataset_name}/connectivity.pt', map_location=device)

                adjacency = connectivity.t().clone().detach()
                adj_t = torch.abs(adjacency) > 0
                edge_index = adj_t.nonzero().t().contiguous()

                i, j = torch.triu_indices(n_neurons, n_neurons, requires_grad=False, device=device)
                A = model.W.clone().detach() / correction
                A[i, i] = 0

                fig, ax = fig_init()
                gt_weight = to_numpy(adjacency)
                pred_weight = to_numpy(A) / second_correction
                plt.scatter(gt_weight, pred_weight, s=0.1, c=mc, alpha=0.1)
                plt.xlabel(r'true $W_{ij}$', fontsize=68)
                plt.ylabel(r'learned $W_{ij}$', fontsize=68)
                if n_neurons == 8000:
                    plt.xlim([-0.05, 0.05])
                    plt.ylim([-0.05, 0.05])
                else:
                    # plt.xlim([-0.2, 0.2])
                    # plt.ylim([-0.2, 0.2])
                    plt.xlim([-0.15, 0.15])
                    plt.ylim([-0.15, 0.15])

                x_data = np.reshape(gt_weight, (n_neurons * n_neurons))
                y_data = np.reshape(pred_weight, (n_neurons * n_neurons))
                lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                residuals = y_data - linear_model(x_data, *lin_fit)
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                r_squared_list.append(r_squared)
                slope_list.append(lin_fit[0])

                if n_neurons == 8000:
                    plt.text(-0.042, 0.042, f'$R^2$: {np.round(r_squared, 3)}', fontsize=34)
                    plt.text(-0.042, 0.036, f'slope: {np.round(lin_fit[0], 2)}', fontsize=34)
                else:
                    # plt.text(-0.17, 0.15, f'$R^2$: {np.round(r_squared, 3)}', fontsize=34)
                    # plt.text(-0.17, 0.12, f'slope: {np.round(lin_fit[0], 2)}', fontsize=34)
                    plt.text(-0.13, 0.13, f'$R^2$: {np.round(r_squared, 3)}', fontsize=34)
                    plt.text(-0.13, 0.11, f'slope: {np.round(lin_fit[0], 2)}', fontsize=34)

                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/comparison_{num}.png", dpi=80)
                plt.close()

                if has_field:

                    if 'short_term_plasticity' in field_type:
                        fig, ax = fig_init()
                        t = torch.linspace(1, 100000, 1, dtype=torch.float32, device=device).unsqueeze(1)
                        prediction = model_f(t) ** 2
                        prediction = prediction.t()
                        plt.imshow(to_numpy(prediction), aspect='auto', cmap='gray')
                        plt.title(r'learned $FMLP(t)_i$', fontsize=68)
                        plt.xlabel(r'$t$', fontsize=68)
                        plt.ylabel(r'$i$', fontsize=68)
                        plt.xticks([10000,100000], [10000, 100000], fontsize=48)
                        plt.yticks([0, 512, 1024], [0, 512, 1024], fontsize=48)
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/results/all/yi_{num}.png", dpi=80)
                        plt.close()

                        prediction = prediction * torch.tensor(second_correction,device=device) / 10

                        fig, ax = fig_init()
                        ids = np.arange(0,100000,100).astype(int)
                        plt.scatter(to_numpy(modulation[:,ids]), to_numpy(prediction[:,ids]), s=1, color=mc, alpha=0.05)
                        # plt.xlim([0,0.5])
                        # plt.ylim([0,2])
                        # plt.xticks([0,0.5], [0,0.5], fontsize=48)
                        # plt.yticks([0,1,2], [0,1,2], fontsize=48)
                        x_data = to_numpy(modulation[:,ids]).flatten()
                        y_data = to_numpy(prediction[:,ids]).flatten()
                        lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                        residuals = y_data - linear_model(x_data, *lin_fit)
                        ss_res = np.sum(residuals ** 2)
                        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot)
                        ax.text(0.05, 0.94, f'$R^2$: {r_squared:0.2f}', transform=ax.transAxes,
                                verticalalignment='top', horizontalalignment='left', fontsize=32)
                        ax.text(0.05, 0.88, f'slope: {lin_fit[0]:0.2f}', transform=ax.transAxes,
                                verticalalignment='top', horizontalalignment='left', fontsize=32)
                        plt.xlabel(r'true $y_i(t)$', fontsize=68)
                        plt.ylabel(r'learned $y_i(t)$', fontsize=68)
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/results/all/comparison_yi_{num}.png", dpi=80)
                        plt.close()

                    else:

                        fig, ax = fig_init()
                        pred = model_f(time=file_id_ / len(file_id_list), enlarge=True) ** 2
                        # pred = torch.reshape(pred, (n_nodes_per_axis, n_nodes_per_axis))
                        pred = torch.reshape(pred, (640, 640))
                        pred = to_numpy(torch.sqrt(pred))
                        pred = np.flipud(pred)
                        pred = np.rot90(pred, 1)
                        pred = np.fliplr(pred)
                        plt.imshow(pred, cmap='grey')
                        plt.ylabel(r'learned $FMLP(s_i, t)$', fontsize=68)
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/results/all/field_{num}.png", dpi=80)
                        plt.close()

                if 'derivative' in field_type:

                    y = torch.linspace(0, 1, 400)
                    x = torch.linspace(-6, 6, 400)
                    grid_y, grid_x = torch.meshgrid(y, x)
                    grid = torch.stack((grid_x, grid_y), dim=-1)
                    grid = grid.to(device)
                    pred_modulation = model.lin_modulation(grid) / 40
                    tau = 100
                    alpha = 0.02
                    true_derivative = (1 - grid_y) / tau - alpha * grid_y * torch.abs(grid_x)

                    fig, ax = fig_init()
                    plt.title(r'learned $MLP_2(x_i, y_i)$', fontsize=68)
                    plt.imshow(to_numpy(pred_modulation))
                    plt.xticks([])
                    plt.yticks([])
                    plt.xlabel(r'$v_i$', fontsize=68)
                    plt.ylabel(r'$y_i$', fontsize=68)
                    plt.tight_layout
                    plt.savefig(f"./{log_dir}/results/all/derivative_yi_{num}.png", dpi=80)
                    plt.close()

                    fig, ax = fig_init()
                    plt.scatter(to_numpy(true_derivative.flatten()), to_numpy(pred_modulation.flatten()), s=5, color=mc, alpha=0.1)
                    x_data = to_numpy(true_derivative.flatten())
                    y_data = to_numpy(pred_modulation.flatten())
                    lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                    residuals = y_data - linear_model(x_data, *lin_fit)
                    ss_res = np.sum(residuals ** 2)
                    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot)
                    ax.text(0.05, 0.94, f'$R^2$: {r_squared:0.2f}', transform=ax.transAxes,
                            verticalalignment='top', horizontalalignment='left', fontsize=32)
                    ax.text(0.05, 0.88, f'slope: {lin_fit[0]:0.2f}', transform=ax.transAxes,
                            verticalalignment='top', horizontalalignment='left', fontsize=32)
                    plt.xlabel(r'true $\dot{y_i}(t)$', fontsize=68)
                    plt.ylabel(r'learned $\dot{y_i}(t)$', fontsize=68)

                    # plt.xticks([-0.1, 0], [-0.1, 0], fontsize=48)
                    # plt.yticks([-0.1, 0], [-0.1, 0], fontsize=48)
                    # plt.xlim([-0.2,0.025])
                    # plt.ylim([-0.2,0.025])

                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/all/comparison_derivative_yi_{num}.png", dpi=80)
                    plt.close()

                if (model.update_type == 'generic') & (model_config.signal_model_name == 'PDE_N5'):

                    k = np.random.randint(n_frames - 50)
                    x = torch.tensor(x_list[0][k], device=device)

                    fig, ax = fig_init()
                    msg_list = []
                    u = torch.linspace(-xnorm.squeeze(), xnorm.squeeze(), 400).to(device)
                    for sample in range(n_neurons):
                        id0 = np.random.randint(0, n_neurons)
                        id1 = np.random.randint(0, n_neurons)
                        f = x[id0, 8:9]
                        embedding0 = model.a[id0, :] * torch.ones((400, config.graph_model.embedding_dim),
                                                                  device=device)
                        embedding1 = model.a[id1, :] * torch.ones((400, config.graph_model.embedding_dim),
                                                                  device=device)
                        in_features = torch.cat((u[:, None], embedding0, embedding1), dim=1)
                        msg = model.lin_edge(in_features.float()) ** 2 * correction
                        in_features = torch.cat((torch.zeros((400, 1), device=device), embedding0, msg,
                                                 f * torch.ones((400, 1), device=device)), dim=1)
                        plt.plot(to_numpy(u), to_numpy(msg), c=cmap.color(to_numpy(x[id0, 5]).astype(int)), linewidth=2, alpha=0.25)
                        # plt.scatter(to_numpy(u), to_numpy(model.lin_phi(in_features)), s=5, c='r', alpha=0.15)
                        # plt.scatter(to_numpy(u), to_numpy(f*msg), s=1, c='w', alpha=0.1)
                        msg_list.append(msg)
                    plt.tight_layout()
                    msg_list = torch.stack(msg_list).squeeze()
                    y_min, y_max = msg_list.min().item(), msg_list.max().item()
                    plt.xlabel(r'$v_i$', fontsize=68)
                    plt.ylabel(r'learned MLPs', fontsize=68)
                    plt.ylim([y_min - y_max/2, y_max * 1.5])
                    plt.tight_layout()
                    plt.savefig(f'./{log_dir}/results/all/MLP1_{num}.png', dpi=80)
                    plt.close()

                im0 = imageio.imread(f"./{log_dir}/results/all/comparison_{num}.png")
                im1 = imageio.imread(f"./{log_dir}/results/all/embedding_{num}.png")
                im2 = imageio.imread(f"./{log_dir}/results/all/MLP0_{num}.png")
                im3 = imageio.imread(f"./{log_dir}/results/all/MLP1_{num}.png")

                fig = plt.figure(figsize=(16, 16))
                plt.axis('off')
                plt.subplot(2, 2, 1)
                plt.axis('off')
                plt.imshow(im0)
                plt.xticks([])
                plt.yticks([])
                plt.subplot(2, 2, 2)
                plt.axis('off')
                plt.imshow(im1)
                plt.xticks([])
                plt.yticks([])
                plt.axis('off')
                plt.subplot(2, 2, 3)
                plt.axis('off')
                plt.imshow(im2)
                plt.xticks([])
                plt.yticks([])
                plt.subplot(2, 2, 4)
                plt.axis('off')
                plt.imshow(im3)
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()

                plt.savefig(f"./{log_dir}/results/training/fig_{num}.png", dpi=80)
                plt.close()



        fig, ax = fig_init(formatx='%.0f', formaty='%.2f')
        plt.plot(r_squared_list, linewidth=4, c=mc)
        plt.xlim([0, 100])
        plt.ylim([0, 1.1])
        plt.yticks(fontsize=48)
        plt.xticks([0, 100], [0, 20], fontsize=48)
        plt.ylabel('$R^2$', fontsize=64)
        plt.xlabel('epoch', fontsize=64)
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/results/R2.png', dpi=300)
        plt.close()
        np.save(f'./{log_dir}/results/R2.npy', r_squared_list)

        slope_list = np.array(slope_list) / p[0][0]
        fig, ax = fig_init(formatx='%.0f', formaty='%.2f')
        plt.plot(slope_list*10, linewidth=4, c=mc)
        plt.xlim([0, 100])
        plt.ylim([0, 1.1])
        plt.yticks(fontsize=48)
        plt.xticks([0, 100], [0, 20], fontsize=48)
        plt.ylabel('slope', fontsize=64)
        plt.xlabel('epoch', fontsize=64)
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/results/slope.png', dpi=300)
        plt.close()

    else:

        files = glob.glob(f'./{log_dir}/results/*.png')
        for f in files:
            os.remove(f)

        # fig_init(formatx='%.0f', formaty='%.0f')
        # plt.hist(distrib, bins=100, color=mc, alpha=0.5)
        # plt.ylabel('counts', fontsize=64)
        # plt.xlabel('$x_{ij}$', fontsize=64)
        # plt.xticks(fontsize=24)
        # plt.yticks(fontsize=24)
        # plt.tight_layout()
        # plt.savefig(f'./{log_dir}/results/signal_distribution.png', dpi=300)
        # plt.close()
        # print(f'mean: {np.mean(distrib):0.2f}  std: {np.std(distrib):0.2f}')
        # logger.info(f'mean: {np.mean(distrib):0.2f}  std: {np.std(distrib):0.2f}')
        #
        # plt.figure(figsize=(15, 10))
        # ax = sns.heatmap(to_numpy(activity), center=0, cmap='viridis', cbar_kws={'fraction': 0.046})
        # cbar = ax.collections[0].colorbar
        # cbar.ax.tick_params(labelsize=32)
        # ax.invert_yaxis()
        # plt.ylabel('neurons', fontsize=64)
        # plt.xlabel('time', fontsize=64)
        # plt.xticks([10000, 99000], [10000, 100000], fontsize=48)
        # plt.yticks([0, 999], [1, 1000], fontsize=48)
        # plt.xticks(rotation=0)
        # plt.tight_layout()
        # plt.savefig(f'./{log_dir}/results/kinograph.png', dpi=300)
        # plt.close()


        if False: #os.path.exists(f"./{log_dir}/neuron_gt_list.pt"):

            os.makedirs(f"./{log_dir}/results/activity", exist_ok=True)

            neuron_gt_list = torch.load(f"./{log_dir}/neuron_gt_list.pt", map_location=device)
            neuron_pred_list = torch.load(f"./{log_dir}/neuron_pred_list.pt", map_location=device)

            neuron_gt_list = torch.cat(neuron_gt_list, 0)
            neuron_pred_list = torch.cat(neuron_pred_list, 0)
            neuron_gt_list = torch.reshape(neuron_gt_list, (1600, n_neurons))
            neuron_pred_list = torch.reshape(neuron_pred_list, (1600, n_neurons))

            n = [20, 30, 100, 150, 260, 270, 520, 620, 720, 820]

            r_squared_list = []
            slope_list = []
            for i in trange(0,1500,10, ncols=90):
                plt.figure(figsize=(20, 10))
                ax = plt.subplot(121)
                plt.plot(neuron_gt_list[:, n[0]].detach().cpu().numpy(), c='w', linewidth=8, label='true', alpha=0.25)
                plt.plot(neuron_pred_list[0:i, n[0]].detach().cpu().numpy(), linewidth=4, c='w', label='learned')
                plt.legend(fontsize=24)
                plt.plot(neuron_gt_list[:, n[1:10]].detach().cpu().numpy(), c='w', linewidth=8, alpha=0.25)
                plt.plot(neuron_pred_list[0:i, n[1:10]].detach().cpu().numpy(), linewidth=4)
                plt.xlim([0, 1500])
                plt.xlabel('time index', fontsize=48)
                plt.ylabel(r'$v_i$', fontsize=48)
                plt.xticks(fontsize=24)
                plt.yticks(fontsize=24)
                plt.ylim([-15,15])
                plt.text(40, 13, 'N=10', fontsize=34)
                plt.text(40, 11, f'time: {i}', fontsize=34)
                ax = plt.subplot(122)
                plt.scatter(to_numpy(neuron_gt_list[i, :]), to_numpy(neuron_pred_list[i, :]), s=10, c=mc)
                plt.xlim([-15,15])
                plt.ylim([-15,15])
                plt.xticks(fontsize=24)
                plt.yticks(fontsize=24)
                x_data = to_numpy(neuron_gt_list[i, :])
                y_data = to_numpy(neuron_pred_list[i, :])
                lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                residuals = y_data - linear_model(x_data, *lin_fit)
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                r_squared_list.append(r_squared)
                slope_list.append(lin_fit[0])
                plt.xlabel(r'true $v_i$', fontsize=48)
                plt.text(-13, 13, 'N=1024', fontsize=34)
                plt.ylabel(r'learned $v_i$', fontsize=48)
                plt.text(-13, 11, f'$R^2$: {np.round(r_squared, 3)}', fontsize=34)
                plt.text(-13, 9, f'slope: {np.round(lin_fit[0], 2)}', fontsize=34)
                plt.tight_layout()
                plt.savefig(f'./{log_dir}/results/activity/comparison_{i}.png', dpi=80)
                plt.close()

            plt.figure(figsize=(10, 10))
            plt.plot(r_squared_list, linewidth=4, label='$R^2$')
            plt.plot(slope_list, linewidth=4, label='slope')
            plt.xticks([0,75,150],[0,375,750],fontsize=24)
            plt.yticks(fontsize=24)
            plt.ylim([0,1.4])
            plt.xlim([0,150])
            plt.xlabel(r'time', fontsize=48)
            plt.title(r'true vs learned $v_i$', fontsize=48)
            plt.legend(fontsize=24)
            plt.tight_layout()
            plt.savefig(f'./{log_dir}/results/activity_comparison.png', dpi=80)
            plt.close()

        connectivity = torch.load(f'./graphs_data/{dataset_name}/connectivity.pt', map_location=device)

        adjacency = connectivity.t().clone().detach()
        adj_t = torch.abs(adjacency) > 0
        edge_index = adj_t.nonzero().t().contiguous()

        edge_index = adj_t.nonzero().t().contiguous()
        weights = to_numpy(adjacency.flatten())
        pos = np.argwhere(weights != 0)
        weights = weights[pos]

        fig_init()
        plt.hist(weights, bins=1000, color=mc, alpha=0.5)
        plt.ylabel(r'counts', fontsize=64)
        plt.xlabel(r'$W$', fontsize=64)
        plt.yticks(fontsize=24)
        plt.xticks(fontsize=24)
        plt.xlim([-0.1, 0.1])
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/results/weights_distribution.png', dpi=300)
        plt.close()

        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(to_numpy(adjacency), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046},
                         vmin=-0.1, vmax=0.1)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=32)
        plt.xticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
        plt.yticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
        plt.xticks(rotation=0)
        plt.subplot(2, 2, 1)
        ax = sns.heatmap(to_numpy(adjacency[0:20, 0:20]), cbar=False, center=0, square=True, cmap='bwr', vmin=-0.1, vmax=0.1)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/results/true connectivity.png', dpi=300)
        plt.close()

        plt.figure(figsize=(10, 10))
        if False: # config.graph_model.signal_model_name == 'PDE_N8':
            with open(f'graphs_data/{dataset_name}/larynx_neuron_list.json', 'r') as file:
                larynx_neuron_list = json.load(file)
            with open(f'graphs_data/{dataset_name}/all_neuron_list.json', 'r') as file:
                activity_neuron_list = json.load(file)
            map_larynx_matrix, n = map_matrix(larynx_neuron_list, activity_neuron_list, adjacency)
        else:
            n = np.random.randint(0, n_neurons, 50)
        for i in range(len(n)):
            plt.plot(to_numpy(activity[n[i].astype(int), :]), linewidth=1)
        plt.xlabel('time', fontsize=64)
        plt.ylabel('$x_{i}$', fontsize=64)
        plt.xlim([0,n_frames])
        # plt.xticks([10000, 99000], [10000, 100000], fontsize=48)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        plt.title(r'$v_i$ samples',fontsize=48)
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/results/activity.png', dpi=300)
        plt.close()

        true_model, bc_pos, bc_dpos = choose_model(config=config, W=adjacency, device=device)

        for epoch in epoch_list:

            net = f'{log_dir}/models/best_model_with_{n_runs-1}_graphs_{epoch}.pt'
            model, bc_pos, bc_dpos = choose_training_model(config, device)
            state_dict = torch.load(net, map_location=device)
            model.load_state_dict(state_dict['model_state_dict'])
            model.edges = edge_index
            print(f'net: {net}')

            i, j = torch.triu_indices(n_neurons, n_neurons, requires_grad=False, device=device)
            A = model.W.clone().detach()
            A[i, i] = 0

            plt.figure(figsize=(10, 10))
            ax = sns.heatmap(to_numpy(A), center=0, square=True, cmap='bwr',
                             cbar_kws={'fraction': 0.046}, vmin=-0.5, vmax=0.5)
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=32)
            plt.xticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
            plt.yticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
            plt.xticks(rotation=0)
            # plt.subplot(2, 2, 1)
            # ax = sns.heatmap(to_numpy(A[0:20, 0:20]), cbar=False, center=0, square=True, cmap='bwr', vmin=-0.1, vmax=0.1)
            # plt.xticks(rotation=0)
            # plt.xticks([])
            # plt.yticks([])
            plt.tight_layout()
            plt.savefig(f'./{log_dir}/results/first learned connectivity.png', dpi=300)
            plt.close()

            A = A[:n_neurons,:n_neurons]

            if has_field:
                net = f'{log_dir}/models/best_model_f_with_{n_runs - 1}_graphs_{epoch}.pt'
                state_dict = torch.load(net, map_location=device)
                model_f.load_state_dict(state_dict['model_state_dict'])

                if 'short_term_plasticity' in field_type:

                    fig, ax = fig_init()
                    t = torch.linspace(1, 1000, 1, dtype=torch.float32, device=device).unsqueeze(1)
                    prediction = model_f(t) ** 2
                    prediction = prediction.t()
                    plt.imshow(to_numpy(prediction), aspect='auto')
                    plt.title(r'learned $MLP_2(i,t)$', fontsize=68)
                    plt.xlabel(r'$t$', fontsize=68)
                    plt.ylabel(r'$i$', fontsize=68)
                    # plt.xticks([10000, 100000], [10000, 100000], fontsize=48)
                    # plt.yticks([0, 512, 1024], [0, 512, 1024], fontsize=48)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/learned_plasticity.png", dpi=80)
                    plt.close()

                    modulation_short = modulation[:, np.linspace(0, 100000, 1000).astype(int)]
                    activity_short = activity[:, np.linspace(0, 100000, 1000).astype(int)]

                    fig, ax = fig_init()
                    plt.scatter(to_numpy(modulation_short), to_numpy(prediction), s=1, color=mc, alpha=0.1)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/short_comparison.png", dpi=80)
                    plt.close()

                    time_step = 32
                    start = 400
                    end = 600
                    derivative_prediction = prediction[:, time_step:] - prediction[:, :-time_step]
                    derivative_prediction = derivative_prediction * 1000
                    x_ = activity_short[:, start:end].flatten()
                    y_ = modulation_short[:, start:end].flatten()
                    derivative_ = derivative_prediction[:, start-time_step//2:end-time_step//2].flatten()
                    fig, ax = fig_init()
                    plt.scatter(to_numpy(x_), to_numpy(y_), s=1, c=to_numpy(derivative_),
                                alpha=0.1, vmin=-100,vmax=100, cmap='viridis')
                    plt.colorbar()
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/plasticity_map.png", dpi=80)
                    plt.close()

                    model_pysrr = PySRRegressor(
                        niterations=30,  # < Increase me for better results
                        binary_operators=["+", "-", "*", "/"],
                        random_state=0,
                        temp_equation_file=False
                    )
                    rr = torch.concatenate((y_[:, None], x_[:, None]), dim=1)
                    model_pysrr.fit(to_numpy(rr), to_numpy(derivative_[:, None]))

                    tau = 100
                    alpha = 0.02
                    true_derivative_ = (1 - y_) / tau - alpha * y_ * torch.abs(x_)
                    fig, ax = fig_init()
                    plt.scatter(to_numpy(x_), to_numpy(y_), s=10, c=to_numpy(true_derivative_),
                                alpha=1, cmap='viridis')
                    plt.colorbar()
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/true_plasticity_map.png", dpi=80)
                    plt.close()

            if has_ghost:
                net = f'{log_dir}/models/best_model_f_with_{n_runs - 1}_graphs_{epoch}.pt'
                state_dict = torch.load(net, map_location=device)
                model_missing_activity.load_state_dict(state_dict['model_state_dict'])

            fig, ax = fig_init()
            if has_ghost:
                plt.scatter(to_numpy(model.a[:n_neurons, 0]), to_numpy(model.a[:n_neurons, 1]), s=150, color=cmap.color(0), edgecolors='none')
            else:
                for n in range(n_neuron_types,-1,-1):
                    pos = torch.argwhere(type_list == n).squeeze()
                    plt.scatter(to_numpy(model.a[pos, 0]), to_numpy(model.a[pos, 1]), s=200, color=cmap.color(n), alpha=0.25, edgecolors='none')
            if 'latex' in style:
                plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=68)
                plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=68)
            else:
                plt.xlabel(r'$a_{0}$', fontsize=68)
                plt.ylabel(r'$a_{1}$', fontsize=68)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/all_embedding.png", dpi=170.7)
            plt.close()

            if 'excitation' in model_config.update_type:
                embedding = model.a[:n_neurons]
                excitation = torch.tensor(x_list[0][:, :, 10: 10 + model.excitation_dim],device=device)
                excitation = torch.reshape(excitation, (excitation.shape[0]*excitation.shape[1], excitation.shape[2]))
                fig = plt.figure(figsize=(10, 9))
                excitation_ = torch.unique(excitation,dim=0)
                for k, exc in enumerate(excitation_):
                    ax = fig.add_subplot(2, 2, k + 1)
                    plt.text(0.2, 0.95, f'{to_numpy(exc)}', fontsize=14, ha='center', va='center', transform=ax.transAxes)
                    in_features = torch.cat([embedding, exc * torch.ones_like(embedding[:,0:1])], dim=1)
                    out = model.lin_exc(in_features.float())
                    # plt.scatter(to_numpy(embedding[:, 0])*0, to_numpy(out), s=100, c='w', alpha=0.15, edgecolors='none')
                    plt.scatter(to_numpy(embedding[:, 0]), to_numpy(embedding[:, 1]), s=100, c=to_numpy(out), alpha=1, edgecolors='none')
                    plt.colorbar()
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/excitation.png", dpi=170.7)
                plt.close()

            fig, ax = fig_init()
            rr = torch.linspace(-xnorm.squeeze() * 2 , xnorm.squeeze() * 2 , 1000).to(device)
            func_list = []
            for n in trange(0,n_neurons, ncols=90):
                if (model_config.signal_model_name == 'PDE_N4') | (model_config.signal_model_name == 'PDE_N5'):
                    embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                    in_features = get_in_features(rr, embedding_, model_config.signal_model_name, max_radius)
                else:
                    in_features = rr[:,None]
                with torch.no_grad():
                    func = model.lin_edge(in_features.float())
                if config.graph_model.lin_edge_positive:
                    func = func ** 2
                func_list.append(func)
                plt.plot(to_numpy(rr), to_numpy(func), 2, color=cmap.color(to_numpy(type_list)[n].astype(int)),
                         linewidth=2 // ( 1 + (n_neuron_types>16)*1.0), alpha=0.25)
            func_list = torch.stack(func_list).squeeze()
            y_min, y_max = func_list.min().item(), func_list.max().item()
            plt.xlabel(r'$v_i$', fontsize=68)
            plt.ylabel(r'Learned $\psi^*(\mathbf{a}_i, v_i)$', fontsize=68)
            # if (model_config.signal_model_name == 'PDE_N4') | (model_config.signal_model_name == 'PDE_N5'):
            #     plt.ylim([-0.5,0.5])
            # plt.xlim([-to_numpy(xnorm)*2, to_numpy(xnorm)*2])
            plt.ylim([y_min,y_max*1.1])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/raw_psi.png", dpi=170.7)
            plt.close()

            upper = func_list[:,950:1000].flatten()
            upper = torch.sort(upper, descending=True).values
            correction = 1 / torch.mean(upper[:upper.shape[0]//10])
            # correction = 1 / torch.mean(torch.mean(func_list[:,900:1000], dim=0))
            print(f'upper: {to_numpy(1/correction):0.4f}  correction: {to_numpy(correction):0.2f}')
            torch.save(correction, f'{log_dir}/correction.pt')

            matrix_correction = torch.mean(func_list[:,950:1000], dim=1)
            A_corrected = A * matrix_correction[:, None]
            plt.figure(figsize=(10, 10))
            ax = sns.heatmap(to_numpy(A_corrected), center=0, square=True, cmap='bwr',
                             cbar_kws={'fraction': 0.046}, vmin=-0.1, vmax=0.1)
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=32)
            plt.xticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
            plt.yticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
            plt.xticks(rotation=0)
            # plt.subplot(2, 2, 1)
            # ax = sns.heatmap(to_numpy(A[0:20, 0:20]), cbar=False, center=0, square=True, cmap='bwr', vmin=-0.1, vmax=0.1)
            # plt.xticks(rotation=0)
            # plt.xticks([])
            # plt.yticks([])
            plt.tight_layout()
            plt.savefig(f'./{log_dir}/results/corrected learned connectivity.png', dpi=300)
            plt.close()

            print('update functions ...')
            if model_config.signal_model_name == 'PDE_N5':
                psi_list = []
                if model.update_type == 'generic':
                    r_list = ['','generic']
                elif model.update_type == '2steps':
                    r_list = ['','2steps']

                r_list = ['']
                for r in r_list:
                    fig, ax = fig_init()
                    rr = torch.linspace(-xnorm.squeeze()*2, xnorm.squeeze()*2, 1500).to(device)
                    ax.set_frame_on(False)
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    for k in range(n_neuron_types):
                        ax = fig.add_subplot(2, 2, k + 1)
                        for spine in ax.spines.values():
                            spine.set_edgecolor(cmap.color(k))  # Set the color of the outline
                            spine.set_linewidth(3)
                        for m in range(n_neuron_types):
                            true_func = true_model.func(rr, k, m, 'phi')
                            plt.plot(to_numpy(rr), to_numpy(true_func), c=mc, linewidth=1, label='original', alpha=0.21)
                        for n in range(n_neuron_types):
                            for m in range(250):
                                pos0 = to_numpy(torch.argwhere(type_list == k).squeeze())
                                pos1 = to_numpy(torch.argwhere(type_list == n).squeeze())
                                n0 = np.random.randint(len(pos0))
                                n0 = pos0[n0,0]
                                n1 = np.random.randint(len(pos1))
                                n1 = pos1[n1,0]
                                embedding0 = model.a[n0, :] * torch.ones((1500, config.graph_model.embedding_dim), device=device)
                                embedding1 = model.a[n1, :] * torch.ones((1500, config.graph_model.embedding_dim), device=device)
                                in_features = torch.cat((rr[:,None],embedding0, embedding1), dim=1)
                                if config.graph_model.lin_edge_positive:
                                    func = model.lin_edge(in_features.float()) ** 2 * correction
                                else:
                                    func = model.lin_edge(in_features.float()) * correction
                                if r == '2steps':
                                    field = torch.ones_like(rr[:,None])
                                    u = torch.zeros_like(rr[:,None])
                                    in_features2 = torch.cat([u, func, field], dim=1)
                                    func = model.lin_phi2(in_features2)
                                elif r == 'generic':
                                    field = torch.ones_like(rr[:,None])
                                    u = torch.zeros_like(rr[:,None])
                                    in_features = torch.cat([u, embedding0, func.detach().clone(), field], dim=1)
                                    func = model.lin_phi(in_features)
                                psi_list.append(func)
                                plt.plot(to_numpy(rr), to_numpy(func), 2, color=cmap.color(n),linewidth=1, alpha=0.25)
                        # plt.ylim([-1.1, 1.1])
                        plt.xlim([-to_numpy(xnorm)*2, to_numpy(xnorm)*2])
                        plt.xticks(fontsize=18)
                        plt.yticks(fontsize=18)
                        # plt.ylabel(r'learned $\psi^*(\mathbf{a}_i, a_j, v_i)$', fontsize=24)
                        # plt.xlabel(r'$v_i$', fontsize=24)
                        # plt.ylim([-1.5, 1.5])
                        # plt.xlim([-5, 5])

                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/learned_psi_{r}.png", dpi=170.7)
                    plt.close()
                psi_list = torch.stack(psi_list)
                psi_list = psi_list.squeeze()
            else:
                psi_list = []
                fig, ax = fig_init()
                rr = torch.linspace(-xnorm.squeeze(), xnorm.squeeze(), 1500).to(device)

                if (model_config.signal_model_name == 'PDE_N4'):
                    for n in range(n_neuron_types):
                        true_func = true_model.func(rr, n, 'phi')
                        plt.plot(to_numpy(rr), to_numpy(true_func), c = mc, linewidth = 16, label = 'original', alpha = 0.21)
                else:
                    true_func = true_model.func(rr, 0, 'phi')
                    plt.plot(to_numpy(rr), to_numpy(true_func), c = mc, linewidth = 16, label = 'original', alpha = 0.21)

                for n in trange(0,n_neurons, ncols=90):
                    if (model_config.signal_model_name == 'PDE_N4') | (model_config.signal_model_name == 'PDE_N5'):
                        embedding_ = model.a[n, :] * torch.ones((1500, config.graph_model.embedding_dim), device=device)
                        in_features = get_in_features(rr, embedding_, model_config.signal_model_name, max_radius)
                    else:
                        in_features = rr[:, None]
                    with torch.no_grad():
                        if config.graph_model.lin_edge_positive:
                            func = model.lin_edge(in_features.float()) ** 2 * correction
                        else:
                            func = model.lin_edge(in_features.float()) * correction
                        psi_list.append(func)
                    if (model_config.signal_model_name == 'PDE_N4') | (model_config.signal_model_name == 'PDE_N5'):
                        plt.plot(to_numpy(rr), to_numpy(func), 2, color=cmap.color(to_numpy(type_list)[n].astype(int)), linewidth=2, alpha=0.25)
                    else:
                        plt.plot(to_numpy(rr), to_numpy(func), 2, color=mc, linewidth=2, alpha=0.25)

                plt.xlabel(r'$v_i$', fontsize=68)
                if (model_config.signal_model_name == 'PDE_N4'):
                    plt.ylabel(r'learned $\psi^*(\mathbf{a}_i, v_i)$', fontsize=68)
                elif model_config.signal_model_name == 'PDE_N5':
                    plt.ylabel(r'learned $\psi^*(\mathbf{a}_i, a_j, v_i)$', fontsize=68)
                else:
                    plt.ylabel(r'learned $\psi^*(v_i)$', fontsize=68)
                if config.graph_model.lin_edge_positive:
                    plt.ylim([-0.2, 1.2])
                else:
                    plt.ylim([-1.6, 1.6])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/learned_psi.png", dpi=170.7)
                plt.close()
                psi_list = torch.stack(psi_list)
                psi_list = psi_list.squeeze()

            print('interaction functions ...')

            fig, ax = fig_init()
            for n in trange(n_neuron_types, ncols=90):
                if model_config.signal_model_name == 'PDE_N5':
                    true_func = true_model.func(rr, n, n, 'update')
                else:
                    true_func = true_model.func(rr, n, 'update')
                plt.plot(to_numpy(rr), to_numpy(true_func), c=mc, linewidth=16, label='original', alpha=0.21)
            phi_list = []
            for n in trange(n_neurons, ncols=90):
                embedding_ = model.a[n, :] * torch.ones((1500, config.graph_model.embedding_dim), device=device)
                # in_features = torch.cat((rr[:, None], embedding_), dim=1)
                in_features = get_in_features_update(rr[:, None], model, embedding_, device)
                with torch.no_grad():
                    func = model.lin_phi(in_features.float())
                func = func[:, 0]
                phi_list.append(func)
                plt.plot(to_numpy(rr), to_numpy(func) * to_numpy(ynorm),
                         color=cmap.color(to_numpy(type_list[n]).astype(int)), linewidth=2, alpha=0.25)
            phi_list = torch.stack(phi_list)
            func_list_ = to_numpy(phi_list)
            plt.xlabel(r'$v_i$', fontsize=68)
            plt.ylabel(r'learned $\phi^*(\mathbf{a}_i, v_i)$', fontsize=68)
            plt.tight_layout()
            # plt.xlim([-to_numpy(xnorm), to_numpy(xnorm)])
            plt.ylim(config.plotting.ylim)
            plt.savefig(f'./{log_dir}/results/learned phi.png', dpi=300)
            plt.close()

            print('UMAP reduction ...')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                trans = umap.UMAP(n_neighbors=50, n_components=2, transform_queue_size=0,
                                  random_state=config.training.seed).fit(func_list_)
                proj_interaction = trans.transform(func_list_)

            proj_interaction = (proj_interaction - np.min(proj_interaction)) / (np.max(proj_interaction) - np.min(proj_interaction) + 1e-10)
            fig, ax = fig_init()
            for n in trange(n_neuron_types, ncols=90):
                pos = torch.argwhere(type_list == n)
                pos = to_numpy(pos)
                if len(pos) > 0:
                    plt.scatter(proj_interaction[pos, 0],
                                proj_interaction[pos, 1], s=200, alpha=0.1)
            plt.xlabel(r'UMAP 0', fontsize=68)
            plt.ylabel(r'UMAP 1', fontsize=68)
            plt.xlim([-0.2, 1.2])
            plt.ylim([-0.2, 1.2])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/UMAP.png", dpi=170.7)
            plt.close()

            config.training.cluster_distance_threshold = 0.1
            config.training.cluster_method = 'distance_embedding'
            embedding = to_numpy(model.a.squeeze())
            labels, n_clusters, new_labels = sparsify_cluster(config.training.cluster_method, proj_interaction, embedding,
                                                              config.training.cluster_distance_threshold, type_list,
                                                              n_neuron_types, embedding_cluster)
            accuracy = metrics.accuracy_score(to_numpy(type_list), new_labels[:n_neurons])
            print(f'accuracy: {accuracy:0.4f}   n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}  ')
            logger.info(f'accuracy: {accuracy:0.4f}   n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method} ')

            # config.training.cluster_method = 'kmeans_auto_embedding'
            # labels, n_clusters, new_labels = sparsify_cluster(config.training.cluster_method, proj_interaction, embedding,
            #                                                   config.training.cluster_distance_threshold, type_list,
            #                                                   n_neuron_types, embedding_cluster)
            # accuracy = metrics.accuracy_score(to_numpy(type_list), new_labels)
            # print(f'accuracy: {accuracy:0.4f}   n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}  ')
            # logger.info(f'accuracy: {accuracy:0.4f}   n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method} ')

            plt.figure(figsize=(10, 10))
            plt.scatter(to_numpy(X1_first[:n_neurons, 0]), to_numpy(X1_first[:n_neurons, 1]), s=150, color=cmap.color(to_numpy(type_list).astype(int)))
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/true_types.png", dpi=170.7)
            plt.close()

            plt.figure(figsize=(10, 10))
            plt.scatter(to_numpy(X1_first[:n_neurons, 0]), to_numpy(X1_first[:n_neurons, 1]), s=150, color=cmap.color(new_labels[:n_neurons].astype(int)))
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/learned_types.png", dpi=170.7)
            plt.close()

            fig, ax = fig_init()
            gt_weight = to_numpy(connectivity)
            pred_weight = to_numpy(A)
            plt.scatter(gt_weight, pred_weight / 10 , s=0.1, c=mc, alpha=0.1)
            plt.xlabel(r'true $W_{ij}$', fontsize=68)
            plt.ylabel(r'learned $W_{ij}$', fontsize=68)
            if n_neurons == 8000:
                plt.xlim([-0.05,0.05])
                plt.ylim([-0.05,0.05])
            else:
                plt.xlim([-0.2,0.2])
                plt.ylim([-0.2,0.2])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/first_comparison.png", dpi=87)
            plt.close()

            x_data = np.reshape(gt_weight, (n_neurons * n_neurons))
            y_data =  np.reshape(pred_weight, (n_neurons * n_neurons))
            lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
            residuals = y_data - linear_model(x_data, *lin_fit)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            print(f'R^2$: {r_squared:0.4f}  slope: {np.round(lin_fit[0], 4)}')
            logger.info(f'R^2$: {np.round(r_squared, 4)}  slope: {np.round(lin_fit[0], 4)}')

            second_correction = lin_fit[0]
            print(f'second_correction: {second_correction:0.2f}')
            np.save(f'{log_dir}/second_correction.npy', second_correction)

            fig, ax = fig_init()
            gt_weight = to_numpy(connectivity)
            pred_weight = to_numpy(A)
            plt.scatter(gt_weight, pred_weight / second_correction, s=0.1, c=mc, alpha=0.1)
            plt.xlabel(r'true $W_{ij}$', fontsize=68)
            plt.ylabel(r'learned $W_{ij}$', fontsize=68)
            if n_neurons == 8000:
                plt.xlim([-0.05,0.05])
                plt.ylim([-0.05,0.05])
            else:
                plt.xlim([-0.2,0.2])
                plt.ylim([-0.2,0.2])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/second_comparison.png", dpi=87)
            plt.close()

            plt.figure(figsize=(10, 10))
            # plt.title(r'learned $W_{ij}$', fontsize=68)
            ax = sns.heatmap(to_numpy(A)/second_correction, center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046})
            cbar = ax.collections[0].colorbar
            # here set the labelsize by 20
            cbar.ax.tick_params(labelsize=32)
            plt.xticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
            plt.yticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
            plt.xticks(rotation=0)
            plt.subplot(2, 2, 1)
            ax = sns.heatmap(to_numpy(A[0:20, 0:20]/second_correction), cbar=False, center=0, square=True, cmap='bwr')
            plt.xticks(rotation=0)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(f'./{log_dir}/results/final learned connectivity.png', dpi=300)
            plt.close()

            if has_ghost:

                print('plot learned activity ...')
                os.makedirs(f"./{log_dir}/results/learned_activity", exist_ok=True)
                for n in range(n_runs):
                    fig, ax = fig_init(fontsize=24, formatx='%.0f', formaty='%.0f')
                    t = torch.zeros((1, 800, 1), dtype=torch.float32, device=device)
                    t[0] = torch.linspace(0, 1, 800, dtype=torch.float32, device=device)[:, None]
                    prediction = model_missing_activity[n](t)
                    prediction = prediction.squeeze().t()
                    plt.imshow(to_numpy(prediction), aspect='auto',cmap='viridis')
                    plt.colorbar()
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/learned_activity/learned_activity_{n}.png", dpi=80)
                    plt.close()




            if has_field:

                print('plot field ...')
                os.makedirs(f"./{log_dir}/results/field", exist_ok=True)

                if 'derivative' in field_type:

                    y = torch.linspace(0, 1, 400)
                    x = torch.linspace(-6, 6, 400)
                    grid_y, grid_x = torch.meshgrid(y, x)
                    grid = torch.stack((grid_x, grid_y), dim=-1)
                    grid = grid.to(device)
                    pred_modulation = model.lin_modulation(grid)
                    tau = 100
                    alpha = 0.02
                    true_derivative = (1 - grid_y) / tau - alpha * grid_y * torch.abs(grid_x)

                    fig, ax = fig_init()
                    plt.title(r'$\dot{y_i}$', fontsize=68)
                    # plt.title(r'$\dot{y_i}=(1-y)/100 - 0.02 x_iy_i$', fontsize=48)
                    plt.imshow(to_numpy(true_derivative))
                    plt.xticks([0, 100, 200, 300, 400], [-6, -3, 0, 3, 6], fontsize=48)
                    plt.yticks([0, 100, 200, 300, 400], [0, 0.25, 0.5, 0.75, 1], fontsize=48)
                    plt.xlabel(r'$v_i$', fontsize=68)
                    plt.ylabel(r'$y_i$', fontsize=68)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/true_field_derivative.png", dpi=80)
                    plt.close()

                    fig, ax = fig_init()
                    plt.title(r'learned $\dot{y_i}$', fontsize=68)
                    plt.imshow(to_numpy(pred_modulation))
                    plt.xticks([0, 100, 200, 300, 400], [-6, -3, 0, 3, 6], fontsize=48)
                    plt.yticks([0, 100, 200, 300, 400], [0, 0.25, 0.5, 0.75, 1], fontsize=48)
                    plt.xlabel(r'$v_i$', fontsize=68)
                    plt.ylabel(r'$y_i$', fontsize=68)
                    # plt.colorbar()
                    plt.tight_layout
                    plt.savefig(f"./{log_dir}/results/field_derivative.png", dpi=80)
                    plt.close()

                    # fig = plt.figure(figsize=(12, 12))
                    # ind_list = [320]
                    # ids = np.arange(0, 100000, 100)
                    # ax = fig.add_subplot(2, 1, 1)
                    # for ind in ind_list:
                    #     plt.plot(to_numpy(modulation[ind, ids]))
                    #     plt.plot(to_numpy(model.b[ind, 0:1000]**2))

                if ('short_term_plasticity' in field_type) | ('modulation_permutation' in field_type):

                    for frame in trange(0, n_frames, n_frames // 100, ncols=90):
                        t = torch.tensor([frame/ n_frames], dtype=torch.float32, device=device)
                        if (model_config.update_type == '2steps'):
                                m_ = model_f(t) ** 2
                                m_ = m_[:,None]
                                in_features= torch.cat((torch.zeros_like(m_), torch.ones_like(m_)*xnorm, m_), dim=1)
                                m = model.lin_phi2(in_features)
                        else:
                            m = model_f(t) ** 2

                        if 'permutation' in model_config.field_type:
                            inverse_permutation_indices = torch.load(f'./graphs_data/{dataset_name}/inverse_permutation_indices.pt', map_location=device)
                            modulation_ = m[inverse_permutation_indices]
                        else:
                            modulation_ = m
                        modulation_ = torch.reshape(modulation_, (32, 32)) * torch.tensor(second_correction, device=device) / 10

                        fig = plt.figure(figsize=(10, 10.5))
                        plt.axis('off')
                        plt.xticks([])
                        plt.xticks([])
                        im_ = to_numpy(modulation_)
                        im_ = np.rot90(im_, k=-1)
                        im_ = np.flipud(im_)
                        im_ = np.fliplr(im_)
                        plt.imshow(im_, cmap='gray')
                        # plt.title(r'neuromodulation $b_i$', fontsize=48)
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/results/field/xi_{frame}.png", dpi=80)
                        plt.close()

                        # x = x_list[0][frame]
                        # fig = plt.figure(figsize=(10, 10.5))
                        # plt.axis('off')
                        # plt.xticks([])
                        # plt.xticks([])
                        # plt.scatter(x[:,1], x[:,2], s=160, c=to_numpy(modulation[:,frame]),
                        #             vmin=0, vmax=2, cmap='viridis')
                        # plt.title(r'neuromodulation $b_i$', fontsize=48)
                        # plt.tight_layout()
                        # plt.savefig(f"./{log_dir}/results/field/bi_{frame}.png", dpi=80)
                        # plt.close()
                        # fig = plt.figure(figsize=(10, 10.5))
                        # plt.axis('off')
                        # plt.xticks([])
                        # plt.xticks([])
                        # plt.scatter(x[:,1], x[:,2], s=160, c=x[:,6],
                        #             vmin=-20, vmax=20, cmap='viridis')
                        # plt.title(r'$v_i$', fontsize=48)
                        # plt.tight_layout()
                        # plt.savefig(f"./{log_dir}/results/field/xi_{frame}.png", dpi=80)
                        # plt.close()

                    fig, ax = fig_init()
                    t = torch.linspace(0, 1, 100000, dtype=torch.float32, device=device).unsqueeze(1)

                    prediction = model_f(t) ** 2
                    prediction = prediction.t()
                    plt.imshow(to_numpy(prediction), aspect='auto')
                    plt.title(r'learned $MLP_2(i,t)$', fontsize=68)
                    plt.xlabel(r'$t$', fontsize=68)
                    plt.ylabel(r'$i$', fontsize=68)
                    plt.xticks([10000, 100000], [10000, 100000], fontsize=48)
                    plt.yticks([0, 512, 1024], [0, 512, 1024], fontsize=48)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/learned_plasticity.png", dpi=80)
                    plt.close()

                    fig, ax = fig_init()
                    plt.imshow(to_numpy(modulation), aspect='auto')
                    plt.title(r'$y_i$', fontsize=68)
                    plt.xlabel(r'$t$', fontsize=68)
                    plt.ylabel(r'$i$', fontsize=68)
                    plt.xticks([10000, 100000], [10000, 100000], fontsize=48)
                    plt.yticks([0, 512, 1024], [0, 512, 1024], fontsize=48)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/true_plasticity.png", dpi=80)
                    plt.close()

                    prediction = prediction * torch.tensor(second_correction, device=device) / 10

                    fig, ax = fig_init()
                    ids = np.arange(0, 100000, 100).astype(int)
                    plt.scatter(to_numpy(modulation[:, ids]), to_numpy(prediction[:, ids]), s=0.1, color=mc, alpha=0.05)
                    # plt.xlim([0, 1])
                    # plt.ylim([0, 2])
                    # plt.xticks([0, 0.5], [0, 0.5], fontsize=48)
                    # plt.yticks([0, 1, 2], [0, 1, 2], fontsize=48)
                    x_data = to_numpy(modulation[:, ids]).flatten()
                    y_data = to_numpy(prediction[:, ids]).flatten()
                    lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                    residuals = y_data - linear_model(x_data, *lin_fit)
                    ss_res = np.sum(residuals ** 2)
                    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot)
                    ax.text(0.05, 0.94, f'$R^2$: {r_squared:0.2f}', transform=ax.transAxes,
                            verticalalignment='top', horizontalalignment='left', fontsize=32)
                    ax.text(0.05, 0.88, f'slope: {lin_fit[0]:0.2f}', transform=ax.transAxes,
                            verticalalignment='top', horizontalalignment='left', fontsize=32)
                    plt.xlabel(r'true $y_i(t)$', fontsize=68)
                    plt.ylabel(r'learned $y_i(t)$', fontsize=68)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/comparison_yi.png", dpi=80)
                    plt.close()

                else:
                    net = f'{log_dir}/models/best_model_f_with_{n_runs - 1}_graphs_{epoch}.pt'
                    state_dict = torch.load(net, map_location=device)
                    model_f.load_state_dict(state_dict['model_state_dict'])
                    im = imread(f"graphs_data/{config.simulation.node_value_map}")

                    x = x_list[0][0]

                    slope_list = list([])
                    im_list = list([])
                    pred_list = list([])

                    for frame in trange(0, n_frames, n_frames // 100, ncols=90):

                        fig, ax = fig_init()
                        im_ = np.zeros((44, 44))
                        if (frame >= 0) & (frame < n_frames):
                            im_ = im[int(frame / n_frames * 256)].squeeze()
                        plt.imshow(im_, cmap='gray', vmin=0, vmax=2)
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/results/field/true_field{epoch}_{frame}.png", dpi=80)
                        plt.close()

                        pred = model_f(time=frame / n_frames, enlarge=False) ** 2 * second_correction / 10
                        pred = torch.reshape(pred, (n_nodes_per_axis, n_nodes_per_axis))

                        pred = to_numpy(pred)
                        pred = np.flipud(pred)
                        pred = np.rot90(pred, 1)
                        pred = np.fliplr(pred)
                        fig, ax = fig_init()
                        plt.imshow(pred, cmap='gray', vmin=0, vmax=2)
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/results/field/reconstructed_field_LR {epoch}_{frame}.png", dpi=80)
                        plt.close()

                        x_data = np.reshape(im_, (n_nodes_per_axis * n_nodes_per_axis))
                        y_data = np.reshape(pred, (n_nodes_per_axis * n_nodes_per_axis))
                        lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                        residuals = y_data - linear_model(x_data, *lin_fit)
                        ss_res = np.sum(residuals ** 2)
                        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot)
                        # print(f'R^2$: {r_squared:0.4f}  slope: {np.round(lin_fit[0], 4)}')
                        slope_list.append(lin_fit[0])

                        fig, ax = fig_init()
                        plt.scatter(im_, pred, s=10, c=mc)
                        plt.xlim([0.3, 1.6])
                        # plt.ylim([0.3, 1.6])
                        plt.xlabel(r'true neuromodulation', fontsize=48)
                        plt.ylabel(r'learned neuromodulation', fontsize=48)
                        plt.text(0.35, 1.5, f'$R^2$: {r_squared:0.2f}  slope: {np.round(lin_fit[0], 2)}', fontsize=42)
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/results/field/comparison {epoch}_{frame}.png", dpi=80)
                        plt.close()
                        im_list.append(im_)
                        pred_list.append(pred)

                        pred = model_f(time=frame / n_frames, enlarge=True) ** 2 * second_correction / 10 # /lin_fit[0]
                        pred = torch.reshape(pred, (640, 640))
                        pred = to_numpy(pred)
                        pred = np.flipud(pred)
                        pred = np.rot90(pred, 1)
                        pred = np.fliplr(pred)
                        fig, ax = fig_init()
                        plt.imshow(pred, cmap='gray')
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/results/field/reconstructed_field_HR {epoch}_{frame}.png", dpi=80)
                        plt.close()

                    im_list = np.array(np.array(im_list))
                    pred_list = np.array(np.array(pred_list))

                    im_list_ = np.reshape(im_list,(100,1024))
                    pred_list_ = np.reshape(pred_list,(100,1024))
                    im_list_ = np.rot90(im_list_)
                    pred_list_ = np.rot90(pred_list_)
                    im_list_ = scipy.ndimage.zoom(im_list_, (1024 / im_list_.shape[0], 1024 / im_list_.shape[1]))
                    pred_list_ = scipy.ndimage.zoom(pred_list_, (1024 / pred_list_.shape[0], 1024 / pred_list_.shape[1]))

                    plt.figure(figsize=(20, 10))
                    plt.subplot(1, 2, 1)
                    plt.title('true field')
                    plt.imshow(im_list_, cmap='grey')
                    plt.xticks([])
                    plt.yticks([])
                    plt.subplot(1, 2, 2)
                    plt.title('reconstructed field')
                    plt.imshow(pred_list_, cmap='grey')
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/pic_comparison {epoch}.png", dpi=80)
                    plt.close()

                    fig, ax = fig_init()
                    plt.scatter(im_list, pred_list, s=1, c=mc, alpha=0.1)
                    plt.xlim([0.3, 1.6])
                    plt.ylim([0.3, 1.6])
                    plt.xlabel(r'true $\Omega_i$', fontsize=68)
                    plt.ylabel(r'learned $\Omega_i$', fontsize=68)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/all_comparison {epoch}.png", dpi=80)
                    plt.close()

                    x_data = np.reshape(im_list, (100 * n_nodes_per_axis * n_nodes_per_axis))
                    y_data = np.reshape(pred_list, (100 * n_nodes_per_axis * n_nodes_per_axis))
                    lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                    residuals = y_data - linear_model(x_data, *lin_fit)
                    ss_res = np.sum(residuals ** 2)
                    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot)
                    print(f'field R^2$: {r_squared:0.4f}  slope: {np.round(lin_fit[0], 4)}')

            if 'PDE_N6' in model_config.signal_model_name:

                modulation = torch.tensor(x_list[0], device=device)
                modulation = modulation[:, :, 8:9].squeeze()
                modulation = modulation.t()
                modulation = modulation.clone().detach()
                modulation = to_numpy(modulation)

                modulation = scipy.ndimage.zoom(modulation, (1024 / modulation.shape[0], 1024 / modulation.shape[1]))
                pred_list_ = to_numpy(model.b**2)
                pred_list_ = scipy.ndimage.zoom(pred_list_, (1024 / pred_list_.shape[0], 1024 / pred_list_.shape[1]))

                plt.figure(figsize=(20, 10))
                plt.subplot(1, 2, 1)
                plt.title('true field')
                plt.imshow(modulation, cmap='grey')
                plt.xticks([])
                plt.yticks([])
                plt.subplot(1, 2, 2)
                plt.title('reconstructed field')
                plt.imshow(pred_list_, cmap='grey')
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/pic_comparison {epoch}.png", dpi=80)
                plt.close()

                for frame in trange(0, modulation.shape[1], modulation.shape[1] // 257, ncols=90):
                    im = modulation[:, frame]
                    im = np.reshape(im, (32, 32))
                    plt.figure(figsize=(8, 8))
                    plt.axis('off')
                    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/field/true_field_{frame}.png", dpi=80)
                    plt.close()

            if (model.update_type == 'generic') & (model_config.signal_model_name == 'PDE_N5'):

                k = np.random.randint(n_frames - 50)
                x = torch.tensor(x_list[0][k], device=device)
                if has_field:
                    if 'visual' in field_type:
                        x[:n_nodes, 8:9] = model_f(time=k / n_frames) ** 2
                        x[n_nodes:n_neurons, 8:9] = 1
                    elif 'learnable_short_term_plasticity' in field_type:
                        alpha = (k % model.embedding_step) / model.embedding_step
                        x[:, 8] = alpha * model.b[:, k // model.embedding_step + 1] ** 2 + (1 - alpha) * model.b[:,
                                                                                                         k // model.embedding_step] ** 2
                    elif ('short_term_plasticity' in field_type) | ('modulation_permutation' in field_type):
                        t = torch.tensor([k / n_frames], dtype=torch.float32, device=device)
                        x[:, 8] = model_f(t) ** 2
                    else:
                        x[:, 8:9] = model_f(time=k / n_frames) ** 2
                else:
                    x[:, 8:9] = torch.ones_like(x[:, 0:1])
                dataset = data.Data(x=x, edge_index=edge_index)
                pred, in_features_ = model(data=dataset, return_all=True)
                feature_list = ['u', 'embedding0', 'embedding1', 'msg', 'field']
                for n in range(in_features_.shape[1]):
                    print(f'feature {feature_list[n]}: {to_numpy(torch.mean(in_features_[:, n])):0.4f}  std: {to_numpy(torch.std(in_features_[:, n])):0.4f}')

                fig, ax = fig_init()
                plt.hist(to_numpy(in_features_[:, -1]), 150)
                plt.tight_layout()
                plt.close()

                fig, ax = fig_init()
                f = torch.reshape(x[:n_nodes, 8:9], (n_nodes_per_axis, n_nodes_per_axis))
                plt.imshow(to_numpy(f), cmap='viridis', vmin=-1, vmax=10)
                plt.tight_layout()
                plt.close()


                fig, ax = fig_init()
                msg_list = []
                u = torch.linspace(-xnorm.squeeze(), xnorm.squeeze(), 400).to(device)
                for sample in range(n_neurons):
                    id0 = np.random.randint(0, n_neurons)
                    id1 = np.random.randint(0, n_neurons)
                    f = x[id0, 8:9]
                    embedding0 = model.a[id0, :] * torch.ones((400, config.graph_model.embedding_dim), device=device)
                    embedding1 = model.a[id1, :] * torch.ones((400, config.graph_model.embedding_dim), device=device)
                    in_features = torch.cat((u[:, None], embedding0, embedding1), dim=1)
                    msg = model.lin_edge(in_features.float()) ** 2 * correction
                    in_features = torch.cat((torch.zeros((400, 1), device=device), embedding0, msg,
                                             f * torch.ones((400, 1), device=device)), dim=1)
                    plt.plot(to_numpy(u), to_numpy(msg), c=cmap.color(to_numpy(x[id0, 5]).astype(int)), linewidth=2, alpha=0.15)
                    # plt.scatter(to_numpy(u), to_numpy(model.lin_phi(in_features)), s=5, c='r', alpha=0.15)
                    # plt.scatter(to_numpy(u), to_numpy(f*msg), s=1, c='w', alpha=0.1)
                    msg_list.append(msg)
                plt.tight_layout()
                msg_list = torch.stack(msg_list).squeeze()
                y_min, y_max = msg_list.min().item(), msg_list.max().item()
                plt.xlabel(r'$v_i$', fontsize=68)
                plt.ylabel(r'learned $\mathrm{MLP_0}$', fontsize=68)
                plt.ylim([y_min - y_max / 2, y_max * 1.5])
                plt.tight_layout()
                plt.savefig(f'./{log_dir}/results/learned_multiple_psi_{epoch}.png', dpi=300)
                plt.close()

                fig, ax = fig_init()
                u = torch.linspace(-xnorm.squeeze(), xnorm.squeeze(), 400).to(device)
                for n in range(n_neuron_types):
                    for m in range(n_neuron_types):
                        true_func = true_model.func(u, n, m, 'phi')
                        plt.plot(to_numpy(u), to_numpy(true_func), c=cmap.color(n), linewidth=3)
                plt.xlabel(r'$v_i$', fontsize=68)
                plt.ylabel(r'true functions', fontsize=68)
                plt.ylim([y_min - y_max / 2, y_max * 1.5])
                plt.tight_layout()
                plt.savefig(f'./{log_dir}/results/true_multiple_psi.png', dpi=300)
                plt.close()

                msg_start = torch.mean(in_features_[:, 3]) - torch.std(in_features_[:, 3])
                msg_end = torch.mean(in_features_[:, 3]) + torch.std(in_features_[:, 3])
                msgs = torch.linspace(msg_start, msg_end, 400).to(device)
                fig, ax = fig_init()
                func_list = []
                rr_list = []
                for sample in range(n_neurons):
                    id0 = np.random.randint(0, n_neurons)
                    embedding0 = model.a[id0, :] * torch.ones((400, config.graph_model.embedding_dim), device=device)
                    in_features = torch.cat((torch.zeros((400, 1), device=device), embedding0, msgs[:,None], torch.ones((400, 1), device=device)), dim=1)
                    pred = model.lin_phi(in_features)
                    plt.plot(to_numpy(msgs), to_numpy(pred), c=cmap.color(to_numpy(x[id0, 5]).astype(int)),  linewidth=2, alpha=0.25)
                    func_list.append(pred)
                    rr_list.append(msgs)
                plt.xlabel(r'$sum_i$', fontsize=68)
                plt.ylabel(r'$\mathrm{MLP_0}(\mathbf{a}_i, x_i=0, sum_i, g_i=1)$', fontsize=48)
                plt.tight_layout()
                plt.savefig(f'./{log_dir}/results/learned_multivariate_phi_{epoch}.png', dpi=300)
                plt.close()


                print('symbolic regression ...')

                text_trap = StringIO()
                sys.stdout = text_trap

                model_pysrr = PySRRegressor(
                    niterations=30,  # < Increase me for better results
                    binary_operators=["+", "*"],
                    unary_operators=[
                        "cos",
                        "exp",
                        "sin",
                        "tanh"
                    ],
                    random_state=0,
                    temp_equation_file=False
                )

                # rr_ = torch.rand((4000, 2), device=device)
                # func_ = rr_[:,0] * rr_[:,1]
                # model_pysrr.fit(to_numpy(rr_), to_numpy(func_))
                # model_pysrr.sympy

                func_list = torch.stack(func_list).squeeze()
                rr_list = torch.stack(rr_list).squeeze()
                func = torch.reshape(func_list, (func_list.shape[0] * func_list.shape[1], 1))
                rr = torch.reshape(rr_list, (func_list.shape[0] * func_list.shape[1], 1))
                idx = torch.randperm(len(rr))[:5000]

                model_pysrr.fit(to_numpy(rr[idx]), to_numpy(func[idx]))

                sys.stdout = sys.__stdout__


                # if model_config.signal_model_name == 'PDE_N4':
                #
                #     fig, ax = fig_init()
                #     for m in range(n_neuron_types):
                #         u = torch.linspace(-xnorm.squeeze() * 2, xnorm.squeeze() * 2, 400).to(device)
                #         true_func = true_model.func(u, m, 'phi')
                #         embedding0 = model.a[m * n_neurons // n_neuron_types, :] * torch.ones(
                #             (400, config.graph_model.embedding_dim), device=device)
                #         field = torch.ones((400, 1), device=device)
                #         in_features = torch.cat((u[:, None], embedding0), dim=1)
                #         if config.graph_model.lin_edge_positive:
                #             MLP0_func = model.lin_edge(in_features.float()) ** 2 * correction
                #         in_features = torch.cat((u[:, None] * 0, embedding0, MLP0_func, field), dim=1)
                #         MLP1_func = model.lin_phi(in_features)
                #         plt.plot(to_numpy(u), to_numpy(true_func), c='g', linewidth=3, label='true')
                #         plt.plot(to_numpy(u), to_numpy(MLP0_func), c='r', linewidth=3, label='MLP')
                #         plt.plot(to_numpy(u), to_numpy(MLP1_func), c='w', linewidth=3, label='MLPoMLP')
                #         # plt.legend(fontsize=24)
                #     plt.tight_layout()
                #     plt.savefig(f'./{log_dir}/results/generic_MLP0_{epoch}.png', dpi=300)
                #     plt.close()

            if False:
                print ('symbolic regression ...')

                def get_pyssr_function(model_pysrr, rr, func):

                    text_trap = StringIO()
                    sys.stdout = text_trap

                    model_pysrr.fit(to_numpy(rr[:, None]), to_numpy(func[:, None]))

                    sys.stdout = sys.__stdout__

                    return model_pysrr.sympy

                model_pysrr = PySRRegressor(
                    niterations=30,  # < Increase me for better results
                    binary_operators=["+", "*"],
                    unary_operators=[
                        "cos",
                        "exp",
                        "sin",
                        "tanh"
                    ],
                    random_state=0,
                    temp_equation_file=False
                )

                match model_config.signal_model_name:

                    case 'PDE_N2':

                        func = torch.mean(psi_list, dim=0).squeeze()

                        symbolic = get_pyssr_function(model_pysrr, rr, func)

                        for n in range(0,7):
                            print(symbolic(n))
                            logger.info(symbolic(n))

                    case 'PDE_N4':

                        for k in range(n_neuron_types):
                            print('  ')
                            print('  ')
                            print('  ')
                            print(f'psi{k} ................')
                            logger.info(f'psi{k} ................')

                            pos = np.argwhere(labels == k)
                            pos = pos.squeeze()

                            func = psi_list[pos]
                            func = torch.mean(psi_list[pos], dim=0)

                            symbolic = get_pyssr_function(model_pysrr, rr, func)

                            # for n in range(0, 5):
                            #     print(symbolic(n))
                            #     logger.info(symbolic(n))

                    case 'PDE_N5':

                        for k in range(4**2):

                            print('  ')
                            print('  ')
                            print('  ')
                            print(f'psi {k//4} {k%4}................')
                            logger.info(f'psi {k//4} {k%4} ................')

                            pos =np.arange(k*250,(k+1)*250)
                            func = psi_list[pos]
                            func = torch.mean(psi_list[pos], dim=0)

                            symbolic = get_pyssr_function(model_pysrr, rr, func)

                            # for n in range(0, 7):
                            #     print(symbolic(n))
                            #     logger.info(symbolic(n))

                for k in range(n_neuron_types):
                    print('  ')
                    print('  ')
                    print('  ')
                    print(f'phi{k} ................')
                    logger.info(f'phi{k} ................')

                    pos = np.argwhere(labels == k)
                    pos = pos.squeeze()

                    func = phi_list[pos]
                    func = torch.mean(phi_list[pos], dim=0)

                    symbolic = get_pyssr_function(model_pysrr, rr, func)

                    # for n in range(4, 7):
                    #     print(symbolic(n))
                    #     logger.info(symbolic(n))



def plot_synaptic3(config, epoch_list, log_dir, logger, cc, style, extended, device):

    dataset_name = config.dataset

    model_config = config.graph_model

    n_frames = config.simulation.n_frames
    n_runs = config.training.n_runs
    n_neuron_types = config.simulation.n_neuron_types
    p = config.simulation.params
    omega = model_config.omega
    cmap = CustomColorMap(config=config)
    dimension = config.simulation.dimension
    max_radius = config.simulation.max_radius
    EmbeddingCluster(config)
    field_type = model_config.field_type
    if field_type != '':
        n_nodes = config.simulation.n_nodes
        n_nodes_per_axis = int(np.sqrt(n_nodes))
        has_field = True
    else:
        has_field = False

    x_list = []
    y_list = []
    for run in trange(1, ncols=90):
        if os.path.exists(f'graphs_data/{dataset_name}/x_list_{run}.pt'):
            x = torch.load(f'graphs_data/{dataset_name}/x_list_{run}.pt', map_location=device)
            y = torch.load(f'graphs_data/{dataset_name}/y_list_{run}.pt', map_location=device)
            x = to_numpy(torch.stack(x))
            y = to_numpy(torch.stack(y))
        else:
            x = np.load(f'graphs_data/{dataset_name}/x_list_{run}.npy')
            y = np.load(f'graphs_data/{dataset_name}/y_list_{run}.npy')
        x_list.append(x)
        y_list.append(y)
    vnorm = torch.load(os.path.join(log_dir, 'vnorm.pt'))
    ynorm = torch.load(os.path.join(log_dir, 'ynorm.pt'))
    print(f'vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')

    print('update variables ...')
    x = x_list[0][n_frames - 1]
    n_neurons = x.shape[0]
    print(f'N neurons: {n_neurons}')
    logger.info(f'N neurons: {n_neurons}')
    config.simulation.n_neurons = n_neurons
    type_list = torch.tensor(x[:, 1 + 2 * dimension:2 + 2 * dimension], device=device)

    activity = torch.tensor(x_list[0],device=device)
    activity = activity[:, :, 6:7].squeeze()
    distrib = to_numpy(activity.flatten())
    activity = activity.t()

    type = x_list[0][0][:, 5]
    type_stack = torch.tensor(type, dtype=torch.float32, device=device)
    type_stack = type_stack[:,None].repeat(n_frames,1)


    if has_field:
        model_f = Siren_Network(image_width=n_nodes_per_axis, in_features=model_config.input_size_nnr, out_features=model_config.output_size_nnr, hidden_features=model_config.hidden_dim_nnr,
                                        hidden_layers=model_config.n_layers_nnr, outermost_linear=True, device=device, first_omega_0=omega, hidden_omega_0=omega)
        model_f.to(device=device)
        model_f.train()

    if 'black' in style:
        mc = 'w'
    else:
        mc = 'k'

    if epoch_list[0] == 'all':

        files = glob.glob(f"{log_dir}/models/*.pt")
        files.sort(key=os.path.getmtime)

        model, bc_pos, bc_dpos = choose_training_model(config, device)

        true_model, bc_pos, bc_dpos = choose_model(config=config, W=[], device=device)

        # plt.rcParams['text.usetex'] = False
        # plt.rc('font', family='sans-serif')
        # plt.rc('text', usetex=False)
        # matplotlib.rcParams['savefig.pad_inches'] = 0

        files = glob.glob(f"{log_dir}/models/best_model_with_{n_runs-1}_graphs_*.pt")
        files.sort(key=sort_key)

        flag = True
        file_id = 0
        while (flag):
            if sort_key(files[file_id]) >0:
                flag = False
                file_id = file_id - 1
            file_id += 1

        files = files[file_id:]

        # file_id_list0 = np.arange(0, file_id, file_id // 90)
        # file_id_list1 = np.arange(file_id, len(files), (len(files) - file_id) // 40)
        # file_id_list = np.concatenate((file_id_list0, file_id_list1))

        file_id_list = np.arange(0, len(files), (len(files)/100)).astype(int)
        r_squared_list = []
        slope_list = []

        with torch.no_grad():
            for file_id_ in trange(0, 100, ncols=90):
                file_id = file_id_list[file_id_]

                epoch = files[file_id].split('graphs')[1][1:-3]
                net = f"{log_dir}/models/best_model_with_{n_runs-1}_graphs_{epoch}.pt"
                state_dict = torch.load(net, map_location=device)
                model.load_state_dict(state_dict['model_state_dict'])
                model.eval()

                if has_field:
                    net = f'{log_dir}/models/best_model_f_with_{n_runs-1}_graphs_{epoch}.pt'
                    state_dict = torch.load(net, map_location=device)
                    model_f.load_state_dict(state_dict['model_state_dict'])

                amax = torch.max(model.a, dim=0).values
                amin = torch.min(model.a, dim=0).values
                (model.a - amin) / (amax - amin)

                # fig, ax = fig_init()
                # for n in range(n_neuron_types):
                #     c1 = cmap.color(n)
                #     c2 = cmap.color((n+1)%4)
                #     c_list = np.linspace(c1, c2, 100)
                #     for k in range(250*n,250*(n+1)):
                #         plt.scatter(to_numpy(model.a[k*100:(k+1)*100, 0:1]), to_numpy(model.a[k*100:(k+1)*100, 1:2]), s=10, color=c_list, alpha=0.1, edgecolors='none')
                # if 'latex' in style:
                #     plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}(t)$', fontsize=68)
                #     plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}(t)$', fontsize=68)
                # else:
                #     plt.xlabel(r'$a_{i0}(t)$', fontsize=68)
                #     plt.ylabel(r'$a_{i1}(t)$', fontsize=68)
                # plt.xlim([0.94, 1.08])
                # plt.ylim([0.9, 1.10])
                # # plt.xlim([0.7, 1.2])
                # # plt.ylim([0.7, 1.2])
                # plt.tight_layout()
                # plt.savefig(f"./{log_dir}/results/all/all_embedding_0_{epoch}.png", dpi=80)
                # plt.close()

                fig, ax = fig_init()
                for k in range(n_neuron_types):
                    # plt.scatter(to_numpy(model.a[:, 0]), to_numpy(model.a[:, 1]), s=1, color=mc, alpha=0.5, edgecolors='none')
                    plt.scatter(to_numpy(model.a[k*25000:(k+1)*25000, 0]), to_numpy(model.a[k*25000:(k+1)*25000, 1]), s=1, color=cmap.color(k),alpha=0.5, edgecolors='none')
                    # plt.scatter(to_numpy(model.a[k * 25000: k * 25000 + 100, 0]),
                    #             to_numpy(model.a[k * 25000: k * 25000 + 100, 1]), s=10, color=c_list, alpha=1)
                if 'latex' in style:
                    plt.xlabel(r'$\ensuremath{\mathbf{a}}_{0}(t)$', fontsize=68)
                    plt.ylabel(r'$\ensuremath{\mathbf{a}}_{1}(t)$', fontsize=68)
                else:
                    plt.xlabel(r'$a_{0}(t)$', fontsize=68)
                    plt.ylabel(r'$a_{1}(t)$', fontsize=68)
                # plt.xlim([0.94, 1.08])
                # plt.ylim([0.9, 1.10])
                plt.xlim([0.7, 1.2])
                plt.ylim([0.7, 1.2])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/all_embedding_1_{epoch}.png", dpi=80)
                plt.close()

                correction = torch.load(f'{log_dir}/correction.pt',map_location=device)
                second_correction = np.load(f'{log_dir}/second_correction.npy')

                i, j = torch.triu_indices(n_neurons, n_neurons, requires_grad=False, device=device)
                A = model.W.clone().detach() / correction
                A[i, i] = 0

                fig, ax = fig_init()
                ax = sns.heatmap(to_numpy(A)/second_correction, center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046}, vmin=-0.1,vmax=0.1)
                cbar = ax.collections[0].colorbar
                cbar.ax.tick_params(labelsize=48)
                plt.xticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
                plt.yticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
                plt.subplot(2, 2, 1)
                ax = sns.heatmap(to_numpy(A[0:20, 0:20])/second_correction, cbar=False, center=0, square=True, cmap='bwr', vmin=-0.1, vmax=0.1)
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                # plt.savefig(f"./{log_dir}/results/all/W_{epoch}.png", dpi=80)
                plt.close()

                rr = torch.tensor(np.linspace(-5, 5, 1000)).to(device)
                func_list = []
                k_list = [0, 250, 500, 750]
                fig, ax = fig_init()
                plt.axis('off')
                for it, k in enumerate(k_list):
                    ax = plt.subplot(2, 2, it + 1)
                    c1 = cmap.color(it)
                    c2 = cmap.color((it + 1) % 4)
                    c_list = np.linspace(c1, c2, 100)
                    for n in range(k * 100, (k + 1) * 100):
                        embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                        in_features = get_in_features(rr, embedding_, model_config.signal_model_name, max_radius)
                        with torch.no_grad():
                            func = model.lin_phi(in_features.float())
                        func_list.append(func)
                        # plt.plot(to_numpy(rr), to_numpy(func), 2, color=c_list[n%100], alpha=0.25)
                        # linewidth=4, alpha=0.15-0.15*(n%100)/100)
                        plt.plot(to_numpy(rr), to_numpy(func), 2, color=cmap.color(it), alpha=0.25)
                    # true_func = true_model.func(rr, it, 'update')
                    # plt.plot(to_numpy(rr), to_numpy(true_func), c=mc, linewidth=1)
                    # true_func = true_model.func(rr, it + 1, 'update')
                    # plt.plot(to_numpy(rr), to_numpy(true_func), c=mc, linewidth=1)
                    # plt.xlabel(r'$v_i$', fontsize=24)
                    # plt.ylabel(r'Learned $\phi^*(\mathbf{a}_i(t), v_i)$', fontsize=68)
                    if k==0:
                        plt.ylabel(r'Learned $\mathrm{MLP_0}(\mathbf{a}_i(t), v_i)$', fontsize=32)
                    plt.ylim([-8, 8])
                    plt.xlim([-5, 5])
                    plt.xticks([])
                    plt.yticks([])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/MLP0_{epoch}.png", dpi=80)
                plt.close()

                fig, ax = fig_init()
                for n in range(0, n_neurons):
                    in_features = rr[:, None]
                    with torch.no_grad():
                        func = model.lin_edge(in_features.float()) * correction
                        plt.plot(to_numpy(rr), to_numpy(func), 2, color=mc, linewidth=2, alpha=0.25)
                plt.xlabel(r'$x_j$', fontsize=68)
                # plt.ylabel(r'learned $\psi^*(v_i)$', fontsize=68)
                plt.ylabel(r'learned $\mathrm{MLP_1}(v_j)$', fontsize=68)
                plt.ylim([-1.1, 1.1])
                plt.xlim(config.plotting.xlim)
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/MLP1_{epoch}.png", dpi=80)
                plt.close()

                adjacency = torch.load(f'./graphs_data/{dataset_name}/adjacency.pt', map_location=device)
                adjacency_ = adjacency.t().clone().detach()
                adj_t = torch.abs(adjacency_) > 0
                edge_index = adj_t.nonzero().t().contiguous()

                i, j = torch.triu_indices(n_neurons, n_neurons, requires_grad=False, device=device)
                A = model.W.clone().detach() / correction
                A[i, i] = 0

                fig, ax = fig_init()
                gt_weight = to_numpy(adjacency)
                pred_weight = to_numpy(A)
                plt.scatter(gt_weight, pred_weight / 10 , s=0.1, c=mc, alpha=0.1)
                plt.xlabel(r'true $W_{ij}$', fontsize=68)
                plt.ylabel(r'learned $W_{ij}$', fontsize=68)
                if n_neurons == 8000:
                    plt.xlim([-0.05, 0.05])
                    plt.ylim([-0.05, 0.05])
                else:
                    plt.xlim([-0.2, 0.2])
                    plt.ylim([-0.2, 0.2])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/comparison_{epoch}.png", dpi=80)
                plt.close()

                x_data = np.reshape(gt_weight, (n_neurons * n_neurons))
                y_data = np.reshape(pred_weight, (n_neurons * n_neurons))
                lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                residuals = y_data - linear_model(x_data, *lin_fit)
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                r_squared_list.append(r_squared)
                slope_list.append(lin_fit[0])

                if has_field:

                    fig, ax = fig_init()
                    pred = model_f(time=file_id_ / len(file_id_list), enlarge=True) ** 2
                    # pred = torch.reshape(pred, (n_nodes_per_axis, n_nodes_per_axis))
                    pred = torch.reshape(pred, (640, 640))
                    pred = to_numpy(torch.sqrt(pred))
                    pred = np.flipud(pred)
                    pred = np.rot90(pred, 1)
                    pred = np.fliplr(pred)
                    plt.imshow(pred, cmap='grey')
                    plt.ylabel(r'learned $MLP_2(x_i, t)$', fontsize=68)
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/all/field_{epoch}.png", dpi=80)
                    plt.close()

        fig, ax = fig_init(formatx='%.0f', formaty='%.2f')
        plt.plot(r_squared_list, linewidth=4, c=mc)
        plt.xlim([0, 100])
        plt.ylim([0, 1.1])
        plt.yticks(fontsize=48)
        plt.xticks([0, 100], [0, 20], fontsize=48)
        plt.ylabel('$R^2$', fontsize=64)
        plt.xlabel('epoch', fontsize=64)
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/results/R2.png', dpi=300)
        plt.close()
        np.save(f'./{log_dir}/results/R2.npy', r_squared_list)

        slope_list = np.array(slope_list) / p[0][0]
        fig, ax = fig_init(formatx='%.0f', formaty='%.2f')
        plt.plot(slope_list, linewidth=4, c=mc)
        plt.xlim([0, 100])
        plt.ylim([0, 1.1])
        plt.yticks(fontsize=48)
        plt.xticks([0, 100], [0, 20], fontsize=48)
        plt.ylabel('slope', fontsize=64)
        plt.xlabel('epoch', fontsize=64)
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/results/slope.png', dpi=300)
        plt.close()

    elif epoch_list[0] == 'time':

        files = glob.glob(f"{log_dir}/models/*")
        files.sort(key=sort_key)
        filename = files[-1]
        filename = filename.split('/')[-1]
        filename = filename.split('graphs')[-1][1:-3]

        epoch = filename
        net = f'{log_dir}/models/best_model_with_{n_runs - 1}_graphs_{epoch}.pt'
        model, bc_pos, bc_dpos = choose_training_model(config, device)
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        print(f'net: {net}')

        adjacency = torch.load(f'./graphs_data/{dataset_name}/adjacency.pt', map_location=device)
        true_model, bc_pos, bc_dpos = choose_model(config=config, W=adjacency, device=device)

        for n in trange(100, ncols=90):

            indices = np.arange(n_neurons)*100+n

            fig, ax = fig_init()
            plt.scatter(to_numpy(model.a[:, 0]), to_numpy(model.a[:, 1]), s=1, color=mc, alpha=0.01)
            for k in range(n_neuron_types):
                plt.scatter(to_numpy(model.a[indices[k * 250:(k + 1) * 250], 0]),
                            to_numpy(model.a[indices[k * 250:(k + 1) * 250], 1]), s=100, color=cmap.color(k), alpha=0.5,
                            edgecolors='none')
            if 'latex' in style:
                plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i}(t)$', fontsize=68)
                plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i}(t)$', fontsize=68)
            else:
                plt.xlabel(r'$a_{i}(t)$', fontsize=68)
                plt.ylabel(r'$a_{i}(t)$', fontsize=68)
            # plt.xlim([0.92, 1.08])
            # plt.ylim([0.9, 1.10])
            # plt.text(0.93, 1.08, f'time: {n}', fontsize=48)

            plt.xlim([0.7, 1.2])
            plt.ylim([0.7, 1.2])
            plt.text(0.72, 1.16, f'time: {n}', fontsize=48)

            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/all2/all_embedding_1_{n}.png", dpi=80)
            plt.close()


            rr = torch.tensor(np.linspace(-5, 5, 1000)).to(device)
            func_list = []
            fig, ax = fig_init()
            plt.axis('off')
            ax = plt.subplot(2, 2, 1)
            plt.ylabel(r'learned $\mathrm{MLP_0}(\mathbf{a}_i(t), v_i)$', fontsize=38)
            for it, k in enumerate(indices):
                if (it%250 == 0) and (it>0):
                    ax = plt.subplot(2, 2, it//250+1)
                plt.xticks([])
                plt.yticks([])
                embedding_ = model.a[k, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                in_features = get_in_features(rr, embedding_, model_config.signal_model_name, max_radius)
                with torch.no_grad():
                    func = model.lin_phi(in_features.float())
                    plt.plot(to_numpy(rr), to_numpy(func), 2, color=cmap.color(it//250), alpha=0.5)
                plt.ylim([-8,8])
                plt.xlim([-5,5])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/all2/phi_{n}.png", dpi=80)
            plt.close()

            # rr = torch.tensor(np.linspace(-5, 5, 1000)).to(device)
            # k_list = [0, 250, 500, 750]
            # fig, ax = fig_init()
            # plt.axis('off')
            # for it, k in enumerate(k_list):
            #     ax = plt.subplot(2, 2, it + 1)
            #     for n in range(k * 100, (k + 25) * 100):
            #         embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
            #         in_features = get_in_features(rr, embedding_, model_config.signal_model_name, max_radius)
            #         with torch.no_grad():
            #             func = model.lin_phi(in_features.float())
            #         plt.plot(to_numpy(rr), to_numpy(func), 2, color=mc, alpha=0.025)
            #     plt.xlabel(r'$v_i$', fontsize=16)
            #     # plt.ylabel(r'Learned $\phi^*(\mathbf{a}_i(t), v_i)$', fontsize=68)
            #     plt.ylabel(r'Learned $\mathrm{MLP_0}(\mathbf{a}_i(t), v_i)$', fontsize=16)
            #
            #
            # for k in range(n_neuron_types):
            #     ax = plt.subplot(2, 2, k + 1)
            #     n = indices[k * 250:(k + 1) * 250]
            #     plt.scatter(to_numpy(model.a[indices[k * 250:(k + 1) * 250], 0]),
            #                 to_numpy(model.a[indices[k * 250:(k + 1) * 250], 1]), s=10, color=cmap.color(k),
            #                 alpha=0.5,
            #                 edgecolors='none')
            #
            #     plt.ylim([-8, 8])
            #     plt.xlim([-5, 5])
            #     plt.tight_layout()
            #
            # plt.savefig(f"./{log_dir}/results/all/MLP0_{epoch}.png", dpi=80)
            # plt.close()

    else:

        fig_init(formatx='%.0f', formaty='%.0f')
        plt.hist(distrib, bins=100, color=mc, alpha=0.5)
        plt.ylabel('counts', fontsize=64)
        plt.xlabel('$x_{ij}$', fontsize=64)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/results/signal_distribution.png', dpi=300)
        plt.close()
        print(f'mean: {np.mean(distrib):0.2f}  std: {np.std(distrib):0.2f}')
        logger.info(f'mean: {np.mean(distrib):0.2f}  std: {np.std(distrib):0.2f}')

        # plt.figure(figsize=(15, 10))
        # ax = sns.heatmap(to_numpy(activity), center=0, cmap='viridis', cbar_kws={'fraction': 0.046})
        # cbar = ax.collections[0].colorbar
        # cbar.ax.tick_params(labelsize=32)
        # ax.invert_yaxis()
        # plt.ylabel('neurons', fontsize=64)
        # plt.xlabel('time', fontsize=64)
        # plt.xticks([1000, 9900], [1000, 10000], fontsize=48)
        # plt.yticks([0, 999], [1, 1000], fontsize=48)
        # plt.xticks(rotation=0)
        # plt.tight_layout()
        # plt.savefig(f'./{log_dir}/results/kinograph.png', dpi=300)
        # plt.close()
        #
        # plt.figure(figsize=(15, 10))
        # n = np.random.permutation(n_neurons)
        # for i in range(25):
        #     plt.plot(to_numpy(activity[n[i].astype(int), :]), linewidth=2)
        # plt.xlabel('time', fontsize=64)
        # plt.ylabel('$x_{i}$', fontsize=64)
        # plt.xticks([0, 10000], fontsize=48)
        # plt.yticks(fontsize=48)
        # plt.tight_layout()
        # plt.savefig(f'./{log_dir}/results/firing rate.png', dpi=300)
        # plt.close()

        adjacency = torch.load(f'./graphs_data/{dataset_name}/adjacency.pt', map_location=device)
        adjacency_ = adjacency.t().clone().detach()
        adj_t = torch.abs(adjacency_) > 0
        edge_index = adj_t.nonzero().t().contiguous()
        weights = to_numpy(adjacency.flatten())
        pos = np.argwhere(weights != 0)
        weights = weights[pos]

        fig_init()
        plt.hist(weights, bins=1000, color=mc, alpha=0.5)
        plt.ylabel(r'counts', fontsize=64)
        plt.xlabel(r'$W$', fontsize=64)
        plt.yticks(fontsize=24)
        plt.xticks(fontsize=24)
        plt.xlim([-0.1, 0.1])
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/results/weights_distribution.png', dpi=300)
        plt.close()

        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(to_numpy(adjacency), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046},
                         vmin=-0.1, vmax=0.1)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=32)
        plt.xticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
        plt.yticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
        plt.xticks(rotation=0)
        plt.subplot(2, 2, 1)
        ax = sns.heatmap(to_numpy(adjacency[0:20, 0:20]), cbar=False, center=0, square=True, cmap='bwr', vmin=-0.1, vmax=0.1)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/results/true connectivity.png', dpi=300)
        plt.close()

        true_model, bc_pos, bc_dpos = choose_model(config=config, W=adjacency, device=device)

        for epoch in epoch_list:

            net = f'{log_dir}/models/best_model_with_{n_runs-1}_graphs_{epoch}.pt'
            model, bc_pos, bc_dpos = choose_training_model(config, device)
            state_dict = torch.load(net, map_location=device)
            model.load_state_dict(state_dict['model_state_dict'])
            model.edges = edge_index
            print(f'net: {net}')

            if has_field:

                im = imread(f"graphs_data/{config.simulation.node_value_map}")

                net = f'{log_dir}/models/best_model_f_with_{n_runs-1}_graphs_{epoch}.pt'
                state_dict = torch.load(net, map_location=device)
                model_f.load_state_dict(state_dict['model_state_dict'])

                os.makedirs(f"./{log_dir}/results/field", exist_ok=True)
                x = x_list[0][0]

                slope_list = list([])
                im_list=list([])
                pred_list=list([])

                for frame in trange(0, n_frames, n_frames//100, ncols=90):

                    fig, ax = fig_init()
                    im_ = np.zeros((44,44))
                    if (frame>=0) & (frame<n_frames):
                        im_ =  im[int(frame / n_frames * 256)].squeeze()
                    plt.imshow(im_,cmap='gray',vmin=0,vmax=2)
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/field/true_field{epoch}_{frame}.png", dpi=80)
                    plt.close()


                    pred = model_f(time=frame / n_frames, enlarge=True) ** 2
                    # pred = torch.reshape(pred, (n_nodes_per_axis, n_nodes_per_axis))
                    pred = torch.reshape(pred, (640, 640))
                    pred = to_numpy(pred)
                    pred = np.flipud(pred)
                    pred = np.rot90(pred, 1)
                    pred = np.fliplr(pred)
                    fig, ax = fig_init()
                    plt.imshow(pred,cmap='gray',vmin=0,vmax=2)
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/field/reconstructed_field_HR {epoch}_{frame}.png", dpi=80)
                    plt.close()

                    pred = model_f(time=frame / n_frames, enlarge=False) ** 2
                    pred = torch.reshape(pred, (n_nodes_per_axis, n_nodes_per_axis))
                    pred = to_numpy(pred)
                    pred = np.flipud(pred)
                    pred = np.rot90(pred, 1)
                    pred = np.fliplr(pred)
                    fig, ax = fig_init()
                    plt.imshow(pred,cmap='gray',vmin=0,vmax=2)
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/field/reconstructed_field_LR {epoch}_{frame}.png", dpi=80)
                    plt.close()

                    x_data = np.reshape(im_, (n_nodes_per_axis * n_nodes_per_axis))
                    y_data = np.reshape(pred, (n_nodes_per_axis * n_nodes_per_axis))
                    lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                    residuals = y_data - linear_model(x_data, *lin_fit)
                    ss_res = np.sum(residuals ** 2)
                    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot)
                    print(f'R^2$: {r_squared:0.4f}  slope: {np.round(lin_fit[0], 4)}')
                    slope_list.append(lin_fit[0])

                    fig, ax = fig_init()
                    plt.scatter(im_,pred, s=10, c=mc)
                    plt.xlim([0.3,1.6])
                    plt.ylim([0.3,1.6])
                    plt.xlabel(r'true $\Omega_i$', fontsize=68)
                    plt.ylabel(r'learned $\Omega_i$', fontsize=68)
                    plt.text(0.5, 1.4, f'$R^2$: {r_squared:0.2f}  slope: {np.round(lin_fit[0], 2)}', fontsize=48)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/field/comparison {epoch}_{frame}.png", dpi=80)
                    plt.close()
                    im_list.append(im_)
                    pred_list.append(pred)

                im_list = np.array(np.array(im_list))
                pred_list = np.array(np.array(pred_list))

                fig, ax = fig_init()
                plt.scatter(im_list, pred_list, s=1, c=mc, alpha=0.1)
                plt.xlim([0.3, 1.6])
                plt.ylim([0.3, 1.6])
                plt.xlabel(r'true $\Omega_i$', fontsize=68)
                plt.ylabel(r'learned $\Omega_i$', fontsize=68)
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/field/all_comparison {epoch}.png", dpi=80)
                plt.close()

                x_data = np.reshape(im_list, (100 * n_nodes_per_axis * n_nodes_per_axis))
                y_data = np.reshape(pred_list, (100 * n_nodes_per_axis * n_nodes_per_axis))
                lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                residuals = y_data - linear_model(x_data, *lin_fit)
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                print(f'R^2$: {r_squared:0.4f}  slope: {np.round(lin_fit[0], 4)}')

            if model_config.embedding_dim == 4:
                for k in range(n_neuron_types):
                    fig = plt.figure(figsize=(10, 9))
                    ax = fig.add_subplot(111, projection='3d')
                    # ax.scatter(to_numpy(model.a[:, 0]), to_numpy(model.a[:, 1]), to_numpy(model.a[:, 2]), s=1, color=mc, alpha=0.01, edgecolors='none')
                    ax.scatter(to_numpy(model.a[k * 25000:(k + 1) * 25000, 0]),
                            to_numpy(model.a[k * 25000:(k + 1) * 25000, 1]), to_numpy(model.a[k * 25000:(k + 1) * 25000, 2]), s=0.1, color=cmap.color(k), alpha=0.5)
                    # ax.scatter(to_numpy(model.a[k*25000:k*25000+100, 0]), to_numpy(model.a[k*25000:k*25000+100, 1]), to_numpy(model.a[k*25000:k*25000+100, 1]), color=mc)
                    plt.ylim([0, 2])
                    plt.xlim([0, 2])
                    ax.set_zlim([-2, 3.5])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/embedding_{k}_{epoch}.png", dpi=80)
                    plt.close()

                fig = plt.figure(figsize=(10, 9))
                ax = fig.add_subplot(111, projection='3d')

                for k in range(n_neuron_types):
                    # ax.scatter(to_numpy(model.a[:, 0]), to_numpy(model.a[:, 1]), to_numpy(model.a[:, 2]), s=1, color=mc, alpha=0.01, edgecolors='none')
                    ax.scatter(to_numpy(model.a[k * 25000:(k + 1) * 25000, 0]),
                            to_numpy(model.a[k * 25000:(k + 1) * 25000, 1]), to_numpy(model.a[k * 25000:(k + 1) * 25000, 2]), s=10, color=cmap.color(k), alpha=0.1, edgecolors='none')
                    # ax.scatter(to_numpy(model.a[k*25000:k*25000+100, 0]), to_numpy(model.a[k*25000:k*25000+100, 1]), to_numpy(model.a[k*25000:k*25000+100, 1]), color=mc)
                plt.ylim([0.5, 2])
                plt.xlim([0, 1.5])
                ax.set_zlim([-2, 2.5])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all_embedding_{epoch}.png", dpi=80)
                plt.close()

            else:

                fig, ax = fig_init()
                for n in range(n_neuron_types):
                    c1 = cmap.color(n)
                    c2 = cmap.color((n+1)%4)
                    c_list = np.linspace(c1, c2, 100)
                    for k in range(250*n,250*(n+1)):
                        plt.scatter(to_numpy(model.a[k*100:(k+1)*100, 0:1]), to_numpy(model.a[k*100:(k+1)*100, 1:2]), s=10, color=c_list, alpha=0.1, edgecolors='none')
                if 'latex' in style:
                    plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}(t)$', fontsize=68)
                    plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}(t)$', fontsize=68)
                else:
                    plt.xlabel(r'$a_{i0}(t)$', fontsize=68)
                    plt.ylabel(r'$a_{i1}(t)$', fontsize=68)
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all_embedding_0_{epoch}.png", dpi=80)
                plt.close()

                fig, ax = fig_init()
                for k in range(n_neuron_types):
                    # plt.scatter(to_numpy(model.a[:, 0]), to_numpy(model.a[:, 1]), s=1, color=mc, alpha=0.5, edgecolors='none')
                    plt.scatter(to_numpy(model.a[k*25000:(k+1)*25000, 0]), to_numpy(model.a[k*25000:(k+1)*25000, 1]), s=1, color=cmap.color(k),alpha=0.5, edgecolors='none')
                    # plt.scatter(to_numpy(model.a[k * 25000: k * 25000 + 100, 0]),
                    #             to_numpy(model.a[k * 25000: k * 25000 + 100, 1]), s=10, color=c_list, alpha=1)
                if 'latex' in style:
                    plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}(t)$', fontsize=68)
                    plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}(t)$', fontsize=68)
                else:
                    plt.xlabel(r'$a_{i0}(t)$', fontsize=68)
                    plt.ylabel(r'$a_{i1}(t)$', fontsize=68)
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all_embedding_1_{epoch}.png", dpi=80)
                plt.close()

                for k in range(n_neuron_types):
                    fig, ax = fig_init()
                    # plt.scatter(to_numpy(model.a[0:100000, 0]), to_numpy(model.a[0:100000, 1]), s=1, color=mc, alpha=0.25, edgecolors='none')
                    plt.scatter(to_numpy(model.a[k*25000:(k+1)*25000, 0]), to_numpy(model.a[k*25000:(k+1)*25000, 1]), s=1, color=cmap.color(k),alpha=0.5, edgecolors='none')
                    if 'latex' in style:
                        plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=68)
                        plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=68)
                    else:
                        plt.xlabel(r'$a_{0}$', fontsize=68)
                        plt.ylabel(r'$a_{1}$', fontsize=68)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/embedding_{k}_{epoch}.png", dpi=80)
                    plt.close()


            rr = torch.tensor(np.linspace(-5, 5, 1000)).to(device)
            func_list = []
            k_list=[0,250,500,750]
            for it, k in enumerate(k_list):
                c1 = cmap.color(it)
                c2 = cmap.color((it + 1) % 4)
                c_list = np.linspace(c1, c2, 100)
                fig, ax = fig_init()
                for n in trange(k*100,(k+1)*100, ncols=90):
                    embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                    in_features = get_in_features(rr, embedding_, model_config.signal_model_name, max_radius)
                    with torch.no_grad():
                        func = model.lin_phi(in_features.float())
                    func_list.append(func)
                    # plt.plot(to_numpy(rr), to_numpy(func), 2, color=c_list[n%100], alpha=0.25)
                             # linewidth=4, alpha=0.15-0.15*(n%100)/100)
                    plt.plot(to_numpy(rr), to_numpy(func), 2, color=cmap.color(it), alpha=0.25)
                true_func = true_model.func(rr, it, 'update')
                plt.plot(to_numpy(rr), to_numpy(true_func), c=mc, linewidth=1)
                true_func = true_model.func(rr, it+1, 'update')
                plt.plot(to_numpy(rr), to_numpy(true_func), c=mc, linewidth=1)
                plt.xlabel(r'$v_i$', fontsize=68)
                plt.ylabel(r'Learned $\phi^*(\mathbf{a}_i(t), v_i)$', fontsize=68)
                plt.ylim([-8,8])
                plt.xlim([-5,5])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/phi_{k}.png", dpi=170.7)
                plt.close()

            fig, ax = fig_init()
            rr = torch.tensor(np.linspace(-5, 5, 1000)).to(device)
            func_list = []
            for n in trange(0,n_neurons,n_neurons, ncols=90):
                if (model_config.signal_model_name == 'PDE_N4') | (model_config.signal_model_name == 'PDE_N5'):
                    embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                    in_features = get_in_features(rr, embedding_, model_config.signal_model_name, max_radius)
                else:
                    in_features = rr[:, None]
                with torch.no_grad():
                    func = model.lin_edge(in_features.float())
                if (model_config.signal_model_name == 'PDE_N4') | (model_config.signal_model_name == 'PDE_N5'):
                    if n<250:
                        func_list.append(func)
                else:
                    func_list.append(func)
                plt.plot(to_numpy(rr), to_numpy(func), 2, color=cmap.color(to_numpy(type_list)[n].astype(int)),
                         linewidth=8 // ( 1 + (n_neuron_types>16)*1.0), alpha=0.25)
            func_list = torch.stack(func_list)
            plt.xlabel(r'$v_i$', fontsize=68)
            plt.ylabel(r'Learned $\psi^*(\mathbf{a}_i, v_i)$', fontsize=68)
            plt.xlim([-5,5])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/raw_psi.png", dpi=170.7)
            plt.close()

            correction = 1 / torch.mean(torch.mean(func_list[:,900:1000], dim=0))
            print(f'correction: {correction:0.2f}')
            torch.save(correction, f'{log_dir}/correction.pt')

            psi_list = []
            fig, ax = fig_init()
            rr = torch.tensor(np.linspace(-7.5, 7.5, 1500)).to(device)
            if model_config.signal_model_name == 'PDE_N4':
                for n in range(n_neuron_types):
                    true_func = true_model.func(rr, n, 'phi')
                    plt.plot(to_numpy(rr), to_numpy(true_func), c = 'k', linewidth = 16, label = 'original', alpha = 0.21)
            else:
                true_func = true_model.func(rr, 0, 'phi')
                plt.plot(to_numpy(rr), to_numpy(true_func), c = 'k', linewidth = 16, label = 'original', alpha = 0.21)

            for n in trange(0,n_neurons, ncols=90):
                in_features = rr[:, None]
                with torch.no_grad():
                    func = model.lin_edge(in_features.float()) * correction
                    psi_list.append(func)
                    plt.plot(to_numpy(rr), to_numpy(func), 2, color=mc, linewidth=2, alpha=0.25)
            plt.xlabel(r'$v_i$', fontsize=68)
            plt.ylabel(r'learned $\psi^*(v_i)$', fontsize=68)
            plt.ylim([-1.1, 1.1])
            plt.xlim(config.plotting.xlim)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/learned_psi.png", dpi=170.7)
            plt.close()


            i, j = torch.triu_indices(n_neurons, n_neurons, requires_grad=False, device=device)
            A = model.W.clone().detach() / correction
            A[i, i] = 0

            fig, ax = fig_init()
            gt_weight = to_numpy(adjacency)
            pred_weight = to_numpy(A)
            plt.scatter(gt_weight, pred_weight / 10 , s=0.1, c=mc, alpha=0.1)
            plt.xlabel(r'true $W_{ij}$', fontsize=68)
            plt.ylabel(r'learned $W_{ij}$', fontsize=68)
            if n_neurons == 8000:
                plt.xlim([-0.05,0.05])
                plt.ylim([-0.05,0.05])
            else:
                plt.xlim([-0.2,0.2])
                plt.ylim([-0.2,0.2])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/comparison_{epoch}.png", dpi=87)
            plt.close()

            x_data = np.reshape(gt_weight, (n_neurons * n_neurons))
            y_data =  np.reshape(pred_weight, (n_neurons * n_neurons))
            lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
            residuals = y_data - linear_model(x_data, *lin_fit)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            print(f'R^2$: {r_squared:0.4f}  slope: {np.round(lin_fit[0], 4)}')
            logger.info(f'R^2$: {np.round(r_squared, 4)}  slope: {np.round(lin_fit[0], 4)}')

            second_correction = lin_fit[0]
            print(f'second_correction: {second_correction:0.2f}')
            np.save(f'{log_dir}/second_correction.npy', second_correction)

            plt.figure(figsize=(10, 10))
            # plt.title(r'learned $W_{ij}$', fontsize=68)
            ax = sns.heatmap(to_numpy(A)/second_correction, center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046}, vmin=-0.1,vmax=0.1)
            cbar = ax.collections[0].colorbar
            # here set the labelsize by 20
            cbar.ax.tick_params(labelsize=32)
            plt.xticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
            plt.yticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
            plt.xticks(rotation=0)
            plt.subplot(2, 2, 1)
            ax = sns.heatmap(to_numpy(A[0:20, 0:20]/second_correction), cbar=False, center=0, square=True, cmap='bwr', vmin=-0.1, vmax=0.1)
            plt.xticks(rotation=0)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(f'./{log_dir}/results/learned connectivity.png', dpi=300)
            plt.close()



def plot_synaptic_CElegans(config, epoch_list, log_dir, logger, cc, style, extended, device):

    dataset_name = config.dataset


    model_config = config.graph_model

    n_frames = config.simulation.n_frames
    n_runs = config.training.n_runs
    n_neuron_types = config.simulation.n_neuron_types
    delta_t = config.simulation.delta_t
    p = config.simulation.params
    omega = model_config.omega
    cmap = CustomColorMap(config=config)
    dimension = config.simulation.dimension
    max_radius = config.simulation.max_radius
    time_step =  config.training.time_step
    field_type = model_config.field_type

    multi_connectivity = config.training.multi_connectivity
    has_missing_activity = config.training.has_missing_activity

    all_neuron_list = json.load(open(f'graphs_data/{dataset_name}/all_neuron_list.json', "r"))
    larynx_neuron_list = json.load(open(f'graphs_data/{dataset_name}/larynx_neuron_list.json', "r"))
    sensory_neuron_list = json.load(open(f'graphs_data/{dataset_name}/sensory_neuron_list.json', "r"))
    inter_neuron_list = json.load(open(f'graphs_data/{dataset_name}/inter_neuron_list.json', "r"))
    motor_neuron_list = json.load(open(f'graphs_data/{dataset_name}/motor_neuron_list.json', "r"))

    neuron_types_list = ['larynx', 'sensory', 'inter', 'motor', ]
    odor_list = ['butanone', 'pentanedione', 'NaCL']

    LinearSegmentedColormap.from_list('black_green', ['black', 'green'])

    x_list = []
    y_list = []
    for run in trange(0,n_runs, ncols=90):
        if os.path.exists(f'graphs_data/{dataset_name}/x_list_{run}.pt'):
            x = torch.load(f'graphs_data/{dataset_name}/x_list_{run}.pt', map_location=device)
            y = torch.load(f'graphs_data/{dataset_name}/y_list_{run}.pt', map_location=device)
            x = to_numpy(torch.stack(x))
            y = to_numpy(torch.stack(y))
        else:
            x = np.load(f'graphs_data/{dataset_name}/x_list_{run}.npy')
            y = np.load(f'graphs_data/{dataset_name}/y_list_{run}.npy')
            if os.path.exists(f'graphs_data/{dataset_name}/raw_x_list_{run}.npy'):
                np.load(f'graphs_data/{dataset_name}/raw_x_list_{run}.npy')
        x_list.append(x)
        y_list.append(y)
    vnorm = torch.load(os.path.join(log_dir, 'vnorm.pt'))
    ynorm = torch.load(os.path.join(log_dir, 'ynorm.pt'))
    if os.path.exists(os.path.join(log_dir, 'xnorm.pt')):
        xnorm = torch.load(os.path.join(log_dir, 'xnorm.pt'))
    else:
        xnorm = torch.tensor([5], device=device)
    print(f'xnorm: {to_numpy(xnorm)}, vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')

    edges = torch.load(f'./graphs_data/{dataset_name}/edge_index.pt', map_location=device)
    edges.clone().detach()

    print('update variables ...')
    x = x_list[0][n_frames - 5]
    n_neurons = x.shape[0]
    print(f'N neurons: {n_neurons}')
    logger.info(f'N neurons: {n_neurons}')
    config.simulation.n_neurons = n_neurons
    type_list = torch.tensor(x[:, 1 + 2 * dimension:2 + 2 * dimension], device=device)

    activity = torch.tensor(x_list[0][:, :, 6:7],device=device)
    activity = activity.squeeze()
    activity = activity.t()

    activity_list = []
    for n in range(n_runs):
        activity_ = torch.tensor(x_list[n][:, :, 6:7], device=device)
        activity_ = activity_.squeeze().t()
        activity_list.append(activity_)

    if 'black' in style:
        mc = 'w'
    else:
        mc = 'k'

    if field_type != '':
        has_field = True
        n_nodes = config.simulation.n_nodes
        n_nodes_per_axis = int(np.sqrt(n_nodes))
        if ('short_term_plasticity' in field_type) | ('modulation' in field_type):
            model_f = nn.ModuleList([
                Siren(in_features=model_config.input_size_nnr, out_features=model_config.output_size_nnr,
                      hidden_features=model_config.hidden_dim_nnr,
                      hidden_layers=model_config.n_layers_nnr, first_omega_0=model_config.omega,
                      hidden_omega_0=model_config.omega,
                      outermost_linear=model_config.outermost_linear_nnr)
                for n in range(n_runs)
            ])
        else:
            model_f = Siren_Network(image_width=n_nodes_per_axis, in_features=model_config.input_size_nnr, out_features=model_config.output_size_nnr, hidden_features=model_config.hidden_dim_nnr,
                                            hidden_layers=model_config.n_layers_nnr, outermost_linear=True, device=device, first_omega_0=omega, hidden_omega_0=omega)
        model_f.to(device=device)
        model_f.train()
        modulation = torch.tensor(x_list[0], device=device)
        modulation = modulation[:, :, 8:9].squeeze()
        modulation = modulation.t()
        modulation = modulation.clone().detach()
        (modulation[:, 1:] - modulation[:, :-1]) / delta_t
    else:
        has_field = False
        model_f = None

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
    else:
        model_missing_activity = None

    if epoch_list[0] == 'all':
        files = glob.glob(f"{log_dir}/models/*.pt")
        files.sort(key=os.path.getmtime)

        model, bc_pos, bc_dpos = choose_training_model(config, device)

        # plt.rcParams['text.usetex'] = False
        # plt.rc('font', family='sans-serif')
        # plt.rc('text', usetex=False)
        # matplotlib.rcParams['savefig.pad_inches'] = 0

        files, file_id_list = get_training_files(log_dir, n_runs)

        r_squared_list = []
        slope_list = []
        it = -1
        with torch.no_grad():
            for file_id_ in trange(0,len(file_id_list), ncols=90):
                it = it + 1
                num = str(it).zfill(4)
                file_id = file_id_list[file_id_]
                epoch = files[file_id].split('graphs')[1][1:-3]
                net = f"{log_dir}/models/best_model_with_{n_runs-1}_graphs_{epoch}.pt"

                print(net)

                state_dict = torch.load(net, map_location=device)
                model.load_state_dict(state_dict['model_state_dict'])
                # if train_config.with_connectivity_mask:
                #     inv_mask = torch.load(f'./graphs_data/{dataset_name}/inv_mask.pt', map_location=device)
                #     with torch.no_grad():
                #         model.W.copy_(model.W * inv_mask)
                model.eval()

                if has_field:
                    net = f'{log_dir}/models/best_model_f_with_{n_runs-1}_graphs_{epoch}.pt'
                    state_dict = torch.load(net, map_location=device)
                    model_f.load_state_dict(state_dict['model_state_dict'])

                amax = torch.max(model.a, dim=0).values
                amin = torch.min(model.a, dim=0).values
                model_a = (model.a - amin) / (amax - amin)

                fig, ax = fig_init()
                for n in range(n_neuron_types-1,-1,-1):
                    pos = torch.argwhere(type_list == n).squeeze()
                    plt.scatter(to_numpy(model_a[pos[1:], 0]), to_numpy(model_a[pos[1:], 1]), s=50, color=cmap.color(n), alpha=1.0, edgecolors='none')
                    plt.scatter(to_numpy(model_a[pos[0], 0]), to_numpy(model_a[pos[0], 1]), s=50, color=cmap.color(n), alpha=1.0, edgecolors='none',
                                label=neuron_types_list[n])
                plt.xlabel(r'$a_{0}$', fontsize=68)
                plt.ylabel(r'$a_{1}$', fontsize=68)
                plt.xlim([-0.1, 1.1])
                plt.ylim([-0.1, 1.1])
                plt.legend(loc='best', fontsize=18, markerscale=2)
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/embedding_{num}.png", dpi=80)
                plt.close()

                if os.path.exists(f'{log_dir}/correction.pt'):
                    correction = torch.load(f'{log_dir}/correction.pt',map_location=device)
                    second_correction = np.load(f'{log_dir}/second_correction.npy')
                else:
                    correction = torch.tensor(1.0, device=device)
                    second_correction = 1.0

                i, j = torch.triu_indices(n_neurons, n_neurons, requires_grad=False, device=device)
                A = model.W.clone().detach() / correction
                A[i, i] = 0
                A = A.t()

                fig, ax = fig_init()
                ax = sns.heatmap(to_numpy(A)/second_correction, center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046}, vmin=-4,vmax=4)
                cbar = ax.collections[0].colorbar
                cbar.ax.tick_params(labelsize=48)
                plt.xticks([0, n_neurons - 1], [1, n_neurons], fontsize=24)
                plt.yticks([0, n_neurons - 1], [1, n_neurons], fontsize=24)
                plt.subplot(2, 2, 1)

                larynx_pred_weight, index_larynx = map_matrix(larynx_neuron_list, all_neuron_list, A)
                ax = sns.heatmap(to_numpy(larynx_pred_weight) / second_correction, cbar=False, center=0, square=True, cmap='bwr', vmin=-4, vmax=4)

                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/W_{num}.png", dpi=80)
                plt.close()

                rr = torch.linspace(-xnorm.squeeze(), xnorm.squeeze(), 1000).to(device)
                if model_config.signal_model_name == 'PDE_N5':
                    fig, ax = fig_init()
                    plt.axis('off')
                    for k in range(n_neuron_types):
                        ax = fig.add_subplot(2, 2, k + 1)
                        for spine in ax.spines.values():
                            spine.set_edgecolor(cmap.color(k))  # Set the color of the outline
                            spine.set_linewidth(3)
                        if k==0:
                            plt.ylabel(r'learned $\mathrm{MLP_1}( a_i, a_j, v_j)$', fontsize=32)
                        for n in range(n_neuron_types):
                            for m in range(250):
                                pos0 = to_numpy(torch.argwhere(type_list == k).squeeze())
                                pos1 = to_numpy(torch.argwhere(type_list == n).squeeze())
                                n0 = np.random.randint(len(pos0))
                                n0 = pos0[n0, 0]
                                n1 = np.random.randint(len(pos1))
                                n1 = pos1[n1, 0]
                                embedding0 = model.a[n0, :] * torch.ones((1000, config.graph_model.embedding_dim),
                                                                         device=device)
                                embedding1 = model.a[n1, :] * torch.ones((1000, config.graph_model.embedding_dim),
                                                                         device=device)
                                in_features = torch.cat((rr[:, None], embedding0, embedding1), dim=1)
                                func = model.lin_edge(in_features.float()) * correction
                                plt.plot(to_numpy(rr), to_numpy(func), 2, color=cmap.color(n), linewidth=3, alpha=0.25)
                        plt.ylim([-1.6, 1.6])
                        plt.xlim([-5, 5])
                        plt.xticks([])
                        plt.yticks([])
                    plt.xlabel(r'$x_j$', fontsize=68)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/all/MLP1_{num}.png", dpi=80)
                    plt.close()
                elif (model_config.signal_model_name == 'PDE_N4'):
                    fig, ax = fig_init()
                    for k in range(n_neuron_types):
                        for m in range(250):
                            pos0 = to_numpy(torch.argwhere(type_list == k).squeeze())
                            n0 = np.random.randint(len(pos0))
                            n0 = pos0[n0, 0]
                            embedding0 = model.a[n0, :] * torch.ones((1000, config.graph_model.embedding_dim),
                                                                     device=device)
                            # in_features = torch.cat((rr[:, None], embedding0), dim=1)
                            in_features = get_in_features(rr, embedding0, model_config.signal_model_name, max_radius)
                            if config.graph_model.lin_edge_positive:
                                func = model.lin_edge(in_features.float()) ** 2 * correction
                            else:
                                func = model.lin_edge(in_features.float()) * correction
                            plt.plot(to_numpy(rr), to_numpy(func), color=cmap.color(k), linewidth=2, alpha=0.25)
                    plt.xlabel(r'$x_j$', fontsize=68)
                    plt.ylabel(r'learned $\mathrm{MLP_1}(\mathbf{a}_j, v_j)$', fontsize=68)
                    if config.graph_model.lin_edge_positive:
                        plt.ylim([-0.2, 1.2])
                    else:
                        plt.ylim([-1.6, 1.6])
                    plt.xlim([-to_numpy(xnorm)//2, to_numpy(xnorm)//2])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/all/MLP1_{num}.png", dpi=80)
                    plt.close()
                elif (model_config.signal_model_name == 'PDE_N8'):
                    rr = torch.linspace(0, 10, 1000).to(device)
                    fig, ax = fig_init()
                    for idx, k in enumerate(np.linspace(4, 10, 13)):  # Corrected step size to generate 13 evenly spaced values
                        for n in range(0,n_neurons,4):
                            embedding_i = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                            embedding_j = model.a[np.random.randint(n_neurons), :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                            if model.embedding_trial:
                                in_features = torch.cat((rr[:, None], torch.ones_like(rr[:, None])*k, embedding_i, embedding_j, model.b[0].repeat(1000, 1)), dim=1)
                            else:
                                in_features = torch.cat((rr[:, None], torch.ones_like(rr[:, None])*k, embedding_i, embedding_j), dim=1)
                            with torch.no_grad():
                                func = model.lin_edge(in_features.float())
                            if config.graph_model.lin_edge_positive:
                                func = func ** 2
                            plt.plot(to_numpy(rr-k), to_numpy(func), 2, color=cmap.color(idx), linewidth=2, alpha=0.25)
                    plt.xlabel(r'$x_i-x_j$', fontsize=68)
                    # plt.ylabel(r'learned $\psi^*(\mathbf{a}_i, v_i)$', fontsize=68)
                    plt.ylabel(r'$\mathrm{MLP_1}(\mathbf{a}_i, a_j, v_i, v_j)$', fontsize=68)
                    plt.ylim([0,4])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/all/MLP1_{num}.png", dpi=80)
                    plt.close()
                else:
                    fig, ax = fig_init()
                    in_features = rr[:, None]
                    with torch.no_grad():
                        func = model.lin_edge(in_features.float()) * correction
                    if config.graph_model.lin_edge_positive:
                        func = func ** 2
                    plt.plot(to_numpy(rr), to_numpy(func), color=mc, linewidth=8, label=r'learned')
                    plt.xlabel(r'$x_j$', fontsize=68)
                    # plt.ylabel(r'learned $\psi^*(\mathbf{a}_i, v_i)$', fontsize=68)
                    plt.ylabel(r'learned $\mathrm{MLP_1}(\mathbf{a}_j, v_j)$', fontsize=68)
                    plt.ylim([-1.5, 1.5])
                    plt.xlim([-5,5])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/all/MLP1_{epoch}.png", dpi=80)
                    plt.close()

                fig, ax = fig_init()
                for n in range(n_neurons):
                    embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                    # in_features = torch.cat((rr[:, None], embedding_), dim=1)
                    in_features = get_in_features_update(rr[:, None], model, embedding_, device)
                    with torch.no_grad():
                        func = model.lin_phi(in_features.float())
                    func = func[:, 0]
                    plt.plot(to_numpy(rr), to_numpy(func) * to_numpy(ynorm), color=cmap.color(to_numpy(type_list[n]).astype(int)), linewidth=2, alpha=0.25) #
                plt.ylim([-4,4])
                plt.xlabel(r'$v_i$', fontsize=68)
                # plt.ylabel(r'learned $\phi^*(\mathbf{a}_i, v_i)$', fontsize=68)
                plt.ylabel(r'learned $\mathrm{MLP_0}(\mathbf{a}_i, v_i)$', fontsize=68)

                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/MLP0_{num}.png", dpi=80)
                plt.close()

                adjacency = torch.load(f'./graphs_data/{dataset_name}/adjacency.pt', map_location=device)
                adjacency_ = adjacency.t().clone().detach()
                adj_t = torch.abs(adjacency_) > 0
                edge_index = adj_t.nonzero().t().contiguous()

                i, j = torch.triu_indices(n_neurons, n_neurons, requires_grad=False, device=device)
                A = model.W.clone().detach() / correction
                A[i, i] = 0

                fig, ax = fig_init()
                gt_weight = to_numpy(adjacency)
                pred_weight = to_numpy(A) / second_correction
                plt.scatter(gt_weight, pred_weight, s=0.1, c=mc, alpha=0.1)
                plt.xlabel(r'true $W_{ij}$', fontsize=68)
                plt.ylabel(r'learned $W_{ij}$', fontsize=68)
                if n_neurons == 8000:
                    plt.xlim([-0.05, 0.05])
                    plt.ylim([-0.05, 0.05])
                else:
                    # plt.xlim([-0.2, 0.2])
                    # plt.ylim([-0.2, 0.2])
                    plt.xlim([-0.15, 0.15])
                    plt.ylim([-0.15, 0.15])

                x_data = np.reshape(gt_weight, (n_neurons * n_neurons))
                y_data = np.reshape(pred_weight, (n_neurons * n_neurons))
                lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                residuals = y_data - linear_model(x_data, *lin_fit)
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                r_squared_list.append(r_squared)
                slope_list.append(lin_fit[0])

                if n_neurons == 8000:
                    plt.text(-0.042, 0.042, f'$R^2$: {np.round(r_squared, 3)}', fontsize=34)
                    plt.text(-0.042, 0.036, f'slope: {np.round(lin_fit[0], 2)}', fontsize=34)
                else:
                    # plt.text(-0.17, 0.15, f'$R^2$: {np.round(r_squared, 3)}', fontsize=34)
                    # plt.text(-0.17, 0.12, f'slope: {np.round(lin_fit[0], 2)}', fontsize=34)
                    plt.text(-0.13, 0.13, f'$R^2$: {np.round(r_squared, 3)}', fontsize=34)
                    plt.text(-0.13, 0.11, f'slope: {np.round(lin_fit[0], 2)}', fontsize=34)

                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/all/comparison_{num}.png", dpi=80)
                plt.close()

                if has_field:

                    if 'short_term_plasticity' in field_type:
                        fig, ax = fig_init()
                        t = torch.linspace(1, 100000, 1, dtype=torch.float32, device=device).unsqueeze(1)
                        prediction = model_f(t) ** 2
                        prediction = prediction.t()
                        plt.imshow(to_numpy(prediction), aspect='auto', cmap='gray')
                        plt.title(r'learned $FMLP(t)_i$', fontsize=68)
                        plt.xlabel(r'$t$', fontsize=68)
                        plt.ylabel(r'$i$', fontsize=68)
                        plt.xticks([10000,100000], [10000, 100000], fontsize=48)
                        plt.yticks([0, 512, 1024], [0, 512, 1024], fontsize=48)
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/results/all/yi_{num}.png", dpi=80)
                        plt.close()

                        prediction = prediction * torch.tensor(second_correction,device=device) / 10

                        fig, ax = fig_init()
                        ids = np.arange(0,100000,100).astype(int)
                        plt.scatter(to_numpy(modulation[:,ids]), to_numpy(prediction[:,ids]), s=1, color=mc, alpha=0.05)
                        # plt.xlim([0,0.5])
                        # plt.ylim([0,2])
                        # plt.xticks([0,0.5], [0,0.5], fontsize=48)
                        # plt.yticks([0,1,2], [0,1,2], fontsize=48)
                        x_data = to_numpy(modulation[:,ids]).flatten()
                        y_data = to_numpy(prediction[:,ids]).flatten()
                        lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                        residuals = y_data - linear_model(x_data, *lin_fit)
                        ss_res = np.sum(residuals ** 2)
                        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot)
                        ax.text(0.05, 0.94, f'$R^2$: {r_squared:0.2f}', transform=ax.transAxes,
                                verticalalignment='top', horizontalalignment='left', fontsize=32)
                        ax.text(0.05, 0.88, f'slope: {lin_fit[0]:0.2f}', transform=ax.transAxes,
                                verticalalignment='top', horizontalalignment='left', fontsize=32)
                        plt.xlabel(r'true $y_i(t)$', fontsize=68)
                        plt.ylabel(r'learned $y_i(t)$', fontsize=68)
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/results/all/comparison_yi_{num}.png", dpi=80)
                        plt.close()

                    else:

                        fig, ax = fig_init()
                        pred = model_f(time=file_id_ / len(file_id_list), enlarge=True) ** 2
                        # pred = torch.reshape(pred, (n_nodes_per_axis, n_nodes_per_axis))
                        pred = torch.reshape(pred, (640, 640))
                        pred = to_numpy(torch.sqrt(pred))
                        pred = np.flipud(pred)
                        pred = np.rot90(pred, 1)
                        pred = np.fliplr(pred)
                        plt.imshow(pred, cmap='grey')
                        plt.ylabel(r'learned $FMLP(s_i, t)$', fontsize=68)
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/results/all/field_{num}.png", dpi=80)
                        plt.close()

                if 'derivative' in field_type:

                    y = torch.linspace(0, 1, 400)
                    x = torch.linspace(-6, 6, 400)
                    grid_y, grid_x = torch.meshgrid(y, x)
                    grid = torch.stack((grid_x, grid_y), dim=-1)
                    grid = grid.to(device)
                    pred_modulation = model.lin_modulation(grid) / 40
                    tau = 100
                    alpha = 0.02
                    true_derivative = (1 - grid_y) / tau - alpha * grid_y * torch.abs(grid_x)

                    fig, ax = fig_init()
                    plt.title(r'learned $MLP_2(x_i, y_i)$', fontsize=68)
                    plt.imshow(to_numpy(pred_modulation))
                    plt.xticks([])
                    plt.yticks([])
                    plt.xlabel(r'$v_i$', fontsize=68)
                    plt.ylabel(r'$y_i$', fontsize=68)
                    plt.tight_layout
                    plt.savefig(f"./{log_dir}/results/all/derivative_yi_{num}.png", dpi=80)
                    plt.close()

                    fig, ax = fig_init()
                    plt.scatter(to_numpy(true_derivative.flatten()), to_numpy(pred_modulation.flatten()), s=5, color=mc, alpha=0.1)
                    x_data = to_numpy(true_derivative.flatten())
                    y_data = to_numpy(pred_modulation.flatten())
                    lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                    residuals = y_data - linear_model(x_data, *lin_fit)
                    ss_res = np.sum(residuals ** 2)
                    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot)
                    ax.text(0.05, 0.94, f'$R^2$: {r_squared:0.2f}', transform=ax.transAxes,
                            verticalalignment='top', horizontalalignment='left', fontsize=32)
                    ax.text(0.05, 0.88, f'slope: {lin_fit[0]:0.2f}', transform=ax.transAxes,
                            verticalalignment='top', horizontalalignment='left', fontsize=32)
                    plt.xlabel(r'true $\dot{y_i}(t)$', fontsize=68)
                    plt.ylabel(r'learned $\dot{y_i}(t)$', fontsize=68)

                    # plt.xticks([-0.1, 0], [-0.1, 0], fontsize=48)
                    # plt.yticks([-0.1, 0], [-0.1, 0], fontsize=48)
                    # plt.xlim([-0.2,0.025])
                    # plt.ylim([-0.2,0.025])

                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/all/comparison_derivative_yi_{num}.png", dpi=80)
                    plt.close()

                if (model.update_type == 'generic') & (model_config.signal_model_name == 'PDE_N5'):

                    k = np.random.randint(n_frames - 50)
                    x = torch.tensor(x_list[0][k], device=device)

                    fig, ax = fig_init()
                    msg_list = []
                    u = torch.linspace(-xnorm.squeeze(), xnorm.squeeze(), 400).to(device)
                    for sample in range(n_neurons):
                        id0 = np.random.randint(0, n_neurons)
                        id1 = np.random.randint(0, n_neurons)
                        f = x[id0, 8:9]
                        embedding0 = model.a[id0, :] * torch.ones((400, config.graph_model.embedding_dim),
                                                                  device=device)
                        embedding1 = model.a[id1, :] * torch.ones((400, config.graph_model.embedding_dim),
                                                                  device=device)
                        in_features = torch.cat((u[:, None], embedding0, embedding1), dim=1)
                        msg = model.lin_edge(in_features.float()) ** 2 * correction
                        in_features = torch.cat((torch.zeros((400, 1), device=device), embedding0, msg,
                                                 f * torch.ones((400, 1), device=device)), dim=1)
                        plt.plot(to_numpy(u), to_numpy(msg), c=cmap.color(to_numpy(x[id0, 5]).astype(int)), linewidth=2, alpha=0.25)
                        # plt.scatter(to_numpy(u), to_numpy(model.lin_phi(in_features)), s=5, c='r', alpha=0.15)
                        # plt.scatter(to_numpy(u), to_numpy(f*msg), s=1, c='w', alpha=0.1)
                        msg_list.append(msg)
                    plt.tight_layout()
                    msg_list = torch.stack(msg_list).squeeze()
                    y_min, y_max = msg_list.min().item(), msg_list.max().item()
                    plt.xlabel(r'$v_i$', fontsize=68)
                    plt.ylabel(r'learned MLPs', fontsize=68)
                    plt.ylim([y_min - y_max/2, y_max * 1.5])
                    plt.tight_layout()
                    plt.savefig(f'./{log_dir}/results/all/MLP1_{num}.png', dpi=80)
                    plt.close()


                im0 = imread(f"./{log_dir}/results/all/W_{num}.png")
                # im0 = imread(f"./{log_dir}/results/all/comparison_{num}.png")
                im1 = imread(f"./{log_dir}/results/all/embedding_{num}.png")
                im2 = imread(f"./{log_dir}/results/all/MLP0_{num}.png")
                im3 = imread(f"./{log_dir}/results/all/MLP1_{num}.png")
                fig = plt.figure(figsize=(16, 16))
                plt.axis('off')
                plt.subplot(2, 2, 1)
                plt.axis('off')
                plt.imshow(im0)
                plt.xticks([])
                plt.yticks([])
                plt.subplot(2, 2, 2)
                plt.axis('off')
                plt.imshow(im1)
                plt.xticks([])
                plt.yticks([])
                plt.axis('off')
                plt.subplot(2, 2, 3)
                plt.axis('off')
                plt.imshow(im2)
                plt.xticks([])
                plt.yticks([])
                plt.subplot(2, 2, 4)
                plt.axis('off')
                plt.imshow(im3)
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()

                plt.savefig(f"./{log_dir}/results/training/fig_{num}.png", dpi=80)
                plt.close()

        fig, ax = fig_init(formatx='%.0f', formaty='%.2f')
        plt.plot(r_squared_list, linewidth=4, c=mc)
        plt.xlim([0, 100])
        plt.ylim([0, 1.1])
        plt.yticks(fontsize=48)
        plt.xticks([0, 100], [0, 20], fontsize=48)
        plt.ylabel('$R^2$', fontsize=64)
        plt.xlabel('epoch', fontsize=64)
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/results/R2.png', dpi=300)
        plt.close()
        np.save(f'./{log_dir}/results/R2.npy', r_squared_list)

        slope_list = np.array(slope_list) / p[0][0]
        fig, ax = fig_init(formatx='%.0f', formaty='%.2f')
        plt.plot(slope_list*10, linewidth=4, c=mc)
        plt.xlim([0, 100])
        plt.ylim([0, 1.1])
        plt.yticks(fontsize=48)
        plt.xticks([0, 100], [0, 20], fontsize=48)
        plt.ylabel('slope', fontsize=64)
        plt.xlabel('epoch', fontsize=64)
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/results/slope.png', dpi=300)
        plt.close()

    else:

        files = glob.glob(f'./{log_dir}/results/*.png')
        for f in files:
            os.remove(f)

        adjacency = torch.load(f'./graphs_data/{dataset_name}/adjacency.pt', map_location=device)
        adjacency_ = adjacency.t().clone().detach()
        adj_t = torch.abs(adjacency_) > 0
        edge_index = adj_t.nonzero().t().contiguous()
        weights = to_numpy(adjacency.flatten())
        pos = np.argwhere(weights != 0)
        weights = weights[pos]

        fig_init()
        plt.hist(weights, bins=1000, color=mc, alpha=0.5)
        plt.ylabel(r'counts', fontsize=64)
        plt.xlabel(r'$W$', fontsize=64)
        plt.yticks(fontsize=24)
        plt.xticks(fontsize=24)
        plt.xlim([-0.1, 0.1])
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/results/weights_distribution.png', dpi=300)
        plt.close()

        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(to_numpy(adjacency), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046}, vmin=-0.1, vmax=0.1)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=32)
        plt.xticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
        plt.yticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/results/true connectivity.png', dpi=300)
        plt.close()

        map_larynx_matrix, n = map_matrix(larynx_neuron_list, all_neuron_list, adjacency)
        fig, axes = plt.subplots(5, 4, figsize=(16, 20))
        axes = axes.flatten()
        for i, activity in enumerate(activity_list[:20]):
            for j in range(len(n)):
                axes[i].plot(to_numpy(activity[n[j].astype(int), :]), linewidth=1)
            axes[i].set_title(f'dataset {i}', fontsize=12)
            axes[i].set_xlim([0, n_frames])
            axes[i].set_ylim([0, 10])
        fig.suptitle('larynx neuron activity', fontsize=16)
        fig.text(0.5, 0.04, 'time', ha='center', fontsize=14)
        fig.text(0.04, 0.5, 'activity', va='center', rotation='vertical', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/results/activity_larynx_grid.png', dpi=300)
        plt.close()

        n = np.random.randint(0, n_neurons, 50)
        fig, axes = plt.subplots(5, 4, figsize=(16, 20))
        axes = axes.flatten()
        for i, activity in enumerate(activity_list[:20]):
            for j in range(len(n)):
                axes[i].plot(to_numpy(activity[n[j].astype(int), :]), linewidth=1)
            axes[i].set_title(f'dataset {i}', fontsize=12)
            axes[i].set_xlim([0, n_frames])
            axes[i].set_ylim([0, 10])
        fig.suptitle('sample neuron activity', fontsize=16)
        fig.text(0.5, 0.04, 'time', ha='center', fontsize=14)
        fig.text(0.04, 0.5, 'activity', va='center', rotation='vertical', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/results/activity_grid.png', dpi=300)
        plt.close()

        true_model, bc_pos, bc_dpos = choose_model(config=config, W=adjacency, device=device)

        for epoch in epoch_list:

            net = f'{log_dir}/models/best_model_with_{n_runs-1}_graphs_{epoch}.pt'
            model, bc_pos, bc_dpos = choose_training_model(config, device)
            state_dict = torch.load(net, map_location=device)
            model.load_state_dict(state_dict['model_state_dict'])
            model.edges = edge_index
            print(f'net: {net}')

            fig, ax = fig_init()
            for n in range(n_neurons):
                if x_list[0][100][n, 6] != config.simulation.baseline_value:
                    plt.scatter(to_numpy(model.a[n, 0]), to_numpy(model.a[n, 1]), s=100, color=cmap.color(int(type_list[n])), alpha=1.0, edgecolors='none')
            if 'latex' in style:
                plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=68)
                plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=68)
            else:
                plt.xlabel(r'$a_{0}$', fontsize=68)
                plt.ylabel(r'$a_{1}$', fontsize=68)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/embedding.png", dpi=170.7)
            plt.close()
            fig, axes = fig_init()
            for n in range(n_neurons):
                if x_list[0][100][n, 6] != config.simulation.baseline_value:
                    plt.scatter(to_numpy(model.a[:n_neurons, 0]), to_numpy(model.a[:n_neurons, 1]), alpha=0.1, s=50, color='k', edgecolors='none')
                    plt.text(to_numpy(model.a[n, 0]), to_numpy(model.a[n, 1]) - 0.01, all_neuron_list[n], fontsize=6)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/embedding_text.png", dpi=170.7)
            plt.close()

            # plt.close(fig_2d)
            # plt.close(fig_scatter)

            if False:
                if os.path.exists(f"./{log_dir}/results/top_pairs_by_run.pkl"):
                    with open(f"./{log_dir}/results/top_pairs_by_run.pkl", 'rb') as f:
                        top_pairs_by_run = pickle.load(f)
                else:
                    top_pairs_by_run = {}
                    for run in range(n_runs):
                        print(f"top 20 pairs in CElegans #{run}:")
                        top_pairs = find_top_responding_pairs(
                            model, all_neuron_list, to_numpy(adjacency), to_numpy(model.W[run]),
                            signal_range=(0, 10), resolution=100, device=device, top_k=20
                        )
                        top_pairs_by_run[run] = top_pairs
                    with open(f"./{log_dir}/results/top_pairs_by_run.pkl", 'wb') as f:
                        pickle.dump(top_pairs_by_run, f)
                os.makedirs(f"./{log_dir}/results/odor_heatmaps/", exist_ok=True)
                if os.path.exists(f"./{log_dir}/results/odor_responses_by_run.pkl"):
                    with open(f"./{log_dir}/results/odor_responses_by_run.pkl", 'rb') as f:
                        odor_responses_by_run = pickle.load(f)
                else:
                    odor_responses_by_run = {}  # Initialize the dictionary BEFORE the loop

                    for run in range(n_runs):
                        print(f"20 responding neurons in CElegans #{run}:")
                        neuron_responses = analyze_odor_responses_by_neuron(
                            model=model, x_list=x_list, edges=edges, n_runs=n_runs, n_frames=n_frames,
                            time_step=time_step,
                            all_neuron_list=all_neuron_list, has_missing_activity=has_missing_activity,
                            model_missing_activity=model_missing_activity,
                            has_neural_field=has_field, model_f=model_f,
                            n_samples=100, device=device, run=run
                        )

                        # Process the raw tensor data to extract top responding neurons
                        processed_responses = {}
                        odor_list = ['butanone', 'pentanedione', 'NaCL']

                        for i, odor in enumerate(odor_list):
                            if odor in neuron_responses:
                                # Calculate mean response across samples for each neuron
                                mean_response = torch.mean(neuron_responses[odor], dim=0)  # [n_neurons]

                                # Get top 20 responding neurons
                                top_20_indices = torch.topk(mean_response, k=20).indices.cpu().numpy()
                                top_20_names = [all_neuron_list[idx] for idx in top_20_indices]
                                top_20_values = [mean_response[idx].item() for idx in top_20_indices]

                                processed_responses[odor] = {
                                    'names': top_20_names,
                                    'indices': top_20_indices.tolist(),
                                    'values': top_20_values
                                }

                                print(f"\ntop 20 responding neurons for {odor}:")
                                for j, (name, idx, val) in enumerate(
                                        zip(top_20_names, top_20_indices, top_20_values)):
                                    print(f"  {j + 1}. {name} : {val:.4f}")

                        # Store responses for this run (moved outside the odor loop)
                        odor_responses_by_run[run] = processed_responses

                        fig = plot_odor_heatmaps(neuron_responses)
                        plt.savefig(f"./{log_dir}/results/odor_heatmaps/odor_heatmaps_{run}.png", dpi=150,
                                    bbox_inches='tight')
                        plt.close()

                    # Save the complete dictionary AFTER the loop
                    with open(f"./{log_dir}/results/odor_responses_by_run.pkl", 'wb') as f:
                        pickle.dump(odor_responses_by_run, f)
                results = run_neural_architecture_pipeline(top_pairs_by_run, odor_responses_by_run, all_neuron_list)
                results['summary_figure'].savefig(f"./{log_dir}/results/neural_architecture_summary.png",dpi=150, bbox_inches='tight')
                plt.close()
                results['architecture_analysis']['architecture_data']
                results['hub_analysis']
                results['pathway_analysis']
                results['summary_figure']
                # Finds neurons that appear in BOTH high connectivity AND high odor responses
                # These are the "bridge" neurons between detection and integration
                # TODO: run_preprocessing_analysis is not implemented yet
                # preprocessing_results = run_preprocessing_analysis(
                #     top_pairs_by_run,
                #     odor_responses_by_run,
                #     results['architecture_analysis'],  # From your previous analysis
                #     all_neuron_list
                # )
                # preprocessing_results['preprocessing_figure'].savefig(
                #     f"./{log_dir}/results/preprocessing_analysis.png",
                #     dpi=150, bbox_inches='tight'
                # )
                # plt.close()
                # with open(f"./{log_dir}/results/preprocessing_analysis.pkl", 'wb') as f:
                #     pickle.dump(preprocessing_results, f)

            # Line plots for specific neurons
            selected_neurons = ['ADAL', 'ADAR', 'AVAL', 'AVAR']  # 4 neurons of interest
            fig_lines = analyze_mlp_edge_lines(
                model,
                selected_neurons,
                all_neuron_list,
                to_numpy(adjacency),  # Your 300x300 adjacency matrix
                signal_range=(0, 10),
                resolution=100,
                device=device
            ) # Example neuron names
            fig_lines.savefig(f"./{log_dir}/results/function_edge_lines_ADA_AVA.png", dpi=300, bbox_inches='tight')
            plt.close(fig_lines)


            for neuron_OI in selected_neurons:
                fig_lines, _ = analyze_mlp_edge_lines_weighted_with_max(
                    model,
                    neuron_OI,  # Single neuron of interest
                    all_neuron_list,
                    to_numpy(adjacency),
                    to_numpy(model.W[0]),  # Your 300x300 weight matrix
                    signal_range=(0, 10),
                    resolution=100,
                    device=device
                )
                fig_lines.savefig(f"./{log_dir}/results/function_edge_lines_{neuron_OI}.png", dpi=300, bbox_inches='tight')
                plt.close(fig_lines)


            fig_lines = analyze_mlp_edge_lines(model, larynx_neuron_list, all_neuron_list, to_numpy(adjacency), signal_range=(0, 10), resolution=100, device=device)
            fig_lines.savefig(f"./{log_dir}/results/function_edge_lines_larynx.png", dpi=300, bbox_inches='tight')
            plt.close(fig_lines)

            fig_lines = analyze_mlp_edge_lines(model, sensory_neuron_list, all_neuron_list, to_numpy(adjacency), signal_range=(0, 10), resolution=100, device=device)
            fig_lines.savefig(f"./{log_dir}/results/function_edge_lines_sensory.png", dpi=300, bbox_inches='tight')
            plt.close(fig_lines)

            fig_lines = analyze_mlp_edge_lines(model, motor_neuron_list, all_neuron_list, to_numpy(adjacency), signal_range=(0, 10), resolution=100, device=device)
            fig_lines.savefig(f"./{log_dir}/results/function_edge_motor_neuron.png", dpi=300, bbox_inches='tight')
            plt.close(fig_lines)

            fig_lines = analyze_mlp_edge_lines(model, inter_neuron_list, all_neuron_list, to_numpy(adjacency), signal_range=(0, 10), resolution=100, device=device)
            fig_lines.savefig(f"./{log_dir}/results/function_edge_inter_neuron.png", dpi=300, bbox_inches='tight')
            plt.close(fig_lines)


            fig = analyze_mlp_phi_synaptic(model, n_sample_pairs=1000, resolution=100, device=device)
            fig.savefig(f"./{log_dir}/results/function_update.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

            if has_missing_activity:
                net = f'{log_dir}/models/best_model_missing_activity_with_{n_runs - 1}_graphs_{epoch}.pt'
                state_dict = torch.load(net, map_location=device)
                model_missing_activity.load_state_dict(state_dict['model_state_dict'])

                fig, axes = plt.subplots(5, 4, figsize=(20, 25))
                axes = axes.flatten()
                for run in range(20):
                    n_frames_temp = n_frames - 10
                    if n_frames_temp > 1000:
                        t = torch.linspace(0, 1, n_frames_temp // 100, dtype=torch.float32, device=device).unsqueeze(1)
                    else:
                        t = torch.linspace(0, 1, n_frames_temp, dtype=torch.float32, device=device).unsqueeze(1)
                    prediction = model_missing_activity[run](t).t() + config.simulation.baseline_value
                    activity = torch.tensor(x_list[run][:, :, 6:7], device=device).squeeze().t()
                    im = axes[run].imshow(to_numpy(prediction), aspect='auto', cmap='viridis', vmin=0, vmax=10)
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/neural_fields_grid.png", dpi=150)
                plt.close()

                fig, axes = plt.subplots(5, 4, figsize=(20, 25))
                axes = axes.flatten()
                for run in range(20):
                    n_frames_temp = n_frames - 10
                    t = torch.linspace(0, 1, n_frames_temp // 100 if n_frames_temp > 1000 else n_frames_temp,
                                       dtype=torch.float32, device=device).unsqueeze(1)
                    prediction = model_missing_activity[run](t).t() + config.simulation.baseline_value
                    activity = torch.tensor(x_list[run][:, :, 6:7], device=device).squeeze().t()
                    pos = np.argwhere(x_list[run][0][:, 6] == 6)
                    axes[run].scatter(to_numpy(activity[pos, :prediction.shape[1]]),
                                      to_numpy(prediction[pos, :]), s=0.5, alpha=0.3, c=mc)
                    axes[run].set_xlim([0,10])
                    axes[run].set_ylim([0,10])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/neural_fields_comparison.png", dpi=150)
                plt.close()

            if multi_connectivity:
                os.makedirs(f"./{log_dir}/results/W", exist_ok=True)
                for k in range(min(20, model.W.shape[0] - 1)):
                    fig, ax = fig_init()
                    plt.axis('off')
                    i, j = torch.triu_indices(n_neurons, n_neurons, requires_grad=False, device=device)
                    A = model.W[k].clone().detach()
                    A[i, i] = 0
                    pos = np.argwhere(x_list[k][100][:, 6] == config.simulation.baseline_value)
                    A[pos,:] = 0
                    A = torch.reshape(A, (n_neurons, n_neurons))
                    plt.imshow(to_numpy(A), aspect='auto', cmap='bwr', vmin=-0.5, vmax=0.5)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/W/W_{k}.png", dpi=80)
                    plt.close()

                larynx_weights =[]
                fig, axes = plt.subplots(4, 5, figsize=(20, 16))
                axes = axes.flatten()
                for k in range(min(20, model.W.shape[0] - 1)):
                    i, j = torch.triu_indices(n_neurons, n_neurons, requires_grad=False, device=device)
                    A = model.W[k].clone().detach()
                    A[i, i] = 0
                    pos = np.argwhere(x_list[k][100][:, 6] == config.simulation.baseline_value)
                    A[pos,:] = 0
                    larynx_pred_weight, index_larynx = map_matrix(larynx_neuron_list, all_neuron_list, A)
                    sns.heatmap(to_numpy(larynx_pred_weight), ax=axes[k], center=0, square=True,
                                vmin=-0.5, vmax=0.5, cmap='bwr', cbar=False, xticklabels=False, yticklabels=False)
                    larynx_weights.append(to_numpy(larynx_pred_weight))
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/W_larynx_grid.png", dpi=80)
                plt.close()

                larynx_stack = np.stack(larynx_weights)
                larynx_mean = np.mean(larynx_stack, axis=0)
                larynx_std = np.std(larynx_stack, axis=0)
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                sns.heatmap(larynx_mean, ax=axes[0], center=0, square=True, cmap='bwr',
                            cbar=True, vmin=-0.5, vmax=0.5, xticklabels=False, yticklabels=False)
                sns.heatmap(larynx_std, ax=axes[1], square=True, cmap='viridis',
                            cbar=True, xticklabels=False, yticklabels=False)
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/W_larynx_mean_std.png", dpi=80)
                plt.close()

            else:

                i, j = torch.triu_indices(n_neurons, n_neurons, requires_grad=False, device=device)
                A = model.W.clone().detach()
                A[i, i] = 0
                A = A.t()
                fig, ax = fig_init()
                ax = sns.heatmap(to_numpy(A) , center=0, square=True, cmap='bwr',
                                 cbar_kws={'fraction': 0.046}, vmin=-4, vmax=4)
                cbar = ax.collections[0].colorbar
                cbar.ax.tick_params(labelsize=48)
                plt.xticks([0, n_neurons - 1], [1, n_neurons], fontsize=24)
                plt.yticks([0, n_neurons - 1], [1, n_neurons], fontsize=24)
                plt.subplot(2, 2, 1)
                larynx_pred_weight, index_larynx = map_matrix(larynx_neuron_list, all_neuron_list, A)
                ax = sns.heatmap(to_numpy(larynx_pred_weight) , cbar=False, center=0, square=True,
                                 cmap='bwr', vmin=-4, vmax=4)
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/W_{epoch}.png", dpi=80)
                plt.close()

            if has_field:
                net = f'{log_dir}/models/best_model_f_with_{n_runs - 1}_graphs_{epoch}.pt'
                state_dict = torch.load(net, map_location=device)
                model_f.load_state_dict(state_dict['model_state_dict'])

            os.makedirs(f"./{log_dir}/results/pairs", exist_ok=True)
            print("right-Left neuron pairs (with indexes):")

            rl_pairs = find_suffix_pairs_with_index(all_neuron_list, 'R', 'L')
            d_i1_i2 = []
            for (i1, n1), (i2, n2) in rl_pairs:
                if (x_list[0][100][i1, 6] != config.simulation.baseline_value) & (x_list[0][100][i2, 6] != config.simulation.baseline_value) :
                    print(f"{n1} (index {i1}) - {n2} (index {i2})")
                dist = torch.sum((model.a[i1, :] - model.a[i2, :]) ** 2)
                d_i1_i2.append(dist)

            d_i1_i2 = torch.stack(d_i1_i2).squeeze()
            sorted_indices = torch.argsort(d_i1_i2)
            rl_pairs_sorted = [rl_pairs[i] for i in sorted_indices]
            d_i1_i2_sorted = d_i1_i2[sorted_indices]

            pair = 0
            for ((i1, n1), (i2, n2)), dist in zip(rl_pairs_sorted, d_i1_i2_sorted):
                if (x_list[0][100][i1, 6] != 6) & (x_list[0][100][i2, 6] != 6) :
                    fig, ax = fig_init()
                    plt.scatter(to_numpy(model.a[:n_neurons, 0]), to_numpy(model.a[:n_neurons, 1]), s=50, color='k',
                                edgecolors='none', alpha=0.25)
                    print(f"{n1} (index {i1}) - {n2} (index {i2}): distance = {dist.item():.4f}")
                    plt.scatter(to_numpy(model.a[i1, 0]), to_numpy(model.a[i1, 1]), s=100, color='g', edgecolors='none')
                    plt.scatter(to_numpy(model.a[i2, 0]), to_numpy(model.a[i2, 1]), s=100, color='r', edgecolors='none')
                    plt.text(to_numpy(model.a[i1, 0]) - 0.05, to_numpy(model.a[i1, 1]) - 0.07,
                             all_neuron_list[i1], fontsize=14, c='g')
                    plt.text(to_numpy(model.a[i2, 0]) - 0.05, to_numpy(model.a[i2, 1]) - 0.07,
                             all_neuron_list[i2], fontsize=14, c='r')
                    ax = fig.add_subplot(3, 3, 1)
                    plt.plot(to_numpy(activity[i1, :]), linewidth=0.5, c='g')
                    plt.plot(to_numpy(activity[i2, :]), linewidth=0.5, c='r')
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/pairs/all_embedding_text_{pair}.png", dpi=80)
                    pair = pair + 1
                    plt.close()

            fig, ax = fig_init(formatx='%.2f', formaty='%d')
            plt.hist(to_numpy(d_i1_i2_sorted), bins=100, color=mc)
            plt.xlabel(r'$d_{aR,aL}$', fontsize=68)
            plt.ylabel('counts', fontsize=68)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/embedding_d_pair.png", dpi=80)
            plt.close()


            # fig, ax = fig_init()
            # rr = torch.linspace(-xnorm.squeeze() * 4, xnorm.squeeze() * 4, 1000).to(device)
            # func_list = []
            # for n in trange(0,n_neurons,n_neurons//100):
            #     if (model_config.signal_model_name == 'PDE_N4') | (model_config.signal_model_name == 'PDE_N5'):
            #         embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
            #         in_features = get_in_features(rr, embedding_, model_config.signal_model_name, max_radius)
            #     else:
            #         in_features = rr[:,None]
            #     with torch.no_grad():
            #         func = model.lin_edge(in_features.float())
            #     if config.graph_model.lin_edge_positive:
            #         func = func ** 2
            #     func_list.append(func)
            #     plt.plot(to_numpy(rr), to_numpy(func), 2, color=cmap.color(to_numpy(type_list)[n].astype(int)),
            #              linewidth=2 // ( 1 + (n_neuron_types>16)*1.0), alpha=0.25)
            # func_list = torch.stack(func_list).squeeze()
            # y_min, y_max = func_list.min().item(), func_list.max().item()
            # plt.xlabel(r'$v_i$', fontsize=68)
            # plt.ylabel(r'Learned $\psi^*(\mathbf{a}_i, v_i)$', fontsize=68)
            # # if (model_config.signal_model_name == 'PDE_N4') | (model_config.signal_model_name == 'PDE_N5'):
            # #     plt.ylim([-0.5,0.5])
            # # plt.xlim([-to_numpy(xnorm)*2, to_numpy(xnorm)*2])
            # plt.ylim([y_min,y_max*1.1])
            # plt.tight_layout()
            # plt.savefig(f"./{log_dir}/results/raw_psi.png", dpi=170.7)
            # plt.close()
            #
            # upper = func_list[:,950:1000].flatten()
            # upper = torch.sort(upper, descending=True).values
            # correction = 1 / torch.mean(upper[:upper.shape[0]//10])
            # # correction = 1 / torch.mean(torch.mean(func_list[:,900:1000], dim=0))
            # print(f'upper: {to_numpy(1/correction):0.4f}  correction: {to_numpy(correction):0.2f}')
            # torch.save(correction, f'{log_dir}/correction.pt')
            #
            # matrix_correction = torch.mean(func_list[:,950:1000], dim=1)
            # A_corrected = A
            # for i in range(n_neurons):
            #     A_corrected[i, :] = A[i, :] * matrix_correction[i]
            # plt.figure(figsize=(10, 10))
            # ax = sns.heatmap(to_numpy(A_corrected), center=0, square=True, cmap='bwr',
            #                  cbar_kws={'fraction': 0.046}, vmin=-0.1, vmax=0.1)
            # cbar = ax.collections[0].colorbar
            # cbar.ax.tick_params(labelsize=32)
            # plt.xticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
            # plt.yticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
            # plt.xticks(rotation=0)
            # # plt.subplot(2, 2, 1)
            # # ax = sns.heatmap(to_numpy(A[0:20, 0:20]), cbar=False, center=0, square=True, cmap='bwr', vmin=-0.1, vmax=0.1)
            # # plt.xticks(rotation=0)
            # # plt.xticks([])
            # # plt.yticks([])
            # plt.tight_layout()
            # plt.savefig(f'./{log_dir}/results/corrected learned connectivity.png', dpi=300)
            # plt.close()
            #
            # print('update functions ...')
            # if model_config.signal_model_name == 'PDE_N5':
            #     psi_list = []
            #     if model.update_type == 'generic':
            #         r_list = ['','generic']
            #     elif model.update_type == '2steps':
            #         r_list = ['','2steps']
            #
            #     r_list = ['']
            #     for r in r_list:
            #         fig, ax = fig_init()
            #         rr = torch.linspace(-xnorm.squeeze()*2, xnorm.squeeze()*2, 1500).to(device)
            #         ax.set_frame_on(False)
            #         ax.get_xaxis().set_visible(False)
            #         ax.get_yaxis().set_visible(False)
            #         for k in range(n_neuron_types):
            #             ax = fig.add_subplot(2, 2, k + 1)
            #             for spine in ax.spines.values():
            #                 spine.set_edgecolor(cmap.color(k))  # Set the color of the outline
            #                 spine.set_linewidth(3)
            #             for m in range(n_neuron_types):
            #                 true_func = true_model.func(rr, k, m, 'phi')
            #                 plt.plot(to_numpy(rr), to_numpy(true_func), c=mc, linewidth=1, label='original', alpha=0.21)
            #             for n in range(n_neuron_types):
            #                 for m in range(250):
            #                     pos0 = to_numpy(torch.argwhere(type_list == k).squeeze())
            #                     pos1 = to_numpy(torch.argwhere(type_list == n).squeeze())
            #                     n0 = np.random.randint(len(pos0))
            #                     n0 = pos0[n0,0]
            #                     n1 = np.random.randint(len(pos1))
            #                     n1 = pos1[n1,0]
            #                     embedding0 = model.a[n0, :] * torch.ones((1500, config.graph_model.embedding_dim), device=device)
            #                     embedding1 = model.a[n1, :] * torch.ones((1500, config.graph_model.embedding_dim), device=device)
            #                     in_features = torch.cat((rr[:,None],embedding0, embedding1), dim=1)
            #                     if config.graph_model.lin_edge_positive:
            #                         func = model.lin_edge(in_features.float()) ** 2 * correction
            #                     else:
            #                         func = model.lin_edge(in_features.float()) * correction
            #                     if r == '2steps':
            #                         field = torch.ones_like(rr[:,None])
            #                         u = torch.zeros_like(rr[:,None])
            #                         in_features2 = torch.cat([u, func, field], dim=1)
            #                         func = model.lin_phi2(in_features2)
            #                     elif r == 'generic':
            #                         field = torch.ones_like(rr[:,None])
            #                         u = torch.zeros_like(rr[:,None])
            #                         in_features = torch.cat([u, embedding0, func.detach().clone(), field], dim=1)
            #                         func = model.lin_phi(in_features)
            #                     psi_list.append(func)
            #                     plt.plot(to_numpy(rr), to_numpy(func), 2, color=cmap.color(n),linewidth=1, alpha=0.25)
            #             # plt.ylim([-1.1, 1.1])
            #             plt.xlim([-to_numpy(xnorm)*2, to_numpy(xnorm)*2])
            #             plt.xticks(fontsize=18)
            #             plt.yticks(fontsize=18)
            #             # plt.ylabel(r'learned $\psi^*(\mathbf{a}_i, a_j, v_i)$', fontsize=24)
            #             # plt.xlabel(r'$v_i$', fontsize=24)
            #             # plt.ylim([-1.5, 1.5])
            #             # plt.xlim([-5, 5])
            #
            #         plt.tight_layout()
            #         plt.savefig(f"./{log_dir}/results/learned_psi_{r}.png", dpi=170.7)
            #         plt.close()
            #     psi_list = torch.stack(psi_list)
            #     psi_list = psi_list.squeeze()
            # else:
            #     psi_list = []
            #     fig, ax = fig_init()
            #     rr = torch.linspace(-xnorm.squeeze(), xnorm.squeeze(), 1500).to(device)
            #     if is_CElegans:
            #         rr = torch.linspace(-xnorm.squeeze() * 4, xnorm.squeeze() * 4, 1500).to(device)
            #     else:
            #         rr = torch.linspace(-xnorm.squeeze(), xnorm.squeeze(), 1500).to(device)
            #     if not(is_CElegans):
            #         if (model_config.signal_model_name == 'PDE_N4'):
            #             for n in range(n_neuron_types):
            #                 true_func = true_model.func(rr, n, 'phi')
            #                 plt.plot(to_numpy(rr), to_numpy(true_func), c = mc, linewidth = 16, label = 'original', alpha = 0.21)
            #         else:
            #             true_func = true_model.func(rr, 0, 'phi')
            #             plt.plot(to_numpy(rr), to_numpy(true_func), c = mc, linewidth = 16, label = 'original', alpha = 0.21)
            #
            #     for n in trange(0,n_neurons):
            #         if (model_config.signal_model_name == 'PDE_N4') | (model_config.signal_model_name == 'PDE_N5'):
            #             embedding_ = model.a[n, :] * torch.ones((1500, config.graph_model.embedding_dim), device=device)
            #             in_features = get_in_features(rr, embedding_, model_config.signal_model_name, max_radius)
            #         else:
            #             in_features = rr[:, None]
            #         with torch.no_grad():
            #             if config.graph_model.lin_edge_positive:
            #                 func = model.lin_edge(in_features.float()) ** 2 * correction
            #             else:
            #                 func = model.lin_edge(in_features.float()) * correction
            #             psi_list.append(func)
            #         if (model_config.signal_model_name == 'PDE_N4') | (model_config.signal_model_name == 'PDE_N5'):
            #             plt.plot(to_numpy(rr), to_numpy(func), 2, color=cmap.color(to_numpy(type_list)[n].astype(int)), linewidth=2, alpha=0.25)
            #         else:
            #             plt.plot(to_numpy(rr), to_numpy(func), 2, color=mc, linewidth=2, alpha=0.25)
            #
            #     plt.xlabel(r'$v_i$', fontsize=68)
            #     if (model_config.signal_model_name == 'PDE_N4'):
            #         plt.ylabel(r'learned $\psi^*(\mathbf{a}_i, v_i)$', fontsize=68)
            #     elif model_config.signal_model_name == 'PDE_N5':
            #         plt.ylabel(r'learned $\psi^*(\mathbf{a}_i, a_j, v_i)$', fontsize=68)
            #     else:
            #         plt.ylabel(r'learned $\psi^*(v_i)$', fontsize=68)
            #     if config.graph_model.lin_edge_positive:
            #         plt.ylim([-0.2, 1.2])
            #     else:
            #         plt.ylim([-1.6, 1.6])
            #     plt.tight_layout()
            #     plt.savefig(f"./{log_dir}/results/learned_psi.png", dpi=170.7)
            #     plt.close()
            #     psi_list = torch.stack(psi_list)
            #     psi_list = psi_list.squeeze()
            #
            # print('interaction functions ...')
            #
            # fig, ax = fig_init()
            # if not (is_CElegans):
            #     for n in trange(n_neuron_types):
            #         if model_config.signal_model_name == 'PDE_N5':
            #             true_func = true_model.func(rr, n, n, 'update')
            #         else:
            #             true_func = true_model.func(rr, n, 'update')
            #         plt.plot(to_numpy(rr), to_numpy(true_func), c=mc, linewidth=16, label='original', alpha=0.21)
            # phi_list = []
            # for n in trange(n_neurons):
            #     embedding_ = model.a[n, :] * torch.ones((1500, config.graph_model.embedding_dim), device=device)
            #     # in_features = torch.cat((rr[:, None], embedding_), dim=1)
            #     in_features = get_in_features_update(rr[:, None], n_neurons, embedding_, model.update_type, device)
            #     with torch.no_grad():
            #         func = model.lin_phi(in_features.float())
            #     func = func[:, 0]
            #     phi_list.append(func)
            #     plt.plot(to_numpy(rr), to_numpy(func) * to_numpy(ynorm),
            #              color=cmap.color(to_numpy(type_list[n]).astype(int)), linewidth=2, alpha=0.25)
            # phi_list = torch.stack(phi_list)
            # func_list_ = to_numpy(phi_list)
            # plt.xlabel(r'$v_i$', fontsize=68)
            # plt.ylabel(r'learned $\phi^*(\mathbf{a}_i, v_i)$', fontsize=68)
            # plt.tight_layout()
            # # plt.xlim([-to_numpy(xnorm), to_numpy(xnorm)])
            # if not (is_CElegans):
            #     plt.ylim(config.plotting.ylim)
            # plt.savefig(f'./{log_dir}/results/learned phi.png', dpi=300)
            # plt.close()
            #
            # print('UMAP reduction ...')
            # with warnings.catch_warnings():
            #     warnings.simplefilter('ignore')
            #     trans = umap.UMAP(n_neighbors=50, n_components=2, transform_queue_size=0,
            #                       random_state=config.training.seed).fit(func_list_)
            #     proj_interaction = trans.transform(func_list_)
            #
            # proj_interaction = (proj_interaction - np.min(proj_interaction)) / (np.max(proj_interaction) - np.min(proj_interaction) + 1e-10)
            # fig, ax = fig_init()
            # for n in trange(n_neuron_types):
            #     pos = torch.argwhere(type_list == n)
            #     pos = to_numpy(pos)
            #     if len(pos) > 0:
            #         plt.scatter(proj_interaction[pos, 0],
            #                     proj_interaction[pos, 1], s=200, alpha=0.1)
            # plt.xlabel(r'UMAP 0', fontsize=68)
            # plt.ylabel(r'UMAP 1', fontsize=68)
            # plt.xlim([-0.2, 1.2])
            # plt.ylim([-0.2, 1.2])
            # plt.tight_layout()
            # plt.savefig(f"./{log_dir}/results/UMAP.png", dpi=170.7)
            # plt.close()
            #
            # config.training.cluster_distance_threshold = 0.1
            # config.training.cluster_method = 'distance_embedding'
            # embedding = to_numpy(model.a.squeeze())
            # labels, n_clusters, new_labels = sparsify_cluster(config.training.cluster_method, proj_interaction, embedding,
            #                                                   config.training.cluster_distance_threshold, type_list,
            #                                                   n_neuron_types, embedding_cluster)
            # accuracy = metrics.accuracy_score(to_numpy(type_list), new_labels[:n_neurons])
            # print(f'accuracy: {accuracy:0.4f}   n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}  ')
            # logger.info(f'accuracy: {accuracy:0.4f}   n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method} ')
            #
            # # config.training.cluster_method = 'kmeans_auto_embedding'
            # # labels, n_clusters, new_labels = sparsify_cluster(config.training.cluster_method, proj_interaction, embedding,
            # #                                                   config.training.cluster_distance_threshold, type_list,
            # #                                                   n_neuron_types, embedding_cluster)
            # # accuracy = metrics.accuracy_score(to_numpy(type_list), new_labels)
            # # print(f'accuracy: {accuracy:0.4f}   n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}  ')
            # # logger.info(f'accuracy: {accuracy:0.4f}   n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method} ')
            #
            # plt.figure(figsize=(10, 10))
            # plt.scatter(to_numpy(X1_first[:n_neurons, 0]), to_numpy(X1_first[:n_neurons, 1]), s=150, color=cmap.color(to_numpy(type_list).astype(int)))
            # plt.xticks([])
            # plt.yticks([])
            # plt.axis('off')
            # plt.tight_layout()
            # plt.savefig(f"./{log_dir}/results/true_types.png", dpi=170.7)
            # plt.close()
            #
            # plt.figure(figsize=(10, 10))
            # plt.scatter(to_numpy(X1_first[:n_neurons, 0]), to_numpy(X1_first[:n_neurons, 1]), s=150, color=cmap.color(new_labels[:n_neurons].astype(int)))
            # plt.xticks([])
            # plt.yticks([])
            # plt.axis('off')
            # plt.tight_layout()
            # plt.savefig(f"./{log_dir}/results/learned_types.png", dpi=170.7)
            # plt.close()
            #
            # fig, ax = fig_init()
            # gt_weight = to_numpy(adjacency)
            # pred_weight = to_numpy(A)
            # plt.scatter(gt_weight, pred_weight / 10 , s=0.1, c=mc, alpha=0.1)
            # plt.xlabel(r'true $W_{ij}$', fontsize=68)
            # plt.ylabel(r'learned $W_{ij}$', fontsize=68)
            # if n_neurons == 8000:
            #     plt.xlim([-0.05,0.05])
            #     plt.ylim([-0.05,0.05])
            # else:
            #     plt.xlim([-0.2,0.2])
            #     plt.ylim([-0.2,0.2])
            # plt.tight_layout()
            # plt.savefig(f"./{log_dir}/results/first_comparison.png", dpi=87)
            # plt.close()
            #
            # x_data = np.reshape(gt_weight, (n_neurons * n_neurons))
            # y_data =  np.reshape(pred_weight, (n_neurons * n_neurons))
            # lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
            # residuals = y_data - linear_model(x_data, *lin_fit)
            # ss_res = np.sum(residuals ** 2)
            # ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            # r_squared = 1 - (ss_res / ss_tot)
            # print(f'R^2$: {r_squared:0.4f}  slope: {np.round(lin_fit[0], 4)}')
            # logger.info(f'R^2$: {np.round(r_squared, 4)}  slope: {np.round(lin_fit[0], 4)}')
            #
            # second_correction = lin_fit[0]
            # print(f'second_correction: {second_correction:0.2f}')
            # np.save(f'{log_dir}/second_correction.npy', second_correction)
            #
            # fig, ax = fig_init()
            # gt_weight = to_numpy(adjacency)
            # pred_weight = to_numpy(A)
            # plt.scatter(gt_weight, pred_weight / second_correction, s=0.1, c=mc, alpha=0.1)
            # plt.xlabel(r'true $W_{ij}$', fontsize=68)
            # plt.ylabel(r'learned $W_{ij}$', fontsize=68)
            # if n_neurons == 8000:
            #     plt.xlim([-0.05,0.05])
            #     plt.ylim([-0.05,0.05])
            # else:
            #     plt.xlim([-0.2,0.2])
            #     plt.ylim([-0.2,0.2])
            # plt.tight_layout()
            # plt.savefig(f"./{log_dir}/results/second_comparison.png", dpi=87)
            # plt.close()
            #
            # plt.figure(figsize=(10, 10))
            # # plt.title(r'learned $W_{ij}$', fontsize=68)
            # ax = sns.heatmap(to_numpy(A)/second_correction, center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046})
            # cbar = ax.collections[0].colorbar
            # # here set the labelsize by 20
            # cbar.ax.tick_params(labelsize=32)
            # plt.xticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
            # plt.yticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
            # plt.xticks(rotation=0)
            # plt.subplot(2, 2, 1)
            # ax = sns.heatmap(to_numpy(A[0:20, 0:20]/second_correction), cbar=False, center=0, square=True, cmap='bwr')
            # plt.xticks(rotation=0)
            # plt.xticks([])
            # plt.yticks([])
            # plt.tight_layout()
            # plt.savefig(f'./{log_dir}/results/final learned connectivity.png', dpi=300)
            # plt.close()

            if False: # has_field:

                print('plot field ...')
                os.makedirs(f"./{log_dir}/results/field", exist_ok=True)

                if 'derivative' in field_type:

                    y = torch.linspace(0, 1, 400)
                    x = torch.linspace(-6, 6, 400)
                    grid_y, grid_x = torch.meshgrid(y, x)
                    grid = torch.stack((grid_x, grid_y), dim=-1)
                    grid = grid.to(device)
                    pred_modulation = model.lin_modulation(grid)
                    tau = 100
                    alpha = 0.02
                    true_derivative = (1 - grid_y) / tau - alpha * grid_y * torch.abs(grid_x)

                    fig, ax = fig_init()
                    plt.title(r'$\dot{y_i}$', fontsize=68)
                    # plt.title(r'$\dot{y_i}=(1-y)/100 - 0.02 x_iy_i$', fontsize=48)
                    plt.imshow(to_numpy(true_derivative))
                    plt.xticks([0, 100, 200, 300, 400], [-6, -3, 0, 3, 6], fontsize=48)
                    plt.yticks([0, 100, 200, 300, 400], [0, 0.25, 0.5, 0.75, 1], fontsize=48)
                    plt.xlabel(r'$v_i$', fontsize=68)
                    plt.ylabel(r'$y_i$', fontsize=68)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/true_field_derivative.png", dpi=80)
                    plt.close()

                    fig, ax = fig_init()
                    plt.title(r'learned $\dot{y_i}$', fontsize=68)
                    plt.imshow(to_numpy(pred_modulation))
                    plt.xticks([0, 100, 200, 300, 400], [-6, -3, 0, 3, 6], fontsize=48)
                    plt.yticks([0, 100, 200, 300, 400], [0, 0.25, 0.5, 0.75, 1], fontsize=48)
                    plt.xlabel(r'$v_i$', fontsize=68)
                    plt.ylabel(r'$y_i$', fontsize=68)
                    # plt.colorbar()
                    plt.tight_layout
                    plt.savefig(f"./{log_dir}/results/field_derivative.png", dpi=80)
                    plt.close()

                    # fig = plt.figure(figsize=(12, 12))
                    # ind_list = [320]
                    # ids = np.arange(0, 100000, 100)
                    # ax = fig.add_subplot(2, 1, 1)
                    # for ind in ind_list:
                    #     plt.plot(to_numpy(modulation[ind, ids]))
                    #     plt.plot(to_numpy(model.b[ind, 0:1000]**2))

                if ('short_term_plasticity' in field_type) | ('modulation' in field_type):

                    for frame in trange(0, n_frames, n_frames // 100):
                        t = torch.tensor([frame/ n_frames], dtype=torch.float32, device=device)
                        if (model_config.update_type == '2steps'):
                                m_ = model_f(t) ** 2
                                m_ = m_[:,None]
                                in_features= torch.cat((torch.zeros_like(m_), torch.ones_like(m_)*xnorm, m_), dim=1)
                                m = model.lin_phi2(in_features)
                        else:
                            m = model_f[0](t) ** 2

                        if 'permutation' in model_config.field_type:
                            inverse_permutation_indices = torch.load(f'./graphs_data/{dataset_name}/inverse_permutation_indices.pt', map_location=device)
                            modulation_ = m[inverse_permutation_indices]
                        else:
                            modulation_ = m
                        modulation_ = torch.reshape(modulation_, (32, 32)) * torch.tensor(second_correction, device=device) / 10

                        fig = plt.figure(figsize=(10, 10.5))
                        plt.axis('off')
                        plt.xticks([])
                        plt.xticks([])
                        im_ = to_numpy(modulation_)
                        im_ = np.rot90(im_, k=-1)
                        im_ = np.flipud(im_)
                        im_ = np.fliplr(im_)
                        plt.imshow(im_, cmap='gray')
                        # plt.title(r'neuromodulation $b_i$', fontsize=48)
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/results/field/xi_{frame}.png", dpi=80)
                        plt.close()

                        # x = x_list[0][frame]
                        # fig = plt.figure(figsize=(10, 10.5))
                        # plt.axis('off')
                        # plt.xticks([])
                        # plt.xticks([])
                        # plt.scatter(x[:,1], x[:,2], s=160, c=to_numpy(modulation[:,frame]),
                        #             vmin=0, vmax=2, cmap='viridis')
                        # plt.title(r'neuromodulation $b_i$', fontsize=48)
                        # plt.tight_layout()
                        # plt.savefig(f"./{log_dir}/results/field/bi_{frame}.png", dpi=80)
                        # plt.close()
                        # fig = plt.figure(figsize=(10, 10.5))
                        # plt.axis('off')
                        # plt.xticks([])
                        # plt.xticks([])
                        # plt.scatter(x[:,1], x[:,2], s=160, c=x[:,6],
                        #             vmin=-20, vmax=20, cmap='viridis')
                        # plt.title(r'$v_i$', fontsize=48)
                        # plt.tight_layout()
                        # plt.savefig(f"./{log_dir}/results/field/xi_{frame}.png", dpi=80)
                        # plt.close()

                    fig, ax = fig_init()
                    t = torch.linspace(0, 1, 100000, dtype=torch.float32, device=device).unsqueeze(1)

                    prediction = model_f(t) ** 2
                    prediction = prediction.t()
                    plt.imshow(to_numpy(prediction), aspect='auto')
                    plt.title(r'learned $MLP_2(i,t)$', fontsize=68)
                    plt.xlabel(r'$t$', fontsize=68)
                    plt.ylabel(r'$i$', fontsize=68)
                    plt.xticks([10000, 100000], [10000, 100000], fontsize=48)
                    plt.yticks([0, 512, 1024], [0, 512, 1024], fontsize=48)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/learned_plasticity.png", dpi=80)
                    plt.close()

                    fig, ax = fig_init()
                    plt.imshow(to_numpy(modulation), aspect='auto')
                    plt.title(r'$y_i$', fontsize=68)
                    plt.xlabel(r'$t$', fontsize=68)
                    plt.ylabel(r'$i$', fontsize=68)
                    plt.xticks([10000, 100000], [10000, 100000], fontsize=48)
                    plt.yticks([0, 512, 1024], [0, 512, 1024], fontsize=48)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/true_plasticity.png", dpi=80)
                    plt.close()

                    prediction = prediction * torch.tensor(second_correction, device=device) / 10

                    fig, ax = fig_init()
                    ids = np.arange(0, 100000, 100).astype(int)
                    plt.scatter(to_numpy(modulation[:, ids]), to_numpy(prediction[:, ids]), s=0.1, color=mc, alpha=0.05)
                    # plt.xlim([0, 1])
                    # plt.ylim([0, 2])
                    # plt.xticks([0, 0.5], [0, 0.5], fontsize=48)
                    # plt.yticks([0, 1, 2], [0, 1, 2], fontsize=48)
                    x_data = to_numpy(modulation[:, ids]).flatten()
                    y_data = to_numpy(prediction[:, ids]).flatten()
                    lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                    residuals = y_data - linear_model(x_data, *lin_fit)
                    ss_res = np.sum(residuals ** 2)
                    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot)
                    ax.text(0.05, 0.94, f'$R^2$: {r_squared:0.2f}', transform=ax.transAxes,
                            verticalalignment='top', horizontalalignment='left', fontsize=32)
                    ax.text(0.05, 0.88, f'slope: {lin_fit[0]:0.2f}', transform=ax.transAxes,
                            verticalalignment='top', horizontalalignment='left', fontsize=32)
                    plt.xlabel(r'true $y_i(t)$', fontsize=68)
                    plt.ylabel(r'learned $y_i(t)$', fontsize=68)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/comparison_yi.png", dpi=80)
                    plt.close()

                else:
                    net = f'{log_dir}/models/best_model_f_with_{n_runs - 1}_graphs_{epoch}.pt'
                    state_dict = torch.load(net, map_location=device)
                    model_f.load_state_dict(state_dict['model_state_dict'])
                    im = imread(f"graphs_data/{config.simulation.node_value_map}")

                    x = x_list[0][0]

                    slope_list = list([])
                    im_list = list([])
                    pred_list = list([])

                    for frame in trange(0, n_frames, n_frames // 100):

                        fig, ax = fig_init()
                        im_ = np.zeros((44, 44))
                        if (frame >= 0) & (frame < n_frames):
                            im_ = im[int(frame / n_frames * 256)].squeeze()
                        plt.imshow(im_, cmap='gray', vmin=0, vmax=2)
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/results/field/true_field{epoch}_{frame}.png", dpi=80)
                        plt.close()

                        pred = model_f(time=frame / n_frames, enlarge=False) ** 2 * second_correction / 10
                        pred = torch.reshape(pred, (n_nodes_per_axis, n_nodes_per_axis))

                        pred = to_numpy(pred)
                        pred = np.flipud(pred)
                        pred = np.rot90(pred, 1)
                        pred = np.fliplr(pred)
                        fig, ax = fig_init()
                        plt.imshow(pred, cmap='gray', vmin=0, vmax=2)
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/results/field/reconstructed_field_LR {epoch}_{frame}.png", dpi=80)
                        plt.close()

                        x_data = np.reshape(im_, (n_nodes_per_axis * n_nodes_per_axis))
                        y_data = np.reshape(pred, (n_nodes_per_axis * n_nodes_per_axis))
                        lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                        residuals = y_data - linear_model(x_data, *lin_fit)
                        ss_res = np.sum(residuals ** 2)
                        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot)
                        # print(f'R^2$: {r_squared:0.4f}  slope: {np.round(lin_fit[0], 4)}')
                        slope_list.append(lin_fit[0])

                        fig, ax = fig_init()
                        plt.scatter(im_, pred, s=10, c=mc)
                        plt.xlim([0.3, 1.6])
                        # plt.ylim([0.3, 1.6])
                        plt.xlabel(r'true neuromodulation', fontsize=48)
                        plt.ylabel(r'learned neuromodulation', fontsize=48)
                        plt.text(0.35, 1.5, f'$R^2$: {r_squared:0.2f}  slope: {np.round(lin_fit[0], 2)}', fontsize=42)
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/results/field/comparison {epoch}_{frame}.png", dpi=80)
                        plt.close()
                        im_list.append(im_)
                        pred_list.append(pred)

                        pred = model_f(time=frame / n_frames, enlarge=True) ** 2 * second_correction / 10 # /lin_fit[0]
                        pred = torch.reshape(pred, (640, 640))
                        pred = to_numpy(pred)
                        pred = np.flipud(pred)
                        pred = np.rot90(pred, 1)
                        pred = np.fliplr(pred)
                        fig, ax = fig_init()
                        plt.imshow(pred, cmap='gray')
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"./{log_dir}/results/field/reconstructed_field_HR {epoch}_{frame}.png", dpi=80)
                        plt.close()

                    im_list = np.array(np.array(im_list))
                    pred_list = np.array(np.array(pred_list))

                    im_list_ = np.reshape(im_list,(100,1024))
                    pred_list_ = np.reshape(pred_list,(100,1024))
                    im_list_ = np.rot90(im_list_)
                    pred_list_ = np.rot90(pred_list_)
                    im_list_ = scipy.ndimage.zoom(im_list_, (1024 / im_list_.shape[0], 1024 / im_list_.shape[1]))
                    pred_list_ = scipy.ndimage.zoom(pred_list_, (1024 / pred_list_.shape[0], 1024 / pred_list_.shape[1]))

                    plt.figure(figsize=(20, 10))
                    plt.subplot(1, 2, 1)
                    plt.title('true field')
                    plt.imshow(im_list_, cmap='grey')
                    plt.xticks([])
                    plt.yticks([])
                    plt.subplot(1, 2, 2)
                    plt.title('reconstructed field')
                    plt.imshow(pred_list_, cmap='grey')
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/pic_comparison {epoch}.png", dpi=80)
                    plt.close()

                    fig, ax = fig_init()
                    plt.scatter(im_list, pred_list, s=1, c=mc, alpha=0.1)
                    plt.xlim([0.3, 1.6])
                    plt.ylim([0.3, 1.6])
                    plt.xlabel(r'true $\Omega_i$', fontsize=68)
                    plt.ylabel(r'learned $\Omega_i$', fontsize=68)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/all_comparison {epoch}.png", dpi=80)
                    plt.close()

                    x_data = np.reshape(im_list, (100 * n_nodes_per_axis * n_nodes_per_axis))
                    y_data = np.reshape(pred_list, (100 * n_nodes_per_axis * n_nodes_per_axis))
                    lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                    residuals = y_data - linear_model(x_data, *lin_fit)
                    ss_res = np.sum(residuals ** 2)
                    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot)
                    print(f'field R^2$: {r_squared:0.4f}  slope: {np.round(lin_fit[0], 4)}')

            # if 'PDE_N6' in model_config.signal_model_name:
            #
            #     modulation = torch.tensor(x_list[0], device=device)
            #     modulation = modulation[:, :, 8:9].squeeze()
            #     modulation = modulation.t()
            #     modulation = modulation.clone().detach()
            #     modulation = to_numpy(modulation)
            #
            #     modulation = scipy.ndimage.zoom(modulation, (1024 / modulation.shape[0], 1024 / modulation.shape[1]))
            #     pred_list_ = to_numpy(model.b**2)
            #     pred_list_ = scipy.ndimage.zoom(pred_list_, (1024 / pred_list_.shape[0], 1024 / pred_list_.shape[1]))
            #
            #     plt.figure(figsize=(20, 10))
            #     plt.subplot(1, 2, 1)
            #     plt.title('true field')
            #     plt.imshow(modulation, cmap='grey')
            #     plt.xticks([])
            #     plt.yticks([])
            #     plt.subplot(1, 2, 2)
            #     plt.title('reconstructed field')
            #     plt.imshow(pred_list_, cmap='grey')
            #     plt.xticks([])
            #     plt.yticks([])
            #     plt.tight_layout()
            #     plt.tight_layout()
            #     plt.savefig(f"./{log_dir}/results/pic_comparison {epoch}.png", dpi=80)
            #     plt.close()
            #
            #     for frame in trange(0, modulation.shape[1], modulation.shape[1] // 257):
            #         im = modulation[:, frame]
            #         im = np.reshape(im, (32, 32))
            #         plt.figure(figsize=(8, 8))
            #         plt.axis('off')
            #         plt.imshow(im, cmap='gray', vmin=0, vmax=1)
            #         plt.tight_layout()
            #         plt.savefig(f"./{log_dir}/results/field/true_field_{frame}.png", dpi=80)
            #         plt.close()
            #
            # if (model.update_type == 'generic') & (model_config.signal_model_name == 'PDE_N5'):
            #
            #     k = np.random.randint(n_frames - 50)
            #     x = torch.tensor(x_list[0][k], device=device)
            #     if has_field:
            #         if 'visual' in field_type:
            #             x[:n_nodes, 8:9] = model_f(time=k / n_frames) ** 2
            #             x[n_nodes:n_neurons, 8:9] = 1
            #         elif 'learnable_short_term_plasticity' in field_type:
            #             alpha = (k % model.embedding_step) / model.embedding_step
            #             x[:, 8] = alpha * model.b[:, k // model.embedding_step + 1] ** 2 + (1 - alpha) * model.b[:,
            #                                                                                              k // model.embedding_step] ** 2
            #         elif ('short_term_plasticity' in field_type) | ('modulation_permutation' in field_type):
            #             t = torch.tensor([k / n_frames], dtype=torch.float32, device=device)
            #             x[:, 8] = model_f(t) ** 2
            #         else:
            #             x[:, 8:9] = model_f(time=k / n_frames) ** 2
            #     else:
            #         x[:, 8:9] = torch.ones_like(x[:, 0:1])
            #     dataset = data.Data(x=x, edge_index=edge_index)
            #     pred, in_features_ = model(data=dataset, return_all=True)
            #     feature_list = ['u', 'embedding0', 'embedding1', 'msg', 'field']
            #     for n in range(in_features_.shape[1]):
            #         print(f'feature {feature_list[n]}: {to_numpy(torch.mean(in_features_[:, n])):0.4f}  std: {to_numpy(torch.std(in_features_[:, n])):0.4f}')
            #
            #     fig, ax = fig_init()
            #     plt.hist(to_numpy(in_features_[:, -1]), 150)
            #     plt.tight_layout()
            #     plt.close()
            #
            #     fig, ax = fig_init()
            #     f = torch.reshape(x[:n_nodes, 8:9], (n_nodes_per_axis, n_nodes_per_axis))
            #     plt.imshow(to_numpy(f), cmap='viridis', vmin=-1, vmax=10)
            #     plt.tight_layout()
            #     plt.close()
            #
            #
            #     fig, ax = fig_init()
            #     msg_list = []
            #     u = torch.linspace(-xnorm.squeeze(), xnorm.squeeze(), 400).to(device)
            #     for sample in range(n_neurons):
            #         id0 = np.random.randint(0, n_neurons)
            #         id1 = np.random.randint(0, n_neurons)
            #         f = x[id0, 8:9]
            #         embedding0 = model.a[id0, :] * torch.ones((400, config.graph_model.embedding_dim), device=device)
            #         embedding1 = model.a[id1, :] * torch.ones((400, config.graph_model.embedding_dim), device=device)
            #         in_features = torch.cat((u[:, None], embedding0, embedding1), dim=1)
            #         msg = model.lin_edge(in_features.float()) ** 2 * correction
            #         in_features = torch.cat((torch.zeros((400, 1), device=device), embedding0, msg,
            #                                  f * torch.ones((400, 1), device=device)), dim=1)
            #         plt.plot(to_numpy(u), to_numpy(msg), c=cmap.color(to_numpy(x[id0, 5]).astype(int)), linewidth=2, alpha=0.15)
            #         # plt.scatter(to_numpy(u), to_numpy(model.lin_phi(in_features)), s=5, c='r', alpha=0.15)
            #         # plt.scatter(to_numpy(u), to_numpy(f*msg), s=1, c='w', alpha=0.1)
            #         msg_list.append(msg)
            #     plt.tight_layout()
            #     msg_list = torch.stack(msg_list).squeeze()
            #     y_min, y_max = msg_list.min().item(), msg_list.max().item()
            #     plt.xlabel(r'$v_i$', fontsize=68)
            #     plt.ylabel(r'learned $\mathrm{MLP_0}$', fontsize=68)
            #     plt.ylim([y_min - y_max / 2, y_max * 1.5])
            #     plt.tight_layout()
            #     plt.savefig(f'./{log_dir}/results/learned_multiple_psi_{epoch}.png', dpi=300)
            #     plt.close()
            #
            #     fig, ax = fig_init()
            #     u = torch.linspace(-xnorm.squeeze(), xnorm.squeeze(), 400).to(device)
            #     for n in range(n_neuron_types):
            #         for m in range(n_neuron_types):
            #             true_func = true_model.func(u, n, m, 'phi')
            #             plt.plot(to_numpy(u), to_numpy(true_func), c=cmap.color(n), linewidth=3)
            #     plt.xlabel(r'$v_i$', fontsize=68)
            #     plt.ylabel(r'true functions', fontsize=68)
            #     plt.ylim([y_min - y_max / 2, y_max * 1.5])
            #     plt.tight_layout()
            #     plt.savefig(f'./{log_dir}/results/true_multiple_psi.png', dpi=300)
            #     plt.close()
            #
            #     msg_start = torch.mean(in_features_[:, 3]) - torch.std(in_features_[:, 3])
            #     msg_end = torch.mean(in_features_[:, 3]) + torch.std(in_features_[:, 3])
            #     msgs = torch.linspace(msg_start, msg_end, 400).to(device)
            #     fig, ax = fig_init()
            #     func_list = []
            #     rr_list = []
            #     for sample in range(n_neurons):
            #         id0 = np.random.randint(0, n_neurons)
            #         embedding0 = model.a[id0, :] * torch.ones((400, config.graph_model.embedding_dim), device=device)
            #         in_features = torch.cat((torch.zeros((400, 1), device=device), embedding0, msgs[:,None], torch.ones((400, 1), device=device)), dim=1)
            #         pred = model.lin_phi(in_features)
            #         plt.plot(to_numpy(msgs), to_numpy(pred), c=cmap.color(to_numpy(x[id0, 5]).astype(int)),  linewidth=2, alpha=0.25)
            #         func_list.append(pred)
            #         rr_list.append(msgs)
            #     plt.xlabel(r'$sum_i$', fontsize=68)
            #     plt.ylabel(r'$\mathrm{MLP_0}(\mathbf{a}_i, x_i=0, sum_i, g_i=1)$', fontsize=48)
            #     plt.tight_layout()
            #     plt.savefig(f'./{log_dir}/results/learned_multivariate_phi_{epoch}.png', dpi=300)
            #     plt.close()
            #
            #
            #     print('symbolic regression ...')
            #
            #     text_trap = StringIO()
            #     sys.stdout = text_trap
            #
            #     model_pysrr = PySRRegressor(
            #         niterations=30,  # < Increase me for better results
            #         binary_operators=["+", "*"],
            #         unary_operators=[
            #             "cos",
            #             "exp",
            #             "sin",
            #             "tanh"
            #         ],
            #         random_state=0,
            #         temp_equation_file=False
            #     )
            #
            #     # rr_ = torch.rand((4000, 2), device=device)
            #     # func_ = rr_[:,0] * rr_[:,1]
            #     # model_pysrr.fit(to_numpy(rr_), to_numpy(func_))
            #     # model_pysrr.sympy
            #
            #     func_list = torch.stack(func_list).squeeze()
            #     rr_list = torch.stack(rr_list).squeeze()
            #     func = torch.reshape(func_list, (func_list.shape[0] * func_list.shape[1], 1))
            #     rr = torch.reshape(rr_list, (func_list.shape[0] * func_list.shape[1], 1))
            #     idx = torch.randperm(len(rr))[:5000]
            #
            #     model_pysrr.fit(to_numpy(rr[idx]), to_numpy(func[idx]))
            #
            #     sys.stdout = sys.__stdout__
            #
            #
            #     # if model_config.signal_model_name == 'PDE_N4':
            #     #
            #     #     fig, ax = fig_init()
            #     #     for m in range(n_neuron_types):
            #     #         u = torch.linspace(-xnorm.squeeze() * 2, xnorm.squeeze() * 2, 400).to(device)
            #     #         true_func = true_model.func(u, m, 'phi')
            #     #         embedding0 = model.a[m * n_neurons // n_neuron_types, :] * torch.ones(
            #     #             (400, config.graph_model.embedding_dim), device=device)
            #     #         field = torch.ones((400, 1), device=device)
            #     #         in_features = torch.cat((u[:, None], embedding0), dim=1)
            #     #         if config.graph_model.lin_edge_positive:
            #     #             MLP0_func = model.lin_edge(in_features.float()) ** 2 * correction
            #     #         in_features = torch.cat((u[:, None] * 0, embedding0, MLP0_func, field), dim=1)
            #     #         MLP1_func = model.lin_phi(in_features)
            #     #         plt.plot(to_numpy(u), to_numpy(true_func), c='g', linewidth=3, label='true')
            #     #         plt.plot(to_numpy(u), to_numpy(MLP0_func), c='r', linewidth=3, label='MLP')
            #     #         plt.plot(to_numpy(u), to_numpy(MLP1_func), c='w', linewidth=3, label='MLPoMLP')
            #     #         # plt.legend(fontsize=24)
            #     #     plt.tight_layout()
            #     #     plt.savefig(f'./{log_dir}/results/generic_MLP0_{epoch}.png', dpi=300)
            #     #     plt.close()
            #
            # if False:
            #     print ('symbolic regression ...')
            #
            #     def get_pyssr_function(model_pysrr, rr, func):
            #
            #         text_trap = StringIO()
            #         sys.stdout = text_trap
            #
            #         model_pysrr.fit(to_numpy(rr[:, None]), to_numpy(func[:, None]))
            #
            #         sys.stdout = sys.__stdout__
            #
            #         return model_pysrr.sympy
            #
            #     model_pysrr = PySRRegressor(
            #         niterations=30,  # < Increase me for better results
            #         binary_operators=["+", "*"],
            #         unary_operators=[
            #             "cos",
            #             "exp",
            #             "sin",
            #             "tanh"
            #         ],
            #         random_state=0,
            #         temp_equation_file=False
            #     )
            #
            #     match model_config.signal_model_name:
            #
            #         case 'PDE_N2':
            #
            #             func = torch.mean(psi_list, dim=0).squeeze()
            #
            #             symbolic = get_pyssr_function(model_pysrr, rr, func)
            #
            #             for n in range(0,7):
            #                 print(symbolic(n))
            #                 logger.info(symbolic(n))
            #
            #         case 'PDE_N4':
            #
            #             for k in range(n_neuron_types):
            #                 print('  ')
            #                 print('  ')
            #                 print('  ')
            #                 print(f'psi{k} ................')
            #                 logger.info(f'psi{k} ................')
            #
            #                 pos = np.argwhere(labels == k)
            #                 pos = pos.squeeze()
            #
            #                 func = psi_list[pos]
            #                 func = torch.mean(psi_list[pos], dim=0)
            #
            #                 symbolic = get_pyssr_function(model_pysrr, rr, func)
            #
            #                 # for n in range(0, 5):
            #                 #     print(symbolic(n))
            #                 #     logger.info(symbolic(n))
            #
            #         case 'PDE_N5':
            #
            #             for k in range(4**2):
            #
            #                 print('  ')
            #                 print('  ')
            #                 print('  ')
            #                 print(f'psi {k//4} {k%4}................')
            #                 logger.info(f'psi {k//4} {k%4} ................')
            #
            #                 pos =np.arange(k*250,(k+1)*250)
            #                 func = psi_list[pos]
            #                 func = torch.mean(psi_list[pos], dim=0)
            #
            #                 symbolic = get_pyssr_function(model_pysrr, rr, func)
            #
            #                 # for n in range(0, 7):
            #                 #     print(symbolic(n))
            #                 #     logger.info(symbolic(n))
            #
            #     for k in range(n_neuron_types):
            #         print('  ')
            #         print('  ')
            #         print('  ')
            #         print(f'phi{k} ................')
            #         logger.info(f'phi{k} ................')
            #
            #         pos = np.argwhere(labels == k)
            #         pos = pos.squeeze()
            #
            #         func = phi_list[pos]
            #         func = torch.mean(phi_list[pos], dim=0)
            #
            #         symbolic = get_pyssr_function(model_pysrr, rr, func)
            #
            #         # for n in range(4, 7):
            #         #     print(symbolic(n))
            #         #     logger.info(symbolic(n))



def plot_synaptic_flyvis(config, epoch_list, log_dir, logger, cc, style, extended, device):
    dataset_name = config.dataset
    model_config = config.graph_model
    config_indices = config.dataset.split('fly_N9_')[1] if 'fly_N9_' in config.dataset else 'evolution'

    n_frames = config.simulation.n_frames
    n_runs = config.training.n_runs
    n_neurons = config.simulation.n_neurons
    n_neuron_types = config.simulation.n_neuron_types
    delta_t = config.simulation.delta_t
    n_edges = config.simulation.n_edges

    colors_65 = sns.color_palette("Set3", 12) * 6  # pastel, repeat until 65
    colors_65 = colors_65[:65]

    config.simulation.max_radius if hasattr(config.simulation, 'max_radius') else 2.5
    dimension = config.simulation.dimension

    log_file = os.path.join(log_dir, 'results.log')
    if os.path.exists(log_file):
        os.remove(log_file)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create file handler only, no console output
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Clear any existing handlers

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logger.addHandler(file_handler)

    # Prevent propagation to root logger (which might have console handlers)
    logger.propagate = False

    print(f'experiment description: {config.description}')
    logger.info(f'experiment description: {config.description}')

    # Load neuron group mapping for flyvis

    cmap = CustomColorMap(config=config)

    if 'black' in style:
        plt.style.use('dark_background')
        mc = 'w'
    else:
        plt.style.use('default')
        mc = 'k'

    x_list = []
    y_list = []
    time.sleep(0.5)
    print('load simulation data...')
    for run in range(0, n_runs):
        if os.path.exists(f'graphs_data/{dataset_name}/x_list_{run}.pt'):
            x = torch.load(f'graphs_data/{dataset_name}/x_list_{run}.pt', map_location=device)
            y = torch.load(f'graphs_data/{dataset_name}/y_list_{run}.pt', map_location=device)
            x = to_numpy(torch.stack(x))
            y = to_numpy(torch.stack(y))
        else:
            x = np.load(f'graphs_data/{dataset_name}/x_list_{run}.npy')
            y = np.load(f'graphs_data/{dataset_name}/y_list_{run}.npy')
        x_list.append(x)
        y_list.append(y)

    ynorm = torch.load(os.path.join(log_dir, 'ynorm.pt'), map_location=device)
    if os.path.exists(os.path.join(log_dir, 'xnorm.pt')):
        xnorm = torch.load(os.path.join(log_dir, 'xnorm.pt'))
    else:
        xnorm = torch.tensor([5], device=device)

    print(f'xnorm: {to_numpy(xnorm)}, ynorm: {to_numpy(ynorm)}')
    logger.info(f'xnorm: {to_numpy(xnorm)}, ynorm: {to_numpy(ynorm)}')

    # Load data with new format
    # connectivity = torch.load(f'./graphs_data/{dataset_name}/connectivity.pt', map_location=device)
    gt_weights = torch.load(f'./graphs_data/{dataset_name}/weights.pt', map_location=device)
    gt_taus = torch.load(f'./graphs_data/{dataset_name}/taus.pt', map_location=device)
    gt_V_Rest = torch.load(f'./graphs_data/{dataset_name}/V_i_rest.pt', map_location=device)
    edges = torch.load(f'./graphs_data/{dataset_name}/edge_index.pt', map_location=device)
    true_weights = torch.zeros((n_neurons, n_neurons), dtype=torch.float32, device=edges.device)
    true_weights[edges[1], edges[0]] = gt_weights

    x = x_list[0][n_frames - 10]
    type_list = torch.tensor(x[:, 2 + 2 * dimension:3 + 2 * dimension], device=device)
    n_types = len(np.unique(to_numpy(type_list)))
    region_list = torch.tensor(x[:, 1 + 2 * dimension:2 + 2 * dimension], device=device)
    n_region_types = len(np.unique(to_numpy(region_list)))
    n_neurons = len(type_list)

    # Neuron type index to name mapping
    index_to_name = {
        0: 'Am', 1: 'C2', 2: 'C3', 3: 'CT1(Lo1)', 4: 'CT1(M10)', 5: 'L1', 6: 'L2', 7: 'L3', 8: 'L4', 9: 'L5',
        10: 'Lawf1', 11: 'Lawf2', 12: 'Mi1', 13: 'Mi10', 14: 'Mi11', 15: 'Mi12', 16: 'Mi13', 17: 'Mi14',
        18: 'Mi15', 19: 'Mi2', 20: 'Mi3', 21: 'Mi4', 22: 'Mi9', 23: 'R1', 24: 'R2', 25: 'R3', 26: 'R4',
        27: 'R5', 28: 'R6', 29: 'R7', 30: 'R8', 31: 'T1', 32: 'T2', 33: 'T2a', 34: 'T3', 35: 'T4a',
        36: 'T4b', 37: 'T4c', 38: 'T4d', 39: 'T5a', 40: 'T5b', 41: 'T5c', 42: 'T5d', 43: 'Tm1',
        44: 'Tm16', 45: 'Tm2', 46: 'Tm20', 47: 'Tm28', 48: 'Tm3', 49: 'Tm30', 50: 'Tm4', 51: 'Tm5Y',
        52: 'Tm5a', 53: 'Tm5b', 54: 'Tm5c', 55: 'Tm9', 56: 'TmY10', 57: 'TmY13', 58: 'TmY14',
        59: 'TmY15', 60: 'TmY18', 61: 'TmY3', 62: 'TmY4', 63: 'TmY5a', 64: 'TmY9'
    }

    activity = torch.tensor(x_list[0][:, :, 3:4], device=device)
    activity = activity.squeeze().t()
    mu_activity = torch.mean(activity, dim=1)
    sigma_activity = torch.std(activity, dim=1)

    print(f'neurons: {n_neurons}  edges: {edges.shape[1]}  neuron types: {n_types}  region types: {n_region_types}')
    logger.info(f'neurons: {n_neurons}  edges: {edges.shape[1]}  neuron types: {n_types}  region types: {n_region_types}')
    os.makedirs(f'{log_dir}/results/', exist_ok=True)

    sorted_neuron_type_names = [index_to_name.get(i, f'Type{i}') for i in range(n_neuron_types)]

    target_type_name_list = ['R1', 'R7', 'C2', 'Mi11', 'Tm1', 'Tm4', 'Tm30']
    activity_results = plot_neuron_activity_analysis(activity, target_type_name_list, type_list, index_to_name, n_neurons, n_frames, delta_t, f'{log_dir}/results/')
    plot_ground_truth_distributions(to_numpy(edges), to_numpy(gt_weights), to_numpy(gt_taus), to_numpy(gt_V_Rest), to_numpy(type_list), n_types, sorted_neuron_type_names, f'{log_dir}/results/')

    if ('Ising' in extended) | ('ising' in extended):
        analyze_ising_model(x_list, delta_t, log_dir, logger, to_numpy(edges))

    if epoch_list[0] == 'all':

        movie_synaptic_flyvis(config, log_dir, n_runs, device, x_list, y_list, edges, gt_weights, gt_taus, gt_V_Rest,
                              type_list, n_neurons, n_types, colors_65, mu_activity, sigma_activity, cmap, mc, ynorm,
                              logger)

    else:
        config_indices = config.dataset.split('fly_N9_')[1] if 'fly_N9_' in config.dataset else 'evolution'
        files, file_id_list = get_training_files(log_dir, n_runs)

        for epoch in epoch_list:

            net = f'{log_dir}/models/best_model_with_{n_runs - 1}_graphs_{epoch}.pt'
            model = Signal_Propagation_FlyVis(aggr_type=model_config.aggr_type, config=config, device=device)
            state_dict = torch.load(net, map_location=device)
            model.load_state_dict(state_dict['model_state_dict'])
            model.edges = edges
            print(f'net: {net}')
            logger.info(f'net: {net}')

            # Plot 1: Loss curve
            if os.path.exists(os.path.join(log_dir, 'loss.pt')):
                fig = plt.figure(figsize=(8, 6))
                ax = plt.gca()
                for spine in ax.spines.values():
                    spine.set_alpha(0.75)
                list_loss = torch.load(os.path.join(log_dir, 'loss.pt'))
                plt.plot(list_loss, color=mc, linewidth=2)
                plt.xlim([0, len(list_loss)])
                plt.ylabel('Loss')
                plt.xlabel('Epochs')
                plt.title('Training Loss')
                plt.tight_layout()
                plt.savefig(f'{log_dir}/results/loss.png', dpi=300)
                plt.close()

            # Plot 2: Embedding using model.a
            fig = plt.figure(figsize=(10, 9))
            ax = plt.gca()
            for spine in ax.spines.values():
                spine.set_alpha(0.75)
            for n in range(n_types):
                pos = torch.argwhere(type_list == n)
                plt.scatter(to_numpy(model.a[pos, 0]), to_numpy(model.a[pos, 1]), s=24, color=colors_65[n], alpha=0.8,
                            edgecolors='none')
            plt.xlabel(r'$\mathbf{a}_{i0}$', fontsize=48)
            plt.ylabel(r'$\mathbf{a}_{i1}$', fontsize=48)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            # results = clustering_gmm(to_numpy(model.a), type_list, n_components=75)
            # print(f"GMM n_components=75: {results['n_components']} components, "
            #     f"accuracy=\033[32m{results['accuracy']:.3f}\033[0m")
            # logger.info(f"eps={eps}: {results['n_clusters_found']} clusters, "
            #             f"accuracy={results['accuracy']:.3f}")
            # plt.text(0.05, 0.95, f"N: {n_neurons}\naccuracy: {results['accuracy']:.2f}",
            #         transform=plt.gca().transAxes, fontsize=32,
            #         verticalalignment='top', color=mc)
            plt.tight_layout()
            plt.savefig(f'{log_dir}/results/embedding_{config_indices}.png', dpi=300)
            plt.close()

            # Plot 3: Edge function visualization
            fig = plt.figure(figsize=(10, 9))
            ax = plt.gca()
            for spine in ax.spines.values():
                spine.set_alpha(0.75)
            for n in trange(n_neurons, ncols=90):
                rr = torch.linspace(config.plotting.xlim[0], config.plotting.xlim[1], 1000, device=device)
                embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                if ('PDE_N9_A' in config.graph_model.signal_model_name) | ('PDE_N9_D' in config.graph_model.signal_model_name):
                    in_features = torch.cat((rr[:, None], embedding_,), dim=1)
                elif ('PDE_N9_B' in config.graph_model.signal_model_name):
                    in_features = torch.cat((rr[:, None] * 0, rr[:, None], embedding_, embedding_), dim=1)
                with torch.no_grad():
                    func = model.lin_edge(in_features.float())
                    if config.graph_model.lin_edge_positive:
                        func = func ** 2
                plt.plot(to_numpy(rr), to_numpy(func), 2,
                            color=cmap.color(to_numpy(type_list)[n].astype(int)),
                            linewidth=1, alpha=0.1)
            plt.xlabel('$v_j$', fontsize=48)
            plt.ylabel(r'$\mathrm{MLP_1}(\mathbf{a}_j, v_j)$', fontsize=48)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.xlim([-1,2.5])
            plt.ylim([-config.plotting.xlim[1]/10, 2.5])
            plt.text(0.05, 0.95, f"N: {n_neurons}",
                    transform=plt.gca().transAxes, fontsize=32,
                    verticalalignment='top', color=mc)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/edge_functions_{config_indices}_all.png", dpi=300)
            plt.close()


            slopes_lin_edge_list = []
            fig = plt.figure(figsize=(10, 9))
            for n in trange(n_neurons, ncols=90):
                if mu_activity[n] + 1 * sigma_activity[n] > 0:
                    rr = torch.linspace(max(mu_activity[n] - 2 * sigma_activity[n],0), mu_activity[n] + 2 * sigma_activity[n], 1000, device=device)
                    embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                    if ('PDE_N9_A' in config.graph_model.signal_model_name) | ('PDE_N9_D' in config.graph_model.signal_model_name):
                        in_features = torch.cat((rr[:, None], embedding_,), dim=1)
                    elif ('PDE_N9_B' in config.graph_model.signal_model_name):
                        in_features = torch.cat((rr[:, None] * 0, rr[:, None], embedding_, embedding_), dim=1)
                    with torch.no_grad():
                        func = model.lin_edge(in_features.float())
                        if config.graph_model.lin_edge_positive:
                            func = func ** 2
                    plt.plot(to_numpy(rr), to_numpy(func), 2,
                                color=cmap.color(to_numpy(type_list)[n].astype(int)),
                                linewidth=1, alpha=0.1)
                    # rr_numpy = to_numpy(rr[rr.shape[0]//2+1:])
                    # func_numpy = to_numpy(func[rr.shape[0]//2+1:].squeeze())
                    rr_numpy = to_numpy(rr)
                    func_numpy = to_numpy(func.squeeze())
                    try:
                        lin_fit, _ = curve_fit(linear_model, rr_numpy, func_numpy)
                        slope = lin_fit[0]
                        offset = lin_fit[1]
                    except:
                        coeffs = np.polyfit(rr_numpy, func_numpy, 1)
                        slope = coeffs[0]
                        offset = coeffs[1]
                    slopes_lin_edge_list.append(slope)
                else:
                    slopes_lin_edge_list.append(1)
            plt.xlabel('$v_j$', fontsize=48)
            plt.ylabel(r'$\mathrm{MLP_1}(\mathbf{a}_j, v_j)$', fontsize=48)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.xlim([-1,5])
            plt.ylim([-config.plotting.xlim[1]/10, config.plotting.xlim[1]*2])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/edge_functions_{config_indices}_domain.png", dpi=300)
            plt.close()


            fig = plt.figure(figsize=(10, 9))
            ax = plt.gca()
            for spine in ax.spines.values():
                spine.set_alpha(0.75)
            slopes_lin_edge_array = np.array(slopes_lin_edge_list)
            plt.scatter(np.arange(n_neurons), slopes_lin_edge_array,
                        c=cmap.color(to_numpy(type_list).astype(int)), s=2, alpha=0.5)
            plt.xlabel('neuron index', fontsize=48)
            plt.ylabel(r'$r_j$', fontsize=48)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

            plt.xticks(fontsize=18)
            plt.yticks(fontsize=24)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/edge_function_slope_{config_indices}.png", dpi=300)
            plt.close()

            # Plot 5: Phi function visualization

            if 'plots' in extended:
                fig = plt.figure(figsize=(10, 9))
                for n in trange(n_neurons, ncols=90):
                    rr = torch.linspace(config.plotting.xlim[0], config.plotting.xlim[1], 1000, device=device)
                    embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                    in_features = torch.cat((rr[:, None], embedding_, rr[:, None] * 0, torch.zeros_like(rr[:, None])), dim=1)
                    with torch.no_grad():
                        func = model.lin_phi(in_features.float())
                        plt.plot(to_numpy(rr), to_numpy(func), 2, color=cmap.color(to_numpy(type_list)[n].astype(int)), linewidth=1, alpha=0.1)
                plt.xlim([-2.5,2.5])
                plt.ylim([-100,100])
                plt.xlabel('$v_i$', fontsize=48)
                plt.ylabel(r'$\mathrm{MLP_0}(\mathbf{a}_i, v_i)$', fontsize=48)
                plt.xticks(fontsize=24)
                plt.yticks(fontsize=24)
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/phi_functions_{config_indices}_all.png", dpi=300)
                plt.close()

            func_list = []
            slopes_lin_phi_list = []
            offsets_list = []
            fig = plt.figure(figsize=(10, 9))
            for n in trange(n_neurons, ncols=90):
                rr = torch.linspace(mu_activity[n] - 2 * sigma_activity[n], mu_activity[n] + 2 * sigma_activity[n], 1000, device=device)
                embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                in_features = torch.cat((rr[:, None], embedding_, rr[:, None] * 0, torch.zeros_like(rr[:, None])), dim=1)
                with torch.no_grad():
                    func = model.lin_phi(in_features.float())
                plt.plot(to_numpy(rr), to_numpy(func), 2,
                        color=cmap.color(to_numpy(type_list)[n].astype(int)),
                        linewidth=1, alpha=0.1)
                func_list.append(func)
                rr_numpy = to_numpy(rr)
                func_numpy = to_numpy(func.squeeze())
                try:
                    lin_fit, _ = curve_fit(linear_model, rr_numpy, func_numpy)
                    slope = lin_fit[0]
                    offset = lin_fit[1]
                except:
                    coeffs = np.polyfit(rr_numpy, func_numpy, 1)
                    slope = coeffs[0]
                    offset = coeffs[1]
                slopes_lin_phi_list.append(slope)
                offsets_list.append(offset)
            plt.xlim(config.plotting.xlim)
            plt.ylim(config.plotting.ylim)
            plt.xlabel('$v_i$', fontsize=48)
            plt.ylabel(r'$\mathrm{MLP_0}(\mathbf{a}_i, v_i)$', fontsize=48)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.text(0.05, 0.95, f"N: {n_neurons}",
                transform=plt.gca().transAxes, fontsize=32,
                verticalalignment='top', color=mc)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/phi_functions_{config_indices}_domain.png", dpi=300)
            plt.close()

            slopes_lin_phi_array = np.array(slopes_lin_phi_list)
            offsets_array = np.array(offsets_list)
            gt_taus = to_numpy(gt_taus[:n_neurons])
            learned_tau = np.where(slopes_lin_phi_array != 0, 1.0 / -slopes_lin_phi_array, 1)
            learned_tau = learned_tau[:n_neurons]
            learned_tau = np.clip(learned_tau, 0, 1)

            fig = plt.figure(figsize=(10, 9))
            plt.scatter(gt_taus, learned_tau, c=mc, s=1, alpha=0.3)
            lin_fit, lin_fitv = curve_fit(linear_model, gt_taus, learned_tau)
            residuals = learned_tau - linear_model(gt_taus, *lin_fit)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((learned_tau - np.mean(learned_tau)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            plt.text(0.05, 0.95, f'R²: {r_squared:.2f}\nslope: {lin_fit[0]:.2f}\nN: {n_edges}',
                     transform=plt.gca().transAxes, verticalalignment='top', fontsize=32)
            plt.xlabel(r'true $\tau$', fontsize=48)
            plt.ylabel(r'learned $\widehat{\tau}$', fontsize=48)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.xlim([0, 0.35])
            plt.ylim([0, 0.35])
            plt.tight_layout()
            plt.savefig(f'{log_dir}/results/tau_comparison_{config_indices}.png', dpi=300)
            plt.close()

            print(f"tau reconstruction R²: \033[92m{r_squared:.3f}\033[0m  slope: {lin_fit[0]:.2f}")
            logger.info(f"tau reconstruction R²: {r_squared:.3f}  slope: {lin_fit[0]:.2f}")
            torch.save(torch.tensor(learned_tau, dtype=torch.float32, device=device), f'{log_dir}/results/tau.pt')

            # V_rest comparison (reconstructed vs ground truth)
            learned_V_rest = np.where(slopes_lin_phi_array != 0, -offsets_array / slopes_lin_phi_array, 1)
            learned_V_rest = learned_V_rest[:n_neurons]
            gt_V_rest = to_numpy(gt_V_Rest[:n_neurons])
            fig = plt.figure(figsize=(10, 9))
            plt.scatter(gt_V_rest, learned_V_rest, c=mc, s=1, alpha=0.3)
            lin_fit, lin_fitv = curve_fit(linear_model, gt_V_rest, learned_V_rest)
            residuals = learned_V_rest - linear_model(gt_V_rest, *lin_fit)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((learned_V_rest - np.mean(learned_V_rest)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            plt.text(0.05, 0.95, f'R²: {r_squared:.2f}\nslope: {lin_fit[0]:.2f}\nN: {n_edges}',
                     transform=plt.gca().transAxes, verticalalignment='top', fontsize=32)
            plt.xlabel(r'true $V_{rest}$', fontsize=48)
            plt.ylabel(r'learned $\widehat{V}_{rest}$', fontsize=48)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.xlim([0, 0.8])
            plt.ylim([0, 0.8])
            plt.tight_layout()
            plt.savefig(f'{log_dir}/results/V_rest_comparison_{config_indices}.png', dpi=300)
            plt.close()

            print(f"V_rest reconstruction R²: \033[92m{r_squared:.3f}\033[0m  slope: {lin_fit[0]:.2f}")
            logger.info(f"V_rest reconstruction R²: {r_squared:.3f}  slope: {lin_fit[0]:.2f}")

            torch.save(torch.tensor(learned_V_rest, dtype=torch.float32, device=device), f'{log_dir}/results/V_rest.pt')

            fig = plt.figure(figsize=(10, 9))
            ax = plt.subplot(2, 1, 1)
            plt.scatter(np.arange(n_neurons), learned_tau,
                        c=cmap.color(to_numpy(type_list).astype(int)), s=2, alpha=0.5)
            plt.ylabel(r'$\widehat{\tau}_i$', fontsize=48)
            plt.xticks([])   # no xticks for top plot
            plt.yticks(fontsize=24)
            ax = plt.subplot(2, 1, 2)
            plt.scatter(np.arange(n_neurons), learned_V_rest,
                        c=cmap.color(to_numpy(type_list).astype(int)), s=2, alpha=0.5)
            plt.xlabel('neuron index', fontsize=48)
            plt.ylabel(r'$\widehat{V}^{\mathrm{rest}}_i$', fontsize=48)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=24)
            plt.ylim([0, 1])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/phi_functions_{config_indices}_params.png", dpi=300)
            plt.close()

            for target_type_name in target_type_name_list:  # Change this to any desired type name
                target_type_index = None
                for idx, name in index_to_name.items():
                    if name == target_type_name:
                        target_type_index = idx
                        break
                if target_type_index is not None:
                    type_mask = (to_numpy(type_list).squeeze() == target_type_index)
                    neurons_of_type = np.where(type_mask)[0]
                    fig = plt.figure(figsize=(10, 9))
                    for n in neurons_of_type:
                        with torch.no_grad():
                            rr = torch.linspace(config.plotting.xlim[0], config.plotting.xlim[1], 1000, device=device)
                            in_features = torch.cat((rr[:, None], embedding_, rr[:, None] * 0, torch.zeros_like(rr[:, None])), dim=1)
                            func = model.lin_phi(in_features.float())
                            plt.plot(to_numpy(rr), to_numpy(func), 2,
                                        color=cmap.color(to_numpy(type_list)[n].astype(int)),
                                        linewidth=1, alpha=0.01)
                            rr = torch.linspace(mu_activity[n] - 2 * sigma_activity[n], mu_activity[n] + 2 * sigma_activity[n], 1000, device=device)
                            embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                            in_features = torch.cat((rr[:, None], embedding_, rr[:, None] * 0, torch.zeros_like(rr[:, None])), dim=1)
                            func = model.lin_phi(in_features.float())
                            plt.plot(to_numpy(rr), to_numpy(func), 2,
                                    color=cmap.color(to_numpy(type_list)[n].astype(int)),
                                    linewidth=1, alpha=0.2)
                    # plt.xlim([to_numpy(mu_activity[n] - 3 * sigma_activity[n]), to_numpy(mu_activity[n] + 3 * sigma_activity[n])])
                    plt.xlim(config.plotting.xlim)
                    plt.ylim(config.plotting.ylim)
                    plt.xlabel('$v_i$', fontsize=48)
                    plt.ylabel(r'$\mathrm{MLP_0}(\mathbf{a}_i, v_i)$', fontsize=48)

                    # Calculate mean tau and V_rest values for neurons of this type
                    type_gt_taus = gt_taus[neurons_of_type]
                    type_learned_tau = learned_tau[neurons_of_type]
                    type_gt_V_rest = gt_V_rest[neurons_of_type]
                    type_learned_V_rest = learned_V_rest[neurons_of_type]

                    mean_gt_tau = np.mean(type_gt_taus)
                    mean_learned_tau = np.mean(type_learned_tau)
                    mean_gt_V_rest = np.mean(type_gt_V_rest)
                    mean_learned_V_rest = np.mean(type_learned_V_rest)

                    # Plot true function in red: y = -x/tau + V_rest/tau
                    if len(neurons_of_type) > 0:
                        # Use a representative neuron for the activity range
                        n_rep = neurons_of_type[0]
                        x_min = to_numpy(mu_activity[n_rep] - 25 * sigma_activity[n_rep])
                        x_max = to_numpy(mu_activity[n_rep] + 25 * sigma_activity[n_rep])
                        x_true = np.linspace(x_min, x_max, 1000)
                        y_true = -x_true / mean_gt_tau + mean_gt_V_rest / mean_gt_tau
                        plt.plot(x_true, y_true, 'r-', linewidth=1, alpha=0.5, label='true function')

                    # Add text with tau and V_rest information
                    text_info = f'$\\tau$ (true): {mean_gt_tau:.3f}\n$\\tau$ (learned): {mean_learned_tau:.3f}\n$V_{{rest}}$ (true): {mean_gt_V_rest:.3f}\n$V_{{rest}}$ (learned): {mean_learned_V_rest:.3f}'
                    plt.text(0.02, 0.98, text_info, transform=plt.gca().transAxes,
                            verticalalignment='top', horizontalalignment='left', fontsize=14)

                    plt.title(f'{target_type_name} id:{target_type_index}  ({len(neurons_of_type)} traces)',fontsize=48)
                    plt.xticks(fontsize=24)
                    plt.yticks(fontsize=24)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/phi_functions_{target_type_name}.png", dpi=300)
                    plt.close()


            # Plot 4: Weight comparison using model.W and gt_weights
            fig = plt.figure(figsize=(10, 9))
            learned_weights = to_numpy(model.W.squeeze())
            true_weights = to_numpy(gt_weights)
            plt.scatter(true_weights, learned_weights, c=mc, s=0.1, alpha=0.1)
            lin_fit, lin_fitv = curve_fit(linear_model, true_weights, learned_weights)
            residuals = learned_weights - linear_model(true_weights, *lin_fit)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((learned_weights - np.mean(learned_weights)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            plt.text(0.05, 0.95, f'R²: {r_squared:.3f}\nslope: {lin_fit[0]:.2f}',
                     transform=plt.gca().transAxes, verticalalignment='top', fontsize=24)
            plt.xlabel(r'true $\mathbf{W}_{ij}$', fontsize=48)
            plt.ylabel(r'learned $\widehat{\mathbf{W}}_{ij}$', fontsize=48)
            plt.xticks(fontsize = 24)
            plt.yticks(fontsize = 24)
            plt.tight_layout()
            plt.savefig(f'{log_dir}/results/comparison_raw.png', dpi=300)
            plt.close()
            print(f"first weights fit R²: {r_squared:.2f}  slope: {np.round(lin_fit[0], 4)}")
            logger.info(f"first weights fit R²: {r_squared:.2f}  slope: {np.round(lin_fit[0], 4)}")

            # k_list = [1]

            k_list = np.linspace(n_frames // 10, n_frames-100, 8, dtype=int).tolist()

            dataset_batch = []
            ids_batch = []
            mask_batch = []
            ids_index = 0
            mask_index = 0

            for batch in range(len(k_list)):

                k = k_list[batch]
                x = torch.tensor(x_list[0][k], dtype=torch.float32, device=device)
                ids = np.arange(n_neurons)

                if not (torch.isnan(x).any()):

                    mask = torch.arange(edges.shape[1])

                    y = torch.tensor(y_list[run][k], device=device) / ynorm

                    if not (torch.isnan(y).any()):

                        dataset = data.Data(x=x, edge_index=edges)
                        dataset_batch.append(dataset)

                        if len(dataset_batch) == 1:
                            data_id = torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * run
                            x_batch = x[:, 3:4]
                            y_batch = y
                            ids_batch = ids
                            mask_batch = mask
                        else:
                            data_id = torch.cat(
                                (data_id, torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * run), dim=0)
                            x_batch = torch.cat((x_batch, x[:, 4:5]), dim=0)
                            y_batch = torch.cat((y_batch, y), dim=0)
                            ids_batch = np.concatenate((ids_batch, ids + ids_index), axis=0)
                            mask_batch = torch.cat((mask_batch, mask + mask_index), dim=0)

                        ids_index += x.shape[0]
                        mask_index += edges.shape[1]

            with torch.no_grad():
                batch_loader = DataLoader(dataset_batch, batch_size=len(k_list), shuffle=False)
                for batch in batch_loader:
                    pred, in_features, msg = model(batch, data_id=data_id, mask=mask_batch, return_all=True)

            v = in_features[:, 0:1].clone().detach()
            embedding = in_features[:, 1:3].clone().detach()
            msg = in_features[:, 3:4].clone().detach()
            excitation = in_features[:, 4:5].clone().detach()

            msg.requires_grad_(True)
            # Concatenate input features for the final layer
            in_features = torch.cat([v, embedding, msg, excitation], dim=1)
            out = model.lin_phi(in_features)

            grad_msg = torch.autograd.grad(
                outputs=out,
                inputs=msg,
                grad_outputs=torch.ones_like(out),
                retain_graph=True,
                create_graph=True  # optional, only if you want to compute higher-order grads later
            )[0]

            # print (f'grad_msg shape: {grad_msg.shape}')
            # print (f'model.W: {model.W.shape}')

            # plt.figure(figsize=(12, 6))
            # plt.plot(to_numpy(grad_msg[0:n_neurons]), c=mc, linewidth=1)
            # plt.xlabel('neuron index')
            # plt.ylabel('gradient')
            # plt.tight_layout()
            # plt.savefig(f'{log_dir}/results/msg_gradients_{epoch}.png', dpi=300)
            # plt.close()

            plt.figure(figsize=(12, 6))

            n_batches = grad_msg.shape[0] // n_neurons
            grad_values = grad_msg.view(n_batches, n_neurons)
            grad_values = grad_values.median(dim=0).values
            grad_values = to_numpy(grad_values).squeeze()

            # grad_values = to_numpy(grad_msg[0:n_neurons]).squeeze()

            # Flatten to 1D
            neuron_indices = np.arange(n_neurons)
            # Create scatter plot colored by neuron type
            for n in range(n_types):
                type_mask = (to_numpy(type_list).squeeze() == n)  # Flatten to 1D
                if np.any(type_mask):
                    plt.scatter(neuron_indices[type_mask], grad_values[type_mask],
                                c=colors_65[n], s=1, alpha=0.8)

                    # Add text label for each neuron type
                    if np.sum(type_mask) > 0:
                        mean_x = np.mean(neuron_indices[type_mask])
                        mean_y = np.mean(grad_values[type_mask])
                        plt.text(mean_x, mean_y, index_to_name.get(n, f'T{n}'),
                                 fontsize=6, ha='center', va='center')
            plt.xlabel('neuron index')
            plt.ylabel('gradient')
            plt.tight_layout()
            # plt.savefig(f'{log_dir}/results/msg_gradients_{epoch}.png', dpi=300)
            plt.close()

            grad_msg_flat = grad_msg.squeeze()
            assert grad_msg_flat.shape[0] == n_neurons * len(k_list), "Gradient and neuron count mismatch"
            target_neuron_ids = edges[1, :] % (model.n_edges + model.n_extra_null_edges)
            grad_msg_per_edge = grad_msg_flat[target_neuron_ids]
            grad_msg_per_edge = grad_msg_per_edge.unsqueeze(1)  # [434112, 1]

            slopes_lin_phi_array = torch.tensor(slopes_lin_phi_array, dtype=torch.float32, device=device)
            slopes_lin_phi_per_edge = slopes_lin_phi_array[target_neuron_ids]

            slopes_lin_edge_array = np.array(slopes_lin_edge_list)
            slopes_lin_edge_array = torch.tensor(slopes_lin_edge_array, dtype=torch.float32, device=device)
            prior_neuron_ids = edges[0, :] % (model.n_edges + model.n_extra_null_edges)  # j
            slopes_lin_edge_per_edge = slopes_lin_edge_array[prior_neuron_ids]

            corrected_W_ = -model.W / slopes_lin_phi_per_edge[:, None] * grad_msg_per_edge
            corrected_W = -model.W / slopes_lin_phi_per_edge[:, None] * grad_msg_per_edge * slopes_lin_edge_per_edge.unsqueeze(1)
            torch.save(corrected_W, f'{log_dir}/results/corrected_W.pt')

            learned_weights = to_numpy(corrected_W.squeeze())
            true_weights = to_numpy(gt_weights)

            # --- Outlier removal: drop weights beyond 3*MAD ---
            residuals = learned_weights - true_weights
            # mad = np.median(np.abs(residuals - np.median(residuals))) + 1e-12
            # z = 0.6745 * (residuals - np.median(residuals)) / mad
            # mask = np.abs(z) <= 10  # keep only inliers
            mask = np.abs(residuals) <= 5  # keep only inliers

            true_in = true_weights[mask]
            learned_in = learned_weights[mask]

            if extended:

                learned_in_ = to_numpy(corrected_W_.squeeze())
                learned_in_ = learned_in_[mask]

                fig = plt.figure(figsize=(10, 9))
                plt.scatter(true_in, learned_in_, c=mc, s=0.1, alpha=0.1)
                lin_fit, _ = curve_fit(linear_model, true_in, learned_in_)
                residuals_ = learned_in_ - linear_model(true_in, *lin_fit)
                ss_res = np.sum(residuals_ ** 2)
                ss_tot = np.sum((learned_in_ - np.mean(learned_in_)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                plt.text(0.05, 0.95,
                        f'R²: {r_squared:.3f}\nslope: {lin_fit[0]:.2f}',
                        transform=plt.gca().transAxes, verticalalignment='top', fontsize=24)
                plt.xlabel(r'true $\mathbf{W}_{ij}$', fontsize=48)
                plt.ylabel(r'learned $\widehat{\mathbf{W}}_{ij}r_j$', fontsize=48)
                plt.xticks(fontsize = 24)
                plt.yticks(fontsize = 24)
                plt.tight_layout()
                plt.savefig(f'{log_dir}/results/comparison_rj.png', dpi=300)
                plt.close()


            fig = plt.figure(figsize=(10, 9))
            plt.scatter(true_in, learned_in, c=mc, s=0.5, alpha=0.1)
            lin_fit, _ = curve_fit(linear_model, true_in, learned_in)
            residuals_ = learned_in - linear_model(true_in, *lin_fit)
            ss_res = np.sum(residuals_ ** 2)
            ss_tot = np.sum((learned_in - np.mean(learned_in)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            plt.text(0.05, 0.95,
                     f'R²: {r_squared:.2f}\nslope: {lin_fit[0]:.2f}\nN: {n_edges}',
                     transform=plt.gca().transAxes, verticalalignment='top', fontsize=32)
            plt.xlabel(r'true $\mathbf{W}_{ij}$', fontsize=48)
            # plt.ylabel(r'learned -$\widehat{\mathbf{W}}_{ij} \, g_i r_j \, \widehat{\tau}_i$', fontsize=48)
            plt.ylabel(r'learned $\widehat{\mathbf{W}}_{ij}^*$', fontsize=48)
            plt.xticks(fontsize = 24)
            plt.yticks(fontsize = 24)
            plt.xlim([-1,2])
            plt.ylim([-1,2])
            plt.tight_layout()
            plt.savefig(f'{log_dir}/results/corrected_comparison.png', dpi=300)
            plt.close()

            print(f"second weights fit R²: \033[92m{r_squared:.4f}\033[0m  slope: {np.round(lin_fit[0], 4)}")
            logger.info(f"second weights fit R²: {r_squared:.4f}  slope: {np.round(lin_fit[0], 4)}")
            print(f'median residuals: {np.median(residuals):.4f}')
            inlier_residuals = residuals[mask]
            print(f'inliers: {len(inlier_residuals)}  mean residual: {np.mean(inlier_residuals):.4f}  std: {np.std(inlier_residuals):.4f}  min,max: {np.min(inlier_residuals):.4f}, {np.max(inlier_residuals):.4f}')
            outlier_residuals = residuals[~mask]
            if len(outlier_residuals) > 0:
                print(
                    f'outliers: {len(outlier_residuals)}  mean residual: {np.mean(outlier_residuals):.4f}  std: {np.std(outlier_residuals):.4f}  min,max: {np.min(outlier_residuals):.4f}, {np.max(outlier_residuals):.4f}')
            else:
                print('outliers: 0  (no outliers detected)')






            # plot analyze_neuron_type_reconstruction
            results_per_neuron = analyze_neuron_type_reconstruction(
                config=config,
                model=model,
                edges=to_numpy(edges),
                true_weights=true_weights,  #  ground truth weights
                gt_taus=gt_taus,  #  ground truth tau values
                gt_V_Rest=gt_V_rest,  #  ground truth V_rest values
                learned_weights=learned_weights,
                learned_tau = learned_tau,
                learned_V_rest=learned_V_rest, # Learned V_rest
                type_list=to_numpy(type_list),
                n_frames=n_frames,
                dimension=dimension,
                n_neuron_types=n_neuron_types,
                device=device,
                log_dir=log_dir,
                dataset_name=dataset_name,
                logger=logger,
                index_to_name=index_to_name
            )

            plot_reconstruction_correlations(activity_results=activity_results, results_per_neuron=results_per_neuron, gt_taus=gt_taus, gt_V_Rest=gt_V_Rest, type_list=type_list, index_to_name=index_to_name, log_dir=log_dir)

            print('alternative clustering methods...')






            # compute connectivity statistics for true weights
            print('computing connectivity statistics...')
            w = true_weights.flatten()
            w_in_mean_true = np.zeros(n_neurons)
            w_in_std_true = np.zeros(n_neurons)
            w_out_mean_true = np.zeros(n_neurons)
            w_out_std_true = np.zeros(n_neurons)
            edges_np = to_numpy(edges)

            for i in trange(n_neurons, ncols=90):
                in_w = w[edges_np[1] == i]
                out_w = w[edges_np[0] == i]
                w_in_mean_true[i] = in_w.mean() if len(in_w) > 0 else 0
                w_in_std_true[i] = in_w.std() if len(in_w) > 0 else 0
                w_out_mean_true[i] = out_w.mean() if len(out_w) > 0 else 0
                w_out_std_true[i] = out_w.std() if len(out_w) > 0 else 0

            # compute connectivity statistics for learned weights
            w = learned_weights.flatten()
            w_in_mean_learned = np.zeros(n_neurons)
            w_in_std_learned = np.zeros(n_neurons)
            w_out_mean_learned = np.zeros(n_neurons)
            w_out_std_learned = np.zeros(n_neurons)

            for i in trange(n_neurons, ncols=90):
                in_w = w[edges_np[1] == i]
                out_w = w[edges_np[0] == i]
                w_in_mean_learned[i] = in_w.mean() if len(in_w) > 0 else 0
                w_in_std_learned[i] = in_w.std() if len(in_w) > 0 else 0
                w_out_mean_learned[i] = out_w.mean() if len(out_w) > 0 else 0
                w_out_std_learned[i] = out_w.std() if len(out_w) > 0 else 0

            # all 4 connectivity stats combined
            W_learned = np.column_stack([w_in_mean_learned, w_in_std_learned,
                                        w_out_mean_learned, w_out_std_learned])
            W_true = np.column_stack([w_in_mean_true, w_in_std_true,
                                    w_out_mean_true, w_out_std_true])

            # learned combinations
            learned_combos = {
                'a': to_numpy(model.a),
                'τ': learned_tau.reshape(-1, 1),
                'V': learned_V_rest.reshape(-1, 1),
                'W': W_learned,
                '(τ,V)': np.column_stack([learned_tau, learned_V_rest]),
                '(τ,V,W)': np.column_stack([learned_tau, learned_V_rest, W_learned]),
                '(a,τ,V,W)': np.column_stack([to_numpy(model.a), learned_tau, learned_V_rest, W_learned]),
            }

            # true combinations
            true_combos = {
                'τ': gt_taus.reshape(-1, 1),
                'V': gt_V_rest.reshape(-1, 1),
                'W': W_true,
                '(τ,V)': np.column_stack([gt_taus, gt_V_rest]),
                '(τ,V,W)': np.column_stack([gt_taus, gt_V_rest, W_true]),
            }

            # cluster learned
            print('\nclustering learned features...')
            learned_results = {}
            for name, feat_array in learned_combos.items():
                result = clustering_gmm(feat_array, type_list, n_components=75)
                learned_results[name] = result['accuracy']
                print(f"{name}: {result['accuracy']:.3f}")

            # Cluster true
            print('\nclustering true features...')
            true_results = {}
            for name, feat_array in true_combos.items():
                result = clustering_gmm(feat_array, type_list, n_components=75)
                true_results[name] = result['accuracy']
                print(f"{name}: {result['accuracy']:.3f}")

            # Plot two-panel figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            # Learned features - fixed order
            learned_order = ['a', 'τ', 'V', 'W', '(τ,V)', '(τ,V,W)', '(a,τ,V,W)']
            learned_vals = [learned_results[k] for k in ['a', 'τ', 'V', 'W', '(τ,V)', '(τ,V,W)', '(a,τ,V,W)']]
            colors_l = ['#d62728' if v < 0.6 else '#ff7f0e' if v < 0.85 else '#2ca02c' for v in learned_vals]
            ax1.barh(range(len(learned_order)), learned_vals, color=colors_l)
            ax1.set_yticks(range(len(learned_order)))
            ax1.set_yticklabels(learned_order, fontsize=11)
            ax1.set_xlabel('clustering accuracy', fontsize=12)
            ax1.set_title('learned features', fontsize=14, fontweight='bold')
            ax1.set_xlim([0, 1])
            ax1.grid(axis='x', alpha=0.3)
            ax1.invert_yaxis()
            for i, v in enumerate(learned_vals):
                ax1.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=10)
            # True features - fixed order
            true_order = ['τ', 'V', 'W', '(τ,V)', '(τ,V,W)']
            true_vals = [true_results[k] for k in ['τ', 'V', 'W', '(τ,V)', '(τ,V,W)']]
            colors_t = ['#d62728' if v < 0.6 else '#ff7f0e' if v < 0.85 else '#2ca02c' for v in true_vals]
            ax2.barh(range(len(true_order)), true_vals, color=colors_t)
            ax2.set_yticks(range(len(true_order)))
            ax2.set_yticklabels(true_order, fontsize=11)
            ax2.set_xlabel('clustering accuracy', fontsize=12)
            ax2.set_title('true features', fontsize=14, fontweight='bold')
            ax2.set_xlim([0, 1])
            ax2.grid(axis='x', alpha=0.3)
            ax2.invert_yaxis()
            for i, v in enumerate(true_vals):
                ax2.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=10)
            plt.tight_layout()
            plt.savefig(f'{log_dir}/results/clustering_comprehensive.png', dpi=300, bbox_inches='tight')
            plt.close()

            a_aug = np.column_stack([to_numpy(model.a), learned_tau, learned_V_rest,
                                    w_in_mean_learned, w_in_std_learned, w_out_mean_learned, w_out_std_learned])
            print('\nGMM learned a tau V_rest weights W:')

            best_acc = 0
            best_n = 0
            for n_comp in [50, 75, 100, 125, 150]:
                results = clustering_gmm(a_aug, type_list, n_components=n_comp)
                print(f"n_components={n_comp}: accuracy=\033[32m{results['accuracy']:.3f}\033[0m, ARI={results['ari']:.3f}, NMI={results['nmi']:.3f}")
                if results['accuracy'] > best_acc:
                    best_acc = results['accuracy']
                    best_n = n_comp

            print(f"best: n_components={best_n}, accuracy=\033[92m{best_acc:.3f}\033[0m")
            logger.info(f"GMM best: n_components={best_n}, accuracy={best_acc:.3f}")

            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            a_umap = reducer.fit_transform(a_aug)

            # Get cluster labels from best GMM
            results = clustering_gmm(a_aug, type_list, n_components=best_n)
            cluster_labels = GaussianMixture(n_components=best_n, random_state=42).fit_predict(a_aug)

            plt.figure(figsize=(10, 9))
            ax = plt.gca()
            for spine in ax.spines.values():
                spine.set_alpha(0.75)
            # for n in range(n_types):
            #     pos = to_numpy(torch.argwhere(type_list == n).squeeze())
            #     plt.scatter(a_umap[pos, 0], a_umap[pos, 1], s=24, color=colors_65[n], alpha=0.8, edgecolors='none')
            from matplotlib.colors import ListedColormap
            cmap_65 = ListedColormap(colors_65)
            plt.scatter(a_umap[:, 0], a_umap[:, 1], c=cluster_labels, s=24, cmap=cmap_65, alpha=0.8, edgecolors='none')


            # Add cluster centroids
            for c in range(best_n):
                mask = cluster_labels == c
                if mask.sum() > 0:
                    cx, cy = np.median(a_umap[mask, 0]), np.median(a_umap[mask, 1])
                    plt.text(cx, cy, str(c), fontsize=8, ha='center', va='center')

            plt.xlabel(r'UMAP$_1$', fontsize=48)
            plt.ylabel(r'UMAP$_2$', fontsize=48)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.text(0.05, 0.95, f"N: {n_neurons}\naccuracy: {best_acc:.2f}",
                    transform=plt.gca().transAxes, fontsize=32, verticalalignment='top')
            plt.tight_layout()
            plt.savefig(f'{log_dir}/results/embedding_augmented_{config_indices}.png', dpi=300)
            plt.close()

            # Spectral clustering
            # print('spectral:')
            # for n_clust in [50, 75, 100, 125, 150]:
            #     results = clustering_spectral(a_aug, type_list, n_clusters=n_clust)
            #     print(f"  n_clusters={n_clust}: accuracy=\033[32m{results['accuracy']:.3f}\033[0m, ARI={results['ari']:.3f}, NMI={results['nmi']:.3f}")



def plot_synaptic_flyvis_calcium(config, epoch_list, log_dir, logger, cc, style, extended, device):
    dataset_name = config.dataset
    config.dataset.split('fly_N9_')[1] if 'fly_N9_' in config.dataset else 'evolution'

    n_frames = config.simulation.n_frames
    n_runs = config.training.n_runs
    n_neurons = config.simulation.n_neurons

    colors_65 = sns.color_palette("Set3", 12) * 6  # pastel, repeat until 65
    colors_65 = colors_65[:65]

    config.simulation.max_radius if hasattr(config.simulation, 'max_radius') else 2.5
    dimension = config.simulation.dimension

    log_file = os.path.join(log_dir, 'results.log')
    if os.path.exists(log_file):
        os.remove(log_file)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create file handler only, no console output
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Clear any existing handlers

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logger.addHandler(file_handler)

    # Prevent propagation to root logger (which might have console handlers)
    logger.propagate = False

    print(f'experiment description: {config.description}')
    logger.info(f'experiment description: {config.description}')

    # Load neuron group mapping for flyvis

    CustomColorMap(config=config)

    if 'black' in style:
        plt.style.use('dark_background')
        mc = 'w'
    else:
        plt.style.use('default')
        mc = 'k'

    x_list = []
    y_list = []
    time.sleep(0.5)
    print('load simulation data...')
    for run in range(0, n_runs):
        if os.path.exists(f'graphs_data/{dataset_name}/x_list_{run}.pt'):
            x = torch.load(f'graphs_data/{dataset_name}/x_list_{run}.pt', map_location=device)
            y = torch.load(f'graphs_data/{dataset_name}/y_list_{run}.pt', map_location=device)
            x = to_numpy(torch.stack(x))
            y = to_numpy(torch.stack(y))
        else:
            x = np.load(f'graphs_data/{dataset_name}/x_list_{run}.npy')
            y = np.load(f'graphs_data/{dataset_name}/y_list_{run}.npy')
        x_list.append(x)
        y_list.append(y)

    ynorm = torch.load(os.path.join(log_dir, 'ynorm.pt'), map_location=device)
    if os.path.exists(os.path.join(log_dir, 'xnorm.pt')):
        xnorm = torch.load(os.path.join(log_dir, 'xnorm.pt'))
    else:
        xnorm = torch.tensor([5], device=device)

    print(f'xnorm: {to_numpy(xnorm)}, ynorm: {to_numpy(ynorm)}')
    logger.info(f'xnorm: {to_numpy(xnorm)}, ynorm: {to_numpy(ynorm)}')

    # Load data with new format
    # connectivity = torch.load(f'./graphs_data/{dataset_name}/connectivity.pt', map_location=device)
    gt_weights = torch.load(f'./graphs_data/{dataset_name}/weights.pt', map_location=device)
    torch.load(f'./graphs_data/{dataset_name}/taus.pt', map_location=device)
    torch.load(f'./graphs_data/{dataset_name}/V_i_rest.pt', map_location=device)
    edges = torch.load(f'./graphs_data/{dataset_name}/edge_index.pt', map_location=device)
    true_weights = torch.zeros((n_neurons, n_neurons), dtype=torch.float32, device=edges.device)
    true_weights[edges[1], edges[0]] = gt_weights

    x = x_list[0][n_frames - 10]
    type_list = torch.tensor(x[:, 2 + 2 * dimension:3 + 2 * dimension], device=device)
    len(np.unique(to_numpy(type_list)))
    region_list = torch.tensor(x[:, 1 + 2 * dimension:2 + 2 * dimension], device=device)
    len(np.unique(to_numpy(region_list)))
    n_neurons = len(type_list)

    # Neuron type index to name mapping

    activity = torch.tensor(x_list[0][:, :, 3:4], device=device)
    activity = activity.squeeze().t()
    torch.mean(activity, dim=1)
    torch.std(activity, dim=1)

    os.makedirs(f'{log_dir}/results/', exist_ok=True)


    if epoch_list[0] == 'all':

        print ('not implemented yet ...')

    else:
        config.dataset.split('fly_N9_')[1]
        files, file_id_list = get_training_files(log_dir, n_runs)

        for epoch in epoch_list:

            net = f'{log_dir}/models/best_model_with_{n_runs - 1}_graphs_{epoch}.pt'
            model = Calcium_Latent_Dynamics(config=config, device=device)

            state_dict = torch.load(net, map_location=device)
            model.load_state_dict(state_dict['model_state_dict'])
            model.eval()
            print(f'net: {net}')
            logger.info(f'net: {net}')

            # Plot 1: Loss curve
            if os.path.exists(os.path.join(log_dir, 'loss.pt')):
                plt.figure(figsize=(8, 6))
                ax = plt.gca()
                for spine in ax.spines.values():
                    spine.set_alpha(0.75)
                list_loss = torch.load(os.path.join(log_dir, 'loss.pt'))
                plt.plot(list_loss, color=mc, linewidth=2)
                plt.xlim([0, len(list_loss)])
                plt.ylabel('Loss')
                plt.xlabel('Epochs')
                plt.title('Training Loss')
                plt.tight_layout()
                plt.savefig(f'{log_dir}/results/loss.png', dpi=300)
                plt.close()

            recons_loss_list = []
            baseline_loss_list = []
            for it in trange(0, n_frames-1, ncols=90):

                x = torch.tensor(x_list[run][it,:,7:8], dtype=torch.float32, device=device).squeeze()
                y = torch.tensor(x_list[run][it+1,:,7:8], device=device).squeeze()   # auto-encoder_loss

                with torch.no_grad():
                    pred, mu, logvar = model(x)
                recon_loss = (pred-y).norm(2)
                baseline_loss = (x-y).norm(2)

                recons_loss_list.append(to_numpy(recon_loss))
                baseline_loss_list.append(to_numpy(baseline_loss))

            recons_loss_list = np.array(recons_loss_list)
            baseline_loss_list = np.array(baseline_loss_list)

            # print mean and std
            print(f'reconstruction loss: {np.mean(recons_loss_list):.4f} +/- {np.std(recons_loss_list):.4f}')
            print(f'baseline loss: {np.mean(baseline_loss_list):.4f} +/- {np.std(baseline_loss_list):.4f}')









def plot_synaptic_zebra(config, epoch_list, log_dir, logger, cc, style, extended, device):

    dataset_name = config.dataset
    simulation_config = config.simulation
    model_config = config.graph_model
    training_config = config.training

    n_neuron_types = simulation_config.n_neuron_types
    n_neurons = simulation_config.n_neurons
    n_runs = training_config.n_runs
    n_frames = simulation_config.n_frames
    delta_t = simulation_config.delta_t
    plot_batch_size = config.plotting.plot_batch_size

    CustomColorMap(config=config)
    dimension = simulation_config.dimension


    log_file = os.path.join(log_dir, 'results.log')
    if os.path.exists(log_file):
        os.remove(log_file)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create file handler only, no console output
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Clear any existing handlers

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logger.addHandler(file_handler)

    # Prevent propagation to root logger (which might have console handlers)
    logger.propagate = False

    print(f'experiment description: {config.description}')
    logger.info(f'experiment description: {config.description}')

    if 'black' in style:
        plt.style.use('dark_background')
        mc = 'w'
    else:
        plt.style.use('default')
        mc = 'k'

    x_list = []
    y_list = []

    print('load data...')

    run = 0
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

    net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs_{epoch_list[0]}.pt"

    model = Signal_Propagation_Zebra(aggr_type=model_config.aggr_type, config=config, device=device)
    state_dict = torch.load(net, map_location=device, weights_only=True)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    print('recons...')

    generated_x_list = []


    # tail_list = np.array(to_numpy(x_list[0][:,0,11:14]))
    # print (tail_list.shape)

    ones = torch.ones((n_neurons, 1), dtype=torch.float32, device=device)
    for it in trange(0, min(n_frames,7800), 1, ncols=90):
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
            # if (it % step == 0) & (visualize == True):
            #     # plot field comparison
            #     output_path = f"./{log_dir}/results/Fig/Fig_{run}_{it_idx:06d}.png"
            #     plot_field_comparison(x, model, it, n_frames, ones, output_path, 50, plot_batch_size)
            #     it_idx += 1

    generated_x_list = np.array(generated_x_list)
    print(f"generated {len(generated_x_list)} frames total")
    print(f"saving ./{log_dir}/results/recons_field.npy")
    np.save(f"./{log_dir}/results/recons_field.npy", generated_x_list)

    reconstructed = generated_x_list
    true = to_numpy(x_list[0][:,:,6:7])
    reconstructed = reconstructed.squeeze()
    true = true.squeeze()

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

    print('plot 5x4 panel for neuron (ground truth vs approximated)...')

    fig, axes = plt.subplots(5, 4, figsize=(18, 12))
    axes_flat = axes.flatten()

    neuron_ids = np.linspace(0, n_neurons-1, 20, dtype=int).tolist()
    start_frame = 0
    end_frame = 500

    N_slices = 72
    x_list[0][0,:,3:4].min()
    x_list[0][0,:,3:4].max()
    0.28 / N_slices
    delta_t/N_slices

    for idx, neuron_id in enumerate(neuron_ids):
        if idx >= 20:
            break

        ax = axes_flat[idx]

        true_data = true[:, neuron_id]
        pred_data = reconstructed[:, neuron_id]

        y_min = min(np.min(true_data[start_frame:end_frame]), np.min(pred_data[start_frame:end_frame]))
        y_max = max(np.max(true_data[start_frame:end_frame]), np.max(pred_data[start_frame:end_frame]))
        y_range = y_max - y_min
        y_padding = y_range * 0.25  # 10% padding

        ax.plot(true_data, linewidth=2, color='green', alpha=0.4)
        ax.text(0.025, 0.95, f'neuron id: {neuron_id}\nz: {to_numpy(x_list[0][0, neuron_id, 3]):0.3f}', transform=ax.transAxes, ha='left', va='top', fontsize=12, color='green')
        if 'true_only' not in style:
            ax.plot(pred_data, linewidth=1, color=mc, alpha=1.0)
            rmse = np.sqrt(np.mean((true_data - pred_data)**2))
            ax.text(0.525, 0.95, f'RMSE: {rmse:.3f}', transform=ax.transAxes, ha='left', va='top', fontsize=12, color=mc)
            ax.set_xlim([start_frame, end_frame])
            # ax.set_ylim([y_min - y_padding, y_max + y_padding])
            ax.set_ylim([0,1.0])
        ax.set_xticks([])
        ax.set_yticks([])
        if idx == 16:  # Bottom left corner - add axis labels with much larger font
            ax.set_xlabel('frame', fontsize=22)
            ax.set_ylabel('calcium', fontsize=22)
            ax.set_xticks([start_frame, (start_frame + end_frame) / 2, end_frame])

            # ax.set_xticklabels(['start_frame', 'end_frame'], fontsize=14)
            # Use dynamic y-range for this panel too
            ax.set_yticks([y_min - y_padding, (y_min + y_max) / 2, y_max + y_padding])
            ax.set_yticklabels([f'{y_min - y_padding:.2f}', f'{(y_min + y_max) / 2:.2f}', f'{y_max + y_padding:.2f}'], fontsize=14)

    plt.tight_layout()
    plt.show()
    plt.savefig(f"./{log_dir}/results/activity_5x4_panel_comparison.png", dpi=150)
    plt.close()



    neuron_id = 60395
    up_sampled_factor = 10
    up_sampled_time_points = torch.linspace(0,end_frame,end_frame*up_sampled_factor, device=device).unsqueeze(1)
    ones = torch.ones((up_sampled_time_points.shape[0],1), dtype=torch.float32, device=device)
    x_ = x[neuron_id, 1:4] * ones
    in_features = torch.cat((x_/model.NNR_f_xy_period, up_sampled_time_points/model.NNR_f_T_period), 1)
    reconstructed_up_sampled = model.NNR_f(in_features)**2


    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], hspace=0.3)

    # Top row - main plot spanning all columns
    ax_top = fig.add_subplot(gs[0, :])
    true_data = true[:, neuron_id]
    pred_data = reconstructed[:, neuron_id]
    time_points = np.arange(len(true_data)) * delta_t
    residuals = true_data - pred_data
    rmse = np.sqrt(np.mean(residuals**2))

    ax_top.plot(time_points, true_data, linewidth=1, color='green', alpha=0.4)
    ax_top.scatter(time_points, true_data, s=2, color='green', alpha=0.4, label='ground truth')
    ax_top.scatter(time_points, pred_data, s=2, color=mc, alpha=1.0, label='reconstructed')
    ax_top.scatter(to_numpy(up_sampled_time_points), to_numpy(reconstructed_up_sampled), s=1, color=mc, alpha=0.5)
    ax_top.text(0.025, 0.95, f'neuron id: {neuron_id}\nz: {to_numpy(x_list[0][0, neuron_id, 3]):0.3f}\nrmse: {rmse:.3f}',
            transform=ax_top.transAxes, ha='left', va='top', fontsize=12,
            color='black' if mc=='w' else 'white')
    ax_top.set_xlabel('frame', fontsize=12)
    ax_top.set_ylabel('calcium', fontsize=12)
    ax_top.set_xlim([start_frame, end_frame])
    ax_top.set_ylim([0, 1.0])
    ax_top.legend()

    # Bottom left - histogram
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax1.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('residual', fontsize=12)
    ax1.set_ylabel('count', fontsize=12)
    shapiro_p = stats.shapiro(residuals)[1]
    ax1.set_title(f'μ={np.mean(residuals):.3f}, σ={np.std(residuals):.3f}, p={shapiro_p:.3f}', fontsize=10)

    # Bottom middle - Q-Q plot
    ax2 = fig.add_subplot(gs[1, 1])
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title('q-q plot', fontsize=10)
    ax2.set_xlabel('theoretical quantiles', fontsize=12)
    ax2.set_ylabel('sample quantiles', fontsize=12)

    # Bottom right - autocorrelation
    ax3 = fig.add_subplot(gs[1, 2])
    from statsmodels.tsa.stattools import acf
    lags = min(40, len(residuals)//4)
    autocorr = acf(residuals, nlags=lags)
    ax3.stem(range(len(autocorr)), autocorr, linefmt='b-', markerfmt='bo', basefmt=' ')
    ax3.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ci = 1.96/np.sqrt(len(residuals))
    ax3.axhline(ci, color='r', linestyle='--', alpha=0.5)
    ax3.axhline(-ci, color='r', linestyle='--', alpha=0.5)
    ax3.set_xlabel('lag', fontsize=12)
    ax3.set_ylabel('acf', fontsize=12)
    ax3.set_title('autocorrelation', fontsize=10)
    ax3.set_ylim([-0.3, 1.0])

    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/activity_single_neuron_comparison.png", dpi=150)
    plt.close()


    # Comprehensive residual analysis for all neurons - 5x4 grid
    fig = plt.figure(figsize=(24, 16))
    outer_grid = fig.add_gridspec(5, 4, hspace=0.4, wspace=0.3)

    neuron_ids = np.linspace(0, n_neurons-1, 20, dtype=int).tolist()
    start_frame = 0
    end_frame = 500

    for idx, neuron_id in enumerate(neuron_ids):
        if idx >= 20:
            break

        # Create inner grid for each neuron (2x3 layout)
        inner_grid = outer_grid[idx].subgridspec(2, 3, height_ratios=[2, 1], hspace=0.3)

        true_data = true[start_frame:end_frame, neuron_id]
        pred_data = reconstructed[start_frame:end_frame, neuron_id]
        time_points = np.arange(len(true_data)) * delta_t
        residuals = true_data - pred_data
        rmse = np.sqrt(np.mean(residuals**2))

        # Top row - time series spanning all columns
        ax_top = fig.add_subplot(inner_grid[0, :])
        ax_top.plot(time_points, true_data, linewidth=1, color='green', alpha=0.4, label='truth')
        ax_top.scatter(time_points, pred_data, s=1, color='black', alpha=0.8, label='pred')
        ax_top.text(0.02, 0.95, f'n={neuron_id}, z={to_numpy(x_list[0][0, neuron_id, 3]):0.2f}, rmse={rmse:.3f}',
                transform=ax_top.transAxes, ha='left', va='top', fontsize=7)
        ax_top.set_ylim([0, 1.0])
        ax_top.set_xticks([])
        ax_top.set_yticks([])

        # Bottom left - histogram
        ax_hist = fig.add_subplot(inner_grid[1, 0])
        ax_hist.hist(residuals, bins=20, edgecolor='black', alpha=0.7, density=True, color='skyblue')
        mu, sigma = np.mean(residuals), np.std(residuals)
        x_norm = np.linspace(residuals.min(), residuals.max(), 50)
        ax_hist.plot(x_norm, stats.norm.pdf(x_norm, mu, sigma), 'r-', lw=1)
        ax_hist.axvline(0, color='k', linestyle='--', alpha=0.5, lw=0.5)
        ax_hist.set_xticks([])
        ax_hist.set_yticks([])
        shapiro_p = stats.shapiro(residuals)[1] if len(residuals) > 3 else 0
        ax_hist.text(0.5, 0.95, f'σ={np.std(residuals):.3f},  p={shapiro_p:.2f}', transform=ax_hist.transAxes,
                    ha='center', va='top', fontsize=6)

        # Bottom middle - Q-Q plot
        ax_qq = fig.add_subplot(inner_grid[1, 1])
        stats.probplot(residuals, dist="norm", plot=ax_qq)
        ax_qq.get_lines()[0].set_markersize(2)
        ax_qq.get_lines()[0].set_markerfacecolor('blue')
        ax_qq.get_lines()[0].set_alpha(0.5)
        ax_qq.set_xticks([])
        ax_qq.set_yticks([])
        ax_qq.set_xlabel('')
        ax_qq.set_ylabel('')
        ax_qq.set_title('')

        # Bottom right - ACF
        ax_acf = fig.add_subplot(inner_grid[1, 2])
        from statsmodels.tsa.stattools import acf
        lags = min(30, len(residuals)//4)
        autocorr = acf(residuals, nlags=lags)
        markerline, stemlines, baseline = ax_acf.stem(range(len(autocorr)), autocorr,
                                                    linefmt='b-', markerfmt='bo', basefmt=' ')
        markerline.set_markersize(2)
        stemlines.set_linewidth(0.5)
        ax_acf.axhline(0, color='k', linestyle='-', linewidth=0.5)
        ci = 1.96/np.sqrt(len(residuals))
        ax_acf.axhline(ci, color='r', linestyle='--', alpha=0.3, lw=0.5)
        ax_acf.axhline(-ci, color='r', linestyle='--', alpha=0.3, lw=0.5)
        ax_acf.set_ylim([-0.3, 1.0])
        ax_acf.set_xticks([])
        ax_acf.set_yticks([])

    plt.savefig(f"./{log_dir}/results/residual_analysis_comprehensive.png", dpi=200, bbox_inches='tight')
    plt.show()
    plt.close()


def analyze_neuron_type_reconstruction(config, model, edges, true_weights, gt_taus, gt_V_Rest,
                                       learned_weights, learned_tau, learned_V_rest, type_list, n_frames, dimension,
                                       n_neuron_types, device, log_dir, dataset_name, logger, index_to_name):

    print('stratified analysis by neuron type...')

    colors_65 = sns.color_palette("Set3", 12) * 6  # pastel, repeat until 65
    colors_65 = colors_65[:65]

    rmse_weights = []
    rmse_taus = []
    rmse_vrests = []
    n_connections = []

    for neuron_type in range(n_neuron_types):

        type_indices = np.where(type_list[edges[1,:]] == neuron_type)[0]
        gt_w_type = true_weights[type_indices]
        learned_w_type = learned_weights[type_indices]
        n_conn = len(type_indices)

        type_indices = np.where(type_list == neuron_type)[0]
        gt_tau_type = gt_taus[type_indices]
        gt_vrest_type = gt_V_Rest[type_indices]

        learned_tau_type = learned_tau[type_indices]
        learned_vrest_type = learned_V_rest[type_indices]

        rmse_w = np.sqrt(np.mean((gt_w_type - learned_w_type)** 2))
        rmse_tau = np.sqrt(np.mean((gt_tau_type - learned_tau_type)** 2))
        rmse_vrest = np.sqrt(np.mean((gt_vrest_type - learned_vrest_type)** 2))

        rmse_weights.append(rmse_w)
        rmse_taus.append(rmse_tau)
        rmse_vrests.append(rmse_vrest)
        n_connections.append(n_conn)

    n_neurons = len(type_list)

    # Per-neuron RMSE for tau
    rmse_tau_per_neuron = np.abs(learned_tau - gt_taus)
    # Per-neuron RMSE for V_rest
    rmse_vrest_per_neuron = np.abs(learned_V_rest - gt_V_Rest)
    # Per-neuron RMSE for weights (incoming connections)
    rmse_weights_per_neuron = np.zeros(n_neurons)
    for neuron_idx in range(n_neurons):
        incoming_edges = np.where(edges[1, :] == neuron_idx)[0]
        if len(incoming_edges) > 0:
            true_w = true_weights[incoming_edges]
            learned_w = learned_weights[incoming_edges]
            rmse_weights_per_neuron[neuron_idx] = np.sqrt(np.mean((learned_w - true_w)**2))

    # Convert to arrays
    rmse_weights = np.array(rmse_weights)
    rmse_taus = np.array(rmse_taus)
    rmse_vrests = np.array(rmse_vrests)

    unique_types_in_order = []
    seen_types = set()
    for i in range(len(type_list)):
        neuron_type_id = type_list[i].item() if hasattr(type_list[i], 'item') else int(type_list[i])
        if neuron_type_id not in seen_types:
            unique_types_in_order.append(neuron_type_id)
            seen_types.add(neuron_type_id)

    # Create neuron type names in the same order as they appear in data
    sorted_neuron_type_names = [index_to_name.get(type_id, f'Type{type_id}') for type_id in unique_types_in_order]
    unique_types_in_order = np.array(unique_types_in_order)
    sort_indices = unique_types_in_order.astype(int)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    x_pos = np.arange(len(sort_indices))

    # Plot weights RMSE
    ax1 = axes[0]
    ax1.bar(x_pos, rmse_weights[sort_indices], color='skyblue', alpha=0.7)
    ax1.set_ylabel('RMSE weights', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 2.5])
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(sorted_neuron_type_names, rotation=90, ha='right', fontsize=6)
    ax1.grid(False)
    ax1.tick_params(axis='y', labelsize=12)

    for i, (tick, rmse_w) in enumerate(zip(ax1.get_xticklabels(), rmse_weights[sort_indices])):
        if rmse_w > 0.5:
            tick.set_color('red')
            tick.set_fontsize(8)

    # Panel 2 (tau)
    ax2 = axes[1]
    ax2.bar(x_pos, rmse_taus[sort_indices], color='lightcoral', alpha=0.7)
    ax2.set_ylabel(r'RMSE $\tau$', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 0.3])
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(sorted_neuron_type_names, rotation=90, ha='right', fontsize=6)
    ax2.grid(False)
    ax2.tick_params(axis='y', labelsize=12)

    # Calculate mean ground truth taus per neuron type
    mean_gt_taus = []
    for neuron_type in range(n_neuron_types):
        type_indices = np.where(type_list == neuron_type)[0]
        gt_tau_type = gt_taus[type_indices]
        mean_gt_taus.append(np.mean(np.abs(gt_tau_type)))

    mean_gt_taus = np.array(mean_gt_taus)

    for i, (tick, rmse_tau) in enumerate(zip(ax2.get_xticklabels(), rmse_taus[sort_indices])):
        if rmse_tau > 0.03:
            tick.set_color('red')
            tick.set_fontsize(8)

    # Panel 3 (V_rest)
    ax3 = axes[2]
    ax3.bar(x_pos, rmse_vrests[sort_indices], color='lightgreen', alpha=0.7)
    ax3.set_ylabel(r'RMSE $V_{rest}$', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 0.8])
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(sorted_neuron_type_names, rotation=90, ha='right', fontsize=6)
    ax3.grid(False)
    ax3.tick_params(axis='y', labelsize=12)

    # Calculate mean ground truth V_rest per neuron type
    mean_gt_vrests = []
    for neuron_type in range(n_neuron_types):
        type_indices = np.where(type_list == neuron_type)[0]
        gt_vrest_type = gt_V_Rest[type_indices]
        mean_gt_vrests.append(np.mean(np.abs(gt_vrest_type)))

    mean_gt_vrests = np.array(mean_gt_vrests)
    for i, (tick, rmse_vrest) in enumerate(zip(ax3.get_xticklabels(), rmse_vrests[sort_indices])):
        if rmse_vrest > 0.08:
            tick.set_color('red')
            tick.set_fontsize(8)

    plt.tight_layout()
    plt.savefig(f'./{log_dir}/results/neuron_type_reconstruction.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Log summary statistics
    logger.info("Neuron type reconstruction analysis:")
    logger.info(f"Mean weights RMSE: {np.mean(rmse_weights):.3f} ± {np.std(rmse_weights):.3f}")
    logger.info(f"Mean tau RMSE: {np.mean(rmse_taus):.3f} ± {np.std(rmse_taus):.3f}")
    logger.info(f"Mean V_rest RMSE: {np.mean(rmse_vrests):.3f} ± {np.std(rmse_vrests):.3f}")

    # Return per-neuron results (NEW)
    return {
        'rmse_weights_per_neuron': rmse_weights_per_neuron,
        'rmse_tau_per_neuron': rmse_tau_per_neuron,
        'rmse_vrest_per_neuron': rmse_vrest_per_neuron,
        'rmse_weights_per_type': rmse_weights,
        'rmse_tau_per_type': rmse_taus,
        'rmse_vrest_per_type': rmse_vrests
    }


def plot_reconstruction_correlations(activity_results, results_per_neuron, gt_taus, gt_V_Rest,
                                   type_list, index_to_name, log_dir, alpha_individual=0.0):
   """
   Plot correlations between neuron statistics and reconstruction performance.
   Parameters:
   -----------
   activity_results : dict with 'mu_activity', 'sigma_activity'
   results_per_neuron : dict with 'rmse_weights_per_neuron', 'rmse_tau_per_neuron', 'rmse_vrest_per_neuron'
   gt_taus : ground truth tau values
   gt_V_Rest : ground truth V_rest values
   type_list : neuron type labels
   index_to_name : mapping from type index to name
   log_dir : directory to save plots
   alpha_individual : transparency for individual neurons (default 0.2)
   """
   mu_activity = activity_results['mu_activity']
   sigma_activity = activity_results['sigma_activity']
   rmse_weights = results_per_neuron['rmse_weights_per_neuron']
   rmse_tau = results_per_neuron['rmse_tau_per_neuron']
   rmse_vrest = results_per_neuron['rmse_vrest_per_neuron']

   gt_taus = to_numpy(gt_taus)
   gt_V_Rest = to_numpy(gt_V_Rest)
   type_list = to_numpy(type_list).squeeze()

   n_neurons = len(mu_activity)
   n_types = len(np.unique(type_list))

   # Create colormap for neuron types
   cmap = plt.cm.get_cmap('tab20b')
   if n_types > 20:
       cmap = plt.cm.get_cmap('hsv')
   colors = [cmap(i/n_types) for i in range(n_types)]
   neuron_colors = [colors[int(t)] for t in type_list]

   # Compute type-averaged values
   unique_types = np.unique(type_list)
   type_mean_mu = np.zeros(len(unique_types))
   type_mean_sigma = np.zeros(len(unique_types))
   type_mean_tau = np.zeros(len(unique_types))
   type_mean_vrest = np.zeros(len(unique_types))
   type_mean_rmse_weights = np.zeros(len(unique_types))
   type_mean_rmse_tau = np.zeros(len(unique_types))
   type_mean_rmse_vrest = np.zeros(len(unique_types))

   for i, type_id in enumerate(unique_types):
       type_mask = (type_list == type_id)
       type_mean_mu[i] = np.mean(mu_activity[type_mask])
       type_mean_sigma[i] = np.mean(sigma_activity[type_mask])
       type_mean_tau[i] = np.mean(gt_taus[type_mask])
       type_mean_vrest[i] = np.mean(gt_V_Rest[type_mask])
       type_mean_rmse_weights[i] = np.mean(rmse_weights[type_mask])
       type_mean_rmse_tau[i] = np.mean(rmse_tau[type_mask])
       type_mean_rmse_vrest[i] = np.mean(rmse_vrest[type_mask])

   # Create figure with spacing between columns
   fig = plt.figure(figsize=(20, 16))
   gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.4,
                        left=0.06, right=0.94, top=0.96, bottom=0.06)

   # Define row labels and data
   row_data = [
       (mu_activity, type_mean_mu, r'$\mu_{activity}$'),
       (sigma_activity, type_mean_sigma, r'$\sigma_{activity}$'),
       (gt_taus * 1000, type_mean_tau * 1000, r'$\tau$ [ms]'),  # Convert to ms
       (gt_V_Rest, type_mean_vrest, r'$V_{rest}$')
   ]

   # Define column labels and data
   col_data = [
       (rmse_weights, type_mean_rmse_weights, 'RMSE weights'),
       (rmse_tau, type_mean_rmse_tau, r'RMSE $\tau$'),
       (rmse_vrest, type_mean_rmse_vrest, r'RMSE $V_{rest}$')
   ]

   # Create all panels
   for row_idx, (x_data, x_type_mean, x_label) in enumerate(row_data):
       for col_idx, (y_data, y_type_mean, y_label) in enumerate(col_data):
           ax = fig.add_subplot(gs[row_idx, col_idx])

           # Filter out inf/nan values for individuals
           valid_mask = np.isfinite(x_data) & np.isfinite(y_data) & (y_data < 200)
           x_valid = x_data[valid_mask]
           y_valid = y_data[valid_mask]
           colors_valid = [neuron_colors[i] for i in range(n_neurons) if valid_mask[i]]

           # Plot individual neurons with low alpha
           ax.scatter(x_valid, y_valid, c=colors_valid,
                              alpha=alpha_individual, s=10, edgecolors='none')

           # Plot type-averaged values with high visibility
           for i, type_id in enumerate(unique_types):
               ax.scatter(x_type_mean[i], y_type_mean[i],
                         c=[colors[int(type_id)]], s=100,
                         edgecolors='black', linewidth=1.5,
                         alpha=1.0, zorder=10)

           # Labels and formatting
           ax.set_xlabel(x_label, fontsize=12)
           ax.set_ylabel('', fontsize=12)
           ax.grid(True, alpha=0.2)
           ax.tick_params(labelsize=10)

           # Set y-axis limit for RMSE
           ax.set_ylim([0, min(150, np.percentile(y_valid, 99) * 1.1)])

           # Add column title for first row
           if row_idx == 0:
               ax.set_title(y_label, fontsize=16, pad=10)

   plt.tight_layout()
   plt.savefig(f'{log_dir}/results/reconstruction_correlations.png',
               dpi=300, bbox_inches='tight')
   plt.close()


def movie_synaptic_flyvis(config, log_dir, n_runs, device, x_list, y_list, edges, gt_weights, gt_taus, gt_V_Rest,
                          type_list, n_neurons, n_types, colors_65, mu_activity, sigma_activity, cmap, mc, ynorm,
                          logger):
    """Create training evolution movies for flyvis analysis including individual subplot movies."""

    config_indices = config.dataset.split('fly_N9_')[1] if 'fly_N9_' in config.dataset else 'evolution'
    files, file_id_list = get_training_files(log_dir, n_runs)

    fps = 10
    metadata = dict(title='Model evolution', artist='Matplotlib', comment='Model evolution over epochs')

    # Create main combined movie
    create_combined_movie(config, log_dir, config_indices, files, file_id_list, n_runs, device, x_list, y_list,
                          edges, gt_weights, gt_taus, gt_V_Rest, type_list, n_neurons, n_types, colors_65,
                          mu_activity, sigma_activity, cmap, mc, ynorm, logger, fps, metadata)

    # Create individual subplot movies
    # create_individual_movies(config, log_dir, config_indices, files, file_id_list, n_runs, device, x_list, y_list,
    #                          edges, gt_weights, gt_taus, gt_V_Rest, type_list, n_neurons, n_types, colors_65,
    #                          mu_activity, sigma_activity, cmap, mc, ynorm, logger, fps, metadata)


def create_combined_movie(config, log_dir, config_indices, files, file_id_list, n_runs, device, x_list, y_list,
                          edges, gt_weights, gt_taus, gt_V_Rest, type_list, n_neurons, n_types, colors_65,
                          mu_activity, sigma_activity, cmap, mc, ynorm, logger, fps, metadata):
    """Create the main 1x4 subplot movie."""

    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig = plt.figure(figsize=(20, 20))  # Wider figure for 4 columns
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    mp4_path = f'{log_dir}/results/training_{config_indices}.mp4'

    with writer.saving(fig, mp4_path, dpi=80):
        for file_id_ in trange(len(file_id_list), ncols=90):
            plt.clf()  # Clear the figure

            # Load model for this epoch
            model, epoch = load_model_for_epoch(config, log_dir, files, file_id_, n_runs, device, edges, logger)

            # Analyze model functions
            slopes_lin_phi_list, offsets_list, slopes_lin_edge_list, _ = analyze_model_functions(
                model, config, n_neurons, mu_activity, sigma_activity, device, x_list, y_list, edges, ynorm)

            # Create 4 subplots in 1 row
            create_weight_subplot(fig, model, gt_weights, mc, 2, 2, 1)
            create_embedding_subplot(fig, model, type_list, n_types, colors_65, 2, 2, 2)
            create_lin_phi_subplot(fig, model, config, n_neurons, mu_activity, sigma_activity, cmap, type_list, device, 2, 2, 3)
            create_lin_edge_subplot(fig, model, config, n_neurons, mu_activity, sigma_activity, cmap, type_list, device, 2, 2, 4)

            # plt.suptitle(f'Epoch {epoch}', fontsize=20)
            plt.tight_layout()
            writer.grab_frame()

    print(f"Combined MP4 saved as: {mp4_path}")


def create_individual_movies(config, log_dir, config_indices, files, file_id_list, n_runs, device, x_list, y_list,
                             edges, gt_weights, gt_taus, gt_V_Rest, type_list, n_neurons, n_types, colors_65,
                             mu_activity, sigma_activity, cmap, mc, ynorm, logger, fps, metadata):
    """Create individual movies for each subplot component."""

    movie_configs = [
        ('weight_reconstruction', (8, 8), create_weight_subplot),
        ('embedding_recons', (8, 8), create_embedding_subplot),
        ('tau_recons', (8, 8), create_tau_subplot),
        ('V_rest_recons', (8, 8), create_vrest_subplot),
        ('lin_phi_recons', (8, 8), create_lin_phi_subplot),
        ('lin_edge_recons', (8, 8), create_lin_edge_subplot)
    ]

    for movie_name, figsize, subplot_func in movie_configs:
        writer = FFMpegWriter(fps=fps, metadata=metadata)
        fig = plt.figure(figsize=figsize)
        mp4_path = f'{log_dir}/results/{movie_name}_{config_indices}.mp4'
        png_path = f'{log_dir}/results/{movie_name}_{config_indices}_first_frame.png'

        logger.info(f'Creating {movie_name} movie...')

        idx = 0

        with writer.saving(fig, mp4_path, dpi=80):
            for file_id_ in tqdm(file_id_list, desc=f'creating {movie_name}'):
                plt.clf()  # Clear the figure

                # Load model for this epoch
                model, epoch = load_model_for_epoch(config, log_dir, files, file_id_, n_runs, device, edges, logger)

                # Analyze model functions
                slopes_lin_phi_list, offsets_list, slopes_lin_edge_list, _ = analyze_model_functions(
                    model, config, n_neurons, mu_activity, sigma_activity, device, x_list, y_list, edges, ynorm)

                # Create the specific subplot
                if movie_name == 'weight_reconstruction':
                    create_weight_subplot(fig, model, gt_weights, mc, 1, 1, 1)
                elif movie_name == 'embedding_recons':
                    create_embedding_subplot(fig, model, type_list, n_types, colors_65, 1, 1, 1)
                elif movie_name == 'tau_recons':
                    create_tau_subplot(fig, slopes_lin_phi_list, gt_taus, n_neurons, mc, 1, 1, 1)
                elif movie_name == 'V_rest_recons':
                    create_vrest_subplot(fig, slopes_lin_phi_list, offsets_list, gt_V_Rest, n_neurons, mc, 1, 1, 1)
                elif movie_name == 'lin_phi_recons':
                    create_lin_phi_subplot(fig, model, config, n_neurons, mu_activity, sigma_activity, cmap, type_list,
                                           device, 1, 1, 1)
                elif movie_name == 'lin_edge_recons':
                    create_lin_edge_subplot(fig, model, config, n_neurons, mu_activity, sigma_activity, cmap, type_list,
                                            device, 1, 1, 1)

                # plt.title(f'Epoch {epoch}', fontsize=16)
                plt.tight_layout()
                writer.grab_frame()

                if idx == 0:
                    plt.savefig(png_path, dpi=300, bbox_inches='tight')
                idx += 1

        logger.info(f'{movie_name} saved as: {mp4_path}')


def load_model_for_epoch(config, log_dir, files, file_id_, n_runs, device, edges, logger):
    """Load model for specific epoch."""
    epoch = files[file_id_].split('graphs')[1][1:-3]
    net = f'{log_dir}/models/best_model_with_{n_runs - 1}_graphs_{epoch}.pt'
    model = Signal_Propagation_FlyVis(aggr_type=config.graph_model.aggr_type, config=config, device=device)
    state_dict = torch.load(net, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    model.edges = edges
    logger.info(f'net: {net}')
    return model, epoch


def analyze_model_functions(model, config, n_neurons, mu_activity, sigma_activity, device, x_list, y_list, edges,
                            ynorm):
    """Analyze model functions and return slopes and corrected weights."""
    slopes_lin_phi_list, offsets_list = analyze_lin_phi_functions(model, config, n_neurons, mu_activity, sigma_activity,
                                                                  device)
    slopes_lin_edge_list = analyze_lin_edge_functions(model, config, n_neurons, mu_activity, sigma_activity, device)
    corrected_W = calculate_corrected_weights(model, config, x_list, y_list, edges, ynorm, device, slopes_lin_phi_list,
                                              slopes_lin_edge_list)
    return slopes_lin_phi_list, offsets_list, slopes_lin_edge_list, corrected_W


def analyze_lin_phi_functions(model, config, n_neurons, mu_activity, sigma_activity, device):
    """Analyze lin_phi functions and return slopes and offsets."""
    slopes_lin_phi_list = []
    offsets_list = []

    for n in range(n_neurons):
        rr = torch.linspace(mu_activity[n] - 2 * sigma_activity[n], mu_activity[n] + 2 * sigma_activity[n], 1000,
                            device=device)
        embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
        in_features = torch.cat((rr[:, None], embedding_, rr[:, None] * 0, torch.zeros_like(rr[:, None])), dim=1)

        with torch.no_grad():
            func = model.lin_phi(in_features.float())

        rr_numpy = to_numpy(rr)
        func_numpy = to_numpy(func.squeeze())
        try:
            lin_fit, _ = curve_fit(linear_model, rr_numpy, func_numpy)
            slope, offset = lin_fit[0], lin_fit[1]
        except:
            coeffs = np.polyfit(rr_numpy, func_numpy, 1)
            slope, offset = coeffs[0], coeffs[1]

        slopes_lin_phi_list.append(slope)
        offsets_list.append(offset)

    return slopes_lin_phi_list, offsets_list


def analyze_lin_edge_functions(model, config, n_neurons, mu_activity, sigma_activity, device):
    """Analyze lin_edge functions and return slopes."""
    slopes_lin_edge_list = []

    for n in range(n_neurons):
        rr = torch.linspace(mu_activity[n] - 2 * sigma_activity[n], mu_activity[n] + 2 * sigma_activity[n], 1000,
                            device=device)
        embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)

        if ('PDE_N9_A' in config.graph_model.signal_model_name) | ('PDE_N9_D' in config.graph_model.signal_model_name):
            in_features = torch.cat((rr[:, None], embedding_,), dim=1)
        elif ('PDE_N9_B' in config.graph_model.signal_model_name):
            in_features = torch.cat((rr[:, None] * 0, rr[:, None], embedding_, embedding_), dim=1)

        with torch.no_grad():
            func = model.lin_edge(in_features.float())
            if config.graph_model.lin_edge_positive:
                func = func ** 2

        rr_numpy = to_numpy(rr[rr.shape[0] // 2 + 1:])
        func_numpy = to_numpy(func[rr.shape[0] // 2 + 1:].squeeze())
        try:
            lin_fit, _ = curve_fit(linear_model, rr_numpy, func_numpy)
            slope = lin_fit[0]
        except:
            coeffs = np.polyfit(rr_numpy, func_numpy, 1)
            slope = coeffs[0]

        slopes_lin_edge_list.append(slope)

    return slopes_lin_edge_list


def calculate_corrected_weights(model, config, x_list, y_list, edges, ynorm, device, slopes_lin_phi_list,
                                slopes_lin_edge_list):
    """Calculate corrected weights using gradient analysis."""
    # [Implementation of the corrected weight calculation logic from the original code]
    # This would include the data construction and gradient calculation steps
    # Returning placeholder for brevity
    return torch.zeros((edges.shape[1], 1), device=device)


def create_weight_subplot(fig, model, gt_weights, mc, rows, cols, pos):
    """Create weight comparison subplot using uncorrected weights."""
    ax = fig.add_subplot(rows, cols, pos)
    learned_weights = to_numpy(model.W.squeeze())
    true_weights = to_numpy(gt_weights)

    # Fit linear model for R² calculation
    from scipy.optimize import curve_fit
    lin_fit, _ = curve_fit(linear_model, true_weights, learned_weights)
    residuals = learned_weights - linear_model(true_weights, *lin_fit)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((learned_weights - np.mean(learned_weights)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    plt.text(0.05, 0.95, f'R²: {r_squared:.3f}\nslope: {lin_fit[0]:.2f}\nN: {len(true_weights)}',
             transform=ax.transAxes, verticalalignment='top', fontsize=24)

    plt.scatter(true_weights, learned_weights , c=mc, s=0.1, alpha=0.1)
    ax.set_xlabel('true $W_{ij}$', fontsize=32)
    ax.set_ylabel('learned $W_{ij}$', fontsize=32)
    ax.set_xlim([-2, 2])
    ax.set_ylim([-8, 8])
    ax.tick_params(axis='both', which='major', labelsize=24)


def create_embedding_subplot(fig, model, type_list, n_types, colors_65, rows, cols, pos):
    """Create embedding subplot."""
    ax = fig.add_subplot(rows, cols, pos)
    embedding_plot = to_numpy(model.a)

    for n in range(n_types):
        type_mask = (to_numpy(type_list).squeeze() == n)
        if np.any(type_mask):
            ax.scatter(embedding_plot[type_mask, 0], embedding_plot[type_mask, 1],
                       c=colors_65[n], s=6, alpha=0.25, edgecolors='none')
    ax.set_xlabel('$a_0$', fontsize=32)
    ax.set_ylabel('$a_1$', fontsize=32)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


def create_lin_phi_subplot(fig, model, config, n_neurons, mu_activity, sigma_activity, cmap, type_list, device, rows,
                           cols, pos):
    """Create lin_phi function subplot."""
    ax = fig.add_subplot(rows, cols, pos)

    for n in range(n_neurons):
        if n % 20 == 0:
            rr = torch.linspace(config.plotting.xlim[0], config.plotting.xlim[1], 1000, device=device)
            embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
            in_features = torch.cat((rr[:, None], embedding_, rr[:, None] * 0, torch.zeros_like(rr[:, None])), dim=1)
            with torch.no_grad():
                func = model.lin_phi(in_features.float())
                ax.plot(to_numpy(rr), to_numpy(func), color=cmap.color(to_numpy(type_list)[n].astype(int)),
                        linewidth=1, alpha=0.2)

    ax.set_xlim(config.plotting.xlim)
    ax.set_ylim(config.plotting.ylim)
    ax.set_xlabel('$v_i$', fontsize=32)
    ax.set_ylabel(r'learned $\mathrm{MLP_0}(\mathbf{a}_i, v_i)$', fontsize=32)
    ax.tick_params(axis='both', which='major', labelsize=24)


def create_lin_edge_subplot(fig, model, config, n_neurons, mu_activity, sigma_activity, cmap, type_list, device, rows,
                            cols, pos):
    """Create lin_edge function subplot."""
    ax = fig.add_subplot(rows, cols, pos)

    for n in range(n_neurons):
        if n % 20 == 0:
            rr = torch.linspace(config.plotting.xlim[0], config.plotting.xlim[1], 1000, device=device)
            embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
            if ('PDE_N9_A' in config.graph_model.signal_model_name) | (
                    'PDE_N9_D' in config.graph_model.signal_model_name):
                in_features = torch.cat((rr[:, None], embedding_,), dim=1)
            elif ('PDE_N9_B' in config.graph_model.signal_model_name):
                in_features = torch.cat((rr[:, None] * 0, rr[:, None], embedding_, embedding_), dim=1)

            with torch.no_grad():
                func = model.lin_edge(in_features.float())
                if config.graph_model.lin_edge_positive:
                    func = func ** 2
                ax.plot(to_numpy(rr), to_numpy(func), color=cmap.color(to_numpy(type_list)[n].astype(int)),
                        linewidth=1, alpha=0.2)

    ax.set_xlim(config.plotting.xlim)
    ax.set_ylim([-config.plotting.xlim[1] / 10, config.plotting.xlim[1] * 1.2])
    ax.set_xlabel('$v_i$', fontsize=32)
    ax.set_ylabel(r'learned $\mathrm{MLP_1}(\mathbf{a}_j, v_i)$', fontsize=32)
    ax.tick_params(axis='both', which='major', labelsize=24)


def create_tau_subplot(fig, slopes_lin_phi_list, gt_taus, n_neurons, mc, rows, cols, pos):
    """Create tau comparison subplot."""
    ax = fig.add_subplot(rows, cols, pos)
    slopes_lin_phi_array_np = np.array(slopes_lin_phi_list)
    learned_tau = np.where(slopes_lin_phi_array_np != 0, 1.0 / -slopes_lin_phi_array_np, 1)
    learned_tau = learned_tau[:n_neurons]
    learned_tau = np.clip(learned_tau, 0, 1)
    gt_taus_numpy = to_numpy(gt_taus[:n_neurons])

    lin_fit, _ = curve_fit(linear_model, gt_taus_numpy, learned_tau)
    residuals = learned_tau - linear_model(gt_taus_numpy, *lin_fit)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((learned_tau - np.mean(learned_tau)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    ax.scatter(gt_taus_numpy, learned_tau, c=mc, s=1, alpha=0.25)
    ax.text(0.05, 0.95, f'R²: {r_squared:.3f}\nslope: {lin_fit[0]:.2f}\nN: {len(gt_taus)}',
            transform=ax.transAxes, verticalalignment='top', fontsize=24)
    ax.set_xlabel('true $\\tau$', fontsize=32)
    ax.set_ylabel('learned $\\tau$', fontsize=32)
    ax.set_xlim([0, 0.35])
    ax.set_ylim([0, 0.35])
    ax.tick_params(axis='both', which='major', labelsize=24)


def create_vrest_subplot(fig, slopes_lin_phi_list, offsets_list, gt_V_Rest, n_neurons, mc, rows, cols, pos):
    """Create V_rest comparison subplot."""
    ax = fig.add_subplot(rows, cols, pos)
    slopes_lin_phi_array_np = np.array(slopes_lin_phi_list)
    offsets_array = np.array(offsets_list)
    learned_V_rest = np.where(slopes_lin_phi_array_np != 0, -offsets_array / slopes_lin_phi_array_np, 1)

    gt_V_rest_numpy = to_numpy(gt_V_Rest[:n_neurons])
    lin_fit, _ = curve_fit(linear_model, gt_V_rest_numpy, learned_V_rest)
    residuals = learned_V_rest - linear_model(gt_V_rest_numpy, *lin_fit)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((learned_V_rest - np.mean(learned_V_rest)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    ax.scatter(gt_V_rest_numpy, learned_V_rest, c=mc, s=1, alpha=0.25)
    ax.text(0.05, 0.95, f'R²: {r_squared:.3f}\nslope: {lin_fit[0]:.2f}\nN: {len(gt_V_rest_numpy)}',
            transform=ax.transAxes, verticalalignment='top', fontsize=24)
    ax.set_xlabel('true $V_{rest}$', fontsize=32)
    ax.set_ylabel(r'learned $\widehat{V}_{rest}$', fontsize=32)
    ax.set_xlim([-0.05, 0.9])
    ax.set_ylim([-0.05, 0.9])
    ax.tick_params(axis='both', which='major', labelsize=24)


def plot_ising_comparison_from_saved(config_list, labels=None, output_path='fig/ising_noise_comparison.png'):
    valid_configs = []

    for config_file_ in config_list:
        try:
            config_file, pre_folder = add_pre_folder(config_file_)
            data_path = f'./log/{config_file}/results/info_ratio_results.npz'
            if not os.path.exists(data_path):
                print(f"Warning: {data_path} not found")
                continue
            valid_configs.append(config_file)
        except Exception as e:
            print(f"Error processing {config_file_}: {e}")
            continue

    if len(valid_configs) != 3:
        raise ValueError(f"Need exactly 3 valid configs, got {len(valid_configs)}")

    # Default sigma labels with lowercase noise descriptions
    if labels is None:
        labels = [r'$\sigma=10^{-6}$ (low noise)',
                 r'$\sigma=0.25$ (moderate noise)',
                 r'$\sigma=2.5$ (large noise)']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(wspace=0.4)

    for col, (config_file, label) in enumerate(zip(valid_configs, labels)):
        data_path = f'./log/{config_file}/results/info_ratio_results.npz'
        data = np.load(data_path)

        obs_rates = data['observed_rates'].flatten()
        pair_rates = data['predicted_rates_pairwise'].flatten()
        indep_rates = data['predicted_rates_independent'].flatten()

        # Rate scatter plots
        ax = axes[col]

        # Plot data
        ax.loglog(obs_rates, pair_rates, 'r.', alpha=0.1, markersize=1.5)
        ax.loglog(obs_rates, indep_rates, 'g.', alpha=0.1, markersize=1.5)
        ax.plot([1e-4, 1e1], [1e-4, 1e1], 'k--', alpha=0.5, linewidth=1)

        ax.set_xlabel(r'observed rate (s$^{-1}$)', fontsize=18)
        ax.set_ylabel(r'predicted rate (s$^{-1}$)' if col == 0 else '', fontsize=18)
        ax.set_title(f'{label}\n$I_N$={data["I_N_median"]:.2f} bits', fontsize=16)
        ax.set_xlim(1e-4, 1e1)
        ax.set_ylim(1e-4, 1e1)
        ax.tick_params(labelsize=14)

        # Legend only on the first panel - top left
        if col == 0:
            red_patch = plt.Rectangle((0, 0), 1, 1, facecolor='red', alpha=1.0)
            green_patch = plt.Rectangle((0, 0), 1, 1, facecolor='green', alpha=1.0)
            ax.legend([red_patch, green_patch], ['pairwise ($P_2$)', 'independent ($P_1$)'],
                     loc='upper left', fontsize=14, frameon=False)

    # Add panel labels - moved upwards
    for i, ax in enumerate(axes):
        ax.text(-0.15, 1.15, f'{chr(97+i)})', transform=ax.transAxes, fontsize=20, va='top', ha='left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.show()





def compare_gnn_results(config_list, varied_parameter):
    """
    Compare GNN experiments by reading config files and results.log files
    Focuses on: weights/tau/V_rest R², clustering accuracy, loss curves
    """

    # Global style
    plt.style.use('ggplot')
    plt.rcParams['axes.facecolor'] = 'black'
    plt.rcParams['figure.facecolor'] = 'black'
    plt.rcParams['axes.edgecolor'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    plt.rcParams['grid.color'] = 'gray'
    plt.rcParams['text.color'] = 'white'

    plt.style.use('default')

    from NeuralGraph.config import NeuralGraphConfig
    from NeuralGraph.models.utils import add_pre_folder

    results = []

    # Read & parse per-config files
    for config_file_ in config_list:
        try:
            config_file, pre_folder = add_pre_folder(config_file_)
            config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')

            # Resolve varied parameter value
            if varied_parameter is None:
                parts = config_file_.split('_')
                if len(parts) >= 2:
                    param_value = parts[-1]
                else:
                    print(f"warning: cannot extract indices from config name '{config_file_}'")
                    continue
            else:
                if '.' not in varied_parameter:
                    raise ValueError("parameter must be in 'section.parameter' format")
                section_name, param_name = varied_parameter.split('.', 1)
                section = getattr(config, section_name, None)
                if section is None:
                    raise ValueError(f"config section '{section_name}' not found")
                param_value = getattr(section, param_name, None)
                if param_value is None:
                    print(f"warning: parameter '{param_name}' not found")
                    continue

            # Parse results.log
            results_log_path = os.path.join('./log', config_file, 'results.log')
            if not os.path.exists(results_log_path):
                print(f"warning: {results_log_path} not found")
                continue

            with open(results_log_path, 'r') as f:
                content = f.read()

            # Parse standard metrics
            r2_match = re.search(r'second weights fit\s+R²:\s*([\d.-]+)', content)
            r2 = float(r2_match.group(1)) if r2_match else None

            tau_r2_match = re.search(r'tau reconstruction R²:\s*([\d.-]+)', content)
            tau_r2 = float(tau_r2_match.group(1)) if tau_r2_match else None

            vrest_r2_match = re.search(r'V_rest reconstruction R²:\s*([\d.-]+)', content)
            vrest_r2 = float(vrest_r2_match.group(1)) if vrest_r2_match else None

            acc_match = re.search(r'accuracy=([\d.-]+)', content)
            best_clustering_acc = float(acc_match.group(1)) if acc_match else None

            results.append({
                'config': config_file_,
                'param_value': param_value,
                'r2': r2,
                'tau_r2': tau_r2,
                'vrest_r2': vrest_r2,
                'best_clustering_acc': best_clustering_acc,
            })

        except Exception as e:
            print(f"error processing {config_file_}: {e}")

    # Group by parameter value
    grouped_results = defaultdict(list)
    for r in results:
        if all(r[k] is not None for k in ['r2', 'tau_r2', 'vrest_r2', 'best_clustering_acc']):
            grouped_results[r['param_value']].append(r)

    # Aggregate into summary
    summary_results = []
    for param_val, group in grouped_results.items():
        r2_values = [r['r2'] for r in group]
        tau_r2_values = [r['tau_r2'] for r in group]
        vrest_r2_values = [r['vrest_r2'] for r in group]
        acc_values = [r['best_clustering_acc'] for r in group]

        summary_results.append({
            'param_value': param_val,
            'r2_mean': np.mean(r2_values),
            'tau_r2_mean': np.mean(tau_r2_values),
            'vrest_r2_mean': np.mean(vrest_r2_values),
            'acc_mean': np.mean(acc_values),
            'r2_std': np.std(r2_values) if len(r2_values) > 1 else 0,
            'tau_r2_std': np.std(tau_r2_values) if len(tau_r2_values) > 1 else 0,
            'vrest_r2_std': np.std(vrest_r2_values) if len(vrest_r2_values) > 1 else 0,
            'acc_std': np.std(acc_values) if len(acc_values) > 1 else 0,
            'n_configs': len(group)
        })

    # Sort results
    summary_results.sort(key=lambda x: x['param_value'])

    # Display parameter name
    if varied_parameter is None:
        param_display_name = "config_indices"
    else:
        param_display_name = varied_parameter.split('.')[1]

    # Print summary table
    print("-" * 75)
    print(f"{'parameter':<15} {'weights R²':<15} {'tau R²':<15} {'V_rest R²':<15} {'clustering':<15}")
    print("-" * 75)

    for r in summary_results:
        r2_str = f"{r['r2_mean']:.3f}±{r['r2_std']:.3f}" if r['r2_std'] > 0 else f"{r['r2_mean']:.3f}"
        tau_str = f"{r['tau_r2_mean']:.3f}±{r['tau_r2_std']:.3f}" if r['tau_r2_std'] > 0 else f"{r['tau_r2_mean']:.3f}"
        vrest_str = f"{r['vrest_r2_mean']:.3f}±{r['vrest_r2_std']:.3f}" if r['vrest_r2_std'] > 0 else f"{r['vrest_r2_mean']:.3f}"
        acc_str = f"{r['acc_mean']:.3f}±{r['acc_std']:.3f}" if r['acc_std'] > 0 else f"{r['acc_mean']:.3f}"

        print(f"{str(r['param_value']):<15} {r2_str:<15} {tau_str:<15} {vrest_str:<15} {acc_str:<15}")


    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.90, bottom=0.10, wspace=0.32, hspace=0.45)
    ax1.text(-0.08, 1.05, 'a)', transform=ax1.transAxes, fontsize=18, va='top', ha='right')
    ax2.text(-0.08, 1.05, 'b)', transform=ax2.transAxes, fontsize=18, va='top', ha='right')
    ax3.text(-0.08, 1.05, 'c)', transform=ax3.transAxes, fontsize=18, va='top', ha='right')
    ax4.text(-0.08, 1.05, 'd)', transform=ax4.transAxes, fontsize=18, va='top', ha='right')

    param_values = [r['param_value'] for r in summary_results]
    # param_display_name = 'noise level'

    x_values = [float(p) for p in param_values]
    use_log = min(x_values) > 0 and max(x_values)/min(x_values) > 10 and varied_parameter != None

    for ax, ydata, label, color in [
        (ax1, [r['r2_mean'] for r in summary_results], 'weights R²', 'blue'),
        (ax2, [r['tau_r2_mean'] for r in summary_results], 'tau R²', 'green'),
        (ax3, [r['vrest_r2_mean'] for r in summary_results], 'V_rest R²', 'orange'),
        (ax4, [r['acc_mean'] for r in summary_results], 'clustering accuracy', 'red'),
    ]:

        if use_log:
            plot_fn = ax.semilogx
            plot_fn(x_values, ydata, 'o', color=color, linewidth=2, markersize=8)
            ax.set_xlim(left=0, right=5)
        else:
            plot_fn = ax.plot
            plot_fn(x_values, ydata, 'o', color=color, linewidth=2, markersize=8)

        ax.set_xlabel(param_display_name, fontsize=18)

        if label == 'clustering accuracy':
            ax.set_ylabel('classification accuracy', fontsize=18)
        elif label == 'weights R²':
            ax.set_ylabel(r'learned $\widehat{\mathbf{W}}_{ij}\quadR²$', fontsize=18)
        elif label == 'tau R²':
            ax.set_ylabel(r'learned $\widehat{\tau}_i \quad R^2$', fontsize=18)
        elif label == 'V_rest R²':
            ax.set_ylabel(r'learned $\widehat{V}^{rest}_i\quad R^2$', fontsize=18)
        ax.set_ylim(0, 1.1)
        if use_log:
            ax.set_xscale('log')
        # ax.set_xlim(left=1e-6, right=1)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    sigma_labels = [r"$\sigma=1E-6$", r"$\sigma=0.25$", r"$\sigma=2.5$"]
    if not use_log and len(x_values) == 3:
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xticks(x_values)
            ax.set_xticklabels(sigma_labels, fontsize=18)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.label.set_size(18)
        ax.yaxis.label.set_size(18)
        ax.tick_params(axis='both', which='major', labelsize=10)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.axvspan(0.25, 100, color='green', alpha=0.35, zorder=-1, linewidth=0)

    plt.tight_layout()
    plt.savefig(f'fig/gnn_comparison_{param_display_name}.png', dpi=400, bbox_inches='tight')
    plt.show()

    return summary_results


def compare_ising_results(config_list, varied_parameter):
    """
    Compare Ising/information theory metrics across experiments
    Focuses on: I_N, I_2, I_2/I_N ratio, higher-order correlations
    """

    from NeuralGraph.config import NeuralGraphConfig
    from NeuralGraph.models.utils import add_pre_folder

    def parse_ising_metrics(content):
        """Extract Ising metrics from results.log content"""
        metrics = {}

        # Parse I_N
        match = re.search(r'I_N:\s*median=([0-9eE+\-\.]+),\s*IQR=\[\s*([0-9eE+\-\.]+),\s*([0-9eE+\-\.]+)\s*\]', content)
        if match:
            metrics['I_N_median'] = float(match.group(1))
            metrics['I_N_q1'] = float(match.group(2))
            metrics['I_N_q3'] = float(match.group(3))

        # Parse I2
        match = re.search(r'I2:\s*median=([0-9eE+\-\.]+),\s*IQR=\[\s*([0-9eE+\-\.]+),\s*([0-9eE+\-\.]+)\s*\]', content)
        if match:
            metrics['I2_median'] = float(match.group(1))
            metrics['I2_q1'] = float(match.group(2))
            metrics['I2_q3'] = float(match.group(3))

        # Parse I_HOC
        match = re.search(r'I_HOC:\s*median=([0-9eE+\-\.]+),\s*IQR=\[\s*([0-9eE+\-\.]+),\s*([0-9eE+\-\.]+)\s*\]', content)
        if match:
            metrics['I_HOC_median'] = float(match.group(1))
            metrics['I_HOC_q1'] = float(match.group(2))
            metrics['I_HOC_q3'] = float(match.group(3))

        # Parse ratio
        match = re.search(r'(?:ratio|I_2/I_N):\s*median=([0-9eE+\-\.]+),\s*IQR=\[\s*([0-9eE+\-\.]+),\s*([0-9eE+\-\.]+)\s*\]', content)
        if match:
            metrics['ratio_median'] = float(match.group(1))
            metrics['ratio_q1'] = float(match.group(2))
            metrics['ratio_q3'] = float(match.group(3))

        # Parse C_3 (connected triplets)
        match = re.search(r'C_3:\s*(?:mean=)?([0-9eE+\-\.]+)\s*±\s*([0-9eE+\-\.]+)', content)
        if match:
            metrics['C3_mean'] = float(match.group(1))
            metrics['C3_std'] = float(match.group(2))

        return metrics

    results = []

    # Parse each config
    for config_file_ in config_list:
        try:
            config_file, _ = add_pre_folder(config_file_)
            config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')

            # Get parameter value
            if varied_parameter is None:
                parts = config_file_.split('_')
                param_value = f"{parts[-2]}_{parts[-1]}" if len(parts) >= 2 else config_file_
            else:
                section_name, param_name = varied_parameter.split('.', 1)
                section = getattr(config, section_name, None)
                param_value = getattr(section, param_name, config_file_) if section else config_file_

            # Read results.log
            results_log_path = os.path.join('./log', config_file, 'results.log')
            if not os.path.exists(results_log_path):
                continue

            with open(results_log_path, 'r') as f:
                content = f.read()

            # Parse Ising metrics
            metrics = parse_ising_metrics(content)
            if metrics:
                metrics['param_value'] = param_value
                metrics['config'] = config_file_
                results.append(metrics)

        except Exception as e:
            print(f"Error processing {config_file_}: {e}")

    if not results:
        # print("No Ising metrics found")
        return None

    # Sort by parameter value
    try:
        results.sort(key=lambda x: float(x['param_value']))
    except:
        results.sort(key=lambda x: str(x['param_value']))

    # Display name
    if varied_parameter is None:
        param_display_name = "config"
    else:
        param_display_name = varied_parameter.split('.')[1]

    # Print summary table
    print(f"\n=== Ising Analysis Comparison: {param_display_name} ===")
    print(f"{'Parameter':<12} {'I_N (bits)':<15} {'I_2 (bits)':<15} {'I_HOC (bits)':<15} {'I_2/I_N':<12} {'C_3':<15}")
    print("-" * 85)

    for r in results:
        i_n = f"{r.get('I_N_median', 0):.3f}" if 'I_N_median' in r else "N/A"
        i_2 = f"{r.get('I2_median', 0):.3f}" if 'I2_median' in r else "N/A"
        i_hoc = f"{r.get('I_HOC_median', 0):.3f}" if 'I_HOC_median' in r else "N/A"
        ratio = f"{r.get('ratio_median', 0):.3f}" if 'ratio_median' in r else "N/A"
        c3 = f"{r.get('C3_mean', 0):.4f}±{r.get('C3_std', 0):.4f}" if 'C3_mean' in r else "N/A"

        print(f"{str(r['param_value']):<12} {i_n:<15} {i_2:<15} {i_hoc:<15} {ratio:<12} {c3:<15}")

    # Create plots
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # Add panel labels a), b), c), d)
    axes[0, 0].text(-0.08, 1.05, 'a)', transform=axes[0, 0].transAxes, fontsize=18, va='top', ha='right')
    axes[0, 1].text(-0.08, 1.05, 'b)', transform=axes[0, 1].transAxes, fontsize=18, va='top', ha='right')
    axes[1, 0].text(-0.08, 1.05, 'c)', transform=axes[1, 0].transAxes, fontsize=18, va='top', ha='right')
    # axes[1, 1].text(-0.15, 1.08, 'd)', transform=axes[1, 1].transAxes, fontsize=18, va='top', ha='right')

    param_values = [r['param_value'] for r in results]
    param_display_name = 'noise level'

    # Try to convert to numeric for better plotting
    try:
        x_values = [float(p) for p in param_values]
        use_log = min(x_values) > 0 and max(x_values)/min(x_values) > 10
    except:
        x_values = range(len(param_values))
        use_log = False

    # Panel 1: I_N and I_2
    ax = axes[0, 0]
    if 'I_N_median' in results[0]:
        i_n_vals = [r.get('I_N_median', 0) for r in results]
        i_2_vals = [r.get('I2_median', 0) for r in results]

        plot_fn = ax.semilogx if use_log else ax.plot
        plot_fn(x_values, i_n_vals, 'o', label='$I_N$', linewidth=2, markersize=8)
        plot_fn(x_values, i_2_vals, 's', label='$I^{(2)}$', linewidth=2, markersize=8)
        # Add shaded region for sigma in [0.25, right edge]
        ax.axvspan(0.25, 100, color='green', alpha=0.35, zorder=-1, linewidth=0)
        ax.set_xlabel(param_display_name, fontsize=18)
        ax.set_ylabel('information (bits)', fontsize=18)
        ax.legend(fontsize=18)
        ax.set_xlim(left=0, right=5)
        # ax.grid(True, alpha=0.3)

    # Panel 2: I_2/I_N ratio
    ax = axes[0, 1]
    if 'ratio_median' in results[0]:
        ratio_vals = [r.get('ratio_median', 0) for r in results]
        plot_fn = ax.semilogx if use_log else ax.plot
        plot_fn(x_values, ratio_vals, 'o', color='green', linewidth=2, markersize=8)
        # Add shaded region for sigma in [0.25, right edge]
        ax.axvspan(0.25, 100, color='green', alpha=0.35, zorder=-1, linewidth=0)
        ax.set_xlabel(param_display_name, fontsize=18)
        ax.set_ylabel('$I^{(2)}/I_N$', fontsize=18)
        ax.set_ylim(0, 1.1)
        ax.set_xlim(left=0, right=5)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        # ax.grid(True, alpha=0.3)

    # Panel 3: Higher-order correlation
    ax = axes[1, 0]
    if 'I_HOC_median' in results[0]:
        i_hoc_vals = [r.get('I_HOC_median', 0) for r in results]
        plot_fn = ax.semilogx if use_log else ax.plot
        plot_fn(x_values, i_hoc_vals, 'o', color='purple', linewidth=2, markersize=8)
        # Add shaded region for sigma in [0.25, right edge]
        ax.axvspan(0.25, 100, color='green', alpha=0.35, zorder=-1, linewidth=0)
        ax.set_xlabel(param_display_name, fontsize=18)
        ax.set_ylabel('$I_{HOC}$ (bits)', fontsize=18)
        ax.set_xlim(left=0, right=5)
        # ax.grid(True, alpha=0.3)

    # Panel 4: Connected triplets C_3
    ax = axes[1, 1]
    # if 'C3_mean' in results[0]:
    #     c3_vals = [r.get('C3_mean', 0) for r in results]
    #     c3_err = [r.get('C3_std', 0) for r in results]

    #     if use_log:
    #         ax.errorbar(x_values, c3_vals, yerr=c3_err, fmt='o', color='teal',
    #                    linewidth=2, markersize=8, capsize=5)
    #         ax.set_xscale('log')
    #     else:
    #         ax.errorbar(x_values, c3_vals, yerr=c3_err, fmt='o', color='teal',
    #                    linewidth=2, markersize=8, capsize=5)
    #     ax.set_xlabel(param_display_name, fontsize=18)
    #     ax.set_ylabel('$C_3$ (triplet correlation)', fontsize=18)
    #     # ax.grid(True, alpha=0.3)

    ax.axis('off')

    # Set x-axis labels if not numeric
    if not isinstance(x_values[0], (int, float)):
        for ax in axes.flat:
            ax.set_xticks(x_values)
            ax.set_xticklabels(param_values, rotation=45, ha='right')

    # plt.suptitle(f'Information Structure vs {param_display_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'fig/ising_comparison_{param_display_name}.png', dpi=400, bbox_inches='tight')
    plt.show()

    return results


def compare_experiments(config_list, varied_parameter=None):
    """
    Run both GNN and Ising comparisons

    Args:
        config_list: list of config file names
        varied_parameter: 'section.parameter' format or None for config indices
    """

    # print("\n" + "="*80)
    # print("EXPERIMENT COMPARISON")
    # print("="*80)

    # Run GNN comparison
    gnn_results = compare_gnn_results(config_list, varied_parameter)

    # Run Ising comparison
    ising_results = compare_ising_results(config_list, varied_parameter)

    # Optional: Create combined summary plot if both analyses succeeded
    if gnn_results and ising_results:
        create_combined_summary_plot(gnn_results, ising_results, varied_parameter)

    return {'gnn': gnn_results, 'ising': ising_results}


def create_combined_summary_plot(gnn_results, ising_results, varied_parameter):
    """Create a combined plot showing both GNN performance and information metrics"""

    # This is where you could create your custom combined visualization
    # For example, showing GNN accuracy alongside I_HOC to demonstrate
    # the relationship between higher-order correlations and performance

    pass  # Implement as needed





def plot_neuron_activity_analysis(activity, target_type_name_list, type_list, index_to_name, n_neurons, n_frames, delta_t, output_path):

   # Calculate mean and std for each neuron
   mu_activity = torch.mean(activity, dim=1)
   sigma_activity = torch.std(activity, dim=1)

   # Create the plot (keeping original visualization)
   plt.figure(figsize=(16, 8))
   plt.errorbar(np.arange(n_neurons), to_numpy(mu_activity), yerr=to_numpy(sigma_activity),
                fmt='o', ecolor='lightgray', alpha=0.6, elinewidth=1, capsize=0,
                markersize=3, color='red')

   # Group neurons by type and add labels at type boundaries (similar to plot_ground_truth_distributions)
   type_boundaries = {}
   current_type = None
   for i in range(n_neurons):
       neuron_type_id = to_numpy(type_list[i]).item()
       if neuron_type_id != current_type:
           if current_type is not None:
               type_boundaries[current_type] = (type_boundaries[current_type][0], i - 1)
           type_boundaries[neuron_type_id] = (i, i)
           current_type = neuron_type_id

   # Close the last type boundary
   if current_type is not None:
       type_boundaries[current_type] = (type_boundaries[current_type][0], n_neurons - 1)

   # Add vertical lines and x-tick labels for each neuron type
   tick_positions = []
   tick_labels = []

   for neuron_type_id, (start_idx, end_idx) in type_boundaries.items():
       center_pos = (start_idx + end_idx) / 2
       neuron_type_name = index_to_name.get(neuron_type_id, f'Type{neuron_type_id}')

       tick_positions.append(center_pos)
       tick_labels.append(neuron_type_name)

       # Add vertical line at type boundary
       if start_idx > 0:
           plt.axvline(x=start_idx, color='gray', linestyle='--', alpha=0.3)

   # Set x-ticks with neuron type names rotated 90 degrees
   plt.xticks(tick_positions, tick_labels, rotation=90, fontsize=10)
   plt.ylabel(r'neuron voltage $v_i(t)\quad\mu_i \pm \sigma_i$', fontsize=16)
   plt.yticks(fontsize=18)

   plt.tight_layout()
   plt.savefig(f'./{output_path}/activity_mu_sigma.png', dpi=300, bbox_inches='tight')
   plt.close()

   # Return per-neuron statistics (NEW)
   return {
       'mu_activity': to_numpy(mu_activity),
       'sigma_activity': to_numpy(sigma_activity)
   }


def plot_ground_truth_distributions(edges, true_weights, gt_taus, gt_V_Rest, type_list, n_neuron_types,
                                    sorted_neuron_type_names, output_path):
    """
    Create a 4-panel vertical figure showing ground truth parameter distributions per neuron type
    with neuron type names as x-axis labels
    """

    fig, axes = plt.subplots(4, 1, figsize=(12, 16))

    # Get type boundaries for labels
    type_boundaries = {}
    current_type = None
    n_neurons = len(type_list)

    for i in range(n_neurons):
        neuron_type_id = int(type_list[i])
        if neuron_type_id != current_type:
            if current_type is not None:
                type_boundaries[current_type] = (type_boundaries[current_type][0], i - 1)
            type_boundaries[neuron_type_id] = (i, i)
            current_type = neuron_type_id

    # Close the last type boundary
    if current_type is not None:
        type_boundaries[current_type] = (type_boundaries[current_type][0], n_neurons - 1)

    def add_type_labels_and_setup_axes(ax, y_values, title):
        # Add mean line for each type and collect type positions
        type_positions = []
        type_names = []

        for neuron_type_id, (start_idx, end_idx) in type_boundaries.items():
            center_pos = (start_idx + end_idx) / 2
            type_positions.append(center_pos)
            neuron_type_name = sorted_neuron_type_names[int(neuron_type_id)] if int(neuron_type_id) < len(
                sorted_neuron_type_names) else f'Type{neuron_type_id}'
            type_names.append(neuron_type_name)

            # Add mean line for this type
            type_mean = np.mean(y_values[start_idx:end_idx + 1])
            ax.hlines(type_mean, start_idx, end_idx, colors='red', linewidth=3)

        # Set x-ticks to neuron type names
        ax.set_xticks(type_positions)
        ax.set_xticklabels(type_names, rotation=90, fontsize=8)
        ax.tick_params(axis='y', labelsize=16)

    # Panel 1: Scatter plot of true weights per connection with neuron index
    ax1 = axes[0]
    connection_targets = edges[1, :]
    connection_weights = true_weights

    ax1.scatter(connection_targets, connection_weights, c='white', s=0.1)
    ax1.set_ylabel('true weights', fontsize=16)

    # For weights, compute means per target neuron
    weight_means_per_neuron = np.zeros(n_neurons)
    for i in range(n_neurons):
        incoming_edges = np.where(edges[1, :] == i)[0]
        if len(incoming_edges) > 0:
            weight_means_per_neuron[i] = np.mean(true_weights[incoming_edges])

    add_type_labels_and_setup_axes(ax1, weight_means_per_neuron, 'distribution of true weights by neuron type')

    # Panel 2: Number of connections per neuron
    ax2 = axes[1]
    n_connections_per_neuron = np.zeros(n_neurons)
    for i in range(n_neurons):
        n_connections_per_neuron[i] = np.sum(edges[1, :] == i)

    ax2.scatter(np.arange(n_neurons), n_connections_per_neuron, c='white', s=0.1)
    ax2.set_ylabel('number of connections', fontsize=16)
    add_type_labels_and_setup_axes(ax2, n_connections_per_neuron, 'number of incoming connections by neuron type')

    # Panel 3: Scatter plot of true tau values per neuron
    ax3 = axes[2]
    ax3.scatter(np.arange(n_neurons), gt_taus * 1000, c='white', s=0.1)
    ax3.set_ylabel(r'true $\tau$ values [ms]', fontsize=16)
    add_type_labels_and_setup_axes(ax3, gt_taus * 1000, r'distribution of true $\tau$ by neuron type')

    # Panel 4: Scatter plot of true V_rest values per neuron
    ax4 = axes[3]
    ax4.scatter(np.arange(n_neurons), gt_V_Rest, c='white', s=0.1)
    ax4.set_ylabel(r'true $v_{rest}$ values [a.u.]', fontsize=16)
    add_type_labels_and_setup_axes(ax4, gt_V_Rest, r'distribution of true $v_{rest}$ by neuron type')

    plt.tight_layout()
    plt.savefig(f'{output_path}/ground_truth_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    return fig


def plot_loss_curves(log_dir, ylim=None):
    """
    Iterates through all folders in the specified directory, loads 'loss.pt' files,
    and plots all loss curves on a single plot.

    Parameters:
    - log_dir (str): Path to the directory containing subfolders with 'loss.pt' files.
    - output_file (str): Path to save the resulting plot.
    - ylim (tuple, optional): Y-axis limits for the plot (e.g., (0, 0.0075)).
    """
    loss_data = {}

    # Iterate through all folders in the specified directory
    for folder in os.listdir(log_dir):
        folder_path = os.path.join(log_dir, folder)
        if os.path.isdir(folder_path):  # Check if it's a directory
            loss_file = os.path.join(folder_path, 'loss.pt')
            if os.path.exists(loss_file):  # Check if 'loss.pt' exists
                try:
                    # Load the loss values
                    loss_values = torch.load(loss_file)
                    loss_data[folder] = loss_values
                except Exception as e:
                    print(f"Error loading {loss_file}: {e}")

    # Plot all loss lists on a single plot
    plt.figure(figsize=(10, 6))
    for folder, loss_values in loss_data.items():
        plt.plot(loss_values, label=folder)

    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Loss Curves', fontsize=16)
    plt.legend(fontsize=10)
    plt.grid(True)
    if ylim:
        plt.ylim(ylim)
    plt.tight_layout()
    plt.savefig(log_dir+'/loss_curves.png', dpi=150)
    plt.close()




def data_plot(config, config_file, epoch_list, style, extended, device):

    # plt.rcParams['text.usetex'] = False  # LaTeX disabled
    # rc('font', **{'family': 'serif', 'serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif', 'serif']})
    # matplotlib.rcParams['savefig.pad_inches'] = 0

    if 'black' in style:
        plt.style.use('dark_background')
        mc ='w'
    else:
        plt.style.use('default')
        mc = 'k'

    if 'latex' in style:
        plt.rcParams['text.usetex'] = False  # LaTeX disabled
        rc('font', **{'family': 'serif', 'serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif', 'serif']})

    matplotlib.rcParams['savefig.pad_inches'] = 0

    log_dir, logger = create_log_dir(config=config, erase=False)

    os.makedirs(os.path.join(log_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'results/all'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'results/training'), exist_ok=True)

    files = glob.glob(f"{log_dir}/results/training/*")
    for f in files:
        os.remove(f)
    os.makedirs(f"./{log_dir}/results/field", exist_ok=True)
    files = glob.glob(f"{log_dir}/results/field/*")
    for f in files:
        os.remove(f)

    if epoch_list==['best']:
        files = glob.glob(f"{log_dir}/models/*")
        files.sort(key=sort_key)
        filename = files[-1]
        filename = filename.split('/')[-1]
        filename = filename.split('graphs')[-1][1:-3]

        epoch_list=[filename]
        print(f'best model: {epoch_list}')
        logger.info(f'best model: {epoch_list}')

    if os.path.exists(f'{log_dir}/loss.pt'):
        loss = torch.load(f'{log_dir}/loss.pt')
        fig, ax = fig_init(formatx='%.0f', formaty='%.2f')
        plt.plot(loss, color=mc, linewidth=4)
        plt.xlim([0, 20])
        plt.ylabel('loss', fontsize=68)
        plt.xlabel('epochs', fontsize=68)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/loss.png", dpi=170.7)
        plt.close()
        # print('final loss {:.3e}'.format(loss[-1]))
        # logger.info('final loss {:.3e}'.format(loss[-1]))



    if 'CElegans' in config.dataset:
        plot_synaptic_CElegans(config, epoch_list, log_dir, logger, 'viridis', style, extended, device)
    elif 'fly' in config.dataset:
        if config.simulation.calcium_type != 'none':
            plot_synaptic_flyvis_calcium(config, epoch_list, log_dir, logger, 'viridis', style, extended, device)
        else:
            plot_synaptic_flyvis(config, epoch_list, log_dir, logger, 'viridis', style, extended, device)
    elif 'zebra' in config.dataset:
        plot_synaptic_zebra(config, epoch_list, log_dir, logger, 'viridis', style, extended, device)
    elif ('PDE_N3' in config.graph_model.signal_model_name):
        plot_synaptic3(config, epoch_list, log_dir, logger, 'viridis', style, extended, device)
    else:
        plot_signal(config, epoch_list, log_dir, logger, 'viridis', style, extended, device)

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def plot_results_figure(config_file_, config_indices, panel_suffix='domain'):
    """
    Generate a 2x3 figure panel for a given configuration.

    Args:
        config_file_: Config file name (e.g., 'fly_N9_44_24')
        config_indices: Index string for specific plots (e.g., '44_6')
        panel_suffix: Suffix for edge_functions panel ('domain' or 'all')
    """
    # Setup config
    config_file, pre_folder = add_pre_folder(config_file_)
    config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
    config.dataset = pre_folder + config.dataset
    config.config_file = pre_folder + config_file_
    print(f'figure results {config_file_}...')

    # Generate plots
    # data_plot(config=config, config_file=config_file, epoch_list=['best'],
    #           style='white color', extended='plots', device=device)

    # Setup paths
    log_dir = f'log/fly/{config_file_}'
    panels = {
        'a': f"{log_dir}/results/corrected_comparison.png",
        'b': f"{log_dir}/results/embedding_{config_indices}.png",
        'c': f"{log_dir}/results/edge_functions_{config_indices}_all.png",
        'd': f"{log_dir}/results/phi_functions_{config_indices}_domain.png",
        'e': f"{log_dir}/results/tau_comparison_{config_indices}.png",
        'f': f"{log_dir}/results/V_rest_comparison_{config_indices}.png"
    }

    # Create figure
    fig = plt.figure(figsize=(18, 12))

    for idx, (label, path) in enumerate(panels.items()):
        ax = fig.add_subplot(2, 3, idx+1)
        img = imageio.imread(path)
        plt.imshow(img)
        plt.axis('off')
        ax.text(0.1, 1.01, f'{label})', transform=ax.transAxes,
                fontsize=24, va='bottom', ha='right')

    plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02,
                        wspace=0.02, hspace=0.04)
    plt.savefig(f"./fig_paper/results_{config_file_.split('_')[-2]}_{config_file_.split('_')[-1]}.png",
                dpi=300, bbox_inches='tight')
    plt.savefig(f"./fig_paper/results_{config_file_.split('_')[-2]}_{config_file_.split('_')[-1]}.pdf",
                dpi=300, bbox_inches='tight')
    plt.close()


def get_figures(index):

    plt.style.use('default')

    match index:

        case 'results_44_6':
             plot_results_figure('fly_N9_44_24', '44_6', 'domain')
        case 'results_51_2':
             plot_results_figure('fly_N9_51_2', '37_2', 'domain')
        case 'results_22_10':
             plot_results_figure('fly_N9_22_10', '18_4_0', 'domain')

        case 'extra_edges':
            config_list = ['fly_N9_51_9', 'fly_N9_51_10', 'fly_N9_51_11', 'fly_N9_51_12']

            for config_file_ in config_list:
                config_file, pre_folder = add_pre_folder(config_file_)
                config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
                config.dataset = pre_folder + config.dataset
                config.config_file = pre_folder + config_file_
                logdir = f'log/fly/{config_file_}'
                data_test(
                    config,
                    visualize=True,
                    style="white color name",
                    verbose=False,
                    best_model='best',
                    run=0,
                    test_mode="full",
                    sample_embedding=False,
                    step=25000,
                    device=device,
                    particle_of_interest=0,
                )

        case 'figure_1_cosyne_2026':

            # config_file_ = 'fly_N9_22_10'
            # config_file, pre_folder = add_pre_folder(config_file_)
            # config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
            # config.dataset = pre_folder + config.dataset
            # config.config_file = pre_folder + config_file_
            # logdir = 'log/fly/fly_N9_22_10'
            # config.simulation.visual_input_type = "DAVIS"

            # data_test(
            #     config,
            #     visualize=False,
            #     style="white color name",
            #     verbose=False,
            #     best_model='best',
            #     run=0,
            #     test_mode="full",
            #     sample_embedding=False,
            #     step=25000,
            #     device=device,
            #     particle_of_interest=0,
            # )

            # config_file_ = 'fly_N9_22_10'
            # config_file, pre_folder = add_pre_folder(config_file_)
            # config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
            # config.dataset = pre_folder + config.dataset
            # config.config_file = pre_folder + config_file_
            # logdir = 'log/fly/fly_N9_22_10'
            # config.simulation.visual_input_type = "DAVIS"

            # data_test(
            #     config,
            #     visualize=False,
            #     style="white color name",
            #     verbose=False,
            #     best_model='best',
            #     run=0,
            #     test_mode="full test_ablation_50",
            #     sample_embedding=False,
            #     step=25000,
            #     device=device,
            #     particle_of_interest=0,
            # )

            # config_file_ = 'fly_N9_44_6'
            # config_file, pre_folder = add_pre_folder(config_file_)
            # config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
            # config.dataset = pre_folder + config.dataset
            # config.config_file = pre_folder + config_file_
            # logdir = 'log/fly/fly_N9_44_6'
            # config.simulation.visual_input_type = "DAVIS"

            # data_test(
            #     config,
            #     visualize=False,
            #     style="white color name",
            #     verbose=False,
            #     best_model='best',
            #     run=0,
            #     test_mode="full",
            #     sample_embedding=False,
            #     step=50,
            #     device=device,
            #     particle_of_interest=0,
            # )

            # config_file_ = 'fly_N9_51_2'
            # config_file, pre_folder = add_pre_folder(config_file_)
            # config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
            # config.dataset = pre_folder + config.dataset
            # config.config_file = pre_folder + config_file_
            # logdir = 'log/fly/fly_N9_51_2'

            # data_test(
            #     config,
            #     visualize=False,
            #     style="black color name true_only",
            #     verbose=False,
            #     best_model='best',
            #     run=0,
            #     test_mode="full",
            #     sample_embedding=False,
            #     step=50,
            #     device=device,
            #     particle_of_interest=0,
            # )


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

            print('plot figure 1...')
            x = np.load('graphs_data/fly/fly_N9_18_4_0/x_list_0.npy')
            type_list = x[-1,:, 6].astype(int)
            len(type_list)

            selected_types = [5, 12, 19, 23, 31, 35, 39, 43, 50, 55]
            neuron_indices = []
            for stype in selected_types:
                indices = np.where(type_list == stype)[0]
                if len(indices) == 0:
                    print(f"Type {stype} ({index_to_name[stype]}) not found in type_list")
                else:
                    neuron_indices.append(indices[0])
                    print(f"Type {stype} ({index_to_name[stype]}): neuron {indices[0]}")

            print(f"Found {len(neuron_indices)} neurons out of {len(selected_types)}")
            print(f"Unique types in type_list: {np.unique(type_list)}")

            logdirs = {'a': 'log/fly/fly_N9_22_10',
                    'b': 'log/fly/fly_N9_22_10',
                    'c': 'log/fly/fly_N9_44_6'}

            start_frame = 88000
            end_frame = 88500

            plt.style.use('default')
            fig, axes = plt.subplots(1, 3, figsize=(20, 12))

            for idx, (key, log_dir) in enumerate(logdirs.items()):
                print(f'processing {key}) {log_dir}...')
                ax = axes[idx]

                if idx==1:
                    true = np.load(f"./{log_dir}/results/activity_modified.npy")
                    pred = np.load(f"./{log_dir}/results/activity_modified_pred.npy")
                else:
                    true = np.load(f"./{log_dir}/results/activity_true.npy")
                    pred = np.load(f"./{log_dir}/results/activity_pred.npy")

                true_slice = true[neuron_indices, start_frame:end_frame].copy()
                pred_slice = pred[neuron_indices, start_frame:end_frame].copy()
                del true, pred

                step_v = 1.5

                for i in range(10):
                    baseline = np.mean(true_slice[i]) if idx==1 else np.mean(pred_slice[i])
                    lw = 6 if idx==2 else 10
                    ax.plot(true_slice[i] - baseline + i * step_v, linewidth=lw, c='green', alpha=0.5,
                            label='true' if i == 0 and idx==2 else None)

                for i in range(10):
                    baseline = np.mean(true_slice[i]) if idx==1 else np.mean(pred_slice[i])
                    ax.plot(pred_slice[i] - baseline + i * step_v, linewidth=2, c='black',
                            label='pred' if i == 0 and idx==2 else None)

                if idx == 0:
                    for i in range(10):
                        ax.text(-100, i * step_v, index_to_name[selected_types[i]],
                                fontsize=24, va='center')

                ax.set_ylim([-step_v, 10 * step_v])
                ax.set_yticks([])

                # Panel labels
                ax.text(-0.12 if idx==0 else -0.05, 1.02, f'{chr(97+idx)})', transform=ax.transAxes,
                        fontsize=24, va='top')

                # Spine removal
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                if idx > 0:
                    ax.spines['left'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)

                # X-axis only for panel a
                if idx == 0:
                    ax.set_xticks([0, end_frame - start_frame])
                    ax.set_xticklabels([start_frame, end_frame], fontsize=20)
                    ax.set_xlabel('frame', fontsize=24)
                else:
                    ax.set_xticks([])

            plt.tight_layout()
            plt.subplots_adjust(left=0.05)
            plt.savefig('./fig_paper/Fig1.pdf', dpi=400, bbox_inches='tight')
            plt.savefig('./fig_paper/Fig1.png', dpi=400, bbox_inches='tight')
            plt.close()

        case 'N9_44_6':
            config_file_ = 'fly_N9_44_6'
            config_file, pre_folder = add_pre_folder(config_file_)
            config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
            config.dataset = pre_folder + config.dataset
            config.config_file = pre_folder + config_file_
            logdir = 'log/fly/fly_N9_44_6'

            # config.training.noise_model_level = 0.0
            config.simulation.visual_input_type = "DAVIS"

            data_test(
                config,
                visualize=True,
                style="white color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="full",
                sample_embedding=False,
                step=50,
                device=device,
                particle_of_interest=0,
            )
            copyfile(f'./{logdir}/results/activity_8x8_panel_comparison.png',
                     f'./{logdir}/results/activity_8x8_panel_comparison_signal_pred_wo_noise.png')


            data_test(
                config,
                visualize=True,
                style="black color name true_only",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="",
                sample_embedding=False,
                step=50,
                device=device,
                particle_of_interest=0,
            )
            copyfile(f'./{logdir}/results/activity_8x8_panel_comparison.png',
                     f'./{logdir}/results/activity_8x8_panel_comparison_signal_DAVIS_true_wo_noise.png')

            data_test(
                config,
                visualize=True,
                style="black color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="",
                sample_embedding=False,
                step=50,
                device=device,
                particle_of_interest=0,
            )
            copyfile(f'./{logdir}/results/activity_8x8_panel_comparison.png',
                     f'./{logdir}/results/activity_8x8_panel_comparison_signal_DAVIS_pred_wo_noise.png')

            data_test(
                config,
                visualize=True,
                style="black color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="test_ablation_50",
                sample_embedding=False,
                step=50,
                device=device,
                particle_of_interest=0,
            )
            copyfile(f'./{logdir}/results/activity_8x8_panel_comparison.png',
                     f'./{logdir}/results/activity_8x8_panel_comparison_signal_DAVIS_pred_abalation_50_wo_noise.png')


            os.remove(f'./{logdir}/results/activity_8x8_panel_comparison.png')

        case 'N9_51_2':
            config_file_ = 'fly_N9_51_2'
            config_file, pre_folder = add_pre_folder(config_file_)
            config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
            config.dataset = pre_folder + config.dataset
            config.config_file = pre_folder + config_file_
            logdir = 'log/fly/fly_N9_51_2'

            data_test(
                config,
                visualize=True,
                style="black color name true_only",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="",
                sample_embedding=False,
                step=50,
                device=device,
                particle_of_interest=0,
            )
            copyfile(f'./{logdir}/results/activity_8x8_panel_comparison.png',
                     f'./{logdir}/results/activity_8x8_panel_comparison_true_signal.png')
            data_test(
                config,
                visualize=True,
                style="black color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="",
                sample_embedding=False,
                step=50,
                device=device,
                particle_of_interest=0,
            )
            copyfile(f'./{logdir}/results/activity_8x8_panel_comparison.png',
                     f'./{logdir}/results/activity_8x8_panel_comparison_signal_pred.png')
            data_test(
                config,
                visualize=True,
                style="black color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="test_ablation_50",
                sample_embedding=False,
                step=50,
                device=device,
                particle_of_interest=0,
            )
            copyfile(f'./{logdir}/results/activity_8x8_panel_comparison.png',
                     f'./{logdir}/results/activity_8x8_panel_comparison_signal_pred_abalation_50.png')
            copyfile('overlay_all_W.png', f'./{logdir}/results/overlay_all_W.png')
            os.remove('overlay_all_W.png')

            config.simulation.visual_input_type = ""
            data_test(
                config,
                visualize=True,
                style="black color name true_only",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="",
                sample_embedding=False,
                step=50,
                device=device,
                particle_of_interest=0,
            )
            copyfile(f'./{logdir}/results/activity_8x8_panel_comparison.png',
                     f'./{logdir}/results/activity_8x8_panel_comparison_signal_optical_flow.png')
            data_test(
                config,
                visualize=True,
                style="black color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="",
                sample_embedding=False,
                step=50,
                device=device,
                particle_of_interest=0,
            )
            copyfile(f'./{logdir}/results/activity_8x8_panel_comparison.png',
                     f'./{logdir}/results/activity_8x8_panel_comparison_signal_pred_optical_flow.png')
            os.remove(f'./{logdir}/results/activity_8x8_panel_comparison.png')

        case 'N9_22_10':
            config_file_ = 'fly_N9_22_10'
            config_file, pre_folder = add_pre_folder(config_file_)
            config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
            config.dataset = pre_folder + config.dataset
            config.config_file = pre_folder + config_file_
            logdir = 'log/fly/fly_N9_22_10'
            config.simulation.visual_input_type = "DAVIS"
            data_test(
                config,
                visualize=True,
                style="white color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="full",
                sample_embedding=False,
                step=25000,
                device=device,
                particle_of_interest=0,
            )
            copyfile(f'./{logdir}/results/activity_8x8_panel_comparison.png',
                     f'./{logdir}/results/activity_8x8_panel_comparison_DAVIS.png')
            copyfile(f'./{logdir}/results/activity_5x4_panel_comparison.png',
                     f'./{logdir}/results/activity_5x4_panel_comparison_DAVIS.png')
            data_test(
                config,
                visualize=True,
                style="white color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="full test_ablation_50",
                sample_embedding=False,
                step=50,
                device=device,
                particle_of_interest=0,
            )
            copyfile(f'./{logdir}/results/activity_8x8_panel_comparison.png',
                     f'./{logdir}/results/activity_8x8_panel_comparison_DAVIS_ablation_50.png')
            copyfile(f'./{logdir}/results/activity_5x4_panel_comparison.png',
                     f'./{logdir}/results/activity_5x4_panel_comparison_DAVIS_ablation_50.png')
            data_test(
                config,
                visualize=True,
                style="black color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="test_ablation_50",
                sample_embedding=False,
                step=50,
                device=device,
                particle_of_interest=0,
            )
            copyfile(f'./{logdir}/results/activity_8x8_panel_comparison.png',
                     f'./{logdir}/results/activity_8x8_panel_comparison_signal_pred_abalation_50.png')

            config.simulation.visual_input_type = "DAVIS"
            data_test(
                config,
                visualize=True,
                style="black color name true_only",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="",
                sample_embedding=False,
                step=50,
                device=device,
                particle_of_interest=0,
            )
            copyfile(f'./{logdir}/results/activity_8x8_panel_comparison.png',
                     f'./{logdir}/results/activity_8x8_panel_comparison_signal_DAVIS.png')
            data_test(
                config,
                visualize=True,
                style="black color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="",
                sample_embedding=False,
                step=50,
                device=device,
                particle_of_interest=0,
            )
            copyfile(f'./{logdir}/results/activity_8x8_panel_comparison.png',
                     f'./{logdir}/results/activity_8x8_panel_comparison_signal_pred_DAVIS.png')

            os.remove(f'./{logdir}/results/activity_8x8_panel_comparison.png')



        case 'new_network_1':
            config_file_ = 'signal_N2_3'
            config_file, pre_folder = add_pre_folder(config_file_)
            config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
            config.dataset = pre_folder + config.dataset
            config.config_file = pre_folder + config_file_
            data_test(
                config,
                visualize=True,
                style="white latex color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="test_ablation_0",
                sample_embedding=False,
                step=100,
                device=device,
                particle_of_interest=0,
                new_params = [0.25, 10, 20, 30, 60],
            )

        case 'new_network_2':
            config_file_ = 'signal_N2_3'
            config_file, pre_folder = add_pre_folder(config_file_)
            config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
            config.dataset = pre_folder + config.dataset
            config.config_file = pre_folder + config_file_
            data_test(
                config,
                visualize=True,
                style="white latex color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="test_ablation_0",
                sample_embedding=False,
                step=100,
                device=device,
                particle_of_interest=0,
                new_params = [0.5, 60, 40, 0, 0],
            )

        case 'ablation_weights':
            config_file_ = 'signal_N2_3'
            config_file, pre_folder = add_pre_folder(config_file_)
            config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
            config.dataset = pre_folder + config.dataset
            config.config_file = pre_folder + config_file_
            data_test(
                config,
                visualize=True,
                style="white latex color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="test_ablation_0",
                sample_embedding=False,
                step=100,
                device=device,
                particle_of_interest=0,
            )
            data_test(
                config,
                visualize=True,
                style="white latex color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="test_ablation_50",
                sample_embedding=False,
                step=100,
                device=device,
                particle_of_interest=0,
            )
            data_test(
                config,
                visualize=True,
                style="white latex color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="test_ablation_90",
                sample_embedding=False,
                step=100,
                device=device,
                particle_of_interest=0,
            )

        case 'ablation_cells':
            config_file_ = 'signal_N2_3'
            config_file, pre_folder = add_pre_folder(config_file_)
            config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
            config.dataset = pre_folder + config.dataset
            config.config_file = pre_folder + config_file_
            data_test(
                config,
                visualize=True,
                style="white latex color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="test_inactivity_0",
                sample_embedding=False,
                step=100,
                device=device,
                particle_of_interest=0,
            )
            data_test(
                config,
                visualize=True,
                style="white latex color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="test_inactivity_25",
                sample_embedding=False,
                step=100,
                device=device,
                particle_of_interest=0,
            )
            data_test(
                config,
                visualize=True,
                style="white latex color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="test_inactivity_50",
                sample_embedding=False,
                step=100,
                device=device,
                particle_of_interest=0,
            )

        case 'permutation_types':
            config_file_ = 'signal_N2_3'
            config_file, pre_folder = add_pre_folder(config_file_)
            config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
            config.dataset = pre_folder + config.dataset
            config.config_file = pre_folder + config_file_
            data_test(
                config,
                visualize=True,
                style="white latex color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="test_permutation_0",
                sample_embedding=False,
                step=100,
                device=device,
                particle_of_interest=0,
            )
            data_test(
                config,
                visualize=True,
                style="white latex color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="test_permutation_50",
                sample_embedding=False,
                step=100,
                device=device,
                particle_of_interest=0,
            )
            data_test(
                config,
                visualize=True,
                style="white latex color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="test_permutation_90",
                sample_embedding=False,
                step=100,
                device=device,
                particle_of_interest=0,
            )

        case 'weight_vs_noise':

            config_list = ['fly_N9_44_15', 'fly_N9_44_16', 'fly_N9_44_17', 'fly_N9_44_18', 'fly_N9_44_19', 'fly_N9_44_20', 'fly_N9_44_21', 'fly_N9_44_22', 'fly_N9_44_23', 'fly_N9_44_24', 'fly_N9_44_25', 'fly_N9_44_26']
            compare_experiments(config_list,'training.noise_model_level')

            # copy file ising_comparison_noise level.png
            shutil.copy('fig/ising_comparison_noise level.png', 'fig_paper/ising_comparison_noise_level.png')
            shutil.copy('fig/gnn_comparison_noise_model_level.png', 'fig_paper/gnn_comparison_noise_model_level.png')

        case 'correction_weight':

            config_file_ = 'fly_N9_22_10'
            config_file, pre_folder = add_pre_folder(config_file_)
            config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
            config.dataset = pre_folder + config.dataset
            config.config_file = pre_folder + config_file_
            print('figure correction_weight...')

            data_plot(config=config, config_file=config_file, epoch_list=['best'], style='white color', extended='plots', device=device)

            log_dir = 'log/fly/fly_N9_22_10'
            config_indices = '18_4_0'

            fig = plt.figure(figsize=(12, 10))

            ax1 = fig.add_subplot(3, 3, 1)
            panel_pic_path =f"./{log_dir}/results/edge_functions_{config_indices}_all.png"
            img = imageio.imread(panel_pic_path)
            plt.imshow(img)
            plt.axis('off')
            ax1.text(0.1, 1.01, 'a)', transform=ax1.transAxes, fontsize=18, va='bottom', ha='right')

            ax2 = fig.add_subplot(3, 3, 2)
            panel_pic_path =f"./{log_dir}/results/edge_functions_{config_indices}_domain.png"
            img = imageio.imread(panel_pic_path)
            plt.imshow(img)
            plt.axis('off')
            ax2.text(0.1, 1.01, 'b)', transform=ax2.transAxes, fontsize=18, va='bottom', ha='right')

            ax3 = fig.add_subplot(3, 3, 3)
            panel_pic_path = f"./{log_dir}/results/edge_function_slope_{config_indices}.png"
            img = imageio.imread(panel_pic_path)
            plt.imshow(img)
            plt.axis('off')
            ax3.text(0.1, 1.01, 'c)', transform=ax3.transAxes, fontsize=18, va='bottom', ha='right')

            # Second row
            ax4 = fig.add_subplot(3, 3, 4)
            panel_pic_path = f"./{log_dir}/results/phi_functions_{config_indices}_all.png"

            img = imageio.imread(panel_pic_path)
            plt.imshow(img)
            plt.axis('off')
            ax4.text(0.1, 1.01, 'd)', transform=ax4.transAxes, fontsize=18, va='bottom', ha='right')

            ax5 = fig.add_subplot(3, 3, 5)
            panel_pic_path = f"./{log_dir}/results/phi_functions_{config_indices}_domain.png"
            img = imageio.imread(panel_pic_path)
            plt.imshow(img)
            plt.axis('off')
            ax5.text(0.1, 1.01, 'e)', transform=ax5.transAxes, fontsize=18, va='bottom', ha='right')

            ax6 = fig.add_subplot(3, 3, 6)
            panel_pic_path = f"./{log_dir}/results/phi_functions_{config_indices}_params.png"
            img = imageio.imread(panel_pic_path)
            plt.imshow(img)
            plt.axis('off')
            ax6.text(0.1, 1.01, 'f)', transform=ax6.transAxes, fontsize=18, va='bottom', ha='right')

            # Third row
            ax7 = fig.add_subplot(3, 3, 7)
            panel_pic_path = f"./{log_dir}/results/comparison_raw.png"
            if os.path.exists(panel_pic_path):
                img = imageio.imread(panel_pic_path)
                plt.imshow(img)
            else:
                plt.text(0.5, 0.5, 'Neuron\nEmbedding', ha='center', va='center', fontsize=16, transform=ax7.transAxes)
            plt.axis('off')
            ax7.text(0.1, 1.01, 'g)', transform=ax7.transAxes, fontsize=18, va='bottom', ha='right')

            ax8 = fig.add_subplot(3, 3, 8)
            panel_pic_path = f"./{log_dir}/results/comparison_rj.png"
            if os.path.exists(panel_pic_path):
                img = imageio.imread(panel_pic_path)
                plt.imshow(img)
            else:
                plt.text(0.5, 0.5, 'Type\nReconstruction', ha='center', va='center', fontsize=16, transform=ax8.transAxes)
            plt.axis('off')
            ax8.text(0.1, 1.01, 'h)', transform=ax8.transAxes, fontsize=18, va='bottom', ha='right')

            ax9 = fig.add_subplot(3, 3, 9)
            panel_pic_path = f"./{log_dir}/results/corrected_comparison.png"
            if os.path.exists(panel_pic_path):
                img = imageio.imread(panel_pic_path)
                plt.imshow(img)
            else:
                plt.text(0.5, 0.5, 'Reconstruction\nCorrelations', ha='center', va='center', fontsize=16, transform=ax9.transAxes)
            plt.axis('off')
            ax9.text(0.1, 1.01, 'i)', transform=ax9.transAxes, fontsize=18, va='bottom', ha='right')

            plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02, wspace=0.02, hspace=0.04)
            plt.savefig("./fig_paper/figure_correction_weight.png", dpi=300, bbox_inches='tight')
            plt.close()

        case 'correction_weight_noise':

            config_file_ = 'fly_N9_44_6'
            config_file, pre_folder = add_pre_folder(config_file_)
            config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
            config.dataset = pre_folder + config.dataset
            config.config_file = pre_folder + config_file_
            print('figure correction_weight...')

            data_plot(config=config, config_file=config_file, epoch_list=['best'], style='black color', extended='plots', device=device)

            log_dir = 'log/fly/fly_N9_44_6'
            config_indices = '44_6'

            fig = plt.figure(figsize=(10, 9))

            ax1 = fig.add_subplot(3, 3, 1)
            panel_pic_path =f"./{log_dir}/results/edge_functions_{config_indices}_all.png"
            img = imageio.imread(panel_pic_path)
            plt.imshow(img)
            plt.axis('off')
            ax1.text(0.1, 1.01, 'a)', transform=ax1.transAxes, fontsize=18, va='bottom', ha='right')

            ax2 = fig.add_subplot(3, 3, 2)
            panel_pic_path =f"./{log_dir}/results/edge_functions_{config_indices}_domain.png"
            img = imageio.imread(panel_pic_path)
            plt.imshow(img)
            plt.axis('off')
            ax2.text(0.1, 1.01, 'b)', transform=ax2.transAxes, fontsize=18, va='bottom', ha='right')

            ax3 = fig.add_subplot(3, 3, 3)
            panel_pic_path = f"./{log_dir}/results/edge_function_slope_{config_indices}.png"
            img = imageio.imread(panel_pic_path)
            plt.imshow(img)
            plt.axis('off')
            ax3.text(0.1, 1.01, 'c)', transform=ax3.transAxes, fontsize=18, va='bottom', ha='right')

            # Second row
            ax4 = fig.add_subplot(3, 3, 4)
            panel_pic_path = f"./{log_dir}/results/phi_functions_{config_indices}_all.png"

            img = imageio.imread(panel_pic_path)
            plt.imshow(img)
            plt.axis('off')
            ax4.text(0.1, 1.01, 'd)', transform=ax4.transAxes, fontsize=18, va='bottom', ha='right')

            ax5 = fig.add_subplot(3, 3, 5)
            panel_pic_path = f"./{log_dir}/results/phi_functions_{config_indices}_domain.png"
            img = imageio.imread(panel_pic_path)
            plt.imshow(img)
            plt.axis('off')
            ax5.text(0.1, 1.01, 'e)', transform=ax5.transAxes, fontsize=18, va='bottom', ha='right')

            ax6 = fig.add_subplot(3, 3, 6)
            panel_pic_path = f"./{log_dir}/results/phi_functions_{config_indices}_params.png"
            img = imageio.imread(panel_pic_path)
            plt.imshow(img)
            plt.axis('off')
            ax6.text(0.1, 1.01, 'f)', transform=ax6.transAxes, fontsize=18, va='bottom', ha='right')

            # Third row
            ax7 = fig.add_subplot(3, 3, 7)
            panel_pic_path = f"./{log_dir}/results/comparison_raw.png"
            if os.path.exists(panel_pic_path):
                img = imageio.imread(panel_pic_path)
                plt.imshow(img)
            else:
                plt.text(0.5, 0.5, 'Neuron\nEmbedding', ha='center', va='center', fontsize=16, transform=ax7.transAxes)
            plt.axis('off')
            ax7.text(0.1, 1.01, 'g)', transform=ax7.transAxes, fontsize=18, va='bottom', ha='right')

            ax8 = fig.add_subplot(3, 3, 8)
            panel_pic_path = f"./{log_dir}/results/comparison_rj.png"
            if os.path.exists(panel_pic_path):
                img = imageio.imread(panel_pic_path)
                plt.imshow(img)
            else:
                plt.text(0.5, 0.5, 'Type\nReconstruction', ha='center', va='center', fontsize=16, transform=ax8.transAxes)
            plt.axis('off')
            ax8.text(0.1, 1.01, 'h)', transform=ax8.transAxes, fontsize=18, va='bottom', ha='right')

            ax9 = fig.add_subplot(3, 3, 9)
            panel_pic_path = f"./{log_dir}/results/corrected_comparison.png"
            if os.path.exists(panel_pic_path):
                img = imageio.imread(panel_pic_path)
                plt.imshow(img)
            else:
                plt.text(0.5, 0.5, 'Reconstruction\nCorrelations', ha='center', va='center', fontsize=16, transform=ax9.transAxes)
            plt.axis('off')
            ax9.text(0.1, 1.01, 'i)', transform=ax9.transAxes, fontsize=18, va='bottom', ha='right')

            plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02, wspace=0.02, hspace=0.04)
            plt.savefig("./fig_paper/figure_correction_weight_noise.png", dpi=300, bbox_inches='tight')
            plt.close()










if __name__ == '__main__':

    warnings.filterwarnings("ignore", category=FutureWarning)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


    print(' ')
    print(f'device {device}')

    # try:
    #     matplotlib.use("Qt5Agg")
    # except:
    #     pass


    # config_list = ['signal_N2_1', 'signal_N2_2'] #, 'signal_N2_3', 'signal_N2_4', 'signal_N2_5', 'signal_N2_6', 'signal_N2_7', 'signal_N2_8']

    # config_list = ['fly_N9_55_1', 'fly_N9_55_2', 'fly_N9_55_3', 'fly_N9_55_4', 'fly_N9_55_5', 'fly_N9_55_6', 'fly_N9_55_7', 'fly_N9_55_8', 'fly_N9_55_9', 'fly_N9_55_10', 'fly_N9_55_11', 'fly_N9_55_12']
    # compare_experiments(config_list, None)

    #
    #  #, 'fly_N9_22_11', 'fly_N9_22_12', 'fly_N9_22_13', 'fly_N9_22_14', 'fly_N9_22_15', 'fly_N9_22_16', 'fly_N9_22_17',
    # config_list = ['fly_N9_44_16', 'fly_N9_44_17', 'fly_N9_44_18', 'fly_N9_44_19', 'fly_N9_44_20', 'fly_N9_44_21', 'fly_N9_44_22', 'fly_N9_44_23', 'fly_N9_44_24', 'fly_N9_44_25', 'fly_N9_44_26']
    # compare_experiments(config_list,'training.noise_model_level')

    # config_list = ['fly_N9_44_24']
    # config_list = ['fly_N9_51_2']
    config_list = ['fly_N9_62_5_1','fly_N9_62_5_2']

    for config_file_ in config_list:
        print(' ')
        config_file, pre_folder = add_pre_folder(config_file_)
        config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
        config.dataset = pre_folder + config.dataset
        config.config_file = pre_folder + config_file_
        print(f'\033[94mconfig_file  {config.config_file}\033[0m')
        folder_name = './log/' + pre_folder + '/tmp_results/'
        os.makedirs(folder_name, exist_ok=True)
        data_plot(config=config, config_file=config_file, epoch_list=['best'], style='black color', extended='plots', device=device)

    # compare_experiments(config_list, 'training.batch_size')

    # get_figures('weight_vs_noise')
    # get_figures('correction_weight')
    # get_figures('correction_weight_noise')

    # get_figures('ablation_weights')]
    # get_figures('permutation_types')
    # get_figures('new_network_1')
    # get_figures('new_network_2')

    # get_figures('extra_edges')
    # get_figures('N9_22_10')
    # get_figures('results_22_10')
    # get_figures('results_44_24')
    # get_figures('N9_44_6')
    # get_figures('N9_51_2')

    # get_figures('results_51_2')
    # get_figures('figure_1_cosyne_2026')



