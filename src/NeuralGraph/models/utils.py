import matplotlib.pyplot as plt
import os
import torch
import torch_geometric.data as data

from NeuralGraph.models import Signal_Propagation
from NeuralGraph.models.Signal_Propagation_MLP import Signal_Propagation_MLP
from NeuralGraph.utils import to_numpy, fig_init, map_matrix, choose_boundary_values
import warnings
import numpy as np

# Optional import
try:
    import umap
except ImportError:
    umap = None

import seaborn as sns
from scipy.optimize import curve_fit
import json
from pathlib import Path
from collections import Counter

def linear_model(x, a, b):
    return a * x + b

def get_embedding(model_a=None, dataset_number = 0):
    embedding = []
    embedding.append(model_a[dataset_number])
    embedding = to_numpy(torch.stack(embedding).squeeze())

    return embedding

def get_embedding_time_series(model=None, dataset_number=None, cell_id=None, n_neurons=None, n_frames=None, has_cell_division=None):
    embedding = []
    embedding.append(model.a[dataset_number])
    embedding = to_numpy(torch.stack(embedding).squeeze())

    indexes = np.arange(n_frames) * n_neurons + cell_id

    return embedding[indexes]

def get_type_time_series(new_labels=None, dataset_number=None, cell_id=None, n_neurons=None, n_frames=None, has_cell_division=None):

    indexes = np.arange(n_frames) * n_neurons + cell_id

    return new_labels[indexes]

def get_in_features_update(rr=None, model=None, embedding = None, device=None):

    n_neurons = model.n_neurons
    model_update_type = model.update_type

    if embedding == None:
        embedding = model.a[0:n_neurons]
        if model.embedding_trial:
            embedding = torch.cat((embedding, model.b[0].repeat(n_neurons, 1)), dim=1)


    if rr == None:
        if 'generic' in model_update_type:
            if 'excitation' in model_update_type:
                in_features = torch.cat((
                    torch.zeros((n_neurons, 1), device=device),
                    embedding,
                    torch.zeros((n_neurons, 1), device=device),
                    torch.ones((n_neurons, 1), device=device),
                    torch.zeros((n_neurons, model.excitation_dim), device=device)
                ), dim=1)
            else:
                in_features = torch.cat((
                    torch.zeros((n_neurons, 1), device=device),
                    embedding,
                    torch.ones((n_neurons, 1), device=device),
                    torch.ones((n_neurons, 1), device=device)
                ), dim=1)
        else:
            in_features = torch.cat((torch.zeros((n_neurons, 1), device=device), embedding), dim=1)
    else:
        if 'generic' in model_update_type:
            if 'excitation' in model_update_type:
                in_features = torch.cat((
                    rr,
                    embedding,
                    torch.zeros((rr.shape[0], 1), device=device),
                    torch.ones((rr.shape[0], 1), device=device),
                    torch.zeros((rr.shape[0], model.excitation_dim), device=device)
                ), dim=1)
            else:
                in_features = torch.cat((
                    rr,
                    embedding,
                    torch.ones((rr.shape[0], 1), device=device),
                    torch.ones((rr.shape[0], 1), device=device)
                ), dim=1)
        else:
            in_features = torch.cat((rr, embedding), dim=1)

    return in_features

def get_in_features_lin_edge(x, model, model_config, xnorm, n_neurons, device):

    signal_model_name = model_config.signal_model_name

    if signal_model_name in ['PDE_N4', 'PDE_N7', 'PDE_N11']:
        in_features_prev = torch.cat((x[:n_neurons, 6:7] - xnorm / 150, model.a[:n_neurons]), dim=1)
        in_features = torch.cat((x[:n_neurons, 6:7], model.a[:n_neurons]), dim=1)
        in_features_next = torch.cat((x[:n_neurons, 6:7] + xnorm / 150, model.a[:n_neurons]), dim=1)
        if model.embedding_trial:
            in_features_prev = torch.cat((in_features_prev, model.b[0].repeat(n_neurons, 1)), dim=1)
            in_features = torch.cat((in_features, model.b[0].repeat(n_neurons, 1)), dim=1)
            in_features_next = torch.cat((in_features_next, model.b[0].repeat(n_neurons, 1)), dim=1)
    elif signal_model_name == 'PDE_N5':
        if model.embedding_trial:
            in_features = torch.cat((x[:n_neurons, 6:7], model.a[:n_neurons], model.b[0].repeat(n_neurons, 1), model.a[:n_neurons], model.b[0].repeat(n_neurons, 1)), dim=1)
            in_features_next = torch.cat((x[:n_neurons, 6:7] + xnorm / 150, model.a[:n_neurons], model.b[0].repeat(n_neurons, 1), model.a[:n_neurons], model.b[0].repeat(n_neurons, 1)), dim=1)
        else:
            in_features = torch.cat((x[:n_neurons, 6:7], model.a[:n_neurons], model.a[:n_neurons]), dim=1)
            in_features_next = torch.cat((x[:n_neurons, 6:7] + xnorm / 150, model.a[:n_neurons], model.a[:n_neurons]), dim=1)
    elif ('PDE_N9_A' in signal_model_name) | (signal_model_name == 'PDE_N9_C') | (signal_model_name == 'PDE_N9_D') :
        in_features = torch.cat((x[:, 3:4], model.a), dim=1)
        in_features_next = torch.cat((x[:,3:4] * 1.05, model.a), dim=1)
    elif signal_model_name == 'PDE_N9_B':
        perm_indices = torch.randperm(n_neurons, device=model.a.device)
        in_features = torch.cat((x[:, 3:4], x[:, 3:4], model.a, model.a[perm_indices]), dim=1)
        in_features_next = torch.cat((x[:, 3:4], x[:, 3:4] * 1.05, model.a, model.a[perm_indices]), dim=1)
    elif signal_model_name == 'PDE_N8':
        if model.embedding_trial:
            perm_indices = torch.randperm(n_neurons, device=model.a.device)
            in_features = torch.cat((x[:n_neurons, 6:7], x[:n_neurons, 6:7], model.a[:n_neurons], model.b[0].repeat(n_neurons, 1), model.a[perm_indices[:n_neurons]], model.b[0].repeat(n_neurons, 1)), dim=1)
            in_features_next = torch.cat((x[:n_neurons, 6:7], x[:n_neurons, 6:7]*1.05, model.a[:n_neurons], model.b[0].repeat(n_neurons, 1), model.a[perm_indices[:n_neurons]], model.b[0].repeat(n_neurons, 1)), dim=1)
        else:
            perm_indices = torch.randperm(n_neurons, device=model.a.device)
            in_features = torch.cat((x[:n_neurons, 6:7], x[:n_neurons, 6:7], model.a[:n_neurons], model.a[perm_indices[:n_neurons]]), dim=1)
            in_features_next = torch.cat((x[:n_neurons, 6:7], x[:n_neurons, 6:7] * 1.05, model.a[:n_neurons], model.a[perm_indices[:n_neurons]]), dim=1)
    else:
        in_features = x[:n_neurons, 6:7]
        in_features_next = x[:n_neurons, 6:7] + xnorm / 150
        in_features_prev = x[:n_neurons, 6:7] - xnorm / 150

    return in_features, in_features_next

def get_in_features(rr=None, embedding=None, model=[], model_name = [], max_radius=[]):

    if model.embedding_trial:
        embedding = torch.cat((embedding, model.b[0].repeat(embedding.shape[0], 1)), dim=1)

    match model_name:
        case 'PDE_A' | 'PDE_Cell_A':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding), dim=1)
        case 'PDE_ParticleField_A':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding), dim=1)
        case 'PDE_A_bis':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding, embedding), dim=1)
        case 'PDE_B' | 'PDE_Cell_B':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     torch.abs(rr[:, None]) / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                     0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
        case 'PDE_ParticleField_B':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                     0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
        case 'PDE_GS':
            in_features = torch.cat(
                (rr[:, None] / max_radius, 0 * rr[:, None], rr[:, None] / max_radius, 10 ** embedding), dim=1)
        case 'PDE_G':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, 0 * rr[:, None],
                                     0 * rr[:, None],
                                     0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
        case 'PDE_E':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding, embedding), dim=1)
        case 'PDE_N2' | 'PDE_N3' | 'PDE_N6' :
            in_features = rr[:, None]
        case 'PDE_N4' | 'PDE_N7' | 'PDE_N11':
            in_features = torch.cat((rr[:, None], embedding), dim=1)
        case 'PDE_N8':
            in_features = torch.cat((rr[:, None]*0, rr[:, None], embedding, embedding), dim=1)
        case 'PDE_N5':
            in_features = torch.cat((rr[:, None], embedding, embedding), dim=1)
        case 'PDE_K':
            in_features = torch.cat((0 * rr[:, None], rr[:, None] / max_radius), dim=1)
        case 'PDE_F':
            in_features = torch.cat((0 * rr[:, None], rr[:, None] / max_radius, rr[:, None] / max_radius, embedding, embedding), dim=-1)
        case 'PDE_M':
            in_features = torch.cat((rr[:, None] / max_radius, rr[:, None] / max_radius, embedding, embedding), dim=-1)

    return in_features

def plot_training_flyvis(x_list, model, config, epoch, N, log_dir, device, cmap, type_list,
                         gt_weights, edges, n_neurons=None, n_neuron_types=None):
    signal_model_name = config.graph_model.signal_model_name

    if n_neurons is None:
        n_neurons = len(type_list)


    plt.style.use('default')

    # Plot 1: Embedding scatter plot
    plt.figure(figsize=(8, 8))
    for n in range(n_neuron_types):
        pos = torch.argwhere(type_list == n)
        plt.scatter(to_numpy(model.a[pos, 0]), to_numpy(model.a[pos, 1]), s=5, color=cmap.color(n), alpha=0.7, edgecolors='none')
    plt.xlabel('embedding 0', fontsize=18)
    plt.ylabel('embedding 1', fontsize=18)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/embedding/{epoch}_{N}.tif", dpi=87)
    plt.close()

    x_data = to_numpy(gt_weights)
    y_data = to_numpy(model.W.squeeze())
    lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
    residuals = y_data - linear_model(x_data, *lin_fit)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    # print(f'R^2$: {np.round(r_squared, 3)}  slope: {np.round(lin_fit[0], 2)}')


    # Check Dale's Law for learned weights
    dale_results = check_dales_law(
        edges=edges,
        weights=model.W,
        type_list=type_list,
        n_neurons=n_neurons,
        verbose=False,
        logger=None
    )

    # Plot 2: Weight comparison scatter plot
    plt.figure(figsize=(8, 8))

    plt.scatter(to_numpy(gt_weights), to_numpy(model.W.squeeze()), s=0.1, c='k', alpha=0.01)
    plt.xlabel(r'true $W_{ij}$', fontsize=18)
    plt.ylabel(r'learned $W_{ij}$', fontsize=18)

    # Add R² and slope
    plt.text(-0.9, 4.5, f'$R^2$: {np.round(r_squared, 3)}\nslope: {np.round(lin_fit[0], 2)}', fontsize=12)

    # Add Dale's Law statistics
    dale_text = (f"Excitatory neurons (all W>0): {dale_results['n_excitatory']} "
                 f"({100*dale_results['n_excitatory']/n_neurons:.1f}%)\n"
                 f"Inhibitory neurons (all W<0): {dale_results['n_inhibitory']} "
                 f"({100*dale_results['n_inhibitory']/n_neurons:.1f}%)\n"
                 f"Mixed/zero neurons (violates Dale's Law): {dale_results['n_mixed']} "
                 f"({100*dale_results['n_mixed']/n_neurons:.1f}%)")
    plt.text(-0.9, -4.5, dale_text, fontsize=8, verticalalignment='bottom')

    plt.xlim([-1, 5])
    plt.ylim([-5, 5])
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/matrix/comparison_{epoch}_{N}.tif",
                dpi=87, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Plot 3: Edge function visualization
    plt.figure(figsize=(8, 8))
    rr = torch.linspace(config.plotting.xlim[0], config.plotting.xlim[1], 1000, device=device)
    for n in range(n_neurons):
        embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
        if ('PDE_N9_A' in signal_model_name) | ('PDE_N9_C' in signal_model_name) | ('PDE_N9_D' in signal_model_name):
            in_features = torch.cat((rr[:, None], embedding_,), dim=1)
        elif ('PDE_N9_B' in signal_model_name):
            in_features = torch.cat((rr[:, None] * 0, rr[:, None], embedding_, embedding_), dim=1)
        with torch.no_grad():
            func = model.lin_edge(in_features.float())
        if config.graph_model.lin_edge_positive:
            func = func ** 2
        if (n % 20 == 0):
            plt.plot(to_numpy(rr), to_numpy(func), 2,
                     color=cmap.color(to_numpy(type_list)[n].astype(int)),
                     linewidth=1, alpha=0.1)
    plt.xlim(config.plotting.xlim)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/function/lin_edge/func_{epoch}_{N}.tif", dpi=87)
    plt.close()

    # Plot 4: Phi function visualization
    plt.figure(figsize=(8, 8))
    for n in range(n_neurons):
        embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
        if ('PDE_N9_C' in signal_model_name):
            in_features = torch.cat((rr[:, None], embedding_, torch.zeros_like(rr[:, None])), dim=1)
        else:
            if model.training_time_window>0:
                in_features = torch.cat((rr[:, None].repeat(1, model.training_time_window-1), embedding_, rr[:, None] * 0, torch.zeros_like(rr[:, None]), torch.zeros((rr.shape[0],model.training_time_window))),dim=1)
            else:
                in_features = torch.cat((rr[:, None], embedding_, rr[:, None] * 0, torch.zeros_like(rr[:, None])), dim=1)
        with torch.no_grad():
            func = model.lin_phi(in_features.float())
        if (n % 20 == 0):
            plt.plot(to_numpy(rr), to_numpy(func), 2,
                     color=cmap.color(to_numpy(type_list)[n].astype(int)),
                     linewidth=1, alpha=0.1)
    plt.xlim(config.plotting.xlim)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/function/lin_phi/func_{epoch}_{N}.tif", dpi=87)
    plt.close()


def plot_training_signal(config, model, x, connectivity, log_dir, epoch, N, n_neurons, type_list, cmap, mc, device):

    if 'PDE_N3' in config.graph_model.signal_model_name:

        fig, ax = fig_init()
        plt.scatter(to_numpy(model.a[:-200, 0]), to_numpy(model.a[:-200, 1]), s=1, color='k', alpha=0.1, edgecolor='none')

    else:
        fig = plt.figure(figsize=(8, 8))
        for n in range(n_neurons):
            if x[n, 6] != config.simulation.baseline_value:
                plt.scatter(to_numpy(model.a[n, 0]), to_numpy(model.a[n, 1]), s=100,
                            color=cmap.color(int(type_list[n])), alpha=1.0, edgecolors='none')

    plt.xlabel(r'$a_0$', fontsize=48)
    plt.ylabel(r'$a_1$', fontsize=48)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/embedding/{epoch}_{N}.tif", dpi=87)
    plt.close()

    gt_weight = to_numpy(connectivity)

    if config.training.multi_connectivity:
        pred_weight = to_numpy(model.W[0, :n_neurons, :n_neurons].clone().detach())
    else:
        pred_weight = to_numpy(model.W[:n_neurons, :n_neurons].clone().detach())

    if config.simulation.n_excitatory_neurons > 0:
         gt_weight = gt_weight[:-1,:-1]
         pred_weight = pred_weight[:-1,:-1]

    if 'PDE_N11' in config.graph_model.signal_model_name:
        weight_variable = '$J_{ij}$'
        signal_variable = '$h_i$'
    else:
        weight_variable = '$W_{ij}$'
        signal_variable = '$v_i$'

    if n_neurons<1000:
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(121)
        ax = sns.heatmap(gt_weight, center=0, vmin=-0.5, vmax=0.5, square=True, cmap='bwr', cbar_kws={'fraction': 0.046})
        plt.xticks([0, n_neurons - 1], [1, n_neurons], fontsize=8)
        plt.yticks([0, n_neurons - 1], [1, n_neurons], fontsize=8)
        plt.ylabel('postsynaptic', fontsize=16)
        plt.xlabel('presynaptic', fontsize=16)
        plt.title(f'true {weight_variable}', fontsize=16)
        ax = fig.add_subplot(122)
        ax = sns.heatmap(pred_weight / 10, center=0, vmin=-0.5, vmax=0.5, square=True, cmap='bwr', cbar_kws={'fraction': 0.046})
        plt.xticks([0, n_neurons - 1], [1, n_neurons], fontsize=8)
        plt.yticks([0, n_neurons - 1], [1, n_neurons], fontsize=8)
        plt.ylabel('postsynaptic', fontsize=16)
        plt.xlabel('presynaptic', fontsize=16)
        plt.title(f'learned {weight_variable}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/matrix/matrix_{epoch}_{N}.tif", dpi=80)
        plt.close()

    fig = plt.figure(figsize=(8, 8))
    fig, ax = fig_init()
    if n_neurons<1000:
        plt.scatter(gt_weight, pred_weight / 10, s=1.0, c=mc, alpha=1.0)
    else:
        plt.scatter(gt_weight, pred_weight / 10, s=0.1, c=mc, alpha=0.1)
    plt.xlabel(r'true $J_{ij}$', fontsize=48)
    plt.ylabel(r'learned $J_{ij}$', fontsize=48)
    if n_neurons == 8000:
        plt.xlim([-0.05, 0.05])
    else:
        plt.ylim([-0.2, 0.2])
        plt.xlim([-0.2, 0.2])
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/matrix/comparison_{epoch}_{N}.tif", dpi=87)
    plt.close()

    if ('PDE_N8' in config.graph_model.signal_model_name):
        os.makedirs(f"./{log_dir}/tmp_training/matrix/larynx", exist_ok=True)
        data_folder_name = f'./graphs_data/{config.dataset}/'
        with open(data_folder_name+"all_neuron_list.json", "r") as f:
            all_neuron_list = json.load(f)
        with open(data_folder_name+"larynx_neuron_list.json", "r") as f:
            larynx_neuron_list = json.load(f)
        larynx_pred_weight, index_larynx = map_matrix(larynx_neuron_list, all_neuron_list, pred_weight)
        larynx_gt_weight, _ = map_matrix(larynx_neuron_list, all_neuron_list, gt_weight)
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(121)
        ax = sns.heatmap(larynx_pred_weight, center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046})
        ax.set_xticks(range(len(larynx_neuron_list)))
        ax.set_xticklabels(larynx_neuron_list, fontsize=12, rotation=90)
        ax.set_yticks(range(len(larynx_neuron_list)))
        ax.set_yticklabels(larynx_neuron_list, fontsize=12)
        plt.ylabel('postsynaptic')
        plt.xlabel('presynaptic')
        ax = fig.add_subplot(122)
        ax = sns.heatmap(larynx_gt_weight, center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046})
        ax.set_xticks(range(len(larynx_neuron_list)))
        ax.set_xticklabels(larynx_neuron_list, fontsize=12, rotation=90)
        ax.set_yticks(range(len(larynx_neuron_list)))
        ax.set_yticklabels(larynx_neuron_list, fontsize=12)
        plt.ylabel('postsynaptic')
        plt.xlabel('presynaptic')
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/matrix/larynx/matrix_{epoch}_{N}.tif", dpi=87)
        plt.close()

        rr = torch.linspace(config.plotting.xlim[0], config.plotting.xlim[1], 1000, device=device)
        fig = plt.figure(figsize=(8, 8))
        for idx, k in enumerate(np.linspace(config.plotting.xlim[0], config.plotting.xlim[1], 10)):  # Corrected step size to generate 13 evenly spaced values
            for n in range(0, n_neurons, 4):
                embedding_i = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                embedding_j = model.a[np.random.randint(n_neurons), :] * torch.ones(
                    (1000, config.graph_model.embedding_dim), device=device)
                if model.embedding_trial:
                    in_features = torch.cat((torch.ones_like(rr[:, None]) * k, rr[:, None], embedding_i, embedding_j,model.b[0].repeat(1000, 1)), dim=1)
                else:
                    in_features = torch.cat((rr[:, None], torch.ones_like(rr[:, None]) * k, embedding_i, embedding_j), dim=1)
                with torch.no_grad():
                    func = model.lin_edge(in_features.float())
                if config.graph_model.lin_edge_positive:
                    func = func ** 2
                plt.plot(to_numpy(rr - k), to_numpy(func), 2, color=cmap.color(idx), linewidth=2, alpha=0.25)
        plt.xlabel(r'$x_i-x_j$', fontsize=18)
        plt.ylabel(r'$MLP_1(a_i, a_j, x_i, x_j)$', fontsize=18)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/function/lin_edge/func_{epoch}_{N}.tif", dpi=87)
        plt.close()

    else:

        fig = plt.figure(figsize=(8, 8))
        rr = torch.linspace(config.plotting.xlim[0], config.plotting.xlim[1], 1000, device=device)
        func_list = []
        for n in range(n_neurons):
            if config.graph_model.signal_model_name in ['PDE_N4', 'PDE_N7', 'PDE_N11']:
                embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                if model.embedding_trial:
                    embedding_ = torch.cat((embedding_, model.b[0].repeat(1000, 1)), dim=1)
                in_features = torch.cat((rr[:, None], embedding_), dim=1)
            elif 'PDE_N5' in config.graph_model.signal_model_name:
                embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                if model.embedding_trial:
                    in_features = torch.cat(
                        (rr[:, None], embedding_, model.b[0].repeat(1000, 1), embedding_, model.b[0].repeat(1000, 1)),
                        dim=1)
                else:
                    in_features = torch.cat((rr[:, None], embedding_, embedding_), dim=1)
            elif 'PDE_N8' in config.graph_model.signal_model_name:
                embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                if model.embedding_trial:
                    in_features = torch.cat((rr[:, None] * 0, rr[:, None], embedding_, model.b[0].repeat(1000, 1),
                                             embedding_, model.b[0].repeat(1000, 1)), dim=1)
                else:
                    in_features = torch.cat((rr[:, None] * 0, rr[:, None], embedding_, embedding_), dim=1)
            else:
                in_features = rr[:, None]
            with torch.no_grad():
                func = model.lin_edge(in_features.float())
            if config.graph_model.lin_edge_positive:
                func = func ** 2
            func_list.append(to_numpy(func))
            if (n % 2 == 0):
                plt.plot(to_numpy(rr), to_numpy(func), 2, color=cmap.color(to_numpy(type_list)[n].astype(int)),
                         linewidth=2, alpha=0.25)
        plt.xlim(config.plotting.xlim)
        all_func = np.concatenate(func_list)
        plt.ylim([np.min(all_func), np.max(all_func)])
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)

        xlabel = signal_variable.replace('_i', '_j')
        if config.training.training_single_type:
            ylabel = rf'$\mathrm{{MLP_1}}({xlabel[1:-1]})$'
        else:
            ylabel = rf'$\mathrm{{MLP_1}}(a_j, {xlabel[1:-1]})$'
        plt.xlabel(xlabel, fontsize=48)
        plt.ylabel(ylabel, fontsize=48)
        plt.tight_layout()

        plt.savefig(f"./{log_dir}/tmp_training/function/lin_edge/func_{epoch}_{N}.tif", dpi=87)
        plt.close()

    rr = torch.linspace(config.plotting.xlim[0], config.plotting.xlim[1], 1000, device=device)
    fig = plt.figure(figsize=(8, 8))
    func_list = []
    for n in range(n_neurons):
        embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
        if 'generic' in config.graph_model.update_type:
            in_features = torch.cat((rr[:, None], embedding_, rr[:, None] * 0, rr[:, None] * 0), dim=1)
        else:
            in_features = torch.cat((rr[:, None], embedding_), dim=1)
        with torch.no_grad():
            func = model.lin_phi(in_features.float())
        func_list.append(to_numpy(func))
        if (n % 2 == 0):
            plt.plot(to_numpy(rr), to_numpy(func), 2,
                     color=cmap.color(to_numpy(type_list)[n].astype(int)),
                     linewidth=1, alpha=0.1)
    plt.xlim(config.plotting.xlim)
    all_func = np.concatenate(func_list)
    plt.ylim([np.min(all_func), np.max(all_func)])
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    if config.training.training_single_type:
        ylabel = rf'$\mathrm{{MLP_0}}({signal_variable[1:-1]})$'
    else:
        ylabel = rf'$\mathrm{{MLP_0}}(a_i, {signal_variable[1:-1]})$'
    plt.xlabel(signal_variable, fontsize=48)
    plt.ylabel(ylabel, fontsize=48)

    plt.tight_layout()

    plt.savefig(f"./{log_dir}/tmp_training/function/lin_phi/func_{epoch}_{N}.tif", dpi=87)
    plt.close()

    if config.simulation.n_excitatory_neurons > 0:
        gt_weight = to_numpy(connectivity[:-1,-1])
        pred_weight = to_numpy(model.W[:-1,-1])

        fig = plt.figure(figsize=(8, 8))
        fig, ax = fig_init()
        plt.scatter(gt_weight, pred_weight, s=10, c=mc)
        plt.xlabel(r'true $e_i$', fontsize=48)
        plt.ylabel(r'learned $e_i$', fontsize=48)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/matrix/comparison_ei_{epoch}_{N}.tif", dpi=87)
        plt.close()

        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(121)

        with torch.no_grad():
            kk = torch.arange(0, config.simulation.n_frames, dtype=torch.float32, device=device) / model.NNR_f_T_period
            excitation_field = model.NNR_f(kk[:,None])
            model_a = model.a[-1] * torch.ones((10000,1), device=device)
            in_features = torch.cat([excitation_field, model_a], dim=1)
            msg = model.lin_edge(in_features)

        excitation=to_numpy(msg.squeeze())


        frame_ = np.arange(0, len(excitation)) / len(excitation)
        gt_excitation=np.cos((2*np.pi)*config.simulation.oscillation_frequency*frame_)
        plt.plot(gt_excitation, c='g', linewidth=5, alpha=0.5)
        plt.plot(excitation, c=mc, linewidth=1)
        plt.xlabel('time', fontsize=48)
        plt.ylabel('excitation', fontsize=48)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        ax = fig.add_subplot(122)
        plt.plot(gt_excitation, c='g', linewidth=5, alpha=0.5)
        plt.plot(excitation, c=mc, linewidth=1)
        plt.xlabel('time', fontsize=48)
        plt.ylabel('excitation', fontsize=48)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlim([0, 2000])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/field/excitation_{epoch}_{N}.tif", dpi=87)
        plt.close()


def plot_training_signal_field(x, n_nodes, kk, time_step, x_list, run, model, field_type, model_f, edges, y_list, ynorm, delta_t, n_frames, log_dir, epoch, N, recurrent_parameters, modulation, device):


    if 'learnable_short_term_plasticity' in field_type:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(2, 2, 1)
        plt.imshow(to_numpy(modulation), aspect='auto')
        ax = fig.add_subplot(2, 2, 2)
        plt.imshow(to_numpy(model.b ** 2), aspect='auto')
        ax.text(0.01, 0.99, f'recurrent_parameter {recurrent_parameters[0]:0.3f} ', transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left', color='w')
        ax = fig.add_subplot(2, 2, 3)
        plt.scatter(to_numpy(modulation[:, np.arange(0, n_frames, n_frames//1000)]), to_numpy(model.b[:, 0:1000] ** 2), s=0.1, color='k', alpha=0.01)
        x_data = to_numpy(modulation[:, np.arange(0, n_frames, n_frames//1000)]).flatten()
        y_data = to_numpy(model.b[:, 0:1000] ** 2).flatten()
        lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
        residuals = y_data - linear_model(x_data, *lin_fit)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        ax.text(0.01, 0.99, f'$R^2$ {r_squared:0.3f}   slope {lin_fit[0]:0.3f}', transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left')
        ind_list = [10, 124, 148, 200, 250, 300]
        ax = fig.add_subplot(4, 2, 6)
        for ind in ind_list:
            plt.plot(to_numpy(modulation[ind, :]))
        ax = fig.add_subplot(4, 2, 8)
        for ind in ind_list:
            plt.plot(to_numpy(model.b[ind, :] ** 2))
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/field/field_{epoch}_{N}.tif", dpi=80)
        plt.close()

    elif ('short_term_plasticity' in field_type) | ('modulation' in field_type):
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(2, 2, 1)
        plt.imshow(to_numpy(modulation), aspect='auto')
        ax = fig.add_subplot(2, 2, 2)
        if n_frames > 1000:
            t = torch.linspace(0, 1, n_frames//100, dtype=torch.float32, device=device).unsqueeze(1)
        else:
            t = torch.linspace(0, 1, n_frames, dtype=torch.float32, device=device).unsqueeze(1)
        prediction = model_f[0](t) ** 2
        prediction = prediction.t()
        plt.imshow(to_numpy(prediction), aspect='auto', cmap='viridis')
        ax = fig.add_subplot(2, 2, 3)
        # if n_frames > 1000:
        #     ids = np.arange(0, n_frames, 100).astype(int)
        # else:
        #     ids = np.arange(0, n_frames, 1).astype(int)
        # plt.scatter(to_numpy(modulation[:, ids[:-1]]), to_numpy(prediction[:modulation.shape[0], :]), s=0.1, color='k', alpha=0.01)
        # x_data = to_numpy(modulation[:, ids[:-1]]).flatten()
        # y_data = to_numpy(prediction[:modulation.shape[0], :]).flatten()
        # lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
        # residuals = y_data - linear_model(x_data, *lin_fit)
        # ss_res = np.sum(residuals ** 2)
        # ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        # r_squared = 1 - (ss_res / ss_tot)
        # ax.text(0.01, 0.99, f'$R^2$ {r_squared:0.3f}   slope {lin_fit[0]:0.3f}', transform=ax.transAxes,
        #         verticalalignment='top', horizontalalignment='left')
        # ind_list = [10, 24, 48, 120, 150, 180]
        # ax = fig.add_subplot(4, 2, 6)
        # for ind in ind_list:
        #     plt.plot(to_numpy(modulation[ind, :]))
        # ax = fig.add_subplot(4, 2, 8)
        # for ind in ind_list:
        #     plt.plot(to_numpy(prediction[ind, :]))
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/field/field_{epoch}_{N}.tif", dpi=80)
        plt.close()

    else:
        n_nodes_per_axis = int(np.sqrt(n_nodes))
        if 'visual' in field_type:
            tmp = torch.reshape(x[:n_nodes, 8:9], (n_nodes_per_axis, n_nodes_per_axis))
        else:
            tmp = torch.reshape(x[:, 8:9], (n_nodes_per_axis, n_nodes_per_axis))
        tmp = to_numpy(torch.sqrt(tmp))
        tmp = np.rot90(tmp, k=1)
        fig = plt.figure(figsize=(12, 12))
        plt.imshow(tmp, cmap='grey')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/field/field_{epoch}_{N}.tif", dpi=80)
        plt.close()

def plot_training_signal_missing_activity(n_frames, k, x_list, baseline_value, model_missing_activity, log_dir, epoch, N, device):

        if n_frames > 1000:
            t = torch.linspace(0, 1, n_frames//100, dtype=torch.float32, device=device).unsqueeze(1)
        else:
            t = torch.linspace(0, 1, n_frames, dtype=torch.float32, device=device).unsqueeze(1)
        prediction = model_missing_activity[0](t)
        prediction = prediction.t()
        fig = plt.figure(figsize=(16, 16))
        fig.add_subplot(2, 2, 1)
        plt.title('neural field')
        plt.imshow(to_numpy(prediction), aspect='auto', cmap='viridis')
        fig.add_subplot(2, 2, 2)
        plt.title('true activity')
        activity = torch.tensor(x_list[0][:, :, 6:7], device=device)
        activity = activity.squeeze()
        activity = activity.t()
        plt.imshow(to_numpy(activity), aspect='auto', cmap='viridis')
        plt.tight_layout()
        fig.add_subplot(2, 2, 3)
        plt.title('learned missing activity')
        pos = np.argwhere(x_list[0][k][:, 6] != baseline_value)
        prediction_ = prediction.clone().detach()
        prediction_[pos[:,0]]=0
        plt.imshow(to_numpy(prediction_), aspect='auto', cmap='viridis')
        fig.add_subplot(2, 2, 4)
        plt.title('learned observed activity')
        pos = np.argwhere(x_list[0][k][:, 6] == baseline_value)
        prediction_ = prediction.clone().detach()
        prediction_[pos[:,0]]=0
        plt.imshow(to_numpy(prediction_), aspect='auto', cmap='viridis')
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/field/missing_activity_{epoch}_{N}.tif", dpi=80)
        plt.close()

def analyze_edge_function(rr=[], vizualize=False, config=None, model_MLP=[], model=None, n_nodes=0, n_neurons=None, ynorm=None, type_list=None, cmap=None, update_type=None, device=None):

    dimension = config.simulation.dimension

    if config.graph_model.particle_model_name != '':
        config_model = config.graph_model.particle_model_name
    elif config.graph_model.signal_model_name != '':
        config_model = config.graph_model.signal_model_name
    elif config.graph_model.mesh_model_name != '':
        config_model = config.graph_model.mesh_model_name

    if rr==[]:
        if 'PDE_N' in config_model:
            rr = torch.tensor(np.linspace(-5, 5, 1000)).to(device)
        else:
            rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device) # noqa: F821

    print('interaction functions ...')
    func_list = []
    for n in range(n_neurons):

        if len(model.a.shape)==3:
            model_a= model.a[1, n, :]
        else:
            model_a = model.a[n, :]

        if (update_type != 'NA') & model.embedding_trial:
            embedding_ = torch.cat((model_a, model.b[0].clone().detach().repeat(n_neurons, 1)), dim=1) * torch.ones((1000, 2*dimension), device=device)
        else:
            embedding_ = model_a * torch.ones((1000, dimension), device=device)

        if update_type == 'NA':
            in_features = get_in_features(rr=rr, embedding=embedding_, model=model, model_name=config_model, max_radius=max_radius) # noqa: F821
        else:
            in_features = get_in_features_update(rr=rr[:, None], embedding=embedding_, model=model, device=device)
        with torch.no_grad():
            func = model_MLP(in_features.float())[:, 0]

        func_list.append(func)

        should_plot = vizualize and (
                n_neurons <= 200 or
                (n % (n_neurons // 200) == 0) or
                (config.graph_model.particle_model_name == 'PDE_GS') or
                ('PDE_N' in config_model)
        )

        if should_plot:
            plt.plot(
                to_numpy(rr),
                to_numpy(func) * to_numpy(ynorm),
                2,
                color=cmap.color(type_list[n].astype(int)),
                linewidth=1,
                alpha=0.25
            )

    func_list = torch.stack(func_list)
    func_list_ = to_numpy(func_list)

    if vizualize:
        if config.graph_model.particle_model_name == 'PDE_GS':
            plt.xscale('log')
            plt.yscale('log')
        if config.graph_model.particle_model_name == 'PDE_G':
            plt.xlim([1E-3, 0.02])
        if config.graph_model.particle_model_name == 'PDE_E':
            plt.xlim([0, 0.05])
        if 'PDE_N' in config.graph_model.particle_model_name:
            plt.xlim(config.plotting.xlim)


        # ylim = [np.min(func_list_)/1.05, np.max(func_list_)*1.05]
        plt.ylim(config.plotting.ylim)

    print('UMAP reduction ...')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if func_list_.shape[0] > 1000:
            new_index = np.random.permutation(func_list_.shape[0])
            new_index = new_index[0:min(1000, func_list_.shape[0])]
            trans = umap.UMAP(n_neighbors=500, n_components=2, transform_queue_size=0, random_state=config.training.seed).fit(func_list_[new_index])
            proj_interaction = trans.transform(func_list_)
        else:
            trans = umap.UMAP(n_neighbors=50, n_components=2, transform_queue_size=0).fit(func_list_)
            proj_interaction = trans.transform(func_list_)

    return func_list, proj_interaction

def choose_training_model(model_config=None, device=None):

    dataset_name = model_config.dataset
    aggr_type = model_config.graph_model.aggr_type
    dimension = model_config.simulation.dimension

    bc_pos, bc_dpos = choose_boundary_values(model_config.simulation.boundary)

    model=[]
    model_name = model_config.graph_model.particle_model_name
    match model_name:
        case 'PDE_R':
            model = Signal_Propagation(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos,
                                     dimension=dimension)
        case 'PDE_MPM' | 'PDE_MPM_A':
            model = Signal_Propagation(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos,
                                    dimension=dimension)
        case  'PDE_Cell' | 'PDE_Cell_area':
            model = Signal_Propagation(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos, dimension=dimension)
            model.edges = []
        case 'PDE_ParticleField_A' | 'PDE_ParticleField_B':
            model = Signal_Propagation(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos, dimension=dimension)
            model.edges = []
        case 'PDE_Agents' | 'PDE_Agents_A' | 'PDE_Agents_B' | 'PDE_Agents_C':
            model = Signal_Propagation(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos, dimension=dimension)
        case 'PDE_A' | 'PDE_A_bis' | 'PDE_B' | 'PDE_B_mass' | 'PDE_B_bis' | 'PDE_E' | 'PDE_G' | 'PDE_K' | 'PDE_T':
            model = Signal_Propagation(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos, dimension=dimension)
            model.edges = []
            if 'PDE_K' in model_name:
                model.connection_matrix = torch.load(f'./graphs_data/{dataset_name}/connection_matrix_list.pt', map_location=device)
        case 'PDE_GS':
            model = Signal_Propagation(aggr_type=aggr_type, config=model_config, device=device)
            t = np.arange(model_config.simulation.n_neurons)
            t1 = np.repeat(t, model_config.simulation.n_neurons)
            t2 = np.tile(t, model_config.simulation.n_neurons)
            e = np.stack((t1, t2), axis=0)
            pos = np.argwhere(e[0, :] - e[1, :] != 0)
            e = e[:, pos]
            model.edges = torch.tensor(e, dtype=torch.long, device=device)
        case 'PDE_GS2':
            model = Signal_Propagation(aggr_type=aggr_type, config=model_config, device=device)
            t = np.arange(model_config.simulation.n_neurons)
            t1 = np.repeat(t, model_config.simulation.n_neurons)
            t2 = np.tile(t, model_config.simulation.n_neurons)
            e = np.stack((t1, t2), axis=0)
            pos = np.argwhere(e[0, :] - e[1, :] != 0)
            e = e[:, pos]
            model.edges = torch.tensor(e, dtype=torch.long, device=device)
        case 'PDE_Cell_A' | 'PDE_Cell_B' | 'PDE_Cell_B_area' | 'PDE_Cell_A_area':
            model = Signal_Propagation(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos, dimension=dimension)
        case 'PDE_F_A' |'PDE_F_B'|'PDE_F_C'|'PDE_F_D'|'PDE_F_E' :
            model = Signal_Propagation(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos, dimension=dimension)
        case 'PDE_MLPs' | 'PDE_MLPs_A' | 'PDE_MLPs_A_bis' | 'PDE_MLPs_A_ter' | 'PDE_MLPs_B'| 'PDE_MLPs_B_0' |'PDE_MLPs_B_1' | 'PDE_MLPs_B_4'| 'PDE_MLPs_B_10' |'PDE_MLPs_C' | 'PDE_MLPs_D' | 'PDE_MLPs_E' | 'PDE_MLPs_F':
            model = Signal_Propagation(aggr_type=aggr_type, config=model_config, device=device,
                                                bc_dpos=bc_dpos, dimension=dimension)
        case 'PDE_M' | 'PDE_M2':
            model = Signal_Propagation(aggr_type=aggr_type, config=model_config, bc_dpos=bc_dpos, dimension=dimension, device=device)
        case 'PDE_MM' | 'PDE_MM_1layer' | 'PDE_MM_2layers' | 'PDE_MM_3layers':
            model = Signal_Propagation(aggr_type=aggr_type, config=model_config, bc_dpos=bc_dpos, dimension=dimension, device=device)
        case 'PDE_MS':
            model = Signal_Propagation(aggr_type=aggr_type, config=model_config,bc_dpos=bc_dpos, dimension=dimension, device=device)

    model_name = model_config.graph_model.mesh_model_name
    match model_name:
        case 'DiffMesh':
            model = Signal_Propagation(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos)
            model.edges = []
        case 'WaveMesh':
            model = Signal_Propagation(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos)
            model.edges = []
        case 'WaveMeshSmooth':
            model = Signal_Propagation(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos)
            model.edges = []
        case 'RD_Mesh' | 'RD_Mesh2' | 'RD_Mesh3' | 'RD_Mesh4':
            model = Signal_Propagation(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos)
            model.edges = []

    model_name = model_config.graph_model.signal_model_name
    match model_name:
        case 'PDE_N9_MLP':
            model = Signal_Propagation_MLP(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos)
            model.edges = []
        case 'PDE_N2' | 'PDE_N3' | 'PDE_N4' | 'PDE_N5' | 'PDE_N6' | 'PDE_N7' | 'PDE_N9' | 'PDE_N8' | 'PDE_N11':
            model = Signal_Propagation(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos)
            model.edges = []
        case 'PDE_WBI':
            from NeuralGraph.models import WBI_Communication
            model = WBI_Communication(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos)
            model.edges = []

    if model==[]:
        raise ValueError(f'Unknown model {model_name}')

    return model, bc_pos, bc_dpos

def constant_batch_size(batch_size):
    def get_batch_size(epoch):
        return batch_size

    return get_batch_size

def increasing_batch_size(batch_size):
    def get_batch_size(epoch):
        return 1 if epoch < 1 else batch_size

    return get_batch_size


def set_trainable_parameters(model=[], lr_embedding=[], lr=[],  lr_update=[], lr_W=[], lr_modulation=[], learning_rate_NNR=[], learning_rate_NNR_f=[], learning_rate_NNR_E=[], learning_rate_NNR_b=[]):

    trainable_params = [param for _, param in model.named_parameters() if param.requires_grad]
    n_total_params = sum(p.numel() for p in trainable_params)

    # Only count model.a if it exists and requires gradients (not frozen by training_single_type)
    if hasattr(model, 'a') and model.a.requires_grad:
        n_total_params = n_total_params + torch.numel(model.a)


    if lr_update==[]:
        lr_update = lr

    param_groups = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if name == 'a':
                param_groups.append({'params': parameter, 'lr': lr_embedding})
            elif (name=='b') or ('lin_modulation' in name):
                param_groups.append({'params': parameter, 'lr': lr_modulation})
            elif 'lin_phi' in name:
                param_groups.append({'params': parameter, 'lr': lr_update})
            elif 'W' in name:
                param_groups.append({'params': parameter, 'lr': lr_W})
            elif 'NNR_f' in name:
                param_groups.append({'params': parameter, 'lr': learning_rate_NNR_f})
            elif 'NNR' in name:
                param_groups.append({'params': parameter, 'lr': learning_rate_NNR})
            else:
                param_groups.append({'params': parameter, 'lr': lr})

    optimizer = torch.optim.Adam(param_groups)

    return optimizer, n_total_params


def set_trainable_division_parameters(model, lr):
    trainable_params = [param for _, param in model.named_parameters() if param.requires_grad]
    n_total_params = sum(p.numel() for p in trainable_params) + torch.numel(model.t)

    embedding = model.t
    optimizer = torch.optim.Adam([embedding], lr=lr)

    _, *parameters = trainable_params
    for parameter in parameters:
        optimizer.add_param_group({'params': parameter, 'lr': lr})

    return optimizer, n_total_params

def get_index_particles(x, n_neuron_types, dimension):
    index_particles = []
    for n in range(n_neuron_types):
        if dimension == 2:
            index = np.argwhere(x[:, 5].detach().cpu().numpy() == n)
        elif dimension == 3:
            index = np.argwhere(x[:, 7].detach().cpu().numpy() == n)
        index_particles.append(index.squeeze())
    return index_particles

def get_type_list(x, dimension):
    type_list = x[:, 1 + 2 * dimension:2 + 2 * dimension].clone().detach()
    return type_list

def sample_synaptic_data_and_predict(model, x_list, edges, n_runs, n_frames, time_step, device,
                            has_missing_activity=False, model_missing_activity=None,
                            has_neural_field=False, model_f=None,
                            run=None, k=None):
    """
    Sample data from x_list and get model predictions

    Args:
        model: trained GNN model
        x_list: list of data arrays [n_runs][n_frames]
        edges: edge indices for graph
        n_runs, n_frames, time_step: data dimensions
        device: torch device
        has_missing_activity: whether to fill missing activity
        model_missing_activity: model for missing activity (if needed)
        has_neural_field: whether to compute neural field
        model_f: field model (if needed)
        run: specific run index (if None, random)
        k: specific frame index (if None, random)

    Returns:
        dict with pred, in_features, x, dataset, data_id, k_batch
    """
    # Sample random run and frame if not specified
    if run is None:
        run = np.random.randint(n_runs)
    if k is None:
        k = np.random.randint(n_frames - 4 - time_step)

    # Get data
    x = torch.tensor(x_list[run][k], dtype=torch.float32, device=device)

    # Handle missing activity if needed
    if has_missing_activity and model_missing_activity is not None:
        pos = torch.argwhere(x[:, 6] == 6)
        if len(pos) > 0:
            t = torch.tensor([k / n_frames], dtype=torch.float32, device=device)
            missing_activity = model_missing_activity[run](t).squeeze()
            x[pos, 6] = missing_activity[pos]

    # Handle neural field if needed
    if has_neural_field and model_f is not None:
        t = torch.tensor([k / n_frames], dtype=torch.float32, device=device)
        x[:, 8] = model_f[run](t) ** 2

    # Create dataset
    dataset = data.Data(x=x, edge_index=edges)
    data_id = torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * run
    k_batch = torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * k

    # Get predictions
    pred, in_features = model(dataset, data_id=data_id, k=k_batch, return_all=True)

    return {
        'pred': pred,
        'in_features': in_features,
        'x': x,
        'dataset': dataset,
        'data_id': data_id,
        'k_batch': k_batch,
        'run': run,
        'k': k
    }

def analyze_odor_responses_by_neuron(model, x_list, edges, n_runs, n_frames, time_step, device,
                                     all_neuron_list, has_missing_activity=False, model_missing_activity=None,
                                     has_neural_field=False, model_f=None, n_samples=50, run=0):
    """
    Analyze odor responses by comparing lin_phi output with and without excitation
    Returns top responding neurons by name for each odor
    """
    odor_list = ['butanone', 'pentanedione', 'NaCL']

    # Store responses: difference between excitation and baseline
    odor_responses = {odor: [] for odor in odor_list}
    valid_samples = 0

    model.eval()
    with torch.no_grad():
        sample = 0
        while valid_samples < n_samples:
            result = sample_synaptic_data_and_predict(
                model, x_list, edges, n_runs, n_frames, time_step, device,
                has_missing_activity, model_missing_activity,
                has_neural_field, model_f, run
            )

            if not (torch.isnan(result['x']).any()):
                # Get baseline response (no excitation)
                x_baseline = result['x'].clone()
                x_baseline[:, 10:13] = 0  # no excitation
                dataset_baseline = data.Data(x=x_baseline, edge_index=edges)
                pred_baseline = model(dataset_baseline, data_id=result['data_id'],
                                      k=result['k_batch'], return_all=False)

                for i, odor in enumerate(odor_list):
                    x_odor = result['x'].clone()
                    x_odor[:, 10:13] = 0
                    x_odor[:, 10 + i] = 1  # activate specific odor

                    dataset_odor = data.Data(x=x_odor, edge_index=edges)
                    pred_odor = model(dataset_odor, data_id=result['data_id'],
                                      k=result['k_batch'], return_all=False)

                    odor_diff = pred_odor - pred_baseline
                    odor_responses[odor].append(odor_diff.cpu())

                valid_samples += 1

            sample += 1
            if sample > n_samples * 10:
                break

        # Convert to tensors [n_samples, n_neurons]
        for odor in odor_list:
            odor_responses[odor] = torch.stack(odor_responses[odor]).squeeze()

    # Identify top responding neurons for each odor
    top_neurons = {}
    for odor in odor_list:
        # Calculate mean response across samples for each neuron
        mean_response = torch.mean(odor_responses[odor], dim=0)  # [n_neurons]

        # Get top 3 responding neurons (highest positive response)
        top_20_indices = torch.topk(mean_response, k=20).indices.cpu().numpy()
        top_20_names = [all_neuron_list[idx] for idx in top_20_indices]
        top_20_values = [mean_response[idx].item() for idx in top_20_indices]

        top_neurons[odor] = {
            'names': top_20_names,
            'indices': top_20_indices.tolist(),
            'values': top_20_values
        }

        print(f"\ntop 20 responding neurons for {odor}:")
        for i, (name, idx, val) in enumerate(zip(top_20_names, top_20_indices, top_20_values)):
            print(f"  {i + 1}. {name} : {val:.4f}")

    return odor_responses  # Return only odor_responses to match original function signature

def plot_odor_heatmaps(odor_responses):
    """
    Plot 3 separate heatmaps showing mean response per neuron for each odor
    """
    odor_list = ['butanone', 'pentanedione', 'NaCL']
    n_neurons = odor_responses['butanone'].shape[1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, odor in enumerate(odor_list):
        # Compute mean response per neuron
        mean_responses = torch.mean(odor_responses[odor], dim=0).numpy()  # [n_neurons]

        # Reshape to 2D for heatmap (assuming square-ish layout)
        side_length = int(np.ceil(np.sqrt(n_neurons)))
        padded_responses = np.pad(mean_responses, (0, side_length ** 2 - n_neurons), 'constant')
        response_matrix = padded_responses.reshape(side_length, side_length)

        # Plot heatmap
        sns.heatmap(response_matrix, ax=axes[i], cmap='bwr', center=0,
                    cbar=False, square=True, xticklabels=False, yticklabels=False)
        axes[i].set_title(f'{odor} mean response')

    plt.tight_layout()
    return fig

def overlay_umap_refit_with_W_list(
    w_list,
    out_prefix="/groups/saalfeld/home/allierc/Py/NeuralGraph/graphs_data/fly/",   # folder containing flyvis_connectomes_W.npz
    figure_path=None,                                            # e.g. ".../overlay_all.png"
    show=True,
    # UMAP params
    neighbors=15,
    min_dist=0.1,
    metric="cosine",
    seed=0,
    # labeling
    label_bg=True,
    labels=None,                 # list of text labels (same length as number of new vectors)
    label_fontsize=7,
    label_y_offset_frac=0.015,
    # styling for new points
    markers=None,                # list like ["*", "D", "o", ...]
    sizes=None,                  # list like [140, 120, 120, ...]
    edgecolors="k",
    linewidths=1.2,
    colors=None,                 # list of facecolors
    verbose=True,
):
    """
    Load saved training W, append multiple new vector(s), refit UMAP on [W_ref; w_new...], and plot.

    Parameters
    ----------
    w_list : array-like
        One of:
          - list/tuple of 1D arrays, each shaped [E]
          - 2D array shaped [n_new, E]
          - single 1D array [E] (treated as one vector)
    labels : list of str (optional)
        Text labels for new points. Defaults to ["NEW_0", "NEW_1", ...].
    markers/sizes/colors : per-point style lists (optional)

    Returns
    -------
    dict with keys:
        emb_bg  : (n_train, 2) UMAP coords of saved training points
        emb_new : (n_new, 2)   UMAP coords of the new points
        ids_bg  : (n_train,)    ids for background (if present in npz)
        reducer : fitted UMAP object
    """
    out = Path(out_prefix)
    W_file = out / "flyvis_connectomes_W.npz"
    if not W_file.exists():
        raise FileNotFoundError(
            f"Missing training matrix: {W_file}\n"
            "Save it once in the collector, e.g.: "
            "np.savez_compressed(f'{out}_W.npz', W=W.astype(np.float32), model_ids=np.array(ok_ids,'<U3'))"
        )

    # --- load saved training matrix (and ids if present) ---
    W_npz = np.load(W_file, allow_pickle=False)
    if "W" in W_npz:
        W_ref = np.asarray(W_npz["W"], dtype=np.float32)
    elif "w" in W_npz:  # fallback key
        W_ref = np.asarray(W_npz["w"], dtype=np.float32)
    else:
        raise KeyError(f"{W_file} must contain array 'W' (or 'w').")
    ids_bg = W_npz.get("model_ids", np.array([f"{i:03d}" for i in range(W_ref.shape[0])], dtype="<U8"))

    # --- normalize incoming w_list to a 2D array (n_new, E) ---
    if isinstance(w_list, (list, tuple)):
        new_vecs = [np.asarray(w, dtype=np.float32).reshape(1, -1) for w in w_list]
        w_new = np.vstack(new_vecs) if len(new_vecs) > 0 else np.zeros((0, W_ref.shape[1]), np.float32)
    else:
        w_arr = np.asarray(w_list, dtype=np.float32)
        if w_arr.ndim == 1:
            w_new = w_arr.reshape(1, -1)
        elif w_arr.ndim == 2:
            w_new = w_arr
        else:
            raise ValueError("w_list must be (list/tuple of 1D), a 1D array, or a 2D array.")
    if w_new.shape[1] != W_ref.shape[1]:
        raise ValueError(f"Feature mismatch: new has {w_new.shape[1]} features; saved W has {W_ref.shape[1]}.")

    n_new = w_new.shape[0]
    if labels is None:
        labels = [f"NEW_{i}" for i in range(n_new)]
    # default styles
    if markers is None:
        # cycle a few nice markers
        base = ["*", "D", "o", "s", "^", "P", "X", "v"]
        markers = [base[i % len(base)] for i in range(n_new)]
    if sizes is None:
        sizes = [140] * n_new
    if colors is None:
        # None -> matplotlib cycles; we’ll pass no color and let scatter choose per call
        colors = [None] * n_new

    # --- concatenate and fit UMAP fresh on [W_ref; w_new] ---
    W_all = np.vstack([W_ref, w_new])
    reducer = umap.UMAP(
        n_neighbors=neighbors,
        min_dist=min_dist,
        n_components=2,
        metric=metric,
        random_state=seed,
        init="spectral",
        verbose=verbose
    ).fit(W_all)

    emb_all = reducer.embedding_.astype(np.float32, copy=False)
    n_train = W_ref.shape[0]
    emb_bg  = emb_all[:n_train]
    emb_new = emb_all[n_train:]

    # --- plot ---
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(emb_bg[:, 0], emb_bg[:, 1], s=28, alpha=0.65, label="FlyVis (refit)")
    if label_bg:
        y_range = float(np.ptp(emb_all[:, 1])) if emb_all.size else 0.0
        dy = (label_y_offset_frac * y_range) if y_range > 0 else 0.02
        for i in range(n_train):
            ax.text(emb_bg[i, 0], emb_bg[i, 1] + dy, str(ids_bg[i]),
                    fontsize=label_fontsize, ha="center", va="bottom")

    # plot each new point with its own style + label
    for i in range(n_new):
        ax.scatter(
            emb_new[i:i+1, 0], emb_new[i:i+1, 1],
            s=sizes[i], marker=markers[i],
            edgecolors=edgecolors, linewidths=linewidths,
            label=labels[i],
            c=None if colors[i] is None else [colors[i]],
            zorder=3
        )

    ax.set_title("UMAP (refit) — FlyVis + new vector(s)")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(loc="best", frameon=True, fontsize=8)
    fig.tight_layout()
    if figure_path:
        fig.savefig(figure_path, dpi=220)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"emb_bg": emb_bg, "emb_new": emb_new, "ids_bg": ids_bg, "reducer": reducer}

from sklearn.neighbors import NearestNeighbors
import joblib

def overlay_barycentric_into_umap(
    w_list,
    out_prefix="/groups/saalfeld/home/allierc/Py/NeuralGraph/flyvis_connectomes",
    figure_path=None,
    show=True,
    metric="cosine",
    k=15,
    label_bg=True,
    labels=None,
    label_fontsize=7,
    label_y_offset_frac=0.015,
    eps_self=1e-12,   # for exact match snapping
):
    """
    Project new vectors into an existing UMAP background using K-NN barycentric weights.
    Requires the same W_ref the reducer was trained on (order must match).

    Files expected (from your collector):
      - {out_prefix}_W.npz               with arrays: W (or w), model_ids
      - {out_prefix}_umap_model.joblib   reducer fitted on W_ref
    """
    out = Path(out_prefix)
    W_file = out.with_name(out.name + "_W.npz")
    model_file = out.with_name(out.name + "_umap_model.joblib")

    # load training matrix + ids
    W_npz = np.load(W_file, allow_pickle=False)
    W_ref = np.asarray(W_npz["W"] if "W" in W_npz else W_npz["w"], dtype=np.float32)
    ids_bg = W_npz.get("model_ids", np.array([f"{i:03d}" for i in range(W_ref.shape[0])], dtype="<U8"))

    # load reducer to get the background embedding
    reducer = joblib.load(model_file)
    emb_bg = reducer.embedding_.astype(np.float32, copy=False)

    # prepare new vectors
    if isinstance(w_list, (list, tuple)):
        w_new = np.vstack([np.asarray(w, np.float32).reshape(1, -1) for w in w_list])
    else:
        arr = np.asarray(w_list, np.float32)
        w_new = arr.reshape(1, -1) if arr.ndim == 1 else arr
    if w_new.shape[1] != W_ref.shape[1]:
        raise ValueError(f"Feature mismatch: new has {w_new.shape[1]}, saved W has {W_ref.shape[1]}.")

    # KNN on the original high-dim space
    nbrs = NearestNeighbors(n_neighbors=min(k, W_ref.shape[0]), metric=metric)
    nbrs.fit(W_ref)

    emb_new = np.zeros((w_new.shape[0], 2), dtype=np.float32)
    for i, v in enumerate(w_new):
        # exact/self match snapping
        dists, idxs = nbrs.kneighbors(v[None, :], return_distance=True)
        dists = dists.ravel(); idxs = idxs.ravel()

        # if the closest neighbor is *itself* (zero distance), snap
        if dists[0] <= eps_self:
            emb_new[i] = emb_bg[idxs[0]]
            continue

        # inverse-distance weights (add tiny epsilon to avoid div by 0)
        wts = 1.0 / (dists + 1e-12)
        wts = wts / (wts.sum() + 1e-12)

        # barycentric combination of neighbor coordinates
        emb_new[i] = (wts[:, None] * emb_bg[idxs]).sum(axis=0)

    # plot
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(emb_bg[:, 0], emb_bg[:, 1], s=28, alpha=0.65, label="FlyVis (fixed)")
    if label_bg:
        y_range = float(np.ptp(emb_bg[:, 1])) if emb_bg.size else 0.0
        dy = (label_y_offset_frac * y_range) if y_range > 0 else 0.02
        for i in range(emb_bg.shape[0]):
            ax.text(emb_bg[i, 0], emb_bg[i, 1] + dy, str(ids_bg[i]),
                    fontsize=label_fontsize, ha="center", va="bottom")

    if labels is None:
        labels = [f"NEW_{i}" for i in range(w_new.shape[0])]
    for i in range(w_new.shape[0]):
        ax.scatter(emb_new[i, 0], emb_new[i, 1], s=160, marker="*",
                   edgecolors="k", linewidths=1.2, label=labels[i])

    ax.set_title("UMAP (fixed) + KNN barycentric projection")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(loc="best", frameon=True, fontsize=8)
    fig.tight_layout()
    if figure_path: fig.savefig(figure_path, dpi=220)
    if show: plt.show()
    else: plt.close(fig)

    return {"emb_bg": emb_bg, "emb_new": emb_new, "ids_bg": ids_bg}

def get_n_hop_neighborhood_with_stats(target_ids, edges_all, n_hops, verbose=False):
    """Get n-hop neighborhood with optional detailed statistics per hop"""

    current = set(target_ids)
    all_neurons = set(target_ids)

    if verbose:
        print("\n=== N-hop Neighborhood Expansion ===")
        print(f"Starting with {len(target_ids)} core neurons")

    # Track stats per hop
    hop_stats = []

    for hop in range(n_hops):
        next_hop = set()
        edge_count = 0

        for node in current:
            # Find predecessors (neurons that send to current)
            mask = edges_all[1, :] == node
            predecessors = edges_all[0, mask].cpu().numpy()
            next_hop.update(predecessors)
            edge_count += len(predecessors)

        # New neurons added at this hop
        new_neurons = next_hop - all_neurons
        all_neurons.update(next_hop)

        if verbose:
            # Calculate edges to neurons at this hop
            edges_to_current = torch.isin(edges_all[1, :],
                                         torch.tensor(list(all_neurons), device=edges_all.device))
            total_edges = edges_to_current.sum().item()

            # Store stats
            hop_stats.append({
                'hop': hop + 1,
                'new_neurons': len(new_neurons),
                'total_neurons': len(all_neurons),
                'edges_this_hop': edge_count,
                'total_edges': total_edges,
                'expansion_factor': len(all_neurons) / len(target_ids)
            })

            print(f"\nHop {hop + 1}:")
            print(f"  New neurons added: {len(new_neurons):,}")
            print(f"  Total neurons now: {len(all_neurons):,} ({100*len(all_neurons)/13741:.1f}% of network)")
            print(f"  Edges from this hop: {edge_count:,}")
            print(f"  Total edges needed: {total_edges:,} ({100*total_edges/edges_all.shape[1]:.1f}% of all edges)")
            print(f"  Expansion factor: {len(all_neurons)/len(target_ids):.2f}x")
            print(f"  Compute cost estimate: {len(all_neurons) * total_edges / 1e6:.2f}M operations")

        current = next_hop

        if len(current) == 0:
            if verbose:
                print("  -> No more expansion possible")
            break

    if verbose:
        print("\n=== Summary ===")
        print(f"Total neurons: {len(all_neurons):,} / 13,741 ({100*len(all_neurons)/13741:.1f}%)")
        print(f"Total edges: {total_edges:,} / {edges_all.shape[1]:,} ({100*total_edges/edges_all.shape[1]:.1f}%)")
        print(f"Memory estimate: {(len(all_neurons) * 8 + total_edges * 8) / 1e6:.2f} MB")

    return np.array(sorted(all_neurons))



def analyze_type_neighbors(
    type_name: str,
    edges_all: torch.Tensor,        # shape (2, E); row0=pre, row1=post; on some device
    type_list: torch.Tensor,        # shape (N,1) or (N,); integer type indices aligned with node IDs
    n_hops: int = 3,
    direction: str = 'in',          # 'in' | 'out' | 'both'
    verbose: bool = False
):
    """
    Pick one neuron of the given type and expand n hops to collect per-hop type compositions.
    Returns a dict with the target info, per-hop stats, and a short summary.
    """

    device = edges_all.device
    type_vec = type_list.squeeze(-1).long().to(device)  # (N,)

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

    # --- map type_name -> type_index (simple, case-insensitive) ---
    def _norm(s): return ''.join(ch for ch in s.lower() if ch.isalnum())
    name_to_index = {}
    for k, v in index_to_name.items():
        name_to_index[_norm(v)] = int(k)

    tkey = _norm(type_name)
    if tkey not in name_to_index:
        raise ValueError(f"Unknown type name: {type_name}")

    target_type_idx = name_to_index[tkey]
    target_type_name = index_to_name.get(target_type_idx, f"Type{target_type_idx}")

    # --- pick one neuron of that type (first occurrence) ---
    cand = torch.nonzero(type_vec == target_type_idx, as_tuple=True)[0]
    if cand.numel() == 0:
        return {
            "target_type_idx": target_type_idx,
            "target_type_name": target_type_name,
            "note": "No neuron of this type present.",
            "per_hop": [],
            "summary": {"total_neurons": 0, "total_hops_realized": 0, "direction": direction}
        }
    target_id = int(cand[0].item())

    # --- neighborhood expansion ---
    visited = torch.tensor([target_id], device=device, dtype=torch.long)
    current = visited.clone()
    per_hop = []

    for hop in range(1, n_hops + 1):
        if direction == 'in':
            mask = torch.isin(edges_all[1], current)
            nxt = edges_all[0, mask]
        elif direction == 'out':
            mask = torch.isin(edges_all[0], current)
            nxt = edges_all[1, mask]
        else:  # 'both'
            mask_in = torch.isin(edges_all[1], current)
            mask_out = torch.isin(edges_all[0], current)
            nxt = torch.cat([edges_all[0, mask_in], edges_all[1, mask_out]], dim=0)

        if nxt.numel() == 0:
            break

        nxt = torch.unique(nxt)
        # remove already visited
        new = nxt[~torch.isin(nxt, visited)]
        if new.numel() == 0:
            break

        # types for newly discovered nodes
        new_types = type_vec[new]
        new_ids = new.tolist()
        new_type_idxs = new_types.tolist()
        new_type_names = [index_to_name.get(int(t), f"Type{int(t)}") for t in new_type_idxs]

        # per-hop counts
        cnt = Counter(new_type_names)
        n_new = int(new.numel())
        type_counts = dict(cnt)
        type_perc = {k: v / n_new for k, v in type_counts.items()}

        per_hop.append({
            "hop": hop,
            "new_neuron_ids": new_ids,
            "type_indices": new_type_idxs,
            "type_names": new_type_names,
            "type_counts": type_counts,
            "type_perc": type_perc,
            "n_new": n_new,
        })

        if verbose:
            print(f"hop {hop}: new={n_new}  unique types={len(cnt)}  top={cnt.most_common(3)}")

        # advance
        visited = torch.unique(torch.cat([visited, new], dim=0))
        current = new

    # --- summary (simple) ---
    cumulative = Counter()
    total_new = 0
    for h in per_hop:
        cumulative.update(h["type_counts"])
        total_new += h["n_new"]
    cumulative_perc = {k: (v / total_new if total_new else 0.0) for k, v in cumulative.items()}

    return {
        "target_id": target_id,
        "target_type_idx": target_type_idx,
        "target_type_name": target_type_name,
        "per_hop": per_hop,
        "summary": {
            "total_neurons": int(visited.numel()),
            "total_hops_realized": len(per_hop),
            "direction": direction,
            "cumulative_type_counts": dict(cumulative),
            "cumulative_type_perc": cumulative_perc,
        },
    }

def plot_weight_comparison(w_true, w_modified, output_path, xlabel='true $W$', ylabel='modified $W$', color='white'):
    w_true_np = w_true.detach().cpu().numpy().flatten()
    w_modified_np = w_modified.detach().cpu().numpy().flatten()
    plt.figure(figsize=(8, 8))
    plt.scatter(w_true_np, w_modified_np, s=8, alpha=0.5, color=color, edgecolors='none')
    # Fit linear model
    lin_fit, _ = curve_fit(linear_model, w_true_np, w_modified_np)
    slope = lin_fit[0]
    lin_fit[1]
    # R2 calculation
    residuals = w_modified_np - linear_model(w_true_np, *lin_fit)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((w_modified_np - np.mean(w_modified_np)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    # Plot identity line
    plt.plot([w_true_np.min(), w_true_np.max()], [w_true_np.min(), w_true_np.max()], 'r--', linewidth=2, label='identity')
    # Add text
    plt.text(w_true_np.min(), w_true_np.max(), f'$R^2$: {r_squared:.3f}\nslope: {slope:.2f}', fontsize=18, va='top', ha='left')
    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return slope, r_squared


def check_dales_law(edges, weights, type_list=None, n_neurons=None, verbose=True, logger=None):
    """
    Check if synaptic weights satisfy Dale's Law.

    Dale's Law: Each neuron releases the same neurotransmitter at all synapses.
    This means all outgoing weights from a neuron should have the same sign.

    Parameters:
    -----------
    edges : torch.Tensor
        Edge index tensor of shape [2, n_edges] where edges[0] are source neurons
    weights : torch.Tensor
        Weight tensor of shape [n_edges] or [n_edges, 1]
    type_list : torch.Tensor, optional
        Neuron type indices of shape [n_neurons] or [n_neurons, 1]
    n_neurons : int, optional
        Total number of neurons (inferred from edges if not provided)
    verbose : bool, default=True
        If True, print detailed statistics
    logger : logging.Logger, optional
        Logger for recording results

    Returns:
    --------
    dict with keys:
        - 'n_excitatory': Number of purely excitatory neurons (all W>0)
        - 'n_inhibitory': Number of purely inhibitory neurons (all W<0)
        - 'n_mixed': Number of mixed neurons (violates Dale's Law)
        - 'n_violations': Number of Dale's Law violations
        - 'violations': List of dicts with violation details
        - 'neuron_signs': Dict mapping neuron_idx to sign (1=excitatory, -1=inhibitory, 0=mixed)
    """
    # Neuron type name mapping (from FlyVis connectome)
    index_to_name = {
        0: 'Am', 1: 'C2', 2: 'C3', 3: 'CT1(Lo1)', 4: 'CT1(M10)',
        5: 'L1', 6: 'L2', 7: 'L3', 8: 'L4', 9: 'L5',
        10: 'Lawf1', 11: 'Lawf2', 12: 'Mi1', 13: 'Mi15', 14: 'Mi4',
        15: 'Mi9', 16: 'T1', 17: 'T2', 18: 'T2a', 19: 'T3',
        20: 'T4a', 21: 'T4b', 22: 'T4c', 23: 'T4d', 24: 'T5a',
        25: 'T5b', 26: 'T5c', 27: 'T5d', 28: 'Tm1', 29: 'Tm2',
        30: 'Tm3', 31: 'Tm4', 32: 'Tm9', 33: 'TmY10', 34: 'TmY13',
        35: 'TmY14', 36: 'TmY15', 37: 'TmY18', 38: 'TmY3',
        39: 'TmY4', 40: 'TmY5a', 41: 'TmY9'
    }

    # Flatten weights if needed
    if weights.dim() > 1:
        weights = weights.squeeze()

    # Infer n_neurons if not provided
    if n_neurons is None:
        n_neurons = int(edges.max().item()) + 1

    # Check Dale's Law for each neuron
    dale_violations = []
    neuron_signs = {}

    for neuron_idx in range(n_neurons):
        # Find all outgoing edges from this neuron
        outgoing_mask = edges[0, :] == neuron_idx
        outgoing_weights = weights[outgoing_mask]

        if len(outgoing_weights) > 0:
            n_positive = (outgoing_weights > 0).sum().item()
            n_negative = (outgoing_weights < 0).sum().item()
            n_zero = (outgoing_weights == 0).sum().item()

            # Dale's Law: all non-zero weights should have same sign
            if n_positive > 0 and n_negative > 0:
                violation_info = {
                    'neuron': neuron_idx,
                    'n_positive': n_positive,
                    'n_negative': n_negative,
                    'n_zero': n_zero
                }

                # Add type information if available
                if type_list is not None:
                    type_id = type_list[neuron_idx].item()
                    type_name = index_to_name.get(type_id, f'Unknown_{type_id}')
                    violation_info['type_id'] = type_id
                    violation_info['type_name'] = type_name

                dale_violations.append(violation_info)
                neuron_signs[neuron_idx] = 0  # Mixed
            elif n_positive > 0:
                neuron_signs[neuron_idx] = 1  # Excitatory
            elif n_negative > 0:
                neuron_signs[neuron_idx] = -1  # Inhibitory
            else:
                neuron_signs[neuron_idx] = 0  # All zero

    # Compute statistics
    n_excitatory = sum(1 for s in neuron_signs.values() if s == 1)
    n_inhibitory = sum(1 for s in neuron_signs.values() if s == -1)
    n_mixed = sum(1 for s in neuron_signs.values() if s == 0)

    # Print results if verbose
    if verbose:
        print("\n=== Dale's Law Check ===")
        print(f"Total neurons: {n_neurons}")
        print(f"Excitatory neurons (all W>0): {n_excitatory} ({100*n_excitatory/n_neurons:.1f}%)")
        print(f"Inhibitory neurons (all W<0): {n_inhibitory} ({100*n_inhibitory/n_neurons:.1f}%)")
        print(f"Mixed/zero neurons (violates Dale's Law): {n_mixed} ({100*n_mixed/n_neurons:.1f}%)")
        print(f"Dale's Law violations: {len(dale_violations)}")

        if logger:
            logger.info("=== Dale's Law Check ===")
            logger.info(f"Total neurons: {n_neurons}")
            logger.info(f"Excitatory: {n_excitatory} ({100*n_excitatory/n_neurons:.1f}%)")
            logger.info(f"Inhibitory: {n_inhibitory} ({100*n_inhibitory/n_neurons:.1f}%)")
            logger.info(f"Violations: {len(dale_violations)}")

        if len(dale_violations) > 0:
            print("\nFirst 10 violations:")
            for i, v in enumerate(dale_violations[:10]):
                if 'type_name' in v:
                    print(f"  Neuron {v['neuron']} ({v['type_name']}): "
                          f"{v['n_positive']} positive, {v['n_negative']} negative, {v['n_zero']} zero weights")
                    if logger:
                        logger.info(f"  Neuron {v['neuron']} ({v['type_name']}): "
                                    f"{v['n_positive']} positive, {v['n_negative']} negative")
                else:
                    print(f"  Neuron {v['neuron']}: "
                          f"{v['n_positive']} positive, {v['n_negative']} negative, {v['n_zero']} zero weights")

            # Group violations by neuron type if available
            if type_list is not None and any('type_name' in v for v in dale_violations):
                type_violations = Counter([v['type_name'] for v in dale_violations if 'type_name' in v])
                print("\nViolations by neuron type:")
                for type_name, count in sorted(type_violations.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {type_name}: {count} violations")
                    if logger:
                        logger.info(f"  {type_name}: {count} violations")
        else:
            print("✓ Weights perfectly satisfy Dale's Law!")
            if logger:
                logger.info("✓ Weights perfectly satisfy Dale's Law!")

        print("=" * 60 + "\n")

    return {
        'n_excitatory': n_excitatory,
        'n_inhibitory': n_inhibitory,
        'n_mixed': n_mixed,
        'n_violations': len(dale_violations),
        'violations': dale_violations,
        'neuron_signs': neuron_signs
    }


class RegularizationTracker:
    """
    Centralized class for computing, tracking, and plotting regularization terms during training.

    Supports regularization terms:
    - W_L1, W_L2: Weight matrix sparsity and L2 regularization
    - W_sign: Dale's Law enforcement (same-sign outgoing weights)
    - edge_diff, edge_norm: Edge function monotonicity and normalization
    - edge_grad, phi_grad: Gradient penalties for MLP smoothness
    - edge_weight, phi_weight: L1/L2 on MLP parameters
    - phi_zero: Constraint that lin_phi(0) = 0
    - update_msg_diff, update_u_diff, update_msg_sign: Update function constraints
    """

    def __init__(self, config, model, device, n_neurons):
        """
        Initialize the regularization tracker.

        Args:
            config: Configuration object with training and model settings
            model: The neural network model
            device: Torch device
            n_neurons: Number of neurons in the network
        """
        self.config = config
        self.model = model
        self.device = device
        self.n_neurons = n_neurons

        train_config = config.training
        model_config = config.graph_model

        # Store coefficients
        self.coeff_W_L1 = train_config.coeff_W_L1
        self.coeff_W_L2 = train_config.coeff_W_L2
        self.coeff_W_sign = train_config.coeff_W_sign
        self.W_sign_temperature = train_config.W_sign_temperature
        self.coeff_edge_diff = train_config.coeff_edge_diff
        self.coeff_edge_norm = train_config.coeff_edge_norm
        self.coeff_edge_gradient_penalty = train_config.coeff_edge_gradient_penalty
        self.coeff_phi_gradient_penalty = train_config.coeff_phi_gradient_penalty
        self.coeff_edge_weight_L1 = train_config.coeff_edge_weight_L1
        self.coeff_edge_weight_L2 = train_config.coeff_edge_weight_L2
        self.coeff_phi_weight_L1 = train_config.coeff_phi_weight_L1
        self.coeff_phi_weight_L2 = train_config.coeff_phi_weight_L2
        self.coeff_update_msg_diff = train_config.coeff_update_msg_diff
        self.coeff_update_u_diff = train_config.coeff_update_u_diff
        self.coeff_update_msg_sign = train_config.coeff_update_msg_sign

        # Model config
        self.lin_edge_positive = model_config.lin_edge_positive
        self.embedding_dim = model_config.embedding_dim

        # Initialize loss components dictionary
        self.loss_components = {
            'loss': [],
            'regul_total': [],
            'W_L1': [],
            'W_L2': [],
            'W_sign': [],
            'edge_grad': [],
            'phi_grad': [],
            'edge_diff': [],
            'edge_norm': [],
            'edge_weight': [],
            'phi_weight': [],
            'update_msg_diff': [],
            'update_u_diff': [],
            'update_msg_sign': [],
        }

        # Current iteration tracking
        self._regul_total = 0
        self._regul_terms = {}
        self._tracking = False

    def should_track(self, iteration, plot_frequency):
        """Determine if this iteration should track regularization components."""
        return (iteration % plot_frequency == 0) or (iteration == 0)

    def _add_regul(self, regul_term, component_name):
        """Internal helper to accumulate regularization and optionally track components."""
        val = regul_term.item()
        self._regul_total += val
        if self._tracking:
            self._regul_terms[component_name] += val
        return regul_term

    def compute_regularization(self, *, x, ids, xnorm, edges, epoch=0,
                                in_features=None, in_features_next=None,
                                track=False):
        """
        Compute all regularization terms.

        Args:
            x: Input tensor
            ids: Neuron indices for this batch
            xnorm: Normalization factor
            edges: Edge index tensor
            epoch: Current epoch (some regularizations only apply after epoch 0)
            in_features: Pre-computed input features for lin_edge (optional)
            in_features_next: Pre-computed next input features for lin_edge (optional)
            track: Whether to track individual regularization components

        Returns:
            Total regularization loss tensor
        """
        # Reset counters
        self._regul_total = 0
        self._tracking = track
        if track:
            self._regul_terms = {key: 0 for key in self.loss_components.keys()
                                if key not in ['loss', 'regul_total']}

        loss = torch.zeros(1, device=self.device)
        model = self.model

        # W L1 sparsity
        if self.coeff_W_L1 > 0:
            regul_term = model.W.norm(1) * self.coeff_W_L1
            loss = loss + self._add_regul(regul_term, 'W_L1')

        # W L2 regularization
        if self.coeff_W_L2 > 0:
            regul_term = model.W.norm(2) * self.coeff_W_L2
            loss = loss + self._add_regul(regul_term, 'W_L2')

        # Edge weight L1/L2 regularization
        if (self.coeff_edge_weight_L1 + self.coeff_edge_weight_L2) > 0:
            for param in model.lin_edge.parameters():
                regul_term = param.norm(1) * self.coeff_edge_weight_L1 + param.norm(2) * self.coeff_edge_weight_L2
                loss = loss + self._add_regul(regul_term, 'edge_weight')

        # Phi weight L1/L2 regularization
        # Note: original code uses norm(2) for L1 coefficient (likely a bug, but keeping for consistency)
        if (self.coeff_phi_weight_L1 + self.coeff_phi_weight_L2) > 0:
            for param in model.lin_phi.parameters():
                regul_term = param.norm(2) * self.coeff_phi_weight_L1 + param.norm(2) * self.coeff_phi_weight_L2
                loss = loss + self._add_regul(regul_term, 'phi_weight')

        # Edge diff (monotonicity constraint)
        if self.coeff_edge_diff > 0 and in_features is not None and in_features_next is not None:
            if self.lin_edge_positive:
                msg0 = model.lin_edge(in_features[ids].clone()) ** 2
                msg1 = model.lin_edge(in_features_next[ids].clone()) ** 2
            else:
                msg0 = model.lin_edge(in_features[ids].clone())
                msg1 = model.lin_edge(in_features_next[ids].clone())
            regul_term = torch.relu(msg0 - msg1).norm(2) * self.coeff_edge_diff
            loss = loss + self._add_regul(regul_term, 'edge_diff')


        # Edge gradient penalty (smoothness)
        if self.coeff_edge_gradient_penalty > 0 and in_features is not None:
            in_features_sample = in_features[ids].clone()
            in_features_sample.requires_grad_(True)

            if self.lin_edge_positive:
                msg_sample = model.lin_edge(in_features_sample) ** 2
            else:
                msg_sample = model.lin_edge(in_features_sample)

            grad_edge = torch.autograd.grad(
                outputs=msg_sample.sum(),
                inputs=in_features_sample,
                create_graph=True
            )[0]

            regul_term = (grad_edge.norm(2) ** 2) * self.coeff_edge_gradient_penalty
            loss = loss + self._add_regul(regul_term, 'edge_grad')

        # Edge norm constraint 
        if self.coeff_edge_norm > 0 and in_features is not None:
            in_features[:, 0] = 2 * xnorm
            if self.lin_edge_positive:
                msg = model.lin_edge(in_features[ids].clone()) ** 2
            else:
                msg = model.lin_edge(in_features[ids].clone())
            regul_term = (msg - 2 * xnorm).norm(2) * self.coeff_edge_norm
            loss = loss + self._add_regul(regul_term, 'edge_norm')



        # Phi gradient penalty (smoothness)
        if self.coeff_phi_gradient_penalty > 0:
            in_features_phi = get_in_features_update(rr=None, model=model, device=self.device)
            in_features_phi_sample = in_features_phi[ids].clone()
            in_features_phi_sample.requires_grad_(True)

            pred_phi_sample = model.lin_phi(in_features_phi_sample)

            grad_phi = torch.autograd.grad(
                outputs=pred_phi_sample.sum(),
                inputs=in_features_phi_sample,
                create_graph=True
            )[0]

            regul_term = (grad_phi.norm(2) ** 2) * self.coeff_phi_gradient_penalty
            loss = loss + self._add_regul(regul_term, 'phi_grad')

        # W sign (Dale's Law)
        if (self.coeff_W_sign > 0) and (epoch > 0):
            weights = model.W.squeeze()
            source_neurons = edges[0]

            n_pos = torch.zeros(self.n_neurons, device=self.device)
            n_neg = torch.zeros(self.n_neurons, device=self.device)
            n_total = torch.zeros(self.n_neurons, device=self.device)

            pos_mask = torch.sigmoid(self.W_sign_temperature * weights)
            neg_mask = torch.sigmoid(-self.W_sign_temperature * weights)

            n_pos.scatter_add_(0, source_neurons, pos_mask)
            n_neg.scatter_add_(0, source_neurons, neg_mask)
            n_total.scatter_add_(0, source_neurons, torch.ones_like(weights))

            violation = torch.where(n_total > 0,
                                   (n_pos / n_total) * (n_neg / n_total),
                                   torch.zeros_like(n_total))

            loss_W_sign = violation.sum()
            regul_term = loss_W_sign * self.coeff_W_sign
            loss = loss + self._add_regul(regul_term, 'W_sign')

        return loss

    def compute_update_regularization(self, in_features, ids_batch):
        """
        Compute update function regularization terms (requires model forward pass output).

        Args:
            in_features: Input features from model forward pass
            ids_batch: Batch indices

        Returns:
            Total update regularization loss tensor
        """
        loss = torch.zeros(1, device=self.device)
        model = self.model

        # Update msg diff (monotonicity on message input)
        if self.coeff_update_msg_diff > 0:
            pred_msg = model.lin_phi(in_features.clone().detach())
            in_features_msg_next = in_features.clone().detach()
            in_features_msg_next[:, self.embedding_dim + 1] = in_features_msg_next[:, self.embedding_dim + 1] * 1.05
            pred_msg_next = model.lin_phi(in_features_msg_next)
            regul_term = torch.relu(pred_msg[ids_batch] - pred_msg_next[ids_batch]).norm(2) * self.coeff_update_msg_diff
            loss = loss + self._add_regul(regul_term, 'update_msg_diff')

        # Update u diff (monotonicity on voltage input)
        if self.coeff_update_u_diff > 0:
            pred_u = model.lin_phi(in_features.clone().detach())
            in_features_u_next = in_features.clone().detach()
            in_features_u_next[:, 0] = in_features_u_next[:, 0] * 1.05
            pred_u_next = model.lin_phi(in_features_u_next)
            regul_term = torch.relu(pred_u_next[ids_batch] - pred_u[ids_batch]).norm(2) * self.coeff_update_u_diff
            loss = loss + self._add_regul(regul_term, 'update_u_diff')

        # Update msg sign (sign consistency)
        if self.coeff_update_msg_sign > 0:
            in_features_modified = in_features.clone().detach()
            in_features_modified[:, 0] = 0
            pred_msg = model.lin_phi(in_features_modified)
            msg = in_features[:, self.embedding_dim + 1].clone().detach()
            regul_term = (torch.tanh(pred_msg / 0.1) - torch.tanh(msg / 0.1)).norm(2) * self.coeff_update_msg_sign
            loss = loss + self._add_regul(regul_term, 'update_msg_sign')

        return loss

    def get_regul_total(self):
        """Return the total regularization loss for the current iteration."""
        return self._regul_total

    def track_regularization(self, loss):
        """
        Track loss components for the current iteration.

        Args:
            loss: Total loss value (including regularization)
        """
        current_loss = loss.item() if torch.is_tensor(loss) else loss

        # Store normalized values
        self.loss_components['loss'].append((current_loss - self._regul_total) / self.n_neurons)
        self.loss_components['regul_total'].append(self._regul_total / self.n_neurons)

        if self._tracking:
            for key in self._regul_terms:
                if key in self.loss_components:
                    self.loss_components[key].append(self._regul_terms[key] / self.n_neurons)

    def plot_loss(self, log_dir, epoch, Niter, total_loss=None, total_loss_regul=None, debug=False):
        """
        Plot loss components.

        Args:
            log_dir: Directory to save plots
            epoch: Current epoch
            Niter: Current iteration
            total_loss: Accumulated total loss (for debug)
            total_loss_regul: Accumulated regularization loss (for debug)
            debug: Whether to print debug information
        """
        from NeuralGraph.generators.utils import plot_signal_loss

        current_loss = self.loss_components['loss'][-1] if self.loss_components['loss'] else None
        current_regul = self.loss_components['regul_total'][-1] if self.loss_components['regul_total'] else None

        plot_signal_loss(
            self.loss_components,
            log_dir,
            epoch=epoch,
            Niter=Niter,
            debug=debug,
            current_loss=current_loss,
            current_regul=current_regul,
            total_loss=total_loss,
            total_loss_regul=total_loss_regul
        )

    def update_coefficients(self, epoch):
        """
        Update regularization coefficients based on epoch (for annealing).

        Args:
            epoch: Current epoch number
        """
        train_config = self.config.training

        # Anneal coefficients that change with epoch: starts at 0, increases toward max value
        self.coeff_edge_weight_L1 = train_config.coeff_edge_weight_L1 * (1 - np.exp(-train_config.coeff_edge_weight_L1_rate * epoch))
        self.coeff_phi_weight_L1 = train_config.coeff_phi_weight_L1 * (1 - np.exp(-train_config.coeff_phi_weight_L1_rate * epoch))
        self.coeff_W_L1 = train_config.coeff_W_L1 * (1 - np.exp(-train_config.coeff_W_L1_rate * epoch))
        self.coeff_W_L2 = train_config.coeff_W_L2


def analyze_neighbor_hops(type_name, edges_all, type_list, n_hops=10, direction='in', verbose=True):
    """
    Analyze neighbor connectivity by hop count for a given neuron type.

    Args:
        type_name: Name of the neuron type to analyze
        edges_all: Edge index tensor [2, n_edges]
        type_list: Tensor of neuron types (N,1) or (N,)
        n_hops: Number of hops to analyze
        direction: 'in' for incoming connections, 'out' for outgoing
        verbose: Whether to print detailed information

    Returns:
        Dictionary with hop_counts, cumulative, total_excl_target, total_incl_target
    """
    res = analyze_type_neighbors(
        type_name=type_name,
        edges_all=edges_all,
        type_list=type_list,
        n_hops=n_hops,
        direction=direction,
        verbose=verbose
    )

    hop_counts = [h["n_new"] for h in res["per_hop"]]
    total_excl_target = sum(hop_counts)
    total_incl_target = 1 + total_excl_target
    cumulative_by_hop = np.cumsum(hop_counts).tolist()

    if verbose:
        for hop in res["per_hop"]:
            print('hop ', hop["hop"], ':', hop["n_new"], hop["type_counts"])
        print("per-hop:", hop_counts)
        print("cumulative:", cumulative_by_hop)
        print("total excl target:", total_excl_target)

    return {
        'hop_counts': hop_counts,
        'cumulative': cumulative_by_hop,
        'total_excl_target': total_excl_target,
        'total_incl_target': total_incl_target,
        'per_hop_details': res["per_hop"]
    }
