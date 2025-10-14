import matplotlib.pyplot as plt
import torch

from matplotlib.ticker import FormatStrFormatter
from NeuralGraph.models import *
from NeuralGraph.utils import *

from GNN_Main import *
import matplotlib as mpl
import networkx as nx
from torch_geometric.utils.convert import to_networkx
import warnings
import numpy as np
import time
import tqdm
from tifffile import imread, imwrite as imsave

import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.linear_model import LogisticRegression
from NeuralGraph.fitting_models import linear_model
import json
from matplotlib.animation import FFMpegWriter
from scipy.signal import find_peaks
from pathlib import Path
from joblib import load

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
        case 'PDE_N4' | 'PDE_N7':
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
                         gt_weights, n_neurons=None, n_neuron_types=None):
    signal_model_name = config.graph_model.signal_model_name
    n_input_neurons = config.simulation.n_input_neurons

    if n_neurons is None:
        n_neurons = len(type_list)

    if config.graph_model.field_type =='visual':
        n_frames = config.simulation.n_frames
        k = 100
        reconstructed_field = to_numpy(model.visual_NNR(torch.tensor([k / n_frames], dtype=torch.float32, device=device)) ** 2)
        gt_field = x_list[0][k,:n_input_neurons,4:5]

        X1 = x_list[0][k,:n_input_neurons,1:3]

        # Setup for saving MP4
        fps = 10  # frames per second for the video
        metadata = dict(title='Field Evolution', artist='Matplotlib', comment='NN Reconstruction over time')
        writer = FFMpegWriter(fps=fps, metadata=metadata)
        fig = plt.figure(figsize=(8, 4))

        # Start the writer context
        if os.path.exists(f"./{log_dir}/tmp_training/field/field_movie_{epoch}_{N}.mp4"):
            os.remove(f"./{log_dir}/tmp_training/field/field_movie_{epoch}_{N}.mp4")
        with writer.saving(fig, f"./{log_dir}/tmp_training/field/field_movie_{epoch}_{N}.mp4", dpi=100):
            for k in range(0, 400, 1):

                # Inference and data extraction
                reconstructed_field = to_numpy(
                    model.visual_NNR(torch.tensor([k / n_frames], dtype=torch.float32, device=device)) ** 2)
                gt_field = x_list[0][k, :n_input_neurons, 4:5]
                X1 = x_list[0][k, :n_input_neurons, 1:3]

                vmin = reconstructed_field.min()
                vmax = reconstructed_field.max()
                fig.clf()  # Clear the figure

                # Ground truth
                ax1 = fig.add_subplot(1, 2, 1)
                sc1 = ax1.scatter(X1[:, 0], X1[:, 1], s=256, c=gt_field, cmap="viridis", marker='h', vmin=-2,
                                  vmax=2)
                ax1.set_xticks([])
                ax1.set_yticks([])
                ax1.set_title("Ground Truth")

                # Reconstructed
                ax2 = fig.add_subplot(1, 2, 2)
                sc2 = ax2.scatter(X1[:, 0], X1[:, 1], s=256, c=reconstructed_field, cmap="viridis", marker='h',
                                  vmin=vmin, vmax=vmax)
                ax2.set_xticks([])
                ax2.set_yticks([])
                ax2.set_title("Reconstructed")

                plt.tight_layout()
                writer.grab_frame()


    # Plot 1: Embedding scatter plot
    fig = plt.figure(figsize=(8, 8))
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


    # Plot 2: Weight comparison scatter plot
    fig = plt.figure(figsize=(8, 8))

    plt.scatter(to_numpy(gt_weights), to_numpy(model.W.squeeze()), s=0.1, c='k', alpha=0.01)
    plt.xlabel(r'true $W_{ij}$', fontsize=18)
    plt.ylabel(r'learned $W_{ij}$', fontsize=18)
    plt.text(-0.9, 4.5, f'R^2: {np.round(r_squared, 3)}\nslope: {np.round(lin_fit[0], 2)}', fontsize=12)
    plt.xlim([-1, 5])
    plt.ylim([-5, 5])
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/matrix/comparison_{epoch}_{N}.tif",
                dpi=87, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Plot 3: Edge function visualization
    fig = plt.figure(figsize=(8, 8))
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
    fig = plt.figure(figsize=(8, 8))
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

def plot_training_signal(config, model, x, adjacency, log_dir, epoch, N, n_neurons, type_list, cmap, device):

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

    gt_weight = to_numpy(adjacency)

    if config.training.multi_connectivity:
        pred_weight = to_numpy(model.W[0, :n_neurons, :n_neurons].clone().detach())
    else:
        pred_weight = to_numpy(model.W[:n_neurons, :n_neurons].clone().detach())

    if n_neurons<1000:
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(121)
        ax = sns.heatmap(gt_weight, center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046})
        plt.xticks([0, n_neurons - 1], [1, n_neurons], fontsize=8)
        plt.yticks([0, n_neurons - 1], [1, n_neurons], fontsize=8)
        plt.ylabel('postsynaptic', fontsize=16)
        plt.xlabel('presynaptic', fontsize=16)
        plt.title('true weight matrix', fontsize=16)
        ax = fig.add_subplot(122)
        ax = sns.heatmap(pred_weight, center=0, square=True, cmap='bwr', vmin=-1, vmax=1, cbar_kws={'fraction': 0.046})
        plt.xticks([0, n_neurons - 1], [1, n_neurons], fontsize=8)
        plt.yticks([0, n_neurons - 1], [1, n_neurons], fontsize=8)
        plt.ylabel('postsynaptic', fontsize=16)
        plt.xlabel('presynaptic', fontsize=16)
        plt.title('predicted weight matrix', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/matrix/comparison_{epoch}_{N}.tif", dpi=50)
        plt.close()
    else:
        fig = plt.figure(figsize=(8, 8))
        fig, ax = fig_init()
        if n_neurons<1000:
            plt.scatter(gt_weight, pred_weight / 10, s=1.0, c='k', alpha=1.0)
        else:
            plt.scatter(gt_weight, pred_weight / 10, s=0.1, c='k', alpha=0.1)
        plt.xlabel(r'true $W_{ij}$', fontsize=48)
        plt.ylabel(r'learned $W_{ij}$', fontsize=48)
        if n_neurons == 8000:
            plt.xlim([-0.05, 0.05])
        else:
            plt.ylim([-0.2, 0.2])
            plt.xlim([-0.2, 0.2])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/matrix/comparison_{epoch}_{N}.tif", dpi=87)
        plt.close()

    if ('PDE_N8' in config.graph_model.signal_model_name):
        dataset = config.dataset
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
            if (n % 2 == 0):
                plt.plot(to_numpy(rr), to_numpy(func), 2, color=cmap.color(to_numpy(type_list)[n].astype(int)),
                         linewidth=2, alpha=0.25)
        plt.xlim(config.plotting.xlim)
        plt.ylim(config.plotting.ylim)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.xlabel('$v_j$', fontsize=48)
        plt.ylabel('$\mathrm{MLP_1}(a_j, v_j)$', fontsize=48)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/function/lin_edge/func_{epoch}_{N}.tif", dpi=87)
        plt.close()

    rr = torch.linspace(config.plotting.xlim[0], config.plotting.xlim[1], 1000, device=device)
    fig = plt.figure(figsize=(8, 8))
    for n in range(n_neurons):
        embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
        if 'generic' in config.graph_model.update_type:
            in_features = torch.cat((rr[:, None], embedding_, rr[:, None] * 0, rr[:, None] * 0), dim=1)
        else:
            in_features = torch.cat((rr[:, None], embedding_), dim=1)
        with torch.no_grad():
            func = model.lin_phi(in_features.float())
        if (n % 2 == 0):
            plt.plot(to_numpy(rr), to_numpy(func), 2,
                     color=cmap.color(to_numpy(type_list)[n].astype(int)),
                     linewidth=1, alpha=0.1)
    plt.xlim(config.plotting.xlim)
    plt.ylim(config.plotting.ylim)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlabel('$v_i$', fontsize=48)
    plt.ylabel('$\mathrm{MLP_0}(a_i, v_i)$', fontsize=48)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/function/lin_phi/func_{epoch}_{N}.tif", dpi=87)
    plt.close()

def plot_training_signal_field(x, n_nodes, recurrent_loop, kk, time_step, x_list, run, model, field_type, model_f, edges, y_list, ynorm, delta_t, n_frames, log_dir, epoch, N, recurrent_parameters, modulation, device):
    if recurrent_loop > 1:
        x = torch.tensor(x_list[run][kk], device=device).clone().detach()
        ids = np.arange(kk, kk + recurrent_loop * time_step, time_step)
        true_activity_list = np.transpose(x_list[run][ids.astype(int), :, 6:7].squeeze())
        true_modulation_list = np.transpose(x_list[run][ids.astype(int), :, 8:9].squeeze())
        loss = 0
        pred_activity_list = list([])
        pred_modulation_list = list([])
        for loop in range(recurrent_loop):
            pred_activity_list.append(x[:, 6:7].clone().detach())
            if (loop == 0) & ('learnable_short_term_plasticity' in field_type):
                alpha = (kk % model.embedding_step) / model.embedding_step
                x[:, 8] = alpha * model.b[:, kk // model.embedding_step + 1] ** 2 + (1 - alpha) * model.b[:, kk // model.embedding_step] ** 2
            elif ('short_term_plasticity' in field_type):
                t = torch.zeros((1, 1, 1), dtype=torch.float32, device=device)
                t[:, 0, :] = torch.tensor(kk / n_frames, dtype=torch.float32, device=device)
                x[:, 8] = model_f(t.clone().detach()) ** 2
            pred_modulation_list.append(x[:, 8:9].clone().detach())
            dataset = data.Data(x=x, edge_index=edges)
            y = torch.tensor(y_list[run][kk], device=device) / ynorm
            pred = model(dataset)
            loss = loss + (pred - y).norm(2)
            kk = kk + time_step
            if 'learnable_short_term_plasticity' in field_type:
                in_modulation = torch.cat((x[:, 6:7], x[:, 8:9]), dim=1)
                pred_modulation = model.lin_modulation(in_modulation)
                x[:, 8:9] = x[:, 8:9] + delta_t * time_step * pred_modulation
            x[:, 6:7] = x[:, 6:7] + delta_t * time_step * pred
        pred_activity_list = torch.stack(pred_activity_list).squeeze().t()
        pred_modulation_list = torch.stack(pred_modulation_list).squeeze().t()
        kk = kk - time_step * recurrent_loop
        fig = plt.figure(figsize=(12, 12))
        ind_list = [10, 124, 148, 200, 250, 300]
        ax = fig.add_subplot(2, 1, 1)
        ids = np.arange(0, recurrent_loop * time_step, time_step)
        for ind in ind_list:
            plt.plot(ids, true_activity_list[ind, :], c='k', alpha=0.5, linewidth=8)
            plt.plot(ids, to_numpy(pred_activity_list[ind, :]))
        plt.text(0.05, 0.95, f'k: {kk}   loss: {np.round(loss.item(), 3)}', ha='left', va='top', transform=ax.transAxes, fontsize=10)
        if 'learnable_short_term_plasticity' in field_type:
            ax = fig.add_subplot(2, 1, 2)
            for ind in ind_list:
                plt.plot(ids, true_modulation_list[ind, :], c='k', alpha=0.5, linewidth=8)
                plt.plot(ids, to_numpy(pred_modulation_list[ind, :]))
        plt.savefig(f"./{log_dir}/tmp_training/field/Field_{epoch}_{N}.tif")
        plt.close()

    if 'learnable_short_term_plasticity' in field_type:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(2, 2, 1)
        plt.imshow(to_numpy(modulation), aspect='auto')
        ax = fig.add_subplot(2, 2, 2)
        plt.imshow(to_numpy(model.b ** 2), aspect='auto')
        ax.text(0.01, 0.99, f'recurrent_parameter {recurrent_parameters[0]:0.3f} ', transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left', color='w')
        ax.text(0.01, 0.95, f'loop {recurrent_loop} ', transform=ax.transAxes,
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
        ax = fig.add_subplot(2, 2, 1)
        plt.title('neural field')
        plt.imshow(to_numpy(prediction), aspect='auto', cmap='viridis')
        ax = fig.add_subplot(2, 2, 2)
        plt.title('true activity')
        activity = torch.tensor(x_list[0][:, :, 6:7], device=device)
        activity = activity.squeeze()
        activity = activity.t()
        plt.imshow(to_numpy(activity), aspect='auto', cmap='viridis')
        plt.tight_layout()
        ax = fig.add_subplot(2, 2, 3)
        plt.title('learned missing activity')
        pos = np.argwhere(x_list[0][k][:, 6] != baseline_value)
        prediction_ = prediction.clone().detach()
        prediction_[pos[:,0]]=0
        plt.imshow(to_numpy(prediction_), aspect='auto', cmap='viridis')
        ax = fig.add_subplot(2, 2, 4)
        plt.title('learned observed activity')
        pos = np.argwhere(x_list[0][k][:, 6] == baseline_value)
        prediction_ = prediction.clone().detach()
        prediction_[pos[:,0]]=0
        plt.imshow(to_numpy(prediction_), aspect='auto', cmap='viridis')
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/field/missing_activity_{epoch}_{N}.tif", dpi=80)
        plt.close()

def analyze_edge_function(rr=[], vizualize=False, config=None, model_MLP=[], model=None, n_nodes=0, n_neurons=None, ynorm=None, type_list=None, cmap=None, update_type=None, device=None):

    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    dimension = config.simulation.dimension

    if config.graph_model.particle_model_name != '':
        config_model = config.graph_model.particle_model_name
    elif config.graph_model.signal_model_name != '':
        config_model = config.graph_model.signal_model_name
    elif config.graph_model.mesh_model_name != '':
        config_model = config.graph_model.mesh_model_name

    if rr==[]:
        if config_model == 'PDE_G':
            rr = torch.tensor(np.linspace(0, max_radius * 1.3, 1000)).to(device)
        elif config_model == 'PDE_GS':
            rr = torch.tensor(np.logspace(7, 9, 1000)).to(device)
        elif config_model == 'PDE_E':
            rr = torch.tensor(np.linspace(min_radius, max_radius, 1000)).to(device)
        elif 'PDE_N' in config_model:
            rr = torch.tensor(np.linspace(-5, 5, 1000)).to(device)
        else:
            rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)

    print('interaction functions ...')
    func_list = []
    for n in range(n_neurons):

        if len(model.a.shape)==3:
            model_a= model.a[1, n, :]
        else:
            model_a = model.a[n, :]

        if config.training.do_tracking:
            embedding_ = model_a * torch.ones((1000, dimension), device=device)
        else:
            if (update_type != 'NA') & model.embedding_trial:
                embedding_ = torch.cat((model_a, model.b[0].clone().detach().repeat(n_neurons, 1)), dim=1) * torch.ones((1000, 2*dimension), device=device)
            else:
                embedding_ = model_a * torch.ones((1000, dimension), device=device)

        if update_type == 'NA':
            in_features = get_in_features(rr=rr, embedding=embedding_, model=model, model_name=config_model, max_radius=max_radius)
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
            model = Interaction_Mouse(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos,
                                     dimension=dimension)
        case 'PDE_MPM' | 'PDE_MPM_A':
            model = Interaction_MPM(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos,
                                    dimension=dimension)
        case  'PDE_Cell' | 'PDE_Cell_area':
            model = Interaction_Cell(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos, dimension=dimension)
            model.edges = []
        case 'PDE_ParticleField_A' | 'PDE_ParticleField_B':
            model = Interaction_Particle_Field(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos, dimension=dimension)
            model.edges = []
        case 'PDE_Agents' | 'PDE_Agents_A' | 'PDE_Agents_B' | 'PDE_Agents_C':
            model = Interaction_Agent(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos, dimension=dimension)
        case 'PDE_A' | 'PDE_A_bis' | 'PDE_B' | 'PDE_B_mass' | 'PDE_B_bis' | 'PDE_E' | 'PDE_G' | 'PDE_K' | 'PDE_T':
            model = Interaction_Particle(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos, dimension=dimension)
            model.edges = []
            if 'PDE_K' in model_name:
                model.connection_matrix = torch.load(f'./graphs_data/{dataset_name}/connection_matrix_list.pt', map_location=device)
        case 'PDE_GS':
            model = Interaction_Planet(aggr_type=aggr_type, config=model_config, device=device)
            t = np.arange(model_config.simulation.n_neurons)
            t1 = np.repeat(t, model_config.simulation.n_neurons)
            t2 = np.tile(t, model_config.simulation.n_neurons)
            e = np.stack((t1, t2), axis=0)
            pos = np.argwhere(e[0, :] - e[1, :] != 0)
            e = e[:, pos]
            model.edges = torch.tensor(e, dtype=torch.long, device=device)
        case 'PDE_GS2':
            model = Interaction_Planet2(aggr_type=aggr_type, config=model_config, device=device)
            t = np.arange(model_config.simulation.n_neurons)
            t1 = np.repeat(t, model_config.simulation.n_neurons)
            t2 = np.tile(t, model_config.simulation.n_neurons)
            e = np.stack((t1, t2), axis=0)
            pos = np.argwhere(e[0, :] - e[1, :] != 0)
            e = e[:, pos]
            model.edges = torch.tensor(e, dtype=torch.long, device=device)
        case 'PDE_Cell_A' | 'PDE_Cell_B' | 'PDE_Cell_B_area' | 'PDE_Cell_A_area':
            model = Interaction_Cell(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos, dimension=dimension)
        case 'PDE_F_A' |'PDE_F_B'|'PDE_F_C'|'PDE_F_D'|'PDE_F_E' :
            model = Interaction_Smooth_Particle(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos, dimension=dimension)
        case 'PDE_MLPs' | 'PDE_MLPs_A' | 'PDE_MLPs_A_bis' | 'PDE_MLPs_A_ter' | 'PDE_MLPs_B'| 'PDE_MLPs_B_0' |'PDE_MLPs_B_1' | 'PDE_MLPs_B_4'| 'PDE_MLPs_B_10' |'PDE_MLPs_C' | 'PDE_MLPs_D' | 'PDE_MLPs_E' | 'PDE_MLPs_F':
            model = Interaction_PDE_Particle(aggr_type=aggr_type, config=model_config, device=device,
                                                bc_dpos=bc_dpos, dimension=dimension)
        case 'PDE_M' | 'PDE_M2':
            model = Interaction_Particle2(aggr_type=aggr_type, config=model_config, bc_dpos=bc_dpos, dimension=dimension, device=device)
        case 'PDE_MM' | 'PDE_MM_1layer' | 'PDE_MM_2layers' | 'PDE_MM_3layers':
            model = Interaction_Particle3(aggr_type=aggr_type, config=model_config, bc_dpos=bc_dpos, dimension=dimension, device=device)
        case 'PDE_MS':
            model = Interaction_Falling_Water_Smooth(aggr_type=aggr_type, config=model_config,bc_dpos=bc_dpos, dimension=dimension, device=device)

    model_name = model_config.graph_model.mesh_model_name
    match model_name:
        case 'DiffMesh':
            model = Mesh_Laplacian(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos)
            model.edges = []
        case 'WaveMesh':
            model = Mesh_Laplacian(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos)
            model.edges = []
        case 'WaveMeshSmooth':
            model = Mesh_Smooth(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos)
            model.edges = []
        case 'RD_Mesh' | 'RD_Mesh2' | 'RD_Mesh3' | 'RD_Mesh4':
            model = Mesh(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos)
            model.edges = []

    model_name = model_config.graph_model.signal_model_name
    match model_name:
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
    n_total_params = sum(p.numel() for p in trainable_params) + torch.numel(model.a)

    # embedding = model.a
    # optimizer = torch.optim.Adam([embedding], lr=lr_embedding)
    #
    # _, *parameters = trainable_params
    # for parameter in parameters:
    #     optimizer.add_param_group({'params': parameter, 'lr': lr})

    if lr_update==[]:
        lr_update = lr

    optimizer = torch.optim.Adam([model.a], lr=lr_embedding)
    for name, parameter in model.named_parameters():
        if (parameter.requires_grad) & (name!='a'):
            if (name=='b') or ('lin_modulation' in name):
                optimizer.add_param_group({'params': parameter, 'lr': lr_modulation})
                # print(f'lr_modulation: {name} {lr_modulation}')
            elif 'lin_phi' in name:
                optimizer.add_param_group({'params': parameter, 'lr': lr_update})
                # print(f'lr_W: {name} {lr_W}')
            elif 'W' in name:
                optimizer.add_param_group({'params': parameter, 'lr': lr_W})
            elif 'NNR_f' in name:
                optimizer.add_param_group({'params': parameter, 'lr': learning_rate_NNR_f})
                # print(f'lr_W: {name} {lr_W}')
            elif 'NNR' in name:
                optimizer.add_param_group({'params': parameter, 'lr': learning_rate_NNR})
                # print(f'lr_W: {name} {lr_W}')
            else:
                optimizer.add_param_group({'params': parameter, 'lr': lr})
                # print(f'lr: {name} {lr}')

    return optimizer, n_total_params

def set_trainable_parameters_vae(
    model=[],                    # global fallback LR (like your original `lr`)
    lr_encoder=[],
    lr_latent_update=[],
    lr_decoder=[],         # optional WD for all groups
):

    trainable_params = [p for _, p in model.named_parameters() if p.requires_grad]
    n_total_params = sum(p.numel() for p in trainable_params)

    param_groups = []
    for name, parameter in model.named_parameters():
        if (not parameter.requires_grad):
            continue
        if "encoder" in name:
            param_groups.append({"params": [parameter], "lr": lr_encoder})
        elif "update_latent" in name:
            param_groups.append({"params": [parameter], "lr": lr_latent_update})
        elif "decoder" in name:
            param_groups.append({"params": [parameter], "lr": lr_decoder})
        else:
            param_groups.append({"params": [parameter], "lr": lr})

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
    embeddings_by_neuron = []
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

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
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

def get_n_hop_neighborhood(target_ids, edges_all, n_hops):
    """Get n-hop neighborhood of target neurons"""
    current = set(target_ids)
    all_neurons = set(target_ids)
    
    for hop in range(n_hops):
        next_hop = set()
        for node in current:
            # Find predecessors (neurons that send to current)
            mask = edges_all[1, :] == node
            predecessors = edges_all[0, mask].cpu().numpy()
            next_hop.update(predecessors)
        
        all_neurons.update(next_hop)
        current = next_hop
    
    return np.array(sorted(all_neurons))