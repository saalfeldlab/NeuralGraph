import os
import umap
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter
from torch_geometric.nn import MessagePassing
import torch_geometric.utils as pyg_utils
import imageio.v2 as imageio
from matplotlib import rc
from NeuralGraph.utils import set_size
from scipy.ndimage import median_filter

# os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2023/bin/x86_64-linux'

# from data_loaders import *

from GNN_Main import *

from NeuralGraph.fitting_models import *
from NeuralGraph.sparsify import *
from NeuralGraph.models.utils import *
from NeuralGraph.models.plot_utils import *
from NeuralGraph.models.MLP import *
from NeuralGraph.utils import to_numpy, CustomColorMap
import matplotlib as mpl
from io import StringIO
import sys
from scipy.stats import pearsonr
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.mixture import GaussianMixture
import warnings
import seaborn as sns
import glob
import shutil
from skimage import measure, morphology, segmentation, filters
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import pickle
import json

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, accuracy_score
from scipy.optimize import linear_sum_assignment
# from pysr import PySRRegressor
from datetime import datetime
from NeuralGraph.spectral_utils.myspectral_funcs import estimate_spectrum, compute_spectral_coefs
from scipy.special import logsumexp


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
    for run in trange(n_runs):
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


def plot_synaptic_CElegans(config, epoch_list, log_dir, logger, cc, style, device):

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

    black_to_green = LinearSegmentedColormap.from_list('black_green', ['black', 'green'])

    x_list = []
    y_list = []
    for run in trange(0,n_runs):
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

    edges = torch.load(f'./graphs_data/{dataset_name}/edge_index.pt', map_location=device)
    edges_all = edges.clone().detach()

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
        d_modulation = (modulation[:, 1:] - modulation[:, :-1]) / delta_t
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
            for file_id_ in trange(0,len(file_id_list)):
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

                larynx_pred_weight, index_larynx = map_matrix(larynx_neuron_list, activity_neuron_list, A)
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
                            plt.ylabel(r'learned $MLP_1( a_i, a_j, x_j)$', fontsize=32)
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
                    plt.ylabel(r'learned $MLP_1(a_j, x_j)$', fontsize=68)
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
                    # plt.ylabel(r'learned $\psi^*(a_i, x_i)$', fontsize=68)
                    plt.ylabel(r'$MLP_1(a_i, a_j, x_i, x_j)$', fontsize=68)
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
                    # plt.ylabel(r'learned $\psi^*(a_i, x_i)$', fontsize=68)
                    plt.ylabel(r'learned $MLP_1(a_j, x_j)$', fontsize=68)
                    plt.ylim([-1.5, 1.5])
                    plt.xlim([-5,5])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/all/MLP1_{epoch}.png", dpi=80)
                    plt.close()

                fig, ax = fig_init()
                func_list = []
                config_model = config.graph_model.signal_model_name
                for n in range(n_neurons):
                    embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                    # in_features = torch.cat((rr[:, None], embedding_), dim=1)
                    in_features = get_in_features_update(rr[:, None], model, embedding_, device)
                    with torch.no_grad():
                        func = model.lin_phi(in_features.float())
                    func = func[:, 0]
                    plt.plot(to_numpy(rr), to_numpy(func) * to_numpy(ynorm), color=cmap.color(to_numpy(type_list[n]).astype(int)), linewidth=2, alpha=0.25) #
                plt.ylim([-4,4])
                plt.xlabel(r'$x_i$', fontsize=68)
                # plt.ylabel(r'learned $\phi^*(a_i, x_i)$', fontsize=68)
                plt.ylabel(r'learned $MLP_0(a_i, x_i)$', fontsize=68)

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
                    plt.xlabel(r'$x_i$', fontsize=68)
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
                    plt.xlabel(r'$x_i$', fontsize=68)
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
                architecture_types = results['architecture_analysis']['architecture_data']
                hub_neurons = results['hub_analysis']
                pathway_patterns = results['pathway_analysis']
                summary_plot = results['summary_figure']
                # Finds neurons that appear in BOTH high connectivity AND high odor responses
                # These are the "bridge" neurons between detection and integration
                preprocessing_results = run_preprocessing_analysis(
                    top_pairs_by_run,
                    odor_responses_by_run,
                    results['architecture_analysis'],  # From your previous analysis
                    all_neuron_list
                )
                preprocessing_results['preprocessing_figure'].savefig(
                    f"./{log_dir}/results/preprocessing_analysis.png",
                    dpi=150, bbox_inches='tight'
                )
                plt.close()
                with open(f"./{log_dir}/results/preprocessing_analysis.pkl", 'wb') as f:
                    pickle.dump(preprocessing_results, f)

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
            # plt.xlabel(r'$x_i$', fontsize=68)
            # plt.ylabel(r'Learned $\psi^*(a_i, x_i)$', fontsize=68)
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
            #             # plt.ylabel(r'learned $\psi^*(a_i, a_j, x_i)$', fontsize=24)
            #             # plt.xlabel(r'$x_i$', fontsize=24)
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
            #     plt.xlabel(r'$x_i$', fontsize=68)
            #     if (model_config.signal_model_name == 'PDE_N4'):
            #         plt.ylabel(r'learned $\psi^*(a_i, x_i)$', fontsize=68)
            #     elif model_config.signal_model_name == 'PDE_N5':
            #         plt.ylabel(r'learned $\psi^*(a_i, a_j, x_i)$', fontsize=68)
            #     else:
            #         plt.ylabel(r'learned $\psi^*(x_i)$', fontsize=68)
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
            # plt.xlabel(r'$x_i$', fontsize=68)
            # plt.ylabel(r'learned $\phi^*(a_i, x_i)$', fontsize=68)
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
                    plt.xlabel(r'$x_i$', fontsize=68)
                    plt.ylabel(r'$y_i$', fontsize=68)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/true_field_derivative.png", dpi=80)
                    plt.close()

                    fig, ax = fig_init()
                    plt.title(r'learned $\dot{y_i}$', fontsize=68)
                    plt.imshow(to_numpy(pred_modulation))
                    plt.xticks([0, 100, 200, 300, 400], [-6, -3, 0, 3, 6], fontsize=48)
                    plt.yticks([0, 100, 200, 300, 400], [0, 0.25, 0.5, 0.75, 1], fontsize=48)
                    plt.xlabel(r'$x_i$', fontsize=68)
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
                        # plt.title(r'$x_i$', fontsize=48)
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
            #     plt.xlabel(r'$x_i$', fontsize=68)
            #     plt.ylabel(r'learned $MLP_0$', fontsize=68)
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
            #     plt.xlabel(r'$x_i$', fontsize=68)
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
            #     plt.ylabel(r'$MLP_0(a_i, x_i=0, sum_i, g_i=1)$', fontsize=48)
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


def plot_synaptic_flyvis(config, epoch_list, log_dir, logger, cc, style, device):
    dataset_name = config.dataset
    model_config = config.graph_model
    config_indices = config.dataset.split('fly_N9_')[1] if 'fly_N9_' in config.dataset else 'evolution'

    n_frames = config.simulation.n_frames
    n_runs = config.training.n_runs
    n_neurons = config.simulation.n_neurons
    n_neuron_types = config.simulation.n_neuron_types
    n_input_neurons = config.simulation.n_input_neurons
    field_type = model_config.field_type
    delta_t = config.simulation.delta_t

    colors_65 = sns.color_palette("Set3", 12) * 6  # pastel, repeat until 65
    colors_65 = colors_65[:65]

    max_radius = config.simulation.max_radius if hasattr(config.simulation, 'max_radius') else 2.5
    dimension = config.simulation.dimension

    log_file = os.path.join(log_dir, 'results.log')
    if os.path.exists(log_file):
        os.remove(log_file)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=log_file,
        format='%(asctime)s %(message)s',
        filemode='w'
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    print(f'experiment description: {config.description}')
    logger.info(f'experiment description: {config.description}')

    # Load neuron group mapping for flyvis
    group_names = ['R1-R6', 'R7-R8', 'L1-L5', 'Lamina_Inter', 'Mi_Early', 'Mi_Mid', 'Mi_Late',
                   'Tm_Early', 'Tm5_Family', 'Tm_Mid', 'Tm_Late', 'TmY', 'T4a_Up', 'T4b_Right',
                   'T4c_Down', 'T4d_Left', 'T5_OFF', 'Tangential', 'Wide_Field', 'Other']

    region_colors = {
        'Retina': ['R1-R6', 'R7-R8'],
        'Lamina': ['L1-L5', 'Lamina_Inter'],
        'Medulla_Mi': ['Mi_Early', 'Mi_Mid', 'Mi_Late'],
        'Medulla_Tm': ['Tm_Early', 'Tm5_Family', 'Tm_Mid', 'Tm_Late', 'TmY'],
        'T4_Motion': ['T4a_Up', 'T4b_Right', 'T4c_Down', 'T4d_Left'],
        'T5_Motion': ['T5_OFF'],
        'Other': ['Tangential', 'Wide_Field', 'Other']
    }
    cmap = CustomColorMap(config=config)

    if 'black' in style:
        mc = 'w'
    else:
        mc = 'k'

    x_list = []
    y_list = []
    time.sleep(0.5)
    print('load data ...')
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

    vnorm = torch.load(os.path.join(log_dir, 'vnorm.pt'))
    ynorm = torch.load(os.path.join(log_dir, 'ynorm.pt'))
    if os.path.exists(os.path.join(log_dir, 'xnorm.pt')):
        xnorm = torch.load(os.path.join(log_dir, 'xnorm.pt'))
    else:
        xnorm = torch.tensor([5], device=device)

    print(f'xnorm: {to_numpy(xnorm)}, vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')
    logger.info(f'xnorm: {to_numpy(xnorm)}, vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')

    # Load data with new format
    # connectivity = torch.load(f'./graphs_data/{dataset_name}/connectivity.pt', map_location=device)
    gt_weights = torch.load(f'./graphs_data/{dataset_name}/weights.pt', map_location=device)
    gt_taus = torch.load(f'./graphs_data/{dataset_name}/taus.pt', map_location=device)
    gt_V_Rest = torch.load(f'./graphs_data/{dataset_name}/V_i_rest.pt', map_location=device)

    edges = torch.load(f'./graphs_data/{dataset_name}/edge_index.pt', map_location=device)
    true_weights = torch.zeros((n_neurons, n_neurons), dtype=torch.float32, device=edges.device)
    true_weights[edges[1], edges[0]] = gt_weights

    if  True: #os.path.exists(f"./{log_dir}/results/E_panels.png"):
        print (f'skipping computation of energy plot ...')
    else:
        energy_stride = 1
        s, h, J, E = sparse_ising_fit_fast(x=x_list[0], voltage_col=3, top_k=50, block_size=2000, energy_stride=energy_stride)

        # Create weights matrix
        true_weights = torch.zeros((n_neurons, n_neurons), dtype=torch.float32, device=edges.device)
        true_weights[edges[1], edges[0]] = gt_weights

        precision, recall, f1 = analysis_J_W(J, to_numpy(true_weights))

        # Panel 4: Compute approximate probabilities for observed states
        # P(s) ~ exp(-E(s)/T), approximate T=1 for simplicity
        T = 1.0
        logP = -E / T
        logP -= logsumexp(logP)  # normalize in log space
        P_s = np.exp(logP) # normalize to get a probability distribution

        # Flatten all non-zero couplings for histogram
        J_vals = [v for Ji in J for v in Ji.values()]
        J_vals = np.array(J_vals, dtype=np.float32)

        # Create 2x2 figure
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # Panel 1: Energy over time
        axs[0, 0].plot(np.arange(0, len(E) * energy_stride, energy_stride), E, lw=1.0, color=mc)
        axs[0, 0].set_xlabel("Frame", fontsize=14)
        axs[0, 0].set_ylabel("Energy", fontsize=14)
        axs[0, 0].set_title("Ising energy over frames", fontsize=14)
        axs[0, 0].set_xlim(0, 600)
        axs[0, 0].tick_params(axis='both', which='major', labelsize=12)

        # Panel 2: Energy histogram
        axs[0, 1].hist(E, bins=100, color='salmon', edgecolor=mc, density=True)
        axs[0, 1].set_xlabel("Energy", fontsize=14)
        axs[0, 1].set_ylabel("Density", fontsize=14)
        axs[0, 1].set_title("Ising energy distribution", fontsize=14)
        axs[0, 1].tick_params(axis='both', which='major', labelsize=12)

        # Panel 3: Couplings histogram
        axs[1, 0].hist(J_vals, bins=100, color='skyblue', edgecolor=mc, density=True)
        axs[1, 0].set_xlabel(r"Coupling strength $J_{ij}$", fontsize=14)
        axs[1, 0].set_ylabel("Density", fontsize=14)
        axs[1, 0].set_title("Sparse Couplings Histogram", fontsize=14)
        axs[1, 0].tick_params(axis='both', which='major', labelsize=12)
        axs[1, 0].text(0.05, 0.95, f'Precision: {precision:.3f}', transform=axs[1, 0].transAxes,
                       fontsize=12, verticalalignment='top')

        axs[1, 1].axis('off')  # Empty panel

        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/E_panels.png", dpi=150)
        plt.close(fig)

        np.save(f"./{log_dir}/results/E.npy", E)
        np.save(f"./{log_dir}/results/s.npy", s)
        np.save(f"./{log_dir}/results/h.npy", h)
        np.save(f"./{log_dir}/results/J.npy", J)

    x = x_list[0][n_frames - 10]
    type_list = torch.tensor(x[:, 2 + 2 * dimension:3 + 2 * dimension], device=device)
    n_types = len(np.unique(to_numpy(type_list)))
    # print (f'{n_types} neuron types in datasets')
    region_list = torch.tensor(x[:, 1 + 2 * dimension:2 + 2 * dimension], device=device)
    n_region_types = len(np.unique(to_numpy(region_list)))
    n_neurons = len(type_list)

    index_to_name = {
        0: 'Am', 1: 'C2', 2: 'C3', 3: 'CT1(Lo1)', 4: 'CT1(M10)', 5: 'L1', 6: 'L2', 7: 'L3', 8: 'L4',
        9: 'L5',
        10: 'Lawf1', 11: 'Lawf2', 12: 'Mi1', 13: 'Mi10', 14: 'Mi11', 15: 'Mi12', 16: 'Mi13', 17: 'Mi14',
        18: 'Mi15', 19: 'Mi2',
        20: 'Mi3', 21: 'Mi4', 22: 'Mi9', 23: 'R1', 24: 'R2', 25: 'R3', 26: 'R4', 27: 'R5', 28: 'R6',
        29: 'R7',
        30: 'R8', 31: 'T1', 32: 'T2', 33: 'T2a', 34: 'T3', 35: 'T4a', 36: 'T4b', 37: 'T4c', 38: 'T4d',
        39: 'T5a',
        40: 'T5b', 41: 'T5c', 42: 'T5d', 43: 'Tm1', 44: 'Tm16', 45: 'Tm2', 46: 'Tm20', 47: 'Tm28',
        48: 'Tm3', 49: 'Tm30',
        50: 'Tm4', 51: 'Tm5Y', 52: 'Tm5a', 53: 'Tm5b', 54: 'Tm5c', 55: 'Tm9', 56: 'TmY10', 57: 'TmY13',
        58: 'TmY14', 59: 'TmY15',
        60: 'TmY18', 61: 'TmY3', 62: 'TmY4', 63: 'TmY5a', 64: 'TmY9'
    }

    activity = torch.tensor(x_list[0][:, :, 3:4], device=device)
    activity = activity.squeeze()
    activity = activity.t()

    mu_activity = torch.mean(activity, dim=1)  # shape: (n_neurons,)
    sigma_activity = torch.std(activity, dim=1)

    plt.figure(figsize=(15, 10))  # Made wider to accommodate labels
    plt.errorbar(np.arange(n_neurons), to_numpy(mu_activity), yerr=to_numpy(sigma_activity), fmt='o',
                 ecolor='lightgray', alpha=0.1, elinewidth=1, capsize=0, markersize=2, color='red')

    # Add neuron type labels for unique positions only
    type_positions = {}
    for i in range(n_neurons):
        neuron_type_id = to_numpy(type_list[i]).item()
        if neuron_type_id not in type_positions:
            type_positions[neuron_type_id] = i

    # Add labels at representative positions for each neuron type
    for neuron_type_id, pos in type_positions.items():
        neuron_type_name = index_to_name.get(neuron_type_id, f'Type{neuron_type_id}')
        plt.text(pos+100, to_numpy(mu_activity[pos]) + to_numpy(sigma_activity[pos]) * 1.1 + 0.1, neuron_type_name,
                 rotation=90, ha='center', va='bottom', fontsize=8, alpha=0.7)

    plt.xlabel('neuron', fontsize=24)
    plt.ylabel(r'$\mu_i \pm \sigma_i$', fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(r'$\mu_i \pm \sigma_i$ for each neuron', fontsize=24)
    plt.tight_layout()
    plt.savefig(f'./{log_dir}/results/activity_{config_indices}_mu_sigma.png', dpi=300)
    plt.close()

    plt.figure(figsize=(10, 10))
    n = np.random.randint(0, n_neurons, 10)
    for i in range(len(n)):
        plt.plot(to_numpy(activity[n[i].astype(int), :]), linewidth=1)
    plt.xlabel('time', fontsize=24)
    plt.ylabel(r'$x_i$', fontsize=24)
    plt.xlim([0, n_frames // 400])
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim([-6, 6])
    plt.savefig(f'./{log_dir}/results/activity.png', dpi=300)
    plt.close()

    # Additional plot for specific neuron type (e.g., 'Am')
    target_type_name_list = ['R1', 'R7', 'Am', 'L1', 'L2', 'TmY4', 'TmY5a', 'TmY9', 'T4a', 'T4b', 'T4c', 'T4d']

    for target_type_name in target_type_name_list:  # Change this to any desired type name
        target_type_index = None
        for idx, name in index_to_name.items():
            if name == target_type_name:
                target_type_index = idx
                break
        if target_type_index is not None:
            type_mask = (to_numpy(type_list).squeeze() == target_type_index)
            neurons_of_type = np.where(type_mask)[0]
            if len(neurons_of_type) > 0:
                # Select up to 10 neurons of this type
                n_neurons_to_plot = min(10, len(neurons_of_type))
                selected_neurons = neurons_of_type[:n_neurons_to_plot]

                plt.figure(figsize=(10, 10))
                for i, neuron_idx in enumerate(selected_neurons):
                    plt.plot(to_numpy(activity[neuron_idx, :]), linewidth=1,
                             label=f'{target_type_name}_{i}' if n_neurons_to_plot <= 5 else None)

                plt.xlabel('time', fontsize=24)
                plt.ylabel(r'$x_i$', fontsize=24)
                plt.xlim([0, n_frames // 400])
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                plt.title(f'x_i samples - {target_type_name} neurons ({n_neurons_to_plot}/{len(neurons_of_type)})',
                          fontsize=24)
                if n_neurons_to_plot <= 5:
                    plt.legend(fontsize=24)
                plt.ylim([-5, 5])
                plt.tight_layout()
                plt.savefig(f'./{log_dir}/results/activity_{target_type_name}.png', dpi=300)
                plt.close()


                N_samples = 1000

                xnt = to_numpy(activity[neuron_idx, :])
                ynt = None
                fs = 1 / delta_t
                window = "hann"
                nperseg = int(fs * 0.8)

                # print (f'test: {N_samples / nperseg}')

                noverlap = None
                nfft = None
                detrend = "constant"
                return_onesided = True
                scaling = "spectrum"
                abs = True
                return_coefs = True

                num_levels = 10
                reps = 2

                xnt = xnt[0:N_samples]
                pxy, freqs, coefs_xnkf = estimate_spectrum(xnt, ynt =ynt, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft, detrend=detrend, return_onesided=return_onesided, scaling=scaling, abs=abs, return_coefs=return_coefs)


                # coefs_xnkfs, freqs = myspectral_funcs.compute_multiscale_spectral_coefs(xnt=y_ns[None], fs=fps,
                #                                                                          window="hann", noverlap=None,
                #                                                                          detrend="constant",
                #                                                                          return_onesided=True,
                #                                                                          scaling="spectrum", axis=0,
                #                                                                          num_levels=num_levels,
                #                                                                          reps=reps)

                coefs_xnkf = coefs_xnkf.compute()
                pxy = pxy.compute()
                min_freq = freqs[0]
                max_freq = freqs[-1]
                spectrogram = np.abs(coefs_xnkf[0]).T[::-1]

                fig, axs = plt.subplots(2, 1, figsize=(12, 6))
                cm = plt.get_cmap("coolwarm")
                xax = np.linspace(0, xnt.shape[0] * delta_t, spectrogram.shape[1])
                im = axs[0].matshow(spectrogram, extent=[xax[0], xax[-1], min_freq, max_freq],
                                    cmap=cm, aspect="auto", origin="lower")
                axs[0].set_xlabel("Time (s)")
                axs[0].set_ylabel("Frequency (Hz)")
                # fig.colorbar(im, ax=axs[0])
                xax = np.arange(0, xnt.shape[0]) * delta_t
                axs[1].plot(xax, xnt, color=mc, linewidth=0.5)
                axs[1].set_xlabel("Time (s)")
                axs[1].set_xlim([0, xax[-1]])
                fig.tight_layout()
                plt.savefig(f'./{log_dir}/results/spectrogram_{target_type_name}.png', dpi=300)
                plt.close(fig)
                # print(f'./{log_dir}/results/spectrogram_{target_type_name}.png')
                # print(f'plotted {n_neurons_to_plot} out of {len(neurons_of_type)} {target_type_name} neurons')
                logger.info(f'plotted {n_neurons_to_plot} out of {len(neurons_of_type)} {target_type_name} neurons')
            else:
                print(f'no neurons found for type {target_type_name}')
                logger.info(f'no neurons found for type {target_type_name}')
        else:
            print(f'type {target_type_name} not found in index_to_name dictionary')
            logger.info(f'type {target_type_name} not found in index_to_name dictionary')

    print(f'neurons: {n_neurons}')
    print(f'edges: {edges.shape[1]}')
    print(f'neuron types: {n_types}')
    print(f'region types: {n_region_types}')
    logger.info(f'number of neurons: {n_neurons}')
    logger.info(f'true edges: {edges.shape[1]}')
    logger.info(f'true number of neuron types: {n_types}')
    logger.info(f'true number of region types: {n_region_types}')

    os.makedirs(f'{log_dir}/results/', exist_ok=True)

    if epoch_list[0] == 'all':

        config_indices = config.dataset.split('fly_N9_')[1] if 'fly_N9_' in config.dataset else 'evolution'
        files, file_id_list = get_training_files(log_dir, n_runs)
        
        fps = 10  # frames per second for the video
        metadata = dict(title='Model evolution', artist='Matplotlib', comment='Model evolution over epochs')
        writer = FFMpegWriter(fps=fps, metadata=metadata)
        fig = plt.figure(figsize=(18, 24))
        plt.subplots_adjust(hspace=3.0)
        mp4_path = f'{log_dir}/results/training_{config_indices}.mp4'

        with writer.saving(fig, mp4_path, dpi=80):
            for file_id_ in trange(len(file_id_list)):
                epoch = files[file_id_].split('graphs')[1][1:-3]

                net = f'{log_dir}/models/best_model_with_{n_runs - 1}_graphs_{epoch}.pt'
                model = Signal_Propagation_FlyVis(aggr_type=model_config.aggr_type, config=config, device=device)
                state_dict = torch.load(net, map_location=device)
                model.load_state_dict(state_dict['model_state_dict'])
                model.edges = edges
                logger.info(f'net: {net}')

                fig.clf()  # Clear the figure

                ax4 = fig.add_subplot(3, 2, 3)
                slopes_lin_phi_list = []
                offsets_list = []
                func_list = []

                for n in range(n_neurons):
                    if (n % 20 == 0):
                        rr = torch.linspace(config.plotting.xlim[0], config.plotting.xlim[1], 1000, device=device)
                        embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                        in_features = torch.cat(
                            (rr[:, None], embedding_, rr[:, None] * 0, torch.zeros_like(rr[:, None])), dim=1)
                        with torch.no_grad():
                            func = model.lin_phi(in_features.float())
                            ax4.plot(to_numpy(rr), to_numpy(func), 2,
                                     color=cmap.color(to_numpy(type_list)[n].astype(int)),
                                     linewidth=1, alpha=0.025)

                    rr = torch.linspace(mu_activity[n] - 2 * sigma_activity[n], mu_activity[n] + 2 * sigma_activity[n],
                                        1000, device=device)
                    embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                    in_features = torch.cat((rr[:, None], embedding_, rr[:, None] * 0, torch.zeros_like(rr[:, None])),
                                            dim=1)
                    with torch.no_grad():
                        func = model.lin_phi(in_features.float())
                    if (n % 20 == 0):
                        ax4.plot(to_numpy(rr), to_numpy(func), 2,
                                 color=cmap.color(to_numpy(type_list)[n].astype(int)),
                                 linewidth=1, alpha=0.2)
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

                ax4.set_xlim(config.plotting.xlim)
                ax4.set_ylim(config.plotting.ylim)
                ax4.set_xlabel('$x_i$', fontsize=32)
                ax4.set_ylabel('learned $MLP_0(a_i, x_i)$', fontsize=32)
                ax4.tick_params(axis='both', which='major', labelsize=24)

                # Plot 3: Lin_edge functions (middle right) and calculate slopes
                ax3 = fig.add_subplot(3, 2, 4)
                slopes_lin_edge_list = []

                for n in range(n_neurons):
                    if (n % 20 == 0):
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
                        ax3.plot(to_numpy(rr), to_numpy(func), 2,
                                 color=cmap.color(to_numpy(type_list)[n].astype(int)),
                                 linewidth=1, alpha=0.05)

                    rr = torch.linspace(mu_activity[n] - 2 * sigma_activity[n], mu_activity[n] + 2 * sigma_activity[n],
                                        1000, device=device)
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
                    ax3.plot(to_numpy(rr), to_numpy(func), 2,
                             color=cmap.color(to_numpy(type_list)[n].astype(int)),
                             linewidth=1, alpha=0.2)

                    rr_numpy = to_numpy(rr[rr.shape[0] // 2 + 1:])
                    func_numpy = to_numpy(func[rr.shape[0] // 2 + 1:].squeeze())
                    try:
                        lin_fit, _ = curve_fit(linear_model, rr_numpy, func_numpy)
                        slope = lin_fit[0]
                        offset = lin_fit[1]
                    except:
                        coeffs = np.polyfit(rr_numpy, func_numpy, 1)
                        slope = coeffs[0]
                        offset = coeffs[1]
                    slopes_lin_edge_list.append(slope)

                ax3.set_xlim(config.plotting.xlim)
                ax3.set_ylim([-config.plotting.xlim[1] / 10, config.plotting.xlim[1] * 2])
                ax3.set_xlabel('$x_i$', fontsize=32)
                ax3.set_ylabel('learned $MLP_1(a_j, x_i)$', fontsize=32)
                ax3.tick_params(axis='both', which='major', labelsize=24)

                # Calculate corrected_W using proper data construction
                k_list = [1]
                dataset_batch = []
                ids_batch = []
                mask_batch = []
                ids_index = 0
                mask_index = 0
                run = 0

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
                    create_graph=True
                )[0]

                grad_msg_flat = grad_msg.squeeze()
                target_neuron_ids = edges[1, :] % (model.n_edges + model.n_extra_null_edges)
                grad_msg_per_edge = grad_msg_flat[target_neuron_ids]
                grad_msg_per_edge = grad_msg_per_edge.unsqueeze(1)

                slopes_lin_phi_array = torch.tensor(slopes_lin_phi_list, dtype=torch.float32, device=device)
                slopes_lin_phi_per_edge = slopes_lin_phi_array[target_neuron_ids]

                slopes_lin_edge_array = torch.tensor(slopes_lin_edge_list, dtype=torch.float32, device=device)
                prior_neuron_ids = edges[0, :] % (model.n_edges + model.n_extra_null_edges)
                slopes_lin_edge_per_edge = slopes_lin_edge_array[prior_neuron_ids]

                corrected_W = -model.W / slopes_lin_phi_per_edge[:,
                                         None] * grad_msg_per_edge * slopes_lin_edge_per_edge.unsqueeze(1)

                # Plot 1: Corrected weight comparison (top left)
                ax1 = fig.add_subplot(3, 2, 1)
                learned_weights = to_numpy(corrected_W.squeeze())
                true_weights = to_numpy(gt_weights)
                if len(true_weights) > 0 and len(learned_weights) > 0:
                    ax1.scatter(true_weights, learned_weights, c=mc, s=1, alpha=0.05)
                    lin_fit, lin_fitv = curve_fit(linear_model, true_weights, learned_weights)
                    residuals = learned_weights - linear_model(true_weights, *lin_fit)
                    ss_res = np.sum(residuals ** 2)
                    ss_tot = np.sum((learned_weights - np.mean(learned_weights)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot)
                    ax1.text(0.05, 0.95, f'R²: {r_squared:.3f}\nslope: {lin_fit[0]:.2f}\nN: {len(true_weights)}',
                             transform=ax1.transAxes, verticalalignment='top', fontsize=24)
                ax1.set_xlabel('true $W_{ij}$', fontsize=32)
                ax1.set_ylabel('learned $W_{ij}$', fontsize=32)
                ax1.set_xlim([-2, 4.5])
                ax1.set_ylim([-2, 4.5])
                ax1.tick_params(axis='both', which='major', labelsize=24)

                # Plot 2: Embedding (top right)
                with torch.no_grad():
                    ax2 = fig.add_subplot(3, 2, 2)
                    embedding_plot = to_numpy(model.a)
                    for n in range(n_types):
                        type_mask = (to_numpy(type_list).squeeze() == n)
                        if np.any(type_mask):
                            ax2.scatter(embedding_plot[type_mask, 0], embedding_plot[type_mask, 1],
                                        c=colors_65[n], s=6, alpha=0.25, edgecolors='none')
                    ax2.set_xlabel('$a_0$', fontsize=32)
                    ax2.set_ylabel('$a_1$', fontsize=32)
                    ax2.set_xticks([])
                    ax2.set_yticks([])

                # Plot 5: Tau comparison (bottom left)
                ax5 = fig.add_subplot(3, 2, 5)
                slopes_lin_phi_array_np = np.array(slopes_lin_phi_list)
                reconstructed_tau = np.where(slopes_lin_phi_array_np != 0, 1.0 / -slopes_lin_phi_array_np, 1)
                reconstructed_tau = reconstructed_tau[:n_neurons]
                reconstructed_tau = np.clip(reconstructed_tau, 0, 1)
                gt_taus_numpy = to_numpy(gt_taus[:n_neurons])
                lin_fit, lin_fitv = curve_fit(linear_model, gt_taus_numpy, reconstructed_tau)
                residuals = reconstructed_tau - linear_model(gt_taus_numpy, *lin_fit)
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((reconstructed_tau - np.mean(reconstructed_tau)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                ax5.scatter(gt_taus_numpy , reconstructed_tau, c=mc, s=1, alpha=0.25)
                ax5.text(0.05, 0.95, f'R²: {r_squared:.3f}\nslope: {lin_fit[0]:.2f}\nN: {len(gt_taus)}',
                         transform=ax5.transAxes, verticalalignment='top', fontsize=24)
                ax5.set_xlabel('true $\\tau$', fontsize=32)
                ax5.set_ylabel('learned $\\tau$', fontsize=32)
                ax5.set_xlim([0, 0.35])
                ax5.set_ylim([0, 0.35])
                ax5.tick_params(axis='both', which='major', labelsize=24)

                # Plot 6: V_rest comparison (bottom right)
                ax6 = fig.add_subplot(3, 2, 6)
                offsets_array = np.array(offsets_list)
                reconstructed_V_rest = np.where(slopes_lin_phi_array_np != 0, -offsets_array / slopes_lin_phi_array_np, 1)

                gt_V_rest_numpy = to_numpy(gt_V_Rest[:n_neurons])
                lin_fit, lin_fitv = curve_fit(linear_model, gt_V_rest_numpy, reconstructed_V_rest)
                residuals = reconstructed_V_rest - linear_model(gt_V_rest_numpy, *lin_fit)
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((reconstructed_V_rest - np.mean(reconstructed_V_rest)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                ax6.scatter(gt_V_rest_numpy, reconstructed_V_rest, c=mc, s=1, alpha=0.25)
                ax6.text(0.05, 0.95, f'R²: {r_squared:.3f}\nslope: {lin_fit[0]:.2f}\nN: {len(gt_V_rest_numpy)}',
                         transform=ax6.transAxes, verticalalignment='top', fontsize=24)
                ax6.set_xlabel('true $V_{rest}$', fontsize=32)
                ax6.set_ylabel('learned $V_{rest}$', fontsize=32)
                ax6.set_xlim([-0.05, 0.9])
                ax6.set_ylim([-0.05, 0.9])
                ax6.tick_params(axis='both', which='major', labelsize=24)

                plt.tight_layout()

                # Save 3x2 panels as PNG only for first frame
                if file_id_ % 10 == 0:
                    plt.savefig(f'{log_dir}/results/training_{config_indices}.png', dpi=300, bbox_inches='tight')
                writer.grab_frame()

        print(f"MP4 saved as: {mp4_path}")

    else:
        config_indices = config.dataset.split('fly_N9_')[1] if 'fly_N9_' in config.dataset else 'evolution'
        files, file_id_list = get_training_files(log_dir, n_runs)

        if True:
            fps = 10
            metadata = dict(title='Model evolution', artist='Matplotlib', comment='Model evolution over epochs')
            writer = FFMpegWriter(fps=fps, metadata=metadata)
            fig = plt.figure(figsize=(16, 8))
            mp4_path = f'{log_dir}/results/MLP_weights_{config_indices}.mp4'
            if os.path.exists(mp4_path):
                os.remove(mp4_path)
            with writer.saving(fig, mp4_path, dpi=80):
                for file_id_ in trange(len(file_id_list)):
                    epoch = files[file_id_].split('graphs')[1][1:-3]

                    net = f'{log_dir}/models/best_model_with_{n_runs - 1}_graphs_{epoch}.pt'
                    model = Signal_Propagation_FlyVis(
                        aggr_type=model_config.aggr_type,
                        config=config,
                        device=device
                    )
                    state_dict = torch.load(net, map_location=device)
                    model.load_state_dict(state_dict['model_state_dict'])
                    logger.info(f'net: {net}')

                    # Extract layer-wise parameters from lin_phi
                    layer_weights, layer_biases = [], []
                    layer_names = []

                    for name, param in model.lin_phi.named_parameters():
                        if "weight" in name:
                            layer_weights.append(param.detach().cpu())
                            layer_names.append(name)
                        elif "bias" in name:
                            layer_biases.append(param.detach().cpu())

                    n_layers = len(layer_weights)

                    fig.clf()
                    fig.set_size_inches(4 * n_layers, 8)
                    axes = fig.subplots(2, n_layers)
                    if n_layers == 1:
                        axes = axes.reshape(-1, 1)

                    for layer_idx in range(n_layers):
                        w = layer_weights[layer_idx]
                        b = layer_biases[layer_idx]

                        # Plot weights
                        ax_w = axes[0, layer_idx]
                        im_w = ax_w.imshow(w, cmap="seismic", aspect="equal", vmin=-10,vmax=10)
                        # cbar_w = fig.colorbar(im_w, ax=ax_w, fraction=0.046, pad=0.04)
                        # cbar_w.ax.tick_params(labelsize=12)
                        ax_w.set_title(f"Layer {layer_idx + 1} Weights ({w.shape[0]}x{w.shape[1]})", fontsize=14)
                        ax_w.axis('off')

                        # Plot biases as row vector
                        ax_b = axes[1, layer_idx]
                        b_2d = b.unsqueeze(0) if b.dim() == 1 else b
                        im_b = ax_b.imshow(b_2d, cmap="seismic", aspect="equal", vmin=-1,vmax=1)
                        # cbar_b = fig.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
                        # cbar_b.ax.tick_params(labelsize=12)
                        ax_b.set_title(f"Layer {layer_idx + 1} Biases ({len(b)})", fontsize=14)
                        ax_b.axis('off')

                    plt.tight_layout()

                    if (file_id_ == len(file_id_list) - 1) | (file_id_ == 0):
                        plt.savefig(f'{log_dir}/results/MLP_weights_{config_indices}.png',
                                    dpi=300, bbox_inches='tight')

                    writer.grab_frame()
            print(f"MP4 saved as: {mp4_path}")

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
            fig = plt.figure(figsize=(8, 8))
            for n in range(n_types):
                pos = torch.argwhere(type_list == n)
                plt.scatter(to_numpy(model.a[pos, 0]), to_numpy(model.a[pos, 1]), s=2, color=colors_65[n], alpha=0.8,
                            edgecolors='none')
            plt.xlabel('embedding 0', fontsize=24)
            plt.ylabel('embedding 1', fontsize=24)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(f'{log_dir}/results/embedding_{config_indices}.png', dpi=300)
            plt.close()

            if True:
                print('embedding clustering results')
                for eps in [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.05]:
                    results = clustering_evaluation(to_numpy(model.a), type_list, eps=eps)
                    print(f"eps={eps}: {results['n_clusters_found']} clusters, "
                          f"accuracy={results['accuracy']:.3f}")
                    logger.info(f"eps={eps}: {results['n_clusters_found']} clusters, "f"accuracy={results['accuracy']:.3f}")
                time.sleep(1)

            # Plot 3: Edge function visualization
            slopes_lin_edge_list = []
            fig = plt.figure(figsize=(8, 8))
            for n in trange(n_neurons):
                if (n % 20 == 0):
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
                             linewidth=1, alpha=0.05)

                rr = torch.linspace(mu_activity[n] - 2 * sigma_activity[n], mu_activity[n] + 2 * sigma_activity[n], 1000, device=device)
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
                         linewidth=1, alpha=0.2)

                rr_numpy = to_numpy(rr[rr.shape[0]//2+1:])
                func_numpy = to_numpy(func[rr.shape[0]//2+1:].squeeze())
                try:
                    lin_fit, _ = curve_fit(linear_model, rr_numpy, func_numpy)
                    slope = lin_fit[0]
                    offset = lin_fit[1]
                except:
                    coeffs = np.polyfit(rr_numpy, func_numpy, 1)
                    slope = coeffs[0]
                    offset = coeffs[1]
                slopes_lin_edge_list.append(slope)
            plt.xlabel('$x_i$', fontsize=24)
            plt.ylabel('$MLP_1(a_i, x_i)$', fontsize=24)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.xlim(config.plotting.xlim)
            plt.ylim([-config.plotting.xlim[1]/10, config.plotting.xlim[1]*2])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/edge_functions_{config_indices}.png", dpi=300)
            plt.close()

            # Plot 5: Phi function visualization

            func_list = []
            slopes_lin_phi_list = []
            offsets_list = []
            fig = plt.figure(figsize=(8, 8))
            for n in trange(n_neurons):
                if (n % 20 == 0):
                    rr = torch.linspace(config.plotting.xlim[0], config.plotting.xlim[1], 1000, device=device)
                    embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                    in_features = torch.cat((rr[:, None], embedding_, rr[:, None] * 0, torch.zeros_like(rr[:, None])), dim=1)
                    with torch.no_grad():
                        func = model.lin_phi(in_features.float())
                        plt.plot(to_numpy(rr), to_numpy(func), 2,
                                 color=cmap.color(to_numpy(type_list)[n].astype(int)),
                                 linewidth=1, alpha=0.025)
                rr = torch.linspace(mu_activity[n] - 2 * sigma_activity[n], mu_activity[n] + 2 * sigma_activity[n], 1000, device=device)
                embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                in_features = torch.cat((rr[:, None], embedding_, rr[:, None] * 0, torch.zeros_like(rr[:, None])), dim=1)
                with torch.no_grad():
                    func = model.lin_phi(in_features.float())
                if (n % 20 == 0):
                    plt.plot(to_numpy(rr), to_numpy(func), 2,
                             color=cmap.color(to_numpy(type_list)[n].astype(int)),
                             linewidth=1, alpha=0.2)
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
            plt.xlabel('$x_i$', fontsize=18)
            plt.ylabel('$MLP_0(a_i, x_i)$', fontsize=18)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/phi_functions_{config_indices}.png", dpi=300)
            plt.close()

            slopes_lin_phi_array = np.array(slopes_lin_phi_list)
            offsets_array = np.array(offsets_list)
            gt_taus = to_numpy(gt_taus[:n_neurons])
            reconstructed_tau = np.where(slopes_lin_phi_array != 0, 1.0 / -slopes_lin_phi_array, 1)
            reconstructed_tau = reconstructed_tau[:n_neurons]
            reconstructed_tau = np.clip(reconstructed_tau, 0, 1)

            fig = plt.figure(figsize=(8, 8))
            plt.scatter(gt_taus, reconstructed_tau, c=mc, s=0.5, alpha=0.3)
            lin_fit, lin_fitv = curve_fit(linear_model, gt_taus, reconstructed_tau)
            residuals = reconstructed_tau - linear_model(gt_taus, *lin_fit)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((reconstructed_tau - np.mean(reconstructed_tau)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            plt.text(0.05, 0.95, f'R²: {r_squared:.3f}\nslope: {lin_fit[0]:.2f}\nN: {len(gt_taus)}',
                     transform=plt.gca().transAxes, verticalalignment='top', fontsize=16)
            plt.xlabel(r'true $\tau$', fontsize=24)
            plt.ylabel(r'learned $\tau$', fontsize=24)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.xlim([0, 0.35])
            # plt.ylim([0, 0.35])
            plt.tight_layout()
            plt.savefig(f'{log_dir}/results/tau_comparison_{config_indices}.png', dpi=300)
            plt.close()

            print(" ")
            print(f"tau reconstruction R²: {r_squared:.4f}  slope: {np.round(lin_fit[0], 4)}")
            logger.info(f"tau reconstruction R²: {r_squared:.4f}  slope: {np.round(lin_fit[0], 4)}")
            torch.save(torch.tensor(reconstructed_tau, dtype=torch.float32, device=device), f'{log_dir}/results/tau.pt')

            # V_rest comparison (reconstructed vs ground truth)
            reconstructed_V_rest = np.where(slopes_lin_phi_array != 0, -offsets_array / slopes_lin_phi_array, 1)
            reconstructed_V_rest = reconstructed_V_rest[:n_neurons]
            gt_V_rest = to_numpy(gt_V_Rest[:n_neurons])
            fig = plt.figure(figsize=(8, 8))
            plt.scatter(gt_V_rest, reconstructed_V_rest, c=mc, s=0.5, alpha=0.3)
            lin_fit, lin_fitv = curve_fit(linear_model, gt_V_rest, reconstructed_V_rest)
            residuals = reconstructed_V_rest - linear_model(gt_V_rest, *lin_fit)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((reconstructed_V_rest - np.mean(reconstructed_V_rest)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            plt.text(0.05, 0.95, f'R²: {r_squared:.3f}\nslope: {lin_fit[0]:.2f}\nN: {len(gt_V_rest)}',
                     transform=plt.gca().transAxes, verticalalignment='top', fontsize=16)
            plt.xlabel(r'true $V_{rest}$', fontsize=24)
            plt.ylabel(r'learned $V_{rest}$', fontsize=24)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.tight_layout()
            plt.savefig(f'{log_dir}/results/V_rest_comparison_{config_indices}.png', dpi=300)
            plt.close()

            print(f"V_rest reconstruction R²: {r_squared:.4f}  slope: {np.round(lin_fit[0], 4)}")
            logger.info(f"V_rest reconstruction R²: {r_squared:.4f}  slope: {np.round(lin_fit[0], 4)}")

            torch.save(torch.tensor(reconstructed_V_rest, dtype=torch.float32, device=device), f'{log_dir}/results/V_rest.pt')

            if False:
                print('linear fit clustering results')
                for eps in [0.005, 0.0075, 0.01, 0.02, 0.05]:
                    results = clustering_evaluation(to_numpy(model.a), type_list, eps=eps)
                    print(f"eps={eps}: {results['n_clusters_found']} clusters, "
                          f"accuracy={results['accuracy']:.3f}")
                    logger.info(f"eps={eps}: {results['n_clusters_found']} clusters, "f"accuracy={results['accuracy']:.3f}")
            if False:
                print('functionnal clustering results')
                logger.info('functionnal results')
                func_list = torch.stack(func_list).squeeze()
                for eps in [0.05, 0.075, 0.1, 0.2, 0.3]:
                    functional_results = functional_clustering_evaluation(func_list, type_list,
                                                                          eps=eps)  # Current functional result
                    print(
                        f"eps={eps}: {functional_results['n_clusters_found']} clusters, {functional_results['accuracy']:.3f} accuracy")
                    logger.info(
                        f"eps={eps}: {functional_results['n_clusters_found']} clusters, {functional_results['accuracy']:.3f} accuracy")

            # Plot 4: Weight comparison using model.W and gt_weights
            fig = plt.figure(figsize=(8, 8))
            learned_weights = to_numpy(model.W.squeeze())
            true_weights = to_numpy(gt_weights)
            plt.scatter(true_weights, learned_weights, c=mc, s=0.1, alpha=0.1)
            lin_fit, lin_fitv = curve_fit(linear_model, true_weights, learned_weights)
            residuals = learned_weights - linear_model(true_weights, *lin_fit)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((learned_weights - np.mean(learned_weights)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            plt.text(0.05, 0.95, f'R²: {r_squared:.3f}\nslope: {lin_fit[0]:.2f}\nN: {len(true_weights)}',
                     transform=plt.gca().transAxes, verticalalignment='top', fontsize=16)
            plt.xlabel('true W_ij')
            plt.ylabel('learned W_ij')
            plt.tight_layout()
            plt.savefig(f'{log_dir}/results/comparison_{epoch}.png', dpi=300)
            plt.close()
            print(f"first weights fit R²: {r_squared:.4f}  slope: {np.round(lin_fit[0], 4)}")
            logger.info(f"first weights fit R²: {r_squared:.4f}  slope: {np.round(lin_fit[0], 4)}")

            logger.info('weights comparison per type')
            # Plot 4bis: Weight comparison using model.W and gt_weights
            fig = plt.figure(figsize=(8, 8))
            type_edge_list = x[to_numpy(edges[1, :]), 6]
            for n in range(n_types):
                pos_neurons = torch.argwhere(type_list == n)
                pos_neurons = pos_neurons.squeeze()
                pos = np.argwhere(type_edge_list == n)
                pos = pos.astype(int).squeeze()
                plt.scatter(true_weights[pos], learned_weights[pos], c=colors_65[n], s=0.1, alpha=0.01)
                lin_fit, lin_fitv = curve_fit(linear_model, true_weights[pos], learned_weights[pos])
                residuals = learned_weights[pos] - linear_model(true_weights[pos], *lin_fit)
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((learned_weights[pos] - np.mean(learned_weights[pos])) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                true_weights_mean = np.mean(true_weights[pos])
                # print(f"{index_to_name[n]} R²: {r_squared:.4f}  slope: {np.round(lin_fit[0], 4)}  edges: {len(pos)}  weights mean: {true_weights_mean:.4f}")
                logger.info(
                    f"{index_to_name[n]} R²: {r_squared:.4f}  slope: {np.round(lin_fit[0], 4)}  edges: {len(pos)}  weights mean: {true_weights_mean:.4f}")
            plt.xlabel('true W_ij', fontsize=24)
            plt.ylabel('learned W_ij', fontsize=24)
            plt.tight_layout()
            # plt.savefig(f'{log_dir}/results/comparison_color_{epoch}.png', dpi=300)
            plt.close()

            k_list = [1]  # , 2, 3, 4, 5, 6, 7, 8]
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
            grad_values = to_numpy(grad_msg[0:n_neurons]).squeeze()  # Flatten to 1D
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
            # --- Global fit after outlier removal ---
            fig = plt.figure(figsize=(8, 8))
            plt.scatter(true_in, learned_in, c=mc, s=0.1, alpha=0.1)
            lin_fit, _ = curve_fit(linear_model, true_in, learned_in)
            residuals = learned_in - linear_model(true_in, *lin_fit)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((learned_in - np.mean(learned_in)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            plt.text(0.05, 0.95,
                     f'R²: {r_squared:.3f}\nslope: {lin_fit[0]:.2f}\nN: {len(true_in)}',
                     transform=plt.gca().transAxes, verticalalignment='top', fontsize=16)
            plt.xlabel('true $W_{ij}$', fontsize=24)
            plt.ylabel('learned $W_{ij}$', fontsize=24)
            plt.tight_layout()
            plt.savefig(f'{log_dir}/results/corrected_comparison_{epoch}.png', dpi=300)
            plt.close()

            print(f"second weights fit R²: {r_squared:.4f}  slope: {np.round(lin_fit[0], 4)}")
            logger.info(f"second weights fit R²: {r_squared:.4f}  slope: {np.round(lin_fit[0], 4)}")
            print(f'median residuals: {np.median(residuals):.4f}')
            inlier_residuals = residuals[mask]
            print(f'inliers: {len(inlier_residuals)}  mean residual: {np.mean(inlier_residuals):.4f}  std: {np.std(inlier_residuals):.4f}  min,max: {np.min(inlier_residuals):.4f}, {np.max(inlier_residuals):.4f}')
            outlier_residuals = residuals[~mask]
            print(f'outliers: {len(outlier_residuals)}  mean residual: {np.mean(outlier_residuals):.4f}  std: {np.std(outlier_residuals):.4f}  min,max: {np.min(outlier_residuals):.4f}, {np.max(outlier_residuals):.4f}')


            # plot distribution

            if 'visual' in field_type:
                n_frames = config.simulation.n_frames
                k = 100
                reconstructed_field = to_numpy(
                    model.visual_NNR(torch.tensor([k / n_frames], dtype=torch.float32, device=device)) ** 2)
                gt_field = x_list[0][k, :n_input_neurons, 4:5]

                # Setup for saving MP4
                fps = 10  # frames per second for the video
                metadata = dict(title='Field Evolution', artist='Matplotlib', comment='NN Reconstruction over time')
                writer = FFMpegWriter(fps=fps, metadata=metadata)
                fig = plt.figure(figsize=(12, 4))

                # Start the writer context
                if os.path.exists(f'{log_dir}/results/field_movie_{config_indices}.mp4'):
                    os.remove(f'{log_dir}/results/field_movie_{config_indices}.mp4')
                r_squared_list = []
                slope_list = []
                with writer.saving(fig, f'{log_dir}/results/field_movie_{epoch}.mp4', dpi=80):
                    for k in range(0, 4000, 10):
                        # Inference and data extraction
                        reconstructed_field = to_numpy(
                            model.visual_NNR(torch.tensor([k / n_frames], dtype=torch.float32, device=device)) ** 2)
                        gt_field = x_list[0][k, :n_input_neurons, 4:5]
                        X1 = x_list[0][k, :n_input_neurons, 1:3]
                        vmin = reconstructed_field.min()
                        vmax = reconstructed_field.max()
                        fig.clf()  # Clear the figure
                        # Ground truth
                        ax1 = fig.add_subplot(1, 3, 1)
                        sc1 = ax1.scatter(X1[:, 0], X1[:, 1], s=256, c=gt_field, cmap="viridis", marker='h', vmin=-2, vmax=2)
                        ax1.set_xticks([])
                        ax1.set_yticks([])
                        ax1.set_title("ground Truth", fontsize=24)
                        # Reconstructed
                        ax2 = fig.add_subplot(1, 3, 2)
                        sc2 = ax2.scatter(X1[:, 0], X1[:, 1], s=256, c=reconstructed_field, cmap="viridis", marker='h', vmin=vmin, vmax=vmax)
                        ax2.set_xticks([])
                        ax2.set_yticks([])
                        ax2.set_title("reconstructed", fontsize=24)

                        ax3 = fig.add_subplot(1, 3, 3)
                        sc3 = ax3.scatter(gt_field, reconstructed_field, s=1, c=mc)
                        x_data = gt_field.squeeze()
                        y_data = reconstructed_field
                        lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                        residuals = y_data - linear_model(x_data, *lin_fit)
                        ss_res = np.sum(residuals ** 2)
                        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot)
                        r_squared_list.append(r_squared)
                        slope_list.append(lin_fit[0])
                        ax3.text(0.05, 0.95,
                                 f'R²: {r_squared:.3f}\nslope: {lin_fit[0]:.2f}',
                                 transform=ax3.transAxes,
                                 verticalalignment='top',
                                 fontsize=12)
                        ax3.set_xlim([0,1])
                        ax3.set_ylim([0,1])
                        plt.tight_layout()
                        writer.grab_frame()

                print(f"visual field R²: {np.mean(r_squared_list):.3f}  std: {np.std(r_squared_list):.3f}  slope: {np.mean(slope_list):.3f}")
                logger.info(f"visual field R²: {np.mean(r_squared_list):.3f}  std: {np.std(r_squared_list):.3f}  slope: {np.mean(slope_list):.3f}")
            print(" ")


def data_flyvis_compare(config_list, varied_parameter):
    """
    Compare flyvis experiments by reading config files and results.log files
    ONLY CHANGE: Added energy distribution panel (ax6)
    """
    import yaml
    import re
    import os
    from collections import defaultdict
    import numpy as np
    import matplotlib.pyplot as plt
    # Disable LaTeX usage completely
    plt.rcParams['text.usetex'] = False
    # Use default matplotlib fonts instead of LaTeX fonts
    plt.rc('font', family='sans-serif')
    from NeuralGraph.config import NeuralGraphConfig
    from NeuralGraph.models.utils import add_pre_folder

    plt.style.use('ggplot')

    plt.rcParams['axes.facecolor'] = 'black'
    plt.rcParams['figure.facecolor'] = 'black'
    plt.rcParams['axes.edgecolor'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    plt.rcParams['grid.color'] = 'gray'
    plt.rcParams['text.color'] = 'white'

    results = []

    for config_file_ in config_list:
        try:
            # Load config to get parameter value
            config_file, pre_folder = add_pre_folder(config_file_)
            config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')

            # Get parameter value from config using section.parameter format OR config file indices
            if varied_parameter is None:
                parts = config_file_.split('_')
                if len(parts) >= 2:
                    param_value = f"{parts[-2]}_{parts[-1]}"
                else:
                    print(f"warning: cannot extract indices from config name '{config_file_}'")
                    continue
            else:
                if '.' not in varied_parameter:
                    raise ValueError(f"parameter must be in 'section.parameter' format, got: {varied_parameter}")

                section_name, param_name = varied_parameter.split('.', 1)
                section = getattr(config, section_name, None)
                if section is None:
                    raise ValueError(f"config section '{section_name}' not found")

                param_value = getattr(section, param_name, None)
                if param_value is None:
                    print(f"warning: parameter '{param_name}' not found in section '{section_name}' for {config_file_}")
                    continue

            # Get results log path
            results_log_path = os.path.join('./log', config_file, 'results.log')

            if not os.path.exists(results_log_path):
                print(f"warning: {results_log_path} not found")
                continue

            # Parse results.log
            with open(results_log_path, 'r') as f:
                content = f.read()

            # Extract R² from second weights fit
            r2_match = re.search(r'second weights fit\s+R²:\s*([\d.-]+)', content)
            r2 = float(r2_match.group(1)) if r2_match else None

            # Extract tau reconstruction R²
            tau_r2_match = re.search(r'tau reconstruction R²:\s*([\d.-]+)', content)
            tau_r2 = float(tau_r2_match.group(1)) if tau_r2_match else None

            # Extract V_rest reconstruction R²
            vrest_r2_match = re.search(r'V_rest reconstruction R²:\s*([\d.-]+)', content)
            vrest_r2 = float(vrest_r2_match.group(1)) if vrest_r2_match else None

            # Extract best clustering accuracy and corresponding eps
            clustering_results = []
            for line in content.split('\n'):
                if 'eps=' in line and 'accuracy' in line:
                    eps_match = re.search(r'eps=([\d.-]+)', line)
                    acc_match = re.search(r'accuracy[=:]\s*([\d.-]+)', line)
                    if eps_match and acc_match:
                        clustering_results.append({
                            'eps': float(eps_match.group(1)),
                            'accuracy': float(acc_match.group(1))
                        })

            if clustering_results:
                best_clustering_result = max(clustering_results, key=lambda x: x['accuracy'])
                best_clustering_acc = best_clustering_result['accuracy']
                best_eps = best_clustering_result['eps']
            else:
                best_clustering_acc = None
                best_eps = None

            # Check for video files and get their sizes
            video_dir = f'./graphs_data/fly/{config.dataset}'
            config_indices = config.dataset.split('fly_N9_')[1] if 'fly_N9_' in config.dataset else 'no_id'
            ffv1_path = os.path.join(video_dir, f'input_{config_indices}_ffv1.mkv')
            libx264_path = os.path.join(video_dir, f'input_{config_indices}_libx264.mkv')

            ffv1_size_mb = None
            libx264_size_mb = None
            compression_ratio = None

            if os.path.exists(ffv1_path):
                ffv1_size_mb = os.path.getsize(ffv1_path) / (1024 * 1024)  # Convert to MB

            if os.path.exists(libx264_path):
                libx264_size_mb = os.path.getsize(libx264_path) / (1024 * 1024)  # Convert to MB

            # Calculate compression ratio (ffv1 / libx264) if both files exist
            if ffv1_size_mb is not None and libx264_size_mb is not None and libx264_size_mb > 0:
                compression_ratio = ffv1_size_mb / libx264_size_mb

            results.append({
                'config': config_file_,
                'param_value': param_value,
                'r2': r2,
                'tau_r2': tau_r2,
                'vrest_r2': vrest_r2,
                'best_clustering_acc': best_clustering_acc,
                'best_eps': best_eps,
                'ffv1_size_mb': ffv1_size_mb,
                'libx264_size_mb': libx264_size_mb,
                'compression_ratio': compression_ratio
            })

        except Exception as e:
            print(f"error processing {config_file_}: {e}")

    # Group results by parameter value
    grouped_results = defaultdict(list)
    for r in results:
        # Only include results with all key metrics available
        if (r['r2'] is not None and r['tau_r2'] is not None and
                r['vrest_r2'] is not None and r['best_clustering_acc'] is not None and
                r['best_eps'] is not None):
            grouped_results[r['param_value']].append(r)

    # Calculate statistics for each parameter value
    summary_results = []
    for param_val, group in grouped_results.items():
        r2_values = [r['r2'] for r in group]
        tau_r2_values = [r['tau_r2'] for r in group]
        vrest_r2_values = [r['vrest_r2'] for r in group]
        acc_values = [r['best_clustering_acc'] for r in group]
        eps_values = [r['best_eps'] for r in group]

        # Video file size handling
        ffv1_sizes = [r['ffv1_size_mb'] for r in group if r['ffv1_size_mb'] is not None]
        libx264_sizes = [r['libx264_size_mb'] for r in group if r['libx264_size_mb'] is not None]
        compression_ratios = [r['compression_ratio'] for r in group if r['compression_ratio'] is not None]

        configs = [r['config'] for r in group]
        n_configs = len(group)

        # Handle eps display
        unique_eps = list(set(eps_values))
        if len(unique_eps) == 1:
            eps_str = f"{unique_eps[0]}"
        else:
            eps_str = f"{min(unique_eps)}-{max(unique_eps)}"

        if n_configs == 1:
            r2_str = f"{r2_values[0]:.4f}"
            tau_r2_str = f"{tau_r2_values[0]:.4f}"
            vrest_r2_str = f"{vrest_r2_values[0]:.4f}"
            acc_str = f"{acc_values[0]:.3f}"
        else:
            r2_mean, r2_std = np.mean(r2_values), np.std(r2_values)
            tau_r2_mean, tau_r2_std = np.mean(tau_r2_values), np.std(tau_r2_values)
            vrest_r2_mean, vrest_r2_std = np.mean(vrest_r2_values), np.std(vrest_r2_values)
            acc_mean, acc_std = np.mean(acc_values), np.std(acc_values)
            r2_str = f"{r2_mean:.4f}±{r2_std:.4f}"
            tau_r2_str = f"{tau_r2_mean:.4f}±{tau_r2_std:.4f}"
            vrest_r2_str = f"{vrest_r2_mean:.4f}±{vrest_r2_std:.4f}"
            acc_str = f"{acc_mean:.3f}±{acc_std:.3f}"

        summary_results.append({
            'param_value': param_val,
            'r2_values': r2_values,
            'tau_r2_values': tau_r2_values,
            'vrest_r2_values': vrest_r2_values,
            'acc_values': acc_values,
            'ffv1_sizes': ffv1_sizes,
            'libx264_sizes': libx264_sizes,
            'compression_ratios': compression_ratios,
            'r2_mean': np.mean(r2_values),
            'tau_r2_mean': np.mean(tau_r2_values),
            'vrest_r2_mean': np.mean(vrest_r2_values),
            'acc_mean': np.mean(acc_values),
            'ffv1_mean': np.mean(ffv1_sizes) if ffv1_sizes else None,
            'libx264_mean': np.mean(libx264_sizes) if libx264_sizes else None,
            'compression_mean': np.mean(compression_ratios) if compression_ratios else None,
            'r2_str': r2_str,
            'tau_r2_str': tau_r2_str,
            'vrest_r2_str': vrest_r2_str,
            'acc_str': acc_str,
            'eps_str': eps_str,
            'n_configs': n_configs,
            'configs': configs
        })

    # Sort by parameter value
    if varied_parameter is None:
        try:
            summary_results.sort(key=lambda x: [int(part) for part in x['param_value'].split('_')])
        except ValueError:
            summary_results.sort(key=lambda x: x['param_value'])
    else:
        summary_results.sort(key=lambda x: x['param_value'])

    # Print comparison summary
    if varied_parameter is None:
        param_display_name = "config_indices"
        parameter_name = "config file indices"
    else:
        full_param_name = varied_parameter.split('.')[1]
        parts = full_param_name.split('_')
        param_display_name = '_'.join(parts[:2]) if len(parts) >= 2 else parts[0]
        parameter_name = varied_parameter

    print(f"\n=== parameter comparison: {parameter_name} ===")
    print(
        f"{param_display_name:<15} {'weights R²':<15} {'tau R²':<15} {'V_rest R²':<15} {'clustering acc':<18} {'best eps':<10} {'n_configs':<10}")
    print("-" * 100)

    for r in summary_results:
        print(
            f"{r['param_value']:<15} {r['r2_str']:<15} {r['tau_r2_str']:<15} {r['vrest_r2_str']:<15} {r['acc_str']:<18} {r['eps_str']:<10} {r['n_configs']:<10}")
    print("-" * 100)
    if summary_results:
        best_r2_result = max(summary_results, key=lambda x: x['r2_mean'])
        best_tau_r2_result = max(summary_results, key=lambda x: x['tau_r2_mean'])
        best_vrest_r2_result = max(summary_results, key=lambda x: x['vrest_r2_mean'])
        best_acc_result = max(summary_results, key=lambda x: x['acc_mean'])

        print(f"\n🏆 best mean weights R²: {best_r2_result['r2_str']}")
        print(f"   {param_display_name}: {best_r2_result['param_value']}, n={best_r2_result['n_configs']}")

        print(f"\n⚡ best mean tau R²: {best_tau_r2_result['tau_r2_str']}")
        print(f"   {param_display_name}: {best_tau_r2_result['param_value']}, n={best_tau_r2_result['n_configs']}")

        print(f"\n🔋 best mean V_rest R²: {best_vrest_r2_result['vrest_r2_str']}")
        print(f"   {param_display_name}: {best_vrest_r2_result['param_value']}, n={best_vrest_r2_result['n_configs']}")

        print(f"\n🎯 best mean clustering accuracy: {best_acc_result['acc_str']}")
        print(f"   {param_display_name}: {best_acc_result['param_value']}, n={best_acc_result['n_configs']}")
    print("\n" + "=" * 100)

    # Prepare data for plotting
    param_values_str = [str(r['param_value']) for r in summary_results]
    r2_means = [r['r2_mean'] for r in summary_results]
    tau_r2_means = [r['tau_r2_mean'] for r in summary_results]
    vrest_r2_means = [r['vrest_r2_mean'] for r in summary_results]
    acc_means = [r['acc_mean'] for r in summary_results]

    # Calculate error bars (std for n>1, None for n=1)
    r2_errors = []
    tau_r2_errors = []
    vrest_r2_errors = []
    acc_errors = []
    for r in summary_results:
        if r['n_configs'] > 1:
            r2_errors.append(np.std(r['r2_values']))
            tau_r2_errors.append(np.std(r['tau_r2_values']))
            vrest_r2_errors.append(np.std(r['vrest_r2_values']))
            acc_errors.append(np.std(r['acc_values']))
        else:
            r2_errors.append(0)
            tau_r2_errors.append(0)
            vrest_r2_errors.append(0)
            acc_errors.append(0)

    # Create figure with six panels (2x3 layout) - ONLY CHANGED TO ADD ENERGY PANEL
    fig, ((ax1, ax4, ax6), (ax2, ax3, ax5)) = plt.subplots(2, 3, figsize=(20, 12))
    plt.subplots_adjust(hspace=3.0)

    # Weights R² panel
    ax1.errorbar(param_values_str, r2_means, yerr=r2_errors,
                 fmt='o', capsize=5, capthick=2, markersize=8, linewidth=2, color='lightblue')
    ax1.set_xlabel(param_display_name, fontsize=16, color='white')
    ax1.set_ylabel('weights R²', fontsize=24, color='white')
    ax1.set_ylim(0, 1.1)
    ax1.tick_params(colors='white', labelsize=14)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    for i, (x, y, n) in enumerate(zip(param_values_str, r2_means, [r['n_configs'] for r in summary_results])):
        if n > 1:
            ax1.text(x, y + r2_errors[i] + 0.02, f'n={n}', ha='center', va='bottom', fontsize=12, color='white')

    # Tau R² panel
    ax2.errorbar(param_values_str, tau_r2_means, yerr=tau_r2_errors,
                 fmt='s', capsize=5, capthick=2, markersize=8, linewidth=2, color='lightgreen')
    ax2.set_xlabel(param_display_name, fontsize=16, color='white')
    ax2.set_ylabel('tau R²', fontsize=24, color='white')
    ax2.set_ylim(0, 1.1)
    ax2.tick_params(colors='white', labelsize=14)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    for i, (x, y, n) in enumerate(zip(param_values_str, tau_r2_means, [r['n_configs'] for r in summary_results])):
        if n > 1:
            ax2.text(x, y + tau_r2_errors[i] + 0.02, f'n={n}', ha='center', va='bottom', fontsize=12, color='white')

    # V_rest R² panel
    ax3.errorbar(param_values_str, vrest_r2_means, yerr=vrest_r2_errors,
                 fmt='^', capsize=5, capthick=2, markersize=8, linewidth=2, color='lightcoral')
    ax3.set_xlabel(param_display_name, fontsize=16, color='white')
    ax3.set_ylabel('V_rest R²', fontsize=24, color='white')
    ax3.set_ylim(0, 1.1)
    ax3.tick_params(colors='white', labelsize=14)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    for i, (x, y, n) in enumerate(zip(param_values_str, vrest_r2_means, [r['n_configs'] for r in summary_results])):
        if n > 1:
            ax3.text(x, y + vrest_r2_errors[i] + 0.02, f'n={n}', ha='center', va='bottom', fontsize=12, color='white')

    # Clustering accuracy panel (convert to percentage)
    acc_means_pct = [acc * 100 for acc in acc_means]
    acc_errors_pct = [err * 100 for err in acc_errors]

    ax4.errorbar(param_values_str, acc_means_pct, yerr=acc_errors_pct,
                 fmt='D', capsize=5, capthick=2, markersize=8, linewidth=2, color='orange')
    ax4.set_xlabel(param_display_name, fontsize=16, color='white')
    ax4.set_ylabel('clustering accuracy (%)', fontsize=24, color='white')
    ax4.set_ylim(0, 100)
    ax4.tick_params(colors='white', labelsize=14)
    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')

    # Loss curves panel (ax5) - RESTORED ORIGINAL VERSION
    ax5.set_xlabel('epochs', fontsize=16, color='white')
    ax5.set_ylabel('loss', fontsize=24, color='white')
    ax5.tick_params(colors='white', labelsize=14)
    ax5.set_ylim(0, 4000)
    ax5.grid(True, alpha=0.3)

    # Generate distinct colors for different configs
    import matplotlib.cm as cm
    colors = cm.tab10(np.linspace(0, 1, len(config_list)))

    legend_info = []

    for i, config_file_ in enumerate(config_list):
        try:
            config_file, pre_folder = add_pre_folder(config_file_)
            loss_path = os.path.join('./log/fly', config_file, 'loss.pt')

            if os.path.exists(loss_path):
                import torch
                loss_values = torch.load(loss_path, map_location='cpu')
                loss_values = np.array(loss_values)

                epochs = np.arange(len(loss_values))
                ax5.plot(epochs, loss_values, color=colors[i], linewidth=2, alpha=0.8)

                # Get parameter value for this config
                if varied_parameter is None:
                    parts = config_file_.split('_')
                    if len(parts) >= 2:
                        param_val = f"{parts[-2]}_{parts[-1]}"
                    else:
                        param_val = config_file_
                else:
                    try:
                        config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
                        section_name, param_name = varied_parameter.split('.', 1)
                        section = getattr(config, section_name, None)
                        if section is not None:
                            param_val = getattr(section, param_name, config_file_)
                        else:
                            param_val = config_file_
                    except:
                        param_val = config_file_

                legend_info.append(f"{param_val}")

        except Exception as e:
            print(f"Warning: Could not load loss for {config_file_}: {e}")

    # Add legend with parameter values and colors
    if legend_info:
        ax5.legend(legend_info, fontsize=10, loc='upper right')
        legend = ax5.get_legend()
        legend.get_frame().set_facecolor('black')
        legend.get_frame().set_alpha(0.8)
        for text in legend.get_texts():
            text.set_color('white')

    # NEW: Energy Distributions panel (ax6)
    colors = plt.cm.viridis(np.linspace(0, 1, len(summary_results)))
    alpha = 0.7

    energy_plotted = False
    for i, result in enumerate(summary_results):
        # Load energy data for this parameter value
        all_energies = []
        for config_name in result['configs']:
            energy_path = os.path.join('./log/fly/', config_name.replace('config_', '').replace('.yaml', ''), 'results',
                                       'E.npy')
            if os.path.exists(energy_path):
                try:
                    energy_data = np.load(energy_path)
                    all_energies.extend(energy_data)
                except Exception as e:
                    print(f"Warning: Could not load {energy_path}: {e}")

        if all_energies:
            energy_plotted = True
            ax6.hist(all_energies, bins=50, alpha=alpha, color=colors[i],
                     density=True, label=f"{param_display_name}={result['param_value']}")

    if energy_plotted:
        ax6.set_xlabel('Ising Energy', fontsize=16, color='white')
        ax6.set_ylabel('Density', fontsize=24, color='white')
        ax6.set_title('Energy Distributions', fontsize=16, color='white')
        ax6.tick_params(colors='white', labelsize=14)
        ax6.legend(fontsize=10, facecolor='black', edgecolor='white')
        legend = ax6.get_legend()
        for text in legend.get_texts():
            text.set_color('white')
    else:
        ax6.text(0.5, 0.5, 'No Energy Data\nAvailable',
                 transform=ax6.transAxes, ha='center', va='center',
                 fontsize=16, color='white', alpha=0.7)
        ax6.set_xlabel('Energy', fontsize=16, color='white')
        ax6.set_ylabel('Density', fontsize=24, color='white')
        ax6.tick_params(colors='white', labelsize=14)

    plt.subplots_adjust(hspace=0.6)
    plt.tight_layout()

    plot_filename = f'parameter_comparison_{param_display_name}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nplot saved as: {plot_filename}")
    plt.close()

    # Create separate energy distribution comparison plot (2x2)
    create_Ising_distribution_comparison(summary_results, param_display_name)


def create_Ising_distribution_comparison(summary_results, param_display_name):
    """
    Create a 2x2 detailed comparison of Ising energies and couplings across parameter values.
    Panels:
    -----------
    Top-left: linear energy histogram
    Top-right: J couplings histogram
    Bottom-left: CDF of energy
    Bottom-right: violin plot of energy distributions
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    all_energy_data = {}
    all_J_data = {}
    n_params = len(summary_results)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_params))

    # Load all energy and J data
    for i, result in enumerate(summary_results):
        energies = []
        J_vals = []
        for config_name in result['configs']:
            config_base = config_name.replace('config_', '').replace('.yaml', '')
            energy_path = os.path.join('./log/fly', config_base, 'results', 'E.npy')
            J_path = os.path.join('./log/fly', config_base, 'results', 'J.npy')

            if os.path.exists(energy_path):
                try:
                    energies.extend(np.load(energy_path))
                except Exception as e:
                    print(f"Warning: could not load {energy_path}: {e}")

            if os.path.exists(J_path):
                try:
                    J_loaded = np.load(J_path, allow_pickle=True)
                    J_vals.extend([v for Ji in J_loaded for v in Ji.values()])
                except Exception as e:
                    print(f"Warning: could not load {J_path}: {e}")

        if energies:
            all_energy_data[result['param_value']] = {
                'energies': np.array(energies),
                'color': colors[i]
            }
        if J_vals:
            all_J_data[result['param_value']] = {
                'J_vals': np.array(J_vals, dtype=np.float32),
                'color': colors[i]
            }

    if not all_energy_data:
        print("No energy data available for comparison.")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('white')

    # ---------------- Top-left: Energy histogram ----------------
    for param_val, data in all_energy_data.items():
        ax1.hist(data['energies'], bins=50, alpha=0.7, color=data['color'],
                 density=True, label=f"{param_display_name}={param_val}",
                 edgecolor='black', linewidth=0.3)
    ax1.set_xlabel('Ising Energy', fontsize=16)
    ax1.set_ylabel('Density', fontsize=16)
    ax1.set_title('Energy Distributions', fontsize=18)
    ax1.set_ylim(0, 1)  # limit y-axis
    ax1.tick_params(colors='black', labelsize=12)
    ax1.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='black', labelcolor='black')
    ax1.set_facecolor('white')
    ax1.grid(False)

    # ---------------- Top-right: J histogram ----------------
    if all_J_data:
        for param_val, data in all_J_data.items():
            ax2.hist(data['J_vals'], bins=50, alpha=0.7, color=data['color'],
                     density=True, edgecolor='black', linewidth=0.3,
                     label=f"{param_display_name}={param_val}")
        ax2.set_xlabel('Coupling strength $J_{ij}$', fontsize=16)
        ax2.set_ylabel('Density', fontsize=16)
        ax2.set_title('Sparse Couplings Histogram', fontsize=18)
        ax2.tick_params(colors='black', labelsize=12)
        ax2.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='black', labelcolor='black')
        ax2.set_facecolor('white')
        ax2.grid(False)
    else:
        ax2.text(0.5, 0.5, 'No Couplings Data\nAvailable',
                 transform=ax2.transAxes, ha='center', va='center',
                 fontsize=16, color='black', alpha=0.7)
        ax2.set_xlabel('J', fontsize=16)
        ax2.set_ylabel('Density', fontsize=16)
        ax2.set_facecolor('white')
        ax2.grid(False)

    # ---------------- Bottom-left: Energy CDF ----------------
    for param_val, data in all_energy_data.items():
        sorted_energies = np.sort(data['energies'])
        y_vals = np.linspace(0, 1, len(sorted_energies))
        ax3.plot(sorted_energies, y_vals, color=data['color'], linewidth=2.5)
    ax3.set_xlabel('Ising Energy', fontsize=16)
    ax3.set_ylabel('Cumulative Probability', fontsize=16)
    ax3.set_title('Cumulative Distribution Function', fontsize=18)
    ax3.set_xlim(left=min([np.min(d['energies']) for d in all_energy_data.values()]),
                 right=max([np.max(d['energies']) for d in all_energy_data.values()]))
    ax3.set_ylim(0, 1.1)  # normalized full range
    ax3.tick_params(colors='black', labelsize=12)
    ax3.set_facecolor('white')
    ax3.grid(False)

    # ---------------- Bottom-right: Energy violin ----------------
    import matplotlib
    box_data = [data['energies'] for data in all_energy_data.values()]
    box_colors = [data['color'] for data in all_energy_data.values()]
    box_labels = [str(k) for k in all_energy_data.keys()]

    vp = ax4.violinplot(box_data, showmeans=True, showmedians=True)
    for pc, color in zip(vp['bodies'], box_colors):
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha(0.8)
    if 'cmedians' in vp:
        vp['cmedians'].set_color('white')
    if 'cmeans' in vp:
        vp['cmeans'].set_color('yellow')
    ax4.set_xticks(np.arange(1, len(box_labels)+1))
    ax4.set_xticklabels(box_labels, rotation=45, ha='right', color='black')
    ax4.set_xlabel(param_display_name, fontsize=16)
    ax4.set_ylabel('Ising Energy', fontsize=16)
    ax4.set_title('Energy Distribution Summary', fontsize=18)
    ax4.tick_params(colors='black', labelsize=12)
    ax4.set_facecolor('white')
    ax4.grid(False)

    # ---------------- Styling for all axes ----------------
    for ax in [ax1, ax2, ax3, ax4]:
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(1)

    plt.tight_layout()
    output_file = f'Ising_distributions_detailed_{param_display_name}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='black')
    print(f"Saved: {output_file}")
    plt.close()


def plot_synaptic2(config, epoch_list, log_dir, logger, cc, style, device):

    dataset_name = config.dataset
    data_folder_name = config.data_folder_name

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
    is_CElegans = 'CElegans' in dataset_name

    x_list = []
    y_list = []
    for run in trange(1,2):
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

    # activity = torch.tensor(x_list[0],device=device)
    # activity = activity[:, :, 6:7].squeeze()
    # distrib = to_numpy(activity.flatten())
    # activity = activity.t()

    activity = torch.tensor(x_list[0][:, :, 6:7],device=device)
    activity = activity.squeeze()
    distrib = to_numpy(activity.flatten())
    activity = activity.t()

    if os.path.exists(f'graphs_data/{dataset_name}/raw_x_list_{run}.npy'):
        raw_activity = torch.tensor(raw_x[:, :, 6:7], device=device)
        raw_activity = raw_activity.squeeze()
        raw_activity = raw_activity.t()

    # plt.figure(figsize=(15, 10))
    # window_size = 25
    # window_end = 50000
    # ts = to_numpy(activity[600, :])
    # ts_avg = np.convolve(ts, np.ones(window_size) / window_size, mode='valid')
    # plt.plot(ts[window_size // 2:window_end + window_size // 2], linewidth=1)
    # plt.plot(ts_avg, linewidth=2)
    # plt.plot(ts[window_size // 2:window_end + window_size // 2] - ts_avg[0:window_end])
    # plt.xlim([window_end - 5000, window_end])
    # plt.close
    # signal_power = np.mean(ts_avg[window_size // 2:window_end + window_size // 2] ** 2)
    # # Compute the noise power
    # noise_power = np.mean((ts[window_size // 2:window_end + window_size // 2] - ts_avg[0:window_end]) ** 2)
    # # Calculate the signal-to-noise ratio (SNR)
    # snr = signal_power / noise_power
    # print(f"Signal-to-Noise Ratio (SNR): {snr:0.2f} 10log10 {10 * np.log10(snr):0.2f}")

    # # Parameters
    # fs = 1000  # Sampling frequency
    # t = np.arange(0, 1, 1 / fs)  # Time vector
    # frequency = 5  # Frequency of the sine wave
    # desired_snr_db = 10 * np.log10(snr)  # Desired SNR in dB
    # # Generate a clean signal (sine wave)
    # clean_signal = np.sin(2 * np.pi * frequency * t)
    # # Calculate the power of the clean signal
    # signal_power = np.mean(clean_signal ** 2)
    # # Calculate the noise power required to achieve the desired SNR
    # desired_snr_linear = 10 ** (desired_snr_db / 10)
    # noise_power = signal_power / desired_snr_linear
    # # Generate noise with the calculated power
    # noise = np.sqrt(noise_power) * np.random.randn(len(t))
    # # Create a noisy signal by adding noise to the clean signal
    # noisy_signal = clean_signal + noise
    # # Plot the clean signal and the noisy signal
    # plt.figure(figsize=(15, 10))
    # plt.subplot(2, 1, 1)
    # plt.plot(t, clean_signal)
    # plt.title('Clean Signal')
    # plt.subplot(2, 1, 2)
    # plt.plot(t, noisy_signal)
    # plt.plot(t, noise)
    # plt.title(f'Noisy Signal with SNR = {desired_snr_db} dB')
    # plt.tight_layout()
    # plt.show()


    # if os.path.exists(f'./graphs_data/{dataset_name}/X1.pt') > 0:
    #     X1_first = torch.load(f'./graphs_data/{dataset_name}/X1.pt', map_location=device)
    #     X_msg = torch.load(f'./graphs_data/{dataset_name}/X_msg.pt', map_location=device)
    # else:
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
        d_modulation = (modulation[:, 1:] - modulation[:, :-1]) / delta_t

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
            for file_id_ in trange(0,len(file_id_list)):
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
                    plt.scatter(to_numpy(model_a[pos, 0]), to_numpy(model_a[pos, 1]), s=50, color=mc, alpha=1.0, edgecolors='none')   # cmap.color(n)
                plt.xlabel(r'$a_{0}$', fontsize=68)
                plt.ylabel(r'$a_{1}$', fontsize=68)
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
                ax = sns.heatmap(to_numpy(A)/second_correction, center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046}, vmin=-4,vmax=4)
                cbar = ax.collections[0].colorbar
                cbar.ax.tick_params(labelsize=48)
                plt.xticks([0, n_neurons - 1], [1, n_neurons], fontsize=24)
                plt.yticks([0, n_neurons - 1], [1, n_neurons], fontsize=24)
                plt.subplot(2, 2, 1)

                ax = sns.heatmap(to_numpy(A[0:20, 0:20])/second_correction, cbar=False, center=0, square=True, cmap='bwr', vmin=-4, vmax=4)
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
                            plt.ylabel(r'learned $MLP_1( a_i, a_j, x_j)$', fontsize=32)
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
                    plt.ylabel(r'learned $MLP_1(a_j, x_j)$', fontsize=68)
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
                    # plt.ylabel(r'learned $\psi^*(a_i, x_i)$', fontsize=68)
                    plt.ylabel(r'$MLP_1(a_i, a_j, x_i, x_j)$', fontsize=68)
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
                    # plt.ylabel(r'learned $\psi^*(a_i, x_i)$', fontsize=68)
                    plt.ylabel(r'learned $MLP_1(a_j, x_j)$', fontsize=68)
                    plt.ylim([-1.5, 1.5])
                    plt.xlim([-5,5])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/all/MLP1_{epoch}.png", dpi=80)
                    plt.close()

                fig, ax = fig_init()
                func_list = []
                config_model = config.graph_model.signal_model_name
                for n in range(n_neurons):
                    embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                    # in_features = torch.cat((rr[:, None], embedding_), dim=1)
                    in_features = get_in_features_update(rr[:, None], model, embedding_, device)
                    with torch.no_grad():
                        func = model.lin_phi(in_features.float())
                    func = func[:, 0]
                    plt.plot(to_numpy(rr), to_numpy(func) * to_numpy(ynorm), color=cmap.color(to_numpy(type_list[n]).astype(int)), linewidth=2, alpha=0.25) #
                plt.ylim([-4,4])
                plt.xlabel(r'$x_i$', fontsize=68)
                # plt.ylabel(r'learned $\phi^*(a_i, x_i)$', fontsize=68)
                plt.ylabel(r'learned $MLP_0(a_i, x_i)$', fontsize=68)

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
                    plt.xlabel(r'$x_i$', fontsize=68)
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
                    plt.xlabel(r'$x_i$', fontsize=68)
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
            for i in trange(0,1500,10):
                plt.figure(figsize=(20, 10))
                ax = plt.subplot(121)
                plt.plot(neuron_gt_list[:, n[0]].detach().cpu().numpy(), c='w', linewidth=8, label='true', alpha=0.25)
                plt.plot(neuron_pred_list[0:i, n[0]].detach().cpu().numpy(), linewidth=4, c='w', label='learned')
                plt.legend(fontsize=24)
                plt.plot(neuron_gt_list[:, n[1:10]].detach().cpu().numpy(), c='w', linewidth=8, alpha=0.25)
                plt.plot(neuron_pred_list[0:i, n[1:10]].detach().cpu().numpy(), linewidth=4)
                plt.xlim([0, 1500])
                plt.xlabel('time index', fontsize=48)
                plt.ylabel(r'$x_i$', fontsize=48)
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
                plt.xlabel(r'true $x_i$', fontsize=48)
                plt.text(-13, 13, 'N=1024', fontsize=34)
                plt.ylabel(r'learned $x_i$', fontsize=48)
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
            plt.title(r'true vs learned $x_i$', fontsize=48)
            plt.legend(fontsize=24)
            plt.tight_layout()
            plt.savefig(f'./{log_dir}/results/activity_comparison.png', dpi=80)
            plt.close()

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
        # plt.subplot(2, 2, 1)
        # ax = sns.heatmap(to_numpy(adjacency[0:20, 0:20]), cbar=False, center=0, square=True, cmap='bwr', vmin=-0.1, vmax=0.1)
        # plt.xticks([])
        # plt.yticks([])
        plt.tight_layout()
        plt.savefig(f'./{log_dir}/results/true connectivity.png', dpi=300)
        plt.close()

        plt.figure(figsize=(10, 10))
        if True: # config.graph_model.signal_model_name == 'PDE_N8':
            with open(f'graphs_data/{dataset_name}/larynx_neuron_list.json', 'r') as file:
                larynx_neuron_list = json.load(file)
            with open(f'graphs_data/{dataset_name}/all_neuron_list.json', 'r') as file:
                activity_neuron_list = json.load(file)
            map_larynx_matrix, n = map_matrix(larynx_neuron_list, all_neuron_list, adjacency)
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
        plt.title(r'$x_i$ samples',fontsize=48)
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
                fig = plt.figure(figsize=(10, 10))
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
            for n in trange(0,n_neurons,n_neurons//100):
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
            plt.xlabel(r'$x_i$', fontsize=68)
            plt.ylabel(r'Learned $\psi^*(a_i, x_i)$', fontsize=68)
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
            A_corrected = A
            for i in range(n_neurons):
                A_corrected[i, :] = A[i, :] * matrix_correction[i]
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
                        # plt.ylabel(r'learned $\psi^*(a_i, a_j, x_i)$', fontsize=24)
                        # plt.xlabel(r'$x_i$', fontsize=24)
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

                for n in trange(0,n_neurons):
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

                plt.xlabel(r'$x_i$', fontsize=68)
                if (model_config.signal_model_name == 'PDE_N4'):
                    plt.ylabel(r'learned $\psi^*(a_i, x_i)$', fontsize=68)
                elif model_config.signal_model_name == 'PDE_N5':
                    plt.ylabel(r'learned $\psi^*(a_i, a_j, x_i)$', fontsize=68)
                else:
                    plt.ylabel(r'learned $\psi^*(x_i)$', fontsize=68)
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
            for n in trange(n_neuron_types):
                if model_config.signal_model_name == 'PDE_N5':
                    true_func = true_model.func(rr, n, n, 'update')
                else:
                    true_func = true_model.func(rr, n, 'update')
                plt.plot(to_numpy(rr), to_numpy(true_func), c=mc, linewidth=16, label='original', alpha=0.21)
            phi_list = []
            for n in trange(n_neurons):
                embedding_ = model.a[n, :] * torch.ones((1500, config.graph_model.embedding_dim), device=device)
                # in_features = torch.cat((rr[:, None], embedding_), dim=1)
                in_features = get_in_features_update(rr[:, None], n_neurons, embedding_, model.update_type, device)
                with torch.no_grad():
                    func = model.lin_phi(in_features.float())
                func = func[:, 0]
                phi_list.append(func)
                plt.plot(to_numpy(rr), to_numpy(func) * to_numpy(ynorm),
                         color=cmap.color(to_numpy(type_list[n]).astype(int)), linewidth=2, alpha=0.25)
            phi_list = torch.stack(phi_list)
            func_list_ = to_numpy(phi_list)
            plt.xlabel(r'$x_i$', fontsize=68)
            plt.ylabel(r'learned $\phi^*(a_i, x_i)$', fontsize=68)
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
            for n in trange(n_neuron_types):
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
            gt_weight = to_numpy(adjacency)
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
                    plt.xlabel(r'$x_i$', fontsize=68)
                    plt.ylabel(r'$y_i$', fontsize=68)
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/true_field_derivative.png", dpi=80)
                    plt.close()

                    fig, ax = fig_init()
                    plt.title(r'learned $\dot{y_i}$', fontsize=68)
                    plt.imshow(to_numpy(pred_modulation))
                    plt.xticks([0, 100, 200, 300, 400], [-6, -3, 0, 3, 6], fontsize=48)
                    plt.yticks([0, 100, 200, 300, 400], [0, 0.25, 0.5, 0.75, 1], fontsize=48)
                    plt.xlabel(r'$x_i$', fontsize=68)
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

                    for frame in trange(0, n_frames, n_frames // 100):
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
                        # plt.title(r'$x_i$', fontsize=48)
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

                for frame in trange(0, modulation.shape[1], modulation.shape[1] // 257):
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
                plt.xlabel(r'$x_i$', fontsize=68)
                plt.ylabel(r'learned $MLP_0$', fontsize=68)
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
                plt.xlabel(r'$x_i$', fontsize=68)
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
                plt.ylabel(r'$MLP_0(a_i, x_i=0, sum_i, g_i=1)$', fontsize=48)
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


def plot_synaptic3(config, epoch_list, log_dir, logger, cc, style, device):

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
    embedding_cluster = EmbeddingCluster(config)
    field_type = model_config.field_type
    if field_type != '':
        n_nodes = config.simulation.n_nodes
        n_nodes_per_axis = int(np.sqrt(n_nodes))
        has_field = True
    else:
        has_field = False

    x_list = []
    y_list = []
    for run in trange(1):
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
            for file_id_ in trange(0, 100):
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
                model_a = (model.a - amin) / (amax - amin)

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
                    # plt.xlabel(r'$x_i$', fontsize=24)
                    # plt.ylabel(r'Learned $\phi^*(a_i(t), x_i)$', fontsize=68)
                    if k==0:
                        plt.ylabel(r'Learned $MLP_0(a_i(t), x_i)$', fontsize=32)
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
                # plt.ylabel(r'learned $\psi^*(x_i)$', fontsize=68)
                plt.ylabel(r'learned $MLP_1(x_j)$', fontsize=68)
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

        for n in trange(100):

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
            plt.ylabel(r'learned $MLP_0(a_i(t), x_i)$', fontsize=38)
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
            #     plt.xlabel(r'$x_i$', fontsize=16)
            #     # plt.ylabel(r'Learned $\phi^*(a_i(t), x_i)$', fontsize=68)
            #     plt.ylabel(r'Learned $MLP_0(a_i(t), x_i)$', fontsize=16)
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

                for frame in trange(0, n_frames, n_frames//100):

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
                    fig = plt.figure(figsize=(10, 10))
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

                fig = plt.figure(figsize=(10, 10))
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
                for n in trange(k*100,(k+1)*100):
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
                plt.xlabel(r'$x_i$', fontsize=68)
                plt.ylabel(r'Learned $\phi^*(a_i(t), x_i)$', fontsize=68)
                plt.ylim([-8,8])
                plt.xlim([-5,5])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/phi_{k}.png", dpi=170.7)
                plt.close()

            fig, ax = fig_init()
            rr = torch.tensor(np.linspace(-5, 5, 1000)).to(device)
            func_list = []
            for n in trange(0,n_neurons,n_neurons):
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
            plt.xlabel(r'$x_i$', fontsize=68)
            plt.ylabel(r'Learned $\psi^*(a_i, x_i)$', fontsize=68)
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

            for n in trange(0,n_neurons):
                in_features = rr[:, None]
                with torch.no_grad():
                    func = model.lin_edge(in_features.float()) * correction
                    psi_list.append(func)
                    plt.plot(to_numpy(rr), to_numpy(func), 2, color=mc, linewidth=2, alpha=0.25)
            plt.xlabel(r'$x_i$', fontsize=68)
            plt.ylabel(r'learned $\psi^*(x_i)$', fontsize=68)
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


def data_plot(config, config_file, epoch_list, style, device):

    # plt.rcParams['text.usetex'] = True
    # rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    # matplotlib.rcParams['savefig.pad_inches'] = 0

    if 'black' in style:
        plt.style.use('dark_background')
        mc ='w'
    else:
        plt.style.use('default')
        mc = 'k'

    if 'latex' in style:
        plt.rcParams['text.usetex'] = True
        rc('font', **{'family': 'serif', 'serif': ['Palatino']})

    # plt.rc('font', family='sans-serif')
    # plt.rc('font', family='serif')
    # plt.rcParams['font.family'] = 'Times New Roman'
    # plt.rc('text', usetex=False)
    matplotlib.rcParams['savefig.pad_inches'] = 0

    log_dir, logger = create_log_dir(config=config, erase=False)

    os.makedirs(os.path.join(log_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'results/all'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'results/training'), exist_ok=True)
    files = glob.glob(f"{log_dir}/results/training/*")
    for f in files:
        os.remove(f)
    # files = glob.glob(f"{log_dir}/results/all/*")
    # for f in files:
    #     os.remove(f)
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

    if ('PDE_N' in config.graph_model.signal_model_name):
        if ('PDE_N3' in config.graph_model.signal_model_name):
            plot_synaptic3(config, epoch_list, log_dir, logger, 'viridis', style, device)
        else:
            if 'CElegans' in config.dataset:
                plot_synaptic_CElegans(config, epoch_list, log_dir, logger, 'viridis', style, device)
            if 'fly' in config.dataset:
                plot_synaptic_flyvis(config, epoch_list, log_dir, logger, 'viridis', style, device)
            else:
                plot_synaptic2(config, epoch_list, log_dir, logger, 'viridis', style, device)

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def get_figures(index):

    epoch_list = ['20']
    match index:
        
        case 'synaptic_2_fig2':
            config_list = ['signal_N2_a10']
            epoch_list = ['best']
        case 'synaptic_supp1':
            config_list = [f'signal_N2_a{i}' for i in range(0, 11)]
            epoch_list = ['all']
        case 'synaptic_supp6':
            config_list = ['signal_N2_a_SNR1', 'signal_N2_a_SNR3', 'signal_N2_a_SNR7']
            epoch_list = ['best']

        case _:
            config_list = []

    match index:
        case 'synaptic_2':
            for config_file in config_list:
                config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
                data_plot(config=config, config_file=config_file, epoch_list=epoch_list, device=device, style=True)
                data_test(config=config, config_file=config_file, visualize=True, style='latex frame color', verbose=False,
                                  best_model='best', run=0, step=100, test_simulation=False,
                                  sample_embedding=False, device=device)
                print(' ')
                print(' ')
        case 'synaptic_supp1' | 'synaptic_supp6':
            for config_file in config_list:
                config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
                data_plot(config=config, config_file=config_file, epoch_list=['best'], device=device, style=True)
                data_plot(config=config, config_file=config_file, epoch_list=['all'], device=device, style=True)
        case 'synaptic_supp2':
            plt.rcParams['text.usetex'] = True
            rc('font', **{'family': 'serif', 'serif': ['Palatino']})
            matplotlib.rcParams['savefig.pad_inches'] = 0

            config_list = ['signal_N2_a1', 'signal_N2_a2', 'signal_N2_a3', 'signal_N2_a4', 'signal_N2_a5', 'signal_N2_a10']
            it_list = [1, 2, 3, 4, 5, 10]
            fig, ax = fig_init(formatx='%.0f', formaty='%.2f')
            plt.xlim([0, 100])
            plt.ylim([0, 1.1])
            plt.yticks(fontsize=24)
            plt.xticks([0, 100], [0, 20], fontsize=24)
            plt.ylabel(r'$R^2$', fontsize=48)
            plt.xlabel(r'epoch', fontsize=48)
            for it, config_file_ in enumerate(config_list):

                config_file, pre_folder = add_pre_folder(config_file_)
                config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
                l_dir = get_log_dir(config)
                log_dir = os.path.join(l_dir, config_file.split('/')[-1])
                r_squared_list = np.load(f'./{log_dir}/results/R2.npy')
                plt.plot(r_squared_list, linewidth = 4, label=f'{(it_list[it])*10000}')
                plt.legend(fontsize=20)
            plt.tight_layout()
            plt.savefig(f'./{log_dir}/results/R2_all.png', dpi=300)
            plt.close()

        case 'synaptic_supp5':

            plt.rcParams['text.usetex'] = True
            rc('font', **{'family': 'serif', 'serif': ['Palatino']})
            matplotlib.rcParams['savefig.pad_inches'] = 0

            config_list = ['signal_N2_a10','signal_N2_e1','signal_N2_e2','signal_N2_e3','signal_N2_e']
            labels = [r'100\%', r'50\%', r'20\%', r'10\%', r'5\%']
            fig, ax = fig_init(formatx='%.0f', formaty='%.2f')
            plt.xlim([0, 100])
            plt.ylim([0, 1.1])
            plt.yticks(fontsize=24)
            plt.xticks([0, 100], [0, 20], fontsize=24)
            plt.ylabel(r'$R^2$', fontsize=48)
            plt.xlabel(r'epoch', fontsize=48)
            for it, config_file_ in enumerate(config_list):

                config_file, pre_folder = add_pre_folder(config_file_)
                config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
                l_dir = get_log_dir(config)
                log_dir = os.path.join(l_dir, config_file.split('/')[-1])
                r_squared_list = np.load(f'./{log_dir}/results/R2.npy')
                plt.plot(r_squared_list, linewidth = 4, label=labels[it])
                plt.legend(fontsize=20)
            plt.tight_layout()
            plt.savefig(f'./{log_dir}/results/R2_all.png', dpi=300)
            plt.close()


        case '3' | '4' |'4_bis' | 'supp4' | 'supp5' | 'supp6' | 'supp7' | 'supp8' | 'supp9' | 'supp10' | 'supp11' | 'supp12' | 'supp15' |'supp16' |'supp18':
            for config_file in config_list:
                config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
                data_plot(config=config, config_file=config_file, epoch_list=epoch_list, device=device)
                data_test(config=config, config_file=config_file, visualize=True, style='latex frame color', verbose=False,
                                  best_model=20, run=0, step=64, test_simulation=False,
                                  sample_embedding=False, device=device)  # config.simulation.n_frames // 7
                print(' ')
                print(' ')


    print(' ')
    print(' ')

    return config_list,epoch_list


if __name__ == '__main__':

    warnings.filterwarnings("ignore", category=FutureWarning)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


    print(' ')
    print(f'device {device}')

    # try:
    #     matplotlib.use("Qt5Agg")
    # except:
    #     pass

    #config_list = ['signal_N5_v6','signal_N5_v6_0','signal_N5_v6_1','signal_N5_v6_2', 'signal_N5_v6_3', 'signal_N5_v7_1','signal_N5_v7_2','signal_N5_v7_3', 'signal_N5_v8','signal_N5_v9','signal_N5_v10',
    #                'signal_N5_v11','signal_N5_v12','signal_N5_v13','signal_N5_v14','signal_N5_v15']
    # config_list = ['signal_N4_a3','signal_N4_a4']
    # config_list = ['signal_N2_a43_3_1_t8','signal_N2_a43_3_5_t8','signal_N2_a43_3_10_t8','signal_N2_a43_3_20_t8','signal_N2_a43_3_1_t16','signal_N2_a43_3_5_t16',
    #                'signal_N2_a43_3_10_t16','signal_N2_a43_3_20_t16','signal_N2_a43_3_20_t20','signal_N2_a43_3_20_t24','signal_N2_a43_3_20_t28']

    # config_list = ['signal_N4_CElegans_a6', 'signal_N4_CElegans_a7', 'signal_N4_CElegans_a8', 'signal_N4_CElegans_a9',
    # config_list = ['signal_N4_CElegans_a7_1', 'signal_N4_CElegans_a7_2', 'signal_N4_CElegans_a7_3', 'signal_N4_CElegans_a7_4', 'signal_N4_CElegans_a7_5', 'signal_N4_CElegans_a7_6', 'signal_N4_CElegans_a7_7', 'signal_N4_CElegans_a9_1', 'signal_N4_CElegans_a9_2', 'signal_N4_CElegans_a9_3', 'signal_N4_CElegans_a9_4', 'signal_N4_CElegans_a9_5']
    # config_list = ['signal_N2_a43_2_1_t16']

    # config_list = [ 'signal_CElegans_c14_4a', 'signal_CElegans_c14_4b', 'signal_CElegans_c14_4c',  'signal_CElegans_d1', 'signal_CElegans_d2', 'signal_CElegans_d3', ]
    # config_list = config_list = ['signal_CElegans_d2', 'signal_CElegans_d2a', 'signal_CElegans_d3', 'signal_CElegans_d3a', 'signal_CElegans_d3b']

    # config_list = ['fly_N9_18_4_1', 'fly_N9_18_4_0', 'fly_N9_20_0', 'fly_N9_22_1', 'fly_N9_22_2', 'fly_N9_22_3', 'fly_N9_22_4', 'fly_N9_23_1', 'fly_N9_23_2', 'fly_N9_23_3', 'fly_N9_23_4', 'fly_N9_23_5', 'fly_N9_18_4_2', 'fly_N9_18_4_3', 'fly_N9_18_4_4', 'fly_N9_18_4_5', 'fly_N9_18_4_6']

    # plot no noise at all
    # config_list = ['fly_N9_22_1', 'fly_N9_22_2', 'fly_N9_22_3', 'fly_N9_22_4']
    # data_flyvis_compare(config_list, 'training.seed')

    # # plot noise on video input
    # config_list = ['fly_N9_18_4_0_bis', 'fly_N9_18_4_0',  'fly_N9_20_0', 'fly_N9_22_1', 'fly_N9_22_2', 'fly_N9_22_3', 'fly_N9_22_4', 'fly_N9_23_1', 'fly_N9_23_2', 'fly_N9_23_3', 'fly_N9_23_4', 'fly_N9_23_5']
    # data_flyvis_compare(config_list, 'simulation.noise_visual_input')


    # plot noise on video input 50/50
    # config_list = ['fly_N9_18_4_0_bis', 'fly_N9_18_4_0',  'fly_N9_20_0', 'fly_N9_22_1', 'fly_N9_22_2', 'fly_N9_22_3', 'fly_N9_22_4',
    #                'fly_N9_33_1', 'fly_N9_33_1_1', 'fly_N9_33_1_2', 'fly_N9_33_1_3', 'fly_N9_33_3', 'fly_N9_33_4', 'fly_N9_33_5', 'fly_N9_33_5_1']
    # data_flyvis_compare(config_list, 'training.noise_model_level')

    # plot noise on video input 50/50
    # config_list = ['fly_N9_18_4_0_bis', 'fly_N9_18_4_0',  'fly_N9_20_0', 'fly_N9_22_1', 'fly_N9_22_2', 'fly_N9_22_3', 'fly_N9_22_4',
    #                'fly_N9_31_1', 'fly_N9_31_2', 'fly_N9_31_3', 'fly_N9_31_4', 'fly_N9_31_5','fly_N9_31_6', 'fly_N9_31_7']
    # data_flyvis_compare(config_list, 'simulation.only_noise_visual_input')

    # config_list = ['fly_N9_18_4_0_bis', 'fly_N9_18_4_0',  'fly_N9_20_0', 'fly_N9_22_1', 'fly_N9_22_2', 'fly_N9_22_3', 'fly_N9_22_4',
    #                'fly_N9_34_1', 'fly_N9_34_2', 'fly_N9_34_3'] #, 'fly_N9_35_1', 'fly_N9_35_2']
    # data_flyvis_compare(config_list, 'simulation.n_extra_null_edges')

    # config_list = ['fly_N9_18_4_0_bis', 'fly_N9_18_4_0',  'fly_N9_20_0', 'fly_N9_22_1', 'fly_N9_22_2', 'fly_N9_22_3', 'fly_N9_22_4',  'fly_N9_35_1', 'fly_N9_35_2']
    # data_flyvis_compare(config_list, 'simulation.visual_input_type')

    # config_list = ['fly_N9_31_1', 'fly_N9_36_2', 'fly_N9_36_3', 'fly_N9_36_4', 'fly_N9_36_5', 'fly_N9_31_6', 'fly_N9_36_7']
    # data_flyvis_compare(config_list, 'training.recursive_loop')

    # config_list = ['fly_N9_18_4_10', 'fly_N9_18_4_11', 'fly_N9_18_4_12', 'fly_N9_18_4_13', 'fly_N9_18_4_14']

    # config_list = ['fly_N9_30_10', 'fly_N9_30_11', 'fly_N9_30_12', 'fly_N9_30_13', 'fly_N9_30_14', 'fly_N9_30_15']

    # config_list = ['fly_N9_31_5', 'fly_N9_36_1', 'fly_N9_36_2', 'fly_N9_36_3', 'fly_N9_36_4', 'fly_N9_36_5', 'fly_N9_31_6', 'fly_N9_36_7']
    # data_flyvis_compare(config_list, 'training.recursive_loop')

    # config_list = ['fly_N9_18_4_15', 'fly_N9_18_4_16', 'fly_N9_18_4_17', 'fly_N9_18_4_18', 'fly_N9_38_1', 'fly_N9_38_2', 'fly_N9_38_3', 'fly_N9_38_4', 'fly_N9_39_0', 'fly_N9_39_1', 'fly_N9_39_2', 'fly_N9_39_3']
    # data_flyvis_compare(config_list, None)

    # config_list = ['fly_N9_18_4_0', 'fly_N9_33_5', 'fly_N9_18_4_1', 'fly_N9_33_5_1']
    # config_list = ['fly_N9_18_4_14', 'fly_N9_31_5']

    # config_list = ['fly_N9_40_1', 'fly_N9_40_2', 'fly_N9_40_3', 'fly_N9_40_5', 'fly_N9_40_6', 'fly_N9_40_7', 'fly_N9_40_8', 'fly_N9_40_9','fly_N9_40_10','fly_N9_40_10', 'fly_N9_40_12'] #, 'fly_N9_41_1', 'fly_N9_41_2']
    # data_flyvis_compare(config_list, None)

    # config_list = ['fly_N9_37_2', 'fly_N9_37_2_1', 'fly_N9_37_2_2', 'fly_N9_37_2_3', 'fly_N9_37_2_4', 'fly_N9_37_2_5']
    # data_flyvis_compare(config_list, 'training.learning_rate_embedding_start')

    # config_list = ['fly_N9_43_1', 'fly_N9_43_2', 'fly_N9_43_3', 'fly_N9_43_4', 'fly_N9_43_5']
    # data_flyvis_compare(config_list, 'training.loss_noise_level')

    # config_list = ['fly_N9_44_9', 'fly_N9_44_10', 'fly_N9_44_11', 'fly_N9_44_12']
    # config_list = ['fly_N9_22_1', 'fly_N9_44_1', 'fly_N9_44_2', 'fly_N9_44_3', 'fly_N9_44_4', 'fly_N9_44_5', 'fly_N9_44_6', 'fly_N9_44_7']
    # data_flyvis_compare(config_list, 'training.noise_model_level')

    # config_list = ['fly_N9_45_1', 'fly_N9_45_2']

    # config_list = ['fly_N9_46_1', 'fly_N9_46_2', 'fly_N9_46_3', 'fly_N9_46_4', 'fly_N9_46_5', 'fly_N9_46_6']
    # data_flyvis_compare(config_list, None)

    # config_list = ['fly_N9_47_1', 'fly_N9_47_2', 'fly_N9_47_3', 'fly_N9_47_4', 'fly_N9_47_5','fly_N9_47_6']
    # data_flyvis_compare(config_list, None)
    #
    # config_list = ['fly_N9_49_1', 'fly_N9_49_2', 'fly_N9_49_3', 'fly_N9_49_4', 'fly_N9_49_5','fly_N9_49_6', 'fly_N9_49_7', 'fly_N9_49_8', 'fly_N9_49_9', 'fly_N9_49_10', 'fly_N9_49_11', 'fly_N9_49_12', 'fly_N9_49_13', 'fly_N9_49_14']
    # data_flyvis_compare(config_list, 'training.coeff_edge_weight_L1')

    # config_list = ['fly_N9_48_1', 'fly_N9_48_2', 'fly_N9_48_3', 'fly_N9_48_4', 'fly_N9_48_5', 'fly_N9_48_6']
    # data_flyvis_compare(config_list, 'training.coeff_edge_weight_L2')

    # config_list = ['fly_N9_50_1', 'fly_N9_50_2', 'fly_N9_50_3', 'fly_N9_50_4', 'fly_N9_50_5', 'fly_N9_50_6', 'fly_N9_50_7']
    # data_flyvis_compare(config_list, 'training.Ising_filter')

    # config_list = ['fly_N9_51_1', 'fly_N9_51_2', 'fly_N9_51_3', 'fly_N9_51_4', 'fly_N9_51_5', 'fly_N9_51_6', 'fly_N9_51_7']
    # data_flyvis_compare(config_list, 'simulation.n_extra_null_edges')

    # config_list = ['fly_N9_52_1', 'fly_N9_52_2', 'fly_N9_52_3', 'fly_N9_52_4', 'fly_N9_52_5', 'fly_N9_52_6', 'fly_N9_52_7', 'fly_N9_52_8', 'fly_N9_52_9']
    # data_flyvis_compare(config_list, 'simulation.n_frames')

    # config_list = ['fly_N9_53_1', 'fly_N9_53_2', 'fly_N9_53_3', 'fly_N9_53_4', 'fly_N9_53_5', 'fly_N9_53_6', 'fly_N9_53_7', 'fly_N9_53_8']
    # data_flyvis_compare(config_list, None)

    # config_list = ['fly_N9_53_3']
    # config_list = ['fly_N9_22_1', 'fly_N9_22_2', 'fly_N9_22_3', 'fly_N9_22_4', 'fly_N9_22_5', 'fly_N9_22_6', 'fly_N9_22_7', 'fly_N9_22_8']
    # data_flyvis_compare(config_list, None)

    # config_list = ['fly_N9_52_2', 'fly_N9_52_2_1', 'fly_N9_52_2_2', 'fly_N9_52_2_3', 'fly_N9_52_2_4', 'fly_N9_52_2_5', 'fly_N9_52_2_6', 'fly_N9_52_2_7']
    # data_flyvis_compare(config_list, None)

    # config_list = ['fly_N9_52_9_1', 'fly_N9_52_9_2', 'fly_N9_52_9_3', 'fly_N9_52_9_4', 'fly_N9_52_9_5', 'fly_N9_52_9_6', 'fly_N9_52_9_7']
    # data_flyvis_compare(config_list, 'none')

    config_list = ['fly_N9_22_1']
    # #
    # # config_list = ['fly_N9_52_2', 'fly_N9_52_2_1', 'fly_N9_52_2_2', 'fly_N9_52_2_3', 'fly_N9_52_2_4', 'fly_N9_52_2_5',
    # #                'fly_N9_52_2_6', 'fly_N9_52_2_7', 'fly_N9_52_9_1', 'fly_N9_52_9_2', 'fly_N9_52_9_3', 'fly_N9_52_9_4',
    # #                'fly_N9_52_9_5', 'fly_N9_52_9_6', 'fly_N9_52_9_7']
    #
    #
    # #
    for config_file_ in config_list:
        print(' ')

        config_file, pre_folder = add_pre_folder(config_file_)
        config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
        config.dataset = pre_folder + config.dataset
        config.config_file = pre_folder + config_file_

        print(f'\033[94mconfig_file  {config.config_file}\033[0m')

        folder_name = './log/' + pre_folder + '/tmp_results/'
        os.makedirs(folder_name, exist_ok=True)
        data_plot(config=config, config_file=config_file, epoch_list=['best'], style='black color', device=device)









