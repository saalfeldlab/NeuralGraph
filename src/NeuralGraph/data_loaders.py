from NeuralGraph.generators.utils import choose_boundary_values, get_equidistant_points
from NeuralGraph.utils import CustomColorMap, get_neuron_index, map_matrix, to_numpy
import os
from dataclasses import dataclass
from typing import Dict

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import torch
from astropy.units import Unit
from tqdm import tqdm, trange


import json
from tqdm import trange
from skimage.measure import label, regionprops
import scipy.io as sio
import seaborn as sns
from torch_geometric.utils import dense_to_sparse
import pickle
import json
import scipy.io
from skimage.draw import disk
import pandas as pd
import scipy.io
from matplotlib.colors import LinearSegmentedColormap
import torch.nn.functional as F



def linear_model(x, a, b):
    return a * x + b


def extract_object_properties(segmentation_image, fluorescence_image=[], radius=40, offset_channel=[0.0, 0.0]):
    # Label the objects in the segmentation image
    labeled_image = label(segmentation_image)
    fluorescence_image = np.flipud(fluorescence_image)
    # fig = plt.figure(figsize=(13, 10.5))
    # plt.imshow(fluorescence_image)
    # plt.show()

    # Extract properties of the labeled objects
    object_properties = []
    for id, region in enumerate(regionprops(labeled_image, intensity_image=fluorescence_image)):
        # Get the cell ID
        cell_id = id

        pos_x = region.centroid[0]
        pos_y = region.centroid[1]

        # Calculate the area of the object
        area = region.area

        if area>8:

            # Calculate the perimeter of the object
            perimeter = region.perimeter

            # Calculate the aspect ratio of the bounding box
            aspect_ratio = region.major_axis_length / (region.minor_axis_length + 1e-6)

            # Calculate the orientation of the object
            orientation = region.orientation

            rr, cc = disk((pos_x+offset_channel[0], pos_y+offset_channel[1]), radius, shape=fluorescence_image.shape)

            # Ensure the coordinates are within bounds
            valid_coords = (rr >= 0) & (rr < fluorescence_image.shape[0]) & (cc >= 0) & (
                        cc < fluorescence_image.shape[1])

            rr_valid = rr[valid_coords]
            cc_valid = cc[valid_coords]

            # Extract the fluorescence values inside the circular mask
            fluo_sum_radius = np.sum(fluorescence_image[rr_valid, cc_valid])
            fluo_sum_segmentation = region.mean_intensity * area


            object_properties.append((id, pos_x, pos_y, area, perimeter, aspect_ratio, orientation, fluo_sum_segmentation, fluo_sum_radius))

    # tmp = fluorescence_image
    # tmp[rr_valid_104, cc_valid_104] = tmp[rr_valid_104, cc_valid_104] + 0.25
    # fig = plt.figure(figsize=(13, 10.5))
    # plt.imshow(tmp)
    #
    #
    # fig = plt.figure(figsize=(13, 10.5))
    # plt.imshow(fluorescence_image)
    # for i in range(len(object_properties)):
    #     pos_x = object_properties[i][1]
    #     pos_y = object_properties[i][2]
    #     plt.scatter(pos_y, pos_x, s=100, c=object_properties[i][7], cmap='viridis', vmin=0, vmax=4000, alpha=0.75)
    #     plt.text(pos_y, pos_x, f'{i}', fontsize=10, color='w')
    # plt.show()


    return object_properties


def find_closest_neighbors(track, x):
    closest_neighbors = []
    for row in track:
        distances = torch.sqrt(torch.sum((x[:, 1:3] - row[1:3]) ** 2, dim=1))
        closest_index = torch.argmin(distances)
        closest_neighbors.append(closest_index.item())
    return closest_neighbors


def get_index_particles(x, n_neurons, dimension):
    index_particles = []
    for n in range(n_neurons):
        if dimension == 2:
            index = np.argwhere(x[:, 5].detach().cpu().numpy() == n)
        elif dimension == 3:
            index = np.argwhere(x[:, 7].detach().cpu().numpy() == n)
        index_particles.append(index.squeeze())
    return index_particles


def skip_to(file, start_line):
    with open(file) as f:
        pos = 0
        cur_line = f.readline()
        while cur_line != start_line:
            pos += 1
            cur_line = f.readline()

        return pos + 1


def process_trace(trace):
    '''
    Returns activity traces with normalization based on mean and standard devation.
    '''
    worm_trace = (trace - np.nanmean(trace))/np.nanstd(trace)
    return worm_trace


def process_activity(activity_worms):
    '''
    Returns a list of matrices corresponding to the data missing in the activity columns of the activity_worms dataframes and
    a matrix of the activity with NaNs replaced by 0's
    '''
    missing_data, activity_data = [],[]
    for id in range(len(activity_worms)):
        worm = (activity_worms[id] - activity_worms[id].mean())/activity_worms[id].std()
        act_matrix = worm
        missing_act = np.zeros(act_matrix.shape)
        missing_act[np.isnan(act_matrix)] = 1
        act_matrix[np.isnan(act_matrix)] = 0
        missing_data.append(missing_act)
        activity_data.append(act_matrix)
    return activity_data, missing_data


def load_worm_Kato_data(config, device=None, visualize=None, step=None, cmap=None):

    # data from https://osf.io/2395t/    Global Brain Dynamics Embed the Motor Command Sequence of Caenorhabditis elegans


    data_folder_name = config.data_folder_name
    dataset_name = config.dataset
    connectome_folder_name = config.connectome_folder_name

    simulation_config = config.simulation
    train_config = config.training
    n_frames = simulation_config.n_frames

    n_runs = train_config.n_runs

    delta_t = simulation_config.delta_t
    bc_pos, bc_dpos = choose_boundary_values('no')
    cmap = CustomColorMap(config=config)

    folder = f'./graphs_data/{dataset_name}/'
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f'./graphs_data/{dataset_name}/Fig/', exist_ok=True)

    with h5py.File(data_folder_name, 'r') as f:
        # View top-level structure
        print(list(f.keys()))
        wt_data = f['WT_NoStim']
        print(list(wt_data.keys()))

    with h5py.File(data_folder_name, 'r') as f:
        # Access the deltaFOverF dataset
        delta_f_over_f = f['WT_NoStim']['deltaFOverF']

        # Iterate through the object references
        for i, ref in enumerate(delta_f_over_f):
            # Dereference the object reference
            dereferenced_data = f[ref[0]]
            # Check if the dereferenced object is a dataset
            if isinstance(dereferenced_data, h5py.Dataset):
                print(f"Dereferenced Dataset {i}: {dereferenced_data[()].shape}")
            elif isinstance(dereferenced_data, h5py.Group):
                print(f"Dereferenced Group {i}: Contains keys {list(dereferenced_data.keys())}")

        wt_data = f['WT_NoStim']['NeuronNames']
        first_ref_data = f[wt_data[1][0]]  # This should point to another object that stores data
        neuron_references = first_ref_data[:]  # Read all the references from this dataset
        for i, neuron_ref in enumerate(neuron_references):
            # Dereference each object reference to get the actual neuron name
            dereferenced_neuron = f[neuron_ref[0]]
            neuron_name = dereferenced_neuron[()]  # Get the actual neuron name

            # Convert the neuron name from numbers (if they are ASCII values) to characters
            decoded_name = ''.join(chr(int(num[0])) for num in neuron_name)

            print(f"Neuron {i + 1} name:", decoded_name)


def plot_worm_adjacency_matrix(weights, all_neuron_list, title, output_path):
    """
    Plots the adjacency matrix and weights for the given chemical and electrical weights.

    Parameters:
        chem_weights (torch.Tensor): Chemical weights matrix.
        eassym_weights (torch.Tensor): Electrical weights matrix.
        all_neuron_list (list): List of neuron names.
        output_path (str): Path to save the output plot.
    """


    fig = plt.figure(figsize=(30, 15))

    # Plot adjacency matrix
    ax = fig.add_subplot(121)
    sns.heatmap(weights > 0, center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046})
    ax.set_xticks(range(len(all_neuron_list)))
    ax.set_xticklabels(all_neuron_list, fontsize=6, rotation=90)
    ax.set_yticks(range(len(all_neuron_list)))
    ax.set_yticklabels(all_neuron_list, fontsize=6)
    plt.title(title, fontsize=18)
    plt.xlabel('presynaptic', fontsize=18)
    plt.ylabel('postsynaptic', fontsize=18)

    # Plot weights
    ax = fig.add_subplot(122)
    sns.heatmap(weights, center=0, square=True, cmap='bwr', vmin=0, vmax=30, cbar_kws={'fraction': 0.046})
    ax.set_xticks(range(len(all_neuron_list)))
    ax.set_xticklabels(all_neuron_list, fontsize=6, rotation=90)
    ax.set_yticks(range(len(all_neuron_list)))
    ax.set_yticklabels(all_neuron_list, fontsize=6)
    plt.title('weights', fontsize=18)
    plt.xlabel('presynaptic', fontsize=18)
    plt.ylabel('postsynaptic', fontsize=18)

    plt.tight_layout()
    plt.savefig(output_path, dpi=500)
    plt.close()


def load_wormvae_data(config, device=None, visualize=None, step=None, cmap=None):

    data_folder_name = config.data_folder_name
    dataset_name = config.dataset
    connectome_folder_name = config.connectome_folder_name

    simulation_config = config.simulation
    train_config = config.training
    n_frames = simulation_config.n_frames

    n_runs = train_config.n_runs
    baseline = simulation_config.baseline_value

    delta_t = simulation_config.delta_t
    bc_pos, bc_dpos = choose_boundary_values('no')
    cmap = CustomColorMap(config=config)

    folder = f'./graphs_data/{dataset_name}/'
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f'./graphs_data/{dataset_name}/Fig/', exist_ok=True)
    os.makedirs(f'./graphs_data/{dataset_name}/Fig/AVFL/', exist_ok=True)
    os.makedirs(f'./graphs_data/{dataset_name}/Fig/I1L/', exist_ok=True)

    # Loading Data from class Worm_Data_Loader(Dataset) in https://github.com/TuragaLab/wormvae

    with open(connectome_folder_name+"all_neuron_names.json", "r") as f:
        all_neuron_list = json.load(f)
    all_neuron_list = [str(neuron) for neuron in all_neuron_list]
    with open(f"graphs_data/{dataset_name}/all_neuron_list.json", "w") as f:
        json.dump(all_neuron_list, f)
    with open(connectome_folder_name+"activity_neuron_list.pkl", "rb") as f:
        activity_neuron_list = pickle.load(f)
    activity_neuron_list = [str(neuron) for neuron in activity_neuron_list]
    with open(f"graphs_data/{dataset_name}/activity_neuron_list.json", "w") as f:
        json.dump(activity_neuron_list, f)

    # Find neurons in all_neuron_list but not in activity_neuron_list
    not_recorded_neurons = list(set(all_neuron_list) - set(activity_neuron_list))

    print(f"neurons with activity data: {len(activity_neuron_list)}")
    print(f"neurons without activity data: {len(not_recorded_neurons)}")
    print (f"total {len(all_neuron_list)} {len(not_recorded_neurons) + len(activity_neuron_list)}")
    # all_neuron_list = [*activity_neuron_list, *not_recorded_neurons]

    print ('load data from Worm_Data_Loader ...')
    odor_channels = 3
    step = 0.25
    n_runs = 21
    n_neurons = 189
    T = 960
    N_length = 109
    T_start = 0
    activity_datasets = np.zeros((n_runs, n_neurons, T))
    odor_datasets = np.zeros((n_runs, odor_channels, T))

    print ('load traces ...')

    trace_variable = sio.loadmat(data_folder_name)
    trace_arr = trace_variable['traces']
    is_L = trace_variable['is_L']
    stimulate_seconds = trace_variable['stim_times']
    stims = trace_variable['stims']

    mean_value = np.nanmean(activity_datasets)
    min_value = np.nanmin(activity_datasets)
    max_value = np.nanmax(activity_datasets)
    std_value = np.nanstd(activity_datasets)
    print(f'mean: {mean_value}, min: {min_value}, max: {max_value}, std: {std_value}')

    for idata in trange(n_runs):
        ineuron = 0
        for ifile in range(N_length):
            if trace_arr[ifile][0].shape[1] == 42:
                data = trace_arr[ifile][0][0][idata]
                if data.shape[0] < 1:
                    activity_datasets[idata][ineuron][:] = np.nan
                else:
                    activity_datasets[idata][ineuron][0:data[0].shape[0]] = data[0]
                ineuron += 1
                data = trace_arr[ifile][0][0][idata + 21]
                if data.shape[0] < 1:
                    activity_datasets[idata][ineuron][:] = np.nan
                else:
                    activity_datasets[idata][ineuron][0:data[0].shape[0]] = data[0]
                ineuron += 1
            else:
                data = trace_arr[ifile][0][0][idata]
                if data.shape[0] < 1:
                    activity_datasets[idata][ineuron][:] = np.nan
                else:
                    activity_datasets[idata][ineuron][0:data[0].shape[0]] = data[0]
                ineuron += 1

    activity_datasets = activity_datasets[:, :, T_start:]
    neuron_OI = get_neuron_index('AVFL', activity_neuron_list)
    for data_OI in range(n_runs):
        activity = activity_datasets[data_OI, neuron_OI, :]
        fig = plt.figure(figsize=(20, 2))
        activity = activity_datasets[data_OI, neuron_OI, :]
        plt.plot(activity, linewidth=1, c='b')
        activity = activity_datasets[data_OI, neuron_OI+1, :]
        plt.plot(activity, linewidth=1, c='r')
        plt.title(f'{data_OI} {neuron_OI} {activity_neuron_list[neuron_OI]}', fontsize=18)
        plt.savefig(f"graphs_data/{dataset_name}/Fig/AVFL/Fig_{data_OI:03d}_{neuron_OI:03d}.tif", dpi=80)
        plt.close()
    neuron_OI = get_neuron_index('I1L', activity_neuron_list)
    for data_OI in range(n_runs):
        activity = activity_datasets[data_OI, neuron_OI, :]
        fig = plt.figure(figsize=(20, 2))
        activity = activity_datasets[data_OI, neuron_OI, :]
        plt.plot(activity, linewidth=1, c='b')
        activity = activity_datasets[data_OI, neuron_OI+1, :]
        plt.plot(activity, linewidth=1, c='r')
        plt.title(f'{data_OI} {neuron_OI} {activity_neuron_list[neuron_OI]}', fontsize=18)
        plt.savefig(f"graphs_data/{dataset_name}/Fig/I1L/Fig_{data_OI:03d}_{neuron_OI:03d}.tif", dpi=80)
        plt.close()

    # add baseline
    activity_worm = activity_datasets + baseline
    # activity_with_zeros, missing_matrix = process_activity(activity_worm)
    # activity_worm = process_trace(activity_worm)

    time = np.arange(start=0, stop=T * step, step=step)
    odor_list = ['butanone', 'pentanedione', 'NaCL']
    for idata in range(n_runs):
        for it_stimu in range(stimulate_seconds.shape[0]):
            tim1_ind = time > stimulate_seconds[it_stimu][0]
            tim2_ind = time < stimulate_seconds[it_stimu][1]
            odor_on = np.multiply(tim1_ind.astype(int), tim2_ind.astype(int))
            stim_odor = stims[idata][it_stimu] - 1
            odor_datasets[idata][stim_odor][:] = odor_on

    odor_worms = odor_datasets[:, :, T_start:]

    os.makedirs(f"graphs_data/{dataset_name}/Fig/Fig/", exist_ok=True)
    os.makedirs(f"graphs_data/{dataset_name}/Fig/Kinograph/", exist_ok=True)
    os.makedirs(f"graphs_data/{dataset_name}/Fig/Denoise/", exist_ok=True)

    print ('load connectome ...')

    chem_weights = torch.load(connectome_folder_name + 'chem_weights.pt')
    eassym_weights = torch.load(connectome_folder_name + 'eassym_weights.pt')
    chem_sparsity = torch.load(connectome_folder_name + 'chem_sparsity.pt')
    esym_sparsity = torch.load(connectome_folder_name + 'esym_sparsity.pt')
    map_Turuga_matrix = chem_weights+eassym_weights
    map_Turuga_matrix = map_Turuga_matrix.to(device=device)
    # plot_worm_adjacency_matrix(to_numpy(map_Turuga_matrix), all_neuron_list, 'adjacency matrix Turuga 2022', f"graphs_data/{dataset_name}/full_Turuga_adjacency_matrix.png")

    print('load connectomes from other data ...')

    # Comparison with data from https://wormwiring.org/pages/adjacency.html
    # Cook 2019 Whole-animal connectomes of both Caenorhabditis

    file_path = '/groups/saalfeld/home/allierc/signaling/Celegans/Cook_2019/SI_5_corrected_July_2020_bis.xlsx'
    sheet_name = 'male chemical'
    Cook_neuron_chem_names = pd.read_excel(file_path, sheet_name=sheet_name, usecols='C', skiprows=3, nrows=382, header=None)
    Cook_neuron_chem_names = [str(label) for label in Cook_neuron_chem_names.squeeze()]
    Cook_matrix_chem = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=3, nrows=382, usecols='D:NU', header=None)
    Cook_matrix_chem = np.array(Cook_matrix_chem)
    Cook_matrix_chem = np.nan_to_num(Cook_matrix_chem, nan=0.0)
    Cook_matrix_chem = torch.tensor(Cook_matrix_chem, dtype=torch.float32, device=device).t()
    file_path = '/groups/saalfeld/home/allierc/signaling/Celegans/Cook_2019/SI_5_corrected_July_2020_bis.xlsx'
    sheet_name = 'male gap jn symmetric'
    Cook_neuron_elec_names = pd.read_excel(file_path, sheet_name=sheet_name, usecols='C', skiprows=3, nrows=586, header=None)
    Cook_neuron_elec_names = [str(label) for label in Cook_neuron_elec_names.squeeze()]
    Cook_matrix_elec = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=3, nrows=586, usecols='D:VQ', header=None)
    Cook_matrix_elec = np.array(Cook_matrix_elec)
    Cook_matrix_elec = np.nan_to_num(Cook_matrix_elec, nan=0.0)
    Cook_matrix_elec = torch.tensor(Cook_matrix_elec, dtype=torch.float32, device=device).t()
    map_Cook_matrix_chem , index = map_matrix(all_neuron_list, Cook_neuron_chem_names, Cook_matrix_chem)
    map_Cook_matrix_elec , index = map_matrix(all_neuron_list, Cook_neuron_elec_names, Cook_matrix_elec)
    map_Cook_matrix = map_Cook_matrix_chem + map_Cook_matrix_elec
    # plot_worm_adjacency_matrix(to_numpy(map_Cook_matrix), all_neuron_list, 'adjacency matrix Cook 2019', f"graphs_data/{dataset_name}/full_Cook_adjacency_matrix.png")

    # Comparison with data from https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.0020095
    data_Kaiser = scipy.io.loadmat('/groups/saalfeld/home/allierc/signaling/Celegans/Kaiser_2006/celegans277.mat')
    positions = data_Kaiser['celegans277positions']
    labels_raw = data_Kaiser['celegans277labels']
    Kaiser_neuron_names = [str(label[0]) for label in labels_raw.squeeze()]
    Kaiser_matrix = np.array(data_Kaiser['celegans277matrix'])
    Kaiser_matrix = torch.tensor(Kaiser_matrix, dtype=torch.float32, device=device)
    map_Kaiser_matrix , index = map_matrix(all_neuron_list, Kaiser_neuron_names, Kaiser_matrix)
    # plot_worm_adjacency_matrix(to_numpy(map_Kaiser_matrix), all_neuron_list, 'adjacency matrix Kaiser 2006', f"graphs_data/{dataset_name}/full_Kaiser_adjacency_matrix.png")

    # Comparison with data from https://github.com/openworm/VarshneyEtAl2011
    # Structural Properties of the <i>Caenorhabditis elegans</i> Neuronal Network
    file_path = '/groups/saalfeld/home/allierc/signaling/Celegans/Varshney_2011/ConnOrdered_040903.mat'
    mat_data = scipy.io.loadmat(file_path)
    chemical_connectome = mat_data['A_init_t_ordered']
    electrical_connectome = mat_data['Ag_t_ordered']
    neuron_names_raw = mat_data['Neuron_ordered']
    Varshney_matrix = np.array((chemical_connectome+electrical_connectome).todense())
    Varshney_matrix = torch.tensor(Varshney_matrix, dtype=torch.float32, device=device).t()
    Varshney_neuron_names = [str(cell[0][0]) for cell in neuron_names_raw]
    map_Varshney_matrix , index = map_matrix(all_neuron_list, Varshney_neuron_names, Varshney_matrix)
    # plot_worm_adjacency_matrix(to_numpy(map_Varshney_matrix), all_neuron_list, 'adjacency matrix Varshney 2011', f"graphs_data/{dataset_name}/full_Varshney_adjacency_matrix.png")

    # Comparison with data from 'Connectomes across development reveal principles of brain maturation'
    # https://www.nature.com/articles/s41586-021-03778-4
    file_path = '/groups/saalfeld/home/allierc/signaling/Celegans/Zhen_2021/41586_2021_3778_MOESM4_ESM.xlsx'
    sheet_name = 'Dataset7'
    Zhen_neuron_names = pd.read_excel(file_path, sheet_name=sheet_name, usecols='C', skiprows=4, nrows=224, header=None)
    Zhen_neuron_names = Zhen_neuron_names.squeeze()  # convert to Series for convenience
    Zhen_matrix = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=4, nrows=224, usecols='D:GA', header=None)
    # Zhen_matrix = Zhen_matrix.T
    Zhen_matrix_7 = torch.tensor(np.array(Zhen_matrix), dtype=torch.float32, device=device)
    sheet_name = 'Dataset8'
    Zhen_matrix = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=4, nrows=224, usecols='D:GA', header=None)
    # Zhen_matrix = Zhen_matrix.T
    Zhen_matrix_8 = torch.tensor(np.array(Zhen_matrix), dtype=torch.float32, device=device)
    Zhen_matrix = Zhen_matrix_7 + Zhen_matrix_8  # combine both datasets
    map_Zhen_matrix , index = map_matrix(all_neuron_list, Zhen_neuron_names, Zhen_matrix_7)
    # plot_worm_adjacency_matrix(to_numpy(map_Zhen_matrix), all_neuron_list, 'adjacency matrix Mei Zhen 2021 (7)', f"graphs_data/{dataset_name}/full_Zhen_adjacency_matrix_7.png")
    map_Zhen_matrix , index = map_matrix(all_neuron_list, Zhen_neuron_names, Zhen_matrix_8)
    # plot_worm_adjacency_matrix(to_numpy(map_Zhen_matrix), all_neuron_list, 'adjacency matrix Mei Zhen 2021 (8)', f"graphs_data/{dataset_name}/full_Zhen_adjacency_matrix_8.png")

    print('generate mask ...')

    mask_matrix = ((map_Zhen_matrix>0) | (map_Varshney_matrix>0) | (map_Kaiser_matrix>0) | (map_Cook_matrix>0) | (map_Turuga_matrix>0)) * 1.0
    torch.save(mask_matrix, f'./graphs_data/{dataset_name}/adjacency.pt')
    print (f'filling factor {torch.sum(mask_matrix)/mask_matrix.shape[0]**2:0.3f}')
    # zero_rows = torch.all(mask_matrix == 0, dim=1)
    # zero_columns = torch.all(mask_matrix == 0, dim=0)
    # mask_matrix[zero_rows] = 1
    # mask_matrix[:, zero_columns] = 1

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    sns.heatmap(to_numpy(mask_matrix), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046})
    ax.set_xticks(range(len(all_neuron_list)))
    ax.set_xticklabels(all_neuron_list, fontsize=6, rotation=90)
    ax.set_yticks(range(len(all_neuron_list)))
    ax.set_yticklabels(all_neuron_list, fontsize=6)
    plt.title('mask', fontsize=18)
    plt.xlabel('pre Neurons', fontsize=18)
    plt.ylabel('post Neurons', fontsize=18)
    plt.tight_layout()
    plt.savefig(f"graphs_data/{dataset_name}/mask_adjacency_matrix.png", dpi=170)
    plt.close()

    # print('generate partial adjacency matrices ...')
    # # generate partial adjacency matrices for activity neurons
    # map_Cook_matrix_chem , index = map_matrix(activity_neuron_list, Cook_neuron_chem_names, Cook_matrix_chem)
    # map_Cook_matrix_elec , index = map_matrix(activity_neuron_list, Cook_neuron_elec_names, Cook_matrix_elec)
    # map_Cook_matrix = map_Cook_matrix_chem + map_Cook_matrix_elec
    # plot_worm_adjacency_matrix(to_numpy(map_Cook_matrix), activity_neuron_list, 'partial adjacency matrix Cook 2019', f"graphs_data/{dataset_name}/partial_Cook_adjacency_matrix.png")
    #
    # map_Varshney_matrix , index = map_matrix(activity_neuron_list, Varshney_neuron_names, Varshney_matrix)
    # plot_worm_adjacency_matrix(to_numpy(map_Varshney_matrix), activity_neuron_list, 'partial adjacency matrix Varshney 2011', f"graphs_data/{dataset_name}/partial_Varshney_adjacency_matrix.png")
    #
    # map_Zhen_matrix_7 , index = map_matrix(activity_neuron_list, Zhen_neuron_names, Zhen_matrix_7)
    # plot_worm_adjacency_matrix(to_numpy(map_Zhen_matrix7), activity_neuron_list, 'partial adjacency matrix Mei Zhen 2021', f"graphs_data/{dataset_name}/partial_Zhen_adjacency_matrix_7.png")
    # map_Zhen_matrix_8 , index = map_matrix(activity_neuron_list, Zhen_neuron_names, Zhen_matrix_8)
    # plot_worm_adjacency_matrix(to_numpy(map_Zhen_matrix8), activity_neuron_list, 'partial adjacency matrix Mei Zhen 2021', f"graphs_data/{dataset_name}/partial_Zhen_adjacency_matrix_8.png")
    #
    # map_Kaiser_matrix , index = map_matrix(activity_neuron_list, Kaiser_neuron_names, Kaiser_matrix)
    # plot_worm_adjacency_matrix(to_numpy(map_Kaiser_matrix), activity_neuron_list, 'partial adjacency matrix Kaiser 2006', f"graphs_data/{dataset_name}/partial full_Kaiser_adjacency_matrix.png")

    sensory_neuron_list = Cook_neuron_chem_names[20:103]
    with open(f"graphs_data/{dataset_name}/sensory_neuron_list.json", "w") as f:
        json.dump(sensory_neuron_list, f)
    inter_neuron_list = Cook_neuron_chem_names[103:184]
    with open(f"graphs_data/{dataset_name}/inter_neuron_list.json", "w") as f:
        json.dump(inter_neuron_list, f)
    motor_neuron_list = Cook_neuron_chem_names[184:292]
    with open(f"graphs_data/{dataset_name}/motor_neuron_list.json", "w") as f:
        json.dump(motor_neuron_list, f)
    larynx_neuron_list = Cook_neuron_chem_names[0:20]
    with open(f"graphs_data/{dataset_name}/larynx_neuron_list.json", "w") as f:
        json.dump(larynx_neuron_list, f)
    map_larynx_matrix , index = map_matrix(larynx_neuron_list, all_neuron_list, mask_matrix)

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    sns.heatmap(to_numpy(map_larynx_matrix), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046})
    ax.set_xticks(range(len(larynx_neuron_list)))
    ax.set_xticklabels(larynx_neuron_list, fontsize=14, rotation=90)
    ax.set_yticks(range(len(larynx_neuron_list)))
    ax.set_yticklabels(larynx_neuron_list, fontsize=14)
    plt.title('larynx adjacency', fontsize=18)
    plt.xlabel('postsynaptic', fontsize=18)
    plt.ylabel('presynaptic', fontsize=18)
    plt.tight_layout()
    plt.savefig(f"graphs_data/{dataset_name}/mask_larynx_adjacency_matrix.png", dpi=170)
    plt.close()

    # generate data for GNN training
    # create fully connected edges
    n_neurons = len(all_neuron_list)
    edge_index, edge_attr = dense_to_sparse(torch.ones((n_neurons)) - torch.eye(n_neurons))
    torch.save(edge_index.to(device), f'./graphs_data/{dataset_name}/edge_index.pt')
    activity_idx = []
    for k in range(len(activity_neuron_list)):
        neuron_OI = get_neuron_index(activity_neuron_list[k], all_neuron_list)
        activity_idx.append(neuron_OI)
    activity_idx = np.array(activity_idx)

    xc, yc = get_equidistant_points(n_points=n_neurons)
    pos = torch.tensor(np.stack((xc, yc), axis=1), dtype=torch.float32, device=device) / 2
    perm = torch.randperm(pos.size(0))
    X1 = to_numpy(pos[perm])

    # type 0 larynx
    # type 1 sensory
    # type 2 inter
    # type 3 motor
    # type 4 other
    T1 = np.ones((n_neurons, 1)) * 4
    type_dict = {}
    for name in larynx_neuron_list:
        type_dict[name] = 0
    for name in sensory_neuron_list:
        type_dict[name] = 1
    for name in inter_neuron_list:
        type_dict[name] = 2
    for name in motor_neuron_list:
        type_dict[name] = 3

    # Default to type 4 ("other") if not found
    T1[activity_idx] = np.array([[type_dict.get(name, 4)] for name in activity_neuron_list])

    if train_config.denoiser:
        print('denoise data with gaussian_smooth')
        def gaussian_smooth(data, sigma=2.0):
            kernel_size = int(6 * sigma + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1

            # Create 1D Gaussian kernel on the same device as data
            device = data.device  # Get the device of input data
            x = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
            kernel = torch.exp(-0.5 * (x / sigma) ** 2)
            kernel = kernel / kernel.sum()

            # Reshape data: (neurons, time) -> (neurons, 1, time) for conv1d
            data_reshaped = data.unsqueeze(1)  # (189, 1, 960)

            # Apply padding
            data_padded = F.pad(data_reshaped, (kernel_size // 2, kernel_size // 2), mode='reflect')

            # Apply convolution - kernel needs to be (out_channels, in_channels, kernel_size)
            kernel_reshaped = kernel.view(1, 1, -1)  # (1, 1, kernel_size)

            # Convolve each neuron independently
            smoothed = F.conv1d(data_padded, kernel_reshaped, padding=0)

            return smoothed.squeeze(1)  # Remove the channel dimension: (189, 960)
        for run in trange(n_runs):

            activity = torch.tensor(activity_worm[run, :, :], dtype=torch.float32, device=device)
            activity_filtered = gaussian_smooth(activity, sigma=2.0)

            # Plotting code
            fig, axes = plt.subplots(4, 1, figsize=(12, 8))
            neurons_to_plot = [50, 60, 80, 100]

            for i, neuron_idx in enumerate(neurons_to_plot):
                axes[i].plot(to_numpy(activity[neuron_idx]), 'gray', alpha=0.7, label='Original')
                axes[i].plot(to_numpy(activity_filtered[neuron_idx]), 'blue', label='Denoised')
                axes[i].set_title(f'Neuron {neuron_idx}')
                axes[i].legend()

            plt.tight_layout()
            plt.savefig(f"graphs_data/{dataset_name}/Fig/Denoise/plot_{run}.png")
            plt.close()

            activity_worm[run, :, :] = to_numpy(activity_filtered)



    for run in range(config.training.n_runs):

        x_list = []
        y_list = []

        for it in trange(0, n_frames-2):
            x = np.zeros((n_neurons, 13))
            x[:, 0] = np.arange(n_neurons)
            x[:, 1:3] = X1
            x[:, 5:6] = T1
            x[:, 6] = 6
            x[activity_idx, 6] = activity_worm[run,:,it]
            x[:, 10:13] = odor_worms[run,:,it]
            x_list.append(x)

            y = (activity_worm[run,:,it+1]- activity_worm[run,:,it]) / delta_t
            y_list.append(y)

            if visualize & (run == 0) & (it % 2 == 0) & (it >= 0):
                plt.style.use('dark_background')

                plt.figure(figsize=(10, 10))
                plt.axis('off')
                values = x[:, 6]
                normed_vals = (values - 4) / (8 - 4)  # (min=4, max=8)

                black_to_green = LinearSegmentedColormap.from_list('black_green', ['black', 'green'])
                black_to_yellow = LinearSegmentedColormap.from_list('black_yellow', ['black', 'yellow'])

                plt.scatter(X1[:, 0], X1[:, 1], s=700, c=normed_vals, cmap=black_to_green)

                plt.scatter(-0.45, 0.5, s=700, c=x[0, 10] + 0.1, cmap= black_to_yellow, vmin=0,vmax=1)
                plt.scatter(-0.4, 0.5, s=700, c=x[0, 11] + 0.1, cmap= black_to_yellow, vmin=0,vmax=1)
                plt.scatter(-0.35, 0.5, s=700, c=x[0, 12] + 0.1, cmap= black_to_yellow, vmin=0,vmax=1)

                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(f"graphs_data/{dataset_name}/Fig/Fig/Fig_{run}_{it:03d}.tif", dpi=80)
                plt.close()

        x_list = np.array(x_list)
        y_list = np.array(y_list)
        np.save(f'graphs_data/{dataset_name}/x_list_{run}.npy', x_list)
        np.save(f'graphs_data/{dataset_name}/y_list_{run}.npy', y_list)

        activity = torch.tensor(x_list[:, :, 6:7], device=device)
        activity = activity.squeeze().t().cpu().numpy()

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(221)
        plt.imshow(activity, aspect='auto', vmin =0, vmax=8, cmap='viridis')
        plt.title(f'dataset {idata}', fontsize=18)
        plt.xlabel('time', fontsize=18)
        plt.ylabel('neurons', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        ax = fig.add_subplot(222)
        plt.title(f'missing data', fontsize=18)
        test_im = activity * 0
        pos = np.argwhere(activity == 6)
        test_im[pos[:, 0], pos[:, 1]] = 1
        pos = np.argwhere(np.isnan(activity))
        test_im[pos[:, 0], pos[:, 1]] = 2
        pos = np.argwhere(np.isinf(activity))
        test_im[pos[:, 0], pos[:, 1]] = 3
        plt.imshow(test_im[:,500:], aspect='auto',vmin =0, vmax=3, cmap='viridis')
        plt.xticks([])
        plt.yticks([])
        ax = fig.add_subplot(223)
        plt.imshow(odor_worms[idata], aspect='auto', vmin =0, vmax=1, cmap='viridis', interpolation='nearest')
        plt.xlabel('time', fontsize=18)
        plt.ylabel('odor', fontsize=18)
        plt.title(f'odor stimuli', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(f"graphs_data/{dataset_name}/Fig/Kinograph/Fig_{run}.tif", dpi=80)  # 170.7)
        plt.close()


def load_zebrafish_data(config, device=None, visualize=None, step=None, cmap=None, style=None):
    data_folder_name = config.data_folder_name
    dataset_name = config.dataset

    simulation_config = config.simulation
    train_config = config.training

    n_frames = simulation_config.n_frames
    n_neurons = simulation_config.n_neurons

    visual_input_type = simulation_config.visual_input_type

    delta_t = simulation_config.delta_t
    delta_x = 0.406 # in microns
    delta_y = 0.406 # in microns
    delta_z = 4 # in microns

    cmap = CustomColorMap(config=config)


    if 'black' in style:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')

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
        -1: "none"  # no condition / padding
    }

    if visual_input_type=='':
        print("using all visual input types")
    else:
        print(f"using visual input type: {visual_input_type}")


    print(f"loading zebrafish data from {data_folder_name} for dataset {dataset_name}...")

    conditions = np.load(data_folder_name+'/conditions.npy', allow_pickle=True)
    print(f"conditions shape: {conditions.shape}, sample: {conditions[0]}")
    traces = np.load(data_folder_name+'/traces.npy', allow_pickle=True)
    print(f"traces shape: {traces.shape}, sample: {traces[0][:5]} \n\n")
    positions = np.load(data_folder_name+'/mapped_positions.npy', allow_pickle=True)
    print(f"positions shape: {positions.shape}, sample: {positions[0]}")

    positions = positions / 1000
    positions_swapped = positions[:, [1, 0, 2]]
    positions_swapped[:, 0] *= -1
    positions[:,0] *= delta_x
    positions[:,1] *= delta_y
    positions[:,2] *= delta_z  

    print(f"min position: {np.min(positions, axis=0)}  max position: {np.max(positions, axis=0)}, ")
    print(f"number of frames: {n_frames}, number of neurons: {n_neurons}")


    path = "/groups/saalfeld/home/allierc/signaling/Zapbench/zapbench_numpy/stimuli_and_ephys.10chFlt"



    # memory-map the file (fast, no copy)
    mm = np.memmap(path, dtype=np.float32, mode='r')
    # reshape to (10, T) with T = number of time samples
    if mm.size % 10 != 0:
        raise ValueError("File size not divisible by 10 channels!")
    stim_ephys = mm.reshape(-1, 10).T   # shape: (10, T)

    print("shape:", stim_ephys.shape)   # (10, T_highres)
    print("dtype:", stim_ephys.dtype)

    # downsample to match imaging rate

    down_sampling = stim_ephys.shape[1] // n_frames

    stim_ephys = stim_ephys[:, ::down_sampling]  # shape: (10, T_highres/20)
    print("downsampled shape:", stim_ephys.shape)
    print("downsampled dtype:", stim_ephys.dtype)

    np.save(f'/groups/saalfeld/home/allierc/signaling/Zapbench/zapbench_numpy/stim_ephys.npy', stim_ephys)




    folder = f'./graphs_data/{dataset_name}/'
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f'./graphs_data/{dataset_name}/Fig/', exist_ok=True)

    n_neurons = positions.shape[0]
    x = np.zeros((n_neurons, 12))
    x[:, 0] = np.arange(n_neurons)
    x[:, 1:4] = positions[:, 0:3]
    x[:, 5:6] = 0
    x[:, 6] = traces[0]
    x[:, 7] = conditions[0]
    x[:, 8:11] = 0

    x_list = []
    y_list = []

    for n in trange(n_frames):
        x[:, 0] = np.arange(n_neurons)
        x[:, 1:4] = positions[:, 0:3]
        x[:, 6] = traces[n]
        x[:, 7] = conditions[n]

        if (visual_input_type=='') | (visual_input_type==condition_names[int(conditions[n])]):
            x_list.append(x.copy())
            
        if n == 0: 
            vmin, vmax = np.percentile(traces[n], [2, 98])

        if n < n_frames - 1:
            y = (traces[n+1]- traces[n]) / delta_t
            if (visual_input_type=='') | (visual_input_type==condition_names[int(conditions[n])]):
                y_list.append(y)

        if (visualize) & (n % step == 0):
            if (visual_input_type=='') | (visual_input_type==condition_names[int(conditions[n])]):
                fig  = plt.figure(figsize=(17, 10))
                plt.axis('off')
                plt.scatter(x[:,1], x[:,2], s=5, c=x[:,6], vmin=vmin, vmax=vmax, cmap='plasma')
                label = f"frame: {n} \n{condition_names[int(conditions[n])]}  "
                plt.text(0.05, 0.45, label, fontsize=24, color='white')
                plt.xlim([0., 0.8])
                plt.ylim([0, 0.51])
                plt.tight_layout()
                plt.savefig(f"graphs_data/{dataset_name}/Fig/xy_{n:06d}.png", dpi=40)
                plt.close()

    print('saving data...')
    x_list = np.array(x_list)
    y_list = np.array(y_list)
    run = 0
    np.save(f'graphs_data/{dataset_name}/x_list_{run}.npy', x_list)
    np.save(f'graphs_data/{dataset_name}/y_list_{run}.npy', y_list)
    print(f"x_list shape: {x_list.shape}")
    print(f"y_list shape: {y_list.shape}")
    print('data saved.')

    print(f'length of the dataset: {len(x_list)}')

    if visualize == -1:
        # sanity checks

        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        if traces.shape[1] != positions.shape[0]:
            raise ValueError(f"trace0 length ({traces.shape[0]}) must match positions.shape[0] ({positions.shape[0]})")

        for id_fig in tqdm(range(0,n_frames)):

            activity = np.asarray(traces[id_fig], float)
            finite = np.isfinite(activity)
            if not finite.all():
                med = np.nanmedian(activity[finite])
                activity = np.where(finite, activity, med)

            # robust color limits (ignore extremes)
            if id_fig == 0: 
                vmin, vmax = np.percentile(activity, [2, 98])

            cloud = pv.PolyData(positions_swapped)
            cloud["activity"] = activity
            plotter = pv.Plotter(off_screen=True, window_size=(500, 300))
            plotter.set_background('black')
            plotter.add_points(
                cloud,
                scalars="activity",
                cmap="plasma",          # plasma LUT
                clim=(vmin, vmax),      # << fix: stop auto-rescaling
                render_points_as_spheres=True,
                point_size=5,           # << fix: make points visible (try 2–5)
                opacity=0.75,            # << fix: no transparency on black bg
                lighting=False,         # << fix: avoid dark shading on tiny spheres
                show_scalar_bar=False,   # helpful to verify range
            )
            condition_id = conditions[id_fig]
            label = f"frame: {id_fig} \n{condition_names[int(condition_id)]}  "
            plotter.add_text(   
                label,
                position="upper_left",   # or "upper_right", "lower_left", "lower_right"
                font_size=12,
                color="white",
            )
            plotter.view_vector((0, 0, 1))
            try:
                plotter.enable_anti_aliasing()
            except Exception:
                pass
            plotter.camera.zoom(1.6)
            num = f"{id_fig:06}"
            plotter.screenshot(f'./graphs_data/{dataset_name}/Fig/xy_{num}.png')
            plotter.close()
        
            plotter = pv.Plotter(off_screen=True, window_size=(800, 500))
            plotter.set_background('black')
            plotter.add_points(
                cloud,
                scalars="activity",
                cmap="plasma",          # plasma LUT
                clim=(vmin, vmax),      # << fix: stop auto-rescaling
                render_points_as_spheres=True,
                point_size=5,           # << fix: make points visible (try 2–5)
                opacity=0.75,            # << fix: no transparency on black bg
                lighting=False,         # << fix: avoid dark shading on tiny spheres
                show_scalar_bar=False,   # helpful to verify range
            )
            plotter.view_vector((1, 0, 0))
            try:
                plotter.enable_anti_aliasing()
            except Exception:
                pass
            plotter.camera.zoom(1.6)

            num = f"{id_fig:06}"
            plotter.screenshot(f'./graphs_data/{dataset_name}/Fig/xz_{num}.png')
            plotter.close()


def ensure_local_path_exists(path):
    """
    Ensure that the local path exists. If it doesn't, create the directory structure.

    :param path: The path to be checked and created if necessary.
    :return: The absolute path of the created directory.
    """

    os.makedirs(path, exist_ok=True)
    return os.path.join(os.getcwd(), path)


@dataclass
class CsvDescriptor:
    """A class to describe the location of data in a dataset as a column of a CSV file."""
    filename: str
    column_name: str
    type: np.dtype
    unit: Unit


def load_csv_from_descriptors(
        column_descriptors: Dict[str, CsvDescriptor],
        **kwargs
) -> pd.DataFrame:
    """
    Load data from a CSV file based on a set of column descriptors.

    :param column_descriptors: A dictionary mapping field names to CsvDescriptors.
    :param kwargs: Additional keyword arguments to pass to pd.read_csv.
    :return: A pandas DataFrame containing the loaded data.
    """
    different_files = set(descriptor.filename for descriptor in column_descriptors.values())
    columns = []

    for file in different_files:
        dtypes = {descriptor.column_name: descriptor.type for descriptor in column_descriptors.values()
                  if descriptor.filename == file}
        print(f"Loading data from '{file}':")
        for column_name, dtype in dtypes.items():
            print(f"  - column {column_name} as {dtype}")
        columns.append(pd.read_csv(file, dtype=dtypes, usecols=list(dtypes.keys()), **kwargs))

    data = pd.concat(columns, axis='columns')
    data.rename(columns={descriptor.column_name: name for name, descriptor in column_descriptors.items()}, inplace=True)

    return data

