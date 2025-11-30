
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import subprocess
import torch
import torch_geometric.data as data
import xarray as xr
from NeuralGraph.generators import PDE_N2, PDE_N3, PDE_N4, PDE_N5, PDE_N6, PDE_N7, PDE_N11
from NeuralGraph.utils import choose_boundary_values, get_equidistant_points, to_numpy, large_tensor_nonzero
from scipy import stats
from scipy.spatial import Delaunay
from time import sleep
from tifffile import imread
from torch_geometric.utils import get_mesh_laplacian, dense_to_sparse
from torch_geometric.utils.convert import to_networkx
from tqdm import trange
import seaborn as sns

# Optional imports
try:
    from fa2_modified import ForceAtlas2
except ImportError:
    ForceAtlas2 = None


def choose_model(config=[], W=[], device=[]):
    model_signal_name = config.graph_model.signal_model_name
    aggr_type = config.graph_model.aggr_type
    short_term_plasticity_mode = config.simulation.short_term_plasticity_mode

    bc_pos, bc_dpos = choose_boundary_values(config.simulation.boundary)

    params = config.simulation.params
    p = torch.tensor(params, dtype=torch.float32, device=device).squeeze()


    match config.simulation.phi:
        case 'tanh':
            phi=torch.tanh
        case 'relu':
            phi=torch.relu
        case 'sigmoid':
            phi=torch.sigmoid
        case _:
            phi=torch.sigmoid

    match model_signal_name:
        case 'PDE_N2':
            model = PDE_N2(aggr_type=aggr_type, p=p, W=W, phi=phi)
        case 'PDE_N3':
            model = PDE_N3(aggr_type=aggr_type, p=p, W=W, phi=phi)
        case 'PDE_N4':
            model = PDE_N4(aggr_type=aggr_type, p=p, W=W, phi=phi)
        case 'PDE_N5':
            model = PDE_N5(aggr_type=aggr_type, p=p, W=W, phi=phi)
        case 'PDE_N6':
            model = PDE_N6(aggr_type=aggr_type, p=p, W=W, phi=phi, short_term_plasticity_mode = short_term_plasticity_mode)
        case 'PDE_N7':
            model = PDE_N7(aggr_type=aggr_type, p=p, W=W, phi=phi, short_term_plasticity_mode = short_term_plasticity_mode)
        case 'PDE_N11':
            func_p = config.simulation.func_params
            model = PDE_N11(config=config, aggr_type=aggr_type, p=p, W=W, phi=phi, func_p=func_p, device=device)



    return model, bc_pos, bc_dpos


def initialize_random_values(n, device):
    return torch.ones(n, 1, device=device) + torch.rand(n, 1, device=device)


def init_neurons(config=[], scenario='none', ratio=1, device=[]):
    simulation_config = config.simulation
    n_neurons = simulation_config.n_neurons * ratio
    n_neuron_types = simulation_config.n_neuron_types
    dimension = simulation_config.dimension

    dpos_init = simulation_config.dpos_init


    xc, yc = get_equidistant_points(n_points=n_neurons)
    pos = torch.tensor(np.stack((xc, yc), axis=1), dtype=torch.float32, device=device) / 2
    perm = torch.randperm(pos.size(0))
    pos = pos[perm]

    dpos = dpos_init * torch.randn((n_neurons, dimension), device=device)
    dpos = torch.clamp(dpos, min=-torch.std(dpos), max=+torch.std(dpos))

    type = torch.zeros(int(n_neurons / n_neuron_types), device=device)

    for n in range(1, n_neuron_types):
        type = torch.cat((type, n * torch.ones(int(n_neurons / n_neuron_types), device=device)), 0)
    if type.shape[0] < n_neurons:
        type = torch.cat((type, n * torch.ones(n_neurons - type.shape[0], device=device)), 0)

    if (config.graph_model.signal_model_name == 'PDE_N6') | (config.graph_model.signal_model_name == 'PDE_N7'):
        features = torch.cat((torch.rand((n_neurons, 1), device=device), 0.1 * torch.randn((n_neurons, 1), device=device),
                              torch.ones((n_neurons, 1), device=device), torch.zeros((n_neurons, 1), device=device)), 1)
    elif 'excitation_single' in config.graph_model.field_type:
        features = torch.zeros((n_neurons, 2), device=device)
    else:
        features = torch.cat((torch.randn((n_neurons, 1), device=device) * 5 , 0.1 * torch.randn((n_neurons, 1), device=device)), 1)

    type = type[:, None]
    particle_id = torch.arange(n_neurons, device=device)
    particle_id = particle_id[:, None]
    age = torch.zeros((n_neurons,1), device=device)

    return pos, dpos, type, features, age, particle_id


def random_rotation_matrix(device='cpu'):
    # Random Euler angles
    roll = torch.rand(1, device=device) * 2 * torch.pi
    pitch = torch.rand(1, device=device) * 2 * torch.pi
    yaw = torch.rand(1, device=device) * 2 * torch.pi

    cos_r, sin_r = torch.cos(roll), torch.sin(roll)
    cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)

    # Rotation matrices around each axis
    R_x = torch.tensor([
        [1, 0, 0],
        [0, cos_r, -sin_r],
        [0, sin_r, cos_r]
    ], device=device).squeeze()

    R_y = torch.tensor([
        [cos_p, 0, sin_p],
        [0, 1, 0],
        [-sin_p, 0, cos_p]
    ], device=device).squeeze()

    R_z = torch.tensor([
        [cos_y, -sin_y, 0],
        [sin_y, cos_y, 0],
        [0, 0, 1]
    ], device=device).squeeze()

    # Combined rotation matrix: R = R_z * R_y * R_x
    R = R_z @ R_y @ R_x
    return R


def get_index(n_neurons, n_neuron_types):
    index_particles = []
    for n in range(n_neuron_types):
        index_particles.append(
            np.arange((n_neurons // n_neuron_types) * n, (n_neurons // n_neuron_types) * (n + 1)))
    return index_particles


def get_time_series(x_list, cell_id, feature):

    match feature:
        case 'velocity_x':
            feature = 3
        case 'velocity_y':
            feature = 4
        case 'type' | 'state':
            feature = 5
        case 'age':
            feature = 8
        case 'mass':
            feature = 10

        case _:  # default
            feature = 0

    time_series = []
    for it in range(len(x_list)):
        x = x_list[it].clone().detach()
        pos_cell = torch.argwhere(x[:, 0] == cell_id)
        if len(pos_cell) > 0:
            time_series.append(x[pos_cell, feature].squeeze())
        else:
            time_series.append(torch.tensor([0.0]))

    return to_numpy(torch.stack(time_series))


def init_mesh(config, device):

    simulation_config = config.simulation
    model_config = config.graph_model

    n_nodes = simulation_config.n_nodes
    n_neurons = simulation_config.n_neurons
    node_value_map = simulation_config.node_value_map
    field_grid = model_config.field_grid
    max_radius = simulation_config.max_radius

    n_nodes_per_axis = int(np.sqrt(n_nodes))
    xs = torch.linspace(1 / (2 * n_nodes_per_axis), 1 - 1 / (2 * n_nodes_per_axis), steps=n_nodes_per_axis)
    ys = torch.linspace(1 / (2 * n_nodes_per_axis), 1 - 1 / (2 * n_nodes_per_axis), steps=n_nodes_per_axis)
    x_mesh, y_mesh = torch.meshgrid(xs, ys, indexing='xy')
    x_mesh = torch.reshape(x_mesh, (n_nodes_per_axis ** 2, 1))
    y_mesh = torch.reshape(y_mesh, (n_nodes_per_axis ** 2, 1))
    mesh_size = 1 / n_nodes_per_axis
    pos_mesh = torch.zeros((n_nodes, 2), device=device)
    pos_mesh[0:n_nodes, 0:1] = x_mesh[0:n_nodes]
    pos_mesh[0:n_nodes, 1:2] = y_mesh[0:n_nodes]

    i0 = imread(f'graphs_data/{node_value_map}')
    if len(i0.shape) == 2:
        # i0 = i0[0,:, :]
        i0 = np.flipud(i0)
        values = i0[(to_numpy(pos_mesh[:, 1]) * 255).astype(int), (to_numpy(pos_mesh[:, 0]) * 255).astype(int)]

    mask_mesh = (x_mesh > torch.min(x_mesh) + 0.02) & (x_mesh < torch.max(x_mesh) - 0.02) & (y_mesh > torch.min(y_mesh) + 0.02) & (y_mesh < torch.max(y_mesh) - 0.02)

    if 'grid' in field_grid:
        pos_mesh = pos_mesh
    else:
        if 'pattern_Null.tif' in simulation_config.node_value_map:
            pos_mesh = pos_mesh + torch.randn(n_nodes, 2, device=device) * mesh_size / 24
        else:
            pos_mesh = pos_mesh + torch.randn(n_nodes, 2, device=device) * mesh_size / 8

    match config.graph_model.mesh_model_name:
        case 'RD_Gray_Scott_Mesh':
            node_value = torch.zeros((n_nodes, 2), device=device)
            node_value[:, 0] -= 0.5 * torch.tensor(values / 255, device=device)
            node_value[:, 1] = 0.25 * torch.tensor(values / 255, device=device)
        case 'RD_FitzHugh_Nagumo_Mesh':
            node_value = torch.zeros((n_nodes, 2), device=device) + torch.rand((n_nodes, 2), device=device) * 0.1
        case 'RD_Mesh' | 'RD_Mesh2' | 'RD_Mesh3' :
            node_value = torch.rand((n_nodes, 3), device=device)
            s = torch.sum(node_value, dim=1)
            for k in range(3):
                node_value[:, k] = node_value[:, k] / s
        case 'DiffMesh' | 'WaveMesh' | 'Particle_Mesh_A' | 'Particle_Mesh_B' | 'WaveSmoothParticle':
            node_value = torch.zeros((n_nodes, 2), device=device)
            node_value[:, 0] = torch.tensor(values / 255 * 5000, device=device)
        case 'PDE_O_Mesh':
            node_value = torch.zeros((n_neurons, 5), device=device)
            node_value[0:n_neurons, 0:1] = x_mesh[0:n_neurons]
            node_value[0:n_neurons, 1:2] = y_mesh[0:n_neurons]
            node_value[0:n_neurons, 2:3] = torch.randn(n_neurons, 1, device=device) * 2 * np.pi  # theta
            node_value[0:n_neurons, 3:4] = torch.ones(n_neurons, 1, device=device) * np.pi / 200  # d_theta
            node_value[0:n_neurons, 4:5] = node_value[0:n_neurons, 3:4]  # d_theta0
            pos_mesh[:, 0] = node_value[:, 0] + (3 / 8) * mesh_size * torch.cos(node_value[:, 2])
            pos_mesh[:, 1] = node_value[:, 1] + (3 / 8) * mesh_size * torch.sin(node_value[:, 2])
        case '' :
            node_value = torch.zeros((n_nodes, 2), device=device)



    type_mesh = torch.zeros((n_nodes, 1), device=device)

    node_id_mesh = torch.arange(n_nodes, device=device)
    node_id_mesh = node_id_mesh[:, None]
    dpos_mesh = torch.zeros((n_nodes, 2), device=device)

    x_mesh = torch.concatenate((node_id_mesh.clone().detach(), pos_mesh.clone().detach(), dpos_mesh.clone().detach(),
                                type_mesh.clone().detach(), node_value.clone().detach()), 1)

    pos = to_numpy(x_mesh[:, 1:3])
    tri = Delaunay(pos, qhull_options='QJ')
    face = torch.from_numpy(tri.simplices)
    face_longest_edge = np.zeros((face.shape[0], 1))

    sleep(0.5)
    for k in trange(face.shape[0]):
        # compute edge distances
        x1 = pos[face[k, 0], :]
        x2 = pos[face[k, 1], :]
        x3 = pos[face[k, 2], :]
        a = np.sqrt(np.sum((x1 - x2) ** 2))
        b = np.sqrt(np.sum((x2 - x3) ** 2))
        c = np.sqrt(np.sum((x3 - x1) ** 2))
        A = np.max([a, b]) / np.min([a, b])
        B = np.max([a, c]) / np.min([a, c])
        C = np.max([c, b]) / np.min([c, b])
        face_longest_edge[k] = np.max([A, B, C])

    face_kept = np.argwhere(face_longest_edge < 5)
    face_kept = face_kept[:, 0]
    face = face[face_kept, :]
    face = face.t().contiguous()
    face = face.to(device, torch.long)

    pos_3d = torch.cat((x_mesh[:, 1:3], torch.ones((x_mesh.shape[0], 1), device=device)), dim=1)
    edge_index_mesh, edge_weight_mesh = get_mesh_laplacian(pos=pos_3d, face=face, normalization="None")
    edge_weight_mesh = edge_weight_mesh.to(dtype=torch.float32)
    mesh_data = {'mesh_pos': pos_3d, 'face': face, 'edge_index': edge_index_mesh, 'edge_weight': edge_weight_mesh,
                 'mask': mask_mesh, 'size': mesh_size}

    if (config.graph_model.particle_model_name == 'PDE_ParticleField_A')  | (config.graph_model.particle_model_name == 'PDE_ParticleField_B'):
        type_mesh = 0 * type_mesh

    a_mesh = torch.zeros_like(type_mesh)
    type_mesh = type_mesh.to(dtype=torch.float32)

    if 'Smooth' in config.graph_model.mesh_model_name:
        distance = torch.sum((pos_mesh[:, None, :] - pos_mesh[None, :, :]) ** 2, dim=2)
        adj_t = ((distance < max_radius ** 2) & (distance >= 0)).float() * 1
        mesh_data['edge_index'] = adj_t.nonzero().t().contiguous()


    return pos_mesh, dpos_mesh, type_mesh, node_value, a_mesh, node_id_mesh, mesh_data


def init_synapse_map(config, x, edge_attr_adjacency, edge_index, device):
    # TODO: This function appears to be dead code with missing parameters (dataset_name, N1, V1, T1, H1, A1)
    dataset = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index, edge_attr=edge_attr_adjacency)
    G = to_networkx(dataset, remove_self_loops=True, to_undirected=True)
    forceatlas2 = ForceAtlas2(
        # Behavior alternatives
        outboundAttractionDistribution=True,  # Dissuade hubs
        linLogMode=False,  # NOT IMPLEMENTED
        adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
        edgeWeightInfluence=1.0,

        # Performance
        jitterTolerance=1.0,  # Tolerance
        barnesHutOptimize=True,
        barnesHutTheta=1.2,
        multiThreaded=False,  # NOT IMPLEMENTED

        # Tuning
        scalingRatio=2.0,
        strongGravityMode=False,
        gravity=1.0,

        # Log
        verbose=True)

    positions = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=500)
    positions = np.array(list(positions.values()))
    X1 = torch.tensor(positions, dtype=torch.float32, device=device)
    X1 = X1 - torch.mean(X1, 0)

    # torch.save(X1, f'./graphs_data/graphs_{dataset_name}/X1.pt')  # TODO: dataset_name parameter missing

    # TODO: N1, V1, T1, H1, A1 parameters missing - this code appears to be incomplete
    # x = torch.concatenate((N1.clone().detach(), X1.clone().detach(), V1.clone().detach(), T1.clone().detach(),
    #                        H1.clone().detach(), A1.clone().detach()), 1)
    x = None  # Placeholder since original code is incomplete

    # pos = nx.spring_layout(G, weight='weight', seed=42, k=1)
    # for k,p in pos.items():
    #     X1[k,:] = torch.tensor([p[0],p[1]], device=device)
    
    
def init_connectivity(connectivity_file, connectivity_type, connectivity_distribution, connectivity_filling_factor, T1, n_neurons, n_neuron_types, dataset_name, device):

    if 'adjacency.pt' in connectivity_file:
        connectivity = torch.load(connectivity_file, map_location=device)
    elif 'mat' in connectivity_file:
        mat = scipy.io.loadmat(connectivity_file)
        connectivity = torch.tensor(mat['A'], device=device)
    elif 'zarr' in connectivity_file:
        print('loading zarr ...')
        dataset = xr.open_zarr(connectivity_file)
        trained_weights = dataset["trained"]  # alpha * sign * N
        print(f'weights {trained_weights.shape}')
        dataset["untrained"]  # sign * N
        values = trained_weights[0:n_neurons,0:n_neurons]
        values = np.array(values)
        values = values / np.max(values)
        connectivity = torch.tensor(values, dtype=torch.float32, device=device)
        values=[]
    elif 'tif' in connectivity_file:
        # TODO: constructRandomMatrices function not implemented
        raise NotImplementedError("constructRandomMatrices function not implemented for tif files")
        # connectivity = constructRandomMatrices(n_neurons=n_neurons, density=1.0, connectivity_mask=f"./graphs_data/{connectivity_file}" ,device=device)
        # n_neurons = connectivity.shape[0]
        # TODO: config parameter not passed to this function
        # config.simulation.n_neurons = n_neurons

    elif connectivity_type != 'none':
        if 'chaotic' in connectivity_type:
            # Chaotic network 
            connectivity = np.random.randn(n_neurons,n_neurons) * np.sqrt(1/n_neurons)
        elif 'ring attractor' in connectivity_type:
            # Ring attractor network 
            th = np.linspace(0, 2 * np.pi, n_neurons, endpoint=False)   # Preferred firing location (angle)
            J1 = 1.0
            J0 = 0.5
            connectivity = (J1 * np.cos(th[:, None] - th[None, :]) + J0) / n_neurons   # Synaptic weight matrix
        elif 'rank 1' in connectivity_type:
            # Rank 1 network 
            u1 = np.random.rand(n_neurons)
            u2 = np.random.rand(n_neurons)
            connectivity = np.outer(u1,u2)
        elif 'successor' in connectivity_type:
            # Successor Representation
            T = np.eye(n_neurons, k=1)
            gamma = 0.98
            connectivity = np.linalg.inv(np.eye(n_neurons) - gamma*T)
        elif 'null' in connectivity_type:
            connectivity = np.zeros((n_neurons, n_neurons))

        connectivity = torch.tensor(connectivity, dtype=torch.float32, device=device)
        # make sure diagonal is zeros
        i, j = torch.triu_indices(n_neurons, n_neurons, requires_grad=False, device=device)
        connectivity[i, i] = 0

    else:
        if 'Gaussian' in connectivity_distribution:
            connectivity = torch.randn((n_neurons, n_neurons), dtype=torch.float32, device=device)
            connectivity = connectivity / np.sqrt(n_neurons)
            print(f"Gaussian   1/sqrt(N)  {1/np.sqrt(n_neurons)}    std {torch.std(connectivity.flatten())}")
        elif 'Lorentz' in connectivity_distribution:
            s = np.random.standard_cauchy(n_neurons**2)
            s[(s < -25) | (s > 25)] = 0
            if n_neurons < 2000:
                s = s / n_neurons**0.7
            elif n_neurons <4000:
                s = s / n_neurons**0.675
            elif n_neurons < 8000:
                s = s / n_neurons**0.67
            elif n_neurons == 8000:
                s = s / n_neurons**0.66
            elif n_neurons > 8000:
                s = s / n_neurons**0.5
            print(f"Lorentz   1/sqrt(N)  {1/np.sqrt(n_neurons):0.3f}    std {np.std(s):0.3f}")

            connectivity = torch.tensor(s, dtype=torch.float32, device=device)
            connectivity = torch.reshape(connectivity, (n_neurons, n_neurons))
        elif 'uniform' in connectivity_distribution:
            connectivity = torch.rand((n_neurons, n_neurons), dtype=torch.float32, device=device)
            connectivity = connectivity - 0.5
        i, j = torch.triu_indices(n_neurons, n_neurons, requires_grad=False, device=device)
        connectivity[i, i] = 0

    if connectivity_filling_factor != 1:
        mask = torch.rand(connectivity.shape) > connectivity_filling_factor
        connectivity[mask] = 0
        mask = (connectivity != 0).float()

        # Calculate effective filling factor
        total_possible = connectivity.shape[0] * connectivity.shape[1]
        actual_connections = mask.sum().item()
        effective_filling_factor = actual_connections / total_possible

        print(f"target filling factor: {connectivity_filling_factor}")
        print(f"effective filling factor: {effective_filling_factor:.6f}")
        print(f"actual connections: {int(actual_connections)}/{total_possible}")

        if n_neurons > 10000:
            edge_index = large_tensor_nonzero(mask)
            print(f'edge_index {edge_index.shape}')
        else:
            edge_index = mask.nonzero().t().contiguous()

    else:
        adj_matrix = torch.ones((n_neurons)) - torch.eye(n_neurons)
        edge_index, edge_attr = dense_to_sparse(adj_matrix)
        mask = (adj_matrix != 0).float()

    if 'structured' in connectivity_distribution:
        parts = connectivity_distribution.split('_')
        float_value1 = float(parts[-2])  # repartition pos/neg
        float_value2 = float(parts[-1])  # filling factor

        matrix_sign = torch.tensor(stats.bernoulli(float_value1).rvs(n_neuron_types ** 2) * 2 - 1,
                                   dtype=torch.float32, device=device)
        matrix_sign = matrix_sign.reshape(n_neuron_types, n_neuron_types)

        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(to_numpy(connectivity), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046},
                         vmin=-0.1, vmax=0.1)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=32)
        plt.xticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
        plt.yticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
        plt.xticks(rotation=0)
        plt.subplot(2, 2, 1)
        ax = sns.heatmap(to_numpy(connectivity[0:20, 0:20]), cbar=False, center=0, square=True, cmap='bwr', vmin=-0.1,
                         vmax=0.1)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f'graphs_data/{dataset_name}/adjacency_0.png', dpi=300)
        plt.close()

        T1_ = to_numpy(T1.squeeze())
        xy_grid = np.stack(np.meshgrid(T1_, T1_), -1)
        connectivity = torch.abs(connectivity)
        T1_ = to_numpy(T1.squeeze())
        xy_grid = np.stack(np.meshgrid(T1_, T1_), -1)
        sign_matrix = matrix_sign[xy_grid[..., 0], xy_grid[..., 1]]
        connectivity *= sign_matrix

        plt.imshow(to_numpy(sign_matrix))
        plt.savefig(f"graphs_data/{dataset_name}/large_connectivity_sign.tif", dpi=130)
        plt.close()

        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(to_numpy(connectivity), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046},
                         vmin=-0.1, vmax=0.1)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=32)
        plt.xticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
        plt.yticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
        plt.xticks(rotation=0)
        plt.subplot(2, 2, 1)
        ax = sns.heatmap(to_numpy(connectivity[0:20, 0:20]), cbar=False, center=0, square=True, cmap='bwr', vmin=-0.1,
                         vmax=0.1)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f'graphs_data/{dataset_name}/adjacency_1.png', dpi=300)
        plt.close()

        flat_sign_matrix = sign_matrix.flatten()
        num_elements = len(flat_sign_matrix)
        num_ones = int(num_elements * float_value2)
        indices = np.random.choice(num_elements, num_ones, replace=False)
        flat_sign_matrix[:] = 0
        flat_sign_matrix[indices] = 1
        sign_matrix = flat_sign_matrix.reshape(sign_matrix.shape)

        connectivity *= sign_matrix

        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(to_numpy(connectivity), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046},
                         vmin=-0.1, vmax=0.1)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=32)
        plt.xticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
        plt.yticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
        plt.xticks(rotation=0)
        plt.subplot(2, 2, 1)
        ax = sns.heatmap(to_numpy(connectivity[0:20, 0:20]), cbar=False, center=0, square=True, cmap='bwr', vmin=-0.1,
                         vmax=0.1)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f'graphs_data/{dataset_name}/adjacency_2.png', dpi=300)
        plt.close()

        total_possible = connectivity.shape[0] * connectivity.shape[1]
        actual_connections = (connectivity != 0).sum().item()
        effective_filling_factor = actual_connections / total_possible

        print(f"target filling factor: {float_value2}")
        print(f"effective filling factor: {effective_filling_factor:.6f}")
        print(f"actual connections: {actual_connections}/{total_possible}")

    edge_index = edge_index.to(device=device)

    return edge_index, connectivity, mask


def generate_compressed_video_mp4(output_dir, run=0, framerate=10, output_name=None, crf=23, log_dir=None):
    """
    Generate a compressed video using ffmpeg's libx264 codec in MP4 format.
    Automatically handles odd dimensions by scaling to even dimensions.

    Parameters:
        output_dir (str): Path to directory containing Fig/Fig_*.png.
        run (int): Run index to use in filename pattern.
        framerate (int): Desired video framerate.
        output_name (str): Name of output .mp4 file.
        crf (int): Constant Rate Factor for quality (0-51, lower = better quality, 23 is default).
        log_dir (str): If provided, save mp4 to log_dir instead of output_dir.
    """

    fig_dir = os.path.join(output_dir, "Fig")
    input_pattern = os.path.join(fig_dir, f"Fig_{run}_%06d.png")

    # Save to log_dir if provided, otherwise to output_dir
    save_dir = log_dir if log_dir is not None else output_dir
    output_path = os.path.join(save_dir, f"{output_name}.mp4")

    # Video filter to ensure even dimensions (required for yuv420p)
    # This scales the video so both width and height are divisible by 2

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",  # Suppress verbose output
        "-framerate", str(framerate),
        "-i", input_pattern,
        "-vf", "scale='trunc(iw/2)*2:trunc(ih/2)*2'",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        output_path,
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"compressed video (libx264) saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating video: {e}")
    except FileNotFoundError:
        print("ffmpeg not found. Please install ffmpeg to generate videos.")


def plot_signal_loss(loss_dict, log_dir, epoch=None, Niter=None, debug=False,
                     current_loss=None, current_regul=None, total_loss=None,
                     total_loss_regul=None):
    """
    Plot stratified loss components over training iterations.

    Creates a two-panel figure showing loss and regularization terms in both
    linear and log scale. Saves to {log_dir}/tmp_training/loss.tif.

    Parameters:
    -----------
    loss_dict : dict
        Dictionary containing loss component lists with keys:
        - 'loss': Loss without regularization
        - 'regul_total': Total regularization loss
        - 'W_L1': W L1 sparsity penalty
        - 'W_L2': W L2 regularization penalty
        - 'edge_grad': Edge gradient penalty
        - 'phi_grad': Phi gradient penalty
        - 'edge_diff': Edge monotonicity penalty
        - 'edge_norm': Edge normalization
        - 'edge_weight': Edge MLP weight regularization
        - 'phi_weight': Phi MLP weight regularization
        - 'W_sign': W sign consistency penalty
    log_dir : str
        Directory to save the figure
    epoch : int, optional
        Current epoch number
    Niter : int, optional
        Number of iterations per epoch
    debug : bool, optional
        If True, print debug information about loss components
    current_loss : float, optional
        Current iteration total loss (for debug)
    current_regul : float, optional
        Current iteration regularization (for debug)
    total_loss : float, optional
        Accumulated total loss (for debug)
    total_loss_regul : float, optional
        Accumulated regularization loss (for debug)
    """
    if len(loss_dict['loss']) == 0:
        return

    # Debug output if requested
    if debug and current_loss is not None and current_regul is not None:
        current_pred_loss = current_loss - current_regul

        # Get current iteration component values (last element in each list)
        comp_sum = (loss_dict['W_L1'][-1] + loss_dict['W_L2'][-1] + loss_dict['edge_grad'][-1] +
                   loss_dict['phi_grad'][-1] + loss_dict['edge_diff'][-1] +
                   loss_dict['edge_norm'][-1] +
                   loss_dict['edge_weight'][-1] + loss_dict['phi_weight'][-1] +
                   loss_dict['W_sign'][-1])

        print(f"\n=== DEBUG Loss Components (Epoch {epoch}, Iter {Niter}) ===")
        print("Current iteration:")
        print(f"  loss.item() (total): {current_loss:.6f}")
        print(f"  regul_this_iter: {current_regul:.6f}")
        print(f"  prediction_loss (loss - regul): {current_pred_loss:.6f}")
        print("\nRegularization breakdown:")
        print(f"  W_L1: {loss_dict['W_L1'][-1]:.6f}")
        print(f"  W_L2: {loss_dict['W_L2'][-1]:.6f}")
        print(f"  W_sign: {loss_dict['W_sign'][-1]:.6f}")
        print(f"  edge_grad: {loss_dict['edge_grad'][-1]:.6f}")
        print(f"  phi_grad: {loss_dict['phi_grad'][-1]:.6f}")
        print(f"  edge_diff: {loss_dict['edge_diff'][-1]:.6f}")
        print(f"  edge_norm: {loss_dict['edge_norm'][-1]:.6f}")
        print(f"  edge_weight: {loss_dict['edge_weight'][-1]:.6f}")
        print(f"  phi_weight: {loss_dict['phi_weight'][-1]:.6f}")
        print(f"  Sum of components: {comp_sum:.6f}")
        if total_loss is not None and total_loss_regul is not None:
            print("\nAccumulated (for reference):")
            print(f"  total_loss (accumulated): {total_loss:.6f}")
            print(f"  total_loss_regul (accumulated): {total_loss_regul:.6f}")
        if current_loss > 0:
            print(f"\nRatio: regul / loss (current iter) = {current_regul / current_loss:.4f}")
        if current_pred_loss < 0:
            print("\n⚠️  WARNING: Negative prediction loss! regul > total loss")
        print("="*60)

    fig_loss, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Add epoch and iteration info as text annotation
    info_text = ""
    if epoch is not None:
        info_text += f"Epoch: {epoch}"
    if Niter is not None:
        if info_text:
            info_text += " | "
        info_text += f"Iterations/epoch: {Niter}"
    if info_text:
        fig_loss.suptitle(info_text, fontsize=20, y=0.995)

    # Linear scale
    ax1.plot(loss_dict['loss'], color='b', linewidth=2, label='Loss (no regul)', alpha=0.8)
    ax1.plot(loss_dict['regul_total'], color='k', linewidth=2, label='Total Regularization', alpha=0.8)
    ax1.plot(loss_dict['W_L1'], color='r', linewidth=1.5, label='W L1 Sparsity', alpha=0.7)
    ax1.plot(loss_dict['W_L2'], color='darkred', linewidth=1.5, label='W L2 Regul', alpha=0.7)
    ax1.plot(loss_dict['W_sign'], color='navy', linewidth=1.5, label='W Sign (Dale)', alpha=0.7)
    ax1.plot(loss_dict['edge_grad'], color='g', linewidth=1.5, label='Edge Gradient Penalty', alpha=0.7)
    ax1.plot(loss_dict['phi_grad'], color='m', linewidth=1.5, label='Phi Gradient Penalty', alpha=0.7)
    ax1.plot(loss_dict['edge_diff'], color='orange', linewidth=1.5, label='Edge Monotonicity', alpha=0.7)
    ax1.plot(loss_dict['edge_norm'], color='brown', linewidth=1.5, label='Edge Norm', alpha=0.7)
    ax1.plot(loss_dict['edge_weight'], color='pink', linewidth=1.5, label='Edge Weight Regul', alpha=0.7)
    ax1.plot(loss_dict['phi_weight'], color='lime', linewidth=1.5, label='Phi Weight Regul', alpha=0.7)
    ax1.set_xlabel('Iteration', fontsize=16)
    ax1.set_ylabel('Loss', fontsize=16)
    ax1.set_title('Loss vs Iteration (Linear Scale)', fontsize=18)
    ax1.legend(fontsize=10, loc='best', ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=14)

    # Log scale
    ax2.plot(loss_dict['loss'], color='b', linewidth=2, label='Loss (no regul)', alpha=0.8)
    ax2.plot(loss_dict['regul_total'], color='k', linewidth=2, label='Total Regularization', alpha=0.8)
    ax2.plot(loss_dict['W_L1'], color='r', linewidth=1.5, label='W L1 Sparsity', alpha=0.7)
    ax2.plot(loss_dict['W_L2'], color='darkred', linewidth=1.5, label='W L2 Regul', alpha=0.7)
    ax2.plot(loss_dict['W_sign'], color='navy', linewidth=1.5, label='W Sign (Dale)', alpha=0.7)
    ax2.plot(loss_dict['edge_grad'], color='g', linewidth=1.5, label='Edge Gradient Penalty', alpha=0.7)
    ax2.plot(loss_dict['phi_grad'], color='m', linewidth=1.5, label='Phi Gradient Penalty', alpha=0.7)
    ax2.plot(loss_dict['edge_diff'], color='orange', linewidth=1.5, label='Edge Monotonicity', alpha=0.7)
    ax2.plot(loss_dict['edge_norm'], color='brown', linewidth=1.5, label='Edge Norm', alpha=0.7)
    ax2.plot(loss_dict['edge_weight'], color='pink', linewidth=1.5, label='Edge Weight Regul', alpha=0.7)
    ax2.plot(loss_dict['phi_weight'], color='lime', linewidth=1.5, label='Phi Weight Regul', alpha=0.7)
    ax2.set_xlabel('Iteration', fontsize=16)
    ax2.set_ylabel('Loss', fontsize=16)
    ax2.set_yscale('log')
    ax2.set_title('Loss vs Iteration (Log Scale)', fontsize=18)
    ax2.legend(fontsize=10, loc='best', ncol=2)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.tick_params(labelsize=14)

    plt.tight_layout()
    plt.savefig(f'{log_dir}/tmp_training/loss.tif', dpi=150)
    plt.close()

