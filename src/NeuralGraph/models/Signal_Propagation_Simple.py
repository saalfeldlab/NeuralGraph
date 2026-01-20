import torch
import torch.nn as nn
import torch_geometric as pyg
from NeuralGraph.models.MLP import MLP


class Signal_Propagation_Simple(pyg.nn.MessagePassing):
    """
    Simple graph neural network for learning neural signal dynamics.

    This version uses FULL MATRIX MULTIPLICATION for PDE_N2 (like ParticleGraph),
    instead of sparse message passing over edge_index.

    Key difference from Signal_Propagation:
    - For PDE_N2: msg = torch.matmul(W * mask, lin_edge(u))
    - This allows learning connectivity even when edge_index only contains sparse edges

    Based on interaction networks (Battaglia et al., NeurIPS 2016).
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html

    learns the first derivative of neural activity (du/dt) using:
    - MLP0 (lin_edge): message function on edges
    - MLP1 (lin_phi): node update function
    - W: learnable connectivity matrix (n_neurons x n_neurons)
    - a: learnable node embeddings (n_neurons x embedding_dim)

    inputs
    ----------
    data : torch_geometric.data.Data
        x[:,0]: particle_id
        x[:,3]: u (neural activity)
        x[:,4]: external_input
    data_id : int
        dataset/trial index
    k : int
        frame index

    returns
    -------
    pred : torch.Tensor
        first derivative du/dt (n_neurons x 1)
    """

    def __init__(self, aggr_type=None, config=None, device=None, bc_dpos=None, projections=None):
        super(Signal_Propagation_Simple, self).__init__(aggr=aggr_type)

        simulation_config = config.simulation
        model_config = config.graph_model
        train_config = config.training

        self.device = device
        self.model = model_config.signal_model_name
        self.embedding_dim = model_config.embedding_dim
        self.n_neurons = simulation_config.n_neurons
        self.n_dataset = config.training.n_runs
        self.n_frames = simulation_config.n_frames
        self.MLP_activation = config.graph_model.MLP_activation

        self.input_size = model_config.input_size
        self.output_size = model_config.output_size
        self.hidden_dim = model_config.hidden_dim
        self.n_layers = model_config.n_layers

        self.n_layers_update = model_config.n_layers_update
        self.hidden_dim_update = model_config.hidden_dim_update
        self.input_size_update = model_config.input_size_update

        self.batch_size = config.training.batch_size
        self.update_type = model_config.update_type

        self.bc_dpos = bc_dpos
        self.adjacency_matrix = simulation_config.adjacency_matrix

        # MLPs
        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.n_layers,
                            hidden_size=self.hidden_dim, activation=self.MLP_activation, device=self.device)

        self.lin_phi = MLP(input_size=self.input_size_update, output_size=self.output_size, nlayers=self.n_layers_update,
                            hidden_size=self.hidden_dim_update, activation=self.MLP_activation, device=self.device)

        # Node embeddings
        if train_config.training_single_type:
            self.register_buffer('a', torch.ones((self.n_neurons, self.embedding_dim), device=self.device, requires_grad=False, dtype=torch.float32))
        else:
            self.a = nn.Parameter(torch.ones(self.n_neurons, self.embedding_dim, device=self.device, requires_grad=True, dtype=torch.float32))

        # Connectivity matrix W - initialized with randn like ParticleGraph
        W_init = torch.randn((int(self.n_neurons), int(self.n_neurons)), device=self.device, dtype=torch.float32)
        W_init.fill_diagonal_(0)
        self.W = nn.Parameter(W_init, requires_grad=True)

        # Mask for diagonal (no self-connections)
        self.register_buffer('mask', torch.ones((int(self.n_neurons), int(self.n_neurons)), requires_grad=False, dtype=torch.float32))
        self.mask.fill_diagonal_(0)


    def forward(self, data=[], data_id=[], k=[], return_all=False):
        self.return_all = return_all
        x, edge_index = data.x, data.edge_index

        self.data_id = data_id.squeeze().long().clone().detach()

        u = data.x[:, 3:4]

        particle_id = x[:, 0].long()
        embedding = self.a[particle_id, :]

        in_features = torch.cat([u, embedding], dim=1)

        # KEY DIFFERENCE: Use FULL matrix multiplication for PDE_N2
        # This is what ParticleGraph does - operates on full W matrix, not sparse edges
        if self.model == 'PDE_N2':
            msg = torch.matmul(self.W * self.mask, self.lin_edge(u))
        else:
            # For other models, use message passing
            msg = self.propagate(edge_index, u=u, embedding=embedding, data_id=self.data_id[:, None], W=self.W)

        pred = self.lin_phi(in_features) + msg

        if return_all:
            return pred, in_features
        else:
            return pred


    def message(self, edge_index_i, edge_index_j, u_i, u_j, embedding_i, embedding_j, data_id_i, W):
        """Message function for non-PDE_N2 models (sparse message passing)."""

        if self.model == 'PDE_N4':
            in_features = torch.cat([u_j, embedding_j], dim=1)
        elif self.model == 'PDE_N5':
            in_features = torch.cat([u_j, embedding_i, embedding_j], dim=1)
        else:
            in_features = u_j

        lin_edge = self.lin_edge(in_features)
        T = W * self.mask

        if self.batch_size == 1:
            return T[edge_index_i, edge_index_j][:, None] * lin_edge
        else:
            return T[edge_index_i % (W.shape[0]), edge_index_j % (W.shape[0])][:, None] * lin_edge


    def update(self, aggr_out):
        return aggr_out


    def psi(self, r, p):
        return p * r
