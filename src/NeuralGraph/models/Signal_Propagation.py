import torch
import torch.nn as nn
import torch_geometric as pyg
from NeuralGraph.models.MLP import MLP


class Signal_Propagation(pyg.nn.MessagePassing):
    """
    graph neural network for learning neural signal dynamics.

    based on interaction networks (Battaglia et al., NeurIPS 2016).
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html

    learns the first derivative of neural activity (du/dt) using:
    - MLP0 (lin_edge): message function on edges
    - MLP1 (lin_phi): node update function
    - W: learnable connectivity matrix (n_neurons x n_neurons)
    - a: learnable node embeddings (n_neurons x embedding_dim)

    external input modes (x[:,4]):
    - "none": no external input
    - "additive": du/dt = MLP1(u, a) + W @ MLP0(u, a) + external_input
    - "multiplicative": du/dt = MLP1(u, a) + W @ MLP0(u, a) * external_input

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

    PARAMS_DOC = {
        "model_name": "Signal_Propagation",
        "description": "GNN for neural signal dynamics: du/dt = lin_phi(u, a) + W @ lin_edge(u, a)",
        "equations": {
            "message": "msg_j = W[i,j] * lin_edge(in_features_j)",
            "update_default": "du/dt = lin_phi(u, a) + sum(msg)",
            "update_generic": "du/dt = lin_phi(u, a, sum(msg), external_input)",
        },
        "graph_model_config": {
            "description": "Parameters in the graph_model: section of the YAML config. "
                           "input_size and input_size_update are DERIVED from the model variant "
                           "and embedding_dim — they must NOT be set independently.",
            "lin_edge (MLP0)": {
                "description": "Edge message function — computes per-edge features before W multiplication",
                "input_size": {
                    "description": "DERIVED — do not set manually. Depends on signal_model_name:",
                    "PDE_N2": "input_size = 1  (u_j only)",
                    "PDE_N3": "input_size = 1  (u_j only)",
                    "PDE_N4": "input_size = 1 + embedding_dim  (u_j, a_j)",
                    "PDE_N5": "input_size = 1 + 2*embedding_dim  (u_j, a_i, a_j)",
                    "PDE_N7": "input_size = 1 + embedding_dim  (u_j, a_j)",
                    "PDE_N8": "input_size = 2 + 2*embedding_dim  (u_i, u_j, a_i, a_j)",
                    "PDE_N11": "input_size = 1 + embedding_dim  (u_j, a_j)",
                },
                "output_size": "1 (scalar edge weight)",
                "hidden_dim": {"description": "Hidden layer width", "typical_range": [32, 128], "default": 64},
                "n_layers": {"description": "Number of MLP layers", "typical_range": [2, 5], "default": 3},
            },
            "lin_phi (MLP1)": {
                "description": "Node update function — computes du/dt from node state + aggregated messages",
                "input_size_update": {
                    "description": "DERIVED — do not set manually. Depends on update_type:",
                    "update_type='none'": "input_size_update = 1 + embedding_dim  (u, a)",
                    "update_type='generic'": "input_size_update = 1 + embedding_dim + output_size + 1  (u, a, msg, external)",
                },
                "output_size": "1 (du/dt scalar)",
                "hidden_dim_update": {"description": "Hidden layer width", "typical_range": [32, 128], "default": 64},
                "n_layers_update": {"description": "Number of MLP layers", "typical_range": [2, 5], "default": 3},
            },
            "embedding": {
                "embedding_dim": {"description": "Dimension of learnable node embedding a_i", "typical_range": [1, 8], "default": 2},
            },
        },
        "simulation_params": {
            "description": "Parameters in the simulation: section of the YAML config (data generation)",
            "slots": [
                {"index": 0, "name": "tau", "description": "Time constant (params[0][0])", "typical_range": [0.5, 2.0]},
                {"index": 1, "name": "noise_level", "description": "Process noise (params[0][1])", "typical_range": [0.0, 0.1]},
                {"index": 2, "name": "g", "description": "Coupling gain — scales connectivity matrix W (params[0][2])", "typical_range": [1.0, 15.0]},
                {"index": 3, "name": "bias", "description": "Constant input bias (params[0][3])", "typical_range": [-1.0, 1.0]},
                {"index": 4, "name": "self_coupling", "description": "Diagonal self-coupling strength (params[0][4])", "typical_range": [0.5, 2.0]},
                {"index": 5, "name": "unused", "description": "Reserved (params[0][5])"},
            ],
        },
        "training_params": {
            "description": "Parameters in the training: section that affect model architecture or loss",
            "tunable": [
                {"name": "learning_rate_W_start", "description": "Learning rate for connectivity W", "typical_range": [1e-4, 5e-2]},
                {"name": "learning_rate_start", "description": "Learning rate for MLPs", "typical_range": [1e-4, 5e-3]},
                {"name": "learning_rate_embedding_start", "description": "Learning rate for embeddings a", "typical_range": [1e-4, 5e-3]},
                {"name": "coeff_W_L1", "description": "L1 sparsity penalty on W", "typical_range": [1e-6, 1e-3]},
                {"name": "coeff_edge_diff", "description": "Regularizer: lin_edge output variance penalty", "typical_range": [0, 500]},
                {"name": "batch_size", "description": "Number of time frames per batch", "typical_range": [1, 16]},
            ],
        },
    }

    def __init__(self, aggr_type=None, config=None, device=None, bc_dpos=None, projections=None):
        super(Signal_Propagation, self).__init__(aggr=aggr_type)

        simulation_config = config.simulation
        model_config = config.graph_model
        train_config = config.training

        self.device = device
        self.model = model_config.signal_model_name
        self.embedding_dim = model_config.embedding_dim
        self.n_neurons = simulation_config.n_neurons
        self.n_dataset = config.training.n_runs
        self.n_frames = simulation_config.n_frames
        self.embedding_trial = config.training.embedding_trial
        self.multi_connectivity = config.training.multi_connectivity
        self.MLP_activation = config.graph_model.MLP_activation

        self.input_size = model_config.input_size
        self.output_size = model_config.output_size
        self.hidden_dim = model_config.hidden_dim
        self.n_layers = model_config.n_layers

        self.lin_edge_positive = model_config.lin_edge_positive

        self.n_layers_update = model_config.n_layers_update
        self.hidden_dim_update = model_config.hidden_dim_update
        self.input_size_update = model_config.input_size_update

        self.input_size_modulation = model_config.input_size_modulation
        self.output_size_modulation = model_config.output_size_modulation
        self.hidden_dim_modulation = model_config.hidden_dim_modulation
        self.n_layers_modulation = model_config.n_layers_modulation

        self.batch_size = config.training.batch_size
        self.update_type = model_config.update_type

        self.bc_dpos = bc_dpos
        self.adjacency_matrix = simulation_config.adjacency_matrix
        self.excitation_dim = model_config.excitation_dim

        self.n_layers_excitation = model_config.n_layers_excitation
        self.hidden_dim_excitation = model_config.hidden_dim_excitation
        self.input_size_excitation = model_config.input_size_excitation

        self.low_rank_factorization = config.training.low_rank_factorization
        self.low_rank = config.training.low_rank

        self.external_input_mode = getattr(config.simulation, 'external_input_mode', 'none')


        if self.model == 'PDE_N3':
            self.embedding_evolves = True
        else:
            self.embedding_evolves = False

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.n_layers,
                            hidden_size=self.hidden_dim, activation=self.MLP_activation, device=self.device)

        self.lin_phi = MLP(input_size=self.input_size_update, output_size=self.output_size, nlayers=self.n_layers_update,
                            hidden_size=self.hidden_dim_update, activation=self.MLP_activation, device=self.device)

        if self.embedding_trial:
            self.b = nn.Parameter(
                torch.ones((int(self.n_dataset), self.embedding_dim), device=self.device, requires_grad=True, dtype=torch.float32))

        if self.model == 'PDE_N3':
            self.a = nn.Parameter(torch.ones((int(self.n_neurons*100 + 1000), self.embedding_dim), device=self.device, requires_grad=True,dtype=torch.float32))
            self.embedding_step =  self.n_frames // 100
        else:
            if train_config.training_single_type:
                self.register_buffer('a', torch.ones((self.n_neurons, self.embedding_dim), device=self.device, requires_grad=False, dtype=torch.float32))
            else:
                self.a = nn.Parameter(torch.ones(self.n_neurons, self.embedding_dim, device=self.device, requires_grad=True, dtype=torch.float32))

        if (self.model == 'PDE_N6') | (self.model == 'PDE_N7'):
            self.b = nn.Parameter(torch.ones((int(self.n_neurons), 1000 + 10), device=self.device, requires_grad=True,dtype=torch.float32)*0.44)
            self.embedding_step = self.n_frames // 1000
            self.lin_modulation = MLP(input_size=self.input_size_modulation, output_size=self.output_size_modulation, nlayers=self.n_layers_modulation,
                                hidden_size=self.hidden_dim_modulation, device=self.device)
            

        if self.multi_connectivity:
            self.W = nn.Parameter(torch.zeros((int(self.n_dataset),int(self.n_neurons),int(self.n_neurons)), device=self.device, requires_grad=True, dtype=torch.float32))
        
        else:

            if self.low_rank_factorization:

                WL_init = torch.randn((int(self.n_neurons),int(self.low_rank)), device=self.device, dtype=torch.float32)
                WL_init[:min(self.n_neurons, self.low_rank), :min(self.n_neurons, self.low_rank)].fill_diagonal_(0)
                self.WL = nn.Parameter(WL_init, requires_grad=True)

                WR_init = torch.randn((int(self.low_rank),int(self.n_neurons)), device=self.device, dtype=torch.float32)
                WR_init[:min(self.low_rank, self.n_neurons), :min(self.low_rank, self.n_neurons)].fill_diagonal_(0)
                self.WR = nn.Parameter(WR_init, requires_grad=True)

                # W as buffer for saving/post-analysis (updated each forward pass)
                self.register_buffer('W', torch.zeros((int(self.n_neurons),int(self.n_neurons)), dtype=torch.float32))

            else:

                W_init = torch.randn((int(self.n_neurons),int(self.n_neurons)), device=self.device, dtype=torch.float32)
                W_init.fill_diagonal_(0)
                self.W = nn.Parameter(W_init, requires_grad=True)


        self.register_buffer('mask', torch.ones((int(self.n_neurons),int(self.n_neurons)), requires_grad=False, dtype=torch.float32))
        self.mask.fill_diagonal_(0)

    def get_interp_a(self, k, particle_id):

        id = particle_id * 100 + k // self.embedding_step
        alpha = (k % self.embedding_step) / self.embedding_step

        return alpha * self.a[id.squeeze()+1, :] + (1 - alpha) * self.a[id.squeeze(), :]
    

    def forward_excitation(self,  k = []):


        kk = torch.full((1, 1), float(k), device=self.device, dtype=torch.float32)

        in_features = torch.tensor(kk / self.NNR_f_T_period, dtype=torch.float32, device=self.device)
        excitation_field = self.NNR_f(in_features)

        return excitation_field


    def forward(self, data=[], data_id=[], k = [], return_all=False):
        self.return_all = return_all
        x, edge_index = data.x, data.edge_index

        self.data_id = data_id.squeeze().long().clone().detach()

        u = data.x[:, 3:4]
        external_input = x[:, 4:5]

        if self.model == 'PDE_N3':
            particle_id = x[:, 0:1].long()
            embedding = self.get_interp_a(k, particle_id)
        else:
            particle_id = x[:, 0].long()
            embedding = self.a[particle_id, :]
            if self.embedding_trial:
                embedding = torch.cat((self.b[self.data_id, :], embedding), dim=1)

        if self.low_rank_factorization:
            W = self.WL @ self.WR
            self.W.copy_(W.detach())  # update buffer for saving/post-analysis
        else:
            W = self.W

        msg = self.propagate(edge_index, u=u, embedding=embedding, data_id=self.data_id[:,None], W=W)


        if 'generic' in self.update_type:        # MLP1(u, embedding, \sum MLP0(u, embedding), field)
            in_features = torch.cat([u, embedding, msg, external_input], dim=1)
            pred = self.lin_phi(in_features)
        else:
            in_features = torch.cat([u, embedding], dim=1)

            if self.external_input_mode == "multiplicative":
                pred = self.lin_phi(in_features) + msg * external_input
            elif self.external_input_mode == "additive":
                pred = self.lin_phi(in_features) + msg + external_input
            else:
                pred = self.lin_phi(in_features) + msg

        


        if return_all:
            return pred, in_features
        else:
            return pred


    def message(self, edge_index_i, edge_index_j, u_i, u_j, embedding_i, embedding_j, data_id_i, W):

        if (self.model=='PDE_N4') | (self.model=='PDE_N7') | (self.model=='PDE_N11'):
            in_features = torch.cat([u_j, embedding_j], dim=1)
        elif (self.model=='PDE_N5'):
            in_features = torch.cat([u_j, embedding_i, embedding_j], dim=1)
        elif (self.model=='PDE_N8'):
            in_features = torch.cat([u_i, u_j, embedding_i, embedding_j], dim=1)
        else:
            in_features = u_j

        lin_edge = self.lin_edge(in_features)
        if self.lin_edge_positive:
            lin_edge = lin_edge**2

        if self.multi_connectivity:
            if self.batch_size == 1:
                return W[data_id_i.squeeze(), edge_index_i, edge_index_j][:, None] * self.mask[edge_index_i, edge_index_j][:, None] * lin_edge
            else:
                return W[data_id_i.squeeze(), edge_index_i % (W.shape[1]), edge_index_j % (W.shape[1])][:, None] * self.mask[edge_index_i % (W.shape[1]), edge_index_j % (W.shape[1])][:, None] * lin_edge
        else:

            T = W * self.mask

            if (self.batch_size==1):
                return T[edge_index_i, edge_index_j][:, None] * lin_edge
            else:
                return T[edge_index_i%(W.shape[0]), edge_index_j%(W.shape[0])][:,None] * lin_edge

        # pos = torch.argwhere(edge_index_i==0)
        # neurons_sender_to_0 = edge_index_j[pos]
        # neurons_sender_to_0[0:20]
        # torch.sum(self.W[data_id_i.squeeze()[pos], edge_index_i[pos], edge_index_j[pos]][:, None] * self.mask[edge_index_i[pos], edge_index_j[pos]][:, None] * lin_edge[pos])


    def update(self, aggr_out):
        return aggr_out

    def psi(self, r, p):
        return p * r




# if (self.model=='PDE_N4') | (self.model=='PDE_N5'):
#     msg = self.propagate(edge_index, u=u, embedding=embedding, field=field)
# elif self.model=='PDE_N6':
#     msg = torch.matmul(self.W * self.mask, self.lin_edge(u)) * field
# else:
#     msg = torch.matmul(self.W * self.mask, self.lin_edge(u))
#     if self.return_all:
#         self.msg = torch.matmul(self.W * self.mask, self.lin_edge(u))
# if (self.model=='PDE_N2') & (self.batch_size==1):
#     msg = torch.matmul(self.W * self.mask, self.lin_edge(u))

# self.n_layers_update2 = model_config.n_layers_update2
# self.hidden_dim_update2 = model_config.hidden_dim_update2
# self.input_size_update2 = model_config.input_size_update2

# if self.update_type=='2steps':
#     self.lin_phi2 = MLP(input_size=self.input_size_update2, output_size=self.output_size,
#                        nlayers=self.n_layers_update2,
#                        hidden_size=self.hidden_dim_update2, device=self.device)

# if self.update_type == '2steps':                  # MLP2( MLP1(u, embedding), \sum MLP0(u, embedding), field)
#     in_features1 = torch.cat([u, embedding], dim=1)
#     pred1 = self.lin_phi(in_features1)
#     field = x[:, 8:9]
#     in_features2 = torch.cat([pred1, msg, field], dim=1)
# pred = self.lin_phi2(in_features2)