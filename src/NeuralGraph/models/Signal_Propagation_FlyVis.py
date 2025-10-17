import torch
import torch.nn as nn
import torch_geometric as pyg
from NeuralGraph.models.MLP import MLP
from NeuralGraph.utils import to_numpy
import numpy as np
from NeuralGraph.models.Siren_Network import *

class Signal_Propagation_FlyVis(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Model learning the first derivative of a scalar field on a mesh.
    The node embedding is defined by a table self.a
    Note the Laplacian coeeficients are in data.edge_attr

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    pred : float
        the first derivative of a scalar field on a mesh (dimension 3).
    """

    def __init__(
        self, aggr_type='add', config=None, device=None ):
        super(Signal_Propagation_FlyVis, self).__init__(aggr=aggr_type)

        simulation_config = config.simulation
        model_config = config.graph_model

        self.device = device
        self.model = model_config.signal_model_name
        self.dimension = simulation_config.dimension
        self.embedding_dim = model_config.embedding_dim
        self.n_neurons = simulation_config.n_neurons
        self.n_input_neurons = simulation_config.n_input_neurons
        self.n_dataset = config.training.n_runs
        self.n_frames = simulation_config.n_frames
        self.field_type = model_config.field_type
        self.embedding_trial = config.training.embedding_trial
        self.multi_connectivity = config.training.multi_connectivity
        self.calcium_type = simulation_config.calcium_type
        self.MLP_activation = config.graph_model.MLP_activation

        self.training_time_window = config.training.time_window

        self.input_size = model_config.input_size
        self.output_size = model_config.output_size
        self.hidden_dim = model_config.hidden_dim
        self.n_layers = model_config.n_layers

        self.n_layers_update = model_config.n_layers_update
        self.hidden_dim_update = model_config.hidden_dim_update
        self.input_size_update = model_config.input_size_update

        self.n_edges = simulation_config.n_edges
        self.n_extra_null_edges = simulation_config.n_extra_null_edges
        self.lin_edge_positive = model_config.lin_edge_positive

        self.batch_size = config.training.batch_size
        self.update_type = model_config.update_type

        self.lin_edge = MLP(
            input_size=self.input_size,
            output_size=self.output_size,
            nlayers=self.n_layers,
            hidden_size=self.hidden_dim,
            activation=self.MLP_activation,
            device=self.device,
        )

        self.lin_phi = MLP(
            input_size=self.input_size_update,
            output_size=self.output_size,
            nlayers=self.n_layers_update,
            hidden_size=self.hidden_dim_update,
            activation=self.MLP_activation,
            device=self.device,
        )

        # embedding
        self.a = nn.Parameter(
            torch.tensor(
                np.ones((int(self.n_neurons), self.embedding_dim)),
                         device=self.device,
                         requires_grad=True, dtype=torch.float32))

        self.W = nn.Parameter(
            torch.zeros(
                self.n_edges + self.n_extra_null_edges,
                device=self.device,
                requires_grad=True,
                dtype=torch.float32,
            )[:, None]
        )

        if 'visual' in model_config.field_type:
            self.NNR_f = Siren(in_features=model_config.input_size_nnr_f, out_features=model_config.output_size_nnr_f,
                        hidden_features=model_config.hidden_dim_nnr_f,
                        hidden_layers=model_config.n_layers_nnr_f, first_omega_0=model_config.omega_f,
                        hidden_omega_0=model_config.omega_f,
                        outermost_linear=model_config.outermost_linear_nnr_f)
            self.NNR_f.to(self.device)

            self.NNR_f_xy_period = model_config.nnr_f_xy_period / (2*np.pi)
            self.NNR_f_T_period = model_config.nnr_f_T_period / (2*np.pi)

            self.learn_visual = True
        else:
            self.learn_visual = False

    def forward_visual(self, x=[], k = []):
        kk = torch.full((x.size(0), 1), float(k), device=self.device, dtype=torch.float32)
        in_features = torch.cat((x[:,1:1+self.dimension] / self.NNR_f_xy_period, kk / self.NNR_f_T_period), dim=1)
        reconstructed_field = self.NNR_f(in_features[:self.n_input_neurons]) ** 2

        return reconstructed_field

    def forward(self, data=[], data_id=[], mask=[], k = [],return_all=False):
        self.return_all = return_all
        x, edge_index = data.x, data.edge_index

        self.data_id = data_id.squeeze().long().clone().detach()
        self.mask = mask.squeeze().long().clone().detach()

        if self.calcium_type!="none":
            v = data.x[:, 7:8]      # voltage is replaced by calcium concentration (observable)
        else:
            v = data.x[:, 3:4]
 
        excitation = data.x[:, 4:5]

        particle_id = x[:, 0].long()
        embedding = self.a[particle_id].squeeze()

        msg = self.propagate(
            edge_index, v=v, embedding=embedding, data_id=self.data_id[:, None]
        )

        if self.learn_visual:
            in_features = torch.cat([v, embedding, msg], dim=1)
            pred = self.lin_phi(in_features) + excitation
        else:
            in_features = torch.cat([v, embedding, msg, excitation], dim=1)
            pred = self.lin_phi(in_features)
        
        if return_all:
            return pred, in_features, msg
        else:
            return pred

    def message(self, edge_index_i, edge_index_j, v_i, v_j, embedding_i, embedding_j, data_id_i):
        if (self.model=='PDE_N9_B'):
            in_features = torch.cat([v_i, v_j, embedding_i, embedding_j], dim=1)
        else:
            in_features = torch.cat([v_j, embedding_j], dim=1)

        lin_edge = self.lin_edge(in_features)
        if self.lin_edge_positive:
            lin_edge = lin_edge**2

        return self.W[self.mask % (self.n_edges+ self.n_extra_null_edges)] * lin_edge

    def update(self, aggr_out):
        return aggr_out
