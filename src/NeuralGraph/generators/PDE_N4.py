
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
import torch
from NeuralGraph.utils import *


class PDE_N4(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Compute network signaling, the transfer functions are neuron-dependent
    
    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    du : float
    the update rate of the signals (dim 1)
        
    """

    def __init__(self, aggr_type=[], p=[], W=[], phi=[]):
        super(PDE_N4, self).__init__(aggr=aggr_type)

        self.p = p
        self.W = W
        self.phi = phi

    def forward(self, data=[], has_field=False, data_id=[]):
        x, edge_index = data.x, data.edge_index
        neuron_type = x[:, 5].long()

        n_neuron_types = torch.max(neuron_type).item() + 1
        n_parameters = self.p.size(0)

        if n_neuron_types > n_parameters:
            parameters = self.p[neuron_type // (n_neuron_types//n_parameters)]
            g = parameters[:, 0:1]
            s = parameters[:, 1:2]
            c = parameters[:, 2:3]
            t_ = torch.linspace(1, 8, n_neuron_types)[:, None].to(x.device)
            t = t_[neuron_type]
        else:
            parameters = self.p[neuron_type]
            g = parameters[:, 0:1]
            s = parameters[:, 1:2]
            c = parameters[:, 2:3]
            t = parameters[:, 3:4]
            if parameters.shape[1] < 5:
                b = torch.zeros_like(t)
            else:
                b = parameters[:, 4:5]

        u = x[:, 6:7]
        if has_field:
            field = x[:, 8:9]
        else:
            field = torch.ones_like(x[:, 6:7])

        msg = self.propagate(edge_index, u=u, t=t, b=b, field=field)

        du = -c * u + s * torch.tanh(u) + g * msg

        return du

    def message(self, edge_index_i, edge_index_j, u_j, t_j, b_j, field_i):

        T = self.W

        return T[edge_index_i, edge_index_j][:, None] * self.phi((u_j-b_j) / t_j) * field_i

    def func(self, u, type, function):

        if function=='phi':

            t = self.p[type, 3:4]

            if self.p.shape[1] < 5:
                b = torch.zeros_like(t)
            else:
                b = self.p[type, 4:5]

            return self.phi((u-b)/t)

        elif function=='update':
            g, s, c = self.p[type, 0:1], self.p[type, 1:2], self.p[type, 2:3]
            return -c * u + s * torch.tanh(u)
