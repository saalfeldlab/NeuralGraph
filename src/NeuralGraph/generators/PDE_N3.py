
import numpy as np
import torch_geometric as pyg
import torch

class PDE_N3(pyg.nn.MessagePassing):
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

    def __init__(self, config=None, aggr_type=[], p=[], W=[], phi=[], device=None):
        super(PDE_N3, self).__init__(aggr=aggr_type)

        self.p = p
        self.W = W
        self.phi = phi
        self.device = device

        # oscillation parameters
        self.n_neurons = config.simulation.n_neurons
        self.A = config.simulation.oscillation_max_amplitude
        self.e = self.A * (torch.rand((self.n_neurons, 1), device=self.device) * 2 - 1)
        self.w = torch.tensor(config.simulation.oscillation_frequency, dtype=torch.float32, device=self.device)
        self.has_oscillations = (config.simulation.input_type == 'oscillatory')
        self.max_frame = config.simulation.n_frames + 1

    def forward(self, data=[], has_field=False, alpha=1.0, data_id=[], frame=[]):
        x, _edge_index = data.x, data.edge_index
        # edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        neuron_type = x[:, 5].long()
        parameters = alpha * self.p[neuron_type + 1] + (1-alpha) * self.p[neuron_type]
        g = parameters[:, 0:1]
        s = parameters[:, 1:2]
        c = parameters[:, 2:3]

        u = x[:, 6:7]

        # msg = self.propagate(edge_index, u=u, edge_attr=edge_attr)
        msg = torch.matmul(self.W, self.phi(u))

        if self.has_oscillations:
            du = -c * u + s * torch.tanh(u) + g * msg + self.e * torch.cos((2*np.pi)*self.w*frame / self.max_frame)
        else:
            du = -c * u + s * torch.tanh(u) + g * msg

        return du


    def message(self, u_j, edge_attr):

        self.activation = self.phi(u_j)
        self.u_j = u_j

        return edge_attr[:,None] * self.phi(u_j)


    def func(self, u, type, function):

        if function=='phi':
            return self.phi(u)

        elif function=='update':
            _g, s, c = self.p[type, 0:1], self.p[type, 1:2], self.p[type, 2:3]
            return -c * u + s * torch.tanh(u)
