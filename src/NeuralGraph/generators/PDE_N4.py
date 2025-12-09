
import numpy as np
import torch_geometric as pyg
import torch


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

    def __init__(self, config=None, aggr_type=[], p=[], W=[], phi=[], device=None):
        super(PDE_N4, self).__init__(aggr=aggr_type)

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
        self.has_triggered = (config.simulation.input_type == 'triggered')
        self.max_frame = config.simulation.n_frames + 1

        # triggered oscillation parameters
        if self.has_triggered:
            self.triggered_n_input = config.simulation.triggered_n_input_neurons
            self.triggered_strength = config.simulation.triggered_impulse_strength
            self.triggered_min_start = config.simulation.triggered_min_start_frame
            self.triggered_max_start = config.simulation.triggered_max_start_frame
            self.triggered_duration = config.simulation.triggered_duration_frames
            # randomly select which neurons receive input
            self.input_neurons = torch.randperm(self.n_neurons, device=self.device)[:self.triggered_n_input]
            # random trigger frame for each simulation
            self.trigger_frame = torch.randint(self.triggered_min_start, self.triggered_max_start + 1, (1,), device=self.device).item()

    def forward(self, data=[], has_field=False, data_id=[], frame=[]):
        x, edge_index = data.x, data.edge_index
        neuron_type = x[:, 5].long()

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

        if self.has_oscillations:
            du = -c * u + s * torch.tanh(u) + g * msg + self.e * torch.cos((2*np.pi)*self.w*frame / self.max_frame)
        elif self.has_triggered:
            du = -c * u + s * torch.tanh(u) + g * msg
            # add impulse input at trigger frame to selected neurons
            if isinstance(frame, int) and frame == self.trigger_frame:
                impulse = torch.zeros((self.n_neurons, 1), device=self.device)
                impulse[self.input_neurons] = self.triggered_strength
                du = du + impulse
            # add oscillatory response after trigger for duration frames
            if isinstance(frame, int) and self.trigger_frame <= frame < self.trigger_frame + self.triggered_duration:
                t_since_trigger = frame - self.trigger_frame
                osc = self.e * torch.sin((2*np.pi)*self.w*t_since_trigger / self.triggered_duration)
                du = du + osc
        else:
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
            _g, s, c = self.p[type, 0:1], self.p[type, 1:2], self.p[type, 2:3]
            return -c * u + s * torch.tanh(u)
