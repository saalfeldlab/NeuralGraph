import torch
import torch_geometric as pyg
from NeuralGraph.utils import to_numpy


class RD_FitzHugh_Nagumo(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Compute the reaction diffusion according to FitzHugh_Nagumo model.

    Inputs
    ----------
    data : a torch_geometric.data object
    Note the Laplacian coeeficients are in data.edge_attr

    Returns
    -------
    increment : float
        the first derivative of two scalar fields u and v
    """

    def __init__(self, aggr_type=[], c=[], bc_dpos=[]):
        super(RD_FitzHugh_Nagumo, self).__init__(aggr='add')  # "mean" aggregation.

        self.c = c
        self.beta = 1E-2
        self.bc_dpos = bc_dpos

        self.a1 = 5E-3
        self.a2 = -2.8E-3
        self.a3 = 5E-3
        self.a4 = 0.02
        self.a5 = 0.125

    def forward(self, data, device):
        particle_type = to_numpy(x[:, 5])
        c = self.c[particle_type]
        c = c[:, None]

        u = data.x[:, 6]
        v = data.x[:, 7]
        laplace_u = c * self.beta * self.propagate(data.edge_index, u=u, discrete_laplacian=data.edge_attr)

        # This is equivalent to the nonlinear reaction diffusion equation:
        #   du = a3 * laplace_u + a4 * (v - v^3 - u * v + noise)
        #   dv = a1 * u + a2 * v
        d_u = self.a3 * laplace_u + self.a4 * (v - v ** 3 - u * v + torch.randn(4225, device=device))
        d_v = (self.a1 * u + self.a2 * v)

        d_uv = self.a5 * torch.column_stack((d_u, d_v))
        return d_uv

    def message(self, u_i, u_j, discrete_laplacian):
        return discrete_laplacian * u_j

    def psi(self, I, p):
        return I
