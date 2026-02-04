
import torch
import torch.nn as nn


def mlp(sizes, activation=nn.Tanh, final_activation=None):
    """Build a simple feedforward MLP."""
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
        elif final_activation is not None:
            layers.append(final_activation())
    return nn.Sequential(*layers)


class PDE_M1(nn.Module):
    """Stoichiometric kinetics ODE for metabolic networks.

    Given a fixed stoichiometric matrix S (n_met x n_rxn), learn the reaction
    rate functions from metabolite concentrations.

    The stoichiometric graph is bipartite (metabolites <-> reactions):
      - substrate edges: metabolite -> reaction with |s_ij| (consumed)
      - all edges: metabolite -> reaction with signed s_ij

    Forward pass:
      1. For each substrate edge, build message from [concentration, |stoich|]
      2. Aggregate messages per reaction via sum
      3. Compute rate v_j = softplus(rate_mlp(h_j))  (non-negative rates)
      4. dx/dt = sum over edges: s_ij * v_j

    X tensor layout (same as neural models for framework compatibility):
      x[:, 0]   = index (metabolite ID)
      x[:, 1:3] = positions (x, y) for visualisation
      x[:, 3]   = concentration
      x[:, 4]   = 0 (unused)
      x[:, 5]   = 0 (unused)
      x[:, 6]   = metabolite_type
      x[:, 7]   = 0 (unused)

    Parameters
    ----------
    config : NeuralGraphConfig
    stoich_graph : dict  with keys
        'sub': (met_sub, rxn_sub, sto_sub)   substrate edges
        'all': (met_all, rxn_all, sto_all)   all stoichiometric edges
    device : torch.device
    """

    def __init__(self, config=None, stoich_graph=None, device=None):
        super().__init__()

        n_met = config.simulation.n_metabolites
        n_rxn = config.simulation.n_reactions
        hidden = config.graph_model.hidden_dim
        msg_dim = config.graph_model.output_size

        self.n_met = n_met
        self.n_rxn = n_rxn
        self.device = device

        # learnable MLPs
        self.msg_mlp = mlp([2, hidden, msg_dim], activation=nn.Tanh)
        self.rate_mlp = mlp([msg_dim, hidden, 1], activation=nn.Tanh)
        self.softplus = nn.Softplus(beta=1.0)

        # store stoichiometric graph (fixed, not learned)
        (met_sub, rxn_sub, sto_sub) = stoich_graph['sub']
        (met_all, rxn_all, sto_all) = stoich_graph['all']

        self.register_buffer('met_sub', met_sub)
        self.register_buffer('rxn_sub', rxn_sub)
        self.register_buffer('sto_sub', sto_sub)
        self.register_buffer('met_all', met_all)
        self.register_buffer('rxn_all', rxn_all)
        self.register_buffer('sto_all', sto_all)

        # store model parameters for saving (like model.p in PDE_N4)
        self.p = torch.zeros(1)

    def forward(self, data=None, has_field=False, frame=None):
        """Compute dx/dt for all metabolites.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Must contain x tensor with concentration at x[:, 3]

        Returns
        -------
        dxdt : Tensor of shape (n_met, 1)
        """
        x = data.x
        concentrations = x[:, 3]  # metabolite concentrations

        # 1. gather substrate concentrations and stoichiometric coefficients
        x_src = concentrations[self.met_sub].unsqueeze(-1)  # (n_sub_edges, 1)
        s_abs = self.sto_sub.unsqueeze(-1)                  # (n_sub_edges, 1)
        msg_in = torch.cat([x_src, s_abs], dim=-1)          # (n_sub_edges, 2)

        # 2. compute messages
        msg = self.msg_mlp(msg_in)  # (n_sub_edges, msg_dim)

        # 3. aggregate messages per reaction
        h_rxn = torch.zeros(self.n_rxn, msg.shape[1], dtype=msg.dtype, device=msg.device)
        h_rxn.index_add_(0, self.rxn_sub, msg)

        # 4. compute non-negative reaction rates
        v = self.softplus(self.rate_mlp(h_rxn).squeeze(-1))  # (n_rxn,)

        # 5. compute dx/dt via stoichiometric matrix: dx_i/dt = sum_j S_ij * v_j
        contrib = self.sto_all * v[self.rxn_all]  # (n_all_edges,)
        dxdt = torch.zeros(self.n_met, dtype=contrib.dtype, device=contrib.device)
        dxdt.index_add_(0, self.met_all, contrib)

        return dxdt.unsqueeze(-1)  # (n_met, 1)

    def get_rates(self, data):
        """Return reaction rates for diagnostics."""
        x = data.x
        concentrations = x[:, 3]

        x_src = concentrations[self.met_sub].unsqueeze(-1)
        s_abs = self.sto_sub.unsqueeze(-1)
        msg_in = torch.cat([x_src, s_abs], dim=-1)
        msg = self.msg_mlp(msg_in)

        h_rxn = torch.zeros(self.n_rxn, msg.shape[1], dtype=msg.dtype, device=msg.device)
        h_rxn.index_add_(0, self.rxn_sub, msg)

        v = self.softplus(self.rate_mlp(h_rxn).squeeze(-1))
        return v
