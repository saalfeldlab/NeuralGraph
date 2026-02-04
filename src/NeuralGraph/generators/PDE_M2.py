
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


class PDE_M2(nn.Module):
    """Stoichiometric kinetics ODE with external rate modulation.

    Extends PDE_M1 by allowing an external signal (e.g. a movie mapped onto
    metabolite positions) to modulate the per-reaction rate constants.

    The external input lives in x[:, 4] (one value per metabolite).  For each
    reaction j the modulation factor is the **mean external input across its
    substrate metabolites**, analogous to how PDE_N4 multiplies the synaptic
    message by the external input at the post-synaptic neuron.

    Modulation modes (set via ``config.simulation.external_input_mode``):
      - ``"multiplicative"``:  v_j = k_j * ext_mean_j * softplus(rate_mlp(h_j))
      - ``"additive"``:        dx_i/dt += external_input_i   (direct flux)
      - ``"none"``:            identical to PDE_M1

    X tensor layout:
      x[:, 0]   = index (metabolite ID)
      x[:, 1:3] = positions (x, y)
      x[:, 3]   = concentration
      x[:, 4]   = external_input (modulation signal)
      x[:, 5]   = 0 (unused)
      x[:, 6]   = metabolite_type
      x[:, 7]   = 0 (unused)
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
        self.external_input_mode = getattr(
            config.simulation, 'external_input_mode', 'none'
        )

        # learnable MLPs
        self.msg_mlp = mlp([2, hidden, msg_dim], activation=nn.Tanh)
        self.rate_mlp = mlp([msg_dim, hidden, 1], activation=nn.Tanh)
        self.softplus = nn.Softplus(beta=1.0)

        # per-reaction rate constants: log-uniform in [0.01, 10]
        log_k = torch.empty(n_rxn)
        log_k.uniform_(-2.0, 1.0)
        self.log_k = nn.Parameter(log_k)

        # store stoichiometric graph (fixed, not learned)
        (met_sub, rxn_sub, sto_sub) = stoich_graph['sub']
        (met_all, rxn_all, sto_all) = stoich_graph['all']

        self.register_buffer('met_sub', met_sub)
        self.register_buffer('rxn_sub', rxn_sub)
        self.register_buffer('sto_sub', sto_sub)
        self.register_buffer('met_all', met_all)
        self.register_buffer('rxn_all', rxn_all)
        self.register_buffer('sto_all', sto_all)

        # pre-compute number of substrates per reaction for averaging
        n_sub_per_rxn = torch.zeros(n_rxn, dtype=torch.float32)
        n_sub_per_rxn.index_add_(0, rxn_sub, torch.ones_like(rxn_sub, dtype=torch.float32))
        n_sub_per_rxn.clamp_(min=1.0)
        self.register_buffer('n_sub_per_rxn', n_sub_per_rxn)

        self.p = torch.zeros(1)

    def forward(self, data=None, has_field=False, frame=None):
        """Compute dx/dt for all metabolites.

        Returns
        -------
        dxdt : Tensor of shape (n_met, 1)
        """
        x = data.x
        concentrations = x[:, 3]
        external_input = x[:, 4]

        # 1. gather substrate concentrations and stoichiometric coefficients
        x_src = concentrations[self.met_sub].unsqueeze(-1)
        s_abs = self.sto_sub.unsqueeze(-1)
        msg_in = torch.cat([x_src, s_abs], dim=-1)

        # 2. compute messages
        msg = self.msg_mlp(msg_in)

        # 3. aggregate messages per reaction
        h_rxn = torch.zeros(self.n_rxn, msg.shape[1], dtype=msg.dtype, device=msg.device)
        h_rxn.index_add_(0, self.rxn_sub, msg)

        # 4. compute base reaction rates
        k = torch.pow(10.0, self.log_k)
        base_v = self.softplus(self.rate_mlp(h_rxn).squeeze(-1))

        # 5. apply external modulation
        if self.external_input_mode == "multiplicative":
            # mean external input at substrate metabolites of each reaction
            ext_src = external_input[self.met_sub]
            ext_agg = torch.zeros(self.n_rxn, dtype=ext_src.dtype, device=ext_src.device)
            ext_agg.index_add_(0, self.rxn_sub, ext_src)
            ext_mean = ext_agg / self.n_sub_per_rxn
            v = k * ext_mean * base_v
        else:
            v = k * base_v

        # 6. compute dx/dt via stoichiometric matrix
        contrib = self.sto_all * v[self.rxn_all]
        dxdt = torch.zeros(self.n_met, dtype=contrib.dtype, device=contrib.device)
        dxdt.index_add_(0, self.met_all, contrib)

        # 7. additive external input (direct flux injection)
        if self.external_input_mode == "additive":
            dxdt = dxdt + external_input

        return 0.005 * dxdt.unsqueeze(-1)

    def get_rates(self, data):
        """Return reaction rates for diagnostics."""
        x = data.x
        concentrations = x[:, 3]
        external_input = x[:, 4]

        x_src = concentrations[self.met_sub].unsqueeze(-1)
        s_abs = self.sto_sub.unsqueeze(-1)
        msg_in = torch.cat([x_src, s_abs], dim=-1)
        msg = self.msg_mlp(msg_in)

        h_rxn = torch.zeros(self.n_rxn, msg.shape[1], dtype=msg.dtype, device=msg.device)
        h_rxn.index_add_(0, self.rxn_sub, msg)

        k = torch.pow(10.0, self.log_k)
        base_v = self.softplus(self.rate_mlp(h_rxn).squeeze(-1))

        if self.external_input_mode == "multiplicative":
            ext_src = external_input[self.met_sub]
            ext_agg = torch.zeros(self.n_rxn, dtype=ext_src.dtype, device=ext_src.device)
            ext_agg.index_add_(0, self.rxn_sub, ext_src)
            ext_mean = ext_agg / self.n_sub_per_rxn
            v = k * ext_mean * base_v
        else:
            v = k * base_v

        return v
