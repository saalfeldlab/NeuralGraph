import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def mlp(sizes, activation=nn.Tanh, final_activation=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
        elif final_activation is not None:
            layers.append(final_activation())
    return nn.Sequential(*layers)

class StoichKineticsODE(nn.Module):
    def __init__(self, n_met=4, n_rxn=3, msg_dim=16, hidden=32):
        super().__init__()
        self.n_met = n_met
        self.n_rxn = n_rxn
        self.msg_mlp = mlp([2, hidden, msg_dim], activation=nn.Tanh)
        self.rate_mlp = mlp([msg_dim, hidden, 1], activation=nn.Tanh)
        self.softplus = nn.Softplus(beta=1.0)

    @staticmethod
    def build_demo_graph():
        stoich_edges = [
            (0, 0, -1.0),
            (1, 0, +1.0),
            (1, 1, -1.0),
            (2, 1, -1.0),
            (3, 1, +1.0),
            (3, 2, -1.0),
            (0, 2, +1.0),
        ]
        sub_edges = [(m, r, s) for (m, r, s) in stoich_edges if s < 0.0]

        met_sub = torch.tensor([e[0] for e in sub_edges], dtype=torch.long)
        rxn_sub = torch.tensor([e[1] for e in sub_edges], dtype=torch.long)
        sto_sub = torch.tensor([abs(e[2]) for e in sub_edges], dtype=torch.float32)

        met_all = torch.tensor([e[0] for e in stoich_edges], dtype=torch.long)
        rxn_all = torch.tensor([e[1] for e in stoich_edges], dtype=torch.long)
        sto_all = torch.tensor([e[2] for e in stoich_edges], dtype=torch.float32)

        return (met_sub, rxn_sub, sto_sub), (met_all, rxn_all, sto_all)

    def forward(self, x_met, graph):
        (met_sub, rxn_sub, sto_sub), (met_all, rxn_all, sto_all) = graph

        x_src = x_met[met_sub].unsqueeze(-1)
        s_abs = sto_sub.unsqueeze(-1)
        msg_in = torch.cat([x_src, s_abs], dim=-1)
        msg = self.msg_mlp(msg_in)

        h_rxn = torch.zeros((self.n_rxn, msg.shape[1]), dtype=msg.dtype)
        h_rxn.index_add_(0, rxn_sub, msg)

        v = self.softplus(self.rate_mlp(h_rxn).squeeze(-1))

        contrib = sto_all * v[rxn_all]
        dxdt = torch.zeros((self.n_met,), dtype=contrib.dtype)
        dxdt.index_add_(0, met_all, contrib)
        return dxdt, v

def draw_simple_timeseries_png(
    t,
    Y,
    labels,
    out_path,
    *,
    figsize=(7, 4),
    s=18,
    alpha=0.85,
    with_lines=True,
    line_alpha=0.35,
    linewidth=1.0,
    title="Concentrations over time",
    xlabel="time",
    ylabel="concentration",
):
    """
    Matplotlib scatter (and optional lines) for time series.
    Safe for torch tensors that require grad.
    """

    # --- SAFELY convert to numpy ---
    if torch.is_tensor(t):
        t = t.detach().cpu().numpy()
    else:
        t = np.asarray(t)

    if torch.is_tensor(Y):
        Y = Y.detach().cpu().numpy()
    else:
        Y = np.asarray(Y)

    assert Y.ndim == 2, "Y must be (T, K)"
    assert t.ndim == 1 and t.shape[0] == Y.shape[0]
    assert len(labels) == Y.shape[1]

    plt.figure(figsize=figsize)

    for k in range(Y.shape[1]):
        if with_lines:
            plt.plot(t, Y[:, k], linewidth=linewidth, alpha=line_alpha)
        plt.scatter(t, Y[:, k], s=s, alpha=alpha, label=labels[k])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

# Run simulation
torch.manual_seed(0)
met_names = ["A", "B", "C", "D"]
rxn_names = ["R1:A->B", "R2:B+C->D", "R3:D->A"]

model = StoichKineticsODE()
graph = model.build_demo_graph()

x = torch.tensor([1.0, 0.2, 0.8, 0.1], dtype=torch.float32)
dt = 0.05
n_steps = 80

traj = [x.clone()]
v_traj = []
for _ in range(n_steps):
    dxdt, v = model(x, graph)
    x = torch.clamp(x + dt * dxdt, min=0.0)
    traj.append(x.clone())
    v_traj.append(v.detach().clone())

X = torch.stack(traj, dim=0)
t = torch.arange(X.shape[0], dtype=torch.float32) * dt

out_path = "./stoich_4met_3rxn_concentrations.png"

draw_simple_timeseries_png(
    t,
    X,
    ["A", "B", "C", "D"],
    out_path,
    with_lines=True
)

print("Saved plot:", out_path)
print("\nFinal concentrations:")
for i, name in enumerate(met_names):
    print(f"  {name}: {X[-1, i].item():.4f}")

print("\nFinal reaction rates (last step):")
last_v = v_traj[-1]
for j, name in enumerate(rxn_names):
    print(f"  {name}: {last_v[j].item():.4f}")
