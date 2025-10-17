import torch
import torch.nn as nn
from NeuralGraph.models.MLP import MLP
from torchdiffeq import odeint

class Signal_Propagation_MLP_NODE(nn.Module):
    """Neural ODE baseline: dv/dt = f_θ(v, I)"""
    
    def __init__(self, aggr_type='add', config=None, device=None):
        super(Signal_Propagation_MLP_NODE, self).__init__()

        """Neural ODE baseline for neural dynamics prediction
        
        Reference:
        - Neural Ordinary Differential Equations (Chen et al., NeurIPS 2018)
        https://arxiv.org/abs/1806.07366
        
        Model: dv/dt = MLP_θ([v₁, ..., vₙ, I₁, ..., Iₘ])
        
        Pure blackbox approach - no connectivity structure.
        Learns dynamics directly from flattened state + inputs.
        """
        
        simulation_config = config.simulation
        model_config = config.graph_model
        self.device = device
        
        self.n_neurons = simulation_config.n_neurons
        self.n_input_neurons = simulation_config.n_input_neurons
        self.calcium_type = simulation_config.calcium_type
        self.delta_t = simulation_config.delta_t
        
        # ODE function: f(t, v) predicts dv/dt
        input_dim = self.n_neurons + self.n_input_neurons
        
        self.ode_func = MLP(
            input_size=input_dim,
            output_size=self.n_neurons,
            nlayers=model_config.n_layers,
            hidden_size=model_config.hidden_dim,
            activation=model_config.MLP_activation,
            device=self.device,
        )
        
        self.a = nn.Parameter(
            torch.randn(self.n_neurons, model_config.embedding_dim).to(self.device),
            requires_grad=True,
        )
    
    def f(self, t, v_flat, I_flat):
        """ODE function: dv/dt = f(v, I)"""
        x = torch.cat([v_flat, I_flat])
        return self.ode_func(x)
    
    def forward(self, x=[], data_id=[], mask=[], k=[], return_all=False):
        # Extract state
        if self.calcium_type != "none":
            v = x[:, 7:8]
        else:
            v = x[:, 3:4]
        
        excitation = x[:self.n_input_neurons, 4:5]
        
        v_flat = v.flatten()
        I_flat = excitation.flatten()
        
        # Single Euler step (match GNN/MLP training)
        dv_dt = self.f(None, v_flat, I_flat)
        
        return dv_dt.view(-1, 1)
    
    def rollout_step(self, v, I, dt, method='euler'):
        """Single integration step for rollout"""
        in_features = torch.cat([v.flatten(), I.flatten()])
        k1 = self.ode_func(in_features)
        
        if method == 'euler':
            v_next = v + dt * k1.view(-1, 1)
        elif method == 'rk4':  # 4th order Runge-Kutta
            k2 = self.ode_func(torch.cat([(v + 0.5*dt*k1.view(-1,1)).flatten(), I.flatten()]))
            k3 = self.ode_func(torch.cat([(v + 0.5*dt*k2.view(-1,1)).flatten(), I.flatten()]))
            k4 = self.ode_func(torch.cat([(v + dt*k3.view(-1,1)).flatten(), I.flatten()]))
            v_next = v + (dt/6) * (k1 + 2*k2 + 2*k3 + k4).view(-1, 1)
        
        return v_next