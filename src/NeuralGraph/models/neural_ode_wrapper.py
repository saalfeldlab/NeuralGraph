"""
Neural ODE wrapper for Signal_Propagation_FlyVis.

Uses torchdiffeq's adjoint method for memory-efficient training:
- Memory O(1) in rollout steps L (vs O(L) for BPTT)
- Backward pass uses adjoint ODE solve
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint
from torch_geometric.loader import DataLoader


class GNNODEFunc(nn.Module):
    """
    Wraps GNN model as ODE vector field: dv/dt = f(t, v).

    Column layout (as in Signal_Propagation_FlyVis):
        - Column 3: neural activity (v)
        - Column 4: visual input (excitation)
    """

    def __init__(self, model, data_template, data_id, neurons_per_sample, batch_size,
                 mask_batch=None, has_visual_field=False, x_list=None,
                 run=0, device=None, k_batch=None):
        super().__init__()
        self.model = model
        self.data_template = data_template
        self.data_id = data_id
        self.neurons_per_sample = neurons_per_sample
        self.batch_size = batch_size
        self.mask_batch = mask_batch
        self.has_visual_field = has_visual_field
        self.x_list = x_list
        self.run = run
        self.device = device or torch.device('cpu')
        self.k_batch = k_batch  # per-sample k values, shape (batch_size,)
        self.delta_t = 1.0

    def set_time_params(self, delta_t):
        self.delta_t = delta_t

    def forward(self, t, v):
        """Compute dv/dt = GNN(v). Called by ODE solver at each integration step."""
        data = self.data_template.clone()
        v_reshaped = v.view(-1, 1)

        x_new = data.x.clone()
        x_new[:, 3:4] = v_reshaped

        # k_offset: discrete time step offset from continuous time t
        k_offset = int((t / self.delta_t).item()) if t.numel() == 1 else 0

        if self.has_visual_field and hasattr(self.model, 'forward_visual'):
            # For visual field, process each batch sample separately
            for b in range(self.batch_size):
                start_idx = b * self.neurons_per_sample
                end_idx = (b + 1) * self.neurons_per_sample
                k_current = int(self.k_batch[b].item()) + k_offset

                visual_input = self.model.forward_visual(x_new[start_idx:end_idx], k_current)
                n_input = getattr(self.model, 'n_input_neurons', self.neurons_per_sample)
                x_new[start_idx:start_idx + n_input, 4:5] = visual_input
                x_new[start_idx + n_input:end_idx, 4:5] = 0

        elif self.x_list is not None:
            # Update visual input for each batch sample from x_list
            for b in range(self.batch_size):
                start_idx = b * self.neurons_per_sample
                end_idx = (b + 1) * self.neurons_per_sample
                k_current = int(self.k_batch[b].item()) + k_offset

                if k_current < len(self.x_list[self.run]):
                    x_next = torch.tensor(
                        self.x_list[self.run][k_current],
                        dtype=torch.float32,
                        device=self.device
                    )
                    x_new[start_idx:end_idx, 4:5] = x_next[:, 4:5]

        data.x = x_new

        pred = self.model(
            data,
            data_id=self.data_id,
            mask=self.mask_batch,
            return_all=False
        )

        return pred.view(-1)


def integrate_neural_ode(model, v0, data_template, data_id, time_steps, delta_t,
                         neurons_per_sample, batch_size, mask_batch=None, has_visual_field=False,
                         x_list=None, run=0, device=None, k_batch=None,
                         ode_method='dopri5', rtol=1e-4, atol=1e-5,
                         adjoint=True, noise_level=0.0):
    """
    Integrate GNN dynamics using Neural ODE.

    New parameters (ODE-specific):
        ode_method : str - solver ('dopri5', 'rk4', 'euler', etc.)
        rtol, atol : float - tolerances for adaptive solvers
        adjoint : bool - use adjoint method for O(1) memory

    Returns:
        v_final : final state (N,)
        v_trajectory : states at all time points (time_steps+1, N)
    """

    # adjoint: O(1) memory, standard: faster but O(L) memory
    solver = odeint_adjoint if adjoint else odeint

    ode_func = GNNODEFunc(
        model=model,
        data_template=data_template,
        data_id=data_id,
        neurons_per_sample=neurons_per_sample,
        batch_size=batch_size,
        mask_batch=mask_batch,
        has_visual_field=has_visual_field,
        x_list=x_list,
        run=run,
        device=device,
        k_batch=k_batch
    )
    ode_func.set_time_params(delta_t)

    # t_span: evaluation points [0, dt, 2*dt, ..., time_steps*dt]
    t_span = torch.linspace(
        0, time_steps * delta_t, time_steps + 1,
        device=device, dtype=v0.dtype
    )

    # v_trajectory shape: (time_steps+1, N)
    v_trajectory = solver(
        ode_func,
        v0.flatten(),
        t_span,
        method=ode_method,
        rtol=rtol,
        atol=atol
    )

    if noise_level > 0 and model.training:
        v_trajectory = v_trajectory + noise_level * torch.randn_like(v_trajectory)

    v_final = v_trajectory[-1]

    return v_final, v_trajectory


def neural_ode_loss_flyvis(model, dataset_batch, x_list, run, k_batch,
                           time_step, batch_size, n_neurons, ids_batch,
                           delta_t, device, mask_batch=None,
                           data_id=None, has_visual_field=False,
                           y_batch=None, noise_level=0.0, ode_method='dopri5',
                           rtol=1e-4, atol=1e-5, adjoint=True):
    """
    Compute loss using Neural ODE integration.
    Replaces explicit autoregressive rollout in data_train_flyvis.
    """


    batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
    data_template = next(iter(batch_loader))

    v0 = data_template.x[:, 3].flatten()
    neurons_per_sample = dataset_batch[0].x.shape[0]

    # Extract per-sample k values (one per batch sample)
    k_per_sample = torch.tensor([
        k_batch[b * neurons_per_sample, 0].item()
        for b in range(batch_size)
    ], device=device)

    v_final, v_trajectory = integrate_neural_ode(
        model=model,
        v0=v0,
        data_template=data_template,
        data_id=data_id,
        time_steps=time_step,
        delta_t=delta_t,
        neurons_per_sample=neurons_per_sample,
        batch_size=batch_size,
        mask_batch=mask_batch,
        has_visual_field=has_visual_field,
        x_list=x_list,
        run=run,
        device=device,
        k_batch=k_per_sample,
        ode_method=ode_method,
        rtol=rtol,
        atol=atol,
        adjoint=adjoint,
        noise_level=noise_level
    )

    pred_x = v_final.view(-1, 1)
    loss = ((pred_x[ids_batch] - y_batch[ids_batch]) / (delta_t * time_step)).norm(2)

    return loss, pred_x
