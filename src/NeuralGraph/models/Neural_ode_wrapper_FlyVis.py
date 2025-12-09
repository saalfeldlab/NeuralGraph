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

# Debug flag - set to True to enable debug prints
DEBUG_ODE = True


class GNNODEFunc_FlyVis(nn.Module):
    """
    Wraps GNN model as ODE vector field: dv/dt = f(t, v).

    Column layout (as in Signal_Propagation_FlyVis):
        - Column 3: neural activity (v)
        - Column 4: visual input (excitation)
    """

    def __init__(self, model, data_template, data_id, neurons_per_sample, batch_size,
                 has_visual_field=False, x_list=None,
                 run=0, device=None, k_batch=None, state_clamp=10.0, stab_lambda=0.0):
        super().__init__()
        self.model = model
        self.data_template = data_template
        self.data_id = data_id
        self.neurons_per_sample = neurons_per_sample
        self.batch_size = batch_size
        self.has_visual_field = has_visual_field
        self.x_list = x_list
        self.run = run
        self.device = device or torch.device('cpu')
        self.k_batch = k_batch  # per-sample k values, shape (batch_size,)
        self.delta_t = 1.0
        self.state_clamp = state_clamp  # clamp state to [-state_clamp, state_clamp]
        self.stab_lambda = stab_lambda  # damping: dv = GNN(v) - lambda*v

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
            return_all=False
        )

        dv = pred.view(-1)

        return dv


def integrate_neural_ode_FlyVis(model, v0, data_template, data_id, time_steps, delta_t,
                         neurons_per_sample, batch_size, has_visual_field=False,
                         x_list=None, run=0, device=None, k_batch=None,
                         ode_method='dopri5', rtol=1e-4, atol=1e-5,
                         adjoint=True, noise_level=0.0, state_clamp=10.0, stab_lambda=0.0):
    """
    Integrate GNN dynamics using Neural ODE.

    New parameters (ODE-specific):
        ode_method : str - solver ('dopri5', 'rk4', 'euler', etc.)
        rtol, atol : float - tolerances for adaptive solvers
        adjoint : bool - use adjoint method for O(1) memory
        state_clamp : float - clamp state to [-state_clamp, state_clamp] during integration
        stab_lambda : float - damping coefficient for stability

    Returns:
        v_final : final state (N,)
        v_trajectory : states at all time points (time_steps+1, N)
    """

    # adjoint: O(1) memory, standard: faster but O(L) memory
    solver = odeint_adjoint if adjoint else odeint

    ode_func = GNNODEFunc_FlyVis(
        model=model,
        data_template=data_template,
        data_id=data_id,
        neurons_per_sample=neurons_per_sample,
        batch_size=batch_size,
        has_visual_field=has_visual_field,
        x_list=x_list,
        run=run,
        device=device,
        k_batch=k_batch,
        state_clamp=state_clamp,
        stab_lambda=stab_lambda
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


def neural_ode_loss_FlyVis(model, dataset_batch, x_list, run, k_batch,
                           time_step, batch_size, n_neurons, ids_batch,
                           delta_t, device,
                           data_id=None, has_visual_field=False,
                           y_batch=None, noise_level=0.0, ode_method='dopri5',
                           rtol=1e-4, atol=1e-5, adjoint=True,
                           iteration=0, state_clamp=10.0, stab_lambda=0.0):
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

    if DEBUG_ODE and (iteration % 500 == 0):
        print(f"\n=== Neural ODE Debug (iter {iteration}) ===")
        print(f"time_step={time_step}, delta_t={delta_t}, batch_size={batch_size}")
        print(f"neurons_per_sample={neurons_per_sample}, n_neurons={n_neurons}")
        print(f"v0 shape={v0.shape}, v0 mean={v0.mean().item():.4f}, std={v0.std().item():.4f}")
        print(f"k_per_sample={k_per_sample.tolist()}")
        print(f"data_template.x shape={data_template.x.shape}")
        print(f"data_template.edge_index shape={data_template.edge_index.shape}")
        print(f"ode_method={ode_method}, adjoint={adjoint}")

    v_final, v_trajectory = integrate_neural_ode_FlyVis(
        model=model,
        v0=v0,
        data_template=data_template,
        data_id=data_id,
        time_steps=time_step,
        delta_t=delta_t,
        neurons_per_sample=neurons_per_sample,
        batch_size=batch_size,
        has_visual_field=has_visual_field,
        x_list=x_list,
        run=run,
        device=device,
        k_batch=k_per_sample,
        ode_method=ode_method,
        rtol=rtol,
        atol=atol,
        adjoint=adjoint,
        noise_level=noise_level,
        state_clamp=state_clamp,
        stab_lambda=stab_lambda
    )

    pred_x = v_final.view(-1, 1)
    loss = ((pred_x[ids_batch] - y_batch[ids_batch]) / (delta_t * time_step)).norm(2)

    if DEBUG_ODE and (iteration % 500 == 0):
        print(f"v_final mean={v_final.mean().item():.4f}, std={v_final.std().item():.4f}")
        print(f"y_batch mean={y_batch[ids_batch].mean().item():.4f}, std={y_batch[ids_batch].std().item():.4f}")
        print(f"pred_x mean={pred_x[ids_batch].mean().item():.4f}, std={pred_x[ids_batch].std().item():.4f}")
        print(f"loss={loss.item():.4f}")
        print("=== End Debug ===\n")

    return loss, pred_x


def debug_verify_forward_pass(model, data_template, data_id, device):
    """
    Debug function to verify the forward pass produces valid outputs.
    Run this once at start of training to sanity-check model behavior.
    """
    print("\n=== Forward Pass Verification ===")

    with torch.no_grad():
        pred = model(data_template, data_id=data_id, return_all=False)

    print(f"Input v (col 3): mean={data_template.x[:, 3].mean().item():.6f}, "
          f"std={data_template.x[:, 3].std().item():.6f}")
    print(f"Input excitation (col 4): mean={data_template.x[:, 4].mean().item():.6f}, "
          f"std={data_template.x[:, 4].std().item():.6f}")
    print(f"Output pred (dv/dt): mean={pred.mean().item():.6f}, std={pred.std().item():.6f}")

    if hasattr(model, 'W'):
        print(f"W: shape={model.W.shape}, mean={model.W.mean().item():.6f}, "
              f"std={model.W.std().item():.6f}, min={model.W.min().item():.6f}, "
              f"max={model.W.max().item():.6f}")

    print(f"edge_index: shape={data_template.edge_index.shape}")
    print("=== End Forward Pass Verification ===\n")


def debug_check_gradients(model, loss, iteration):
    """
    Debug function to check gradient flow after loss.backward().
    Call this after loss.backward() to verify gradients are flowing properly.
    """
    print(f"\n=== Gradient Check (iter {iteration}) ===")

    # Check W gradients (the key parameter we're training)
    if hasattr(model, 'W'):
        if model.W.grad is not None:
            w_grad = model.W.grad
            print(f"W.grad: shape={w_grad.shape}, mean={w_grad.mean().item():.8f}, "
                  f"std={w_grad.std().item():.8f}, max={w_grad.abs().max().item():.8f}, "
                  f"nonzero={(w_grad.abs() > 1e-10).sum().item()}/{w_grad.numel()}")
        else:
            print("W.grad is None - NO GRADIENT FLOWING TO W!")

    # Check phi_edge network gradients
    if hasattr(model, 'phi_edge'):
        phi_grads = []
        for name, param in model.phi_edge.named_parameters():
            if param.grad is not None:
                phi_grads.append(param.grad.abs().mean().item())
        if phi_grads:
            print(f"phi_edge grads: mean={sum(phi_grads)/len(phi_grads):.8f}")
        else:
            print("phi_edge: No gradients!")

    # Check embedding gradients
    if hasattr(model, 'embedding'):
        if model.embedding.weight.grad is not None:
            emb_grad = model.embedding.weight.grad
            print(f"embedding.grad: mean={emb_grad.mean().item():.8f}, "
                  f"std={emb_grad.std().item():.8f}")
        else:
            print("embedding.grad is None")

    print(f"loss value: {loss.item():.6f}")
    print("=== End Gradient Check ===\n")


def debug_compare_ode_vs_recurrent(model, dataset_batch, x_list, run, k_batch,
                                    time_step, batch_size, n_neurons, ids_batch,
                                    delta_t, device, data_id, x_batch):
    """
    Debug function to compare ODE integration vs manual Euler steps.
    Call this to verify ODE wrapper produces same results as recurrent.
    """
    print("\n" + "="*60)
    print("DEBUG: Comparing ODE vs Recurrent (Euler) integration")
    print("="*60)

    batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
    data_template = next(iter(batch_loader))
    neurons_per_sample = dataset_batch[0].x.shape[0]

    # Get k values
    k_per_sample = torch.tensor([
        k_batch[b * neurons_per_sample, 0].item()
        for b in range(batch_size)
    ], device=device)

    v0 = data_template.x[:, 3].flatten()

    print(f"Initial v0: mean={v0.mean().item():.6f}, std={v0.std().item():.6f}")
    print(f"time_step={time_step}, delta_t={delta_t}")

    # --- Manual Euler integration (like recurrent training) ---
    print("\n--- Manual Euler Integration ---")
    pred_x_euler = x_batch[:, 0:1].clone()

    for step in range(time_step):
        # Create data with current state
        dataset_batch_step = []
        for b in range(batch_size):
            start_idx = b * neurons_per_sample
            end_idx = (b + 1) * neurons_per_sample
            dataset_batch[b].x[:, 3:4] = pred_x_euler[start_idx:end_idx].reshape(-1, 1)

            # Update visual input
            k_current = int(k_per_sample[b].item()) + step
            if k_current < len(x_list[run]):
                x_next = torch.tensor(x_list[run][k_current], dtype=torch.float32, device=device)
                dataset_batch[b].x[:, 4:5] = x_next[:, 4:5]

            dataset_batch_step.append(dataset_batch[b])

        batch_loader_step = DataLoader(dataset_batch_step, batch_size=batch_size, shuffle=False)
        for batch in batch_loader_step:
            with torch.no_grad():
                pred = model(batch, data_id=data_id, return_all=False)

        pred_x_euler = pred_x_euler + delta_t * pred
        print(f"  Step {step}: pred mean={pred.mean().item():.6f}, pred_x mean={pred_x_euler.mean().item():.6f}")

    # --- ODE integration ---
    print("\n--- ODE Integration ---")
    with torch.no_grad():
        v_final_ode, v_trajectory = integrate_neural_ode_FlyVis(
            model=model,
            v0=v0,
            data_template=data_template,
            data_id=data_id,
            time_steps=time_step,
            delta_t=delta_t,
            neurons_per_sample=neurons_per_sample,
            batch_size=batch_size,
            has_visual_field=False,
            x_list=x_list,
            run=run,
            device=device,
            k_batch=k_per_sample,
            ode_method='euler',
            rtol=1e-4,
            atol=1e-5,
            adjoint=False,
            noise_level=0.0
        )

    for t_idx in range(v_trajectory.shape[0]):
        print(f"  t={t_idx}: v mean={v_trajectory[t_idx].mean().item():.6f}")

    pred_x_ode = v_final_ode.view(-1, 1)

    # --- Compare ---
    print("\n--- Comparison ---")
    print(f"Euler final: mean={pred_x_euler[ids_batch].mean().item():.6f}, std={pred_x_euler[ids_batch].std().item():.6f}")
    print(f"ODE final:   mean={pred_x_ode[ids_batch].mean().item():.6f}, std={pred_x_ode[ids_batch].std().item():.6f}")

    diff = (pred_x_euler[ids_batch] - pred_x_ode[ids_batch]).abs()
    print(f"Difference:  mean={diff.mean().item():.6f}, max={diff.max().item():.6f}")

    # Check W gradients
    if hasattr(model, 'W') and model.W.grad is not None:
        print(f"\nW grad: mean={model.W.grad.mean().item():.6f}, std={model.W.grad.std().item():.6f}")

    print("="*60 + "\n")

    return pred_x_euler, pred_x_ode
