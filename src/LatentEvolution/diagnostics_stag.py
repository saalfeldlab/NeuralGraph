"""
Validation diagnostics for LatentStagModel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
import torch
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from LatentEvolution.latent_stag import LatentStagModel, StagModelParams

from torch.utils.tensorboard import SummaryWriter
from LatentEvolution.diagnostics import (
    compute_linear_interpolation_baseline,
    compute_rollout_stability_metrics,
    plot_long_rollout_mse,
    plot_time_aligned_mse,
)


def optimize_z0_batched(
    model: LatentStagModel,
    target_data: torch.Tensor,
    stimulus: torch.Tensor,
    latent_dims: int,
    num_starts: int,
    start_indices: list[int],
    fit_window: int,
    n_steps: int = 2,
    lr: float = 1.0,
) -> torch.Tensor:
    """
    optimize z0 for multiple start points in parallel using LBFGS.
    """
    device = target_data.device
    stim_dims = stimulus.shape[1]

    # extract fit windows for all starts
    fit_targets = torch.stack([
        target_data[idx + 1 : idx + 1 + fit_window]
        for idx in start_indices
    ], dim=0)  # (num_starts, fit_window, num_neurons)

    fit_stims = torch.stack([
        stimulus[idx : idx + fit_window]
        for idx in start_indices
    ], dim=0)  # (num_starts, fit_window, stim_dims)

    # pre-encode stimulus
    with torch.no_grad():
        stim_flat = fit_stims.reshape(-1, stim_dims)
        stim_latent_flat = model.stimulus_encoder(stim_flat)
        stim_latent_dim = stim_latent_flat.shape[-1]
        stim_latent = stim_latent_flat.reshape(num_starts, fit_window, stim_latent_dim)

    # learnable z0
    z0 = torch.randn(num_starts, latent_dims, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([z0], lr=lr, max_iter=20, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        z = z0
        predictions = []
        for t in range(fit_window):
            z = model.evolver(z, stim_latent[:, t, :])
            pred = model.decoder(z)
            predictions.append(pred)
        pred_trace = torch.stack(predictions, dim=1)
        loss = torch.nn.functional.mse_loss(pred_trace, fit_targets)
        loss.backward()
        return loss

    for _ in range(n_steps):
        optimizer.step(closure)

    return z0.detach()


def rollout_from_z0_batched(
    model: LatentStagModel,
    z0: torch.Tensor,
    stimulus: torch.Tensor,
    start_indices: list[int],
    n_steps: int,
) -> torch.Tensor:
    """
    rollout from z0 for multiple start points in parallel.
    """
    num_starts = z0.shape[0]
    latent_dims = z0.shape[1]
    stim_dims = stimulus.shape[1]

    # extract rollout stimulus for all starts
    rollout_stims = torch.stack([
        stimulus[idx : idx + n_steps]
        for idx in start_indices
    ], dim=0)

    # pre-encode stimulus
    with torch.no_grad():
        stim_flat = rollout_stims.reshape(-1, stim_dims)
        stim_latent_flat = model.stimulus_encoder(stim_flat)
        stim_latent = stim_latent_flat.reshape(num_starts, n_steps, -1)

        z = z0
        latent_trace = []
        for t in range(n_steps):
            z = model.evolver(z, stim_latent[:, t, :])
            latent_trace.append(z)
        latent_trace = torch.stack(latent_trace, dim=1)

        latent_flat = latent_trace.reshape(-1, latent_dims)
        rollout_flat = model.decoder(latent_flat)
        num_neurons = rollout_flat.shape[-1]
        rollout = rollout_flat.reshape(num_starts, n_steps, num_neurons)

    return rollout


def compute_constant_baseline(
    val_data: torch.Tensor,
    start_indices: list[int],
    n_steps: int,
) -> np.ndarray:
    """
    compute constant prediction baseline: x(t) = x(0).
    """
    mse_sum = np.zeros(n_steps)
    for idx in start_indices:
        x_0 = val_data[idx]
        gt = val_data[idx + 1 : idx + 1 + n_steps]
        pred = x_0.unsqueeze(0).expand(n_steps, -1)
        mse = torch.pow(pred - gt, 2).mean(dim=1).cpu().numpy()
        mse_sum += mse
    return mse_sum / len(start_indices)


def run_validation_diagnostics(
    val_data: torch.Tensor,
    val_stim: torch.Tensor,
    model: LatentStagModel,
    cfg: StagModelParams,
    writer: SummaryWriter,
    epoch: int,
    dataset_name: str = "validation",
    n_starts: int = 10,
    n_rollout_steps: int = 2000,
    fit_window: int | None = None,
) -> dict[str, float]:
    """
    run validation diagnostics for LatentStagModel.
    """
    time_units = cfg.training.time_units
    evolve_multiple_steps = cfg.training.evolve_multiple_steps

    if fit_window is None:
        fit_window = time_units * evolve_multiple_steps

    # pick random start indices
    max_start = val_data.shape[0] - n_rollout_steps - fit_window - 1
    rng = np.random.default_rng(seed=epoch)
    start_indices = sorted(rng.integers(0, max_start, size=n_starts).tolist())

    # optimize z0 and rollout
    model.eval()
    z0 = optimize_z0_batched(
        model=model,
        target_data=val_data,
        stimulus=val_stim,
        latent_dims=cfg.latent_dims,
        num_starts=n_starts,
        start_indices=start_indices,
        fit_window=fit_window,
    )

    rollout = rollout_from_z0_batched(
        model=model,
        z0=z0,
        stimulus=val_stim,
        start_indices=start_indices,
        n_steps=n_rollout_steps,
    )

    # compute mse: (n_starts, n_steps, n_neurons)
    gt = torch.stack([
        val_data[idx + 1 : idx + 1 + n_rollout_steps]
        for idx in start_indices
    ], dim=0)
    mse_array = torch.pow(rollout - gt, 2).cpu().numpy()  # (n_starts, n_steps, n_neurons)

    # baselines
    constant_baseline = compute_constant_baseline(val_data, start_indices, n_rollout_steps)
    linear_interp_baseline = compute_linear_interpolation_baseline(
        val_data, start_indices, n_rollout_steps, time_units, evolve_multiple_steps
    )

    # determine tensorboard prefix (match latent.py convention)
    if dataset_name == "validation":
        tb_prefix = "Val"
    elif dataset_name.startswith("cv_"):
        tb_prefix = f"CrossVal/{dataset_name[3:]}"
    else:
        tb_prefix = dataset_name

    # metrics (average over neurons first, then over starts)
    mse_avg_neurons = mse_array.mean(axis=2)  # (n_starts, n_steps)
    total_steps = time_units * evolve_multiple_steps
    mse_fit = mse_avg_neurons[:, :total_steps].mean()
    mse_beyond = mse_avg_neurons[:, total_steps:].mean()
    metrics = {
        f"{tb_prefix}/mse_fit_window": float(mse_fit),
        f"{tb_prefix}/mse_beyond": float(mse_beyond),
        f"{tb_prefix}/mse_overall": float(mse_avg_neurons.mean()),
    }

    for key, val in metrics.items():
        writer.add_scalar(key, val, epoch)

    # long rollout plot (reuse from diagnostics.py)
    null_models = {"constant baseline": constant_baseline}
    fig_long = plot_long_rollout_mse(
        mse_array=mse_array,
        rollout_type="latent",
        n_steps=n_rollout_steps,
        n_starts=n_starts,
        null_models=null_models,
    )
    writer.add_figure(f"{tb_prefix}/multi_start_{n_rollout_steps}step_latent_rollout_mses_by_time", fig_long, epoch)
    plt.close(fig_long)

    # compute and log stability metrics
    stability_metrics = compute_rollout_stability_metrics(
        mse_array=mse_array,
        time_units=time_units,
        evolve_multiple_steps=evolve_multiple_steps,
        rollout_type="latent",
        n_steps=n_rollout_steps,
    )
    for metric_name, metric_value in stability_metrics.items():
        metrics[f"{tb_prefix}/{metric_name}"] = metric_value
        writer.add_scalar(f"{tb_prefix}/{metric_name}", metric_value, epoch)

    # zoomed time-aligned plot (reuse from diagnostics.py)
    fig_zoomed = plot_time_aligned_mse(
        mse_array=mse_array,
        constant_baseline=constant_baseline,
        linear_interp_baseline=linear_interp_baseline,
        time_units=time_units,
        evolve_multiple_steps=evolve_multiple_steps,
        rollout_type="latent",
    )
    writer.add_figure(f"{tb_prefix}/time_aligned_mse_latent", fig_zoomed, epoch)
    plt.close(fig_zoomed)

    print(f"  {dataset_name}: mse_fit={mse_fit:.4e}, mse_beyond={mse_beyond:.4e}")

    return metrics
