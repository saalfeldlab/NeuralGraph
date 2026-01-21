"""
Post-training diagnostics for latent space models.

This module provides utilities for analyzing model performance
and behavior after training is complete.
"""

from __future__ import annotations

from enum import StrEnum, auto
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
import torch
import matplotlib.pyplot as plt
import time

if TYPE_CHECKING:
    from LatentEvolution.latent import ModelParams, LatentModel
    from LatentEvolution.load_flyvis import NeuronData


class PlotMode(StrEnum):
    """Control plotting that happens during training vs post training"""
    TRAINING = auto()
    POST_RUN = auto()

    @property
    def save_figures(self):
        return self == PlotMode.POST_RUN

    @property
    def neuron_traces(self):
        return self == PlotMode.POST_RUN


# rollout types to evaluate - set to ("latent",) to skip activity rollouts
ROLLOUT_TYPES: tuple[str, ...] = ("latent",)
# ROLLOUT_TYPES: tuple[str, ...] = ("activity", "latent")  # uncomment to include activity rollouts


def plot_recon_error_labeled(true_trace, recon_trace, neuron_data: NeuronData):
    """
    Create a labeled plot of reconstruction error with text labels for each cell type.

    This creates a single plot showing variance across neurons, colored by cell type,
    with text labels at the centroid of each cell type's points.
    Each point represents a neuron.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Compute variance across time for each neuron (axis=0 -> one value per neuron)
    var_trace = np.nanvar(true_trace, axis=0)
    err = recon_trace - true_trace
    var_err = np.nanvar(err, axis=0)

    # Create colormap with enough distinct colors for all cell types (65 types)
    num_types = len(neuron_data.TYPE_NAMES)
    cmap_names = ['tab20', 'tab20b', 'tab20c', 'Set3']
    colors = []
    for j in range(num_types):
        cmap_idx = (j // 20) % len(cmap_names)
        color_idx = j % 20
        cmap = plt.cm.get_cmap(cmap_names[cmap_idx])
        colors.append(cmap(color_idx / 20))

    # Plot each cell type and compute centroids for labels
    for type_idx, type_name in enumerate(neuron_data.TYPE_NAMES):
        neuron_indices = neuron_data.indices_per_type[type_idx]
        x_vals = var_trace[neuron_indices]
        y_vals = var_err[neuron_indices]

        # Scatter plot for this cell type
        ax.scatter(
            x_vals,
            y_vals,
            marker=".",
            alpha=0.6,
            color=colors[type_idx],
            s=30
        )

        # Compute centroid for label placement
        centroid_x = np.mean(x_vals)
        centroid_y = np.mean(y_vals)

        # Add text label at centroid with smaller font for 65 types
        # Use smaller padding and no background box for less visual clutter
        ax.text(
            centroid_x,
            centroid_y,
            type_name,
            fontsize=6,
            fontweight='bold',
            ha='center',
            va='center',
            color='black',
            bbox=dict(boxstyle='round,pad=0.2', facecolor=colors[type_idx], alpha=0.8, edgecolor='black', linewidth=0.3)
        )

    ax.set_xlabel("Raw variance", fontsize=12)
    ax.set_ylabel("Unexplained variance", fontsize=12)
    ax.set_title(f"Variance across neurons (labeled by cell type, n={num_types} types)", fontsize=14)
    fig.tight_layout()
    return fig


def plot_long_rollout_mse(
    mse_array,
    rollout_type: str,
    n_steps: int,
    n_starts: int,
    null_models: dict[str, np.ndarray] | None = None
) -> tuple[plt.Figure, dict[str, float]]:
    """
    plot mse over time for model rollout and optional null models.

    args:
        mse_array: model mse array of shape (n_starts, n_steps, n_neurons)
        rollout_type: "latent" or "activity"
        n_steps: number of rollout steps
        n_starts: number of starting points
        null_models: dict mapping legend names to mse traces (n_steps,)
    """
    # average over neurons first, then compute stats over starts
    mse_avg_over_neurons = mse_array.mean(axis=2)  # (n_starts, n_steps)

    mse_min_across_starts = mse_avg_over_neurons.min(axis=0)  # (n_steps,)
    mse_max_across_starts = mse_avg_over_neurons.max(axis=0)  # (n_steps,)
    mse_mean_across_starts = mse_avg_over_neurons.mean(axis=0)  # (n_steps,)

    # Create plot with log scale
    fig, ax = plt.subplots(figsize=(12, 6))
    time_steps = np.arange(n_steps)

    # plot null models first (so they're in background)
    if null_models:
        for label, trace in null_models.items():
            ax.plot(time_steps, trace, linewidth=2, label=label,
                    linestyle='--', alpha=0.7)

    # Plot min/max as shaded region
    ax.fill_between(
        time_steps,
        mse_min_across_starts,
        mse_max_across_starts,
        alpha=0.25,
        label='model min/max across starts',
        color='C0'
    )

    # Plot mean line
    ax.plot(time_steps, mse_mean_across_starts, linewidth=2, label='model mean across starts', color='C0')

    ax.set_xlabel('Rollout Time Steps')
    ax.set_ylabel('MSE (averaged over neurons)')
    ax.set_yscale('log')
    ax.set_ylim(1e-4, 1e1)

    # Create clear title indicating rollout type
    rollout_label = f"{rollout_type} Space Rollout"
    ax.set_title(
        f'Multi-Start Long Rollout MSE - {rollout_label}\n'
        f'{n_starts} random starts, {n_steps} steps each',
        fontsize=14,
        fontweight='bold'
    )
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()

    return fig


def compute_rollout_stability_metrics(
    mse_array: np.ndarray,
    time_units: int,
    evolve_multiple_steps: int,
    rollout_type: str,
    n_steps: int,
) -> dict[str, float]:
    """
    Compute stability metrics from rollout MSE array (worst case over starts).

    MSE is first averaged over neurons for each start point, then the worst case
    (max) over starts is used for metrics. This captures starts that blow up.

    MSE metrics (except mse_final) are computed only up to the divergence point
    (first step where mse > 1.0) to avoid polluting metrics with diverged values.

    Args:
        mse_array: Array of shape (n_starts, n_steps, n_neurons)
        time_units: Number of evolver steps per observation interval
        evolve_multiple_steps: Number of observation multiples in training
        rollout_type: "latent" or "activity" for metric naming
        n_steps: Total rollout steps

    Returns:
        Dictionary of stability metrics
    """
    # average over neurons, then take max over starts -> shape (n_steps,)
    mse_by_time = mse_array.mean(axis=2).max(axis=0)

    training_horizon = time_units * evolve_multiple_steps
    prefix = f"rollout_{rollout_type}_{n_steps}step"

    # initialize all metrics to nan
    metrics = {
        f"{prefix}_mse_at_loss_points": float("nan"),
        f"{prefix}_mse_intervening": float("nan"),
        f"{prefix}_mse_beyond_training": float("nan"),
        f"{prefix}_mse_final": float("nan"),
        f"{prefix}_first_divergence_step": float("nan"),
        f"{prefix}_mse_max": float("nan"),
    }

    # first compute divergence point (first step where mse > 1)
    divergence_indices = np.where(mse_by_time > 1.0)[0]
    if len(divergence_indices) > 0:
        first_divergence_step = int(divergence_indices[0])
    else:
        first_divergence_step = n_steps
    metrics[f"{prefix}_first_divergence_step"] = first_divergence_step

    # truncate to pre-divergence window for other metrics
    valid_steps = first_divergence_step
    mse_by_time_valid = mse_by_time[:valid_steps] if valid_steps > 0 else mse_by_time[:1]

    # 1. mse at loss points (where training loss is applied)
    # loss applied at steps: time_units, 2*time_units, ..., evolve_multiple_steps*time_units
    # in 0-indexed: time_units-1, 2*time_units-1, ...
    loss_point_indices = [m * time_units - 1 for m in range(1, evolve_multiple_steps + 1)]
    loss_point_indices = [i for i in loss_point_indices if i < valid_steps]
    if loss_point_indices:
        metrics[f"{prefix}_mse_at_loss_points"] = float(mse_by_time_valid[loss_point_indices].mean())

    # 2. mse at intervening steps (within training horizon, excluding loss points)
    # only meaningful if time_units > 1
    if time_units > 1:
        all_training_indices = set(range(min(training_horizon, valid_steps)))
        loss_point_set = set(loss_point_indices)
        intervening_indices = sorted(all_training_indices - loss_point_set)
        if intervening_indices:
            metrics[f"{prefix}_mse_intervening"] = float(mse_by_time_valid[intervening_indices].mean())

    # 3. mse beyond training horizon (but before divergence)
    if valid_steps > training_horizon:
        metrics[f"{prefix}_mse_beyond_training"] = float(mse_by_time_valid[training_horizon:].mean())

    # 4. mse at final time step (always uses full rollout, not truncated)
    metrics[f"{prefix}_mse_final"] = float(mse_by_time[-1])

    # 5. max mse in valid window (before divergence)
    metrics[f"{prefix}_mse_max"] = float(mse_by_time_valid.max())

    return metrics

def compute_multi_start_rollout_mse(
    model: LatentModel,
    val_data: torch.Tensor,
    val_stim: torch.Tensor,
    neuron_data: NeuronData,
    n_steps: int = 2000,
    n_starts: int = 10,
    rollout_type: str = "latent",
    plot_mode: PlotMode = PlotMode.TRAINING,
    time_units: int = 1,
    evolve_multiple_steps: int = 1,
) -> tuple[np.ndarray, dict[str, plt.Figure], dict[str, float]]:
    """
    Compute MSE over time from multiple random starting points.

    For each starting point, runs a rollout and computes MSE at each time step
    for each neuron. Aggregates across starting points to show MSE variability
    across neurons as a function of rollout time.

    Args:
        model: The trained LatentModel
        val_data: Validation data of shape (T, N) where N is number of neurons
        val_stim: Validation stimulus of shape (T, S)
        n_steps: Number of rollout steps (default: 2000)
        n_starts: Number of random starting points (default: 10)
        rollout_type: "latent" for latent space rollout, "activity" for activity space
        time_units: Number of evolver steps per observation interval (from config)
        evolve_multiple_steps: Number of observation multiples in training (from config)

    Returns:
        mse_array: Array of shape (n_starts, n_steps, n_neurons) with MSE at each time/neuron
        figures: figures
        metrics: Dictionary with stability metrics
    """
    assert n_starts > 0
    # Pick random starting points ensuring we have enough data
    max_start = val_data.shape[0] - n_steps - 1
    if max_start < n_starts:
        raise ValueError(
            f"Not enough data for {n_starts} starting points with {n_steps} steps. "
            f"Need at least {n_steps + n_starts + 1} time points, have {val_data.shape[0]}"
        )

    rng = np.random.default_rng(seed=0)
    start_indices = rng.choice(max_start, size=n_starts, replace=False)

    n_neurons = val_data.shape[1]
    mse_array = np.zeros((n_starts, n_steps, n_neurons))

    # null model: constant prediction x(t) = x(0)
    # accumulate mse per time step (averaged over neurons and starts)
    null_model_mse_by_time = np.zeros(n_steps)

    best_mse = float("inf")
    worst_mse = -float("inf")

    best_segment_data = None
    worst_segment_data = None

    for i, start_idx in enumerate(start_indices):

        # Get initial state and segments
        initial_state = val_data[start_idx]
        stimulus_segment = val_stim[start_idx:start_idx + n_steps]
        real_segment = val_data[start_idx + 1:start_idx + n_steps + 1]

        # compute null model mse: predict x(t) = x(0) for all t
        # error = real_segment - initial_state (broadcast automatically)
        null_squared_error = torch.pow(real_segment - initial_state, 2)
        # average over neurons for each time step, accumulate over starts
        null_model_mse_by_time += null_squared_error.mean(dim=1).detach().cpu().numpy()

        # Perform rollout
        if rollout_type == "latent":
            predicted_segment = evolve_n_steps_latent(model, initial_state, stimulus_segment, n_steps)
        else:  # activity
            predicted_segment = evolve_n_steps(model, initial_state, stimulus_segment, n_steps)

        # Compute MSE per time step per neuron (squared error, not averaged)
        squared_error = torch.pow(predicted_segment - real_segment, 2).detach().cpu().numpy()
        mse_array[i] = squared_error
        mean_mse = np.nanmean(squared_error)
        if best_segment_data is None:
            worst_segment_data = (start_idx, real_segment.detach().cpu().numpy(), predicted_segment.detach().cpu().numpy())
            best_segment_data = (start_idx, real_segment.detach().cpu().numpy(), predicted_segment.detach().cpu().numpy())
            best_mse = mean_mse
            worst_mse = mean_mse
        else:
            if mean_mse > worst_mse:
                worst_segment_data = (start_idx, real_segment.detach().cpu().numpy(), predicted_segment.detach().cpu().numpy())
                worst_mse = mean_mse
            if mean_mse < best_mse:
                best_segment_data = (start_idx, real_segment.detach().cpu().numpy(), predicted_segment.detach().cpu().numpy())
                best_mse = mean_mse
        # Clear GPU memory after each rollout
        del predicted_segment, real_segment, squared_error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # finalize null model mse (average over starts)
    null_model_mse_by_time /= n_starts

    figures = {}

    # plot worst segment
    start_idx, real_segment, predicted_segment = worst_segment_data
    if plot_mode.neuron_traces:
        figures[f"worst_{n_steps}step_rollout_{rollout_type}_traces"] = plot_rollout_traces_from_results(
                real_segment, predicted_segment, neuron_data, rollout_type, start_idx=start_idx
            )
        plt.close()
    # plot mse vs variance cell type labelled
    figures[f"worst_{n_steps}step_rollout_{rollout_type}_mse_var_scatter"] = plot_recon_error_labeled(real_segment, predicted_segment, neuron_data)

    # plot best segment
    start_idx, real_segment, predicted_segment = best_segment_data
    if plot_mode.neuron_traces:
        figures[f"best_{n_steps}step_rollout_{rollout_type}_traces"] = plot_rollout_traces_from_results(
                real_segment, predicted_segment, neuron_data, rollout_type, start_idx=start_idx
            )
        plt.close()
    # plot mse vs variance cell type labelled
    figures[f"best_{n_steps}step_rollout_{rollout_type}_mse_var_scatter"] = plot_recon_error_labeled(real_segment, predicted_segment, neuron_data)

    # plot mse over time
    null_models = {"constant prediction": null_model_mse_by_time}
    fig = plot_long_rollout_mse(mse_array, rollout_type, n_steps, n_starts, null_models)
    figures[f"multi_start_{n_steps}step_{rollout_type}_rollout_mses_by_time"] = fig
    plt.close()

    # compute stability metrics
    metrics = compute_rollout_stability_metrics(
        mse_array, time_units, evolve_multiple_steps, rollout_type, n_steps
    )

    return mse_array, figures, metrics


def run_validation_diagnostics(
    run_dir: Path,
    val_data: torch.Tensor,
    neuron_data: NeuronData,
    val_stim: torch.Tensor,
    model: LatentModel,
    config: ModelParams,
    plot_mode: PlotMode,
) -> tuple[dict[str, float|int], dict[str, plt.Figure]]:
    """
    Perform validation diagnostics on the trained model.

    Args:
        run_dir: Directory to save diagnostic figures
        val_data: Validation data tensor of shape (T_val, N)
        neuron_data: NeuronData instance with neuron type information
        val_stim: Validation stimulus tensor of shape (T_val, S)
        model: The trained LatentModel instance
        config: ModelParams configuration object
        save_figures: Whether to save figures to disk (default: False)
        skip_multi_start_rollout: Skip multi-start rollout evaluation (default: False)

    Returns:
        metrics: Dictionary of scalar metrics
        figures: Dictionary of matplotlib figures
    """

    overall_start = time.time()

    metrics = {}
    figures = {}

    # make run dir if it doesn't exist
    run_dir.mkdir(parents=True, exist_ok=True)

    time_units = config.training.time_units
    evolve_multiple_steps = config.training.evolve_multiple_steps

    # Multi-start rollout evaluation
    for rollout_type in ROLLOUT_TYPES:
        with torch.no_grad():
            _, new_figs, new_metrics = compute_multi_start_rollout_mse(
                model, val_data, val_stim, neuron_data, n_steps=2000, n_starts=20, rollout_type=rollout_type,
                plot_mode=plot_mode, time_units=time_units, evolve_multiple_steps=evolve_multiple_steps
            )
        metrics.update(new_metrics)
        figures.update(new_figs)

    if plot_mode.save_figures:
        for key, fig in figures.items():
            fig.savefig(run_dir / f"{key}.jpg", dpi=100)

    elapsed = time.time() - overall_start
    print(f"validation diagnostics complete ({elapsed:.1f}s)")
    return metrics, figures


def evolve_n_steps(model: LatentModel, initial_state: torch.Tensor, stimulus: torch.Tensor, n_steps: int) -> torch.Tensor:
    """
    Evolve the model by n time steps using the predicted state at each step.

    This performs an autoregressive rollout starting from a single initial state.
    At each step, encodes the current state, evolves in latent space, and decodes.

    Args:
        model: The LatentModel to evolve
        initial_state: Initial state tensor of shape (neurons,)
        stimulus: Stimulus tensor of shape (T, stimulus_dim) where T >= n_steps
        n_steps: Number of time steps to evolve

    Returns:
        predicted_trace: Tensor of shape (n_steps, neurons) with predicted states
    """
    predicted_trace = []
    current_state = initial_state.unsqueeze(0)  # shape (1, neurons)

    for t in range(n_steps):
        current_stimulus = stimulus[t:t+1]  # shape (1, stimulus_dim)
        next_state = model(current_state, current_stimulus)
        predicted_trace.append(next_state.squeeze(0))
        current_state = next_state

    return torch.stack(predicted_trace, dim=0)


def evolve_n_steps_latent(model: LatentModel, initial_state: torch.Tensor, stimulus: torch.Tensor, n_steps: int) -> torch.Tensor:
    """
    Evolve the model by n time steps entirely in latent space.

    This performs an autoregressive rollout starting from a single initial state.
    Encodes once, evolves in latent space for n steps, then decodes all states.

    Args:
        model: The LatentModel to evolve
        initial_state: Initial state tensor of shape (neurons,)
        stimulus: Stimulus tensor of shape (T, stimulus_dim) where T >= n_steps
        n_steps: Number of time steps to evolve

    Returns:
        predicted_trace: Tensor of shape (n_steps, neurons) with predicted states
    """
    current_latent = model.encoder(initial_state.unsqueeze(0))  # shape (1, latent_dim)
    stimulus_latent = model.stimulus_encoder(stimulus)  # shape (n_steps, stim_latent_dim)

    latent_trace = []
    for t in range(n_steps):
        current_stimulus_latent = stimulus_latent[t:t+1]  # shape (1, stim_latent_dim)
        current_latent = model.evolver(current_latent, current_stimulus_latent)
        latent_trace.append(current_latent.squeeze(0))

    latent_trace = torch.stack(latent_trace, dim=0)
    predicted_trace = model.decoder(latent_trace)

    return predicted_trace


def plot_rollout_traces_from_results(
    real_segment: torch.Tensor | np.ndarray,
    predicted_segment: torch.Tensor | np.ndarray,
    neuron_data: NeuronData,
    rollout_type: str,
    start_idx: int = 100,
) -> plt.Figure:
    """
    Plot neuron traces during multi-step rollout, one neuron per cell type.

    Creates one large figure with traces for one randomly selected neuron from each cell type.

    Args:
        real_segment: Real trace segment of shape (n_steps, neurons)
        predicted_segment: Predicted trace segment of shape (n_steps, neurons)
        neuron_data: NeuronData instance
        start_idx: Starting index used for the rollout (for labeling)

    Returns:
        Matplotlib figure with rollout traces
    """
    # Pick one neuron from each cell type (same approach as plot_neuron_reconstruction)
    rng = np.random.default_rng(seed=0)
    ntypes = len(neuron_data.TYPE_NAMES)

    fig, axes = plt.subplots(ntypes, 1, sharex=True, figsize=(24, 3 * ntypes))

    # defin ylim
    ixs = [
        rng.choice(neuron_data.indices_per_type[itype]) for itype in range(len(neuron_data.TYPE_NAMES))
    ]
    is_torch = isinstance(real_segment, torch.Tensor) and isinstance(predicted_segment, torch.Tensor)
    llim = real_segment[:, ixs].min()
    ulim = real_segment[:, ixs].max()
    if is_torch:
        llim = llim.detach().cpu().numpy()
        ulim = ulim.detach().cpu().numpy()

    # broaden a bit more
    llim -= 0.5
    ulim += 0.5

    for itype, tname in enumerate(neuron_data.TYPE_NAMES):
        ix = ixs[itype]

        if is_torch:
            real_trace_cpu = real_segment[:, ix].detach().cpu().numpy()
            pred_trace_cpu = predicted_segment[:, ix].detach().cpu().numpy()
        else:
            real_trace_cpu = real_segment[:, ix]
            pred_trace_cpu = predicted_segment[:, ix]
        # Plot real trace: black solid
        axes[itype].plot(real_trace_cpu, lw=1, color='black', label='Real')
        # ylim = axes[itype].get_ylim()

        # Plot predicted trace: orange solid
        axes[itype].plot(pred_trace_cpu, lw=1, color='#ff7f0e', label='Predicted (rollout)')

        # Error shading
        time_steps = np.arange(len(real_trace_cpu))
        axes[itype].fill_between(time_steps, real_trace_cpu, pred_trace_cpu,
                                alpha=0.2, color='#ff7f0e')
        axes[itype].set_ylim(llim, ulim)
        total_var = np.nanvar(real_trace_cpu)
        unexplained_var = np.nanvar(real_trace_cpu - pred_trace_cpu)
        r2 = 1 - unexplained_var / total_var

        axes[itype].set_title(f"{tname}: ix={int(ix)}, R2 = {r2:.2f}, var = {total_var:.2e} ({rollout_type} rollout)")
        axes[itype].legend(loc='upper right')
        axes[itype].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Rollout Time Steps")
    fig.suptitle(f"Multi-Step {rollout_type} Rollout Traces (autoregressive from single start at t={start_idx})", fontsize=16, y=1.1)
    fig.tight_layout()

    return fig
