"""
Post-training diagnostics for latent space models.

This module provides utilities for analyzing model performance
and behavior after training is complete.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
import torch
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from LatentEvolution.latent import ModelParams, LatentModel
    from LatentEvolution.load_flyvis import NeuronData

def plot_neuron_reconstruction(
    true_trace: np.ndarray,
    recon_trace: np.ndarray,
    neuron_data: NeuronData,
    xlim: tuple[int, int] | None = None,
    separate_per_type: bool = False,
    target_height_px: int = 250,
):
    """
    Plot neuron reconstruction traces.

    Args:
        true_trace: Ground truth neuron traces
        recon_trace: Reconstructed neuron traces
        neuron_data: NeuronData instance with type information
        xlim: Optional x-axis limits
        separate_per_type: If True, return dict of figures (one per neuron type).
                          If False, return single combined figure.
        target_height_px: Target height in pixels for separate per-type figures (default: 250)

    Returns:
        If separate_per_type=False: single matplotlib Figure
        If separate_per_type=True: dict mapping neuron type name to Figure
    """
    ntypes = len(neuron_data.TYPE_NAMES)
    rng = np.random.default_rng(seed=0)

    if separate_per_type:
        # Return one figure per neuron type
        # Calculate figsize based on target height in pixels
        dpi = 100
        fig_height_inches = target_height_px / dpi
        fig_width_inches = fig_height_inches * 4  # Maintain 4:1 aspect ratio

        figures = {}
        for itype, tname in enumerate(neuron_data.TYPE_NAMES):
            ix = rng.choice(neuron_data.indices_per_type[itype])

            fig, ax = plt.subplots(1, 1, figsize=(fig_width_inches, fig_height_inches), dpi=dpi)

            # Real trace: black solid
            ax.plot(true_trace[:, ix], lw=1, color='black', label="Real")
            ylim = ax.get_ylim()

            # Reconstructed trace: orange solid
            ax.plot(recon_trace[:, ix], lw=1, color='#ff7f0e', label="Point-wise reconstruction")

            # Error shading
            time_steps = np.arange(len(true_trace[:, ix]))
            ax.fill_between(time_steps, true_trace[:, ix], recon_trace[:, ix],
                           alpha=0.2, color='#ff7f0e')
            ax.set_ylim(*ylim)
            ax.set_title(f"{tname}: ix={int(ix)}")
            ax.set_xlabel("Time steps")
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

            if xlim is not None:
                ax.set_xlim(*xlim)

            fig.tight_layout()
            figures[tname] = fig

        return figures
    else:
        # Return single combined figure (original behavior)
        fig, ax = plt.subplots(ntypes, 1, sharex=True, figsize=(24, 3*ntypes))

        for itype, tname in enumerate(neuron_data.TYPE_NAMES):
            ix = rng.choice(neuron_data.indices_per_type[itype])
            title = f"{tname}: ix={int(ix)}"
            # Real trace: black solid
            ax[itype].plot(true_trace[:, ix], lw=1, color='black', label='Real')
            # fix the range based on the true trace
            ylim = ax[itype].get_ylim()

            # Reconstructed trace: orange solid
            ax[itype].plot(recon_trace[:, ix], lw=1, color='#ff7f0e', label='Point-wise reconstruction')

            # Error shading
            time_steps = np.arange(len(true_trace[:, ix]))
            ax[itype].fill_between(time_steps, true_trace[:, ix], recon_trace[:, ix],
                                  alpha=0.2, color='#ff7f0e')
            ax[itype].set_ylim(*ylim)
            ax[itype].set_title(title)
            ax[itype].legend(loc='upper right')
        if xlim is not None:
            ax[-1].set_xlim(*xlim)
        ax[-1].set_xlabel("Time steps")
        fig.tight_layout()
        return fig

def plot_recon_error(true_trace, recon_trace, neuron_data: NeuronData):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    # time variation, neuron variation
    for i in (0, 1):
        var_trace = np.var(true_trace, axis=i)
        err = recon_trace - true_trace
        var_err = np.var(err, axis=i)

        if i == 0:  # neuron variation - color by cell type
            # Create colormap with enough distinct colors for all cell types (65 types)
            # Use multiple colormaps combined for better distinction
            num_types = len(neuron_data.TYPE_NAMES)
            cmap_names = ['tab20', 'tab20b', 'tab20c', 'Set3']
            colors = []
            for j in range(num_types):
                cmap_idx = (j // 20) % len(cmap_names)
                color_idx = j % 20
                cmap = plt.cm.get_cmap(cmap_names[cmap_idx])
                colors.append(cmap(color_idx / 20))

            # Plot each cell type with its own color
            for type_idx, _type_name in enumerate(neuron_data.TYPE_NAMES):
                neuron_indices = neuron_data.indices_per_type[type_idx]
                ax[i].scatter(
                    var_trace[neuron_indices],
                    var_err[neuron_indices],
                    marker=".",
                    alpha=0.6,
                    color=colors[type_idx],
                    s=20
                )
        else:  # time variation - no coloring
            ax[i].scatter(var_trace, var_err, marker=".", alpha=0.5)

        ax[i].set_xlabel("Raw variance")
        ax[i].set_ylabel("Unexplained variance")
        ax[i].set_title("Variance across " + ["time", "neurons"][i])

    # Skip legend due to 65 cell types - would be too large
    # Instead add note that labeled version is available
    ax[1].text(0.02, 0.98, f'{num_types} cell types (see labeled plot)',
               transform=ax[1].transAxes, fontsize=8, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    fig.tight_layout()
    return fig


def plot_recon_error_labeled(true_trace, recon_trace, neuron_data: NeuronData):
    """
    Create a labeled plot of reconstruction error with text labels for each cell type.

    This creates a single plot showing variance across neurons, colored by cell type,
    with text labels at the centroid of each cell type's points.
    Each point represents a neuron.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Compute variance across time for each neuron (axis=0 -> one value per neuron)
    var_trace = np.var(true_trace, axis=0)
    err = recon_trace - true_trace
    var_err = np.var(err, axis=0)

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


@torch.compile(fullgraph=True, mode="reduce-overhead")
def evolve_many_time_steps_latent(model: LatentModel, val_data, val_stim, tmax: int):
    """
    Perform autoregressive rollout in latent space for multiple time horizons.

    For each dt from 0 to tmax-1, predicts dt steps ahead from each time point
    using autoregressive evolution in latent space (similar to evolve_n_steps
    but for multiple starting points and time horizons simultaneously).

    Args:
        model: The LatentModel
        val_data: Validation data of shape (T, neurons)
        val_stim: Validation stimulus of shape (T, stim_dim)
        tmax: Maximum number of steps to evolve

    Returns:
        recons: Reconstructed data of shape (tmax, T-tmax, neurons)
                where recons[dt, t] is the prediction for time t+dt starting from time t
    """
    with torch.no_grad():
        # Encode all data and stimulus to latent space
        proj_t = model.encoder(val_data)
        latent_dim = proj_t.shape[1]
        proj_stim_t = model.stimulus_encoder(val_stim)

        num_time_points = proj_t.shape[0] - tmax
        results = []

        for dt in range(tmax):
            if dt == 0:
                # 0-step prediction: just encode (no evolution)
                results.append(proj_t[:num_time_points])
            else:
                # dt-step prediction: autoregressive rollout in latent space
                # Start from ground truth latent state at each time point
                current_latent = proj_t[:num_time_points].clone()

                # Evolve dt steps forward autoregressively
                for step in range(dt):
                    # Get stimulus for current step (offset by step)
                    current_stim = proj_stim_t[step:step+num_time_points]
                    # Concatenate latent state and stimulus
                    evolver_input = torch.cat([current_latent, current_stim], dim=1)
                    # Evolve one step in latent space
                    evolver_output = model.evolver(evolver_input)
                    # Extract new latent state for next iteration
                    current_latent = evolver_output[:, :latent_dim]

                results.append(current_latent)

        # Stack results: (tmax, num_time_points, latent_dim)
        results = torch.stack(results, dim=0)
        # Decode all latent states at once
        recons = model.decoder(results.reshape((-1, latent_dim))).reshape((tmax, num_time_points, val_data.shape[1]))
    return recons


@torch.compile(fullgraph=True, mode="reduce-overhead")
def evolve_many_time_steps(model: LatentModel, val_data, val_stim, tmax: int):
    """
    Perform autoregressive rollout in activity space for multiple time horizons.

    For each dt from 0 to tmax-1, predicts dt steps ahead from each time point
    using autoregressive evolution in activity space (encoding and decoding at each step).
    This is the activity space equivalent of evolve_many_time_steps_latent.

    Args:
        model: The LatentModel
        val_data: Validation data of shape (T, neurons)
        val_stim: Validation stimulus of shape (T, stim_dim)
        tmax: Maximum number of steps to evolve

    Returns:
        recons: Reconstructed data of shape (tmax, T-tmax, neurons)
                where recons[dt, t] is the prediction for time t+dt starting from time t
    """
    with torch.no_grad():
        num_time_points = val_data.shape[0] - tmax
        results = []

        for dt in range(tmax):
            if dt == 0:
                # 0-step prediction: just reconstruct through encoder/decoder
                recon = model.decoder(model.encoder(val_data[:num_time_points]))
                results.append(recon)
            else:
                # dt-step prediction: autoregressive rollout in activity space
                # Start from ground truth activity state at each time point
                current_state = val_data[:num_time_points].clone()

                # Evolve dt steps forward autoregressively (encode/decode at each step)
                for step in range(dt):
                    # Get stimulus for current step (offset by step)
                    current_stim = val_stim[step:step+num_time_points]
                    # Encode current state, evolve in latent, decode back to activity
                    # This uses the model's forward pass which does: encode -> evolve -> decode
                    current_state = model(current_state, current_stim)

                results.append(current_state)

        # Stack results: (tmax, num_time_points, neurons)
        recons = torch.stack(results, dim=0)
    return recons

def plot_mses(val_data: torch.Tensor, recons: torch.Tensor, rollout_type: str = "latent"):
    """
    Plot MSE vs time steps for multi-step rollout.

    Args:
        val_data: Validation data tensor
        recons: Reconstructed data from rollout
        rollout_type: Type of rollout - "latent" for latent space rollout,
                      "activity" for activity space rollout (encode/decode at each step)

    Returns:
        fig: Matplotlib figure
        mse_metrics: Dictionary of MSE metrics
    """
    tmax = recons.shape[0]
    model_mses = torch.zeros(tmax, device=recons.device)
    for dt in range(tmax):
        model_mses[dt] = torch.nn.functional.mse_loss(recons[dt], val_data[dt:-(tmax-dt)])
    model_mses = model_mses.detach().cpu().numpy()

    # constant model serves as a null here
    null_mses = torch.zeros(tmax, device=recons.device)
    for dt in range(tmax):
        null_mses[dt] = torch.nn.functional.mse_loss(val_data[:-tmax], val_data[dt:-(tmax-dt)])
    null_mses = null_mses.detach().cpu().numpy()

    # Set labels and title based on rollout type
    if rollout_type == "latent":
        model_label = "latent space rollout"
        title = "MSE vs Time Steps (Latent Space Rollout)\nAveraged over all starting points"
        metric_prefix = "latent"
    else:  # activity
        model_label = "activity space rollout"
        title = "MSE vs Time Steps (Activity Space Rollout)\nAveraged over all starting points"
        metric_prefix = "activity"

    fig = plt.figure()
    plt.plot(np.arange(tmax), model_mses, label=model_label)
    plt.plot(np.arange(tmax), null_mses, ls="dashed", label="constant model: x(t+1)=x(t)")
    plt.legend()
    plt.xticks(np.arange(tmax))
    plt.grid(True)
    plt.xlabel("Time steps evolved")
    plt.ylabel("MSE")
    plt.title(title)
    fig.tight_layout()

    mse_metrics = {f"model_mse_evolve_{i}_steps_{metric_prefix}": float(model_mses[i]) for i in range(tmax)}
    mse_metrics.update(
        {f"null_mse_evolve_{i}_steps": float(null_mses[i]) for i in range(tmax)}
    )
    return fig, mse_metrics




def compute_per_neuron_mse(
    model: LatentModel,
    data: torch.Tensor,
    stim: torch.Tensor,
    time_units: int,
) -> np.ndarray:
    """
    Compute per-neuron MSE between model predictions and targets.

    Args:
        model: The trained LatentModel instance
        data: Data tensor of shape (T, N) where N is number of neurons
        stim: Stimulus tensor of shape (T, S)
        time_units: Number of time steps to evolve

    Returns:
        per_neuron_mse: Array of shape (N,) with MSE for each neuron
    """
    with torch.no_grad():
        x_t = data[:-time_units]
        stim_t = stim[:-time_units]
        x_t_plus = data[time_units:]
        predictions = model(x_t, stim_t)

        # Compute MSE per neuron (average over time dimension)
        per_neuron_mse = ((predictions - x_t_plus) ** 2).mean(dim=0).cpu().numpy()

    return per_neuron_mse


def compute_multi_start_rollout_mse(
    model: LatentModel,
    val_data: torch.Tensor,
    val_stim: torch.Tensor,
    n_steps: int = 2000,
    n_starts: int = 10,
    rollout_type: str = "latent",
) -> tuple[np.ndarray, plt.Figure, dict[str, float]]:
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

    Returns:
        mse_array: Array of shape (n_starts, n_steps, n_neurons) with MSE at each time/neuron
        fig: Matplotlib figure showing min/max/mean MSE across neurons vs time
        metrics: Dictionary with summary statistics
    """
    print(f"  Computing multi-start {rollout_type} rollout ({n_starts} starts, {n_steps} steps each)...")

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

    for i, start_idx in enumerate(start_indices):
        print(f"    Rollout {i+1}/{n_starts} from start_idx={start_idx}...")

        # Get initial state and segments
        initial_state = val_data[start_idx]
        stimulus_segment = val_stim[start_idx:start_idx + n_steps]
        real_segment = val_data[start_idx + 1:start_idx + n_steps + 1]

        # Perform rollout
        if rollout_type == "latent":
            predicted_segment = evolve_n_steps_latent(model, initial_state, stimulus_segment, n_steps)
        else:  # activity
            predicted_segment = evolve_n_steps(model, initial_state, stimulus_segment, n_steps)

        # Compute MSE per time step per neuron (squared error, not averaged)
        squared_error = torch.pow(predicted_segment - real_segment, 2).detach().cpu().numpy()
        mse_array[i] = squared_error

        # Clear GPU memory after each rollout
        del predicted_segment, real_segment, squared_error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Compute statistics: average over starting points, then get min/max/mean across neurons
    print("    Computing statistics across starting points and neurons...")
    mse_avg_over_starts = mse_array.mean(axis=0)  # (n_steps, n_neurons)

    mse_min_across_neurons = mse_avg_over_starts.min(axis=1)  # (n_steps,)
    mse_max_across_neurons = mse_avg_over_starts.max(axis=1)  # (n_steps,)
    mse_mean_across_neurons = mse_avg_over_starts.mean(axis=1)  # (n_steps,)

    # Create plot with log scale
    print("    Creating plot...")
    fig, ax = plt.subplots(figsize=(12, 6))
    time_steps = np.arange(n_steps)

    ax.fill_between(
        time_steps,
        mse_min_across_neurons,
        mse_max_across_neurons,
        alpha=0.3,
        label='Min/Max across neurons',
        color='C0'
    )
    ax.plot(time_steps, mse_mean_across_neurons, linewidth=2, label='Mean across neurons', color='C0')

    ax.set_xlabel('Rollout Time Steps')
    ax.set_ylabel('MSE (averaged over starting points)')
    ax.set_yscale('log')
    ax.set_title(
        f'Multi-Start Rollout MSE ({rollout_type} space)\n'
        f'{n_starts} random starts, {n_steps} steps each'
    )
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()

    # Compute summary metrics
    metrics = {
        f"multi_start_long_{rollout_type}_rollout_mses_by_time_final_mean": float(mse_mean_across_neurons[-1]),
        f"multi_start_long_{rollout_type}_rollout_mses_by_time_final_min": float(mse_min_across_neurons[-1]),
        f"multi_start_long_{rollout_type}_rollout_mses_by_time_final_max": float(mse_max_across_neurons[-1]),
        f"multi_start_long_{rollout_type}_rollout_mses_by_time_mean_over_time": float(mse_mean_across_neurons.mean()),
    }

    print(f"    Multi-start {rollout_type} rollout complete")
    return mse_array, fig, metrics


def run_validation_diagnostics(
    run_dir: Path,
    val_data: torch.Tensor,
    neuron_data: NeuronData,
    val_stim: torch.Tensor,
    model: LatentModel,
    config: ModelParams,
    save_figures: bool = False,
    skip_neuron_traces: bool = False,
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
        skip_neuron_traces: Skip generating neuron trace figures (default: False)

    Returns:
        metrics: Dictionary of scalar metrics
        figures: Dictionary of matplotlib figures
    """
    print("Running validation diagnostics...")
    print(f"  Val data shape: {val_data.shape}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    metrics = {}
    figures = {}

    true_trace = val_data.detach().cpu().numpy()
    recon_trace = model.decoder(model.encoder(val_data)).detach().cpu().numpy()

    # make run dir if it doesn't exist
    run_dir.mkdir(parents=True, exist_ok=True)

    # Neuron traces (full trace) - only generated for post-training analysis
    if not skip_neuron_traces:
        fig = plot_neuron_reconstruction(
            true_trace, recon_trace, neuron_data, xlim=None, separate_per_type=False
        )
        figures["neuron_traces"] = fig
        if save_figures:
            fig.savefig(run_dir / "neuron_traces.jpg", dpi=100)
            plt.close(fig)

    # Neuron traces (zoomed) - only generated for post-training analysis
    if not skip_neuron_traces:
        fig = plot_neuron_reconstruction(
            true_trace, recon_trace, neuron_data, xlim=(100, 1100), separate_per_type=False
        )
        figures["neuron_traces_zoom"] = fig
        if save_figures:
            fig.savefig(run_dir / "neuron_traces_zoom.jpg", dpi=100)
            plt.close(fig)

    # Reconstruction error stratified (colored by cell type)
    fig = plot_recon_error(true_trace, recon_trace, neuron_data)
    figures["reconstruction_variance"] = fig
    if save_figures:
        fig.savefig(run_dir / "reconstruction_variance.jpg", dpi=100)
        plt.close(fig)

    # Reconstruction error with cell type labels
    fig_labeled = plot_recon_error_labeled(true_trace, recon_trace, neuron_data)
    figures["reconstruction_variance_labeled"] = fig_labeled
    if save_figures:
        fig_labeled.savefig(run_dir / "reconstruction_variance_labeled.jpg", dpi=150)
        plt.close(fig_labeled)

    # MSE evolution over time steps - Latent space rollout
    recons_latent = evolve_many_time_steps_latent(model, val_data, val_stim, tmax=20)
    fig_latent, mse_metrics_latent = plot_mses(val_data, recons_latent, rollout_type="latent")
    metrics.update(mse_metrics_latent)
    figures["mses_by_time_steps_latent"] = fig_latent
    if save_figures:
        fig_latent.savefig(run_dir / "mses_by_time_steps_latent.jpg", dpi=100)
        plt.close(fig_latent)

    # Free GPU memory from latent rollout (~10 GB)
    del recons_latent
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # MSE evolution over time steps - Activity space rollout
    recons_activity = evolve_many_time_steps(model, val_data, val_stim, tmax=20)
    fig_activity, mse_metrics_activity = plot_mses(val_data, recons_activity, rollout_type="activity")
    metrics.update(mse_metrics_activity)
    figures["mses_by_time_steps_activity"] = fig_activity
    if save_figures:
        fig_activity.savefig(run_dir / "mses_by_time_steps_activity.jpg", dpi=100)
        plt.close(fig_activity)

    # Free GPU memory from activity rollout (~10 GB)
    del recons_activity
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Multi-step rollout evaluation - compute rollout once
    print("  Starting multi-step rollout evaluation...")
    with torch.no_grad():
        real_segment, predicted_segment, mse_per_step, cumulative_mse = compute_rollout(
            model, val_data, val_stim, n_steps=2000, start_idx=100
        )
    print("  Rollout computation complete")

    # Generate MSE figure (always)
    print("  Generating rollout MSE figure...")
    rollout_mse_fig, rollout_metrics = plot_rollout_mse_from_results(
        mse_per_step, cumulative_mse
    )
    metrics.update(rollout_metrics)
    figures["rollout_mse"] = rollout_mse_fig
    print("  Rollout MSE figure complete")

    # Generate neuron trace figure (only for post-training analysis)
    if not skip_neuron_traces:
        print("  Generating rollout neuron trace figure...")
        rollout_traces_fig = plot_rollout_traces_from_results(
            real_segment, predicted_segment, neuron_data, start_idx=100
        )
        figures["rollout_traces"] = rollout_traces_fig
        print("  Rollout trace figure complete")
        if save_figures:
            print("  Saving rollout trace figure to disk...")
            rollout_traces_fig.savefig(run_dir / "rollout_traces.jpg", dpi=100)
            plt.close(rollout_traces_fig)
            print("  Rollout trace figure saved")

    # Multi-step rollout evaluation in latent space
    print("  Starting multi-step rollout evaluation in latent space...")
    with torch.no_grad():
        real_segment_latent, predicted_segment_latent, mse_per_step_latent, cumulative_mse_latent = compute_rollout_latent(
            model, val_data, val_stim, n_steps=2000, start_idx=100
        )
    print("  Latent rollout computation complete")

    # Generate neuron trace figure for latent rollout (only for post-training analysis)
    if not skip_neuron_traces:
        print("  Generating latent rollout neuron trace figure...")
        rollout_latent_traces_fig = plot_rollout_traces_from_results(
            real_segment_latent, predicted_segment_latent, neuron_data, start_idx=100
        )
        figures["rollout_latent_traces"] = rollout_latent_traces_fig
        print("  Latent rollout trace figure complete")
        if save_figures:
            print("  Saving latent rollout trace figure to disk...")
            rollout_latent_traces_fig.savefig(run_dir / "rollout_latent_traces.jpg", dpi=100)
            plt.close(rollout_latent_traces_fig)
            print("  Latent rollout trace figure saved")

    # Multi-start rollout evaluation - Activity space
    print("  Starting multi-start rollout evaluation (activity space)...")
    with torch.no_grad():
        _, multi_rollout_activity_fig, multi_rollout_activity_metrics = compute_multi_start_rollout_mse(
            model, val_data, val_stim, n_steps=2000, n_starts=10, rollout_type="activity"
        )
    metrics.update(multi_rollout_activity_metrics)
    figures["multi_start_long_activity_rollout_mses_by_time"] = multi_rollout_activity_fig
    if save_figures:
        multi_rollout_activity_fig.savefig(run_dir / "multi_start_long_activity_rollout_mses_by_time.jpg", dpi=100)
        plt.close(multi_rollout_activity_fig)
    print("  Multi-start rollout (activity) complete")

    # Multi-start rollout evaluation - Latent space
    print("  Starting multi-start rollout evaluation (latent space)...")
    with torch.no_grad():
        _, multi_rollout_latent_fig, multi_rollout_latent_metrics = compute_multi_start_rollout_mse(
            model, val_data, val_stim, n_steps=2000, n_starts=10, rollout_type="latent"
        )
    metrics.update(multi_rollout_latent_metrics)
    figures["multi_start_long_latent_rollout_mses_by_time"] = multi_rollout_latent_fig
    if save_figures:
        multi_rollout_latent_fig.savefig(run_dir / "multi_start_long_latent_rollout_mses_by_time.jpg", dpi=100)
        plt.close(multi_rollout_latent_fig)
    print("  Multi-start rollout (latent) complete")

    print("Validation diagnostics complete.")
    return metrics, figures


@torch.compile(fullgraph=True, mode="reduce-overhead")
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
    print(f"    Starting rollout for {n_steps} steps...")
    predicted_trace = []
    current_state = initial_state.unsqueeze(0)  # shape (1, neurons)

    for t in range(n_steps):
        if t % 500 == 0:
            print(f"    Rollout step {t}/{n_steps}...")
        # Get the stimulus for this time step
        current_stimulus = stimulus[t:t+1]  # shape (1, stimulus_dim)

        # Evolve by one step (encodes, evolves in latent, decodes)
        next_state = model(current_state, current_stimulus)

        predicted_trace.append(next_state.squeeze(0))

        # Use predicted state as input for next step
        current_state = next_state

    print("    Rollout steps complete, stacking results...")
    return torch.stack(predicted_trace, dim=0)


@torch.compile(fullgraph=True, mode="reduce-overhead")
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
    print(f"    Starting latent rollout for {n_steps} steps...")

    # Encode initial state to latent space
    current_latent = model.encoder(initial_state.unsqueeze(0))  # shape (1, latent_dim)
    latent_dim = current_latent.shape[1]

    # Encode all stimulus
    stimulus_latent = model.stimulus_encoder(stimulus)  # shape (n_steps, stim_latent_dim)

    # Collect latent states
    latent_trace = []

    for t in range(n_steps):
        if t % 500 == 0:
            print(f"    Latent rollout step {t}/{n_steps}...")

        # Get the stimulus for this time step
        current_stimulus_latent = stimulus_latent[t:t+1]  # shape (1, stim_latent_dim)

        # Concatenate latent state and stimulus
        evolver_input = torch.cat([current_latent, current_stimulus_latent], dim=1)

        # Evolve one step in latent space
        evolver_output = model.evolver(evolver_input)

        # Extract new latent state
        current_latent = evolver_output[:, :latent_dim]

        latent_trace.append(current_latent.squeeze(0))

    print("    Latent rollout steps complete, stacking and decoding...")
    # Stack all latent states: (n_steps, latent_dim)
    latent_trace = torch.stack(latent_trace, dim=0)

    # Decode all latent states at once
    predicted_trace = model.decoder(latent_trace)

    return predicted_trace


def compute_rollout(
    model: LatentModel,
    real_trace: torch.Tensor,
    stimulus: torch.Tensor,
    n_steps: int,
    start_idx: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute multi-step rollout from a single starting point.

    Performs autoregressive rollout and compares with ground truth.

    Args:
        model: The LatentModel to evolve
        real_trace: Real state tensor of shape (T, neurons)
        stimulus: Stimulus tensor of shape (T, stimulus_dim)
        n_steps: Number of time steps to predict
        start_idx: Starting index in the trace

    Returns:
        real_segment: Real trace segment of shape (n_steps, neurons)
        predicted_segment: Predicted trace segment of shape (n_steps, neurons)
        mse_per_step: MSE at each time step, shape (n_steps,)
        cumulative_mse: Cumulative average MSE up to each time step, shape (n_steps,)
    """
    print(f"    Preparing rollout data (n_steps={n_steps}, start_idx={start_idx})...")
    # Get initial state
    initial_state = real_trace[start_idx]

    # Get stimulus segment
    stimulus_segment = stimulus[start_idx:start_idx + n_steps]

    # Get real trace segment (ground truth for the next n_steps)
    real_segment = real_trace[start_idx + 1:start_idx + n_steps + 1]

    # Predict n steps (autoregressive rollout)
    predicted_segment = evolve_n_steps(model, initial_state, stimulus_segment, n_steps)

    print("    Computing MSE metrics...")
    # Compute MSE per time step
    mse_per_step = torch.pow(predicted_segment - real_segment, 2).mean(dim=1)

    # Compute cumulative MSE
    cumulative_mse = torch.cumsum(mse_per_step, dim=0) / torch.arange(1, n_steps + 1, device=mse_per_step.device)

    return real_segment, predicted_segment, mse_per_step, cumulative_mse


def compute_rollout_latent(
    model: LatentModel,
    real_trace: torch.Tensor,
    stimulus: torch.Tensor,
    n_steps: int,
    start_idx: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute multi-step rollout from a single starting point, entirely in latent space.

    Performs autoregressive rollout in latent space and compares with ground truth.
    Unlike compute_rollout, this encodes once, evolves in latent space, then decodes all.

    Args:
        model: The LatentModel to evolve
        real_trace: Real state tensor of shape (T, neurons)
        stimulus: Stimulus tensor of shape (T, stimulus_dim)
        n_steps: Number of time steps to predict
        start_idx: Starting index in the trace

    Returns:
        real_segment: Real trace segment of shape (n_steps, neurons)
        predicted_segment: Predicted trace segment of shape (n_steps, neurons)
        mse_per_step: MSE at each time step, shape (n_steps,)
        cumulative_mse: Cumulative average MSE up to each time step, shape (n_steps,)
    """
    print(f"    Preparing latent rollout data (n_steps={n_steps}, start_idx={start_idx})...")
    # Get initial state
    initial_state = real_trace[start_idx]

    # Get stimulus segment
    stimulus_segment = stimulus[start_idx:start_idx + n_steps]

    # Get real trace segment (ground truth for the next n_steps)
    real_segment = real_trace[start_idx + 1:start_idx + n_steps + 1]

    # Predict n steps (autoregressive rollout in latent space)
    predicted_segment = evolve_n_steps_latent(model, initial_state, stimulus_segment, n_steps)

    print("    Computing MSE metrics...")
    # Compute MSE per time step
    mse_per_step = torch.pow(predicted_segment - real_segment, 2).mean(dim=1)

    # Compute cumulative MSE
    cumulative_mse = torch.cumsum(mse_per_step, dim=0) / torch.arange(1, n_steps + 1, device=mse_per_step.device)

    return real_segment, predicted_segment, mse_per_step, cumulative_mse


def plot_rollout_mse_from_results(
    mse_per_step: torch.Tensor,
    cumulative_mse: torch.Tensor,
) -> tuple[plt.Figure, dict[str, float]]:
    """
    Plot MSE growth during multi-step rollout from precomputed results.

    Args:
        mse_per_step: MSE at each rollout step, shape (n_steps,)
        cumulative_mse: Cumulative average MSE, shape (n_steps,)

    Returns:
        fig: Matplotlib figure with MSE plots
        metrics: Dictionary with rollout MSE metrics
    """
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot MSE per time step
    axes[0].plot(mse_per_step.detach().cpu().numpy())
    axes[0].set_xlabel('Rollout Time Step')
    axes[0].set_ylabel('MSE (averaged over neurons)')
    axes[0].set_title('MSE at Each Rollout Step\n(autoregressive from single start)')
    axes[0].grid(True, alpha=0.3)

    # Plot cumulative MSE
    axes[1].plot(cumulative_mse.detach().cpu().numpy(), color='orange')
    axes[1].set_xlabel('Rollout Time Step')
    axes[1].set_ylabel('Cumulative Average MSE')
    axes[1].set_title('Cumulative Average MSE During Rollout\n(autoregressive from single start)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Compute metrics
    metrics = {
        "rollout_final_mse": float(mse_per_step[-1].item()),
        "rollout_cumulative_mse": float(cumulative_mse[-1].item()),
        "rollout_mean_mse": float(mse_per_step.mean().item()),
    }

    return fig, metrics


def plot_rollout_traces_from_results(
    real_segment: torch.Tensor,
    predicted_segment: torch.Tensor,
    neuron_data: NeuronData,
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

    print(f"    Creating figure with {ntypes} subplots, size=(24, {3 * ntypes})...")
    # Create one large figure
    fig, axes = plt.subplots(ntypes, 1, sharex=True, figsize=(24, 3 * ntypes))

    print(f"    Plotting {ntypes} neuron traces...")
    for itype, tname in enumerate(neuron_data.TYPE_NAMES):
        if itype % 10 == 0:
            print(f"      Plotting cell type {itype}/{ntypes}...")
        ix = rng.choice(neuron_data.indices_per_type[itype])

        real_trace_cpu = real_segment[:, ix].detach().cpu().numpy()
        pred_trace_cpu = predicted_segment[:, ix].detach().cpu().numpy()

        # Plot real trace: black solid
        axes[itype].plot(real_trace_cpu, lw=1, color='black', label='Real')
        ylim = axes[itype].get_ylim()

        # Plot predicted trace: orange solid
        axes[itype].plot(pred_trace_cpu, lw=1, color='#ff7f0e', label='Predicted (rollout)')

        # Error shading
        time_steps = np.arange(len(real_trace_cpu))
        axes[itype].fill_between(time_steps, real_trace_cpu, pred_trace_cpu,
                                alpha=0.2, color='#ff7f0e')
        axes[itype].set_ylim(*ylim)
        axes[itype].set_title(f"{tname}: ix={int(ix)} (autoregressive rollout from t={start_idx})")
        axes[itype].legend(loc='upper right')
        axes[itype].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Rollout Time Steps")
    fig.suptitle(f"Multi-Step Rollout Traces (autoregressive from single start at t={start_idx})", fontsize=16, y=0.995)

    print("    Running tight_layout()...")
    fig.tight_layout()

    return fig
