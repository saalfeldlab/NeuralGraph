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

            # Plot true trace
            p = ax.plot(true_trace[:, ix], lw=3, alpha=0.5, label="True")
            ylim = ax.get_ylim()

            # Plot reconstructed trace
            ax.plot(recon_trace[:, ix], color=p[-1].get_color(), label="Reconstructed")
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
            p = ax[itype].plot(true_trace[:, ix], lw=3, alpha=0.5)
            # fix the range based on the true trace
            ylim = ax[itype].get_ylim()

            # reconstructed trace
            ax[itype].plot(recon_trace[:, ix], color=p[-1].get_color())
            ax[itype].set_ylim(*ylim)
            ax[itype].set_title(title)
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
def evolve_many_time_steps(model: LatentModel, val_data, val_stim, tmax: int):
    with torch.no_grad():
        proj_t = model.encoder(val_data)
        latent_dim = proj_t.shape[1]
        proj_stim_t = model.stimulus_encoder(val_stim)

        # time evolve by "0"
        # drop the last tmax time steps, maybe off by 1 here?
        results = [proj_t[:-tmax]]

        evolver_input = torch.concatenate([proj_t, proj_stim_t], dim=1)
        for dt in range(1, tmax):
            evolver_output = model.evolver(evolver_input)
            results.append(evolver_output[:-tmax, :latent_dim])
            evolver_input = evolver_output
            evolver_input[:-dt, latent_dim:] = proj_stim_t[dt:, :]
        results = torch.stack(results, dim=0)

        recons = model.decoder(results.reshape((-1, latent_dim))).reshape((tmax, -1, val_data.shape[1]))
    return recons

def plot_mses(val_data: torch.Tensor, recons: torch.Tensor):
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

    fig = plt.figure()
    plt.plot(np.arange(tmax), model_mses, label="latent model")
    plt.plot(np.arange(tmax), null_mses, ls="dashed", label="constant model (null)")
    plt.legend()
    plt.xticks(np.arange(tmax))
    plt.grid(True)
    plt.xlabel("Time steps evolved")
    plt.ylabel("MSE")
    fig.tight_layout()
    mse_metrics = {f"model_mse_evolve_{i}_steps": float(model_mses[i]) for i in range(tmax)}
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

    # MSE evolution over time steps
    recons = evolve_many_time_steps(model, val_data, val_stim, tmax=10)
    fig, mse_metrics = plot_mses(val_data, recons)
    metrics.update(mse_metrics)
    figures["mses_by_time_steps"] = fig
    if save_figures:
        fig.savefig(run_dir / "mses_by_time_steps.jpg", dpi=100)
        plt.close(fig)

    print("Validation diagnostics complete.")
    return metrics, figures
