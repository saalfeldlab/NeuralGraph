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

def plot_neuron_reconstruction(true_trace: np.ndarray, recon_trace: np.ndarray, neuron_data: NeuronData, xlim: tuple[int, int] | None = None):
    ntypes = len(neuron_data.TYPE_NAMES)

    rng = np.random.default_rng(seed=0)

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

def plot_recon_error(true_trace, recon_trace):
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    # time variation, neuron variation
    for i in (0, 1):
        var_trace = np.var(true_trace, axis=i)
        err = recon_trace - true_trace
        var_err = np.var(err, axis=i)
        ax[i].scatter(var_trace, var_err, marker=".", alpha=0.5)
        ax[i].set_xlabel("Raw variance")
        ax[i].set_ylabel("Unexplained variance")
        ax[i].set_title("Variance across " + ["time", "neurons"][i])
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




def run_validation_diagnostics(
    run_dir: Path,
    val_data: torch.Tensor,
    neuron_data: NeuronData,
    val_stim: torch.Tensor,
    model: LatentModel,
    config: ModelParams,
    save_figures: bool = False,
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

    # Neuron traces (full trace)
    fig = plot_neuron_reconstruction(true_trace, recon_trace, neuron_data, xlim=None)
    figures["neuron_traces"] = fig
    if save_figures:
        fig.savefig(run_dir / "neuron_traces.jpg", dpi=100)
        plt.close(fig)

    # Neuron traces (zoomed)
    fig = plot_neuron_reconstruction(true_trace, recon_trace, neuron_data, xlim=(100, 1100))
    figures["neuron_traces_zoom"] = fig
    if save_figures:
        fig.savefig(run_dir / "neuron_traces_zoom.jpg", dpi=100)
        plt.close(fig)

    # Reconstruction error stratified
    fig = plot_recon_error(true_trace, recon_trace)
    figures["reconstruction_variance"] = fig
    if save_figures:
        fig.savefig(run_dir / "reconstruction_variance.jpg", dpi=100)
        plt.close(fig)

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
