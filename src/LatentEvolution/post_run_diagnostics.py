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

def plot_neuron_reconstruction(true_trace: np.ndarray, recon_trace: np.ndarray, neuron_data: NeuronData):
    ntypes = len(neuron_data.TYPE_NAMES)

    rng = np.random.default_rng(seed=0)

    fig, ax = plt.subplots(ntypes, 1, sharex=True, figsize=(16, 3*ntypes))

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
    mses = torch.zeros(tmax, device=recons.device)
    for dt in range(tmax):
        mses[dt] = torch.nn.functional.mse_loss(recons[dt], val_data[dt:-(tmax-dt)])
    mses = mses.detach().cpu().numpy()

    fig = plt.figure()
    plt.plot(np.arange(tmax), mses)
    plt.xticks(np.arange(tmax))
    plt.grid(True)
    plt.xlabel("Time steps evolved")
    plt.ylabel("MSE")
    return fig




def post_training_diagnostics(
    run_dir: Path,
    val_data: torch.Tensor,
    neuron_data: NeuronData,
    val_stim: torch.Tensor,
    model: LatentModel,
    config: ModelParams,
) -> None:
    """
    Perform post-training diagnostics on the trained model.

    Args:
        train_data: Training data tensor of shape (T_train, N)
        val_data: Validation data tensor of shape (T_val, N)
        test_data: Test data tensor of shape (T_test, N)
        train_stim: Training stimulus tensor of shape (T_train, S)
        val_stim: Validation stimulus tensor of shape (T_val, S)
        test_stim: Test stimulus tensor of shape (T_test, S)
        model: The trained LatentModel instance
        config: ModelParams configuration object
    """
    # TODO: Implement diagnostic analyses
    # - Latent space visualization
    # - Reconstruction quality analysis
    # - Temporal prediction accuracy
    # - Model capacity metrics
    # - Generalization analysis

    print("Running post-training diagnostics...")
    print(f"  Val data shape: {val_data.shape}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    model.eval()

    true_trace = val_data.detach().cpu().numpy()
    recon_trace = model.decoder(model.encoder(val_data)).detach().cpu().numpy()

    # Neuron traces
    fig = plot_neuron_reconstruction(true_trace, recon_trace, neuron_data)
    fig.savefig(run_dir / "neuron_traces.jpg", dpi=100)

    # Reconstruction error stratified
    fig = plot_recon_error(true_trace, recon_trace)
    fig.savefig(run_dir / "reconstruction_variance.jpg", dpi=100)

    recons = evolve_many_time_steps(model, val_data, val_stim, tmax=10)
    fig = plot_mses(val_data, recons)
    fig.savefig(run_dir / "mses_by_time_steps.jpg", dpi=100)

    print("Post-training diagnostics complete.")
