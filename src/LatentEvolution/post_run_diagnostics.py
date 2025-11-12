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

def plot_neuron_reconstruction(val_data: torch.Tensor, model: LatentModel, neuron_data: NeuronData):
    true_trace = val_data.detach().cpu().numpy()
    recon = model.decoder(model.encoder(val_data)).detach().cpu().numpy()
    ntypes = len(neuron_data.TYPE_NAMES)

    rng = np.random.default_rng(seed=0)

    fig, ax = plt.subplots(1, ntypes, sharex=True, figsize=(8, 3*ntypes))

    for itype, tname in enumerate(neuron_data.TYPE_NAMES):
        ix = rng.choice(neuron_data.indices_per_type[itype])
        title = f"{tname}: {ix=}"
        p = ax[itype].plot(true_trace[:, ix], lw=3, alpha=0.5)
        # fix the range based on the true trace
        ylim = ax[itype].get_ylim()

        # reconstructed trace
        ax[itype].plot(recon[:, ix], color=p[-1].get_color())
        ax[itype].set_ylim(*ylim)
        ax[itype].set_title(title)

    ax[-1].set_xlabel("Time steps")
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

    # Neuron traces
    fig = plot_neuron_reconstruction(val_data, model, neuron_data)
    fig.savefig(run_dir / "neuron_traces.jpg", dpi=100)

    print("Post-training diagnostics complete.")
