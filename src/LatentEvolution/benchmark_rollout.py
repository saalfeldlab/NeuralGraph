"""Benchmark script comparing LatentModel vs LatentStagModel rollouts."""

import torch
import time
import yaml
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from LatentEvolution.latent_stag import (
    LatentStagModel,
    StagModelParams,
)
from LatentEvolution.latent import LatentModel
from LatentEvolution.eed_model import ModelParams
from LatentEvolution.load_flyvis import load_column_slice, FlyVisSim
from LatentEvolution.diagnostics import evolve_n_steps_latent
from LatentEvolution.training_config import StimulusFrequency
from LatentEvolution.acquisition import compute_neuron_phases, StaggeredRandomMode


def plot_rollout_mse_comparison(
    latent_model_mse: np.ndarray,
    stag_model_mse: np.ndarray,
    fit_window: int,
    time_units: int,
    ems: int,
    num_start_points: int,
) -> Figure:
    """
    plot mse traces for both models, overlaid per start point.

    args:
        latent_model_mse: mse per time step per start point (num_start_points, n_steps)
        stag_model_mse: mse per time step per start point (num_start_points, n_steps)
        fit_window: number of steps used for z0 fitting
        time_units: observation interval
        ems: evolve_multiple_steps
        num_start_points: number of start points
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    n_steps = latent_model_mse.shape[1]
    time_axis = np.arange(n_steps)

    # use a colormap for different start points
    colors = plt.cm.tab10(np.linspace(0, 1, num_start_points))

    # plot first 5 start points to avoid crowding
    # solid = LatentModel, dashed = LatentStagModel
    num_to_plot = min(5, num_start_points)
    for i in range(num_to_plot):
        color = colors[i]
        # only add label for first start point to avoid legend clutter
        if i == 0:
            ax.plot(time_axis, latent_model_mse[i], linestyle='-', color=color,
                    alpha=0.7, linewidth=1.5, label='LatentModel (solid)')
            ax.plot(time_axis, stag_model_mse[i], linestyle='--', color=color,
                    alpha=0.7, linewidth=1.5, label='LatentStagModel (dashed)')
        else:
            ax.plot(time_axis, latent_model_mse[i], linestyle='-', color=color,
                    alpha=0.7, linewidth=1.5)
            ax.plot(time_axis, stag_model_mse[i], linestyle='--', color=color,
                    alpha=0.7, linewidth=1.5)

    # mark the fit window boundary
    ax.axvline(x=fit_window, color='gray', linestyle=':', alpha=0.7, label=f'fit window (t={fit_window})')

    ax.set_xlabel('time step', fontsize=14)
    ax.set_ylabel('MSE (averaged over neurons)', fontsize=14)
    ax.set_yscale('log')
    ax.set_xlim(0, 2 * fit_window)
    ax.set_title(
        f'rollout MSE comparison: LatentModel vs LatentStagModel\n'
        f'(tu={time_units}, ems={ems}, fit_window={fit_window}, {num_start_points} starts)',
        fontsize=16,
        fontweight='bold',
    )
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()

    return fig


def plot_mse_scatter(
    latent_model_mse: np.ndarray,
    stag_model_mse: np.ndarray,
    fit_window: int,
) -> Figure:
    """
    scatter plot comparing latentmodel vs latentstag mse per start point.

    args:
        latent_model_mse: (num_start_points, n_steps)
        stag_model_mse: (num_start_points, n_steps)
        fit_window: boundary between fit and extrapolation
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # compute mean mse per start for fit window and beyond
    latent_fit = latent_model_mse[:, :fit_window].mean(axis=1)
    latent_beyond = latent_model_mse[:, fit_window:].mean(axis=1)
    stag_fit = stag_model_mse[:, :fit_window].mean(axis=1)
    stag_beyond = stag_model_mse[:, fit_window:].mean(axis=1)

    # scatter: circles for fit window, triangles for beyond
    ax.scatter(latent_fit, stag_fit, marker='o', s=60, alpha=0.7,
               label=f'[0, {fit_window}] (fit window)', color='C0')
    ax.scatter(latent_beyond, stag_beyond, marker='^', s=60, alpha=0.7,
               label=f'[{fit_window}, ...] (extrapolation)', color='C1')

    # diagonal line (y=x)
    all_vals = np.concatenate([latent_fit, latent_beyond, stag_fit, stag_beyond])
    vmin, vmax = all_vals.min() * 0.9, all_vals.max() * 1.1
    ax.plot([vmin, vmax], [vmin, vmax], 'k--', alpha=0.5, label='y=x')

    ax.set_xlabel('LatentModel MSE', fontsize=14)
    ax.set_ylabel('LatentStagModel MSE', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_aspect('equal')
    ax.set_title('MSE comparison: LatentModel vs LatentStagModel\n(each point = one start)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()

    return fig


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}', flush=True)

    # load config from yaml
    config_path = Path(__file__).parent / "latent_stag_20step.yaml"
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    stag_cfg = StagModelParams(**config_dict)

    latent_dims = stag_cfg.latent_dims
    num_neurons = stag_cfg.num_neurons
    stim_dims = stag_cfg.stimulus_encoder_params.num_input_dims
    time_units = stag_cfg.training.time_units
    ems = stag_cfg.training.evolve_multiple_steps
    fit_window = time_units * ems  # 100 steps
    n_rollout_steps = 2000

    # checkpoint path
    checkpoint_path = "/groups/saalfeld/home/kumarv4/repos/NeuralGraph/runs/test_acq_20260122_42c9742/5846e4/model_final.pt"
    print(f'loading weights from: {checkpoint_path}', flush=True)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

    # -------------------------------------------------------------------------
    # load LatentModel (full model with encoder)
    # -------------------------------------------------------------------------
    # need to load config from the checkpoint run directory
    checkpoint_config_path = Path(checkpoint_path).parent / "config.yaml"
    with open(checkpoint_config_path) as f:
        latent_model_config = yaml.safe_load(f)
    latent_cfg = ModelParams(**latent_model_config)

    latent_model = LatentModel(latent_cfg).to(device)
    latent_model.load_state_dict(state_dict, strict=True)
    latent_model.eval()
    print(f'LatentModel params: {sum(p.numel() for p in latent_model.parameters()):,}', flush=True)

    # -------------------------------------------------------------------------
    # load LatentStagModel (decoder, evolver, stimulus_encoder only)
    # -------------------------------------------------------------------------
    stag_model = LatentStagModel(stag_cfg).to(device)

    # extract decoder, evolver, stimulus_encoder weights (skip encoder)
    stag_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('decoder.') or key.startswith('evolver.') or key.startswith('stimulus_encoder.'):
            stag_state_dict[key] = value

    stag_model.load_state_dict(stag_state_dict, strict=True)
    stag_model.eval()
    print(f'LatentStagModel params: {sum(p.numel() for p in stag_model.parameters()):,}', flush=True)

    print(f'latent_dims: {latent_dims}, num_neurons: {num_neurons}, stim_dims: {stim_dims}', flush=True)
    print(f'time_units: {time_units}, ems: {ems}, fit_window: {fit_window}', flush=True)

    # -------------------------------------------------------------------------
    # load real data from davis dataset (10 random start points)
    # -------------------------------------------------------------------------
    data_path = "/groups/saalfeld/home/kumarv4/repos/NeuralGraph/graphs_data/fly/fly_N9_62_0_davis_calcium/x_list_0"
    print(f'loading data from: {data_path}', flush=True)

    # query dataset size via tensorstore
    import tensorstore as ts
    ts_path = Path(data_path) / 'timeseries.zarr'
    spec = {'driver': 'zarr', 'kvstore': {'driver': 'file', 'path': str(ts_path)}}
    store = ts.open(spec).result()
    dataset_len = store.shape[0]
    print(f'dataset length: {dataset_len} time steps', flush=True)

    # pick 20 random start points with room for rollout
    num_start_points = 20
    total_len = fit_window + n_rollout_steps + 1
    max_start = dataset_len - total_len
    rng = np.random.default_rng(seed=123)
    start_indices = sorted(rng.integers(0, max_start, size=num_start_points).tolist())
    print(f'start indices: {start_indices}', flush=True)
    print(f'total_len needed per start: {total_len}', flush=True)

    # load data for all start points
    voltage_list = []
    stim_list = []
    for start_idx in start_indices:
        voltage_slice = load_column_slice(data_path, FlyVisSim.VOLTAGE.value, start_idx, start_idx + total_len)
        stim_slice = load_column_slice(data_path, FlyVisSim.STIMULUS.value, start_idx, start_idx + total_len, neuron_limit=stim_dims)
        voltage_list.append(torch.tensor(voltage_slice, dtype=torch.float32, device=device))
        stim_list.append(torch.tensor(stim_slice, dtype=torch.float32, device=device))

    # stack into batched tensors: (batch, time, neurons/stim_dims)
    voltage_data_batched = torch.stack(voltage_list, dim=0)  # (10, total_len, num_neurons)
    stim_data_batched = torch.stack(stim_list, dim=0)  # (10, total_len, stim_dims)
    print(f'voltage shape: {voltage_data_batched.shape}, stim shape: {stim_data_batched.shape}', flush=True)

    # -------------------------------------------------------------------------
    # run LatentModel rollout for all start points
    # -------------------------------------------------------------------------
    print(f'\n--- LatentModel {n_rollout_steps} step rollout (x{num_start_points} start points) ---', flush=True)

    # warmup with first start point
    print('warmup...', flush=True)
    with torch.no_grad():
        _ = evolve_n_steps_latent(
            latent_model,
            voltage_data_batched[0, 0],  # first start point, t=0
            stim_data_batched[0, :100],
            100,
            StimulusFrequency.ALL,
            time_units=1,
        )
        if device.type == 'cuda':
            torch.cuda.synchronize()

    # run rollout for each start point (sequential - LatentModel doesn't batch easily)
    print('timed run...', flush=True)
    latent_model_rollouts = []
    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()
        for b in range(num_start_points):
            rollout = evolve_n_steps_latent(
                latent_model,
                voltage_data_batched[b, 0],  # initial state for this start point
                stim_data_batched[b, :n_rollout_steps],  # stimulus
                n_rollout_steps,
                StimulusFrequency.ALL,
                time_units=1,
            )
            latent_model_rollouts.append(rollout)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        latent_model_elapsed = time.time() - start_time

    # stack: (batch, n_rollout_steps, num_neurons)
    latent_model_rollouts = torch.stack(latent_model_rollouts, dim=0)
    print(f'LatentModel rollout shape: {latent_model_rollouts.shape}', flush=True)
    print(f'LatentModel rollout elapsed: {latent_model_elapsed:.2f}s ({latent_model_elapsed/n_rollout_steps/num_start_points*1000:.2f}ms/step)', flush=True)

    # compute mse vs ground truth per start point (avg over neurons only)
    gt_rollouts = voltage_data_batched[:, 1:n_rollout_steps + 1, :]  # (batch, n_rollout_steps, num_neurons)
    latent_model_mse_per_start = torch.pow(latent_model_rollouts - gt_rollouts, 2).mean(dim=2).cpu().numpy()  # (batch, n_steps)
    latent_model_mse_mean = latent_model_mse_per_start.mean(axis=0)  # avg over starts
    print(f'LatentModel vs ground truth: mean={latent_model_mse_mean.mean():.6f}, max={latent_model_mse_mean.max():.6f}', flush=True)

    # -------------------------------------------------------------------------
    # set up staggered acquisition for z0 optimization
    # -------------------------------------------------------------------------
    print('\n--- setting up staggered acquisition ---', flush=True)

    # compute neuron phases for staggered acquisition
    acquisition_mode = StaggeredRandomMode(seed=42)
    neuron_phases = compute_neuron_phases(
        num_neurons=num_neurons,
        time_units=time_units,
        acquisition_mode=acquisition_mode,
        device=device,
    )
    assert neuron_phases is not None, "staggered mode should return phases"
    print(f'neuron phases: min={neuron_phases.min().item()}, max={neuron_phases.max().item()}', flush=True)

    # create observation mask: (fit_window, num_neurons) boolean
    # neuron n is observed at time t if (t - phase[n]) % time_units == 0
    time_indices = torch.arange(fit_window, device=device).unsqueeze(1)  # (fit_window, 1)
    phases = neuron_phases.unsqueeze(0)  # (1, num_neurons)
    observation_mask = ((time_indices - phases) % time_units) == 0  # (fit_window, num_neurons)

    obs_per_neuron = observation_mask.sum(dim=0)
    print(f'observations per neuron: {obs_per_neuron[0].item()} (should be {ems})', flush=True)
    print(f'total observations: {observation_mask.sum().item()} (should be {num_neurons * ems})', flush=True)

    # precompute nonzero indices for the mask (avoids dynamic shapes in compiled function)
    obs_indices = observation_mask.nonzero(as_tuple=True)  # (time_indices, neuron_indices)
    obs_t = obs_indices[0]  # time indices
    obs_n = obs_indices[1]  # neuron indices
    print(f'obs_indices: {len(obs_t)} pairs', flush=True)

    # -------------------------------------------------------------------------
    # optimize z0 for LatentStagModel with staggered MSE (batched over start points)
    # -------------------------------------------------------------------------
    print(f'\n--- LatentStagModel z0 optimization (staggered MSE, {num_start_points} start points) ---', flush=True)

    # batched target and stimulus: (batch, fit_window, neurons/stim_dims)
    fit_target_batched = voltage_data_batched[:, 1:1 + fit_window, :]  # (batch, fit_window, num_neurons)
    fit_stimulus_batched = stim_data_batched[:, 0:fit_window, :]  # (batch, fit_window, stim_dims)

    # pre-encode stimulus for all start points
    # stimulus_encoder expects (time, stim_dims), so we process per batch or reshape
    with torch.no_grad():
        # reshape to (batch * fit_window, stim_dims), encode, reshape back
        stim_flat = fit_stimulus_batched.reshape(-1, stim_dims)
        stim_latent_flat = stag_model.stimulus_encoder(stim_flat)
        stim_latent_dim = stim_latent_flat.shape[-1]
        stim_latent_batched = stim_latent_flat.reshape(num_start_points, fit_window, stim_latent_dim)
    print(f'stim_latent_batched shape: {stim_latent_batched.shape}', flush=True)

    # define batched loss function
    # z0: (batch, latent_dims)
    # stim_latent: (batch, fit_window, stim_latent_dim)
    # target: (batch, fit_window, num_neurons)
    # returns sum of losses over batch
    def compute_loss_batched(z0, stim_latent, target, t_idx, n_idx):
        z = z0  # (batch, latent_dims)
        predictions = []
        for t in range(fit_window):
            # stim_latent[:, t, :] -> (batch, stim_latent_dim)
            z = stag_model.evolver(z, stim_latent[:, t, :])
            pred = stag_model.decoder(z)  # (batch, num_neurons)
            predictions.append(pred)
        # pred_trace: (fit_window, batch, num_neurons) -> transpose to (batch, fit_window, num_neurons)
        pred_trace = torch.stack(predictions, dim=1)  # (batch, fit_window, num_neurons)
        # index with precomputed (time, neuron) pairs - same mask for all batches
        # pred_trace[:, t_idx, n_idx] -> (batch, num_obs)
        pred_obs = pred_trace[:, t_idx, n_idx]
        target_obs = target[:, t_idx, n_idx]
        loss = torch.nn.functional.mse_loss(pred_obs, target_obs)
        return loss

    # no compilation - use raw function
    print('using uncompiled loss function...', flush=True)

    # actual z0 optimization using LBFGS (2 steps)
    fit_steps = 2
    print(f'\n--- fit_steps={fit_steps}, using LBFGS (no compilation) ---', flush=True)

    z0 = torch.randn(num_start_points, latent_dims, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([z0], lr=1.0, max_iter=20, line_search_fn='strong_wolfe')

    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()

    for step in range(fit_steps + 1):
        def closure():
            optimizer.zero_grad()
            loss = compute_loss_batched(z0, stim_latent_batched, fit_target_batched, obs_t, obs_n)
            loss.backward()
            return loss

        # report loss
        with torch.no_grad():
            loss = compute_loss_batched(z0, stim_latent_batched, fit_target_batched, obs_t, obs_n)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f'step {step:3d}: loss={loss.item():.4f}, t={elapsed:.1f}s', flush=True)

        if step == fit_steps:
            break

        optimizer.step(closure)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    fit_elapsed = time.time() - start
    print(f'z0 fit elapsed: {fit_elapsed:.2f}s ({fit_elapsed/num_start_points:.2f}s per start point)', flush=True)

    z0 = z0.detach()

    # -------------------------------------------------------------------------
    # rollout from fitted z0 (batched, same time range as LatentModel)
    # -------------------------------------------------------------------------
    print(f'\n--- LatentStagModel rollout from fitted z0 ({num_start_points} start points) ---', flush=True)

    # pre-encode rollout stimulus for all start points
    rollout_stimulus_batched = stim_data_batched[:, :n_rollout_steps, :]  # (batch, n_rollout_steps, stim_dims)
    with torch.no_grad():
        stim_flat = rollout_stimulus_batched.reshape(-1, stim_dims)
        rollout_stim_latent_flat = stag_model.stimulus_encoder(stim_flat)
        rollout_stim_latent_batched = rollout_stim_latent_flat.reshape(num_start_points, n_rollout_steps, -1)
    print(f'rollout_stim_latent_batched shape: {rollout_stim_latent_batched.shape}', flush=True)

    # warmup
    print('warmup...', flush=True)
    with torch.no_grad():
        z = z0.clone()
        for t in range(100):
            z = stag_model.evolver(z, rollout_stim_latent_batched[:, t, :])
        if device.type == 'cuda':
            torch.cuda.synchronize()

    # timed run (batched)
    print('timed run...', flush=True)
    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()

        z = z0  # (batch, latent_dims)
        latent_trace = []
        for t in range(n_rollout_steps):
            z = stag_model.evolver(z, rollout_stim_latent_batched[:, t, :])
            latent_trace.append(z)
        # latent_trace: list of (batch, latent_dims) -> (batch, n_rollout_steps, latent_dims)
        latent_trace = torch.stack(latent_trace, dim=1)
        # decode all at once: reshape to (batch * n_rollout_steps, latent_dims)
        latent_flat = latent_trace.reshape(-1, latent_dims)
        stag_rollout_flat = stag_model.decoder(latent_flat)
        stag_model_rollouts = stag_rollout_flat.reshape(num_start_points, n_rollout_steps, num_neurons)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        stag_rollout_elapsed = time.time() - start_time

    print(f'LatentStagModel rollout shape: {stag_model_rollouts.shape}', flush=True)
    print(f'LatentStagModel rollout elapsed: {stag_rollout_elapsed:.2f}s ({stag_rollout_elapsed/n_rollout_steps/num_start_points*1000:.2f}ms/step)', flush=True)

    # -------------------------------------------------------------------------
    # compare rollouts (per start point)
    # -------------------------------------------------------------------------
    print('\n--- comparison ---', flush=True)

    # mse vs ground truth per start point (avg over neurons only)
    stag_mse_per_start = torch.pow(stag_model_rollouts - gt_rollouts, 2).mean(dim=2).cpu().numpy()  # (batch, n_steps)
    stag_mse_mean = stag_mse_per_start.mean(axis=0)  # avg over starts

    # mse between the two model rollouts
    model_vs_model_mse = torch.nn.functional.mse_loss(latent_model_rollouts, stag_model_rollouts).item()
    print(f'LatentModel vs LatentStagModel (direct comparison, {n_rollout_steps} steps): MSE = {model_vs_model_mse:.6f}', flush=True)

    print('\n--- summary ---', flush=True)
    print(f'averaged over {num_start_points} start points:', flush=True)

    # separate mse for fit window [0, fit_window] and beyond [fit_window, ...]
    latent_mse_fit = latent_model_mse_per_start[:, :fit_window].mean()
    latent_mse_beyond = latent_model_mse_per_start[:, fit_window:].mean()
    stag_mse_fit = stag_mse_per_start[:, :fit_window].mean()
    stag_mse_beyond = stag_mse_per_start[:, fit_window:].mean()

    print(f'LatentModel:     [0, {fit_window}] mse={latent_mse_fit:.6f}, [{fit_window}, ...] mse={latent_mse_beyond:.6f}', flush=True)
    print(f'LatentStagModel: [0, {fit_window}] mse={stag_mse_fit:.6f}, [{fit_window}, ...] mse={stag_mse_beyond:.6f}', flush=True)

    # -------------------------------------------------------------------------
    # plot comparison
    # -------------------------------------------------------------------------
    print('\n--- generating plot ---', flush=True)

    fig = plot_rollout_mse_comparison(
        latent_model_mse=latent_model_mse_per_start,
        stag_model_mse=stag_mse_per_start,
        fit_window=fit_window,
        time_units=time_units,
        ems=ems,
        num_start_points=num_start_points,
    )

    plot_path = Path(__file__).parent / "rollout_mse_comparison.png"
    fig.savefig(plot_path, dpi=150)
    print(f'saved plot to: {plot_path}', flush=True)
    plt.close(fig)

    # scatter plot comparing models
    fig_scatter = plot_mse_scatter(
        latent_model_mse=latent_model_mse_per_start,
        stag_model_mse=stag_mse_per_start,
        fit_window=fit_window,
    )
    plot_path_scatter = Path(__file__).parent / "mse_scatter.png"
    fig_scatter.savefig(plot_path_scatter, dpi=150)
    print(f'saved plot to: {plot_path_scatter}', flush=True)
    plt.close(fig_scatter)
