"""
zapbench z0 bank + encoder + decoder training.

trains a sparse z0 bank with linear encoder/decoder on zapbench data:
1. decoder training: z0_bank + decoder (piecewise constant reconstruction)
2. encoder training: linear encoder to match z0_bank targets

usage:
  python zapbench_z0_enc_dec.py <expt_code> [--overrides]

  example:
    python zapbench_z0_enc_dec.py my_expt --latent_dim 128 --z0_spacing_frames 2.0
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tyro
import yaml
from sklearn.manifold import TSNE
from tqdm import tqdm

from LatentEvolution.hparam_paths import create_run_directory, get_git_commit_hash
from LatentEvolution.training_utils import seed_everything
from LatentEvolution.zapbench_config import DataSplit
from LatentEvolution.zapbench_data import load_sparse_activity

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


class Config(BaseModel):
    """configuration for z0 bank + encoder + decoder training."""
    # data
    traces_path: str = "/groups/saalfeld/saalfeldlab/zapbench-release/volumes/20240930/traces"
    ephys_path: str = "/groups/saalfeld/saalfeldlab/zapbench-processed/ephys.zarr"
    bin_size_ms: float = 100.0

    # model
    latent_dim: int = 64
    z0_spacing_frames: float = 4.0  # spacing in frames (fractional allowed)

    # decoder training (z0_bank + decoder)
    decoder_epochs: int = 100
    decoder_lr: float = 1e-3
    decoder_batch_size: int = 64  # observations per batch
    tv_weight: float = 0.0  # total variation regularization on z0_bank

    # encoder training
    enc_epochs: int = 100
    enc_lr: float = 1e-3
    enc_batch_size: int = 64  # bins per batch

    # plotting
    tsne_max_points: int = 2000
    l2_num_starts: int = 20
    l2_max_delta: int = 100

    # misc
    seed: int = 135717
    max_conditions: int = 0  # 0 = all conditions, >0 = limit to first N

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# sparse data structure per condition
# ---------------------------------------------------------------------------


@dataclass
class ConditionData:
    """sparse data for one condition.

    stores observations in a flat sorted structure for efficient bin lookup.
    """
    name: str
    obs_times: torch.Tensor  # (N, K) sorted bin indices per neuron
    obs_vals: torch.Tensor   # (N, K) values per neuron
    num_bins: int
    num_neurons: int
    num_frames: int  # K
    avg_bins_per_frame: float

    # sparse bin->observations lookup
    bins_sorted: torch.Tensor     # (num_obs,) sorted bin indices
    neuron_ids_sorted: torch.Tensor  # (num_obs,) neuron ids
    vals_sorted: torch.Tensor     # (num_obs,) values
    bin_starts: torch.Tensor      # (num_bins,) start index for each bin
    bin_ends: torch.Tensor        # (num_bins,) end index for each bin


def build_condition_data(
    name: str,
    obs_times: torch.Tensor,
    obs_vals: torch.Tensor,
    num_bins: int,
    device: torch.device,
) -> ConditionData:
    """build ConditionData with sparse bin->observations structure.

    args:
        name: condition name
        obs_times: (N, K) sparse observation times on cpu
        obs_vals: (N, K) sparse values on cpu
        num_bins: total number of time bins
        device: target device

    returns:
        ConditionData with all tensors on device
    """
    N, K = obs_times.shape

    # move to device
    obs_times = obs_times.to(device)
    obs_vals = obs_vals.to(device)

    # compute avg bins per frame (for z0 spacing)
    avg_bins_per_frame = num_bins / K

    # flatten for sparse structure
    bins_flat = obs_times.flatten()  # (N * K,)
    neuron_ids = torch.arange(N, device=device).unsqueeze(1).expand(N, K).flatten()
    vals_flat = obs_vals.flatten()

    # sort by bin index
    sort_idx = torch.argsort(bins_flat)
    bins_sorted = bins_flat[sort_idx]
    neuron_ids_sorted = neuron_ids[sort_idx]
    vals_sorted = vals_flat[sort_idx]

    # bin boundaries via searchsorted
    bin_starts = torch.searchsorted(bins_sorted, torch.arange(num_bins, device=device))
    bin_ends = torch.searchsorted(bins_sorted, torch.arange(num_bins, device=device) + 1)

    return ConditionData(
        name=name,
        obs_times=obs_times,
        obs_vals=obs_vals,
        num_bins=num_bins,
        num_neurons=N,
        num_frames=K,
        avg_bins_per_frame=avg_bins_per_frame,
        bins_sorted=bins_sorted,
        neuron_ids_sorted=neuron_ids_sorted,
        vals_sorted=vals_sorted,
        bin_starts=bin_starts,
        bin_ends=bin_ends,
    )


def get_bin_observations(
    cond: ConditionData,
    t: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """get neuron ids and values observed at bin t.

    returns:
        obs_n: (num_obs,) neuron indices
        obs_v: (num_obs,) observed values
    """
    start = cond.bin_starts[t].item()
    end = cond.bin_ends[t].item()
    return cond.neuron_ids_sorted[start:end], cond.vals_sorted[start:end]


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------


def load_all_conditions(
    traces_path: str,
    ephys_path: str,
    bin_size_ms: float,
    split: DataSplit,
    split_type: Literal["train", "val", "test"],
    device: torch.device,
    max_conditions: int = 0,
) -> list[ConditionData]:
    """load all conditions for given split type.

    args:
        traces_path: path to traces zarr
        ephys_path: path to ephys.zarr
        bin_size_ms: bin width in milliseconds
        split: data split configuration
        split_type: which split to load
        device: target device
        max_conditions: limit to first N conditions (0 = all)

    returns:
        list of ConditionData, one per condition with data in this split
    """
    ranges = split.get_ranges(split_type)
    if max_conditions > 0:
        ranges = ranges[:max_conditions]
    conditions = []

    for cond_name, abs_start, abs_end in ranges:
        log.info(f"  loading {cond_name} ({split_type})...")

        # load sparse data
        obs_times, obs_vals, _counts, num_bins = load_sparse_activity(
            traces_path, ephys_path, bin_size_ms,
            time_slice=slice(abs_start, abs_end),
        )

        # build condition data structure
        cond_data = build_condition_data(
            name=cond_name,
            obs_times=obs_times,
            obs_vals=obs_vals,
            num_bins=num_bins,
            device=device,
        )
        conditions.append(cond_data)

        log.info(f"    bins={num_bins}, frames={cond_data.num_frames}, "
                 f"neurons={cond_data.num_neurons}, "
                 f"avg_bins_per_frame={cond_data.avg_bins_per_frame:.1f}")

    return conditions


# ---------------------------------------------------------------------------
# model components
# ---------------------------------------------------------------------------


def create_z0_bank(
    num_z0s: int,
    latent_dim: int,
    device: torch.device,
) -> nn.Parameter:
    """create z0 bank parameter."""
    return nn.Parameter(torch.randn(num_z0s, latent_dim, device=device) * 0.1)


def get_z0_normalized(z0_bank: nn.Parameter) -> torch.Tensor:
    """return z0_bank normalized to unit sphere."""
    return z0_bank / z0_bank.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def create_decoder(
    num_neurons: int,
    latent_dim: int,
    device: torch.device,
) -> tuple[nn.Parameter, nn.Parameter]:
    """create linear decoder parameters W_dec, b_dec."""
    W_dec = nn.Parameter(torch.randn(num_neurons, latent_dim, device=device) * 0.01)
    b_dec = nn.Parameter(torch.zeros(num_neurons, device=device))
    return W_dec, b_dec


def create_encoder(
    num_neurons: int,
    latent_dim: int,
    device: torch.device,
) -> nn.Parameter:
    """create linear encoder parameter W_enc."""
    return nn.Parameter(torch.randn(latent_dim, num_neurons, device=device) * 0.01)


def decode(z: torch.Tensor, W_dec: nn.Parameter, b_dec: nn.Parameter) -> torch.Tensor:
    """decode latent z to all neurons."""
    return z @ W_dec.T + b_dec


def encode_sparse(
    obs_n: torch.Tensor,
    obs_v: torch.Tensor,
    W_enc: nn.Parameter,
    num_neurons: int,
) -> torch.Tensor:
    """encode partial observations to latent.

    args:
        obs_n: (num_obs,) neuron indices
        obs_v: (num_obs,) observed values
        W_enc: (L, N) encoder weights
        num_neurons: total number of neurons

    returns:
        z: (L,) latent encoding, normalized to unit sphere
    """
    # build sparse activity vector
    x = torch.zeros(num_neurons, device=W_enc.device)
    x[obs_n] = obs_v

    # linear encoding
    z = W_enc @ x  # (L,)

    # normalize to unit sphere
    return z / z.norm().clamp(min=1e-6)


# ---------------------------------------------------------------------------
# decoder training
# ---------------------------------------------------------------------------


def train_decoder(
    conditions: list[ConditionData],
    cfg: Config,
    device: torch.device,
) -> tuple[nn.Parameter, nn.Parameter, nn.Parameter, torch.Tensor, int]:
    """train z0_bank and decoder.

    args:
        conditions: list of ConditionData
        cfg: config
        device: target device

    returns:
        z0_bank: trained z0 bank parameter
        W_dec: decoder weights
        b_dec: decoder bias
        z0_bin_indices: bin indices for each z0
        bin_stride: spacing between z0s in bins
    """
    # get dimensions from first condition
    num_neurons = conditions[0].num_neurons
    avg_bins_per_frame = np.mean([c.avg_bins_per_frame for c in conditions])

    # compute z0 spacing in bins
    bin_stride = int(cfg.z0_spacing_frames * avg_bins_per_frame)
    bin_stride = max(1, bin_stride)  # at least 1 bin

    # find max bins across conditions for z0 bank size
    max_bins = max(c.num_bins for c in conditions)
    num_z0s = (max_bins + bin_stride - 1) // bin_stride

    log.info("decoder training:")
    log.info(f"  avg_bins_per_frame={avg_bins_per_frame:.1f}, bin_stride={bin_stride}")
    log.info(f"  num_z0s={num_z0s}, max_bins={max_bins}")

    # create parameters
    z0_bank = create_z0_bank(num_z0s, cfg.latent_dim, device)
    W_dec, b_dec = create_decoder(num_neurons, cfg.latent_dim, device)

    # z0 bin indices
    z0_bin_indices = torch.arange(0, num_z0s * bin_stride, bin_stride, device=device)

    # optimizer
    optimizer = torch.optim.Adam([z0_bank, W_dec, b_dec], lr=cfg.decoder_lr)

    # compute per-neuron std for potential normalization
    all_vals = torch.cat([c.vals_sorted for c in conditions])
    global_std = all_vals.std().item()
    log.info(f"  global std: {global_std:.4f}")

    # compute sampling weights (by number of valid bins per condition)
    weights = torch.tensor([c.num_bins for c in conditions], dtype=torch.float, device=device)
    weights = weights / weights.sum()

    # count total observations per epoch
    total_obs = sum(c.num_bins for c in conditions)
    batches_per_epoch = total_obs // cfg.decoder_batch_size

    log.info(f"  {cfg.decoder_epochs} epochs, {batches_per_epoch} batches/epoch")

    pbar = tqdm(range(cfg.decoder_epochs), desc="decoder")
    for _epoch in pbar:
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_tv = 0.0

        for _ in range(batches_per_epoch):
            # sample condition
            cond_idx = torch.multinomial(weights, 1).item()
            cond = conditions[cond_idx]

            # sample batch of bins
            batch_bins = torch.randint(0, cond.num_bins, (cfg.decoder_batch_size,), device=device)

            batch_loss = torch.tensor(0.0, device=device)
            batch_count = 0

            for t_tensor in batch_bins:
                t = t_tensor.item()
                obs_n, obs_v = get_bin_observations(cond, t)
                if len(obs_n) == 0:
                    continue

                # get z0 for this bin
                z0_idx = min(t // bin_stride, num_z0s - 1)
                z0_norm = get_z0_normalized(z0_bank)
                z = z0_norm[z0_idx]

                # decode and compute loss only on observed neurons
                x_pred_full = decode(z, W_dec, b_dec)
                x_pred = x_pred_full[obs_n]

                batch_loss = batch_loss + ((x_pred - obs_v) ** 2).mean()
                batch_count += 1

            if batch_count == 0:
                continue

            recon_loss = batch_loss / batch_count

            # TV regularization on z0_bank
            z0_norm = get_z0_normalized(z0_bank)
            tv_loss = ((z0_norm[1:] - z0_norm[:-1]) ** 2).mean()

            loss = recon_loss + cfg.tv_weight * tv_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_tv += tv_loss.item()

        avg_loss = epoch_loss / batches_per_epoch
        avg_recon = epoch_recon / batches_per_epoch
        avg_tv = epoch_tv / batches_per_epoch
        pbar.set_postfix(loss=f"{avg_loss:.4f}", recon=f"{avg_recon:.4f}", tv=f"{avg_tv:.4f}")

    return z0_bank, W_dec, b_dec, z0_bin_indices, bin_stride


# ---------------------------------------------------------------------------
# encoder training
# ---------------------------------------------------------------------------


def train_encoder(
    conditions: list[ConditionData],
    z0_bank: nn.Parameter,
    bin_stride: int,
    cfg: Config,
    device: torch.device,
) -> nn.Parameter:
    """train encoder to match z0_bank targets.

    args:
        conditions: list of ConditionData
        z0_bank: trained z0 bank (frozen during encoder training)
        bin_stride: spacing between z0s in bins
        cfg: config
        device: target device

    returns:
        W_enc: trained encoder weights
    """
    num_neurons = conditions[0].num_neurons
    num_z0s = z0_bank.shape[0]

    log.info("encoder training:")

    # create encoder
    W_enc = create_encoder(num_neurons, cfg.latent_dim, device)

    # get frozen z0 targets
    with torch.no_grad():
        z0_targets = get_z0_normalized(z0_bank).detach()

    # optimizer (only encoder)
    optimizer = torch.optim.Adam([W_enc], lr=cfg.enc_lr)

    # sampling weights
    weights = torch.tensor([c.num_bins for c in conditions], dtype=torch.float, device=device)
    weights = weights / weights.sum()

    # batches per epoch
    total_bins = sum(c.num_bins for c in conditions)
    batches_per_epoch = total_bins // cfg.enc_batch_size

    log.info(f"  {cfg.enc_epochs} epochs, {batches_per_epoch} batches/epoch")

    pbar = tqdm(range(cfg.enc_epochs), desc="encoder")
    for _epoch in pbar:
        epoch_loss = 0.0

        for _ in range(batches_per_epoch):
            # sample condition
            cond_idx = torch.multinomial(weights, 1).item()
            cond = conditions[cond_idx]

            # sample batch of bins
            batch_bins = torch.randint(0, cond.num_bins, (cfg.enc_batch_size,), device=device)

            batch_loss = torch.tensor(0.0, device=device)
            batch_count = 0

            for t_tensor in batch_bins:
                t = t_tensor.item()
                obs_n, obs_v = get_bin_observations(cond, t)
                if len(obs_n) < 10:  # skip bins with too few observations
                    continue

                # encode partial observations
                z_pred = encode_sparse(obs_n, obs_v, W_enc, num_neurons)

                # target: z0_bank value for this bin
                z0_idx = min(t // bin_stride, num_z0s - 1)
                z_target = z0_targets[z0_idx]

                # L2 loss on unit sphere
                batch_loss = batch_loss + ((z_pred - z_target) ** 2).sum()
                batch_count += 1

            if batch_count == 0:
                continue

            loss = batch_loss / batch_count

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / batches_per_epoch
        pbar.set_postfix(loss=f"{avg_loss:.4f}")

    return W_enc


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------


def plot_l2_distance(
    conditions: list[ConditionData],
    W_enc: nn.Parameter,
    cfg: Config,
    output_path: Path,
) -> None:
    """plot L2 distance vs delta_t for each condition.

    args:
        conditions: list of ConditionData
        W_enc: trained encoder
        cfg: config
        output_path: output file path
    """
    num_conds = len(conditions)
    fig, axes = plt.subplots(num_conds, 1, figsize=(10, 4 * num_conds), squeeze=False)

    for idx, cond in enumerate(conditions):
        ax = axes[idx, 0]

        # collect encodings for random start bins
        start_bins = torch.randperm(cond.num_bins)[:cfg.l2_num_starts]
        all_distances = []
        all_deltas = []

        with torch.no_grad():
            for start_tensor in start_bins:
                start = start_tensor.item()
                obs_n_start, obs_v_start = get_bin_observations(cond, start)
                if len(obs_n_start) < 10:
                    continue
                z_start = encode_sparse(obs_n_start, obs_v_start, W_enc, cond.num_neurons)

                for delta in range(1, min(cfg.l2_max_delta, cond.num_bins - start)):
                    t = start + delta
                    obs_n, obs_v = get_bin_observations(cond, t)
                    if len(obs_n) < 10:
                        continue
                    z_t = encode_sparse(obs_n, obs_v, W_enc, cond.num_neurons)

                    l2_dist = (z_start - z_t).norm().item()
                    all_distances.append(l2_dist)
                    all_deltas.append(delta)

        if len(all_distances) > 0:
            # bin by delta_t and plot mean/std
            deltas = np.array(all_deltas)
            distances = np.array(all_distances)

            unique_deltas = np.unique(deltas)
            means = [distances[deltas == d].mean() for d in unique_deltas]
            stds = [distances[deltas == d].std() for d in unique_deltas]

            ax.errorbar(unique_deltas, means, yerr=stds, alpha=0.7, capsize=2)

        ax.set_xlabel("delta_t (bins)")
        ax.set_ylabel("L2 distance")
        ax.set_title(f"{cond.name}")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info(f"  saved {output_path}")


def plot_tsne(
    conditions: list[ConditionData],
    W_enc: nn.Parameter,
    cfg: Config,
    output_path: Path,
) -> None:
    """plot t-SNE colored by phase and time for each condition.

    args:
        conditions: list of ConditionData
        W_enc: trained encoder
        cfg: config
        output_path: output file path
    """
    num_conds = len(conditions)
    fig, axes = plt.subplots(num_conds, 2, figsize=(14, 4 * num_conds), squeeze=False)

    for idx, cond in enumerate(conditions):
        ax_phase = axes[idx, 0]
        ax_time = axes[idx, 1]

        # subsample bins
        num_points = min(cfg.tsne_max_points, cond.num_bins)
        sample_bins = torch.randperm(cond.num_bins)[:num_points]

        embeddings = []
        times = []
        valid_bins = []

        with torch.no_grad():
            for t_tensor in sample_bins:
                t = t_tensor.item()
                obs_n, obs_v = get_bin_observations(cond, t)
                if len(obs_n) < 10:
                    continue
                z = encode_sparse(obs_n, obs_v, W_enc, cond.num_neurons)
                embeddings.append(z.cpu().numpy())
                times.append(t)
                valid_bins.append(t)

        if len(embeddings) < 10:
            ax_phase.text(0.5, 0.5, "not enough data", ha="center", va="center", transform=ax_phase.transAxes)
            ax_time.text(0.5, 0.5, "not enough data", ha="center", va="center", transform=ax_time.transAxes)
            continue

        embeddings = np.stack(embeddings)
        times = np.array(times)

        # run t-SNE
        tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings) - 1), random_state=cfg.seed)
        coords = tsne.fit_transform(embeddings)

        # compute phase (for cyclical coloring)
        # assume avg_bins_per_frame gives us a natural period
        period = cond.avg_bins_per_frame * 10  # ~10 frames as period
        phases = (times % period) / period

        # plot by phase
        scatter1 = ax_phase.scatter(coords[:, 0], coords[:, 1], c=phases, cmap="hsv", alpha=0.6, s=10)
        ax_phase.set_title(f"{cond.name} - phase")
        ax_phase.set_xlabel("t-SNE 1")
        ax_phase.set_ylabel("t-SNE 2")
        plt.colorbar(scatter1, ax=ax_phase, label="phase")

        # plot by time
        scatter2 = ax_time.scatter(coords[:, 0], coords[:, 1], c=times, cmap="viridis", alpha=0.6, s=10)
        ax_time.set_title(f"{cond.name} - time")
        ax_time.set_xlabel("t-SNE 1")
        ax_time.set_ylabel("t-SNE 2")
        plt.colorbar(scatter2, ax=ax_time, label="bin index")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info(f"  saved {output_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    """CLI entry point."""
    # parse expt_code from first argument
    if len(sys.argv) < 2:
        print("usage: python zapbench_z0_enc_dec.py <expt_code> [--overrides]", flush=True)
        print("  example: python zapbench_z0_enc_dec.py my_expt --latent_dim 128", flush=True)
        sys.exit(1)

    expt_code = sys.argv[1]
    if not re.match(r"^[A-Za-z0-9_]+$", expt_code):
        print(f"error: expt_code must match [A-Za-z0-9_]+, got: {expt_code}", flush=True)
        sys.exit(1)

    # remaining args are tyro overrides
    tyro_args = sys.argv[2:]

    # parse config with tyro
    cfg = tyro.cli(Config, args=tyro_args)

    # get git commit hash and create run directory
    commit_hash = get_git_commit_hash()
    run_dir = create_run_directory(
        expt_code=expt_code,
        tyro_args=tyro_args,
        model_class=Config,
        commit_hash=commit_hash,
    )

    # save command line
    with open(run_dir / "command_line.txt", "w") as f:
        f.write("\n".join(sys.argv))

    # save config
    config_path = run_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg.model_dump(), f, sort_keys=False, indent=2)

    log.info(f"run directory: {run_dir.resolve()}")
    log.info(f"config saved to {config_path}")
    print(f"run directory: {run_dir.resolve()}", flush=True)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device: {device}")

    seed_everything(cfg.seed)

    # load data
    split = DataSplit()

    log.info("loading train data...")
    train_conditions = load_all_conditions(
        cfg.traces_path, cfg.ephys_path, cfg.bin_size_ms, split, "train", device,
        max_conditions=cfg.max_conditions,
    )

    log.info("loading val data...")
    val_conditions = load_all_conditions(
        cfg.traces_path, cfg.ephys_path, cfg.bin_size_ms, split, "val", device,
        max_conditions=cfg.max_conditions,
    )

    # train decoder (z0_bank + decoder)
    log.info("training decoder...")
    z0_bank, W_dec, b_dec, z0_bin_indices, bin_stride = train_decoder(
        train_conditions, cfg, device,
    )

    # train encoder
    log.info("training encoder...")
    W_enc = train_encoder(train_conditions, z0_bank, bin_stride, cfg, device)

    # create plots directory
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # generate diagnostic plots
    log.info("generating plots...")

    # training data plots
    plot_l2_distance(train_conditions, W_enc, cfg, plots_dir / "train_l2_distance.png")
    plot_tsne(train_conditions, W_enc, cfg, plots_dir / "train_tsne.png")

    # validation data plots
    plot_l2_distance(val_conditions, W_enc, cfg, plots_dir / "val_l2_distance.png")
    plot_tsne(val_conditions, W_enc, cfg, plots_dir / "val_tsne.png")

    # save model
    model_path = run_dir / "model.pt"
    torch.save({
        "W_enc": W_enc.detach().cpu(),
        "W_dec": W_dec.detach().cpu(),
        "b_dec": b_dec.detach().cpu(),
        "z0_bank": z0_bank.detach().cpu(),
        "z0_bin_indices": z0_bin_indices.cpu(),
        "bin_stride": bin_stride,
        "config": cfg.model_dump(),
    }, model_path)
    log.info(f"saved model to {model_path}")

    # create completion flag
    (run_dir / "complete").touch()

    log.info("done")


if __name__ == "__main__":
    main()
