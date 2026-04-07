"""zapbench demo: minimal training loop with linear EED model.

demonstrates key infrastructure:
- sparse data loading from zarr
- GPU interpolation
- background CPU validation
- staggered observation masking

no artifacts created - prints to stdout only.

usage:
    python zapbench_demo.py
"""

import logging
import queue
import threading
import time

import numpy as np
import torch
import torch.nn as nn

from LatentEvolution.zapbench_config import (
    DataSplit,
    ConditionSplit,
)
from LatentEvolution.zapbench_train import (
    ConditionData,
    load_all_data,
    ValidationResult,
    log_validation_result,
)

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# hardcoded config
# ---------------------------------------------------------------------------

TRACES_PATH = "/groups/saalfeld/saalfeldlab/zapbench-release/volumes/20240930/traces"
EPHYS_PATH = "/groups/saalfeld/saalfeldlab/zapbench-processed/ephys.zarr"
BIN_SIZE_MS = 100.0

LATENT_DIMS = 16
FITTING_WINDOW = 25  # bins (~2.5 seconds at 100ms bins)
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-4

# data split: explicit frame ranges
TRAIN_FRAMES = 100  # limit training data for fast demo


# ---------------------------------------------------------------------------
# linear EED model
# ---------------------------------------------------------------------------


class LinearEED(nn.Module):
    """linear encoder-evolver-decoder.

    encoder: x @ W_enc + b_enc  (N -> L)
    decoder: z @ W_dec + b_dec  (L -> N)
    evolver: z + z @ W_evo      (L -> L, W_evo initialized to 0)
    """

    def __init__(self, num_neurons: int, latent_dims: int = 64):
        super().__init__()
        self.num_neurons = num_neurons
        self.latent_dims = latent_dims

        self.encoder = nn.Linear(num_neurons, latent_dims)
        self.decoder = nn.Linear(latent_dims, num_neurons)
        self.evolver_w = nn.Linear(latent_dims, latent_dims, bias=False)

        # zero init evolver so it starts as identity
        nn.init.zeros_(self.evolver_w.weight)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """encode neural activity to latent. x: (B, N) -> z: (B, L)"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """decode latent to neural activity. z: (B, L) -> x: (B, N)"""
        return self.decoder(z)

    def evolve(self, z: torch.Tensor) -> torch.Tensor:
        """evolve latent one step. z: (B, L) -> z_next: (B, L)"""
        return z + self.evolver_w(z)


# ---------------------------------------------------------------------------
# train step (uncompiled for demo - avoids torch.compile CPU overhead)
# ---------------------------------------------------------------------------


def train_step(
    model: LinearEED,
    batch: torch.Tensor,
    obs_mask: torch.Tensor,
) -> torch.Tensor:
    """training step: encode, evolve, decode, masked loss."""
    device = batch.device
    T = batch.shape[1]

    z = model.encode(batch[:, 0, :])

    loss = torch.tensor(0.0, device=device)
    for t in range(T):
        x_pred = model.decode(z)
        error_sq = (x_pred - batch[:, t, :]) ** 2
        mask_t = obs_mask[:, t, :]
        loss = loss + (error_sq * mask_t).sum() / mask_t.sum().clamp(min=1)
        z = model.evolve(z)

    return loss


# ---------------------------------------------------------------------------
# data split: explicit construction
# ---------------------------------------------------------------------------

# explicit split: dots for train (first 100 frames), gain for val, flash for test
DEMO_SPLIT = DataSplit(conditions=[
    ConditionSplit(condition_name="dots", train=(0, TRAIN_FRAMES), val=None, test=None),
    ConditionSplit(condition_name="gain", train=None, val=(0, 34), test=None),
    ConditionSplit(condition_name="flash", train=None, val=None, test=(0, 34)),
])


# ---------------------------------------------------------------------------
# validation (runs on CPU in background thread)
# same logic as zapbench_train.run_validation_cpu but uses LinearEED
# ---------------------------------------------------------------------------


def run_validation_cpu(
    num_neurons: int,
    latent_dims: int,
    state_dict: dict,
    val_conditions: list[ConditionData],
    result_queue: queue.Queue,
    epoch: int,
    max_rollout_frames: int | None = None,
) -> None:
    """run validation on CPU, put ValidationResult in queue."""
    # create model on CPU
    model = LinearEED(num_neurons, latent_dims).cpu()
    model.load_state_dict(state_dict)
    model.eval()

    results_mse: dict[str, np.ndarray] = {}
    results_mae: dict[str, np.ndarray] = {}
    baseline_mse: dict[str, np.ndarray] = {}
    baseline_mae: dict[str, np.ndarray] = {}

    with torch.no_grad():
        for cond in val_conditions:
            T = cond.activity.shape[0]
            start = cond.valid_start
            rollout_len = T - start

            if rollout_len <= 0:
                continue

            gt = cond.activity[start:].float()  # (rollout_len, N)
            frame_idx = cond.frame_index[start:]  # (rollout_len, N)
            obs_mask = frame_idx >= 0

            # encode initial state
            z = model.encode(gt[0:1])

            # collect predictions
            preds = torch.empty(rollout_len, cond.num_neurons)
            for t in range(rollout_len):
                preds[t] = model.decode(z)[0]
                z = model.evolve(z)

            # compute errors
            errors_sq = (preds - gt) ** 2
            errors_abs = (preds - gt).abs()

            # per-frame metrics
            num_frames = cond.num_frames if max_rollout_frames is None else min(cond.num_frames, max_rollout_frames)
            frame_mse_sum = torch.zeros(num_frames)
            frame_mae_sum = torch.zeros(num_frames)
            frame_counts = torch.zeros(num_frames)

            bin_indices, neuron_indices = torch.where(obs_mask)
            frame_indices_abs = frame_idx[bin_indices, neuron_indices].long()
            min_frame = frame_indices_abs.min().item() if len(frame_indices_abs) > 0 else 0
            frame_indices = frame_indices_abs - min_frame

            valid_mask = frame_indices < num_frames
            bin_indices = bin_indices[valid_mask]
            neuron_indices = neuron_indices[valid_mask]
            frame_indices = frame_indices[valid_mask]

            obs_errors_sq = errors_sq[bin_indices, neuron_indices]
            obs_errors_abs = errors_abs[bin_indices, neuron_indices]

            frame_mse_sum.scatter_add_(0, frame_indices, obs_errors_sq)
            frame_mae_sum.scatter_add_(0, frame_indices, obs_errors_abs)
            frame_counts.scatter_add_(0, frame_indices, torch.ones_like(obs_errors_sq))

            frame_mse = frame_mse_sum / frame_counts.clamp(min=1)
            frame_mae = frame_mae_sum / frame_counts.clamp(min=1)

            results_mse[cond.name] = frame_mse.numpy()
            results_mae[cond.name] = frame_mae.numpy()

            # baseline: predict mean of first 4 frames
            baseline_frames = 4
            frame_idx_rel = frame_idx - min_frame
            first_frames_mask = (frame_idx_rel >= 0) & (frame_idx_rel < baseline_frames)
            first_frames_obs = obs_mask & first_frames_mask
            neuron_sum = torch.zeros(cond.num_neurons)
            neuron_count = torch.zeros(cond.num_neurons)
            obs_bin, obs_neuron = torch.where(first_frames_obs)
            neuron_sum.scatter_add_(0, obs_neuron, gt[obs_bin, obs_neuron])
            neuron_count.scatter_add_(0, obs_neuron, torch.ones(len(obs_neuron)))
            mean_per_neuron = neuron_sum / neuron_count.clamp(min=1)

            baseline_errors_sq = (mean_per_neuron.unsqueeze(0) - gt) ** 2
            baseline_errors_abs = (mean_per_neuron.unsqueeze(0) - gt).abs()

            bl_mse_sum = torch.zeros(num_frames)
            bl_mae_sum = torch.zeros(num_frames)
            bl_obs_errors_sq = baseline_errors_sq[bin_indices, neuron_indices]
            bl_obs_errors_abs = baseline_errors_abs[bin_indices, neuron_indices]
            bl_mse_sum.scatter_add_(0, frame_indices, bl_obs_errors_sq)
            bl_mae_sum.scatter_add_(0, frame_indices, bl_obs_errors_abs)

            baseline_mse[cond.name] = (bl_mse_sum / frame_counts.clamp(min=1)).numpy()
            baseline_mae[cond.name] = (bl_mae_sum / frame_counts.clamp(min=1)).numpy()

    mean_mse = float(np.mean([m.mean() for m in results_mse.values()])) if results_mse else 0.0
    mean_mae = float(np.mean([m.mean() for m in results_mae.values()])) if results_mae else 0.0

    result_queue.put(ValidationResult(
        epoch=epoch,
        mean_mse=mean_mse,
        mean_mae=mean_mae,
        per_condition_mse=results_mse,
        per_condition_mae=results_mae,
        baseline_mse=baseline_mse,
        baseline_mae=baseline_mae,
    ))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    if not torch.cuda.is_available():
        log.warning("CUDA not available - training on CPU will be slow!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device: {device}")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # load data using shared infrastructure
    log.info("loading data...")
    train_data, val_data, test_data = load_all_data(
        traces_path=TRACES_PATH,
        ephys_path=EPHYS_PATH,
        bin_size_ms=BIN_SIZE_MS,
        split=DEMO_SPLIT,
        fitting_window=FITTING_WINDOW,
        device=device,
    )
    assert train_data is not None

    num_neurons = train_data.num_neurons
    log.info(f"num_neurons: {num_neurons}")
    log.info(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # model
    model = LinearEED(num_neurons, LATENT_DIMS).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    log.info(f"model parameters: {num_params:,}")

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # training setup
    batches_per_epoch = train_data.total_weight // BATCH_SIZE
    log.info(f"batches per epoch: {batches_per_epoch}")

    rng = torch.Generator(device=device)
    time_offsets = torch.arange(FITTING_WINDOW, device=device)

    # validation queue and thread
    val_queue: queue.Queue = queue.Queue()
    val_thread: threading.Thread | None = None

    def start_validation(epoch: int) -> threading.Thread:
        state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        t = threading.Thread(
            target=run_validation_cpu,
            args=(num_neurons, LATENT_DIMS, state_dict, val_data, val_queue, epoch),
        )
        t.start()
        return t

    def check_validation() -> ValidationResult | None:
        nonlocal val_thread
        if val_thread is not None and not val_thread.is_alive():
            result = val_queue.get_nowait()
            val_thread = None
            return result
        return None

    # start initial validation (epoch 0, untrained model)
    val_thread = start_validation(0)

    # training loop
    log.info(f"training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for _ in range(batches_per_epoch):
            batch, mask = train_data.sample_batch(BATCH_SIZE, time_offsets, rng)

            optimizer.zero_grad()
            loss = train_step(model, batch, mask)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / batches_per_epoch
        epoch_time = time.time() - epoch_start

        log.info(f"epoch {epoch}: loss={avg_loss:.4f}, time={epoch_time:.1f}s")

        # check if validation finished
        val_result = check_validation()
        if val_result is not None:
            log.info(f"validation (cpu) finished: val_mse={val_result.mean_mse:.6f} (epoch {val_result.epoch})")

        # start next validation if not busy
        if val_thread is None:
            val_thread = start_validation(epoch + 1)

    # wait for final validation
    if val_thread is not None:
        val_thread.join()
        val_result = val_queue.get()
        log_validation_result(val_result, FITTING_WINDOW, writer=None, prefix="val")

    # run test evaluation
    log.info("running test evaluation...")
    test_queue: queue.Queue = queue.Queue()
    test_thread = threading.Thread(
        target=run_validation_cpu,
        args=(num_neurons, LATENT_DIMS,
              {k: v.cpu().clone() for k, v in model.state_dict().items()},
              test_data, test_queue, EPOCHS),
    )
    test_thread.start()
    test_thread.join()
    test_result = test_queue.get()
    log_validation_result(test_result, FITTING_WINDOW, writer=None, prefix="test")

    log.info("done")


if __name__ == "__main__":
    main()
