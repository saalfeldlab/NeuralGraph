"""
integration of chunked streaming into latent.py training loop.

provides instrumentation and helper functions for using RandomChunkLoader
with zarr datasets.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from NeuralGraph.zarr_io import load_column_slice


# -------------------------------------------------------------------
# Latency Instrumentation
# -------------------------------------------------------------------


@dataclass
class ChunkLatencyStats:
    """tracks latencies for chunked data loading."""

    # chunk loading latencies
    chunk_load_times: list[float] = field(default_factory=list)
    chunk_transfer_times: list[float] = field(default_factory=list)
    chunk_get_times: list[float] = field(default_factory=list)

    # per-batch training latencies (sampled)
    batch_forward_times: list[float] = field(default_factory=list)
    batch_backward_times: list[float] = field(default_factory=list)
    batch_step_times: list[float] = field(default_factory=list)

    def record_chunk_get(self, duration_s: float):
        """record time to get next chunk (includes waiting for background load)."""
        self.chunk_get_times.append(duration_s)

    def record_batch_times(self, forward_s: float, backward_s: float, step_s: float):
        """record training times for a batch."""
        self.batch_forward_times.append(forward_s)
        self.batch_backward_times.append(backward_s)
        self.batch_step_times.append(step_s)

    def get_summary(self) -> dict[str, float]:
        """get summary statistics (means in milliseconds)."""
        return {
            'chunk_get_mean_ms': float(np.mean(self.chunk_get_times) * 1000) if self.chunk_get_times else 0.0,
            'chunk_get_max_ms': float(np.max(self.chunk_get_times) * 1000) if self.chunk_get_times else 0.0,
            'chunk_get_min_ms': float(np.min(self.chunk_get_times) * 1000) if self.chunk_get_times else 0.0,
            'batch_forward_mean_ms': float(np.mean(self.batch_forward_times) * 1000) if self.batch_forward_times else 0.0,
            'batch_backward_mean_ms': float(np.mean(self.batch_backward_times) * 1000) if self.batch_backward_times else 0.0,
            'batch_step_mean_ms': float(np.mean(self.batch_step_times) * 1000) if self.batch_step_times else 0.0,
            'num_chunks': float(len(self.chunk_get_times)),
            'num_batches_sampled': float(len(self.batch_forward_times)),
        }

    def print_summary(self):
        """print summary statistics."""
        summary = self.get_summary()
        print("\n=== Chunked Streaming Latency Stats ===")
        print(f"chunks loaded: {summary['num_chunks']}")
        print(f"chunk get time: mean={summary['chunk_get_mean_ms']:.1f}ms, "
              f"min={summary['chunk_get_min_ms']:.1f}ms, max={summary['chunk_get_max_ms']:.1f}ms")
        if summary['num_batches_sampled'] > 0:
            print(f"batch timings ({summary['num_batches_sampled']} samples):")
            print(f"  forward:  {summary['batch_forward_mean_ms']:.2f}ms")
            print(f"  backward: {summary['batch_backward_mean_ms']:.2f}ms")
            print(f"  step:     {summary['batch_step_mean_ms']:.2f}ms")


# -------------------------------------------------------------------
# Zarr Loader Function
# -------------------------------------------------------------------


def create_zarr_loader(
    data_path: str,
    column_idx: int,
    stim_column_idx: int,
    num_stim_dims: int,
) -> Callable[[int, int], tuple[np.ndarray, np.ndarray]]:
    """create a load function for RandomChunkLoader that reads from zarr.

    args:
        data_path: path to zarr dataset (e.g., "graphs_data/fly/fly_N9_62_1/x_list_0")
        column_idx: column index for neural data (e.g., FlyVisSim.VOLTAGE.value)
        stim_column_idx: column index for stimulus (e.g., FlyVisSim.STIMULUS.value)
        num_stim_dims: number of stimulus dimensions to load

    returns:
        function(start_idx, end_idx) -> (data, stim) as numpy arrays
    """
    def load_fn(start_idx: int, end_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """load a slice from zarr."""
        # load neural data
        data = load_column_slice(data_path, column_idx, start_idx, end_idx)

        # load stimulus
        stim = load_column_slice(
            data_path, stim_column_idx, start_idx, end_idx,
            neuron_limit=num_stim_dims
        )

        return data, stim

    return load_fn


# -------------------------------------------------------------------
# Batch Sampling Within Chunk
# -------------------------------------------------------------------


def sample_batch_within_chunk(
    chunk_data: torch.Tensor,
    _chunk_stim: torch.Tensor,
    wmat_indices: torch.Tensor,
    wmat_indptr: torch.Tensor,
    batch_size: int,
    total_steps: int,
    num_neurons_to_zero: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """sample a random batch within the current chunk.

    args:
        chunk_data: current chunk on GPU (chunk_size, num_neurons)
        _chunk_stim: current stimulus chunk on GPU (unused, kept for API consistency)
        wmat_indices: connectome column indices
        wmat_indptr: connectome row pointers
        batch_size: number of samples in batch
        total_steps: number of evolution steps (time_units * evolve_multiple_steps)
        num_neurons_to_zero: number of neurons for augmentation
        device: torch device

    returns:
        (batch_indices, selected_neurons, needed_indices)
    """
    chunk_size = chunk_data.shape[0]
    num_neurons = chunk_data.shape[1]

    # sample random start indices within chunk bounds
    max_start_idx = chunk_size - total_steps
    if max_start_idx <= 0:
        raise ValueError(
            f"chunk_size ({chunk_size}) must be > total_steps ({total_steps})"
        )

    batch_indices = torch.randint(
        low=0, high=max_start_idx, size=(batch_size,), device=device
    )

    # generate augmentation indices
    if num_neurons_to_zero > 0:
        selected_neurons = torch.randint(
            low=0, high=num_neurons, size=(num_neurons_to_zero,), device=device
        )
        needed_indices = torch.concatenate(
            [wmat_indices[wmat_indptr[i]:wmat_indptr[i+1]] for i in selected_neurons]
        )
        needed_indices = torch.unique(torch.concatenate([needed_indices, selected_neurons]))
    else:
        selected_neurons = torch.empty(0, dtype=torch.long, device=device)
        needed_indices = torch.empty(0, dtype=torch.long, device=device)

    return batch_indices, selected_neurons, needed_indices


# -------------------------------------------------------------------
# Training Loop Integration Helper
# -------------------------------------------------------------------


def calculate_chunk_params(
    total_timesteps: int,
    chunk_size: int,
    batch_size: int,
    data_passes_per_epoch: int,
) -> tuple[int, int, int]:
    """calculate chunking parameters for an epoch.

    args:
        total_timesteps: total number of timesteps in dataset
        chunk_size: size of each chunk (e.g., 65536)
        batch_size: batch size
        data_passes_per_epoch: how many times to pass over data per epoch

    returns:
        (chunks_per_epoch, batches_per_chunk, batches_per_epoch)
    """
    # total batches per epoch (based on data passes)
    batches_per_epoch = (total_timesteps // batch_size) * data_passes_per_epoch
    if batches_per_epoch == 0:
        raise ValueError(
            f"dataset too small: total_timesteps ({total_timesteps}) * "
            f"data_passes_per_epoch ({data_passes_per_epoch}) / batch_size ({batch_size}) = 0 batches"
        )

    # if dataset smaller than chunk_size, treat entire dataset as one chunk
    if total_timesteps < chunk_size:
        chunks_per_epoch = data_passes_per_epoch  # load once per data pass
        batches_per_chunk = total_timesteps // batch_size
    else:
        # batches per chunk
        batches_per_chunk = chunk_size // batch_size
        if batches_per_chunk == 0:
            raise ValueError(
                f"chunk_size ({chunk_size}) must be >= batch_size ({batch_size})"
            )

        # chunks needed per epoch
        chunks_per_epoch = batches_per_epoch // batches_per_chunk
        if chunks_per_epoch == 0:
            raise ValueError(
                f"not enough batches for even one chunk: batches_per_epoch ({batches_per_epoch}) < "
                f"batches_per_chunk ({batches_per_chunk}). either increase data_passes_per_epoch, "
                f"decrease chunk_size, or use a smaller chunk_size"
            )

    return chunks_per_epoch, batches_per_chunk, batches_per_epoch
