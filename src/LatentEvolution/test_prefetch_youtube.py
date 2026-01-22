"""
diagnostic script to test chunk prefetching with real youtube zarr data.

simulates training by doing GPU computation on loaded chunks to see if
background loading stays ahead under realistic GPU memory pressure.
"""

import time
from pathlib import Path

import torch

from LatentEvolution.chunk_loader import RandomChunkLoader
from LatentEvolution.chunk_streaming import create_zarr_loader
from LatentEvolution.load_flyvis import FlyVisSim


def simulate_gpu_training(chunk_data: torch.Tensor, target_duration_s: float = 1.0):
    """simulate gpu training using actual chunk data.

    args:
        chunk_data: loaded chunk on gpu (chunk_size, num_neurons)
        target_duration_s: how long to run computation (seconds)
    """
    device = chunk_data.device
    num_neurons = chunk_data.shape[1]

    # create fake weight matrix for simulation
    weights = torch.randn(num_neurons, num_neurons // 2, device=device)

    start = time.time()
    iterations = 0
    while time.time() - start < target_duration_s:
        # simulate forward pass: matmul + nonlinearity
        output = torch.mm(chunk_data, weights)
        output = torch.relu(output)

        # simulate backward pass: gradient computation
        grad = torch.randn_like(output)
        _ = (output * grad).sum()  # simulate loss computation

        if device.type == 'cuda':
            torch.cuda.synchronize()  # ensure computation completes

        iterations += 1

    return iterations


def test_prefetch_youtube():
    """test prefetching with youtube dataset."""

    # check GPU availability
    if not torch.cuda.is_available():
        print("error: CUDA not available, this test requires GPU", flush=True)
        return

    device = torch.device('cuda')
    print(f"using device: {device}", flush=True)

    # youtube dataset config
    data_path = "graphs_data/fly/fly_N9_62_1_youtube-vos_calcium/x_list_0"
    column_idx = FlyVisSim.CALCIUM.value
    stim_column_idx = FlyVisSim.STIMULUS.value
    num_stim_dims = 1736

    # check dataset exists
    if not Path(data_path).exists():
        print(f"error: dataset not found at {data_path}")
        return

    print(f"loading from: {data_path}", flush=True)
    print(f"column: {column_idx} (CALCIUM)", flush=True)
    print(f"stimulus dims: {num_stim_dims}", flush=True)

    # create zarr loader
    zarr_load_fn = create_zarr_loader(
        data_path=data_path,
        column_idx=column_idx,
        stim_column_idx=stim_column_idx,
        num_stim_dims=num_stim_dims,
    )

    # chunk loader parameters
    chunk_size = 65536
    train_start = 0
    train_end = 967680  # typical train split
    train_total_timesteps = train_end - train_start

    print(f"chunk_size: {chunk_size}", flush=True)
    print(f"train timesteps: {train_total_timesteps}", flush=True)
    print("simulating 1s GPU computation per chunk\n", flush=True)

    # test with different prefetch values
    for prefetch in [2, 4, 6]:
        print(f"\n{'='*70}", flush=True)
        print(f"testing with prefetch={prefetch}", flush=True)
        print(f"{'='*70}\n", flush=True)

        loader = RandomChunkLoader(
            load_fn=lambda start, end: zarr_load_fn(train_start + start, train_start + end),
            total_timesteps=train_total_timesteps,
            chunk_size=chunk_size,
            device=device,  # cuda device
            prefetch=prefetch,
            seed=42
        )

        # simulate one epoch with 10 chunks
        num_chunks = 10
        loader.start_epoch(num_chunks=num_chunks)

        chunk_get_times = []
        queue_sizes_before = []
        total_iterations = 0

        print(f"{'Chunk':<6} {'QueueSize':<11} {'GetTime(ms)':<13} {'Iters':<8} {'Shape':<20}", flush=True)
        print("-" * 80, flush=True)

        for i in range(num_chunks):
            # check queue size before getting chunk
            queue_size = loader.cpu_queue.qsize()
            queue_sizes_before.append(queue_size)

            # get chunk (measure time)
            get_start = time.time()
            chunk_data, chunk_stim = loader.get_next_chunk()
            get_time = time.time() - get_start
            chunk_get_times.append(get_time)

            if chunk_data is None:
                print(f"{i:<6} end of epoch", flush=True)
                break

            # simulate GPU training (1 second per chunk)
            iterations = simulate_gpu_training(chunk_data, target_duration_s=1.0)
            total_iterations += iterations

            print(f"{i:<6} {queue_size:<11} {get_time*1000:<13.1f} {iterations:<8} {str(chunk_data.shape):<20}", flush=True)

        loader.cleanup()

        # summary
        print(f"\n{'='*80}", flush=True)
        print(f"summary for prefetch={prefetch}", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"mean get_time:       {sum(chunk_get_times) / len(chunk_get_times) * 1000:.1f}ms", flush=True)
        print(f"min get_time:        {min(chunk_get_times) * 1000:.1f}ms", flush=True)
        print(f"max get_time:        {max(chunk_get_times) * 1000:.1f}ms", flush=True)
        print(f"total iterations:    {total_iterations}", flush=True)
        print(f"queue sizes:         {queue_sizes_before}", flush=True)
        print(f"queue empty count:   {queue_sizes_before.count(0)}/{len(queue_sizes_before)} times", flush=True)
        print(f"queue full count:    {queue_sizes_before.count(prefetch)}/{len(queue_sizes_before)} times", flush=True)

        # interpretation
        if queue_sizes_before.count(0) > num_chunks // 2:
            print(f"\n⚠️  WARNING: Queue empty {queue_sizes_before.count(0)}/{num_chunks} times", flush=True)
            print("    Background thread is NOT keeping up with training", flush=True)
        else:
            print("\n✓  Queue had items ready most of the time", flush=True)
            print("    Background prefetching is working!", flush=True)


if __name__ == "__main__":
    test_prefetch_youtube()
