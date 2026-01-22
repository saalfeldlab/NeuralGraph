"""
diagnostic script to test chunk prefetching with real youtube zarr data.

simulates training by sleeping for 3.7s per chunk to see if background
loading stays ahead.
"""

import time
from pathlib import Path

from LatentEvolution.chunk_loader import RandomChunkLoader
from LatentEvolution.chunk_streaming import create_zarr_loader
from LatentEvolution.load_flyvis import FlyVisSim


def test_prefetch_youtube():
    """test prefetching with youtube dataset."""

    # youtube dataset config
    data_path = "graphs_data/fly/fly_N9_62_1_youtube-vos_calcium/x_list_0"
    column_idx = FlyVisSim.CALCIUM.value
    stim_column_idx = FlyVisSim.STIMULUS.value
    num_stim_dims = 1736

    # check dataset exists
    if not Path(data_path).exists():
        print(f"error: dataset not found at {data_path}")
        return

    print(f"loading from: {data_path}")
    print(f"column: {column_idx} (CALCIUM)")
    print(f"stimulus dims: {num_stim_dims}")

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

    print(f"chunk_size: {chunk_size}")
    print(f"train timesteps: {train_total_timesteps}")
    print("simulating 3.7s training per chunk\n")

    # test with different prefetch values
    for prefetch in [2, 4, 6]:
        print(f"\n{'='*70}")
        print(f"testing with prefetch={prefetch}")
        print(f"{'='*70}\n")

        loader = RandomChunkLoader(
            load_fn=lambda start, end: zarr_load_fn(train_start + start, train_start + end),
            total_timesteps=train_total_timesteps,
            chunk_size=chunk_size,
            device='cpu',  # use cpu to isolate disk i/o
            prefetch=prefetch,
            seed=42
        )

        # simulate one epoch with 10 chunks
        num_chunks = 10
        loader.start_epoch(num_chunks=num_chunks)

        chunk_get_times = []
        queue_sizes_before = []

        print(f"{'Chunk':<6} {'QueueSize':<11} {'GetTime(ms)':<13} {'Shape':<20}")
        print("-" * 70)

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
                print(f"{i:<6} end of epoch")
                break

            print(f"{i:<6} {queue_size:<11} {get_time*1000:<13.1f} {str(chunk_data.shape):<20}")

            # simulate training (3.7 seconds per chunk)
            time.sleep(3.7)

        loader.cleanup()

        # summary
        print(f"\n{'='*70}")
        print(f"summary for prefetch={prefetch}")
        print(f"{'='*70}")
        print(f"mean get_time:       {sum(chunk_get_times) / len(chunk_get_times) * 1000:.1f}ms")
        print(f"min get_time:        {min(chunk_get_times) * 1000:.1f}ms")
        print(f"max get_time:        {max(chunk_get_times) * 1000:.1f}ms")
        print(f"queue sizes:         {queue_sizes_before}")
        print(f"queue empty count:   {queue_sizes_before.count(0)}/{len(queue_sizes_before)} times")
        print(f"queue full count:    {queue_sizes_before.count(prefetch)}/{len(queue_sizes_before)} times")

        # interpretation
        if queue_sizes_before.count(0) > num_chunks // 2:
            print(f"\n⚠️  WARNING: Queue empty {queue_sizes_before.count(0)}/{num_chunks} times")
            print("    Background thread is NOT keeping up with training")
        else:
            print("\n✓  Queue had items ready most of the time")
            print("    Background prefetching is working!")


if __name__ == "__main__":
    test_prefetch_youtube()
