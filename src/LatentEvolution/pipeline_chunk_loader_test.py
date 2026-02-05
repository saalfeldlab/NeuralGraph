"""
unit tests for pipeline_chunk_loader module.

tests verify:
- 3-stage pipeline (cpu_loader -> gpu_transfer -> main) works correctly
- overlap between stages provides speedup
- gpu_prefetch controls buffering on device
- edge cases: early termination, cleanup, multiple epochs
"""

import random
import time
import unittest
from typing import Callable

import numpy as np
import torch

from LatentEvolution.chunk_streaming import generate_random_chunks
from LatentEvolution.pipeline_chunk_loader import PipelineChunkLoader, PipelineProfiler


class MockDataSource:
    """simulates a large dataset on disk with configurable delays."""

    def __init__(
        self,
        total_timesteps: int,
        num_neurons: int,
        num_stim_dims: int,
        load_delay_ms: float = 0.0
    ):
        self.total_timesteps = total_timesteps
        self.num_neurons = num_neurons
        self.num_stim_dims = num_stim_dims
        self.load_delay_ms = load_delay_ms

        self.load_calls = []
        self.load_count = 0

    def load_slice(self, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        """load a slice of data (simulated)."""
        if self.load_delay_ms > 0:
            time.sleep(self.load_delay_ms / 1000.0)

        self.load_calls.append((start, end))
        self.load_count += 1

        size = end - start
        data = np.zeros((size, self.num_neurons), dtype=np.float32)
        stim = np.zeros((size, self.num_stim_dims), dtype=np.float32)

        return data, stim


class TestPipelineChunkLoader(unittest.TestCase):
    """test suite for PipelineChunkLoader."""

    def test_basic_loading(self):
        """test basic chunk loading on cpu."""
        source = MockDataSource(
            total_timesteps=100000,
            num_neurons=1000,
            num_stim_dims=100
        )

        loader = PipelineChunkLoader(
            load_fn=source.load_slice,
            device='cpu',
            prefetch=1,
            gpu_prefetch=1,
        )

        chunks = [(i * 10000, (i + 1) * 10000) for i in range(5)]
        loader.start_epoch(chunks)

        for _ in range(5):
            chunk_start, (chunk_data, chunk_stim) = loader.get_next_chunk()
            self.assertIsNotNone(chunk_start)
            self.assertEqual(chunk_data.shape, (10000, 1000))
            self.assertEqual(chunk_stim.shape, (10000, 100))
            self.assertEqual(chunk_data.device.type, 'cpu')

        # verify end of epoch
        end_start, end_payload = loader.get_next_chunk()
        self.assertIsNone(end_start)
        self.assertIsNone(end_payload)

        self.assertEqual(source.load_count, 5)
        loader.cleanup()

    def test_random_chunk_generation(self):
        """test that generate_random_chunks produces valid aligned chunks."""
        total_timesteps = 100000
        chunk_size = 5000
        time_units = 10

        chunks = generate_random_chunks(
            total_timesteps=total_timesteps,
            chunk_size=chunk_size,
            num_chunks=20,
            time_units=time_units,
            rng=random.Random(123),
        )

        self.assertEqual(len(chunks), 20)

        start_indices = [start for start, _ in chunks]
        self.assertGreater(len(set(start_indices)), 1, "all chunks from same location")

        for start, end in chunks:
            self.assertGreaterEqual(start, 0)
            self.assertLessEqual(start, total_timesteps - chunk_size)
            self.assertEqual(end, start + chunk_size)
            self.assertEqual(start % time_units, 0,
                f"chunk start {start} not aligned to time_units={time_units}")

    def test_pipeline_overlap(self):
        """test that 3-stage pipeline provides overlap speedup."""
        # each stage takes ~100ms (cpu->cpu transfer is instant)
        load_time_ms = 100
        train_time_ms = 100

        source = MockDataSource(
            total_timesteps=50000,
            num_neurons=1000,
            num_stim_dims=100,
            load_delay_ms=load_time_ms
        )

        loader = PipelineChunkLoader(
            load_fn=source.load_slice,
            device='cpu',
            prefetch=2,  # buffer 2 chunks in cpu_queue
            gpu_prefetch=2,  # buffer 2 chunks in gpu_queue
        )

        chunks = [(i * 5000, (i + 1) * 5000) for i in range(10)]
        loader.start_epoch(chunks)

        total_start = time.time()

        for i in range(10):
            get_start = time.time()
            _, _ = loader.get_next_chunk()
            get_time = time.time() - get_start

            # simulate training
            time.sleep(train_time_ms / 1000.0)

            print(f"chunk {i}: get={get_time*1000:.1f}ms")

        total_time = time.time() - total_start

        # sequential: 10 * (100ms load + 100ms train) = 2000ms
        # with overlap: ~1000ms + startup overhead
        sequential_time = 10 * (load_time_ms + train_time_ms) / 1000.0

        print(f"\ntotal time: {total_time*1000:.1f}ms")
        print(f"sequential would be: {sequential_time*1000:.1f}ms")
        print(f"speedup: {sequential_time / total_time:.2f}x")

        # expect speedup from overlap (relaxed threshold for system variability)
        self.assertLess(total_time, sequential_time * 0.85, "no overlap detected")

        loader.cleanup()

    def test_gpu_prefetch_buffering(self):
        """test that gpu_prefetch controls how many chunks buffer on device."""
        source = MockDataSource(
            total_timesteps=30000,
            num_neurons=500,
            num_stim_dims=50,
            load_delay_ms=30
        )

        # with gpu_prefetch=2, gpu_queue can hold 2 chunks
        loader = PipelineChunkLoader(
            load_fn=source.load_slice,
            device='cpu',
            prefetch=2,
            gpu_prefetch=2
        )

        chunks = [(i * 3000, (i + 1) * 3000) for i in range(5)]
        loader.start_epoch(chunks)

        # give pipeline time to fill up
        time.sleep(0.2)

        # first chunks should be ready immediately
        for i in range(5):
            start = time.time()
            _, _ = loader.get_next_chunk()
            get_time = time.time() - start

            print(f"chunk {i}: get_time={get_time*1000:.1f}ms")

            # first couple should be very fast (already buffered)
            if i < 2:
                self.assertLess(get_time, 0.05, f"chunk {i} should be buffered")

        loader.cleanup()

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cuda_transfer(self):
        """test cuda pipeline (if gpu available)."""
        source = MockDataSource(
            total_timesteps=20000,
            num_neurons=2000,
            num_stim_dims=200
        )

        loader = PipelineChunkLoader(
            load_fn=source.load_slice,
            device='cuda',
            prefetch=1,
            gpu_prefetch=2
        )

        chunks = [(i * 5000, (i + 1) * 5000) for i in range(3)]
        loader.start_epoch(chunks)

        for _ in range(3):
            _, (chunk_data, chunk_stim) = loader.get_next_chunk()
            self.assertEqual(chunk_data.device.type, 'cuda')
            self.assertEqual(chunk_stim.device.type, 'cuda')
            self.assertEqual(chunk_data.shape, (5000, 2000))

        loader.cleanup()

    def test_statistics(self):
        """test that loader tracks statistics correctly."""
        source = MockDataSource(
            total_timesteps=15000,
            num_neurons=100,
            num_stim_dims=50
        )

        loader = PipelineChunkLoader(
            load_fn=source.load_slice,
            device='cpu'
        )

        chunks = [(i * 3000, (i + 1) * 3000) for i in range(5)]
        loader.start_epoch(chunks)

        for _ in range(5):
            loader.get_next_chunk()

        stats = loader.get_stats()
        self.assertEqual(stats['chunks_loaded'], 5)
        self.assertEqual(stats['chunks_transferred'], 5)

        loader.cleanup()

    def test_cleanup_stops_pipeline(self):
        """test that cleanup stops both threads."""
        source = MockDataSource(
            total_timesteps=50000,
            num_neurons=100,
            num_stim_dims=50,
            load_delay_ms=100
        )

        loader = PipelineChunkLoader(
            load_fn=source.load_slice,
            device='cpu'
        )

        chunks = [(i * 5000, (i + 1) * 5000) for i in range(20)]
        loader.start_epoch(chunks)

        # get only 2 chunks
        loader.get_next_chunk()
        loader.get_next_chunk()

        loader.cleanup()

        # should not have loaded all 20 chunks
        self.assertLess(source.load_count, 20, "cleanup did not stop pipeline")

    def test_multiple_epochs(self):
        """test starting new epoch immediately after previous."""
        source = MockDataSource(
            total_timesteps=20000,
            num_neurons=500,
            num_stim_dims=50,
            load_delay_ms=30
        )

        loader = PipelineChunkLoader(
            load_fn=source.load_slice,
            device='cpu',
            prefetch=2,
            gpu_prefetch=2
        )

        # epoch 1
        chunks1 = [(i * 5000, (i + 1) * 5000) for i in range(3)]
        loader.start_epoch(chunks1)
        for _ in range(3):
            _, (chunk_data, _chunk_stim) = loader.get_next_chunk()
            self.assertIsNotNone(chunk_data)

        # epoch 2 immediately
        chunks2 = [(i * 5000, (i + 1) * 5000) for i in range(3)]
        loader.start_epoch(chunks2)
        for _ in range(3):
            _, (chunk_data, _chunk_stim) = loader.get_next_chunk()
            self.assertIsNotNone(chunk_data)

        loader.cleanup()

    def test_early_break_from_epoch(self):
        """test breaking out of epoch early."""
        source = MockDataSource(
            total_timesteps=30000,
            num_neurons=500,
            num_stim_dims=50,
            load_delay_ms=50
        )

        loader = PipelineChunkLoader(
            load_fn=source.load_slice,
            device='cpu',
            prefetch=2,
            gpu_prefetch=2
        )

        # epoch 1: ask for 5, consume only 2
        chunks1 = [(i * 5000, (i + 1) * 5000) for i in range(5)]
        loader.start_epoch(chunks1)
        loader.get_next_chunk()
        loader.get_next_chunk()

        # epoch 2 should work
        chunks2 = [(i * 5000, (i + 1) * 5000) for i in range(3)]
        loader.start_epoch(chunks2)
        for _ in range(3):
            _, (chunk_data, _chunk_stim) = loader.get_next_chunk()
            self.assertIsNotNone(chunk_data)

        loader.cleanup()

    def test_chunk_alignment_with_time_units(self):
        """test that generate_random_chunks aligns to time_units."""
        time_units = 10
        chunks = generate_random_chunks(
            total_timesteps=100000,
            chunk_size=5000,
            num_chunks=20,
            time_units=time_units,
            rng=random.Random(123),
        )

        chunk_starts = [start for start, _ in chunks]
        for start in chunk_starts:
            self.assertEqual(start % time_units, 0,
                f"chunk start {start} not aligned to time_units={time_units}")

        self.assertGreater(len(set(chunk_starts)), 1, "all chunks from same location")

    def test_deterministic_with_seed(self):
        """test that same seed produces same chunk sequence."""
        kwargs = dict(total_timesteps=30000, chunk_size=5000, num_chunks=3, time_units=1)

        chunks1 = generate_random_chunks(**kwargs, rng=random.Random(42))
        chunks2 = generate_random_chunks(**kwargs, rng=random.Random(42))

        self.assertEqual(chunks1, chunks2, "same seed should produce same sequence")

    def test_single_chunk(self):
        """test edge case: only one chunk requested."""
        source = MockDataSource(
            total_timesteps=10000,
            num_neurons=100,
            num_stim_dims=50
        )

        loader = PipelineChunkLoader(
            load_fn=source.load_slice,
            device='cpu',
            gpu_prefetch=2
        )

        loader.start_epoch([(0, 5000)])

        _, (chunk_data, _chunk_stim) = loader.get_next_chunk()
        self.assertIsNotNone(chunk_data)

        end_start, end_payload = loader.get_next_chunk()
        self.assertIsNone(end_start)
        self.assertIsNone(end_payload)

        loader.cleanup()

    def test_chunk_size_larger_than_dataset(self):
        """test chunk_size > dataset (returns full dataset)."""
        source = MockDataSource(
            total_timesteps=10000,
            num_neurons=100,
            num_stim_dims=50
        )

        loader = PipelineChunkLoader(
            load_fn=source.load_slice,
            device='cpu'
        )

        loader.start_epoch([(0, 10000), (0, 10000)])

        for _ in range(2):
            _, (chunk_data, _chunk_stim) = loader.get_next_chunk()
            self.assertEqual(chunk_data.shape, (10000, 100))

        loader.cleanup()


class TestPipelineTimingCombinations(unittest.TestCase):
    """stress test pipeline with random timing combinations."""

    def test_random_timing_combinations(self):
        """test pipeline with randomly varying delays for each stage.

        runs for ~10s, randomly sampling short/long delays for:
        - cpu_loader (disk i/o)
        - gpu_transfer (cpu->gpu)
        - training (main thread)

        this tests all bottleneck scenarios:
        - training-bound (slow training, fast load)
        - cpu-bound (slow disk, fast training)
        - gpu-bound (slow transfer, fast training)
        - balanced (all similar)
        """
        import random as rand

        # timing parameters (ms)
        short_delay = 10
        long_delay = 80

        # test duration
        target_duration_s = 10.0
        chunks_per_iteration = 5

        rng = rand.Random(42)
        iterations = 0
        total_chunks = 0
        start_time = time.time()

        print("\n" + "=" * 70)
        print("pipeline timing stress test (~10s)")
        print("=" * 70)

        while time.time() - start_time < target_duration_s:
            # randomly sample delays for this iteration
            load_delay = rng.choice([short_delay, long_delay])
            transfer_delay = rng.choice([short_delay, long_delay])
            train_delay = rng.choice([short_delay, long_delay])

            # identify bottleneck for logging
            delays = {'load': load_delay, 'transfer': transfer_delay, 'train': train_delay}
            bottleneck = max(delays.keys(), key=lambda k: delays[k])

            source = MockDataSourceWithTransferDelay(
                total_timesteps=50000,
                num_neurons=100,
                num_stim_dims=50,
                load_delay_ms=load_delay,
                transfer_delay_ms=transfer_delay,
            )

            loader = PipelineChunkLoaderWithTransferDelay(
                load_fn=source.load_slice,
                transfer_delay_fn=source.get_transfer_delay,
                device='cpu',
                prefetch=2,
                gpu_prefetch=2,
            )

            chunks = [(i * 5000, (i + 1) * 5000) for i in range(chunks_per_iteration)]
            iter_start = time.time()
            loader.start_epoch(chunks)

            chunks_received = 0
            for _ in range(chunks_per_iteration):
                _, payload = loader.get_next_chunk()
                if payload is None:
                    break
                chunks_received += 1

                # simulate training
                time.sleep(train_delay / 1000.0)

            # verify end of epoch
            _, end_payload = loader.get_next_chunk()
            self.assertIsNone(end_payload, "expected end of epoch")

            loader.cleanup()

            iter_time = time.time() - iter_start
            total_chunks += chunks_received
            iterations += 1

            # verify we got all chunks
            self.assertEqual(chunks_received, chunks_per_iteration,
                f"expected {chunks_per_iteration} chunks, got {chunks_received}")

            print(f"iter {iterations:2d}: load={load_delay:2d}ms transfer={transfer_delay:2d}ms "
                  f"train={train_delay:2d}ms -> {bottleneck:>8s}-bound | "
                  f"{iter_time*1000:.0f}ms total")

        elapsed = time.time() - start_time
        print("=" * 70)
        print(f"completed {iterations} iterations, {total_chunks} chunks in {elapsed:.1f}s")
        print(f"average: {total_chunks/elapsed:.1f} chunks/s")
        print("=" * 70)

        # basic sanity checks
        self.assertGreater(iterations, 5, "should complete multiple iterations in 10s")
        self.assertEqual(total_chunks, iterations * chunks_per_iteration)


class MockDataSourceWithTransferDelay(MockDataSource):
    """mock data source that also provides transfer delay."""

    def __init__(
        self,
        total_timesteps: int,
        num_neurons: int,
        num_stim_dims: int,
        load_delay_ms: float = 0.0,
        transfer_delay_ms: float = 0.0,
    ):
        super().__init__(total_timesteps, num_neurons, num_stim_dims, load_delay_ms)
        self.transfer_delay_ms = transfer_delay_ms

    def get_transfer_delay(self) -> float:
        """return transfer delay in ms."""
        return self.transfer_delay_ms


class PipelineChunkLoaderWithTransferDelay(PipelineChunkLoader):
    """pipeline loader that injects artificial transfer delay for testing."""

    def __init__(
        self,
        transfer_delay_fn: Callable[[], float],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.transfer_delay_fn = transfer_delay_fn

    def _gpu_transfer_worker(self):
        """gpu_transfer thread with injected delay."""
        while True:
            if self.stop_flag:
                break

            item = self.cpu_queue.get()

            if item is None:
                self.gpu_queue.put(None)
                break

            start_idx, cpu_tensors = item

            # inject artificial transfer delay
            delay_ms = self.transfer_delay_fn()
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)

            # transfer to device
            gpu_tensors = self._do_transfer(cpu_tensors)

            self.chunks_transferred += 1
            self.gpu_queue.put((start_idx, gpu_tensors))


class TestPipelineProfiler(unittest.TestCase):
    """test suite for PipelineProfiler."""

    def test_profiler_records_events(self):
        """test that profiler records events from pipeline."""
        source = MockDataSource(
            total_timesteps=20000,
            num_neurons=100,
            num_stim_dims=50,
            load_delay_ms=10,
        )

        profiler = PipelineProfiler()
        profiler.start()

        loader = PipelineChunkLoader(
            load_fn=source.load_slice,
            device='cpu',
            prefetch=2,
            gpu_prefetch=2,
            profiler=profiler,
        )

        chunks = [(i * 5000, (i + 1) * 5000) for i in range(3)]
        loader.start_epoch(chunks)

        for _ in range(3):
            _, payload = loader.get_next_chunk()
            if payload is None:
                break
            # simulate training
            time.sleep(0.01)

        loader.cleanup()
        profiler.stop()

        # verify events were recorded
        self.assertGreater(len(profiler.events), 0)

        # check event names
        event_names = {e.name for e in profiler.events}
        self.assertIn("disk_load", event_names)
        self.assertIn("gpu_transfer", event_names)
        self.assertIn("gpu_queue_wait", event_names)

        # verify stats
        stats = profiler.get_stats()
        self.assertIn("disk_load", stats)
        self.assertEqual(stats["disk_load"]["count"], 3)

    def test_profiler_chrome_trace_format(self):
        """test that chrome trace output has correct format."""
        profiler = PipelineProfiler()
        profiler.start()

        with profiler.event("test_event", "test", thread="main"):
            time.sleep(0.001)

        profiler.stop()

        trace = profiler.to_chrome_trace()

        # verify structure
        self.assertIn("traceEvents", trace)
        events = trace["traceEvents"]

        # should have metadata events + our event
        self.assertGreater(len(events), 1)

        # find our event
        test_events = [e for e in events if e.get("name") == "test_event"]
        self.assertEqual(len(test_events), 1)

        event = test_events[0]
        self.assertEqual(event["cat"], "test")
        self.assertEqual(event["ph"], "X")  # complete event
        self.assertIn("ts", event)
        self.assertIn("dur", event)
        self.assertEqual(event["tid"], 0)  # main thread

    def test_profiler_disabled_by_default(self):
        """test that profiler doesn't record when not started."""
        profiler = PipelineProfiler()

        # don't call start()
        with profiler.event("test_event", "test"):
            pass

        self.assertEqual(len(profiler.events), 0)


if __name__ == "__main__":
    unittest.main()
