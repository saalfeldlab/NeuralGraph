"""
unit tests for chunk_loader module.

tests verify:
- random chunk loading works correctly
- background thread loads chunks asynchronously
- overlap between loading and "training" happens
- cuda stream usage works (if gpu available)
"""

import time
import unittest
import numpy as np
import torch

from LatentEvolution.chunk_loader import RandomChunkLoader


# -------------------------------------------------------------------
# Mock Data Source
# -------------------------------------------------------------------


class MockDataSource:
    """simulates a large dataset on disk with load delays."""

    def __init__(
        self,
        total_timesteps: int,
        num_neurons: int,
        num_stim_dims: int,
        load_delay_ms: float = 0.0
    ):
        """
        args:
            total_timesteps: size of dataset
            num_neurons: number of neurons
            num_stim_dims: stimulus dimensions
            load_delay_ms: simulated disk i/o delay in milliseconds
        """
        self.total_timesteps = total_timesteps
        self.num_neurons = num_neurons
        self.num_stim_dims = num_stim_dims
        self.load_delay_ms = load_delay_ms

        # track which ranges were loaded
        self.load_calls = []
        self.load_count = 0

    def load_slice(self, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        """load a slice of data (simulated)."""
        # simulate disk i/o delay
        if self.load_delay_ms > 0:
            time.sleep(self.load_delay_ms / 1000.0)

        # record call
        self.load_calls.append((start, end))
        self.load_count += 1

        # return synthetic data (use zeros for speed - random generation is too slow)
        size = end - start
        data = np.zeros((size, self.num_neurons), dtype=np.float32)
        stim = np.zeros((size, self.num_stim_dims), dtype=np.float32)

        return data, stim


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------


class TestChunkLoader(unittest.TestCase):
    """test suite for RandomChunkLoader."""

    def test_basic_chunk_loading(self):
        """test basic chunk loading without gpu."""
        source = MockDataSource(
            total_timesteps=100000,
            num_neurons=1000,
            num_stim_dims=100
        )

        loader = RandomChunkLoader(
            load_fn=source.load_slice,
            total_timesteps=source.total_timesteps,
            chunk_size=10000,
            device='cpu',
            prefetch=1,
            seed=42
        )

        # load 5 random chunks
        num_chunks = 5
        loader.start_epoch(num_chunks)

        loaded_chunks = []
        for _ in range(num_chunks):
            chunk_start, chunk_data, chunk_stim = loader.get_next_chunk()
            self.assertIsNotNone(chunk_start)
            self.assertIsNotNone(chunk_data)
            self.assertIsNotNone(chunk_stim)
            self.assertEqual(chunk_data.shape, (10000, 1000))
            self.assertEqual(chunk_stim.shape, (10000, 100))
            self.assertEqual(chunk_data.device.type, 'cpu')
            loaded_chunks.append((chunk_start, chunk_data, chunk_stim))

        # verify end of epoch
        end_start, end_data, end_stim = loader.get_next_chunk()
        self.assertIsNone(end_start)
        self.assertIsNone(end_data)
        self.assertIsNone(end_stim)

        # verify 5 chunks were loaded
        self.assertEqual(source.load_count, 5)

        loader.cleanup()

    def test_random_chunk_indices(self):
        """test that chunks are loaded from random locations."""
        source = MockDataSource(
            total_timesteps=100000,
            num_neurons=100,
            num_stim_dims=50
        )

        loader = RandomChunkLoader(
            load_fn=source.load_slice,
            total_timesteps=source.total_timesteps,
            chunk_size=5000,
            device='cpu',
            seed=123
        )

        num_chunks = 10
        loader.start_epoch(num_chunks)

        # collect all loaded chunks
        for _ in range(num_chunks):
            loader.get_next_chunk()

        # verify chunks are from different locations
        start_indices = [start for start, end in source.load_calls]
        self.assertGreater(len(set(start_indices)), 1, "all chunks from same location (not random)")

        # verify all chunks are valid
        for start, end in source.load_calls:
            self.assertGreaterEqual(start, 0)
            self.assertLessEqual(start, source.total_timesteps - 5000)
            self.assertEqual(end, start + 5000)

        loader.cleanup()

    def test_overlap_with_mock_training(self):
        """test that loading overlaps with mock training (using sleeps)."""
        # simulate disk i/o: 100ms per chunk
        # simulate training: 100ms per chunk
        load_time_ms = 100
        train_time_ms = 100

        source = MockDataSource(
            total_timesteps=50000,
            num_neurons=1000,
            num_stim_dims=100,
            load_delay_ms=load_time_ms
        )

        loader = RandomChunkLoader(
            load_fn=source.load_slice,
            total_timesteps=source.total_timesteps,
            chunk_size=5000,
            device='cpu',
            prefetch=10  # large buffer to avoid blocking background thread
        )

        num_chunks = 20
        loader.start_epoch(num_chunks)

        total_start = time.time()

        for i in range(num_chunks):
            # get chunk (should overlap with previous training)
            get_start = time.time()
            _, chunk_data, chunk_stim = loader.get_next_chunk()
            get_time = time.time() - get_start

            # simulate deterministic "training" on this chunk
            train_start = time.time()
            time.sleep(train_time_ms / 1000.0)
            train_time = time.time() - train_start

            print(f"chunk {i}: get={get_time*1000:.1f}ms, train={train_time*1000:.1f}ms")

        total_time = time.time() - total_start

        # expected times:
        # sequential: 20 * (100ms load + 100ms train) = 4000ms
        # with overlap: max(20*100ms, 20*100ms) + startup = 2000ms + overhead
        sequential_time = num_chunks * (load_time_ms + train_time_ms) / 1000.0
        expected_overlap_time = max(num_chunks * load_time_ms, num_chunks * train_time_ms) / 1000.0

        print(f"\ntotal time: {total_time*1000:.1f}ms")
        print(f"sequential would be: {sequential_time*1000:.1f}ms")
        print(f"expected with overlap: ~{expected_overlap_time*1000:.1f}ms")
        print(f"speedup: {sequential_time / total_time:.2f}x")

        # verify overlap happened (should be much faster than sequential)
        # with load=100ms and train=100ms, overlap should give ~2x speedup
        # with more chunks, cold start overhead is amortized
        self.assertLess(total_time, sequential_time * 0.65, "no overlap detected (too slow)")

        loader.cleanup()

    def test_prefetch_buffer(self):
        """test that prefetch buffer works correctly."""
        source = MockDataSource(
            total_timesteps=30000,
            num_neurons=500,
            num_stim_dims=50,
            load_delay_ms=50
        )

        # with prefetch=2, should load 2 chunks ahead
        loader = RandomChunkLoader(
            load_fn=source.load_slice,
            total_timesteps=source.total_timesteps,
            chunk_size=3000,
            device='cpu',
            prefetch=2
        )

        loader.start_epoch(num_chunks=4)

        # give background thread time to prefetch
        time.sleep(0.15)  # enough time to load 2-3 chunks

        # now get chunks - first 2 should be instant (already in queue)
        for i in range(4):
            start = time.time()
            _, chunk_data, chunk_stim = loader.get_next_chunk()
            get_time = time.time() - start

            print(f"chunk {i}: get_time={get_time*1000:.1f}ms")

            # first chunk should be very fast (already prefetched)
            if i == 0:
                self.assertLess(get_time, 0.02, f"first chunk should be prefetched, took {get_time*1000:.1f}ms")

        loader.cleanup()

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cuda_transfer(self):
        """test cuda stream transfers (if gpu available)."""
        source = MockDataSource(
            total_timesteps=20000,
            num_neurons=2000,
            num_stim_dims=200
        )

        loader = RandomChunkLoader(
            load_fn=source.load_slice,
            total_timesteps=source.total_timesteps,
            chunk_size=5000,
            device='cuda',
            prefetch=1
        )

        loader.start_epoch(num_chunks=3)

        for _ in range(3):
            _, chunk_data, chunk_stim = loader.get_next_chunk()
            self.assertIsNotNone(chunk_data)
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

        loader = RandomChunkLoader(
            load_fn=source.load_slice,
            total_timesteps=source.total_timesteps,
            chunk_size=3000,
            device='cpu'
        )

        loader.start_epoch(num_chunks=5)

        for _ in range(5):
            loader.get_next_chunk()

        stats = loader.get_stats()
        self.assertEqual(stats['chunks_loaded'], 5)
        self.assertEqual(stats['chunks_transferred'], 5)

        loader.cleanup()

    def test_chunk_size_larger_than_dataset(self):
        """test that chunk_size > dataset works (returns full dataset)."""
        source = MockDataSource(
            total_timesteps=10000,
            num_neurons=100,
            num_stim_dims=50
        )

        # chunk_size larger than dataset should work (returns full dataset)
        loader = RandomChunkLoader(
            load_fn=source.load_slice,
            total_timesteps=source.total_timesteps,
            chunk_size=20000,  # larger than dataset
            device='cpu'
        )

        loader.start_epoch(num_chunks=2)

        # should get 2 chunks, each containing the full dataset
        for _ in range(2):
            _, chunk_data, chunk_stim = loader.get_next_chunk()
            self.assertIsNotNone(chunk_data)
            self.assertEqual(chunk_data.shape, (10000, 100))  # full dataset

        loader.cleanup()

    def test_cleanup_stops_loading(self):
        """test that cleanup stops background thread."""
        source = MockDataSource(
            total_timesteps=50000,
            num_neurons=100,
            num_stim_dims=50,
            load_delay_ms=100  # slow loading
        )

        loader = RandomChunkLoader(
            load_fn=source.load_slice,
            total_timesteps=source.total_timesteps,
            chunk_size=5000,
            device='cpu'
        )

        # start loading many chunks
        loader.start_epoch(num_chunks=20)

        # get only 2 chunks
        loader.get_next_chunk()
        loader.get_next_chunk()

        # cleanup should stop background thread
        loader.cleanup()

        # should not have loaded all 20 chunks
        self.assertLess(source.load_count, 20, "cleanup did not stop background loading")

    def test_deterministic_with_seed(self):
        """test that same seed produces same chunk sequence."""
        source = MockDataSource(
            total_timesteps=30000,
            num_neurons=100,
            num_stim_dims=50
        )

        # first run with seed=42
        loader1 = RandomChunkLoader(
            load_fn=source.load_slice,
            total_timesteps=source.total_timesteps,
            chunk_size=5000,
            device='cpu',
            seed=42
        )
        loader1.start_epoch(num_chunks=3)
        for _ in range(3):
            loader1.get_next_chunk()
        calls1 = source.load_calls.copy()
        loader1.cleanup()

        # reset source
        source.load_calls.clear()

        # second run with seed=42
        loader2 = RandomChunkLoader(
            load_fn=source.load_slice,
            total_timesteps=source.total_timesteps,
            chunk_size=5000,
            device='cpu',
            seed=42
        )
        loader2.start_epoch(num_chunks=3)
        for _ in range(3):
            loader2.get_next_chunk()
        calls2 = source.load_calls.copy()
        loader2.cleanup()

        # should load same chunks in same order
        self.assertEqual(calls1, calls2, "same seed should produce same chunk sequence")


    def test_multiple_epochs_back_to_back(self):
        """test starting new epoch immediately after previous completes."""
        source = MockDataSource(
            total_timesteps=20000,
            num_neurons=500,
            num_stim_dims=50,
            load_delay_ms=50
        )

        loader = RandomChunkLoader(
            load_fn=source.load_slice,
            total_timesteps=source.total_timesteps,
            chunk_size=5000,
            device='cpu',
            prefetch=2
        )

        # epoch 1
        loader.start_epoch(num_chunks=3)
        for _ in range(3):
            _, chunk_data, chunk_stim = loader.get_next_chunk()
            self.assertIsNotNone(chunk_data)

        # epoch 2 immediately after (thread should be cleaned up)
        loader.start_epoch(num_chunks=3)
        for _ in range(3):
            _, chunk_data, chunk_stim = loader.get_next_chunk()
            self.assertIsNotNone(chunk_data)

        loader.cleanup()

    def test_early_break_from_epoch(self):
        """test breaking out of epoch early (simulates warmup duration mismatch)."""
        source = MockDataSource(
            total_timesteps=30000,
            num_neurons=500,
            num_stim_dims=50,
            load_delay_ms=100
        )

        loader = RandomChunkLoader(
            load_fn=source.load_slice,
            total_timesteps=source.total_timesteps,
            chunk_size=5000,
            device='cpu',
            prefetch=2
        )

        # epoch 1: ask for 5 chunks but only consume 2
        loader.start_epoch(num_chunks=5)
        for _ in range(2):
            _, chunk_data, chunk_stim = loader.get_next_chunk()
            self.assertIsNotNone(chunk_data)
        # break early (don't consume all 5 chunks)

        # wait a bit for background thread to finish
        time.sleep(0.6)  # 5 chunks * 100ms = 500ms

        # epoch 2 should work (thread should be done)
        loader.start_epoch(num_chunks=3)
        for _ in range(3):
            _, chunk_data, chunk_stim = loader.get_next_chunk()
            self.assertIsNotNone(chunk_data)

        loader.cleanup()

    def test_chunk_alignment_with_time_units(self):
        """test that chunk starts are aligned to time_units multiples."""
        source = MockDataSource(
            total_timesteps=100000,
            num_neurons=500,
            num_stim_dims=50
        )

        time_units = 10
        loader = RandomChunkLoader(
            load_fn=source.load_slice,
            total_timesteps=source.total_timesteps,
            chunk_size=5000,
            device='cpu',
            time_units=time_units,
            seed=123
        )

        num_chunks = 20
        loader.start_epoch(num_chunks)

        # collect all chunk starts
        chunk_starts = []
        for _ in range(num_chunks):
            chunk_start, chunk_data, chunk_stim = loader.get_next_chunk()
            self.assertIsNotNone(chunk_start)
            chunk_starts.append(chunk_start)

        # verify all chunk starts are multiples of time_units
        for start in chunk_starts:
            self.assertEqual(start % time_units, 0,
                f"chunk start {start} is not aligned to time_units={time_units}")

        # verify chunks are from different aligned locations (randomness preserved)
        self.assertGreater(len(set(chunk_starts)), 1,
            "all chunks from same location (not random)")

        loader.cleanup()

    def test_acquisition_mode_batch_sampling_bounds(self):
        """test that batch sampling with acquisition modes stays within chunk bounds."""
        from LatentEvolution.acquisition import (
            compute_neuron_phases,
            sample_batch_indices,
            TimeAlignedMode,
            StaggeredRandomMode,
        )

        source = MockDataSource(
            total_timesteps=50000,
            num_neurons=1000,
            num_stim_dims=100
        )

        time_units = 5
        chunk_size = 10000
        total_steps = time_units * 2  # evolve_multiple_steps = 2

        loader = RandomChunkLoader(
            load_fn=source.load_slice,
            total_timesteps=source.total_timesteps,
            chunk_size=chunk_size,
            device='cpu',
            time_units=time_units,
            seed=42
        )

        loader.start_epoch(num_chunks=5)

        for _ in range(5):
            chunk_start, chunk_data, chunk_stim = loader.get_next_chunk()
            if chunk_data is None:
                break

            # test time_aligned mode
            neuron_phases_aligned = compute_neuron_phases(
                num_neurons=source.num_neurons,
                time_units=time_units,
                acquisition_mode=TimeAlignedMode(),
                device='cpu'
            )

            obs_indices_aligned = sample_batch_indices(
                chunk_size=chunk_size,
                total_steps=total_steps,
                time_units=time_units,
                batch_size=32,
                num_neurons=source.num_neurons,
                neuron_phases=neuron_phases_aligned,
                device='cpu'
            )

            # all observation indices should be within chunk bounds (batch_size, num_neurons)
            self.assertTrue(torch.all(obs_indices_aligned >= 0),
                "time_aligned: observation indices below 0")
            self.assertTrue(torch.all(obs_indices_aligned < chunk_size),
                f"time_aligned: observation indices >= chunk_size ({chunk_size})")

            # for time_aligned, all neurons in a batch should have same observation time
            # so all columns should be identical
            first_col = obs_indices_aligned[:, 0].numpy()
            for neuron_idx in range(source.num_neurons):
                col = obs_indices_aligned[:, neuron_idx].numpy()
                self.assertTrue(np.array_equal(first_col, col),
                    "time_aligned: not all neurons observed at same time")

            # every observation index must be divisible by time_units
            self.assertTrue(np.all(first_col % time_units == 0),
                f"time_aligned: not all observation indices divisible by time_units={time_units}")

            # check spacing between batch samples
            sorted_first_col = np.sort(first_col)
            if len(sorted_first_col) > 1:
                diffs = np.diff(sorted_first_col)
                # all should be multiples of time_units
                self.assertTrue(np.all(diffs % time_units == 0),
                    f"time_aligned: batch spacing not multiples of time_units={time_units}")

            # test staggered_random mode
            neuron_phases_staggered = compute_neuron_phases(
                num_neurons=source.num_neurons,
                time_units=time_units,
                acquisition_mode=StaggeredRandomMode(seed=123),
                device='cpu'
            )

            obs_indices_staggered = sample_batch_indices(
                chunk_size=chunk_size,
                total_steps=total_steps,
                time_units=time_units,
                batch_size=32,
                num_neurons=source.num_neurons,
                neuron_phases=neuron_phases_staggered,
                device='cpu'
            )

            # all observation indices should be within chunk bounds
            self.assertTrue(torch.all(obs_indices_staggered >= 0),
                "staggered_random: observation indices below 0")
            self.assertTrue(torch.all(obs_indices_staggered < chunk_size),
                f"staggered_random: observation indices >= chunk_size ({chunk_size})")

            # for staggered_random, within each batch row, indices should differ by phases
            # check that each batch starts at a multiple of time_units
            batch_starts = obs_indices_staggered[:, 0] - neuron_phases_staggered[0]
            self.assertTrue(torch.all(batch_starts % time_units == 0),
                "staggered_random: batch starts not aligned to time_units")

            # every observation index minus its neuron's phase should be divisible by time_units
            # this verifies observations are at: phase_n + k*tu for each neuron n
            obs_minus_phases = obs_indices_staggered - neuron_phases_staggered.unsqueeze(0)  # (batch, neurons)
            self.assertTrue(torch.all(obs_minus_phases % time_units == 0),
                f"staggered_random: (obs_index - phase) not divisible by time_units={time_units}")

        loader.cleanup()

    def test_staggered_complete_coverage(self):
        """test that staggered_random gives complete neuron coverage within time_units window."""
        from LatentEvolution.acquisition import (
            compute_neuron_phases,
            StaggeredRandomMode,
        )

        num_neurons = 100
        time_units = 10
        device = 'cpu'

        # generate staggered phases
        phases = compute_neuron_phases(
            num_neurons=num_neurons,
            time_units=time_units,
            acquisition_mode=StaggeredRandomMode(seed=42),
            device=device
        )

        # verify phases are in valid range [0, time_units-1]
        self.assertTrue(torch.all(phases >= 0))
        self.assertTrue(torch.all(phases < time_units))

        # check that phases cover a good distribution (not all neurons at the same phase)
        unique_phases = torch.unique(phases)
        self.assertGreater(len(unique_phases), 1,
            "all neurons have same phase (no staggering)")

    def test_acquisition_with_chunk_boundaries(self):
        """test that acquisition modes work correctly near chunk boundaries."""
        from LatentEvolution.acquisition import (
            compute_neuron_phases,
            sample_batch_indices,
            StaggeredRandomMode,
        )

        source = MockDataSource(
            total_timesteps=30000,
            num_neurons=200,
            num_stim_dims=50
        )

        time_units = 7
        chunk_size = 5000
        total_steps = time_units * 3

        loader = RandomChunkLoader(
            load_fn=source.load_slice,
            total_timesteps=source.total_timesteps,
            chunk_size=chunk_size,
            device='cpu',
            time_units=time_units,
            seed=99
        )

        loader.start_epoch(num_chunks=3)

        neuron_phases = compute_neuron_phases(
            num_neurons=source.num_neurons,
            time_units=time_units,
            acquisition_mode=StaggeredRandomMode(seed=55),
            device='cpu'
        )

        for _ in range(3):
            chunk_start, chunk_data, chunk_stim = loader.get_next_chunk()
            if chunk_data is None:
                break

            # verify chunk_start is aligned
            self.assertEqual(chunk_start % time_units, 0,
                f"chunk_start {chunk_start} not aligned to time_units={time_units}")

            # sample many batches to stress test boundaries
            for _ in range(10):
                obs_indices = sample_batch_indices(
                    chunk_size=chunk_size,
                    total_steps=total_steps,
                    time_units=time_units,
                    batch_size=16,
                    num_neurons=source.num_neurons,
                    neuron_phases=neuron_phases,
                    device='cpu'
                )

                # verify no out-of-bounds access
                self.assertTrue(torch.all(obs_indices >= 0),
                    "observation index below 0")
                self.assertTrue(torch.all(obs_indices < chunk_size),
                    f"observation index >= chunk_size ({chunk_size})")

                # verify we can actually index the chunk data (would fail if out of bounds)
                try:
                    _ = chunk_data[obs_indices]
                except IndexError as e:
                    self.fail(f"indexing chunk_data with obs_indices failed: {e}")

        loader.cleanup()


if __name__ == "__main__":
    unittest.main()
