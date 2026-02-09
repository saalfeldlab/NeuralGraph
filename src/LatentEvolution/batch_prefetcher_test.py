"""unit tests for batch prefetcher."""

import time
import unittest

import torch

from LatentEvolution.batch_prefetcher import BatchPrefetcher


class TestBatchPrefetcher(unittest.TestCase):
    """tests for BatchPrefetcher."""

    def test_basic_prefetch(self):
        """verify prefetcher returns correct number of batches."""
        batch_shape = (4, 10, 32)
        num_batches = 5
        counter = {"value": 0}

        def sample_fn(buf: torch.Tensor):
            # fill buffer with incrementing values
            buf.fill_(counter["value"])
            counter["value"] += 1

        def transfer_fn(buf: torch.Tensor):
            # simulate CPU-only transfer for testing
            return buf.clone()

        prefetcher = BatchPrefetcher(
            sample_fn=sample_fn,
            transfer_fn=transfer_fn,
            batch_shape=batch_shape,
            num_buffers=2,
        )
        prefetcher.start(num_batches)

        received = []
        while True:
            batch = prefetcher.get()
            if batch is None:
                break
            received.append(batch[0, 0, 0].item())

        prefetcher.stop()

        self.assertEqual(len(received), num_batches)
        self.assertEqual(received, [0, 1, 2, 3, 4])

    def test_buffer_reuse(self):
        """verify pinned buffers are reused."""
        batch_shape = (2, 5, 8)
        num_batches = 10
        num_buffers = 2
        buffer_ids = []

        def sample_fn(buf: torch.Tensor):
            buffer_ids.append(id(buf))

        def transfer_fn(buf: torch.Tensor):
            return buf.clone()

        prefetcher = BatchPrefetcher(
            sample_fn=sample_fn,
            transfer_fn=transfer_fn,
            batch_shape=batch_shape,
            num_buffers=num_buffers,
        )
        prefetcher.start(num_batches)

        for _ in range(num_batches):
            batch = prefetcher.get()
            self.assertIsNotNone(batch)

        # verify only num_buffers unique buffer ids
        unique_ids = set(buffer_ids)
        self.assertEqual(len(unique_ids), num_buffers)

        prefetcher.stop()

    def test_stop_early(self):
        """verify prefetcher can be stopped early."""
        batch_shape = (2, 5, 8)
        num_batches = 100

        def sample_fn(buf: torch.Tensor):
            time.sleep(0.01)  # simulate slow sampling

        def transfer_fn(buf: torch.Tensor):
            return buf.clone()

        prefetcher = BatchPrefetcher(
            sample_fn=sample_fn,
            transfer_fn=transfer_fn,
            batch_shape=batch_shape,
            num_buffers=2,
        )
        prefetcher.start(num_batches)

        # get only a few batches
        for _ in range(3):
            batch = prefetcher.get()
            self.assertIsNotNone(batch)

        # stop early
        prefetcher.stop()

        # should not hang
        self.assertIsNone(prefetcher.worker_thread)

    def test_overlap_timing(self):
        """verify sampling overlaps with consumption."""
        batch_shape = (2, 5, 8)
        num_batches = 5
        sample_time_ms = 50
        consume_time_ms = 30

        def sample_fn(buf: torch.Tensor):
            time.sleep(sample_time_ms / 1000)

        def transfer_fn(buf: torch.Tensor):
            return buf.clone()

        prefetcher = BatchPrefetcher(
            sample_fn=sample_fn,
            transfer_fn=transfer_fn,
            batch_shape=batch_shape,
            num_buffers=3,  # enough buffers for overlap
        )

        start = time.perf_counter()
        prefetcher.start(num_batches)

        for _ in range(num_batches):
            batch = prefetcher.get()
            self.assertIsNotNone(batch)
            time.sleep(consume_time_ms / 1000)  # simulate training

        prefetcher.stop()
        elapsed_ms = (time.perf_counter() - start) * 1000

        # without overlap: 5 * (50 + 30) = 400ms
        # with overlap: ~50 + 5 * 50 = 300ms (first batch + max(sample, consume))
        # allow some slack for thread scheduling
        self.assertLess(elapsed_ms, 380)

    def test_shape_correct(self):
        """verify output batches have correct shape."""
        batch_shape = (8, 20, 64)

        def sample_fn(buf: torch.Tensor):
            buf.normal_()

        def transfer_fn(buf: torch.Tensor):
            return buf.clone()

        prefetcher = BatchPrefetcher(
            sample_fn=sample_fn,
            transfer_fn=transfer_fn,
            batch_shape=batch_shape,
            num_buffers=2,
        )
        prefetcher.start(3)

        for _ in range(3):
            batch = prefetcher.get()
            self.assertEqual(batch.shape, batch_shape)

        prefetcher.stop()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_gpu_transfer(self):
        """verify batches are transferred to GPU."""
        batch_shape = (4, 10, 32)
        device = torch.device("cuda")

        def sample_fn(buf: torch.Tensor):
            buf.fill_(42.0)

        def transfer_fn(buf: torch.Tensor):
            return buf.to(device, non_blocking=True)

        prefetcher = BatchPrefetcher(
            sample_fn=sample_fn,
            transfer_fn=transfer_fn,
            batch_shape=batch_shape,
            num_buffers=2,
        )
        prefetcher.start(3)

        for _ in range(3):
            batch = prefetcher.get()
            self.assertEqual(batch.device.type, "cuda")
            self.assertEqual(batch[0, 0, 0].item(), 42.0)

        prefetcher.stop()


if __name__ == "__main__":
    unittest.main()
