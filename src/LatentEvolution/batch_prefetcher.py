"""
batch prefetcher: overlap CPU sampling with GPU compute using pinned buffer pool.

architecture:
    sampler thread:  free_queue -> sample batch -> GPU transfer -> filled_queue
    main thread:     filled_queue -> train -> return buffer to free_queue

    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  free_queue        sampler thread         filled_queue         │
    │  ┌──────────┐     ┌──────────────┐       ┌──────────────┐      │
    │  │ pinned_0 │────>│ sample B     │──────>│ gpu_batch_0  │      │
    │  │ pinned_1 │     │ samples into │       │ gpu_batch_1  │      │
    │  │ pinned_2 │     │ pinned buf   │       │ gpu_batch_2  │      │
    │  └────▲─────┘     │ then .to(gpu)│       └──────┬───────┘      │
    │       │           └──────────────┘              │              │
    │       │                                         ▼              │
    │       │                                  ┌──────────────┐      │
    │       └──────────────────────────────────│  main thread │      │
    │         (return pinned buf after use)    │    train     │      │
    │                                          └──────────────┘      │
    └─────────────────────────────────────────────────────────────────┘

usage:
    prefetcher = BatchPrefetcher(
        sample_fn=lambda buf: sample_into_buffer(data, buf, rng),
        transfer_fn=lambda buf: buf.to(device, non_blocking=True),
        batch_shape=(32, 100, 71721),
        num_buffers=3,
    )
    prefetcher.start(num_batches=1000)

    for _ in range(1000):
        batch = prefetcher.get()  # blocks until ready
        loss = train_step(model, batch)
        loss.backward()
        optimizer.step()

    prefetcher.stop()
"""

from queue import Queue
from threading import Thread
from typing import Callable

import torch


class BatchPrefetcher:
    """prefetches batches using background thread and pinned buffer pool."""

    def __init__(
        self,
        sample_fn: Callable[[torch.Tensor], None],
        transfer_fn: Callable[[torch.Tensor], torch.Tensor],
        batch_shape: tuple[int, int, int],
        num_buffers: int = 3,
    ):
        """initialize batch prefetcher.

        args:
            sample_fn: function(pinned_buffer) -> None.
                fills pinned_buffer in-place with sampled data.
            transfer_fn: function(pinned_buffer) -> gpu_tensor.
                transfers pinned buffer to GPU, returns GPU tensor.
            batch_shape: (batch_size, seq_len, num_features) shape for buffers.
            num_buffers: number of pinned buffers in pool. more buffers = more
                slack for variance in sampling time, but more CPU memory.
        """
        self.sample_fn = sample_fn
        self.transfer_fn = transfer_fn
        self.batch_shape = batch_shape
        self.num_buffers = num_buffers

        # allocate pinned buffer pool
        self.buffers = [
            torch.empty(batch_shape, dtype=torch.float32, pin_memory=True)
            for _ in range(num_buffers)
        ]

        # queues for producer-consumer pattern
        self.free_queue: Queue[torch.Tensor] = Queue()
        self.filled_queue: Queue[tuple[torch.Tensor, torch.Tensor] | None] = Queue()

        # thread control
        self.worker_thread: Thread | None = None
        self.stop_flag = False
        self.num_batches = 0

    def start(self, num_batches: int):
        """start prefetching batches.

        args:
            num_batches: total number of batches to prefetch.
        """
        self.stop_flag = False
        self.num_batches = num_batches

        # clear queues
        self._drain_queue(self.free_queue)
        self._drain_queue(self.filled_queue)

        # put all buffers in free queue
        for buf in self.buffers:
            self.free_queue.put(buf)

        # start worker thread
        self.worker_thread = Thread(
            target=self._worker,
            daemon=True,
            name="batch_prefetcher",
        )
        self.worker_thread.start()

    def _worker(self):
        """background worker: sample batches and transfer to GPU."""
        for _ in range(self.num_batches):
            if self.stop_flag:
                break

            # get a free buffer (blocks if none available)
            with torch.profiler.record_function("prefetch_wait_free"):
                pinned_buffer = self.free_queue.get()

            if self.stop_flag:
                break

            # sample into pinned buffer
            with torch.profiler.record_function("prefetch_sample"):
                self.sample_fn(pinned_buffer)

            # transfer to GPU
            with torch.profiler.record_function("prefetch_transfer"):
                gpu_batch = self.transfer_fn(pinned_buffer)

            # put in filled queue for main thread
            self.filled_queue.put((gpu_batch, pinned_buffer))

        # signal completion
        self.filled_queue.put(None)

    def get(self) -> torch.Tensor | None:
        """get next prefetched batch.

        returns:
            GPU tensor of shape batch_shape, or None if done.
        """
        with torch.profiler.record_function("prefetch_get"):
            item = self.filled_queue.get()

        if item is None:
            return None

        gpu_batch, pinned_buffer = item

        # return pinned buffer to pool for reuse
        self.free_queue.put(pinned_buffer)

        return gpu_batch

    def _drain_queue(self, queue: Queue):
        """drain all items from a queue."""
        while not queue.empty():
            try:
                queue.get_nowait()
            except Exception:
                break

    def stop(self):
        """stop prefetching and cleanup."""
        self.stop_flag = True

        # drain queues to unblock worker
        self._drain_queue(self.free_queue)
        self._drain_queue(self.filled_queue)

        # put a buffer to unblock worker if waiting on free_queue
        if self.buffers:
            try:
                self.free_queue.put_nowait(self.buffers[0])
            except Exception:
                pass

        # wait for worker
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=2.0)
            self.worker_thread = None
