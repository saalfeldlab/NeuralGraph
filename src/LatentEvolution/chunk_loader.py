"""
chunked data loader with background prefetching for large datasets.

loads random overlapping windows from disk to cpu (pinned memory) in a background
thread, then transfers to gpu asynchronously. enables overlap of disk i/o, cpu→gpu
transfer, and model training.
"""

from queue import Queue
from threading import Thread
import random
from typing import Callable, Optional

import torch
import numpy as np


class RandomChunkLoader:
    """loads random overlapping chunks with background prefetching.

    architecture:
        background thread: disk → cpu (pinned memory) → queue
        main thread: queue → gpu (async transfer)

    this allows overlap of:
        - disk i/o (background thread)
        - cpu→gpu transfer (cuda stream)
        - model training (main thread)

    example:
        >>> loader = RandomChunkLoader(
        ...     load_fn=lambda start, end: (data[start:end], stim[start:end]),
        ...     total_timesteps=1000000,
        ...     chunk_size=65536,
        ...     device='cuda'
        ... )
        >>> loader.start_epoch(num_chunks=15)
        >>> for _ in range(15):
        ...     chunk_data, chunk_stim = loader.get_next_chunk()
        ...     # train on chunk...
        >>> loader.cleanup()
    """

    def __init__(
        self,
        load_fn: Callable[[int, int], tuple[np.ndarray, np.ndarray]],
        total_timesteps: int,
        chunk_size: int,
        device: torch.device | str = 'cuda',
        prefetch: int = 1,
        seed: Optional[int] = None,
    ):
        """initialize chunk loader.

        args:
            load_fn: function(start_idx, end_idx) -> (data, stim) as numpy arrays
            total_timesteps: total number of timesteps in dataset
            chunk_size: size of each chunk to load
            device: pytorch device to transfer chunks to
            prefetch: number of chunks to buffer in queue (1 or 2 recommended)
            seed: random seed for chunk sampling (optional)
        """
        self.load_fn = load_fn
        self.total_timesteps = total_timesteps
        self.chunk_size = chunk_size
        self.device = torch.device(device) if isinstance(device, str) else device
        self.prefetch = prefetch

        # random number generator for chunk sampling
        self.rng = random.Random(seed)

        # maximum valid start index (0 if dataset < chunk_size)
        self.max_start_idx = max(0, total_timesteps - chunk_size)

        # queue holds (chunk_data, chunk_stim) on cpu pinned memory
        self.cpu_queue: Queue = Queue(maxsize=prefetch)

        # cuda stream for async gpu transfer
        self.transfer_stream: Optional[torch.cuda.Stream] = None
        if self.device.type == 'cuda':
            self.transfer_stream = torch.cuda.Stream()

        # background thread control
        self.loader_thread: Optional[Thread] = None
        self.stop_flag = False

        # statistics
        self.chunks_loaded = 0
        self.chunks_transferred = 0

    def _load_random_chunk_to_cpu(self) -> tuple[torch.Tensor, torch.Tensor]:
        """load a random window to cpu pinned memory.

        returns:
            (data_pinned, stim_pinned): tensors on cpu with pinned memory (if cuda available)
        """
        # pick random start index
        start_idx = self.rng.randint(0, self.max_start_idx)
        end_idx = min(start_idx + self.chunk_size, self.total_timesteps)

        # load from disk via user-provided function
        data_np, stim_np = self.load_fn(start_idx, end_idx)

        # convert to tensors
        data_cpu = torch.from_numpy(data_np)
        stim_cpu = torch.from_numpy(stim_np)

        # pin memory for fast gpu transfer (only if cuda available)
        if torch.cuda.is_available():
            data_cpu = data_cpu.pin_memory()
            stim_cpu = stim_cpu.pin_memory()

        self.chunks_loaded += 1

        return data_cpu, stim_cpu

    def _background_loader(self, num_chunks: int):
        """background thread: load random chunks to cpu queue.

        args:
            num_chunks: number of random chunks to load
        """
        for _ in range(num_chunks):
            if self.stop_flag:
                break

            # load to cpu
            cpu_data, cpu_stim = self._load_random_chunk_to_cpu()

            # put in queue (blocks if queue is full - provides backpressure)
            self.cpu_queue.put((cpu_data, cpu_stim))

        # signal completion
        self.cpu_queue.put(None)

    def start_epoch(self, num_chunks: int):
        """start background loading for an epoch.

        args:
            num_chunks: number of random chunks to load this epoch
        """
        # stop previous thread if still alive and clean up
        if self.loader_thread is not None:
            if self.loader_thread.is_alive():
                # signal thread to stop
                self.stop_flag = True
                # drain queue to unblock thread if it's blocked on put()
                while not self.cpu_queue.empty():
                    try:
                        self.cpu_queue.get_nowait()
                    except:
                        break
                # wait for thread to finish
                self.loader_thread.join(timeout=2.0)
                if self.loader_thread.is_alive():
                    print("warning: previous loader thread did not stop cleanly")
            self.loader_thread = None

        # clear any remaining items in queue (e.g., None sentinel from previous epoch)
        while not self.cpu_queue.empty():
            try:
                self.cpu_queue.get_nowait()
            except:
                break

        # reset state for new epoch
        self.stop_flag = False
        self.chunks_loaded = 0
        self.chunks_transferred = 0

        # start background loader thread
        self.loader_thread = Thread(
            target=self._background_loader,
            args=(num_chunks,),
            daemon=True
        )
        self.loader_thread.start()

    def get_next_chunk(self) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """get next random chunk on gpu (blocks until ready).

        returns:
            (chunk_data, chunk_stim): tensors on gpu, or (None, None) if epoch done
        """
        # get from cpu queue (blocks if empty - waits for background loader)
        item = self.cpu_queue.get()

        if item is None:
            # end of epoch
            return None, None

        cpu_data, cpu_stim = item

        # transfer to gpu
        if self.device.type == 'cuda' and self.transfer_stream is not None:
            # async transfer using cuda stream
            with torch.cuda.stream(self.transfer_stream):
                gpu_data = cpu_data.to(self.device, non_blocking=True)
                gpu_stim = cpu_stim.to(self.device, non_blocking=True)

            # synchronize to ensure transfer completes
            self.transfer_stream.synchronize()
        else:
            # cpu or mps device - blocking transfer
            gpu_data = cpu_data.to(self.device)
            gpu_stim = cpu_stim.to(self.device)

        self.chunks_transferred += 1

        return gpu_data, gpu_stim

    def cleanup(self):
        """stop background loading and cleanup resources."""
        self.stop_flag = True

        # drain queue to unblock background thread
        while not self.cpu_queue.empty():
            try:
                self.cpu_queue.get_nowait()
            except:
                break

        # wait for thread to finish
        if self.loader_thread is not None:
            self.loader_thread.join(timeout=5.0)
            if self.loader_thread.is_alive():
                print("warning: background loader thread did not terminate cleanly")

        self.loader_thread = None

    def get_stats(self) -> dict[str, int]:
        """get loader statistics.

        returns:
            dict with 'chunks_loaded' and 'chunks_transferred'
        """
        return {
            'chunks_loaded': self.chunks_loaded,
            'chunks_transferred': self.chunks_transferred,
        }
