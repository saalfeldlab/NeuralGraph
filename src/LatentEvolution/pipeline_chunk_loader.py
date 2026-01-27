"""
pipeline chunk loader with 3-stage parallelism for large datasets.

architecture:
    cpu_loader thread:   disk -> cpu (pinned memory) -> cpu_queue
    gpu_transfer thread: cpu_queue -> gpu -> gpu_queue
    main thread:         gpu_queue -> training

this enables overlap of disk i/o, cpu->gpu transfer, and training.
"""

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
from pathlib import Path
from queue import Queue
from threading import Thread
import random
import threading
import time
from typing import Callable, Optional, Generator

import torch
import numpy as np


# -------------------------------------------------------------------
# Profiler
# -------------------------------------------------------------------


@dataclass
class TraceEvent:
    """a single trace event for chrome tracing."""
    name: str
    category: str
    start_us: int  # microseconds since profiler start
    duration_us: int
    tid: int  # thread id
    pid: int = 1  # process id (default 1)
    args: dict = field(default_factory=dict)


class PipelineProfiler:
    """records pipeline events for chrome tracing visualization.

    usage:
        profiler = PipelineProfiler()
        profiler.start()

        with profiler.event("disk_load", "io", thread="cpu_loader"):
            load_data()

        profiler.save("trace.json")
        # load in chrome://tracing or https://ui.perfetto.dev/
    """

    THREAD_IDS = {"main": 0, "cpu_loader": 1, "gpu_transfer": 2}

    def __init__(self):
        self.events: list[TraceEvent] = []
        self.start_time_ns: int = 0
        self.lock = threading.Lock()
        self._enabled = False

    def start(self):
        """start profiling."""
        self.start_time_ns = time.perf_counter_ns()
        self._enabled = True
        self.events.clear()

    def stop(self):
        """stop profiling."""
        self._enabled = False

    def is_enabled(self) -> bool:
        """check if profiling is enabled."""
        return self._enabled

    @contextmanager
    def event(
        self, name: str, category: str = "pipeline", thread: Optional[str] = None, **kwargs
    ) -> Generator[None, None, None]:
        """context manager to record an event."""
        if not self._enabled:
            yield
            return

        start_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            end_ns = time.perf_counter_ns()
            duration_ns = end_ns - start_ns

            if thread is not None:
                tid = self.THREAD_IDS.get(thread, hash(thread) % 100)
            else:
                tid = threading.current_thread().ident or 0

            trace_event = TraceEvent(
                name=name,
                category=category,
                start_us=(start_ns - self.start_time_ns) // 1000,
                duration_us=duration_ns // 1000,
                tid=tid,
                args=kwargs if kwargs else {},
            )

            with self.lock:
                self.events.append(trace_event)

    def to_chrome_trace(self) -> dict:
        """convert events to chrome tracing format."""
        trace_events = []

        # thread name metadata
        for name, tid in self.THREAD_IDS.items():
            trace_events.append({
                "name": "thread_name", "ph": "M", "pid": 1, "tid": tid,
                "args": {"name": name},
            })

        # process name metadata
        trace_events.append({
            "name": "process_name", "ph": "M", "pid": 1, "tid": 0,
            "args": {"name": "PipelineChunkLoader"},
        })

        # events
        for event in self.events:
            trace_event = {
                "name": event.name, "cat": event.category, "ph": "X",
                "ts": event.start_us, "dur": event.duration_us,
                "pid": event.pid, "tid": event.tid,
            }
            if event.args:
                trace_event["args"] = event.args
            trace_events.append(trace_event)

        return {"traceEvents": trace_events}

    def save(self, path: str | Path):
        """save trace to JSON file."""
        with open(Path(path), "w") as f:
            json.dump(self.to_chrome_trace(), f)

    def get_stats(self) -> dict[str, dict[str, float]]:
        """compute summary statistics per event name."""
        stats: dict[str, list[float]] = defaultdict(list)
        for event in self.events:
            stats[event.name].append(event.duration_us / 1000.0)

        result = {}
        for name, durations in stats.items():
            result[name] = {
                "count": len(durations),
                "mean_ms": sum(durations) / len(durations),
                "min_ms": min(durations),
                "max_ms": max(durations),
                "total_ms": sum(durations),
            }
        return result

    def print_stats(self):
        """print summary statistics."""
        stats = self.get_stats()
        print("\npipeline profiler stats:")
        print("-" * 60)
        for name, s in sorted(stats.items()):
            print(f"  {name:20s}: n={s['count']:4d}  "
                  f"mean={s['mean_ms']:7.2f}ms  "
                  f"min={s['min_ms']:7.2f}ms  "
                  f"max={s['max_ms']:7.2f}ms")
        print("-" * 60)


# -------------------------------------------------------------------
# Chunk Loader
# -------------------------------------------------------------------


class PipelineChunkLoader:
    """loads random chunks using a 3-stage pipeline.

    ┌─────────────────────────────────────────────────────────────────────┐
    │  cpu_loader thread     gpu_transfer thread        main thread       │
    │  ──────────────────    ───────────────────        ───────────       │
    │                                                                     │
    │  ┌─────────────┐       ┌─────────────────┐       ┌─────────────┐   │
    │  │    disk     │       │   cpu_queue     │       │  gpu_queue  │   │
    │  │   (mmap)    │       │ (pinned memory) │       │(gpu tensors)│   │
    │  └──────┬──────┘       └────────┬────────┘       └──────┬──────┘   │
    │         │                       │                       │          │
    │         ▼                       ▼                       ▼          │
    │  ┌─────────────┐       ┌─────────────────┐       ┌─────────────┐   │
    │  │ load slice  │       │  cuda stream    │       │   training  │   │
    │  │ numpy->torch│──────>│  cpu.to(gpu)    │──────>│    loop     │   │
    │  │ pin_memory  │       │  synchronize    │       │             │   │
    │  └─────────────┘       └─────────────────┘       └─────────────┘   │
    │                                                                     │
    │  queue sizes control memory and parallelism:                        │
    │    cpu_queue.maxsize = prefetch      (chunks in pinned memory)      │
    │    gpu_queue.maxsize = gpu_prefetch  (chunks on gpu)                │
    └─────────────────────────────────────────────────────────────────────┘

    example:
        >>> loader = PipelineChunkLoader(
        ...     load_fn=lambda start, end: (data[start:end], stim[start:end]),
        ...     total_timesteps=1000000,
        ...     chunk_size=65536,
        ...     device='cuda',
        ...     gpu_prefetch=2,  # double buffer for overlap
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
        time_units: int = 1,
        gpu_prefetch: int = 1,
        profiler: Optional[PipelineProfiler] = None,
    ):
        """initialize pipeline chunk loader.

        args:
            load_fn: function(start_idx, end_idx) -> (data, stim) as numpy arrays
            total_timesteps: total number of timesteps in dataset
            chunk_size: size of each chunk to load
            device: pytorch device to transfer chunks to
            prefetch: number of chunks to buffer in cpu_queue (pinned memory)
            seed: random seed for chunk sampling (optional)
            time_units: alignment constraint - chunk starts must be multiples of this
            gpu_prefetch: number of chunks to buffer in gpu_queue (on device). set to 2
                for double buffering to overlap cpu->gpu transfer with training.
            profiler: optional PipelineProfiler for chrome tracing
        """
        self.load_fn = load_fn
        self.total_timesteps = total_timesteps
        self.chunk_size = chunk_size
        self.device = torch.device(device) if isinstance(device, str) else device
        self.prefetch = prefetch
        self.time_units = time_units
        self.gpu_prefetch = gpu_prefetch
        self.profiler = profiler

        # random number generator for chunk sampling
        self.rng = random.Random(seed)

        # maximum valid start index (0 if dataset < chunk_size), aligned to time_units
        max_unaligned_start = max(0, total_timesteps - chunk_size)
        self.max_start_idx = (max_unaligned_start // time_units) * time_units

        # cpu_queue: cpu_loader thread -> gpu_transfer thread (pinned memory)
        self.cpu_queue: Queue = Queue(maxsize=prefetch)

        # gpu_queue: gpu_transfer thread -> main thread (device tensors)
        self.gpu_queue: Queue = Queue(maxsize=gpu_prefetch)

        # cuda stream for async gpu transfer (used by gpu_transfer thread)
        self.transfer_stream: Optional[torch.cuda.Stream] = None
        if self.device.type == 'cuda':
            self.transfer_stream = torch.cuda.Stream()

        # thread handles
        self.cpu_loader_thread: Optional[Thread] = None
        self.gpu_transfer_thread: Optional[Thread] = None
        self.stop_flag = False

        # statistics
        self.chunks_loaded = 0
        self.chunks_transferred = 0

    def _load_random_chunk_to_cpu(self) -> tuple[int, torch.Tensor, torch.Tensor]:
        """load a random window to cpu pinned memory.

        returns:
            (start_idx, data_pinned, stim_pinned)
        """
        # pick random start index aligned to time_units
        num_valid_starts = (self.max_start_idx // self.time_units) + 1
        aligned_idx = self.rng.randint(0, num_valid_starts - 1)
        start_idx = aligned_idx * self.time_units
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

        return start_idx, data_cpu, stim_cpu

    def _cpu_loader_worker(self, num_chunks: int):
        """cpu_loader thread: disk -> cpu_queue.

        args:
            num_chunks: number of random chunks to load
        """
        for chunk_idx in range(num_chunks):
            if self.stop_flag:
                break

            # profile disk load
            if self.profiler and self.profiler.is_enabled():
                with self.profiler.event("disk_load", "io", thread="cpu_loader", chunk=chunk_idx):
                    start_idx, cpu_data, cpu_stim = self._load_random_chunk_to_cpu()
            else:
                start_idx, cpu_data, cpu_stim = self._load_random_chunk_to_cpu()

            self.cpu_queue.put((start_idx, cpu_data, cpu_stim))

        # signal completion to gpu_transfer thread
        self.cpu_queue.put(None)

    def _gpu_transfer_worker(self):
        """gpu_transfer thread: cpu_queue -> gpu_queue."""
        chunk_idx = 0
        while True:
            if self.stop_flag:
                break

            # get from cpu_queue (blocks until ready)
            item = self.cpu_queue.get()

            if item is None:
                # propagate sentinel to main thread
                self.gpu_queue.put(None)
                break

            start_idx, cpu_data, cpu_stim = item

            # profile gpu transfer
            if self.profiler and self.profiler.is_enabled():
                with self.profiler.event("gpu_transfer", "transfer", thread="gpu_transfer", chunk=chunk_idx):
                    gpu_data, gpu_stim = self._do_transfer(cpu_data, cpu_stim)
            else:
                gpu_data, gpu_stim = self._do_transfer(cpu_data, cpu_stim)

            self.chunks_transferred += 1
            self.gpu_queue.put((start_idx, gpu_data, gpu_stim))
            chunk_idx += 1

    def _do_transfer(self, cpu_data: torch.Tensor, cpu_stim: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """transfer tensors from cpu to device."""
        if self.device.type == 'cuda' and self.transfer_stream is not None:
            with torch.cuda.stream(self.transfer_stream):
                gpu_data = cpu_data.to(self.device, non_blocking=True)
                gpu_stim = cpu_stim.to(self.device, non_blocking=True)
            self.transfer_stream.synchronize()
        else:
            # cpu or mps - blocking transfer
            gpu_data = cpu_data.to(self.device)
            gpu_stim = cpu_stim.to(self.device)
        return gpu_data, gpu_stim

    def start_epoch(self, num_chunks: int):
        """start pipeline for an epoch.

        args:
            num_chunks: number of random chunks to load this epoch
        """
        # cleanup any previous epoch
        self._stop_threads()

        # clear queues
        self._drain_queue(self.cpu_queue)
        self._drain_queue(self.gpu_queue)

        # reset state
        self.stop_flag = False
        self.chunks_loaded = 0
        self.chunks_transferred = 0

        # start cpu_loader thread
        self.cpu_loader_thread = Thread(
            target=self._cpu_loader_worker,
            args=(num_chunks,),
            daemon=True,
            name='cpu_loader',
        )
        self.cpu_loader_thread.start()

        # start gpu_transfer thread
        self.gpu_transfer_thread = Thread(
            target=self._gpu_transfer_worker,
            daemon=True,
            name='gpu_transfer',
        )
        self.gpu_transfer_thread.start()

    def get_next_chunk(self) -> tuple[Optional[int], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """get next chunk from gpu_queue (blocks until ready).

        returns:
            (chunk_start, chunk_data, chunk_stim) or (None, None, None) if epoch done
        """
        # profile queue wait time
        if self.profiler and self.profiler.is_enabled():
            with self.profiler.event("gpu_queue_wait", "sync", thread="main"):
                item = self.gpu_queue.get()
        else:
            item = self.gpu_queue.get()

        if item is None:
            return None, None, None

        return item

    def _drain_queue(self, queue: Queue):
        """drain all items from a queue."""
        while not queue.empty():
            try:
                queue.get_nowait()
            except:
                break

    def _stop_threads(self):
        """signal threads to stop and wait for them."""
        if self.cpu_loader_thread is None and self.gpu_transfer_thread is None:
            return

        self.stop_flag = True

        # drain queues to unblock threads
        self._drain_queue(self.cpu_queue)
        self._drain_queue(self.gpu_queue)

        # wait for threads
        if self.cpu_loader_thread is not None:
            self.cpu_loader_thread.join(timeout=2.0)
            if self.cpu_loader_thread.is_alive():
                print("warning: cpu_loader thread did not stop cleanly")
            self.cpu_loader_thread = None

        if self.gpu_transfer_thread is not None:
            # put sentinel to unblock gpu_transfer if it's waiting on cpu_queue
            try:
                self.cpu_queue.put_nowait(None)
            except:
                pass
            self.gpu_transfer_thread.join(timeout=2.0)
            if self.gpu_transfer_thread.is_alive():
                print("warning: gpu_transfer thread did not stop cleanly")
            self.gpu_transfer_thread = None

    def cleanup(self):
        """stop pipeline and cleanup resources."""
        self._stop_threads()

    def get_stats(self) -> dict[str, int]:
        """get loader statistics.

        returns:
            dict with 'chunks_loaded' and 'chunks_transferred'
        """
        return {
            'chunks_loaded': self.chunks_loaded,
            'chunks_transferred': self.chunks_transferred,
        }
