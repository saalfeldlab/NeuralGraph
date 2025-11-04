"""
GPU statistics monitoring utilities using nvidia-smi.
"""

import subprocess
from typing import Optional


def is_nvidia_gpu_available() -> bool:
    """Check if nvidia-smi is available (indicating NVIDIA GPU presence)."""
    try:
        subprocess.check_output(
            ["nvidia-smi"],
            stderr=subprocess.DEVNULL,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_gpu_name() -> Optional[str]:
    """Get the name of the GPU from nvidia-smi."""
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return result.split('\n')[0]  # Get first GPU if multiple
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_gpu_utilization() -> float:
    """Get current GPU utilization percentage (0-100)."""
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return float(result.split('\n')[0])  # Get first GPU if multiple
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return 0.0


def get_gpu_memory_used_mb() -> float:
    """Get current GPU memory used in MB."""
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return float(result.split('\n')[0])  # Get first GPU if multiple
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return 0.0


def get_gpu_memory_total_mb() -> float:
    """Get total GPU memory available in MB."""
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return float(result.split('\n')[0])  # Get first GPU if multiple
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return 0.0


class GPUMonitor:
    """
    Context manager for monitoring GPU statistics during training.

    Tracks:
    - Average GPU utilization per epoch
    - Maximum GPU memory usage across all epochs
    - Total GPU memory available
    - GPU device name
    """

    def __init__(self):
        self.enabled = is_nvidia_gpu_available()
        self.gpu_name = get_gpu_name() if self.enabled else None
        self.total_memory_mb = get_gpu_memory_total_mb() if self.enabled else 0.0
        self.max_memory_mb = 0.0
        self.epoch_utilizations = []

    def sample_epoch_start(self):
        """Sample GPU stats at the start of an epoch."""
        if not self.enabled:
            return

        # Track memory usage
        current_memory = get_gpu_memory_used_mb()
        if current_memory > self.max_memory_mb:
            self.max_memory_mb = current_memory

    def sample_epoch_end(self):
        """Sample GPU stats at the end of an epoch."""
        if not self.enabled:
            return

        # Track utilization and memory
        utilization = get_gpu_utilization()
        self.epoch_utilizations.append(utilization)

        current_memory = get_gpu_memory_used_mb()
        if current_memory > self.max_memory_mb:
            self.max_memory_mb = current_memory

    def get_average_utilization(self) -> Optional[float]:
        """Get average GPU utilization across all epochs."""
        if not self.enabled or not self.epoch_utilizations:
            return None
        return sum(self.epoch_utilizations) / len(self.epoch_utilizations)

    def get_metrics(self) -> dict:
        """Get all collected GPU metrics."""
        if not self.enabled:
            return {
                "gpu_type": "N/A",
                "total_gpu_memory_mb": "N/A",
                "max_gpu_memory_mb": "N/A",
                "avg_gpu_utilization_percent": "N/A",
            }

        return {
            "gpu_type": self.gpu_name or "Unknown",
            "total_gpu_memory_mb": round(self.total_memory_mb, 2),
            "max_gpu_memory_mb": round(self.max_memory_mb, 2),
            "avg_gpu_utilization_percent": round(self.get_average_utilization() or 0.0, 2),
        }
