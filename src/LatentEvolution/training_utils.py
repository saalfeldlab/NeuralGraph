"""
shared training utilities for neural dynamics models.

includes loss accumulation, seeding, and device selection.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict
import random

import numpy as np
import torch


@dataclass
class LossAccumulator:
    """generic loss accumulator with enum-keyed dict storage."""
    loss_types: type[Enum]  # enum class defining loss component types
    components: Dict[Enum, float] = field(init=False)
    count: int = field(init=False, default=0)

    def __post_init__(self):
        self.components = {lt: 0.0 for lt in self.loss_types}

    def accumulate(self, loss_dict: Dict[Enum, torch.Tensor]) -> None:
        """accumulate losses from dict mapping enum -> tensor."""
        for loss_type, value in loss_dict.items():
            self.components[loss_type] += value.detach().item()
        self.count += 1

    def mean(self) -> Dict[Enum, float]:
        """return mean of each component."""
        if self.count == 0:
            return {k: 0.0 for k in self.components}
        return {k: v / self.count for k, v in self.components.items()}

    def __getitem__(self, loss_type: Enum) -> float:
        """allow bracket access: losses[LossType.TOTAL]"""
        return self.components[loss_type]


def seed_everything(seed: int):
    """set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """cross-platform device selection."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("using apple mps backend for training.")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print(f"using cuda device: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("using cpu for training.")
        return torch.device("cpu")
