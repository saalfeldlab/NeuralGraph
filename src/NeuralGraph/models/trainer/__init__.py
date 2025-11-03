"""Trainer module - extracted from graph_trainer.py"""

from .data_train import data_train
from .data_train_signal import data_train_signal
from .data_train_flyvis import data_train_flyvis
from .data_train_flyvis_calcium import data_train_flyvis_calcium
from .data_train_zebra import data_train_zebra
from .data_train_zebra_fluo import data_train_zebra_fluo
from .data_test import data_test
from .data_test_signal import data_test_signal
from .data_test_flyvis import data_test_flyvis
from .data_test_zebra import data_test_zebra

__all__ = [
    "data_train",
    "data_train_signal",
    "data_train_flyvis",
    "data_train_flyvis_calcium",
    "data_train_zebra",
    "data_train_zebra_fluo",
    "data_test",
    "data_test_signal",
    "data_test_flyvis",
    "data_test_zebra"
]
