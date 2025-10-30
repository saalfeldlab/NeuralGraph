"""Module to load flyvis simulation data."""

from enum import IntEnum
from typing import NamedTuple
import numpy as np


class FlyVisSim(IntEnum):
    """Column interpretation in flyvis simulation outputs."""

    INDEX = 0
    XPOS = 1
    YPOS = 2
    VOLTAGE = 3
    STIMULUS = 4
    GROUP_TYPE = 5
    TYPE = 6
    CALCIUM = 7
    FLUORESCENCE = 8


class NeuronData(NamedTuple):
    """Flyvis neuron info."""

    ix: np.ndarray[tuple[int], np.dtype[np.int32]]
    pos: np.ndarray[tuple[int, int], np.dtype[np.float32]]
    group_type: np.ndarray[tuple[int], np.dtype[np.uint8]]
    type: np.ndarray[tuple[int], np.dtype[np.uint8]]


class SimulationResults(NamedTuple):
    neuron_data: NeuronData
    data: np.ndarray[tuple[int, int, int], np.dtype[np.float32]]

    @staticmethod
    def load(path: str):
        # this takes a while
        # T x N x 9 array
        x = np.load(path)

        assert (x[0, :, FlyVisSim.GROUP_TYPE] <= np.iinfo(np.uint8).max).all()
        assert (x[0, :, FlyVisSim.TYPE] <= np.iinfo(np.uint8).max).all()

        # split off time-independent piece
        return SimulationResults(
            neuron_data=NeuronData(
                ix=x[0, :, FlyVisSim.INDEX].astype(np.int32),
                pos=x[0, :, [FlyVisSim.XPOS, FlyVisSim.YPOS]],
                group_type=x[0, :, FlyVisSim.GROUP_TYPE].astype(np.uint8),
                type=x[0, :, FlyVisSim.TYPE].astype(np.uint8),
            ),
            data=x,
        )

    def __getitem__(
        self, col: FlyVisSim
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        """Access underlying simulation data"""
        return self.data[:, :, col]
