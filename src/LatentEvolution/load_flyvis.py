"""Module to load flyvis simulation data."""

from enum import IntEnum
import numpy as np
from pydantic import BaseModel, field_validator, ConfigDict
import torch

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


# columns that don't change over time (stored once in metadata.zarr)
STATIC_COLUMNS = (
    FlyVisSim.INDEX,
    FlyVisSim.XPOS,
    FlyVisSim.YPOS,
    FlyVisSim.GROUP_TYPE,
    FlyVisSim.TYPE,
)

# columns that change each frame (stored in timeseries.zarr)
DYNAMIC_COLUMNS = (
    FlyVisSim.VOLTAGE,
    FlyVisSim.STIMULUS,
    FlyVisSim.CALCIUM,
    FlyVisSim.FLUORESCENCE,
)


class NeuronData:
    """Flyvis neuron info."""

    TYPE_NAMES = [
        "Am", "C2", "C3", "CT1(Lo1)", "CT1(M10)",
        "L1", "L2", "L3", "L4", "L5", "Lawf1", "Lawf2",
        "Mi1", "Mi10", "Mi11", "Mi12", "Mi13", "Mi14", "Mi15", "Mi2", "Mi3", "Mi4", "Mi9",
        "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8",
        "T1", "T2", "T2a", "T3", "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d",
        "Tm1", "Tm16", "Tm2", "Tm20", "Tm28", "Tm3", "Tm30", "Tm4", "Tm5Y",
        "Tm5a", "Tm5b", "Tm5c", "Tm9", "TmY10", "TmY13", "TmY14",
        "TmY15", "TmY18", "TmY3", "TmY4", "TmY5a", "TmY9"
    ]

    def __init__(self, x:np.ndarray[tuple[int, int, int], np.dtype[np.float32]]):
        self.ix=x[0, :, FlyVisSim.INDEX].astype(np.int32)
        self.pos=x[0, :, [FlyVisSim.XPOS, FlyVisSim.YPOS]]
        self.group_type=x[0, :, FlyVisSim.GROUP_TYPE].astype(np.uint8)
        self.type=x[0, :, FlyVisSim.TYPE].astype(np.uint8)
        self._compute_indices_per_type()

    def _compute_indices_per_type(self):
        """compute indices for each neuron type."""
        order = np.argsort(self.type)
        uniq_types, start_index = np.unique(self.type[order], return_index=True)
        num_neuron_types = len(uniq_types)
        assert (uniq_types == np.arange(num_neuron_types)).all(), "breaks assumptions"
        breaks = np.zeros(len(uniq_types)+1, dtype=np.int64)
        breaks[:-1] = start_index
        breaks[-1] = len(self.type)
        self.indices_per_type = [
            order[breaks[i]:breaks[i+1]] for i in range(num_neuron_types)
        ]

    @classmethod
    def from_metadata(cls, metadata: np.ndarray) -> "NeuronData":
        """create NeuronData from V2 metadata array.

        args:
            metadata: (N, 5) array with columns [INDEX, XPOS, YPOS, GROUP_TYPE, TYPE]

        returns:
            NeuronData instance
        """
        obj = cls.__new__(cls)
        obj.ix = metadata[:, 0].astype(np.int32)
        obj.pos = metadata[:, 1:3].copy()
        obj.group_type = metadata[:, 3].astype(np.uint8)
        obj.type = metadata[:, 4].astype(np.uint8)
        obj._compute_indices_per_type()
        return obj


class DataSplit(BaseModel):
    """Split the time series into train/validation sets."""

    train_start: int
    train_end: int
    validation_start: int
    validation_end: int

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @field_validator("*")
    @classmethod
    def check_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Indices in data_split must be non-negative.")
        return v

    @field_validator("train_end")
    @classmethod
    def check_order(cls, v, info):
        # very basic ordering sanity check
        d = info.data
        if "train_start" in d and v <= d["train_start"]:
            raise ValueError("train_end must be greater than train_start.")
        return v

def load_connectome_graph(data_path: str):
    """FlyVis connectome.

    Matrix convention
    wmat[i] lists the neurons j that influence i
    rows = post-synaptic, cols = pre-synaptic
    but flyvis paper has opposite conventions
    """

    edge_index = torch.load(f"{data_path}/edge_index.pt", map_location="cpu").numpy()[::-1].copy()
    weights =  torch.load(f"{data_path}/weights.pt", map_location="cpu").numpy()
    wmat = torch.sparse_coo_tensor(edge_index, weights).to_sparse_csr()
    return wmat
