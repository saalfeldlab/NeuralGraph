"""Module to load flyvis simulation data."""

from enum import IntEnum
from pathlib import Path

import numpy as np
import torch
import tensorstore as ts

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


# mapping from dynamic column indices to timeseries column position
_DYNAMIC_COL_TO_TS = {col.value: i for i, col in enumerate(DYNAMIC_COLUMNS)}


def load_column_slice(
    path: str | Path,
    column: int,
    time_start: int,
    time_end: int,
    neuron_limit: int | None = None,
) -> np.ndarray:
    """load a time series column slice directly from zarr format.

    this avoids loading the full (T, N, 9) array when you only need one column.

    args:
        path: base path to zarr data (without extension)
        column: dynamic column index (VOLTAGE=3, STIMULUS=4, CALCIUM=7, FLUORESCENCE=8)
        time_start: start time index
        time_end: end time index
        neuron_limit: optional limit on neurons (first N)

    returns:
        numpy array of shape (time_end - time_start, N) or (time_end - time_start, neuron_limit)

    raises:
        AssertionError: if column is a static column (use load_metadata instead)
    """
    assert column in _DYNAMIC_COL_TO_TS, (
        f"column {column} is static, use load_metadata() instead"
    )

    path = Path(path)
    base_path = path.with_suffix('') if path.suffix in ('.npy', '.zarr') else path

    ts_col = _DYNAMIC_COL_TO_TS[column]
    neuron_slice = slice(None, neuron_limit) if neuron_limit else slice(None)

    ts_path = base_path / 'timeseries.zarr'
    spec = {
        'driver': 'zarr',
        'kvstore': {'driver': 'file', 'path': str(ts_path)},
    }
    store = ts.open(spec).result()
    data = store[time_start:time_end, neuron_slice, ts_col].read().result()

    return np.ascontiguousarray(data)


def load_metadata(path: str | Path) -> np.ndarray:
    """load metadata from V2 zarr format.

    args:
        path: base path to zarr data

    returns:
        numpy array of shape (N, 5) with static columns
    """
    path = Path(path)
    base_path = path.with_suffix('') if path.suffix in ('.npy', '.zarr') else path
    meta_path = base_path / 'metadata.zarr'

    spec = {
        'driver': 'zarr',
        'kvstore': {'driver': 'file', 'path': str(meta_path)},
    }
    return ts.open(spec).result().read().result()
