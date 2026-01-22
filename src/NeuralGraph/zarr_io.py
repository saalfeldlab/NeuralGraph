"""zarr/tensorstore I/O utilities for simulation data.

provides:
- ZarrSimulationWriter: incremental writer that appends frames during generation
- ZarrSimulationWriterV2: split metadata/timeseries writer for efficient storage
- detect_format: check if .npy or .zarr exists at path
- load_simulation_data: auto-detect format and load accordingly
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import tensorstore as ts


# flyvis format constants (for V2 writer compatibility)
# static columns: INDEX=0, XPOS=1, YPOS=2, GROUP_TYPE=5, TYPE=6
# dynamic columns: VOLTAGE=3, STIMULUS=4, CALCIUM=7, FLUORESCENCE=8
STATIC_COLS = [0, 1, 2, 5, 6]
DYNAMIC_COLS = [3, 4, 7, 8]
N_STATIC = len(STATIC_COLS)
N_DYNAMIC = len(DYNAMIC_COLS)


class ZarrSimulationWriter:
    """incremental writer - appends frames during generation (legacy V1 format).

    usage:
        writer = ZarrSimulationWriter(path, n_neurons=1000, n_features=8)
        for frame in simulation:
            writer.append(frame)
        writer.finalize()
    """

    def __init__(
        self,
        path: str | Path,
        n_neurons: int,
        n_features: int,
        chunks: tuple[int, int, int] | None = None,
        dtype: np.dtype = np.float32,
    ):
        """initialize zarr writer.

        args:
            path: output path (without extension, .zarr will be added)
            n_neurons: number of neurons (second dimension)
            n_features: number of features per neuron (third dimension)
            chunks: chunk sizes (time, neurons, features). use -1 for full dimension.
            dtype: data type for storage
        """
        self.path = Path(path)
        if not str(self.path).endswith('.zarr'):
            self.path = Path(str(self.path) + '.zarr')

        self.n_neurons = n_neurons
        self.n_features = n_features
        self.dtype = dtype

        # determine chunk sizes
        if chunks is None:
            # default: 500 frames, full neurons, full features
            chunks = (500, n_neurons, n_features)
        else:
            # replace -1 with actual dimensions
            chunks = (
                chunks[0],
                n_neurons if chunks[1] == -1 else chunks[1],
                n_features if chunks[2] == -1 else chunks[2],
            )
        self.chunks = chunks

        # buffer for accumulating frames before writing
        self._buffer: list[np.ndarray] = []
        self._total_frames = 0
        self._store: ts.TensorStore | None = None
        self._initialized = False

    def _initialize_store(self, first_frame: np.ndarray):
        """initialize tensorstore with zarr format."""
        # ensure parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # remove existing zarr directory if present
        if self.path.exists():
            import shutil
            shutil.rmtree(self.path)

        # create zarr store with tensorstore
        # start with initial capacity, will resize as needed
        initial_time_capacity = max(self.chunks[0] * 10, 1000)

        spec = {
            'driver': 'zarr',
            'kvstore': {
                'driver': 'file',
                'path': str(self.path),
            },
            'metadata': {
                'dtype': '<f4' if self.dtype == np.float32 else '<f8',
                'shape': [initial_time_capacity, self.n_neurons, self.n_features],
                'chunks': list(self.chunks),
                'compressor': {
                    'id': 'blosc',
                    'cname': 'zstd',
                    'clevel': 3,
                    'shuffle': 2,  # bitshuffle
                },
            },
            'create': True,
            'delete_existing': True,
        }

        self._store = ts.open(spec).result()
        self._initialized = True

    def append(self, frame: np.ndarray):
        """append a single frame to the buffer.

        args:
            frame: array of shape (n_neurons, n_features)
        """
        if frame.shape != (self.n_neurons, self.n_features):
            raise ValueError(
                f"frame shape {frame.shape} doesn't match expected "
                f"({self.n_neurons}, {self.n_features})"
            )

        self._buffer.append(frame.astype(self.dtype, copy=False))

        # flush buffer when it reaches chunk size
        if len(self._buffer) >= self.chunks[0]:
            self._flush_buffer()

    def _flush_buffer(self):
        """write buffered frames to zarr store."""
        if not self._buffer:
            return

        # stack frames into array
        data = np.stack(self._buffer, axis=0)
        n_frames = data.shape[0]

        if not self._initialized:
            self._initialize_store(self._buffer[0])

        # check if we need to resize
        current_shape = self._store.shape
        needed_size = self._total_frames + n_frames
        if needed_size > current_shape[0]:
            # resize to accommodate new data (with some headroom)
            new_size = max(needed_size, current_shape[0] * 2)
            self._store = self._store.resize(
                exclusive_max=[new_size, self.n_neurons, self.n_features]
            ).result()

        # write data
        self._store[self._total_frames:self._total_frames + n_frames].write(data).result()

        self._total_frames += n_frames
        self._buffer.clear()

    def finalize(self):
        """finalize the zarr store - flush remaining buffer and resize to exact size."""
        # flush any remaining buffered data
        self._flush_buffer()

        if self._store is not None and self._total_frames > 0:
            # resize to exact final size
            self._store = self._store.resize(
                exclusive_max=[self._total_frames, self.n_neurons, self.n_features]
            ).result()

        return self._total_frames


class ZarrSimulationWriterV2:
    """split metadata/timeseries writer for efficient storage (V2 format).

    separates static columns (INDEX, XPOS, YPOS, GROUP_TYPE, TYPE) from
    dynamic columns (VOLTAGE, STIMULUS, CALCIUM, FLUORESCENCE) to avoid
    redundant storage of position data.

    storage structure:
        path/
            metadata.zarr    # (N, 5) static columns, stored once
            timeseries.zarr  # (T, N, 4) dynamic columns

    usage:
        writer = ZarrSimulationWriterV2(path, n_neurons=14011, n_features=9)
        for frame in simulation:
            writer.append(frame)  # frame is (N, 9)
        writer.finalize()
    """

    def __init__(
        self,
        path: str | Path,
        n_neurons: int,
        n_features: int = 9,
        time_chunks: int = 2000,
        dtype: np.dtype = np.float32,
    ):
        """initialize V2 zarr writer.

        args:
            path: output directory path
            n_neurons: number of neurons
            n_features: total features per neuron (must be 9 for flyvis)
            time_chunks: chunk size along time dimension for timeseries
            dtype: data type for storage
        """
        self.path = Path(path)
        self.n_neurons = n_neurons
        self.n_features = n_features
        self.time_chunks = time_chunks
        self.dtype = dtype

        # paths for sub-arrays
        self.metadata_path = self.path / 'metadata.zarr'
        self.timeseries_path = self.path / 'timeseries.zarr'

        # state
        self._metadata_saved = False
        self._metadata: np.ndarray | None = None
        self._buffer: list[np.ndarray] = []
        self._total_frames = 0
        self._ts_store: ts.TensorStore | None = None
        self._ts_initialized = False

    def _save_metadata(self, frame: np.ndarray):
        """save static metadata from first frame."""
        # ensure directory exists
        self.path.mkdir(parents=True, exist_ok=True)

        # remove existing if present
        if self.metadata_path.exists():
            import shutil
            shutil.rmtree(self.metadata_path)

        # extract static columns: [INDEX, XPOS, YPOS, GROUP_TYPE, TYPE]
        self._metadata = frame[:, STATIC_COLS].astype(self.dtype)  # (N, 5)

        # save metadata as zarr (small, no need for chunking)
        spec = {
            'driver': 'zarr',
            'kvstore': {
                'driver': 'file',
                'path': str(self.metadata_path),
            },
            'metadata': {
                'dtype': '<f4' if self.dtype == np.float32 else '<f8',
                'shape': list(self._metadata.shape),
                'chunks': list(self._metadata.shape),  # single chunk
                'compressor': {
                    'id': 'blosc',
                    'cname': 'zstd',
                    'clevel': 3,
                    'shuffle': 2,
                },
            },
            'create': True,
            'delete_existing': True,
        }

        store = ts.open(spec).result()
        store.write(self._metadata).result()
        self._metadata_saved = True

    def _initialize_timeseries_store(self):
        """initialize timeseries zarr store."""
        if self.timeseries_path.exists():
            import shutil
            shutil.rmtree(self.timeseries_path)

        initial_time_capacity = max(self.time_chunks * 10, 1000)

        spec = {
            'driver': 'zarr',
            'kvstore': {
                'driver': 'file',
                'path': str(self.timeseries_path),
            },
            'metadata': {
                'dtype': '<f4' if self.dtype == np.float32 else '<f8',
                'shape': [initial_time_capacity, self.n_neurons, N_DYNAMIC],
                'chunks': [self.time_chunks, self.n_neurons, 1],  # column-chunked
                'compressor': {
                    'id': 'blosc',
                    'cname': 'zstd',
                    'clevel': 3,
                    'shuffle': 2,
                },
            },
            'create': True,
            'delete_existing': True,
        }

        self._ts_store = ts.open(spec).result()
        self._ts_initialized = True

    def append(self, frame: np.ndarray):
        """append a single frame.

        args:
            frame: array of shape (n_neurons, n_features) with all 9 columns
        """
        if frame.shape != (self.n_neurons, self.n_features):
            raise ValueError(
                f"frame shape {frame.shape} doesn't match expected "
                f"({self.n_neurons}, {self.n_features})"
            )

        # save metadata from first frame
        if not self._metadata_saved:
            self._save_metadata(frame)

        # extract dynamic columns: [VOLTAGE, STIMULUS, CALCIUM, FLUORESCENCE]
        dynamic_data = frame[:, DYNAMIC_COLS].astype(self.dtype)
        self._buffer.append(dynamic_data)

        # flush when buffer reaches chunk size
        if len(self._buffer) >= self.time_chunks:
            self._flush_buffer()

    def _flush_buffer(self):
        """write buffered timeseries data."""
        if not self._buffer:
            return

        data = np.stack(self._buffer, axis=0)  # (chunk_size, N, 4)
        n_frames = data.shape[0]

        if not self._ts_initialized:
            self._initialize_timeseries_store()

        # resize if needed
        current_shape = self._ts_store.shape
        needed_size = self._total_frames + n_frames
        if needed_size > current_shape[0]:
            new_size = max(needed_size, current_shape[0] * 2)
            self._ts_store = self._ts_store.resize(
                exclusive_max=[new_size, self.n_neurons, N_DYNAMIC]
            ).result()

        # write
        self._ts_store[self._total_frames:self._total_frames + n_frames].write(data).result()

        self._total_frames += n_frames
        self._buffer.clear()

    def finalize(self):
        """finalize - flush buffer and resize to exact size."""
        self._flush_buffer()

        if self._ts_store is not None and self._total_frames > 0:
            self._ts_store = self._ts_store.resize(
                exclusive_max=[self._total_frames, self.n_neurons, N_DYNAMIC]
            ).result()

        return self._total_frames


def detect_format(path: str | Path) -> Literal['npy', 'zarr_v2', 'zarr_v1', 'none']:
    """check what format exists at path.

    args:
        path: base path without extension

    returns:
        'npy' if .npy file exists
        'zarr_v2' if V2 zarr directory exists (with metadata.zarr + timeseries.zarr)
        'zarr_v1' if V1 zarr directory exists (with .zarray file)
        'none' if nothing exists
    """
    path = Path(path)

    # strip any existing extension
    base_path = path.with_suffix('') if path.suffix in ('.npy', '.zarr') else path

    npy_path = Path(str(base_path) + '.npy')
    zarr_v1_path = Path(str(base_path) + '.zarr')

    # check for V2 zarr format (directory with metadata.zarr and timeseries.zarr)
    if base_path.exists() and base_path.is_dir():
        metadata_path = base_path / 'metadata.zarr'
        timeseries_path = base_path / 'timeseries.zarr'
        if metadata_path.exists() and timeseries_path.exists():
            return 'zarr_v2'

    # check for V1 zarr format (directory with .zarray file)
    if zarr_v1_path.exists() and zarr_v1_path.is_dir():
        zarray_path = zarr_v1_path / '.zarray'
        if zarray_path.exists():
            return 'zarr_v1'

    # check for npy
    if npy_path.exists():
        return 'npy'

    return 'none'


def _load_zarr_v1(path: Path) -> np.ndarray:
    """load V1 zarr format (simple zarr array)."""
    zarr_path = Path(str(path) + '.zarr')
    spec = {
        'driver': 'zarr',
        'kvstore': {'driver': 'file', 'path': str(zarr_path)},
    }
    return ts.open(spec).result().read().result()


def _load_zarr_v2(path: Path) -> np.ndarray:
    """load V2 zarr split format and reconstruct full (T, N, 9) array."""
    metadata_path = path / 'metadata.zarr'
    timeseries_path = path / 'timeseries.zarr'

    # load metadata (N, 5)
    meta_spec = {
        'driver': 'zarr',
        'kvstore': {'driver': 'file', 'path': str(metadata_path)},
    }
    metadata = ts.open(meta_spec).result().read().result()  # (N, 5)

    # load timeseries (T, N, 4)
    ts_spec = {
        'driver': 'zarr',
        'kvstore': {'driver': 'file', 'path': str(timeseries_path)},
    }
    timeseries = ts.open(ts_spec).result().read().result()  # (T, N, 4)

    T, N = timeseries.shape[:2]

    # reconstruct full array
    full = np.empty((T, N, 9), dtype=np.float32)

    # static columns - broadcast from (N,) to (T, N)
    full[:, :, 0] = metadata[:, 0]  # INDEX
    full[:, :, 1] = metadata[:, 1]  # XPOS
    full[:, :, 2] = metadata[:, 2]  # YPOS
    full[:, :, 5] = metadata[:, 3]  # GROUP_TYPE
    full[:, :, 6] = metadata[:, 4]  # TYPE

    # dynamic columns
    full[:, :, 3] = timeseries[:, :, 0]  # VOLTAGE
    full[:, :, 4] = timeseries[:, :, 1]  # STIMULUS
    full[:, :, 7] = timeseries[:, :, 2]  # CALCIUM
    full[:, :, 8] = timeseries[:, :, 3]  # FLUORESCENCE

    return full


def load_simulation_data(path: str | Path) -> np.ndarray:
    """load simulation data from zarr or npy format.

    args:
        path: base path (with or without extension)

    returns:
        numpy array with simulation data

    raises:
        FileNotFoundError: if no data found at path
    """
    path = Path(path)
    fmt = detect_format(path)

    # get base path without extension
    base_path = path.with_suffix('') if path.suffix in ('.npy', '.zarr') else path

    if fmt == 'none':
        raise FileNotFoundError(f"no .npy or .zarr found at {base_path}")

    if fmt == 'npy':
        npy_path = Path(str(base_path) + '.npy')
        return np.load(npy_path)

    if fmt == 'zarr_v1':
        return _load_zarr_v1(base_path)

    # zarr_v2 format - load and reconstruct full array
    return _load_zarr_v2(base_path)


def load_zarr_lazy(path: str | Path) -> ts.TensorStore:
    """load zarr file as tensorstore handle for lazy access.

    args:
        path: path to zarr directory

    returns:
        tensorstore handle
    """
    path = Path(path)
    if not str(path).endswith('.zarr'):
        path = Path(str(path) + '.zarr')

    spec = {
        'driver': 'zarr',
        'kvstore': {
            'driver': 'file',
            'path': str(path),
        },
    }

    return ts.open(spec).result()
