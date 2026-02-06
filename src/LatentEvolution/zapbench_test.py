"""tests for zapbench sparse loading and interpolation."""

import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import tensorstore as ts
import torch

from LatentEvolution.zapbench import (
    load_and_interpolate,
    load_sparse_activity,
    interpolate_sparse,
)


def _write_zarr(path: str, array: np.ndarray):
    """write a numpy array to zarr via tensorstore."""
    spec = {
        "driver": "zarr",
        "kvstore": {"driver": "file", "path": path},
        "metadata": {"dtype": array.dtype.str, "shape": list(array.shape)},
        "create": True,
        "delete_existing": True,
    }
    store = ts.open(spec).result()
    store.write(array).result()


def _create_ephys_zarr(ephys_dir: str, cell_ephys_index: np.ndarray, sampling_freq_hz: float):
    """create ephys.zarr directory with cell_ephys_index and zarr.json."""
    os.makedirs(ephys_dir, exist_ok=True)

    # write zarr.json with sampling frequency
    zarr_json = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {"sampling_frequency_hz": sampling_freq_hz},
    }
    with open(os.path.join(ephys_dir, "zarr.json"), "w") as f:
        json.dump(zarr_json, f)

    # write cell_ephys_index array (T, N)
    cell_index_path = os.path.join(ephys_dir, "cell_ephys_index")
    _write_zarr(cell_index_path, cell_ephys_index.astype(np.int64))


class TestInterpolateSparse(unittest.TestCase):
    """tests for the sparse searchsorted interpolation."""

    def _make_sparse(self, obs_times_list, obs_vals_list, T):
        """build padded sparse tensors from per-neuron lists."""
        N = len(obs_times_list)
        K_max = max(len(t) for t in obs_times_list)
        obs_times = torch.full((N, K_max), T + 1, dtype=torch.long)
        obs_vals = torch.zeros(N, K_max)
        counts = torch.zeros(N, dtype=torch.long)
        for n in range(N):
            k = len(obs_times_list[n])
            counts[n] = k
            obs_times[n, :k] = torch.tensor(obs_times_list[n])
            obs_vals[n, :k] = torch.tensor(obs_vals_list[n])
        return obs_times, obs_vals, counts

    def test_no_gaps(self):
        """fully observed matrix is returned unchanged."""
        T = 3
        ot, ov, counts = self._make_sparse(
            [[0, 1, 2], [0, 1, 2]],
            [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]],
            T,
        )
        result = interpolate_sparse(ot, ov, counts, T)
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.assertTrue(torch.allclose(result, expected))

    def test_single_gap_linear_interp(self):
        """one gap between two observations is linearly interpolated."""
        T = 3
        # neuron 0: observed at t=0 (val=0) and t=2 (val=6)
        # neuron 1: fully observed
        ot, ov, counts = self._make_sparse(
            [[0, 2], [0, 1, 2]],
            [[0.0, 6.0], [1.0, 2.0, 3.0]],
            T,
        )
        result = interpolate_sparse(ot, ov, counts, T)
        self.assertAlmostEqual(result[1, 0].item(), 3.0)
        self.assertAlmostEqual(result[1, 1].item(), 2.0)

    def test_multiple_gaps(self):
        """two gaps between observations."""
        T = 4
        ot, ov, counts = self._make_sparse([[0, 3]], [[0.0, 9.0]], T)
        result = interpolate_sparse(ot, ov, counts, T)
        self.assertAlmostEqual(result[1, 0].item(), 3.0)
        self.assertAlmostEqual(result[2, 0].item(), 6.0)

    def test_boundary_no_prior_observation(self):
        """gap at start extrapolates from next observed value."""
        T = 3
        ot, ov, counts = self._make_sparse([[2]], [[5.0]], T)
        result = interpolate_sparse(ot, ov, counts, T)
        self.assertAlmostEqual(result[0, 0].item(), 5.0)
        self.assertAlmostEqual(result[1, 0].item(), 5.0)

    def test_boundary_no_next_observation(self):
        """gap at end extrapolates from prior observed value."""
        T = 3
        ot, ov, counts = self._make_sparse([[0]], [[3.0]], T)
        result = interpolate_sparse(ot, ov, counts, T)
        self.assertAlmostEqual(result[1, 0].item(), 3.0)
        self.assertAlmostEqual(result[2, 0].item(), 3.0)

    def test_staggered_pattern(self):
        """alternating observation pattern across neurons."""
        T = 5
        ot, ov, counts = self._make_sparse(
            [[0, 2, 4], [1, 3]],
            [[10.0, 30.0, 50.0], [20.0, 40.0]],
            T,
        )
        result = interpolate_sparse(ot, ov, counts, T)
        self.assertAlmostEqual(result[1, 0].item(), 20.0)
        self.assertAlmostEqual(result[3, 0].item(), 40.0)
        self.assertAlmostEqual(result[0, 1].item(), 20.0)
        self.assertAlmostEqual(result[2, 1].item(), 30.0)
        self.assertAlmostEqual(result[4, 1].item(), 40.0)


class TestLoadSparseActivity(unittest.TestCase):
    """tests for loading sparse activity from zarr."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.traces_path = os.path.join(self.tmpdir, "traces.zarr")
        self.ephys_path = os.path.join(self.tmpdir, "ephys.zarr")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_basic_load(self):
        """load sparse activity from zarr."""
        # 4 frames, 3 neurons, 1000 Hz sampling
        # sample indices: neuron n at frame t has index t * 100 + n * 10
        T, N = 4, 3
        traces = np.arange(T * N, dtype=np.float32).reshape(T, N)
        cell_ephys_index = np.array([
            [0, 10, 20],
            [100, 110, 120],
            [200, 210, 220],
            [300, 310, 320],
        ], dtype=np.int64)

        _write_zarr(self.traces_path, traces)
        _create_ephys_zarr(self.ephys_path, cell_ephys_index, sampling_freq_hz=1000.0)

        obs_times, obs_vals, counts, num_bins = load_sparse_activity(
            self.traces_path, self.ephys_path, bin_size_ms=50.0,
        )

        # shape is (N, T)
        self.assertEqual(obs_times.shape, (3, 4))
        self.assertEqual(obs_vals.shape, (3, 4))
        self.assertTrue(torch.all(counts == 4))

    def test_time_slice(self):
        """load subset of frames via time_slice."""
        T, N = 10, 2
        traces = np.arange(T * N, dtype=np.float32).reshape(T, N)
        cell_ephys_index = np.arange(T * N, dtype=np.int64).reshape(T, N) * 10

        _write_zarr(self.traces_path, traces)
        _create_ephys_zarr(self.ephys_path, cell_ephys_index, sampling_freq_hz=1000.0)

        obs_times, obs_vals, counts, num_bins = load_sparse_activity(
            self.traces_path, self.ephys_path, bin_size_ms=50.0,
            time_slice=slice(3, 7),
        )

        # 4 frames loaded
        self.assertEqual(obs_times.shape, (2, 4))
        self.assertTrue(torch.all(counts == 4))

    def test_time_slice_clamped(self):
        """time_slice exceeding bounds is clamped."""
        T, N = 10, 3
        traces = np.ones((T, N), dtype=np.float32)
        cell_ephys_index = np.zeros((T, N), dtype=np.int64)

        _write_zarr(self.traces_path, traces)
        _create_ephys_zarr(self.ephys_path, cell_ephys_index, sampling_freq_hz=1000.0)

        # request [8, 100) but only 10 frames exist -> clamps to [8, 10)
        obs_times, obs_vals, counts, num_bins = load_sparse_activity(
            self.traces_path, self.ephys_path, bin_size_ms=10.0,
            time_slice=slice(8, 100),
        )

        self.assertEqual(obs_times.shape, (3, 2))
        self.assertTrue(torch.all(counts == 2))


class TestLoadAndInterpolate(unittest.TestCase):
    """end-to-end test for load_and_interpolate."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.traces_path = os.path.join(self.tmpdir, "traces.zarr")
        self.ephys_path = os.path.join(self.tmpdir, "ephys.zarr")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_staggered_interpolation(self):
        """staggered acquisition times are correctly interpolated."""
        # 3 frames, 2 neurons at 1000 Hz sampling
        # bin_size = 50ms
        # neuron 0: samples at 0, 100, 200 -> times 0, 100, 200 ms -> bins 0, 2, 4
        # neuron 1: samples at 50, 150, 250 -> times 50, 150, 250 ms -> bins 1, 3, 5
        traces = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        cell_ephys_index = np.array([
            [0, 50],
            [100, 150],
            [200, 250],
        ], dtype=np.int64)

        _write_zarr(self.traces_path, traces)
        _create_ephys_zarr(self.ephys_path, cell_ephys_index, sampling_freq_hz=1000.0)

        result = load_and_interpolate(
            self.traces_path, self.ephys_path, bin_size_ms=50.0,
        )

        # bins span from 0 to 5 (6 bins total, indices relative to min)
        self.assertEqual(result.shape[0], 6)
        self.assertEqual(result.shape[1], 2)

        # neuron 0: observed at bins 0, 2, 4 with vals 1, 3, 5
        # bin 1 interpolates between 1 and 3 -> 2
        # bin 3 interpolates between 3 and 5 -> 4
        self.assertAlmostEqual(result[0, 0].item(), 1.0)
        self.assertAlmostEqual(result[1, 0].item(), 2.0)
        self.assertAlmostEqual(result[2, 0].item(), 3.0)
        self.assertAlmostEqual(result[3, 0].item(), 4.0)
        self.assertAlmostEqual(result[4, 0].item(), 5.0)

        # neuron 1: observed at bins 1, 3, 5 with vals 2, 4, 6
        # bin 0 extrapolates from bin 1 -> 2
        # bin 2 interpolates between 2 and 4 -> 3
        # bin 4 interpolates between 4 and 6 -> 5
        self.assertAlmostEqual(result[0, 1].item(), 2.0)
        self.assertAlmostEqual(result[1, 1].item(), 2.0)
        self.assertAlmostEqual(result[2, 1].item(), 3.0)
        self.assertAlmostEqual(result[3, 1].item(), 4.0)
        self.assertAlmostEqual(result[4, 1].item(), 5.0)
        self.assertAlmostEqual(result[5, 1].item(), 6.0)

    def test_with_time_slice(self):
        """load_and_interpolate respects time_slice."""
        T, N = 10, 2
        traces = np.arange(T * N, dtype=np.float32).reshape(T, N)
        # all neurons acquired at same time within each frame
        cell_ephys_index = np.tile(np.arange(T) * 100, (N, 1)).T.astype(np.int64)

        _write_zarr(self.traces_path, traces)
        _create_ephys_zarr(self.ephys_path, cell_ephys_index, sampling_freq_hz=1000.0)

        result = load_and_interpolate(
            self.traces_path, self.ephys_path, bin_size_ms=100.0,
            time_slice=slice(2, 5),
        )

        # 3 frames, 100ms bins, all neurons same time -> 3 bins
        self.assertEqual(result.shape, (3, 2))


if __name__ == "__main__":
    unittest.main()
