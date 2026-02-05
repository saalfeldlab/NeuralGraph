"""tests for zapbench load_staggered_activity and interpolate_irregular."""

import os
import shutil
import tempfile
import unittest

import numpy as np
import tensorstore as ts
import torch

from LatentEvolution.zapbench import (
    load_and_interpolate,
    load_staggered_activity,
    interpolate_staggered_activity,
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


def _write_acq_zarr(path: str, array: np.ndarray):
    """write acq array transposed to (N, T) as stored on disk."""
    _write_zarr(path, array.T)


class TestLoadStaggeredActivity(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.traces_path = os.path.join(self.tmpdir, "traces.zarr")
        self.acq_path = os.path.join(self.tmpdir, "acq.zarr")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_single_frame_two_neurons(self):
        """one frame, two neurons with different offsets."""
        # frame_period=100ms, bin_size=50ms
        # neuron 0 at offset 10ms -> absolute 10ms -> bin 0
        # neuron 1 at offset 60ms -> absolute 60ms -> bin 1
        traces = np.array([[1.0, 2.0]], dtype=np.float32)
        acq = np.array([[10.0, 60.0]], dtype=np.float64)
        _write_zarr(self.traces_path, traces)
        _write_acq_zarr(self.acq_path, acq)

        staggered, observed = load_staggered_activity(
            self.traces_path, self.acq_path,
            bin_size_ms=50.0, frame_period_ms=100.0,
        )
        # num_bins = ceil(1 * 100 / 50) = 2
        self.assertEqual(staggered.shape, (2, 2))
        self.assertAlmostEqual(staggered[0, 0].item(), 1.0)
        self.assertEqual(staggered[0, 1].item(), 0.0)
        self.assertEqual(staggered[1, 0].item(), 0.0)
        self.assertAlmostEqual(staggered[1, 1].item(), 2.0)
        expected_obs = torch.tensor([[True, False], [False, True]], device="cpu")
        self.assertTrue(torch.equal(observed, expected_obs))

    def test_two_frames_staggered_pattern(self):
        """two frames, verifying the staggered pattern across bins."""
        # frame_period=20ms, bin_size=10ms -> 4 bins per 2 frames
        # frame 0: neuron 0 at 0+2=2ms -> bin 0, neuron 1 at 0+12=12ms -> bin 1
        # frame 1: neuron 0 at 20+2=22ms -> bin 2, neuron 1 at 20+12=32ms -> bin 3
        traces = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        acq = np.array([[2.0, 12.0], [2.0, 12.0]], dtype=np.float64)
        _write_zarr(self.traces_path, traces)
        _write_acq_zarr(self.acq_path, acq)

        staggered, observed = load_staggered_activity(
            self.traces_path, self.acq_path,
            bin_size_ms=10.0, frame_period_ms=20.0,
        )
        # num_bins = ceil(2 * 20 / 10) = 4
        self.assertEqual(staggered.shape, (4, 2))
        self.assertAlmostEqual(staggered[0, 0].item(), 10.0)
        self.assertFalse(observed[0, 1].item())
        self.assertFalse(observed[1, 0].item())
        self.assertAlmostEqual(staggered[1, 1].item(), 20.0)
        self.assertAlmostEqual(staggered[2, 0].item(), 30.0)
        self.assertFalse(observed[2, 1].item())
        self.assertFalse(observed[3, 0].item())
        self.assertAlmostEqual(staggered[3, 1].item(), 40.0)

    def test_time_slice(self):
        """loading a subset of frames via time_slice."""
        traces = np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            dtype=np.float32,
        )
        acq = np.array(
            [[0.0, 5.0], [0.0, 5.0], [0.0, 5.0], [0.0, 5.0]],
            dtype=np.float64,
        )
        _write_zarr(self.traces_path, traces)
        _write_acq_zarr(self.acq_path, acq)

        # load only frames 2-3, frame_period=10ms, bin_size=10ms
        staggered, observed = load_staggered_activity(
            self.traces_path, self.acq_path,
            bin_size_ms=10.0, frame_period_ms=10.0,
            time_slice=slice(2, 4),
        )
        # 2 frames -> num_bins = ceil(2 * 10 / 10) = 2
        self.assertEqual(staggered.shape, (2, 2))
        self.assertAlmostEqual(staggered[0, 0].item(), 5.0)
        self.assertAlmostEqual(staggered[0, 1].item(), 6.0)
        self.assertAlmostEqual(staggered[1, 0].item(), 7.0)
        self.assertAlmostEqual(staggered[1, 1].item(), 8.0)

    def test_output_size_deterministic(self):
        """output size depends only on T, frame_period, bin_size."""
        traces = np.ones((5, 3), dtype=np.float32)
        acq = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ], dtype=np.float64)
        _write_zarr(self.traces_path, traces)
        _write_acq_zarr(self.acq_path, acq)

        stag1, _ = load_staggered_activity(
            self.traces_path, self.acq_path,
            bin_size_ms=20.0, frame_period_ms=100.0,
        )

        acq2 = np.full((5, 3), 99.0, dtype=np.float64)
        _write_acq_zarr(self.acq_path, acq2)

        stag2, _ = load_staggered_activity(
            self.traces_path, self.acq_path,
            bin_size_ms=20.0, frame_period_ms=100.0,
        )

        self.assertEqual(stag1.shape, stag2.shape)
        # ceil(5 * 100 / 20) = 25
        self.assertEqual(stag1.shape[0], 25)


class TestInterpolateIrregular(unittest.TestCase):

    def test_no_gaps(self):
        """fully observed matrix is returned unchanged."""
        data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        obs = torch.ones(3, 2, dtype=torch.bool)
        result = interpolate_staggered_activity(data, obs)
        self.assertTrue(torch.allclose(result, data))

    def test_single_gap_linear_interp(self):
        """one gap between two observations is linearly interpolated."""
        # neuron 0: observed at t=0 (val=0) and t=2 (val=6), gap at t=1
        # neuron 1: fully observed
        data = torch.tensor([[0.0, 1.0], [0.0, 2.0], [6.0, 3.0]])
        obs = torch.tensor([[True, True], [False, True], [True, True]])
        result = interpolate_staggered_activity(data, obs)
        self.assertAlmostEqual(result[1, 0].item(), 3.0)  # lerp(0, 6, 0.5)
        self.assertAlmostEqual(result[1, 1].item(), 2.0)  # already observed

    def test_multiple_gaps(self):
        """two gaps between observations."""
        # t=0: val=0 (observed), t=1,2: gap, t=3: val=9 (observed)
        data = torch.tensor([[0.0], [0.0], [0.0], [9.0]])
        obs = torch.tensor([[True], [False], [False], [True]])
        result = interpolate_staggered_activity(data, obs)
        self.assertAlmostEqual(result[0, 0].item(), 0.0)
        self.assertAlmostEqual(result[1, 0].item(), 3.0)  # lerp(0, 9, 1/3)
        self.assertAlmostEqual(result[2, 0].item(), 6.0)  # lerp(0, 9, 2/3)
        self.assertAlmostEqual(result[3, 0].item(), 9.0)

    def test_boundary_no_prior_observation(self):
        """gap at start with no prior observation uses next observed value."""
        data = torch.tensor([[0.0], [0.0], [5.0]])
        obs = torch.tensor([[False], [False], [True]])
        result = interpolate_staggered_activity(data, obs)
        # constant extrapolation from t=2
        self.assertAlmostEqual(result[0, 0].item(), 5.0)
        self.assertAlmostEqual(result[1, 0].item(), 5.0)

    def test_boundary_no_next_observation(self):
        """gap at end with no next observation uses prior observed value."""
        data = torch.tensor([[3.0], [0.0], [0.0]])
        obs = torch.tensor([[True], [False], [False]])
        result = interpolate_staggered_activity(data, obs)
        self.assertAlmostEqual(result[1, 0].item(), 3.0)
        self.assertAlmostEqual(result[2, 0].item(), 3.0)

    def test_staggered_pattern(self):
        """alternating observation pattern across neurons."""
        # neuron 0 observed at t=0,2,4; neuron 1 observed at t=1,3
        data = torch.tensor([
            [10.0, 0.0],
            [0.0, 20.0],
            [30.0, 0.0],
            [0.0, 40.0],
            [50.0, 0.0],
        ])
        obs = torch.tensor([
            [True, False],
            [False, True],
            [True, False],
            [False, True],
            [True, False],
        ])
        result = interpolate_staggered_activity(data, obs)
        # neuron 0: t=1 -> lerp(10, 30, 0.5) = 20, t=3 -> lerp(30, 50, 0.5) = 40
        self.assertAlmostEqual(result[1, 0].item(), 20.0)
        self.assertAlmostEqual(result[3, 0].item(), 40.0)
        # neuron 1: t=0 -> extrapolate from t=1 = 20, t=2 -> lerp(20, 40, 0.5) = 30
        self.assertAlmostEqual(result[0, 1].item(), 20.0)
        self.assertAlmostEqual(result[2, 1].item(), 30.0)
        # neuron 1: t=4 -> extrapolate from t=3 = 40
        self.assertAlmostEqual(result[4, 1].item(), 40.0)


class TestInterpolateSparse(unittest.TestCase):
    """tests for the sparse searchsorted interpolation path."""

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
        # (3, 2) fully observed
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


class TestLoadAndInterpolate(unittest.TestCase):
    """end-to-end test: load_and_interpolate matches dense path."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.traces_path = os.path.join(self.tmpdir, "traces.zarr")
        self.acq_path = os.path.join(self.tmpdir, "acq.zarr")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _dense_reference(self, bin_size_ms, frame_period_ms, time_slice=None):
        """produce interpolated result via the dense path."""
        staggered, observed = load_staggered_activity(
            self.traces_path, self.acq_path,
            bin_size_ms=bin_size_ms, frame_period_ms=frame_period_ms,
            time_slice=time_slice,
        )
        return interpolate_staggered_activity(staggered, observed)

    def test_matches_dense_path(self):
        """load_and_interpolate matches dense load + interpolate."""
        traces = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        acq = np.array([[2.0, 12.0], [2.0, 12.0]], dtype=np.float64)
        _write_zarr(self.traces_path, traces)
        _write_acq_zarr(self.acq_path, acq)

        expected = self._dense_reference(bin_size_ms=10.0, frame_period_ms=20.0)
        result = load_and_interpolate(
            self.traces_path, self.acq_path,
            bin_size_ms=10.0, frame_period_ms=20.0,
        )
        self.assertTrue(
            torch.allclose(expected, result, atol=1e-5),
            f"max diff: {(expected - result).abs().max().item():.2e}",
        )

    def test_with_time_slice(self):
        """load_and_interpolate with time_slice matches dense path."""
        traces = np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            dtype=np.float32,
        )
        acq = np.array(
            [[0.0, 5.0], [0.0, 5.0], [0.0, 5.0], [0.0, 5.0]],
            dtype=np.float64,
        )
        _write_zarr(self.traces_path, traces)
        _write_acq_zarr(self.acq_path, acq)

        expected = self._dense_reference(
            bin_size_ms=10.0, frame_period_ms=10.0, time_slice=slice(1, 3),
        )
        result = load_and_interpolate(
            self.traces_path, self.acq_path,
            bin_size_ms=10.0, frame_period_ms=10.0,
            time_slice=slice(1, 3),
        )
        self.assertTrue(
            torch.allclose(expected, result, atol=1e-5),
            f"max diff: {(expected - result).abs().max().item():.2e}",
        )

    def test_staggered_pattern(self):
        """staggered offsets produce correct interpolated output."""
        traces = np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            dtype=np.float32,
        )
        acq = np.array(
            [[0.0, 50.0], [0.0, 50.0], [0.0, 50.0]],
            dtype=np.float64,
        )
        _write_zarr(self.traces_path, traces)
        _write_acq_zarr(self.acq_path, acq)

        expected = self._dense_reference(bin_size_ms=50.0, frame_period_ms=100.0)
        result = load_and_interpolate(
            self.traces_path, self.acq_path,
            bin_size_ms=50.0, frame_period_ms=100.0,
        )
        self.assertTrue(
            torch.allclose(expected, result, atol=1e-5),
            f"max diff: {(expected - result).abs().max().item():.2e}",
        )


if __name__ == "__main__":
    unittest.main()
