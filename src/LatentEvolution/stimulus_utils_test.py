"""
unit tests for stimulus_utils module.
"""

import unittest
import torch

from LatentEvolution.stimulus_utils import downsample_stimulus
from LatentEvolution.training_config import StimulusFrequency


class TestDownsampleStimulus(unittest.TestCase):
    """tests for downsample_stimulus function."""

    def setUp(self):
        """set up test fixtures."""
        self.device = torch.device("cpu")
        # (total_steps=20, batch_size=2, dim_stim_latent=5)
        self.sample_stimulus = torch.randn(20, 2, 5, device=self.device)

    def test_all_mode_returns_unchanged(self):
        """test that ALL mode returns the input unchanged."""
        result = downsample_stimulus(
            self.sample_stimulus,
            tu=5,
            num_multiples=4,
            stimulus_frequency=StimulusFrequency.ALL,
        )
        self.assertTrue(torch.equal(result, self.sample_stimulus))
        self.assertEqual(result.shape, self.sample_stimulus.shape)

    def test_none_mode_returns_zeros(self):
        """test that NONE mode returns zeros with same shape."""
        result = downsample_stimulus(
            self.sample_stimulus,
            tu=5,
            num_multiples=4,
            stimulus_frequency=StimulusFrequency.NONE,
        )
        self.assertTrue(torch.all(result == 0))
        self.assertEqual(result.shape, self.sample_stimulus.shape)

    def test_constant_mode_shape(self):
        """test that TIME_UNITS_CONSTANT mode returns correct shape."""
        result = downsample_stimulus(
            self.sample_stimulus,
            tu=5,
            num_multiples=4,
            stimulus_frequency=StimulusFrequency.TIME_UNITS_CONSTANT,
        )
        self.assertEqual(result.shape, self.sample_stimulus.shape)

    def test_constant_mode_values_held(self):
        """test that TIME_UNITS_CONSTANT mode holds values constant within centered intervals."""
        # create known stimulus: each sample has distinct value
        # with tu=10, samples at t=0, 10, 20, 30
        proj_stim = torch.zeros(40, 1, 3, device=self.device)
        proj_stim[0, :, :] = 1.0   # sample at t=0
        proj_stim[10, :, :] = 2.0  # sample at t=10
        proj_stim[20, :, :] = 3.0  # sample at t=20
        proj_stim[30, :, :] = 4.0  # sample at t=30

        result = downsample_stimulus(
            proj_stim,
            tu=10,
            num_multiples=4,
            stimulus_frequency=StimulusFrequency.TIME_UNITS_CONSTANT,
        )

        # each sample is centered around its time point
        # sample at t=0: used for [0, 5) (first sample starts at 0)
        # sample at t=10: used for [5, 15)
        # sample at t=20: used for [15, 25)
        # sample at t=30: used for [25, 40) (last sample extends to end)
        self.assertTrue(torch.all(result[0:5] == 1.0))
        self.assertTrue(torch.all(result[5:15] == 2.0))
        self.assertTrue(torch.all(result[15:25] == 3.0))
        self.assertTrue(torch.all(result[25:40] == 4.0))

    def test_constant_mode_centered_intervals(self):
        """test that TIME_UNITS_CONSTANT mode uses centered intervals around sample points."""
        # test with tu=20: samples at 0, 20, 40, 60
        proj_stim = torch.zeros(80, 1, 1, device=self.device)
        proj_stim[0, :, :] = 10.0   # sample at t=0
        proj_stim[20, :, :] = 20.0  # sample at t=20
        proj_stim[40, :, :] = 30.0  # sample at t=40
        proj_stim[60, :, :] = 40.0  # sample at t=60

        result = downsample_stimulus(
            proj_stim,
            tu=20,
            num_multiples=4,
            stimulus_frequency=StimulusFrequency.TIME_UNITS_CONSTANT,
        )

        # centered intervals (half_tu = 10):
        # t=0:  [0, 10)    -> value 10.0
        # t=20: [10, 30)   -> value 20.0
        # t=40: [30, 50)   -> value 30.0
        # t=60: [50, 80)   -> value 40.0 (last sample extends to end)
        self.assertTrue(torch.all(result[0:10] == 10.0))
        self.assertTrue(torch.all(result[10:30] == 20.0))
        self.assertTrue(torch.all(result[30:50] == 30.0))
        self.assertTrue(torch.all(result[50:80] == 40.0))

    def test_interpolate_mode_shape(self):
        """test that TIME_UNITS_INTERPOLATE mode returns correct shape."""
        result = downsample_stimulus(
            self.sample_stimulus,
            tu=5,
            num_multiples=4,
            stimulus_frequency=StimulusFrequency.TIME_UNITS_INTERPOLATE,
        )
        self.assertEqual(result.shape, self.sample_stimulus.shape)

    def test_interpolate_mode_boundary_values(self):
        """test that TIME_UNITS_INTERPOLATE mode matches boundary values."""
        # create stimulus with known values at boundaries
        proj_stim = torch.zeros(20, 1, 2, device=self.device)
        proj_stim[0, 0, :] = torch.tensor([0.0, 0.0])
        proj_stim[5, 0, :] = torch.tensor([1.0, 2.0])
        proj_stim[10, 0, :] = torch.tensor([2.0, 4.0])
        proj_stim[15, 0, :] = torch.tensor([3.0, 6.0])

        result = downsample_stimulus(
            proj_stim,
            tu=5,
            num_multiples=4,
            stimulus_frequency=StimulusFrequency.TIME_UNITS_INTERPOLATE,
        )

        # check boundary values are preserved (at start of each interval)
        self.assertTrue(torch.allclose(result[0, 0, :], torch.tensor([0.0, 0.0], device=self.device)))
        self.assertTrue(torch.allclose(result[5, 0, :], torch.tensor([1.0, 2.0], device=self.device)))
        self.assertTrue(torch.allclose(result[10, 0, :], torch.tensor([2.0, 4.0], device=self.device)))
        # last interval uses constant (no final boundary), so check it matches previous boundary
        self.assertTrue(torch.allclose(result[15, 0, :], torch.tensor([3.0, 6.0], device=self.device)))

    def test_interpolate_mode_midpoint_values(self):
        """test that TIME_UNITS_INTERPOLATE mode correctly interpolates midpoints."""
        # create simple linear stimulus
        proj_stim = torch.zeros(20, 1, 1, device=self.device)
        proj_stim[0, 0, 0] = 0.0
        proj_stim[5, 0, 0] = 10.0
        proj_stim[10, 0, 0] = 20.0
        proj_stim[15, 0, 0] = 30.0

        result = downsample_stimulus(
            proj_stim,
            tu=5,
            num_multiples=4,
            stimulus_frequency=StimulusFrequency.TIME_UNITS_INTERPOLATE,
        )

        # check interpolation in first interval (0 to 10)
        # at step 2 (midpoint between 0 and 5), should be ~4.0
        # weights go from 0.0, 0.2, 0.4, 0.6, 0.8 for tu=5
        expected_step_2 = 0.0 * 0.6 + 10.0 * 0.4  # w=0.4 at step 2
        self.assertTrue(torch.allclose(result[2, 0, 0], torch.tensor(expected_step_2), atol=0.1))

    def test_edge_case_no_final_boundary(self):
        """test edge case when total_steps = num_multiples * tu (no final boundary)."""
        # this is the common case in training
        proj_stim = torch.randn(20, 2, 3, device=self.device)

        result = downsample_stimulus(
            proj_stim,
            tu=5,
            num_multiples=4,
            stimulus_frequency=StimulusFrequency.TIME_UNITS_INTERPOLATE,
        )

        # should handle gracefully without indexing errors
        self.assertEqual(result.shape, (20, 2, 3))

    def test_edge_case_with_final_boundary(self):
        """test case when we have extra data point for final boundary."""
        # 21 steps = 4 intervals * 5 tu + 1 extra for final boundary
        proj_stim = torch.randn(21, 2, 3, device=self.device)

        result = downsample_stimulus(
            proj_stim,
            tu=5,
            num_multiples=4,
            stimulus_frequency=StimulusFrequency.TIME_UNITS_INTERPOLATE,
        )

        # should only return 20 steps (4 complete intervals)
        self.assertEqual(result.shape, (20, 2, 3))

    def test_different_tu_values(self):
        """test with different time_units values."""
        for tu in [1, 2, 5, 10, 20]:
            total_steps = tu * 5  # 5 multiples
            proj_stim = torch.randn(total_steps, 2, 3, device=self.device)

            result = downsample_stimulus(
                proj_stim,
                tu=tu,
                num_multiples=5,
                stimulus_frequency=StimulusFrequency.TIME_UNITS_CONSTANT,
            )

            self.assertEqual(result.shape, proj_stim.shape)

    def test_invalid_stimulus_frequency_raises(self):
        """test that invalid stimulus frequency raises ValueError."""
        # use a mock invalid enum value
        class FakeStimulusFrequency:
            pass

        with self.assertRaisesRegex(ValueError, "unknown stimulus frequency"):
            downsample_stimulus(
                self.sample_stimulus,
                tu=5,
                num_multiples=4,
                stimulus_frequency=FakeStimulusFrequency(),
            )

    def test_preserves_device(self):
        """test that output is on the same device as input."""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            proj_stim = torch.randn(20, 2, 3, device=device)

            result = downsample_stimulus(
                proj_stim,
                tu=5,
                num_multiples=4,
                stimulus_frequency=StimulusFrequency.TIME_UNITS_INTERPOLATE,
            )

            self.assertEqual(result.device, device)

    def test_preserves_dtype(self):
        """test that output preserves input dtype."""
        for dtype in [torch.float32, torch.float64]:
            proj_stim = torch.randn(20, 2, 3, device=self.device, dtype=dtype)

            result = downsample_stimulus(
                proj_stim,
                tu=5,
                num_multiples=4,
                stimulus_frequency=StimulusFrequency.TIME_UNITS_CONSTANT,
            )

            self.assertEqual(result.dtype, dtype)

    def test_batch_dimension_independence(self):
        """test that each batch is processed independently."""
        # create stimulus with different values for each batch
        proj_stim = torch.zeros(20, 3, 2, device=self.device)
        proj_stim[:, 0, :] = 1.0  # batch 0
        proj_stim[:, 1, :] = 2.0  # batch 1
        proj_stim[:, 2, :] = 3.0  # batch 2

        result = downsample_stimulus(
            proj_stim,
            tu=5,
            num_multiples=4,
            stimulus_frequency=StimulusFrequency.TIME_UNITS_CONSTANT,
        )

        # each batch should maintain its distinct values
        self.assertTrue(torch.all(result[:, 0, :] == 1.0))
        self.assertTrue(torch.all(result[:, 1, :] == 2.0))
        self.assertTrue(torch.all(result[:, 2, :] == 3.0))


if __name__ == "__main__":
    unittest.main()
