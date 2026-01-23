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
        """test that TIME_UNITS_CONSTANT mode holds values constant within intervals."""
        # create known stimulus: each interval has distinct value
        proj_stim = torch.zeros(20, 1, 3, device=self.device)
        proj_stim[0, :, :] = 1.0  # first interval
        proj_stim[5, :, :] = 2.0  # second interval
        proj_stim[10, :, :] = 3.0  # third interval
        proj_stim[15, :, :] = 4.0  # fourth interval

        result = downsample_stimulus(
            proj_stim,
            tu=5,
            num_multiples=4,
            stimulus_frequency=StimulusFrequency.TIME_UNITS_CONSTANT,
        )

        # check that each interval holds the value from its boundary
        self.assertTrue(torch.all(result[0:5] == 1.0))
        self.assertTrue(torch.all(result[5:10] == 2.0))
        self.assertTrue(torch.all(result[10:15] == 3.0))
        self.assertTrue(torch.all(result[15:20] == 4.0))

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
        self.assertTrue(torch.allclose(result[0, 0, :], torch.tensor([0.0, 0.0])))
        self.assertTrue(torch.allclose(result[5, 0, :], torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.allclose(result[10, 0, :], torch.tensor([2.0, 4.0])))
        # last interval uses constant (no final boundary), so check it matches previous boundary
        self.assertTrue(torch.allclose(result[15, 0, :], torch.tensor([3.0, 6.0])))

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
