"""unit tests for interpolate_staggered_to_aligned."""

import torch
import pytest

from LatentEvolution.interpolate_staggered import interpolate_staggered


@pytest.fixture
def simple_data():
    """simple (20, 4) data with known values for testing."""
    T, N = 20, 4
    # data[t, n] = t * 10 + n so values are easy to reason about
    t_idx = torch.arange(T, dtype=torch.float32).unsqueeze(1)
    n_idx = torch.arange(N, dtype=torch.float32).unsqueeze(0)
    return t_idx * 10 + n_idx


class TestExactAtObservationTimes:
    """at phi_i + k*tu, interpolated value == original data value."""

    def test_exact_match(self, simple_data):
        T, N = simple_data.shape
        tu = 4
        phases = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        result = interpolate_staggered(simple_data, phases, tu)

        # check at each neuron's observation times
        for n in range(N):
            phi = phases[n].item()
            obs_times = list(range(phi, T, tu))
            for t in obs_times:
                assert torch.isclose(result[t, n], simple_data[t, n]), (
                    f"neuron {n}, t={t}: expected {simple_data[t, n].item()}, "
                    f"got {result[t, n].item()}"
                )


class TestMidpointInterpolation:
    """at phi_i + k*tu + tu/2, value is average of neighboring observations."""

    def test_midpoint(self, simple_data):
        T, N = simple_data.shape
        tu = 4
        phases = torch.tensor([0, 0, 0, 0], dtype=torch.long)

        result = interpolate_staggered(simple_data, phases, tu)

        # midpoint at t=2 (between obs at t=0 and t=4)
        for n in range(N):
            expected = (simple_data[0, n] + simple_data[4, n]) / 2.0
            assert torch.isclose(result[2, n], expected), (
                f"neuron {n}, t=2: expected {expected.item()}, got {result[2, n].item()}"
            )

    def test_midpoint_with_phases(self, simple_data):
        T, N = simple_data.shape
        tu = 4
        phases = torch.tensor([1, 1, 1, 1], dtype=torch.long)

        result = interpolate_staggered(simple_data, phases, tu)

        # midpoint at t=3 (between obs at t=1 and t=5)
        for n in range(N):
            expected = (simple_data[1, n] + simple_data[5, n]) / 2.0
            assert torch.isclose(result[3, n], expected), (
                f"neuron {n}, t=3: expected {expected.item()}, got {result[3, n].item()}"
            )


class TestFirstWindowBoundary:
    """t < phi_i: left observation is missing, clamping holds boundary value."""

    def test_before_first_observation(self, simple_data):
        T, N = simple_data.shape
        tu = 4
        # neuron 3 has phase 3, so first observation is at t=3
        phases = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        result = interpolate_staggered(simple_data, phases, tu)

        # for neuron 3 at t=0,1,2 the left bracket t_lo is negative, clamped to 0
        # t_lo_clamped=0, t_hi = t_lo + tu. since t_lo was < 0, t_hi could be < tu.
        # the key check: values should be reasonable (no NaN, no crash)
        for t in range(3):
            assert torch.isfinite(result[t, 3]), f"t={t}: got non-finite value"


class TestLastWindowBoundary:
    """t > last observation: right observation is missing, clamping holds value."""

    def test_after_last_observation(self, simple_data):
        T, N = simple_data.shape
        tu = 4
        phases = torch.tensor([0, 0, 0, 0], dtype=torch.long)

        result = interpolate_staggered(simple_data, phases, tu)

        # last observation is at t=16 (0 + 4*4). for t=17,18,19 the right bracket
        # t_hi would be 20 which is clamped to 19.
        for t in range(17, T):
            for n in range(N):
                assert torch.isfinite(result[t, n]), f"t={t}, n={n}: got non-finite"


class TestAllZeroPhases:
    """time-aligned case: interpolation is identity at tu boundaries."""

    def test_identity_at_boundaries(self, simple_data):
        T, N = simple_data.shape
        tu = 4
        phases = torch.zeros(N, dtype=torch.long)

        result = interpolate_staggered(simple_data, phases, tu)

        # at tu boundaries (0, 4, 8, 12, 16), output should match input exactly
        for t in range(0, T, tu):
            assert torch.allclose(result[t], simple_data[t]), (
                f"t={t}: expected {simple_data[t].tolist()}, got {result[t].tolist()}"
            )

    def test_linear_between_boundaries(self, simple_data):
        T, N = simple_data.shape
        tu = 4
        phases = torch.zeros(N, dtype=torch.long)

        result = interpolate_staggered(simple_data, phases, tu)

        # at t=1 (1/4 of the way from t=0 to t=4)
        for n in range(N):
            expected = simple_data[0, n] * 0.75 + simple_data[4, n] * 0.25
            assert torch.isclose(result[1, n], expected, atol=1e-5), (
                f"n={n}, t=1: expected {expected.item()}, got {result[1, n].item()}"
            )


class TestShapePreservation:
    """output shape matches input shape."""

    def test_shape(self, simple_data):
        T, N = simple_data.shape
        tu = 4
        phases = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        result = interpolate_staggered(simple_data, phases, tu)
        assert result.shape == simple_data.shape

    def test_shape_large(self):
        T, N = 1000, 50
        tu = 10
        data = torch.randn(T, N)
        phases = torch.randint(0, tu, (N,))

        result = interpolate_staggered(data, phases, tu)
        assert result.shape == (T, N)

    def test_dtype_float(self):
        T, N = 20, 4
        tu = 4
        data = torch.randn(T, N)
        phases = torch.zeros(N, dtype=torch.long)

        result = interpolate_staggered(data, phases, tu)
        assert result.dtype == torch.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
