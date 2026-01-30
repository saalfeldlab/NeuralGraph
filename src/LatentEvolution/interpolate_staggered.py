"""
linear interpolation of staggered observation data to time-aligned grid.

staggered acquisition means neuron i is observed at times phi_i, phi_i + tu,
phi_i + 2*tu, ... this module creates a fake time-aligned dataset by linearly
interpolating between each neuron's bracketing staggered observation times.
"""

import torch


def interpolate_staggered_to_aligned(
    data: torch.Tensor,
    neuron_phases: torch.Tensor,
    time_units: int,
) -> torch.Tensor:
    """
    interpolate staggered observations onto an aligned time grid.

    for each timestep t and neuron i with phase phi_i:
    - find bracketing observations: t_lo = phi_i + floor((t - phi_i) / tu) * tu
    - t_hi = t_lo + tu
    - weight w = (t - t_lo) / tu
    - result = (1 - w) * data[t_lo, i] + w * data[t_hi, i]
    - boundary: clamp indices to [0, T-1]

    args:
        data: (T, N) full simulation data at all timesteps
        neuron_phases: (N,) phase offsets in [0, tu-1] for each neuron
        time_units: observation interval (tu)

    returns:
        (T, N) interpolated time-aligned data
    """
    T, N = data.shape
    device = data.device

    # t: (T, 1), phases: (1, N)
    t = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)  # (T, 1)
    phi = neuron_phases.unsqueeze(0).float()  # (1, N)

    tu = float(time_units)

    # find t_lo: the largest staggered observation time <= t
    # t_lo = phi + floor((t - phi) / tu) * tu
    t_lo = phi + torch.floor((t - phi) / tu) * tu  # (T, N)

    # t_hi = t_lo + tu
    t_hi = t_lo + tu  # (T, N)

    # clamp to valid index range [0, T-1]
    t_lo_idx = t_lo.clamp(0, T - 1).long()  # (T, N)
    t_hi_idx = t_hi.clamp(0, T - 1).long()  # (T, N)

    # weight: fraction of the way from t_lo to t_hi
    # when t_lo == t_hi (boundary clamped), w doesn't matter since both values are the same
    w = ((t - t_lo) / tu).clamp(0.0, 1.0)  # (T, N)

    # gather values at t_lo and t_hi for each neuron
    neuron_idx = torch.arange(N, device=device).unsqueeze(0).expand(T, N)  # (T, N)
    val_lo = data[t_lo_idx, neuron_idx]  # (T, N)
    val_hi = data[t_hi_idx, neuron_idx]  # (T, N)

    # linear interpolation
    result = (1.0 - w) * val_lo + w * val_hi  # (T, N)

    return result
