"""
utilities for stimulus processing and downsampling.
"""

import torch

from LatentEvolution.training_config import StimulusFrequency


def downsample_stimulus(
    proj_stim_t: torch.Tensor,
    tu: int,
    num_multiples: int,
    stimulus_frequency: StimulusFrequency,
) -> torch.Tensor:
    """
    downsample encoded stimulus based on frequency mode.

    args:
        proj_stim_t: (total_steps, b, dim_stim_latent) encoded stimulus at all time points
        tu: time_units (observation interval)
        num_multiples: evolve_multiple_steps
        stimulus_frequency: StimulusFrequency enum mode

    returns:
        downsampled_proj_stim_t: (total_steps, b, dim_stim_latent) downsampled stimulus
    """
    total_steps, batch_size, dim_stim_latent = proj_stim_t.shape
    device = proj_stim_t.device

    if stimulus_frequency == StimulusFrequency.ALL:
        # use all time points (no downsampling)
        return proj_stim_t

    elif stimulus_frequency == StimulusFrequency.NONE:
        # no stimulus: return zeros
        return torch.zeros_like(proj_stim_t)

    elif stimulus_frequency == StimulusFrequency.TIME_UNITS_CONSTANT:
        # sample at 0, tu, 2*tu, ..., num_multiples*tu and hold constant
        # extract samples at time unit boundaries
        sample_indices = torch.arange(num_multiples + 1, device=device) * tu  # [0, tu, 2*tu, ...]
        proj_samples = proj_stim_t[sample_indices, :, :]  # (num_samples, b, dim_stim_latent)

        # repeat each sample tu times
        downsampled = torch.repeat_interleave(proj_samples, tu, dim=0)[:total_steps]
        return downsampled

    elif stimulus_frequency == StimulusFrequency.TIME_UNITS_INTERPOLATE:
        # sample at time unit boundaries and linearly interpolate
        sample_indices = torch.arange(num_multiples + 1, device=device) * tu
        proj_samples = proj_stim_t[sample_indices, :, :]  # (num_samples, b, dim_stim_latent)

        # interpolate in latent space between consecutive samples
        proj_stim_list = []
        for m in range(num_multiples):
            start_proj = proj_samples[m]      # (b, dim_stim_latent)
            end_proj = proj_samples[m + 1]    # (b, dim_stim_latent)

            # linear interpolation weights for tu steps
            # exclude 1.0 to avoid duplicates at boundaries
            weights = torch.linspace(0, 1, tu + 1, device=device)[:-1]  # (tu,)

            for w in weights:
                interp = (1 - w) * start_proj + w * end_proj  # (b, dim_stim_latent)
                proj_stim_list.append(interp)

        downsampled = torch.stack(proj_stim_list, dim=0)  # (total_steps, b, dim_stim_latent)
        return downsampled

    else:
        raise ValueError(f"unknown stimulus frequency: {stimulus_frequency}")
