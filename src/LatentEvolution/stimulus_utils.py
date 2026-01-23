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
        # sample at 0, tu, 2*tu, ..., and hold constant
        # only sample at indices that exist in the input tensor
        num_samples = min(num_multiples + 1, (total_steps + tu - 1) // tu)  # how many complete intervals we can sample
        sample_indices = torch.arange(num_samples, device=device) * tu  # [0, tu, 2*tu, ...]
        proj_samples = proj_stim_t[sample_indices, :, :]  # (num_samples, b, dim_stim_latent)

        # repeat each sample tu times
        downsampled = torch.repeat_interleave(proj_samples, tu, dim=0)[:total_steps]
        return downsampled

    elif stimulus_frequency == StimulusFrequency.TIME_UNITS_INTERPOLATE:
        # sample at time unit boundaries and linearly interpolate
        # check if we have enough data for the final boundary point
        final_boundary_idx = num_multiples * tu

        if final_boundary_idx >= total_steps:
            # not enough data for final boundary - use constant mode for last interval
            # this happens when total_steps = num_multiples * tu (no extra boundary point)
            sample_indices = torch.arange(num_multiples, device=device) * tu  # [0, tu, 2*tu, ..., (num_multiples-1)*tu]
            proj_samples = proj_stim_t[sample_indices, :, :]  # (num_multiples, b, dim_stim_latent)

            proj_stim_list = []
            # interpolate for all intervals except the last
            for m in range(num_multiples - 1):
                start_proj = proj_samples[m]
                end_proj = proj_samples[m + 1]
                weights = torch.linspace(0, 1, tu + 1, device=device)[:-1]
                for w in weights:
                    interp = (1 - w) * start_proj + w * end_proj
                    proj_stim_list.append(interp)

            # for the last interval, hold constant (no boundary point to interpolate to)
            last_proj = proj_samples[-1]
            remaining_steps = total_steps - len(proj_stim_list)
            for _ in range(remaining_steps):
                proj_stim_list.append(last_proj)

            downsampled = torch.stack(proj_stim_list, dim=0)
        else:
            # we have enough data for full interpolation including final boundary
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
