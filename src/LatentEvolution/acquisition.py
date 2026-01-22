"""
data acquisition mode logic for latent evolution training.

handles different sampling strategies:
- all_time_points: all data available (current default behavior)
- time_aligned: observations only at 0, tu, 2tu, ... for all neurons
- staggered_random: each neuron observed every tu steps at random phase offsets
"""

from typing import Literal, Union, Annotated
from pydantic import BaseModel, Field, ConfigDict
import torch


# -------------------------------------------------------------------
# acquisition mode config classes
# -------------------------------------------------------------------


class AllTimePointsMode(BaseModel):
    """current behavior: all data points available for training."""
    mode: Literal["all_time_points"] = "all_time_points"
    model_config = ConfigDict(extra="forbid")


class TimeAlignedMode(BaseModel):
    """observations only at 0, tu, 2tu, ... all neurons observed simultaneously."""
    mode: Literal["time_aligned"] = "time_aligned"
    model_config = ConfigDict(extra="forbid")


class StaggeredRandomMode(BaseModel):
    """each neuron observed every tu steps, but at different random phases."""
    mode: Literal["staggered_random"] = "staggered_random"
    seed: int = Field(
        42,
        description="random seed for assigning neuron phases"
    )
    model_config = ConfigDict(extra="forbid")


# discriminated union type
AcquisitionMode = Annotated[
    Union[AllTimePointsMode, TimeAlignedMode, StaggeredRandomMode],
    Field(discriminator='mode')
]


# -------------------------------------------------------------------
# pre-computation (once per training run)
# -------------------------------------------------------------------


def compute_neuron_phases(
    num_neurons: int,
    time_units: int,
    acquisition_mode: AcquisitionMode,
    device: torch.device,
) -> torch.Tensor | None:
    """
    pre-compute neuron phase offsets.

    args:
        num_neurons: number of neurons in the model
        time_units: observation interval (tu parameter)
        acquisition_mode: acquisition mode configuration
        device: pytorch device

    returns:
        None for all_time_points
        zeros (num_neurons,) for time_aligned (all neurons phase 0)
        random phases (num_neurons,) in [0, tu-1] for staggered_random
    """
    if isinstance(acquisition_mode, AllTimePointsMode):
        return None
    elif isinstance(acquisition_mode, TimeAlignedMode):
        return torch.zeros(num_neurons, dtype=torch.long, device=device)
    elif isinstance(acquisition_mode, StaggeredRandomMode):
        rng = torch.Generator(device=device).manual_seed(acquisition_mode.seed)
        phases = torch.randint(
            0, time_units, (num_neurons,), generator=rng, device=device
        )
        return phases
    else:
        raise ValueError(f"unknown acquisition mode: {acquisition_mode}")


# -------------------------------------------------------------------
# batch sampling
# -------------------------------------------------------------------


def sample_batch_indices(
    chunk_size: int,
    total_steps: int,
    time_units: int,
    batch_size: int,
    num_neurons: int,
    neuron_phases: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    """
    sample observation indices for a batch, respecting acquisition mode constraints.

    assumes chunk_start is a multiple of time_units.

    args:
        chunk_size: number of timesteps in chunk
        total_steps: time_units * evolve_multiple_steps
        time_units: observation interval
        batch_size: number of samples in batch
        num_neurons: number of neurons
        neuron_phases: pre-computed phases or None for all_time_points
        device: pytorch device

    returns:
        observation_indices: (batch_size, num_neurons) tensor of observation times relative to chunk start
    """
    max_start_in_chunk = chunk_size - total_steps

    if neuron_phases is None:
        # all_time_points: sample any index in chunk
        # all neurons observed at same time
        batch_start_indices = torch.randint(
            0, max_start_in_chunk, size=(batch_size,), device=device
        )
        # broadcast to (batch_size, num_neurons)
        observation_indices = batch_start_indices.unsqueeze(1).expand(batch_size, num_neurons)
    else:
        # time_aligned or staggered_random: sample only at multiples of tu
        num_valid = max_start_in_chunk // time_units
        if num_valid == 0:
            raise ValueError(
                f"no valid starting indices in chunk. "
                f"chunk_size={chunk_size}, total_steps={total_steps}, time_units={time_units}"
            )
        sampled_multiples = torch.randint(0, num_valid, size=(batch_size,), device=device)
        batch_start_indices = sampled_multiples * time_units  # (batch_size,)

        # add neuron phases: (batch_size, 1) + (1, num_neurons) -> (batch_size, num_neurons)
        observation_indices = batch_start_indices.unsqueeze(1) + neuron_phases.unsqueeze(0)

    return observation_indices
