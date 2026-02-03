"""
shared training configuration classes for neural dynamics models.

includes profiling, training hyperparameters, and cross-validation configs.
"""

from enum import Enum, auto
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
import torch

from LatentEvolution.acquisition import AcquisitionMode, AllTimePointsMode
from LatentEvolution.stimulus_ae_model import StimulusAETrainingConfig


class StimulusFrequency(Enum):
    """stimulus frequency for training.

    controls how stimulus is provided during evolution steps between observations.
    """
    ALL = auto()                    # use stimulus at every time step (current behavior)
    NONE = auto()                   # no stimulus provided (set to zero)
    TIME_UNITS_CONSTANT = auto()    # use stimulus at time_units intervals, hold constant between
    TIME_UNITS_INTERPOLATE = auto() # use stimulus at time_units intervals, linearly interpolate between


class DataSplit(BaseModel):
    """split the time series into train/validation sets."""

    train_start: int
    train_end: int
    validation_start: int
    validation_end: int

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @field_validator("*")
    @classmethod
    def check_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("indices in data_split must be non-negative.")
        return v

    @field_validator("train_end")
    @classmethod
    def check_order(cls, v, info):
        # very basic ordering sanity check
        d = info.data
        if "train_start" in d and v <= d["train_start"]:
            raise ValueError("train_end must be greater than train_start.")
        return v


class ProfileConfig(BaseModel):
    """configuration for pytorch profiler to generate chrome traces."""
    wait: int = Field(
        1, description="number of epochs to skip before starting profiler warmup"
    )
    warmup: int = Field(
        1, description="number of epochs for profiler warmup"
    )
    active: int = Field(
        1, description="number of epochs to actively profile"
    )
    repeat: int = Field(
        0, description="number of times to repeat the profiling cycle"
    )
    record_shapes: bool = Field(
        True, description="record tensor shapes in the trace"
    )
    profile_memory: bool = Field(
        True, description="profile memory usage"
    )
    with_stack: bool = Field(
        False, description="record source code stack traces (increases overhead)"
    )
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class UnconnectedToZeroConfig(BaseModel):
    """augmentation: add synthetic unconnected neurons with zero activity."""
    num_neurons: int = Field(0, description="number of unconnected neurons to add")
    loss_coeff: float = Field(1.0, description="scalar weighting of the loss for unconnected neurons")
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class TrainingConfig(BaseModel):
    time_units: int = Field(
        1,
        description="observation interval: activity data available every n steps. evolver unrolled n times during training.",
        json_schema_extra={"short_name": "tu"}
    )
    acquisition_mode: AcquisitionMode = Field(
        default_factory=AllTimePointsMode,
        description="data acquisition mode. controls which timesteps have observable data for each neuron.",
        json_schema_extra={"short_name": "acq"}
    )
    stimulus_frequency: StimulusFrequency = Field(
        StimulusFrequency.ALL,
        description="stimulus frequency. controls how stimulus is provided during evolution steps.",
        json_schema_extra={"short_name": "stim_freq"}
    )
    intermediate_loss_steps: list[int] = Field(
        default_factory=list,
        description="deprecated: intermediate steps feature has been removed. must be empty list.",
        json_schema_extra={"short_name": "ils"}
    )
    evolve_multiple_steps: int = Field(
        1,
        description="number of time_units multiples to evolve. loss applied at each multiple.",
        json_schema_extra={"short_name": "ems"}
    )
    epochs: int = Field(10, json_schema_extra={"short_name": "ep"})
    batch_size: int = Field(32, json_schema_extra={"short_name": "bs"})
    learning_rate: float = Field(1e-3, json_schema_extra={"short_name": "lr"})
    optimizer: str = Field("Adam", description="optimizer name from torch.optim", json_schema_extra={"short_name": "opt"})
    train_step: str = Field("train_step", description="compiled train step function")
    simulation_config: str | None = Field(
        None,
        description="name of simulation config (e.g., 'fly_N9_62_1'). mutually exclusive with training_data_path."
    )
    training_data_path: str | None = Field(
        None,
        description="absolute path to data directory. mutually exclusive with simulation_config. should point to x_list_0 directory."
    )
    column_to_model: str = "CALCIUM"
    use_tf32_matmul: bool = Field(
        False, description="enable fast tf32 multiplication on certain nvidia gpus"
    )
    seed: int = Field(42, json_schema_extra={"short_name": "seed"})
    data_split: DataSplit
    data_passes_per_epoch: int = 1
    diagnostics_freq_epochs: int = Field(
        0, description="run validation diagnostics every n epochs (0 = only at end of training)"
    )
    save_checkpoint_every_n_epochs: int = Field(
        10, description="save model checkpoint every n epochs (0 = disabled)"
    )
    save_best_checkpoint: bool = Field(
        True, description="save checkpoint when validation loss improves"
    )
    loss_function: str = Field(
        "mse_loss", description="loss function name from torch.nn.functional (e.g., 'mse_loss', 'huber_loss', 'l1_loss')"
    )
    grad_clip_max_norm: float = Field(
        0.0, description="max gradient norm for clipping (0 = disabled)", json_schema_extra={"short_name": "gc"}
    )
    reconstruction_warmup_epochs: int = Field(
        0, description="number of warmup epochs to train encoder/decoder only (reconstruction loss) before the main training loop. these are additional epochs, not counted in 'epochs'.", json_schema_extra={"short_name": "recon_wu"}
    )
    pretrain_stimulus_ae: bool = Field(
        False, description="enable stimulus autoencoder pretraining. pretrains stimulus encoder/decoder for reconstruction, then freezes encoder during main training.", json_schema_extra={"short_name": "psa"}
    )
    stimulus_ae: StimulusAETrainingConfig = Field(
        default_factory=StimulusAETrainingConfig, description="stimulus autoencoder training hyperparameters. used when pretrain_stimulus_ae is True."
    )
    unconnected_to_zero: UnconnectedToZeroConfig = Field(default_factory=UnconnectedToZeroConfig)
    early_stop_intervening_mse: bool = Field(
        False, description="enable early stopping based on max intervening mse metric (0 to tu-1)", json_schema_extra={"short_name": "es_int"}
    )
    early_stop_patience_epochs: int = Field(
        10, description="number of epochs to wait for 10% improvement in max intervening mse before stopping", json_schema_extra={"short_name": "es_patience"}
    )
    early_stop_min_divergence: int = Field(
        1000, description="minimum first divergence step required for early stopping to activate", json_schema_extra={"short_name": "es_min_div"}
    )
    z0_consistency_loss: float = Field(
        1.0, description="weight for z0 consistency loss. enforces that evolved latent matches z0_bank at subsequent windows.", json_schema_extra={"short_name": "z0c"}
    )
    encoder_consistency_loss: float = Field(
        1.0, description="weight for encoder consistency loss. enforces z ≈ encoder(decoder(z)) at tu boundaries.", json_schema_extra={"short_name": "enc_c"}
    )
    recon_loss_wt: float = Field(
        1.0, description="weight for reconstruction loss.", json_schema_extra={"short_name": "recon_w"}
    )
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @field_validator("optimizer")
    @classmethod
    def validate_optimizer(cls, v: str) -> str:
        if not hasattr(torch.optim, v):
            raise ValueError(f"unknown optimizer '{v}' in torch.optim")
        return v

    @field_validator("loss_function")
    @classmethod
    def validate_loss_function(cls, v: str) -> str:
        if not hasattr(torch.nn.functional, v):
            raise ValueError(f"unknown loss function '{v}' in torch.nn.functional")
        return v

    @model_validator(mode='after')
    def validate_training_config(self):
        if len(self.intermediate_loss_steps) > 0:
            raise ValueError("intermediate_loss_steps is deprecated and must be empty list")
        if self.evolve_multiple_steps < 1:
            raise ValueError("evolve_multiple_steps must be >= 1")

        # validate acquisition mode compatibility
        from LatentEvolution.acquisition import StaggeredRandomMode
        if isinstance(self.acquisition_mode, StaggeredRandomMode):
            if self.unconnected_to_zero.num_neurons > 0:
                raise ValueError(
                    "unconnected_to_zero augmentation is incompatible with staggered_random acquisition mode. "
                    "staggered mode observes neurons at different times, breaking the connectome assumption."
                )

        # validate stimulus frequency compatibility
        if self.time_units == 1 and self.stimulus_frequency != StimulusFrequency.ALL:
            raise ValueError(
                f"stimulus_frequency must be ALL when time_units=1. "
                f"got stimulus_frequency={self.stimulus_frequency.name}, time_units={self.time_units}"
            )

        return self


class CrossValidationConfig(BaseModel):
    """configuration for cross-dataset validation."""
    simulation_config: str
    name: str | None = None  # optional human-readable name
    data_split: DataSplit | None = None  # data split

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
