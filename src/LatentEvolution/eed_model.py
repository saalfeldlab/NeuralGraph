"""
EED (Encoder-Evolver-Decoder) building blocks.

This module contains the Evolver model and parameter configuration classes.
MLP building blocks live in mlp.py; stimulus encoder/decoder params live in
stimulus_ae_model.py.
"""

import torch
import torch.nn as nn
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from LatentEvolution.training_config import TrainingConfig, ProfileConfig, CrossValidationConfig
from LatentEvolution.mlp import MLP, MLPWithSkips, MLPParams
from LatentEvolution.stimulus_ae_model import StimulusEncoderParams


# -------------------------------------------------------------------
# Pydantic Config Classes
# -------------------------------------------------------------------


class EvolverParams(BaseModel):
    num_hidden_units: int
    num_hidden_layers: int
    l1_reg_loss: float = 0.0
    activation: str = Field(
        "Tanh", description="Activation function for evolver. Tanh recommended for stability in multi-step rollouts."
    )
    learnable_diagonal: bool = Field(
        False, description="DEPRECATED: This feature has been removed. Must be False."
    )
    use_input_skips: bool = Field(False, description="If True, use MLPWithSkips instead of standard MLP")
    use_mlp_with_matrix: bool = Field(
        False, description="DEPRECATED: This feature was not effective. Must be False."
    )
    time_units: int = Field(
        1, description="DEPRECATED: Use training.time_units instead. Kept for backwards compatibility."
    )
    zero_init: bool = Field(
        True, description="If True, zero-initialize final layer so evolver starts as identity (z_{t+1} = z_t). Provides stability but may slow dynamics learning."
    )
    tv_reg_loss: float = Field(
        0.0, description="total variation regularization on evolver updates. penalizes ||Δz|| to stabilize dynamics and prevent explosive rollouts. typical range: 1e-5 to 1e-3."
    )
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @field_validator("activation")
    @classmethod
    def validate_activation(cls, v: str) -> str:
        if not hasattr(nn, v):
            raise ValueError(f"Unknown activation '{v}' in torch.nn")
        return v

    @field_validator("learnable_diagonal")
    @classmethod
    def validate_learnable_diagonal(cls, v: bool) -> bool:
        if v:
            raise ValueError("`learnable_diagonal` is deprecated.")
        return False

    @field_validator("use_mlp_with_matrix")
    @classmethod
    def validate_use_mlp_with_matrix(cls, v: bool) -> bool:
        if v:
            raise ValueError("`use_mlp_with_matrix` is deprecated: this feature was not effective.")
        return False


class EncoderParams(BaseModel):
    num_hidden_units: int
    num_hidden_layers: int
    l1_reg_loss: float = 0.0
    use_input_skips: bool = Field(False, description="If True, use MLPWithSkips instead of standard MLP")
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class DecoderParams(BaseModel):
    num_hidden_units: int
    num_hidden_layers: int
    l1_reg_loss: float = 0.0
    use_input_skips: bool = Field(False, description="If True, use MLPWithSkips instead of standard MLP")
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# -------------------------------------------------------------------
# PyTorch Models
# -------------------------------------------------------------------


class Evolver(nn.Module):
    def __init__(self, latent_dims: int, stim_dims: int, evolver_params: EvolverParams, use_batch_norm: bool = True, activation: str = "ReLU"):
        super().__init__()

        evolver_activation = evolver_params.activation

        # Use MLPWithSkips if flag is set, similar to encoder/decoder
        evolver_cls = MLPWithSkips if evolver_params.use_input_skips else MLP

        self.evolver = evolver_cls(
            MLPParams(
                num_input_dims=latent_dims + stim_dims,
                num_hidden_layers=evolver_params.num_hidden_layers,
                num_hidden_units=evolver_params.num_hidden_units,
                num_output_dims=latent_dims,
                activation=evolver_activation,
                use_batch_norm=use_batch_norm,
            )
        )

        # optionally zero-init final layer so evolver starts as identity (z_{t+1} = z_t)
        if evolver_params.zero_init:
            if evolver_params.use_input_skips:
                nn.init.zeros_(self.evolver.output_layer.weight)
                nn.init.zeros_(self.evolver.output_layer.bias)
            else:
                nn.init.zeros_(self.evolver.layers[-1].weight)
                nn.init.zeros_(self.evolver.layers[-1].bias)

    def forward(self, proj_t, proj_stim_t):
        """Evolve one time step in latent space."""
        proj_t_next = proj_t + self.evolver(torch.cat([proj_t, proj_stim_t], dim=1))
        return proj_t_next


# -------------------------------------------------------------------
# Model Configuration
# -------------------------------------------------------------------


class ModelParams(BaseModel):
    latent_dims: int = Field(..., json_schema_extra={"short_name": "ld"})
    num_neurons: int
    use_batch_norm: bool = True
    activation: str = Field("ReLU", description="activation function from torch.nn")
    encoder_params: EncoderParams
    decoder_params: DecoderParams
    evolver_params: EvolverParams
    stimulus_encoder_params: StimulusEncoderParams
    training: TrainingConfig
    profiling: ProfileConfig | None = Field(
        None, description="optional profiler configuration to generate chrome traces for performance analysis"
    )
    cross_validation_configs: list[CrossValidationConfig] = Field(
        default_factory=lambda: [CrossValidationConfig(simulation_config="fly_N9_62_0")],
        description="list of datasets to validate on after training"
    )

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @field_validator("activation")
    @classmethod
    def validate_activation(cls, v: str) -> str:
        if not hasattr(nn, v):
            raise ValueError(f"unknown activation '{v}' in torch.nn")
        return v

    @model_validator(mode='after')
    def validate_encoder_decoder_symmetry(self):
        """ensure encoder and decoder have symmetric mlp parameters."""
        if self.encoder_params.num_hidden_units != self.decoder_params.num_hidden_units:
            raise ValueError(
                f"encoder and decoder must have the same num_hidden_units. "
                f"got encoder={self.encoder_params.num_hidden_units}, decoder={self.decoder_params.num_hidden_units}"
            )
        if self.encoder_params.num_hidden_layers != self.decoder_params.num_hidden_layers:
            raise ValueError(
                f"encoder and decoder must have the same num_hidden_layers. "
                f"got encoder={self.encoder_params.num_hidden_layers}, decoder={self.decoder_params.num_hidden_layers}"
            )
        if self.encoder_params.use_input_skips != self.decoder_params.use_input_skips:
            raise ValueError(
                f"encoder and decoder must have the same use_input_skips setting. "
                f"got encoder={self.encoder_params.use_input_skips}, decoder={self.decoder_params.use_input_skips}"
            )

        return self

    def flatten(self, sep: str = ".") -> dict[str, int | float | str | bool]:
        """
        flatten the modelparams into a single-level dictionary.

        args:
            sep: separator to use for nested keys (default: ".")

        returns:
            a flat dictionary with nested keys joined by the separator.

        example:
            >>> params.flatten()
            {'latent_dims': 10, 'encoder_params.num_hidden_units': 64, ...}
        """
        def _flatten_dict(
            d: dict[str, int | float | str | bool | dict],
            parent_key: str = "",
        ) -> dict[str, int | float | str | bool]:
            items: list[tuple[str, int | float | str | bool]] = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(_flatten_dict(v, new_key).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        return _flatten_dict(self.model_dump())
