"""
EED (Encoder-Evolver-Decoder) building blocks.

This module contains the foundational PyTorch model components (MLP, Evolver)
and their associated Pydantic parameter configuration classes.
"""

import torch
import torch.nn as nn
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from LatentEvolution.training_config import TrainingConfig, ProfileConfig, CrossValidationConfig


# -------------------------------------------------------------------
# Pydantic Config Classes
# -------------------------------------------------------------------


class MLPParams(BaseModel):
    num_input_dims: int
    num_output_dims: int
    num_hidden_layers: int
    num_hidden_units: int
    activation: str = Field("ReLU", description="Activation function from torch.nn")
    use_batch_norm: bool = True
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @field_validator("activation")
    @classmethod
    def validate_activation(cls, v: str) -> str:
        if not hasattr(nn, v):
            raise ValueError(f"Unknown activation '{v}' in torch.nn")
        return v


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


class StimulusEncoderParams(BaseModel):
    num_input_dims: int
    num_hidden_layers: int
    num_hidden_units: int
    num_output_dims: int
    use_input_skips: bool = Field(False, description="If True, use MLPWithSkips instead of standard MLP")


class DecoderParams(BaseModel):
    num_hidden_units: int
    num_hidden_layers: int
    l1_reg_loss: float = 0.0
    use_input_skips: bool = Field(False, description="If True, use MLPWithSkips instead of standard MLP")
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# -------------------------------------------------------------------
# PyTorch Models
# -------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, params: MLPParams):
        super().__init__()
        self.layers = nn.ModuleList()
        input_dims = params.num_input_dims

        for _ in range(params.num_hidden_layers):
            self.layers.append(nn.Linear(input_dims, params.num_hidden_units))
            if params.use_batch_norm:
                self.layers.append(nn.BatchNorm1d(params.num_hidden_units))
            self.layers.append(getattr(nn, params.activation)())
            input_dims = params.num_hidden_units

        if params.num_hidden_layers:
            self.layers.append(
                nn.Linear(params.num_hidden_units, params.num_output_dims)
            )
        else:
            self.layers.append(nn.Linear(params.num_input_dims, params.num_output_dims))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MLPWithSkips(nn.Module):
    """
    MLP where each hidden layer receives a direct linear projection of the input.
    The input projection (without activation) is concatenated with the previous layer's output.
    """
    def __init__(self, params: MLPParams):
        super().__init__()
        self.num_hidden_layers = params.num_hidden_layers

        if self.num_hidden_layers == 0:
            # No hidden layers, just direct mapping
            self.output_layer = nn.Linear(params.num_input_dims, params.num_output_dims)
            return

        # Linear projections from input to each hidden layer (no activation)
        self.input_projections = nn.ModuleList()
        for _ in range(params.num_hidden_layers):
            self.input_projections.append(nn.Linear(params.num_input_dims, params.num_hidden_units))

        # Main hidden layers
        self.linear_layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        for i in range(params.num_hidden_layers):
            # First layer: concat(input, input_projection)
            # Subsequent layers: concat(prev_hidden, input_projection)
            if i == 0:
                layer_input_dims = params.num_input_dims + params.num_hidden_units
            else:
                layer_input_dims = params.num_hidden_units + params.num_hidden_units

            self.linear_layers.append(nn.Linear(layer_input_dims, params.num_hidden_units))
            self.activations.append(getattr(nn, params.activation)())

        # Output layer also gets concatenated input projection
        self.input_projection_final = nn.Linear(params.num_input_dims, params.num_hidden_units)
        self.output_layer = nn.Linear(params.num_hidden_units + params.num_hidden_units, params.num_output_dims)

    def forward(self, x):
        if self.num_hidden_layers == 0:
            return self.output_layer(x)

        original_input = x
        hidden = x

        # Process each hidden layer
        for i in range(self.num_hidden_layers):
            # Get linear projection of input (no activation)
            input_proj = self.input_projections[i](original_input)

            # Concatenate with current hidden state
            concat = torch.cat([hidden, input_proj], dim=-1)

            # Apply linear transformation and activation
            hidden = self.linear_layers[i](concat)
            hidden = self.activations[i](hidden)

        # Final output layer with input projection
        input_proj_final = self.input_projection_final(original_input)
        concat_final = torch.cat([hidden, input_proj_final], dim=-1)
        output = self.output_layer(concat_final)

        return output


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
