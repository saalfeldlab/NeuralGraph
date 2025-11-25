"""
EED (Encoder-Evolver-Decoder) building blocks.

This module contains the foundational PyTorch model components (MLP, Evolver)
and their associated Pydantic parameter configuration classes.
"""

import torch
import torch.nn as nn
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


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
    time_units: int
    num_hidden_units: int
    num_hidden_layers: int
    l1_reg_loss: float = 0.0
    learnable_diagonal: bool = Field(
        False, description="DEPRECATED: This feature has been removed. Must be False."
    )
    use_input_skips: bool = Field(False, description="If True, use MLPWithSkips instead of standard MLP")
    use_mlp_with_matrix: bool = Field(
        False, description="If True, use architecture: x = MLP(concat[x, Ax]) where A is a learnable matrix"
    )
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @field_validator("learnable_diagonal")
    @classmethod
    def validate_learnable_diagonal(cls, v: bool) -> bool:
        if v:
            raise ValueError("`learnable_diagonal` is deprecated.")
        return False

    @model_validator(mode='after')
    def validate_mutually_exclusive_architectures(self):
        """Ensure use_input_skips and use_mlp_with_matrix are mutually exclusive."""
        if self.use_input_skips and self.use_mlp_with_matrix:
            raise ValueError(
                "use_input_skips and use_mlp_with_matrix are mutually exclusive architecture choices. "
                "Only one can be True at a time."
            )
        return self


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
        self.time_units = evolver_params.time_units
        self.use_mlp_with_matrix = evolver_params.use_mlp_with_matrix
        dim = latent_dims + stim_dims

        # Learnable matrix for matrix concatenation architecture
        if self.use_mlp_with_matrix:
            self.matrix = nn.Parameter(torch.randn(dim, dim) * 0.01)

        # Use MLPWithSkips if flag is set, similar to encoder/decoder
        evolver_cls = MLPWithSkips if evolver_params.use_input_skips else MLP

        # Adjust input dimensions based on architecture
        mlp_input_dims = 2 * dim if self.use_mlp_with_matrix else dim

        self.evolver = evolver_cls(
            MLPParams(
                num_input_dims=mlp_input_dims,
                num_hidden_layers=evolver_params.num_hidden_layers,
                num_hidden_units=evolver_params.num_hidden_units,
                num_output_dims=dim,
                activation=activation,
                use_batch_norm=use_batch_norm,
            )
        )

    def forward(self, x):
        for _ in range(self.time_units):
            if self.use_mlp_with_matrix:
                # New architecture: x = MLP(concat[x, Ax])
                Ax = x @ self.matrix
                concat = torch.cat([x, Ax], dim=-1)
                x = self.evolver(concat)
            else:
                # Standard residual: x = x + MLP(x)
                x = self.evolver(x)
        return x
