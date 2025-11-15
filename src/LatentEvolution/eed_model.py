"""
EED (Encoder-Evolver-Decoder) building blocks.

This module contains the foundational PyTorch model components (MLP, Evolver)
and their associated Pydantic parameter configuration classes.
"""

import torch
import torch.nn as nn
from pydantic import BaseModel, Field, field_validator, ConfigDict


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
        False, description="Use learnable diagonal matrix for residual (x -> Ax + mlp(x) instead of x + mlp(x))"
    )
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


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
        self.use_learnable_diagonal = evolver_params.learnable_diagonal
        dim = latent_dims + stim_dims

        # Learnable diagonal matrix for residual connection
        if self.use_learnable_diagonal:
            self.diagonal = nn.Parameter(torch.ones(dim))

        self.evolver = MLP(
            MLPParams(
                num_input_dims=dim,
                num_hidden_layers=evolver_params.num_hidden_layers,
                num_hidden_units=evolver_params.num_hidden_units,
                num_output_dims=dim,
                activation=activation,
                use_batch_norm=use_batch_norm,
            )
        )

    def forward(self, x):
        for _ in range(self.time_units):
            if self.use_learnable_diagonal:
                x = self.diagonal * x + self.evolver(x)
            else:
                x = x + self.evolver(x)
        return x
