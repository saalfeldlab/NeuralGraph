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
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class StimulusEncoderParams(BaseModel):
    num_input_dims: int
    num_hidden_layers: int
    num_hidden_units: int
    num_output_dims: int


class DecoderParams(BaseModel):
    num_hidden_units: int
    num_hidden_layers: int
    l1_reg_loss: float = 0.0
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
