"""zapbench EED (Encoder-Evolver-Decoder) model.

architecture:
  x(t) [N] --> encoder --> z(t) [L]
                              |
                    evolver: z(t+1) = z(t) + f(z(t))
                              |
                    (repeat for fitting_window steps)
                              |
                        decoder: z(t+k) --> x_hat(t+k) [N]
"""

import torch
import torch.nn as nn

from LatentEvolution.mlp import MLPWithSkips, MLPParams
from LatentEvolution.zapbench_config import ModelConfig


class EEDModel(nn.Module):
    """encoder-evolver-decoder model for zapbench neural activity prediction."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # encoder: (N,) -> (L,)
        self.encoder = MLPWithSkips(MLPParams(
            num_input_dims=config.num_neurons,
            num_output_dims=config.latent_dims,
            num_hidden_layers=config.encoder_decoder.hidden_layers,
            num_hidden_units=config.encoder_decoder.hidden_units,
            activation=config.encoder_decoder.activation,
            use_batch_norm=False,
        ))

        # decoder: (L,) -> (N,)
        self.decoder = MLPWithSkips(MLPParams(
            num_input_dims=config.latent_dims,
            num_output_dims=config.num_neurons,
            num_hidden_layers=config.encoder_decoder.hidden_layers,
            num_hidden_units=config.encoder_decoder.hidden_units,
            activation=config.encoder_decoder.activation,
            use_batch_norm=False,
        ))

        # evolver: (L,) -> (L,) with residual connection
        self.evolver_mlp = MLPWithSkips(MLPParams(
            num_input_dims=config.latent_dims,
            num_output_dims=config.latent_dims,
            num_hidden_layers=config.evolver.hidden_layers,
            num_hidden_units=config.evolver.hidden_units,
            activation=config.evolver.activation,
            use_batch_norm=False,
        ))

        # optionally zero-init evolver output so it starts as identity
        if config.evolver.zero_init:
            self._zero_init_evolver()

    def _zero_init_evolver(self):
        """zero-initialize evolver final layer so z_{t+1} = z_t initially."""
        nn.init.zeros_(self.evolver_mlp.output_layer.weight)
        nn.init.zeros_(self.evolver_mlp.output_layer.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """encode neural activity to latent space.

        args:
            x: (B, N) neural activity

        returns:
            z: (B, L) latent representation
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """decode latent representation to neural activity.

        args:
            z: (B, L) latent representation

        returns:
            x: (B, N) predicted neural activity
        """
        return self.decoder(z)

    def evolve(self, z: torch.Tensor) -> torch.Tensor:
        """evolve latent state one time step.

        args:
            z: (B, L) current latent state

        returns:
            z_next: (B, L) next latent state
        """
        return z + self.evolver_mlp(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """encode, evolve one step, decode.

        args:
            x: (B, N) neural activity at time t

        returns:
            x_next: (B, N) predicted neural activity at time t+1
        """
        z = self.encode(x)
        z_next = self.evolve(z)
        x_next = self.decode(z_next)
        return x_next


