"""
stimulus autoencoder model and pretraining.

depends only on mlp.py (and standard libraries).
dependency direction: eed_model.py and training_config.py import from here.
"""

from datetime import datetime
from enum import Enum, auto
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel, Field, ConfigDict, field_validator

from LatentEvolution.mlp import MLP, MLPWithSkips, MLPParams


# -------------------------------------------------------------------
# config classes
# -------------------------------------------------------------------


class StimulusEncoderParams(BaseModel):
    num_input_dims: int
    num_hidden_layers: int
    num_hidden_units: int
    num_output_dims: int
    use_input_skips: bool = Field(False, description="If True, use MLPWithSkips instead of standard MLP")


class StimulusAETrainingConfig(BaseModel):
    """training config for stimulus autoencoder pretraining."""
    epochs: int = Field(10, description="number of pretraining epochs")
    batch_size: int = Field(256, description="batch size for pretraining")
    learning_rate: float = Field(1e-4, description="learning rate for pretraining")
    optimizer: str = Field("Adam", description="optimizer name from torch.optim")
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @field_validator("optimizer")
    @classmethod
    def validate_optimizer(cls, v: str) -> str:
        if not hasattr(torch.optim, v):
            raise ValueError(f"unknown optimizer '{v}' in torch.optim")
        return v


# -------------------------------------------------------------------
# loss type
# -------------------------------------------------------------------


class StimulusAELossType(Enum):
    RECON = auto()


# -------------------------------------------------------------------
# model
# -------------------------------------------------------------------


class StimulusAE(nn.Module):
    """stimulus autoencoder: symmetric encoder + decoder for stimulus reconstruction."""

    def __init__(
        self,
        encoder_params: StimulusEncoderParams,
        activation: str = "ReLU",
    ):
        super().__init__()

        mlp_cls = MLPWithSkips if encoder_params.use_input_skips else MLP
        self.encoder = mlp_cls(
            MLPParams(
                num_input_dims=encoder_params.num_input_dims,
                num_hidden_units=encoder_params.num_hidden_units,
                num_hidden_layers=encoder_params.num_hidden_layers,
                num_output_dims=encoder_params.num_output_dims,
                use_batch_norm=False,
                activation=activation,
            )
        )

        # decoder is symmetric: same architecture, swapped input/output dims
        self.decoder = mlp_cls(
            MLPParams(
                num_input_dims=encoder_params.num_output_dims,
                num_hidden_units=encoder_params.num_hidden_units,
                num_hidden_layers=encoder_params.num_hidden_layers,
                num_output_dims=encoder_params.num_input_dims,
                use_batch_norm=False,
                activation=activation,
            )
        )

    def forward(self, stim):
        latent = self.encoder(stim)
        recon = self.decoder(latent)
        return recon


# -------------------------------------------------------------------
# train step (compiled)
# -------------------------------------------------------------------


def _train_step_stimulus_ae_nocompile(
    model: StimulusAE,
    stim_batch: torch.Tensor,
) -> dict[StimulusAELossType, torch.Tensor]:
    """single train step for stimulus autoencoder."""
    recon = model(stim_batch)
    recon_loss = torch.nn.functional.mse_loss(recon, stim_batch)
    return {
        StimulusAELossType.RECON: recon_loss,
    }


train_step_stimulus_ae = torch.compile(
    _train_step_stimulus_ae_nocompile, fullgraph=True, mode="reduce-overhead"
)


# -------------------------------------------------------------------
# training loop
# -------------------------------------------------------------------


def pretrain_stimulus_ae(
    stim_np: np.ndarray,
    encoder_params: StimulusEncoderParams,
    train_cfg: StimulusAETrainingConfig,
    activation: str,
    device: torch.device,
    run_dir: Path,
    writer=None,
) -> StimulusAE:
    """pretrain stimulus autoencoder on stimulus data.

    loads the numpy stimulus array onto gpu, trains the autoencoder,
    saves a checkpoint, and returns the trained model.
    gpu stimulus tensor is freed when this function returns.

    args:
        stim_np: numpy stimulus array, shape (T, stim_dims)
        encoder_params: stimulus encoder config
        train_cfg: training hyperparameters
        activation: activation function name
        device: torch device
        run_dir: directory to save checkpoint
        writer: optional tensorboard SummaryWriter

    returns:
        trained StimulusAE model
    """
    stim_data = torch.from_numpy(stim_np).to(device)
    print(f"stimulus ae training data: {stim_data.shape}")

    num_timesteps = stim_data.shape[0]

    model = StimulusAE(encoder_params, activation).to(device)
    print(f"stimulus ae parameters: {sum(p.numel() for p in model.parameters()):,}")
    model.train()

    OptimizerClass = getattr(torch.optim, train_cfg.optimizer)
    optimizer = OptimizerClass(model.parameters(), lr=train_cfg.learning_rate)

    batches_per_epoch = num_timesteps // train_cfg.batch_size

    for epoch in range(train_cfg.epochs):
        epoch_start = datetime.now()
        epoch_loss = 0.0
        num_batches = 0

        for _ in range(batches_per_epoch):
            optimizer.zero_grad()
            batch_idx = torch.randint(0, num_timesteps, (train_cfg.batch_size,), device=device)
            stim_batch = stim_data[batch_idx]
            losses = train_step_stimulus_ae(model, stim_batch)
            losses[StimulusAELossType.RECON].backward()
            optimizer.step()
            epoch_loss += losses[StimulusAELossType.RECON].item()
            num_batches += 1

        mean_loss = epoch_loss / max(num_batches, 1)
        epoch_duration = (datetime.now() - epoch_start).total_seconds()
        print(f"stim_ae {epoch+1}/{train_cfg.epochs} | recon: {mean_loss:.4e} | duration: {epoch_duration:.2f}s")

        if writer is not None:
            writer.add_scalar("StimulusAE/recon", mean_loss, epoch)
            writer.add_scalar("StimulusAE/epoch_duration", epoch_duration, epoch)

    # save checkpoint
    stim_ae_path = run_dir / "stimulus_ae.pt"
    torch.save(model.state_dict(), stim_ae_path)
    print(f"saved stimulus ae checkpoint to {stim_ae_path}")

    return model


# -------------------------------------------------------------------
# standalone config + cli entry point
# -------------------------------------------------------------------


class StimulusAEConfig(BaseModel):
    """config for running stimulus ae pretraining."""
    encoder_params: StimulusEncoderParams
    training: StimulusAETrainingConfig
    activation: str = Field("ReLU", description="activation function from torch.nn")
    data_path: str = Field(..., description="path to zarr data directory")
    train_start: int = Field(..., description="start timestep for training data")
    train_end: int = Field(..., description="end timestep for training data")
    seed: int = Field(42, description="random seed")
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def _load_config_from_yaml(yaml_path: str) -> StimulusAEConfig:
    """load stimulus ae config from a latent model yaml file.

    extracts stimulus_encoder_params, stimulus_ae training config,
    activation, simulation_config, data_split, and seed from the
    full model yaml (e.g., latent_20step.yaml).
    """
    import yaml

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # resolve data path from simulation_config
    sim_config = data["training"]["simulation_config"]
    data_path = f"graphs_data/fly/{sim_config}/x_list_0"

    ds = data["training"]["data_split"]

    stim_ae_cfg = data["training"].get("stimulus_ae")
    if stim_ae_cfg is None:
        raise ValueError(
            f"no training.stimulus_ae section found in {yaml_path}. "
            "add stimulus_ae config under training to use standalone pretraining."
        )

    return StimulusAEConfig(
        encoder_params=StimulusEncoderParams(**data["stimulus_encoder_params"]),
        training=StimulusAETrainingConfig(**stim_ae_cfg),
        activation=data.get("activation", "ReLU"),
        data_path=data_path,
        train_start=ds["train_start"],
        train_end=ds["train_end"],
        seed=data["training"].get("seed", 42),
    )


if __name__ == "__main__":
    import sys
    import tyro

    from LatentEvolution.training_utils import seed_everything, get_device

    msg = """usage: python stimulus_ae_model.py <default_config> [overrides...]

arguments:
  default_config  base config file (e.g., latent_20step.yaml)

example:
  python stimulus_ae_model.py latent_20step.yaml
  python stimulus_ae_model.py latent_20step.yaml --training.epochs 20

to view available overrides:
  python stimulus_ae_model.py latent_20step.yaml --help"""

    if len(sys.argv) < 2:
        print(msg)
        sys.exit(1)

    default_yaml = sys.argv[1]
    default_path = Path(__file__).resolve().parent / default_yaml
    if not default_path.exists():
        print(f"error: config file not found: {default_path}")
        sys.exit(1)

    default_cfg = _load_config_from_yaml(str(default_path))
    tyro_args = sys.argv[2:]
    cfg = tyro.cli(StimulusAEConfig, default=default_cfg, args=tyro_args)

    seed_everything(cfg.seed)
    device = get_device()

    from LatentEvolution.load_flyvis import FlyVisSim, load_column_slice

    stim_np = load_column_slice(
        cfg.data_path,
        FlyVisSim.STIMULUS.value,
        cfg.train_start,
        cfg.train_end,
        neuron_limit=cfg.encoder_params.num_input_dims,
    )

    run_dir = Path("runs/stimulus_ae")
    run_dir.mkdir(parents=True, exist_ok=True)

    stim_ae = pretrain_stimulus_ae(
        stim_np=stim_np,
        encoder_params=cfg.encoder_params,
        train_cfg=cfg.training,
        activation=cfg.activation,
        device=device,
        run_dir=run_dir,
    )
    print("done")
