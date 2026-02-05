"""
Latent staggered model: decode-only architecture with learned initial latents.

Requires staggered acquisition mode where different neurons are observed at
different phases within each time_units cycle.

This model doesn't have a memory bank of latent states. Instead,
we interpolate staggered activities to create a time-aligned activity. This
allows us to reuse the LatentModel for aligned activity.
"""

from pathlib import Path
from datetime import datetime
from enum import Enum, auto
import gc
import sys
import re
import signal

import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import yaml
import tyro

from LatentEvolution.hparam_paths import create_run_directory, get_git_commit_hash
from LatentEvolution.interpolate_staggered import interpolate_staggered_compiled
from LatentEvolution.training_utils import (
    LossAccumulator,
    seed_everything,
    get_device,
)
from LatentEvolution.chunk_streaming import calculate_chunk_params, generate_random_chunks
from LatentEvolution.acquisition import (
    compute_neuron_phases,
)
from LatentEvolution.latent import load_dataset, load_val_only, LatentModel, ModelParams
from LatentEvolution.stimulus_ae_model import pretrain_stimulus_ae
from LatentEvolution.load_flyvis import FlyVisSim, load_column_slice
from LatentEvolution.pipeline_chunk_loader import PipelineProfiler
from LatentEvolution.diagnostics_stag import run_validation_diagnostics_interp


# -------------------------------------------------------------------
# Loss Types
# -------------------------------------------------------------------


class LossType(Enum):
    """loss component types for staggered model."""
    TOTAL = auto()


# -------------------------------------------------------------------
# Training Step
# -------------------------------------------------------------------


@torch.compile(fullgraph=True, mode="reduce-overhead")
def train_step(
    model: LatentModel,
    train_data: torch.Tensor,
    train_stim: torch.Tensor,
    batch_start_times: torch.Tensor,
    neuron_phases: torch.Tensor,
    cfg: ModelParams,
) -> dict[LossType, torch.Tensor]:
    """
    training step for staggered model.

    args:
        model: the staggered model (decoder, evolver, stim encoder)
        train_data: chunk data (chunk_timesteps, num_neurons)
        train_stim: chunk stimulus (chunk_timesteps, stim_dims)
        batch_start_times: (batch_size,) batch start times
        neuron_phases: (num_neurons,) when in cycle is neuron observed
        cfg: model configuration

    returns:
        dict of loss components
    """
    device = train_data.device

    losses: dict[LossType, torch.Tensor] = {}

    tu = cfg.training.time_units
    num_multiples = cfg.training.evolve_multiple_steps
    total_steps = tu * num_multiples


    # batch_size, num_neurons = observation_indices.shape
    # neuron_indices = torch.arange(num_neurons, device=device).unsqueeze(0).expand(batch_size, num_neurons)

    x_t_interp = train_data[batch_start_times, :] # (B, N)
    proj_t = model.encoder(x_t_interp) # (B, L)

    batch_start_phase = (batch_start_times.unsqueeze(1) - neuron_phases.unsqueeze(0)) % tu # B x N

    # stimulus for the full window
    stim_indices = batch_start_times.unsqueeze(0) + torch.arange(total_steps, device=device).unsqueeze(1)
    stim_t = train_stim[stim_indices, :]  # (total_steps, batch_size, stim_dims)
    dim_stim = train_stim.shape[1]
    dim_stim_latent = cfg.stimulus_encoder_params.num_output_dims

    # encode all stimulus: (total_steps, batch_size, stim_latent_dims)
    proj_stim_t = model.stimulus_encoder(stim_t.reshape((-1, dim_stim))).reshape((total_steps, -1, dim_stim_latent))

    loss = torch.tensor(0.0, device=device)
    for t in range(total_steps):
        x_t_recon = model.decoder(proj_t)
        # evolve by t time steps and compute error
        error = x_t_recon - train_data[batch_start_times+t, :]
        # train_data is only valid at observation points specified by neuron_phases
        loss += torch.pow(torch.where((batch_start_phase + t) % tu == 0, error, 0.0), 2).mean()
        proj_t = model.evolver(proj_t, proj_stim_t[t])

    losses[LossType.TOTAL] = loss

    return losses


# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------


def train(cfg: ModelParams, run_dir: Path):
    """training loop for staggered latent model."""
    seed_everything(cfg.training.seed)

    # --- Signal handling for graceful termination ---
    terminate_flag = {"value": False}

    def handle_sigusr2(signum, frame):
        terminate_flag["value"] = True
        print("\nSIGUSR2 received - will terminate after current epoch")

    signal.signal(signal.SIGUSR2, handle_sigusr2)

    commit_hash = get_git_commit_hash()

    log_path = run_dir / "stdout.log"
    err_path = run_dir / "stderr.log"
    with open(log_path, "w", buffering=1) as log_file, open(err_path, "w", buffering=1) as err_log:
        sys.stdout = log_file
        sys.stderr = err_log

        print(f"run directory: {run_dir.resolve()}")

        # save config
        config_path = run_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(cfg.model_dump(mode='json'), f, sort_keys=False, indent=2)
        print(f"saved config to {config_path}")

        # device setup
        device = get_device()
        if cfg.training.use_tf32_matmul and device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            print("tf32 matmul precision: enabled ('high')")

        # model
        model = LatentModel(cfg).to(device)
        print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")
        model.train()

        # pipeline profiler for chrome tracing (only profile first N epochs to limit file size)
        profile_first_n_epochs = 5
        profiler = PipelineProfiler()
        profiler.start()

        # load data (reuse from latent.py)
        dt = cfg.training.time_units
        chunk_loader, val_data, val_stim, _neuron_data, train_total_timesteps = load_dataset(
            simulation_config=cfg.training.simulation_config,
            column_to_model=cfg.training.column_to_model,
            data_split=cfg.training.data_split,
            num_input_dims=cfg.stimulus_encoder_params.num_input_dims,
            device=device,
            training_data_path=cfg.training.training_data_path,
            gpu_prefetch=2,  # double buffer for cpu->gpu transfer overlap
            profiler=profiler,
        )
        print(f"training data: {train_total_timesteps} timesteps (pipeline chunked streaming)")

        # optimizer
        OptimizerClass = getattr(torch.optim, cfg.training.optimizer)
        optimizer = OptimizerClass(
            [
                {'params': model.parameters()},
            ],
            lr=cfg.training.learning_rate
        )

        # tensorboard
        writer = SummaryWriter(log_dir=run_dir)
        print(f"tensorboard --logdir={run_dir}")

        # chunking
        chunk_size = 65536
        chunks_per_epoch, batches_per_chunk, batches_per_epoch = calculate_chunk_params(
            total_timesteps=train_total_timesteps,
            chunk_size=chunk_size,
            batch_size=cfg.training.batch_size,
            data_passes_per_epoch=cfg.training.data_passes_per_epoch,
        )
        print(f"chunking: {chunks_per_epoch} chunks/epoch, {batches_per_chunk} batches/chunk, {batches_per_epoch} total batches/epoch")

        # neuron phases (required for staggered acquisition)
        neuron_phases = compute_neuron_phases(
            num_neurons=cfg.num_neurons,
            time_units=dt,
            acquisition_mode=cfg.training.acquisition_mode,
            device=device,
        )
        assert neuron_phases is not None, "staggered_random acquisition mode requires neuron phases"
        torch.save(neuron_phases, run_dir / "neuron_phases.pt")
        print(f"acquisition mode: staggered_random, phases for {cfg.num_neurons} neurons")

        # --- load cross-validation datasets ---
        cv_datasets: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for cv_config in cfg.cross_validation_configs:
            cv_name = cv_config.name or cv_config.simulation_config
            data_split = cv_config.data_split or cfg.training.data_split
            cv_val_data, cv_val_stim = load_val_only(
                simulation_config=cv_config.simulation_config,
                column_to_model=cfg.training.column_to_model,
                data_split=data_split,
                num_input_dims=cfg.stimulus_encoder_params.num_input_dims,
                device=device,
            )
            cv_datasets[cv_name] = (cv_val_data, cv_val_stim)
            print(f"loaded cross-validation dataset: {cv_name} (val shape: {cv_val_data.shape})")

        # --- Stimulus autoencoder pretraining ---
        if cfg.training.pretrain_stimulus_ae:
            print("\n=== stimulus autoencoder pretraining ===")
            if cfg.training.training_data_path is not None:
                stim_data_path = cfg.training.training_data_path
            else:
                stim_data_path = f"graphs_data/fly/{cfg.training.simulation_config}/x_list_0"

            stim_np = load_column_slice(
                stim_data_path,
                FlyVisSim.STIMULUS.value,
                cfg.training.data_split.train_start,
                cfg.training.data_split.train_end,
                neuron_limit=cfg.stimulus_encoder_params.num_input_dims,
            )

            stim_ae = pretrain_stimulus_ae(
                stim_np=stim_np,
                encoder_params=cfg.stimulus_encoder_params,
                train_cfg=cfg.training.stimulus_ae,
                activation=cfg.activation,
                device=device,
                run_dir=run_dir,
                writer=writer,
            )
            del stim_np
            gc.collect()

            # copy pretrained encoder weights into model and freeze
            model.stimulus_encoder.load_state_dict(stim_ae.encoder.state_dict())
            model.stimulus_encoder.requires_grad_(False)
            print("=== stimulus encoder frozen ===\n")

        train_step_fn = globals()[cfg.training.train_step]
        total_steps = dt * cfg.training.evolve_multiple_steps

        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        training_start = datetime.now()
        epoch_durations = []

        # epoch loop
        for epoch in range(cfg.training.epochs):
          # stop profiler after first N epochs and save trace immediately
          if epoch == profile_first_n_epochs and profiler.is_enabled():
              profiler.stop()
              trace_path = run_dir / "pipeline_trace.json"
              profiler.save(trace_path)
              print(f"profiler stopped after epoch {epoch}, saved to {trace_path}")
              profiler.print_stats()

          with profiler.event("epoch", "training", thread="main", epoch=epoch):
            epoch_start = datetime.now()
            losses = LossAccumulator(LossType)

            chunks = generate_random_chunks(
                total_timesteps=train_total_timesteps, chunk_size=65536,
                num_chunks=chunks_per_epoch, time_units=dt,
            )
            chunk_loader.start_epoch(chunks)

            for chunk_idx in range(chunks_per_epoch):
              with profiler.event("chunk", "pipeline", thread="main", chunk=chunk_idx):
                chunk_start, (chunk_data, chunk_stim) = chunk_loader.get_next_chunk()
                if chunk_data is None or chunk_start is None or chunk_stim is None:
                    break
                chunk_size = chunk_data.shape[0]
                # interpolate staggered acquisition data
                chunk_interp = interpolate_staggered_compiled(
                    chunk_data, neuron_phases, cfg.training.time_units
                ).clone()

                with profiler.event("train", "compute", thread="main"):
                    for _ in range(batches_per_chunk):
                        optimizer.zero_grad()

                        # pick random starts - not aligned with tu
                        batch_start_indices = torch.randint(
                            0, chunk_size - total_steps, size=(cfg.training.batch_size,), device=device
                        )

                        loss_dict = train_step_fn(
                            model, chunk_interp, chunk_stim, batch_start_indices, neuron_phases, cfg
                        )
                        loss_dict[LossType.TOTAL].backward()

                        optimizer.step()
                        losses.accumulate(loss_dict)

            mean_losses = losses.mean()
            epoch_end = datetime.now()
            epoch_duration = (epoch_end - epoch_start).total_seconds()
            epoch_durations.append(epoch_duration)
            total_elapsed = (epoch_end - training_start).total_seconds()

            print(
                f"epoch {epoch+1}/{cfg.training.epochs} | "
                f"loss: {mean_losses[LossType.TOTAL]:.4e} | "
                f"duration: {epoch_duration:.2f}s"
            )

            for loss_type, loss_value in mean_losses.items():
                writer.add_scalar(f"Loss/train_{loss_type.name.lower()}", loss_value, epoch)
            writer.add_scalar("Time/epoch_duration", epoch_duration, epoch)
            writer.add_scalar("Time/total_elapsed", total_elapsed, epoch)

            # run validation diagnostics
            if cfg.training.diagnostics_freq_epochs > 0 and (epoch + 1) % cfg.training.diagnostics_freq_epochs == 0:
                model.eval()

                # main validation dataset
                diag_start = datetime.now()
                val_metrics, val_figures = run_validation_diagnostics_interp(
                    val_data=val_data,
                    val_stim=val_stim,
                    model=model,
                    cfg=cfg,
                    epoch=epoch,
                    neuron_phases=neuron_phases,
                )
                diag_duration = (datetime.now() - diag_start).total_seconds()

                for metric_name, metric_value in val_metrics.items():
                    writer.add_scalar(f"Val/{metric_name}", metric_value, epoch)
                for fig_name, fig in val_figures.items():
                    writer.add_figure(f"Val/{fig_name}", fig, epoch)
                    plt.close(fig)

                writer.add_scalar("Time/diagnostics_duration", diag_duration, epoch)
                print(f"  validation diagnostics: {diag_duration:.1f}s")

                # cross-validation datasets
                for cv_name, (cv_val_data, cv_val_stim) in cv_datasets.items():
                    cv_start = datetime.now()
                    cv_metrics, cv_figures = run_validation_diagnostics_interp(
                        val_data=cv_val_data,
                        val_stim=cv_val_stim,
                        model=model,
                        cfg=cfg,
                        epoch=epoch,
                        neuron_phases=neuron_phases,
                    )
                    cv_duration = (datetime.now() - cv_start).total_seconds()

                    for metric_name, metric_value in cv_metrics.items():
                        writer.add_scalar(f"CrossVal/{cv_name}/{metric_name}", metric_value, epoch)
                    for fig_name, fig in cv_figures.items():
                        writer.add_figure(f"CrossVal/{cv_name}/{fig_name}", fig, epoch)
                        plt.close(fig)

                    print(f"  cv/{cv_name} diagnostics: {cv_duration:.1f}s")

                model.train()

            if cfg.training.save_checkpoint_every_n_epochs > 0 and (epoch + 1) % cfg.training.save_checkpoint_every_n_epochs == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1:04d}.pt"
                torch.save({
                    'model': model.state_dict()
                }, checkpoint_path)
                print(f"  -> saved checkpoint at epoch {epoch+1}")

            # save latest checkpoint (overwrite each epoch)
            latest_checkpoint_path = checkpoint_dir / "checkpoint_latest.pt"
            torch.save({
                'model': model.state_dict()
            }, latest_checkpoint_path)

            # check for graceful termination signal
            if terminate_flag["value"]:
                print(f"\n=== graceful termination at epoch {epoch+1} ===")
                break

        # training complete
        training_end = datetime.now()
        total_training_duration = (training_end - training_start).total_seconds()
        avg_epoch_duration = sum(epoch_durations) / len(epoch_durations) if epoch_durations else 0.0

        model_path = run_dir / "model_final.pt"
        torch.save({
            'model': model.state_dict()
        }, model_path)
        print(f"saved final model to {model_path}")

        metrics = {
            "final_train_loss": float(mean_losses[LossType.TOTAL]),
            "commit_hash": commit_hash,
            "training_duration_seconds": round(total_training_duration, 2),
            "avg_epoch_duration_seconds": round(avg_epoch_duration, 2),

        }
        metrics_path = run_dir / "final_metrics.yaml"
        with open(metrics_path, "w") as f:
            yaml.dump(metrics, f, sort_keys=False, indent=2)
        print(f"saved metrics to {metrics_path}")

        chunk_loader.cleanup()

        # save profiler if still running (epochs < profile_first_n_epochs)
        if profiler.is_enabled():
            profiler.stop()
            trace_path = run_dir / "pipeline_trace.json"
            profiler.save(trace_path)
            print(f"saved pipeline trace to {trace_path}")
            profiler.print_stats()

        writer.close()
        print("training complete")

        return terminate_flag["value"]


# -------------------------------------------------------------------
# CLI Entry Point
# -------------------------------------------------------------------


if __name__ == "__main__":
    msg = """Usage: python latent_stag.py <expt_code> <default_config> [overrides...]

Arguments:
  expt_code       Experiment code (must match [A-Za-z0-9_]+)
  default_config  Base config file (e.g., latent_stag_20step.yaml)

Example:
  python latent_stag.py my_experiment latent_stag_20step.yaml --training.epochs 100"""

    if len(sys.argv) < 3:
        print(msg)
        sys.exit(1)

    expt_code = sys.argv[1]
    default_yaml = sys.argv[2]

    if not re.match("[A-Za-z0-9_]+", expt_code):
        print(f"error: expt_code must match [A-Za-z0-9_]+, got: {expt_code}")
        sys.exit(1)

    default_path = Path(__file__).resolve().parent / default_yaml
    if not default_path.exists():
        print(f"error: config file not found: {default_path}")
        sys.exit(1)

    tyro_args = sys.argv[3:]
    commit_hash = get_git_commit_hash()

    run_dir = create_run_directory(
        expt_code=expt_code,
        tyro_args=tyro_args,
        model_class=ModelParams,
        commit_hash=commit_hash,
    )

    with open(run_dir / "command_line.txt", "w") as out:
        out.write("\n".join(sys.argv))

    with open(default_path, "r") as f:
        data = yaml.safe_load(f)
    default_cfg = ModelParams(**data)

    cfg = tyro.cli(ModelParams, default=default_cfg, args=tyro_args)

    was_terminated = train(cfg, run_dir)

    # add a completion/termination flag
    flag_file = "terminated" if was_terminated else "complete"
    with open(run_dir / flag_file, "w"):
        pass
