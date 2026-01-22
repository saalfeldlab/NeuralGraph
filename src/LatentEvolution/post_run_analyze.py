"""
Post-training analysis script for generating diagnostic figures.

This script can be pointed to a specific run directory to generate all
post-training diagnostic figures. It loads the trained model and datasets,
then creates figures within the run directory.

If training is complete (model_final.pt exists), figures are saved directly
to the run directory. If training is incomplete (only checkpoint_best.pt exists),
figures are saved to a TEMPORARY subdirectory to avoid overwriting final results.
"""

from pathlib import Path
import sys
import argparse
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from LatentEvolution.latent import LatentModel, ModelParams, get_device, load_val_only
from LatentEvolution.diagnostics import PlotMode, run_validation_diagnostics
from LatentEvolution.load_flyvis import NeuronData
from LatentEvolution.load_flyvis import load_metadata


def main(run_dir: Path, epoch: int | None = None) -> None:
    """
    Generate post-training diagnostic figures for a completed or ongoing run.

    Args:
        run_dir: Path to the run directory containing config.yaml and model checkpoints
        epoch: Optional epoch number to analyze a specific checkpoint
    """
    run_dir = run_dir.resolve()

    if not run_dir.exists():
        print(f"Error: Run directory does not exist: {run_dir}")
        sys.exit(1)

    # Load config
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        print(f"Error: config.yaml not found in {run_dir}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    cfg = ModelParams(**config_data)

    print(f"Loaded config from {config_path}")

    # Determine which model to use and where to save outputs
    model_final_path = run_dir / "model_final.pt"
    checkpoint_best_path = run_dir / "checkpoints" / "checkpoint_best.pt"

    if epoch is not None:
        # Use specific epoch checkpoint
        checkpoint_epoch_path = run_dir / "checkpoints" / f"checkpoint_epoch_{epoch:04d}.pt"
        if not checkpoint_epoch_path.exists():
            print(f"Error: Checkpoint not found: {checkpoint_epoch_path}")
            sys.exit(1)
        model_path = checkpoint_epoch_path
        out_dir = run_dir / f"analysis_epoch_{epoch:04d}"
        print(f"Using epoch {epoch} checkpoint: {model_path}")
        print(f"Figures will be saved to: {out_dir}")
    elif model_final_path.exists():
        model_path = model_final_path
        out_dir = run_dir
        print(f"Using final model: {model_path}")
    elif checkpoint_best_path.exists():
        model_path = checkpoint_best_path
        out_dir = run_dir / "TEMPORARY"
        print(f"Using best checkpoint (TEMPORARY): {model_path}")
        print(f"Figures will be saved to: {out_dir}")
    else:
        print("Error: No model found. Expected either:")
        print(f"  - {model_final_path}")
        print(f"  - {checkpoint_best_path}")
        sys.exit(1)

    # Get device
    device = get_device()

    # Create TensorBoard writer (writes to same directory as training)
    writer = SummaryWriter(log_dir=run_dir)
    print(f"TensorBoard logging to {run_dir}")

    # Load model
    model = LatentModel(cfg).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model to {device}")

    # Load main validation dataset
    print(f"\nLoading main dataset: {cfg.training.simulation_config}")
    val_data, val_stim = load_val_only(
        simulation_config=cfg.training.simulation_config,
        column_to_model=cfg.training.column_to_model,
        data_split=cfg.training.data_split,
        num_input_dims=cfg.stimulus_encoder_params.num_input_dims,
        device=device,
    )

    # Load neuron metadata
    data_path = f"graphs_data/fly/{cfg.training.simulation_config}/x_list_0"
    metadata = load_metadata(data_path)
    neuron_data = NeuronData.from_metadata(metadata)

    # Run diagnostics on main validation dataset
    print("\n=== Running diagnostics on main validation dataset ===")
    run_validation_diagnostics(
        run_dir=out_dir,
        val_data=val_data,
        neuron_data=neuron_data,
        val_stim=val_stim,
        model=model,
        config=cfg,
        plot_mode=PlotMode.POST_RUN,
    )
    print(f"Saved main validation figures to {out_dir}")

    # Run cross-validation diagnostics
    if cfg.cross_validation_configs:
        print("\n=== Running Cross-Dataset Validation ===")

        for cv_config in cfg.cross_validation_configs:
            cv_name = cv_config.name or cv_config.simulation_config
            print(f"\nEvaluating on {cv_name} ({cv_config.simulation_config})...")

            data_split = cv_config.data_split or cfg.training.data_split
            # Load cross-validation dataset (only need validation split)
            cv_val_data, cv_val_stim = load_val_only(
                simulation_config=cv_config.simulation_config,
                column_to_model=cfg.training.column_to_model,
                data_split=data_split,  # Use same time ranges
                num_input_dims=cfg.stimulus_encoder_params.num_input_dims,
                device=device,
            )

            # Run diagnostics on cross-validation dataset
            cv_out_dir = out_dir / "cross_validation" / cv_name
            cv_metrics, cv_figures = run_validation_diagnostics(
                run_dir=cv_out_dir,
                val_data=cv_val_data,
                neuron_data=neuron_data,
                val_stim=cv_val_stim,
                model=model,
                config=cfg,
                plot_mode=PlotMode.POST_RUN
            )
            print(f"Saved cross-validation figures to {cv_out_dir}")

            # Log cross-validation scalar metrics to TensorBoard
            for metric_name, metric_value in cv_metrics.items():
                writer.add_scalar(f"CrossVal/{cv_name}/{metric_name}", metric_value, 0)
            print(f"Logged {len(cv_metrics)} cross-validation scalar metrics to TensorBoard")

            # Log MSE figures to TensorBoard (skip rollout traces)
            mse_figure_names = [
                'mses_by_time_steps_latent',
                'mses_by_time_steps_activity',
                'multi_start_long_latent_rollout_mses_by_time',
                'multi_start_long_activity_rollout_mses_by_time'
            ]
            for fig_name in mse_figure_names:
                if fig_name in cv_figures:
                    writer.add_figure(f"CrossVal/{cv_name}/{fig_name}", cv_figures[fig_name], 0)
            print("Logged MSE figures to TensorBoard (skipped rollout traces)")

    # Close TensorBoard writer
    writer.close()
    print("\nTensorBoard logging completed")

    print("\n=== Post-training analysis complete ===")
    print(f"All figures saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate post-training diagnostic figures for a run directory"
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to the run directory containing config.yaml and model checkpoints"
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch number to analyze (uses checkpoints/checkpoint_epoch_XXXX.pt)"
    )
    args = parser.parse_args()

    main(args.run_dir, epoch=args.epoch)
