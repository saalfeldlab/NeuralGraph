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

from LatentEvolution.latent import LatentModel, ModelParams, get_device, load_dataset
from LatentEvolution.diagnostics import run_validation_diagnostics


def main(run_dir: Path) -> None:
    """
    Generate post-training diagnostic figures for a completed or ongoing run.

    Args:
        run_dir: Path to the run directory containing config.yaml and model checkpoints
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

    if model_final_path.exists():
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

    # Load model
    model = LatentModel(cfg).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model to {device}")

    # Load main training dataset
    print(f"\nLoading main dataset: {cfg.training.simulation_config}")
    _, val_data, _, _, val_stim, _, neuron_data = load_dataset(
        simulation_config=cfg.training.simulation_config,
        column_to_model=cfg.training.column_to_model,
        data_split=cfg.training.data_split,
        num_input_dims=cfg.stimulus_encoder_params.num_input_dims,
        device=device,
    )

    # Run diagnostics on main validation dataset
    print("\n=== Running diagnostics on main validation dataset ===")
    run_validation_diagnostics(
        run_dir=out_dir,
        val_data=val_data,
        neuron_data=neuron_data,
        val_stim=val_stim,
        model=model,
        config=cfg,
        save_figures=True,
    )
    print(f"Saved main validation figures to {out_dir}")

    # Run cross-validation diagnostics
    if cfg.cross_validation_configs:
        print("\n=== Running Cross-Dataset Validation ===")

        for cv_config in cfg.cross_validation_configs:
            cv_name = cv_config.name or cv_config.simulation_config
            print(f"\nEvaluating on {cv_name} ({cv_config.simulation_config})...")

            # Load cross-validation dataset (only need validation split)
            _, cv_val_data, _, _, cv_val_stim, _, cv_neuron_data = load_dataset(
                simulation_config=cv_config.simulation_config,
                column_to_model=cfg.training.column_to_model,
                data_split=cfg.training.data_split,  # Use same time ranges
                num_input_dims=cfg.stimulus_encoder_params.num_input_dims,
                device=device,
            )

            # Run diagnostics on cross-validation dataset
            cv_out_dir = out_dir / "cross_validation" / cv_name
            run_validation_diagnostics(
                run_dir=cv_out_dir,
                val_data=cv_val_data,
                neuron_data=cv_neuron_data,
                val_stim=cv_val_stim,
                model=model,
                config=cfg,
                save_figures=True,
            )
            print(f"Saved cross-validation figures to {cv_out_dir}")

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
    args = parser.parse_args()

    main(args.run_dir)
