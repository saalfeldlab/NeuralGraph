"""Model checkpoint loading utilities."""

from pathlib import Path

import torch
import yaml


def get_device() -> torch.device:
    """Cross-platform device selection."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("Using Apple MPS backend for training.")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("Using CPU for training.")
        return torch.device("cpu")


def load_model_from_checkpoint(
    checkpoint_path: Path | str,
    config_path: Path | str | None = None,
    device: torch.device | None = None,
):
    """
    Load a model from a checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint file (e.g., "runs/my_exp/run_id/checkpoints/checkpoint_best.pt")
        config_path: Optional path to config.yaml. If None, looks for config.yaml in run directory.
        device: Device to load model onto. If None, uses get_device().

    Returns:
        Loaded LatentModel instance in eval mode

    Example:
        >>> from pathlib import Path
        >>> model = load_model_from_checkpoint(
        ...     "runs/my_experiment/20251105_abc123_def456/checkpoints/checkpoint_best.pt"
        ... )
        >>> # Use model for inference
        >>> model.eval()
        >>> with torch.no_grad():
        ...     output = model(x, stim)
    """
    # Import here to avoid circular dependency
    from LatentEvolution.latent import LatentModel, ModelParams

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Infer config path if not provided
    if config_path is None:
        # Assume checkpoint is in: run_dir/checkpoints/checkpoint_*.pt
        run_dir = checkpoint_path.parent.parent
        config_path = run_dir / "config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load config
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    cfg = ModelParams(**config_dict)

    # Get device
    if device is None:
        device = get_device()

    # Create model
    model = LatentModel(cfg).to(device)

    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    # Set to eval mode
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"  Config: {config_path}")
    print(f"  Device: {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model
