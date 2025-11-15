"""
Utilities for creating hierarchical run directories based on hyperparameter overrides.
"""

from pathlib import Path
from datetime import datetime
from uuid import uuid4
from pydantic import BaseModel


def parse_tyro_overrides(tyro_args: list[str]) -> list[tuple[str, str]]:
    """
    Parse tyro arguments to extract (param_path, value) pairs in order.

    Args:
        tyro_args: Command line arguments (e.g., ['--training.learning-rate', '0.001', '--latent-dims', '64'])

    Returns:
        List of (param_name, value) tuples in order, with hyphens converted to underscores.
        Boolean flags are handled as follows:
            --param-name        -> ('param_name', 'True')
            --no-param-name     -> ('param_name', 'False')
            --param-name True   -> ('param_name', 'True')
        Example: [('training.learning_rate', '0.001'), ('latent_dims', '64')]
    """
    overrides = []
    i = 0
    while i < len(tyro_args):
        if tyro_args[i].startswith('--'):
            # Remove the '--' prefix
            arg = tyro_args[i][2:]

            # Check if this is a negated boolean flag (--no-param-name)
            if arg.startswith('no-'):
                # Remove 'no-' prefix and convert hyphens to underscores
                param_name = arg[3:].replace('-', '_')
                overrides.append((param_name, 'False'))
                i += 1
            elif i + 1 < len(tyro_args) and not tyro_args[i + 1].startswith('--'):
                # Has an explicit value
                param_name = arg.replace('-', '_')
                value = tyro_args[i + 1]
                overrides.append((param_name, value))
                i += 2
            else:
                # Flag without value - treat as boolean True
                param_name = arg.replace('-', '_')
                overrides.append((param_name, 'True'))
                i += 1
        else:
            i += 1
    return overrides


def get_short_name_for_field(model_class: type[BaseModel], field_path: str) -> str | None:
    """
    Get short name from json_schema_extra if defined for a field path.

    Args:
        model_class: The pydantic model class (e.g., ModelParams)
        field_path: Dot-separated field path (e.g., 'training.learning_rate' or 'latent_dims')

    Returns:
        Short name if defined in json_schema_extra, otherwise None
    """
    parts = field_path.split('.')
    current_model = model_class

    for i, part in enumerate(parts):
        if part not in current_model.model_fields:
            return None

        field_info = current_model.model_fields[part]

        # If this is the last part, check for short_name
        if i == len(parts) - 1:
            if field_info.json_schema_extra and isinstance(field_info.json_schema_extra, dict):
                short_name = field_info.json_schema_extra.get('short_name')
                assert isinstance(short_name, str | None)
                return short_name
            return None

        # Otherwise, navigate to the nested model
        field_type = field_info.annotation
        # Handle Optional types, Union, etc.
        if hasattr(field_type, '__origin__'):
            # For Union types, try to find BaseModel subclass
            if hasattr(field_type, '__args__'):
                for arg in field_type.__args__:
                    try:
                        if isinstance(arg, type) and issubclass(arg, BaseModel):
                            field_type = arg
                            break
                    except TypeError:
                        continue
            else:
                field_type = field_type.__args__[0]

        try:
            if not issubclass(field_type, BaseModel):
                return None
        except TypeError:
            return None

        current_model = field_type

    return None


def build_hparam_path(tyro_args: list[str], model_class: type[BaseModel]) -> Path:
    """
    Build nested directory path from tyro overrides using short names where available.

    Args:
        tyro_args: Command line arguments from tyro
        model_class: The pydantic model class to extract short names from

    Returns:
        Path object with nested directory structure (e.g., Path('lr0.001/bs32/ld64'))
        Returns Path('.') if no overrides found.
    """
    overrides = parse_tyro_overrides(tyro_args)
    if not overrides:
        return Path('.')

    hparam_dirs = []
    for param_path, value in overrides:
        short_name = get_short_name_for_field(model_class, param_path)
        if short_name:
            hparam_dirs.append(f"{short_name}{value}")
        else:
            # Use full path with dots replaced by underscores
            sanitized_path = param_path.replace('.', '_')
            hparam_dirs.append(f"{sanitized_path}{value}")

    return Path(*hparam_dirs)


def create_run_directory(
    expt_code: str,
    tyro_args: list[str],
    model_class: type[BaseModel],
    commit_hash: str,
    base_dir: Path = Path("runs"),
) -> Path:
    """
    Create hierarchical run directory with structure:
    <base_dir>/<expt_code>_<date>_<commit_hash>/<param1>/<param2>/.../<uuid>/

    Args:
        expt_code: Experiment code/name
        tyro_args: Command line arguments from tyro
        model_class: Pydantic model class for extracting short names
        commit_hash: Git commit hash
        base_dir: Base directory for runs (default: "runs")

    Returns:
        Path to the created run directory
    """
    date_str = datetime.now().strftime("%Y%m%d")
    expt_dir_name = f"{expt_code}_{date_str}_{commit_hash}"

    # Build hyperparameter path
    hparam_path = build_hparam_path(tyro_args, model_class)

    # Generate short UUID for uniqueness
    run_uuid = str(uuid4())[:6]

    # Construct full path
    run_dir = base_dir / expt_dir_name / hparam_path / run_uuid

    # Create directory
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir
