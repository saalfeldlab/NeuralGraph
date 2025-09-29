import argparse
import importlib
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")

# Ensure local 'src' is importable when running as a script
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Dynamic imports to remain robust to execution environments
_cfg_mod = importlib.import_module("NeuralGraph.config")
NeuralGraphConfig = getattr(_cfg_mod, "NeuralGraphConfig")

_gen_mod = importlib.import_module("NeuralGraph.generators.graph_data_generator")
data_generate = getattr(_gen_mod, "data_generate")

_train_mod = importlib.import_module("NeuralGraph.models.graph_trainer")
data_train = getattr(_train_mod, "data_train")
data_test = getattr(_train_mod, "data_test")

_utils_mod = importlib.import_module("NeuralGraph.utils")
add_pre_folder = getattr(_utils_mod, "add_pre_folder")
set_device = getattr(_utils_mod, "set_device")


def _parse_args():
    parser = argparse.ArgumentParser(description="NeuralGraph multimodel runner")
    parser.add_argument(
        "-o",
        "--option",
        nargs="+",
        required=True,
        help=(
            "task token and one or more config names.\n"
            "Examples:\n"
            "  -o generate,train,test fly_N9_22_10 fly_N9_44_24\n"
            "  -o train fly_N9_22_10\n"
        ),
    )
    parser.add_argument(
        "--model-start",
        type=int,
        default=0,
        help="Start model id (inclusive), zero-padded to 3 digits",
    )
    parser.add_argument(
        "--model-end",
        type=int,
        default=49,
        help="End model id (inclusive)",
    )
    parser.add_argument(
        "--ensemble-id",
        type=str,
        default=None,
        help="Override ensemble id (zero-padded to 4 digits if numeric)",
    )
    parser.add_argument(
        "--best-model",
        type=str,
        default="best",
        help="Checkpoint tag to load at test time (default: 'best')",
    )
    return parser.parse_args()


def _model_id_list(start_id: int, end_id: int):
    start = int(start_id)
    end = int(end_id)
    if end < start:
        raise ValueError("model-end must be >= model-start")
    return [f"{i:03d}" for i in range(start, end + 1)]


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)

    args = _parse_args()

    # Task token and config list from -o
    task = args.option[0]
    config_names = args.option[1:]
    if len(config_names) == 0:
        raise SystemExit("No config names provided after task token in -o/--option")

    model_ids = _model_id_list(args.model_start, args.model_end)

    for config_file_ in config_names:
        print(" ")
        config_root = os.path.dirname(os.path.abspath(__file__)) + "/config"
        cfg_rel, pre_folder = add_pre_folder(config_file_)
        cfg = NeuralGraphConfig.from_yaml(f"{config_root}/{cfg_rel}.yaml")

        base_dataset = cfg.dataset
        base_config_file = config_file_

        for mid in model_ids:
            # Clone-like behavior by mutating fields prior to each stage
            if args.ensemble_id is not None:
                cfg.simulation.ensemble_id = args.ensemble_id
            cfg.simulation.model_id = mid

            # Separate per-model data and log outputs by suffixing
            cfg.dataset = pre_folder + base_dataset + f"__mid_{mid}"
            cfg.config_file = pre_folder + base_config_file + f"__mid_{mid}"

            device = set_device(cfg.training.device)

            print(f"\033[92mconfig_file  {cfg.config_file}\033[0m")
            print(f"\033[92mdevice  {device}\033[0m")

            if "generate" in task:
                data_generate(
                    cfg,
                    device=device,
                    visualize=False,
                    run_vizualized=0,
                    style="black color",
                    alpha=1,
                    erase=False,
                    bSave=True,
                    step=25,
                )

            if "train" in task:
                data_train(config=cfg, erase=False, best_model=args.best_model, device=device)

            if "test" in task:
                data_test(
                    config=cfg,
                    visualize=True,
                    style="black color name",
                    verbose=False,
                    best_model=args.best_model,
                    run=0,
                    test_mode="",
                    sample_embedding=False,
                    step=2,
                    device=device,
                    particle_of_interest=0,
                    new_params=None,
                )


