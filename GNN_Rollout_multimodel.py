import argparse
import os
import warnings

import matplotlib

from NeuralGraph.config import NeuralGraphConfig
from NeuralGraph.models.graph_trainer import data_test
from NeuralGraph.utils import add_pre_folder, set_device


matplotlib.use("Agg")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FlyVis rollouts (data_test_flyvis) for many models."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="One or more config names (e.g. fly_N9_22_10 fly_N9_44_24)",
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
    parser.add_argument(
        "--n-rollout-frames",
        type=int,
        default=600,
        help="Number of rollout frames to simulate (passed to data_test_flyvis).",
    )
    parser.add_argument(
        "--no-rollout-noise",
        action="store_true",
        help="Disable process noise during rollout (override noise_model_level=0 at test time).",
    )
    parser.add_argument(
        "--override_stored_rollout",
        action="store_true",
        help="Override stored rollout results with new rollout results.",
    )
    return parser.parse_args()


def _model_id_list(start_id: int, end_id: int) -> list[str]:
    start = int(start_id)
    end = int(end_id)
    if end < start:
        raise ValueError("model-end must be >= model-start")
    return [f"{i:03d}" for i in range(start, end + 1)]


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)

    args = _parse_args()

    model_ids = _model_id_list(args.model_start, args.model_end)

    for config_file_ in args.configs:
        print(" ")
        config_root = os.path.dirname(os.path.abspath(__file__)) + "/config"
        cfg_rel, pre_folder = add_pre_folder(config_file_)
        cfg = NeuralGraphConfig.from_yaml(f"{config_root}/{cfg_rel}.yaml")

        base_dataset = cfg.dataset
        base_config_file = config_file_

        for mid in model_ids:
            if args.ensemble_id is not None:
                cfg.simulation.ensemble_id = args.ensemble_id
            cfg.simulation.model_id = mid

            cfg.dataset = pre_folder + base_dataset + f"__mid_{mid}"
            cfg.config_file = pre_folder + base_config_file + f"__mid_{mid}"

            # Skip if no checkpoints exist for this model
            log_dir = os.path.join("log", cfg.config_file)
            models_dir = os.path.join(log_dir, "models")
            if not os.path.isdir(models_dir) or not os.listdir(models_dir):
                print(f"\033[93m[rollout] No models in {models_dir}; skipping mid={mid}\033[0m")
                continue

            # Skip if rollout results already exist
            results_dir = os.path.join(log_dir, "results")
            rollout_log = os.path.join(log_dir, "results_rollout.log")
            # the standard activity results are not saved anymore, test_mode="" is changed silently to test_ablation_0, causing the numerically identical rollouts to be saved under filenames with modified suffix
            # activity_true = os.path.join(results_dir, "activity_true.npy")
            # activity_pred = os.path.join(results_dir, "activity_pred.npy")
            activity_mod = os.path.join(results_dir, "activity_modified.npy")
            activity_mod_pred = os.path.join(results_dir, "activity_modified_pred.npy")

            # already_has_standard = os.path.exists(activity_true) and os.path.exists(activity_pred)
            already_has_modified = os.path.exists(activity_mod) and os.path.exists(activity_mod_pred)
            if os.path.exists(rollout_log) and already_has_modified and not args.override_stored_rollout: #(already_has_standard or already_has_modified):
                print(f"\033[93m[rollout] Existing rollout results in {log_dir}; skipping mid={mid}\033[0m")
                continue

            device = set_device(cfg.training.device)

            print(f"\033[92m[rollout] config_file {cfg.config_file}\033[0m")
            print(f"\033[92m[rollout] device {device}\033[0m")

            # Routes to data_test_flyvis for 'fly' datasets and
            # produces results_rollout.log and activity_{true,pred}.npy.
            data_test(
                config=cfg,
                visualize=False,
                style="black color name",
                verbose=False,
                best_model=args.best_model,
                run=0,
                test_mode="",
                sample_embedding=False,
                step=10,
                n_rollout_frames=args.n_rollout_frames,
                device=device,
                particle_of_interest=0,
                new_params=None,
                rollout_without_noise=args.no_rollout_noise,
            )


