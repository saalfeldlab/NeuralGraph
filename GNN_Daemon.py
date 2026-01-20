import matplotlib
matplotlib.use('Agg')  # set non-interactive backend before other imports
import argparse
import os
import time
import glob
import shutil
from datetime import datetime

# redirect PyTorch JIT cache to /scratch instead of /tmp (per IT request)
if os.path.isdir('/scratch'):
    os.environ['TMPDIR'] = '/scratch/allierc'
    os.makedirs('/scratch/allierc', exist_ok=True)


from NeuralGraph.config import NeuralGraphConfig
from NeuralGraph.generators.graph_data_generator import data_generate
from NeuralGraph.models.graph_trainer import data_train, data_test
from NeuralGraph.utils import set_device, add_pre_folder
from GNN_PlotFigure import data_plot

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")


def log_event(log_file, config_name, event_type):
    """Append an event to the log file with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a') as f:
        f.write(f"{timestamp} | {event_type} | {config_name}\n")


def is_file_stable(filepath, wait_time=2):
    """Check if file is fully written by comparing size over time."""
    try:
        size1 = os.path.getsize(filepath)
        time.sleep(wait_time)
        size2 = os.path.getsize(filepath)
        return size1 == size2
    except OSError:
        return False


def process_config(config_file_path, device, log_file):
    """Process a single config file through generate, train, test, plot pipeline."""

    config_root = os.path.dirname(os.path.abspath(__file__)) + "/config"
    config_name = os.path.basename(config_file_path)
    config_file_ = os.path.splitext(config_name)[0]  # remove .yaml extension

    # Log start
    log_event(log_file, config_name, "START")
    print(f"\n{'='*60}")
    print(f"Processing: {config_name}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    try:
        config_file, pre_folder = add_pre_folder(config_file_)

        # load config from the processing directory
        processing_path = f"{config_root}/processing/{config_name}"
        config = NeuralGraphConfig.from_yaml(processing_path)
        config.dataset = pre_folder + config.dataset
        config.config_file = pre_folder + config_file_

        # create analysis log file for metrics (used by Claude loop)
        root_dir = os.path.dirname(os.path.abspath(__file__))
        analysis_log_path = f"{root_dir}/{config_file_}_analysis.log"
        analysis_log = open(analysis_log_path, 'w')

        task = 'generate_train_test_plot'

        if "generate" in task:
            print("\n--- GENERATE ---")
            data_generate(
                config,
                device=device,
                visualize=False,
                run_vizualized=0,
                style="color",
                alpha=1,
                erase=True,
                bSave=True,
                step=2,
            )

        if "train" in task:
            print("\n--- TRAIN ---")
            data_train(
                config=config,
                erase=True,
                best_model=None,
                style='',
                device=device,
                log_file=analysis_log,
            )

        if "test" in task:
            print("\n--- TEST ---")
            config.training.noise_model_level = 0.0
            data_test(
                config=config,
                visualize=False,
                style="color name continuous_slice",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="",
                sample_embedding=False,
                step=10,
                n_rollout_frames=1000,
                device=device,
                particle_of_interest=0,
                new_params=None,
                log_file=analysis_log,
            )

        if 'plot' in task:
            print("\n--- PLOT ---")
            folder_name = './log/' + pre_folder + '/tmp_results/'
            os.makedirs(folder_name, exist_ok=True)
            data_plot(config=config, config_file=config_file, epoch_list=['best'],
                     style='black color', extended='plots', device=device, apply_weight_correction=True, log_file=analysis_log)

        # close analysis log file
        analysis_log.close()

        # Move to done
        done_path = f"{config_root}/done/{config_name}"
        shutil.move(processing_path, done_path)
        print(f"Config file {config_name} moved to done/")

        # Log stop (success)
        log_event(log_file, config_name, "STOP (SUCCESS)")
        print(f"\n{'='*60}")
        print(f"Completed: {config_name}")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        print("Waiting for config file to be copied to new/ ...")

    except Exception as e:
        # Log stop (error)
        log_event(log_file, config_name, f"STOP (ERROR: {str(e)[:50]})")
        print(f"\nERROR processing {config_name}: {e}")

        # Move to done even on error (to avoid reprocessing)
        try:
            processing_path = f"{config_root}/processing/{config_name}"
            done_path = f"{config_root}/done/{config_name}"
            if os.path.exists(processing_path):
                shutil.move(processing_path, done_path)
                print(f"Config file {config_name} moved to done/ (with error)")
        except Exception:
            pass
        print("Waiting for config file to be copied to new/ ...")


def daemon_loop(target_config=None):
    """Main daemon loop that watches config/new/ for config files.

    Args:
        target_config: If specified, only process this specific config file (e.g., 'signal_sparsity_Claude').
                      If None, process any yaml file that appears in new/.
    """

    warnings.filterwarnings("ignore", category=FutureWarning)

    config_root = os.path.dirname(os.path.abspath(__file__)) + "/config"
    new_dir = f"{config_root}/new"
    processing_dir = f"{config_root}/processing"
    done_dir = f"{config_root}/done"
    log_file = f"{config_root}/daemon_log.txt"

    # Create directories if they don't exist
    os.makedirs(new_dir, exist_ok=True)
    os.makedirs(processing_dir, exist_ok=True)
    os.makedirs(done_dir, exist_ok=True)

    # Determine target filename
    if target_config:
        target_filename = f"{target_config}.yaml" if not target_config.endswith('.yaml') else target_config
    else:
        target_filename = None

    # Initialize device once
    device = None

    print(f"\n{'#'*60}")
    print("GNN Daemon Started")
    print(f"Watching: {new_dir}")
    if target_filename:
        print(f"Target config: {target_filename}")
    else:
        print("Target config: ANY (processing all yaml files)")
    print(f"Log file: {log_file}")
    print(f"{'#'*60}\n")

    log_event(log_file, target_filename or "DAEMON", "DAEMON_START")

    print(f"Waiting for {target_filename or 'config file'} to be copied to new/ ...")

    while True:
        try:
            # Look for the target yaml file in new/
            if target_filename:
                # Only look for the specific target file
                config_file_path = f"{new_dir}/{target_filename}"
                if not os.path.exists(config_file_path):
                    time.sleep(10)
                    continue
                config_name = target_filename
            else:
                # Look for any yaml files in new/
                yaml_files = glob.glob(f"{new_dir}/*.yaml")
                if not yaml_files:
                    time.sleep(10)
                    continue
                # Sort by modification time (oldest first)
                yaml_files.sort(key=os.path.getmtime)
                config_file_path = yaml_files[0]
                config_name = os.path.basename(config_file_path)

            print(f"Config file {config_name} copied to new/")

            # Check if file is fully written
            if is_file_stable(config_file_path):
                print(f"Config file {config_name} detected and stable")

                # Move to processing
                processing_path = f"{processing_dir}/{config_name}"
                shutil.move(config_file_path, processing_path)
                print(f"Config file {config_name} moved to processing/")

                # Initialize device if needed
                if device is None:
                    # Load config to get device setting
                    temp_config = NeuralGraphConfig.from_yaml(processing_path)
                    device = set_device(temp_config.training.device)

                # Process the config
                process_config(processing_path, device, log_file)
            else:
                print(f"Config file {config_name} not stable yet, waiting...")

            # Wait before checking again
            time.sleep(10)

        except KeyboardInterrupt:
            print("\nDaemon stopped by user")
            log_event(log_file, target_filename or "DAEMON", "DAEMON_STOP")
            break
        except Exception as e:
            print(f"Error in daemon loop: {e}")
            time.sleep(10)  # Wait before retrying


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN Daemon - watches for config files and processes them")
    parser.add_argument(
        "-o", "--option", nargs="+", help="option: task config_name (e.g., 'generate_train_test_plot signal_sparsity_Claude')"
    )
    args = parser.parse_args()

    target_config = None
    if args.option and len(args.option) >= 2:
        # args.option[0] is task (ignored, daemon always does generate_train_test_plot)
        # args.option[1] is the target config name
        target_config = args.option[1]
        print(f"Target config from args: {target_config}")

    daemon_loop(target_config=target_config)
