import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before other imports
import argparse
import os
import subprocess
import time

# Redirect PyTorch JIT cache to /scratch instead of /tmp (per IT request)
if os.path.isdir('/scratch'):
    os.environ['TMPDIR'] = '/scratch/allierc'
    os.makedirs('/scratch/allierc', exist_ok=True)



from NeuralGraph.config import NeuralGraphConfig
from NeuralGraph.generators.graph_data_generator import data_generate
from NeuralGraph.models.graph_trainer import data_train, data_test, data_train_INR
from NeuralGraph.utils import set_device, add_pre_folder
from NeuralGraph.models.NGP_trainer import data_train_NGP
from GNN_PlotFigure import data_plot

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
# os.environ["MPLBACKEND"] = "Agg"
# os.environ["QT_API"] = "pyside6"
# os.environ["VISPY_BACKEND"] = "pyside6"

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    parser = argparse.ArgumentParser(description="NeuralGraph")
    parser.add_argument(
        "-o", "--option", nargs="+", help="Option that takes multiple values"
    )


    device=[]
    args = parser.parse_args()

    if args.option:
        print(f"Options: {args.option}")
    if args.option != None:
        task = args.option[0]
        config_list = [args.option[1]]
        if len(args.option) > 2:
            best_model = args.option[2]
        else:
            best_model = None
        # Parse additional parameters from remaining args (e.g. iterations=20, experiment=dale)
        task_params = {}
        for arg in args.option[2:]:
            if '=' in arg:
                key, value = arg.split('=', 1)
                task_params[key] = int(value) if value.isdigit() else value
    else:
        best_model = ''
        task = 'train_test_plot_Claude'  # 'train', 'test', 'generate', 'plot', 'train_NGP', 'train_INR', 'Claude'
        task_params = {'iterations': 48, 'experiment': 'experiment_convergence'}
        config_list = ['signal_chaotic_Claude']

    # Parse parameters from task_params
    n_iterations = task_params.get('iterations', 5)
    experiment_name = task_params.get('experiment', 'experiment')

    # If Claude in task, determine iteration range; otherwise single iteration
    if 'Claude' in task:
        iteration_range = range(1, n_iterations + 1)
    else:
        iteration_range = range(1, 2)  # single iteration

    for config_file_ in config_list:
        print(" ")
        config_root = os.path.dirname(os.path.abspath(__file__)) + "/config"
        config_file, pre_folder = add_pre_folder(config_file_)

        # Setup for Claude analysis (paths needed before iteration loop)
        if 'Claude' in task:
            root_dir = os.path.dirname(os.path.abspath(__file__))
            experiment_path = f"{root_dir}/{experiment_name}.md"
            analysis_path = f"{root_dir}/analysis_{experiment_name}.md"

            # Check experiment file exists
            if not os.path.exists(experiment_path):
                print(f"\033[91merror: experiment file not found: {experiment_path}\033[0m")
                print(f"\033[93mavailable experiment files:\033[0m")
                for f in os.listdir(root_dir):
                    if f.endswith('.md') and not f.startswith('analysis_') and not f.startswith('README'):
                        print(f"  - {f[:-3]}")
                continue

            # Clear analysis file at start
            with open(analysis_path, 'w') as f:
                f.write(f"# Experiment Log: {config_file_}\n\n")
            print(f"\033[93mcleared {analysis_path}\033[0m")
            print(f"\033[93mexperiment: {experiment_name} ({n_iterations} iterations)\033[0m")

        # Analysis log file in root folder (for Claude to read)
        root_dir = os.path.dirname(os.path.abspath(__file__))
        analysis_log_path = f"{root_dir}/analysis.log"

        for iteration in iteration_range:
            if 'Claude' in task:
                print(f"\n\n\n\033[94miteration {iteration}/{n_iterations}: {config_file_} ===\033[0m")

            # Reload config to pick up any changes from previous iteration
            config = NeuralGraphConfig.from_yaml(f"{config_root}/{config_file}.yaml")
            config.dataset = pre_folder + config.dataset
            config.config_file = pre_folder + config_file_

            if device==[]:
                device = set_device(config.training.device)

            # Open analysis.log for this iteration (append mode for test/plot to add metrics)
            log_file = open(analysis_log_path, 'w')

            if "generate" in task:
                erase = 'Claude' in task  # erase when iterating with Claude
                data_generate(
                    config,
                    device=device,
                    visualize=False,
                    run_vizualized=0,
                    style="black color",
                    alpha=1,
                    erase=erase,
                    bSave=True,
                    step=2,
                    log_file=log_file
                )

            if 'train_NGP' in task:
                # Use new modular NGP trainer pipeline
                data_train_NGP(config=config, device=device)

            elif 'train_INR' in task:
                print()
                # Pre-train nnr_f (SIREN) on external_input data before joint GNN learning
                data_train_INR(config=config, device=device, total_steps=50000)

            elif "train" in task:
                data_train(
                    config=config,
                    erase=False,
                    best_model=best_model,
                    style = 'black',
                    device=device
                )

            if "test" in task:

                config.training.noise_model_level = 0.0

                if 'fly' in config_file_:
                    config.simulation.visual_input_type = 'optical_flow'   #'DAVIS'

                data_test(
                    config=config,
                    visualize=False,
                    style="black color name continuous_slice",
                    verbose=False,
                    best_model='best',
                    run=0,
                    test_mode="",
                    sample_embedding=False,
                    step=10,
                    n_rollout_frames=1000,
                    device=device,
                    particle_of_interest=0,
                    new_params = None,
                    log_file=log_file,
                )

            if 'plot' in task:
                folder_name = './log/' + pre_folder + '/tmp_results/'
                os.makedirs(folder_name, exist_ok=True)
                data_plot(config=config, config_file=config_file, epoch_list=['best'], style='black color', extended='plots', device=device, apply_weight_correction=True, log_file=log_file)

            log_file.close()

            if 'Claude' in task:
                # Claude analysis: reads activity.png and analysis.log, updates config per experiment protocol
                data_folder = f"{root_dir}/graphs_data/{config.dataset}"
                activity_path = f"{data_folder}/activity.png"
                config_path = f"{root_dir}/config/{pre_folder}{config_file_}.yaml"

                # Check files are ready (generated by data_generate above)
                time.sleep(2)  # pause to ensure files are written
                if not os.path.exists(activity_path):
                    print(f"\033[91merror: activity.png not found at {activity_path}\033[0m")
                    continue
                if not os.path.exists(analysis_log_path):
                    print(f"\033[91merror: analysis.log not found at {analysis_log_path}\033[0m")
                    continue
                print(f"\033[92mfiles ready: activity.png, analysis.log\033[0m")

                # Call Claude CLI for analysis
                print(f"\033[93mClaude analysis...\033[0m")

                claude_prompt = f"""Iteration {iteration}/{n_iterations}: Parameter study.

1. Read activity image: {activity_path}
2. Read analysis log: {analysis_log_path}
3. Read protocol: {experiment_path}
4. Append to {analysis_path} using log format from protocol
5. Edit {config_path} to explore next parameter combination per protocol

Config file: {config_file_}"""

                claude_cmd = [
                    'claude',
                    '-p', claude_prompt,
                    '--output-format', 'text',
                    '--max-turns', '20',
                    '--allowedTools',
                    'Read', 'Edit'
                ]

                # Run with real-time output streaming
                process = subprocess.Popen(
                    claude_cmd,
                    cwd=root_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )

                # Stream output line by line
                for line in process.stdout:
                    print(line, end='', flush=True)

                process.wait()
