import matplotlib
matplotlib.use('Agg')  # set non-interactive backend before other imports
import argparse
import glob
import os
import shutil
import subprocess
import time

# redirect PyTorch JIT cache to /scratch instead of /tmp (per IT request)
if os.path.isdir('/scratch'):
    os.environ['TMPDIR'] = '/scratch/allierc'
    os.makedirs('/scratch/allierc', exist_ok=True)


from NeuralGraph.config import NeuralGraphConfig
from NeuralGraph.generators.graph_data_generator import data_generate
from NeuralGraph.models.graph_trainer import data_train, data_test, data_train_INR
from NeuralGraph.models.exploration_tree import compute_ucb_scores
from NeuralGraph.utils import set_device, add_pre_folder
from NeuralGraph.models.NGP_trainer import data_train_NGP
from GNN_PlotFigure import data_plot

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    parser = argparse.ArgumentParser(description="NeuralGraph")
    parser.add_argument(
        "-o", "--option", nargs="+", help="option that takes multiple values"
    )

    print()
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
        # parse additional parameters from remaining args (e.g. iterations=20, experiment=dale)
        task_params = {}
        for arg in args.option[2:]:
            if '=' in arg:
                key, value = arg.split('=', 1)
                task_params[key] = int(value) if value.isdigit() else value
    else:
        best_model = ''
        task = 'generate_train_test_plot_Claude'  # 'train', 'test', 'generate', 'plot', 'train_NGP', 'train_INR', 'Claude'
        task_params = {'iterations': 48, 'experiment': 'experiment_Dale_1'}
        config_list = ['signal_Claude']

    # parse parameters from task_params
    n_iterations = task_params.get('iterations', 5)
    experiment_name = task_params.get('experiment', 'experiment')

    # if Claude in task, determine iteration range; otherwise single iteration
    if 'Claude' in task:
        iteration_range = range(1, n_iterations + 1)
        # copy signal_Claude_first.yaml to signal_Claude.yaml at start
        root_dir = os.path.dirname(os.path.abspath(__file__))
        config_root = root_dir + "/config"
        for cfg in config_list:
            cfg_file, pre = add_pre_folder(cfg)
            first_config = f"{config_root}/{pre}{cfg}_first.yaml"
            target_config = f"{config_root}/{pre}{cfg}.yaml"
            if os.path.exists(first_config):
                shutil.copy2(first_config, target_config)
                print(f"\033[93mcopied {first_config} -> {target_config}\033[0m")
    else:
        iteration_range = range(1, 2)  # single iteration

    for config_file_ in config_list:
        print(" ")
        config_root = os.path.dirname(os.path.abspath(__file__)) + "/config"
        config_file, pre_folder = add_pre_folder(config_file_)

        # setup for Claude analysis (paths needed before iteration loop)
        if 'Claude' in task:
            root_dir = os.path.dirname(os.path.abspath(__file__))
            experiment_path = f"{root_dir}/{experiment_name}.md"
            analysis_path = f"{root_dir}/analysis_{experiment_name}.md"

            # check experiment file exists
            if not os.path.exists(experiment_path):
                print(f"\033[91merror: experiment file not found: {experiment_path}\033[0m")
                print(f"\033[93mavailable experiment files:\033[0m")
                for f in os.listdir(root_dir):
                    if f.endswith('.md') and not f.startswith('analysis_') and not f.startswith('README'):
                        print(f"  - {f[:-3]}")
                continue

            # clear analysis file at start
            with open(analysis_path, 'w') as f:
                f.write(f"# Experiment Log: {config_file_}\n\n")
            print(f"\033[93mcleared {analysis_path}\033[0m")
            print(f"\033[93mexperiment: {experiment_name} ({n_iterations} iterations)\033[0m")

        # analysis log file in root folder (for Claude to read)
        root_dir = os.path.dirname(os.path.abspath(__file__))
        analysis_log_path = f"{root_dir}/analysis.log"

        for iteration in iteration_range:
            if 'Claude' in task:
                print(f"\n\n\n\033[94miteration {iteration}/{n_iterations}: {config_file_} ===\033[0m")

            # reload config to pick up any changes from previous iteration
            config = NeuralGraphConfig.from_yaml(f"{config_root}/{config_file}.yaml")
            config.dataset = pre_folder + config.dataset
            config.config_file = pre_folder + config_file_

            if device==[]:
                device = set_device(config.training.device)

            # open analysis.log for this iteration (append mode for test/plot to add metrics)
            log_file = open(analysis_log_path, 'w')

            if "generate" in task:
                erase = 'Claude' in task  # erase when iterating with claude
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
                # use new modular NGP trainer pipeline
                data_train_NGP(config=config, device=device)

            elif 'train_INR' in task:
                print()
                # pre-train nnr_f (SIREN) on external_input data before joint GNN learning
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
                # save exploration artifacts before Claude analysis
                exploration_dir = f"{root_dir}/log/Claude_exploration/{experiment_name}"
                config_save_dir = f"{exploration_dir}/config"
                scatter_save_dir = f"{exploration_dir}/connectivity_scatter"
                heatmap_save_dir = f"{exploration_dir}/connectivity_heatmap"
                activity_save_dir = f"{exploration_dir}/activity"

                # create directories at start of experiment
                if iteration == 1:
                    # clear and recreate exploration folder
                    if os.path.exists(exploration_dir):
                        shutil.rmtree(exploration_dir)
                    os.makedirs(config_save_dir, exist_ok=True)
                    os.makedirs(scatter_save_dir, exist_ok=True)
                    os.makedirs(heatmap_save_dir, exist_ok=True)
                    os.makedirs(activity_save_dir, exist_ok=True)

                # save config file
                src_config = f"{root_dir}/config/{pre_folder}{config_file_}.yaml"
                dst_config = f"{config_save_dir}/iter_{iteration:03d}.yaml"
                if os.path.exists(src_config):
                    shutil.copy2(src_config, dst_config)

                # save connectivity scatterplot (most recent comparison_*.tif from matrix folder)
                matrix_dir = f"{root_dir}/log/{pre_folder}{config_file_}/tmp_training/matrix"
                scatter_files = glob.glob(f"{matrix_dir}/comparison_*.tif")
                if scatter_files:
                    # get most recent file
                    latest_scatter = max(scatter_files, key=os.path.getmtime)
                    dst_scatter = f"{scatter_save_dir}/iter_{iteration:03d}.tif"
                    shutil.copy2(latest_scatter, dst_scatter)

                # save connectivity heatmap
                data_folder = f"{root_dir}/graphs_data/{config.dataset}"
                src_heatmap = f"{data_folder}/connectivity_matrix.png"
                dst_heatmap = f"{heatmap_save_dir}/iter_{iteration:03d}.png"
                if os.path.exists(src_heatmap):
                    shutil.copy2(src_heatmap, dst_heatmap)

                # save activity plot
                activity_path = f"{data_folder}/activity.png"
                dst_activity = f"{activity_save_dir}/iter_{iteration:03d}.png"
                if os.path.exists(activity_path):
                    shutil.copy2(activity_path, dst_activity)

                # claude analysis: reads activity.png and analysis.log, updates config per experiment protocol
                config_path = f"{root_dir}/config/{pre_folder}{config_file_}.yaml"
                ucb_path = f"{root_dir}/ucb_scores.txt"

                # compute UCB scores including current iteration's connectivity_R2 from analysis.log
                ucb_computed = compute_ucb_scores(analysis_path, ucb_path,
                                                   current_log_path=analysis_log_path,
                                                   current_iteration=iteration)
                if ucb_computed:
                    print(f"\033[92mUCB scores computed: {ucb_path}\033[0m")

                # check files are ready (generated by data_generate above)
                time.sleep(2)  # pause to ensure files are written
                if not os.path.exists(activity_path):
                    print(f"\033[91merror: activity.png not found at {activity_path}\033[0m")
                    continue
                if not os.path.exists(analysis_log_path):
                    print(f"\033[91merror: analysis.log not found at {analysis_log_path}\033[0m")
                    continue
                print(f"\033[92mfiles ready: activity.png, analysis.log\033[0m")

                # call Claude CLI for analysis
                print(f"\033[93mClaude analysis...\033[0m")

                claude_prompt = f"""Iteration {iteration}/{n_iterations}: Parameter study.

1. Read activity image: {activity_path}
2. Read analysis log: {analysis_log_path}
3. Read protocol: {experiment_path}
4. Read UCB scores: {ucb_path}
5. Append to {analysis_path} using log format from protocol
6. Edit {config_path} to explore next parameter combination per protocol

Config file: {config_file_}"""

                claude_cmd = [
                    'claude',
                    '-p', claude_prompt,
                    '--output-format', 'text',
                    '--max-turns', '20',
                    '--allowedTools',
                    'Read', 'Edit'
                ]

                # run with real-time output streaming
                process = subprocess.Popen(
                    claude_cmd,
                    cwd=root_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )

                # stream output line by line
                for line in process.stdout:
                    print(line, end='', flush=True)

                process.wait()
