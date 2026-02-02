import matplotlib
matplotlib.use('Agg')  # set non-interactive backend before other imports
import argparse
import os
import shutil
import subprocess
import time
import re
import yaml

# redirect PyTorch JIT cache to /scratch instead of /tmp (per IT request)
if os.path.isdir('/scratch'):
    os.environ['TMPDIR'] = '/scratch/allierc'
    os.makedirs('/scratch/allierc', exist_ok=True)


import sys

from NeuralGraph.config import NeuralGraphConfig
from NeuralGraph.generators.graph_data_generator import data_generate
from NeuralGraph.models.graph_trainer import data_train, data_test, data_train_INR
from NeuralGraph.models.exploration_tree import compute_ucb_scores
from NeuralGraph.models.plot_exploration_tree import parse_ucb_scores, plot_ucb_tree
from NeuralGraph.models.utils import save_exploration_artifacts
from NeuralGraph.utils import set_device, add_pre_folder
from NeuralGraph.models.NGP_trainer import data_train_NGP
from NeuralGraph.git_code_tracker import track_code_modifications, is_git_repo, get_modified_code_files
from GNN_PlotFigure import data_plot

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

def detect_last_iteration(analysis_path):
    """Detect the last completed iteration from analysis.md.

    Scans for '## Iter N:' entries written by Claude after each iteration.
    Returns the next iteration to run (1-indexed), or 1 if nothing found.
    """
    found_iters = set()
    if os.path.exists(analysis_path):
        with open(analysis_path, 'r') as f:
            for line in f:
                match = re.match(r'^##+ Iter (\d+):', line)
                if match:
                    found_iters.add(int(match.group(1)))
    if not found_iters:
        return 1
    return max(found_iters) + 1


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    parser = argparse.ArgumentParser(description="NeuralGraph")
    parser.add_argument(
        "-o", "--option", nargs="+", help="option that takes multiple values"
    )
    parser.add_argument(
        "--fresh", action="store_true", default=True, help="start from iteration 1 (default)"
    )
    parser.add_argument(
        "--resume", action="store_true", help="auto-resume from last completed iteration"
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
        task_params = {}
        for arg in args.option[2:]:
            if '=' in arg:
                key, value = arg.split('=', 1)
                task_params[key] = int(value) if value.isdigit() else value
    else:
        best_model = ''
        task = 'generate_train_test_plot_Claude_cluster'  # 'train', 'test', 'generate', 'plot', 'train_NGP', 'train_INR', 'Claude', 'code', 'cluster'
        config_list = ['signal_landscape']
        task_params = {'iterations': 2048}



    n_iterations = task_params.get('iterations', 5)
    base_config_name = config_list[0] if config_list else 'signal'
    instruction_name = task_params.get('instruction', f'instruction_{base_config_name}')
    llm_task_name = task_params.get('llm_task', f'{base_config_name}_Claude')

    # Auto-resume or fresh start
    _root = os.path.dirname(os.path.abspath(__file__))
    if args.resume:
        start_iteration = detect_last_iteration(f"{_root}/{llm_task_name}_analysis.md")
        if start_iteration > 1:
            print(f"\033[93mResuming from iteration {start_iteration}\033[0m")
        else:
            print(f"\033[93mNo previous iterations found, starting fresh\033[0m")
    else:
        start_iteration = 1
        _analysis_check = f"{_root}/{llm_task_name}_analysis.md"
        if os.path.exists(_analysis_check):
            print(f"\033[91mWARNING: Fresh start will erase existing results in:\033[0m")
            print(f"\033[91m  {_analysis_check}\033[0m")
            print(f"\033[91m  {_root}/{llm_task_name}_memory.md\033[0m")
            answer = input("\033[91mContinue? (y/n): \033[0m").strip().lower()
            if answer != 'y':
                print("Aborted.")
                sys.exit(0)
        print(f"\033[93mFresh start\033[0m")

    if 'Claude' in task:
        iteration_range = range(start_iteration, n_iterations + 1)


        root_dir = os.path.dirname(os.path.abspath(__file__))
        config_root = root_dir + "/config"

        if start_iteration > 1:
            print(f"\033[93mResuming from iteration {start_iteration}\033[0m")


        for cfg in config_list:
            cfg_file, pre = add_pre_folder(cfg)
            source_config = f"{config_root}/{pre}{cfg}.yaml"
            target_config = f"{config_root}/{pre}{llm_task_name}.yaml"

            # Only copy and initialize config on fresh start (not when resuming)
            if start_iteration == 1 and not args.resume:
                # Erase config from new/processing/done directories
                for subdir in ['new', 'processing', 'done']:
                    cleanup_path = f"{config_root}/{subdir}/{llm_task_name}.yaml"
                    if os.path.exists(cleanup_path):
                        os.remove(cleanup_path)
                        print(f"\033[93mdeleted {cleanup_path}\033[0m")

                if os.path.exists(source_config):
                    shutil.copy2(source_config, target_config)
                    print(f"\033[93mcopied {source_config} -> {target_config}\033[0m")
                    with open(target_config, 'r') as f:
                        config_data = yaml.safe_load(f)
                    claude_cfg = config_data.get('claude', {})
                    claude_n_epochs = claude_cfg.get('n_epochs', 1)
                    claude_data_augmentation_loop = claude_cfg.get('data_augmentation_loop', 100)
                    claude_n_iter_block = claude_cfg.get('n_iter_block', 24)
                    claude_ucb_c = claude_cfg.get('ucb_c', 1.414)
                    claude_node_name = claude_cfg.get('node_name', 'a100')
                    config_data['dataset'] = llm_task_name
                    config_data['training']['n_epochs'] = claude_n_epochs
                    config_data['training']['data_augmentation_loop'] = claude_data_augmentation_loop
                    config_data['description'] = 'designed by Claude'
                    config_data['claude'] = {
                        'n_epochs': claude_n_epochs,
                        'data_augmentation_loop': claude_data_augmentation_loop,
                        'n_iter_block': claude_n_iter_block,
                        'ucb_c': claude_ucb_c,
                        'node_name': claude_node_name
                    }
                    with open(target_config, 'w') as f:
                        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
                    print(f"\033[93mmodified {target_config}: dataset='{llm_task_name}', n_epochs={claude_n_epochs}, data_augmentation_loop={claude_data_augmentation_loop}, n_iter_block={claude_n_iter_block}, ucb_c={claude_ucb_c}, node_name={claude_node_name}\033[0m")
            else:
                print(f"\033[93mpreserving {target_config} (resuming from iter {start_iteration})\033[0m")
                # Load existing config to get claude parameters
                with open(target_config, 'r') as f:
                    config_data = yaml.safe_load(f)
                claude_cfg = config_data.get('claude', {})
                claude_n_epochs = claude_cfg.get('n_epochs', 1)
                claude_data_augmentation_loop = claude_cfg.get('data_augmentation_loop', 100)
                claude_n_iter_block = claude_cfg.get('n_iter_block', 24)
                claude_ucb_c = claude_cfg.get('ucb_c', 1.414)
                claude_node_name = claude_cfg.get('node_name', 'a100')

        n_iter_block = claude_n_iter_block

        print(f"\033[94mCluster node: gpu_{claude_node_name}\033[0m")

        ucb_file = f"{root_dir}/{llm_task_name}_ucb_scores.txt"
        if start_iteration == 1 and not args.resume:
            # only delete UCB file when starting fresh (not resuming)
            if os.path.exists(ucb_file):
                os.remove(ucb_file)
                print(f"\033[93mdeleted {ucb_file}\033[0m")
        else:
            print(f"\033[93mpreserving {ucb_file} (resuming from iter {start_iteration})\033[0m")

        config_list = [llm_task_name]

        # Track if code was modified by Claude (starts False, set True after Claude modifies code)
        code_modified_by_claude = False
        # Check if code modifications are enabled (task contains 'code')
        code_changes_enabled = 'code' in task
        # Check if cluster execution is enabled (task contains 'cluster')
        cluster_enabled = 'cluster' in task
        if code_changes_enabled:
            print("\033[93mCode modifications ENABLED (task contains 'code')\033[0m")
            if cluster_enabled:
                print("\033[93mCluster execution ENABLED (task contains 'cluster')\033[0m")
            else:
                print("\033[90mCluster execution disabled (add 'cluster' to task to enable)\033[0m")
        else:
            print("\033[90mCode modifications disabled (add 'code' to task to enable)\033[0m")
    else:
        iteration_range = range(1, 2)
        code_modified_by_claude = False
        code_changes_enabled = False
        cluster_enabled = False  




    for config_file_ in config_list:
        print(" ")
        config_root = os.path.dirname(os.path.abspath(__file__)) + "/config"
        config_file, pre_folder = add_pre_folder(config_file_)



        if 'Claude' in task:
            root_dir = os.path.dirname(os.path.abspath(__file__))
            instruction_path = f"{root_dir}/{instruction_name}.md"
            analysis_path = f"{root_dir}/{llm_task_name}_analysis.md"
            memory_path = f"{root_dir}/{llm_task_name}_memory.md"

            # check instruction file exists
            if not os.path.exists(instruction_path):
                print(f"\033[91merror: instruction file not found: {instruction_path}\033[0m")
                print("\033[93mavailable instruction files:\033[0m")
                for f in os.listdir(root_dir):
                    if f.endswith('.md') and not f.startswith('analysis_') and not f.startswith('README'):
                        print(f"  - {f[:-3]}")
                continue

            # clear analysis and memory files at start (only if not resuming)
            if start_iteration == 1 and not args.resume:
                with open(analysis_path, 'w') as f:
                    f.write(f"# Experiment Log: {config_file_}\n\n")
                print(f"\033[93mcleared {analysis_path}\033[0m")
                # clear reasoning.log for Claude tasks
                reasoning_path = analysis_path.replace('_analysis.md', '_reasoning.log')
                open(reasoning_path, 'w').close()
                print(f"\033[93mcleared {reasoning_path}\033[0m")
                # initialize working memory file
                with open(memory_path, 'w') as f:
                    f.write(f"# Working Memory: {config_file_}\n\n")
                    f.write("## Knowledge Base (accumulated across all blocks)\n\n")
                    f.write("### Regime Comparison Table\n")
                    f.write("| Block | Regime | E/I | n_frames | n_neurons | n_types | noise | eff_rank | Best R² | Optimal lr_W | Optimal L1 | Degeneracy | Key finding |\n")
                    f.write("| ----- | ------ | --- | -------- | --------- | ------- | ----- | -------- | ------- | ------------ | ---------- | ---------- | ----------- |\n\n")
                    f.write("### Established Principles\n\n")
                    f.write("### Open Questions\n\n")
                    f.write("---\n\n")
                    f.write("## Previous Block Summary\n\n")
                    f.write("---\n\n")
                    f.write("## Current Block (Block 1)\n\n")
                    f.write("### Block Info\n\n")
                    f.write("### Hypothesis\n\n")
                    f.write("### Iterations This Block\n\n")
                    f.write("### Emerging Observations\n\n")
                print(f"\033[93mcleared {memory_path}\033[0m")
            else:
                print(f"\033[93mpreserving {analysis_path} (resuming from iter {start_iteration})\033[0m")
                print(f"\033[93mpreserving {memory_path} (resuming from iter {start_iteration})\033[0m")
                reasoning_path = analysis_path.replace('_analysis.md', '_reasoning.log')
                print(f"\033[93mpreserving {reasoning_path} (resuming from iter {start_iteration})\033[0m")
            print(f"\033[93m{instruction_name} ({n_iterations} iterations, starting at {start_iteration})\033[0m")

        root_dir = os.path.dirname(os.path.abspath(__file__))
        analysis_log_path = f"{root_dir}/{llm_task_name}_analysis.log"



        for iteration in iteration_range:


            if 'Claude' in task:
                print(f"\n\n\n\033[94miteration {iteration}/{n_iterations}: {config_file_} ===\033[0m")
                # block boundary: erase UCB at start of each n_iter_block-iteration block (except iter 1, already handled)
                if iteration > 1 and (iteration - 1) % n_iter_block == 0:
                    ucb_file = f"{root_dir}/{llm_task_name}_ucb_scores.txt"
                    if os.path.exists(ucb_file):
                        os.remove(ucb_file)
                        print(f"\033[93msimulation block boundary: deleted {ucb_file} (new simulation block)\\033[0m")

            # reload config to pick up any changes from previous iteration
            config = NeuralGraphConfig.from_yaml(f"{config_root}/{config_file}.yaml")
            config.dataset = pre_folder + config.dataset
            config.config_file = pre_folder + config_file_

            if device==[]:
                device = set_device(config.training.device)

            # open analysis.log for this iteration (append mode for test/plot to add metrics)
            log_file = open(analysis_log_path, 'w')

            if 'daemon' in task:
                # Daemon mode: submit job to cluster via config/new/ directory
                # Copy current config to config/new/, wait for cluster to process it
                # Use basename to flatten the path (config/signal/foo.yaml -> config/new/foo.yaml)
                config_filename = f"{os.path.basename(config_file)}.yaml"
                new_path = f"{config_root}/new/{config_filename}"
                processing_path = f"{config_root}/processing/{config_filename}"
                done_path = f"{config_root}/done/{config_filename}"

                # Ensure directories exist
                os.makedirs(f"{config_root}/new", exist_ok=True)
                os.makedirs(f"{config_root}/processing", exist_ok=True)
                os.makedirs(f"{config_root}/done", exist_ok=True)

                # Copy config to new/ to submit job
                source_config = f"{config_root}/{config_file}.yaml"
                shutil.copy2(source_config, new_path)
                submit_time = time.time()
                print(f"\033[93msubmitted to daemon: {new_path}\033[0m")

                # Wait for job to complete (file appears in done/)
                print(f"\033[93mWaiting for {config_filename} to be copied into config/done/ ...\033[0m")
                check_interval = 1 * 60  # 1 minute in seconds
                while True:
                    if os.path.exists(done_path):
                        print(f"\033[92mConfig file {config_filename} copied into config/done/\033[0m")
                        time.sleep(5)  # Wait 5s before deleting
                        os.remove(done_path)
                        print(f"\033[93mRemoved from done/: {config_filename}\033[0m")
                        break
                    # Also check if still in processing (job running)
                    if os.path.exists(processing_path):
                        print("  ... job still running (in processing/)")
                    elif os.path.exists(new_path):
                        print("  ... job queued (in new/)")
                    else:
                        print("  ... waiting for daemon to pick up job")
                    time.sleep(check_interval)
                print("\033[92mDaemon job completed, proceeding with Claude analysis...\033[0m")

                # Close log file (metrics will be read from cluster output)
                log_file.close()

                # Read analysis.log generated by daemon (if exists)
                # The daemon writes to the same log path, so metrics should be available
                if os.path.exists(analysis_log_path):
                    print(f"\033[92mMetrics available in: {analysis_log_path}\033[0m")

            else:
                # Local execution mode
                if "generate" in task:
                    erase = 'Claude' in task  # erase when iterating with claude
                    data_generate(
                        config,
                        device=device,
                        visualize=False,
                        run_vizualized=0,
                        style="color",
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
                    # Training execution: cluster > subprocess (code modified) > direct
                    use_subprocess = 'Claude' in task and code_changes_enabled and code_modified_by_claude

                    if cluster_enabled or use_subprocess:
                        # Paths for subprocess/cluster execution
                        config_path = f"{config_root}/{config_file}.yaml"
                        log_dir = f"{root_dir}/log/Claude_exploration/{instruction_name}"
                        os.makedirs(log_dir, exist_ok=True)
                        error_log_path = f"{log_dir}/training_output_latest.log"
                        error_details_path = f"{log_dir}/training_error_latest.log"

                        if cluster_enabled:
                            # Submit training job to cluster via SSH + bsub
                            if use_subprocess:
                                print("\033[93mcode modified by Claude - submitting training to cluster...\033[0m")
                            else:
                                print("\033[93msubmitting training to cluster....\033[0m")

                            # Build the python command
                            train_cmd = f"python train_signal_subprocess.py --config '{config_path}' --device cuda"
                            train_cmd += f" --log_file '{analysis_log_path}'"
                            train_cmd += f" --config_file '{config.config_file}'"
                            train_cmd += f" --error_log '{error_details_path}'"
                            if 'Claude' in task:
                                train_cmd += " --erase"

                            # Create temporary bash script that activates conda and runs training
                            cluster_script_path = f"{log_dir}/cluster_train.sh"
                            # Use absolute path for cluster (home directory on cluster)
                            cluster_home = "/groups/saalfeld/home/allierc"
                            cluster_root_dir = f"{cluster_home}/Graph/NeuralGraph"
                            conda_path = f"{cluster_home}/miniforge3/etc/profile.d/conda.sh"

                            # Update paths in train_cmd to use cluster paths
                            cluster_config_path = config_path.replace(root_dir, cluster_root_dir)
                            cluster_analysis_log = analysis_log_path.replace(root_dir, cluster_root_dir)
                            cluster_error_log = error_details_path.replace(root_dir, cluster_root_dir)

                            cluster_train_cmd = f"python train_signal_subprocess.py --config '{cluster_config_path}' --device cuda"
                            cluster_train_cmd += f" --log_file '{cluster_analysis_log}'"
                            cluster_train_cmd += f" --config_file '{config.config_file}'"
                            cluster_train_cmd += f" --error_log '{cluster_error_log}'"
                            if 'Claude' in task:
                                cluster_train_cmd += " --erase"

                            with open(cluster_script_path, 'w') as f:
                                f.write("#!/bin/bash\n")
                                f.write(f"cd {cluster_root_dir}\n")
                                f.write(f"conda run -n neural-graph {cluster_train_cmd}\n")
                            os.chmod(cluster_script_path, 0o755)

                            # Path to script on cluster
                            cluster_script = cluster_script_path.replace(root_dir, cluster_root_dir)

                            # Submit job to cluster via SSH to login1
                            # -W 6000 = 100 hours max wall time, -K makes bsub wait for job completion
                            ssh_cmd = f"ssh allierc@login1 \"cd {cluster_root_dir} && bsub -n 8 -gpu 'num=1' -q gpu_{claude_node_name} -W 6000 -K 'bash {cluster_script}'\""

                            print(f"\033[96msubmitting via SSH: {ssh_cmd}\033[0m")

                            result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)

                            if result.returncode != 0:
                                print("\033[91mCluster training failed:\033[0m")
                                print(f"stdout: {result.stdout}")
                                print(f"stderr: {result.stderr}")
                                if os.path.exists(error_details_path):
                                    with open(error_details_path, 'r') as f:
                                        print(f.read())
                                raise RuntimeError(f"Cluster training failed at iteration {iteration}")

                            print("\033[92mCluster training completed successfully\033[0m")
                            print(result.stdout)

                        else:
                            # Run training locally in subprocess (code was modified)
                            print("\033[93mcode modified by Claude - running training in subprocess...\033[0m")

                            train_script = os.path.join(root_dir, 'train_signal_subprocess.py')
                            train_cmd = [
                                sys.executable,  # Use same Python interpreter
                                '-u',  # Force unbuffered output for real-time streaming
                                train_script,
                                '--config', config_path,
                                '--device', str(device),
                                '--log_file', analysis_log_path,
                                '--config_file', config.config_file,
                                '--error_log', error_details_path,
                                '--erase'
                            ]

                            # Run training subprocess with repair loop
                            env = os.environ.copy()
                            env['PYTHONUNBUFFERED'] = '1'
                            env['TQDM_DISABLE'] = '1'  # Disable tqdm in subprocess (doesn't stream well)

                            # Code files that Claude might modify
                            code_files = [
                                'src/NeuralGraph/models/graph_trainer.py',
                                'src/NeuralGraph/generators/graph_data_generator.py',
                            ]

                            max_repair_attempts = 10
                            training_success = False
                            error_traceback = None

                            for repair_attempt in range(max_repair_attempts + 1):
                                process = subprocess.Popen(
                                    train_cmd,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    text=True,
                                    bufsize=1,
                                    env=env
                                )

                                # Capture all output for logging while also streaming to console
                                output_lines = []
                                with open(error_log_path, 'w') as output_file:
                                    for line in process.stdout:
                                        output_file.write(line)
                                        output_file.flush()
                                        output_lines.append(line.rstrip())
                                        # Filter: skip tqdm-like lines (progress bars)
                                        if '|' in line and '%' in line and 'it/s' in line:
                                            continue
                                        print(line, end='', flush=True)

                                process.wait()

                                if process.returncode == 0:
                                    training_success = True
                                    break

                                # Training failed - capture error info
                                error_traceback = '\n'.join(output_lines[-50:])  # Last 50 lines

                                if repair_attempt == 0:
                                    print(f"\033[91m\ntraining subprocess failed with code {process.returncode}\033[0m")
                                    print("\033[93mthis may indicate a code modification error.\033[0m\n")

                                    # Show last 20 lines of output for context
                                    print("\033[93mLast 20 lines of output:\033[0m")
                                    print("-" * 80)
                                    for line in output_lines[-20:]:
                                        print(line)
                                    print("-" * 80)

                                    # Show paths to log files
                                    print(f"\nFull output logged to: {error_log_path}")
                                    if os.path.exists(error_details_path):
                                        print(f"Error details logged to: {error_details_path}")
                                        try:
                                            with open(error_details_path, 'r') as f:
                                                error_details = f.read()
                                            if error_details.strip():
                                                print("\n\033[91mDetailed error information:\033[0m")
                                                print(error_details)
                                                error_traceback = error_details + '\n' + error_traceback
                                        except Exception as e:
                                            print(f"Could not read error details: {e}")

                                # Check if code was modified (only attempt repair for code errors)
                                modified_code = get_modified_code_files(root_dir, code_files) if is_git_repo(root_dir) else []

                                if not modified_code and repair_attempt == 0:
                                    print("\033[93mNo code modifications detected - skipping repair attempts\033[0m")
                                    break

                                # Attempt repair only if code was modified
                                if repair_attempt < max_repair_attempts and modified_code:
                                    print(f"\033[93mRepair attempt {repair_attempt + 1}/{max_repair_attempts}: Asking Claude to fix the code error...\033[0m")

                                    repair_prompt = f"""TRAINING CRASHED - Please fix the code error.

Attempt {repair_attempt + 1}/{max_repair_attempts}

Error traceback:
```
{error_traceback[-3000:] if error_traceback else 'No traceback available'}
```

Modified code files that may contain the bug:
{chr(10).join(f'- {root_dir}/{f}' for f in modified_code)}

Instructions:
1. Read the error traceback carefully
2. Identify the bug in the modified code
3. Fix the bug using the Edit tool
4. Do NOT make other changes, only fix the crash

If you cannot fix it, say "CANNOT_FIX" and explain why."""

                                    repair_cmd = [
                                        'claude',
                                        '-p', repair_prompt,
                                        '--output-format', 'text',
                                        '--max-turns', '10',
                                        '--allowedTools', 'Read', 'Edit', 'Write'
                                    ]

                                    repair_result = subprocess.run(repair_cmd, cwd=root_dir, capture_output=True, text=True)
                                    repair_output = repair_result.stdout

                                    if 'CANNOT_FIX' in repair_output:
                                        print("\033[91mClaude cannot fix the error\033[0m")
                                        break

                                    print(f"\033[92mRepair attempt {repair_attempt + 1} complete, retrying training...\033[0m")

                            # If still failing after all attempts, rollback and skip iteration
                            if not training_success:
                                print("\033[91mAll repair attempts failed - rolling back code changes\033[0m")

                                # Rollback modified files using git
                                if is_git_repo(root_dir):
                                    for file_path in code_files:
                                        try:
                                            subprocess.run(['git', 'checkout', 'HEAD', '--', file_path],
                                                          cwd=root_dir, capture_output=True, timeout=10)
                                        except:
                                            pass
                                    print("\033[93mRolled back code to last working state\033[0m")

                                # Log failed modification to memory
                                if os.path.exists(memory_path):
                                    with open(memory_path, 'a') as f:
                                        f.write(f"\n### Failed Code Modification (Iter {iteration})\n")
                                        f.write(f"Error: {error_traceback[-500:] if error_traceback else 'Unknown'}\n")
                                        f.write("**DO NOT retry this modification**\n\n")

                                continue  # Skip to next iteration

                            print("\033[92mtraining subprocess completed successfully\033[0m")
                    else:
                        # No cluster and no code modifications - run training directly (fastest)
                        data_train(
                            config=config,
                            erase='Claude' in task,  # erase old models when iterating with Claude
                            best_model=best_model,
                            style = 'color',
                            device=device,
                            log_file=log_file
                        )

                if "test" in task:

                    config.simulation.noise_model_level = 0.0

                    if 'fly' in config_file_:
                        config.simulation.visual_input_type = 'optical_flow'   #'DAVIS'

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
                        new_params = None,
                        log_file=log_file,
                    )

                if 'plot' in task:
                    folder_name = './log/' + pre_folder + '/tmp_results/'
                    os.makedirs(folder_name, exist_ok=True)
                    data_plot(config=config, config_file=config_file, epoch_list=['best'], style='color', extended='plots', device=device, apply_weight_correction=True, log_file=log_file)

                log_file.close()

            if 'Claude' in task:

                block_number = (iteration - 1) // n_iter_block + 1
                iter_in_block = (iteration - 1) % n_iter_block + 1
                is_block_end = iter_in_block == n_iter_block

                exploration_dir = f"{root_dir}/log/Claude_exploration/{instruction_name}"
                artifact_paths = save_exploration_artifacts(
                    root_dir, exploration_dir, config, config_file_, pre_folder, iteration,
                    iter_in_block=iter_in_block, block_number=block_number
                )
                tree_save_dir = artifact_paths['tree_save_dir']
                protocol_save_dir = artifact_paths['protocol_save_dir']
                activity_path = artifact_paths['activity_path']

                # claude analysis: reads activity.png and analysis.log, updates config per experiment protocol
                config_path = f"{root_dir}/config/{pre_folder}{config_file_}.yaml"
                ucb_path = f"{root_dir}/{llm_task_name}_ucb_scores.txt"

                # read ucb_c from config
                with open(config_path, 'r') as f:
                    raw_config = yaml.safe_load(f)
                ucb_c = raw_config.get('claude', {}).get('ucb_c', 1.414)

                # compute UCB scores for Claude to read
                compute_ucb_scores(analysis_path, ucb_path, c=ucb_c,
                                   current_log_path=analysis_log_path,
                                   current_iteration=iteration,
                                   block_size=n_iter_block)
                print(f"\033[92mUCB scores computed (c={ucb_c}): {ucb_path}\033[0m")

                # check files are ready (generated by data_generate above)
                time.sleep(2)  # pause to ensure files are written
                if not os.path.exists(activity_path):
                    print(f"\033[91merror: activity.png not found at {activity_path}\033[0m")
                    continue
                if not os.path.exists(analysis_log_path):
                    print(f"\033[91merror: analysis.log not found at {analysis_log_path}\033[0m")
                    continue
                if not os.path.exists(ucb_path):
                    print(f"\033[91merror: ucb_scores.txt not found at {ucb_path}\033[0m")
                    continue
                print("\033[92mfiles ready: activity.png, analysis.log, ucb_scores.txt\033[0m")

                # call Claude CLI for analysis
                print("\033[93mClaude analysis...\033[0m")

                claude_prompt = f"""Iteration {iteration}/{n_iterations}
Block info: block {block_number}, iteration {iter_in_block}/{n_iter_block} within block
{">>> BLOCK END <<<" if is_block_end else ""}

Instructions (follow all instructions): {instruction_path}
Working memory: {memory_path}
Full log (append only): {analysis_path}
Activity image: {activity_path}
Metrics log: {analysis_log_path}
UCB scores: {ucb_path}
Current config: {config_path}"""

                claude_cmd = [
                    'claude',
                    '-p', claude_prompt,
                    '--output-format', 'text',
                    '--max-turns', '500',
                    '--allowedTools',
                    'Read', 'Edit', 'Write'
                ]

                # run with real-time output streaming and token expiry detection
                output_lines = []
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
                    output_lines.append(line)

                process.wait()

                # check for OAuth token expiration error
                output_text = ''.join(output_lines)
                if 'OAuth token has expired' in output_text or 'authentication_error' in output_text:
                    print(f"\n\033[91m{'='*60}\033[0m")
                    print(f"\033[91mOAuth token expired at iteration {iteration}\033[0m")
                    print("\033[93mTo resume:\033[0m")
                    print("\033[93m  1. Run: claude /login\033[0m")
                    print(f"\033[93m  2. Then: python GNN_Main.py -o {task} {config_file_} start={iteration}\033[0m")
                    print(f"\033[91m{'='*60}\033[0m")
                    raise SystemExit(1)

                # Save Claude's terminal output to reasoning log (separate from analysis.md)
                reasoning_log_path = analysis_path.replace('_analysis.md', '_reasoning.log')
                if output_text.strip():
                    with open(reasoning_log_path, 'a') as f:
                        f.write(f"\n{'='*60}\n")
                        f.write(f"=== Iteration {iteration} ===\n")
                        f.write(f"{'='*60}\n")
                        f.write(output_text.strip())
                        f.write("\n\n")

                # Git tracking: commit any code modifications made by Claude (only if code changes enabled)
                if code_changes_enabled:
                    if is_git_repo(root_dir):
                        print("\n\033[96mchecking for code modifications to commit\033[0m")
                        git_results = track_code_modifications(
                            root_dir=root_dir,
                            iteration=iteration,
                            analysis_path=analysis_path,
                            reasoning_path=reasoning_log_path
                        )

                        if git_results:
                            for file_path, success, message in git_results:
                                if success:
                                    print(f"\033[92m✓ Git: {message}\033[0m")
                                    # Set flag so next iteration uses subprocess
                                    code_modified_by_claude = True
                                else:
                                    print(f"\033[93m⚠ Git: {message}\033[0m")
                        else:
                            print("\033[90m  No code modifications detected\033[0m")
                    else:
                        # Not a git repo - check for code modifications directly
                        tracked_code_files = ['src/NeuralGraph/models/graph_trainer.py']
                        modified_files = get_modified_code_files(root_dir, tracked_code_files)
                        if modified_files:
                            code_modified_by_claude = True
                            print(f"\033[93m  Code modified (no git): {modified_files}\033[0m")
                        if iteration == 1:
                            print("\033[90m  Not a git repository - code modifications will not be version controlled\033[0m")

                # save instruction file at first iteration of each block
                if iter_in_block == 1:
                    dst_instruction = f"{protocol_save_dir}/block_{block_number:03d}.md"
                    if os.path.exists(instruction_path):
                        shutil.copy2(instruction_path, dst_instruction)

                # save memory file at end of each block (after Claude updates it)
                if is_block_end:
                    memory_save_dir = f"{exploration_dir}/memory"
                    os.makedirs(memory_save_dir, exist_ok=True)
                    dst_memory = f"{memory_save_dir}/block_{block_number:03d}_memory.md"
                    if os.path.exists(memory_path):
                        shutil.copy2(memory_path, dst_memory)
                        print(f"\033[92msaved memory snapshot: {dst_memory}\033[0m")

                # recompute UCB scores after Claude to pick up mutations from analysis markdown
                compute_ucb_scores(analysis_path, ucb_path, c=ucb_c,
                                   current_log_path=analysis_log_path,
                                   current_iteration=iteration,
                                   block_size=n_iter_block)

                # generate UCB tree visualization from ucb_scores.txt
                # For block 1: save every iteration; for block 2+: save only at block end
                should_save_tree = (block_number == 1) or is_block_end
                if should_save_tree:
                    ucb_tree_path = f"{tree_save_dir}/ucb_tree_iter_{iteration:03d}.png"
                    nodes = parse_ucb_scores(ucb_path)
                else:
                    nodes = None  # Skip tree generation for intermediate iterations in blocks > 0
                if nodes:
                    # get simulation info from config for tree annotation
                    sim_info = f"n_neurons={config.simulation.n_neurons}, n_frames={config.simulation.n_frames}"
                    sim_info += f", time_step={config.training.time_step}"
                    if hasattr(config.training, 'recurrent_training'):
                        sim_info += f", recurrent={config.training.recurrent_training}"
                    sim_info += f", connectivity_type={config.simulation.connectivity_type}"
                    if hasattr(config.simulation, 'Dale_law'):
                        sim_info += f", Dale_law={config.simulation.Dale_law}"
                        # Only show E/I ratio if Dale_law is True, otherwise show NA
                        if config.simulation.Dale_law and hasattr(config.simulation, 'Dale_law_factor'):
                            sim_info += f", E/I={config.simulation.Dale_law_factor}"
                        else:
                            sim_info += ", E/I=NA"
                    if hasattr(config.simulation, 'noise_model_level'):
                        sim_info += f", noise_model_level={config.simulation.noise_model_level}"
                    if config.simulation.connectivity_type == 'low_rank' and hasattr(config.simulation, 'connectivity_rank'):
                        sim_info += f", connectivity_rank={config.simulation.connectivity_rank}"
                    if hasattr(config.simulation, 'connectivity_filling_factor'):
                        sim_info += f", filling_factor={config.simulation.connectivity_filling_factor}"
                    if hasattr(config.simulation, 'n_neuron_types'):
                        sim_info += f", n_neuron_types={config.simulation.n_neuron_types}"
                    # Extract g (network gain) from params array if available
                    if hasattr(config.simulation, 'params') and len(config.simulation.params) > 0:
                        g_value = config.simulation.params[0][2]  # g is third column (index 2)
                        sim_info += f", g={g_value}"
                    # Add low_rank training parameters
                    if hasattr(config.training, 'low_rank_factorization') and config.training.low_rank_factorization:
                        sim_info += ", low_rank_factorization=True"
                        if hasattr(config.training, 'low_rank'):
                            sim_info += f", low_rank={config.training.low_rank}"

                    plot_ucb_tree(nodes, ucb_tree_path,
                                  title=f"UCB Tree - Iter {iteration}",
                                  simulation_info=sim_info)

# bsub -n 8 -gpu "num=1" -q gpu_h100 -Is -W 6000 "python GNN_Daemon.py -o generate_train_test_plot signal_landscape_Claude"