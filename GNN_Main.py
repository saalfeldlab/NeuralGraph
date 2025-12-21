import matplotlib
matplotlib.use('Agg')  # set non-interactive backend before other imports
import matplotlib.pyplot as plt
import argparse
import glob
import os
import shutil
import subprocess
import time
import yaml

# redirect PyTorch JIT cache to /scratch instead of /tmp (per IT request)
if os.path.isdir('/scratch'):
    os.environ['TMPDIR'] = '/scratch/allierc'
    os.makedirs('/scratch/allierc', exist_ok=True)


from NeuralGraph.config import NeuralGraphConfig
from NeuralGraph.generators.graph_data_generator import data_generate
from NeuralGraph.models.graph_trainer import data_train, data_test, data_train_INR
from NeuralGraph.models.exploration_tree import compute_ucb_scores, parse_experiment_log, build_tree_structure, plot_data_exploration
from NeuralGraph.models.plot_exploration_tree import parse_ucb_scores, plot_ucb_tree
from NeuralGraph.models.utils import save_exploration_artifacts
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
        task_params = {}
        for arg in args.option[2:]:
            if '=' in arg:
                key, value = arg.split('=', 1)
                task_params[key] = int(value) if value.isdigit() else value
    else:
        best_model = ''
        task = 'generate_train_test_plot_Claude'  # 'train', 'test', 'generate', 'plot', 'train_NGP', 'train_INR', 'Claude'
        config_list = ['signal_chaotic_1']
        task_params = {'iterations': 1024}



    # resume support: start_iteration parameter (default 1)
    start_iteration = 1




    n_iterations = task_params.get('iterations', 5)
    base_config_name = config_list[0] if config_list else 'signal'
    experiment_name = task_params.get('experiment', f'experiment_{base_config_name}')
    llm_task_name = task_params.get('llm_task', f'{base_config_name}_Claude')




    if 'Claude' in task:
        iteration_range = range(start_iteration, n_iterations + 1)
        if start_iteration > 1:
            print(f"\033[93mResuming from iteration {start_iteration}\033[0m")
        root_dir = os.path.dirname(os.path.abspath(__file__))
        config_root = root_dir + "/config"


        for cfg in config_list:
            cfg_file, pre = add_pre_folder(cfg)
            source_config = f"{config_root}/{pre}{cfg}.yaml"
            target_config = f"{config_root}/{pre}{llm_task_name}.yaml"
            if os.path.exists(source_config):
                shutil.copy2(source_config, target_config)
                print(f"\033[93mcopied {source_config} -> {target_config}\033[0m")
                with open(target_config, 'r') as f:
                    config_data = yaml.safe_load(f)
                claude_cfg = config_data.get('claude', {})
                claude_n_epochs = claude_cfg.get('n_epochs', 1)
                claude_data_augmentation_loop = claude_cfg.get('data_augmentation_loop', 100)
                claude_n_iter_block = claude_cfg.get('n_iter_block', 24)
                config_data['dataset'] = llm_task_name
                config_data['training']['n_epochs'] = claude_n_epochs
                config_data['training']['data_augmentation_loop'] = claude_data_augmentation_loop
                config_data['description'] = 'designed by Claude'
                config_data['claude'] = {
                    'n_epochs': claude_n_epochs,
                    'data_augmentation_loop': claude_data_augmentation_loop,
                    'n_iter_block': claude_n_iter_block
                }
                with open(target_config, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
                print(f"\033[93mmodified {target_config}: dataset='{llm_task_name}', n_epochs={claude_n_epochs}, data_augmentation_loop={claude_data_augmentation_loop}, n_iter_block={claude_n_iter_block}\033[0m")

        n_iter_block = claude_n_iter_block

        ucb_file = f"{root_dir}/{llm_task_name}_ucb_scores.txt"
        if start_iteration == 1:
            # only delete UCB file when starting fresh (not resuming)
            if os.path.exists(ucb_file):
                os.remove(ucb_file)
                print(f"\033[93mdeleted {ucb_file}\033[0m")
        else:
            print(f"\033[93mpreserving {ucb_file} (resuming from iter {start_iteration})\033[0m")

        config_list = [llm_task_name]
    else:

        iteration_range = range(1, 2)  




    for config_file_ in config_list:
        print(" ")
        config_root = os.path.dirname(os.path.abspath(__file__)) + "/config"
        config_file, pre_folder = add_pre_folder(config_file_)



        if 'Claude' in task:
            root_dir = os.path.dirname(os.path.abspath(__file__))
            experiment_path = f"{root_dir}/{experiment_name}.md"
            analysis_path = f"{root_dir}/{llm_task_name}_analysis.md"
            memory_path = f"{root_dir}/{llm_task_name}_memory.md"

            # check experiment file exists
            if not os.path.exists(experiment_path):
                print(f"\033[91merror: experiment file not found: {experiment_path}\033[0m")
                print(f"\033[93mavailable experiment files:\033[0m")
                for f in os.listdir(root_dir):
                    if f.endswith('.md') and not f.startswith('analysis_') and not f.startswith('README'):
                        print(f"  - {f[:-3]}")
                continue

            # clear analysis and memory files at start (only if not resuming)
            if start_iteration == 1:
                with open(analysis_path, 'w') as f:
                    f.write(f"# Experiment Log: {config_file_}\n\n")
                print(f"\033[93mcleared {analysis_path}\033[0m")
                # initialize working memory file
                with open(memory_path, 'w') as f:
                    f.write(f"# Working Memory: {config_file_}\n\n")
                    f.write("## Knowledge Base (accumulated across all blocks)\n\n")
                    f.write("### Regime Comparison Table\n")
                    f.write("| Block | Regime | Best R² | Optimal lr_W | Optimal L1 | Key finding |\n")
                    f.write("|-------|--------|---------|--------------|------------|-------------|\n\n")
                    f.write("### Coverage Table\n")
                    f.write("| connectivity_type | Dale_law=False | Dale_law=True |\n")
                    f.write("|-------------------|----------------|---------------|\n")
                    f.write("| chaotic           | ?              | ?             |\n")
                    f.write("| low_rank=20       | ?              | ?             |\n")
                    f.write("| low_rank=50       | ?              | ?             |\n\n")
                    f.write("### Established Principles\n")
                    f.write("- (none yet)\n\n")
                    f.write("### Open Questions\n")
                    f.write("- (none yet)\n\n")
                    f.write("---\n\n")
                    f.write("## Previous Block Summary\n")
                    f.write("(no previous block)\n\n")
                    f.write("---\n\n")
                    f.write("## Current Block (Block 1)\n\n")
                    f.write("### Block Info\n")
                    f.write("Simulation: (to be filled by first iteration)\n")
                    f.write("Iterations: 1 to n_iter_block\n\n")
                    f.write("### Hypothesis\n")
                    f.write("(first block - establishing baseline)\n\n")
                    f.write("### Iterations This Block\n\n")
                    f.write("### Emerging Observations\n")
                    f.write("(none yet)\n")
                print(f"\033[93mcleared {memory_path}\033[0m")
            else:
                print(f"\033[93mpreserving {analysis_path} (resuming from iter {start_iteration})\033[0m")
                print(f"\033[93mpreserving {memory_path} (resuming from iter {start_iteration})\033[0m")
            print(f"\033[93m{experiment_name} ({n_iterations} iterations, starting at {start_iteration})\033[0m")

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
                    erase='Claude' in task,  # erase old models when iterating with Claude
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

                block_number = (iteration - 1) // n_iter_block + 1
                iter_in_block = (iteration - 1) % n_iter_block + 1
                is_block_end = iter_in_block == n_iter_block

                exploration_dir = f"{root_dir}/log/Claude_exploration/{experiment_name}"
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

                # compute UCB scores for Claude to read
                compute_ucb_scores(analysis_path, ucb_path,
                                   current_log_path=analysis_log_path,
                                   current_iteration=iteration,
                                   block_size=n_iter_block)
                print(f"\033[92mUCB scores computed: {ucb_path}\033[0m")

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
                print(f"\033[92mfiles ready: activity.png, analysis.log, ucb_scores.txt\033[0m")

                # call Claude CLI for analysis
                print(f"\033[93mClaude analysis...\033[0m")

                # check if using memory-based protocol (v2 with working memory)
                use_memory_file = os.path.exists(memory_path) and 'memory' in open(experiment_path).read().lower()

                # detect first iteration of block for special instructions
                is_first_iter = (iter_in_block == 1)
                is_first_ever = (iteration == 1)

                if use_memory_file:
                    # build first-iteration instructions
                    first_iter_note = ""
                    if is_first_ever:
                        first_iter_note = """
>>> FIRST ITERATION: Memory file is empty. Fill in Block Info with simulation params from config. Use parent=root, baseline config. <<<"""
                    elif is_first_iter:
                        first_iter_note = """
>>> NEW BLOCK START: Use parent=root for first iteration of new block. Fill Block Info from config. <<<"""

                    claude_prompt = f"""Iteration {iteration}/{n_iterations}: Parameter study.

Block info: block {block_number}, iteration {iter_in_block}/{n_iter_block} within block
{">>> BLOCK END: Last iteration of block! See Block End Checklist in protocol. <<<" if is_block_end else ""}{first_iter_note}

FILES TO READ:
1. Working memory (your persistent knowledge): {memory_path}
2. Activity image: {activity_path}
3. Metrics log: {analysis_log_path}
4. UCB scores: {ucb_path}
5. Current config: {config_path}
6. Protocol (for rules): {experiment_path}

FILES TO WRITE:
1. Append iteration log to: {analysis_path} (full record, never read)
2. Update working memory: {memory_path}
   - Add iteration to "Iterations This Block"
   {"- First iter: fill in Block Info (simulation params from config)" if is_first_iter else ""}
   {"- BLOCK END: Update Knowledge Base, Regime/Coverage tables, replace Previous Block Summary, clear Current Block, write new Hypothesis" if is_block_end else ""}
3. Edit config for next iteration: {config_path}
{"4. Evaluate and possibly edit protocol: " + experiment_path if is_block_end else ""}

Config file: {config_file_}"""
                else:
                    # original prompt (backward compatible - no memory file)
                    claude_prompt = f"""Iteration {iteration}/{n_iterations}: Parameter study.

Block info: block {block_number}, iteration {iter_in_block}/{n_iter_block} within block
{">>> BLOCK END: Last iteration of block! Write summary, edit protocol, change simulation params for next block. <<<" if is_block_end else ""}

1. Read activity image: {activity_path}
2. Read analysis log: {analysis_log_path}
3. Read protocol: {experiment_path}
4. Read UCB scores: {ucb_path}
5. Read current config: {config_path}
6. Append to {analysis_path} using log format from protocol
7. Edit {config_path} to explore next parameter combination per protocol

Config file: {config_file_}"""

                claude_cmd = [
                    'claude',
                    '-p', claude_prompt,
                    '--output-format', 'text',
                    '--max-turns', '20',
                    '--allowedTools',
                    'Read', 'Edit'
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
                    print(f"\033[93mTo resume:\033[0m")
                    print(f"\033[93m  1. Run: claude /login\033[0m")
                    print(f"\033[93m  2. Then: python GNN_Main.py -o {task} {config_file_} start={iteration}\033[0m")
                    print(f"\033[91m{'='*60}\033[0m")
                    raise SystemExit(1)

                # save protocol file at first iteration of each block
                if iter_in_block == 1:
                    dst_protocol = f"{protocol_save_dir}/block_{block_number:03d}.md"
                    if os.path.exists(experiment_path):
                        shutil.copy2(experiment_path, dst_protocol)

                # save memory file at end of each block (after Claude updates it)
                if is_block_end and use_memory_file:
                    memory_save_dir = f"{exploration_dir}/memory"
                    os.makedirs(memory_save_dir, exist_ok=True)
                    dst_memory = f"{memory_save_dir}/block_{block_number:03d}_memory.md"
                    if os.path.exists(memory_path):
                        shutil.copy2(memory_path, dst_memory)
                        print(f"\033[92msaved memory snapshot: {dst_memory}\033[0m")

                # recompute UCB scores after Claude to pick up mutations from analysis markdown
                compute_ucb_scores(analysis_path, ucb_path,
                                   current_log_path=analysis_log_path,
                                   current_iteration=iteration,
                                   block_size=n_iter_block)

                # generate UCB tree visualization from ucb_scores.txt
                ucb_tree_path = f"{tree_save_dir}/ucb_tree_iter_{iteration:03d}.png"
                nodes = parse_ucb_scores(ucb_path)
                if nodes:
                    # get simulation info from config for tree annotation
                    sim_info = f"connectivity_type={config.simulation.connectivity_type}"
                    if hasattr(config.simulation, 'Dale_law'):
                        sim_info += f", Dale_law={config.simulation.Dale_law}"
                    if hasattr(config.simulation, 'noise_model_level'):
                        sim_info += f", noise_model_level={config.simulation.noise_model_level}"
                    if config.simulation.connectivity_type == 'low_rank' and hasattr(config.simulation, 'connectivity_rank'):
                        sim_info += f", connectivity_rank={config.simulation.connectivity_rank}"

                    plot_ucb_tree(nodes, ucb_tree_path,
                                  title=f"UCB Tree - Iter {iteration}",
                                  simulation_info=sim_info)
