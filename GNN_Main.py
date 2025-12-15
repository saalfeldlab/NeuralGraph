import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before other imports
import argparse
import os
import subprocess

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
    else:
        best_model = ''
        task = 'Claude_experiment'  #, 'train', 'test', 'generate', 'plot', 'train_NGP', 'train_INR', 'Claude_experiment'

        config_list = ['signal_chaotic_1']




    for config_file_ in config_list:
        print(" ")
        config_root = os.path.dirname(os.path.abspath(__file__)) + "/config"
        config_file, pre_folder = add_pre_folder(config_file_)
        config = NeuralGraphConfig.from_yaml(f"{config_root}/{config_file}.yaml")
        config.dataset = pre_folder + config.dataset
        config.config_file = pre_folder + config_file_
        # print(f"\033[92mconfig_file:  {config.config_file}\033[0m")
        
        if device==[]:
            device = set_device(config.training.device)
            # print(f"\033[92mdevice:  {device}\033[0m")

        if "generate" in task:
            data_generate(
                config,
                device=device,
                visualize=False,
                run_vizualized=0,
                style="black color",
                alpha=1,
                erase=False,
                bSave=True,
                step=2
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
            )

        if 'plot' in task:
            folder_name = './log/' + pre_folder + '/tmp_results/'
            os.makedirs(folder_name, exist_ok=True)
            data_plot(config=config, config_file=config_file, epoch_list=['best'], style='black color', extended='plots', device=device, apply_weight_correction=True)

        if 'Claude_experiment' in task:
            # Automated experiment workflow for Claude analysis - 20 iterations
            root_dir = os.path.dirname(os.path.abspath(__file__))
            activity_path = f"{root_dir}/graphs_data/{pre_folder}{config.dataset}/activity.png"
            config_path = f"{root_dir}/config/{pre_folder}{config_file_}.yaml"
            analysis_path = f"{root_dir}/analysis.md"
            experiment_path = f"{root_dir}/experiment.md"

            # Clear analysis.md at start
            with open(analysis_path, 'w') as f:
                f.write(f"# Experiment Log: {config_file_}\n\n")
            print(f"\033[93mcleared analysis.md\033[0m")

            for iteration in range(1, 21):
                print(f"\033[94miteration {iteration}/20: {config_file_} ===\033[0m")


                # Step 1: Generate data
                print(f"\033[93mstep 1: generating data...\033[0m")

                # Reload config to pick up any changes from previous iteration
                config = NeuralGraphConfig.from_yaml(f"{config_root}/{config_file}.yaml")
                config.dataset = pre_folder + config.dataset
                config.config_file = pre_folder + config_file_

                data_generate(
                    config,
                    device=device,
                    visualize=False,
                    run_vizualized=0,
                    style="black color",
                    alpha=1,
                    erase=True,  # Erase old data to regenerate with new config
                    bSave=True,
                    step=2
                )

                # Step 2: Call Claude CLI for analysis
                print(f"\n\033[93mstep 2: Claude analysis...\033[0m")

                claude_prompt = f"""Iteration {iteration}/20: Analyze neural activity.

1. Read activity image: {activity_path}
2. Read analysis protocol: {experiment_path}
3. Classify dynamics and determine what config change would help (per experiment.md guidelines)
4. Append to {analysis_path}:
   ## Iter {iteration}: [Steady State/Chaotic]
   Observation: [one line]
   Change: [parameter: old -> new] or None

5. If not chaotic, edit {config_path} based on experiment.md recommendations.

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

                # Print latest analysis entry in yellow
                try:
                    with open(analysis_path, 'r') as f:
                        lines = f.readlines()
                    # Find lines containing current iteration
                    for line in lines:
                        if f'Iter {iteration}' in line:
                            print(f"\033[93m{'─'*60}\033[0m")
                            print(f"\033[93m{line.strip()}\033[0m")
                            print(f"\033[93m{'─'*60}\033[0m")
                            break
                except:
                    pass

                print(f"\033[92m--- Iteration {iteration} complete ---\033[0m")



                  


