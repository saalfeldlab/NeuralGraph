"""
GNN Recurrent Training - Recurrent time_step optimization experiments.

This script runs GNN training with recurrent training mode enabled.
It supports both explicit Euler recurrent training and Neural ODE training.

Recurrent training (time_step > 1):
    - Model predicts T steps ahead using its own predictions
    - Loss computed at step T against ground truth
    - Gradients backpropagate through T steps

Neural ODE training (alternative):
    - Uses continuous ODE solver (torchdiffeq)
    - Adaptive step size based on rtol/atol
    - Adjoint method for memory-efficient backprop

Usage:
    python GNN_recurrent.py                                         # run full pipeline
    python GNN_recurrent.py -o generate signal_N2_recurrent_1       # generate data only
    python GNN_recurrent.py -o train signal_N2_recurrent_1          # train only
    python GNN_recurrent.py -o test signal_N2_recurrent_1           # test only
    python GNN_recurrent.py -o plot signal_N2_recurrent_1           # plot only

Config parameters (in training section):
    recurrent_training: True        # enable recurrent mode
    neural_ODE_training: False      # alternative: Neural ODE mode
    time_step: 4                    # recurrence depth (1, 4, 16, 32, 64)
    noise_recurrent_level: 0.0      # noise at each recurrent step

See instruction_signal_N2_recurrent_1.md for experiment protocol.
"""

import matplotlib
matplotlib.use('Agg')  # set non-interactive backend before other imports
import argparse
import os

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

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    parser = argparse.ArgumentParser(description="NeuralGraph Recurrent Training")
    parser.add_argument(
        "-o", "--option", nargs="+", help="option that takes multiple values"
    )

    print()
    print("=" * 80)
    print("GNN Recurrent Training - Time Step Optimization")
    print("=" * 80)

    device = []
    args = parser.parse_args()

    if args.option:
        print(f"Options: {args.option}")
    if args.option is not None:
        task = args.option[0]
        config_list = [args.option[1]]
        if len(args.option) > 2:
            best_model = args.option[2]
        else:
            best_model = None
    else:
        best_model = ''
        task = 'generate_train_test_plot'
        config_list = ['signal_N2_recurrent_1']

    for config_file_ in config_list:
        print()
        config_root = os.path.dirname(os.path.abspath(__file__)) + "/config"
        config_file, pre_folder = add_pre_folder(config_file_)

        # load config
        config = NeuralGraphConfig.from_yaml(f"{config_root}/{config_file}.yaml")
        config.config_file = config_file
        config.dataset = config_file  # e.g., 'signal/signal_N2_recurrent_1'

        if device == []:
            device = set_device(config.training.device)

        log_dir = f'./log/{config_file}'
        graphs_dir = f'./graphs_data/{config_file}'

        # Print recurrent training configuration
        print("-" * 80)
        print("Recurrent Training Configuration")
        print("-" * 80)
        recurrent_training = getattr(config.training, 'recurrent_training', False)
        neural_ODE_training = getattr(config.training, 'neural_ODE_training', False)
        time_step = getattr(config.training, 'time_step', 1)
        noise_recurrent = getattr(config.training, 'noise_recurrent_level', 0.0)

        if recurrent_training:
            print(f"  Mode: Recurrent Training (explicit Euler)")
            print(f"  time_step: {time_step}")
            print(f"  noise_recurrent_level: {noise_recurrent}")
        elif neural_ODE_training:
            ode_method = getattr(config.training, 'ode_method', 'dopri5')
            ode_adjoint = getattr(config.training, 'ode_adjoint', True)
            print(f"  Mode: Neural ODE Training")
            print(f"  time_step: {time_step}")
            print(f"  ode_method: {ode_method}")
            print(f"  ode_adjoint: {ode_adjoint}")
        else:
            print(f"  Mode: Single-step prediction (no recurrence)")
        print()

        if "generate" in task:
            # Generate synthetic neural activity data
            print()
            print("-" * 80)
            print("STEP 1: GENERATE - Simulating neural activity")
            print("-" * 80)
            print(f"  Simulating {config.simulation.n_neurons} neurons, {config.simulation.n_neuron_types} types")
            print(f"  Generating {config.simulation.n_frames} time frames")
            print(f"  Output: {graphs_dir}/")
            print()
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
            # Train the GNN with recurrent training
            print()
            print("-" * 80)
            print("STEP 2: TRAIN - Training GNN with recurrent mode")
            print("-" * 80)
            print(f"  Training for {config.training.n_epochs} epochs, {config.training.n_runs} run(s)")
            print(f"  Recurrent time_step: {time_step}")
            print(f"  Models: {log_dir}/models/")
            print(f"  Training plots: {log_dir}/tmp_training")
            print()
            data_train(
                config=config,
                erase=True,
                best_model=best_model,
                style='black',
                device=device
            )

        if "test" in task:
            # Test the trained GNN model
            print()
            print("-" * 80)
            print("STEP 3: TEST - Evaluating trained model")
            print("-" * 80)
            print(f"  Testing prediction accuracy and rollout inference")
            print(f"  Output: {log_dir}/results/")
            print()
            config.training.noise_model_level = 0.0

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
                new_params=None,
            )

        if 'plot' in task:
            # Generate plots
            print()
            print("-" * 80)
            print("STEP 4: PLOT - Generating result figures")
            print("-" * 80)
            print(f"  Connectivity comparison (learned vs true)")
            print(f"  Embedding visualization")
            print(f"  MLP function plots")
            print(f"  Output: {log_dir}/results/")
            print()
            folder_name = './log/' + pre_folder + '/tmp_results/'
            os.makedirs(folder_name, exist_ok=True)
            data_plot(config=config, config_file=config_file, epoch_list=['best'], style='black color', extended='plots', device=device, apply_weight_correction=True)

        print()
        print("=" * 80)
        print("Recurrent training complete!")
        print(f"Results saved to: {log_dir}/results/")
        print("=" * 80)
