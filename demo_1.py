"""
Demo 1 - Figure 2 from the paper:
"Graph neural networks uncover structure and function underlying the activity of neural assemblies"

This demo reproduces Figure 2: 1000 densely connected neurons with 4 neuron-dependent update functions.
    - N_neurons: 1000
    - N_types: 4 (parameterized by tau_i={0.5,1} and s_i={1,2})
    - N_frames: 100,000
    - Connectivity: 100% (dense)
    - Noise: none
    - External inputs: none

The simulation follows Equation 2 from the paper:
    dx_i/dt = -x_i/tau_i + s_i*tanh(x_i) + g_i * sum_j(W_ij * psi(x_j))

Usage:
    python demo_1.py                                    # run full pipeline
    python demo_1.py -o generate signal_demo_1          # generate data only
    python demo_1.py -o train signal_demo_1             # train only
    python demo_1.py -o test signal_demo_1              # test only
    python demo_1.py -o plot signal_demo_1              # plot only
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
    parser = argparse.ArgumentParser(description="NeuralGraph Demo 1 - Figure 2 data results")
    parser.add_argument(
        "-o", "--option", nargs="+", help="option that takes multiple values"
    )

    print()
    print("=" * 80)
    print("Demo 1 - Figure 2: 1000 neurons, 4 types, dense connectivity")
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
        config_list = ['signal_demo_1']

    for config_file_ in config_list:
        print()
        config_root = os.path.dirname(os.path.abspath(__file__)) + "/config"
        config_file, pre_folder = add_pre_folder(config_file_)

        # load config
        config = NeuralGraphConfig.from_yaml(f"{config_root}/{config_file}.yaml")
        config.dataset = pre_folder + config.dataset
        config.config_file = pre_folder + config_file_

        if device == []:
            device = set_device(config.training.device)

        log_dir = f'./log/{pre_folder}{config.dataset}'
        graphs_dir = f'./graphs_data/{pre_folder}{config.dataset}'

        if "generate" in task:
            # Generate synthetic neural activity data using the PDE_N2 model
            # This creates the training dataset with 1000 neurons and 100,000 time points
            #
            # Figure 2a: Activity time series used for GNN training
            # Figure 2b: Sample of 10 time series
            # Figure 2c: True connectivity matrix W_ij
            #
            # Output folder: ./graphs_data/signal/signal_demo_1/
            #   - saved graph data (.pt files)
            #   - activity.png: activity time series visualization
            #   - connectivity.png: true connectivity matrix
            print()
            print("-" * 80)
            print("STEP 1: GENERATE - Simulating neural activity (Fig 2a-c)")
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
            # Train the GNN to learn connectivity W, latent embeddings a_i, and functions phi/psi
            # The GNN learns to predict dx/dt from the observed activity x
            #
            # Output folder: ./log/signal/signal_demo_1/
            #   - models/: saved model checkpoints (.pt files)
            #   - tmp_training/: training curves and intermediate results
            print()
            print("-" * 80)
            print("STEP 2: TRAIN - Training GNN to learn W, embeddings, phi, psi")
            print("-" * 80)
            print(f"  Training for {config.training.n_epochs} epochs, {config.training.n_runs} run(s)")
            print(f"  Learning: connectivity W, latent vectors a_i, functions phi* and psi*")
            print(f"  Models: {log_dir}/models/")
            print(f"  Training plots: {log_dir}/tmp_training")
            print(f"  Tensorboard: tensorboard --logdir {log_dir}/")
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
            # Evaluates prediction accuracy and performs rollout inference
            #
            # Output folder: ./log/signal/signal_demo_1/
            #   - results/: test results and rollout predictions
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
            # Generate publication-quality figures matching Figure 2 from the paper
            #
            # Figure 2d: Learned connectivity matrix
            # Figure 2e: Comparison of learned vs true connectivity (R^2, slope)
            # Figure 2f: Learned latent vectors a_i (2D embedding showing 4 clusters)
            # Figure 2g: Learned update functions phi*(a_i, x) - 1000 overlaid curves
            # Figure 2h: Learned transfer function psi*(x) normalized to max=1
            #
            # Output folder: ./log/signal/signal_demo_1/results/
            #   - embedding.png: latent vectors a_i (Fig 2f)
            #   - learned_phi.png: update functions phi* (Fig 2g)
            #   - learned_psi.png: transfer function psi* (Fig 2h)
            #   - connectivity_comparison.png: W learned vs true (Fig 2d,e)
            print()
            print("-" * 80)
            print("STEP 4: PLOT - Generating Figure 2 panels (d-h)")
            print("-" * 80)
            print(f"  Fig 2d: Learned connectivity matrix")
            print(f"  Fig 2e: W learned vs true (R^2, slope)")
            print(f"  Fig 2f: Latent vectors a_i (4 clusters)")
            print(f"  Fig 2g: Update functions phi*(a_i, x)")
            print(f"  Fig 2h: Transfer function psi*(x)")
            print(f"  Output: {log_dir}/results/")
            print()
            folder_name = './log/' + pre_folder + '/tmp_results/'
            os.makedirs(folder_name, exist_ok=True)
            data_plot(config=config, config_file=config_file, epoch_list=['best'], style='black color', extended='plots', device=device, apply_weight_correction=True)

        print()
        print("=" * 80)
        print("Demo 1 complete!")
        print(f"Results saved to: {log_dir}/results/")
        print("=" * 80)
