import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before other imports
import argparse
import os



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
        task = 'train_INR'  #, 'train', 'test', 'generate', 'plot', 'train_NGP', 'train_INR'


        # config_list = [
        #     # N2_2_x
        #     # 'signal_N2_2_1', 'signal_N2_2_2', 'signal_N2_2_3', 'signal_N2_2_4',
        #     # N2_3_x
        #     # 'signal_N2_3_1', 'signal_N2_3_2', 'signal_N2_3_3', 'signal_N2_3_4',
        #     # N11_1_3_x
        #     # 'signal_N11_1_3_1', 'signal_N11_1_3_2', 'signal_N11_1_3_3', 'signal_N11_1_3_4', 'signal_N11_1_3_5',
        #     # N11_5_x_x
        #     'signal_N11_5_1_1', 'signal_N11_5_1_2', 'signal_N11_5_2_1', 'signal_N11_5_2_2', 'signal_N11_5_2_3',-
        #     'signal_N11_5_2_4', 'signal_N11_5_2_5', 'signal_N11_5_2_6', 'signal_N11_5_4_1', 'signal_N11_5_4_2',
        #     'signal_N11_5_5'
        # ]

        # config_list = ['signal_N4_1', 'signal_N4_2', 'signal_N4_3', 'signal_N4_4', 'signal_N2_1']

        config_list = ['signal_N4_4_2','signal_N4_4_3', 'signal_N4_5_1','signal_N4_5_2','signal_N4_5_3']


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
                n_rollout_frames=2000,
                device=device,
                particle_of_interest=0,
                new_params = None,
            )

        if 'plot' in task:
            folder_name = './log/' + pre_folder + '/tmp_results/'
            os.makedirs(folder_name, exist_ok=True)
            data_plot(config=config, config_file=config_file, epoch_list=['best'], style='black color', extended='plots', device=device, apply_weight_correction=True)



                  


