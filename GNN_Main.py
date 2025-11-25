import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before other imports
import argparse
import os



from NeuralGraph.config import NeuralGraphConfig
from NeuralGraph.generators.graph_data_generator import data_generate
from NeuralGraph.models.graph_trainer import data_train, data_test
from NeuralGraph.utils import set_device, add_pre_folder
from NeuralGraph.models.NGP_trainer import data_train_NGP

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
        best_model = None
        task = 'test'  #, 'train', 'test', 'NGP', 'NGP_old'

        # config_list = ['fly_N9_64_1_1', 'fly_N9_64_1_2', 'fly_N9_64_1_3', 'fly_N9_64_1_4',
        #                'fly_N9_64_2_1', 'fly_N9_64_2_2', 'fly_N9_64_2_3', 'fly_N9_64_2_4',
        #                'fly_N9_64_3_1', 'fly_N9_64_3_2', 'fly_N9_64_3_3', 'fly_N9_64_3_4',
        #                'fly_N9_64_4_1', 'fly_N9_64_4_2', 'fly_N9_64_4_3', 'fly_N9_64_4_4'
        #                ]  


        # config_list = ['fly_N9_62_5_9_1', 'fly_N9_62_5_9_2', 'fly_N9_62_5_9_3', 'fly_N9_62_5_9_4', 'fly_N9_62_5_19_1', 'fly_N9_62_5_19_2', 'fly_N9_62_5_19_3', 'fly_N9_62_5_19_4']

        # config_list = ['fly_N9_62_5_10', 'fly_N9_62_5_11', 'fly_N9_62_5_12', 'fly_N9_62_5_13', 'fly_N9_62_5_14', 'fly_N9_62_5_15', 'fly_N9_62_5_16', 'fly_N9_62_5_17', 'fly_N9_62_5_18']

        config_list = ['fly_N9_62_5_9_5', 'fly_N9_62_5_19_5', 'fly_N9_62_5_19_6']

        # config_list = ['fly_N9_62_22_1', 'fly_N9_62_22_2', 'fly_N9_62_22_3', 'fly_N9_62_22_4', 'fly_N9_62_22_5', 'fly_N9_62_22_6', 'fly_N9_62_22_7', 'fly_N9_62_22_8', 'fly_N9_62_22_9', 'fly_N9_62_22_10']
        
        # config_list = ['zebra_N10_34_1']

        # config_list = ['signal_N11_4_4_1', 'signal_N11_4_4_2', 'signal_N11_4_4_3', 'signal_N11_4_4_4', 'signal_N11_4_4_5', 'signal_N11_4_4_6']


        # config_list = ['signal_N11_2_1_1']
        

    for config_file_ in config_list:
        print(" ")
        config_root = os.path.dirname(os.path.abspath(__file__)) + "/config"
        config_file, pre_folder = add_pre_folder(config_file_)
        config = NeuralGraphConfig.from_yaml(f"{config_root}/{config_file}.yaml")
        config.dataset = pre_folder + config.dataset
        config.config_file = pre_folder + config_file_
        device = set_device(config.training.device)

        print(f"\033[92mconfig_file:  {config.config_file}\033[0m")
        print(f"\033[92mdevice:  {device}\033[0m")

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

        if "train" in task:
            data_train(config=config, erase=False, best_model=best_model, device=device)

        if "test" in task:

            config.training.noise_model_level = 0.0
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
                device=device,
                particle_of_interest=0,
                new_params = None,
            )

        if task == 'NGP':
            # Use new modular NGP trainer pipeline
            data_train_NGP(config=config, device=device)


            
                  


