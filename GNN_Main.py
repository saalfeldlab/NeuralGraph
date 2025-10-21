import time
from shutil import copyfile
import argparse
import networkx as nx
import os
import scipy.io
import umap
import torch
import torch.nn as nn
import torch_geometric.data as data
from sklearn import metrics
from tifffile import imread
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import to_networkx
from scipy.optimize import curve_fit
from scipy.spatial import Delaunay
from torchvision.transforms import GaussianBlur
import matplotlib
import matplotlib.pyplot as plt

from matplotlib import rc
from matplotlib.ticker import FuncFormatter
from prettytable import PrettyTable

from NeuralGraph.config import NeuralGraphConfig
from NeuralGraph.generators.graph_data_generator import *
from NeuralGraph.models.graph_trainer import *
from NeuralGraph.models.Siren_Network import *
from NeuralGraph.models.utils import *

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

import os
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
        best_model = 'None' 
        task = 'test'  # 'generate', 'train', 'test'

        config_list = ['fly_N9_64_1_8', 'fly_N9_64_1_9', 'fly_N9_64_2_8', 'fly_N9_64_2_9', 'fly_N9_64_3_6', 'fly_N9_64_3_7', 'fly_N9_64_3_7_1', 'fly_N9_64_3_7_2', 'fly_N9_64_4_2', 'fly_N9_64_4_3', 'fly_N9_64_4_7']  # for quick test
        
        # config_list = ['fly_N9_64_3_7']  # for quick test


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

            data_test(
                config=config,
                visualize=False,
                style="white color name continuous_slice",
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


