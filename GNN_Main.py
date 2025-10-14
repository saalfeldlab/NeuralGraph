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
        best_model = None 
        task = 'train'  # 'generate', 'train', 'test'

        # config_list = ['signal_CElegans_d2', 'signal_CElegans_d2a', 'signal_CElegans_d3', 'signal_CElegans_d3a', 'signal_CElegans_d3b']
        # config_list = ['signal_CElegans_c14_4']
        # config_list = ['fly_N9_53_1', 'fly_N9_53_2', 'fly_N9_53_3', 'fly_N9_53_4', 'fly_N9_53_5', 'fly_N9_53_6', 'fly_N9_53_7', 'fly_N9_53_8']
        # config_list = ['fly_N9_44_26'] #, 'fly_N9_44_16', 'fly_N9_44_17', 'fly_N9_44_18', 'fly_N9_44_19', 'fly_N9_44_20', 'fly_N9_44_21', 'fly_N9_44_22',  'fly_N9_44_23', 'fly_N9_44_24', 'fly_N9_44_25', 'fly_N9_44_26']
        # config_list = ['fly_N9_22_1']

        # config_list = ['fly_N9_22_10'] #, 'fly_N9_22_11', 'fly_N9_22_12', 'fly_N9_22_13', 'fly_N9_22_14', 'fly_N9_22_15', 'fly_N9_22_16', 'fly_N9_22_17', 'fly_N9_22_18']

        # config_list = ['fly_N9_59_1', 'fly_N9_59_2', 'fly_N9_59_3', 'fly_N9_59_4', 'fly_N9_59_5']

        config_list = ['fly_N9_61_2']

        # config_list = ['signal_N11_5_1_3']

        # config_list = ['zebra_N10_33_5_13_3', 'zebra_N10_33_5']

    for config_file_ in config_list:
        print(" ")
        config_root = os.path.dirname(os.path.abspath(__file__)) + "/config"
        config_file, pre_folder = add_pre_folder(config_file_)
        config = NeuralGraphConfig.from_yaml(f"{config_root}/{config_file}.yaml")
        config.dataset = pre_folder + config.dataset
        config.config_file = pre_folder + config_file_
        device = set_device(config.training.device)

        print(f"\033[92mconfig_file  {config.config_file}\033[0m")
        print(f"\033[92mdevice  {device}\033[0m")

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
                step=20
            ) 

        if "train" in task:
            data_train(config=config, erase=False, best_model=best_model, device=device)
            
        if "test" in task:
            data_test(
                config=config,
                visualize=True,
                style="white color name continuous_slice",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="",
                sample_embedding=False,
                step=2,
                device=device,
                particle_of_interest=0,
                new_params = None,
            )  


