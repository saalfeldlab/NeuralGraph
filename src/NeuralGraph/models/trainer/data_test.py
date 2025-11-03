"""Extracted from graph_trainer.py"""
from ._imports import *

def data_test(config=None, config_file=None, visualize=False, style='color frame', verbose=True, best_model=20, step=15,
              ratio=1, run=1, test_mode='', sample_embedding=False, particle_of_interest=1, new_params = None, device=[]):

    dataset_name = config.dataset
    print(f"\033[92mdataset_name: {dataset_name}\033[0m")
    print(f"\033[92mdevice: {config.description}\033[0m")

    if test_mode == "":
        test_mode = "test_ablation_0"

    if 'fly' in config.dataset:
        data_test_flyvis(config, visualize, style, verbose, best_model, step, test_mode, new_params, device)

    elif 'zebra' in config.dataset:
        data_test_zebra(config, visualize, style, verbose, best_model, step, test_mode, device)
    else:
        data_test_signal(config, config_file, visualize, style, verbose, best_model, step, ratio, run, test_mode, sample_embedding, particle_of_interest, new_params, device)

