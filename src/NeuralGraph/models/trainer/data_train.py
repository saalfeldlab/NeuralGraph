"""Extracted from graph_trainer.py"""
from ._imports import *
from .data_train_flyvis import data_train_flyvis
from .data_train_flyvis_calcium import data_train_flyvis_calcium
from .data_train_signal import data_train_signal
from .data_train_zebra import data_train_zebra
from .data_train_zebra_fluo import data_train_zebra_fluo

def data_train(config=None, erase=False, best_model=None, device=None):
    # plt.rcParams['text.usetex'] = True
    # rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    # matplotlib.rcParams['savefig.pad_inches'] = 0

    seed = config.training.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # torch.autograd.set_detect_anomaly(True)

    dataset_name = config.dataset
    print(f"\033[92mdataset_name: {dataset_name}\033[0m")
    print(f"\033[92mdevice: {config.description}\033[0m")

    if 'fly' in config.dataset:
        if config.simulation.calcium_type != 'none':
            data_train_flyvis_calcium(config, erase, best_model, device)
        else:
            data_train_flyvis(config, erase, best_model, device)
    elif 'zebra_fluo' in config.dataset:
        data_train_zebra_fluo(config, erase, best_model, device)
    elif 'zebra' in config.dataset:
        data_train_zebra(config, erase, best_model, device)
    else:
        data_train_signal(config, erase, best_model, device)

