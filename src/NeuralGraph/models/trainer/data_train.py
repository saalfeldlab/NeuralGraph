"""Extracted from graph_trainer.py"""
from ._imports import *

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
        elif 'RNN' in config.graph_model.signal_model_name or 'LSTM' in config.graph_model.signal_model_name:
            data_train_flyvis_RNN(config, erase, best_model, device)
        else:
            data_train_flyvis(config, erase, best_model, device)
    elif 'zebra_fluo' in config.dataset:
        data_train_zebra_fluo(config, erase, best_model, device)
    elif 'zebra' in config.dataset:
        data_train_zebra(config, erase, best_model, device)
    else:
        data_train_signal(config, erase, best_model, device)

