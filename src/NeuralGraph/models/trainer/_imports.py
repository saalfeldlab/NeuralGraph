"""Common imports for all trainer functions"""
import os
import time
import glob

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.animation import FFMpegWriter
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import copy

from NeuralGraph.models.utils import (
    choose_training_model,
    increasing_batch_size,
    constant_batch_size,
    set_trainable_parameters,
    set_trainable_parameters_vae,
    get_in_features_update,
    get_in_features_lin_edge,
    analyze_edge_function,
    get_n_hop_neighborhood_with_stats,
    plot_training_signal,
    plot_training_signal_field,
    plot_training_signal_missing_activity,
    plot_training_flyvis,
    plot_weight_comparison,
    get_index_particles,
)
from NeuralGraph.utils import (
    to_numpy,
    CustomColorMap,
    create_log_dir,
    check_and_clear_memory,
    sort_key,
    fig_init,
    get_equidistant_points,
    open_gcs_zarr,
    compute_trace_metrics,
    get_datavis_root_dir,
)
from NeuralGraph.models.Siren_Network import Siren, Siren_Network
from NeuralGraph.models.Signal_Propagation_FlyVis import Signal_Propagation_FlyVis
from NeuralGraph.models.Signal_Propagation_MLP import Signal_Propagation_MLP
from NeuralGraph.models.Signal_Propagation_MLP_ODE import Signal_Propagation_MLP_ODE
from NeuralGraph.models.Signal_Propagation_Zebra import Signal_Propagation_Zebra
from NeuralGraph.models.Signal_Propagation_Temporal import Signal_Propagation_Temporal
from NeuralGraph.models.Signal_Propagation_RNN import Signal_Propagation_RNN
from NeuralGraph.models.Signal_Propagation_LSTM import Signal_Propagation_LSTM
from NeuralGraph.models.utils_zebra import (
    plot_field_comparison,
    plot_field_comparison_continuous_slices,
    plot_field_comparison_discrete_slices,
    plot_field_discrete_xy_slices_grid,
)

from NeuralGraph.models.Calcium_Latent_Dynamics import Calcium_Latent_Dynamics
from NeuralGraph.sparsify import EmbeddingCluster, sparsify_cluster, clustering_evaluation
from NeuralGraph.fitting_models import linear_model

from sklearn.neighbors import NearestNeighbors
from scipy.optimize import curve_fit

from torch_geometric.utils import dense_to_sparse, to_networkx
from torch_geometric.data import Data as pyg_Data
from torch_geometric.loader import DataLoader
import torch_geometric as pyg
import seaborn as sns
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from scipy.spatial import KDTree
from sklearn import neighbors, metrics
from scipy.ndimage import median_filter
from tifffile import imwrite, imread
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree
from scipy.special import logsumexp
from NeuralGraph.generators.utils import choose_model, generate_compressed_video_mp4, init_connectivity
from NeuralGraph.generators.graph_data_generator import (
    apply_pairwise_knobs_torch,
    assign_columns_from_uv,
    build_neighbor_graph,
    compute_column_labels,
    greedy_blue_mask,
    mseq_bits,
)
from NeuralGraph.generators.davis import AugmentedDavis
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import tensorstore as ts
import napari
from collections import deque
from tqdm import tqdm, trange
import networkx as nx
import scipy
from prettytable import PrettyTable
import imageio
