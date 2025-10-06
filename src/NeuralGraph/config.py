from typing import Optional, Literal, Annotated, Dict
import yaml
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Union

# Sub-config schemas for NeuralGraph


class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension: int = 2
    n_frames: int = 1000
    start_frame: int = 0
    seed: int = 42

    model_id: str = "000"
    ensemble_id: str = "0000"

    sub_sampling: int = 1
    delta_t: float = 1

    boundary: Literal["periodic", "no", "periodic_special", "wall"] = "periodic"
    min_radius: float = 0.0
    max_radius: float = 0.1

    n_neurons: int = 1000
    n_neuron_types: int = 5
    n_input_neurons: int = 0
    n_excitatory_neurons: int = 0
    n_edges: int = 0
    max_edges: float = 1.0e6
    n_extra_null_edges: int = 0

    baseline_value: float = -999.0
    shuffle_neuron_types: bool = False

    noise_visual_input: float = 0.0
    only_noise_visual_input: float = 0.0
    visual_input_type: str = ""
    blank_freq: int = 2  # Frequency of blank frames in visual input
    simulation_initial_state: bool = False

    tile_contrast: float = 0.2
    tile_corr_strength: float = 0.0   # correlation knob for tile_mseq / tile_blue_noise
    tile_flip_prob: float = 0.05      # per-frame random flip probability
    tile_seed: int = 42


    n_grid: int = 128

    n_nodes: Optional[int] = None
    n_node_types: Optional[int] = None
    node_coeff_map: Optional[str] = None
    node_value_map: Optional[str] = "input_data/pattern_Null.tif"

    adjacency_matrix: str = ""
    short_term_plasticity_mode: str = "depression"

    connectivity_file: str = ""
    connectivity_init: list[float] = [-1]
    connectivity_filling_factor: float = 1
    connectivity_type: Literal["none", "distance", "voronoi", "k_nearest"] = "distance"
    connectivity_parameter: float = 1.0
    connectivity_distribution: str = "Gaussian"
    connectivity_distribution_params: float = 1

    excitation_value_map: Optional[str] = None
    excitation: str = "none"

    params: list[list[float]]
    func_params: list[tuple] = None

    phi: str = "tanh"
    tau: float = 1.0
    sigma: float = 0.005

    calcium_type: Literal["none", "leaky", "multi-compartment", "saturation"] = "none"
    calcium_activation: Literal["softplus", "relu", "identity", "tanh"] = "softplus"
    calcium_tau: float = 0.5  # decay time constant (same units as delta_t)
    calcium_alpha: float = 1.0  # scale factor to convert [Ca] to fluorescence
    calcium_beta: float = 0.0  # baseline offset for fluorescence
    calcium_initial: float = 0.0  # initial calcium concentration
    calcium_noise_level: float = 0.0  # optional Gaussian noise added to [Ca] updates
    calcium_saturation_kd: float = 1.0  # for nonlinear saturation models
    calcium_num_compartments: int = 1
    calcium_dow_sample: int = 1  # down-sample [Ca] time series by this factor

    pos_init: str = "uniform"
    dpos_init: float = 0
    diffusion_coefficients: list[list[float]] = None


class GraphModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    particle_model_name: str = ""
    cell_model_name: str = ""
    mesh_model_name: str = ""
    signal_model_name: str = ""
    prediction: Literal["first_derivative", "2nd_derivative"] = "2nd_derivative"
    integration: Literal["Euler", "Runge-Kutta"] = "Euler"

    aggr_type: str
    embedding_dim: int = 2
    embedding_init: str = ""

    field_type: str = ""
    field_grid: Optional[str] = ""

    input_size: int = 1
    output_size: int = 1
    hidden_dim: int = 1
    n_layers: int = 1

    input_size_2: int = 1
    output_size_2: int = 1
    hidden_dim_2: int = 1
    n_layers_2: int = 1

    input_size_decoder: int = 1
    output_size_decoder: int = 1
    hidden_dim_decoder: int = 1
    n_layers_decoder: int = 1

    lin_edge_positive: bool = False

    update_type: Literal[
        "linear",
        "mlp",
        "pre_mlp",
        "2steps",
        "none",
        "no_pos",
        "generic",
        "excitation",
        "generic_excitation",
        "embedding_MLP",
        "test_field",
    ] = "none"

    input_size_update: int = 3
    n_layers_update: int = 3
    hidden_dim_update: int = 64
    output_size_update: int = 1


    kernel_type: str = "mlp"

    input_size_nnr: int = 3
    n_layers_nnr: int = 5
    hidden_dim_nnr: int = 128
    output_size_nnr: int = 1
    outermost_linear_nnr: bool = True
    omega: float = 80.0


    input_size_nnr_f: int = 3
    n_layers_nnr_f: int = 5
    hidden_dim_nnr_f: int = 128
    output_size_nnr_f: int = 1
    outermost_linear_nnr_f: bool = True
    omega_f: float = 80.0

    nnr_f_xy_period: float = 1.0
    nnr_f_T_period: float = 1.0


    input_size_modulation: int = 2
    n_layers_modulation: int = 3
    hidden_dim_modulation: int = 64
    output_size_modulation: int = 1

    input_size_excitation: int = 3
    n_layers_excitation: int = 5
    hidden_dim_excitation: int = 128

    excitation_dim: int = 1

    latent_dim: int = 64
    latent_update_steps: int = 50
    stochastic_latent: bool = True
    latent_init_std: float = 1.0  # only used if you later add 'init from noise' modes

    # encoder sizes (x -> [mu, logvar])
    input_size_encoder: int = 1      # set to n_neurons in your YAML
    n_layers_encoder: int = 3
    hidden_dim_encoder: int = 256
    latent_n_layers_update: int = 2
    latent_hidden_dim_update: int = 64
    output_size_decoder: int = 1      # set to n_neurons in your YAML
    n_layers_decoder: int = 3
    hidden_dim_decoder:  int = 256




class PlottingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    colormap: str = "tab10"
    arrow_length: int = 10
    marker_size: int = 100
    xlim: list[float] = [-0.1, 0.1]
    ylim: list[float] = [-0.1, 0.1]
    embedding_lim: list[float] = [-40, 40]
    speedlim: list[float] = [0, 1]
    pic_folder: str = "none"
    pic_format: str = "jpg"
    pic_size: list[int] = [1000, 1100]
    data_embedding: int = 1
    plot_batch_size: int = 1000


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    device: Annotated[str, Field(pattern=r"^(auto|cpu|cuda:\d+)$")] = "auto"

    n_epochs: int = 20
    n_epochs_init: int = 99999
    epoch_reset: int = -1
    epoch_reset_freq: int = 99999
    batch_size: int = 1
    batch_ratio: float = 1
    small_init_batch_size: bool = True
    embedding_step: int = 1000
    shared_embedding: bool = False
    embedding_trial: bool = False
    remove_self: bool = True

    pretrained_model: str = ""
    pre_trained_W: str = ""

    multi_connectivity: bool = False
    with_connectivity_mask: bool = False
    has_missing_activity: bool = False

    do_tracking: bool = False
    tracking_gt_file: str = ""
    ctrl_tracking: bool = False
    distance_threshold: float = 0.1
    epoch_distance_replace: int = 20

    denoiser: bool = False
    denoiser_type: Literal["none", "window", "LSTM", "Gaussian_filter", "wavelet"] = ("none")
    denoiser_param: float = 1.0

    time_window: int = 0

    n_runs: int = 2
    seed: int = 42
    clamp: float = 0
    pred_limit: float = 1.0e10

    particle_dropout: float = 0
    n_ghosts: int = 0
    ghost_method: Literal["none", "tensor", "MLP"] = "none"
    ghost_logvar: float = -12

    sparsity_freq: int = 5
    sparsity: Literal[
        "none",
        "replace_embedding",
        "replace_embedding_function",
        "replace_state",
        "replace_track",
    ] = "none"
    fix_cluster_embedding: bool = False
    cluster_method: Literal[
        "kmeans",
        "kmeans_auto_plot",
        "kmeans_auto_embedding",
        "distance_plot",
        "distance_embedding",
        "distance_both",
        "inconsistent_plot",
        "inconsistent_embedding",
        "none",
    ] = "distance_plot"
    cluster_distance_threshold: float = 0.01
    cluster_connectivity: Literal["single", "average"] = "single"

    Ising_filter: str = "none"

    learning_rate_start: float = 0.001
    learning_rate_embedding_start: float = 0.001
    learning_rate_update_start: float = 0.0
    learning_rate_modulation_start: float = 0.0001
    learning_rate_W_start: float = 0.0001

    learning_rate_end: float = 0.0005
    learning_rate_embedding_end: float = 0.0001
    learning_rate_modulation_end: float = 0.0001
    Learning_rate_W_end: float = 0.0001

    learning_rate_missing_activity: float = 0.0001
    learning_rate_NNR: float = 0.0001
    learning_rate_NNR_f: float = 0.0001
    training_NNR_start_epoch: int = 0

    coeff_W_L1: float = 0.0
    coeff_W_L1_rate: float = 0.5
    coeff_W_L1_ghost: float = 0
    coeff_W_sign: float = 0

    coeff_entropy_loss: float = 0
    coeff_loss1: float = 1
    coeff_loss2: float = 1
    coeff_loss3: float = 1
    coeff_edge_diff: float = 10
    coeff_update_diff: float = 0
    coeff_update_msg_diff: float = 0
    coeff_update_msg_sign: float = 0
    coeff_update_u_diff: float = 0
    coeff_NNR_f: float = 0

    coeff_permutation: float = 100

    coeff_TV_norm: float = 0
    coeff_missing_activity: float = 0
    coeff_edge_norm: float = 0

    coeff_edge_weight_L1: float = 0
    coeff_edge_weight_L1_rate: float = 0.5
    coeff_phi_weight_L1: float = 0
    coeff_phi_weight_L1_rate: float = 0.5

    coeff_edge_weight_L2: float = 0
    coeff_phi_weight_L2: float = 0

    coeff_Jp_norm: float = 0
    coeff_F_norm: float = 0
    coeff_det_F: float = 1

    diff_update_regul: str = "none"

    coeff_model_a: float = 0
    coeff_model_b: float = 0
    coeff_lin_modulation: float = 0
    coeff_continuous: float = 0

    noise_level: float = 0
    measurement_noise_level: float = 0
    noise_model_level: float = 0
    loss_noise_level: float = 0.0


    rotation_augmentation: bool = False
    translation_augmentation: bool = False
    reflection_augmentation: bool = False
    velocity_augmentation: bool = False
    data_augmentation_loop: int = 40

    recursive_training: bool = False
    recursive_training_start_epoch: int = 0
    recursive_loop: int = 0
    coeff_loop: list[float] = [2, 4, 8, 16, 32, 64]
    time_step: int = 1
    recursive_sequence: str = ""
    recursive_parameters: list[float] = [0, 0]

    regul_matrix: bool = False
    sub_batches: int = 1
    sequence: list[str] = ["to track", "to cell"]

    MPM_trainer : str = "F"


# Main config schema for NeuralGraph


class NeuralGraphConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: Optional[str] = "NeuralGraph"
    dataset: str
    data_folder_name: str = "none"
    connectome_folder_name: str = "none"
    data_folder_mesh_name: str = "none"
    config_file: str = "none"
    simulation: SimulationConfig
    graph_model: GraphModelConfig
    plotting: PlottingConfig
    training: TrainingConfig

    @staticmethod
    def from_yaml(file_name: str):
        with open(file_name, "r") as file:
            raw_config = yaml.safe_load(file)
        return NeuralGraphConfig(**raw_config)

    def pretty(self):
        return yaml.dump(self, default_flow_style=False, sort_keys=False, indent=4)


if __name__ == "__main__":
    config_file = "../../config/arbitrary_3.yaml"  # Insert path to config file
    config = NeuralGraphConfig.from_yaml(config_file)
    print(config.pretty())

    print("Successfully loaded config file. Model description:", config.description)