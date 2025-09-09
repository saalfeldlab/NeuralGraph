#!/usr/bin/env python3
"""
Generate 12 configuration files for HOC experiments
Files: fly_N9_55_1.yaml through fly_N9_55_12.yaml
Saves directly to config/fly/ directory
"""

import os
from pathlib import Path

def generate_config(exp_num, rollout, noise, discount_type):
    """Generate a single config file content"""
    
    # Generate coeff_loop based on discount type
    if rollout == 1:
        coeff_loop = "[1.0]"
    elif discount_type == "none":
        coeff_loop = "[" + ", ".join(["1.0"] * rollout) + "]"
    elif discount_type == "linear":
        # Linear decay: 1.0, 0.8, 0.6, 0.4, 0.2
        linear_values = [1.0 - 0.2*i for i in range(rollout)]
        linear_values = [max(0.2, v) for v in linear_values]  # Floor at 0.2
        coeff_loop = "[" + ", ".join([f"{v:.1f}" for v in linear_values]) + "]"
    else:  # exponential discount
        coeff_loop = "[" + ", ".join([f"{0.7**i:.2f}" for i in range(rollout)]) + "]"
    
    # Set recursive_training flag
    recursive_training = "True" if rollout > 1 else "False"
    
    # Generate description
    description = f"Exp {exp_num}: {rollout}-step rollout, noise {noise}"
    if rollout > 1:
        description += f", {discount_type} discount"
    
    config = f"""description: '{description}'
dataset: 'fly_N9_18_4_0'

simulation:
    connectivity_file: 'to be specified'
    adjacency_matrix: ''
    params: [[1.0,1.0,1.0,1.0]]
    n_neurons: 13741
    n_input_neurons: 1736
    n_neuron_types: 65
    n_edges: 434112
    n_frames: 90720
    delta_t: 0.02
    baseline_value: 0

graph_model:
    signal_model_name: 'PDE_N9_A'
    prediction: 'first_derivative'
    input_size: 3
    output_size: 1
    hidden_dim: 32
    n_layers: 2
    input_size_update: 5
    n_layers_update: 3
    hidden_dim_update: 64
    aggr_type: 'add'
    embedding_dim: 2
    update_type: 'generic'
    lin_edge_positive: True

plotting:
    colormap: 'tab20'
    arrow_length: 1
    xlim: [-2.5, 5]
    ylim: [-100, 100]

training:
    n_epochs: 10
    n_runs: 1
    device: 'auto'
    seed: 241
    recursive_training: {recursive_training}
    recursive_loop: {rollout}
    coeff_loop: {coeff_loop}
    batch_size: 1
    batch_ratio: 1
    sparsity: 'none'
    rotation_augmentation: False
    data_augmentation_loop: 25
    learning_rate_W_start: 1.0E-3
    learning_rate_start: 5.0E-4
    learning_rate_embedding_start: 5.0E-3
    measurement_noise_level: {noise}
    noise_model_level: 0
    coeff_edge_diff: 500
    coeff_update_u_diff: 0
    coeff_update_msg_diff: 0
    coeff_edge_norm: 1000.0
"""
    return config

def main():
    """Generate all config files directly in config/fly/ directory"""
    
    # Define all experiments (removed exp 1&2, added 2 new ones)
    experiments = [
        (1, 3, "1.0E-3", "none"),     # 3-step, low noise, no discount
        (2, 3, "1.0E-3", "exp"),      # 3-step, low noise, exponential
        (3, 3, "1.0E-1", "none"),     # 3-step, high noise, no discount
        (4, 3, "1.0E-1", "exp"),      # 3-step, high noise, exponential
        (5, 5, "1.0E-3", "none"),     # 5-step, low noise, no discount
        (6, 5, "1.0E-3", "exp"),      # 5-step, low noise, exponential
        (7, 5, "1.0E-1", "none"),     # 5-step, high noise, no discount
        (8, 5, "1.0E-1", "exp"),      # 5-step, high noise, exponential
        (9, 3, "1.0E-2", "exp"),      # 3-step, medium noise, exponential
        (10, 5, "1.0E-2", "exp"),     # 5-step, medium noise, exponential
        (11, 3, "1.0E-3", "linear"),  # NEW: 3-step, low noise, linear decay
        (12, 7, "1.0E-3", "exp"),     # NEW: 7-step, low noise, exponential
    ]
    
    # Create output directory
    output_dir = Path("config/fly")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all config files
    print("Generating HOC experiment configuration files...")
    print("="*70)
    print(f"{'File':<20} {'Rollout':<8} {'Noise':<10} {'Discount':<12} {'Recursive'}")
    print("-"*70)
    
    for exp_num, rollout, noise, discount in experiments:
        filename = f"fly_N9_55_{exp_num}.yaml"
        filepath = output_dir / filename
        
        content = generate_config(exp_num, rollout, noise, discount)
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        # Print summary for this experiment
        recursive = "True" if rollout > 1 else "False"
        print(f"{filename:<20} {rollout:<8} {noise:<10} {discount:<12} {recursive}")
    
    print("="*70)
    print(f"✅ Successfully created 12 config files in '{output_dir}/'")
    print("\nNew experiments added:")
    print("  - Exp 11: 3-step with linear decay (tests different weighting)")
    print("  - Exp 12: 7-step rollout (tests longer trajectories)")
    print("\nUsage: python train.py --config config/fly/fly_N9_55_X.yaml")
    print("       where X is the experiment number (1-12)")

if __name__ == "__main__":
    main()