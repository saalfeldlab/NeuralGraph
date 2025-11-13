# Signal N11_2 Experiment Series - Oscillatory Input to Chaotic RNN

## Summary

Testing GNN's ability to learn chaotic RNN dynamics driven by oscillatory external input with varying amplitudes and noise levels.

**Data Generation**: PDE_N11 simulator ([notebooks/rnn_models.ipynb](../../src/NeuralGraph/notebooks/rnn_models.ipynb))
**Model**: Signal_Propagation GNN
**Trainer**: graph_trainer.py (data_train_signal)

---

## Experiment Table

| Config           | Dataset        | Osc Amp | noise_level | n_excit | batch_size | data_aug | R² @ conv | Notes                       |
|:-----------------|:---------------|:-------:|:-----------:|:-------:|:----------:|:--------:|:---------:|:----------------------------|
| signal_N11_2_1_1 | signal_N11_2_1 | 3.0     | 0.0         | 0       | 8          | 200      | [Pending] | Low amplitude baseline      |
| signal_N11_2_1_2 | signal_N11_2_1 | 3.0     | 0.0         | 0       | 8          | 200      | [Pending] | Repeat                      |
| signal_N11_2_1_3 | signal_N11_2_1 | 3.0     | 0.0         | 0       | 8          | 200      | [Pending] | Repeat                      |
| signal_N11_2_1_4 | signal_N11_2_1 | 3.0     | 0.0         | 0       | 8          | 200      | [Pending] | Repeat                      |
| signal_N11_2_1_5 | signal_N11_2_1 | 3.0     | 0.0         | 0       | 8          | 200      | [Pending] | Repeat                      |
| signal_N11_2_1_6 | signal_N11_2_1 | 3.0     | 0.0         | 0       | 8          | 200      | [Pending] | Repeat                      |
| signal_N11_2_2_1 | signal_N11_2_2 | 10.0    | 0.0         | 0       | 8          | 50       | [Pending] | Medium amplitude            |
| signal_N11_2_2_2 | signal_N11_2_2 | 10.0    | 0.0         | 0       | 8          | 50       | [Pending] | Repeat                      |
| signal_N11_2_2_3 | signal_N11_2_2 | 10.0    | 0.0         | 0       | 8          | 50       | [Pending] | Repeat                      |
| signal_N11_2_2_4 | signal_N11_2_2 | 10.0    | 0.0         | 0       | 8          | 50       | [Pending] | Repeat                      |
| signal_N11_2_2_5 | signal_N11_2_2 | 10.0    | 0.0         | 0       | 8          | 50       | [Pending] | Repeat                      |
| signal_N11_2_3_1 | signal_N11_2_3 | 30.0    | 0.0         | 0       | 8          | 200      | [Pending] | High amplitude              |
| signal_N11_2_3_2 | signal_N11_2_3 | 30.0    | 0.0         | 0       | 8          | 200      | [Pending] | Repeat                      |
| signal_N11_2_3_3 | signal_N11_2_3 | 30.0    | 0.0         | 0       | 8          | 200      | [Pending] | Repeat                      |
| signal_N11_2_3_4 | signal_N11_2_3 | 30.0    | 0.0         | 0       | 8          | 200      | [Pending] | Repeat                      |
| signal_N11_2_4_1 | signal_N11_2_4 | 3.0     | 0.5         | 1       | 1          | 50       | [Pending] | +noise +excit +NNR_f        |
| signal_N11_2_5_1 | signal_N11_2_5 | 3.0     | 1.0         | 1       | 1          | 50       | [Pending] | +high noise +excit +NNR_f   |
| signal_N11_2_6_1 | signal_N11_2_6 | 3.0     | 2.0         | 1       | 1          | 50       | [Pending] | +v.high noise +excit +NNR_f |
| signal_N11_2_7_1 | signal_N11_2_7 | 3.0     | 5.0         | 1       | 1          | 50       | [Pending] | +extreme noise +excit +NNR_f|

---

## Configuration

### RNN Dynamics with Oscillatory Input
```python
du/dt = -c*u + g * W * tanh(u) + e * cos(2π * w * frame / max_frame)

# Base parameters
n_neurons: 100
n_neuron_types: 1
delta_t: 0.1
n_frames: 10,000

connectivity_type: 'chaotic'
connectivity_init: [0, 0.1]
connectivity_distribution: 'Gaussian'

# RNN equation parameters
g: 7.0  # Gain (chaotic regime)
c: 1.0  # Decay constant

# Oscillatory input
visual_input_type: 'oscillatory'
oscillation_max_amplitude: [3.0, 10.0, 30.0]
oscillation_frequency: 10.0
```

### GNN Model
```python
signal_model_name: 'PDE_N11'
prediction: 'first_derivative'

# Edge function
input_size: 3
output_size: 1
hidden_dim: 64
n_layers: 3

# Update function
input_size_update: 3
n_layers_update: 3
hidden_dim_update: 64

aggr_type: 'add'
embedding_dim: 2
```

### Training
```python
n_epochs: 10
n_runs: 2
seed: 24
sparsity: 'none'

# Standard configs (N11_2_1, 2_2, 2_3)
batch_size: 8
data_augmentation_loop: 200 (or 50 for N11_2_2)
n_excitatory_neurons: 0

# Special configs (N11_2_4-7)
batch_size: 1
data_augmentation_loop: 50
n_excitatory_neurons: 1
learning_rate_NNR_f: 1.0E-6

# Learning rates
learning_rate_W_start: 1.0E-3
learning_rate_start: 5.0E-4
learning_rate_embedding_start: 5.0E-3

# Regularization
coeff_W_L1: 1.0E-5
coeff_edge_norm: 0.0
```

---

## Experiment Series Breakdown

### N11_2_1_x: Low Amplitude (A = 3.0)
**6 configs** - Baseline oscillatory input
- Clean data (no noise)
- No extra excitatory neurons
- High data augmentation (200)
- Batch size: 8
- Oscillation amplitude comparable to internal dynamics

### N11_2_2_x: Medium Amplitude (A = 10.0)
**5 configs** - Medium oscillatory drive
- Clean data (no noise)
- No extra excitatory neurons
- Lower data augmentation (50)
- Batch size: 8
- Oscillation amplitude dominates internal dynamics

### N11_2_3_x: High Amplitude (A = 30.0)
**4 configs** - Strong external drive
- Clean data (no noise)
- No extra excitatory neurons
- High data augmentation (200)
- Batch size: 8
- Very strong external drive

### N11_2_4-7: Combined Challenges (A = 3.0 + Noise + Excitatory Neuron)
**4 configs** - Testing robustness with multiple factors

Special features in these configs:
- **n_excitatory_neurons: 1**: Adds 1 excitatory neuron to the network
- **NNR_f network**: Additional neural network for modulation
  - `input_size_nnr_f: 1`
  - `hidden_dim_nnr_f: 128`
  - `n_layers_nnr_f: 3`
  - `learning_rate_NNR_f: 1.0E-6`
  - `omega_f: 30`
- **Reduced batch size: 1**: More challenging training
- **Lower data augmentation: 50**: Less data reuse
- **init_training_single_type: True**: All neurons initialized as same type

#### Noise Levels:
- **N11_2_4_1**: noise_level = 0.5 (medium)
- **N11_2_5_1**: noise_level = 1.0 (high)
- **N11_2_6_1**: noise_level = 2.0 (very high)
- **N11_2_7_1**: noise_level = 5.0 (extreme)

---

## Key Questions

### Oscillation Amplitude Effects
- [ ] Can GNN disentangle oscillatory input from internal dynamics?
- [ ] How does amplitude (3.0 vs 10.0 vs 30.0) affect learning quality?
- [ ] At what amplitude does external drive dominate internal chaos?
- [ ] Compare with baseline chaotic RNN (N11_1)

### Noise Robustness with Oscillations
- [ ] How does noise affect learning with oscillatory input?
- [ ] Compare noise robustness: oscillatory (N11_2) vs non-oscillatory (N11_1)
- [ ] Does oscillatory drive help or hurt under noisy conditions?

### Excitatory Neuron Effects
- [ ] Impact of adding 1 excitatory neuron to the network?
- [ ] Role of NNR_f modulation network?
- [ ] Why batch_size=1 for these configs?

### Data Augmentation
- [ ] Why does N11_2_2 use lower augmentation (50 vs 200)?
- [ ] Optimal augmentation for different amplitude regimes?

---

## Observations

### Data Augmentation Strategy
- **High amplitude oscillations (A=3.0, 30.0)**: data_aug = 200
- **Medium amplitude (A=10.0)**: data_aug = 50
- May reflect different learning difficulty at different amplitude regimes

### Batch Size Reduction
- Standard configs (2_1, 2_2, 2_3): batch_size = 8
- Special configs (2_4-7): batch_size = 1
- Smaller batches may be necessary with excitatory neurons and NNR_f network

### Noise + Oscillation Combination
- N11_2_4-7 combine THREE challenges:
  1. Oscillatory external input
  2. Training noise (0.5 to 5.0)
  3. Extra excitatory neuron with modulation network
- Tests robustness under complex, realistic conditions

### Comparison with N11_1
- N11_1: Pure chaotic dynamics + noise
- N11_2_1-3: Chaotic + oscillations (no noise)
- N11_2_4-7: Chaotic + oscillations + noise + excitatory
- Systematic increase in complexity

---

## Expected Results

### Amplitude Effects
- **Low (3.0)**: Should learn both internal and external dynamics
- **Medium (10.0)**: External drive may dominate, easier learning?
- **High (30.0)**: Very strong external drive, internal chaos may be masked

### Noise Effects (N11_2_4-7)
- **Medium noise (0.5)**: Should handle with oscillatory structure
- **High noise (1.0-2.0)**: More challenging
- **Extreme noise (5.0)**: Likely to degrade significantly

### Data Augmentation
- Higher augmentation (200) should help with complex temporal patterns
- Lower augmentation (50) tests if model can learn efficiently

---

## Notes

- **Frequency fixed at 10.0**: One full oscillation cycle over trajectory
- **All use same network architecture** (except NNR_f in 2_4-7)
- **Challenge**: Learning connectivity matrix W when external oscillatory input drives significant portion of dynamics
- **NNR_f network**: Purpose unclear, may modulate excitatory neuron activity
- **Systematic design**: Tests amplitude, noise, and architectural variations
