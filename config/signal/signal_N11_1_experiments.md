# Signal N11_1 Experiment Series - Baseline Chaotic RNN

## Summary

Testing GNN's ability to learn chaotic RNN dynamics with varying noise levels and training configurations (init_training_single_type, frozen embeddings).

**Data Generation**: PDE_N11 simulator ([notebooks/rnn_models.ipynb](../../src/NeuralGraph/notebooks/rnn_models.ipynb))
**Model**: Signal_Propagation GNN
**Trainer**: graph_trainer.py (data_train_signal)

---

## Experiment Table

| Config         | Dataset        | noise_level | lr_embedding | init_single_type | R² @ conv | Notes                    |
|:---------------|:---------------|:-----------:|:------------:|:----------------:|:---------:|:-------------------------|
| signal_N11_1_1 | signal_N11_1   | 0.0         | 5.0E-3       | True             | [Pending] | Baseline                 |
| signal_N11_1_2 | signal_N11_1   | 0.0         | 5.0E-3       | True             | [Pending] | Test init_single_type    |
| signal_N11_1_3 | signal_N11_1   | 0.0         | 5.0E-3       | True             | [Pending] | Frozen embeddings test   |
| signal_N11_1_4 | signal_N11_1_4 | 0.5         | 5.0E-3       | True             | [Pending] | Medium noise             |
| signal_N11_1_5 | signal_N11_1_5 | 1.0         | 5.0E-3       | True             | [Pending] | High noise               |
| signal_N11_1_6 | signal_N11_1_6 | 2.0         | 5.0E-3       | True             | [Pending] | Very high noise          |
| signal_N11_1_7 | signal_N11_1_7 | 5.0         | 5.0E-3       | True             | [Pending] | Extreme noise            |

**Note**: signal_N11_1_3 description mentions `learning_rate_embedding_start: 1.0E-16` but actual config has 5.0E-3 (same as others)

---

## Configuration

### RNN Dynamics (PDE_N11)
```python
du/dt = -c*u + g * W * tanh(u)

# Parameters
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
φ: tanh
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
batch_size: 8
seed: 24
data_augmentation_loop: 50
sparsity: 'none'

# Learning rates
learning_rate_W_start: 1.0E-3
learning_rate_start: 5.0E-4
learning_rate_embedding_start: 5.0E-3

# Regularization
coeff_W_L1: 1.0E-5
coeff_edge_norm: 0.0

# Special flags
init_training_single_type: True
```

---

## Experiment Series Breakdown

### Baseline (signal_N11_1_1)
- Clean data (no noise)
- Standard configuration
- Reference for all comparisons

### Init Training Test (signal_N11_1_2)
- Same as baseline
- Tests `init_training_single_type: True`
- All neurons initialized as same type

### Frozen Embeddings (signal_N11_1_3)
- Description mentions frozen embeddings (lr_embedding: 1.0E-16)
- **Actual config**: lr_embedding: 5.0E-3 (same as baseline)
- May need correction if frozen embeddings intended

### Noise Robustness (signal_N11_1_4 - signal_N11_1_7)
- **noise_level: 0.5** (1_4): Medium noise during training
- **noise_level: 1.0** (1_5): High noise
- **noise_level: 2.0** (1_6): Very high noise
- **noise_level: 5.0** (1_7): Extreme noise

---

## Key Questions

### Noise Robustness
- [ ] How does noise affect convergence speed?
- [ ] Optimal noise level for robustness vs accuracy trade-off?
- [ ] Does noisy training improve generalization?
- [ ] At what noise level does learning break down?

### Training Configuration
- [ ] Effect of `init_training_single_type: True`?
- [ ] Should signal_N11_1_3 use frozen embeddings (1.0E-16)?
- [ ] Compare same dataset (signal_N11_1) across configs 1-3

---

## Observations

**Data Reuse**:
- Configs 1-3 use same dataset (signal_N11_1)
- Tests different training configurations on identical data
- Configs 4-7 use separate noisy datasets

**Pure Chaotic Dynamics**:
- No external input
- No oscillatory drive
- Tests fundamental ability to learn internal RNN dynamics

**Noise Range**:
- 0.0 (clean) → 5.0 (extreme)
- Large range to test robustness limits

---

## Expected Results

- **Low noise (0.0-0.5)**: Should learn well, high R²
- **Medium noise (1.0-2.0)**: May show robustness benefits
- **High noise (5.0)**: Expected to degrade performance
- **Init comparison**: Minimal difference expected between 1_1 and 1_2

---

## Notes

- All experiments use same network architecture
- 10 epochs with 50x data augmentation
- 2 independent runs for robustness
- Chaotic regime (g=7.0) ensures complex dynamics
