# Signal N11_2 Experiment Series - Oscillatory Input to Chaotic RNN

## Summary

Testing GNN's ability to learn chaotic RNN dynamics driven by oscillatory external input with varying amplitudes, noise levels, and architectural variations.

**Data Generation**: PDE_N11 simulator ([notebooks/rnn_models.ipynb](../../src/NeuralGraph/notebooks/rnn_models.ipynb))
**Model**: Signal_Propagation GNN with optional NNR_f modulation network
**Trainer**: graph_trainer.py (data_train_signal)

---

## Experiment Table

| Config           | Dataset        | Osc Amp | noise_level | n_excit | batch_size | data_aug | init_single | frozen_emb | R² @ conv | Notes                       |
|:-----------------|:---------------|:-------:|:-----------:|:-------:|:----------:|:--------:|:-----------:|:----------:|:---------:|:----------------------------|
| signal_N11_2_1_1 | signal_N11_2_1 | 3.0     | 0.0         | 0       | 8          | 200      | No          | No         | [Pending] | Low amp baseline            |
| signal_N11_2_1_2 | signal_N11_2_1 | 3.0     | 0.0         | 1       | 1          | 50       | No          | No         | [Pending] | +excit +NNR_f batch=1       |
| signal_N11_2_1_3 | signal_N11_2_1 | 3.0     | 0.0         | 1       | 1          | 50       | Yes         | No         | [Pending] | +excit +NNR_f +init_single  |
| signal_N11_2_2_1 | signal_N11_2_2 | 10.0    | 0.0         | 0       | 8          | 50       | No          | No         | [Pending] | Medium amp baseline         |
| signal_N11_2_2_2 | signal_N11_2_2 | 10.0    | 0.0         | 1       | 8          | 50       | No          | No         | [Pending] | +excit +NNR_f               |
| signal_N11_2_2_3 | signal_N11_2_2 | 10.0    | 0.0         | 1       | 8          | 50       | Yes         | No         | [Pending] | +excit +NNR_f +init_single  |
| signal_N11_2_3_1 | signal_N11_2_3 | 30.0    | 0.0         | 0       | 8          | 200      | No          | No         | [Pending] | High amp baseline           |
| signal_N11_2_3_2 | signal_N11_2_3 | 30.0    | 0.0         | 1       | 8          | 50       | No          | No         | [Pending] | +excit +NNR_f               |
| signal_N11_2_3_3 | signal_N11_2_3 | 30.0    | 0.0         | 1       | 8          | 50       | No          | Yes        | [Pending] | +excit +NNR_f +frozen_emb   |
| signal_N11_2_4_1 | signal_N11_2_4 | 3.0     | 0.5         | 1       | 1          | 50       | Yes         | No         | [Pending] | +noise +excit +NNR_f        |
| signal_N11_2_5_1 | signal_N11_2_5 | 3.0     | 1.0         | 1       | 1          | 50       | Yes         | No         | [Pending] | +high noise +excit +NNR_f   |
| signal_N11_2_6_1 | signal_N11_2_6 | 3.0     | 2.0         | 1       | 1          | 50       | Yes         | No         | [Pending] | +v.high noise +excit +NNR_f |
| signal_N11_2_7_1 | signal_N11_2_7 | 3.0     | 5.0         | 1       | 1          | 50       | Yes         | No         | [Pending] | +extreme noise +excit +NNR_f|

**Column Definitions:**
- `init_single`: `init_training_single_type` - all neurons initialized as same type
- `frozen_emb`: embedding learning rate set to ~0 (1E-16 in config 2_3_3)

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

# NNR_f network (configs with n_excitatory_neurons=1)
input_size_nnr_f: 1
hidden_dim_nnr_f: 128
n_layers_nnr_f: 3
outermost_linear_nnr_f: True
output_size_nnr_f: 1
nnr_f_T_period: 10000
omega_f: 30
learning_rate_NNR_f: 1.0E-6
```

### Training
```python
n_epochs: 10
n_runs: 2
seed: 24
sparsity: 'none'

# Learning rates
learning_rate_W_start: 1.0E-3
learning_rate_start: 5.0E-4
learning_rate_embedding_start: 5.0E-3  # (or 1E-16 for frozen embeddings)

# Regularization
coeff_W_L1: 1.0E-5
coeff_edge_norm: 0.0
```

---

## Experiment Series Breakdown

### N11_2_1: Low Amplitude (A = 3.0)
**3 configs** - Baseline + architectural variations

| Config | n_excit | batch | data_aug | init_single | Purpose |
|:-------|:-------:|:-----:|:--------:|:-----------:|:--------|
| 2_1_1  | 0       | 8     | 200      | No          | Clean baseline |
| 2_1_2  | 1       | 1     | 50       | No          | Add NNR_f network |
| 2_1_3  | 1       | 1     | 50       | Yes         | NNR_f + single type init |

**Key characteristics:**
- Oscillation amplitude comparable to internal dynamics
- Tests effect of NNR_f modulation network
- Compares batch sizes (8 vs 1)
- Tests `init_training_single_type` impact

### N11_2_2: Medium Amplitude (A = 10.0)
**3 configs** - Medium drive + architectural variations

| Config | n_excit | batch | data_aug | init_single | Purpose |
|:-------|:-------:|:-----:|:--------:|:-----------:|:--------|
| 2_2_1  | 0       | 8     | 50       | No          | Medium amp baseline |
| 2_2_2  | 1       | 8     | 50       | No          | Add NNR_f network |
| 2_2_3  | 1       | 8     | 50       | Yes         | NNR_f + single type init |

**Key characteristics:**
- External drive dominates internal dynamics
- All use batch_size=8 (even with NNR_f)
- Lower data augmentation (50) across all configs

### N11_2_3: High Amplitude (A = 30.0)
**3 configs** - Strong drive + architectural + frozen embedding test

| Config | n_excit | batch | data_aug | init_single | frozen_emb | Purpose |
|:-------|:-------:|:-----:|:--------:|:-----------:|:----------:|:--------|
| 2_3_1  | 0       | 8     | 200      | No          | No         | High amp baseline |
| 2_3_2  | 1       | 8     | 50       | No          | No         | Add NNR_f network |
| 2_3_3  | 1       | 8     | 50       | No          | Yes        | NNR_f + frozen embeddings |

**Key characteristics:**
- Very strong external drive (10x baseline)
- Config 2_3_1 uses high augmentation (200)
- Config 2_3_3 freezes embeddings (lr_emb = 1E-16) - tests if embeddings are necessary
- All use batch_size=8

### N11_2_4-7: Combined Challenges (A = 3.0 + Noise + NNR_f)
**4 configs** - Testing robustness with noise escalation

| Config | noise_level | n_excit | batch | data_aug | init_single | Purpose |
|:-------|:-----------:|:-------:|:-----:|:--------:|:-----------:|:--------|
| 2_4_1  | 0.5         | 1       | 1     | 50       | Yes         | Medium noise |
| 2_5_1  | 1.0         | 1       | 1     | 50       | Yes         | High noise |
| 2_6_1  | 2.0         | 1       | 1     | 50       | Yes         | Very high noise |
| 2_7_1  | 5.0         | 1       | 1     | 50       | Yes         | Extreme noise |

**Special features:**
- All include NNR_f modulation network
- All use `n_excitatory_neurons: 1`
- All use `init_training_single_type: True`
- Reduced batch size: 1 (more challenging)
- Lower data augmentation: 50
- Systematic noise escalation: 0.5 → 1.0 → 2.0 → 5.0

---

## Key Experimental Factors

### 1. Oscillation Amplitude Sweep
- **Low (3.0)**: Comparable to internal dynamics - tests disentanglement
- **Medium (10.0)**: External drive dominates - potentially easier learning
- **High (30.0)**: Very strong drive - tests if internal chaos is masked

### 2. NNR_f Modulation Network
- Added when `n_excitatory_neurons=1`
- 3-layer network (hidden_dim=128) with very low learning rate (1E-6)
- Purpose: Time-dependent modulation with frequency omega_f=30
- Tests whether explicit modulation network helps with oscillatory dynamics

### 3. Batch Size Variations
- **batch_size=8**: Standard configs (2_1_1, 2_2_x, 2_3_x)
- **batch_size=1**: Challenging configs (2_1_2, 2_1_3, 2_4-7)
- Smaller batches when NNR_f network is present (except medium/high amp)

### 4. Data Augmentation Strategy
| Amplitude | Baseline Aug | +NNR_f Aug |
|:----------|:------------:|:----------:|
| 3.0       | 200          | 50         |
| 10.0      | 50           | 50         |
| 30.0      | 200          | 50         |

Pattern: Baseline configs use higher augmentation for extremes (3.0, 30.0), medium amp uses less

### 5. Initialization Strategy
- **`init_training_single_type: True`**: All neurons initialized as same type
- Used in: 2_1_3, 2_2_3, 2_4_1, 2_5_1, 2_6_1, 2_7_1
- Tests impact of homogeneous vs heterogeneous initialization

### 6. Frozen Embeddings Test
- **Config 2_3_3**: Sets `learning_rate_embedding_start: 1E-16` (effectively frozen)
- Only test with frozen embeddings in the entire series
- Tests whether learned embeddings are necessary for high-amplitude regime

---

## Key Questions

### Oscillation Amplitude Effects
- [ ] Can GNN disentangle oscillatory input from internal dynamics at low amplitude?
- [ ] How does amplitude (3.0 vs 10.0 vs 30.0) affect learning quality?
- [ ] At what amplitude does external drive dominate internal chaos?
- [ ] Compare with baseline chaotic RNN (N11_1)

### NNR_f Network Impact
- [ ] Does NNR_f modulation network improve learning of oscillatory dynamics?
- [ ] Why is NNR_f learning rate so low (1E-6)?
- [ ] What role does omega_f=30 play (3x oscillation frequency)?
- [ ] Compare performance: with vs without NNR_f at each amplitude

### Batch Size Effects
- [ ] Why batch_size=1 for low-amp configs with NNR_f but batch_size=8 for medium/high-amp?
- [ ] Does smaller batch size help or hurt with NNR_f network?
- [ ] Compare 2_1_2 (batch=1) vs 2_2_2 (batch=8) - both have NNR_f

### Initialization Impact
- [ ] Effect of `init_training_single_type` on learning?
- [ ] Compare pairs: 2_1_2 vs 2_1_3, 2_2_2 vs 2_2_3

### Frozen Embeddings
- [ ] Can model learn without adapting embeddings (2_3_3)?
- [ ] Are embeddings more/less important at high amplitude?

### Noise Robustness with Oscillations
- [ ] How does noise affect learning with oscillatory input?
- [ ] Noise threshold for acceptable performance?
- [ ] Compare noise robustness: oscillatory (N11_2) vs non-oscillatory (N11_1)
- [ ] Does oscillatory structure help or hurt under noisy conditions?

### Data Augmentation Strategy
- [ ] Why does medium amplitude (10.0) use less augmentation?
- [ ] Why do extremes (3.0, 30.0) use more augmentation for baseline configs?
- [ ] Is this related to learning difficulty or data diversity needs?

---

## Comparison Matrix

### By Amplitude (No NNR_f - Baseline Configs)
| Amplitude | Config  | n_excit | batch | data_aug |
|:----------|:--------|:-------:|:-----:|:--------:|
| 3.0       | 2_1_1   | 0       | 8     | 200      |
| 10.0      | 2_2_1   | 0       | 8     | 50       |
| 30.0      | 2_3_1   | 0       | 8     | 200      |

### By Amplitude (With NNR_f)
| Amplitude | Config  | batch | data_aug | init_single | frozen_emb |
|:----------|:--------|:-----:|:--------:|:-----------:|:----------:|
| 3.0       | 2_1_2   | 1     | 50       | No          | No         |
| 3.0       | 2_1_3   | 1     | 50       | Yes         | No         |
| 10.0      | 2_2_2   | 8     | 50       | No          | No         |
| 10.0      | 2_2_3   | 8     | 50       | Yes         | No         |
| 30.0      | 2_3_2   | 8     | 50       | No          | No         |
| 30.0      | 2_3_3   | 8     | 50       | No          | Yes        |

### Noise Escalation (All with NNR_f, batch=1, init_single=True)
| Noise | Config  | Amplitude | data_aug |
|:------|:--------|:---------:|:--------:|
| 0.0   | 2_1_2   | 3.0       | 50       |
| 0.0   | 2_1_3   | 3.0       | 50       |
| 0.5   | 2_4_1   | 3.0       | 50       |
| 1.0   | 2_5_1   | 3.0       | 50       |
| 2.0   | 2_6_1   | 3.0       | 50       |
| 5.0   | 2_7_1   | 3.0       | 50       |

---

## Expected Results

### Amplitude Effects
- **Low (3.0)**: Should learn both internal and external dynamics; most challenging disentanglement
- **Medium (10.0)**: External drive dominates; potentially easier learning; less augmentation needed
- **High (30.0)**: Internal chaos masked by strong drive; embeddings may be less important (test 2_3_3)

### NNR_f Network
- Should help capture periodic/oscillatory structure
- Low learning rate suggests fine-tuning of modulation
- omega_f=30 (3x oscillation freq) may capture harmonics

### Batch Size Impact
- batch=1 is more challenging but may help with temporal dependencies
- batch=8 provides more stable gradients
- Interesting that medium/high amp configs use batch=8 even with NNR_f

### Noise Effects (2_4-7)
- **Medium noise (0.5)**: Should handle with oscillatory structure
- **High noise (1.0-2.0)**: More challenging, test robustness
- **Extreme noise (5.0)**: Likely significant performance degradation

### Frozen Embeddings (2_3_3)
- If performance similar to 2_3_2, embeddings may not be critical at high amplitude
- If much worse, embeddings are important even when external drive dominates

---

## Notes

- **Frequency fixed at 10.0**: One full oscillation cycle over trajectory
- **All use same base GNN architecture** (hidden_dim=64, n_layers=3)
- **NNR_f architecture**: 128-dimensional hidden, 3 layers, linear output
- **Challenge**: Learning connectivity matrix W when external oscillatory input drives significant portion of dynamics
- **Systematic design**:
  - Amplitude sweep (3 levels: 3.0, 10.0, 30.0)
  - Architectural variations (±NNR_f, ±init_single_type, frozen_emb)
  - Noise escalation (4 levels: 0.5, 1.0, 2.0, 5.0)
  - Batch size variations (1 vs 8)
- **Comparison opportunity**: Compare with N11_1 series (no oscillations) to isolate effect of external periodic drive
- **Total configs**: 13 configs across 7 datasets
