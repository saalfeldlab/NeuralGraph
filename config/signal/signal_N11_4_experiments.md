# Signal N11_4 Experiment Series - Successor Representation Learning

## Summary

Learning successor representation in RNN dynamics with frozen embeddings and sparse connectivity regularization.

**Data Generation**: PDE_N11 simulator ([notebooks/rnn_models.ipynb](../../src/NeuralGraph/notebooks/rnn_models.ipynb))
**Model**: Signal_Propagation GNN
**Trainer**: graph_trainer.py (data_train_signal)

---

## Experiment Table

| Config           | Dataset        | n_frames | start_frame | seed | R² @ convergence | Notes              |
|:-----------------|:---------------|:--------:|:-----------:|:----:|:----------------:|:-------------------|
| signal_N11_4_1   | signal_N11_4_1 | 5,000    | -50         | 24   | [Pending]        | Baseline long traj |
| signal_N11_4_1_1 | signal_N11_4_1 | 1,000    | 0           | 34   | [Pending]        | Short trajectory   |
| signal_N11_4_1_2 | signal_N11_4_1 | 1,000    | 0           | 44   | [Pending]        | Seed variation     |
| signal_N11_4_1_3 | signal_N11_4_1 | 1,000    | 0           | 54   | [Pending]        | Seed variation     |
| signal_N11_4_1_4 | signal_N11_4_1 | 1,000    | 0           | 64   | [Pending]        | Seed variation     |
| signal_N11_4_1_5 | signal_N11_4_1 | 1,000    | 0           | 74   | [Pending]        | Seed variation     |
| signal_N11_4_1_6 | signal_N11_4_1 | 1,000    | 0           | 84   | [Pending]        | Seed variation     |
| signal_N11_4_2   | signal_N11_4_2 | -        | -           | -    | [Pending]        | -                  |

---

## Configuration

### RNN Dynamics (Successor Representation)
```python
du/dt = -c*u + g * W * tanh(u)

# Parameters
n_neurons: 100
delta_t: 0.1
connectivity_type: 'successor'  # Not chaotic!
g: 7.0
c: 1.0
```

### GNN Model
```python
signal_model_name: 'PDE_N11'
prediction: 'first_derivative'
hidden_dim: 64
n_layers: 3
embedding_dim: 2
```

### Training (Special Configuration)
```python
n_epochs: 20  # 2x more than other series
n_runs: 2
batch_size: 8
data_augmentation_loop: 200

# Learning rates
learning_rate_W_start: 1.0E-3
learning_rate_start: 5.0E-4
learning_rate_embedding_start: 1.0E-16  # ✓ Frozen embeddings!

# Regularization
coeff_W_L1: 1.0E-4  # 10x stronger than N11_1/N11_2
```

---

## Key Design Choices

### Frozen Embeddings
- `learning_rate_embedding: 1.0E-16` effectively freezes embeddings
- Forces learning into connectivity matrix W
- Tests if GNN can learn successor representation without adaptive node embeddings

### Stronger L1 Regularization
- `coeff_W_L1: 1.0E-4` (vs 1.0E-5 in other series)
- Promotes sparse connectivity structure
- Encourages interpretable successor representations

### Trajectory Length Comparison
- **Long**: 5,000 frames (signal_N11_4_1)
- **Short**: 1,000 frames (signal_N11_4_1_x)
- Tests data efficiency for learning successor representation

### Multiple Seeds
- Seeds: 24, 34, 44, 54, 64, 74, 84
- Tests robustness of learning across initializations

---

## Key Questions

- [ ] Can GNN learn successor representation with frozen embeddings?
- [ ] Effect of L1 regularization on connectivity structure
- [ ] Comparison: long (5K) vs short (1K) trajectories
- [ ] Robustness across different random seeds
- [ ] Quality of learned successor matrix vs ground truth

---

## Mathematical Background

### Successor Representation
The successor representation M captures expected future state occupancy:
```
M(s,s') = E[Σ γ^t 𝟙(s_t = s') | s_0 = s]
```

In continuous RNN context, this relates to how states propagate through the network over time.

---

## Notes

- **connectivity_type: 'successor'**: Different from chaotic dynamics in N11_1/N11_2
- **More epochs (20 vs 10)**: Compensates for frozen embeddings
- **Shorter trajectories**: Tests if 1K frames sufficient for learning
- **Higher data augmentation**: 200 loops through dataset
