# GNN Class Modification Guide

When modifying the `Signal_Propagation` class or its variants, you must ensure consistency across multiple dependent modules.

## Core Files to Check

| File | What to Check |
|------|---------------|
| `models/Signal_Propagation.py` | Main GNN class definition |
| `models/graph_trainer.py` | Training loop, model instantiation, W access |
| `utils.py` | `LossRegularizer` class, `choose_training_model()` factory |
| `GNN_PlotFigure.py` | Weight plotting, model attribute access |

## Key Attributes and Their Access Patterns

### Connectivity Matrix (W)

The W matrix has three storage modes:

```python
# 1. Standard mode
self.W = nn.Parameter(shape=(n_neurons, n_neurons))

# 2. Multi-connectivity mode (per-run weights)
self.W = nn.Parameter(shape=(n_runs, n_neurons, n_neurons))

# 3. Low-rank factorization mode
self.WL = nn.Parameter(shape=(n_neurons, rank))
self.WR = nn.Parameter(shape=(rank, n_neurons))
self.W = Buffer(shape=(n_neurons, n_neurons))  # updated each forward pass
```

**External code accessing W:**
- `graph_trainer.py`: `model.W.copy_(...)`, `model.W * model.mask`
- `utils.py (LossRegularizer)`: `model.W`, `model.WL @ model.WR`
- `GNN_PlotFigure.py`: `model.W.squeeze()`, `model.W[0, :, :]`

### Embeddings (a, b)

```python
self.a  # Node embeddings: (n_neurons, embedding_dim)
self.b  # Trial embeddings: (n_runs, embedding_dim) if embedding_trial=True
```

### MLPs

```python
self.lin_edge  # Edge message function
self.lin_phi   # Node update function
self.lin_modulation  # Optional modulation (PDE_N6/N7)
```

## Modification Checklist

When changing model attributes:

1. **Adding a new learnable parameter:**
   - [ ] Define in `__init__` as `nn.Parameter` or `register_buffer`
   - [ ] Add to optimizer in `set_trainable_parameters()` (utils.py)
   - [ ] Add regularization term in `LossRegularizer.compute()` if needed
   - [ ] Update plotting in `GNN_PlotFigure.py` if visualization needed

2. **Changing W storage (e.g., low-rank factorization):**
   - [ ] Update `__init__` with new storage format
   - [ ] Update `forward()` to compute W correctly
   - [ ] Update `message()` to use W correctly
   - [ ] Update `LossRegularizer` to handle new format:
     ```python
     if hasattr(model, 'WL') and hasattr(model, 'WR'):
         model_W = model.WL @ model.WR
     else:
         model_W = model.W
     ```
   - [ ] Update `graph_trainer.py` mask application
   - [ ] Update `GNN_PlotFigure.py` weight extraction

3. **Changing forward signature:**
   - [ ] Update all call sites in `graph_trainer.py`
   - [ ] Update any variant classes (Signal_Propagation_MLP, etc.)

4. **Adding config parameters:**
   - [ ] Add to `config.py` dataclass
   - [ ] Extract in `__init__` from appropriate config section
   - [ ] Document in config YAML template

## Config Parameter Flow

```
YAML config file
    ↓
NeuralGraphConfig (config.py)
    ├── simulation: SimulationConfig
    │   ├── n_neurons → model.n_neurons
    │   ├── external_input_mode → model.external_input_mode
    │   └── ...
    ├── graph_model: GraphModelConfig
    │   ├── embedding_dim → model.embedding_dim
    │   ├── input_size → lin_edge input dimension
    │   └── ...
    └── training: TrainingConfig
        ├── low_rank_factorization → model.low_rank_factorization
        ├── low_rank → model.low_rank
        ├── coeff_W_L1 → regularizer coefficients
        └── ...
```

## Input Data Layout

The model expects `data.x` with this column layout:

| Column | Content | Usage |
|--------|---------|-------|
| 0 | particle_id | Index into embeddings `self.a` |
| 1:3 | positions (x, y) | Spatial coordinates |
| 3 | u (signal) | Neural activity state |
| 4 | external_input | External modulation |
| 5 | plasticity | For PDE_N6/N7 |
| 6 | neuron_type | Type classification |
| 7 | calcium | For specialized models |

## Testing a Modification

After making changes:

1. Run data generation to verify forward pass:
   ```bash
   python GNN_Main.py --config config/signal/your_config.yaml --mode generate
   ```

2. Run a few training epochs to verify backward pass:
   ```bash
   python GNN_Main.py --config config/signal/your_config.yaml --mode train
   ```

3. Check regularization terms are computed correctly (look for NaN/Inf)

4. Verify plotting works with modified model:
   ```bash
   python GNN_PlotFigure.py --config config/signal/your_config.yaml
   ```

## Common Pitfalls

- **Shape mismatches in LossRegularizer**: The `get_in_features_lin_edge()` function must return correct column based on model type
- **Low-rank buffer not updated**: Must call `self.W.copy_(W.detach())` in forward pass to keep buffer in sync
- **Multi-connectivity indexing**: Use `W[data_id, edge_i, edge_j]` not `W[edge_i, edge_j]`
- **Mask not applied**: Remember to apply `self.mask` to W to prevent self-loops
