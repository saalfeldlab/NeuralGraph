# X Tensor Layout Refactor for Signal Models

## Goal
Unify signal data X tensor layout with FlyVis for consistency. Remove unused columns (V1 velocity) and move external input logic from PDE models to data generator.

## New X Tensor Layout (Signal Models)

```
x[:, 0]   = index (neuron ID)
x[:, 1:3] = positions (x, y)
x[:, 3]   = signal u (state)           # was x[:, 6]
x[:, 4]   = external_input             # was x[:, 8]
x[:, 5]   = plasticity p (PDE_N6/N7)   # was x[:, 8]
x[:, 6]   = neuron_type                # was x[:, 5]
x[:, 7]   = calcium
```

## Files Modified

### 1. `src/NeuralGraph/generators/graph_data_generator.py`

**Function: `data_generate_synaptic()`**

- Replace concatenation-based x tensor with explicit column assignment:
```python
x = torch.zeros((n_neurons, 8), dtype=torch.float32, device=device)
x[:, 0] = torch.arange(n_neurons, dtype=torch.float32, device=device)  # index
x[:, 1:3] = X1.clone().detach()  # positions
x[:, 3] = H1[:, 0].clone().detach()  # signal state u
x[:, 4] = 0  # external input (set per frame)
x[:, 5] = 1  # plasticity p (init to 1 for pde_n6/n7)
x[:, 6] = T1.squeeze().clone().detach()  # neuron_type
x[:, 7] = 0  # calcium
```

- Move external input computation (oscillatory/triggered) from PDE_N4 into frame loop
- Compute `external_input` each frame and set `x[:, 4] = external_input.squeeze()`
- Update field update logic to use new columns: `x[:, 3]` for signal, `x[:, 5]` for plasticity

### 2. `src/NeuralGraph/generators/PDE_N*.py` (All Signal PDEs)

Update column indices in all PDE models:

| Column | Old | New | Description |
|--------|-----|-----|-------------|
| signal | `x[:, 6:7]` | `x[:, 3:4]` | Signal state u |
| external_input | `x[:, 8:9]` | `x[:, 4:5]` | External input |
| plasticity | `x[:, 8:9]` | `x[:, 5:6]` | For PDE_N6/N7 only |
| neuron_type | `x[:, 5]` | `x[:, 6]` | Neuron type index |

**PDE_N4 specific changes:**
- Remove oscillation/triggered logic from `__init__` and `forward()`
- External input now comes from `x[:, 4:5]` (set by data generator)
- Add `external_input_mode` handling: "additive", "multiplicative", or "none"

### 3. `src/NeuralGraph/models/graph_trainer.py`

**Function: `data_train_signal()`**

Key changes:
- `activity_column` in LossRegularizer: change from `6` to `3`
- Activity extraction: `x[:, 3:4]` instead of `x[:, 6:7]`
- `type_list` stays at `x[:, 6:7]` (neuron_type moved to column 6)
- External input updates: `x[:, 4]` instead of `x[:, 8]`

### 4. `src/NeuralGraph/config.py`

Add new config options to `GraphModelConfig`:
```python
external_input_type: Literal["none", "signals", "visual"] = "none"
external_input_mode: Literal["additive", "multiplicative", "none"] = "none"
```

Add to `TrainingConfig`:
```python
learn_external_input: bool = False
```

### 5. Config YAML files

Add external input configuration:
```yaml
graph_model:
  external_input_type: "signals"   # or "none", "visual"
  external_input_mode: "additive"  # or "multiplicative", "none"

training:
  learn_external_input: False      # True to learn with SIREN/NGP
```

For learning external input with SIREN, add:
```yaml
graph_model:
  inr_type: "siren"
  input_size_nnr: 1
  hidden_dim_nnr: 1024
  n_layers_nnr: 2
  outermost_linear_nnr: True
  output_size_nnr: 1000  # n_neurons
  omega_nnr: 600

training:
  learn_external_input: True
  learning_rate_NNR_f: 1.0E-6
```

### 6. Plotting functions

Update column indices in:
- `src/NeuralGraph/generators/utils.py`: `plot_synaptic_*` functions
- `src/NeuralGraph/models/utils.py`: `plot_training_signal_*` functions
- `GNN_PlotFigure.py`: `plot_signal()`, `plot_synaptic3()`, etc.

## Implementation Order

1. Update `data_generate_synaptic()` - new x tensor layout + move oscillation logic
2. Update all PDE_N* models - column indices + external_input_mode handling
3. Update `data_train_signal()` - column accesses + activity_column parameter
4. Update config.py - add new config fields
5. Update plotting functions
6. Update config YAML files

## Critical: Regenerate Data

After refactoring, **regenerate all data** since old data files have signal at column 6:
```bash
python GNN_Main.py -o generate signal_N4_1
```

Then train:
```bash
python GNN_Main.py -o train signal_N4_1
```

## Key Bug Fix

In `graph_trainer.py`, the `LossRegularizer` initialization must use the correct activity column:
```python
regularizer = LossRegularizer(
    ...
    activity_column=3,  # NEW: signal at column 3 (was 6)
    ...
)
```

This is critical for training convergence - wrong column means regularizer reads neuron_type instead of signal values.
