# Dale's Law Parameter Study

## Goal

Grid search to understand how parameters influence dynamics when Dale's law is applied.

## Starting Point

`signal_chaotic_Claude.yaml` with `Dale_law: False` produces chaotic activity.

## Allowed Modifications

```yaml
params:  # 4 rows for 4 neuron types, each row: [a, b, g, s, w, h]
  [      # a:decay, b:offset, g:gain, s:self, w:width, h:threshold
    [1.0, 0.0, 7.0, 0.0, 1.0, 0.0],  # type 1
    [2.0, 0.0, 7.0, 0.0, 1.0, 1.0],  # type 2
    [3.0, 0.0, 7.0, 0.0, 1.0, 2.0],  # type 3
    [4.0, 0.0, 7.0, 0.0, 1.0, 3.0],  # type 4
  ]
Dale_law: False
Dale_law_factor: 0.5 # fraction of excitatory neurons (only used when Dale_law: True)
n_neuron_types: 1 # number of neuron types (1 to 4), uses corresponding param rows
```

Note: `connectivity_init` is NOT used in the code - connectivity scaling is fixed at `1/sqrt(n_neurons)`.

## Analysis Files

- `activity.png`: visual activity traces
- `analysis.log`: spectral radius and SVD rank(99%)

## Classification

Look at `activity.png` and `analysis.log`:

- **Steady**: flat lines, low SVD rank
- **Oscillatory**: sustained irregular fluctuations, higher SVD rank

## Protocol

Systematically explore parameter space:

1. First confirm baseline is oscillatory with `Dale_law: False`
2. Enable `Dale_law: True` and observe effect
3. Explore parameters:
   - `Dale_law_factor`: 0.3, 0.5, 0.7
   - `gain` (g in params): 5, 7, 10
   - `n_neuron_types`: 1, 2, 4
4. Record observations for each combination

## Log Format

```
## Iter N: [steady/oscillatory]
Config: Dale_law=T/F, factor=X, gain=Y, n_types=Z
Metrics: spectral_radius=X, svd_rank=Y
Observation: [one line]
Change: [param: old -> new]
```
