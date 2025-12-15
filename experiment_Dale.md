# Dale's Law Training Study

## Goal

Explore neural activity dynamics under Dale's law constraints and evaluate the GNN's ability to recover the true connectivity matrix W and faithfully reproduce the observed dynamics.

Key questions:
- How do E/I ratio and neuron type heterogeneity affect dynamics?
- Under what conditions can the GNN recover W and predict activity?

## Starting Point

`signal_chaotic_Claude.yaml` with `Dale_law: False`, `factor=0.5`, `gain=7`, `n_types=1` (baseline without Dale's law).

## Allowed Modifications

```yaml
params:  # 4 rows for 4 neuron types, each row: [a, b, g, s, w, h]
  [      # a:decay, b:offset, g:gain, s:self, w:width, h:threshold
    [1.0, 0.0, 10.0, 0.0, 1.0, 0.0],  # type 1
    [2.0, 0.0, 10.0, 0.0, 1.0, 1.0],  # type 2
    [3.0, 0.0, 10.0, 0.0, 1.0, 2.0],  # type 3
    [4.0, 0.0, 10.0, 0.0, 1.0, 3.0],  # type 4
  ]
Dale_law: True
Dale_law_factor: 0.3 # fraction of excitatory neurons
n_neuron_types: 2 # number of neuron types (1, 2, or 4)
```

## Analysis Files

- `activity.png`: visual activity traces (from data generation)
- `analysis.log`: metrics from all pipeline stages:
  - `spectral_radius`: eigenvalue analysis of connectivity
  - `svd_rank`: SVD rank at 99% variance (activity complexity)
  - `test_R2`: R² between ground truth and rollout prediction
  - `test_pearson`: Pearson correlation per neuron (mean)
  - `connectivity_R2`: R² of learned vs true connectivity weights
  - `final_loss`: final training loss (lower is better)

## Classification

- **Good training**: test_R2 > 0.8, connectivity_R2 > 0.7
- **Moderate**: test_R2 0.5-0.8, connectivity_R2 0.4-0.7
- **Poor**: test_R2 < 0.5 or connectivity_R2 < 0.4

## Protocol

**IMPORTANT: Change only ONE parameter per iteration** to isolate effects.

1. Start with baseline: Dale_law=False, factor=0.5, gain=7, n_types=1
2. Enable Dale_law (keep all else same)
3. Then vary ONE parameter at a time:
   - `n_neuron_types`: 1 → 2 → 4
   - `Dale_law_factor`: 0.5 → 0.3
4. If training fails, revert and try different parameter
5. Record test_R2 and connectivity_R2 for each

## Log Format

```
## Iter N: [good/moderate/poor]
Config: factor=X, gain=Y, n_types=Z
Metrics: spectral_radius=X, svd_rank=Y, test_R2=Z, test_pearson=W, connectivity_R2=V, final_loss=L
Observation: [one line about training quality]
Change: [param: old -> new]
```
