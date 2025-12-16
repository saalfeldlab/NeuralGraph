# Dale's Law Training Study

## Context

You execute one experimental iteration in an iterative exploration loop.

## Goal

Explore neural activity dynamics under Dale's law constraints and evaluate the GNN's ability to recover the true connectivity matrix W (onnectivity_R2) and faithfully reproduce the observed dynamics (test_pearson).
Obvioulsy if the activity generated is flat then the GNN recovery is not possible. Try to favor activity complexity, with vd_rank > 10

Key questions:

- How do E/I ratio and neuron type heterogeneity affect dynamics?
- Under what conditions can the GNN recover W and predict activity?

## Starting Point

`signal_Claude.yaml` with `connectivity_type=chaotic`, `Dale_law: False`, `factor=0.5`, `gain=7`, `n_types=1` (baseline without Dale's law).

## Allowed Modifications

```yaml
# Connectivity structure
connectivity_type: "chaotic" # or "low_rank"
connectivity_rank: 20 # only used when connectivity_type="low_rank", range 5-100

# Neuron parameters: 4 rows for 4 neuron types, each row: [a, b, g, s, w, h]
params: [
    # a:decay, b:offset, g:gain, s:self, w:width, h:threshold
    [1.0, 0.0, 10.0, 0.0, 1.0, 0.0], # type 1
    [2.0, 0.0, 10.0, 0.0, 1.0, 1.0], # type 2
    [3.0, 0.0, 10.0, 0.0, 1.0, 2.0], # type 3
    [4.0, 0.0, 10.0, 0.0, 1.0, 3.0], # type 4
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

- **Converged**: connectivity_R2 > 0.9
- **Partial**: connectivity_R2 0.1-0.9 (grey zone)
- **Failed**: connectivity_R2 < 0.1

## Protocol

**IMPORTANT: Change only ONE parameter per iteration** to isolate effects.

1. Start with baseline: Dale_law=False, factor=0.5, gain=7, n_types=1
2. Enable Dale_law (keep all else same)
3. Then vary ONE parameter at a time
4. If training fails, revert and try different parameter

## Log Format

```
## Iter N: [converged/partial/failed]
Node: id=N, parent=P, V=1, N_total=N
Config: connectivity_type=X, connectivity_rank=R, factor=X, gain=Y, n_types=Z, Dale_law=T/F
Metrics: spectral_radius=X, svd_rank=Y, test_R2=Z, test_pearson=W, connectivity_R2=V, final_loss=L
RankScore: 0.XX  # = (rank - 1) / (N_total - 1), ascending order
Observation: [one line]
Change: [param: old -> new]
```
