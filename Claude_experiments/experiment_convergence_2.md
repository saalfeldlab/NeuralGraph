# GNN Convergence Study

## Context

You execute one experimental iteration in an iterative exploration loop.

## Goal

Find robust training hyperparameters that ensure GNN convergence for a given simulation configurations.
The GNN should reliably recover the connectivity matrix W (connectivity_R2 > 0.7).

**Note**: Training is stochastic - convergence may fail ~1 in 3 runs with the same config. A config is considered "robust" only if it converges consistently across multiple runs.

Key questions:

- What learning rate combinations ensure stable convergence?
- How does L1 regularization (coeff_W_L1) affect W recovery?
- What is the robust operating range for these hyperparameters?

## Starting Point

`signal_chaotic_Claude.yaml` with current training parameters that sometimes fail to converge (connectivity_R2 ≈ 0).

## Allowed Modifications

```yaml
training:
  learning_rate_W_start: 1.0E-3 # LR for connectivity weights W
  learning_rate_start: 5.0E-4 # LR for model parameters
  learning_rate_embedding_start: 2.5E-4 # LR for embeddings

  coeff_W_L1: 5.0E-6 # L1 regularization on W (sparsity)
  batch_size: 8 # batch size for training
```

Parameter ranges (use your judgment to explore):

- `learning_rate_W_start`: 1E-4 to 1E-2
- `learning_rate_start`: 1E-4 to 1E-3
- `coeff_W_L1`: 0 to 1E-4
- `batch_size`: 1, 4, 8, 16

## Analysis Files

- `analysis.log`: metrics from training/test/plot:
  - `test_R2`: R² between ground truth and rollout prediction
  - `test_pearson`: Pearson correlation per neuron (mean)
  - `connectivity_R2`: R² of learned vs true connectivity weights
  - `final_loss`: final training loss (lower is better)

## Classification

- **Converged**: connectivity_R2 > 0.9
- **Partial**: connectivity_R2 0.1-0.9 (grey zone)
- **Failed**: connectivity_R2 < 0.1

## UCB Tree Exploration

`ucb_scores.txt` provides pre-computed UCB scores for all nodes including current iteration:

```
Node 2: UCB=2.175, parent=1, visits=1, R2=0.997 [CURRENT]
Node 1: UCB=2.110, parent=root, visits=2, R2=0.934
```

- **Higher UCB = more promising to explore from** (balances exploitation vs exploration)
- `[CURRENT]` marks the current iteration's node
- `visits`: how many times this node's subtree was explored
- `R2`: connectivity_R2 for this node

## Protocol

**IMPORTANT: Change only ONE parameter per iteration** to isolate effects.

1. Start with baseline: lr_W=1E-3, lr=5E-4, coeff_W_L1=5E-6, batch_size=8
2. Read `ucb_scores.txt` to decide which node to explore from
3. Then vary ONE parameter at a time from chosen parent
4. If training fails, revert and try different parameter

**Handling stochasticity**: If a config fails, retry once with same params before changing. If it fails twice, change parameter. If it succeeds once but failed before, note as "unstable".

Goal: Find robust configurations and map the boundaries of the working parameter space.

**Validation**: When a good config is found (connectivity_R2 > 0.9), repeat it 3 times without changes to confirm robustness. Only mark as "robust" if all 3 runs converge.

**After validation succeeds**: Don't stop! Search for other working points:

- Move away from the current working config to find alternative solutions
- Try significantly different parameter combinations (not just small perturbations)
- Assess the importance of coeff_W_L1
- Goal: discover if there are multiple distinct working regions in parameter space

## Log Format

```
## Iter N: [converged/partial/failed]
Node: id=N, parent=P, V=1, N_total=N  # parent=0 for root node (iteration 1)
Config: lr_W=X, lr=Y, coeff_W_L1=Z, batch_size=B
Metrics: test_R2=A, test_pearson=B, connectivity_R2=C, final_loss=D
RankScore: 0.XX  # = (rank - 1) / (N_total - 1), ascending order
Observation: [one line about convergence quality]
Change: [param: old -> new]
```

## Meta-analysis

Every 10 iterations make a meta-analysis of the results
