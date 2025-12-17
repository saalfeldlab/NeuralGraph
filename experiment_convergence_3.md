# GNN Convergence Study

## Context

You execute one experimental iteration in an iterative exploration loop.

## Goal

Find robust training hyperparameters that ensure GNN convergence for a given simulation configurations.
The GNN should reliably recover the connectivity matrix W (connectivity_R2 > 0.9).

**Note**: Training is stochastic - convergence may fail ~1 in 3 runs with the same config. A config is considered "robust" only if it converges consistently across multiple runs.

Key questions:

- What learning rate combinations ensure stable convergence?
- How does L1 regularization (coeff_W_L1) affect W recovery?
- What is the robust operating range for these hyperparameters?

## Starting Point

with training parameters that sometimes fail to converge (connectivity_R2 <0.1>).

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
  - `test_pearson`: Pearson correlation between ground truth and rollout prediction
  - `connectivity_R2`: R² of learned vs true connectivity weights
  - `final_loss`: final training loss (lower is better)

## Classification

- **Converged**: connectivity_R2 > 0.9
- **Partial**: connectivity_R2 0.1-0.9 (grey zone)
- **Failed**: connectivity_R2 < 0.1

## UCB Tree Exploration (PUCT)

`ucb_scores.txt` provides pre-computed UCB scores using the **PUCT formula**:

```
UCB(u) = R2(u) + c * sqrt(N_total) / (1 + V(u))
```

Where:

- `R2(u)`: connectivity_R2 for node u (exploitation term)
- `c = 1.0`: exploration constant
- `N_total`: total number of nodes in tree
- `V(u)`: visit count (backpropagated from descendants)

Example output:

```
Node 47: UCB=4.535, parent=45, visits=1, R2=1.000
Node 51: UCB=4.535, parent=44, visits=1, R2=1.000
Node 4: UCB=1.138, parent=3, visits=48, R2=1.000
```

**Visit count semantics (PUCT backpropagation)**:

- Each node starts with V=1 (its own creation)
- When a child is created, all ancestors get V += 1
- Leaf nodes: V=1 (just created)
- Root after 50 expansions: V=50

**Interpretation**:

- **Higher UCB = more promising to explore from**
- Leaf nodes (V=1) have high exploration bonus → encourages breadth
- Well-explored branches (high V) have low exploration bonus → discourages revisiting
- High R2 nodes are favored for exploitation

## Protocol

**IMPORTANT: Change only ONE parameter per iteration** to isolate effects.

### Dual-Mode Exploration (70% Success / 30% Failure)

The goal is to map **both** the working region AND the failure boundaries. Use this allocation:

- **70% of iterations**: Exploit/explore from HIGH UCB nodes (R2 > 0.5)

  - Goal: Find robust working configs, validate them
  - Strategy: `exploit` or `explore`

- **30% of iterations**: Probe from LOW R2 nodes or push beyond boundaries
  - Goal: Understand WHY configs fail, find failure boundaries
  - Strategy: `boundary` or `failure-probe`

### How to decide which mode:

```
if iteration % 10 in [3, 6, 9]:  # iterations 3, 6, 9, 13, 16, 19, ...
    MODE = "failure-probe"
    → Pick node with LOWEST R2 that hasn't been probed
    → OR push a working config toward extreme values
else:
    MODE = "success-exploit"
    → Pick node with HIGHEST UCB (standard PUCT)
```

### Failure Probing Strategies

When in `failure-probe` mode:

1. **From failed node**: Try small perturbation to see if failure is sharp or gradual
2. **From working node**: Push ONE parameter to extreme (10x or 0.1x)
3. **Boundary search**: If you found failure at X, try X/2 to find exact boundary

### Success Exploitation (Standard)

1. Start with baseline: lr_W=2E-3, lr=5E-4, coeff_W_L1=5E-6, batch_size=8
2. Read `ucb_scores.txt` to decide which node to explore from
3. Then vary ONE parameter at a time from chosen parent

### Handling Stochastic Failures

When expanding from node P:

- **First attempt fails** (conn_R2 < 0.1): Create new node (id=N+1, parent=P, same config)
  - Log: `Observation: retry of node N (stochastic failure)`
- **Second attempt fails**: Mark as deterministic failure (not stochastic)
  - Log: `Observation: deterministic failure, boundary found`
- **First fails, second succeeds**: Mark config as "unstable"
  - Log: `Observation: unstable (stochastic boundary)`

### Validation

When a good config is found (connectivity_R2 > 0.9), repeat it 3 times without changes to confirm robustness. Only mark as "robust" if all 3 runs converge.

### Key Questions to Answer

1. **Working region**: What parameter ranges reliably converge?
2. **Failure boundaries**: Where exactly does convergence break?
3. **Failure modes**: WHY do configs fail? (too fast? too slow? unstable?)
4. **Sharp vs gradual**: Are boundaries sharp cliffs or gradual degradation?

## Log Format

```
## Iter N: [converged/partial/failed]
Node: id=N, parent=P
Mode: [success-exploit/failure-probe]
Strategy: [exploit/explore/boundary/failure-probe]
Config: lr_W=X, lr=Y, coeff_W_L1=Z, batch_size=B
Metrics: test_R2=A, test_pearson=B, connectivity_R2=C, final_loss=D
Failure mode: [diverged | flat_activity | slow_convergence | numerical_instability]
Mutation: [param]: [old] → [new]
Observation: [one line about result and WHY]
```

## Meta-analysis

Every 10 iterations make a meta-analysis of the results
