# GNN Simulation Landscape Study

## Context

You execute one experimental iteration in an iterative exploration loop.
Each **block of 12 iterations** explores training hyperparameters for a fixed simulation configuration.
At block boundaries (iter 1, 13, 25, 37, ...), you create a new simulation and reset the UCB tree.

## Goal

Map the **simulation landscape**: understand which simulation configurations allow successful GNN training (connectivity_R2 > 0.9) and which are fundamentally harder.

Key questions:

- Which simulation types (chaotic vs low_rank) are easier for GNN to learn?
- How does neuron type diversity affect learnability?
- Can noise_model_level > 0 help verify solvability before attempting noise_model_level=0?
- What training hyperparameters work across different simulation regimes?

## Block Structure

- **Block 0**: iterations 1-12 (first simulation)
- **Block 1**: iterations 13-24 (second simulation)
- **Block 2**: iterations 25-36 (third simulation)
- etc.

At block boundaries, the UCB file will be empty (erased). When UCB file is empty, use `parent=root`.

## Simulation Parameters to Explore

```yaml
simulation:
  connectivity_type: "chaotic" # or "low_rank"
  Dale_law: True # enforce excitatory/inhibitory separation
  Dale_law_factor: 0.5 # fraction excitatory (0.1 to 0.9)
  # low_rank specific:
  # connectivity_rank: 5-100 (only when connectivity_type="low_rank")
```

```yaml
training:
  noise_model_level: 0.0 # 0 to 5 (0 = hard target, >0 = verify solvability)
```

**Dale's Law**: When `Dale_law=True`, each neuron is either excitatory (positive weights) or inhibitory (negative weights). `Dale_law_factor` controls the fraction of excitatory neurons (e.g., 0.8 = 80% excitatory, 20% inhibitory).

**Strategy**: Start with `noise_model_level > 0` to verify the GNN can solve the problem in principle, then reduce to `noise_model_level=0` as the final target.

## Training Parameters (within block)

```yaml
training:
  learning_rate_W_start: 2.0E-3 # LR for connectivity weights W
  learning_rate_start: 1.0E-4 # LR for model parameters
  coeff_W_L1: 1.0E-5 # L1 regularization on W
  batch_size: 8 # batch size
```

Parameter ranges:

- `learning_rate_W_start`: 1E-4 to 1E-2
- `learning_rate_start`: 1E-5 to 1E-3
- `coeff_W_L1`: 0 to 1E-4
- `batch_size`: 1, 4, 8, 16

## Analysis Files

- `analysis.log`: metrics from training/test/plot
- `ucb_scores.txt`: UCB tree for current block only

## Classification

- **Converged**: connectivity_R2 > 0.9
- **Partial**: connectivity_R2 0.1-0.9
- **Failed**: connectivity_R2 < 0.1

## UCB Tree Exploration

`ucb_scores.txt` shows nodes from **current block only**.

Example:

```
=== UCB Scores (Block 2, iters 25-36, N=8, c=1.0) ===

Node 32: UCB=2.156, parent=29, visits=1, R2=0.873
Node 29: UCB=1.892, parent=root, visits=4, R2=0.921
Node 25: UCB=1.234, parent=25, visits=8, R2=0.654
```

### Parent Selection Rule (CRITICAL)

**The `parent` field indicates which node's CONFIG you are modifying not the iteration number**

**Step 1: Find parent node**

- If UCB file is empty → `parent=root`
- Otherwise → select node with **highest UCB** as parent

**Step 2: Choose exploration strategy**

| Condition | Strategy | Action |
|---|---|---|
| Default | **exploit** | Use highest UCB node as parent, try new mutation |
| 3+ consecutive successes (R² ≥ 0.9) | **failure-probe** | Deliberately try extreme parameter to find failure boundary |
| Found good config | **robustness-test** | Re-run same config (no mutation) to verify reproducibility |

**failure-probe**: After multiple successes, intentionally push parameters to extremes (e.g., 10x lr, 0.1x lr) to map where the config breaks. This helps understand the stability region.

**robustness-test**: Duplicate the best iteration with identical config to verify the result is reproducible, not due to lucky initialization.

**Reversion check**: If reverting a parameter to match a previous node's value, use that node as parent.
Example: If reverting `lr` back to `1E-4` (Node 2's value), use `parent=2`.

## Protocol

### Block Start (iter 1, 13, 25, ...)

When starting a new block:

1. **Create new simulation** by modifying simulation parameters in config
2. **Set parent=root** (UCB file will be empty)
3. **Start with baseline training config** or carry over best config from previous block

Simulation exploration order (suggested):

- Block 0: chaotic, Dale_law=True, factor=0.5, noise=0.0
- Block 4: chaotic, Dale_law=True, factor=0.8, noise=0.0 (mostly excitatory)
- Block 5: chaotic, Dale_law=True, factor=0.2, noise=0.0 (mostly inhibitory)
- Block 6: low_rank (rank=20), Dale_law=False, noise=0.0
- etc.

### Block Summary

At the end of each block (iter 12, 24, 36, ...), write a brief summary:

1. Did this simulation regime converge?
2. What training configs worked best?
3. Comparison to previous blocks

```
### Block N Summary (iters X-Y)
Simulation: [connectivity_type], [n_types] types, noise=[level]
Best R2: [value] at iter [N]
Observation: [four lines about what worked/failed for this simulation]
Optimum training parameters: [learning_rate_W_start, learning_rate_start, learning_rate_embedding_start, coeff_W_L1: 1.0E-5]
```

## Log Format

```
## Iter N: [converged/partial/failed]
Node: id=N, parent=P
Mode: [success-exploit/failure-probe]
Strategy: [exploit/explore/boundary]
Config: lr_W=X, lr=Y, lr_emb=Z, coeff_W_L1=W, batch_size=B
Metrics: test_R2=A, test_pearson=B, connectivity_R2=C, final_loss=D
Activity: [brief description of dynamics]
Mutation: [param]: [old] -> [new]
Observation: [one line about result]
```

For block boundaries, add:

```
## Iter N: [status]
--- NEW BLOCK ---
Simulation: connectivity_type=[type], Dale_law=[True/False], Dale_law_factor=[F], noise_model_level=[L]
Node: id=N, parent=root
...
```

## Example Workflow

**Iter 1** (Block 0 start):

```
## Iter 1: partial
--- NEW BLOCK ---
Simulation: connectivity_type=chaotic, Dale_law=False, noise_model_level=2.0
Node: id=1, parent=root
Mode: success-exploit
Strategy: explore
Config: lr_W=2.0E-3, lr=1.0E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.732, test_pearson=0.679, connectivity_R2=0.758, final_loss=1.4e+03
Activity: healthy oscillatory dynamics
Mutation: baseline config (first run)
Observation: partial convergence with noise=2.0; lr may need tuning
```

**Iter 13** (Block 1 start):

```
## Iter 13: partial
--- NEW BLOCK ---
Simulation: connectivity_type=chaotic, Dale_law=False, noise_model_level=0.0
Node: id=13, parent=root
Mode: success-exploit
Strategy: explore
Config: lr_W=2.0E-3, lr=5.0E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: ...
Activity: ...
Mutation: baseline (carried from block 0 best)
Observation: attempting hard target (noise=0) with config that worked for noise=2
```

**Iter 25** (Block 2 start - Dale's Law):

```
## Iter 25: partial
--- NEW BLOCK ---
Simulation: connectivity_type=chaotic, Dale_law=True, Dale_law_factor=0.5, noise_model_level=2.0
Node: id=25, parent=root
Mode: success-exploit
Strategy: explore
Config: lr_W=2.0E-3, lr=5.0E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: ...
Activity: ...
Mutation: baseline (carried from block 1 best)
Observation: first Dale's Law block with balanced E/I (50/50)
```
