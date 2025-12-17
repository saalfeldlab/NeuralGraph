# GNN Simulation Landscape Study

## Context

You execute one experimental iteration in an iterative exploration loop.
Each **block of 12 iterations** explores training hyperparameters for a fixed simulation configuration.
At each iteration you modify the learning parameters in the current config file
At each iteration if necessary you modify the rule decision block of protocol file (lines between ## Parent Selection Rule (CRITICAL) and ## END Parent selection Rule (CRITICAL)), indicate if the protocol file is changed or not
At block boundaries (iter 1, 13, 25, 37, ...), you create a new simulation.

## Goal

Map the **simulation landscape**: understand which simulation configurations allow successful GNN training (connectivity_R2 > 0.9) and which simulation configurations are fundamentally harder.

## Analysis Files

- `analysis.log`: metrics from training/test/plot:
  - `spectral_radius`: eigenvalue analysis of connectivity
  - `svd_rank`: SVD rank at 99% variance (activity complexity)
  - `test_R2`: R² between ground truth and rollout prediction
  - `test_pearson`: Pearson correlation per neuron (mean)
  - `connectivity_R2`: R² of learned vs true connectivity weights
  - `final_loss`: final training loss (lower is better)
- `ucb_scores.txt`: provides pre-computed UCB scores for all nodes including current iteration
  at block boundaries, the UCB file will be empty (erased). When UCB file is empty, use `parent=root`.
  ```
  Node 2: UCB=2.175, parent=1, visits=1, R2=0.997 [CURRENT]
  Node 1: UCB=2.110, parent=root, visits=2, R2=0.934
  ```
  - `Node N`:
  - `UCB`: Upper Confidence Bound score = R² + c×√(log(N_total)/visits); higher = more promising to explore
  - `parent`: which node's config was mutated to create this node (root = baseline config)
  - `visits`: how many times this node or its descendants have been explored
  - `R2`: connectivity_R2 achieved by this node's config

## Classification

- **Converged**: connectivity_R2 > 0.9
- **Partial**: connectivity_R2 0.1-0.9
- **Failed**: connectivity_R2 < 0.1

## Simulation Parameters to explore

```yaml
simulation:
  connectivity_type: "chaotic" # or "low_rank"
  Dale_law: True # enforce excitatory/inhibitory separation
  Dale_law_factor: 0.5 # fraction excitatory/inhibitory (0.1 to 0.9)
  low_rank specific:
  connectivity_rank: 20 # only used when connectivity_type="low_rank", range 5-100
```

## Training Parameters to explore

```yaml
training:
  learning_rate_W_start: 2.0E-3 # LR for connectivity weights W range: 1.0E-4 to 1.0E-2
  learning_rate_start: 1.0E-4 # LR for model parameters range:1.0E-5 to 1.0E-3
  coeff_W_L1: 1.0E-5 # L1 regularization on W range: 1.0E-3 to 1.0E-6
  batch_size: 8 # batch size values: 8, 16 ,32
  noise_model_level: 0.0 # add noise to data, it's a good regularizer values: 0, 0.5, 1
```

## Parent Selection Rule (CRITICAL)

**Step 1: select parent node to ccontinue**

- Use `ucb_scores.txt` to select a new node
- If UCB file is empty → `parent=root`
- Otherwise → select node with **highest UCB** as parent

**Step 2: Choose exploration strategy**

| Condition                           | Strategy            | Action                                                      |
| ----------------------------------- | ------------------- | ----------------------------------------------------------- |
| Default                             | **exploit**         | Use highest UCB node, try new mutation                      |
| 3+ consecutive successes (R² ≥ 0.9) | **failure-probe**   | Deliberately try extreme parameter to find failure boundary |
| Found good config                   | **robustness-test** | Re-run same config (no mutation) to verify reproducibility  |

**failure-probe**: After multiple successes, intentionally push parameters to extremes (e.g., 10x lr, 0.1x lr) to map where the config breaks. This helps understand the stability region.

**robustness-test**: Duplicate the best iteration with identical config to verify the result is reproducible, not due to lucky initialization.

**Reversion check**: If reverting a parameter to match a previous node's value, use that node as parent.
Example: If reverting `lr` back to `1E-4` (Node 2's value), use `parent=2`.

## END Parent selection Rule (CRITICAL)

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
Simulation: connectivity_type=[type], Dale_law=[True/False], Dale_law_factor=[F], connectivity_rank = [R] if connectivity_type='low_rank', noise_model_level=[L]
Node: id=N, parent=root
...
```

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
