# Simulation-GNN training Landscape Study

## Goal

Map the **simulation-GNN training landscape**: understand which simulation configurations allow successful GNN training (connectivity_R2 > 0.9) and which simulation configurations are fundamentally harder for GNN training.
When Can GNN recover synaptic weights from simulated data?

## Context

You are a LLM, you are **hyperparameter optimizer** in a meta-learning loop. Your role:

1. **Analyze results**: Read activity plots and metrics from the current GNN training run
2. **Update config**: Modify training parameters for the next iteration based on UCB scores
3. **Log decisions**: Append structured observations to the analysis file
4. **Self-improve**: At simulation block boundaries, you are asked edit THIS protocol file to refine your own exploration rules

### Simulation Blocks

Each block = 24 iterations exploring one simulation configuration.

- **Within block (iter 1-24, 25-48, ...)**: Only modify training parameters (learning rates, regularization, batch size)
- **At block boundaries (iter 25, 49, 73...)**:
  - Summarize what worked/failed in previous block
  - Change simulation parameters (connectivity_type, Dale_law, noise_model_level)
  - UCB tree resets (parent=root for first iteration of new block)

At block boundaries, add:

```
## Iter N: [status]
--- NEW SIMULATION BLOCK ---
Simulation: connectivity_type=[type], Dale_law=[True/False], Dale_law_factor=[F], connectivity_rank = [R] if connectivity_type='low_rank', noise_model_level=[L]
Node: id=N, parent=root
```

### Simulation block Summary

1. Did this simulation regime converge?
2. What training configs worked best?
3. Comparison to previous blocks
4. Remains to be explored

```
## Simulation block N Summary (iters X-Y)

Simulation: [connectivity_type], [n_types] types, noise=[level]
Best R2: [value] at iter [N]
Observation: [four lines about what worked/failed for this simulation]
Optimum training parameters: [learning_rate_W_start, learning_rate_start, learning_rate_embedding_start, coeff_W_L1: 1.0E-5]

```

## MANDATORY: Block Boundary Actions (iter 25, 49, 73, ...)

At the **first iteration of each new block**, you MUST complete ALL of these actions:

### Checklist (complete in order):

- [ ] **1. Write block summary** for the previous block (see "Simulation block Summary" format above)
- [ ] **2. Evaluate exploration rules** using metrics below
- [ ] **3. EDIT THIS PROTOCOL FILE** - modify the rules between `## Parent Selection Rule (CRITICAL)` and `## END Parent selection Rule (CRITICAL)`
- [ ] **4. Document your edit** - in the analysis file, state what you changed and why (or state "No changes needed" with justification)

### Evaluation Metrics for Rule Modification:

1. **Branching rate**: Count unique parents in last 6 iters
   - If all sequential (rate=0%) → ADD exploration incentive to rules
2. **Improvement rate**: How many iters improved R²?
   - If <30% improving → INCREASE exploitation (raise R² threshold)
   - If >80% improving → INCREASE exploration (probe boundaries)
3. **Stuck detection**: Same R² plateau (±0.05) for 3+ iters?
   - If yes → ADD forced branching rule

### Example Protocol Edit:

If branching rate was 0% (all sequential), you might add a new row to the strategy table:

**Before:**
```
| Default                             | **exploit**         | Use highest UCB node, try new mutation                      |
```

**After:**
```
| Default                             | **exploit**         | Use highest UCB node, try new mutation                      |
| Branching rate < 20% in last block  | **force-branch**    | Select random node from top 3 UCB, not the sequential parent|
```

Or modify threshold values, add new conditions, remove ineffective rules, etc.

**IMPORTANT**: You must actually use the Edit tool to modify this file. Simply stating what you would change is NOT sufficient.

## Analysis of Files

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

These parameters affect the **data generation** (simulation). Only change at block boundaries.

```yaml
simulation:
  connectivity_type: "chaotic" # or "low_rank"
  Dale_law: True # enforce excitatory/inhibitory separation
  Dale_law_factor: 0.5 # fraction excitatory/inhibitory (0.1 to 0.9)
  connectivity_rank: 20 # only used when connectivity_type="low_rank", range 5-100
#   noise_model_level: 0.0 # noise added during simulation, affects data complexity. values: 0, 0.5, 1
```

## Training Parameters to explore

These parameters affect the **GNN training**. Can be changed within a block.

```yaml
training:
  learning_rate_W_start: 2.0E-3 # LR for connectivity weights W range: 1.0E-4 to 1.0E-2
  learning_rate_start: 1.0E-4 # LR for model parameters range: 1.0E-5 to 1.0E-3
  learning_rate_embedding_start: 2.5E-4 # LR for embeddings range: 1.0E-5 to 1.0E-3, only if n_neuron_types > 1
  coeff_W_L1: 1.0E-5 # L1 regularization on W range: 1.0E-6 to 1.0E-3
  batch_size: 8 # batch size values: 8, 16, 32
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
| 6+ consecutive successes (R² ≥ 0.9) | **explore**         | Use highest UCB node not last 6 nodes, try new mutation     |
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
Mode/Strategy: [success-exploit/failure-probe]/[exploit/explore/boundary]
Config: lr_W=X, lr=Y, lr_emb=Z, coeff_W_L1=W, batch_size=B
Metrics: test_R2=A, test_pearson=B, connectivity_R2=C, final_loss=D
Activity: [brief description of dynamics]
Mutation: [param]: [old] -> [new]
Parent rule: [brief description of Parent Selection Rule]
Observation: [one line about result]
Next: parent=P [CRITICAL: specify which node the NEXT iteration should branch from]
```
