# Flyvis GNN Training Landscape Study

## Goal

Map the **flyvis GNN training landscape**: understand which training configurations allow successful GNN learning of the fly visual system connectome. The goal is to recover synaptic weights, time constants, and resting potentials from simulated neural activity.

## Context (CRITICAL)

You are a LLM, you are **hyperparameter optimizer** in a meta-learning loop. Your role:

1. **Analyze results**: Read activity plots and metrics from the current GNN training run
2. **Update config**: Modify training parameters for the next iteration based on Parent Selection Rule (see below)
3. **Log decisions**: Append structured observations to the analysis file
4. **Self-improve**: At simulation block boundaries, you are asked edit THIS protocol file to refine your own exploration rules

## Analysis of Files

- `analysis.log`: metrics from training/test/plot:
  - `test_R2`: R² between ground truth and rollout prediction
  - `test_pearson`: Pearson correlation per neuron (mean)
  - `connectivity_R2`: R² of learned vs true connectivity weights
  - `tau_R2`: R² of learned vs true time constants
  - `V_rest_R2`: R² of learned vs true resting potentials
  - `cluster_accuracy`: accuracy of neuron type clustering
  - `final_loss`: final training loss (lower is better)
- `ucb_scores.txt`: provides pre-computed UCB scores for all nodes including current iteration
  at block boundaries, the UCB file will be empty (erased). When UCB file is empty, use `parent=root`.

```
Node 2: UCB=2.175, parent=1, visits=1, R2=0.997 [CURRENT]
Node 1: UCB=2.110, parent=root, visits=2, R2=0.934

```

- `Node N`:
- `UCB`: Upper Confidence Bound score = R² + c×sqrt(log(N_total)/visits); higher = more promising to explore
- `parent`: which node's config was mutated to create this node (root = baseline config)
- `visits`: how many times this node or its descendants have been explored
- `R2`: connectivity_R2 achieved by this node's config

## Classification

- **Converged**: connectivity_R2 > 0.9
- **Partial**: connectivity_R2 0.1-0.9
- **Failed**: connectivity_R2 < 0.1

## Training Parameters to explore

These parameters affect the **GNN training**. Can be changed within a block (when iter_in_block <> n_iter_block)

```yaml
training:
  learning_rate_W_start: 1.0E-3 # LR for connectivity weights W range: 1.0E-4 to 1.0E-2
  learning_rate_start: 5.0E-4 # LR for model parameters range: 1.0E-5 to 1.0E-3
  learning_rate_embedding_start: 1.0E-3 # LR for embeddings range: 1.0E-5 to 1.0E-3
  coeff_W_L1: 5.0E-5 # L1 regularization on W range: 1.0E-6 to 1.0E-3
  coeff_edge_diff: 500 # edge difference regularization range: 0 to 1000
  coeff_edge_norm: 1 # edge normalization regularization range: 0 to 10
  noise_model_level: 0.05 # noise during training range: 0 to 0.2
  batch_size: 1 # batch size values: 1, 2, 4 (flyvis uses small batches due to large network)
  data_augmentation_loop: 25 # data augmentation range: 10 to 100
```

## Simulation Parameters to explore

These parameters affect the **data generation** (simulation). Only change at block boundaries (when iter_in_block == n_iter_block)

```yaml
simulation:
  visual_input_type: "DAVIS" # or "optical_flow" - type of visual input
  n_frames: 64000 # number of simulation frames range: 32000 to 128000
  delta_t: 0.02 # simulation time step range: 0.01 to 0.05
```

## Parent Selection Rule (CRITICAL)

**Step 1: select parent node to continue**

- Use `ucb_scores.txt` to select a new node
- If UCB file is empty -> `parent=root`
- Otherwise -> select node with **highest UCB** as parent

**Step 2: Choose exploration strategy**

| Condition                                       | Strategy            | Action                                                                  |
| ----------------------------------------------- | ------------------- | ----------------------------------------------------------------------- |
| Default                                         | **exploit**         | Use highest UCB node, try new mutation                                  |
| 3+ consecutive successes (R2 >= 0.9)            | **failure-probe**   | Deliberately try extreme parameter to find failure boundary             |
| n_iter_block/4 consecutive successes (R2 >= 0.9)| **explore**         | Use highest UCB node not in last n_iter_block/4 nodes, try new mutation |
| Found good config                               | **robustness-test** | Re-run same config (no mutation) to verify reproducibility              |

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
Config: lr_W=X, lr=Y, lr_emb=Z, coeff_W_L1=W, noise=N
Metrics: test_R2=A, test_pearson=B, connectivity_R2=C, tau_R2=D, V_rest_R2=E, cluster_accuracy=F
Activity: [brief description of dynamics]
Mutation: [param]: [old] -> [new]
Parent rule: [brief description of Parent Selection Rule]
Observation: [one line about result]
Next: parent=P [CRITICAL: specify which node the NEXT iteration should branch from]
```

### Simulation Blocks

Each block = `n_iter_block` iterations exploring one simulation configuration.
The prompt provides: `Block info: block {block_number}, iteration {iter_in_block}/{n_iter_block} within block`

- `block_number`: which simulation block (1, 2, 3, ...)
- `iter_in_block`: current iteration within this block (1 to n_iter_block)
- `n_iter_block`: total iterations per block

### Within block (iter_in_block < n_iter_block):

Only modify training parameters (learning rates, regularization, noise, batch size)

### Block End (iter_in_block == n_iter_block) Log Format

```
## Simulation Block {block_number} Summary (iters X-Y)
Simulation: visual_input_type=[type], n_frames=[N], delta_t=[T]
Best R2: [value] at iter [N]
Best tau_R2: [value], V_rest_R2: [value], cluster_accuracy: [value]
Converged: [Yes/No]
Observation: [what worked/failed for this simulation]
Optimum training: lr_W=[X], lr=[Y], lr_emb=[Z], coeff_W_L1=[W], noise=[N]

--- NEW SIMULATION BLOCK ---
Next simulation: visual_input_type=[type], n_frames=[N], ...
Node: id=N, parent=root
```

## MANDATORY: Block End Actions (when iter_in_block == n_iter_block)

At the **last iteration of each block** (iter_in_block == n_iter_block), you MUST complete ALL of these actions:

### Checklist (complete in order):

- [ ] **1. Write block summary** (see "Block End Log Format" above)
- [ ] **2. Evaluate exploration rules** using metrics below
- [ ] **3. EDIT THIS PROTOCOL FILE** - modify the rules between `## Parent Selection Rule (CRITICAL)` and `## END Parent selection Rule (CRITICAL)`
- [ ] **4. Document your edit** - in the analysis file, state what you changed and why (or state "No changes needed" with justification)

### Evaluation Metrics for Rule Modification:

1. **Branching rate**: Count unique parents in last n_iter_block/4 iters
   - If all sequential (rate=0%) -> ADD exploration incentive to rules
2. **Improvement rate**: How many iters improved R2?
   - If <30% improving -> INCREASE exploitation (raise R2 threshold)
   - If >80% improving -> INCREASE exploration (probe boundaries)
3. **Stuck detection**: Same R2 plateau (+/-0.05) for 3+ iters?
   - If yes -> ADD forced branching rule

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
