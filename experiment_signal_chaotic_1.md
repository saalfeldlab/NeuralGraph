# Simulation-GNN Training Landscape Study

## Goal

Map the **simulation-GNN training landscape**: understand which simulation configurations allow successful GNN training (connectivity_R2 > 0.9) and which are fundamentally harder.

## Iteration loop structure

Each block = `n_iter_block` iterations exploring one simulation configuration.
The prompt provides: `Block info: block {block_number}, iteration {iter_in_block}/{n_iter_block} within block`

## File structure (CRITICAL)

You maintain TWO files:

### 1. Full Log (append-only record)

**File**: `{config}_analysis.md`

- Append every iteration's full log entry
- Append block summaries
- **Never read this file** — it's for human record only

### 2. Working Memory (active knowledge)

**File**: `{config}_memory.md`

- **READ at start of each iteration**
- **UPDATE at end of each iteration**
- Contains: knowledge base + previous block + current block only
- Fixed size (~500 lines max)

---

## Iteration Workflow Step 1-4, every iteration

### Step 1: Read Working Memory if exists

Read `{config}_memory.md` to recall:

- Established principles
- Previous block findings
- Current block progress

### Step 2: Analyze Current Results

- `analysis.log`: metrics from training/test/plot:
  - `spectral_radius`: eigenvalue analysis of connectivity
  - `svd_rank`: SVD rank at 99% variance (activity complexity)
  - `test_R2`: R² between ground truth and rollout prediction
  - `test_pearson`: Pearson correlation per neuron (mean)
  - `connectivity_R2`: R² of learned vs true connectivity weights
  - `final_loss`: final training loss (lower is better)
- `ucb_scores.txt`: provides pre-computed UCB scores for all nodes including current iteration
  at block boundaries, the UCB file will be empty (erased). When UCB file is empty, use `parent=root`.

Node 2: UCB=2.175, parent=1, visits=1, R2=0.997
Node 1: UCB=2.110, parent=root, visits=2, R2=0.934

### Step 3: Write Outputs

Append to Full Log\*\* (`{config}_analysis.md`) and Current Block sections of `{config}_memory.md` :

Log Form

```
## Iter N: [converged/partial/failed]
Node: id=N, parent=P
Mode/Strategy: [success-exploit/failure-probe]/[exploit/explore/boundary]
Config: lr_W=X, lr=Y, lr_emb=Z, coeff_W_L1=W, batch_size=B, low_rank_factorization=[T/F], low_rank=R, n_frames=NF
Metrics: test_R2=A, test_pearson=B, connectivity_R2=C, final_loss=D
Activity: [brief description]
Mutation: [param]: [old] -> [new]
Parent rule: [one line]
Observation: [one line]
Next: parent=P
```

### Step 4: Edit config file for next iteration

(The config path is provided in the prompt as "Current config")

- Classification

- **Converged**: connectivity_R2 > 0.9
- **Partial**: connectivity_R2 0.1-0.9
- **Failed**: connectivity_R2 < 0.1

- Training Parameters (change within block)

```yaml
training:
  learning_rate_W_start: 2.0E-3 # range: 1E-4 to 1E-2
  learning_rate_start: 1.0E-4 # range: 1E-5 to 1E-3
  learning_rate_embedding_start: 2.5E-4
  coeff_W_L1: 1.0E-5 # range: 1E-6 to 1E-3
  batch_size: 8 # values: 8, 16, 32
  low_rank_factorization: False or True
  low_rank: 20 # range: 5-100
```

- Simulation Parameters (change at block boundaries only)

```yaml
simulation:
  n_frames: 10000
  connectivity_type: "chaotic" # or "low_rank"
  Dale_law: False or True
  Dale_law_factor: 0.5
  connectivity_rank: 20 if low_rank
```

## Parent Selection Rule (CRITICAL)

**Step A: Select parent node**

- Read `ucb_scores.txt`
- If empty → `parent=root`
- Otherwise → select node with **highest UCB**

**Step B: Choose strategy**

| Condition                            | Strategy            | Action                             |
| ------------------------------------ | ------------------- | ---------------------------------- |
| Default                              | **exploit**         | Highest UCB node, try mutation     |
| 3+ consecutive R² ≥ 0.9              | **failure-probe**   | Extreme parameter to find boundary |
| n_iter_block/4 consecutive successes | **explore**         | Select outside recent chain        |
| Good config found                    | **robustness-test** | Re-run same config                 |

## END Parent Selection Rule

## Block Workflow step 1 to 2, at the end of a block iter_in_block==n_iter_block

**STEP 1 COMPULSORY modify Protocol (this file)**

- [ ] Evaluate rules

1. **Branching rate**: Count unique parents in last n_iter_block/4 iters
   - If all sequential (rate=0%) → ADD exploration incentive to rules
2. **Improvement rate**: How many iters improved R²?
   - If <30% improving → INCREASE exploitation (raise R² threshold)
   - If >80% improving → INCREASE exploration (probe boundaries)
3. **Stuck detection**: Same R² plateau (±0.05) for 3+ iters?
   - If yes → ADD forced branching rule

**STEP 2. Update Working Memory** (`{config}_memory.md`):

- Update Knowledge Base with confirmed principles
- Add row to Regime Comparison Table
- Replace Previous Block Summary
- Clear Current Block sections
- Write hypothesis for next block

## Working Memory Structure

```markdown
# Working Memory

## Knowledge Base (accumulated across all blocks)

### Regime Comparison Table

| Block | Regime | Best R² | Optimal lr_W | Optimal L1 | Key finding |
| ----- | chaotic Dale_law=False | ------- | ------------ | ---------- | ----------- |
| ----- | low_rank=50 Dale_law=True

### Established Principles

[Confirmed patterns that apply across regimes]

### Open Questions

[Patterns needing more testing, contradictions]

---

## Previous Block Summary (Block N-1)

[Replaced entirely at each block boundary]

---

## Current Block (Block N)

### Block Info

Simulation: connectivity_type=X, Dale_law=Y, ...
Iterations: M to M+n_iter_block

### Hypothesis

[Prediction for this block, stated before running]

### Iterations This Block

[Current block iterations only — cleared at block boundary]

### Emerging Observations

[Running notes on what's working/failing]
```

---

## Knowledge Base Guidelines

### What to Add to Established Principles

✓ "Constrained connectivity needs lower lr_W" (causal, generalizable)
✓ "L1 > 1e-04 fails for low_rank" (boundary condition)
✓ "Effective rank < 15 requires factorization=True" (theoretical link)
✗ "lr_W=0.01 worked in Block 4" (too specific)
✗ "Block 3 converged" (not a principle)

### What to Add to Open Questions

- Patterns needing more testing
- Contradictions between blocks
- Theoretical predictions not yet verified

## Memory Size Control

Target Knowledge Base: ~100 lines max (grows slowly)
If approaching limit:

---

## Theoretical Background

### Spectral radius

- ρ(W) < 1: activity decays → harder to constrain W
- ρ(W) ≈ 1: edge of chaos → rich dynamics → good recovery
- ρ(W) > 1: unstable

### Effective rank

- High (30+): full W recoverable
- Low (<15): only subspace identifiable → need factorization

### Low-rank connectivity

- W = W_L @ W_R constrains solution space
- Without factorization: spurious full-rank solutions

### Learning rates

- lr_W:lr ratio matters (typically 20:1 to 50:1)
- Too fast φ learning → noisy W gradients
