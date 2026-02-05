# Low-Rank GNN Training Parameter Optimization

**Reference**: See `neural-gnn/paper/main.pdf` for context.

## Goal

Find GNN training hyperparameters that recover the connectivity matrix W from **low-rank neural dynamics** (connectivity_type=low_rank, rank=20, n_neurons=100).

**This is a fixed-regime exploration**: simulation parameters are FROZEN. Only GNN training parameters may be changed. There are NO block boundary simulation changes.

## Known Challenges (from prior exploration)

Low-rank connectivity (rank=20, n=100) produces data with effective rank ~12, which is the hardest regime for W recovery:
- Low eff_rank means fewer distinguishable activity modes → more equivalent W solutions
- Degeneracy is the primary failure mode: the GNN learns correct dynamics (high test_pearson) from wrong W (low connectivity_R2)
- MLP compensation: lin_edge and lin_phi reshape their nonlinear mappings to compensate for incorrect W

## Prior Knowledge (starting points from 188-iteration landscape exploration)

- `lr_W=3E-3` is optimal for low-rank at 10k frames
- `coeff_W_L1=1E-6` is critical — L1=1E-5 degrades dynamics in low eff_rank regimes
- `coeff_edge_diff=10000` constrains lin_edge monotonicity, reducing MLP compensation ability
- Two-phase training helps: no L1 in early epochs lets W converge first, then L1 refines sparsity
- `lr=1E-4` is safe (lr-ceiling-global exception: lr=2E-4 may work in low eff_rank)
- Overtraining causes degeneracy — more epochs is NOT always better
- `batch_size=8` is safe; batch=16 may degrade at L1=1E-5

**These are starting hypotheses to validate and refine, not fixed truths.**

---

## Iteration Loop Structure

Each block = `n_iter_block` iterations. All blocks use the same simulation.
The prompt provides: `Block info: block {block_number}, iteration {iter_in_block}/{n_iter_block} within block`

## File Structure (CRITICAL)

You maintain **TWO** files:

### 1. Full Log (append-only record)

**File**: `{config}_analysis.md`

- Append every iteration's full log entry
- Append block summaries
- **Never read this file** — it's for human record only

### 2. Working Memory

**File**: `{config}_memory.md`

- **READ at start of each iteration**
- **UPDATE at end of each iteration**
- Contains: established principles + previous blocks summary + current block iterations
- Fixed size (~500 lines max)

---

## Iteration Workflow (Steps 1-5, every iteration)

### Step 1: Read Working Memory

Read `{config}_memory.md` to recall:

- Established principles
- Previous block findings
- Current block progress

### Step 2: Analyze Current Results

**Metrics from `analysis.log`:**

- `effective rank (99% var)`: **CRITICAL** - Extract as `eff_rank=N`
- `test_R2`: R² between ground truth and rollout prediction
- `test_pearson`: Pearson correlation
- `connectivity_R2`: R² of learned vs true connectivity weights
- `final_loss`: final training loss
- `cluster_accuracy`: neuron classification
- `kinograph_R2`: mean per-frame R² between GT and GNN rollout kinographs
- `kinograph_SSIM`: structural similarity between kinographs
- `kinograph_Wasserstein`: population mode Wasserstein distance (0 = identical, 1 = one-σ shift)

**Example analysis.log format:**

```
spectral radius: 1.029
--- activity ---
  effective rank (90% var): 8
  effective rank (99% var): 12   <-- Extract this value for eff_rank
```

**Classification:**

- **Converged**: connectivity_R2 > 0.9
- **Partial**: connectivity_R2 0.1-0.9
- **Failed**: connectivity_R2 < 0.1

**Degeneracy Detection (CRITICAL — check every iteration):**

Compute the **degeneracy gap** = `test_pearson - connectivity_R2`:

| test_pearson | connectivity_R2 | Degeneracy gap | Diagnosis |
|:---:|:---:|:---:|---|
| > 0.95 | > 0.9 | < 0.1 | **Healthy** — correct W |
| > 0.95 | 0.3–0.9 | 0.1–0.7 | **Degenerate** — MLP compensation |
| > 0.95 | < 0.3 | > 0.7 | **Severely degenerate** |
| < 0.5 | < 0.5 | ~0 | **Failed** — not degeneracy |

**When degeneracy gap > 0.3, DO NOT trust dynamics metrics as evidence of learning quality.**

**Log degeneracy in the iteration entry when detected:**
```
Degeneracy: gap=0.53 (test_pearson=0.999, conn_R2=0.466) — MLP compensation suspected
```

**Upper Confidence Bound (UCB) scores from `ucb_scores.txt`:**

- Provides computed UCB scores for all exploration nodes within a block
- At block boundaries, the UCB file will be empty (erased). When empty, use `parent=root`

### Step 3: Write Outputs

Append to Full Log (`{config}_analysis.md`) and **Current Block** sections of `{config}_memory.md`:

- In memory.md: Insert iteration log in "Iterations This Block" section (BEFORE "Emerging Observations")
- Update "Emerging Observations" at the END of the file

**Log Form:**

```
## Iter N: [converged/partial/failed]
Node: id=N, parent=P
Mode/Strategy: [exploit/explore/boundary/principle-test/degeneracy-break]
Config: seed=S, lr_W=X, lr=Y, lr_emb=Z, coeff_W_L1=W, coeff_edge_diff=D, n_epochs_init=I, first_coeff_L1=F, batch_size=B
Metrics: test_R2=A, test_pearson=B, connectivity_R2=C, cluster_accuracy=D, final_loss=E, kino_R2=F, kino_SSIM=G, kino_WD=H
Activity: eff_rank=R, spectral_radius=S, [brief description]
Mutation: [param]: [old] -> [new]
Parent rule: [one line]
Observation: [one line]
Next: parent=P
```

**CRITICAL: The `Mutation:` line is parsed by the UCB tree builder. Always include the exact parameter change (e.g., `Mutation: lr_W: 2E-3 -> 5E-3`).**

### Step 4: Parent Selection Rule in UCB tree

Step A: Select parent node

- Read `ucb_scores.txt`
- If empty → `parent=root`
- Otherwise → select node with **highest UCB** as parent

**CRITICAL**: The `parent=P` in the Node line must be the **node ID** (integer), NOT "root" (unless UCB file is empty).

Step B: Choose strategy

| Condition | Strategy | Action |
|---|---|---|
| Default | **exploit** | Highest UCB node, conservative mutation |
| 3+ consecutive R² ≥ 0.9 | **failure-probe** | Extreme parameter to find boundary |
| n_iter_block/4 consecutive successes | **explore** | Select outside recent chain |
| degeneracy gap > 0.3 for 3+ iters | **degeneracy-break** | Increase coeff_edge_diff, L1, or reduce training duration |
| Same R² plateau (±0.05) for 3+ iters | **forced-branch** | Select 2nd-highest UCB, switch param dimension |
| 4+ consecutive same-param mutations | **switch-dimension** | Change a different parameter |
| 2+ distant nodes with R² > 0.9 | **recombine** | Merge best params from both nodes |
| test_R2 > 0.998 plateau for 3+ iters | **dimension-sweep** | Explore untested param dimensions (lr_emb, n_epochs_init, first_coeff_L1, edge_diff) |
| improvement rate < 30% in block | **exploit-tighten** | Keep best config, mutate secondary params conservatively |
| best test_R2 unchanged for 2+ batches | **regime-shift** | Change seed, training_single_type, or n_epochs — shift to orthogonal dimension |
| all perturbations from best degrade | **seed-robustness** | Replay best config at new seed to test generalization |
| best test_R2 unchanged for 2+ blocks | **cross-seed-optimize** | Focus on closing the gap between seeds — test n_epochs_init, batch_size, lr_W fine-tuning at the weaker seed |
| new recipe beats old at 2+ seeds | **universal-recipe-validate** | Test the new recipe at all remaining seeds to confirm universality |
| same config gives R2 range > 0.05 across runs | **variance-reduction** | Test recipe at new seed or with different aug/epochs to find lower-variance variant |
| all primary params exhausted, variance > 0.01 | **batch-size-sweep** | Test batch_size=8 at best per-seed configs — different batch size changes optimization trajectory and may reduce variance |
| best seed R2 > 0.995 and worst seed R2 < 0.99 | **seed-gap-close** | At the weaker seed, try coeff_edge_diff>10000, batch_size=16, or training_single_type=False to close the gap |
| L1 changes tested at 3+ seeds with degradation | **L1-lock** | Stop testing L1 values between 1E-6 and 1E-5 for non-99 seeds — the L1 landscape is a cliff, not a gradient |
| new weak seed found (R2 < 0.90) | **weak-seed-lr_W-sweep** | Sweep lr_W at [3E-3, 4E-3, 6E-3, 7E-3, 8E-3] + test L1=1E-6 at this seed — prior pattern: each weak seed has a unique lr_W or L1 optimum |
| 6+ seeds tested with per-seed optima found | **recipe-catalogue** | Stop searching for universal recipe — catalogue per-seed optima and test robustness at new seeds to expand coverage |
| lr_W peak localized to 1E-3 range at a seed | **peak-refine** | Test ±0.5E-3 around peak lr_W to see if finer tuning helps — only if test_R2 < 0.995 at peak |
| all 7+ seeds ≥0.985 with per-seed tuning | **new-seed-stress** | Test at fresh seeds (e.g., 1000, 2000) using default recipe first, then per-seed tuning if needed — measure recipe transfer |
| 6+ attempts at a seed all R2<0.9 | **radical-rescue** | Try extreme interventions: n_epochs=3+, lr_W≤2E-3, lr=5E-5, n_epochs_init=0, or declare seed unlearnable and move on |
| abandoned seed with conn_R2<0.4 at standard recipe | **hard-seed-rescue** | Apply lr_W=4E-3+L1=1E-6+3ep triple combo — this rescued seed=1000 from 0.516→0.991. previous "unlearnable" declarations used only standard-recipe perturbations (one param at a time). the triple combo changes the optimization landscape fundamentally |

### Step 5: Edit Config File

Edit config file for next iteration.

**CRITICAL: Config Parameter Constraints**

**DO NOT add new parameters to the `claude:` section.** Only these fields are allowed:
- `n_epochs`: int
- `data_augmentation_loop`: int
- `n_iter_block`: int
- `ucb_c`: float (0.5-3.0)

**DO NOT change `simulation:` parameters except `seed`.** The simulation regime is fixed for this exploration.

**Simulation Parameters (only seed is mutable):**

```yaml
training:
  seed: 137                        # changing seed generates a DIFFERENT connectivity matrix W
                                   # use different seeds to test robustness across W samples
```

Changing `seed` produces a new random low-rank connectivity matrix. This lets you track whether a training configuration works for one specific W realization or generalizes across multiple W samples. Log the seed in Config line and note when a mutation is a seed change.

**Training Parameters (the exploration space):**

Mutate ONE parameter at a time for causal understanding.

```yaml
training:
  learning_rate_W_start: 3.0E-3   # range: 1E-4 to 1E-2
  learning_rate_start: 1.0E-4     # range: 1E-5 to 1E-3
  learning_rate_embedding_start: 2.5E-4
  coeff_W_L1: 1.0E-5              # range: 1E-6 to 1E-3
  coeff_edge_diff: 10000          # range: 100 to 50000 — KEY for low-rank
  batch_size: 8                   # values: 8, 16, 32

  # Two-phase training
  n_epochs_init: 2                # epochs in phase 1 (no L1)
  first_coeff_L1: 0               # L1 during phase 1 (typically 0)
  # Note: coeff_W_L1 applies in phase 2 (after n_epochs_init)

  training_single_type: True      # can try False
```

**Claude Exploration Parameters:**

```yaml
claude:
  ucb_c: 1.414    # UCB exploration constant (0.5-3.0)
```

---

## Block Workflow (Steps 1-3, every end of block)

Triggered when `iter_in_block == n_iter_block`

### STEP 1: Edit Instructions (this file)

You **MUST** use the Edit tool to add/modify parent selection rules.

**Evaluate:**
- Branching rate < 20% → ADD exploration rule
- Improvement rate < 30% → INCREASE exploitation
- Same R² plateau for 3+ iters → ADD forced branching
- > 4 consecutive same-param mutations → ADD switch-dimension rule

### STEP 2: Choose Next Block Focus

Since simulation is fixed, blocks explore different **training parameter subspaces**:
- Block 1: lr_W sweep (central parameter)
- Block 2: L1 / coeff_edge_diff interaction
- Block 3: Two-phase training parameters (n_epochs_init, first_coeff_L1)
- Block 4+: Refine based on findings

**At block boundaries, choose which parameter subspace to explore next.**

### STEP 3: Update Working Memory

Update `{config}_memory.md`:

- Update Knowledge Base with confirmed principles
- Replace Previous Block Summary with **short summary** (2-3 lines)
- Clear "Iterations This Block" section
- Write hypothesis for next block

---

# Working Memory Structure

## Knowledge Base (accumulated across all blocks)

### Best Configurations Found

| Blk | lr_W | lr | L1 | edge_diff | n_ep_init | first_L1 | batch | conn_R2 | test_R2 | Finding |
| --- | ---- | -- | -- | --------- | --------- | -------- | ----- | ------- | ------- | ------- |
| 1   | 3E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | ? | ? | baseline |

### Established Principles

[Confirmed patterns — require 3+ supporting iterations]

### Open Questions

[Patterns needing more testing, contradictions]

---

## Previous Block Summary (Block N-1)

[Short summary only]

---

## Current Block (Block N)

### Block Info

Focus: [which parameter subspace]

### Hypothesis

[Prediction for this block]

### Iterations This Block

[Current block iterations — cleared at block boundary]

### Emerging Observations

[Running notes]
**CRITICAL: This section must ALWAYS be at the END of memory file.**

---

## Background

### GNN Architecture (Signal_Propagation)

```
du/dt = lin_phi(u, a) + W @ lin_edge(u, a)
```

- `lin_edge` (MLP): message function on edges
- `lin_phi` (MLP): node update function
- `W`: learnable connectivity matrix (100 × 100)
- `a`: learnable node embeddings

### Two-Phase Training

Phase 1 (first `n_epochs_init` epochs):
- Uses `first_coeff_L1` (typically 0) instead of `coeff_W_L1`
- Lets W converge without L1 pressure — find the right structure first

Phase 2 (remaining epochs):
- Uses `coeff_W_L1` for L1 regularization
- Refines W sparsity pattern while maintaining structure from phase 1

### Training Loss and Regularization

```
L = L_pred + coeff_W_L1·||W||₁ + coeff_edge_diff·L_edge_diff
```

- `coeff_W_L1`: L1 on W (sparsity)
- `coeff_edge_diff`: enforces monotonicity of lin_edge (prevents MLP compensation)
  - Higher values (10000+) constrain the MLP more → W must carry the signal
  - This is the KEY lever against degeneracy in low-rank regime

### Low-Rank Regime Specifics

- True W has rank 20 (100 neurons): W = W_L @ W_R where W_L ∈ ℝ^(100×20), W_R ∈ ℝ^(20×100)
- Effective rank of activity data ~12 (99% variance)
- Spectral radius typically ~1.0 (edge of chaos)
- The GNN learns a full-rank W — it must discover the low-rank structure from the data alone
