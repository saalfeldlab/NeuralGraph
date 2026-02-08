# Sparse GNN Training: N-Scaling Study (n=200 → 1000)

**Reference**: See `neural-gnn/paper/main.pdf` and `config/signal/signal_fig_supp_8_1.yaml` for context.

## Goal

Understand how sparse connectivity recovery scales from n=200 to n=1000 neurons. The reference config (`signal_fig_supp_8_1.yaml`) achieves **R²=0.817** at n=1000 with 50% sparsity — we need to fill the gap and find where recovery becomes possible.

**Key finding**: Previous LLM exploration at n=100 and n=200 plateaued at conn_R2≈0.45-0.49 because it never tested the **reference recipe** (lr_W=1E-4, two-phase training).

## CRITICAL: Reference Recipe (MUST TEST FIRST)

The reference config `signal_fig_supp_8_1.yaml` achieves R²=0.817 at n=1000/100k frames with these parameters:

```yaml
learning_rate_W_start: 1.0E-4   # 30x LOWER than previous LLM exploration!
learning_rate_start: 1.0E-4
learning_rate_embedding_start: 1.0E-4

# Two-phase training (CRITICAL for sparse)
n_epochs_init: 2       # Phase 1: no L1, let W converge
first_coeff_L1: 0      # Phase 1: L1 = 0
coeff_W_L1: 1.0E-5     # Phase 2: apply L1

# Regularization
coeff_edge_diff: 100
coeff_lin_phi_zero: 1.0   # Penalize phi output at zero (NEVER TESTED BEFORE)
```

**Two-phase training is essential**: Phase 1 (n_epochs_init=2) lets W converge without L1 pressure. Phase 2 applies L1 to refine sparsity. **n_epochs is LOCKED to 1** — explore other parameters to improve recovery.

## User instructions to follow

- Explore n_neurons = [200, 400, 600, 800, 1000] systematically
- Scale n_frames proportionally: n_frames = 100 × n_neurons (minimum)
- **ALWAYS start each n with the reference recipe** — do NOT use lr_W=3E-3
- Adjust computation to 2 hours per iteration

## Sparse Regime Characteristics

The true connectivity matrix W is a random Gaussian matrix (W_ij ~ N(0, 1/n)) with 50% of entries zeroed out:

- 100 neurons → 10,000 possible connections → ~5,000 non-zero entries
- W is full-rank (not low-rank) but sparse: half of all entries are exactly zero
- Diagonal is zero (no self-connections)
- Spectral radius ~0.7 (lower than dense due to 50% masking)
- Effective rank of activity data expected to be higher than low-rank regimes (~30-50 for 99% variance)

### Key Differences from Low-Rank Regime

| Property          | Low-Rank (rank=20)                                | Sparse (filling=50%)                                         |
| ----------------- | ------------------------------------------------- | ------------------------------------------------------------ |
| Non-zero entries  | ~all (dense, but constrained to rank-20 subspace) | ~50% (random sparsity pattern)                               |
| True structure    | Low-rank: W = U @ V                               | Sparse: random with binary mask                              |
| Effective rank    | ~12 (low)                                         | ~30-50 (higher)                                              |
| L1 role           | Indirect (L1 doesn't match true structure)        | Direct match (L1 promotes sparsity = true structure)         |
| Degeneracy risk   | High (few modes → many equivalent W)              | Lower (more modes → fewer equivalent W)                      |
| Primary challenge | Recovering rank-20 structure from limited modes   | Recovering correct sparsity pattern (which entries are zero) |

### Expected Challenges

- **Sparsity recovery**: The GNN learns a dense W. L1 must push the correct 50% of entries to zero
- **Threshold sensitivity**: L1 too low → learned W stays dense; L1 too high → real connections get suppressed
- **Scale interaction**: connectivity weights are O(1/sqrt(n)), so L1 must be calibrated to this scale
- **Edge-diff interaction**: coeff_edge_diff constrains MLP compensation, but with higher eff_rank the MLP has less room to compensate anyway
- **False positive/negative tradeoff**: recovering both the zeros AND the non-zero values correctly

## Prior Knowledge (CRITICAL)

**LESSON FROM PREVIOUS EXPLORATION**: The LLM explored n=100 and n=200 for 100+ iterations with lr_W=3E-3 and plateaued at conn_R2≈0.45-0.49. The reference config uses **lr_W=1E-4** (30x lower) and achieves R²=0.817.

**Reference recipe (from signal_fig_supp_8_1.yaml):**
- `lr_W=1E-4` — NOT 3E-3! Much lower learning rate for W
- `lr=1E-4` — same as lr_W (1:1 ratio, not 30:1)
- Two-phase training: `n_epochs_init=2`, `first_coeff_L1=0`, then `coeff_W_L1=1E-5`
- `coeff_lin_phi_zero=1.0` — penalizes phi output at zero (NEW, never tested)
- `coeff_edge_diff=100` — lower than previous exploration (not 10000)
- `batch_size=8` is safe

**N-scaling hypothesis:**
| n_neurons | n_frames | Expected outcome |
| --------- | -------- | ---------------- |
| 200 | 20,000 | Test reference recipe — should improve over 0.45 plateau |
| 400 | 40,000 | Interpolation point |
| 600 | 60,000 | Interpolation point |
| 800 | 80,000 | Interpolation point |
| 1000 | 100,000 | Should reach R²≈0.817 (reference) |

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

| test_pearson | connectivity_R2 | Degeneracy gap | Diagnosis                         |
| :----------: | :-------------: | :------------: | --------------------------------- |
|    > 0.95    |      > 0.9      |     < 0.1      | **Healthy** — correct W           |
|    > 0.95    |     0.3–0.9     |    0.1–0.7     | **Degenerate** — MLP compensation |
|    > 0.95    |      < 0.3      |     > 0.7      | **Severely degenerate**           |
|    < 0.5     |      < 0.5      |       ~0       | **Failed** — not degeneracy       |

**When degeneracy gap > 0.3, DO NOT trust dynamics metrics as evidence of learning quality.**

**Log degeneracy in the iteration entry when detected:**

```
Degeneracy: gap=0.53 (test_pearson=0.999, conn_R2=0.466) — MLP compensation suspected
```

**Sparsity Quality (SPECIFIC TO SPARSE REGIME — check every iteration):**

When connectivity_R2 is partial (0.3-0.9), examine whether the issue is:

- Wrong sparsity pattern (non-zero entries in wrong locations)
- Right sparsity pattern but wrong magnitudes
- Both

If available in logs, note the fraction of correctly identified zeros vs non-zeros.

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
Config: seed=S, lr_W=X, lr=Y, lr_emb=Z, coeff_W_L1=W, coeff_edge_diff=D, n_epochs_init=I, first_coeff_L1=F, batch_size=B, recurrent=[T/F], time_step=T
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
| --------- | -------- | ------ |
| Default | **exploit** | Highest UCB node, conservative mutation |
| 3+ consecutive R² ≥ 0.9 | **failure-probe** | Extreme parameter to find boundary |
| n_iter_block/4 consecutive successes | **explore** | Select outside recent chain |
| reference recipe converged at current n | **n-scale-up** | Increase n_neurons to next level [200→400→600→800→1000], scale n_frames proportionally |
| Same R² plateau (±0.05) for 3+ iters | **forced-branch** | Select 2nd-highest UCB, switch param dimension |

### Step 5: Edit Config File

Edit config file for next iteration.

**CRITICAL: Config Parameter Constraints**

**DO NOT add new parameters to the `claude:` section.** Only these fields are allowed:

- `n_epochs`: int — **LOCKED to 1. DO NOT change n_epochs above 1 under any circumstances.**
- `data_augmentation_loop`: int
- `n_iter_block`: int
- `ucb_c`: float (0.5-3.0)

**CRITICAL: n_epochs = 1 is LOCKED.** Training more epochs is too slow. Do NOT increase n_epochs to 2, 3, 4, or higher. Explore other parameters instead.

**DO NOT change `simulation:` parameters except `seed`.** The simulation regime is fixed for this exploration.

**Simulation Parameters (only seed is mutable):**

```yaml
training:
  seed:
    137 # changing seed generates a DIFFERENT connectivity matrix W
    # use different seeds to test robustness across W samples
```

Changing `seed` produces a new random sparse connectivity matrix. This lets you track whether a training configuration works for one specific W realization or generalizes across multiple W samples. Log the seed in Config line and note when a mutation is a seed change.

**Training Parameters (the exploration space):**

**START WITH REFERENCE RECIPE**, then mutate ONE parameter at a time.

```yaml
training:
  # REFERENCE RECIPE (start here!)
  learning_rate_W_start: 1.0E-4  # CRITICAL: 30x lower than previous exploration
  learning_rate_start: 1.0E-4   # same as lr_W (1:1 ratio)
  learning_rate_embedding_start: 1.0E-4

  # Two-phase training (CRITICAL for sparse)
  n_epochs_init: 2       # Phase 1: epochs without L1
  first_coeff_L1: 0      # Phase 1: L1 = 0 (let W converge)
  coeff_W_L1: 1.0E-5     # Phase 2: apply L1 for sparsity

  # Regularization
  coeff_edge_diff: 100   # lower than previous (not 10000)
  coeff_lin_phi_zero: 1.0  # NEW: penalize phi output at zero

  batch_size: 8
  training_single_type: True
```

**Simulation Parameters (scale with n_neurons):**

```yaml
simulation:
  n_neurons: 200        # start at 200, scale up to 1000
  n_frames: 20000       # n_frames = 100 × n_neurons
  connectivity_filling_factor: 0.5
```

**Claude Exploration Parameters:**

```yaml
claude:
  ucb_c: 1.414 # UCB exploration constant (0.5-3.0)
```

### Step 5.2: Modify GNN Code (PREFERRED when config sweeps plateau)

**SIMPLE RULE: NEVER MODIFY CODE IF 'code' NOT IN TASK**

Config-only sweeps have shown conn_R2≈0.489 regardless of L1 (1E-6 to 1E-3). The ceiling is architectural, not parametric. **Code changes to GNN parameters are the primary lever for breaking through.**

**When to modify code:**

- When config-level parameters produce identical results across sweeps (R² plateau)
- When degeneracy gap > 0.3 persists despite regularization tuning
- When you have a specific architectural hypothesis to test
- NEVER modify code in first 4 iterations of a block (establish baseline first)

**Files you can modify:**

| File                                           | Permission                                   |
| ---------------------------------------------- | -------------------------------------------- |
| `src/NeuralGraph/models/graph_trainer.py`      | **ONLY modify `data_train_signal` function** |
| `src/NeuralGraph/models/Signal_Propagation.py` | Can modify if necessary                      |
| `src/NeuralGraph/models/MLP.py`                | Can modify if necessary                      |
| `src/NeuralGraph/utils.py`                     | Can modify if necessary                      |

**Key model attributes (read-only reference):**

- `model.W` - Connectivity matrix `(n_neurons, n_neurons)`, init: `torch.randn` (std=1.0)
- `model.a` - Node embeddings `(n_neurons, embedding_dim)`, init: `torch.ones`
- `model.lin_edge` - Edge message MLP (hidden layers init: std=0.1)
- `model.lin_phi` - Node update MLP (hidden layers init: std=0.1)

**Priority code changes for sparse regime (ordered by expected impact):**

1. **W initialization scale** — True W has entries ~ N(0, 1/√n). Current init is `torch.randn` (std=1.0), 10× too large for n=100. Try `torch.randn(...) * (1.0 / math.sqrt(n_neurons))` in `Signal_Propagation.py`
2. **Gradient clipping on W** — Large gradients on 10,000 W entries can destabilize training. Add `torch.nn.utils.clip_grad_norm_(model.W, max_norm=1.0)` after `loss.backward()` in `graph_trainer.py`
3. **LR scheduler** — Cosine annealing or step decay for `lr_W` to allow large early updates then fine-tuning. Add scheduler in `data_train_signal`
4. **Proximal L1** — Replace gradient-based L1 with proximal soft-thresholding on W after each step: `model.W.data = torch.sign(W) * torch.clamp(W.abs() - threshold, min=0)`. More effective at producing exact zeros
5. **MLP capacity reduction** — Smaller lin_edge/lin_phi (hidden_dim: 64→32, n_layers: 3→2) forces W to carry more signal, reducing MLP compensation
6. **W diagonal constraint** — Enforce `model.W.data.fill_diagonal_(0)` after each optimizer step (prevent self-loops from absorbing signal)
7. **Loss function** — Add explicit sparsity metrics: penalize entries below a threshold that aren't exactly zero, or add group-lasso-style regularization
8. **Different optimizer for W** — Try SGD with momentum for W (while keeping Adam for MLPs) for sharper L1-induced zeros

**How code reloading works:**

- Training runs in a subprocess for each iteration after code is modified, reloading all modules
- Code changes are immediately effective in the next iteration
- Syntax errors cause iteration failure with error message
- Modified files are automatically committed to git with descriptive messages

**Safety rules (CRITICAL):**

1. **Make minimal changes** — edit only what's necessary
2. **Test in isolation first** — don't combine code + config changes in the same iteration
3. **Document thoroughly** — explain WHY in mutation log
4. **One change at a time** — never modify multiple functions simultaneously
5. **Preserve interfaces** — don't change function signatures

**Logging Code Modifications:**

```
## Iter N: [converged/partial/failed]
Node: id=N, parent=P
Mode/Strategy: code-modification
Config: [unchanged from parent, or specify if also changed]
CODE MODIFICATION:
  File: src/NeuralGraph/models/graph_trainer.py
  Function: data_train_signal
  Change: [what was changed]
  Hypothesis: [why this should help sparse recovery]
Metrics: test_R2=A, test_pearson=B, connectivity_R2=C, cluster_accuracy=D, final_loss=E, kino_R2=F, kino_SSIM=G, kino_WD=H
Mutation: [code] data_train_signal: [short description]
Parent rule: [one line]
Observation: [compare to parent — did code change help?]
Next: parent=P
```

**NEVER:**

- Modify GNN_LLM.py or GNN_LLM_parallel.py (breaks the experiment loop)
- Change function signatures (breaks compatibility)
- Add dependencies requiring new pip packages
- Make multiple simultaneous code changes (can't isolate causality)
- Modify code just to "try something" without a specific hypothesis

**ALWAYS:**

- Explain the hypothesis motivating the code change
- Compare directly to parent iteration (same config, code-only diff)
- Document exactly what changed (file, function, what was added/removed)
- Consider reverting a code change if it doesn't help (git checkout the file)

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

- Block 1: lr_W and coeff_W_L1 sweep (central parameters for sparse recovery)
- Block 2: L1 calibration — find optimal sparsity pressure
- Block 3: coeff_edge_diff / two-phase training interaction
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

| Blk | lr_W | lr   | L1   | edge_diff | n_ep_init | first_L1 | batch | conn_R2 | test_R2 | Finding  |
| --- | ---- | ---- | ---- | --------- | --------- | -------- | ----- | ------- | ------- | -------- |
| 1   | 3E-3 | 1E-4 | 1E-5 | 10000     | 2         | 0        | 8     | ?       | ?       | baseline |

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

- `coeff_W_L1`: L1 on W (sparsity) — **primary lever for sparse regime** (directly promotes correct zero pattern)
- `coeff_edge_diff`: enforces monotonicity of lin_edge (prevents MLP compensation)
  - Higher values (10000+) constrain the MLP more → W must carry the signal

### Sparse Regime Specifics

- True W is chaotic Gaussian (W_ij ~ N(0, 1/√n)) with 50% entries masked to zero
- ~5000 non-zero entries out of 10000 possible (100×100 minus diagonal)
- Spectral radius typically ~0.7 (reduced by sparsity)
- Effective rank of activity data expected ~30-50 (much higher than low-rank regime)
- The GNN learns a dense W — L1 regularization must discover which 50% of entries should be zero
- Unlike low-rank, the non-zero entries have no special structure (random Gaussian)

### Recurrent Training

When `recurrent_training=True` and `time_step=T`, the model is trained to predict T steps ahead using its own predictions (autoregressive rollout):

1. Sample frame k (aligned to `time_step` boundaries)
2. Target = actual state at frame `k + time_step` (not derivative)
3. First step: `pred_x = x + delta_t * model(x) + noise`
4. Steps 2..T: feed `pred_x` back into model, accumulate Euler steps
5. Loss = `||pred_x - y|| / (delta_t * time_step)` (backprop through all T steps)

**Key parameters:**

| Parameter                        | Description                                    | Range            |
| -------------------------------- | ---------------------------------------------- | ---------------- |
| `recurrent_training`             | Enable multi-step rollout                      | True/False       |
| `time_step`                      | Rollout depth (1 = single-step, no recurrence) | 1, 4, 16, 32, 64 |
| `noise_recurrent_level`          | Noise per rollout step (regularization)        | 0 to 0.1         |
| `recurrent_training_start_epoch` | Epoch to begin recurrent training              | 0+               |

**Guidance:**

- Start with `recurrent_training=False` (default) to establish baseline
- Enable at block boundaries: set `recurrent_training=True` + `time_step=4` as first test
- `noise_recurrent_level=0.01-0.05` helps prevent rollout instability
- Higher `time_step` costs proportionally more compute — reduce `data_augmentation_loop` to compensate
- `recurrent_training_start_epoch > 0` allows warmup with single-step before switching to recurrent
