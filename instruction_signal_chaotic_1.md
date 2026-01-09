# Simulation-GNN Training Landscape Study

## Goal

Map the **simulation-GNN training landscape**: understand which neural activity simulation configurations allow successful GNN training (connectivity_R2 > 0.9) and which are fundamentally harder.

## Iteration Loop Structure

Each block = `n_iter_block` iterations exploring one simulation configuration.
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

- `spectral_radius`: eigenvalue analysis of connectivity
- `svd_rank`: SVD rank at 99% variance (activity complexity)
- `test_R2`: R² between ground truth and rollout prediction
- `test_pearson`: Pearson correlation between ground truth and rollout prediction
- `connectivity_R2`: R² of learned vs true connectivity weights
- `cluster_accuracy`: clustering accuracy (neuron type classification)
- `final_loss`: final training loss

**Classification:**

- **Converged**: connectivity_R2 > 0.9
- **Partial**: connectivity_R2 0.1-0.9
- **Failed**: connectivity_R2 < 0.1

**UCB scores from `ucb_scores.txt`:**

- Provides computed UCB scores for all exploration nodes including current iteration
- At block boundaries, the UCB file will be empty (erased). When empty, use `parent=root`

Example:

```
Node 2: UCB=2.175, parent=1, visits=1, R2=0.997
Node 1: UCB=2.110, parent=root, visits=2, R2=0.934
```

### Step 3: Write Outputs

Append to Full Log (`{config}_analysis.md`) and **Current Block** sections of `{config}_memory.md`:

- In memory.md: Insert iteration log in "Iterations This Block" section (BEFORE "Emerging Observations")
- Update "Emerging Observations" at the END of the file

**Log Form:**

```
## Iter N: [converged/partial/failed]
Node: id=N, parent=P
Mode/Strategy: [success-exploit/failure-probe]/[exploit/explore/boundary]
Config: lr_W=X, lr=Y, lr_emb=Z, coeff_W_L1=W, batch_size=B, low_rank_factorization=[T/F], low_rank=R, n_frames=NF
Metrics: test_R2=A, test_pearson=B, connectivity_R2=C, cluster_accuracy=D, final_loss=E
Activity: [brief description]
Mutation: [param]: [old] -> [new]
Parent rule: [one line]
Observation: [one line]
Next: parent=P
```

### Step 4: Parent Selection Rule in UCB tree

Step A: Select parent node

- Read `ucb_scores.txt`
- If empty → `parent=root`
- Otherwise → select node with **highest UCB**

Step B: Choose strategy

| Condition                            | Strategy             | Action                                       |
| ------------------------------------ | -------------------- | -------------------------------------------- |
| Default                              | **exploit**          | Highest UCB node, try mutation               |
| 3+ consecutive R² ≥ 0.9              | **failure-probe**    | Extreme parameter to find boundary           |
| n_iter_block/4 consecutive successes | **explore**          | Select outside recent chain                  |
| Good config found                    | **robustness-test**  | Re-run same config                           |
| 2+ distant nodes with R² > 0.9       | **recombine**        | Merge params from both nodes                 |
| 100% convergence, branching<10%      | **forced-branch**    | Select node in bottom 50% of tree            |
| 4+ consecutive same-param mutations  | **switch-dimension** | Mutate different parameter than recent chain |
| 3+ partial results probing boundary  | **boundary-skip**    | Accept boundary as found, explore elsewhere  |

**Recombination details:**

Trigger: exists Node A and Node B where:

- Both R² > 0.9
- Not parent-child (distance ≥ 2 in tree)
- Different parameter strengths

Action:

- parent = higher R² node
- Mutation = adopt best param from other node

Example:

```
Node 12: lr_W=1E-2, lr=1E-4, R²=0.94  (good lr_W)
Node 38: lr_W=5E-3, lr=2E-3, R²=0.97  (good lr)

Recombine → lr_W=1E-2, lr=2E-3
```

### Step 5: Edit Config File

Edit config file for next iteration of the exploration.
(The config path is provided in the prompt as "Current config")

**CRITICAL: Config Parameter Constraints**

**DO NOT add new parameters to the `claude:` section.** Only these fields are allowed:

- `n_epochs`: int (training epochs per iteration)
- `data_augmentation_loop`: int (data augmentation count)
- `n_iter_block`: int (iterations per block)
- `ucb_c`: float value (0.5-3.0)

Any other parameters belong in the `training:` or `simulation:` sections, NOT in `claude:`.

Adding invalid parameters to `claude:` will cause a validation error and crash the experiment.

**Training Parameters (change within block):**

```yaml
training:
  learning_rate_W_start: 2.0E-3 # range: 1E-4 to 1E-2
  learning_rate_start: 1.0E-4 # range: 1E-5 to 1E-3
  learning_rate_embedding_start: 2.5E-4 # used when n_neuron_types>1
  coeff_W_L1: 1.0E-5 # range: 1E-6 to 1E-3
  batch_size: 8 # values: 8, 16, 32
  low_rank_factorization: False # or True
  low_rank: 20 # range: 5-100
  coeff_edge_diff: 100 # enforces positive monotonicity
```

**Simulation Parameters (change at block boundaries only):**

```yaml
simulation:
  n_frames: 10000
  connectivity_type: "chaotic" # or "low_rank"
  Dale_law: False # or True
  Dale_law_factor: 0.5
  connectivity_rank: 20 # if low_rank
  n_neurons: 100 # can be changed to 1000 and ONLY 1000, if Iter > 512
  n_neuron_types: 1 #  # can be changed to 4 and ONLY 4,  if Iter > 1024
```

**Claude Exploration Parameters:**

These parameters control the UCB exploration strategy. Can be adjusted between blocks to adapt exploration behavior.

```yaml
claude:
  ucb_c: 1.414 # UCB exploration constant (0.5-3.0), adjust between blocks
```

**UCB exploration constant (ucb_c):**

- `ucb_c` controls exploration vs exploitation: UCB(k) = R²_k + c × sqrt(ln(N) / n_k)
- Higher c (>1.5) → more exploration of under-visited branches
- Lower c (<1.0) → more exploitation of high-performing nodes
- Default: 1.414 (√2, standard UCB1)
- Adjust between blocks based on search behavior:
  - If stuck in local optimum (all R² similar, no improvement) → INCREASE ucb_c to 2.0
  - If too much random exploration (jumping between distant nodes) → DECREASE ucb_c to 1.0
  - Typical range: 0.5 to 3.0

---

## Block Workflow (Steps 1-3, every end of block)

Triggered when `iter_in_block == n_iter_block`

### STEP 1: COMPULSORY — Edit Instructions (this file)

You \***\*MUST\*\*** use the Edit tool to add/modify parent selection rules in this file.
Do NOT just write recommendations in the analysis log — actually edit the file.

After editing, state in analysis log: `"INSTRUCTIONS EDITED: added rule [X]"` or `"INSTRUCTIONS EDITED: modified [Y]"`

**Evaluate and modify rules based on:**

**Branching rate:**

- Branching rate < 20% → ADD exploration rule
- Branching rate 20-80% → No change needed
- **Branch rate** = parent ≠ (current_iter - 1), calculate for **entire block**

Example:

```
Block 1 (16 iterations):
Sequential: Iters 1-14 (all parent = node-1)
Branches: Iter 15 (parent=2)
Branches: 1 out of 15 → Branching rate = 7%
```

**Improvement rate:**

- If <30% improving → INCREASE exploitation (raise R² threshold)
- If >80% improving → INCREASE exploration (probe boundaries)

**Stuck detection:**

- Same R² plateau (±0.05) for 3+ iters? → ADD forced branching rule

**Dimension diversity:**

- Count consecutive iterations mutating **same parameter**
- If > 4 consecutive same-param → ADD switch-dimension rule

Example:

```
Iter 2-14: all mutated lr_W → 13 consecutive same-dimension → ADD rule
```

### STEP 2: Choose Next Simulation block

- Check Regime Comparison Table → choose untested combination
- **Do not replicate** previous block unless motivated (testing knowledge transfer)

### STEP 3: Update Working Memory

Update `{config}_memory.md`:

- Update Knowledge Base with confirmed principles
- Add row to Regime Comparison Table
- Replace Previous Block Summary with **short summary** (2-3 lines, NOT individual iterations)
- Clear "Iterations This Block" section
- Write hypothesis for next block

---

## Working Memory Structure

```markdown
# Working Memory

## Knowledge Base (accumulated across all blocks)

### Regime Comparison Table

| Block | Regime                  | E/I | n_frames | n_neurons | n_types | eff_rank | Best R² | Optimal lr_W | Optimal L1 | Key finding |
| ----- | ----------------------- | --- | -------- | --------- | ------- | -------- | ------- | ------------ | ---------- | ----------- |
| 1     | chaotic, Dale_law=False | -   | 10000    | 100       | 1       | 31-35    | 1.000   | 8E-3         | 1E-5       | ...         |

### Established Principles

[Confirmed patterns that apply across regimes]

### Open Questions

[Patterns needing more testing, contradictions]

---

## Previous Block Summary (Block N-1)

[Short summary only - NOT individual iterations. Example:
"Block 1 (chaotic, Dale_law=False): Best R²=1.000 at lr_W=8E-3.
Key finding: lr_W optimal range 4E-3 to 8E-3."]

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
**CRITICAL: This section must ALWAYS be at the END of memory file. When adding new iterations, insert them BEFORE this section.**
```

---

## Knowledge Base Guidelines

### What to Add to Established Principles

Examples:

- ✓ "Constrained connectivity needs lower lr_W" (causal, generalizable)
- ✓ "L1 > 1e-04 fails for low_rank" (boundary condition)
- ✓ "Effective rank < 15 requires factorization=True" (theoretical link)
- ✗ "lr_W=0.01 worked in Block 4" (too specific)
- ✗ "Block 3 converged" (not a principle)

### Scientific Method (CRITICAL)

**Repeatability:**

- Each iteration is a single training run with stochastic initialization
- Results may vary between runs with identical config
- A "failed" run may succeed on retry; a "converged" run may fail

**Evidence hierarchy:**

| Level            | Criterion                              | Action                 |
| ---------------- | -------------------------------------- | ---------------------- |
| **Established**  | Consistent across 3+ iterations/blocks | Add to Principles      |
| **Tentative**    | Observed 1-2 times                     | Add to Open Questions  |
| **Contradicted** | Conflicting evidence                   | Note in Open Questions |

### What to Add to Open Questions

Examples:

- Patterns needing more testing
- Contradictions between blocks
- Theoretical predictions not yet verified

---

## Theoretical Background

### GNN Architecture (Signal_Propagation)

The model learns neural dynamics du/dt using a graph neural network:

```
du/dt = lin_phi(u, a) + W @ lin_edge(u, a)
```

**Components:**

- `lin_edge` (MLP): message function on edges, transforms source neuron activity
- `lin_phi` (MLP): node update function, computes local dynamics
- `W`: learnable connectivity matrix (n_neurons × n_neurons)
- `a`: learnable node embeddings (n_neurons × embedding_dim)

**Forward pass:**

1. For each edge (j→i): message = W[i,j] × lin_edge(u_j, a_j)
2. Aggregate messages: msg_i = Σ_j W[i,j] × lin_edge(u_j, a_j)
3. Update: du/dt_i = lin_phi(u_i, a_i) + msg_i

**Low-rank factorization:**

When `low_rank_factorization=True`: W = W_L @ W_R where W_L ∈ ℝ^(n×r), W_R ∈ ℝ^(r×n)

**Node embeddings for heterogeneity:**

The embedding vector `a_i` allows each neuron to have different dynamics parameters:

- When `n_neuron_types > 1`: embeddings are learnable (requires `lr_emb`)
- When `n_neuron_types = 1`: embeddings are fixed (all neurons identical)
- Embeddings are concatenated with activity: lin_phi(u_i, a_i) and lin_edge(u_j, a_j)
- This allows the MLPs to learn neuron-type-specific transfer functions

### Training Loss and Regularization

**Prediction loss:**

```
L_pred = ||du/dt_pred - du/dt_true||₂
```

**Key regularization terms:**

- `coeff_W_L1`: L1 on W (sparsity). Range: 1E-6 to 1E-4
- `coeff_edge_diff`: enforces monotonicity of lin_edge output (positive: higher u → higher output)
  - Computed as: relu(msg(u) - msg(u+δ))·coeff for sampled u values
  - Stabilizes message function, prevents oscillating gradients

**Learning rates:**

- `learning_rate_W_start` (lr_W): learning rate for connectivity W
- `learning_rate_start` (lr): learning rate for lin_edge and lin_phi
- `learning_rate_embedding_start` (lr_emb): learning rate for node embeddings a

**Total loss:**

```
L = L_pred + coeff_W_L1·||W||₁ + coeff_edge_diff·L_edge_diff + ...
```

### Spectral Radius

- ρ(W) < 1: activity decays → harder to constrain W
- ρ(W) ≈ 1: edge of chaos → rich dynamics → good recovery
- ρ(W) > 1: unstable

### Effective Rank

- High (30+): full W recoverable
- Low (<15): only subspace identifiable → need factorization
- **Block 6 finding**: effective_rank is the primary predictor of achievable R²
  - effective_rank=20 → R²≈0.998 achievable
  - effective_rank=10 → R²≈0.92 ceiling (regardless of training params)

### Low-rank Connectivity

- W = W_L @ W_R constrains solution space
- Without factorization: spurious full-rank solutions

### Learning Rates

- lr_W:lr ratio matters (typically 20:1 to 50:1)
- Too fast φ learning → noisy W gradients
