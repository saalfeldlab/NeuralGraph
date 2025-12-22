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
  - `cluster_accuracy`: GMM clustering accuracy on learned embeddings (neuron type classification)
  - `final_loss`: final training loss (lower is better)
- `ucb_scores.txt`: provides pre-computed UCB scores for all nodes including current iteration
  at block boundaries, the UCB file will be empty (erased). When UCB file is empty, use `parent=root`.

Node 2: UCB=2.175, parent=1, visits=1, R2=0.997
Node 1: UCB=2.110, parent=root, visits=2, R2=0.934

### Step 3: Write Outputs

Append to Full Log (`{config}_analysis.md`) and **Current Block** sections of `{config}_memory.md`:

- In memory.md: Insert iteration log in "Iterations This Block" section (BEFORE "Emerging Observations")
- Update "Emerging Observations" at the END of the file

Log Form

```
## Iter N: [converged/partial/failed]
Node: id=N, parent=P
Mode/Strategy: [success-exploit/failure-probe]/[exploit/explore/boundary]
Config: lr_W=X, lr=Y, lr_emb=Z, coeff_W_L1=W, batch_size=B, low_rank_factorization=[T/F], low_rank=R, n_frames=NF
Metrics: test_R2=A, test_pearson=B, connectivity_R2=C, cluster_accuracy=E, final_loss=D
Activity: [brief description]
Mutation: [param]: [old] -> [new]
Parent rule: [one line]
Observation: [one line]
Next: parent=P
```

### Step 4: Edit config file for next iteration according to Parent Selection Rule

(The config path is provided in the prompt as "Current config")

- **Classification**

**Converged**: connectivity_R2 > 0.9
**Partial**: connectivity_R2 0.1-0.9
**Failed**: connectivity_R2 < 0.1

- **Training Parameters (change within block)**

```yaml
training:
  learning_rate_W_start: 2.0E-3 # range: 1E-4 to 1E-2
  learning_rate_start: 1.0E-4 # range: 1E-5 to 1E-3
  learning_rate_embedding_start: 2.5E-4 # used because n_neuron_types>1 in config to handle heterogeneity
  coeff_W_L1: 1.0E-5 # range: 1E-6 to 1E-3
  batch_size: 8 # values: 8, 16, 32
  low_rank_factorization: False or True
  low_rank: 20 # range: 5-100
  coeff_edge_diff: 100 # enforces positive monotonicity of message function
```

- **Simulation Parameters (change at block boundaries only)**

```yaml
simulation:
  n_frames: 10000
  connectivity_type: "chaotic" # or "low_rank"
  Dale_law: False or True
  Dale_law_factor: 0.5
  connectivity_rank: 20 if low_rank
```

- **Parent Selection Rule**

Step A: Select parent node

- Read `ucb_scores.txt`
- If empty → `parent=root`
- Otherwise → select node with **highest UCB**

Step B: Choose strategy

| Condition                            | Strategy            | Action                             |
| ------------------------------------ | ------------------- | ---------------------------------- |
| Default                              | **exploit**         | Highest UCB node, try mutation     |
| 3+ consecutive R² ≥ 0.9              | **failure-probe**   | Extreme parameter to find boundary |
| n_iter_block/4 consecutive successes | **explore**         | Select outside recent chain        |
| Good config found                    | **robustness-test** | Re-run same config                 |

---

## Block Workflow step 1 to 3, every end of block

block iter_in_block==n_iter_block

**STEP 1: COMPULSORY — Edit Protocol (this file)**

You MUST use the Edit tool to add/modify rules in this file.
Do NOT just write recommendations in the analysis log — actually edit the file.

After editing, state in analysis log: "PROTOCOL EDITED: added rule [X]" or "PROTOCOL EDITED: modified [Y]"

Evaluate and modify rules based on:

- **Branching rate**:

  - Branching rate < 20% ADD exploration rule
  - Branching rate 20-80% No change needed

  **Branch rate** = parent ≠ (current_iter - 1), calculate for **entire block**

  ```
  Block 1 (16 iterations):
  Sequential: Iters 1-14 (all parent = node-1)
  Branches: Iter 15 (parent=2)

  Branches: 1 out of 15 → Branching rate = 7%
  ```

- **Improvement rate**: How many iters improved R²?
  - If <30% improving → INCREASE exploitation (raise R² threshold)
  - If >80% improving → INCREASE exploration (probe boundaries)
- **Stuck detection**: Same R² plateau (±0.05) for 3+ iters?

  - If yes → ADD forced branching rule

- **Dimension diversity**: Count consecutive iterations mutating **same parameter**

  - If > 4 consecutive same-param → ADD switch-dimension rule

  Example:

```
  Iter 2-14: all mutated lr_W → 13 consecutive same-dimension → ADD rule
```

**STEP 2 Choose next simulation**

Check Regime Comparison Table → choose untested combination
**Do not replicate** previous block unless motivated (testing knowledge transfer)

**STEP 3: Update Working Memory** (`{config}_memory.md`):

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

| Block | Regime | Best R² | Optimal lr_W | Optimal L1 | Key finding |
| ----- | chaotic Dale_law=False | ------- | ------------ | ---------- | ----------- |
| ----- | low_rank=50 Dale_law=True

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

example:
✓ "Constrained connectivity needs lower lr_W" (causal, generalizable)
✓ "L1 > 1e-04 fails for low_rank" (boundary condition)
✓ "Effective rank < 15 requires factorization=True" (theoretical link)
✗ "lr_W=0.01 worked in Block 4" (too specific)
✗ "Block 3 converged" (not a principle)

### Scientific Method (CRITICAL)

- **Repeatability**

- Each iteration is a single training run with stochastic initialization
- Results may vary between runs with identical config
- A "failed" run may succeed on retry; a "converged" run may fail

- **Evidence hierarchy**
  | Level | Criterion | Action |
  | **Established** | Consistent across 3+ iterations/blocks | Add to Principles |
  | **Tentative** | Observed 1-2 times | Add to Open Questions |
  | **Contradicted** | Conflicting evidence | Note in Open Questions |

### What to Add to Open Questions

example:

Patterns needing more testing, Contradictions between blocks, Theoretical predictions not yet verified

---

## Theoretical Background to be revised upon landscape exploration

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

### Established Principles

previous experiment involved only one neuron type
here there are four neuron_types with different signaling fucntions

- lr_W:lr ratio is the key parameter; optimal ratio varies by regime
  - chaotic unconstrained: 40-100:1 sufficient (easiest)
  - low_rank=20: 320:1 required (factorization also needed)
  - chaotic with Dale_law: 800:1 required (hardest)
- L1 regularization must scale with connectivity complexity
  - chaotic unconstrained: L1=1E-5 optimal
  - low_rank / Dale_law: L1=1E-6 optimal (10x weaker)
  - L1 lower bound ~5E-7; below this causes catastrophic failure
- batch_size: chaotic robust to 8 or 16; low_rank prefers 8
- Dale_law constraint reduces effective_rank from ~30 to ~10, requiring much higher lr_W
- constraints (low_rank or Dale_law) both require weaker L1 regularization
- **n_frames is critical for low_rank regimes** - n_frames=20000 enables rich activity (effective_rank 28-30)
- **overparameterization helps**: model low_rank > ground truth connectivity_rank works better
- **REVISED**: factorization requirement depends on n_frames:
  - n_frames=10000: factorization=True required for low_rank (effective_rank ~12)
  - n_frames=20000: factorization=False works (effective_rank ~28-30)
- **NEW**: with sufficient data (n_frames=20000), even double constraints (low_rank + Dale_law) achieve perfect R²=1.000
