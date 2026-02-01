# Simulation-GNN Training Landscape Study

**Reference**: See `neural-gnn/paper/main.pdf` for current context understanding.

## Goal

Map the **simulation-GNN training landscape**: understand which neural activity simulation configurations allow successful GNN training (connectivity_R2 > 0.9) and which are fundamentally harder.
In particular is it possible to use GNN to recover connectivity from low-rank data, from low-gain data, from sparse data, from noisy dynamics ?
It is taken for granted that increasing training size (number frames) improves recovery of the neural dynamics parameter. Also increasing the overall complexity of the training data, e.g. its effictive rank, improves recovery. Finally we observe that injecting noise into the simulation (not measurement noise) helps recovery too. This can be however reconsidered. And we need to find other way to make GNN as an efficient tool for neuron recovery.

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
- Understanding of simulation-GNN training landscape
- Current block progress

### Step 2: Analyze Current Results

**Metrics from `analysis.log`:**

- `effective rank (99% var)`: **CRITICAL** - SVD rank at 99% cumulative variance. Extract this value and log it as `eff_rank=N` in the activity field.
- `test_R2`: R² between ground truth and rollout prediction
- `test_pearson`: Pearson correlation between ground truth and rollout prediction
- `connectivity_R2`: R² of learned vs true connectivity weights
- `final_loss`: final training loss
- `cluster_accuracy`: neuron classification
- `kinograph_R2`: mean per-frame R² between GT and GNN rollout kinographs
- `kinograph_SSIM`: structural similarity between kinographs
- `kinograph_Wasserstein`: time-unaligned population mode Wasserstein (PCA-projected, normalized by GT std — dimensionless; 0 = identical modes, 1 = one-σ shift; captures right modes even if timing differs)

**Example analysis.log format:**

```
spectral radius: 1.029
--- activity ---
  effective rank (90% var): 26
  effective rank (99% var): 84   <-- Extract this value for eff_rank
```

**Classification:**

- **Converged**: connectivity_R2 > 0.9
- **Partial**: connectivity_R2 0.1-0.9
- **Failed**: connectivity_R2 < 0.1

**Visual Analysis of Embedding Plot (ONLY if n_neuron_types > 1):**

Examine the latest embedding plot to assess cluster quality:

- **Location**: `log/signal/{config_name}/tmp_training/embedding/`
- **Files**: `{epoch}_{iteration}.png` — find the one with highest iteration number

**What to look for:**

Well-separated clusters (one color per neuron type):

- Each neuron type should form a distinct, tight cluster
- Colors should not overlap significantly
- This indicates embeddings are learning neuron-type information

**Log the embedding observation** in the iteration log as:

```
Embedding: [description, e.g., "4 well-separated clusters" or "colors mixed, no separation" or "2 clusters merged"]
```

**Dual-Objective Optimization (when n_neuron_types > 1):**

When connectivity_R2 > 0.9 but cluster_accuracy < 0.9:

- Focus on embedding optimization: increase lr_emb, adjust training duration
- The connectivity learning is successful, now optimize embedding learning

| connectivity_R2 | cluster_accuracy | Status             | Action                               |
| --------------- | ---------------- | ------------------ | ------------------------------------ |
| > 0.9           | > 0.9            | **FULL CONVERGED** | Success - both objectives met        |
| > 0.9           | < 0.9            | **W-converged**    | Focus on lr_emb, embedding learning  |
| < 0.9           | > 0.9            | **E-converged**    | Focus on lr_W, connectivity learning |
| < 0.9           | < 0.9            | **Partial/Failed** | Standard optimization                |

**Upper Confidence Bound (UCB) scores from `ucb_scores.txt`:**

- Provides computed UCB scores for all exploration nodes within a block including current iteration
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
Config: lr_W=X, lr=Y, lr_emb=Z, coeff_W_L1=W, batch_size=B, low_rank_factorization=[T/F], low_rank=R, n_frames=NF, recurrent=[T/F], time_step=T
Metrics: test_R2=A, test_pearson=B, connectivity_R2=C, cluster_accuracy=D, final_loss=E, kino_R2=F, kino_SSIM=G, kino_WD=H
Activity: eff_rank=R (from analysis.log "effective rank (99% var)"), spectral_radius=S, [brief description]
Embedding: [visual observation from embedding plot when n_types>1, e.g., "4 well-separated clusters" or "colors mixed"]
Mutation: [param]: [old] -> [new]
Parent rule: [one line]
Observation: [one line]
Next: parent=P
```

**CRITICAL: Always extract `effective rank (99% var)` from `analysis.log` and include it in the Activity field as `eff_rank=R`. This value is essential for understanding training difficulty and must be recorded in every iteration.**

**CRITICAL: The `Mutation:` line is parsed by the UCB tree builder. Always include the exact parameter change with old and new values (e.g., `Mutation: lr_W: 2E-3 -> 5E-3`). This line MUST appear in the analysis.md log entry for every iteration. Without it, the UCB tree cannot track what was changed.**

**CRITICAL: When n_neuron_types > 1, always examine the embedding plot and include the `Embedding:` field in the log. Visual inspection of cluster separation is essential for understanding embedding learning quality.**

### Step 4: Parent Selection Rule in UCB tree

Step A: Select parent node

- Read `ucb_scores.txt`
- If empty → `parent=root`
- Otherwise → select node with **highest UCB** as parent

**CRITICAL**: The `parent=P` in the Node line must be the **node ID** (integer) of the selected parent, NOT "root" (unless UCB file is empty). Example: if you select node 3 as parent, write `Node: id=4, parent=3`.

Step B: Choose strategy

| Condition                                                           | Strategy                 | Action                                                                                       |
| ------------------------------------------------------------------- | ------------------------ | -------------------------------------------------------------------------------------------- |
| Default                                                             | **exploit**              | Highest UCB node, try mutation                                                               |
| 3+ consecutive R² ≥ 0.9                                             | **failure-probe**        | Extreme parameter to find boundary                                                           |
| n_iter_block/4 consecutive successes                                | **explore**              | Select outside recent chain                                                                  |
| 2+ distant nodes with R² > 0.9                                      | **recombine**            | Merge params from both nodes                                                                 |
| 4+ consecutive partial (0.1 < R² < 0.9) with improving trend        | **scale-up**             | Increase data_augmentation_loop (5x) to break plateau                                        |
| 4+ consecutive converged (R²>0.9) with same param dimension         | **forced-branch**        | Select 2nd highest UCB node (not recent chain), switch param dimension                       |
| eff_rank < 8 AND spectral_radius < 0.7 AND 3+ consecutive R² < 0.05 | **fixed-point-collapse** | Data fundamentally unrecoverable - skip remaining iterations, modify sim params at block end |
| n_neurons ≥ 500 AND 4+ consecutive partial AND n_epochs < 3         | **epoch-scale-up**       | n≥500 needs more training capacity - increase n_epochs to 3+                                 |
| new regime block AND block 1 convergence rate > 80%                 | **regime-transfer-test** | first batch of new regime: test if block 1 optimal params transfer; vary the regime-specific param (e.g. factorization for low-rank) |
| low_rank regime AND no factorization tested                         | **factorization-probe**  | always test low_rank_factorization=True vs False early in low-rank blocks                    |
| lr_W increased AND dynamics degraded (test_R2 dropped)             | **lr-co-optimize**       | when lr_W increase hurts test_R2, try increasing lr proportionally (lr_W/lr ratio ~10:1)     |
| low_rank_factorization=True AND lr_W < 5E-3                        | **factorization-lr-guard** | factorization requires lr_W>=5E-3 to optimize W=W_L@W_R; do NOT use factorization with lr_W<5E-3 |
| lr increased AND dynamics degraded at moderate lr_W (<=3E-3)       | **lr-ceiling**           | at moderate lr_W (<=3E-3), lr=1E-4 is optimal; do NOT increase lr beyond 1E-4 at lr_W<=3E-3   |
| lr increased AND dynamics degraded at any lr_W                     | **lr-ceiling-global**    | lr=1E-4 is optimal in chaotic (eff_rank=35); EXCEPTION 1: in low eff_rank regimes (eff_rank~12, e.g. Dale_law or low_rank), lr=2E-4 may work without degradation (iter 36); EXCEPTION 2: in noisy regimes (eff_rank>=84), lr=2E-4 safe (iters 56, 60); test in context |
| batch_size increased AND dynamics degraded                         | **batch-size-trade-off** | batch_size=16 may degrade dynamics ~2% at L1=1E-5 but shows no degradation at L1=1E-6 (iter 24); BUT in Dale regime, batch=16 degrades connectivity ~10% (iter 34); regime-dependent |
| low eff_rank regime AND coeff_W_L1 >= 1E-5 AND dynamics degraded  | **L1-reduction**         | in low eff_rank regimes (eff_rank<20), L1=1E-6 is critical for dynamics in low_rank; in Dale regime L1 effect is marginal (iter 35 vs 31); always try L1=1E-6 early in low-rank blocks |
| new regime block AND previous block found L1=1E-6 beneficial      | **L1-transfer-test**     | test L1=1E-6 in new regime to see if L1 reduction benefit transfers across regimes |
| Dale_law=True AND lr_W >= 5E-3                                    | **Dale-lr_W-cliff**      | Dale_law creates sharp lr_W cliff at 4.5E-3-5E-3; lr_W>=5E-3 fails reproducibly (iters 29, 30); safe range [3.5E-3, 4.5E-3]; do NOT use lr_W>=5E-3 with Dale_law |
| eff_rank <= 12 AND constraints (Dale/low_rank) AND lr_W > 4.5E-3 | **constrained-lr_W-guard** | constrained regimes (Dale, low_rank) narrow the safe lr_W range; always probe cliff before exploiting high lr_W |
| n_neuron_types > 1 AND coeff_W_L1 >= 1E-5 AND n_neurons <= 100   | **heterogeneous-L1-guard** | L1=1E-6 critical for embedding at n=100/4types (L1=1E-5 destroys cluster_acc 0.990->0.440, iter 44); BUT at n=200/4types L1=1E-5 is BETTER (cluster=1.000 vs 0.750, iter 149 vs 151); n-dependent L1 overrides heterogeneous rule at n>=200 |
| n_neuron_types > 1 AND lr_emb=1E-3 AND lr_W < 4E-3               | **lr_emb-lr_W-coupling**   | lr_emb=1E-3 overshoots at lr_W<4E-3 (cluster 0.710 at lr_W=2E-3, iter 42); use lr_emb=5E-4 when lr_W<=3E-3; lr_emb/lr_W ratio ~0.2 is safe |
| n_neuron_types > 1 AND n_neurons >= 200 AND lr_emb > 1E-3       | **heterogeneous-lr_emb-ceiling** | lr_emb=2E-3 overshoots at n=200/4types/lr_W=8E-3: cluster 1.000→0.375, dynamics -18.2% (iter 154); lr_emb=1E-3 is ceiling at n>=200; lr_emb/lr_W ratio must be <=0.125 (stricter than n=100 ratio ~0.2) |
| n_neuron_types > 1 AND lr_W > 5E-3 AND n_neurons <= 100          | **heterogeneous-lr_W-cap-n100** | at n=100/4types: lr_W=6E-3 degrades embedding (1.000->0.490, iter 45); lr_W=5E-3 sweet spot; do NOT exceed 5E-3 |
| n_neuron_types > 1 AND n_neurons >= 200 AND lr_W > 8E-3          | **heterogeneous-lr_W-cap-n200** | at n=200/4types: lr_W=8E-3 is optimal for full dual (conn=0.988, cluster=1.000, iter 149); lr_W=1E-2 DEGRADES (-8.7% conn, -75.5pp cluster, iter 153); lr_W=1.2E-2 partial recovery (conn=0.955, cluster=0.750, iter 155) — NON-MONOTONIC; safe range [6E-3, 8E-3]; do NOT use lr_W=1E-2 |
| n_neuron_types > 1 AND batch_size > 8                             | **heterogeneous-batch-guard** | batch_size=16 severely degrades heterogeneous networks — dynamics -12%, embedding -50%, connectivity -3% (iter 48); always use batch_size=8 for n_types>1 |
| noise_model_level >= 0.5 AND lr_W >= 8E-3                        | **noisy-lr_W-ceiling**   | in noisy regimes (noise>=0.5), lr_W=1E-2 degrades dynamics severely (test_R2=0.707, iter 57); optimal lr_W inversely correlates with noise: noise=0.1→2E-3, noise=0.5→4E-3, noise=1.0→2E-3; keep lr_W<=8E-3 at noise>=0.5 |
| noise_model_level > 0 AND lr increased beyond 1E-4               | **noisy-lr-tolerance**   | at noise-inflated eff_rank>=84, lr=2E-4 does NOT degrade dynamics (iter 56, 60); noise widens lr tolerance; OVERRIDES lr-ceiling-global in noisy regimes |
| n_neurons >= 200 AND lr_W < 5E-3                                 | **n-scaling-lr_W-boundary** | convergence boundary scales ~2x with n_neurons doubling: n=100→~1.5E-3, n=200→~3.5E-3; at n>=200 use lr_W>=5E-3 as starting point (iters 61-72); do NOT use lr_W<4E-3 at n>=200 |
| n_neurons >= 200 AND lr_W > optimal_lr_W                         | **n-scaling-dynamics-cliff** | dynamics cliff depends on n AND n_epochs: at 1ep n=100→8E-3, n=200→5.5E-3, n=300→~1.2E-2; at 2ep n=200 cliff shifts to >1.2E-2 (all converge up to 1.2E-2, iter 131); more epochs widen safe lr_W range; n=200/2ep optimal at lr_W=8E-3 (iter 126); n=300/3ep optimal at lr_W=1E-2 |
| n_neurons >= 200 AND n_epochs >= 2 AND lr_W in [5E-3, 1.2E-2]   | **n200-epoch-recipe** | n=200 at 2ep achieves 100% convergence (12/12, block 11); optimal: lr_W=8E-3, lr=2E-4, L1=1E-5; 3ep further boosts conn 0.993→0.994 and dynamics 0.963→0.985; batch=16 safe (negligible degradation); do NOT use L1=1E-6 at n=200 |
| n_neurons >= 200 AND lr > 2E-4 AND lr_W >= 1E-2                 | **lr-lr_W-interaction** | lr tolerance narrows at high lr_W: at n=300/lr_W=1E-2, lr=3E-4 degrades dynamics -4.9% and conn -2.9% (iter 108); lr=2E-4 is safe; at n=200/lr_W=5E-3, lr=3E-4 was safe; higher lr_W tightens lr ceiling; use lr=2E-4 when lr_W>=1E-2 |
| n_neurons >= 200 AND lr <= 1E-4                                  | **n-scaling-lr-tolerance** | n=200 (eff_rank=43) tolerates lr=2-3E-4 at lr_W=5E-3 (iters 67, 72); n=300 tolerates lr=2E-4 at lr_W=1E-2 but NOT lr=3E-4 (iter 108); lr tolerance depends on lr_W — safe at 2E-4 for all n>=200 |
| n_types=1 AND chaotic AND coeff_W_L1 < 1E-5 AND n_neurons <= 200 | **L1-chaotic-homogeneous-guard** | L1=1E-6 harmful for n_types=1 chaotic at n<=200: degrades dynamics; at n=300 L1=1E-6 BENEFICIAL (+4.1% conn, iter 119); at n=600 L1=1E-6 HARMFUL AGAIN (-3.4% conn, iter 142); L1=1E-6 benefit is NARROW to n=300 only; use L1=1E-5 for n<=200 and n>=600, L1=1E-6 for n=300 |
| n_neurons >= 300 AND n_epochs < 3 AND coeff_W_L1 >= 1E-5        | **n300-epoch-L1-synergy** | n=300 convergence requires BOTH n_epochs>=3 AND L1=1E-6: n_epochs=3+L1=1E-5→0.886 (partial); n_epochs=3+L1=1E-6→0.922 (converged); n_epochs=4+L1=1E-6→0.924 (best); L1=1E-6 contributes +4.1% and n_epochs contributes +3%; both needed for convergence (iters 110, 114, 117, 119) |
| n_neurons >= 300 AND n_epochs < 2                                | **n300-epoch-minimum** | n=300 at 10k frames requires n_epochs>=2: best conn at 1 epoch was 0.805 (iter 103), n_epochs=2 boosted to 0.890 (+10.6%, iter 106); loss dropped 4x; n=300 is training-capacity-limited at 1 epoch; use n_epochs>=2 for n>=300 |
| n_neurons >= 300 AND batch_size > 8                              | **n300-batch-guard** | batch_size=16 degrades conn -7.8% at n=300 (0.893→0.823, iter 115); extends heterogeneous-batch-guard to large n; always use batch_size=8 for n>=300 |
| connectivity_filling_factor < 1 AND n_epochs < 2                | **sparse-epoch-minimum**   | sparse connectivity (filling_factor=0.5) at 10k frames requires n_epochs>=2; at 1 epoch no config converges (max R2=0.420); 2 epochs boosts ~10% (0.466); lr_W=1E-2 optimal (no cliff); eff_rank drops to 21, spectral_radius=0.746 subcritical (iters 73-84) |
| connectivity_filling_factor < 1 AND all partial for full block  | **sparse-scale-up**        | sparse 50% at n=100/10k frames/2 epochs yields max R2=0.466 (0% convergence); reference config uses n=1000/100k frames/10 epochs — sparse regime fundamentally needs more data and training to converge; consider n_frames increase at block boundaries |
| connectivity_filling_factor < 1 AND eff_rank < 25               | **sparse-subcritical-guard** | sparse connectivity reduces eff_rank (35->21 at 50% fill) and makes spectral_radius subcritical (0.746); dynamics test_R2 stuck at ~0.11 regardless of training params; this is a data complexity issue not a training issue |
| connectivity_filling_factor < 1 AND noise_model_level > 0 AND conn_R2 plateau for 4+ iters | **sparse-noise-plateau** | noise inflates eff_rank (21→91 at noise=0.5) but does NOT rescue sparse connectivity; conn plateaus at ~0.489 with COMPLETE parameter insensitivity (lr_W 2E-3 to 1.5E-2, L1, n_epochs, aug_loop all identical); subcritical rho=0.746 unchanged by noise; sparse+noise is data-limited not training-limited (iters 85-96) |
| recurrent_training=True AND spectral_radius < 1 AND noise_model_level > 0 | **recurrent-subcritical-guard** | recurrent training (time_step=4) is CATASTROPHIC in noisy subcritical regimes: conn collapsed 0.489→0.054, loss exploded 200→170k (iter 95); multi-step rollout amplifies instability when dynamics are subcritical + noisy; do NOT use recurrent training when spectral_radius<1 and noise>0 |
| n_neurons >= 600 AND lr <= 1E-4                                  | **n600-lr-floor**          | lr=1E-4 is CATASTROPHIC at n=600: conn=0.000, kino diverged to -1.17E14 (iter 143); n=600 REQUIRES lr>=2E-4; at large n, MLP needs sufficient lr to learn dynamics — lr=1E-4 prevents learning entirely; do NOT use lr<2E-4 at n>=600 |
| n_neurons >= 600 AND n_epochs < 10                               | **n600-epoch-minimum**     | n=600 at 10k frames is severely training-capacity-limited: 4ep→0.540, 6ep→0.554, 8ep→0.580, 10ep→0.626; gains are NOT diminishing (~4-8% per +2ep); use n_epochs>=10 for n>=600; extrapolate ~15-20ep needed for convergence |
| n_neurons >= 600 AND coeff_W_L1 < 1E-5                          | **n600-L1-guard**          | L1=1E-6 worse than L1=1E-5 at n=600 across all epoch counts (4ep: 0.511 vs 0.540; 10ep: 0.605 vs 0.626); L1 beneficial threshold shifts BACK above n=300 — use L1=1E-5 for n>=600; L1=1E-6 only beneficial in narrow n=300 range |
| n_neurons >= 600 AND lr_W > 1E-2                                 | **n600-lr_W-ceiling**      | lr_W=1.5E-2 hurts dynamics at n=600 (test_R2=0.810 vs 0.802 at 4ep, but conn drops vs lr_W=1E-2 at higher epochs); lr_W=2E-2 even worse (0.684); lr_W=1E-2 optimal at n=600; do NOT use lr_W>1E-2 at n>=600 |

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

### Step 5: Edit Config File (default) or Modify Code

#### Step 5.1: Edit Config File (default)

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

**Training Parameters (change within block, ONLY one at a time):**

Mutate ONE parameter at a time for better causal understanding.

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
  data_augmentation_loop: int (data augmentation count) # 40 affects training time - improve results

  # Recurrent training parameters (can tune within block)
  recurrent_training: False # enable multi-step rollout training (True/False)
  time_step: 1 # rollout depth: 1 = single-step (default), 4, 16, 32, 64
  noise_recurrent_level: 0.0 # noise injected per rollout step (range: 0 to 0.1)
  recurrent_training_start_epoch: 0 # epoch to begin recurrent training (0 = from start)
```

**User Priors (read every iteration)**

These priors are provided by the user and should be treated as established knowledge when planning experiments.

**Prior 1: n_frames scales with n_neurons (CRITICAL)**

The training dataset size is the **main key** to connectivity recovery at larger scales. The following scaling is known to work:

| n_neurons | n_frames  | Notes |
|-----------|-----------|-------|
| 100       | 1,000     | Minimum sufficient for n=100 |
| 600       | 6,000     | Minimum sufficient for n=600 |
| 1,000     | 100,000   | Reference config value |

The scaling is approximately **n_frames ~ 10 × n_neurons** for small networks, but grows faster at large n. When increasing n_neurons at block boundaries, **adjust n_frames accordingly** rather than keeping it fixed at 10,000. n_epochs can be adjusted between 1 and 5, but increasing n_frames has a much larger impact on recovery than increasing n_epochs.

**Prior 2: Sparse matrices are hard to recover.** Solution exists for n_neurons=1000 and n_frames=100000, then the following parameters work:

```yaml
training:
  n_epochs: 2

  learning_rate_W_start: 1.0E-4
  learning_rate_start: 1.0E-4
  learning_rate_embedding_start: 1.0E-4

  # Two-phase training (like ParticleGraph)
  n_epochs_init: 2 # epochs in phase 1
  first_coeff_L1: 0 # phase 1: no L1
  coeff_W_L1: 1.0E-5 # phase 2: target L1

  # Regularization coefficients
  coeff_edge_diff: 100 # Monotonicity constraint (phase 1 only, disabled in phase 2)
  coeff_update_diff: 0 # Monotonicity constraint on update function
  coeff_lin_phi_zero: 1.0 # Penalize phi output at zero
```

Resume to n_epochs: 1 if not sparse regime.

**Training Time Guidance**

Target: Keep each iteration under **2 hours** when possible.

**If training is taking too long, consider reducing these parameters (in order of impact):**

1. `data_augmentation_loop`: 40 → 20 → 10 (biggest impact on time)
2. `n_epochs`: 20 → 15 → 10
3. `n_frames`: 50000 → 30000 → 10000 (at block boundaries)
4. `batch_size`: 8 → 16 → 32 (larger = faster)

**Approximate training times:**

- `data_augmentation_loop=40` + `n_epochs=20` + `n_frames=50000` ≈ 4-6 hours
- `data_augmentation_loop=20` + `n_epochs=15` + `n_frames=30000` ≈ 1.5-2 hours
- `data_augmentation_loop=10` + `n_epochs=10` + `n_frames=10000` ≈ 30-45 min

**Simulation Parameters (can be changed ONLY at block boundaries):**

```yaml
simulation:
  n_frames: 10000 # scale with n_neurons: n=100→1000, n=600→6000, n=1000→100000 (see User Priors)
  connectivity_type: "chaotic" # or "low_rank"
  Dale_law: False # or True
  Dale_law_factor: 0.5 # between 0 and 1 to explore different excitatory/inhibitory ratios.
  connectivity_rank: 20 # if low_rank between 10 and 90
  connectivity_filling_factor: 1 # test 0.05 0.1 0.2 0.5 percent of non zero conenctivity weights
  n_neurons: 100
  # BEFORE iteration 128, can be changed to 100, 200, 300
  # can be changed to 600, 1000 ONLY after iteration 128
  n_neuron_types: 1 # can be changed between 1 to 4
  # params: [a, b, g, s, w, h] per neuron type, one row per neuron type
  # g (third column): network gain - scales connectivity influence on dynamics (range: 1-10)
  #   IMPORTANT: g must be changed SIMULTANEOUSLY for ALL neuron types
  # s is the self excitation, can be changed 0, 1, or 2
  # others parameters a, b, w and h are fixed
  params:
  [
    [1.0, 0.0, 7.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 7.0, 1.0, 1.0, 0.0],
    [2.0, 0.0, 7.0, 1.0, 1.0, 0.0],
    [2.0, 0.0, 7.0, 2.0, 1.0, 0.0],
  ]
  # params values can be changed BUT the size 4 x 6 must be maintained
  # g, the gain of the neural dynamics is the most important parameter
  # g is in the third column, should be equal value for every neuron type, hence in every row
  # n_neuron_types defines network heterogeneity (change at block boundaries):
  # - 1 = homogeneous network (all neurons have same dynamics)
  # - 2-4 = heterogeneous network (neurons have different dynamics based on type)
  # - params array must have exactly n_neuron_types rows
  noise_model_level: 0 # can be changed to 0.1 0.5 1 and 2 noise added to the Euler integration of the simulation
```

**graph_model section — refer to PARAMS_DOC**

The `graph_model:` parameters have strict dependencies documented in `Signal_Propagation.PARAMS_DOC` (in `src/NeuralGraph/models/Signal_Propagation.py`). Read the PARAMS_DOC before modifying any graph_model parameter. In particular, `input_size` and `input_size_update` are DERIVED from the model variant and `embedding_dim` — they must be updated together if `embedding_dim` changes.

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
- Adjust between blocks based on search behavior

#### Step 5.2: Modify Code (optional, requires 'code' in task)

SIMPLE RULE: NEVER MODIFY CODE IF 'code' NOT IN TASK

**When to modify code:**

- When config-level parameters are insufficient to solve a problem
- When a failure mode indicates a fundamental limitation
- When you have a specific architectural hypothesis to test
- When 3+ iterations suggest a code-level change would help
- NEVER modify code in first 4 iterations of a block

**Files you can modify (if necessary):**

| File                                           | Permission                                   |
| ---------------------------------------------- | -------------------------------------------- |
| `src/NeuralGraph/models/graph_trainer.py`      | **ONLY modify `data_train_signal` function** |
| `src/NeuralGraph/models/Signal_Propagation.py` | Can modify if necessary                      |
| `src/NeuralGraph/utils.py`                     | Can modify if necessary                      |
| `GNN_PlotFigure.py`                            | Can modify if necessary                      |

**Key model attributes (read-only reference):**

- `model.W` - Connectivity matrix `(n_neurons, n_neurons)`
- `model.a` - Node embeddings `(n_neurons, embedding_dim)`
- `model.lin_edge` - Edge message MLP
- `model.lin_phi` - Node update MLP

**How code reloading works:**

- Training runs in a subprocess for each iteration after code is modified, reloading all modules
- Code changes are immediately effective in the next iteration
- Syntax errors cause iteration failure with error message
- Modified files are automatically committed to git with descriptive messages

**Safety rules (CRITICAL):**

1. **Make minimal changes** - edit only what's necessary
2. **Test in isolation first** - don't combine code + config changes
3. **Document thoroughly** - explain WHY in mutation log
4. **One change at a time** - never modify multiple functions simultaneously
5. **Preserve interfaces** - don't change function signatures

**Allowed Training Loop Changes (data_train_signal only):**

- Change optimizer (Adam → AdamW, SGD, RMSprop)
- Add learning rate scheduler (CosineAnnealingLR, ReduceLROnPlateau)
- Add gradient clipping
- Modify loss function (add regularization terms, use different distance metrics)
- Change data sampling strategy
- Add early stopping logic

**Example: Add learning rate schedule**

```python
# After optimizer creation:
optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)  # ADD THIS

# In training loop (after optimizer.step()):
scheduler.step()  # ADD THIS
```

**Example: Add gradient clipping**

```python
# In training loop, after loss.backward():
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ADD THIS
optimizer.step()
```

**Logging Code Modifications:**

In iteration log, use this format:

```
## Iter N: [converged/partial/failed]
Node: id=N, parent=P
Mode/Strategy: code-modification
Config: [unchanged from parent, or specify if also changed]
CODE MODIFICATION:
  File: src/NeuralGraph/models/graph_trainer.py
  Function: data_train_signal
  Change: Added learning rate scheduler
  Hypothesis: LR decay may help convergence in late training
Metrics: test_R2=A, test_pearson=B, connectivity_R2=C, cluster_accuracy=D, final_loss=E
Mutation: [code] data_train_signal: Added LR scheduler
Parent rule: [one line]
Observation: [compare to parent - did code change help?]
Next: parent=P
```

**Constraints and Prohibitions:**

**NEVER:**

- Modify GNN_LLM.py (breaks the experiment loop)
- Change function signatures (breaks compatibility)
- Add dependencies requiring new pip packages
- Make multiple simultaneous code changes (can't isolate causality)
- Modify code just to "try something" without hypothesis

**ALWAYS:**

- Explain the hypothesis motivating the code change
- Compare directly to parent iteration (same config, code-only diff)
- Document exactly what changed (file, line numbers, what was added/removed)
- Consider config-based solutions first

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

# Working Memory Structure

## Knowledge Base (accumulated across all blocks)

### Regime Comparison Table

**eff_rank MUST be from `analysis.log`: `effective rank (99% var): N`**

| Blk | Regime  | E/I | frames | neurons | types | noise | eff_rank | g   | R²   | lr_W | L1   | Finding  |
| --- | ------- | --- | ------ | ------- | ----- | ----- | -------- | --- | ---- | ---- | ---- | -------- |
| 1   | chaotic | -   | 10k    | 100     | 1     | 0     | 31-35    | 7   | 1.00 | 8E-3 | 1E-5 | baseline |

### Established Principles

[Confirmed patterns that apply across regimes]

### Open Questions

[Patterns needing more testing, contradictions]

### Simulation-GNN training landscape

Understanding of simulation-GNN training landscape

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

## Background

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

### Connectivity W

**Spectral Radius**

- ρ(W) < 1: activity decays → harder to constrain W
- ρ(W) ≈ 1: edge of chaos → rich dynamics → good recovery
- ρ(W) > 1: unstable

**Low-rank Connectivity**

- W = W_L @ W_R constrains solution space
- Without factorization: spurious full-rank solutions

**Dale's Law and E/I Balance**

- Dale_law=True enforces excitatory/inhibitory (E/I) constraint on connectivity W
- Dale_law_factor controls E/I ratio: 0.5 means 50% excitatory, 50% inhibitory neurons

**Network Gain (g)**

The network gain `g` (third column in `params` array) scales how strongly connectivity W influences neural dynamics:

- `g` appears in the dynamics equation: `dx/dt = ... + g × Σ_j W_ij × ψ(x_j)`
- Higher g (7-10): stronger network interactions
- Lower g (1-3): weaker network interactions
- `g` must be set **identically for ALL neuron types** in the `params` array
- Range: 1-10 (default: 7)

### Data Complexity

**Effective Rank (eff_rank, svd_rank)**

The effective rank measures the intrinsic dimensionality of neural activity data. It is computed via SVD decomposition of the activity matrix (n_frames × n_neurons):

```
U, S, Vt = SVD(activity)
cumulative_variance = cumsum(S²) / sum(S²)
eff_rank = min k such that cumulative_variance[k] ≥ 0.99
```

The effective rank at 99% variance is logged in `analysis.log` as:

```
effective rank (99% var): 16
```

### Recurrent Training

When `recurrent_training=True` and `time_step=T`, the model is trained to predict T steps ahead using its own predictions (autoregressive rollout):

1. Sample frame k (aligned to `time_step` boundaries)
2. Target = actual state at frame `k + time_step` (not derivative)
3. First step: `pred_x = x + delta_t * model(x) + noise`
4. Steps 2..T: feed `pred_x` back into model, accumulate Euler steps
5. Loss = `||pred_x - y|| / (delta_t * time_step)` (backprop through all T steps)

**Key parameters:**

| Parameter | Description | Range |
| --- | --- | --- |
| `recurrent_training` | Enable multi-step rollout | True/False |
| `time_step` | Rollout depth (1 = single-step, no recurrence) | 1, 4, 16, 32, 64 |
| `noise_recurrent_level` | Noise per rollout step (regularization) | 0 to 0.1 |
| `recurrent_training_start_epoch` | Epoch to begin recurrent training | 0+ |

**Expected trade-offs:**

| time_step | Connectivity R² | Rollout Stability | Training Cost |
| --- | --- | --- | --- |
| 1 | Best | May overfit short-term | Low |
| 4 | Good | Better generalization | Moderate |
| 16 | Moderate | Good long-term | High |
| 32-64 | Lower | Best (if converges) | Very high |

**Guidance:**
- Start with `recurrent_training=False` (default) to establish baseline
- Enable at block boundaries: set `recurrent_training=True` + `time_step=4` as first test
- `noise_recurrent_level=0.01-0.05` helps prevent rollout instability
- Higher `time_step` costs proportionally more compute — reduce `data_augmentation_loop` to compensate
- `recurrent_training_start_epoch > 0` allows warmup with single-step before switching to recurrent

---

## Reference Configs

When exploring new parameter regimes, these configs contain validated parameters from the paper:

| Config                | Neurons | Types | Frames  | Use Case                         |
| --------------------- | ------- | ----- | ------- | -------------------------------- |
| `signal_fig_supp_11`  | 8000    | 4     | 100k    | **Scale up** - large networks    |
| `signal_fig_supp_12`  | 1000    | 32    | 100k    | **Many types** - heterogeneous   |
| `signal_fig_supp_8`   | 1000    | 4     | 100k    | **5% sparsity**                  |
| `signal_fig_supp_8_1` | 1000    | 4     | 100k    | **50% sparsity**                 |
| `signal_fig_supp_8_2` | 1000    | 4     | 100k    | **20% sparsity**                 |
| `signal_fig_supp_10`  | 1000    | 4     | 100k    | **Noise test** (σ=7.2)           |
| `signal_fig_supp_7_*` | 1000    | 4     | 10k-50k | **Dataset size** ablation        |
| `signal_fig_supp_13`  | 1000    | 4     | 100k    | **Identical neurons** hypothesis |

**Location**: `config/signal/signal_fig_supp_*.yaml`

**How to use**: At then end of a block, a start of a new regime, e.g., testing 5% sparsity, read the reference config for examples:

```
Read config/signal/signal_fig_XXX.yaml
```

Then adapt the relevant parameters (connectivity_filling_factor, learning rates, etc.) to your current config.
