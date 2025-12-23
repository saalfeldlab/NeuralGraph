# Working Memory: signal_chaotic_1_Claude

## Knowledge Base (accumulated across all blocks)

### Regime Comparison Table

| Block | Regime                   | E/I  | n_frames | n_neurons | n_types | eff_rank | Best R² | Optimal lr_W  | Optimal L1 | Key finding                                              |
| ----- | ------------------------ | ---- | -------- | --------- | ------- | -------- | ------- | ------------- | ---------- | -------------------------------------------------------- |
| 1     | chaotic, Dale=False      | -    | 10000    | 100       | 1       | 31-35    | 1.000   | 8E-3 to 30E-3 | 1E-5       | lr_W:lr ratio 40-100:1 works; robust regime              |
| 2     | low_rank=20, Dale=False  | -    | 10000    | 100       | 1       | 6-12     | 0.977   | 16E-3         | 1E-6       | lr_W:lr ratio 320:1 needed; factorization required       |
| 3     | chaotic, Dale=True       | 0.5  | 10000    | 100       | 1       | 10       | 0.940   | 80E-3         | 1E-6       | lr_W:lr ratio 800:1 needed; E/I constraint hardest       |
| 4     | low_rank=50, Dale=False  | -    | 20000    | 100       | 1       | 7-21     | 0.989   | 20E-3         | 1E-6       | n_frames=20000 essential; effective_rank 7→21            |
| 5     | low_rank=20, Dale=True   | 0.5  | 20000    | 100       | 1       | 28-30    | 1.000   | 8-10E-3       | 1E-6       | factorization=False works with n_frames=20000            |
| 6     | low_rank=50, Dale=True   | 0.5  | 20000    | 100       | 1       | 10-20    | 0.998   | 25E-3         | 1E-6       | effective_rank variance (10 vs 20) drives R² variance    |
| 7     | chaotic, Dale=True       | 0.5  | 20000    | 100       | 1       | 28-36    | 0.9999  | 40E-3         | 1E-6       | n_frames=20000 solves Block 3; 100% convergence          |
| 8     | low_rank=10, Dale=False  | -    | 30000    | 100       | 1       | 4-7      | 0.901   | 80E-3         | 1E-6       | 7% converge; eff_rank=4 barrier; stochastic eff_rank 6-7 |

### Established Principles

- lr_W:lr ratio is the key parameter; optimal ratio varies by regime
  - chaotic unconstrained: 40-100:1 sufficient (easiest)
  - low_rank: 160-320:1 required
  - chaotic with Dale_law (n_frames=10000): 800:1 required (hardest)
  - chaotic with Dale_law (n_frames=20000): 400:1 sufficient
- L1 regularization must scale with connectivity complexity
  - chaotic unconstrained: L1=1E-5 optimal
  - low_rank / Dale_law: L1=1E-6 optimal (10x weaker)
  - L1 lower bound ~5E-7; below this causes catastrophic failure
- batch_size: chaotic robust to 8, 16, 32; low_rank prefers 8
  - batch_size=32 with lr_W compensation achieves best results
- **n_frames is critical**: n_frames=20000 enables:
  - rich activity (effective_rank 28-36 vs 10)
  - perfect recovery even with double constraints (low_rank + Dale_law)
  - lower lr_W:lr ratio (400:1 vs 800:1 for chaotic+Dale_law)
- **overparameterization helps**: model low_rank > ground truth connectivity_rank
- factorization requirement depends on n_frames:
  - n_frames=10000: factorization=True required for low_rank
  - n_frames=20000: factorization=False works (effective_rank high enough)
- **effective_rank is the primary predictor of achievable R²**
  - effective_rank≥20 → R²≈0.999+ achievable
  - effective_rank<15 → R²≈0.92 ceiling

### Open Questions

- what is the minimum n_frames needed for perfect recovery?
- is there a universal formula: effective_rank > threshold → factorization not needed?
- can n_neurons=1000 enable higher effective_rank for low_rank=10?
- does Dale_law=True + low_rank=10 behave differently?

---

## Previous Block Summary (Block 8)

Block 8 (low_rank=10, Dale_law=False, n_frames=30000): Best R²=0.901, 1/14 converged (7%).
Key finding: low_rank=10 produces effective_rank=4 typically, which is a fundamental learnability barrier. Stochastically reaches eff_rank=6-7 enabling R²~0.87-0.90. n_frames=30000 did NOT boost effective_rank. Optimal config when lucky: lr_W=80E-3, lr=5E-5, L1=1E-6, batch_size=32.

---

## Current Block (Block 9)

### Block Info

Simulation: connectivity_type=low_rank, connectivity_rank=10, Dale_law=True, Dale_law_factor=0.5, n_frames=30000
Iterations: 128 to 143

### Hypothesis

Testing low_rank=10 with Dale_law=True to see if E/I constraints change effective_rank dynamics. Based on Block 8:
- low_rank=10 typically produces effective_rank=4 (unlearnable)
- Dale_law may increase effective_rank by introducing E/I structure
- Or Dale_law may further constrain dynamics, making it worse
- Using n_frames=30000, lr_W=80E-3, lr=5E-5, L1=1E-6, batch_size=32 (Block 8 optimal)
Starting with factorization=True, low_rank=10 matching simulation.

### Iterations This Block

## Iter 128: partial
Node: id=128, parent=127
Mode/Strategy: exploit (new simulation)
Config: lr_W=80E-3, lr=5E-5, L1=1E-6, batch_size=32, factorization=T, low_rank=10, n_frames=30000
Metrics: test_R2=0.279, test_pearson=-0.184, connectivity_R2=0.572, final_loss=6136
Activity: effective_rank(90%)=3, effective_rank(99%)=13, spectral_radius=0.649
Mutation: simulation change: Dale_law False → True (new block)
Observation: Dale_law=True increased effective_rank(99%) from ~6 to 13; R²=0.572 partial; negative test_pearson indicates dynamics mismatch
Next: parent=128, try lr_W=100E-3

## Iter 129: partial
Node: id=129, parent=root
Mode/Strategy: exploit
Config: lr_W=100E-3, lr=5E-5, L1=1E-6, batch_size=32, factorization=T, low_rank=10, n_frames=30000
Metrics: test_R2=0.263, test_pearson=0.064, connectivity_R2=0.508, final_loss=5551
Activity: effective_rank(90%)=3, effective_rank(99%)=12, spectral_radius=0.649
Mutation: lr_W: 80E-3 → 100E-3
Parent rule: UCB selects only available node (root for new block)
Observation: lr_W=100E-3 (ratio 2000:1) degraded R² from 0.572→0.508; ratio too high; effective_rank=12 consistent
Next: parent=129, try lr_W=60E-3 (ratio 1200:1)

## Iter 130: partial
Node: id=130, parent=129
Mode/Strategy: exploit
Config: lr_W=60E-3, lr=5E-5, L1=1E-6, batch_size=32, factorization=T, low_rank=10, n_frames=30000
Metrics: test_R2=0.298, test_pearson=0.245, connectivity_R2=0.585, final_loss=6025
Activity: effective_rank(90%)=3, effective_rank(99%)=13, spectral_radius=0.649
Mutation: lr_W: 100E-3 → 60E-3 (reduced ratio 2000:1 → 1200:1)
Parent rule: UCB selects node 130 (UCB=1.292), highest available
Observation: lr_W=60E-3 (ratio 1200:1) improved R² from 0.508→0.585; confirms lower ratio is better
Next: parent=130, try lr_W=40E-3 (ratio 800:1)

## Iter 131: partial
Node: id=131, parent=130
Mode/Strategy: exploit
Config: lr_W=40E-3, lr=5E-5, L1=1E-6, batch_size=32, factorization=T, low_rank=10, n_frames=30000
Metrics: test_R2=0.398, test_pearson=-0.258, connectivity_R2=0.615, final_loss=6196
Activity: effective_rank(90%)=3, effective_rank(99%)=13, spectral_radius=0.649
Mutation: lr_W: 60E-3 → 40E-3 (reduced ratio 1200:1 → 800:1)
Parent rule: UCB selects node 131 (UCB=1.481), highest available
Observation: lr_W=40E-3 (ratio 800:1) improved R² from 0.585→0.615; trend continues: lower ratio better
Next: parent=131, try lr_W=20E-3 (ratio 400:1)

## Iter 132: partial
Node: id=132, parent=131
Mode/Strategy: exploit
Config: lr_W=20E-3, lr=5E-5, L1=1E-6, batch_size=32, factorization=T, low_rank=10, n_frames=30000
Metrics: test_R2=0.282, test_pearson=0.053, connectivity_R2=0.634, final_loss=6776
Activity: effective_rank(90%)=3, effective_rank(99%)=14, spectral_radius=0.649
Mutation: lr_W: 40E-3 → 20E-3 (reduced ratio 800:1 → 400:1)
Parent rule: UCB selects node 132 (UCB=1.634), highest available
Observation: lr_W=20E-3 (ratio 400:1) improved R² from 0.615→0.634; trend continues; try even lower ratio 200:1
Next: parent=132, try lr_W=10E-3 (ratio 200:1)

## Iter 133: partial
Node: id=133, parent=132
Mode/Strategy: exploit
Config: lr_W=10E-3, lr=5E-5, L1=1E-6, batch_size=32, factorization=T, low_rank=10, n_frames=30000
Metrics: test_R2=0.373, test_pearson=-0.208, connectivity_R2=0.665, final_loss=7474
Activity: effective_rank(90%)=4, effective_rank(99%)=16, spectral_radius=0.649
Mutation: lr_W: 20E-3 → 10E-3 (reduced ratio 400:1 → 200:1)
Parent rule: UCB selects node 133 (UCB=1.783), highest available
Observation: lr_W=10E-3 (ratio 200:1) improved R² from 0.634→0.665; eff_rank(99%) increased 14→16; trend continues
Next: parent=133, try lr_W=5E-3 (ratio 100:1)

## Iter 134: partial
Node: id=134, parent=133
Mode/Strategy: exploit
Config: lr_W=5E-3, lr=5E-5, L1=1E-6, batch_size=32, factorization=T, low_rank=10, n_frames=30000
Metrics: test_R2=0.470, test_pearson=-0.346, connectivity_R2=0.637, final_loss=7908
Activity: effective_rank(90%)=3, effective_rank(99%)=13, spectral_radius=0.649
Mutation: lr_W: 10E-3 → 5E-3 (reduced ratio 200:1 → 100:1)
Parent rule: UCB selects node 134 (UCB=1.861), highest available
Observation: lr_W=5E-3 (ratio 100:1) degraded R² from 0.665→0.637; ratio 100:1 too low; optimal near 200:1
Next: parent=134, try lr=1E-4 (ratio 50:1) to test even lower extreme

## Iter 135: partial
Node: id=135, parent=134
Mode/Strategy: exploit
Config: lr_W=5E-3, lr=1E-4, L1=1E-6, batch_size=32, factorization=T, low_rank=10, n_frames=30000
Metrics: test_R2=0.552, test_pearson=-0.331, connectivity_R2=0.644, final_loss=7530
Activity: effective_rank(90%)=3, effective_rank(99%)=14, spectral_radius=0.649
Mutation: lr: 5E-5 → 1E-4 (reduced ratio 100:1 → 50:1)
Parent rule: UCB selects node 135 (UCB=1.966), highest available
Observation: ratio 50:1 gives R²=0.644, similar to ratio 100:1 (0.637); both worse than ratio 200:1 (0.665); optimal around 200:1
Next: parent=133 (best node R²=0.665), try lr_W=8E-3 with lr=5E-5 (ratio 160:1)

## Iter 136: partial
Node: id=136, parent=133
Mode/Strategy: exploit
Config: lr_W=8E-3, lr=5E-5, L1=1E-6, batch_size=32, factorization=T, low_rank=10, n_frames=30000
Metrics: test_R2=0.416, test_pearson=-0.050, connectivity_R2=0.503, final_loss=6162
Activity: effective_rank(90%)=2, effective_rank(99%)=10, spectral_radius=0.649
Mutation: lr_W: 10E-3 → 8E-3 (reduced ratio 200:1 → 160:1)
Parent rule: tried mutation from best R² node (133, R²=0.665)
Observation: ratio 160:1 (R²=0.503) worse than ratio 200:1 (R²=0.665); confirms ratio 200:1 is optimal; eff_rank dropped to 10
Next: parent=135 (UCB=2.058), try L1=5E-7 to probe lower regularization boundary

## Iter 137: partial
Node: id=137, parent=135
Mode/Strategy: exploit
Config: lr_W=5E-3, lr=1E-4, L1=5E-7, batch_size=32, factorization=T, low_rank=10, n_frames=30000
Metrics: test_R2=0.381, test_pearson=-0.117, connectivity_R2=0.631, final_loss=7584
Activity: effective_rank(90%)=3, effective_rank(99%)=13, spectral_radius=0.649
Mutation: L1: 1E-6 → 5E-7 (reduced regularization)
Parent rule: UCB selects node 137 (UCB=2.131), highest available
Observation: L1=5E-7 degraded R² from 0.644→0.631; confirms L1=1E-6 is optimal lower bound for this regime
Next: parent=133 (best R²=0.665), try batch_size=16 instead of 32

## Iter 138: partial
Node: id=138, parent=133
Mode/Strategy: exploit
Config: lr_W=10E-3, lr=5E-5, L1=1E-6, batch_size=16, factorization=T, low_rank=10, n_frames=30000
Metrics: test_R2=0.464, test_pearson=-0.258, connectivity_R2=0.627, final_loss=8715
Activity: effective_rank(90%)=3, effective_rank(99%)=15, spectral_radius=0.649
Mutation: batch_size: 32 → 16
Parent rule: selected best R² node (133, R²=0.665) for dimension switch
Observation: batch_size=16 degraded R² from 0.665→0.627; batch_size=32 remains optimal for this regime
Next: parent=137 (UCB=2.212), try increasing lr to 2E-4 (ratio 25:1)

## Iter 139: partial
Node: id=139, parent=137
Mode/Strategy: exploit
Config: lr_W=5E-3, lr=2E-4, L1=5E-7, batch_size=32, factorization=T, low_rank=10, n_frames=30000
Metrics: test_R2=0.452, test_pearson=-0.136, connectivity_R2=0.548, final_loss=6500
Activity: effective_rank(90%)=3, effective_rank(99%)=12, spectral_radius=0.649
Mutation: lr: 1E-4 → 2E-4 (reduced ratio 50:1 → 25:1)
Parent rule: UCB selects node 139 (UCB=2.206), highest available
Observation: ratio 25:1 (R²=0.548) worse than ratio 50:1 (R²=0.644); confirms ratio ~50-200:1 is optimal range
Next: parent=138 (UCB=2.285), try low_rank=15 (overparameterization)

## Iter 140: partial
Node: id=140, parent=138
Mode/Strategy: exploit
Config: lr_W=10E-3, lr=5E-5, L1=1E-6, batch_size=32, factorization=T, low_rank=15, n_frames=30000
Metrics: test_R2=0.357, test_pearson=0.306, connectivity_R2=0.742, final_loss=7047
Activity: effective_rank(90%)=4, effective_rank(99%)=15, spectral_radius=0.649
Mutation: low_rank: 10 → 15 (overparameterization)
Parent rule: UCB selects node 140 (UCB=2.474), highest available
Observation: overparameterization (low_rank=15) improved R² from 0.627→0.742 (NEW BLOCK BEST); test_pearson positive (0.306)
Next: parent=140, try low_rank=20 for more overparameterization

## Iter 141: partial
Node: id=141, parent=140
Mode/Strategy: exploit
Config: lr_W=10E-3, lr=5E-5, L1=1E-6, batch_size=32, factorization=T, low_rank=20, n_frames=30000
Metrics: test_R2=0.452, test_pearson=0.028, connectivity_R2=0.794, final_loss=6235
Activity: effective_rank(90%)=3, effective_rank(99%)=14, spectral_radius=0.649
Mutation: low_rank: 15 → 20 (more overparameterization)
Parent rule: UCB selects node 141 (UCB=2.597), highest available
Observation: low_rank=20 improved R² from 0.742→0.794 (NEW BLOCK BEST); overparameterization trend continues
Next: parent=141, try low_rank=25 for even more overparameterization

### Emerging Observations

- Dale_law=True with low_rank=10 produces effective_rank(99%)=10-16 vs ~4-7 without Dale
- R²=0.503-0.742 range achieved - much higher than Block 8's typical 0.07-0.16
- spectral_radius=0.649 is stable (vs 1.036 without Dale)
- lr_W:lr ratio trend: 2000:1 (0.508) → 1200:1 (0.585) → 800:1 (0.615) → 400:1 (0.634) → 200:1 (0.665) → 160:1 (0.503) → 100:1 (0.637) → 50:1 (0.644) → 25:1 (0.548)
- ratio 200:1 (lr_W=10E-3, lr=5E-5) with low_rank=10 achieved R²=0.665
- 13 consecutive partial results; best R²=0.742 at node 140 (overparameterization)
- Dale_law fundamentally changes dynamics: lower spectral_radius (0.649 vs 1.036), higher eff_rank (13 vs 4)
- ratio 160:1 significantly underperformed (0.503) - sensitive optimum at ratio 200:1
- L1=1E-6 optimal; L1=5E-7 degraded; batch_size=32 optimal
- **OVERPARAMETERIZATION HAS SWEET SPOT**: low_rank progression: 10→0.665, 15→0.742, 20→0.794, 25→0.126 (CRASHED)
- optimal overparameterization ~2x (low_rank=20 for connectivity_rank=10); 2.5x (low_rank=25) is too much
- 15 consecutive partial results; best R²=0.794 at node 141 (low_rank=20)
- next: return to best node (141), try lr_W adjustment around optimal config

## Iter 142: partial
Node: id=142, parent=141
Mode/Strategy: exploit
Config: lr_W=10E-3, lr=5E-5, L1=1E-6, batch_size=32, factorization=T, low_rank=25, n_frames=30000
Metrics: test_R2=0.475, test_pearson=0.509, connectivity_R2=0.126, final_loss=9372
Activity: effective_rank(90%)=4, effective_rank(99%)=15, spectral_radius=0.649
Mutation: low_rank: 20 → 25 (more overparameterization)
Parent rule: UCB selects node 142 (UCB=1.997 from previous iteration)
Observation: low_rank=25 CRASHED from 0.794→0.126; excessive overparameterization (2.5x) is harmful; optimal near 2x (low_rank=20)
