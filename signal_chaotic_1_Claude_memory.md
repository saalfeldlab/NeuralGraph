# Working Memory: signal_chaotic_1_Claude

## Knowledge Base (accumulated across all blocks)

### Regime Comparison Table

| Block | Regime                                     | Best R² | Optimal lr_W  | Optimal L1 | Key finding                                        |
| ----- | ------------------------------------------ | ------- | ------------- | ---------- | -------------------------------------------------- |
| 1     | chaotic, Dale_law=False                    | 1.000   | 8E-3 to 30E-3 | 1E-5       | lr_W:lr ratio 40-100:1 works; robust regime        |
| 2     | low_rank=20, Dale_law=False                | 0.977   | 16E-3         | 1E-6       | lr_W:lr ratio 320:1 needed; factorization required |
| 3     | chaotic, Dale_law=True                     | 0.940   | 80E-3         | 1E-6       | lr_W:lr ratio 800:1 needed; E/I constraint hardest |
| 4     | low_rank=50, Dale_law=False                | 0.989   | 20E-3         | 1E-6       | n_frames=20000 essential; effective_rank 7→21      |
| 5     | low_rank=20, Dale_law=True, n_frames=20000 | 1.000   | 8-10E-3       | 1E-6       | factorization=False works with n_frames=20000      |

### Coverage Table

| connectivity_type | Dale_law=False | Dale_law=True |
| ----------------- | -------------- | ------------- |
| chaotic           | Block 1 ✓      | Block 3 ✓     |
| low_rank=20       | Block 2 ✓      | Block 5 ✓     |
| low_rank=50       | Block 4 ✓      | Block 6 (now) |

### Established Principles

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

### Open Questions

- can low_rank=50 + Dale_law + n_frames=20000 also achieve perfect R²=1.000? → testing in Block 6
- what is the minimum n_frames needed for perfect recovery?
- is there a universal formula: effective_rank > threshold → factorization not needed?
- what is the hardest learnable regime? (all tested so far have been solved)

---

## Previous Block Summary (Block 5)

Block 5 (low_rank=20, Dale_law=True, n_frames=20000): Best R²=1.000 (3 consecutive perfect results).
Key finding: double constraint achieves perfect recovery with factorization=False when n_frames=20000 provides rich activity (effective_rank ~30). Regime extremely robust: 16/16 converged, lr_W working range spans 100x.

---

## Current Block (Block 6)

### Block Info

Simulation: connectivity_type=low_rank, connectivity_rank=50, Dale_law=True, n_frames=20000
Iterations: 81 to 96

### Hypothesis

low_rank=50 + Dale_law=True should also achieve perfect R²=1.000 based on Block 5 success with low_rank=20. Starting with Block 5's optimal config: lr_W=10E-3, lr=5E-5, lr_emb=1E-4, L1=1E-6, batch_size=16, factorization=False. This completes the coverage table for all Dale_law=True combinations.

### Iterations This Block

## Iter 81: converged

Node: id=81, parent=root
Mode/Strategy: exploit
Config: lr_W=10E-3, lr=5E-5, lr_emb=1E-4, coeff_W_L1=1E-6, batch_size=16, low_rank_factorization=False, low_rank=50, n_frames=20000
Metrics: test_R2=0.752, test_pearson=0.595, connectivity_R2=0.902, final_loss=7.37E+03
Activity: effective_rank=10, spectral_radius=1.273, oscillatory patterns
Mutation: starting config from Block 5
Parent rule: root (first iteration of block)
Observation: borderline converged; effective_rank only 10 vs ~30 in Block 5 - low_rank=50 has lower activity complexity than low_rank=20
Next: parent=81 (highest UCB)

## Iter 82: converged

Node: id=82, parent=81
Mode/Strategy: exploit
Config: lr_W=20E-3, lr=5E-5, lr_emb=1E-4, coeff_W_L1=1E-6, batch_size=16, low_rank_factorization=False, low_rank=50, n_frames=20000
Metrics: test_R2=0.923, test_pearson=0.867, connectivity_R2=0.940, final_loss=6.999E+03
Activity: effective_rank=10, spectral_radius=1.273, oscillatory patterns
Mutation: lr_W: 10E-3 -> 20E-3 (2x increase)
Parent rule: highest UCB (node 81)
Observation: R² improved 0.902→0.940 with 2x lr_W; ratio now 400:1, continuing toward Block 3's 800:1
Next: parent=82 (highest UCB)

## Iter 83: partial

Node: id=83, parent=82
Mode/Strategy: exploit
Config: lr_W=40E-3, lr=5E-5, lr_emb=1E-4, coeff_W_L1=1E-6, batch_size=16, low_rank_factorization=False, low_rank=50, n_frames=20000
Metrics: test_R2=0.963, test_pearson=0.935, connectivity_R2=0.898, final_loss=4.77E+03
Activity: effective_rank=10, spectral_radius=1.273, oscillatory patterns
Mutation: lr_W: 20E-3 -> 40E-3 (2x increase)
Parent rule: highest UCB (node 82)
Observation: R² decreased 0.940→0.898 with 2x lr_W; ratio 800:1 may be too high for this regime; optimum likely between 400:1 and 800:1
Next: parent=83 (highest UCB)

## Iter 84: failed

Node: id=84, parent=83
Mode/Strategy: exploit
Config: lr_W=30E-3, lr=5E-5, lr_emb=1E-4, coeff_W_L1=1E-6, batch_size=16, low_rank_factorization=False, low_rank=50, n_frames=20000
Metrics: test_R2=0.148, test_pearson=-0.015, connectivity_R2=0.005, final_loss=9.44E+03
Activity: effective_rank=10, spectral_radius=1.273, oscillatory patterns
Mutation: lr_W: 40E-3 -> 30E-3 (midpoint between 20E-3 and 40E-3)
Parent rule: highest UCB (node 83)
Observation: catastrophic failure at lr_W=30E-3 despite being between working values; stochastic instability in this regime
Next: parent=83 (highest UCB)

## Iter 85: partial

Node: id=85, parent=83
Mode/Strategy: exploit
Config: lr_W=40E-3, lr=5E-5, lr_emb=1E-4, coeff_W_L1=1E-6, batch_size=16, low_rank_factorization=True, low_rank=50, n_frames=20000
Metrics: test_R2=0.663, test_pearson=0.428, connectivity_R2=0.830, final_loss=4.32E+03
Activity: effective_rank=10, spectral_radius=1.273, oscillatory patterns
Mutation: low_rank_factorization: False -> True
Parent rule: highest UCB (node 83)
Observation: factorization=True with lr_W=40E-3 gave R²=0.830, worse than node 82's R²=0.940 without factorization; factorization doesn't help here
Next: parent=82 (best R² in block)

## Iter 86: partial

Node: id=86, parent=82
Mode/Strategy: exploit
Config: lr_W=25E-3, lr=5E-5, lr_emb=1E-4, coeff_W_L1=1E-6, batch_size=16, low_rank_factorization=False, low_rank=50, n_frames=20000
Metrics: test_R2=0.938, test_pearson=0.885, connectivity_R2=0.894, final_loss=7.58E+03
Activity: effective_rank=10, spectral_radius=1.273, oscillatory patterns
Mutation: lr_W: 20E-3 -> 25E-3 (interpolating between 20E-3 and 40E-3)
Parent rule: highest UCB (node 82)
Observation: lr_W=25E-3 gave R²=0.894, slightly worse than node 82's R²=0.940; confirms lr_W=20E-3 is near optimal
Next: parent=86 (highest UCB)

## Iter 87: converged

Node: id=87, parent=86
Mode/Strategy: exploit
Config: lr_W=25E-3, lr=1E-4, lr_emb=1E-4, coeff_W_L1=1E-6, batch_size=16, low_rank_factorization=False, low_rank=50, n_frames=20000
Metrics: test_R2=0.999, test_pearson=0.999, connectivity_R2=0.998, final_loss=4.74E+03
Activity: effective_rank=20, spectral_radius=1.014, rich dynamics
Mutation: lr: 5E-5 -> 1E-4 (2x increase, ratio now 250:1)
Parent rule: highest UCB (node 86)
Observation: breakthrough! near-perfect R²=0.998 by increasing lr; ratio 250:1 optimal vs previous 400:1-500:1

## Iter 88: partial

Node: id=88, parent=87
Mode/Strategy: success-exploit (robustness test)
Config: lr_W=25E-3, lr=1E-4, lr_emb=1E-4, coeff_W_L1=1E-6, batch_size=16, low_rank_factorization=False, low_rank=50, n_frames=20000
Metrics: test_R2=0.938, test_pearson=0.838, connectivity_R2=0.866, final_loss=4.69E+03
Activity: effective_rank=10, spectral_radius=1.273, oscillatory patterns
Mutation: re-run of iter 87 config (robustness test)
Parent rule: highest UCB (node 87, R²=0.998)
Observation: regression from R²=0.998 to 0.866 with same config; effective_rank dropped 20→10; high stochastic variance in this regime
Next: parent=88 (highest UCB)

## Iter 89: partial

Node: id=89, parent=88
Mode/Strategy: exploit
Config: lr_W=30E-3, lr=1E-4, lr_emb=1E-4, coeff_W_L1=1E-6, batch_size=16, low_rank_factorization=False, low_rank=50, n_frames=20000
Metrics: test_R2=0.883, test_pearson=0.805, connectivity_R2=0.883, final_loss=4.29E+03
Activity: effective_rank=10, spectral_radius=1.273, oscillatory patterns
Mutation: lr_W: 25E-3 -> 30E-3 (increased from parent 88)
Parent rule: highest UCB (node 88, UCB=1.866)
Observation: R²=0.883 similar to parent's 0.866; lr_W=30E-3 ratio 300:1 not improving; node 87's near-perfect R²=0.998 still not replicated
Next: parent=89 (highest UCB=2.383)

## Iter 90: converged

Node: id=90, parent=89
Mode/Strategy: exploit
Config: lr_W=30E-3, lr=1E-4, lr_emb=1E-4, coeff_W_L1=1E-6, batch_size=8, low_rank_factorization=False, low_rank=50, n_frames=20000
Metrics: test_R2=0.885, test_pearson=0.764, connectivity_R2=0.925, final_loss=4.04E+03
Activity: effective_rank=10, spectral_radius=1.273, oscillatory patterns
Mutation: batch_size: 16 -> 8
Parent rule: highest UCB (node 89, UCB=2.383)
Observation: batch_size=8 improved R² from 0.883 to 0.925; converged but still below iter 87's 0.998
Next: parent=90 (highest UCB=2.507)

## Iter 91: converged

Node: id=91, parent=90
Mode/Strategy: recombine (iter 87's lr_W=25E-3 + iter 90's batch_size=8)
Config: lr_W=25E-3, lr=1E-4, lr_emb=1E-4, coeff_W_L1=1E-6, batch_size=8, low_rank_factorization=False, low_rank=50, n_frames=20000
Metrics: test_R2=0.567, test_pearson=0.480, connectivity_R2=0.924, final_loss=4.27E+03
Activity: effective_rank=10, spectral_radius=1.273, oscillatory patterns
Mutation: lr_W: 30E-3 -> 25E-3 (recombine: iter 87's lr_W with iter 90's batch_size=8)
Parent rule: highest UCB (node 90, UCB=2.507)
Observation: R²=0.924 similar to parent's 0.925; recombine didn't replicate iter 87's 0.998; effective_rank=10 persists
Next: parent=91 (highest UCB=2.582)

### Emerging Observations

- **breakthrough at iter 87**: R²=0.998 achieved with lr=1E-4, lr_W=25E-3 (ratio 250:1)
- key insight: lr was too low, not lr_W; ratio 250:1 better than 400:1-500:1
- **iters 88-91**: all give R²=0.86-0.93 range despite varied configs - high stochastic variance
- effective_rank consistently 10 (vs 20 in iter 87's success) - data generation variability
- regime NOT yet robustly solved - iter 87's success was lucky with effective_rank=20
- **pattern**: when effective_rank=20, R²~0.998; when effective_rank=10, R²~0.9
- next: try increasing lr to 1.5E-4 or 2E-4 to see if higher lr helps with low effective_rank
