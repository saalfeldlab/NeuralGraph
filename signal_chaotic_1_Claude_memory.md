# Working Memory: signal_chaotic_1_Claude

## Knowledge Base (accumulated across all blocks)

### Regime Comparison Table

| Block | Regime                   | E/I  | n_frames | n_neurons | n_types | Best R² | Optimal lr_W  | Optimal L1 | Key finding                                              |
| ----- | ------------------------ | ---- | -------- | --------- | ------- | ------- | ------------- | ---------- | -------------------------------------------------------- |
| 1     | chaotic, Dale=False      | -    | 10000    | 100       | 1       | 1.000   | 8E-3 to 30E-3 | 1E-5       | lr_W:lr ratio 40-100:1 works; robust regime              |
| 2     | low_rank=20, Dale=False  | -    | 10000    | 100       | 1       | 0.977   | 16E-3         | 1E-6       | lr_W:lr ratio 320:1 needed; factorization required       |
| 3     | chaotic, Dale=True       | 0.5  | 10000    | 100       | 1       | 0.940   | 80E-3         | 1E-6       | lr_W:lr ratio 800:1 needed; E/I constraint hardest       |
| 4     | low_rank=50, Dale=False  | -    | 20000    | 100       | 1       | 0.989   | 20E-3         | 1E-6       | n_frames=20000 essential; effective_rank 7→21            |
| 5     | low_rank=20, Dale=True   | 0.5  | 20000    | 100       | 1       | 1.000   | 8-10E-3       | 1E-6       | factorization=False works with n_frames=20000            |
| 6     | low_rank=50, Dale=True   | 0.5  | 20000    | 100       | 1       | 0.998   | 25E-3         | 1E-6       | effective_rank variance (10 vs 20) drives R² (0.92 vs 0.998) |

### Coverage Table

| connectivity_type | Dale_law=False | Dale_law=True |
| ----------------- | -------------- | ------------- |
| chaotic           | Block 1 ✓      | Block 3 ✓     |
| low_rank=20       | Block 2 ✓      | Block 5 ✓     |
| low_rank=50       | Block 4 ✓      | Block 6 ✓     |

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
- with sufficient data (n_frames=20000), even double constraints (low_rank + Dale_law) achieve perfect R²=1.000
- **NEW from Block 6**: effective_rank is the primary predictor of achievable R²
  - effective_rank=20 → R²≈0.998 achievable
  - effective_rank=10 → R²≈0.92 ceiling (regardless of training params)
  - data generation randomness causes effective_rank variance

### Open Questions

- can increasing n_frames to 30000 stabilize effective_rank and eliminate stochastic variance?
- what is the minimum n_frames needed for perfect recovery?
- is there a universal formula: effective_rank > threshold → factorization not needed?
- is the chaotic + Dale_law regime (Block 3, best R²=0.940) improvable with n_frames=20000?

---

## Previous Block Summary (Block 6)

Block 6 (low_rank=50, Dale_law=True, n_frames=20000): Best R²=0.998 (iter 87).
Key finding: effective_rank variance (10 vs 20) caused by stochastic data generation drives R² variance (0.92 vs 0.998). When effective_rank=20, near-perfect recovery achievable; when effective_rank=10, R²≈0.92 ceiling. Optimal ratio 250:1. Factorization=True did not help. 10/16 converged.

---

## Current Block (Block 7)

### Block Info

Simulation: connectivity_type=chaotic, Dale_law=True, n_frames=20000
Iterations: 97 to 112

### Hypothesis

Block 3 (chaotic + Dale_law, n_frames=10000) achieved only R²=0.940. The question is whether n_frames=20000 can improve this regime as it did for low_rank regimes in Blocks 4-6. Based on established principles, chaotic + Dale_law requires the highest lr_W:lr ratio (800:1). Starting with lr_W=80E-3, lr=1E-4, L1=1E-6 from Block 3's optimal config.

### Iterations This Block

## Iter 97: converged
Node: id=97, parent=root
Mode/Strategy: exploit (start of new block)
Config: lr_W=25E-3, lr=1E-4, L1=1E-6, batch_size=8, factorization=F, n_frames=20000
Metrics: test_R2=0.990, test_pearson=0.987, connectivity_R2=0.9998, final_loss=3479
Activity: effective_rank(99%)=34, spectral_radius=1.285
Mutation: n_frames: 10000 -> 20000 (block change)
Observation: n_frames=20000 dramatically improves chaotic+Dale_law - from Block 3's 0.940 to 0.9998
Next: parent=97

## Iter 98: converged
Node: id=98, parent=97
Mode/Strategy: exploit
Config: lr_W=30E-3, lr=1E-4, L1=1E-6, batch_size=8, factorization=F, n_frames=20000
Metrics: test_R2=0.968, test_pearson=0.965, connectivity_R2=0.9998, final_loss=3522
Activity: effective_rank(99%)=35, spectral_radius=1.285
Mutation: lr_W: 25E-3 -> 30E-3
Parent rule: highest UCB (node 97)
Observation: lr_W=30E-3 (ratio 300:1) maintains perfect connectivity recovery
Next: parent=98

## Iter 99: converged
Node: id=99, parent=98
Mode/Strategy: exploit
Config: lr_W=40E-3, lr=1E-4, L1=1E-6, batch_size=8, factorization=F, n_frames=20000
Metrics: test_R2=0.987, test_pearson=0.986, connectivity_R2=0.9996, final_loss=3261
Activity: effective_rank(99%)=32, spectral_radius=1.285
Mutation: lr_W: 30E-3 -> 40E-3
Parent rule: highest UCB (node 99)
Observation: lr_W=40E-3 (ratio 400:1) still works perfectly - regime is very robust to lr_W
Next: parent=99 (failure-probe: test extreme lr_W to find upper boundary)

### Emerging Observations

- **hypothesis confirmed**: n_frames=20000 dramatically improves chaotic+Dale_law regime
- jumped from Block 3's best 0.940 to 0.9998 immediately
- lr_W range 25E-3 to 40E-3 (ratio 250-400:1) all work perfectly - much lower than Block 3's 80E-3 (ratio 800:1)
- effective_rank=32-35 is very high, enabling excellent recovery
- spectral_radius=1.285 indicates slightly unstable dynamics (but still trainable)
- 3/3 converged so far - this regime is very robust with n_frames=20000
- next: failure-probe with extreme lr_W (100E-3 or higher) to find upper boundary
