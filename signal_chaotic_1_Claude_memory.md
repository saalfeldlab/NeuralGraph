# Working Memory: signal_chaotic_1_Claude

## Knowledge Base (accumulated across all blocks)

### Regime Comparison Table
| Block | Regime | eff_rank | Best R² | Optimal lr_W | Optimal L1 | Key finding |
|-------|--------|----------|---------|--------------|------------|-------------|
| 1 | chaotic, Dale_law=False | 31-35 | 1.000 | 4E-3 to 4E-2 | ≤5E-4 | extremely robust; low_rank_factorization=True catastrophic |
| 2 | chaotic, Dale_law=True | 10 | 0.913 | 5E-2 to 1.2E-1 | ≤1E-5 | R² ceiling ~0.91 due to low eff_rank; needs 10-30x higher lr_W |

### Established Principles
1. chaotic full-rank connectivity: lr_W has 10x robust range (4E-3 to 4E-2)
2. L1 regularization boundary: regime-dependent (Dale_law=False: ≤5E-4, Dale_law=True: ≤1E-5)
3. low_rank_factorization=True catastrophically fails for chaotic W regardless of Dale_law
4. batch_size (8, 16, 32) does not affect convergence in chaotic regime
5. Dale_law=True reduces effective_rank ~3x (from ~32 to ~10)
6. effective_rank determines R² ceiling: eff_rank=10 → R²≈0.91 max; eff_rank=32 → R²=1.0 achievable
7. Dale_law=True requires 10-30x higher lr_W than Dale_law=False
8. lr/lr_W ratio matters: optimal ~1000:1 to 2000:1 for Dale_law=True

### Open Questions
- does low_rank connectivity work with low_rank_factorization=True?
- how does connectivity_rank affect achievable R²?
- what is the minimum n_frames needed for convergence?
- does n_neuron_types>1 change parameter sensitivity?

---

## Previous Block Summary

**Block 2** (chaotic, Dale_law=True): 6/16 converged (37.5%), Best R²=0.913.
Key finding: effective_rank=10 creates R² ceiling at ~0.91; lr_W must be 10-30x higher; L1 threshold 10x stricter.

---

## Current Block (Block 3)

### Block Info
Simulation: connectivity_type=low_rank, connectivity_rank=20, Dale_law=False, n_frames=10000, n_neurons=100, n_types=1
Iterations: 33 to 48

### Hypothesis
low_rank connectivity (rank=20) creates structured, lower-dimensional W. Predictions:
1. low_rank_factorization=True may work (unlike chaotic) since W is actually low-rank
2. lower lr_W may be optimal (structured optimization)
3. higher R² ceiling possible (structure easier to learn)

### Iterations This Block

## Iter 33: failed
Node: id=33, parent=root
Mode/Strategy: exploit (first iteration of block)
Config: lr_W=4E-3, lr=1E-4, coeff_W_L1=1E-5, low_rank_factorization=False
Metrics: connectivity_R2=0.038, test_R2=0.335, final_loss=4199.5
Activity: eff_rank=6, spectral_radius=0.962
Observation: low_rank (rank=20) produces eff_rank=6 (lower than Dale_law=True!); lr_W=4E-3 fails completely
Next: parent=33, lr_W=8E-2

## Iter 34: partial
Node: id=34, parent=33
Mode/Strategy: exploit
Config: lr_W=8E-2, lr=1E-4, coeff_W_L1=1E-5, low_rank_factorization=False
Metrics: connectivity_R2=0.674, test_R2=0.830, test_pearson=0.840, final_loss=2302.4
Activity: eff_rank=8, spectral_radius=1.184
Mutation: lr_W: 4E-3 → 8E-2 (20x increase)
Parent rule: highest UCB (node 33, only option)
Observation: 20x lr_W increase: R² 0.038→0.674; still partial, need higher lr_W or factorization
Next: parent=34, lr_W=1.2E-1

## Iter 35: failed
Node: id=35, parent=34
Mode/Strategy: exploit
Config: lr_W=1.2E-1, lr=1E-4, coeff_W_L1=1E-5, low_rank_factorization=False
Metrics: connectivity_R2=0.063, test_R2=0.556, test_pearson=-0.100, final_loss=1683.1
Activity: eff_rank=6, spectral_radius=0.962
Mutation: lr_W: 8E-2 → 1.2E-1 (1.5x increase)
Parent rule: highest UCB (node 34, UCB=1.490)
Observation: lr_W=1.2E-1 too high; R² dropped 0.674→0.063; upper bound of lr_W is between 8E-2 and 1.2E-1
Next: parent=34, try low_rank_factorization=True (W is actually rank-20)

## Iter 36: failed
Node: id=36, parent=34
Mode/Strategy: exploit
Config: lr_W=8E-2, lr=1E-4, coeff_W_L1=1E-5, low_rank_factorization=True, low_rank=20
Metrics: connectivity_R2=0.144, test_R2=0.486, test_pearson=0.148, final_loss=2055.9
Activity: eff_rank=6, spectral_radius=0.962
Mutation: low_rank_factorization: False → True (testing hypothesis)
Parent rule: highest UCB (node 34, UCB=1.381, best R²=0.674)
Observation: low_rank_factorization=True with matching rank=20 still fails; R² dropped 0.674→0.144; factorization not the answer
Next: parent=36 (highest UCB=1.558), try lr_W=4E-2 with factorization=False

### Emerging Observations

- **Surprising finding**: low_rank connectivity (rank=20) produces activity with eff_rank=6-8 — lower than Dale_law=True (eff_rank=10)
- lr_W optimal range narrowing: 8E-2 works (R²=0.674), 1.2E-1 fails
- lr_W=4E-3 (Block 1 optimal) completely fails for low_rank regime
- **Key test result**: low_rank_factorization=True FAILS even when W is actually rank-20 (R²=0.144)
- This strengthens Principle #3: low_rank_factorization=True fails regardless of W structure
- R² ceiling unknown — partial at 0.674 so far, need more exploration
- **Next strategy**: explore lr_W values between 4E-3 and 8E-2

