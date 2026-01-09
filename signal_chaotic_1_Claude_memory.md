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

### Emerging Observations

- **Surprising finding**: low_rank connectivity (rank=20) produces activity with eff_rank=6 — lower than Dale_law=True (eff_rank=10)
- This explains why Block 1's lr_W=4E-3 completely fails here
- Hypothesis: need lr_W similar to or higher than Dale_law=True (8E-2 to 1.2E-1)
- Watch: will low_rank_factorization=True work here since W is actually low-rank?

