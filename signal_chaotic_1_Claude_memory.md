# Working Memory: signal_chaotic_1_Claude

## Knowledge Base (accumulated across all blocks)

### Regime Comparison Table
| Block | Regime | E/I | n_frames | n_neurons | n_types | eff_rank | Best R² | Optimal lr_W | Optimal L1 | Key finding |
|-------|--------|-----|----------|-----------|---------|----------|---------|--------------|------------|-------------|
| 1 | chaotic, Dale_law=False | - | 10000 | 100 | 1 | 31-35 | 0.9999 | 4E-3 | 5E-5 | lr_W=4E-3 most reliable, low_rank_factorization MUST be False |

### Established Principles
1. chaotic regime with effective_rank 30+ is fully learnable without low-rank factorization
2. low_rank_factorization=True causes complete failure for chaotic regime (R²=0.208)
3. lr_W doubling from 2E-3 to 4E-3 critical for initial convergence
4. lr:lr_W ratio 15:1 to 20:1 optimal for chaotic regime
5. spectral radius stable at 0.973 (edge of chaos) across all chaotic iterations
6. stochastic training variability exists (identical configs yield different results)

### Open Questions
1. how does Dale_law=True (E/I separation) affect learning requirements?
2. what happens with low_rank connectivity (connectivity_rank=20)?
3. does effective_rank determine whether low_rank_factorization helps or hurts?

---

## Previous Block Summary (Block 1)

Block 1 (chaotic, Dale_law=False): Best R²=0.9999 at lr_W=4E-3, lr=1E-4 to 3E-4. Key finding: lr_W optimal range 3E-3 to 4E-3, low_rank_factorization MUST be False. Branching rate only 13% - too low.

---

## Current Block (Block 2)

### Block Info
Simulation: connectivity_type=chaotic, Dale_law=True, Dale_law_factor=0.5, n_frames=10000, n_neurons=100, n_neuron_types=1
Iterations: 17 to 32

### Hypothesis
testing Dale's law (E/I separation) in chaotic regime. hypothesis: Dale_law=True adds constraint that may reduce lr_W requirements or change optimal L1. starting with Block 1's optimal params (lr_W=4E-3, lr=2E-4, L1=5E-5) to test transferability.

### Iterations This Block

### Emerging Observations

