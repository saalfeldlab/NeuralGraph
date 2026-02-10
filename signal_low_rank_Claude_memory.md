# Working Memory: signal_low_rank (parallel)

## Knowledge Base (accumulated across all blocks)

### Best Configurations Found

| Blk | gain | rank | lr_W | lr   | L1   | edge_diff | n_ep_init | batch | conn_R2 | test_R2 | Finding  |
| --- | ---- | ---- | ---- | ---- | ---- | --------- | --------- | ----- | ------- | ------- | -------- |
| -   | 7    | 20   | 3E-3 | 1E-4 | 1E-6 | 10000     | 2         | 8     | 0.993   | 0.996   | baseline from prior exploration |

### Gain × Rank Landscape Map

| gain\rank | 10 | 15 | 20 | 25 | 30 |
| --------- | -- | -- | -- | -- | -- |
| 4         | ?  | ?  | ?  | ?  | ?  |
| 5         | ?  | ?  | ?  | ?  | ?  |
| 6         | ?  | ?  | ?  | ?  | ?  |
| 7         | ?  | ?  | 0.993/0.996 | ?  | ?  |
| 8         | ?  | ?  | ?  | ?  | ?  |
| 9         | ?  | ?  | ?  | ?  | ?  |
| 10        | ?  | ?  | ?  | ?  | ?  |

### Established Principles

- `lr_W=3E-3` optimal for gain=7, rank=20 (prior exploration)
- `coeff_W_L1=1E-6` critical for low-rank dynamics; 1E-5 degrades
- `coeff_edge_diff=10000` constrains lin_edge monotonicity

### Open Questions

- Does the gain=7, rank=20 recipe transfer to other (gain, rank) combinations?
- How does lower gain (4) affect learnability? (weaker dynamics, less signal)
- How does higher gain (10) affect rollout stability? (more chaotic)
- How does lower rank (10) affect degeneracy? (more equivalent W solutions)

---

## Previous Block Summary

(first block - no prior blocks)

---

## Current Block (Block 1)

### Block Info

Focus: Initial gain × rank landscape mapping
Slots: 0=baseline(7,20), 1=low-gain(4,20), 2=high-gain(10,20), 3=low-rank(7,10)

### Hypothesis

The known-good recipe (lr_W=3E-3, L1=1E-6, edge_diff=10000) works for gain=7, rank=20. Expect:
- Slot 0: should converge (baseline validation with optimal L1=1E-6)
- Slot 1 (gain=4): may struggle with weak dynamics, might need higher lr_W
- Slot 2 (gain=10): may have rollout instability, dynamics richer but harder to stabilize
- Slot 3 (rank=10): more degenerate solutions possible, edge_diff may need increase

### Initial Config Variations (Batch 1)

| Slot | gain | rank | lr_W | L1   | edge_diff | batch | Rationale |
| ---- | ---- | ---- | ---- | ---- | --------- | ----- | --------- |
| 0    | 7    | 20   | 3E-3 | 1E-6 | 10000     | 8     | baseline recipe validation |
| 1    | 4    | 20   | 4E-3 | 1E-6 | 10000     | 8     | higher lr_W to compensate for weaker dynamics |
| 2    | 10   | 20   | 2E-3 | 1E-6 | 10000     | 8     | lower lr_W for stability with chaotic dynamics |
| 3    | 7    | 10   | 3E-3 | 1E-6 | 15000     | 8     | higher edge_diff to counter degeneracy at low rank |

### Iterations This Block

(awaiting first results)

### Emerging Observations

(awaiting first results)

