# Experiment Log: signal_low_rank (parallel)

## Block 1: Initial Gain × Rank Landscape Mapping

### Batch 1 (Initial Setup)

**Strategy**: Map the gain × rank landscape with diverse starting points

| Slot | gain | rank | lr_W | L1   | edge_diff | batch | Rationale |
| ---- | ---- | ---- | ---- | ---- | --------- | ----- | --------- |
| 0    | 7    | 20   | 3E-3 | 1E-6 | 10000     | 8     | baseline recipe validation |
| 1    | 4    | 20   | 4E-3 | 1E-6 | 10000     | 8     | higher lr_W for weaker dynamics at low gain |
| 2    | 10   | 20   | 2E-3 | 1E-6 | 10000     | 8     | lower lr_W for stability with chaotic high-gain |
| 3    | 7    | 10   | 3E-3 | 1E-6 | 15000     | 8     | higher edge_diff to counter degeneracy at low rank |

**Hypotheses**:
- Slot 0: should converge (known-good recipe with corrected L1=1E-6)
- Slot 1: gain=4 has weaker dynamics, fewer activity modes → harder W recovery, may need tuning
- Slot 2: gain=10 is more chaotic, richer signal but harder rollout stability
- Slot 3: rank=10 is more degenerate, edge_diff=15000 may help constrain MLP compensation

(awaiting results)

