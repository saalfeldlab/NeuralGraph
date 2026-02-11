# Working Memory: signal_low_rank (parallel)

## Knowledge Base (accumulated across all blocks)

### Best Configurations Found

| Blk | gain | rank | lr_W | lr   | L1   | edge_diff | n_ep_init | batch | conn_R2 | test_R2 | Finding  |
| --- | ---- | ---- | ---- | ---- | ---- | --------- | --------- | ----- | ------- | ------- | -------- |
| -   | 7    | 20   | 3E-3 | 1E-4 | 1E-6 | 10000     | 2         | 8     | 0.993   | 0.996   | baseline from prior exploration |
| 3   | 5    | 10   | 5E-3 | 1E-4 | 1E-6 | 15000     | 2         | 8     | 0.886   | 0.871   | was best gain=5 (iter 36) |
| 4   | 5    | 10   | 6E-3 | 1E-4 | 1E-6 | 15000     | 2         | 8     | 0.891   | 0.985   | new best gain=5 (iter 40) |
| 4   | 7    | 30   | 3E-3 | 1E-4 | 1E-6 | 15000     | 2         | 8     | 1.000   | 0.800   | rank=30 perfect W but poor rollout |
| 4   | 7    | 15   | 4E-3 | 1E-4 | 1E-6 | 15000     | 2         | 8     | 0.851   | 0.998   | seed=42 broke rank=15 anomaly |
| 3   | 7    | 25   | 3E-3 | 1E-4 | 1E-6 | 15000     | 2         | 8     | 0.933   | 0.999   | first rank=25 test (iter 33) |
| 4   | 7    | 30   | 3E-3 | 1E-4 | 1E-6 | 10000     | 2         | 8     | 0.994   | 0.992   | rank=30 solved (iter 44) |
| 4   | 7    | 30   | 2E-3 | 1E-4 | 1E-6 | 10000     | 2         | 8     | 0.982   | 0.998   | rank=30 optimal (iter 48) |
| 5   | 8    | 20   | 2E-3 | 1E-4 | 1E-6 | 15000     | 2         | 8     | 1.000   | 0.999   | gain=8 rank=20 optimal (iter 56) |
| 5   | 8    | 20   | 2E-3 | 1E-4 | 1E-6 | 10000     | 2         | 8     | 1.000   | 0.98    | gain=8 edge_diff=10000 also works (iter 57) |
| 5   | 8    | 20   | 2E-3 | 1E-4 | 1E-6 | 10000     | 2         | 8     | 1.000   | 0.97    | gain=8 seed=42 robust (iter 59) |
| 6   | 9    | 10   | 2E-3 | 1E-4 | 1E-6 | 15000     | 2         | 8     | 1.000   | 0.97    | gain=9 rank=10 (iter 73) |
| 6   | 9    | 20   | 2E-3 | 1E-4 | 1E-6 | 15000     | 2         | 8     | 1.000   | 0.996   | gain=9 seed=99 robust (iter 74) |
| 6   | 9    | 30   | 2E-3 | 1E-4 | 1E-6 | 15000     | 2         | 8     | 1.000   | 0.991   | gain=9 rank=30 (iter 78) |
| 6   | 10   | 10   | 2E-3 | 1E-4 | 1E-6 | 15000     | 2         | 8     | 1.000   | 0.991   | gain=10 rank=10 (iter 80) |
| 8   | 10   | 20   | 2E-3 | 1E-4 | 1E-6 | 10000     | 2         | 8     | 1.000   | 0.977   | gain=10 rank=20 optimal edge_diff (iter 85) |
| 8   | 8    | 30   | 2E-3 | 1E-4 | 1E-6 | 15000     | 2         | 8     | 0.9999  | 0.9998  | gain=8 rank=30 SOLVED (iter 86) |
| 8   | 10   | 30   | 2E-3 | 1E-4 | 1E-6 | 15000     | 2         | 8     | 0.9998  | 0.995   | gain=10 rank=30 SOLVED (iter 87) |
| 8   | 10   | 20   | 2E-3 | 1E-4 | 1E-6 | 10000     | 2         | 8     | 1.000   | 0.997   | gain=10 rank=20 seed=42 (iter 89) |
| 8   | 10   | 30   | 2E-3 | 1E-4 | 1E-6 | 10000     | 2         | 8     | 1.000   | 0.9997  | gain=10 rank=30 edge_diff=10000 BEST (iter 91) |
| 8   | 8    | 30   | 2E-3 | 1E-4 | 1E-6 | 10000     | 2         | 8     | 1.000   | 0.9999  | gain=8 rank=30 seed=42 BEST (iter 94) |
| 8   | 10   | 30   | 2E-3 | 1E-4 | 1E-6 | 10000     | 2         | 8     | 0.9999  | 0.9994  | gain=10 rank=30 seed=42 (iter 95) |
| 8   | 9    | 25   | 2E-3 | 1E-4 | 1E-6 | 10000     | 2         | 8     | 0.994   | 0.9999  | gain=9 rank=25 SOLVED (iter 96) |
| 9   | 6    | 15   | 3E-3 | 1E-4 | 1E-6 | 10000     | 2         | 8     | 1.000   | 0.993   | gain=6 rank=15 SOLVED (iter 98) |
| 9   | 8    | 15   | 2E-3 | 1E-4 | 1E-6 | 10000     | 2         | 8     | 1.000   | 0.9995  | gain=8 rank=15 SOLVED (iter 99) |
| 9   | 9    | 25   | 2E-3 | 1E-4 | 1E-6 | 10000     | 2         | 8     | 1.000   | 0.9999  | gain=9 rank=25 seed=42 ROBUST (iter 100) |
| 9   | 8    | 10   | 2E-3 | 1E-4 | 1E-6 | 20000     | 2         | 8     | 1.000   | 0.996   | gain=8 rank=10 SOLVED edge_diff=20000 (iter 101) |
| 9   | 6    | 15   | 3E-3 | 1E-4 | 1E-6 | 10000     | 2         | 8     | 1.000   | 0.9994  | gain=6 rank=15 seed=42 ROBUST (iter 102) |
| 9   | 8    | 15   | 2E-3 | 1E-4 | 1E-6 | 10000     | 2         | 8     | 1.000   | 0.995   | gain=8 rank=15 seed=42 ROBUST (iter 103) |
| 9   | 8    | 10   | 2E-3 | 1E-4 | 1E-6 | 20000     | 2         | 8     | 1.000   | 0.980   | gain=8 rank=10 edge_diff=20000 seed=42 (iter 105) |
| 9   | 9    | 15   | 2E-3 | 1E-4 | 1E-6 | 10000     | 2         | 8     | 1.000   | 0.9997  | gain=9 rank=15 SOLVED (iter 107) |
| 10  | 10   | 25   | 2E-3 | 1E-4 | 1E-6 | 15000     | 2         | 8     | 1.000   | 0.999   | gain=10 rank=25 SOLVED (iter 113) |
| 10  | 10   | 15   | 2E-3 | 1E-4 | 1E-6 | 10000     | 2         | 8     | 1.000   | 1.000   | gain=10 rank=15 seed-robust (iter 114) |
| 10  | 6    | 30   | 3E-3 | 1E-4 | 1E-6 | 10000     | 2         | 8     | 1.000   | 0.995   | gain=6 rank=30 SOLVED (iter 116) |

### Gain × Rank Landscape Map

| gain\rank | 10 | 15 | 20 | 25 | 30 |
| --------- | -- | -- | -- | -- | -- |
| 4         | 0.78/0.97 partial (lr_W=4E-3 peak) | ?  | 0.20/0.77 FAIL | ?  | ?  |
| 5         | 0.89/0.83 partial (lr_W=6E-3 best) | ?  | 0.60/0.99 partial | ?  | ?  |
| 6         | 1.00/0.99 ✓✓ (2 seeds) | **1.00/0.999 ✓✓** (seed-robust)  | 0.97/0.98 ✓ | **0.71/0.999 DEGEN** seed-indep | **1.00/0.995 ✓** NEW (iter 116)  |
| 7         | 1.00/1.00 ✓✓ (3 seeds) | **0.74/0.85 FAIL** lr_W=2E-3 worse | 0.999/0.993 ✓ | **0.95/1.00 ✓** (lr_W=3E-3) | 1.00/0.93 seed-sens |
| 8         | **1.00/0.98 ✓✓** edge_diff=20000 seed-robust | **1.00/0.995+ ✓✓** (seed-robust)  | 1.00/0.97-0.99 ✓ (seed-robust) | ?  | **1.00/0.999+ ✓✓** (seed-robust) |
| 9         | 1.00/0.86-0.97 seed-sens | **1.00/0.9997 ✓✓** NEW  | **1.00/0.996 ✓✓** (seed-robust) | **1.00/1.00 ✓✓** (seed-robust) | **1.00/0.999 ✓** (edge_diff=10000) |
| 10        | 1.00/0.88-0.99 seed-sens (edge_diff=15000) | **1.00/1.00 ✓✓** seed-robust (iter 111/114)  | **1.00/0.98-0.997 ✓✓** (seed-robust) | **1.00/0.999 ✓** edge_diff=15000 (iter 113) | **1.00/0.999+ ✓✓** (seed-robust) |

### Established Principles

- `lr_W=3E-3` optimal for gain=6,7 at rank=10 (iters 9,17,19,21,23,25)
- `lr_W=2E-3` optimal for gain=8,9,10 across all ranks (iters 11, 57, 67, 69, 71)
- `coeff_W_L1=1E-6` critical for low-rank dynamics; 1E-5 degrades
- **edge_diff depends on regime**:
  - rank=15: edge_diff=10000 REQUIRED (edge_diff=15000 crashes rollout, iter 70)
  - rank≥25: edge_diff=10000 optimal (edge_diff=15000 degrades conn_R2, iter 72)
  - gain=8,9 at rank=20: edge_diff=15000 optimal (1.00 vs 0.97, iters 56, 71)
  - rank=10: edge_diff=15000 or 10000 both work
- lower rank reduces DoF and improves learnability (gain=4/5/6: rank=10 >> rank=20)
- **learnability threshold confirmed at gain=5-6**: gain≥6 solves, gain≤5 partial
- **gain=6,7 at rank=10 generalizes across seeds** (42, 99, 137 tested)
- **gain=4/5 plateau is intrinsic** — seed changes don't help (iters 26, 28)
- **lr_W has peaked optimum for low-gain**: gain=4 peaks at 4E-3, gain=5 peaks at 6E-3 (iters 34, 36, 40, 43)
- **rank=15 plateau at 0.85 is intrinsic**: NOT seed-specific. seed=42 best (0.85), seed=137 worse (0.82). edge_diff=10000+lr_W=4E-3 optimal
- **rank=30 solved with edge_diff=10000 but seed-sensitive**: seed=137 works (0.98/1.00), seed=42 fails rollout (0.70) despite good W (iters 44, 45, 48)
- **gain=8 at rank=10 SOLVED**: lr_W=2E-3 + edge_diff=10000 (iter 69, 0.9999/0.99)
- **gain=8,9 at rank=20 is seed-robust**: lr_W=2E-3 + edge_diff=10000 or 15000 both work (iters 57, 59, 67, 71, 74)
- **gain=9 at rank=10 SOLVED**: lr_W=2E-3 + edge_diff=10000 or 15000 both work (iters 73, 77)
- **gain=9 at rank=30 SOLVED**: lr_W=2E-3 + edge_diff=15000 (iter 78, 1.00/0.99)
- **gain=10 at rank=10 seed-sensitive**: lr_W=2E-3 + edge_diff=15000 works for seed=137 (0.99), seed=42 degrades (0.88)
- **gain=9 at rank=30 edge_diff=10000 optimal**: edge_diff=10000 gives 0.999 vs 15000 gives 0.991 (iter 82 vs 78)
- **gain=10 at rank=20 edge_diff=10000 optimal**: edge_diff=10000 gives 0.98 vs 0.96 for edge_diff=15000 (iter 85 vs 84)
- **gain=8 at rank=30 SOLVED**: lr_W=2E-3 + edge_diff=15000 (iter 86, 0.9999/0.9998)
- **gain=10 at rank=30 SOLVED**: lr_W=2E-3 + edge_diff=15000 (iter 87, 0.9998/0.995)
- **gain=9 at rank=10 seed-sensitive**: seed=137 works (0.97), seed=42 degrades rollout (0.86) despite perfect W (iter 88)
- **gain=10 at rank=20 SEED-ROBUST**: edge_diff=10000, seed=42 (0.997) better than seed=137 (0.977) (iter 89)
- **edge_diff=10000 universally optimal at high gain**: gain=8/10 at rank=30 work with edge_diff=10000, often better than 15000 (iters 90, 91)
- **gain=8 at rank=10 rollout issue**: perfect W (1.00) but rollout=0.90-0.92 despite edge_diff=10000/15000 (iters 92, 93). intrinsic instability at low rank + high gain.
- **gain=8/10 at rank=30 SEED-ROBUST**: both seeds (137, 42) work excellently with edge_diff=10000 (iters 94, 95)
- **gain=9 at rank=25 SOLVED**: lr_W=2E-3 + edge_diff=10000 gives 0.994/0.9999 (iter 96)
- **gain=6 at rank=15 SOLVED**: lr_W=3E-3 + edge_diff=10000 gives 1.00/0.993 (iter 98)
- **gain=8 at rank=15 SOLVED**: lr_W=2E-3 + edge_diff=10000 gives 1.00/0.9995 — BREAKS gain=7 rank=15 plateau (iter 99)
- **gain=9 at rank=25 SEED-ROBUST**: seed=42 matches seed=137 (iter 100)
- **gain=8 rank=10 SOLVED with edge_diff=20000**: rollout issue fixed (0.996 vs 0.91 at edge_diff=15000). edge_diff=20000 key for low-rank high-gain rollout stability (iter 101)
- **gain=6,8 at rank=15 SEED-ROBUST**: both seeds (137, 42) work at 0.995+ level (iters 98-99, 102-103)
- **gain=9 at rank=15 SOLVED**: lr_W=2E-3 + edge_diff=10000 gives 1.00/0.9997 (iter 107)
- **rank=15 plateau is gain=7 SPECIFIC**: gain=6/8/9 all solve, only gain=7 stuck at 0.85
- **gain=6 rank=25 UNLEARNABLE**: edge_diff=15000, 20000, lr_W=2E-3, and seed=42 ALL FAILED. V_R2 stuck at 0.67-0.72. intrinsic degeneracy (iters 104-110)
- **gain=10 at rank=15 SOLVED**: lr_W=2E-3 + edge_diff=10000 gives perfect 1.00/0.9998 (iter 111)
- **gain=10 rank=15 SEED-ROBUST**: seed=42 matches seed=137 (1.00 conn_R2, iter 114)
- **gain=10 rank=25 SOLVED with edge_diff=15000**: edge_diff=15000 fixed it (0.9997 vs 0.847 at 10000, iter 113)
- **gain=7 rank=15 plateau INTRINSIC**: lr_W=2E-3 made it WORSE (0.74 vs 0.85). lr_W=3E-3 is optimal for gain=7 (iter 115)
- **gain=6 rank=30 SOLVED**: lr_W=3E-3 + edge_diff=10000 gives 0.9999 conn_R2 (iter 116). rank=30 > rank=25 for gain=6

### Open Questions

- **gain=5 at rank=10**: lr_W=6E-3 is peak (0.89), edge_diff=15000 optimal. plateau confirmed intrinsic at ~0.89
- **rank=15 at gain=7**: plateau (0.85) is GAIN-SPECIFIC. gain=6/8/9/10 all work at rank=15. lr_W=2E-3 might help?
- **gain=6 rank=25 UNLEARNABLE**: edge_diff=20000 and seed=42 BOTH FAILED (iters 109-110). V_R2 stuck at 0.71-0.72. declare unlearnable.
- **gain=6 rank=30 untested**: next gap to explore — may inherit degeneracy from rank=25
- **gain=10 at rank=25**: 0.85 conn_R2 with edge_diff=10000. need edge_diff=15000 to fix.

---

## Previous Block Summary (Block 9)

Block 9 (iters 97-108) major breakthroughs:
- **edge_diff=20000 SOLVES gain=8 rank=10 rollout**: breakthrough! test_R2 jumped from 0.91 to 0.996. seed-robust (seed=42: 0.98)
- **rank=15 fully mapped**: gain=6/8/9 ALL SOLVED (only gain=7 has plateau at 0.85)
- **gain=6 rank=25 INTRINSIC DEGENERACY**: edge_diff=15000 and lr_W=2E-3 both failed. V_R2 stuck at 0.67-0.72

---

## Current Block (Block 10)

### Block Info

Focus: (1) fix gain=6 rank=25 with aggressive edge_diff=20000 or seed change, (2) fill remaining gaps (gain=10 rank=15/25), (3) validate complete landscape

### Hypothesis

Block 10 final landscape completion:
- gain=6 rank=25: edge_diff=20000 or seed=42 may break degeneracy
- gain=10 rank=15/25 should work (high-gain regime easy with lr_W=2E-3)
- universal recipe emerging: lr_W=2E-3 for gain≥8, lr_W=3E-3 for gain=6-7
- edge_diff=10000 works for most, edge_diff=20000 for low-rank high-gain rollout stability

### Iterations This Block

## Iter 113: converged
Node: id=113, parent=112
Config: gain=10, rank=25, seed=137, lr_W=2E-3, edge_diff=15000
Metrics: conn_R2=0.9997, test_R2=0.999, U_R2=0.967, V_R2=0.968
Mutation: edge_diff: 10000 -> 15000
Observation: gain=10 rank=25 SOLVED! edge_diff=15000 fixed it (0.9997 vs 0.847).

## Iter 114: converged
Node: id=114, parent=111
Config: gain=10, rank=15, seed=42, lr_W=2E-3, edge_diff=10000
Metrics: conn_R2=1.000, test_R2=0.9998, U_R2=0.979, V_R2=0.979
Mutation: seed: 137 -> 42
Observation: gain=10 rank=15 SEED-ROBUST! seed=42 matches seed=137.

## Iter 115: failed
Node: id=115, parent=root
Config: gain=7, rank=15, seed=137, lr_W=2E-3, edge_diff=10000
Metrics: conn_R2=0.742, test_R2=0.854, U_R2=0.967, V_R2=0.800
Mutation: lr_W: 3E-3 -> 2E-3
Observation: lr_W=2E-3 DEGRADES gain=7 rank=15! 0.74 vs 0.85. lr_W=3E-3 optimal for gain=7.

## Iter 116: converged
Node: id=116, parent=root
Config: gain=6, rank=30, seed=137, lr_W=3E-3, edge_diff=10000
Metrics: conn_R2=0.9999, test_R2=0.995, U_R2=0.960, V_R2=0.960
Mutation: gain=6, rank=30 (new cell)
Observation: gain=6 rank=30 SOLVED! 0.9999 conn_R2. rank=30 > rank=25 for gain=6.

### Emerging Observations

- **gain=10 rank=25 SOLVED**: edge_diff=15000 was the key (0.9997 vs 0.847). confirms edge_diff=15000 for rank=25 at high gain.
- **gain=10 rank=15 SEED-ROBUST**: both seeds work perfectly. rank=15 column now complete for gain≥8.
- **gain=7 rank=15 plateau CONFIRMED INTRINSIC**: lr_W=2E-3 made it WORSE (0.74 vs 0.85). only gain=7 fails at rank=15.
- **gain=6 rank=30 SOLVED**: 0.9999 conn_R2, symmetric U/V recovery. rank=30 > rank=25 for gain=6.
- **rank=25 column pattern**: gain=6 FAILS (intrinsic), gain=7 unknown, gain=9/10 WORK with edge_diff=15000.
- **landscape nearly complete**: only gain=4/5 rows and gain=7 rank=15 remain problematic.

