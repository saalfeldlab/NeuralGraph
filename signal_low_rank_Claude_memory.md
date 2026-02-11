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
| 10  | 5    | 30   | 3E-3 | 1E-4 | 1E-6 | 10000     | 2         | 8     | 1.000   | 1.000   | **BREAKTHROUGH** gain=5 rank=30 SOLVED (iter 120) |
| 10  | 6    | 30   | 3E-3 | 1E-4 | 1E-6 | 10000     | 2         | 8     | 1.000   | 0.985   | gain=6 rank=30 seed=42 ROBUST (iter 118) |
| 11  | 5    | 30   | 3E-3 | 1E-4 | 1E-6 | 10000     | 2         | 8     | 1.000   | 0.999   | gain=5 rank=30 seed=42 ROBUST (iter 121) |
| 11  | 4    | 30   | 4E-3 | 1E-4 | 1E-6 | 10000     | 2         | 8     | 0.999   | 0.999   | **BREAKTHROUGH** gain=4 rank=30 SOLVED (iter 123) |

### Gain × Rank Landscape Map

| gain\rank | 10 | 15 | 20 | 25 | 30 |
| --------- | -- | -- | -- | -- | -- |
| 4         | 0.78/0.97 partial (lr_W=4E-3 peak) | ?  | 0.20/0.77 FAIL | ?  | **0.999/0.999 ✓✓** BREAKTHROUGH (iter 123) |
| 5         | 0.89/0.83 partial (lr_W=6E-3 best) | ?  | 0.60/0.99 partial | ?  | **1.00/1.00 ✓✓** seed-robust (iters 120,121) |
| 6         | 1.00/0.99 ✓✓ (2 seeds) | **1.00/0.999 ✓✓** (seed-robust)  | 0.97/0.98 ✓ | **0.71/0.999 DEGEN** UNLEARNABLE | **1.00/0.995 ✓✓** seed-robust (iter 116,118)  |
| 7         | 1.00/1.00 ✓✓ (3 seeds) | **0.74/0.85 FAIL** UNLEARNABLE | 0.999/0.993 ✓ | **0.90/1.00 partial** (iter 124) | 1.00/0.93 seed-sens |
| 8         | **1.00/0.98 ✓✓** edge_diff=20000 seed-robust | **1.00/0.995+ ✓✓** (seed-robust)  | 1.00/0.97-0.99 ✓ (seed-robust) | ?  | **1.00/0.999+ ✓✓** (seed-robust) |
| 9         | 1.00/0.86-0.97 seed-sens | **1.00/0.9997 ✓✓** NEW  | **1.00/0.996 ✓✓** (seed-robust) | **1.00/1.00 ✓✓** (seed-robust) | **1.00/0.999 ✓** (edge_diff=10000) |
| 10        | 1.00/0.88-0.99 seed-sens (edge_diff=15000) | **1.00/1.00 ✓✓** seed-robust (iter 111/114)  | **1.00/0.98-0.997 ✓✓** (seed-robust) | **0.90/0.999 partial** seed=42 UNFIX (iter 122) | **1.00/0.999+ ✓✓** (seed-robust) |

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
- **gain=5 rank=30 SOLVED (BREAKTHROUGH)**: lr_W=3E-3 + edge_diff=10000 gives 0.9999 conn_R2 (iter 120). low-gain needs HIGH rank!
- **gain=6 rank=30 SEED-ROBUST**: seed=42 (iter 118) matches seed=137 at 0.9998 conn_R2
- **gain=10 rank=25 SEED-SENSITIVE**: seed=42 (iter 117) degrades to 0.74 despite seed=137 at 0.9997. V_R2 collapse.
- **gain=7 rank=15 plateau CONFIRMED UNBREAKABLE**: lr_W=4E-3 + edge_diff=15000 (iter 119) still only 0.81. all lr_W values give 0.74-0.85.
- **gain=5 rank=30 SEED-ROBUST**: seed=42 (iter 121) matches seed=137 at 0.9999 conn_R2. low-gain + high-rank is universal.
- **gain=4 rank=30 SOLVED (BREAKTHROUGH)**: lr_W=4E-3 + edge_diff=10000 gives 0.9993 conn_R2 (iter 123). pattern confirmed: gain=4/5 need rank=30!
- **gain=10 rank=25 seed=42 UNFIXABLE**: edge_diff=20000 FAILED (iter 122). still 0.896 conn_R2. declare seed=42 at this cell unlearnable.
- **gain=7 rank=25 partial**: 0.898 conn_R2 (iter 124). V_R2=0.886 is limiting. may follow gain=6 rank=25 pattern (degeneracy).

### Open Questions

- **gain=5 at rank=10**: lr_W=6E-3 is peak (0.89), edge_diff=15000 optimal. plateau confirmed intrinsic at ~0.89
- **rank=15 at gain=7 UNBREAKABLE**: declare UNLEARNABLE.
- **gain=6 rank=25 UNLEARNABLE**: declare UNLEARNABLE.
- **gain=10 rank=25 seed=42 UNFIXABLE**: edge_diff=20000 didn't help. declare seed=42 unlearnable at this cell.
- **gain=7 rank=25**: partial (0.90). may need edge_diff=15000 or lr_W tuning.
- **gain=4 rank=30 seed-robustness**: need to test seed=42

---

## Previous Block Summary (Block 10)

Block 10 (iters 109-120) major breakthroughs:
- **gain=5 rank=30 SOLVED (BREAKTHROUGH)**: 0.9999 conn_R2. low-gain needs HIGH rank (30) — rank=10/20 were partial
- **gain=6 rank=30 SEED-ROBUST**: both seeds (137, 42) work at 0.999+ conn_R2
- **gain=10 rank=15/25 mapped**: rank=15 seed-robust, rank=25 seed-sensitive (seed=42 fails at 0.74)
- **gain=7 rank=15 UNBREAKABLE**: lr_W=2E-3, 3E-3, 4E-3 all give 0.74-0.85. declare unlearnable.

---

## Current Block (Block 11)

### Block Info

Focus: (1) validate gain=5 rank=30 seed-robustness, (2) fix gain=10 rank=25 seed=42 with edge_diff=20000, (3) explore gain=4 rank=30, (4) fill gain=7 rank=25 gap

### Hypothesis

Block 11 pattern validation:
- gain=5 rank=30 should be seed-robust (pattern: gain=6 rank=30 is seed-robust)
- edge_diff=20000 may fix gain=10 rank=25 seed=42 (worked for gain=8 rank=10)
- gain=4 rank=30 may work if gain=5 rank=30 works (low-gain needs high rank)
- gain=7 rank=25 is last major untested cell in landscape

### Iterations This Block

## Iter 121: converged
Node: id=121, parent=120
Config: gain=5, rank=30, seed=42, lr_W=3E-3, edge_diff=10000
Metrics: conn_R2=0.9999, test_R2=0.9988, U_R2=0.9596, V_R2=0.9601
Observation: **gain=5 rank=30 SEED-ROBUST confirmed!** seed=42 matches seed=137.

## Iter 122: partial
Node: id=122, parent=117
Config: gain=10, rank=25, seed=42, lr_W=2E-3, edge_diff=20000
Metrics: conn_R2=0.8959, test_R2=0.9994, U_R2=0.9320, V_R2=0.8723
Observation: edge_diff=20000 FAILED to fix gain=10 rank=25 seed=42. V_R2 still degraded. declare seed=42 unlearnable at this cell.

## Iter 123: converged (BREAKTHROUGH)
Node: id=123, parent=120
Config: gain=4, rank=30, seed=137, lr_W=4E-3, edge_diff=10000
Metrics: conn_R2=0.9993, test_R2=0.9992, U_R2=0.9596, V_R2=0.9596
Observation: **BREAKTHROUGH** gain=4 rank=30 SOLVED! pattern confirmed: gain=4/5 need rank=30 (high DoF) while rank=10/20 fail.

## Iter 124: partial
Node: id=124, parent=root
Config: gain=7, rank=25, seed=137, lr_W=3E-3, edge_diff=10000
Metrics: conn_R2=0.8977, test_R2=0.9999, U_R2=0.9549, V_R2=0.8862
Observation: gain=7 rank=25 partial at 0.90 conn_R2. V_R2=0.886 is limiting. similar to gain=6 rank=25 degeneracy pattern.

### Emerging Observations

- **low-gain (4-5) needs rank=30**: both gain=4 and gain=5 at rank=30 SOLVED (0.999+) while rank=10/20 fail. high DoF compensates weak signal.
- **mid-gain (6-7) has rank=25 anomaly**: gain=6 rank=25 UNLEARNABLE, gain=7 rank=25 partial (0.90). mid-gain at rank=25 prone to V collapse.
- **high-gain (8-10) universally easy**: except gain=10 rank=25 seed=42 (UNFIXABLE despite edge_diff=20000).
- **edge_diff scaling**: 10000 default, 15000 for some rank=25 cases, 20000 doesn't help seed-specific failures.

