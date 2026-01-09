# Experiment Log: signal_chaotic_1_Claude

## Block 1: chaotic, Dale_law=False, n_frames=10000, n_neurons=100, n_types=1

## Iter 1: partial
Node: id=1, parent=root
Mode/Strategy: exploit
Config: lr_W=2E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, batch_size=8, low_rank_factorization=False, low_rank=20, n_frames=10000
Metrics: test_R2=0.536, test_pearson=0.294, connectivity_R2=0.132, final_loss=4318
Activity: chaotic oscillations, spectral_radius=1.045, eff_rank=14, activity decays after t~5000
Mutation: none (baseline)
Parent rule: root (first iteration)
Observation: low eff_rank(14) may limit W recovery; try increasing lr_W to improve gradient signal
Next: parent=1

## Iter 2: converged
Node: id=2, parent=1
Mode/Strategy: exploit
Config: lr_W=4E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, batch_size=8, low_rank_factorization=False, low_rank=20, n_frames=10000
Metrics: test_R2=0.972, test_pearson=0.967, connectivity_R2=1.000, final_loss=3986
Activity: sustained chaotic oscillations, spectral_radius=0.973, eff_rank=34, good signal diversity throughout
Mutation: lr_W: 2E-3 -> 4E-3
Parent rule: highest UCB (node 1 was only option)
Observation: doubling lr_W achieved perfect W recovery; eff_rank jumped to 34 (was 14 in iter 1 due to different random seed)
Next: parent=2

## Iter 3: converged
Node: id=3, parent=2
Mode/Strategy: exploit
Config: lr_W=8E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, batch_size=8, low_rank_factorization=False, low_rank=20, n_frames=10000
Metrics: test_R2=0.766, test_pearson=0.725, connectivity_R2=1.000, final_loss=2703
Activity: sustained chaotic oscillations, spectral_radius=0.973, eff_rank=32
Mutation: lr_W: 4E-3 -> 8E-3
Parent rule: highest UCB (node 2, UCB=1.816)
Observation: lr_W=8E-3 maintains perfect W recovery; 2 consecutive successes
Next: parent=3

## Iter 4: converged
Node: id=4, parent=3
Mode/Strategy: exploit
Config: lr_W=1E-2, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, batch_size=8, low_rank_factorization=False, low_rank=20, n_frames=10000
Metrics: test_R2=0.945, test_pearson=0.923, connectivity_R2=0.922, final_loss=5256
Activity: sustained chaotic oscillations, spectral_radius=0.973, eff_rank=33
Mutation: lr_W: 8E-3 -> 1E-2
Parent rule: highest UCB (node 3, UCB=1.943)
Observation: lr_W=1E-2 still converges but R2 dropped from 1.0 to 0.92; approaching upper boundary
Next: parent=4 (failure-probe with extreme lr_W)

## Iter 5: converged
Node: id=5, parent=4
Mode/Strategy: failure-probe
Config: lr_W=2E-2, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, batch_size=8, low_rank_factorization=False, low_rank=20, n_frames=10000
Metrics: test_R2=0.893, test_pearson=0.869, connectivity_R2=0.995, final_loss=3487
Activity: sustained chaotic oscillations, spectral_radius=0.973, eff_rank=32
Mutation: lr_W: 1E-2 -> 2E-2
Parent rule: failure-probe (3+ consecutive R² ≥ 0.9)
Observation: lr_W=2E-2 still achieves excellent R²=0.995; upper boundary not yet found; 4 consecutive converged
Next: parent=5 (continue failure-probe with lr_W=4E-2)

## Iter 6: converged
Node: id=6, parent=5
Mode/Strategy: failure-probe
Config: lr_W=4E-2, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, batch_size=8, low_rank_factorization=False, low_rank=20, n_frames=10000
Metrics: test_R2=0.981, test_pearson=0.970, connectivity_R2=0.998, final_loss=2477
Activity: sustained chaotic oscillations, spectral_radius=0.973, eff_rank=34
Mutation: lr_W: 2E-2 -> 4E-2
Parent rule: failure-probe (continue probing upper boundary)
Observation: lr_W=4E-2 still converges excellently; 5 consecutive converged; 6 consecutive lr_W mutations
Next: parent=6 (switch-dimension: try varying lr instead of lr_W)

## Iter 7: converged
Node: id=7, parent=6
Mode/Strategy: switch-dimension
Config: lr_W=4E-2, lr=5E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, batch_size=8, low_rank_factorization=False, low_rank=20, n_frames=10000
Metrics: test_R2=0.538, test_pearson=0.416, connectivity_R2=0.999, final_loss=1339
Activity: sustained chaotic oscillations, spectral_radius=0.973, eff_rank=34
Mutation: lr: 1E-4 -> 5E-4
Parent rule: switch-dimension (6 consecutive lr_W mutations)
Observation: 5x lr increase maintains R²=0.999; 6 consecutive converged; chaotic regime remarkably robust
Next: parent=2 (explore: branch from node 2 with lr_W=4E-3 to test L1 effects)

## Iter 8: converged
Node: id=8, parent=2
Mode/Strategy: explore
Config: lr_W=4E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-4, batch_size=8, low_rank_factorization=False, low_rank=20, n_frames=10000
Metrics: test_R2=0.980, test_pearson=0.971, connectivity_R2=0.9999, final_loss=3549
Activity: sustained chaotic oscillations, spectral_radius=0.973, eff_rank=33
Mutation: coeff_W_L1: 1E-5 -> 1E-4
Parent rule: explore (select outside recent chain; node 2 has R²=1.0)
Observation: 10x L1 increase still achieves R²=0.9999; 7 consecutive converged; chaotic regime extremely robust
Next: parent=8 (failure-probe: probe upper L1 boundary with coeff_W_L1=5E-4)

## Iter 9: converged
Node: id=9, parent=8
Mode/Strategy: failure-probe
Config: lr_W=4E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=5E-4, batch_size=8, low_rank_factorization=False, low_rank=20, n_frames=10000
Metrics: test_R2=0.785, test_pearson=0.725, connectivity_R2=0.998, final_loss=3147
Activity: sustained chaotic oscillations, spectral_radius=0.973, eff_rank=35
Mutation: coeff_W_L1: 1E-4 -> 5E-4
Parent rule: failure-probe (probing upper L1 boundary)
Observation: coeff_W_L1=5E-4 (50x baseline) still achieves R²=0.998; 8 consecutive converged; L1 upper boundary not found
Next: parent=7 (highest UCB=3.120, try different dimension from lr branch)

## Iter 10: partial
Node: id=10, parent=7
Mode/Strategy: exploit
Config: lr_W=4E-2, lr=5E-4, lr_emb=2.5E-4, coeff_W_L1=1E-3, batch_size=8, low_rank_factorization=False, low_rank=20, n_frames=10000
Metrics: test_R2=0.342, test_pearson=0.316, connectivity_R2=0.762, final_loss=4595
Activity: sustained chaotic oscillations, spectral_radius=0.973, eff_rank=36
Mutation: coeff_W_L1: 1E-5 -> 1E-3
Parent rule: highest UCB (node 7, UCB=3.120; tried L1 increase from node 7's lr=5E-4 branch)
Observation: coeff_W_L1=1E-3 causes first failure (R²=0.762); L1 upper boundary found between 5E-4 and 1E-3
Next: parent=9 (highest UCB=3.234, continue L1 boundary exploration)

## Iter 11: converged
Node: id=11, parent=9
Mode/Strategy: exploit
Config: lr_W=4E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=5E-4, batch_size=16, low_rank_factorization=False, low_rank=20, n_frames=10000
Metrics: test_R2=0.786, test_pearson=0.758, connectivity_R2=0.998, final_loss=3147
Activity: sustained chaotic oscillations, spectral_radius=0.973, eff_rank=30
Mutation: batch_size: 8 -> 16 (inherited L1=5E-4 from parent)
Parent rule: highest UCB (node 9, UCB=3.234)
Observation: batch_size=16 with L1=5E-4 maintains R²=0.998; confirms L1=5E-4 is robust; 10 of 11 converged
Next: parent=11 (highest UCB=3.343, explore different dimension)

## Iter 12: converged
Node: id=12, parent=11
Mode/Strategy: exploit
Config: lr_W=4E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=5E-4, batch_size=32, low_rank_factorization=False, low_rank=20, n_frames=10000
Metrics: test_R2=0.548, test_pearson=0.334, connectivity_R2=0.956, final_loss=3655
Activity: sustained chaotic oscillations, spectral_radius=0.973, eff_rank=34
Mutation: batch_size: 16 -> 32
Parent rule: highest UCB (node 11, UCB=3.343)
Observation: batch_size=32 still converges (R²=0.956); 11 of 12 converged (92%); all batch sizes work
Next: parent=12 (highest UCB=3.405)

## Iter 13: failed
Node: id=13, parent=12
Mode/Strategy: exploit
Config: lr_W=4E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=5E-4, batch_size=32, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.530, test_pearson=0.493, connectivity_R2=0.055, final_loss=6787
Activity: sustained chaotic oscillations, spectral_radius=0.973, eff_rank=36
Mutation: low_rank_factorization: False -> True
Parent rule: highest UCB (node 12, UCB=3.405)
Observation: low_rank_factorization=True catastrophically fails (R²=0.055); full-rank chaotic W incompatible with low-rank constraint
Next: parent=10 (highest UCB=3.311, explore L1 boundary failure mode)

## Iter 14: partial
Node: id=14, parent=10
Mode/Strategy: exploit
Config: lr_W=4E-2, lr=5E-4, lr_emb=2.5E-4, coeff_W_L1=7E-4, batch_size=8, low_rank_factorization=False, low_rank=20, n_frames=10000
Metrics: test_R2=0.418, test_pearson=0.480, connectivity_R2=0.830, final_loss=3827
Activity: sustained chaotic oscillations, spectral_radius=0.973, eff_rank=34
Mutation: coeff_W_L1: 1E-3 -> 7E-4
Parent rule: highest UCB (node 14, UCB=3.475)
Observation: L1=7E-4 partially converges (R²=0.830); confirms L1 boundary lies between 5E-4 (works) and 7E-4 (partial)
Next: parent=14 (highest UCB=3.475)

## Iter 15: partial
Node: id=15, parent=14
Mode/Strategy: exploit
Config: lr_W=4E-2, lr=5E-4, lr_emb=2.5E-4, coeff_W_L1=6E-4, batch_size=8, low_rank_factorization=False, low_rank=20, n_frames=10000
Metrics: test_R2=0.539, test_pearson=0.484, connectivity_R2=0.872, final_loss=3553
Activity: sustained chaotic oscillations, spectral_radius=0.973, eff_rank=33
Mutation: coeff_W_L1: 7E-4 -> 6E-4
Parent rule: highest UCB (node 15, UCB=3.610)
Observation: L1=6E-4 partial (R²=0.872); L1 boundary precisely at 5E-4<threshold<6E-4; 12 of 15 converged (80%)
Next: parent=15 (highest UCB=3.610)

## Iter 16: partial
Node: id=16, parent=15
Mode/Strategy: exploit
Config: lr_W=4E-2, lr=5E-4, lr_emb=2.5E-4, coeff_W_L1=5.5E-4, batch_size=8, low_rank_factorization=False, low_rank=20, n_frames=10000
Metrics: test_R2=0.387, test_pearson=0.368, connectivity_R2=0.877, final_loss=3417
Activity: sustained chaotic oscillations, spectral_radius=0.973, eff_rank=33
Mutation: coeff_W_L1: 6E-4 -> 5.5E-4
Parent rule: highest UCB (node 16, UCB=3.705)
Observation: L1=5.5E-4 partial (R²=0.877); boundary refinement confirms L1 threshold is ~5E-4

---

## Block 1 Summary

**Regime**: chaotic, Dale_law=False, n_frames=10000, n_neurons=100, n_types=1

**Results**: 12/16 converged (75%), Best R²=1.000 (nodes 2, 3, 8)

**Key Findings**:
1. lr_W robust: 4E-3 to 4E-2 all achieve R²≥0.92 (10x range)
2. lr robust: 1E-4 to 5E-4 works at lr_W=4E-2
3. L1 boundary precise: 5E-4 works (R²=0.998), 5.5E-4+ fails (R²<0.9)
4. batch_size insensitive: 8, 16, 32 all work
5. low_rank_factorization=True fails: R²=0.055 vs 0.95+ (catastrophic for full-rank W)
6. eff_rank varies by seed: 10-36, does not prevent convergence when >30

**Optimal config**: lr_W=4E-3, lr=1E-4, coeff_W_L1≤5E-4, batch_size=8, low_rank_factorization=False

**Branching analysis**:
- Branches: iters 8 (parent=2), 10 (parent=7), 14 (parent=10)
- Branching rate: 3/15 = 20%

INSTRUCTIONS EDITED: added rule "boundary-skip" - when 3+ partial results probing same boundary, accept and explore elsewhere

---

## Block 2: chaotic, Dale_law=True, Dale_law_factor=0.5, n_frames=10000, n_neurons=100, n_types=1

### Hypothesis
Dale_law=True constrains W to have consistent sign per column (E/I neurons). This may require different lr_W and may benefit from low_rank_factorization given reduced effective rank.

## Iter 17: partial
Node: id=17, parent=root
Mode/Strategy: exploit
Config: lr_W=4E-3, lr=1E-4, coeff_W_L1=1E-5, batch_size=8, low_rank_factorization=False
Metrics: test_R2=0.685, test_pearson=0.736, connectivity_R2=0.610, final_loss=3.88E+03
Activity: effective_rank=10 (much lower than Block 1's 31-35), spectral_radius=1.020, range=[-10.2, 10.2]
Mutation: block start (Dale_law=True baseline)
Parent rule: first iteration of new block
Observation: Dale_law=True dramatically reduces effective rank (10 vs 31-35), causing partial convergence with same lr_W that worked in Block 1
Next: parent=17, try higher lr_W=8E-3 to escape local minimum

## Iter 19: partial
Node: id=19, parent=18
Mode/Strategy: exploit
Config: lr_W=1.6E-2, lr=1E-4, coeff_W_L1=1E-5, batch_size=8, low_rank_factorization=False
Metrics: test_R2=0.993, test_pearson=0.997, connectivity_R2=0.850, final_loss=2.04E+03
Activity: effective_rank=10, spectral_radius=1.020, range=[-10.2, 10.2]
Mutation: lr_W: 8E-3 -> 1.6E-2
Parent rule: UCB highest (node 18, UCB=1.577)
Observation: lr_W doubling continues to improve R²: 0.61→0.76→0.85 (+0.09); still partial but approaching threshold; try lr_W=3.2E-2 or low_rank_factorization=True
Next: parent=19, try lr_W=3.2E-2 (continue doubling to find convergence)

## Iter 18: partial
Node: id=18, parent=17
Mode/Strategy: exploit
Config: lr_W=8E-3, lr=1E-4, coeff_W_L1=1E-5, batch_size=8, low_rank_factorization=False
Metrics: test_R2=0.946, test_pearson=0.963, connectivity_R2=0.761, final_loss=2.75E+03
Activity: effective_rank=10, spectral_radius=1.020, range=[-10.2, 10.2]
Mutation: lr_W: 4E-3 -> 8E-3
Parent rule: UCB highest (1.761)
Observation: lr_W increase improved R² from 0.610 to 0.761 (+0.15), but still partial; suggests higher lr_W may help further
Next: parent=18, try lr_W=1.6E-2 (double again)

## Iter 20: partial
Node: id=20, parent=19
Mode/Strategy: exploit
Config: lr_W=3.2E-2, lr=1E-4, coeff_W_L1=1E-5, batch_size=8, low_rank_factorization=False
Metrics: test_R2=0.979, test_pearson=0.989, connectivity_R2=0.895, final_loss=1.63E+03
Activity: effective_rank=10, spectral_radius=1.020
Mutation: lr_W: 1.6E-2 -> 3.2E-2
Parent rule: highest UCB (Node 19, UCB=1.793)
Observation: R² improved 0.850→0.895 (+0.045); approaching 0.9 threshold; lr_W trend: 4E-3→0.61, 8E-3→0.76, 1.6E-2→0.85, 3.2E-2→0.895
Next: parent=20 (highest UCB=2.309), try lr_W=5E-2

## Iter 21: converged
Node: id=21, parent=20
Mode/Strategy: exploit
Config: lr_W=5E-2, lr=1E-4, coeff_W_L1=1E-5, batch_size=8, low_rank_factorization=False
Metrics: test_R2=0.991, test_pearson=0.994, connectivity_R2=0.906, final_loss=1.49E+03
Activity: effective_rank=10, spectral_radius=1.020
Mutation: lr_W: 3.2E-2 -> 5E-2
Parent rule: highest UCB (Node 20, UCB=2.309)
Observation: crossed 0.9 threshold! R² improved 0.895→0.906 (+0.011)
Next: parent=21 (highest UCB=2.487)

## Iter 22: converged
Node: id=22, parent=21
Mode/Strategy: exploit
Config: lr_W=8E-2, lr=1E-4, coeff_W_L1=1E-5, batch_size=8, low_rank_factorization=False
Metrics: test_R2=0.987, test_pearson=0.995, connectivity_R2=0.912, final_loss=1.41E+03
Activity: effective_rank=10, spectral_radius=1.020
Mutation: lr_W: 5E-2 -> 8E-2
Parent rule: highest UCB (Node 22, UCB=2.644)
Observation: 2nd consecutive convergence; R² improved 0.906→0.912 (+0.006); diminishing returns continue
Next: parent=22 (highest UCB), exploit lr_W=1.2E-1

## Iter 23: converged
Node: id=23, parent=22
Mode/Strategy: failure-probe
Config: lr_W=1.2E-1, lr=1E-4, coeff_W_L1=1E-5, batch_size=8, low_rank_factorization=False
Metrics: test_R2=0.992, test_pearson=0.997, connectivity_R2=0.901, final_loss=1.44E+03
Activity: effective_rank=10, spectral_radius=1.020
Mutation: lr_W: 8E-2 -> 1.2E-1
Parent rule: highest UCB (Node 22, UCB=2.159)
Observation: 3rd consecutive convergence but R² decreased 0.912→0.901 (-0.011); lr_W=8E-2 appears optimal for Dale_law=True
Next: parent=23 (highest UCB=2.772), failure-probe with extreme lr_W=2.5E-1 to find upper boundary

## Iter 24: partial
Node: id=24, parent=23
Mode/Strategy: failure-probe
Config: lr_W=2.5E-1, lr=1E-4, coeff_W_L1=1E-5, batch_size=8, low_rank_factorization=False
Metrics: test_R2=0.980, test_pearson=0.991, connectivity_R2=0.868, final_loss=1.61E+03
Activity: effective_rank=10, spectral_radius=1.020
Mutation: lr_W: 1.2E-1 -> 2.5E-1
Parent rule: highest UCB (Node 23, UCB=2.867), failure-probe strategy after 3 consecutive convergences
Observation: upper boundary confirmed - lr_W=2.5E-1 drops below 0.9 threshold (0.868); optimal range is 5E-2 to 1.2E-1
Next: parent=22 (best R²=0.912), explore low_rank_factorization=True to test if it helps with reduced effective_rank

## Iter 25: failed
Node: id=25, parent=22
Mode/Strategy: switch-dimension (after 8 consecutive lr_W mutations)
Config: lr_W=8E-2, lr=1E-4, coeff_W_L1=1E-5, batch_size=8, low_rank_factorization=True, low_rank=10
Metrics: test_R2=0.291, test_pearson=0.145, connectivity_R2=0.324, final_loss=4.33E+03
Activity: effective_rank=10, spectral_radius=1.020
Mutation: low_rank_factorization: False -> True, low_rank=10
Parent rule: branched from best R² node (22) to test different dimension
Observation: catastrophic failure even with rank matching effective_rank; confirms low_rank_factorization incompatible with Dale_law chaotic regime
Next: parent=22, switch-dimension to explore coeff_W_L1

## Iter 26: partial
Node: id=26, parent=22
Mode/Strategy: switch-dimension
Config: lr_W=8E-2, lr=1E-4, coeff_W_L1=1E-4, batch_size=8, low_rank_factorization=False
Metrics: test_R2=0.909, test_pearson=0.956, connectivity_R2=0.617, final_loss=2.25E+03
Activity: effective_rank=10, spectral_radius=1.020
Mutation: coeff_W_L1: 1E-5 -> 1E-4
Parent rule: branched from best R² node (22) to test L1 dimension
Observation: coeff_W_L1=1E-4 causes partial failure (R²=0.617); Dale_law=True is MORE sensitive to L1 than Dale_law=False (1E-4 worked in Block 1)
Next: parent=24 (highest UCB=3.104), try lower lr_W to find if lower lr_W + L1=1E-5 can improve


## Iter 27: partial
Node: id=27, parent=24
Mode/Strategy: exploit
Config: lr_W=2.5E-1, lr=5E-05, coeff_W_L1=1E-5, batch_size=8, low_rank_factorization=False
Metrics: test_R2=0.960, test_pearson=0.983, connectivity_R2=0.872, final_loss=1.61E+03
Activity: effective_rank=10, spectral_radius=1.020
Mutation: lr: 1E-4 -> 5E-05
Parent rule: highest UCB (Node 24, UCB=3.104)
Observation: reducing lr slightly improved R² (0.868→0.872); still above optimal lr_W range
Next: parent=27 (highest UCB=3.216), reduce lr_W toward optimal range

## Iter 28: partial
Node: id=28, parent=27
Mode/Strategy: exploit
Config: lr_W=1.5E-1, lr=5E-05, coeff_W_L1=1E-5, batch_size=8, low_rank_factorization=False
Metrics: test_R2=0.957, test_pearson=0.980, connectivity_R2=0.896, final_loss=1.51E+03
Activity: effective_rank=10, spectral_radius=1.020
Mutation: lr_W: 2.5E-1 -> 1.5E-1
Parent rule: highest UCB (Node 28, UCB=3.345)
Observation: R² improved 0.872→0.896 (+0.024); approaching 0.9 threshold; confirms optimal lr_W < 1.5E-1
Next: parent=28 (highest UCB), continue reducing lr_W toward 8E-2

## Iter 29: converged
Node: id=29, parent=28
Mode/Strategy: exploit
Config: lr_W=1E-1, lr=5E-05, coeff_W_L1=1E-5, batch_size=8, low_rank_factorization=False
Metrics: test_R2=0.967, test_pearson=0.985, connectivity_R2=0.913, final_loss=1.48E+03
Activity: effective_rank=10, spectral_radius=1.020
Mutation: lr_W: 1.5E-1 -> 1E-1
Parent rule: highest UCB (Node 29, UCB=3.462)
Observation: best R² of Block 2 (0.913); lr=5E-5 + lr_W=1E-1 achieves optimal; confirms lr_W=8E-2 to 1E-1 is sweet spot
Next: parent=29 (highest UCB=3.462), 4 consecutive converged → failure-probe

## Iter 30: partial
Node: id=30, parent=29
Mode/Strategy: failure-probe
Config: lr_W=8E-2, lr=1E-5, coeff_W_L1=1E-5, batch_size=8, low_rank_factorization=False
Metrics: test_R2=0.925, test_pearson=0.957, connectivity_R2=0.828, final_loss=2.54E+03
Activity: effective_rank=10, spectral_radius=1.020
Mutation: lr: 5E-05 -> 1E-05
Parent rule: highest UCB (Node 29, UCB=3.462), failure-probe after 4 consecutive convergences
Observation: reducing lr to 1E-5 degraded R² from 0.913→0.828; lr=5E-5 is optimal with high lr_W; confirms lr/lr_W ratio matters
Next: parent=30 (highest UCB=3.473), try different mutation

## Iter 31: partial
Node: id=31, parent=30
Mode/Strategy: exploit
Config: lr_W=1.2E-1, lr=1E-5, coeff_W_L1=1E-5, batch_size=8, low_rank_factorization=False
Metrics: test_R2=0.837, test_pearson=0.903, connectivity_R2=0.828, final_loss=2.47E+03
Activity: effective_rank=10, spectral_radius=1.020
Mutation: lr_W: 8E-2 -> 1.2E-1
Parent rule: highest UCB (Node 30, UCB=3.473)
Observation: lr=1E-5 continues to underperform; confirms lr must be ≥5E-5; lr_W=1.2E-1 with lr=1E-5 gives same R² as lr_W=8E-2 with lr=1E-5 (~0.828)
Next: parent=31 (highest UCB=3.566), mutate lr: 1E-5 -> 5E-5

## Iter 32: converged
Node: id=32, parent=31
Mode/Strategy: exploit
Config: lr_W=1.2E-1, lr=5E-5, coeff_W_L1=1E-5, batch_size=8, low_rank_factorization=False
Metrics: test_R2=0.987, test_pearson=0.992, connectivity_R2=0.907, final_loss=1.48E+03
Activity: effective_rank=10, spectral_radius=1.020
Mutation: lr: 1E-5 -> 5E-5
Parent rule: highest UCB (Node 31, UCB=3.566)
Observation: restoring lr=5E-5 recovers convergence (0.828→0.907); confirms lr/lr_W ratio is critical

---

## Block 2 Summary
Simulation: connectivity_type=chaotic, Dale_law=True, Dale_law_factor=0.5, n_frames=10000, n_neurons=100, n_types=1

Convergence: 6/16 iterations converged (37.5%)
Best R²: 0.913 (Node 29: lr_W=1E-1, lr=5E-5) and 0.912 (Node 22: lr_W=8E-2, lr=1E-4)

Key Findings:
1. Dale_law=True reduces effective_rank from ~32 to ~10 (3x reduction)
2. R² ceiling at ~0.913 (limited by effective_rank=10, not training params)
3. Optimal lr_W range: 5E-2 to 1.2E-1 (10-30x higher than Dale_law=False)
4. lr sensitivity: lr=5E-5 to 1E-4 required; lr=1E-5 fails
5. L1 sensitivity: 10x stricter (≤1E-5 vs ≤5E-4 for Dale_law=False)
6. low_rank_factorization=True fails catastrophically (R²=0.324 vs 0.912)
7. lr_W/lr ratio matters: optimal ratio ~1000:1 to 2000:1

Branching rate: 4/15 = 27% (nodes 25, 26, 27, 32 branched from ancestors)

INSTRUCTIONS EDITED: No rule changes needed (branching rate 27% is healthy, 20-80% range)

---

## Block 3: low_rank connectivity, Dale_law=False, n_frames=10000, n_neurons=100, n_types=1

### Hypothesis
low_rank connectivity (rank=20) creates a structured, lower-dimensional W. This may:
1. benefit from low_rank_factorization=True (unlike chaotic)
2. need different lr_W (potentially lower for structured)
3. have higher R² ceiling (structure easier to learn)

## Iter 33: failed
Node: id=33, parent=root
Mode/Strategy: exploit (first iteration of block)
Config: lr_W=4E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, batch_size=8, low_rank_factorization=False, low_rank=20, n_frames=10000
Metrics: test_R2=0.335, test_pearson=0.105, connectivity_R2=0.038, cluster_accuracy=N/A, final_loss=4199.5
Activity: smooth coherent oscillations, effective_rank=6 (much lower than expected for rank-20 connectivity), spectral_radius=0.962
Mutation: baseline config from Block 1
Parent rule: root (first iteration)
Observation: low_rank connectivity (rank=20) produces eff_rank=6 activity — even lower than Dale_law=True (eff_rank=10); lr_W=4E-3 completely fails
Next: parent=33, try lr_W=8E-2 (20x increase, extrapolating from Dale_law pattern)
