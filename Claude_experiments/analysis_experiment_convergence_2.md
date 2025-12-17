# Experiment Log: signal_Claude

## Iter 1: converged
Node: id=4, parent=3, V=1, N_total=4
Config: lr_W=2.0E-3, lr=1.0E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.7742, test_pearson=0.6996, connectivity_R2=0.9817, final_loss=6.35e+02
RankScore: 1.00 (rank 4/4)
Observation: excellent convergence with connectivity_R2=0.9817, strong W recovery
Change: coeff_W_L1: 5.0E-6 -> 1.0E-5 (from previous iteration 3)
Next: validation run 1/3 - keep config unchanged to confirm robustness

## Iter 2: converged (validation 1/3)
Node: id=4, parent=3, V=2, N_total=4
Config: lr_W=2.0E-3, lr=1.0E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9995, test_pearson=0.9990, connectivity_R2=1.0000, final_loss=1.32e+02
RankScore: 1.00 (rank 4/4)
Observation: excellent convergence, connectivity_R2=1.0 (perfect), validation 1/3 passed
Change: none (validation run)
Next: validation run 2/3 - keep config unchanged

## Iter 3: converged (validation 2/3)
Node: id=4, parent=3, V=3, N_total=4
Config: lr_W=2.0E-3, lr=1.0E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9918, test_pearson=0.9897, connectivity_R2=1.0000, final_loss=1.26e+02
RankScore: 1.00 (rank 4/4)
Observation: excellent convergence, connectivity_R2=1.0 (perfect), validation 2/3 passed
Change: none (validation run)
Next: validation run 3/3 - keep config unchanged to complete robustness check

## Iter 4: converged (validation 3/3) ✓ ROBUST
Node: id=4, parent=3, V=4, N_total=4
Config: lr_W=2.0E-3, lr=1.0E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9673, test_pearson=0.9717, connectivity_R2=1.0000, final_loss=1.07e+02
RankScore: 1.00 (rank 4/4)
Observation: validation 3/3 passed - config confirmed ROBUST (3/3 converged with connectivity_R2=1.0)
Change: none (validation run)
Next: explore alternative parameter space - test coeff_W_L1=0 to assess L1 regularization importance

## Iter 5: converged
Node: id=5, parent=4, V=1, N_total=5
Config: lr_W=2.0E-3, lr=1.0E-4, coeff_W_L1=0, batch_size=8
Metrics: test_R2=0.9547, test_pearson=0.9546, connectivity_R2=1.0000, final_loss=1.15e+02
RankScore: 1.00 (rank 5/5)
Observation: perfect connectivity_R2=1.0 even without L1 regularization - L1 not critical for convergence
Change: coeff_W_L1: 1.0E-5 -> 0
Next: test higher L1 regularization (coeff_W_L1=1.0E-4) to map upper boundary of working range

## Iter 6: converged
Node: id=6, parent=4, V=1, N_total=6
Config: lr_W=2.0E-3, lr=1.0E-4, coeff_W_L1=1.0E-4, batch_size=8
Metrics: test_R2=0.9723, test_pearson=0.9728, connectivity_R2=1.0000, final_loss=1.12e+02
RankScore: 1.00 (rank 6/6)
Observation: perfect connectivity_R2=1.0 with high L1 - coeff_W_L1 range [0, 1E-4] all work well
Change: coeff_W_L1: 0 -> 1.0E-4
Next: explore lr_W dimension - test lr_W=5.0E-3 (higher) from robust node 4 config

## Iter 7: converged
Node: id=7, parent=4, V=1, N_total=7
Config: lr_W=5.0E-3, lr=1.0E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9755, test_pearson=0.9671, connectivity_R2=1.0000, final_loss=1.65e+02
RankScore: 1.00 (rank 7/7)
Observation: perfect connectivity_R2=1.0 with higher lr_W=5E-3 - lr_W range [2E-3, 5E-3] works well
Change: lr_W: 2.0E-3 -> 5.0E-3
Next: explore lr dimension - test lr=5.0E-4 (higher) to find boundaries of learning rate space

## Iter 8: converged
Node: id=8, parent=7, V=1, N_total=8
Config: lr_W=5.0E-3, lr=5.0E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9643, test_pearson=0.9600, connectivity_R2=0.9999, final_loss=2.91e+02
RankScore: 1.00 (rank 8/8)
Observation: excellent connectivity_R2=0.9999 with lr=5E-4 (5x higher than robust config) - lr range expanded
Change: lr: 1.0E-4 -> 5.0E-4
Next: explore lr_W upper boundary - test lr_W=1.0E-2 to find upper limit of lr_W

## Iter 9: converged
Node: id=9, parent=8, V=1, N_total=9
Config: lr_W=1.0E-2, lr=1.0E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9102, test_pearson=0.9240, connectivity_R2=0.9999, final_loss=2.55e+02
RankScore: 1.00 (rank 9/9)
Observation: excellent connectivity_R2=0.9999 with lr_W=1E-2 (highest tested) - lr_W range [2E-3, 1E-2] all converge
Change: lr_W: 5.0E-3 -> 1.0E-2
Next: explore lr lower boundary - test lr=1.0E-4 with lr_W=1.0E-2 (lr was already 1E-4, so test batch_size=16 instead to map another dimension)

## Iter 10: converged
Node: id=10, parent=9, V=1, N_total=10
Config: lr_W=1.0E-2, lr=1.0E-4, coeff_W_L1=1.0E-5, batch_size=16
Metrics: test_R2=0.9102, test_pearson=0.9240, connectivity_R2=0.9999, final_loss=1.28e+02
RankScore: 1.00 (rank 10/10)
Observation: excellent connectivity_R2=0.9999 with batch_size=16 - batch_size range [8, 16] works well, lower final_loss than batch_size=8
Change: batch_size: 8 -> 16
Next: test batch_size=4 to find lower boundary of batch_size range

## Iter 11: converged
Node: id=11, parent=10, V=1, N_total=11
Config: lr_W=1.0E-2, lr=1.0E-4, coeff_W_L1=1.0E-5, batch_size=4
Metrics: test_R2=0.9833, test_pearson=0.9811, connectivity_R2=0.9997, final_loss=4.74e+02
RankScore: 1.00 (rank 11/11)
Observation: excellent connectivity_R2=0.9997 with batch_size=4 - batch_size range [4, 16] all converge; higher final_loss than batch_size=8,16
Change: batch_size: 16 -> 4
Next: test batch_size=1 (extreme lower boundary) to find minimum viable batch_size

## Iter 12: converged
Node: id=12, parent=11, V=1, N_total=12
Config: lr_W=1.0E-2, lr=1.0E-4, coeff_W_L1=1.0E-5, batch_size=1
Metrics: test_R2=0.8268, test_pearson=0.8552, connectivity_R2=0.9996, final_loss=1.27e+03
RankScore: 1.00 (rank 12/12)
Observation: excellent connectivity_R2=0.9996 with batch_size=1 (extreme lower) - full batch_size range [1, 16] all converge; final_loss highest at batch_size=1
Change: batch_size: 4 -> 1
Next: test lr_W lower boundary - try lr_W=1.0E-3 from robust node 4 (explore unexplored lr_W lower boundary)

## Iter 13: converged
Node: id=13, parent=4, V=1, N_total=13
Config: lr_W=1.0E-3, lr=1.0E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.8268, test_pearson=0.8552, connectivity_R2=0.9996, final_loss=1.09e+03
RankScore: 1.00 (rank 13/13)
Observation: excellent connectivity_R2=0.9996 with lr_W=1E-3 (lower boundary) - lr_W range [1E-3, 1E-2] all converge
Change: lr_W: 2.0E-3 -> 1.0E-3 (from robust node 4 baseline)
Next: test lr_W=5.0E-4 (extreme lower) to find lr_W lower boundary failure point

## Iter 14: converged
Node: id=14, parent=13, V=1, N_total=14
Config: lr_W=5.0E-4, lr=1.0E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.8268, test_pearson=0.8552, connectivity_R2=0.9996, final_loss=3.88e+03
RankScore: 1.00 (rank 14/14)
Observation: excellent connectivity_R2=0.9996 with lr_W=5E-4 (extreme lower) - lr_W range [5E-4, 1E-2] all converge; final_loss significantly higher (3.88e+03 vs 1.09e+03 at lr_W=1E-3)
Change: lr_W: 1.0E-3 -> 5.0E-4
Next: test lr upper boundary - try lr=1.0E-3 from robust node 4 to find lr upper limit

## Iter 15: converged
Node: id=15, parent=4, V=1, N_total=15
Config: lr_W=2.0E-3, lr=1.0E-3, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.8268, test_pearson=0.8552, connectivity_R2=0.9996, final_loss=2.57e+02
RankScore: 1.00 (rank 15/15)
Observation: excellent connectivity_R2=0.9996 with lr=1E-3 (10x higher than baseline) - lr range [1E-4, 1E-3] all converge; lr_W restored to 2E-3 from robust node 4
Change: lr: 1.0E-4 -> 1.0E-3 (from robust node 4 baseline)
Next: test lr_W=1.0E-4 (extreme lower) from robust node 4 to find lr_W failure point

## Iter 16: converged
Node: id=16, parent=4, V=1, N_total=16
Config: lr_W=1.0E-4, lr=1.0E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.8268, test_pearson=0.8552, connectivity_R2=0.9996, final_loss=8.75e+03
Observation: excellent connectivity_R2=0.9996 with lr_W=1E-4 (extreme lower, 20x lower than baseline) - lr_W range [1E-4, 1E-2] all converge; final_loss very high (8.75e+03) indicating slow convergence
Change: lr_W: 2.0E-3 -> 1.0E-4 (from robust node 4 baseline)
Next: test lr=2.0E-3 (upper boundary) from high-UCB node 15 (lr=1E-3 worked well) to find lr upper failure point

## Iter 17: converged
Node: id=17, parent=15, V=1, N_total=17
Config: lr_W=2.0E-3, lr=2.0E-3, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.8268, test_pearson=0.8552, connectivity_R2=0.9996, final_loss=2.63e+02
Observation: excellent connectivity_R2=0.9996 with lr=2E-3 (20x higher than baseline) - lr range expanded to [1E-4, 2E-3] all converge; final_loss reasonable (2.63e+02)
Change: lr: 1.0E-3 -> 2.0E-3 (from node 15)
Next: test lr=5.0E-3 (extreme upper) from node 17 to find lr upper failure point

## Iter 18: converged
Node: id=18, parent=17, V=1, N_total=18
Config: lr_W=2.0E-3, lr=5.0E-3, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.8268, test_pearson=0.8552, connectivity_R2=0.9996, final_loss=3.67e+02
Observation: excellent connectivity_R2=0.9996 with lr=5E-3 (50x higher than baseline) - lr range expanded to [1E-4, 5E-3] all converge; final_loss slightly higher (3.67e+02)
Change: lr: 2.0E-3 -> 5.0E-3 (from node 17)
Next: test lr=1.0E-2 (extreme upper, same as lr_W) from node 18 to find lr upper failure point

## Iter 19: converged
Node: id=19, parent=18, V=1, N_total=19
Config: lr_W=2.0E-3, lr=1.0E-2, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.8268, test_pearson=0.8552, connectivity_R2=0.9996, final_loss=2.71e+02
Observation: excellent connectivity_R2=0.9996 with lr=1E-2 (100x higher than baseline) - lr range expanded to [1E-4, 1E-2] all converge; final_loss lower than lr=5E-3 (2.71e+02 vs 3.67e+02)
Change: lr: 5.0E-3 -> 1.0E-2 (from node 18)
Next: test lr=2.0E-2 (extreme upper) from node 19 to find lr upper failure point

## Iter 20: converged
Node: id=20, parent=19, V=1, N_total=20
Config: lr_W=2.0E-3, lr=2.0E-2, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.8268, test_pearson=0.8552, connectivity_R2=0.9996, final_loss=3.12e+03
Observation: connectivity_R2=0.9996 with lr=2E-2 (200x higher than baseline) - lr range expanded to [1E-4, 2E-2] all converge; final_loss increased significantly (3.12e+03 vs 2.71e+02 at lr=1E-2) indicating possible upper boundary
Change: lr: 1.0E-2 -> 2.0E-2 (from node 19)
Next: meta-analysis at iteration 20, then explore from high-UCB node (node 5, 6, 12, 14, or 16) to test distinct regions

## Iter 21: converged
Node: id=21, parent=20, V=1, N_total=21
Config: lr_W=2.0E-3, lr=5.0E-2, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.8268, test_pearson=0.8552, connectivity_R2=0.9996, final_loss=1.22e+04
Observation: connectivity_R2=0.9996 with lr=5E-2 (500x higher than baseline) - lr range expanded to [1E-4, 5E-2] all converge; final_loss very high (1.22e+04) indicating degradation at extreme lr
Change: lr: 2.0E-2 -> 5.0E-2 (from node 20)
Next: test lr=1.0E-1 (extreme upper) from node 21 to find lr failure point, or pivot to test lr_W lower boundary lr_W=5E-5

## Iter 22: converged
Node: id=22, parent=21, V=1, N_total=22
Config: lr_W=2.0E-3, lr=1.0E-1, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.8268, test_pearson=0.8552, connectivity_R2=0.9996, final_loss=1.13e+04
Observation: connectivity_R2=0.9996 with lr=1E-1 (1000x higher than baseline) - lr range expanded to [1E-4, 1E-1] all converge; final_loss=1.13e+04 similar to lr=5E-2 (1.22e+04)
Change: lr: 5.0E-2 -> 1.0E-1 (from node 21)
Next: test lr=2.0E-1 (extreme upper) from node 22 to find lr failure point

## Iter 23: converged
Node: id=23, parent=22, V=1, N_total=23
Config: lr_W=2.0E-3, lr=2.0E-1, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.8268, test_pearson=0.8552, connectivity_R2=0.9996, final_loss=1.13e+04
Observation: connectivity_R2=0.9996 with lr=2E-1 (2000x higher than baseline) - lr range expanded to [1E-4, 2E-1] all converge; final_loss=1.13e+04 same as lr=1E-1 (plateau reached)
Change: lr: 1.0E-1 -> 2.0E-1 (from node 22)
Next: test lr=5.0E-1 (extreme upper) from node 23 to find lr failure point - plateau suggests approaching boundary

## Iter 24: converged
Node: id=24, parent=23, V=1, N_total=24
Config: lr_W=2.0E-3, lr=5.0E-1, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.8268, test_pearson=0.8552, connectivity_R2=0.9996, final_loss=1.21e+04
Observation: connectivity_R2=0.9996 with lr=5E-1 (5000x higher than baseline) - lr range expanded to [1E-4, 5E-1] all converge; final_loss=1.21e+04 similar to plateau at lr=1E-1
Change: lr: 2.0E-1 -> 5.0E-1 (from node 23)
Next: test lr=1.0E+0 (lr=1.0) from node 24 to find lr upper failure point

## Iter 25: converged
Node: id=25, parent=24, V=1, N_total=25
Config: lr_W=2.0E-3, lr=1.0E+0, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.8268, test_pearson=0.8552, connectivity_R2=0.9996, final_loss=1.15e+04
Observation: connectivity_R2=0.9996 with lr=1.0 (10000x higher than baseline) - lr range expanded to [1E-4, 1.0] all converge; final_loss=1.15e+04 slightly lower than lr=5E-1
Change: lr: 5.0E-1 -> 1.0E+0 (from node 24)
Next: test lr=2.0E+0 from node 25 to continue finding lr upper failure point

## Iter 26: converged
Node: id=26, parent=25, V=1, N_total=26
Config: lr_W=2.0E-3, lr=2.0E+0, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.8268, test_pearson=0.8552, connectivity_R2=0.9996, final_loss=1.17e+04
Observation: connectivity_R2=0.9996 with lr=2.0 (20000x higher than baseline) - lr range expanded to [1E-4, 2.0] all converge; final_loss=1.17e+04 similar to lr=1.0, plateau continues
Change: lr: 1.0E+0 -> 2.0E+0 (from node 25)
Next: test lr=5.0E+0 from node 26 to continue finding lr upper failure point - plateau at final_loss~1.1e+04 suggests lr may not be the limiting factor

## Iter 27: converged
Node: id=27, parent=26, V=1, N_total=27
Config: lr_W=2.0E-3, lr=5.0E+0, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.8268, test_pearson=0.8552, connectivity_R2=0.9996, final_loss=1.18e+04
Observation: connectivity_R2=0.9996 with lr=5.0 (50000x higher than baseline) - lr range expanded to [1E-4, 5.0] all converge; final_loss=1.18e+04 still at plateau
Change: lr: 2.0E+0 -> 5.0E+0 (from node 26)
Next: test lr=1.0E+1 from node 27 to continue finding lr upper failure point

## Iter 28: converged
Node: id=28, parent=27, V=1, N_total=28
Config: lr_W=2.0E-3, lr=1.0E+1, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.8268, test_pearson=0.8552, connectivity_R2=0.9996, final_loss=1.23e+04
Observation: connectivity_R2=0.9996 with lr=10.0 (100000x higher than baseline) - lr range expanded to [1E-4, 10.0] all converge; final_loss=1.23e+04 still at plateau (~1.1-1.2e+04)
Change: lr: 5.0E+0 -> 1.0E+1 (from node 27)
Next: pivot to explore high-UCB unexplored nodes (5, 6, 12, 14, 16) - lr upper boundary search exhausted without finding failure point at lr=10

## Iter 29: converged
Node: id=29, parent=5, V=1, N_total=29
Config: lr_W=2.0E-3, lr=1.0E-4, coeff_W_L1=0, batch_size=8
Metrics: test_R2=0.8268, test_pearson=0.8552, connectivity_R2=0.9996, final_loss=1.17e+02
Observation: connectivity_R2=0.9996 with coeff_W_L1=0 (from high-UCB node 5) - confirms L1 regularization not critical; final_loss=1.17e+02 similar to baseline
Change: pivot from lr exploration to node 5 (coeff_W_L1=0 config); lr: 1.0E+1 -> 1.0E-4
Next: explore lr_W=1.0E-2 from node 5 (coeff_W_L1=0) to test if high lr_W works without L1 regularization

## Iter 30: converged
Node: id=30, parent=29, V=1, N_total=30
Config: lr_W=1.0E-2, lr=1.0E-4, coeff_W_L1=0, batch_size=8
Metrics: test_R2=0.8268, test_pearson=0.8552, connectivity_R2=0.9996, final_loss=2.38e+02
Observation: connectivity_R2=0.9996 with lr_W=1E-2 and coeff_W_L1=0 - confirms high lr_W works without L1 regularization; final_loss=2.38e+02 (2x higher than lr_W=2E-3)
Change: lr_W: 2.0E-3 -> 1.0E-2 (from node 29)
Next: explore from high-UCB node 6 (coeff_W_L1=1E-4) - test lr_W=5E-3 with high L1 to map interaction effects

## Iter 31: converged
Node: id=31, parent=6, V=1, N_total=31
Config: lr_W=5.0E-3, lr=1.0E-4, coeff_W_L1=1.0E-4, batch_size=8
Metrics: test_R2=0.8268, test_pearson=0.8552, connectivity_R2=0.9996, final_loss=2.74e+02
Observation: connectivity_R2=0.9996 with lr_W=5E-3 and coeff_W_L1=1E-4 (high L1) - confirms high lr_W works with high L1 regularization; final_loss=2.74e+02 similar to lr_W=1E-2 with L1=0
Change: lr_W: 2.0E-3 -> 5.0E-3 (from node 6, coeff_W_L1=1E-4)
Next: explore from high-UCB node 12 (batch_size=1) - test lr_W=5E-3 to map interaction of high lr_W with small batch

## Iter 32: converged
Node: id=32, parent=12, V=1, N_total=32
Config: lr_W=5.0E-3, lr=1.0E-4, coeff_W_L1=1.0E-5, batch_size=1
Metrics: test_R2=0.9721, test_pearson=0.9690, connectivity_R2=0.9998, final_loss=1.02e+03
Observation: connectivity_R2=0.9998 with lr_W=5E-3 and batch_size=1 - confirms high lr_W works with small batch; final_loss=1.02e+03 (higher than batch_size=8 as expected)
Change: lr_W: 1.0E-2 -> 5.0E-3 (from node 12, batch_size=1)
Next: explore from high-UCB node 14 (lr_W=5E-4) - test batch_size=16 to map interaction of low lr_W with large batch

## Iter 33: converged
Node: id=33, parent=14, V=1, N_total=33
Config: lr_W=5.0E-4, lr=1.0E-4, coeff_W_L1=1.0E-5, batch_size=16
Metrics: test_R2=0.9721, test_pearson=0.9690, connectivity_R2=0.9998, final_loss=3.68e+03
Observation: connectivity_R2=0.9998 with lr_W=5E-4 and batch_size=16 - confirms low lr_W works with large batch; final_loss=3.68e+03 (high due to slow lr_W convergence)
Change: batch_size: 8 -> 16 (from node 14, lr_W=5E-4)
Next: explore from highest-UCB node 32 (lr_W=5E-3, batch_size=1) - test lr=5E-4 to map interaction of high lr_W with small batch and higher lr

## Iter 34: converged
Node: id=34, parent=32, V=1, N_total=34
Config: lr_W=5.0E-3, lr=5.0E-4, coeff_W_L1=1.0E-5, batch_size=1
Metrics: test_R2=0.9673, test_pearson=0.9668, connectivity_R2=0.9998, final_loss=1.02e+03
Observation: connectivity_R2=0.9998 with lr_W=5E-3, lr=5E-4, batch_size=1 - confirms high lr_W with higher lr works with small batch; final_loss=1.02e+03 similar to iter 32
Change: lr: 1.0E-4 -> 5.0E-4 (from node 32, batch_size=1)
Next: explore from high-UCB node 33 (lr_W=5E-4, batch_size=16) - test lr=5E-4 to map interaction of low lr_W with large batch and higher lr

## Iter 35: converged
Node: id=35, parent=33, V=1, N_total=35
Config: lr_W=5.0E-4, lr=5.0E-4, coeff_W_L1=1.0E-5, batch_size=16
Metrics: test_R2=0.9673, test_pearson=0.9668, connectivity_R2=0.9998, final_loss=3.43e+03
Observation: connectivity_R2=0.9998 with lr_W=5E-4, lr=5E-4, batch_size=16 - confirms low lr_W with higher lr works with large batch; final_loss=3.43e+03 (similar to iter 33 with lr=1E-4)
Change: lr: 1.0E-4 -> 5.0E-4 (from node 33, batch_size=16)
Next: explore from high-UCB node 16 (lr_W=1E-4, batch_size=8) - test lr=5E-4 to map interaction of extreme low lr_W with higher lr

## Iter 36: converged
Node: id=36, parent=16, V=1, N_total=36
Config: lr_W=1.0E-4, lr=5.0E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9673, test_pearson=0.9668, connectivity_R2=0.9998, final_loss=8.66e+03
Observation: connectivity_R2=0.9998 with lr_W=1E-4 (extreme low) and lr=5E-4 - confirms extreme low lr_W works with higher lr; final_loss=8.66e+03 (high due to slow lr_W convergence, similar to iter 16 with final_loss=8.75e+03)
Change: lr: 1.0E-4 -> 5.0E-4 (from node 16, lr_W=1E-4)
Next: explore from high-UCB node 28 (lr_W=2E-3, lr=10.0) - test coeff_W_L1=0 to map interaction of extreme high lr with no L1 regularization

## Iter 37: converged
Node: id=37, parent=28, V=1, N_total=37
Config: lr_W=2.0E-3, lr=1.0E+1, coeff_W_L1=0, batch_size=8
Metrics: test_R2=0.9673, test_pearson=0.9668, connectivity_R2=0.9998, final_loss=1.21e+04
Observation: connectivity_R2=0.9998 with extreme high lr=10.0 and coeff_W_L1=0 - confirms extreme high lr works without L1 regularization; final_loss=1.21e+04 similar to iter 28 with L1=1E-5
Change: coeff_W_L1: 1.0E-5 -> 0 (from node 28, lr=10.0)
Next: explore from high-UCB node 34 (lr_W=5E-3, lr=5E-4, batch_size=1) - test coeff_W_L1=0 to map interaction of high lr_W with small batch and no L1

## Iter 38: converged
Node: id=38, parent=34, V=1, N_total=38
Config: lr_W=5.0E-3, lr=5.0E-4, coeff_W_L1=0, batch_size=1
Metrics: test_R2=0.8456, test_pearson=0.8621, connectivity_R2=0.9997, final_loss=9.89e+02
Observation: connectivity_R2=0.9997 with lr_W=5E-3, lr=5E-4, batch_size=1, coeff_W_L1=0 - confirms high lr_W with small batch works without L1 regularization; final_loss=9.89e+02 similar to iter 34 with L1=1E-5
Change: coeff_W_L1: 1.0E-5 -> 0 (from node 34, batch_size=1)
Next: explore from high-UCB node 35 (lr_W=5E-4, lr=5E-4, batch_size=16) - test coeff_W_L1=0 to map interaction of low lr_W with large batch and no L1

## Iter 39: converged
Node: id=39, parent=35, V=1, N_total=39
Config: lr_W=5.0E-4, lr=5.0E-4, coeff_W_L1=0, batch_size=16
Metrics: test_R2=0.8456, test_pearson=0.8621, connectivity_R2=0.9997, final_loss=3.41e+03
Observation: connectivity_R2=0.9997 with lr_W=5E-4, lr=5E-4, batch_size=16, coeff_W_L1=0 - confirms low lr_W with large batch works without L1 regularization; final_loss=3.41e+03 similar to iter 35 with L1=1E-5 (3.43e+03)
Change: coeff_W_L1: 1.0E-5 -> 0 (from node 35, batch_size=16)
Next: explore from high-UCB node 30 (lr_W=1E-2, lr=1E-4, coeff_W_L1=0) - test batch_size=4 to map interaction of high lr_W with small batch and no L1

## Iter 40: converged
Node: id=40, parent=30, V=1, N_total=40
Config: lr_W=1.0E-2, lr=1.0E-4, coeff_W_L1=0, batch_size=4
Metrics: test_R2=0.8456, test_pearson=0.8621, connectivity_R2=0.9997, final_loss=4.49e+02
Observation: connectivity_R2=0.9997 with lr_W=1E-2, lr=1E-4, batch_size=4, coeff_W_L1=0 - confirms high lr_W with small batch works without L1; final_loss=4.49e+02 (2x higher than node 30 with batch_size=8 at 2.38e+02)
Change: batch_size: 8 -> 4 (from node 30, lr_W=1E-2, coeff_W_L1=0)
Next: meta-analysis at iteration 40, then explore from high-UCB node 31 (lr_W=5E-3, coeff_W_L1=1E-4) - test batch_size=4 to map interaction of medium-high lr_W with small batch and high L1

## Iter 41: converged
Node: id=41, parent=31, V=1, N_total=41
Config: lr_W=5.0E-3, lr=1.0E-4, coeff_W_L1=1.0E-4, batch_size=4
Metrics: test_R2=0.8456, test_pearson=0.8621, connectivity_R2=0.9997, final_loss=5.24e+02
Observation: connectivity_R2=0.9997 with lr_W=5E-3, lr=1E-4, batch_size=4, coeff_W_L1=1E-4 - confirms medium-high lr_W with small batch works with high L1; final_loss=5.24e+02 similar to node 40 (4.49e+02 with same batch but higher lr_W and no L1)
Change: batch_size: 8 -> 4 (from node 31, lr_W=5E-3, coeff_W_L1=1E-4)
Next: explore from high-UCB node 36 (lr_W=1E-4, lr=5E-4) - test batch_size=4 to map interaction of extreme low lr_W with higher lr and small batch

## Iter 42: converged
Node: id=42, parent=36, V=1, N_total=42
Config: lr_W=1.0E-4, lr=5.0E-4, coeff_W_L1=1.0E-5, batch_size=4
Metrics: test_R2=0.8456, test_pearson=0.8621, connectivity_R2=0.9997, final_loss=1.08e+04
Observation: connectivity_R2=0.9997 with extreme low lr_W=1E-4, lr=5E-4, batch_size=4 - confirms extreme low lr_W works with higher lr and small batch; final_loss=1.08e+04 (very high due to slow lr_W convergence, similar to node 36 at 8.66e+03 with batch_size=8)
Change: batch_size: 8 -> 4 (from node 36, lr_W=1E-4, lr=5E-4)
Next: explore from high-UCB node 37 (lr_W=2E-3, lr=10.0, coeff_W_L1=0) - test batch_size=4 to map interaction of extreme high lr with no L1 and small batch

## Iter 43: converged
Node: id=43, parent=37, V=1, N_total=43
Config: lr_W=2.0E-3, lr=1.0E+1, coeff_W_L1=0, batch_size=4
Metrics: test_R2=0.8456, test_pearson=0.8621, connectivity_R2=0.9997, final_loss=1.72e+04
Observation: connectivity_R2=0.9997 with extreme high lr=10.0, no L1, batch_size=4 - confirms extreme high lr works with small batch and no L1; final_loss=1.72e+04 (higher than node 37 at 1.21e+04 with batch_size=8, ~40% increase from smaller batch)
Change: batch_size: 8 -> 4 (from node 37, lr_W=2E-3, lr=10.0, coeff_W_L1=0)
Next: explore from high-UCB node 38 (lr_W=5E-3, lr=5E-4, coeff_W_L1=0, batch_size=1) - test batch_size=16 to map interaction of high lr_W with large batch and no L1

## Iter 44: converged
Node: id=44, parent=38, V=1, N_total=44
Config: lr_W=5.0E-3, lr=5.0E-4, coeff_W_L1=0, batch_size=16
Metrics: test_R2=0.8456, test_pearson=0.8621, connectivity_R2=0.9997, final_loss=1.19e+02
Observation: connectivity_R2=0.9997 with lr_W=5E-3, lr=5E-4, batch_size=16, coeff_W_L1=0 - confirms high lr_W with large batch works without L1; final_loss=1.19e+02 (much lower than node 38 at 9.89e+02 with batch_size=1, ~8x improvement from larger batch)
Change: batch_size: 1 -> 16 (from node 38, lr_W=5E-3, lr=5E-4, coeff_W_L1=0)
Next: explore from high-UCB node 39 (lr_W=5E-4, lr=5E-4, coeff_W_L1=0, batch_size=16) - test batch_size=1 to map interaction of low lr_W with small batch and no L1

## Iter 45: converged
Node: id=45, parent=39, V=1, N_total=45
Config: lr_W=5.0E-4, lr=5.0E-4, coeff_W_L1=0, batch_size=1
Metrics: test_R2=0.9151, test_pearson=0.9245, connectivity_R2=0.9999, final_loss=1.35e+03
Observation: connectivity_R2=0.9999 with low lr_W=5E-4, lr=5E-4, batch_size=1, coeff_W_L1=0 - confirms low lr_W with small batch works without L1; final_loss=1.35e+03 (lower than node 39 at 3.41e+03 with batch_size=16, ~2.5x reduction from smaller batch possibly due to more gradient updates)
Change: batch_size: 16 -> 1 (from node 39, lr_W=5E-4, lr=5E-4, coeff_W_L1=0)
Next: explore from high-UCB node 40 (lr_W=1E-2, lr=1E-4, coeff_W_L1=0, batch_size=4) - test lr=1E-3 to map interaction of high lr_W with small batch, no L1, and higher lr

## Iter 46: converged
Node: id=46, parent=40, V=1, N_total=46
Config: lr_W=1.0E-2, lr=1.0E-3, coeff_W_L1=0, batch_size=4
Metrics: test_R2=0.9151, test_pearson=0.9245, connectivity_R2=0.9999, final_loss=4.64e+02
Observation: connectivity_R2=0.9999 with high lr_W=1E-2, lr=1E-3, batch_size=4, coeff_W_L1=0 - confirms high lr_W with small batch and higher lr works without L1; final_loss=4.64e+02 (similar to node 40 at 4.49e+02 with lr=1E-4)
Change: lr: 1.0E-4 -> 1.0E-3 (from node 40, lr_W=1E-2, batch_size=4, coeff_W_L1=0)
Next: explore from high-UCB node 45 (lr_W=5E-4, lr=5E-4, batch_size=1, coeff_W_L1=0) - test lr_W=1E-3 to increase lr_W and map interaction with small batch

## Iter 47: converged
Node: id=47, parent=45, V=1, N_total=47
Config: lr_W=1.0E-3, lr=5.0E-4, coeff_W_L1=0, batch_size=1
Metrics: test_R2=0.9755, test_pearson=0.9757, connectivity_R2=0.9999, final_loss=5.75e+02
Observation: connectivity_R2=0.9999 with lr_W=1E-3 (2x increase from parent), lr=5E-4, batch_size=1, coeff_W_L1=0 - confirms medium lr_W with small batch works without L1; final_loss=5.75e+02 (lower than node 45 at 1.35e+03 with lr_W=5E-4, ~2.3x improvement from doubling lr_W)
Change: lr_W: 5.0E-4 -> 1.0E-3 (from node 45, lr=5E-4, batch_size=1, coeff_W_L1=0)
Next: explore from high-UCB node 41 (lr_W=5E-3, lr=1E-4, coeff_W_L1=1E-4, batch_size=4) - test lr=5E-4 to map interaction of high lr_W with small batch and high L1

## Iter 48: converged
Node: id=48, parent=41, V=1, N_total=48
Config: lr_W=5.0E-3, lr=5.0E-4, coeff_W_L1=1.0E-4, batch_size=4
Metrics: test_R2=0.9755, test_pearson=0.9757, connectivity_R2=0.9999, final_loss=5.73e+02
Observation: connectivity_R2=0.9999 with lr_W=5E-3, lr=5E-4, coeff_W_L1=1E-4, batch_size=4 - confirms high lr_W with small batch and high L1 works well with higher lr; final_loss=5.73e+02 (similar to node 41 at 5.24e+02 with lr=1E-4)
Change: lr: 1.0E-4 -> 5.0E-4 (from node 41, lr_W=5E-3, coeff_W_L1=1E-4, batch_size=4)
Next: explore from high-UCB node 42 (lr_W=1E-4, lr=5E-4, coeff_W_L1=1E-5, batch_size=4) - test lr=1E-3 to map interaction of extreme low lr_W with higher lr and small batch

## Iter 49: converged
Node: id=49, parent=42, V=1, N_total=49
Config: lr_W=1.0E-4, lr=1.0E-3, coeff_W_L1=1.0E-5, batch_size=4
Metrics: test_R2=0.9755, test_pearson=0.9757, connectivity_R2=0.9999, final_loss=1.04e+04
Observation: connectivity_R2=0.9999 with extreme low lr_W=1E-4, lr=1E-3 (2x from parent), batch_size=4 - confirms extreme low lr_W works with higher lr and small batch; final_loss=1.04e+04 (similar to node 42 at 1.08e+04 with lr=5E-4)
Change: lr: 5.0E-4 -> 1.0E-3 (from node 42, lr_W=1E-4, batch_size=4)
Next: explore from high-UCB node 43 (lr_W=2E-3, lr=10.0, coeff_W_L1=0, batch_size=4) - test lr_W=5E-3 to map interaction of higher lr_W with extreme high lr and small batch

## Iter 50: converged
Node: id=50, parent=43, V=1, N_total=50
Config: lr_W=5.0E-3, lr=1.0E+1, coeff_W_L1=0, batch_size=4
Metrics: test_R2=0.9755, test_pearson=0.9757, connectivity_R2=0.9999, final_loss=1.78e+04
Observation: connectivity_R2=0.9999 with lr_W=5E-3 (2.5x increase from parent), lr=10.0, batch_size=4, coeff_W_L1=0 - confirms higher lr_W with extreme high lr and small batch works without L1; final_loss=1.78e+04 (similar to node 43 at 1.72e+04, slight increase with higher lr_W)
Change: lr_W: 2.0E-3 -> 5.0E-3 (from node 43, lr=10.0, batch_size=4, coeff_W_L1=0)
Next: meta-analysis at iteration 50, then explore from high-UCB node 44 (lr_W=5E-3, lr=5E-4, coeff_W_L1=0, batch_size=16) - test lr=1E-3 to map interaction of high lr_W with large batch and higher lr

## Iter 51: converged
Node: id=51, parent=44, V=1, N_total=51
Config: lr_W=5.0E-3, lr=1.0E-3, coeff_W_L1=0, batch_size=16
Metrics: test_R2=0.9755, test_pearson=0.9757, connectivity_R2=0.9999, final_loss=1.24e+02
Observation: connectivity_R2=0.9999 with lr_W=5E-3, lr=1E-3 (2x increase from parent), batch_size=16, coeff_W_L1=0 - confirms high lr_W with large batch and higher lr works well; final_loss=1.24e+02 (similar to node 44 at 1.19e+02, slight increase with higher lr)
Change: lr: 5.0E-4 -> 1.0E-3 (from node 44, lr_W=5E-3, batch_size=16, coeff_W_L1=0)
Next: explore from high-UCB node 47 (lr_W=1E-3, lr=5E-4, batch_size=1, coeff_W_L1=0) - test lr=1E-3 to map interaction of medium lr_W with small batch and higher lr

---

## Meta-Analysis (Iterations 41-50)

### Summary of Recent Exploration

| Iter | Node | Config (lr_W, lr, L1, batch) | connectivity_R2 | final_loss | Focus |
|------|------|------------------------------|-----------------|------------|-------|
| 41 | 41 | 5E-3, 1E-4, 1E-4, 4 | 0.9997 | 5.24e+02 | high lr_W + high L1 + small batch |
| 42 | 42 | 1E-4, 5E-4, 1E-5, 4 | 0.9997 | 1.08e+04 | extreme low lr_W + small batch |
| 43 | 43 | 2E-3, 10.0, 0, 4 | 0.9997 | 1.72e+04 | extreme high lr + no L1 + small batch |
| 44 | 44 | 5E-3, 5E-4, 0, 16 | 0.9997 | 1.19e+02 | high lr_W + large batch + no L1 |
| 45 | 45 | 5E-4, 5E-4, 0, 1 | 0.9999 | 1.35e+03 | low lr_W + batch_size=1 + no L1 |
| 46 | 46 | 1E-2, 1E-3, 0, 4 | 0.9999 | 4.64e+02 | high lr_W + higher lr + small batch |
| 47 | 47 | 1E-3, 5E-4, 0, 1 | 0.9999 | 5.75e+02 | medium lr_W + batch_size=1 + no L1 |
| 48 | 48 | 5E-3, 5E-4, 1E-4, 4 | 0.9999 | 5.73e+02 | high lr_W + high L1 + small batch |
| 49 | 49 | 1E-4, 1E-3, 1E-5, 4 | 0.9999 | 1.04e+04 | extreme low lr_W + higher lr |
| 50 | 50 | 5E-3, 10.0, 0, 4 | 0.9999 | 1.78e+04 | high lr_W + extreme high lr + no L1 |

### Key Findings (Iterations 41-50)

1. **100% convergence rate continues**: 50/50 iterations converged with connectivity_R2 > 0.99

2. **Extreme lr (lr=10.0) still works**:
   - iter 43: lr=10.0, connectivity_R2=0.9997, final_loss=1.72e+04
   - iter 50: lr=10.0, connectivity_R2=0.9999, final_loss=1.78e+04
   - extreme high lr just increases final_loss, doesn't break convergence

3. **batch_size=1 still works**:
   - iter 45: batch_size=1, connectivity_R2=0.9999, final_loss=1.35e+03
   - iter 47: batch_size=1, connectivity_R2=0.9999, final_loss=5.75e+02
   - small batch increases variance but converges well

4. **batch_size=16 gives lowest final_loss**:
   - iter 44: batch_size=16, connectivity_R2=0.9997, final_loss=1.19e+02
   - consistently best final_loss with large batch

5. **L1 regularization still optional**:
   - iter 48 (with L1=1E-4): connectivity_R2=0.9999, final_loss=5.73e+02
   - iter 47 (no L1): connectivity_R2=0.9999, final_loss=5.75e+02
   - no measurable difference

### Complete Parameter Space Map (50 Iterations)

| Parameter | Full Range Tested | All Working? | Optimal Value | Notes |
|-----------|-------------------|--------------|---------------|-------|
| lr_W | 1E-4 to 1E-2 | ✓ 100% | 2E-3 to 5E-3 | low lr_W → slow convergence |
| lr | 1E-4 to 10.0 | ✓ 100% | 1E-4 to 1E-3 | high lr → high final_loss |
| coeff_W_L1 | 0 to 1E-4 | ✓ 100% | any (optional) | no effect on convergence |
| batch_size | 1 to 16 | ✓ 100% | 16 (best loss) | small batch → higher loss |

### Conclusions at Iteration 50

1. **Extremely robust system**: 50/50 iterations converged (100% success rate)
2. **No failure region found**: all 50 tested parameter combinations work
3. **Parameter space is flat**: connectivity_R2 > 0.999 everywhere
4. **Final loss varies by ~100x**: from 1.19e+02 (iter 44) to 1.78e+04 (iter 50)
5. **Optimal settings**: lr_W=5E-3, lr=5E-4, coeff_W_L1=0, batch_size=16 → lowest final_loss

### Recommendations for Iterations 51-60

1. **Consider stopping exploration**: parameter space appears fully characterized
2. **Test extreme lr_W boundaries**: lr_W=5E-5 or lr_W=2E-2 (beyond current range)
3. **Validate optimal config**: repeat iter 44 config (lr_W=5E-3, lr=5E-4, batch_size=16, L1=0) 3x
4. **Alternative: explore different simulation configs** (different PDE models, connectivity types)

---

## Meta-Analysis (Iterations 31-40)

### Summary of Recent Exploration

| Iter | Node | Config (lr_W, lr, L1, batch) | connectivity_R2 | final_loss | Focus |
|------|------|------------------------------|-----------------|------------|-------|
| 31 | 31 | 5E-3, 1E-4, 1E-4, 8 | 0.9996 | 2.74e+02 | high lr_W + high L1 |
| 32 | 32 | 5E-3, 1E-4, 1E-5, 1 | 0.9998 | 1.02e+03 | high lr_W + small batch |
| 33 | 33 | 5E-4, 1E-4, 1E-5, 16 | 0.9998 | 3.68e+03 | low lr_W + large batch |
| 34 | 34 | 5E-3, 5E-4, 1E-5, 1 | 0.9998 | 1.02e+03 | high lr_W + higher lr + small batch |
| 35 | 35 | 5E-4, 5E-4, 1E-5, 16 | 0.9998 | 3.43e+03 | low lr_W + higher lr + large batch |
| 36 | 36 | 1E-4, 5E-4, 1E-5, 8 | 0.9998 | 8.66e+03 | extreme low lr_W |
| 37 | 37 | 2E-3, 10.0, 0, 8 | 0.9998 | 1.21e+04 | extreme high lr + no L1 |
| 38 | 38 | 5E-3, 5E-4, 0, 1 | 0.9997 | 9.89e+02 | high lr_W + small batch + no L1 |
| 39 | 39 | 5E-4, 5E-4, 0, 16 | 0.9997 | 3.41e+03 | low lr_W + large batch + no L1 |
| 40 | 40 | 1E-2, 1E-4, 0, 4 | 0.9997 | 4.49e+02 | high lr_W + small batch + no L1 |

### Key Findings (Iterations 31-40)

1. **100% convergence rate continues**: 10/10 iterations converged with connectivity_R2 > 0.99

2. **L1 regularization effect**: minimal
   - With L1 (iter 34): connectivity_R2=0.9998, final_loss=1.02e+03
   - Without L1 (iter 38): connectivity_R2=0.9997, final_loss=9.89e+02
   - Difference negligible - L1 is optional

3. **Interaction effects explored**:
   - high lr_W + small batch: converges (iters 32, 34, 38, 40)
   - low lr_W + large batch: converges but slow (iters 33, 35, 39)
   - extreme high lr (10.0): still converges (iter 37)

4. **Final loss patterns**:
   - Low lr_W (1E-4, 5E-4) → high final_loss (3-9e+03)
   - High lr_W (5E-3, 1E-2) → lower final_loss (2-5e+02)
   - Small batch_size increases final_loss ~2-4x

### Complete Parameter Space Map (40 Iterations)

| Parameter | Full Range Tested | All Working? | Optimal Value |
|-----------|-------------------|--------------|---------------|
| lr_W | 1E-4 to 1E-2 | ✓ | 2E-3 to 5E-3 |
| lr | 1E-4 to 10.0 | ✓ | 1E-4 to 1E-3 |
| coeff_W_L1 | 0 to 1E-4 | ✓ | any (optional) |
| batch_size | 1 to 16 | ✓ | 8 (best loss) |

### Conclusions at Iteration 40

1. **Extremely robust system**: 40/40 iterations converged (100%)
2. **No failure region found**: all tested parameter combinations work
3. **lr not limiting**: even lr=10.0 converges (connectivity_R2=0.9998)
4. **lr_W more important than lr**: lr_W affects convergence speed more
5. **L1 regularization optional**: no measurable benefit for convergence
6. **batch_size affects loss magnitude only**: all sizes converge

### Recommendations for Iterations 41-50

Since no failure points found in standard ranges, explore:
1. More interaction effects (high lr_W + high lr + no L1)
2. Test lr_W=5E-5 (extreme lower boundary)
3. Validate best configurations with repeated runs
4. Consider stopping parameter search - space appears fully robust

---

## Meta-Analysis (Iterations 11-20)

### Summary of Explored Parameter Space (Complete)

| Parameter | Tested Range | Working Range | Notes |
|-----------|--------------|---------------|-------|
| lr_W | 1E-4 to 1E-2 | 1E-4 to 1E-2 | all converge; final_loss increases at extremes |
| lr | 1E-4 to 2E-2 | 1E-4 to 2E-2 | 200x range all converge; lr=1E-2 optimal |
| coeff_W_L1 | 0 to 1E-4 | 0 to 1E-4 | L1 regularization not critical |
| batch_size | 1 to 16 | 1 to 16 | all converge; smaller batch = higher loss |

### Key Findings (Complete Study)

1. **Extremely robust parameter space**: 20/20 iterations converged (100%)

2. **lr_W range**: [1E-4, 1E-2] all work (100x range)
   - lr_W=1E-4: connectivity_R2=0.9996, final_loss=8.75e+03 (slow)
   - lr_W=2E-3: connectivity_R2=1.0000, final_loss=1.07e+02 (optimal)
   - lr_W=1E-2: connectivity_R2=0.9999, final_loss=2.55e+02

3. **lr range**: [1E-4, 2E-2] all work (200x range)
   - lr=1E-4: connectivity_R2=1.0000, final_loss=1.07e+02 (baseline)
   - lr=1E-2: connectivity_R2=0.9996, final_loss=2.71e+02 (optimal high lr)
   - lr=2E-2: connectivity_R2=0.9996, final_loss=3.12e+03 (upper boundary)

4. **batch_size range**: [1, 16] all work
   - batch_size=1: final_loss=1.27e+03 (highest)
   - batch_size=8: final_loss=1.07e+02 (optimal)
   - batch_size=16: final_loss=1.28e+02

5. **coeff_W_L1 range**: [0, 1E-4] all work - L1 not critical

### Optimal Configurations

| Config | lr_W | lr | coeff_W_L1 | batch_size | connectivity_R2 | final_loss |
|--------|------|-----|------------|------------|-----------------|------------|
| Robust baseline | 2E-3 | 1E-4 | 1E-5 | 8 | 1.0000 | 1.07e+02 |
| Fast training | 1E-2 | 5E-4 | 1E-5 | 8 | 0.9999 | 2.91e+02 |
| High lr | 2E-3 | 1E-2 | 1E-5 | 8 | 0.9996 | 2.71e+02 |

### Conclusions

1. **Parameter space is highly robust**: No failure points found in tested ranges
2. **Optimal settings**: lr_W=2E-3, lr=1E-4, batch_size=8 give best final_loss
3. **L1 regularization optional**: coeff_W_L1 can be 0 without affecting convergence
4. **Training is stable**: Stochasticity appears minimal in this parameter region

### Recommendations for Next Phase

Since no failure points have been found, explore extreme boundaries:
1. Test lr_W=5E-5 (extreme lower boundary)
2. Test lr=5E-2 (extreme upper boundary)
3. Test combinations of extreme values to find interaction effects

---

---

## Meta-Analysis (Iterations 1-10)

### Summary of Explored Parameter Space

| Parameter | Tested Range | Working Range | Notes |
|-----------|--------------|---------------|-------|
| lr_W | 2E-3 to 1E-2 | 2E-3 to 1E-2 | all tested values converge |
| lr | 1E-4 to 5E-4 | 1E-4 to 5E-4 | both endpoints work |
| coeff_W_L1 | 0 to 1E-4 | 0 to 1E-4 | L1 regularization not critical |
| batch_size | 8 to 16 | 8 to 16 | both work, 16 gives lower loss |

### Key Findings

1. **Robust baseline found**: lr_W=2E-3, lr=1E-4, coeff_W_L1=1E-5, batch_size=8 validated 3/3 (100%)

2. **L1 regularization (coeff_W_L1)**: Not critical for convergence
   - coeff_W_L1=0: connectivity_R2=1.0000
   - coeff_W_L1=1E-5: connectivity_R2=1.0000
   - coeff_W_L1=1E-4: connectivity_R2=1.0000

3. **lr_W is robust**: Range [2E-3, 1E-2] all achieve connectivity_R2 > 0.99

4. **lr is robust**: Range [1E-4, 5E-4] both work

5. **batch_size**: Both 8 and 16 work; batch_size=16 tends to give slightly lower final_loss

6. **Convergence rate**: 10/10 iterations converged (100%) - parameter space appears well-behaved

### Unexplored Boundaries

- lr_W lower boundary (< 2E-3): not yet tested
- lr upper boundary (> 5E-4): not yet tested
- batch_size lower boundary (< 8): not yet tested
- batch_size=1, 4: not yet tested

### Recommendations for Next 10 Iterations

1. Test batch_size=4 (lower boundary)
2. Test batch_size=1 (extreme lower)
3. Test lr_W=1E-3 (lower boundary)
4. Test lr=1E-3 (upper boundary)
5. Test combinations near boundaries to find failure points

---

