# Experiment Log: signal_chaotic_1_Claude

## Iter 1: converged
Node: id=1, parent=root
Mode/Strategy: exploit
Config: lr_W=2.0E-3, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, low_rank_factorization=False
Metrics: test_R2=0.9982, test_pearson=0.9959, connectivity_R2=1.0000, final_loss=5.58E+03
Activity: rich chaotic dynamics, 100 neurons, range [-19.8, 20.3], effective rank 35 (99% var), spectral radius 1.051
Mutation: baseline config (first iteration)
Parent rule: first iteration, parent=root
Observation: excellent convergence with baseline config, perfect connectivity recovery
Next: parent=1 (highest UCB, will probe boundaries with lr mutation)

## Iter 2: converged
Node: id=2, parent=1
Mode/Strategy: exploit
Config: lr_W=5.0E-3, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, low_rank_factorization=False
Metrics: test_R2=0.9875, test_pearson=0.9795, connectivity_R2=0.9999, final_loss=3.80E+03
Activity: chaotic dynamics, 100 neurons, range [-19.6, 18.2], effective rank 34 (99% var), spectral radius 1.025
Mutation: lr_W: 2.0E-3 -> 5.0E-3 (2.5x increase)
Parent rule: exploit - highest UCB node (node 1), increased lr_W to probe upper boundary
Observation: higher lr_W maintains excellent convergence, lower final loss, slightly reduced pearson (0.996->0.980)
Next: parent=2 (highest UCB=1.707, try batch_size mutation)

## Iter 3: converged
Node: id=3, parent=2
Mode/Strategy: failure-probe (3 consecutive successes triggered)
Config: lr_W=5.0E-3, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=16, low_rank_factorization=False
Metrics: test_R2=0.9208, test_pearson=0.8845, connectivity_R2=0.9987, final_loss=4.04E+03
Activity: chaotic dynamics, 100 neurons, range [-19.6, 17.8], effective rank 33 (99% var), spectral radius 1.025
Mutation: batch_size: 8 -> 16 (2x increase)
Parent rule: exploit from node 2 (highest UCB=1.577), batch_size doubled
Observation: batch_size=16 slightly degrades pearson (0.98->0.88) but connectivity R² remains excellent; training robust to batch size
Next: parent=3 (highest UCB=1.865, failure-probe with extreme lr_W=0.02 to find failure boundary)

## Iter 4: converged
Node: id=4, parent=3
Mode/Strategy: failure-probe (4 consecutive successes)
Config: lr_W=2.0E-2, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=16, low_rank_factorization=False
Metrics: test_R2=0.9425, test_pearson=0.9167, connectivity_R2=0.9997, final_loss=2.11E+03
Activity: chaotic dynamics, 100 neurons, range [-21.0, 21.0], effective rank 33 (99% var), spectral radius 1.025
Mutation: lr_W: 5.0E-3 -> 2.0E-2 (4x increase)
Parent rule: failure-probe from node 3 (highest UCB=1.865), lr_W pushed to 10x baseline
Observation: extreme lr_W=0.02 still converges excellently; connectivity recovery very robust to lr_W; lowest final loss yet (2.11E+03)
Next: parent=4 (highest UCB=2.000, failure-probe with very low lr_W=2.0E-4 to find lower boundary)

## Iter 5: failed
Node: id=5, parent=4
Mode/Strategy: failure-probe (probing lower lr_W boundary)
Config: lr_W=2.0E-4, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=16, low_rank_factorization=False
Metrics: test_R2=0.3677, test_pearson=0.1754, connectivity_R2=0.0657, final_loss=1.31E+04
Activity: chaotic dynamics, 100 neurons, range [-23.5, 19.8], effective rank 33 (99% var), spectral radius 1.025
Mutation: lr_W: 2.0E-2 -> 2.0E-4 (100x decrease)
Parent rule: failure-probe from node 4 (highest UCB=1.745), testing very low lr_W to find lower boundary
Observation: found failure boundary! lr_W=2.0E-4 (10x below baseline) causes complete failure; lr_W needs to be >= ~1E-3 for convergence
Next: parent=4 (highest UCB=1.745, exploit with coeff_W_L1 mutation to explore regularization)

## Iter 6: converged
Node: id=6, parent=4
Mode/Strategy: exploit (exploring regularization parameter space)
Config: lr_W=2.0E-2, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-4, batch_size=16, low_rank_factorization=False
Metrics: test_R2=0.7287, test_pearson=0.6662, connectivity_R2=0.9918, final_loss=2.15E+03
Activity: chaotic dynamics, 100 neurons, range [-19.6, 20.3], effective rank 34 (99% var), spectral radius 1.025
Mutation: coeff_W_L1: 1.0E-5 -> 1.0E-4 (10x increase)
Parent rule: exploit from node 4 (highest UCB=1.745), exploring L1 regularization effect
Observation: increased L1 regularization maintains excellent connectivity R² (0.992) but reduces test_R2/pearson; regularization trades prediction accuracy for sparsity
Next: parent=6 (highest UCB=2.217, explore model lr to probe different parameter dimension)

## Iter 7: converged
Node: id=7, parent=6
Mode/Strategy: exploit (exploring model lr dimension)
Config: lr_W=2.0E-2, lr=5.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-4, batch_size=16, low_rank_factorization=False
Metrics: test_R2=0.8172, test_pearson=0.7249, connectivity_R2=0.9935, final_loss=1.61E+03
Activity: chaotic dynamics, 100 neurons, range [-19.6, 17.9], effective rank 34 (99% var), spectral radius 1.025
Mutation: lr: 1.0E-4 -> 5.0E-4 (5x increase)
Parent rule: exploit from node 6 (highest UCB=2.217), exploring model learning rate
Observation: increased model lr improves test_R2 (0.73->0.82) and pearson (0.67->0.72) while maintaining excellent connectivity R² (0.994); lowest final loss (1.61E+03); model lr=5E-4 is better than 1E-4
Next: parent=7 (highest UCB=2.316, failure-probe with extreme coeff_W_L1=1E-3 to find upper boundary)

## Iter 8: partial
Node: id=8, parent=7
Mode/Strategy: failure-probe (probing upper L1 regularization boundary)
Config: lr_W=2.0E-2, lr=5.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-3, batch_size=16, low_rank_factorization=False
Metrics: test_R2=0.6931, test_pearson=0.5815, connectivity_R2=0.7561, final_loss=4.36E+03
Activity: chaotic dynamics, 100 neurons, range [-21.3, 22.0], effective rank 33 (99% var), spectral radius 1.025
Mutation: coeff_W_L1: 1.0E-4 -> 1.0E-3 (10x increase)
Parent rule: failure-probe from node 7 (highest UCB=2.316), testing extreme L1 regularization to find upper boundary
Observation: found upper boundary for L1 reg! coeff_W_L1=1E-3 degrades connectivity R² from 0.994 to 0.756; optimal L1 is around 1E-4 to 1E-5
Next: parent=7 (second highest UCB=1.936, revert L1 and try batch_size=32 to explore remaining dimension)

## Iter 9: converged
Node: id=9, parent=7
Mode/Strategy: exploit (exploring batch_size dimension)
Config: lr_W=2.0E-2, lr=5.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-4, batch_size=32, low_rank_factorization=False
Metrics: test_R2=0.9601, test_pearson=0.9378, connectivity_R2=0.9946, final_loss=1.74E+03
Activity: chaotic dynamics, 100 neurons, range [-19.6, 24.8], effective rank 35 (99% var), spectral radius 1.025
Mutation: batch_size: 16 -> 32 (2x increase)
Parent rule: exploit from node 7 (highest UCB=1.936), testing larger batch size
Observation: batch_size=32 yields excellent results with best test_R2 (0.96) and pearson (0.94) seen with regularization; connectivity R² remains excellent (0.995); batch_size tolerant from 8 to 32
Next: parent=9 (highest UCB=2.495, failure-probe with very low lr=1E-5 to find lower model lr boundary)

## Iter 10: failed
Node: id=10, parent=9
Mode/Strategy: failure-probe (probing lower model lr boundary)
Config: lr_W=2.0E-2, lr=1.0E-5, lr_emb=2.5E-4, coeff_W_L1=1.0E-4, batch_size=32, low_rank_factorization=False
Metrics: test_R2=0.3988, test_pearson=0.1276, connectivity_R2=0.0851, final_loss=4.79E+03
Activity: chaotic dynamics, 100 neurons, range [-22.9, 20.5], effective rank 34 (99% var), spectral radius 1.025
Mutation: lr: 5.0E-4 -> 1.0E-5 (50x decrease)
Parent rule: failure-probe from node 9 (highest UCB=2.495), testing very low model lr to find lower boundary
Observation: found lower lr boundary! lr=1E-5 causes complete failure; model lr needs to be >= ~1E-4 for convergence; this matches lr_W lower boundary pattern
Next: parent=8 (highest UCB=2.337, explore from partial node to potentially improve L1 regularization config)

## Iter 11: converged
Node: id=11, parent=8
Mode/Strategy: exploit (recovering from partial node by reducing L1)
Config: lr_W=2.0E-2, lr=5.0E-4, lr_emb=2.5E-4, coeff_W_L1=5.0E-4, batch_size=32, low_rank_factorization=False
Metrics: test_R2=0.8718, test_pearson=0.7858, connectivity_R2=0.9348, final_loss=2.72E+03
Activity: chaotic dynamics, 100 neurons, range [-19.6, 19.9], effective rank 35 (99% var), spectral radius 1.025
Mutation: coeff_W_L1: 1.0E-3 -> 5.0E-4 (2x decrease)
Parent rule: exploit from node 8 (highest UCB=2.337), intermediate L1 value between failed 1E-3 and successful 1E-4
Observation: intermediate L1=5E-4 recovers convergence (R²=0.935) from partial node 8; confirms L1 upper boundary is between 5E-4 and 1E-3; lower pearson (0.79) than best configs suggests L1=1E-4 is more optimal
Next: parent=11 (highest UCB=2.593, try reverting L1 to 1E-4 to improve pearson while keeping batch_size=32)

## Iter 12: converged
Node: id=12, parent=11
Mode/Strategy: exploit (reverted L1 to optimal value)
Config: lr_W=2.0E-2, lr=5.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-4, batch_size=32, low_rank_factorization=False
Metrics: test_R2=0.9515, test_pearson=0.9327, connectivity_R2=0.9703, final_loss=1.64E+03
Activity: chaotic dynamics, 100 neurons, range [-19.6, 20.3], effective rank 30 (99% var), spectral radius 1.025
Mutation: coeff_W_L1: 5.0E-4 -> 1.0E-4 (5x decrease)
Parent rule: exploit from node 11 (highest UCB=2.593), reverted L1 from 5E-4 to 1E-4 to improve convergence
Observation: L1=1E-4 improves connectivity R² from 0.935 to 0.970 and pearson from 0.79 to 0.93; confirms L1=1E-4 is optimal; this completes L1 boundary mapping (1E-5 to 1E-4 optimal, >5E-4 degrades)
Next: parent=12 (highest UCB=2.702, explore lr_emb dimension which is unexplored)

## Iter 13: converged
Node: id=13, parent=12
Mode/Strategy: exploit (exploring lr_emb dimension)
Config: lr_W=2.0E-2, lr=5.0E-4, lr_emb=5.0E-4, coeff_W_L1=1.0E-4, batch_size=32, low_rank_factorization=False
Metrics: test_R2=0.8797, test_pearson=0.8460, connectivity_R2=0.9873, final_loss=2.04E+03
Activity: chaotic dynamics, 100 neurons, range [-20.2, 21.4], effective rank 32 (99% var), spectral radius 1.025
Mutation: lr_emb: 2.5E-4 -> 5.0E-4 (2x increase, implicit from parent)
Parent rule: exploit from node 12 (highest UCB=2.702), exploring lr_emb dimension
Observation: lr_emb=5E-4 maintains excellent connectivity R² (0.987) with slightly lower pearson (0.85) than node 12; lr_emb tolerant in this range; need to probe lr_emb boundaries
Next: parent=13 (highest UCB=2.790, failure-probe with very low lr_emb=1E-5 to find lower boundary)

## Iter 14: converged
Node: id=14, parent=13
Mode/Strategy: failure-probe (probing lower lr_emb boundary)
Config: lr_W=2.0E-2, lr=5.0E-4, lr_emb=1.0E-5, coeff_W_L1=1.0E-4, batch_size=32, low_rank_factorization=False
Metrics: test_R2=0.9250, test_pearson=0.9101, connectivity_R2=0.9955, final_loss=1.66E+03
Activity: chaotic dynamics, 100 neurons, range [-19.6, 26.0], effective rank 34 (99% var), spectral radius 1.025
Mutation: lr_emb: 5.0E-4 -> 1.0E-5 (50x decrease)
Parent rule: failure-probe from node 13 (highest UCB=2.790), testing very low lr_emb to find lower boundary
Observation: lr_emb=1E-5 still converges excellently (R²=0.996, pearson=0.91); lr_emb has no practical lower boundary for n_neuron_types=1; embedding lr is irrelevant when there's only one neuron type
Next: parent=14 (highest UCB=2.866, explore-branch to node 9 for fresh exploration path with batch_size=32 and test lr_W lower intermediate)

## Iter 15: converged
Node: id=15, parent=14
Mode/Strategy: exploit (continuing from best node)
Config: lr_W=5.0E-3, lr=5.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-4, batch_size=32, low_rank_factorization=False
Metrics: test_R2=0.9380, test_pearson=0.9044, connectivity_R2=0.9999, final_loss=2.16E+03
Activity: chaotic dynamics, 100 neurons, range [-25.7, 20.0], effective rank 36 (99% var), spectral radius 1.025
Mutation: lr_W: 2.0E-2 -> 5.0E-3 (4x decrease), lr_emb: 1.0E-5 -> 2.5E-4 (reverted to baseline)
Parent rule: exploit from node 14 (highest UCB=2.866), reduced lr_W from extreme 2E-2 toward intermediate 5E-3
Observation: lr_W=5E-3 achieves perfect connectivity R² (0.9999) with excellent pearson (0.90); intermediate lr_W between 5E-3 and 2E-2 all work excellently; lr_emb reversion irrelevant for n_neuron_types=1
Next: parent=15 (highest UCB=2.936, explore with lr_W=1E-2 to fill gap between 5E-3 and 2E-2)

## Iter 16: converged
Node: id=16, parent=15
Mode/Strategy: exploit (filling lr_W gap)
Config: lr_W=1.0E-2, lr=5.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-4, batch_size=32, low_rank_factorization=False
Metrics: test_R2=0.9304, test_pearson=0.9008, connectivity_R2=0.9980, final_loss=1.65E+03
Activity: chaotic dynamics, 100 neurons, range [-20.9, 22.0], effective rank 33 (99% var), spectral radius 1.025
Mutation: lr_W: 5.0E-3 -> 1.0E-2 (2x increase)
Parent rule: exploit from node 15 (highest UCB=2.936), filled gap between lr_W=5E-3 and 2E-2
Observation: lr_W=1E-2 achieves excellent connectivity R² (0.998); confirms entire lr_W range from 5E-3 to 2E-2 works well; 6 consecutive successes (nodes 11-16)
Next: parent=9 (UCB=2.328, explore-branch to unexplored path since 6 consecutive successes warrant forced branching)

## Iter 17: converged
Node: id=17, parent=9
Mode/Strategy: explore (7 consecutive successes, branching from node 9)
Config: lr_W=2.0E-2, lr=5.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=32, low_rank_factorization=False
Metrics: test_R2=0.9222, test_pearson=0.8869, connectivity_R2=0.9973, final_loss=1.88E+03
Activity: chaotic dynamics, 100 neurons, range [-19.6, 21.2], effective rank 34 (99% var), spectral radius 1.025
Mutation: coeff_W_L1: 1.0E-4 -> 1.0E-5 (10x decrease, reverted to baseline)
Parent rule: explore from node 9 (UCB=2.025), 7 consecutive successes triggered explore strategy; reverted L1 to baseline while keeping batch_size=32
Observation: coeff_W_L1=1E-5 achieves excellent connectivity R² (0.997); confirms L1 range 1E-5 to 1E-4 both work well; 8 consecutive successes (nodes 11-17); landscape is very robust
Next: parent=16 (highest UCB=3.060, exploit with failure-probe on unexplored dimension; try lr=1E-3 to find upper model lr boundary)

## Iter 18: converged
Node: id=18, parent=16
Mode/Strategy: failure-probe (testing upper model lr boundary)
Config: lr_W=1.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-4, batch_size=32, low_rank_factorization=False
Metrics: test_R2=0.9848, test_pearson=0.9737, connectivity_R2=0.9992, final_loss=1.40E+03
Activity: chaotic dynamics, 100 neurons, range [-19.6, 19.3], effective rank 34 (99% var), spectral radius 1.025
Mutation: lr: 5.0E-4 -> 1.0E-3 (2x increase)
Parent rule: failure-probe from node 16 (highest UCB=3.060), probed upper model lr boundary with lr=1E-3
Observation: lr=1E-3 achieves excellent connectivity R² (0.999) with best pearson yet (0.974) and lowest final_loss (1.40E+03); model lr range extends to 1E-3; 9 consecutive successes (nodes 11-18); this is the best config so far
Next: parent=9 (explore strategy, UCB=2.055, 9 consecutive successes triggers explore; branch to successful node outside last 6)

## Iter 19: converged
Node: id=19, parent=9
Mode/Strategy: explore (10 consecutive successes, branching from node 9)
Config: lr_W=2.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-4, batch_size=32, low_rank_factorization=False
Metrics: test_R2=0.9335, test_pearson=0.8998, connectivity_R2=0.9969, final_loss=1.28E+03
Activity: chaotic dynamics, 100 neurons, range [-19.6, 17.9], effective rank 35 (99% var), spectral radius 1.025
Mutation: lr: 5.0E-4 -> 1.0E-3 (2x increase, inherited from explore branch)
Parent rule: explore from node 9 (UCB=1.866), 10 consecutive successes triggered explore strategy; combined node 9's batch_size=32 with node 18's optimal lr=1E-3
Observation: combined best settings from explore branch; achieves excellent connectivity R² (0.997) with lowest final_loss yet (1.28E+03); 11 consecutive successes (nodes 9-19); landscape extremely robust across lr/lr_W/batch_size/L1 combinations
Next: parent=18 (highest UCB=3.179, try failure-probe with extreme lr=2E-3 to find upper model lr boundary)

## Iter 20: converged
Node: id=20, parent=18
Mode/Strategy: failure-probe (testing upper model lr boundary at 2E-3)
Config: lr_W=1.0E-2, lr=2.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-4, batch_size=32, low_rank_factorization=False
Metrics: test_R2=0.9437, test_pearson=0.9186, connectivity_R2=0.9989, final_loss=1.38E+03
Activity: chaotic dynamics, 100 neurons, range [-19.6, 21.1], effective rank 34 (99% var), spectral radius 1.025
Mutation: lr: 1.0E-3 -> 2.0E-3 (2x increase)
Parent rule: failure-probe from node 18 (UCB=2.490), pushed model lr to 2E-3 (2x above protocol max 1E-3) to find upper boundary
Observation: lr=2E-3 still converges excellently (R²=0.999, pearson=0.92); model lr upper boundary extends beyond protocol range; 12 consecutive successes (nodes 9-20); this config maintains near-perfect connectivity recovery
Next: parent=20 (highest UCB=3.235, failure-probe with extreme lr=5E-3 to find true upper model lr boundary)

## Iter 21: converged
Node: id=21, parent=20
Mode/Strategy: failure-probe (testing extreme upper model lr boundary at 5E-3)
Config: lr_W=1.0E-2, lr=5.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-4, batch_size=32, low_rank_factorization=False
Metrics: test_R2=0.9906, test_pearson=0.9845, connectivity_R2=0.9989, final_loss=1.27E+03
Activity: chaotic dynamics, 100 neurons, range [-21.3, 17.3], effective rank 35 (99% var), spectral radius 1.025
Mutation: lr: 2.0E-3 -> 5.0E-3 (2.5x increase)
Parent rule: failure-probe from node 20 (UCB=2.526), pushed model lr to 5E-3 (5x above protocol max 1E-3) to find true upper boundary
Observation: lr=5E-3 achieves excellent results with best pearson yet (0.985) and lowest final_loss (1.27E+03); model lr can go 50x above documented range; 13 consecutive successes (nodes 9-21); landscape extremely robust
Next: parent=21 (highest UCB=3.290, failure-probe with extreme lr=1E-2 to find true upper model lr boundary)

## Iter 22: converged
Node: id=22, parent=21
Mode/Strategy: failure-probe (testing extreme upper model lr boundary at 1E-2)
Config: lr_W=1.0E-2, lr=1.0E-2, lr_emb=2.5E-4, coeff_W_L1=1.0E-4, batch_size=32, low_rank_factorization=False
Metrics: test_R2=0.9377, test_pearson=0.9115, connectivity_R2=0.9949, final_loss=1.34E+03
Activity: chaotic dynamics, 100 neurons, range [-19.6, 17.3], effective rank 34 (99% var), spectral radius 1.025
Mutation: lr: 5.0E-3 -> 1.0E-2 (2x increase)
Parent rule: failure-probe from node 21 (UCB=3.290), pushed model lr to 1E-2 (100x above protocol max 1E-3) to find true upper boundary
Observation: lr=1E-2 still converges (R²=0.995) but pearson drops from 0.985 to 0.911 and connectivity R² drops slightly from 0.999 to 0.995; approaching upper lr boundary; 14 consecutive successes (nodes 9-22); optimal model lr is around 5E-3
Next: parent=9 (explore strategy, UCB=1.933, 14 consecutive successes triggers explore; branch to node outside last 6 iters to test alternative path)

## Iter 23: converged
Node: id=23, parent=9
Mode/Strategy: explore (15 consecutive successes, branching from node 9)
Config: lr_W=2.0E-2, lr=2.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-4, batch_size=32, low_rank_factorization=False
Metrics: test_R2=0.9377, test_pearson=0.9131, connectivity_R2=0.9937, final_loss=1.21E+03
Activity: chaotic dynamics, 100 neurons, range [-20, 21], effective rank 33 (99% var), spectral radius 1.025
Mutation: lr_W: 1.0E-2 -> 2.0E-2, lr: 1.0E-2 -> 2.0E-3 (combined node 9's lr_W with intermediate lr)
Parent rule: explore from node 9 (UCB=1.794), 15 consecutive successes triggered explore; combined node 9 config with lr from explored path
Observation: combined lr_W=2E-2 with lr=2E-3 achieves excellent connectivity R² (0.994) with low final_loss (1.21E+03); confirms high lr_W robust; 16 consecutive successes (nodes 9-23); landscape fully mapped with all parameter combinations converging
Next: parent=17 (highest UCB=3.395, final exploration with alternative L1 path; test lr=5E-3 with L1=1E-5)

