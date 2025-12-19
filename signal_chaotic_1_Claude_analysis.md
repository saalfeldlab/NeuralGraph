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

