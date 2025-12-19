# Experiment Log: signal_chaotic_2_Claude

## Iter 1: partial
Node: id=1, parent=root
Mode/Strategy: exploit
Config: lr_W=0.002, lr=0.0001, lr_emb=0.00025, coeff_W_L1=1e-05, batch_size=8, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.4754, test_pearson=0.4804, connectivity_R2=0.2776, final_loss=1.5731e+04
Activity: chaotic dynamics, spectral_radius=1.014 (critical), effective_rank=35, range [-28, 34]
Mutation: baseline (no mutation, first iter)
Parent rule: first iteration, parent=root
Observation: low connectivity_R2 despite moderate test_R2, model predicts dynamics but fails to recover weights
Next: parent=1, increase lr_W to improve weight recovery

## Iter 2: partial
Node: id=2, parent=1
Mode/Strategy: exploit
Config: lr_W=0.005, lr=0.0001, lr_emb=0.00025, coeff_W_L1=1e-05, batch_size=8, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.6852, test_pearson=0.4373, connectivity_R2=0.4576, final_loss=1.1818e+04
Activity: chaotic dynamics, spectral_radius=1.025 (critical), effective_rank=32, range [-29, 25]
Mutation: lr_W: 0.002 -> 0.005
Parent rule: node 2 has highest UCB (1.165), exploit best node
Observation: lr_W increase improved connectivity_R2 significantly (0.278 -> 0.458), test_R2 also improved, continue pushing lr_W
Next: parent=2, push lr_W to upper bound (0.01) to test if higher weight LR continues improving recovery

## Iter 3: partial
Node: id=3, parent=2
Mode/Strategy: exploit
Config: lr_W=0.01, lr=0.0001, lr_emb=0.00025, coeff_W_L1=1e-05, batch_size=8, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.4939, test_pearson=0.2576, connectivity_R2=0.6029, final_loss=8.9171e+03
Activity: chaotic dynamics, spectral_radius=1.025, effective_rank=31, range [-29, 38], sustained oscillations across all neurons
Mutation: lr_W: 0.005 -> 0.01
Parent rule: node 2 had highest UCB (1.035), exploiting best node
Observation: lr_W at upper bound improved connectivity_R2 (0.458 -> 0.603, +32%) but degraded test_R2 and pearson, suggesting overfitting to weights at expense of dynamics
Next: parent=3, try increasing coeff_W_L1 regularization to balance weight recovery vs dynamics prediction

## Iter 4: partial
Node: id=4, parent=3
Mode/Strategy: exploit
Config: lr_W=0.01, lr=0.0001, lr_emb=0.00025, coeff_W_L1=1e-04, batch_size=8, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.7195, test_pearson=0.3709, connectivity_R2=0.4255, final_loss=9.7971e+03
Activity: chaotic dynamics, spectral_radius=1.025, effective_rank=30, range [-29, 31], sustained oscillations
Mutation: coeff_W_L1: 1e-05 -> 1e-04
Parent rule: node 4 highest UCB (1.425), but mutation increased regularization
Observation: 10x L1 regularization hurt connectivity_R2 (0.603 -> 0.426, -29%) while improving test_R2 (0.494 -> 0.720), regularization suppresses weight magnitude recovery
Next: parent=3, revert to best R2 config, try increasing lr (model parameters) instead of regularization

## Iter 5: partial
Node: id=5, parent=3
Mode/Strategy: exploit
Config: lr_W=0.01, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1e-05, batch_size=8, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.8810, test_pearson=0.7847, connectivity_R2=0.7004, final_loss=6.7632e+03
Activity: chaotic dynamics, spectral_radius=1.025, effective_rank=27, range [-29, 26], sustained oscillations across all neurons
Mutation: lr: 0.0001 -> 0.0003
Parent rule: node 3 had highest UCB after reverting from failed regularization path
Observation: 3x increase in model lr significantly improved all metrics - connectivity_R2 (0.603 -> 0.700, +16%), test_R2 (0.494 -> 0.881), pearson (0.258 -> 0.785). best result so far.
Next: parent=5, continue exploiting this direction by pushing lr higher (toward 0.001 upper bound)

## Iter 6: converged
Node: id=6, parent=5
Mode/Strategy: exploit
Config: lr_W=0.01, lr=0.0006, lr_emb=0.00025, coeff_W_L1=1e-05, batch_size=8, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9165, test_pearson=0.7844, connectivity_R2=0.9728, final_loss=3.6218e+03
Activity: chaotic dynamics, spectral_radius=1.025, effective_rank=29, range [-29, 26], sustained irregular oscillations across all neurons
Mutation: lr: 0.0003 -> 0.0006
Parent rule: node 5 highest UCB (1.517), continuing exploitation of successful lr increase path
Observation: doubling model lr achieved convergence! connectivity_R2 jumped from 0.700 to 0.973 (+39%), first converged result. lr=0.0006 with lr_W=0.01 is effective combination.
Next: parent=6, exploit by pushing lr toward upper bound (0.001) or try robustness-test to verify this config

## Iter 7: converged
Node: id=7, parent=6
Mode/Strategy: exploit
Config: lr_W=0.01, lr=0.001, lr_emb=0.00025, coeff_W_L1=1e-05, batch_size=8, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.7042, test_pearson=0.3481, connectivity_R2=0.9781, final_loss=3.2122e+03
Activity: chaotic dynamics, spectral_radius=1.025, effective_rank=31, range [-29, 29], sustained irregular oscillations across all neurons
Mutation: lr: 0.0006 -> 0.001
Parent rule: node 7 highest UCB (2.301), continuing exploitation of converged path
Observation: pushing lr to upper bound (0.001) maintained excellent connectivity_R2 (0.973 -> 0.978, +0.5%) but significantly degraded test_R2 (0.917 -> 0.704, -23%) and pearson (0.784 -> 0.348, -56%). lr=0.001 is too aggressive for dynamics prediction.
Next: parent=6, revert to lr=0.0006 and try lr_emb increase to improve embedding learning while maintaining converged connectivity

## Iter 8: converged
Node: id=8, parent=6
Mode/Strategy: exploit
Config: lr_W=0.01, lr=0.0006, lr_emb=0.0005, coeff_W_L1=1e-05, batch_size=8, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.8669, test_pearson=0.6661, connectivity_R2=0.9868, final_loss=3.2522e+03
Activity: chaotic dynamics, spectral_radius=1.025, effective_rank=31, range [-30, 30], sustained irregular oscillations across all neurons
Mutation: lr_emb: 0.00025 -> 0.0005
Parent rule: node 6 had highest UCB after reverting from aggressive lr path, branching to explore lr_emb
Observation: doubling lr_emb improved connectivity_R2 (0.973 -> 0.987, +1.4%) and maintained good dynamics prediction (test_R2=0.867). best connectivity_R2 so far. 3 consecutive converged iterations (6,7,8).
Next: parent=8, use failure-probe strategy - push lr_W extreme low (0.001) to map failure boundary

