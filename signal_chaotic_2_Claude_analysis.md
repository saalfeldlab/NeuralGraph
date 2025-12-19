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

