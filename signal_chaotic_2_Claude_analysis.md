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

