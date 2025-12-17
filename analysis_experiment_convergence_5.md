# Experiment Log: signal_Claude

## Iter 1: partial
--- NEW BLOCK ---
Simulation: connectivity_type=chaotic, Dale_law=False, noise_model_level=0.0
Node: id=1, parent=root
Mode: baseline
Strategy: explore
Config: lr_W=2.0E-3, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.5393, test_pearson=0.5997, connectivity_R2=0.5248, final_loss=2.0796e+03
Activity: oscillatory dynamics, range [-22.6, 22.3], effective rank 35 (99% var), spectral_radius=1.06
Mutation: baseline config (no mutation)
Observation: partial convergence with chaotic connectivity; connectivity learning stalled at ~52%

## Iter 2: converged
Node: id=2, parent=1
Mode: success-exploit
Strategy: exploit
Config: lr_W=5.0E-3, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9585, test_pearson=0.9336, connectivity_R2=0.9420, final_loss=7.7699e+02
Activity: oscillatory dynamics, range [-26.3, 22.7], effective rank 33 (99% var), spectral_radius=1.025
Mutation: lr_W: 2.0E-3 -> 5.0E-3
Observation: strong convergence achieved by increasing lr_W 2.5x; connectivity learning improved from 52% to 94%

