# Experiment Log: signal_Claude

## Iter 1: partial
--- NEW BLOCK ---
Simulation: connectivity_type=chaotic, Dale_law=False, noise_model_level=0.0
Node: id=1, parent=root
Mode: success-exploit
Strategy: explore
Config: lr_W=2.0E-3, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9226, test_pearson=0.9329, connectivity_R2=0.7345, final_loss=1.65e+03
Activity: healthy oscillatory dynamics, range [-23.3, 20.3], effective rank 36
Mutation: baseline config (first run)
Observation: partial convergence; good activity prediction but connectivity recovery needs improvement

## Iter 2: converged
Node: id=2, parent=1
Mode: success-exploit
Strategy: exploit
Config: lr_W=5.0E-3, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9572, test_pearson=0.9304, connectivity_R2=0.9954, final_loss=4.13e+02
Activity: healthy oscillatory dynamics, range [-26.3, 19.2], effective rank 12 (90%) / 34 (99%)
Mutation: lr_W: 2.0E-3 -> 5.0E-3
Observation: excellent convergence; 2.5x higher lr_W dramatically improved connectivity recovery

