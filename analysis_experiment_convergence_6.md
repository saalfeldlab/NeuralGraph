# Experiment Log: signal_Claude

## Iter 1: converged
--- NEW SIMULATION BLOCK ---
Simulation: connectivity_type=chaotic, Dale_law=False, noise_model_level=0
Node: id=1, parent=root
Mode/Strategy: baseline/exploit
Config: lr_W=2.0E-3, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9990, test_pearson=0.9977, connectivity_R2=0.9989, final_loss=5.35E+02
Activity: rich oscillatory dynamics across 100 neurons, range [-22.5, 19.9], effective rank 11 (90% var)
Mutation: baseline config (no mutation)
Parent rule: UCB empty at block start, parent=root
Observation: excellent baseline convergence with chaotic connectivity, strong connectivity recovery
Next: parent=1 (highest UCB, exploit strategy)

