# Experiment Log: signal_Claude

## Iter 1: converged
--- NEW SIMULATION BLOCK ---
Simulation: connectivity_type=chaotic, Dale_law=False, noise_model_level=0
Node: id=1, parent=root
Mode/Strategy: exploit
Config: lr_W=2.0E-3, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9981, test_pearson=0.9937, connectivity_R2=0.9914, final_loss=5.72E+02
Activity: chaotic dynamics, range [-20.3, 23.9], effective rank 30, 100 neurons distinguishable
Mutation: baseline config (first iteration)
Parent rule: UCB file shows only Node 1 as current; this was baseline from root
Observation: excellent convergence on baseline chaotic config, GNN successfully recovers connectivity
Next: parent=1

