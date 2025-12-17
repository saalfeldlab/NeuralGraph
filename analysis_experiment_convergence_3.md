# Experiment Log: signal_Claude

## Iter 1: partial
Node: id=51, parent=46
Mode: success-exploit
Strategy: explore
Config: lr_W=2.0E-3, lr=1.0E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.7763, test_pearson=0.7461, connectivity_R2=0.7201, final_loss=1.79E+03
Mutation: coeff_W_L1: 5.0E-6 → 1.0E-5 (from parent node 46)
Observation: partial convergence with higher L1 regularization - connectivity recovery degraded from parent's R2=1.0, suggesting coeff_W_L1=1.0E-5 may be too strong

