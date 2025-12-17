# Experiment Log: signal_Claude

## Iter 1: partial
--- NEW BLOCK ---
Simulation: connectivity_type=chaotic, Dale_law=False, noise_model_level=0.0
Node: id=1, parent=root
Mode: success-exploit
Strategy: explore
Config: lr_W=2.0E-3, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9368, test_pearson=0.9151, connectivity_R2=0.7792, final_loss=1.15e+03
Activity: healthy oscillatory dynamics, effective rank 10 (90% var), range [-20, 22]
Mutation: baseline config (first run)
Observation: good activity prediction but connectivity learning stuck at 78%; try higher lr_W

