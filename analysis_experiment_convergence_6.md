# Experiment Log: signal_Claude

## Iter 1: converged
--- NEW SIMULATION BLOCK ---
Simulation: connectivity_type=chaotic, Dale_law=False, noise_model_level=0
Node: id=1, parent=root
Mode/Strategy: exploit
Config: lr_W=2.0E-3, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9989, test_pearson=0.9987, connectivity_R2=0.9991, final_loss=5.26e+02
Activity: smooth oscillatory dynamics, range [-19.7, 22.9], effective rank 32 (99% var), spectral_radius=1.076
Mutation: baseline config (no mutation)
Parent rule: UCB file shows Node 1 as only node, first iteration of block
Observation: excellent convergence on baseline chaotic config, connectivity nearly perfectly recovered
Next: parent=1 (exploit successful config with small mutation)

## Iter 2: converged
Node: id=2, parent=1
Mode/Strategy: exploit
Config: lr_W=3.0E-3, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9893, test_pearson=0.9820, connectivity_R2=0.9830, final_loss=5.45e+02
Activity: smooth oscillatory dynamics, range [-19.6, 22.9], effective rank 33 (99% var), spectral_radius=1.025
Mutation: lr_W: 2.0E-3 -> 3.0E-3
Parent rule: highest UCB was Node 1 (only option), explored lr_W increase
Observation: lr_W increase slightly degraded R2 (0.999->0.983), suggesting 2.0E-3 was closer to optimal
Next: parent=2 (UCB=1.690 > Node 1 UCB=1.471, try different mutation direction)

## Iter 3: converged
Node: id=3, parent=2
Mode/Strategy: exploit
Config: lr_W=3.0E-3, lr=2.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9523, test_pearson=0.9310, connectivity_R2=0.9914, final_loss=1.79e+02
Activity: smooth oscillatory dynamics, range [-22.1, 20.4], effective rank 33 (99% var), spectral_radius=1.025
Mutation: lr: 1.0E-4 -> 2.0E-4
Parent rule: Node 2 had highest UCB=1.690, tried lr increase
Observation: lr increase improved connectivity_R2 (0.983->0.991) and reduced loss (545->179), but test_R2 dropped (0.989->0.952)
Next: parent=3 (highest UCB=1.857, 3 consecutive successes triggers failure-probe strategy)

