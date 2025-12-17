# Experiment Log: signal_Claude

## Iter 1: partial
--- NEW BLOCK ---
Simulation: connectivity_type=chaotic, Dale_law=False, noise_model_level=0.0
Node: id=1, parent=root
Mode: success-exploit
Strategy: explore
Config: lr_W=2.0E-3, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.6408, test_pearson=0.4699, connectivity_R2=0.8621, final_loss=1.2272e+03
Activity: oscillatory dynamics across 100 neurons with moderate variability, effective rank 13 (90% var), spectral radius 1.039
Mutation: baseline config (first run)
Observation: solid partial convergence (R2=0.862) on first try; lr_W increase may push toward convergence

## Iter 2: converged
Node: id=2, parent=1
Mode: success-exploit
Strategy: exploit
Config: lr_W=4.0E-3, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9800, test_pearson=0.9732, connectivity_R2=0.9578, final_loss=5.51e+02
Activity: oscillatory dynamics across 100 neurons, effective rank 12 (90% var), spectral radius 1.025
Mutation: lr_W: 2.0E-3 -> 4.0E-3
Observation: doubling lr_W achieved convergence (R2=0.958); 2 consecutive successes, continue exploiting

## Iter 3: converged
Node: id=3, parent=2
Mode: success-exploit
Strategy: exploit
Config: lr_W=6.0E-3, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9891, test_pearson=0.9838, connectivity_R2=0.9506, final_loss=5.27e+02
Activity: oscillatory dynamics across 100 neurons, effective rank 11 (90% var), range [-26.3, 23.6], spectral radius 1.025
Mutation: lr_W: 4.0E-3 -> 6.0E-3
Observation: lr_W=6E-3 maintains convergence (R2=0.951); 3 consecutive successes, switching to failure-probe

## Iter 4: converged
Node: id=4, parent=3
Mode: failure-probe
Strategy: boundary
Config: lr_W=2.0E-2, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9737, test_pearson=0.9644, connectivity_R2=0.9988, final_loss=2.18e+02
Activity: oscillatory dynamics across 100 neurons, effective rank 11 (90% var), range [-26.3, 23.6], spectral radius 1.025
Mutation: lr_W: 6.0E-3 -> 2.0E-2 (3.3x increase, probing upper boundary)
Observation: lr_W=2E-2 still converges excellently (R2=0.999); upper boundary not yet found, continue probing higher

## Iter 5: converged
Node: id=5, parent=4
Mode: failure-probe
Strategy: boundary
Config: lr_W=5.0E-2, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9609, test_pearson=0.9385, connectivity_R2=0.9929, final_loss=2.96e+02
Activity: oscillatory dynamics across 100 neurons, effective rank 11 (90% var), range [-26.3, 21.6], spectral radius 1.025
Mutation: lr_W: 2.0E-2 -> 5.0E-2 (2.5x increase, continuing boundary probe)
Observation: lr_W=5E-2 still converges (R2=0.993) but pearson dropped 0.964->0.939; may be approaching upper limit

