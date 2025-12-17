# Experiment Log: signal_Claude

## Iter 1: partial
--- NEW BLOCK ---
Simulation: connectivity_type=chaotic, Dale_law=False, noise_model_level=0.0
Node: id=1, parent=root
Mode: success-exploit
Strategy: explore
Config: lr_W=2.0E-3, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9277, test_pearson=0.8998, connectivity_R2=0.7646, final_loss=1.20e+03
Activity: healthy oscillatory dynamics, 100 neurons, effective rank 10-32, range [-21, 21]
Mutation: baseline config (first run)
Observation: good test_R2 but connectivity_R2=0.76 needs improvement; try increasing lr_W

## Iter 2: converged
Node: id=2, parent=1
Mode: success-exploit
Strategy: exploit
Config: lr_W=4.0E-3, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9944, test_pearson=0.9904, connectivity_R2=0.9239, final_loss=6.82e+02
Activity: healthy oscillatory dynamics, 100 neurons, effective rank 11-32, range [-26, 25]
Mutation: lr_W: 2.0E-3 → 4.0E-3
Observation: doubling lr_W achieved convergence; connectivity_R2 improved from 0.76 to 0.92; try further lr_W increase

## Iter 3: converged
Node: id=3, parent=2
Mode: success-exploit
Strategy: exploit
Config: lr_W=6.0E-3, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9773, test_pearson=0.9658, connectivity_R2=0.9957, final_loss=3.96e+02
Activity: stable oscillatory dynamics, 100 neurons, effective rank 12-33, range [-26, 21]
Mutation: lr_W: 4.0E-3 → 6.0E-3
Observation: excellent convergence; connectivity_R2=0.996 near-perfect; lr_W=6E-3 optimal so far; explore lr variations

## Iter 4: converged
Node: id=4, parent=3
Mode: success-exploit
Strategy: exploit
Config: lr_W=6.0E-3, lr=2.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9464, test_pearson=0.9339, connectivity_R2=0.9910, final_loss=2.70e+02
Activity: stable oscillatory dynamics, 100 neurons, effective rank 11-32, range [-26, 20]
Mutation: lr: 1.0E-4 → 2.0E-4
Observation: connectivity_R2 slightly decreased from 0.996 to 0.991 but still excellent; test_R2 dropped; try lr_W=8E-3 with lr=1E-4

## Iter 5: converged
Node: id=5, parent=4
Mode: success-exploit
Strategy: exploit
Config: lr_W=8.0E-3, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9648, test_pearson=0.9447, connectivity_R2=0.9957, final_loss=2.09e+02
Activity: stable oscillatory dynamics, 100 neurons, effective rank 11-32, range [-26, 20]
Mutation: lr_W: 6.0E-3 → 8.0E-3, lr: 2.0E-4 → 1.0E-4
Observation: best connectivity_R2=0.996 matches iter 3; lr_W=8E-3 with lr=1E-4 optimal; 4 consecutive successes - probe upper boundary

