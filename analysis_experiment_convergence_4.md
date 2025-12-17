# Experiment Log: signal_Claude

## Iter 1: partial
--- NEW BLOCK ---
Simulation: connectivity_type=chaotic, Dale_law=False, noise_model_level=0.0
Node: id=1, parent=root
Mode: success-exploit

Strategy: explore
Config: lr_W=2.0E-3, lr=1.0E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.4022, test_pearson=0.2498, connectivity_R2=0.6731, final_loss=1.09e+03
Activity: healthy oscillatory dynamics with effective rank 8-28; range [-25.7, 19.8]
Mutation: baseline config (first run of block 0)
Observation: partial convergence; lr_W may need increase to push connectivity_R2 higher

## Iter 2: converged
Node: id=2, parent=1
Mode: success-exploit
Strategy: exploit
Config: lr_W=5.0E-3, lr=1.0E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9638, test_pearson=0.9398, connectivity_R2=0.9838, final_loss=4.05e+02
Activity: healthy oscillatory dynamics with effective rank 11-32; range [-26.3, 23.9]
Mutation: lr_W: 2.0E-3 -> 5.0E-3
Observation: excellent convergence! increasing lr_W from 2E-3 to 5E-3 dramatically improved R2 from 0.67 to 0.98

## Iter 3: failed
Node: id=3, parent=2
Mode: failure-probe
Strategy: explore
Config: lr_W=5.0E-3, lr=2.0E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.3377, test_pearson=-0.0042, connectivity_R2=0.1410, final_loss=1.58e+03
Activity: healthy oscillatory dynamics with effective rank 11-31; range [-26.3, 26.9]
Mutation: lr: 1.0E-4 -> 2.0E-4
Observation: severe degradation from doubling lr; model lr highly sensitive - 2x increase caused R2 to drop from 0.98 to 0.14

## Iter 4: converged
Node: id=4, parent=2
Mode: success-exploit
Strategy: exploit
Config: lr_W=1.0E-2, lr=1.0E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9488, test_pearson=0.9293, connectivity_R2=0.9982, final_loss=1.95e+02
Activity: healthy oscillatory dynamics with effective rank 11-32; range [-26.3, 23.2]
Mutation: lr_W: 5.0E-3 -> 1.0E-2
Observation: best result so far! further increasing lr_W from 5E-3 to 1E-2 improved connectivity_R2 from 0.984 to 0.998

