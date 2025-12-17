# Experiment Log: signal_Claude

## Iter 1: converged
Node: id=1, parent=root
Mode: success-exploit
Strategy: baseline
Config: lr_W=2.0E-3, lr=1.0E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9987, test_pearson=0.9980, connectivity_R2=0.9999, final_loss=4.44E+02
Activity: range=[-19.1, 20.3], effective_rank=13-36, spectral_radius=1.05
Mutation: baseline config (starting point)
Observation: excellent convergence with baseline - this establishes our working region center

## Iter 2: converged
Node: id=2, parent=1
Mode: success-exploit
Strategy: explore
Config: lr_W=5.0E-3, lr=1.0E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.8466, test_pearson=0.8062, connectivity_R2=0.9999, final_loss=2.09E+02
Activity: range=[-23.0, 20.5], effective_rank=11-33, spectral_radius=1.025
Mutation: lr_W: 2.0E-3 → 5.0E-3 (2.5x increase)
Observation: higher lr_W still converges perfectly (connectivity_R2=0.9999) with lower final_loss, but slightly lower test_R2 - faster W learning trades off some dynamics prediction

## Iter 3: converged
Node: id=3, parent=2
Mode: failure-probe
Strategy: boundary (push lr_W to extreme)
Config: lr_W=1.0E-2, lr=1.0E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9333, test_pearson=0.8991, connectivity_R2=0.9998, final_loss=2.70E+02
Activity: range=[-20.6, 18.6], effective_rank=12-33, spectral_radius=1.025
Mutation: lr_W: 5.0E-3 → 1.0E-2 (2x increase)
Observation: lr_W=1E-2 still converges well - upper boundary not yet found, need to push higher

## Iter 4: converged
Node: id=4, parent=3
Mode: failure-probe
Strategy: boundary (push lr_W to extreme)
Config: lr_W=2.0E-2, lr=1.0E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9026, test_pearson=0.8646, connectivity_R2=0.9993, final_loss=3.59E+02
Activity: range=[-19.6, 20.0], effective_rank=10-33, spectral_radius=1.025
Mutation: lr_W: 1.0E-2 → 2.0E-2 (2x increase)
Observation: lr_W=2E-2 still converges (connectivity_R2=0.9993) but test_R2 dropped to 0.90 - approaching upper boundary, lr_W sweet spot appears to be around 1E-2

## Iter 5: converged
Node: id=5, parent=4
Mode: success-exploit
Strategy: explore
Config: lr_W=2.0E-2, lr=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9011, test_pearson=0.8709, connectivity_R2=0.9930, final_loss=4.16E+02
Activity: range=[-20.4, 20.4], effective_rank=10-33, spectral_radius=1.025
Mutation: lr: 1.0E-4 → 2.5E-4 (2.5x increase)
Observation: higher lr with high lr_W maintains convergence (connectivity_R2=0.993), similar performance - lr=2.5E-4 is compatible with lr_W=2E-2

## Iter 6: converged
Node: id=6, parent=5
Mode: failure-probe
Strategy: boundary (push lr_W to extreme)
Config: lr_W=5.0E-2, lr=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.7126, test_pearson=0.5874, connectivity_R2=0.9916, final_loss=6.12E+02
Activity: range=[-19.6, 21.5], effective_rank=10-33, spectral_radius=1.025
Mutation: lr_W: 2.0E-2 → 5.0E-2 (2.5x increase)
Observation: lr_W=5E-2 still converges for W (connectivity_R2=0.992) but test_R2 dropped to 0.71 - dynamics prediction significantly degraded. this marks the practical upper boundary for lr_W where W learning is too fast relative to dynamics learning

