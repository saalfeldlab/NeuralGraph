# Experiment Log: signal_Claude

## Iter 1: partial
Node: id=1, parent=root
Mode: success-exploit
Strategy: explore
Config: lr_W=2.0E-3, lr=1.0E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.7322, test_pearson=0.6789, connectivity_R2=0.7578, final_loss=1.4041e+03
Activity: good oscillatory dynamics, effective rank 12-36, range [-18.9, 20.2]
Mutation: baseline config (first run)
Observation: partial convergence with decent W recovery; lr may be too low for full convergence

## Iter 2: partial
Node: id=2, parent=1
Mode: success-exploit
Strategy: exploit
Config: lr_W=2.0E-3, lr=5.0E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.7755, test_pearson=0.7311, connectivity_R2=0.8730, final_loss=7.5457e+02
Activity: healthy oscillatory dynamics, effective rank 12-33, range [-26.3, 19.4]
Mutation: lr: 1.0E-4 → 5.0E-4 (5x increase)
Observation: significant improvement in W recovery (0.758 → 0.873); higher lr helps convergence, approaching threshold

