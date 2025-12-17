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

## Iter 3: failed
Node: id=3, parent=2
Mode: failure-probe
Strategy: boundary
Config: lr_W=2.0E-3, lr=5.0E-4, coeff_W_L1=5.0E-6, batch_size=8
Metrics: test_R2=0.3101, test_pearson=0.0612, connectivity_R2=0.0661, final_loss=2.2194e+03
Activity: oscillatory dynamics present, effective rank 11-33, range [-26.3, 23.4]
Failure mode: slow_convergence (loss 2x higher than node 2, W recovery collapsed)
Mutation: coeff_W_L1: 1.0E-5 → 5.0E-6 (halved L1 regularization)
Observation: drastic failure after reducing L1 reg; could be stochastic or L1=5E-6 is below stability threshold - retry to confirm

## Iter 4: converged
Node: id=4, parent=2
Mode: success-exploit
Strategy: exploit
Config: lr_W=2.0E-3, lr=1.0E-3, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9497, test_pearson=0.9195, connectivity_R2=0.9330, final_loss=5.9716e+02
Activity: healthy oscillatory dynamics, effective rank 12-33, range [-26.3, 19.2]
Mutation: lr: 5.0E-4 → 1.0E-3 (2x increase from node 2 baseline)
Observation: first full convergence! lr=1E-3 with L1=1E-5 achieves robust W recovery; branched from node 2 (node 3 failed)

## Iter 5: converged
Node: id=5, parent=4
Mode: success-exploit
Strategy: exploit (validation run 1)
Config: lr_W=2.0E-3, lr=1.0E-3, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9309, test_pearson=0.9076, connectivity_R2=0.9530, final_loss=4.8370e+02
Activity: healthy oscillatory dynamics, effective rank 12-33, range [-26.3, 23.1]
Mutation: none (validation run of node 4 config)
Observation: second consecutive convergence with same config; R2 improved (0.933 → 0.953), loss decreased (597 → 484); config appears robust, need one more validation run

## Iter 6: converged
Node: id=6, parent=5
Mode: failure-probe
Strategy: boundary (probing L1 regularization threshold)
Config: lr_W=2.0E-3, lr=1.0E-3, coeff_W_L1=7.5E-6, batch_size=8
Metrics: test_R2=0.9870, test_pearson=0.9815, connectivity_R2=0.9302, final_loss=7.4145e+02
Activity: healthy oscillatory dynamics, effective rank 12-35, range [-22.1, 21.5]
Mutation: coeff_W_L1: 1.0E-5 → 7.5E-6 (25% reduction to probe L1 boundary)
Observation: convergence maintained at lower L1! boundary is below 7.5E-6; node 3 failed at 5E-6 with lr=5E-4, but lr=1E-3 may recover at lower L1. test_R2 improved significantly (0.931 → 0.987). next: probe L1=5E-6 with lr=1E-3 to check if higher lr compensates for lower L1

