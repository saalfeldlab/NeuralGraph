# Experiment Log: signal_chaotic_Claude

## Iter 0: good (baseline without Dale's law)
Config: Dale_law=False, factor=0.5, gain=7, n_types=1
Metrics: spectral_radius=1.031, svd_rank=34, test_R2=0.9980, test_pearson=0.9941, connectivity_R2=1.0000
Observation: excellent training without Dale's law constraint; chaotic activity with rich dynamics
Change: Dale_law: False -> True, factor: 0.5 -> 0.3, gain: 7 -> 10, n_types: 1 -> 2

## Iter 1: poor
Config: Dale_law=True, factor=0.3, gain=10, n_types=2
Metrics: spectral_radius=2.962, svd_rank=11, test_R2=0.3448, test_pearson=0.1665, connectivity_R2=0.0211
Observation: Dale's law with 2 types yields low-rank dynamics (rank 11 vs 34); transient then flat activity; very poor connectivity recovery
Change: n_types: 2 -> 1

