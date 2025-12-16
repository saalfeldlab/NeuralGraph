# Experiment Log: signal_Claude

## Iter 1: converged
Node: id=1, parent=0, V=1, N_total=1
Config: connectivity_type=chaotic, connectivity_rank=NA, factor=0.5, gain=7, n_types=1, Dale_law=False
Metrics: spectral_radius=1.018, svd_rank=31, test_R2=0.9431, test_pearson=0.8993, connectivity_R2=0.9343, final_loss=1.2045e+03
RankScore: NA
Observation: baseline without Dale's law shows excellent convergence with rich chaotic dynamics (rank 31); activity traces show diverse oscillatory patterns across all 100 neurons
Change: initial baseline

## Iter 2: converged
Node: id=2, parent=1, V=1, N_total=2
Config: connectivity_type=chaotic, connectivity_rank=NA, factor=0.5, gain=7, n_types=1, Dale_law=True
Metrics: spectral_radius=0.895, svd_rank=23, test_R2=0.9652, test_pearson=0.9748, connectivity_R2=0.9970, final_loss=7.4694e+02
RankScore: 1.00
Observation: Dale's law dramatically improves connectivity recovery (0.934→0.997) with lower loss; spectral radius < 1 yields stable dynamics but reduced complexity (rank 23); E/I balance constraint may simplify weight structure for easier learning
Change: Dale_law: False -> True

