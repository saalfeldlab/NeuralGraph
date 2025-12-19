# Experiment Log: signal_Claude_ter

## Iter 1: partial
Node: id=1, parent=root
Mode/Strategy: exploit
Config: lr_W=2.0E-3, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.5178, test_pearson=0.1084, connectivity_R2=0.1098, final_loss=1.4149e+04
Activity: chaotic dynamics with amplitude range [-30.9, 32.4], effective rank 10 (90% var), no external input
Mutation: baseline config (first iteration)
Parent rule: first iteration of block, parent=root
Observation: baseline config achieves partial convergence; connectivity recovery weak despite reasonable test_R2
Next: parent=1 (highest UCB, try increasing lr_W to improve connectivity learning)

## Iter 2: partial
Node: id=2, parent=1
Mode/Strategy: exploit
Config: lr_W=5.0E-3, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.6436, test_pearson=0.2487, connectivity_R2=0.5522, final_loss=1.1236e+04
Activity: chaotic dynamics with amplitude range [-28.8, 26.2], effective rank 10 (90% var), spectral radius 1.025
Mutation: lr_W: 2.0E-3 -> 5.0E-3
Parent rule: Node 1 was highest UCB (0.581), increased lr_W per iter 1 suggestion
Observation: increasing lr_W 2.5x improved connectivity_R2 5x (0.11->0.55); trend suggests higher lr_W may help further
Next: parent=2 (highest UCB=1.259, try increasing lr_W further or adjusting other params)

## Iter 3: partial
Node: id=3, parent=2
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.7530, test_pearson=0.5453, connectivity_R2=0.6136, final_loss=9.4837e+03
Activity: chaotic dynamics with amplitude range [-28.8, 26.6], effective rank 9 (90% var), spectral radius 1.025
Mutation: lr_W: 5.0E-3 -> 1.0E-2
Parent rule: Node 2 was highest UCB (1.259), continued lr_W increase trend
Observation: doubling lr_W again improved connectivity_R2 from 0.55 to 0.61 (+11%); pearson correlation doubled (0.25->0.55); consistent improvement trend
Next: parent=3 (highest UCB=1.480, try lr_W=2.0E-2 or explore other dimensions)

## Iter 4: partial
Node: id=4, parent=3
Mode/Strategy: exploit
Config: lr_W=2.0E-2, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9163, test_pearson=0.8188, connectivity_R2=0.5901, final_loss=7.8702e+03
Activity: chaotic dynamics with amplitude range [-28.8, 31.5], effective rank 8 (90% var), spectral radius 1.025
Mutation: lr_W: 1.0E-2 -> 2.0E-2
Parent rule: Node 3 was highest UCB (1.480), continued lr_W increase trend
Observation: lr_W=2.0E-2 overshot; connectivity_R2 dropped (0.61->0.59) while test_R2 improved (0.75->0.92); lr_W=1.0E-2 appears near optimal for connectivity
Next: parent=3 (revert to lr_W=1.0E-2, explore lr or lr_emb dimension instead)

## Iter 5: converged
Node: id=5, parent=3
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=5.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.5805, test_pearson=0.2536, connectivity_R2=0.9546, final_loss=3.7124e+03
Activity: chaotic dynamics with amplitude range [-29.4, 25.6], effective rank 9 (90% var), spectral radius 1.025
Mutation: lr: 1.0E-4 -> 5.0E-4
Parent rule: reverted to Node 3 (lr_W=1.0E-2) per iter 4 suggestion, explored lr dimension
Observation: increasing lr 5x achieved first convergence (R2=0.955); connectivity_R2 jumped from 0.61 to 0.95 while test_R2 dropped (0.75->0.58); lr appears critical for connectivity learning
Next: parent=5 (highest UCB=2.073, try robustness test or explore lr further)

## Iter 6: converged
Node: id=6, parent=5
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9699, test_pearson=0.9314, connectivity_R2=0.9979, final_loss=3.2784e+03
Activity: chaotic dynamics with amplitude range [-28.8, 28.2], effective rank 10 (90% var), rank 32 (99% var), spectral radius 1.025
Mutation: lr: 5.0E-4 -> 1.0E-3
Parent rule: Node 5 was highest UCB (2.073), continued lr increase trend
Observation: doubling lr achieved near-perfect connectivity recovery (R2=0.998); test_R2 jumped to 0.97 and pearson to 0.93; optimal config found for chaotic regime
Next: parent=6 (highest UCB=2.223, try failure-probe with lr=2.0E-3 to map stability boundary)

## Iter 7: converged
Node: id=7, parent=6
Mode/Strategy: failure-probe
Config: lr_W=1.0E-2, lr=2.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.8900, test_pearson=0.8106, connectivity_R2=0.9858, final_loss=3.3125e+03
Activity: chaotic dynamics with amplitude range [-28.8, 29.2], effective rank 9 (90% var), rank 31 (99% var), spectral radius 1.025
Mutation: lr: 1.0E-3 -> 2.0E-3
Parent rule: Node 6 was highest UCB (2.223), failure-probe per iter 6 suggestion
Observation: lr=2.0E-3 still converges but performance degraded (R2: 0.998->0.986, pearson: 0.93->0.81); approaching upper stability boundary for lr
Next: parent=7 (highest UCB=2.309, continue failure-probe with lr=5.0E-3 to find failure boundary)

## Iter 8: converged
Node: id=8, parent=7
Mode/Strategy: failure-probe
Config: lr_W=1.0E-2, lr=5.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.8617, test_pearson=0.6822, connectivity_R2=0.9849, final_loss=2.9688e+03
Activity: chaotic dynamics with amplitude range [-30.5, 28.9], effective rank 9 (90% var), rank 31 (99% var), spectral radius 1.025
Mutation: lr: 2.0E-3 -> 5.0E-3
Parent rule: Node 7 was highest UCB (2.309), continued failure-probe per iter 7 suggestion
Observation: lr=5.0E-3 still converges but pearson degraded significantly (0.81->0.68); connectivity R2 stable at 0.985; near failure boundary for lr
Next: parent=8 (highest UCB=2.399, continue failure-probe with lr=1.0E-2 to find failure boundary)

## Iter 9: converged
Node: id=9, parent=8
Mode/Strategy: failure-probe
Config: lr_W=1.0E-2, lr=1.0E-2, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.7859, test_pearson=0.5471, connectivity_R2=0.9535, final_loss=3.2287e+03
Activity: chaotic dynamics with amplitude range [-28.8, 25.6], effective rank 8 (90% var), rank 30 (99% var), spectral radius 1.025
Mutation: lr: 5.0E-3 -> 1.0E-2
Parent rule: Node 8 was highest UCB (2.399), continued failure-probe per iter 8 suggestion
Observation: lr=1.0E-2 still converges (R2=0.954) but approaching failure boundary; pearson degraded (0.68->0.55), test_R2 dropped (0.86->0.79); 5th consecutive success, lr upper boundary mapped between 1E-2 and 2E-2
Next: parent=9 (highest UCB=2.454, continue failure-probe with lr=2.0E-2 to confirm failure boundary)

## Iter 10: failed
Node: id=10, parent=9
Mode/Strategy: failure-probe
Config: lr_W=1.0E-2, lr=2.0E-2, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.4473, test_pearson=0.0231, connectivity_R2=0.1970, final_loss=8.0714e+03
Activity: chaotic dynamics with amplitude range [-28.8, 36.1], effective rank 9 (90% var), rank 29 (99% var), spectral radius 1.025
Mutation: lr: 1.0E-2 -> 2.0E-2
Parent rule: Node 9 was highest UCB (2.454), continued failure-probe per iter 9 suggestion
Observation: lr=2.0E-2 confirms failure boundary; connectivity_R2 collapsed (0.95->0.20), pearson near zero (0.02); lr upper limit is between 1.0E-2 and 2.0E-2; optimal lr range mapped as [5.0E-4, 1.0E-2]
Next: parent=4 (highest UCB=2.171, explore lr_W=2.0E-2 branch with optimized lr to test high lr_W regime)

## Iter 11: converged
Node: id=11, parent=4
Mode/Strategy: exploit
Config: lr_W=2.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.8646, test_pearson=0.7454, connectivity_R2=0.9331, final_loss=3.0879e+03
Activity: chaotic dynamics with amplitude range [-28.8, 27.0], effective rank 9 (90% var), rank 31 (99% var), spectral radius 1.025
Mutation: lr: 1.0E-4 -> 1.0E-3 (from node 4's lr=1.0E-4 to optimal lr=1.0E-3)
Parent rule: Node 4 was highest UCB (2.171), combined high lr_W=2.0E-2 with optimized lr=1.0E-3 from iter 6
Observation: high lr_W=2.0E-2 can converge (R2=0.933) when paired with optimal lr=1.0E-3; confirms lr is more critical than lr_W for connectivity recovery; R2 lower than optimal config (0.998 at iter 6)
Next: parent=11 (highest UCB=2.591, explore lr_emb dimension or try lr_W between 1.0E-2 and 2.0E-2)

## Iter 12: converged
Node: id=12, parent=11
Mode/Strategy: exploit
Config: lr_W=2.0E-2, lr=1.0E-3, lr_emb=5.0E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.7514, test_pearson=0.4551, connectivity_R2=0.9731, final_loss=2.8803e+03
Activity: chaotic dynamics with amplitude range [-29.9, 28.8], effective rank 9 (90% var), rank 31 (99% var), spectral radius 1.025
Mutation: lr_emb: 2.5E-4 -> 5.0E-4
Parent rule: Node 11 was highest UCB (2.591), explored lr_emb dimension per iter 11 suggestion
Observation: increasing lr_emb 2x improved connectivity_R2 (0.933->0.973) but test_R2 and pearson degraded (0.86->0.75, 0.75->0.46); lr_emb helps connectivity learning but hurts dynamics prediction; this is 8th consecutive success, should explore failure boundary
Next: parent=12 (highest UCB=2.705, try failure-probe with lr_emb=1.0E-3 or explore lr_W=1.5E-2 midpoint)

## Iter 13: converged
Node: id=13, parent=12
Mode/Strategy: failure-probe
Config: lr_W=2.0E-2, lr=1.0E-3, lr_emb=1.0E-3, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.6685, test_pearson=0.3565, connectivity_R2=0.9113, final_loss=2.5429e+03
Activity: chaotic dynamics with amplitude range [-29.1, 30.3], effective rank 8 (90% var), rank 30 (99% var), spectral radius 1.025
Mutation: lr_emb: 5.0E-4 -> 1.0E-3
Parent rule: Node 12 was highest UCB (2.705), failure-probe per iter 12 suggestion
Observation: lr_emb=1.0E-3 degraded connectivity_R2 (0.973->0.911) and test metrics; approaching lr_emb upper boundary; 9th consecutive success; per protocol, explore strategy - branch from Node 6 (best R2=0.998, not in last 6 nodes)
Next: parent=6 (explore strategy: highest UCB not in last 6 nodes, try coeff_W_L1 dimension to map regularization effects)

## Iter 14: converged
Node: id=14, parent=6
Mode/Strategy: explore
Config: lr_W=1.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-6, batch_size=8
Metrics: test_R2=0.7147, test_pearson=0.4101, connectivity_R2=0.9584, final_loss=3.2463e+03
Activity: chaotic dynamics with amplitude range [-28.8, 27.9], effective rank 9 (90% var), rank 30 (99% var), spectral radius 1.025
Mutation: coeff_W_L1: 1.0E-5 -> 1.0E-6
Parent rule: Node 6 was selected per explore strategy (best R2 not in last 6 nodes); explored coeff_W_L1 dimension
Observation: reducing coeff_W_L1 10x (1.0E-5->1.0E-6) lowered connectivity_R2 from 0.998 to 0.958; optimal regularization appears to be 1.0E-5; 10th consecutive success; should explore higher coeff_W_L1 or batch_size dimension
Next: parent=14 (highest UCB=2.829, try coeff_W_L1=1.0E-4 to test stronger regularization)

## Iter 15: converged
Node: id=15, parent=14
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-4, batch_size=8
Metrics: test_R2=0.7102, test_pearson=0.3817, connectivity_R2=0.9353, final_loss=3.3500e+03
Activity: chaotic dynamics with amplitude range [-28.8, 30.1], effective rank 8 (90% var), rank 30 (99% var), spectral radius 1.025
Mutation: coeff_W_L1: 1.0E-6 -> 1.0E-4
Parent rule: Node 14 was highest UCB (2.829), tested higher regularization per iter 14 suggestion
Observation: coeff_W_L1=1.0E-4 (100x from iter 14) degraded connectivity_R2 (0.958->0.935); both extremes (1.0E-6 and 1.0E-4) worse than baseline 1.0E-5; optimal regularization confirmed at ~1.0E-5; 11th consecutive success
Next: parent=9 (explore strategy: highest UCB=2.244 not in last 6 nodes [10-15], try batch_size=16 to explore final unexplored dimension)

## Iter 16: converged
Node: id=16, parent=9
Mode/Strategy: explore
Config: lr_W=1.0E-2, lr=1.0E-2, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=16
Metrics: test_R2=0.8697, test_pearson=0.7683, connectivity_R2=0.9453, final_loss=2.8069e+03
Activity: chaotic dynamics with amplitude range [-28.8, 36.1], effective rank 9 (90% var), rank 30 (99% var), spectral radius 1.025
Mutation: batch_size: 8 -> 16
Parent rule: Node 9 was selected per explore strategy (highest UCB=2.454 not in last 6 nodes [10-15]); explored batch_size dimension
Observation: batch_size=16 achieves good connectivity_R2=0.945 with high lr=1.0E-2; comparable to Node 9 (R2=0.954 with batch_size=8); larger batch slightly degrades performance; 12th consecutive success; all major parameters mapped
Next: parent=6 (explore strategy: test batch_size=16 with optimal lr=1.0E-3 to see if larger batch improves best config)

## Iter 17: converged
Node: id=17, parent=6
Mode/Strategy: explore
Config: lr_W=1.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=16
Metrics: test_R2=0.8552, test_pearson=0.6491, connectivity_R2=0.9007, final_loss=3.1836e+03
Activity: chaotic dynamics with amplitude range [-28.8, 27.7], effective rank 8 (90% var), rank 29 (99% var), spectral radius 1.025
Mutation: batch_size: 8 -> 16
Parent rule: Node 6 was selected per iter 16 suggestion; tested batch_size=16 with optimal lr=1.0E-3 config
Observation: batch_size=16 with optimal lr=1.0E-3 achieves R2=0.901, significantly below Node 6's R2=0.998 with batch_size=8; larger batch size degrades connectivity learning across all lr values tested; batch_size=8 confirmed optimal; 13th consecutive success
Next: parent=16 (highest UCB=3.007, explore batch_size=32 as extreme failure-probe to map batch_size boundary)

## Iter 18: converged
Node: id=18, parent=16
Mode/Strategy: failure-probe
Config: lr_W=1.0E-2, lr=1.0E-2, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=32
Metrics: test_R2=0.7156, test_pearson=0.4871, connectivity_R2=0.9539, final_loss=2.7147e+03
Activity: chaotic dynamics with amplitude range [-28.8, 28.2], effective rank 10 (90% var), rank 31 (99% var), spectral radius 1.025
Mutation: batch_size: 16 -> 32
Parent rule: Node 16 was highest UCB (3.007), failure-probe batch_size=32 per iter 17 suggestion
Observation: batch_size=32 still converges (R2=0.954) with high lr=1.0E-2; performance comparable to batch_size=16 (R2=0.945); batch_size robust across [8,16,32] when combined with high lr; 14th consecutive success; parameter space thoroughly mapped
Next: parent=6 (robustness-test: re-run optimal config lr_W=1.0E-2, lr=1.0E-3, batch_size=8 to verify reproducibility)

## Iter 19: converged
Node: id=19, parent=6
Mode/Strategy: robustness-test
Config: lr_W=1.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.8372, test_pearson=0.7074, connectivity_R2=0.9560, final_loss=3.4962e+03
Activity: chaotic dynamics with amplitude range [-28.8, 28.3], effective rank 9 (90% var), rank 31 (99% var), spectral radius 1.025
Mutation: none (robustness test of Node 6 config)
Parent rule: Node 6 was selected per iter 18 suggestion for robustness-test of optimal config
Observation: robustness test shows variance; same config as Node 6 (R2=0.998) achieved R2=0.956 here; 15th consecutive success; optimal config is robust (converges) but with ~0.04 R2 variance; all major parameter boundaries mapped
Next: parent=6 (explore: try lr=2.0E-3 to refine boundary between optimal lr=1.0E-3 and degraded lr=5.0E-3)

## Iter 20: converged
Node: id=20, parent=6
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=2.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.8310, test_pearson=0.7085, connectivity_R2=0.9922, final_loss=3.1582e+03
Activity: chaotic dynamics with amplitude range [-31.9, 28.5], effective rank 9 (90% var), rank 31 (99% var), spectral radius 1.025
Mutation: lr: 1.0E-3 -> 2.0E-3
Parent rule: Node 6 was selected per iter 19 suggestion; tested lr=2.0E-3 to refine boundary
Observation: lr=2.0E-3 achieves excellent R2=0.992, close to Node 6's 0.998; confirms lr range [1.0E-3, 2.0E-3] is optimal; 16th consecutive success; per protocol (6+ consecutive successes), switch to explore strategy
Next: parent=11 (explore strategy: highest UCB=2.051 not in last 6 nodes [15-20], test lr_W=1.5E-2 midpoint with high lr_emb)

## Iter 21: converged
Node: id=21, parent=11
Mode/Strategy: explore
Config: lr_W=1.5E-2, lr=1.0E-3, lr_emb=5.0E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.7554, test_pearson=0.5114, connectivity_R2=0.9358, final_loss=2.9002e+03
Activity: chaotic dynamics with amplitude range [-28.8, 26.6], effective rank 8 (90% var), rank 28 (99% var), spectral radius 1.025
Mutation: lr_W: 2.0E-2 -> 1.5E-2 (midpoint test)
Parent rule: Node 11 was selected per iter 20 suggestion (explore strategy, highest UCB not in last 6 nodes); tested lr_W=1.5E-2 midpoint
Observation: lr_W=1.5E-2 achieves R2=0.936, between Node 11 (R2=0.933, lr_W=2.0E-2) and Node 6 (R2=0.998, lr_W=1.0E-2); confirms lr_W=1.0E-2 optimal; 17th consecutive success; 3 iters remaining in block 0
Next: parent=20 (highest UCB=3.283, test lr_W=1.5E-2 with optimal lr=2.0E-3 to see if combining midpoint lr_W with optimal lr improves R2)

## Iter 22: converged
Node: id=22, parent=20
Mode/Strategy: exploit
Config: lr_W=1.5E-2, lr=2.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.8567, test_pearson=0.6447, connectivity_R2=0.9739, final_loss=2.8518e+03
Activity: chaotic dynamics with amplitude range [-30.7, 26.5], effective rank 10 (90% var), rank 33 (99% var), spectral radius 1.025
Mutation: lr_W: 1.0E-2 -> 1.5E-2
Parent rule: Node 20 was highest UCB (3.283) per iter 21 suggestion; tested lr_W=1.5E-2 with optimal lr=2.0E-3
Observation: lr_W=1.5E-2 with lr=2.0E-3 achieves R2=0.974, better than Node 21 (R2=0.936, lr=1.0E-3) but below Node 20 (R2=0.992, lr_W=1.0E-2); confirms lr_W=1.0E-2 remains optimal even with higher lr; 18th consecutive success; 2 iters remaining in block 0
Next: parent=15 (explore strategy: highest UCB=3.281 not in last 6 nodes [17-22], explore coeff_W_L1=1.0E-3 extreme to complete regularization boundary)

## Iter 23: partial
Node: id=23, parent=15
Mode/Strategy: explore
Config: lr_W=1.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-3, batch_size=8
Metrics: test_R2=0.6783, test_pearson=0.3502, connectivity_R2=0.8510, final_loss=5.6317e+03
Activity: chaotic dynamics with amplitude range [-28.8, 26.1], effective rank 8 (90% var), rank 29 (99% var), spectral radius 1.025
Mutation: coeff_W_L1: 1.0E-4 -> 1.0E-3
Parent rule: Node 15 was selected per iter 22 suggestion (explore strategy); tested coeff_W_L1=1.0E-3 extreme
Observation: coeff_W_L1=1.0E-3 too high; R2 dropped from 0.935 to 0.851 (partial); first non-converged result after 18 successes; regularization upper boundary confirmed between 1.0E-4 and 1.0E-3; optimal at ~1.0E-5; final iter of block 0
Next: parent=22 (highest UCB=3.372, last iteration of block 0, test lr_emb=5.0E-4 with lr_W=1.5E-2 and lr=2.0E-3 to complete exploration)

## Iter 24: converged
Node: id=24, parent=22
Mode/Strategy: exploit
Config: lr_W=1.5E-2, lr=2.0E-3, lr_emb=5.0E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.7853, test_pearson=0.6112, connectivity_R2=0.9524, final_loss=2.5423e+03
Activity: chaotic dynamics with amplitude range [-28.8, 27.7], effective rank 8 (90% var), rank 30 (99% var), spectral radius 1.025
Mutation: lr_emb: 2.5E-4 -> 5.0E-4
Parent rule: Node 22 was highest UCB (3.402) per iter 23 suggestion; tested lr_emb=5.0E-4 with lr_W=1.5E-2 and lr=2.0E-3
Observation: lr_emb=5.0E-4 achieves R2=0.952, below Node 22 (R2=0.974, lr_emb=2.5E-4); higher lr_emb degrades connectivity_R2 even with good lr/lr_W; 19 of 24 converged; final iteration of block 0
Next: NEW BLOCK - change simulation parameters for block 1

## Simulation block 0 Summary (iters 1-24)

Simulation: connectivity_type=chaotic, n_types=4, noise=none
Best R2: 0.998 at iter 6
Observation: chaotic regime converges reliably (19/24 = 79%). initial config (lr=1.0E-4) failed to recover connectivity despite reasonable test_R2. key insight: lr is critical for connectivity recovery, optimal range [1.0E-3, 2.0E-3]. lr_W=1.0E-2 optimal, higher values (1.5E-2, 2.0E-2) still converge but with lower R2. regularization (coeff_W_L1) best at 1.0E-5, both higher/lower degrade. batch_size robust [8,16,32], batch_size=8 marginally better.
Optimum training parameters: lr_W=1.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8

### Block 0 Exploration Rule Evaluation
- Branching rate (last 6 iters): 5/6 unique parents = 83% (good exploration)
- Improvement rate: 19/24 converged = 79% (healthy mix of exploit/explore)
- Stuck detection: no plateau detected
- Protocol edit: no changes needed - rules produced effective exploration with high convergence rate and thorough parameter mapping

---

## Iter 25: failed
--- NEW SIMULATION BLOCK ---
Simulation: connectivity_type=low_rank, Dale_law=False, connectivity_rank=20, noise_model_level=0
Node: id=25, parent=root
Mode/Strategy: exploit (baseline for new simulation block)
Config: lr_W=1.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.8779, test_pearson=0.3646, connectivity_R2=0.0092, final_loss=7.1754e+02
Activity: extremely low rank dynamics; effective rank 1 (90% var), rank 2 (99% var); amplitude range [-28.5, 28.2]; spectral radius 0.951; no external input
Mutation: baseline config (optimal from block 0)
Parent rule: first iteration of block 1, parent=root
Observation: low_rank connectivity (rank=20) fundamentally different from chaotic; activity collapses to rank 1-2 despite n_neurons=100; connectivity_R2 near zero (0.009) despite excellent test_R2 (0.88); optimal config from chaotic block fails completely; need much higher lr_W to recover sparse low-rank connectivity structure
Next: parent=25 (only node in block 1, try lr_W=5.0E-2 to increase connectivity learning aggressiveness for low-rank structure)

## Iter 26: partial
Node: id=26, parent=25
Mode/Strategy: exploit
Config: lr_W=5.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.7825, test_pearson=0.6437, connectivity_R2=0.5770, final_loss=3.6175e+03
Activity: low_rank dynamics with improved structure; effective rank 4 (90% var), rank 12 (99% var); amplitude range [-18.4, 20.9]; spectral radius 1.071; 8.3x compression
Mutation: lr_W: 1.0E-2 -> 5.0E-2
Parent rule: Node 25 was highest UCB (0.481); increased lr_W 5x to address low-rank connectivity learning
Observation: aggressive lr_W=5.0E-2 achieved 62x improvement in connectivity_R2 (0.009->0.577); activity structure improved from rank 1-2 to rank 4-12; pearson correlation doubled (0.36->0.64); major progress but still partial convergence
Next: parent=26 (highest UCB=1.284, try lr_W=1.0E-1 to continue aggressive connectivity learning trend for low-rank)

## Iter 27: partial
Node: id=27, parent=26
Mode/Strategy: exploit
Config: lr_W=1.0E-1, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.7543, test_pearson=0.5383, connectivity_R2=0.7943, final_loss=3.4750e+03
Activity: low_rank dynamics; effective rank 4 (90% var), rank 12 (99% var); amplitude range [-30.2, 36.4]; spectral radius 1.037; 8.3x compression
Mutation: lr_W: 5.0E-2 -> 1.0E-1
Parent rule: Node 26 was highest UCB (1.154); continued lr_W increase trend per iter 26 suggestion
Observation: lr_W=1.0E-1 improved connectivity_R2 (0.577->0.794, +38%); test_R2 and pearson slightly degraded; continuing lr_W increase trend; approaching convergence threshold (0.9)
Next: parent=27 (highest UCB=1.660, try lr_W=2.0E-1 to continue trend toward convergence)

## Iter 28: failed
Node: id=28, parent=27
Mode/Strategy: exploit
Config: lr_W=2.0E-1, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9033, test_pearson=0.5283, connectivity_R2=0.0007, final_loss=1.2014e+03
Activity: low_rank dynamics collapsed; effective rank 1 (90% var), rank 3 (99% var); amplitude range [-38.5, 20.0]; spectral radius 1.034; 33x compression
Mutation: lr_W: 1.0E-1 -> 2.0E-1
Parent rule: Node 27 was highest UCB (1.660) per iter 27 suggestion; continued lr_W increase trend
Observation: lr_W=2.0E-1 catastrophically overshot; connectivity_R2 collapsed (0.794->0.001, -99.9%); activity collapsed from rank 4 to rank 1; lr_W upper boundary is between 1.0E-1 and 2.0E-1; revert to lr_W=1.0E-1 and explore lr dimension
Next: parent=27 (highest UCB=1.461, revert to lr_W=1.0E-1 and try lr=2.0E-3 to push connectivity_R2 above 0.9)

## Iter 29: failed
Node: id=29, parent=27
Mode/Strategy: exploit
Config: lr_W=1.0E-1, lr=2.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.7583, test_pearson=0.7426, connectivity_R2=0.0114, final_loss=1.0693e+03
Activity: low_rank dynamics collapsed; effective rank 1 (90% var), rank 2 (99% var); amplitude range [-39.9, 30.9]; spectral radius 1.236; 49.5x compression
Mutation: lr: 1.0E-3 -> 2.0E-3
Parent rule: Node 27 was highest UCB (1.461) per iter 28 suggestion; tried lr=2.0E-3 with lr_W=1.0E-1
Observation: lr=2.0E-3 failed (R2=0.011); activity collapsed to rank 1-2; lr increase with high lr_W destabilizes connectivity learning; Node 27 (lr=1.0E-3, R2=0.794) remains best; need to explore lr_emb or reduce lr to 5.0E-4
Next: parent=27 (highest UCB=1.353, try lr=5.0E-4 to test if lower lr stabilizes connectivity learning with high lr_W)

## Iter 30: failed
Node: id=30, parent=27
Mode/Strategy: exploit
Config: lr_W=1.0E-1, lr=5.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.4440, test_pearson=0.2025, connectivity_R2=0.0042, final_loss=1.2958e+03
Activity: low_rank dynamics collapsed; effective rank 1 (90% var), rank 5 (99% var); amplitude range [-24.5, 26.4]; spectral radius 0.947; 19.8x compression
Mutation: lr: 1.0E-3 -> 5.0E-4
Parent rule: Node 27 was highest UCB (1.284) per iter 29 suggestion; tested lr=5.0E-4 with lr_W=1.0E-1
Observation: lr=5.0E-4 failed (R2=0.004); activity collapsed to rank 1; both lr increase (iter 29) and decrease (iter 30) from Node 27's lr=1.0E-3 failed; lr=1.0E-3 appears critical; need to explore lr_emb dimension or lr_W between 5.0E-2 and 1.0E-1
Next: parent=27 (highest UCB=1.284, try lr_emb=5.0E-4 to explore embedding lr dimension for low_rank regime)

## Iter 31: failed
Node: id=31, parent=27
Mode/Strategy: exploit
Config: lr_W=1.0E-1, lr=1.0E-3, lr_emb=5.0E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.2977, test_pearson=-0.1585, connectivity_R2=0.0356, final_loss=1.8785e+03
Activity: low_rank dynamics collapsed; effective rank 2 (90% var), rank 7 (99% var); amplitude range [-28.1, 26.7]; spectral radius 1.095; 14.1x compression
Mutation: lr_emb: 2.5E-4 -> 5.0E-4
Parent rule: Node 27 was highest UCB (1.284) per iter 30 suggestion; tested lr_emb=5.0E-4 with lr_W=1.0E-1
Observation: lr_emb=5.0E-4 failed (R2=0.036); 4th consecutive failure from Node 27 (iters 28-31); all mutations (lr_W up, lr up, lr down, lr_emb up) from Node 27 failed; lr_W=1.0E-1 may be at edge of stability; try lr_W=7.5E-2 midpoint between Node 26 (lr_W=5.0E-2, R2=0.577) and Node 27 (lr_W=1.0E-1, R2=0.794)
Next: parent=26 (UCB=0.955, try lr_W=7.5E-2 midpoint to find stable region between 5.0E-2 and 1.0E-1)

## Iter 32: failed
Node: id=32, parent=26
Mode/Strategy: exploit
Config: lr_W=7.5E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.4523, test_pearson=0.0524, connectivity_R2=0.0323, final_loss=1.8405e+03
Activity: low_rank dynamics collapsed; effective rank 2 (90% var), rank 7 (99% var); amplitude range [-28.1, 26.7]; spectral radius 1.095; 14.1x compression
Mutation: lr_W: 5.0E-2 -> 7.5E-2
Parent rule: Node 26 was selected per iter 31 suggestion; tested lr_W=7.5E-2 midpoint
Observation: lr_W=7.5E-2 failed (R2=0.032); midpoint between stable Node 26 (lr_W=5.0E-2, R2=0.577) and best Node 27 (lr_W=1.0E-1, R2=0.794) also failed; 5th consecutive failure (iters 28-32); low_rank regime highly unstable above lr_W=5.0E-2; need to explore from Node 26 (last stable config) with different lr values
Next: parent=27 (highest UCB=1.266, try lr=1.5E-3 between 1.0E-3 and 2.0E-3 to find stable lr for high lr_W)

## Iter 33: failed
Node: id=33, parent=27
Mode/Strategy: exploit
Config: lr_W=1.0E-1, lr=1.5E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.3921, test_pearson=0.1807, connectivity_R2=0.0001, final_loss=1.9789e+03
Activity: low_rank dynamics collapsed; effective rank 2 (90% var), rank 7 (99% var); amplitude range [-28.1, 26.7]; spectral radius 1.095; 14.1x compression
Mutation: lr: 1.0E-3 -> 1.5E-3
Parent rule: Node 27 was highest UCB (1.223) per iter 32 suggestion; tested lr=1.5E-3 between 1.0E-3 and 2.0E-3
Observation: lr=1.5E-3 failed catastrophically (R2=0.0001); 6th consecutive failure from Node 27 area (iters 28-33); all lr variations (0.5E-3, 1.0E-3, 1.5E-3, 2.0E-3) with lr_W=1.0E-1 fail except original Node 27; Node 27 (R2=0.794) appears to be a lucky initialization; low_rank regime with high lr_W is highly unstable; need to return to Node 26 (lr_W=5.0E-2, R2=0.577) and explore other dimensions
Next: parent=26 (UCB=0.910, stable partial result; try lr=2.0E-3 to push connectivity_R2 higher without increasing lr_W)

## Iter 34: failed
Node: id=34, parent=26
Mode/Strategy: exploit
Config: lr_W=5.0E-2, lr=2.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.5708, test_pearson=0.2358, connectivity_R2=0.0403, final_loss=1.8153e+03
Activity: low_rank dynamics collapsed; effective rank 2 (90% var), rank 7 (99% var); amplitude range [-28.1, 26.7]; spectral radius 1.095; 14.1x compression
Mutation: lr: 1.0E-3 -> 2.0E-3
Parent rule: Node 26 was selected per iter 33 suggestion (stable partial result); tested lr=2.0E-3 without increasing lr_W
Observation: lr=2.0E-3 with lr_W=5.0E-2 failed (R2=0.040); 7th consecutive failure (iters 28-34); lr increase destabilized even at lower lr_W; Node 26 (lr=1.0E-3, R2=0.577) was stable but lr=2.0E-3 breaks it; low_rank regime requires strict lr=1.0E-3; need to explore coeff_W_L1 or batch_size from Node 26
Next: parent=26 (UCB=0.893, return to stable config and try reducing coeff_W_L1 to 1.0E-6 to reduce regularization pressure on low-rank connectivity)

## Iter 35: failed
Node: id=35, parent=26
Mode/Strategy: exploit
Config: lr_W=5.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-6, batch_size=8
Metrics: test_R2=0.3327, test_pearson=-0.0231, connectivity_R2=0.0461, final_loss=1.6635e+03
Activity: low_rank dynamics collapsed; effective rank 2 (90% var), rank 7 (99% var); amplitude range [-28.1, 26.7]; spectral radius 1.095; 14.1x compression
Mutation: coeff_W_L1: 1.0E-5 -> 1.0E-6
Parent rule: Node 26 was selected per iter 34 suggestion; tested coeff_W_L1=1.0E-6 to reduce regularization
Observation: coeff_W_L1=1.0E-6 failed (R2=0.046); 8th consecutive failure (iters 28-35); both lr variations (iters 34) and coeff_W_L1 reduction failed from Node 26; low_rank regime extremely fragile; only Node 26 (R2=0.577) and Node 27 (R2=0.794) achieved partial convergence; all derivatives from these nodes fail; try batch_size=16 from Node 26 as last unexplored dimension
Next: parent=26 (UCB=0.879, try batch_size=16 as final unexplored parameter from stable Node 26 config)

## Iter 36: failed
Node: id=36, parent=26
Mode/Strategy: exploit
Config: lr_W=5.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=16
Metrics: test_R2=0.3746, test_pearson=0.2007, connectivity_R2=0.0461, final_loss=1.3196e+03
Activity: low_rank dynamics collapsed; effective rank 2 (90% var), rank 7 (99% var); amplitude range [-28.1, 26.7]; spectral radius 1.095; 14.1x compression
Mutation: batch_size: 8 -> 16 (from Node 26's batch_size=8)
Parent rule: Node 26 was selected per iter 35 suggestion; tested batch_size=16 as final unexplored parameter
Observation: batch_size=16 failed (R2=0.046); 9th consecutive failure (iters 28-36); all parameter variations from stable Node 26/27 fail immediately; low_rank regime (rank=20) with current simulation config appears fundamentally hard for GNN; only initial learning (iters 26-27) succeeded, all refinements destabilize; need robustness test - rerun Node 27 exact config to verify if R2=0.794 reproducible or lucky initialization
Next: parent=27 (UCB=1.289, robustness test: rerun Node 27 exact config lr_W=1.0E-1, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8)

## Iter 37: failed
Node: id=37, parent=27
Mode/Strategy: robustness-test
Config: lr_W=1.0E-1, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.4819, test_pearson=0.1698, connectivity_R2=0.0369, final_loss=1.9287e+03
Activity: low_rank dynamics collapsed; effective rank 2 (90% var), rank 7 (99% var); amplitude range [-28.1, 26.7]; spectral radius 1.095; 14.1x compression
Mutation: none (robustness test of Node 27 config)
Parent rule: Node 27 was selected per iter 36 suggestion for robustness test of best config (R2=0.794)
Observation: robustness test confirms Node 27's R2=0.794 was not reproducible (R2=0.037 this run); 10th consecutive failure (iters 28-37); low_rank regime (rank=20) is fundamentally unstable - connectivity recovery highly seed-dependent; best reproducible result is Node 26 (R2=0.577, lr_W=5.0E-2); try lr_W=6.0E-2 between 5.0E-2 (stable) and 7.5E-2 (failed) to find max stable lr_W
Next: parent=26 (UCB=0.854, try lr_W=6.0E-2 midpoint to find maximum stable lr_W for low_rank regime)

## Iter 38: failed
Node: id=38, parent=26
Mode/Strategy: exploit
Config: lr_W=6.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.4312, test_pearson=0.2532, connectivity_R2=0.0400, final_loss=1.7991e+03
Activity: low_rank dynamics collapsed; effective rank 2 (90% var), rank 7 (99% var); amplitude range [-28.1, 26.7]; spectral radius 1.095; 14.1x compression
Mutation: lr_W: 5.0E-2 -> 6.0E-2
Parent rule: Node 26 was selected per iter 37 suggestion; tested lr_W=6.0E-2 midpoint between 5.0E-2 (R2=0.577) and 7.5E-2 (failed)
Observation: lr_W=6.0E-2 failed (R2=0.040); 11th consecutive failure (iters 28-38); even small lr_W increase (5.0E-2->6.0E-2) destabilizes; Node 26 (lr_W=5.0E-2) appears to be max stable lr_W; low_rank (rank=20) connectivity recovery is fundamentally limited to R2~0.58; try robustness test of Node 26 exact config to verify reproducibility before block boundary
Next: parent=26 (UCB=0.844, robustness test: rerun Node 26 exact config lr_W=5.0E-2, lr=1.0E-3 to verify if R2=0.577 is reproducible)

## Iter 39: failed
Node: id=39, parent=26
Mode/Strategy: robustness-test
Config: lr_W=5.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.5313, test_pearson=0.3336, connectivity_R2=0.0464, final_loss=1.7937e+03
Activity: low_rank dynamics collapsed; effective rank 2 (90% var), rank 7 (99% var); amplitude range [-28.1, 26.7]; spectral radius 1.095; 14.1x compression
Mutation: none (robustness test of Node 26 config)
Parent rule: Node 26 was selected per iter 38 suggestion for robustness test of best stable config (R2=0.577)
Observation: robustness test confirms Node 26's R2=0.577 was not reproducible (R2=0.046 this run); 12th consecutive failure (iters 28-39); both Node 26 (R2=0.577) and Node 27 (R2=0.794) were lucky initializations; low_rank regime (rank=20) is fundamentally unstable for connectivity recovery - all results are seed-dependent; remaining 9 iters in block 1 to explore alternative training strategies
Next: parent=26 (UCB=0.835, try extreme lr_W=2.0E-2 below baseline to test if lower lr_W is more stable for low_rank)

## Iter 40: failed
Node: id=40, parent=26
Mode/Strategy: exploit
Config: lr_W=2.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.3569, test_pearson=0.1614, connectivity_R2=0.0628, final_loss=1.7551e+03
Activity: low_rank dynamics collapsed; effective rank 2 (90% var), rank 7 (99% var); amplitude range [-28.1, 26.7]; spectral radius 1.095; 14.1x compression
Mutation: lr_W: 5.0E-2 -> 2.0E-2 (below baseline, testing conservative lr_W)
Parent rule: Node 26 was selected per iter 39 suggestion; tested conservative lr_W=2.0E-2 below Node 26's lr_W=5.0E-2
Observation: lr_W=2.0E-2 failed (R2=0.063); 13th consecutive failure (iters 28-40); conservative lr_W also fails; low_rank (rank=20) regime is fundamentally intractable regardless of lr_W value; activity consistently collapses to rank 2 (90% var); remaining 8 iters in block 1; try lr_W=3.0E-2 between 2.0E-2 (failed) and 5.0E-2 (unstable partial) as systematic grid search
Next: parent=40 (highest UCB=2.063, try lr_W=3.0E-2 to continue systematic exploration of lr_W range)

## Iter 41: failed
Node: id=41, parent=40
Mode/Strategy: exploit
Config: lr_W=3.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.3704, test_pearson=0.1554, connectivity_R2=0.0252, final_loss=1.7508e+03
Activity: low_rank dynamics collapsed; effective rank 2 (90% var), rank 7 (99% var); amplitude range [-28.1, 26.7]; spectral radius 1.095; 14.1x compression
Mutation: lr_W: 2.0E-2 -> 3.0E-2
Parent rule: Node 40 was highest UCB (2.063) per iter 40 suggestion; tested lr_W=3.0E-2 midpoint between 2.0E-2 (failed) and 5.0E-2 (unstable partial)
Observation: lr_W=3.0E-2 failed (R2=0.025); 14th consecutive failure (iters 28-41); systematic lr_W grid search (2.0E-2, 3.0E-2, 5.0E-2, 6.0E-2, 7.5E-2, 1.0E-1) shows only 5.0E-2 achieved any partial success (R2=0.577) but not reproducibly; low_rank (rank=20) regime fundamentally unstable; try lr_W=4.0E-2 between 3.0E-2 (failed) and 5.0E-2 (partial but unstable)
Next: parent=41 (highest UCB, try lr_W=4.0E-2 to complete systematic lr_W exploration)

## Iter 42: failed
Node: id=42, parent=41
Mode/Strategy: exploit
Config: lr_W=4.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.4837, test_pearson=0.2772, connectivity_R2=0.0454, final_loss=1.7548e+03
Activity: low_rank dynamics collapsed; effective rank 2 (90% var), rank 7 (99% var); amplitude range [-28.1, 26.7]; spectral radius 1.095; 14.1x compression
Mutation: lr_W: 3.0E-2 -> 4.0E-2
Parent rule: Node 41 was highest UCB (1.439) per iter 41 suggestion; tested lr_W=4.0E-2 to complete systematic lr_W exploration
Observation: lr_W=4.0E-2 failed (R2=0.045); 15th consecutive failure (iters 28-42); complete lr_W grid search [2.0E-2, 3.0E-2, 4.0E-2, 5.0E-2, 6.0E-2, 7.5E-2, 1.0E-1] all fail except initial Node 26 (5.0E-2) and Node 27 (1.0E-1) which were not reproducible; low_rank (rank=20) regime is fundamentally intractable with current GNN architecture; 6 iters remaining in block 1; try coeff_W_L1=1.0E-4 with lr_W=4.0E-2 to test if stronger regularization helps stabilize low_rank connectivity learning
Next: parent=39 (highest UCB=2.168, try coeff_W_L1=1.0E-4 from Node 26 config to test stronger regularization for low_rank stability)

## Iter 43: failed
Node: id=43, parent=39
Mode/Strategy: exploit
Config: lr_W=5.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-4, batch_size=8
Metrics: test_R2=0.4835, test_pearson=0.0744, connectivity_R2=0.0056, final_loss=2.5287e+03
Activity: low_rank dynamics collapsed; effective rank 2 (90% var), rank 7 (99% var); amplitude range [-28.1, 26.7]; spectral radius 1.095; 14.1x compression
Mutation: coeff_W_L1: 1.0E-5 -> 1.0E-4
Parent rule: Node 39 was highest UCB (2.168) per iter 42 suggestion; tested stronger regularization coeff_W_L1=1.0E-4
Observation: coeff_W_L1=1.0E-4 failed (R2=0.006); 16th consecutive failure (iters 28-43); both weaker regularization (1.0E-6, iter 35, R2=0.046) and stronger regularization (1.0E-4, R2=0.006) fail; coeff_W_L1=1.0E-5 is optimal but insufficient for low_rank (rank=20); 5 iters remaining in block 1; try lr_emb=1.0E-4 (reduced) from Node 35 config to minimize embedding interference with connectivity learning
Next: parent=35 (highest UCB=2.226, try lr_emb=1.0E-4 to test minimal embedding learning for low_rank regime)

## Iter 44: failed
Node: id=44, parent=35
Mode/Strategy: exploit
Config: lr_W=5.0E-2, lr=1.0E-3, lr_emb=1.0E-4, coeff_W_L1=1.0E-6, batch_size=8
Metrics: test_R2=0.3260, test_pearson=0.1408, connectivity_R2=0.0364, final_loss=1.7733e+03
Activity: low_rank dynamics collapsed; effective rank 2 (90% var), rank 7 (99% var); amplitude range [-28.1, 26.7]; spectral radius 1.095; 14.1x compression
Mutation: lr_emb: 2.5E-4 -> 1.0E-4 (reduced embedding lr from Node 35's coeff_W_L1=1.0E-6 config)
Parent rule: Node 35 was highest UCB (2.226) per iter 43 suggestion; tested reduced lr_emb=1.0E-4 with low regularization
Observation: lr_emb=1.0E-4 failed (R2=0.036); 17th consecutive failure (iters 28-44); reduced embedding lr also fails; 4 iters remaining in block 1; complete failure of all parameter combinations in low_rank (rank=20) regime; try Node 36's batch_size=16 config with lr_W=1.0E-1 (Node 27 config) as final high-risk attempt
Next: parent=36 (highest UCB=2.282, try lr_W=1.0E-1 with batch_size=16 as high-risk combination)

## Iter 45: failed
Node: id=45, parent=36
Mode/Strategy: exploit
Config: lr_W=1.0E-1, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=16
Metrics: test_R2=0.4035, test_pearson=0.1107, connectivity_R2=0.0255, final_loss=1.4198e+03
Activity: low_rank dynamics collapsed; effective rank 2 (90% var), rank 7 (99% var); amplitude range [-28.1, 26.7]; spectral radius 1.095; 14.1x compression
Mutation: lr_W: 5.0E-2 -> 1.0E-1 (high lr_W with batch_size=16 from Node 36)
Parent rule: Node 36 was highest UCB (2.282) per iter 44 suggestion; tested lr_W=1.0E-1 with batch_size=16 as high-risk combination
Observation: lr_W=1.0E-1 with batch_size=16 failed (R2=0.025); 18th consecutive failure (iters 28-45); all parameter combinations exhaustively tested in low_rank (rank=20) regime; 3 iters remaining in block 1; try batch_size=32 with aggressive lr_W=1.0E-1 and lr=2.0E-3 as extreme high-risk combination to complete exploration
Next: parent=42 (highest UCB=2.337, try batch_size=32 with lr_W=4.0E-2 and lr=2.0E-3 to test larger batch with moderate lr_W)

## Iter 46: failed
Node: id=46, parent=42
Mode/Strategy: exploit
Config: lr_W=4.0E-2, lr=2.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=32
Metrics: test_R2=0.4340, test_pearson=0.2469, connectivity_R2=0.0498, final_loss=1.0705e+03
Activity: low_rank dynamics collapsed; effective rank 2 (90% var), rank 7 (99% var); amplitude range [-28.1, 26.7]; spectral radius 1.095; 14.1x compression
Mutation: batch_size: 8 -> 32, lr: 1.0E-3 -> 2.0E-3 (batch_size=32 with lr=2.0E-3 from Node 42's lr_W=4.0E-2)
Parent rule: Node 42 was highest UCB (2.337) per iter 45 suggestion; tested batch_size=32 with lr=2.0E-3
Observation: batch_size=32 with lr=2.0E-3 failed (R2=0.050); 19th consecutive failure (iters 28-46); low_rank (rank=20) regime completely intractable; all parameter combinations exhaustively tested (lr_W: 2.0E-2 to 2.0E-1, lr: 5.0E-4 to 2.0E-3, lr_emb: 1.0E-4 to 5.0E-4, coeff_W_L1: 1.0E-6 to 1.0E-4, batch_size: 8 to 32); only initial Node 26 (R2=0.577) and Node 27 (R2=0.794) achieved partial success but not reproducible; 2 iters remaining in block 1
Next: parent=34 (highest UCB=2.386, try lr_W=3.0E-2 with lr=2.0E-3 to complete exploration of moderate lr_W with higher lr)

## Iter 47: failed
Node: id=47, parent=34
Mode/Strategy: exploit
Config: lr_W=3.0E-2, lr=2.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.3934, test_pearson=0.2059, connectivity_R2=0.0456, final_loss=1.7534e+03
Activity: low_rank dynamics collapsed; effective rank 2 (90% var), rank 7 (99% var); amplitude range [-28.1, 26.7]; spectral radius 1.095; 14.1x compression
Mutation: lr_W: 5.0E-2 -> 3.0E-2, lr: 2.0E-3 (from Node 34's lr=2.0E-3, reduced lr_W)
Parent rule: Node 34 was highest UCB (2.386) per iter 46 suggestion; tested moderate lr_W=3.0E-2 with higher lr=2.0E-3
Observation: lr_W=3.0E-2 with lr=2.0E-3 failed (R2=0.046); 20th consecutive failure (iters 28-47); all combinations of moderate lr_W and higher lr fail; low_rank (rank=20) regime completely intractable; 1 iter remaining in block 1; final iteration should try one last unexplored combination
Next: parent=46 (highest UCB=2.448, try lr_emb=5.0E-4 with batch_size=32 and lr_W=4.0E-2 as final exploration before block 2)

## Iter 48: failed
Node: id=48, parent=46
Mode/Strategy: exploit
Config: lr_W=4.0E-2, lr=2.0E-3, lr_emb=5.0E-4, coeff_W_L1=1.0E-5, batch_size=32
Metrics: test_R2=0.4824, test_pearson=0.1737, connectivity_R2=0.0640, final_loss=1.0076e+03
Activity: low_rank dynamics collapsed; effective rank 2 (90% var), rank 7 (99% var); amplitude range [-28.1, 26.7]; spectral radius 1.095; 14.1x compression
Mutation: lr_emb: 2.5E-4 -> 5.0E-4 (from Node 46's batch_size=32, lr=2.0E-3 config)
Parent rule: Node 46 was highest UCB (2.448) per iter 47 suggestion; tested lr_emb=5.0E-4 with batch_size=32 and lr_W=4.0E-2 as final exploration before block 2
Observation: lr_emb=5.0E-4 with batch_size=32 failed (R2=0.064); 21st consecutive failure (iters 28-48); low_rank (rank=20) regime completely intractable; final iteration of block 1
Next: NEW BLOCK - change simulation parameters for block 2

## Simulation block 1 Summary (iters 25-48)

Simulation: connectivity_type=low_rank, connectivity_rank=20, n_types=4, noise=none
Best R2: 0.794 at iter 27 (not reproducible), 0.577 at iter 26 (not reproducible)
Observation: low_rank (rank=20) regime is fundamentally intractable for GNN connectivity recovery. initial iterations (26-27) achieved partial convergence (R2=0.577, 0.794) but these results were not reproducible - robustness tests (iters 37, 39) failed. all 21 parameter variations (iters 28-48) failed catastrophically with R2<0.1. activity consistently collapses to effective rank 2 regardless of training parameters. low-rank connectivity structure creates unstable training dynamics where connectivity recovery is highly seed-dependent. this simulation regime appears fundamentally hard for current GNN architecture - the low-rank structure may require architectural changes (e.g., rank constraints) rather than training parameter tuning.
Optimum training parameters: none reliable - all configurations produce seed-dependent results

### Block 1 Exploration Rule Evaluation
- Branching rate (last 6 iters 43-48): 6/6 unique parents (43→39, 44→35, 45→36, 46→42, 47→34, 48→46) = 100% (excellent exploration, forced by consistent failures)
- Improvement rate: 2/24 partial (iters 26-27) = 8% (extremely low, regime is intractable)
- Stuck detection: 21 consecutive failures (R2<0.1) from iter 28-48
- Protocol edit assessment: current rules worked well - high branching rate explored parameter space exhaustively, but regime is fundamentally intractable; no rule changes needed as rules correctly identified and explored all parameter combinations

---

## Iter 49: [pending]
--- NEW SIMULATION BLOCK ---
Simulation: connectivity_type=low_rank, Dale_law=False, connectivity_rank=50, noise_model_level=0
Node: id=49, parent=root
Mode/Strategy: baseline for new simulation block
Config: lr_W=1.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Rationale: block 2 tests low_rank with higher rank=50 (vs rank=20 in block 1); higher rank may produce richer dynamics (avoiding collapse to rank 2) while still testing low_rank regime; using optimal training params from block 0

