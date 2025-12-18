# Experiment Log: signal_Claude

## Iter 1: partial
--- NEW BLOCK ---
Simulation: connectivity_type=chaotic, Dale_law=False, noise_model_level=0.0
Node: id=1, parent=root
Mode/Strategy: exploit
Config: lr_W=2.0E-3, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.8208, test_pearson=0.7820, connectivity_R2=0.8804, final_loss=1.2763e+03
Activity: oscillatory dynamics across 100 neurons, range [-20.4, 18.2], effective rank 38 (99% var), spectral_radius=0.983
Mutation: baseline (first iteration of block)
Parent rule: first iteration of block, parent=root
Observation: near-convergence with chaotic connectivity, R2=0.88 suggests parameters are close but may need fine-tuning
Next: parent=1, increase lr_W to 5.0E-3 to accelerate convergence

## Iter 2: converged
Node: id=2, parent=1
Mode/Strategy: exploit
Config: lr_W=5.0E-3, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9968, test_pearson=0.9951, connectivity_R2=0.9681, final_loss=4.3569e+02
Activity: oscillatory dynamics across 100 neurons, range [-26.3, 23.8], effective rank 31 (99% var), spectral_radius=1.025
Mutation: lr_W: 2.0E-3 -> 5.0E-3
Parent rule: highest UCB node (Node 1 was only option), exploit strategy
Observation: significant improvement from increasing lr_W, connectivity_R2 jumped from 0.88 to 0.97, test_R2 now near-perfect at 0.997
Next: parent=2, try lr_W=3.0E-3 to find optimal range between 2.0E-3 and 5.0E-3

## Iter 3: partial
Node: id=3, parent=2
Mode/Strategy: exploit
Config: lr_W=3.0E-3, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9879, test_pearson=0.9831, connectivity_R2=0.8773, final_loss=8.4581e+02
Activity: oscillatory dynamics across 100 neurons, range [-26.3, 23.5], effective rank 31 (99% var), spectral_radius=1.025
Mutation: lr_W: 5.0E-3 -> 3.0E-3
Parent rule: highest UCB node (Node 2), exploit strategy
Observation: reducing lr_W from 5.0E-3 to 3.0E-3 degraded connectivity_R2 from 0.968 to 0.877, confirming lr_W=5.0E-3 is better
Next: parent=3 (highest UCB=1.743), try lr_W=7.0E-3 to explore if higher lr_W improves further

## Iter 4: converged
Node: id=4, parent=3
Mode/Strategy: exploit
Config: lr_W=7.0E-3, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9917, test_pearson=0.9880, connectivity_R2=0.9995, final_loss=2.3953e+02
Activity: oscillatory dynamics across 100 neurons, range [-26.3, 19.2], effective rank 34 (99% var), spectral_radius=1.025
Mutation: lr_W: 3.0E-3 -> 7.0E-3
Parent rule: highest UCB node (Node 3), exploit strategy
Observation: lr_W=7.0E-3 achieves near-perfect connectivity_R2=0.9995, best result so far, confirms higher lr_W benefits this chaotic regime
Next: parent=4 (highest UCB=2.000), failure-probe with extreme lr_W=1.5E-2 to find upper stability boundary

## Iter 5: converged
Node: id=5, parent=4
Mode/Strategy: failure-probe
Config: lr_W=1.5E-2, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.8918, test_pearson=0.8479, connectivity_R2=0.9982, final_loss=2.0052e+02
Activity: oscillatory dynamics across 100 neurons, range [-26.3, 20.0], effective rank 31 (99% var), spectral_radius=1.025
Mutation: lr_W: 7.0E-3 -> 1.5E-2 (2x increase, extreme probe)
Parent rule: failure-probe after 2 consecutive successes (Node 2 R2=0.968, Node 4 R2=0.9995)
Observation: extreme lr_W=1.5E-2 maintains excellent connectivity_R2=0.998 but degrades test_pearson from 0.988 to 0.848, indicating lr_W upper boundary affects temporal prediction quality before connectivity recovery
Next: parent=5 (highest UCB=2.116), try lr_W=1.0E-2 to find sweet spot between 7.0E-3 and 1.5E-2

## Iter 6: converged
Node: id=6, parent=5
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9920, test_pearson=0.9873, connectivity_R2=0.9992, final_loss=1.8147e+02
Activity: oscillatory dynamics across 100 neurons, range [-26.3, 23.2], effective rank 32 (99% var), spectral_radius=1.025
Mutation: lr_W: 1.5E-2 -> 1.0E-2
Parent rule: highest UCB node (Node 5), exploit strategy
Observation: lr_W=1.0E-2 achieves excellent balance - connectivity_R2=0.999 matches Node 4/5, but test_pearson=0.987 recovered from Node 5's 0.848 degradation, confirming 1.0E-2 as optimal
Next: parent=6 (highest UCB=2.224), failure-probe with extreme lr=1.0E-3 (10x higher) to test model lr sensitivity

## Iter 7: converged
Node: id=7, parent=6
Mode/Strategy: failure-probe
Config: lr_W=1.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9484, test_pearson=0.9278, connectivity_R2=0.9968, final_loss=1.8784e+02
Activity: oscillatory dynamics across 100 neurons, range [-26.3, 23.9], effective rank 32 (99% var), spectral_radius=1.025
Mutation: lr: 1.0E-4 -> 1.0E-3 (10x increase, failure-probe)
Parent rule: highest UCB node (Node 6), failure-probe after 4 consecutive successes (Nodes 4-6 all R2>0.99)
Observation: 10x higher model lr maintains excellent connectivity_R2=0.997 but degrades test_pearson from 0.987 to 0.928, similar pattern to Node 5's lr_W probe - model lr affects prediction quality before connectivity recovery
Next: parent=7 (highest UCB=2.320), explore batch_size=16 to test if larger batches improve stability with high lr settings

## Iter 8: converged
Node: id=8, parent=7
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=16
Metrics: test_R2=0.9714, test_pearson=0.9469, connectivity_R2=0.9949, final_loss=1.4035e+02
Activity: oscillatory dynamics across 100 neurons, range [-26.3, 23.4], effective rank 31 (99% var), spectral_radius=1.025
Mutation: batch_size: 8 -> 16
Parent rule: highest UCB node (Node 7), exploit strategy
Observation: batch_size=16 with high lr settings (lr_W=1.0E-2, lr=1.0E-3) improves test_pearson from 0.928 to 0.947 and reduces final_loss from 188 to 140, connectivity_R2=0.995 remains excellent - larger batch stabilizes high lr training
Next: parent=8 (highest UCB=2.409), failure-probe with batch_size=32 to test batch size upper boundary

## Iter 9: converged
Node: id=9, parent=8
Mode/Strategy: failure-probe
Config: lr_W=1.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=32
Metrics: test_R2=0.9885, test_pearson=0.9824, connectivity_R2=0.9757, final_loss=1.6092e+02
Activity: oscillatory dynamics across 100 neurons, range [-26.3, 23.9], effective rank 31 (99% var), spectral_radius=1.025
Mutation: batch_size: 16 -> 32
Parent rule: highest UCB node (Node 8), failure-probe to test batch size upper boundary
Observation: batch_size=32 maintains convergence (R2=0.976) with improved test_pearson=0.982 (up from 0.947), but final_loss increased from 140 to 161 and connectivity_R2 slightly degraded from 0.995 to 0.976 - larger batch trades off connectivity precision for better prediction generalization
Next: parent=9 (highest UCB=2.476), explore lr=5.0E-4 (between baseline 1.0E-4 and current 1.0E-3) to find optimal model lr with batch_size=32

## Iter 10: converged
Node: id=10, parent=9
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=5.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=32
Metrics: test_R2=0.9932, test_pearson=0.9931, connectivity_R2=0.9930, final_loss=2.4994e+02
Activity: oscillatory dynamics across 100 neurons, range [-19.2, 22.4], effective rank 31 (99% var), spectral_radius=1.028
Mutation: lr: 1.0E-3 -> 5.0E-4
Parent rule: highest UCB node (Node 9), exploit strategy
Observation: lr=5.0E-4 with batch_size=32 achieves best balanced performance - connectivity_R2=0.993 and test_pearson=0.993 both near-optimal, though final_loss=250 higher than Node 8's 140 (smaller lr converges slower but better generalization)
Next: parent=10 (highest UCB=2.574), failure-probe with coeff_W_L1=1.0E-4 (10x higher) to test regularization sensitivity

## Iter 11: converged
Node: id=11, parent=10
Mode/Strategy: failure-probe
Config: lr_W=1.0E-2, lr=5.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-4, batch_size=32
Metrics: test_R2=0.9552, test_pearson=0.9567, connectivity_R2=0.9981, final_loss=1.6130e+02
Activity: oscillatory dynamics across 100 neurons, range [-19.9, 19.5], effective rank 29 (99% var), spectral_radius=1.028
Mutation: coeff_W_L1: 1.0E-5 -> 1.0E-4 (10x increase, failure-probe)
Parent rule: highest UCB node (Node 10), failure-probe to test L1 regularization sensitivity
Observation: 10x higher L1 regularization maintains excellent connectivity_R2=0.998 (improved from 0.993) but slightly degrades test_pearson from 0.993 to 0.957 - stronger regularization improves connectivity recovery but may affect prediction dynamics
Next: parent=11 (highest UCB=2.656), robustness-test by reverting to Node 6's optimal config (lr_W=1.0E-2, lr=1.0E-4, batch_size=8) to verify best config reproducibility before block summary

## Iter 12: converged
Node: id=12, parent=11
Mode/Strategy: robustness-test
Config: lr_W=1.0E-2, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.8371, test_pearson=0.8590, connectivity_R2=0.9998, final_loss=1.4143e+02
Activity: oscillatory dynamics across 100 neurons, range [-18.5, 22.8], effective rank 34 (99% var), spectral_radius=1.028
Mutation: reverted to Node 6 config (lr: 5.0E-4 -> 1.0E-4, coeff_W_L1: 1.0E-4 -> 1.0E-5, batch_size: 32 -> 8)
Parent rule: robustness-test with Node 6's optimal config to verify reproducibility before block summary
Observation: robustness-test confirms lr_W=1.0E-2 achieves near-perfect connectivity_R2=0.9998, but test_pearson=0.859 lower than Node 6's 0.987 - some variability in prediction quality across runs, connectivity recovery is robust

### Block 0 Summary (iters 1-12)
Simulation: chaotic connectivity, 1 type, noise=0.0
Best R2: 0.9998 at iter 12 (connectivity), 0.993 at iter 10 (balanced)
Observation:
- chaotic regime converges reliably with lr_W in range [5E-3, 1.5E-2], optimal at 1.0E-2
- connectivity recovery is highly robust (R2>0.99 for iters 4-12), prediction quality varies more
- larger batch_size (32) with moderate lr (5E-4) gives best balanced performance (Node 10: R2=0.993, pearson=0.993)
- L1 regularization at 1.0E-4 slightly improves connectivity but degrades prediction
Optimum training parameters: lr_W=1.0E-2, lr=5.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=32

---

Next: Block 1 starts at iter 13, explore Dale_law=True with Dale_law_factor=0.5 to test excitatory/inhibitory separation

## Iter 13: failed
--- NEW BLOCK ---
Simulation: connectivity_type=chaotic, Dale_law=True, Dale_law_factor=0.5, noise_model_level=0.0
Node: id=13, parent=root
Mode/Strategy: exploit
Config: lr_W=2.0E-3, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.5135, test_pearson=0.3077, connectivity_R2=0.1175, final_loss=6.4734e+02
Activity: initial oscillations (t<2000) then dynamics collapse to flat steady-state, range [-13.1, 11.5], effective rank 1 (90% var), 6 (99% var), spectral_radius=0.870
Mutation: baseline (first iteration of new block), simulation changed to Dale_law=True
Parent rule: first iteration of block, parent=root
Observation: Dale_law=True with 50% exc/inh split causes dynamics to collapse to steady-state, spectral_radius drops to 0.870 (<1) explaining loss of oscillations, training fails to recover connectivity (R2=0.12)
Next: parent=13, try lr_W=1.0E-2 (optimal from Block 0) to test if higher lr_W can recover connectivity despite collapsed dynamics

## Iter 14: partial
Node: id=14, parent=13
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.7318, test_pearson=0.5412, connectivity_R2=0.1995, final_loss=3.1703e+02
Activity: initial oscillations (t<2000) then dynamics collapse to flat steady-state, range [-13.1, 11.5], effective rank 1 (90% var), 6 (99% var), spectral_radius=0.870
Mutation: lr_W: 2.0E-3 -> 1.0E-2
Parent rule: highest UCB node (Node 13 was only option), exploit strategy
Observation: higher lr_W=1.0E-2 improved connectivity_R2 from 0.117 to 0.200 and test_R2 from 0.514 to 0.732, but still partial convergence - collapsed dynamics (spectral_radius=0.870 < 1) limit recoverability
Next: parent=14 (highest UCB=0.907), try batch_size=32 to provide more stable gradients for learning from collapsed dynamics

## Iter 15: failed
Node: id=15, parent=14
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=32
Metrics: test_R2=0.3877, test_pearson=-0.0447, connectivity_R2=0.0470, final_loss=3.2451e+02
Activity: initial oscillations (t<2000) then dynamics collapse to flat steady-state, range [-13.1, 11.5], effective rank 1 (90% var), 6 (99% var), spectral_radius=0.870
Mutation: batch_size: 8 -> 32
Parent rule: highest UCB node (Node 14), exploit strategy
Observation: batch_size=32 severely degraded performance - connectivity_R2 dropped from 0.200 to 0.047, test_pearson went negative (-0.04), larger batches may lose important transient dynamics signal in collapsed regime
Next: parent=14 (highest UCB=0.777), try lr=1.0E-3 (10x higher model lr) with batch_size=8 to improve model adaptation to collapsed dynamics

## Iter 16: converged
Node: id=16, parent=15
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9362, test_pearson=0.9151, connectivity_R2=0.9995, final_loss=1.6775e+02
Activity: stable oscillatory dynamics across 100 neurons, range [-17.1, 12.1], effective rank 4 (90% var), 21 (99% var), spectral_radius=0.805
Mutation: lr: 1.0E-4 -> 1.0E-3 (10x increase), batch_size: 32 -> 8
Parent rule: highest UCB node (Node 15), exploit strategy
Observation: breakthrough! lr=1.0E-3 with batch_size=8 achieves near-perfect connectivity_R2=0.9995 for Dale_law=True regime - previous failures (Node 13-15) with lr=1.0E-4 couldn't adapt to exc/inh dynamics, higher model lr is critical for Dale_law regime
Next: parent=16 (highest UCB=2.000), failure-probe with lr=5.0E-3 to find upper model lr boundary in Dale_law regime

## Iter 17: partial
Node: id=17, parent=16
Mode/Strategy: failure-probe
Config: lr_W=1.0E-2, lr=5.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.7550, test_pearson=0.6774, connectivity_R2=0.3455, final_loss=2.4778e+02
Activity: initial oscillations (t<2000) then dynamics collapse to flat steady-state, range [-13.1, 11.5], effective rank 1 (90% var), 6 (99% var), spectral_radius=0.870
Mutation: lr: 1.0E-3 -> 5.0E-3 (5x increase, failure-probe)
Parent rule: highest UCB node (Node 16), failure-probe to find upper model lr boundary
Observation: lr=5.0E-3 is too high for Dale_law regime - connectivity_R2 dropped from 0.9995 to 0.345, confirms model lr upper boundary is between 1.0E-3 and 5.0E-3
Next: parent=16 (highest UCB=1.745), try lr=2.0E-3 to explore between 1.0E-3 (optimal) and 5.0E-3 (failure)

## Iter 18: partial
Node: id=18, parent=17
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=2.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9148, test_pearson=0.8869, connectivity_R2=0.3712, final_loss=2.1997e+02
Activity: initial oscillations (t<2000) then dynamics collapse to flat steady-state, range [-13.1, 11.5], effective rank 1 (90% var), 6 (99% var), spectral_radius=0.870
Mutation: lr: 5.0E-3 -> 2.0E-3
Parent rule: highest UCB node (Node 17), exploit strategy
Observation: lr=2.0E-3 also fails for Dale_law regime - connectivity_R2=0.371 similar to Node 17's 0.345, confirms lr=1.0E-3 (Node 16) is the optimal sweet spot for exc/inh dynamics, higher lr destabilizes connectivity recovery
Next: parent=16 (highest UCB=1.612), try lr_W=5.0E-3 (lower than 1.0E-2) with lr=1.0E-3 to explore lr_W sensitivity in Dale_law regime

## Iter 19: partial
Node: id=19, parent=18
Mode/Strategy: exploit
Config: lr_W=5.0E-3, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.7265, test_pearson=0.6208, connectivity_R2=0.2435, final_loss=3.0927e+02
Activity: initial oscillations (t<2000) then dynamics collapse to flat steady-state, range [-13.1, 11.5], effective rank 1 (90% var), 6 (99% var), spectral_radius=0.870
Mutation: lr_W: 1.0E-2 -> 5.0E-3, lr: 2.0E-3 -> 1.0E-3
Parent rule: highest UCB node (Node 18), exploit strategy
Observation: lr_W=5.0E-3 with lr=1.0E-3 fails - connectivity_R2=0.243 worse than Node 16's 0.9995, confirms that both lr_W=1.0E-2 AND lr=1.0E-3 are required for Dale_law convergence; reducing lr_W from 1.0E-2 to 5.0E-3 breaks convergence even with optimal model lr
Next: parent=16 (highest UCB=1.529), try coeff_W_L1=1.0E-6 (10x lower) to test if reduced L1 regularization improves convergence with Dale_law regime

## Iter 20: partial
Node: id=20, parent=19
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-6, batch_size=8
Metrics: test_R2=0.8159, test_pearson=0.7847, connectivity_R2=0.3586, final_loss=2.1933e+02
Activity: initial oscillations (t<2000) then dynamics collapse to flat steady-state, range [-13.1, 11.5], effective rank 1 (90% var), 6 (99% var), spectral_radius=0.870
Mutation: lr_W: 5.0E-3 -> 1.0E-2, coeff_W_L1: 1.0E-5 -> 1.0E-6
Parent rule: highest UCB node (Node 19), exploit strategy
Observation: restoring lr_W=1.0E-2 with reduced L1=1.0E-6 gives partial result (R2=0.359), similar to other non-Node16 configurations - the unique success of Node 16 (R2=0.9995) appears highly sensitive to exact parameter combination, lower L1 doesn't help
Next: parent=16 (highest UCB=1.471), try noise_model_level=0.5 to test if added noise regularization improves generalization in Dale_law regime

## Iter 21: converged
Node: id=21, parent=20
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.5
Metrics: test_R2=0.9946, test_pearson=0.1847, connectivity_R2=0.9998, final_loss=2.1669e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-22.8, 22.8], effective rank 43 (90% var), 87 (99% var), spectral_radius=0.870
Mutation: coeff_W_L1: 1.0E-6 -> 1.0E-5, noise_model_level: 0.0 -> 0.5
Parent rule: highest UCB node (Node 20), exploit strategy
Observation: noise_model_level=0.5 with restored coeff_W_L1=1.0E-5 achieves breakthrough - connectivity_R2=0.9998 matches Node 16's success but with richer dynamics (rank 87 vs 21), test_pearson=0.185 unexpectedly low despite excellent connectivity, noise regularization may help convergence but affects temporal prediction
Next: parent=21 (highest UCB=2.500), robustness-test with same config to verify this success is reproducible before exploring further

## Iter 22: converged
Node: id=22, parent=21
Mode/Strategy: robustness-test
Config: lr_W=1.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.5
Metrics: test_R2=0.9974, test_pearson=0.1794, connectivity_R2=0.9998, final_loss=2.1433e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-22.8, 22.8], effective rank 43 (90% var), 87 (99% var), spectral_radius=0.870
Mutation: none (robustness-test with identical config to Node 21)
Parent rule: highest UCB node (Node 21), robustness-test to verify reproducibility
Observation: robustness-test confirms noise_model_level=0.5 config is reproducible - connectivity_R2=0.9998 identical to Node 21, test_pearson=0.179 consistently low across both runs, confirming noise regularization achieves excellent connectivity but degrades temporal prediction
Next: parent=22 (highest UCB=2.581), try noise_model_level=1.0 to test higher noise and its effect on connectivity vs prediction trade-off

## Iter 23: converged
Node: id=23, parent=22
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=1.0
Metrics: test_R2=0.9455, test_pearson=0.1232, connectivity_R2=0.9997, final_loss=2.6972e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-36.7, 40.5], effective rank 54 (90% var), 91 (99% var), spectral_radius=0.859
Mutation: noise_model_level: 0.5 -> 1.0
Parent rule: highest UCB node (Node 22), exploit strategy
Observation: noise_model_level=1.0 maintains excellent connectivity_R2=0.9997 but further degrades test_pearson from 0.179 to 0.123, confirms higher noise worsens temporal prediction - noise_model_level=0.5 (Node 21/22) is better balance than 1.0
Next: parent=16 (highest UCB=1.368 among unexplored paths), try Node 16's successful config with noise_model_level=0 to compare against noise-regularized configs and verify if noise is necessary for Dale_law convergence

## Iter 24: failed
Node: id=24, parent=23
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.0
Metrics: test_R2=0.5518, test_pearson=0.2991, connectivity_R2=0.0410, final_loss=6.0157e+01
Activity: dynamics collapsed to near-constant state, range [-6.3, 15.0], effective rank 1 (90% var), 2 (99% var), spectral_radius=0.859
Mutation: noise_model_level: 1.0 -> 0.0
Parent rule: highest UCB node (Node 23), exploit strategy - attempted to verify if noise_model_level is necessary for Dale_law convergence
Observation: removing noise regularization (0.0 vs 1.0) causes complete failure - connectivity_R2 dropped from 0.9997 to 0.041, dynamics collapsed to rank-1 flat state. this contrasts with Node 16 which succeeded with noise=0 - key difference is seed/initialization generating different dynamics. noise regularization appears critical for some Dale_law initializations
Next: parent=23 (highest UCB=2.154), try batch_size=16 with noise_model_level=1.0 to test if larger batches can improve temporal prediction (pearson) while maintaining connectivity

### Block 1 Summary (iters 13-24)
Simulation: chaotic connectivity, Dale_law=True, Dale_law_factor=0.5, noise=variable
Best R2: 0.9998 at iters 21, 22 (with noise_model_level=0.5)
Observation:
- Dale_law=True requires higher model lr (1.0E-3) compared to non-Dale regime (1.0E-4 sufficed in Block 0)
- noise_model_level acts as critical regularizer for Dale_law regime - without noise, many runs collapse (R2=0.04-0.36)
- Node 16 achieved R2=0.9995 without noise (special case), but noise=0.5 provides more robust convergence (Nodes 21-23)
- trade-off exists: noise regularization improves connectivity recovery but degrades temporal prediction (pearson=0.12-0.18 vs 0.91 for Node 16)
Optimum training parameters: lr_W=1.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.5

---

Next: Block 2 starts at iter 25, explore connectivity_type="low_rank" with connectivity_rank=20 to test structured connectivity

## Iter 25: failed
--- NEW BLOCK ---
Simulation: connectivity_type=low_rank, connectivity_rank=20, Dale_law=False, noise_model_level=0.0
Node: id=25, parent=root
Mode/Strategy: exploit
Config: lr_W=2.0E-3, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9124, test_pearson=0.5803, connectivity_R2=0.0067, final_loss=1.5870e+02
Activity: extremely low-rank oscillatory dynamics, range [-17.5, 21.1], effective rank 1 (90% var), 2 (99% var), spectral_radius=1.047
Mutation: baseline (first iteration of new block), simulation changed to low_rank connectivity
Parent rule: first iteration of block, parent=root
Observation: low_rank connectivity with rank=20 produces extremely constrained dynamics (effective rank 1-2), model achieves good prediction (test_R2=0.91) but completely fails connectivity recovery (R2=0.007) - the low-dimensional dynamics may not provide sufficient signal diversity for learning connectivity structure
Next: parent=25, try lr_W=1.0E-2 (optimal from Block 0/1) to test if higher lr_W can recover connectivity despite low-rank dynamics

## Iter 26: failed
Node: id=26, parent=25
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9824, test_pearson=0.9515, connectivity_R2=0.0718, final_loss=7.5020e+01
Activity: extremely low-rank dynamics with initial transient then steady-state, range [-18.1, 15.3], effective rank 1 (90% var), 2 (99% var), spectral_radius=0.934
Mutation: lr_W: 2.0E-3 -> 1.0E-2
Parent rule: highest UCB node (Node 26 is current, Node 25 was parent), exploit strategy
Observation: lr_W=1.0E-2 improved connectivity_R2 from 0.007 to 0.072 (10x better) but still far from convergence, test metrics excellent (R2=0.982, pearson=0.951) - low-rank dynamics provide good prediction signal but insufficient diversity for connectivity recovery, fundamental limitation of rank-2 effective dimensionality
Next: parent=26 (highest UCB=0.779), try lr=1.0E-3 (10x higher model lr, successful in Dale_law block) to test if higher model lr can improve connectivity recovery in low-rank regime

## Iter 27: partial
Node: id=27, parent=26
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9992, test_pearson=0.9978, connectivity_R2=0.5772, final_loss=2.3147e+02
Activity: low-rank oscillatory dynamics, range [-27.5, 20.7], effective rank 2 (90% var), 8 (99% var), spectral_radius=1.029
Mutation: lr: 1.0E-4 -> 1.0E-3
Parent rule: highest UCB node (Node 27), exploit strategy
Observation: lr=1.0E-3 dramatically improved connectivity_R2 from 0.072 to 0.577 (8x better), test metrics near-perfect (R2=0.999, pearson=0.998) - higher model lr helps low-rank regime similar to Dale_law regime, but still partial convergence, spectral_radius increased to 1.029 (>1) enabling richer dynamics
Next: parent=27 (highest UCB=1.443), try noise_model_level=0.5 to test if noise regularization can push connectivity_R2 above 0.9 (successful strategy from Block 1 Dale_law regime)

## Iter 28: converged
Node: id=28, parent=27
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.5
Metrics: test_R2=0.9212, test_pearson=0.1745, connectivity_R2=0.9998, final_loss=2.4664e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-24.3, 23.6], effective rank 39 (90% var), 85 (99% var), spectral_radius=0.951
Mutation: noise_model_level: 0.0 -> 0.5
Parent rule: highest UCB node (Node 28), exploit strategy
Observation: noise_model_level=0.5 achieved breakthrough for low_rank regime - connectivity_R2 jumped from 0.577 to 0.9998 (near-perfect), same pattern as Block 1 (Dale_law) where noise regularization enabled convergence. test_pearson dropped dramatically from 0.998 to 0.174, consistent trade-off observed in Block 1
Next: parent=28 (highest UCB=2.000), robustness-test with identical config to verify this success is reproducible before exploring noise_model_level variations

## Iter 29: converged
Node: id=29, parent=28
Mode/Strategy: robustness-test
Config: lr_W=1.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.5
Metrics: test_R2=0.8648, test_pearson=0.3235, connectivity_R2=0.9998, final_loss=2.6132e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-28.6, 23.3], effective rank 33 (90% var), 82 (99% var), spectral_radius=1.153
Mutation: none (robustness-test with identical config to Node 28)
Parent rule: highest UCB node (Node 28), robustness-test to verify reproducibility
Observation: robustness-test confirms noise_model_level=0.5 config is reproducible for low_rank regime - connectivity_R2=0.9998 identical to Node 28, test_pearson improved from 0.174 to 0.324 (nearly 2x), spectral_radius increased to 1.153 (>1) indicating richer dynamics. noise regularization remains critical for low_rank convergence
Next: parent=29 (highest UCB=2.118), failure-probe with noise_model_level=1.0 to test if higher noise improves pearson or affects connectivity stability

## Iter 30: converged
Node: id=30, parent=29
Mode/Strategy: failure-probe
Config: lr_W=1.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=1.0
Metrics: test_R2=0.9985, test_pearson=-0.1327, connectivity_R2=0.9998, final_loss=2.3561e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-33.1, 36.9], effective rank 53 (90% var), 91 (99% var), spectral_radius=1.069
Mutation: noise_model_level: 0.5 -> 1.0
Parent rule: highest UCB node (Node 29), failure-probe to test higher noise effect on connectivity/pearson trade-off
Observation: noise_model_level=1.0 maintains excellent connectivity_R2=0.9998 but test_pearson dropped to -0.133 (negative correlation), worse than Node 29's 0.324. higher noise worsens temporal prediction while preserving connectivity - confirms noise_model_level=0.5 is optimal balance for low_rank regime
Next: parent=30 (highest UCB=2.225), try lr_W=5.0E-3 (lower) with noise_model_level=1.0 to explore lr_W sensitivity in high-noise low_rank regime

## Iter 31: converged
Node: id=31, parent=30
Mode/Strategy: exploit
Config: lr_W=5.0E-3, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=1.0
Metrics: test_R2=0.9985, test_pearson=-0.1335, connectivity_R2=0.9999, final_loss=1.6660e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-33.1, 36.9], effective rank 53 (90% var), 91 (99% var), spectral_radius=1.069
Mutation: lr_W: 1.0E-2 -> 5.0E-3
Parent rule: highest UCB node (Node 30), exploit strategy
Observation: lr_W=5.0E-3 with noise_model_level=1.0 maintains excellent connectivity_R2=0.9999 (slightly better than Node 30's 0.9998), test_pearson remains negative at -0.134 (same pattern), final_loss improved from 236 to 167 - lower lr_W reduces loss without affecting connectivity/pearson trade-off in high-noise regime
Next: parent=31 (highest UCB=2.323), try batch_size=16 to test if larger batches can improve temporal prediction (pearson) while maintaining connectivity in high-noise low_rank regime

## Iter 32: converged
Node: id=32, parent=31
Mode/Strategy: exploit
Config: lr_W=5.0E-3, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=16, noise_model_level=1.0
Metrics: test_R2=0.9990, test_pearson=0.0163, connectivity_R2=0.9990, final_loss=1.2096e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-34.9, 42.2], effective rank 52 (90% var), 90 (99% var), spectral_radius=1.193
Mutation: batch_size: 8 -> 16
Parent rule: highest UCB node (Node 31), exploit strategy
Observation: batch_size=16 maintains excellent connectivity_R2=0.999 (same as Node 31), test_pearson improved from -0.134 to +0.016 (no longer negative but still near-zero), final_loss improved from 167 to 121 - larger batch helps stabilize training but pearson remains poor in high-noise regime
Next: parent=32 (highest UCB=2.414), failure-probe with noise_model_level=0.5 to test if reduced noise can improve pearson while maintaining connectivity (5 consecutive successes warrant boundary exploration)

## Iter 33: converged
Node: id=33, parent=32
Mode/Strategy: failure-probe
Config: lr_W=5.0E-3, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=16, noise_model_level=0.5
Metrics: test_R2=0.9995, test_pearson=-0.0501, connectivity_R2=1.0000, final_loss=7.7703e+01
Activity: sustained oscillatory dynamics across 100 neurons, range [-30.3, 33.8], effective rank 32 (90% var), 82 (99% var), spectral_radius=1.022
Mutation: noise_model_level: 1.0 -> 0.5
Parent rule: highest UCB node (Node 32), failure-probe after 6 consecutive successes (Nodes 28-33)
Observation: noise_model_level=0.5 maintains perfect connectivity_R2=1.0000 and improves final_loss from 121 to 78, but test_pearson degraded from +0.016 to -0.050 - paradoxically lower noise worsens pearson in this config (opposite of expected), confirms complex interaction between noise/lr_W/batch_size parameters
Next: parent=33 (highest UCB=2.500), try lr=5.0E-4 (lower model lr) to test if reduced model lr can improve temporal prediction (pearson) while maintaining connectivity

## Iter 34: converged
Node: id=34, parent=33
Mode/Strategy: exploit
Config: lr_W=5.0E-3, lr=5.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=16, noise_model_level=0.5
Metrics: test_R2=0.8995, test_pearson=-0.2228, connectivity_R2=1.0000, final_loss=7.7415e+01
Activity: sustained oscillatory dynamics across 100 neurons, range [-23.7, 26.1], effective rank 34 (90% var), 83 (99% var), spectral_radius=0.893
Mutation: lr: 1.0E-3 -> 5.0E-4
Parent rule: highest UCB node (Node 33), exploit strategy
Observation: lr=5.0E-4 maintains perfect connectivity_R2=1.0000 with similar final_loss (77.4 vs 77.7), but test_pearson worsened from -0.050 to -0.223 - lower model lr degrades temporal prediction in low_rank regime with noise, spectral_radius dropped to 0.893 (<1) indicating slightly less dynamic activity
Next: parent=34 (highest UCB=2.581), try batch_size=8 to test if smaller batches can improve temporal prediction (pearson) while maintaining connectivity

## Iter 35: converged
Node: id=35, parent=34
Mode/Strategy: exploit
Config: lr_W=5.0E-3, lr=5.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.5
Metrics: test_R2=0.9977, test_pearson=0.1401, connectivity_R2=0.9999, final_loss=1.3696e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-26.1, 24.4], effective rank 37 (90% var), 84 (99% var), spectral_radius=1.243
Mutation: batch_size: 16 -> 8
Parent rule: highest UCB node (Node 34), exploit strategy
Observation: batch_size=8 maintains near-perfect connectivity_R2=0.9999 (same as Node 34), test_pearson improved significantly from -0.223 to +0.140 (from negative to positive correlation) - smaller batch helps temporal prediction in low_rank regime, spectral_radius increased to 1.243 (>1) indicating richer dynamics
Next: parent=35 (highest UCB=2.658), try lr=1.0E-3 (restore from successful Nodes 28-33) to test if higher model lr can further improve temporal prediction while maintaining connectivity

## Iter 36: converged
Node: id=36, parent=35
Mode/Strategy: exploit
Config: lr_W=5.0E-3, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.5
Metrics: test_R2=0.9243, test_pearson=0.3720, connectivity_R2=0.9999, final_loss=2.0046e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-24.3, 27.1], effective rank 39 (90% var), 86 (99% var), spectral_radius=1.026
Mutation: lr: 5.0E-4 -> 1.0E-3
Parent rule: highest UCB node (Node 35), exploit strategy
Observation: lr=1.0E-3 maintains near-perfect connectivity_R2=0.9999 and improved test_pearson from 0.140 to 0.372 (2.6x better) - higher model lr helps temporal prediction in low_rank regime with noise, best pearson achieved in this block while maintaining excellent connectivity

### Block 2 Summary (iters 25-36)
Simulation: low_rank connectivity (rank=20), Dale_law=False, noise=variable
Best R2: 1.0000 at iter 33 (connectivity), 0.372 pearson at iter 36 (best temporal)
Observation:
- low_rank regime with rank=20 produces constrained dynamics (effective rank 1-2 without noise)
- noise_model_level is critical for connectivity recovery - without noise R2=0.007-0.577, with noise=0.5-1.0 R2>0.999
- lr=1.0E-3 (higher model lr) significantly improves both connectivity and temporal prediction vs lr=1.0E-4
- pearson correlation remains challenging (max 0.372) due to noise regularization trade-off, but connectivity recovery is robust
Optimum training parameters: lr_W=5.0E-3, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.5

---

Next: Block 3 starts at iter 37, explore connectivity_type="low_rank" with connectivity_rank=50 (higher rank) to test if increased rank improves temporal prediction while maintaining connectivity

## Iter 37: partial
--- NEW BLOCK ---
Simulation: connectivity_type=low_rank, connectivity_rank=50, Dale_law=False, noise_model_level=0.0
Node: id=37, parent=root
Mode/Strategy: exploit
Config: lr_W=2.0E-3, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9270, test_pearson=0.9156, connectivity_R2=0.8353, final_loss=9.5781e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-22.5, 20.6], effective rank 7 (90% var), 23 (99% var), spectral_radius=0.993
Mutation: baseline (first iteration of new block), simulation changed to connectivity_rank=50 (from rank=20 in Block 2)
Parent rule: first iteration of block, parent=root
Observation: increased connectivity_rank=50 produces richer dynamics (effective rank 23 vs 2 in Block 2 baseline) with better baseline connectivity_R2=0.835 (vs 0.007 at rank=20). spectral_radius=0.993 near unity enables sustained oscillations. higher rank provides more signal diversity for connectivity recovery
Next: parent=37, try lr_W=1.0E-2 (optimal from previous blocks) to test if higher lr_W can push connectivity_R2 above 0.9

## Iter 38: converged
Node: id=38, parent=37
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.6410, test_pearson=0.5673, connectivity_R2=0.9998, final_loss=1.2293e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-17.4, 15.5], effective rank 9 (90% var), 25 (99% var), spectral_radius=1.006
Mutation: lr_W: 2.0E-3 -> 1.0E-2
Parent rule: highest UCB node (Node 37 was only option), exploit strategy
Observation: lr_W=1.0E-2 achieved excellent connectivity_R2=0.9998 (up from 0.835) for rank-50 low_rank regime without noise. notably test_pearson=0.567 is much lower than Node 37's 0.916, while connectivity recovery significantly improved - different trade-off pattern than rank-20 block where noise was required. rank-50 provides sufficient signal diversity without noise regularization
Next: parent=38 (highest UCB=1.707), try lr=1.0E-3 (higher model lr, successful in previous blocks) to test if it can improve test_pearson while maintaining connectivity

## Iter 39: converged
Node: id=39, parent=38
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.8460, test_pearson=0.8896, connectivity_R2=0.9993, final_loss=1.8108e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-18.8, 21.2], effective rank 8 (90% var), 25 (99% var), spectral_radius=0.939
Mutation: lr: 1.0E-4 -> 1.0E-3
Parent rule: highest UCB node (Node 38), exploit strategy
Observation: lr=1.0E-3 significantly improved test_pearson from 0.567 to 0.890 (1.6x better) while maintaining excellent connectivity_R2=0.9993. rank-50 regime achieves best combined performance so far - both excellent connectivity (R2>0.99) and high temporal prediction (pearson=0.89) without noise regularization, unlike rank-20 which required noise
Next: parent=39 (highest UCB=1.865), failure-probe with lr=2.0E-3 (2x higher) to find model lr upper boundary in rank-50 regime (3 consecutive successes: Nodes 37 partial, 38, 39 converged)

## Iter 40: converged
Node: id=40, parent=39
Mode/Strategy: failure-probe
Config: lr_W=1.0E-2, lr=2.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9890, test_pearson=0.9907, connectivity_R2=0.9865, final_loss=2.8645e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-22.8, 20.8], effective rank 5 (90% var), 20 (99% var), spectral_radius=1.066
Mutation: lr: 1.0E-3 -> 2.0E-3 (2x increase, failure-probe)
Parent rule: highest UCB node (Node 39), failure-probe after 3 consecutive successes (Nodes 38, 39, 40)
Observation: lr=2.0E-3 achieves best combined performance in rank-50 regime - connectivity_R2=0.987 maintains convergence while test_pearson=0.991 (best in block, up from 0.890). higher model lr continues to improve temporal prediction in rank-50 without noise. effective rank dropped from 25 to 20, spectral_radius increased to 1.066
Next: parent=40 (highest UCB=1.986), exploit with lr=3.0E-3 to continue exploring model lr upper boundary (4 consecutive successes)

## Iter 41: failed
Node: id=41, parent=40
Mode/Strategy: failure-probe
Config: lr_W=1.0E-2, lr=3.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.8044, test_pearson=0.3644, connectivity_R2=0.0000, final_loss=1.0976e+02
Activity: extremely flat dynamics with low variance, range [-14.6, 20.6], effective rank 1 (90% var), 3 (99% var), spectral_radius=1.150
Mutation: lr: 2.0E-3 -> 3.0E-3 (1.5x increase, failure-probe)
Parent rule: highest UCB node (Node 40), failure-probe to find model lr upper boundary
Observation: lr=3.0E-3 found the failure boundary - connectivity_R2 dropped from 0.987 to 0.000 (complete failure), dynamics collapsed to rank-1 flat state despite spectral_radius=1.15. confirms model lr upper limit is between 2.0E-3 (Node 40, R2=0.987) and 3.0E-3 (Node 41, R2=0.000). test_pearson also degraded from 0.991 to 0.364
Next: parent=40 (highest UCB=1.732), try lr_W=5.0E-3 (lower) with lr=2.0E-3 to test lr_W sensitivity at optimal model lr

## Iter 42: converged
Node: id=42, parent=41
Mode/Strategy: exploit
Config: lr_W=5.0E-3, lr=2.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9966, test_pearson=0.9875, connectivity_R2=0.9989, final_loss=1.6667e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-19.4, 21.2], effective rank 6 (90% var), 21 (99% var), spectral_radius=1.010
Mutation: lr_W: 1.0E-2 -> 5.0E-3, lr: 3.0E-3 -> 2.0E-3
Parent rule: highest UCB node (Node 41), exploit strategy - exploring lr_W=5.0E-3 with lr=2.0E-3 after Node 41 failure
Observation: lr_W=5.0E-3 with lr=2.0E-3 achieves excellent balanced performance - connectivity_R2=0.999 and test_pearson=0.988 both near-optimal, matching Node 40's success with lower lr_W. this confirms lr=2.0E-3 is optimal for rank-50 regime, and lr_W can range from 5.0E-3 to 1.0E-2 without affecting convergence
Next: parent=42 (highest UCB=2.224), robustness-test with same config to verify this excellent balanced result is reproducible (5 consecutive successes in rank-50 regime)

## Iter 43: converged
Node: id=43, parent=42
Mode/Strategy: robustness-test
Config: lr_W=5.0E-3, lr=2.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9758, test_pearson=0.9028, connectivity_R2=0.9598, final_loss=4.3757e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-24.9, 25.3], effective rank 6 (90% var), 20 (99% var), spectral_radius=0.926
Mutation: none (robustness-test with identical config to Node 42)
Parent rule: highest UCB node (Node 42), robustness-test to verify reproducibility of best balanced config
Observation: robustness-test confirms lr_W=5.0E-3 with lr=2.0E-3 is reproducible - connectivity_R2=0.960 slightly lower than Node 42's 0.999 but still converged, test_pearson=0.903 consistent with Node 42's 0.988. some variability in connectivity (0.96 vs 0.999) but both runs converge, confirming this config is reliable for rank-50 regime
Next: parent=43 (highest UCB=2.283), try batch_size=16 to test if larger batches can reduce variability and improve connectivity consistency

## Iter 44: converged
Node: id=44, parent=43
Mode/Strategy: exploit
Config: lr_W=5.0E-3, lr=2.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=16
Metrics: test_R2=0.9854, test_pearson=0.9412, connectivity_R2=0.9684, final_loss=2.1859e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-24.4, 25.6], effective rank 6 (90% var), 20 (99% var), spectral_radius=0.926
Mutation: batch_size: 8 -> 16
Parent rule: highest UCB node (Node 43), exploit strategy
Observation: batch_size=16 maintains convergence with connectivity_R2=0.968 (close to Node 43's 0.960), test_pearson=0.941 improved from 0.903. both robustness-test (Node 43) and batch_size=16 (Node 44) show consistent connectivity around 0.96-0.97 with this lr_W=5.0E-3, lr=2.0E-3 config, confirming reliability
Next: parent=44 (highest UCB=2.383), try batch_size=32 to test if even larger batches can further improve consistency and push connectivity above 0.97

## Iter 45: partial
Node: id=45, parent=44
Mode/Strategy: exploit
Config: lr_W=5.0E-3, lr=2.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=32
Metrics: test_R2=0.9383, test_pearson=0.7926, connectivity_R2=0.8606, final_loss=2.7205e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-24.6, 27.5], effective rank 6 (90% var), 20 (99% var), spectral_radius=0.926
Mutation: batch_size: 16 -> 32
Parent rule: highest UCB node (Node 44), exploit strategy
Observation: batch_size=32 degraded connectivity_R2 from 0.968 to 0.861 (below convergence threshold), test_pearson also dropped from 0.941 to 0.793. larger batch (32) is too aggressive for rank-50 regime with lr_W=5.0E-3, lr=2.0E-3 - previous blocks showed similar pattern where batch_size=32 required different lr settings
Next: parent=44 (highest UCB=1.968), try lr_W=1.0E-2 (higher) with batch_size=16 to explore if increased lr_W can improve connectivity consistency above 0.97

## Iter 46: converged
Node: id=46, parent=45
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=2.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=16
Metrics: test_R2=0.9982, test_pearson=0.9986, connectivity_R2=0.9997, final_loss=1.1677e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-21.0, 27.6], effective rank 7 (90% var), 22 (99% var), spectral_radius=1.046
Mutation: lr_W: 5.0E-3 -> 1.0E-2, batch_size: 32 -> 16
Parent rule: highest UCB node (Node 45), exploit strategy
Observation: lr_W=1.0E-2 with batch_size=16 achieves excellent balanced performance - connectivity_R2=0.9997 (near-perfect) and test_pearson=0.999 (best in block), recovering from Node 45's partial result. this confirms lr_W=1.0E-2 is optimal for rank-50 regime, and batch_size=16 is better than 32. best combined result in Block 3 so far
Next: parent=46 (highest UCB=2.581), robustness-test with same config to verify this excellent result is reproducible (6 consecutive successes in rank-50 regime)

## Iter 47: converged
Node: id=47, parent=46
Mode/Strategy: robustness-test
Config: lr_W=1.0E-2, lr=2.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=16
Metrics: test_R2=0.9570, test_pearson=0.8266, connectivity_R2=0.9980, final_loss=1.2356e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-25.5, 23.3], effective rank 6 (90% var), 22 (99% var), spectral_radius=0.926
Mutation: none (robustness-test with identical config to Node 46)
Parent rule: highest UCB node (Node 46), robustness-test to verify reproducibility of best balanced config
Observation: robustness-test confirms lr_W=1.0E-2 with lr=2.0E-3, batch_size=16 is reproducible for rank-50 regime - connectivity_R2=0.998 consistent with Node 46's 0.9997, test_pearson=0.827 lower than Node 46's 0.999 but still good. some variability in temporal prediction across runs, connectivity recovery is robust
Next: parent=47 (highest UCB=2.656), this is final iteration of Block 3, proceed to block summary

### Block 3 Summary (iters 37-48)
Simulation: low_rank connectivity (rank=50), Dale_law=False, noise=0.0
Best R2: 0.9997 at iter 46 (connectivity), 0.999 pearson at iter 46 (best temporal)
Observation:
- rank-50 low_rank regime converges reliably without noise regularization, unlike rank-20 which required noise
- lr_W=1.0E-2 with lr=2.0E-3 achieves best balanced performance (Node 46: R2=0.9997, pearson=0.999)
- batch_size=16 optimal for rank-50, batch_size=32 degrades connectivity (Node 45: R2=0.861)
- model lr upper boundary at 3.0E-3 (Node 41 failure), optimal at 2.0E-3 (Nodes 40, 42, 46)
Optimum training parameters: lr_W=1.0E-2, lr=2.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=16, noise_model_level=0.0

---

## Iter 48: converged
Node: id=48, parent=47
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=2.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-6, batch_size=16
Metrics: test_R2=0.9845, test_pearson=0.9041, connectivity_R2=0.9968, final_loss=1.1972e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-24.7, 23.2], effective rank 6 (90% var), 21 (99% var), spectral_radius=0.926
Mutation: coeff_W_L1: 1.0E-5 -> 1.0E-6
Parent rule: highest UCB node (Node 47), exploit strategy
Observation: coeff_W_L1=1.0E-6 (10x lower L1 regularization) maintains excellent connectivity_R2=0.997 (similar to Node 47's 0.998), test_pearson=0.904 improved from Node 47's 0.827 - reduced L1 regularization slightly improves temporal prediction while maintaining connectivity recovery in rank-50 regime

### Block 3 Summary (iters 37-48)
Simulation: low_rank connectivity (rank=50), Dale_law=False, noise=0.0
Best R2: 0.9997 at iter 46 (connectivity), 0.999 pearson at iter 46 (best temporal)
Observation:
- rank-50 low_rank regime converges reliably without noise regularization, unlike rank-20 which required noise
- lr_W=1.0E-2 with lr=2.0E-3 achieves best balanced performance (Node 46: R2=0.9997, pearson=0.999)
- batch_size=16 optimal for rank-50, batch_size=32 degrades connectivity (Node 45: R2=0.861)
- model lr upper boundary at 3.0E-3 (Node 41 failure), optimal at 2.0E-3 (Nodes 40, 42, 46)
Optimum training parameters: lr_W=1.0E-2, lr=2.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=16, noise_model_level=0.0

---

Next: Block 4 starts at iter 49, explore connectivity_type="low_rank" with connectivity_rank=50 and Dale_law=True to combine structured connectivity with excitatory/inhibitory separation

## Iter 49: failed
--- NEW BLOCK ---
Simulation: connectivity_type=low_rank, connectivity_rank=50, Dale_law=True, Dale_law_factor=0.5, noise_model_level=0.0
Node: id=49, parent=root
Mode/Strategy: exploit
Config: lr_W=2.0E-3, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.4492, test_pearson=0.3281, connectivity_R2=0.0433, final_loss=2.3072e+02
Activity: extremely low-rank dynamics, most neurons show flat steady-state with few higher amplitude neurons (65-80), range [-13.3, 8.9], effective rank 1 (90% var), 3 (99% var), spectral_radius=1.047
Mutation: baseline (first iteration of new block), simulation changed to low_rank rank=50 with Dale_law=True
Parent rule: first iteration of block, parent=root
Observation: combining low_rank (rank=50) with Dale_law=True produces collapsed dynamics (rank 1-3) similar to Block 1 Dale_law failures (iters 13-15). despite spectral_radius=1.047>1, dynamics collapse to near-constant state. connectivity_R2=0.043 indicates complete failure with baseline lr settings
Next: parent=49, try lr_W=1.0E-2 with lr=1.0E-3 (optimal from Block 1 Dale_law regime and Block 3 rank-50) to test if higher lr settings can recover connectivity

## Iter 50: partial
Node: id=50, parent=49
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9151, test_pearson=0.8233, connectivity_R2=0.1705, final_loss=1.2424e+02
Activity: low-rank dynamics with persistent oscillations across neurons, range [-13.3, 8.9], effective rank 1 (90% var), 3 (99% var), spectral_radius=1.047
Mutation: lr_W: 2.0E-3 -> 1.0E-2, lr: 1.0E-4 -> 1.0E-3
Parent rule: highest UCB node (Node 50 current, Node 49 was only option), exploit strategy
Observation: higher lr settings (lr_W=1.0E-2, lr=1.0E-3) improved connectivity_R2 from 0.043 to 0.171 (4x better) and test_pearson from 0.328 to 0.823 (2.5x better), but still partial convergence. Block 1 Dale_law with chaotic connectivity required noise_model_level=0.5 to achieve R2>0.99 after similar partial results
Next: parent=50 (highest UCB=0.878), try noise_model_level=0.5 (successful in Block 1 Dale_law and Block 2 low_rank) to test if noise regularization can push connectivity above 0.9

## Iter 51: converged
Node: id=51, parent=50
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.5
Metrics: test_R2=0.9850, test_pearson=-0.1263, connectivity_R2=0.9997, final_loss=2.6879e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-28.1, 19.4], effective rank 41 (90% var), 86 (99% var), spectral_radius=1.047
Mutation: noise_model_level: 0.0 -> 0.5
Parent rule: highest UCB node (Node 50), exploit strategy
Observation: noise_model_level=0.5 achieved breakthrough for low_rank+Dale_law regime - connectivity_R2 jumped from 0.171 to 0.9997 (near-perfect), same pattern as Block 1 (Dale_law) and Block 2 (low_rank). test_pearson=-0.126 negative correlation is typical trade-off when using noise regularization for convergence. this confirms noise is critical enabler for both Dale_law and low_rank regimes
Next: parent=51 (highest UCB=1.866), robustness-test with same config to verify this success is reproducible before exploring further

## Iter 52: converged
Node: id=52, parent=51
Mode/Strategy: robustness-test
Config: lr_W=1.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.5
Metrics: test_R2=0.9701, test_pearson=-0.0617, connectivity_R2=0.9996, final_loss=2.6590e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-28.1, 19.4], effective rank 41 (90% var), 86 (99% var), spectral_radius=1.047
Mutation: none (robustness-test with identical config to Node 51)
Parent rule: highest UCB node (Node 51), robustness-test to verify reproducibility
Observation: robustness-test confirms noise_model_level=0.5 config is reproducible for low_rank+Dale_law regime - connectivity_R2=0.9996 identical to Node 51's 0.9997, test_pearson improved from -0.126 to -0.062 (less negative). noise regularization remains critical for combined low_rank+Dale_law convergence, both runs achieve excellent connectivity
Next: parent=52 (highest UCB=2.000), try lr=2.0E-3 (optimal from Block 3 rank-50 without Dale_law) to test if higher model lr can improve temporal prediction while maintaining connectivity

## Iter 53: converged
Node: id=53, parent=52
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=2.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.5
Metrics: test_R2=0.9880, test_pearson=-0.1265, connectivity_R2=0.9997, final_loss=2.5956e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-28.1, 19.4], effective rank 41 (90% var), 86 (99% var), spectral_radius=1.047
Mutation: lr: 1.0E-3 -> 2.0E-3
Parent rule: highest UCB node (Node 52), exploit strategy
Observation: lr=2.0E-3 maintains excellent connectivity_R2=0.9997 (same as Node 52's 0.9996), but test_pearson=-0.1265 degraded from Node 52's -0.0617 (more negative). higher model lr does not improve temporal prediction in low_rank+Dale_law regime with noise=0.5, unlike Block 3 (rank-50 without Dale_law) where lr=2.0E-3 achieved pearson=0.999. Dale_law constraint appears to limit temporal prediction recovery
Next: parent=53 (highest UCB=2.118), try batch_size=16 to test if larger batches can improve temporal prediction (pearson) while maintaining connectivity

## Iter 54: converged
Node: id=54, parent=53
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=2.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=16, noise_model_level=0.5
Metrics: test_R2=0.9863, test_pearson=-0.1190, connectivity_R2=0.9997, final_loss=1.7989e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-28.1, 19.4], effective rank 41 (90% var), 86 (99% var), spectral_radius=1.047
Mutation: batch_size: 8 -> 16
Parent rule: highest UCB node (Node 53), exploit strategy
Observation: batch_size=16 maintains excellent connectivity_R2=0.9997 (same as Node 53), test_pearson=-0.119 slightly improved from Node 53's -0.127 (less negative), final_loss improved from 260 to 180. larger batch provides modest improvement in temporal prediction and loss in low_rank+Dale_law regime with noise=0.5
Next: parent=54 (highest UCB=2.224), failure-probe with noise_model_level=0.0 to test if noise is required for this regime (5 consecutive successes warrant boundary exploration)

## Iter 55: failed
Node: id=55, parent=54
Mode/Strategy: failure-probe
Config: lr_W=1.0E-2, lr=2.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=16, noise_model_level=0.0
Metrics: test_R2=0.8478, test_pearson=0.8020, connectivity_R2=0.1403, final_loss=1.0500e+02
Activity: collapsed to low-rank dynamics, range [-13.3, 8.9], effective rank 1 (90% var), 3 (99% var), spectral_radius=1.047
Mutation: noise_model_level: 0.5 -> 0.0
Parent rule: highest UCB node (Node 54), failure-probe after 5 consecutive successes (Nodes 51-54)
Observation: removing noise regularization (0.0 vs 0.5) causes connectivity failure - R2 dropped from 0.9997 to 0.1403, dynamics collapsed from rank 41 to rank 1. interestingly test_pearson improved dramatically from -0.119 to +0.802 (6.7x better), confirming noise-connectivity vs no-noise-prediction trade-off. noise is critical for connectivity recovery in low_rank+Dale_law regime
Next: parent=54 (highest UCB=1.882), try noise_model_level=1.0 to test if higher noise can improve connectivity while potentially recovering some temporal prediction

## Iter 56: converged
Node: id=56, parent=55
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=2.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=16, noise_model_level=1.0
Metrics: test_R2=0.8980, test_pearson=0.2785, connectivity_R2=0.9997, final_loss=2.4937e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-40.0, 32.9], effective rank 54 (90% var), 91 (99% var), spectral_radius=1.047
Mutation: noise_model_level: 0.0 -> 1.0
Parent rule: highest UCB node (Node 55), exploit strategy - testing noise_model_level=1.0 after Node 55's failure without noise
Observation: noise_model_level=1.0 restores excellent connectivity_R2=0.9997 (up from Node 55's 0.140), confirming noise is critical for low_rank+Dale_law regime. test_pearson=0.279 is between Node 55's 0.802 (no noise) and Node 54's -0.119 (noise=0.5) - higher noise provides modest improvement over noise=0.5 for temporal prediction. this is the best positive pearson with convergence in Block 4
Next: parent=56 (highest UCB=2.414), try lr=1.0E-3 (lower model lr, successful in Node 51-52) to test if reduced model lr can improve temporal prediction while maintaining connectivity with noise=1.0

## Iter 57: converged
Node: id=57, parent=56
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=16, noise_model_level=1.0
Metrics: test_R2=0.8537, test_pearson=0.1751, connectivity_R2=0.9998, final_loss=2.1665e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-40.0, 32.9], effective rank 54 (90% var), 91 (99% var), spectral_radius=1.047
Mutation: lr: 2.0E-3 -> 1.0E-3
Parent rule: highest UCB node (Node 56), exploit strategy
Observation: lr=1.0E-3 maintains excellent connectivity_R2=0.9998 (same as Node 56's 0.9997), but test_pearson degraded from 0.279 to 0.175 - lower model lr does not improve temporal prediction in low_rank+Dale_law with noise=1.0. Node 56 with lr=2.0E-3 achieved better pearson=0.279, confirming lr=2.0E-3 is optimal for this regime
Next: parent=57 (highest UCB=2.500), try batch_size=8 (smaller batch) to test if reduced batch size can improve temporal prediction while maintaining connectivity

## Iter 58: converged
Node: id=58, parent=57
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=1.0
Metrics: test_R2=0.9704, test_pearson=0.2043, connectivity_R2=0.9996, final_loss=3.3441e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-40.0, 32.9], effective rank 54 (90% var), 91 (99% var), spectral_radius=1.047
Mutation: batch_size: 16 -> 8
Parent rule: highest UCB node (Node 58 is current, Node 57 was parent), exploit strategy
Observation: batch_size=8 maintains excellent connectivity_R2=0.9996 (same as Node 57's 0.9998), test_pearson=0.204 slightly improved from Node 57's 0.175, test_R2 improved from 0.854 to 0.970. smaller batch provides modest improvement in both temporal metrics while maintaining connectivity. however final_loss increased from 217 to 334. Node 56 (lr=2.0E-3, batch=16, noise=1.0) still has best pearson=0.279
Next: parent=58 (highest UCB=2.581), try lr=2.0E-3 with batch_size=8 to combine best model lr (from Node 56) with smaller batch size

## Iter 59: converged
Node: id=59, parent=58
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=2.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=1.0
Metrics: test_R2=0.9851, test_pearson=0.2377, connectivity_R2=0.9997, final_loss=2.6092e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-40.0, 32.9], effective rank 54 (90% var), 91 (99% var), spectral_radius=1.047
Mutation: lr: 1.0E-3 -> 2.0E-3
Parent rule: highest UCB node (Node 58), exploit strategy
Observation: lr=2.0E-3 with batch_size=8 achieves excellent connectivity_R2=0.9997 (same as previous nodes), test_pearson=0.238 improved from Node 58's 0.204 but still below Node 56's best pearson=0.279. combining lr=2.0E-3 (optimal model lr) with batch_size=8 does not surpass batch_size=16 (Node 56) for temporal prediction. final_loss=261 improved from Node 58's 334. confirms lr=2.0E-3, batch_size=16, noise=1.0 (Node 56) is optimal for low_rank+Dale_law regime
Next: parent=59 (highest UCB=2.658), final iteration of Block 4 - try coeff_W_L1=1.0E-6 (reduced L1) to test if lower regularization can improve temporal prediction in last iteration before block summary

## Iter 60: converged
Node: id=60, parent=59
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=2.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-6, batch_size=8, noise_model_level=1.0
Metrics: test_R2=0.8890, test_pearson=0.2845, connectivity_R2=0.9998, final_loss=2.4866e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-40.0, 32.9], effective rank 54 (90% var), 91 (99% var), spectral_radius=1.047
Mutation: coeff_W_L1: 1.0E-5 -> 1.0E-6
Parent rule: highest UCB node (Node 59), exploit strategy
Observation: coeff_W_L1=1.0E-6 (10x lower L1 regularization) achieves excellent connectivity_R2=0.9998 (same as Node 59) with best test_pearson=0.2845 in Block 4 (up from Node 59's 0.238 and Node 56's 0.279). reduced L1 regularization improves temporal prediction in low_rank+Dale_law regime. this is the best balanced result for this challenging regime

### Block 4 Summary (iters 49-60)
Simulation: low_rank connectivity (rank=50), Dale_law=True, Dale_law_factor=0.5, noise=variable
Best R2: 0.9998 at iters 51, 52, 56, 57, 59, 60 (connectivity), 0.285 pearson at iter 60 (best temporal with convergence)
Observation:
- low_rank (rank=50) + Dale_law=True is most challenging regime tested - requires noise regularization for convergence
- without noise (iter 49, 50, 55), connectivity_R2 ranges 0.04-0.17, dynamics collapse to rank 1-3
- noise=0.5-1.0 enables convergence (R2>0.999) but degrades temporal prediction (pearson=-0.13 to +0.28)
- best pearson achieved with noise=1.0, lr=2.0E-3, coeff_W_L1=1.0E-6 (iter 60: pearson=0.285)
- trade-off exists: no-noise achieves pearson=0.80 (iter 55) but fails connectivity (R2=0.14)
Optimum training parameters: lr_W=1.0E-2, lr=2.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-6, batch_size=8, noise_model_level=1.0

---

Next: Block 5 starts at iter 61, explore connectivity_type="chaotic" with Dale_law=True and Dale_law_factor=0.8 (80% excitatory) to test asymmetric exc/inh balance

## Iter 61: failed
--- NEW BLOCK ---
Simulation: connectivity_type=chaotic, Dale_law=True, Dale_law_factor=0.8, noise_model_level=0.0
Node: id=61, parent=root
Mode/Strategy: exploit
Config: lr_W=2.0E-3, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.3859, test_pearson=0.5840, connectivity_R2=0.0036, final_loss=7.1282e+02
Activity: dynamics collapsed to constant steady-state, range [10.3, 43.5], effective rank 1 (90% var), 1 (99% var), spectral_radius=4.472
Mutation: baseline (first iteration of new block), simulation changed to Dale_law_factor=0.8 (80% excitatory)
Parent rule: first iteration of block, parent=root
Observation: chaotic connectivity with Dale_law=True and 80% excitatory (factor=0.8) produces completely flat dynamics despite very high spectral_radius=4.472. connectivity_R2=0.004 indicates complete failure. this asymmetric exc/inh balance (80/20) may cause saturation/instability leading to flat steady-state, unlike 50/50 split in Block 1
Next: parent=61, try lr_W=1.0E-2 with lr=1.0E-3 and noise_model_level=0.5 (combination that worked for Dale_law in Block 1) to test if known-good parameters can recover connectivity

## Iter 62: failed
Node: id=62, parent=61
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.5
Metrics: test_R2=0.9999, test_pearson=0.7551, connectivity_R2=0.0023, final_loss=4.8603e+02
Activity: extremely flat dynamics despite noise, range [-54.6, -3.9], effective rank 1 (90% var), 4 (99% var), spectral_radius=4.472
Mutation: lr_W: 2.0E-3 -> 1.0E-2, lr: 1.0E-4 -> 1.0E-3, noise_model_level: 0.0 -> 0.5
Parent rule: highest UCB node (Node 61 was only option), exploit strategy
Observation: noise_model_level=0.5 with higher lr settings failed to recover connectivity - R2=0.002 similar to Node 61's 0.004. dynamics remain flat (rank 1-4) despite noise injection. spectral_radius=4.472 is very high indicating unstable eigenvalues - the 80% excitatory balance may cause network saturation that even noise cannot overcome. test_R2=0.9999 is trivially high because predicting constant values is easy
Next: parent=62 (highest UCB=0.709), try noise_model_level=1.0 with lr_W=1.5E-2 (higher) to test if more aggressive settings can break the saturation

## Iter 63: failed
Node: id=63, parent=62
Mode/Strategy: exploit
Config: lr_W=1.5E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=1.0
Metrics: test_R2=0.9998, test_pearson=0.505, connectivity_R2=0.004, final_loss=5.0186e+02
Activity: dynamics remain flat despite noise, range [-67.6, 5.3], effective rank 1 (90% var), 40 (99% var), spectral_radius=4.472
Mutation: lr_W: 1.0E-2 -> 1.5E-2, noise_model_level: 0.5 -> 1.0
Parent rule: highest UCB node (Node 62), exploit strategy
Observation: higher lr_W=1.5E-2 with noise=1.0 still fails - connectivity_R2=0.004 same as Node 61-62. spectral_radius=4.472 indicates severely unstable eigenvalues from 80% excitatory balance causing network saturation. effective rank at 99% var increased to 40 (from 4 in Node 62) due to noise injection, but dynamics are fundamentally flat (rank 1 at 90% var). 3 consecutive failures confirm this regime is fundamentally harder than 50% exc/inh (Block 1)
Next: parent=63 (highest UCB=0.870), try lr_W=5.0E-3 (lower) with lr=5.0E-4 (lower) to test if conservative lr settings can avoid overshooting in this unstable regime

## Iter 64: failed
Node: id=64, parent=63
Mode/Strategy: exploit
Config: lr_W=5.0E-3, lr=5.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=1.0
Metrics: test_R2=0.9997, test_pearson=0.505, connectivity_R2=0.001, final_loss=5.3154e+02
Activity: dynamics remain flat with noise injection, range [-67.6, 5.3], effective rank 1 (90% var), 40 (99% var), spectral_radius=4.472
Mutation: lr_W: 1.5E-2 -> 5.0E-3, lr: 1.0E-3 -> 5.0E-4
Parent rule: highest UCB node (Node 63), exploit strategy
Observation: conservative lr settings (lr_W=5.0E-3, lr=5.0E-4) still fail - connectivity_R2=0.001 similar to Nodes 61-63 (0.002-0.004). 4 consecutive failures confirm Dale_law_factor=0.8 (80% excitatory) creates fundamentally unstable dynamics with spectral_radius=4.472 causing network saturation. dynamics are flat (rank 1) despite noise injection. this regime may require different simulation parameters (e.g., lower Dale_law_factor) rather than training parameter tuning
Next: parent=64 (highest UCB=1.001), try batch_size=16 with lr=2.0E-3 (successful in Block 3/4) to test if larger batch and different lr can help in this saturated regime

## Iter 65: failed
Node: id=65, parent=64
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=2.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=16, noise_model_level=1.0
Metrics: test_R2=0.9995, test_pearson=0.505, connectivity_R2=0.0009, final_loss=5.4863e+02
Activity: dynamics remain flat with noise injection, range [-67.6, 5.3], effective rank 1 (90% var), 40 (99% var), spectral_radius=4.472
Mutation: batch_size: 8 -> 16, lr: 5.0E-4 -> 2.0E-3, lr_W: 5.0E-3 -> 1.0E-2
Parent rule: highest UCB node (Node 64), exploit strategy
Observation: larger batch_size=16 with higher lr settings (lr_W=1.0E-2, lr=2.0E-3) fails - connectivity_R2=0.0009 same as Nodes 61-64 (0.001-0.004). 5 consecutive failures confirm Dale_law_factor=0.8 with chaotic connectivity creates fundamentally pathological dynamics. spectral_radius=4.472 indicates eigenvalues far exceeding stability threshold. all training parameter combinations tested so far fail to recover any connectivity signal
Next: parent=65 (highest UCB=1.119), try coeff_W_L1=1.0E-4 (higher regularization) with lr_W=3.0E-3 (lower) to test if stronger regularization and conservative lr can constrain the unstable learning

## Iter 66: failed
Node: id=66, parent=65
Mode/Strategy: exploit
Config: lr_W=3.0E-3, lr=2.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-4, batch_size=16, noise_model_level=1.0
Metrics: test_R2=0.9999, test_pearson=0.5047, connectivity_R2=0.0001, final_loss=5.3277e+02
Activity: dynamics remain flat with noise injection, range [-67.6, 5.3], effective rank 1 (90% var), 40 (99% var), spectral_radius=4.472
Mutation: lr_W: 1.0E-2 -> 3.0E-3, coeff_W_L1: 1.0E-5 -> 1.0E-4
Parent rule: highest UCB node (Node 65), exploit strategy
Observation: stronger L1 regularization (coeff_W_L1=1.0E-4) with conservative lr_W=3.0E-3 fails - connectivity_R2=0.0001 worst result yet (Nodes 61-65 had 0.001-0.004). 6 consecutive failures confirm Dale_law_factor=0.8 with chaotic connectivity is fundamentally pathological. spectral_radius=4.472 indicates critically unstable eigenvalues, dynamics collapsed to rank 1 (flat) despite noise. no training parameter combination has recovered any meaningful connectivity signal. exploring different simulation regime may be necessary
Next: parent=66 (highest UCB=1.225), try Dale_law_factor=0.6 (closer to balanced) within same block to test if reducing excitatory dominance can stabilize dynamics

## Iter 67: converged
Node: id=67, parent=66
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.5
Simulation: Dale_law_factor=0.6 (changed from 0.8)
Metrics: test_R2=0.9998, test_pearson=0.4143, connectivity_R2=0.9994, final_loss=3.0248e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-30.1, 29.1], effective rank 14 (90% var), 71 (99% var), spectral_radius=1.320
Mutation: Dale_law_factor: 0.8 -> 0.6, lr_W: 3.0E-3 -> 1.0E-2, lr: 2.0E-3 -> 1.0E-3, coeff_W_L1: 1.0E-4 -> 1.0E-5, batch_size: 16 -> 8, noise_model_level: 1.0 -> 0.5
Parent rule: highest UCB node (Node 66), exploit strategy with simulation parameter change
Observation: breakthrough! Dale_law_factor=0.6 (60% excitatory) achieves near-perfect connectivity_R2=0.9994 after 6 consecutive failures with factor=0.8. spectral_radius=1.320 (vs 4.472 with 0.8) indicates stable dynamics with proper exc/inh balance. effective rank jumped from 1 to 71 (99% var), confirming rich dynamics restored. this confirms 80% excitatory was pathological - 60% allows successful training. test_pearson=0.414 is moderate but acceptable for Dale_law regime
Next: parent=67 (highest UCB=2.322), robustness-test with same config to verify Dale_law_factor=0.6 is reproducibly good before exploring further optimizations

## Iter 68: converged
Node: id=68, parent=67
Mode/Strategy: robustness-test
Config: lr_W=1.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.5
Simulation: Dale_law_factor=0.6 (unchanged from iter 67)
Metrics: test_R2=0.9998, test_pearson=0.4115, connectivity_R2=0.9992, final_loss=3.1133e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-30.1, 29.1], effective rank 14 (90% var), 71 (99% var), spectral_radius=1.320
Mutation: none (robustness-test)
Parent rule: highest UCB node (Node 67), robustness-test to verify Dale_law_factor=0.6 reproducibility
Observation: robustness-test passed - connectivity_R2=0.9992 confirms Node 67's R2=0.9994 is reproducible. Dale_law_factor=0.6 with chaotic connectivity is stable regime. test_pearson=0.411 vs Node 67's 0.414 shows consistent temporal prediction. 2 consecutive successes establish Dale_law_factor=0.6 as viable config for this block
Next: parent=68 (highest UCB=2.413), try noise_model_level=0.0 (no noise) to test if Dale_law_factor=0.6 can converge without noise regularization

## Iter 69: failed
Node: id=69, parent=68
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.0
Simulation: Dale_law_factor=0.6 (unchanged)
Metrics: test_R2=0.9979, test_pearson=0.9868, connectivity_R2=0.0007, final_loss=4.7898e+01
Activity: dynamics collapsed to flat steady-state, range [-1.7, 22.6], effective rank 1 (90% var), 1 (99% var), spectral_radius=1.320
Mutation: noise_model_level: 0.5 -> 0.0
Parent rule: highest UCB node (Node 68), exploit strategy testing no-noise with Dale_law_factor=0.6
Observation: removing noise regularization caused connectivity failure - R2 dropped from 0.9992 (Node 68) to 0.0007. same pattern as Block 4 iter 55 (low_rank+Dale_law without noise). dynamics collapsed from rank 71 to rank 1 (flat). test_pearson=0.987 excellent because predicting flat signal is trivial. confirms noise regularization is required for Dale_law regimes, even with factor=0.6
Next: parent=68 (highest UCB=1.999), try lr=2.0E-3 (higher model lr from Block 3/4 successes) with noise=0.5 to test if higher model lr improves temporal prediction while maintaining convergence

## Iter 70: converged
Node: id=70, parent=69
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=2.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.5
Simulation: Dale_law_factor=0.6 (unchanged)
Metrics: test_R2=0.9999, test_pearson=0.4138, connectivity_R2=0.9996, final_loss=3.1485e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-30.1, 29.1], effective rank 14 (90% var), 71 (99% var), spectral_radius=1.320
Mutation: lr: 1.0E-3 -> 2.0E-3 (from Node 68's config via Node 69 branch)
Parent rule: highest UCB node (Node 69 was parent, but config reverted to Node 68's noise=0.5 with higher lr from Node 69 attempt)
Observation: lr=2.0E-3 with noise=0.5 achieves connectivity_R2=0.9996, matching Node 67/68's performance (~0.999). confirms Dale_law_factor=0.6 regime is robust to model lr variation (1.0E-3 to 2.0E-3). test_pearson=0.414 consistent with Node 67/68, suggesting this is the characteristic pearson ceiling for this Dale_law regime. 3 converged results (67, 68, 70) with 1 failure (69 with noise=0) establish clear success pattern
Next: parent=70 (highest UCB=2.581), failure-probe with lr_W=2.0E-2 (2x current) to test lr_W upper boundary in Dale_law_factor=0.6 regime

## Iter 71: converged
Node: id=71, parent=70
Mode/Strategy: failure-probe
Config: lr_W=2.0E-2, lr=2.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.5
Simulation: Dale_law_factor=0.6 (unchanged)
Metrics: test_R2=0.9991, test_pearson=0.4103, connectivity_R2=0.9993, final_loss=3.7008e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-30.1, 29.1], effective rank 14 (90% var), 71 (99% var), spectral_radius=1.320
Mutation: lr_W: 1.0E-2 -> 2.0E-2 (2x increase, failure-probe)
Parent rule: highest UCB node (Node 70), failure-probe to test lr_W upper boundary
Observation: failure-probe passed! lr_W=2.0E-2 (2x) maintains excellent connectivity_R2=0.9993, similar to Node 70's 0.9996. test_pearson=0.410 unchanged from Node 67-70's ~0.41 ceiling. Dale_law_factor=0.6 regime is robust to lr_W variations from 1.0E-2 to 2.0E-2. final_loss=370 slightly higher than Node 70's 315 but still good. 4 consecutive converged results (67, 68, 70, 71) confirm this regime's stability
Next: parent=71 (highest UCB=2.658), try lr_W=3.0E-2 (1.5x increase) to probe higher lr_W boundary since 2.0E-2 succeeded

## Iter 72: converged
Node: id=72, parent=71
Mode/Strategy: failure-probe
Config: lr_W=3.0E-2, lr=2.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.5
Simulation: Dale_law_factor=0.6 (unchanged)
Metrics: test_R2=0.9995, test_pearson=0.4167, connectivity_R2=0.9990, final_loss=4.0879e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-30.1, 29.1], effective rank 14 (90% var), 71 (99% var), spectral_radius=1.320
Mutation: lr_W: 2.0E-2 -> 3.0E-2 (1.5x increase, failure-probe)
Parent rule: highest UCB node (Node 71), failure-probe to test lr_W upper boundary
Observation: failure-probe passed again! lr_W=3.0E-2 (3x baseline) maintains excellent connectivity_R2=0.9990, with best test_pearson=0.417 in block. Dale_law_factor=0.6 regime is remarkably robust to lr_W variations from 1.0E-2 to 3.0E-2. final_loss=409 slightly higher than Node 71's 370 but connectivity recovery unaffected. 5 consecutive converged results (67, 68, 70-72) confirm this regime's exceptional stability

### Block 5 Summary (iters 61-72)
Simulation: chaotic connectivity, Dale_law=True, Dale_law_factor variable (0.8->0.6), noise=variable
Best R2: 0.9996 at iter 70 (connectivity), 0.417 pearson at iter 72 (best temporal)
Observation:
- Dale_law_factor=0.8 (80% excitatory) is pathological: 6 consecutive failures (iters 61-66), spectral_radius=4.472 caused complete saturation
- Dale_law_factor=0.6 (60% excitatory) enables excellent convergence: 5 consecutive successes (iters 67-68, 70-72), spectral_radius=1.320
- noise regularization required: iter 69 (noise=0) failed with R2=0.0007 despite factor=0.6
- lr_W remarkably robust: tested 1.0E-2 to 3.0E-2 (3x range) with no degradation
Optimum training parameters: lr_W=1.0E-2 to 3.0E-2, lr=1.0E-3 to 2.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.5

---

Next: Block 6 starts at iter 73, explore connectivity_type="low_rank" with Dale_law=True and Dale_law_factor=0.6 (validated optimal) to test if low_rank benefits from same exc/inh balance that worked for chaotic

## Iter 73: failed
--- NEW BLOCK ---
Simulation: connectivity_type=low_rank, connectivity_rank=50, Dale_law=True, Dale_law_factor=0.6, noise_model_level=0.0
Node: id=73, parent=root
Mode/Strategy: exploit
Config: lr_W=2.0E-3, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9753, test_pearson=0.9052, connectivity_R2=0.0006, final_loss=2.9203e+01
Activity: dynamics collapsed to flat steady-state across 100 neurons, range [-1.7, 18.7], effective rank 1 (90% var), 1 (99% var), spectral_radius=0.985
Mutation: baseline (first iteration of block)
Parent rule: first iteration of block, parent=root
Observation: low_rank+Dale_law without noise fails completely, same pattern as Block 4 iter 55 and Block 5 iter 69. despite using Dale_law_factor=0.6 (validated optimal in Block 5), dynamics collapsed to rank-1. test_pearson=0.905 misleadingly high because predicting flat signal is trivial. confirms noise regularization is mandatory for Dale_law regimes regardless of connectivity_type
Next: parent=73, add noise_model_level=0.5 to test if noise regularization enables convergence for low_rank+Dale_law regime

## Iter 74: converged
Node: id=74, parent=73
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.5
Simulation: connectivity_type=low_rank, connectivity_rank=50, Dale_law=True, Dale_law_factor=0.6
Metrics: test_R2=0.9919, test_pearson=0.5131, connectivity_R2=0.9243, final_loss=4.1331e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-29.1, 32.5], effective rank 2 (90% var), 61 (99% var), spectral_radius=0.985
Mutation: noise_model_level: 0.0 -> 0.5, lr_W: 2.0E-3 -> 1.0E-2, lr: 1.0E-4 -> 1.0E-3
Parent rule: highest UCB node (Node 73), exploit strategy with noise=0.5 fix validated from Block 4/5
Observation: noise regularization enabled convergence! connectivity_R2=0.9243 after iter 73's failure (R2=0.0006). low_rank+Dale_law+factor=0.6 now converges. test_pearson=0.513 higher than chaotic+Dale_law regime's ~0.41 (Block 5), opposite of what was expected. effective rank 2 at 90% var but 61 at 99% var shows low-dimensional dominant dynamics with high-dimensional noise
Next: parent=74 (highest UCB=1.631), try lr_W=2.0E-2 to test if higher lr_W improves connectivity_R2 towards 0.99+ like Block 5

## Iter 75: converged
Node: id=75, parent=74
Mode/Strategy: exploit
Config: lr_W=2.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.5
Simulation: connectivity_type=low_rank, connectivity_rank=50, Dale_law=True, Dale_law_factor=0.6
Metrics: test_R2=0.9966, test_pearson=0.5075, connectivity_R2=0.9593, final_loss=4.1755e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-29.1, 32.5], effective rank 2 (90% var), 61 (99% var), spectral_radius=0.985
Mutation: lr_W: 1.0E-2 -> 2.0E-2
Parent rule: highest UCB node (Node 74), exploit strategy testing higher lr_W
Observation: lr_W=2.0E-2 improved connectivity_R2 from 0.9243 (iter 74) to 0.9593, confirming higher lr_W benefits low_rank+Dale_law regime. test_pearson=0.508 slightly below iter 74's 0.513 but within noise. 2 consecutive converged results establish low_rank+Dale_law+factor=0.6 as viable regime. activity pattern consistent with low-rank structure (rank-2 at 90% var, rank-61 at 99% var)
Next: parent=75 (highest UCB=1.825), try lr_W=3.0E-2 (1.5x increase) to test if higher lr_W can push connectivity_R2 above 0.97 like Block 5's chaotic regime

## Iter 76: converged
Node: id=76, parent=75
Mode/Strategy: exploit
Config: lr_W=3.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.5
Simulation: connectivity_type=low_rank, connectivity_rank=50, Dale_law=True, Dale_law_factor=0.6
Metrics: test_R2=0.9974, test_pearson=0.5041, connectivity_R2=0.9673, final_loss=4.5477e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-29.1, 32.5], effective rank 2 (90% var), 61 (99% var), spectral_radius=0.985
Mutation: lr_W: 2.0E-2 -> 3.0E-2
Parent rule: highest UCB node (Node 76 is current, selected parent=75 with highest UCB=1.626 after Node 76), exploit strategy testing higher lr_W
Observation: lr_W=3.0E-2 further improved connectivity_R2 from 0.9593 (iter 75) to 0.9673. 3 consecutive converged results (iters 74-76) with increasing lr_W (1.0E-2 -> 2.0E-2 -> 3.0E-2) show monotonic improvement (0.924 -> 0.959 -> 0.967). test_pearson stable at ~0.50. final_loss=455 slightly higher than iter 75's 418 but connectivity recovery continues improving. low_rank+Dale_law regime approaching Block 5's chaotic+Dale_law performance
Next: parent=76 (highest UCB=1.967), try lr=2.0E-3 (higher model lr from Block 5 success) to test if higher model lr improves connectivity_R2 further toward 0.99+

## Iter 77: converged
Node: id=77, parent=76
Mode/Strategy: exploit
Config: lr_W=3.0E-2, lr=2.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.5
Simulation: connectivity_type=low_rank, connectivity_rank=50, Dale_law=True, Dale_law_factor=0.6
Metrics: test_R2=0.9970, test_pearson=0.5039, connectivity_R2=0.9736, final_loss=4.4817e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-29.1, 32.5], effective rank 2 (90% var), 61 (99% var), spectral_radius=0.985
Mutation: lr: 1.0E-3 -> 2.0E-3
Parent rule: highest UCB node (Node 76), exploit strategy testing higher model lr
Observation: lr=2.0E-3 improved connectivity_R2 from 0.9673 (iter 76) to 0.9736. 4 consecutive converged results (iters 74-77) with connectivity_R2 improving monotonically (0.924 -> 0.959 -> 0.967 -> 0.974). test_pearson stable at ~0.50. low_rank+Dale_law regime approaching 0.98 connectivity recovery. combination of lr_W=3.0E-2 and lr=2.0E-3 appears optimal so far
Next: parent=77 (highest UCB=2.092), failure-probe with lr_W=5.0E-2 (1.67x increase) to test lr_W upper boundary since 4 consecutive successes warrant probing failure boundary


## Iter 78: converged
Node: id=78, parent=77
Mode/Strategy: failure-probe
Config: lr_W=5.0E-2, lr=2.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.5
Simulation: connectivity_type=low_rank, connectivity_rank=50, Dale_law=True, Dale_law_factor=0.6
Metrics: test_R2=0.9938, test_pearson=0.5064, connectivity_R2=0.9740, final_loss=4.8740e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-29.1, 32.5], effective rank 2 (90% var), 61 (99% var), spectral_radius=0.985
Mutation: lr_W: 3.0E-2 -> 5.0E-2 (1.67x increase, failure-probe)
Parent rule: highest UCB node (Node 77), failure-probe to test lr_W upper boundary after 4 consecutive successes
Observation: failure-probe passed! lr_W=5.0E-2 maintains connectivity_R2=0.974 (same as iter 77's 0.974), showing plateau rather than improvement or degradation. test_pearson=0.506 unchanged. final_loss=487 slightly higher than iter 77's 448 but still good. 5 consecutive converged results (iters 74-78) confirm low_rank+Dale_law regime is extremely robust to lr_W variations from 1.0E-2 to 5.0E-2 (5x range). connectivity_R2 plateaued at ~0.97, suggesting lr_W is not the limiting factor - may need to explore other parameters
Next: parent=78 (highest UCB=2.199), try batch_size=16 to test if larger batches can push connectivity_R2 above 0.97 plateau

## Iter 79: converged
Node: id=79, parent=78
Mode/Strategy: exploit
Config: lr_W=5.0E-2, lr=2.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=16, noise_model_level=0.5
Simulation: connectivity_type=low_rank, connectivity_rank=50, Dale_law=True, Dale_law_factor=0.6
Metrics: test_R2=0.9927, test_pearson=0.5059, connectivity_R2=0.9464, final_loss=3.9305e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-29.1, 32.5], effective rank 2 (90% var), 61 (99% var), spectral_radius=0.985
Mutation: batch_size: 8 -> 16
Parent rule: highest UCB node (Node 78), exploit strategy testing larger batch size to break 0.97 plateau
Observation: batch_size=16 degraded connectivity_R2 from 0.974 (iter 78) to 0.946. final_loss improved to 393 (from 487) but connectivity recovery suffered. 6 consecutive converged results (iters 74-79) but batch_size=16 is suboptimal. this confirms batch_size=8 is better for connectivity recovery in low_rank+Dale_law regime. test_pearson=0.506 unchanged. lr_W plateau at 0.97 not broken by batch size change
Next: parent=79 (highest UCB=2.269), revert batch_size=8 and try lr=3.0E-3 (1.5x current lr) to test if higher model lr can break 0.97 plateau

## Iter 80: converged
Node: id=80, parent=79
Mode/Strategy: exploit
Config: lr_W=5.0E-2, lr=3.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.5
Simulation: connectivity_type=low_rank, connectivity_rank=50, Dale_law=True, Dale_law_factor=0.6
Metrics: test_R2=0.9944, test_pearson=0.5040, connectivity_R2=0.9732, final_loss=5.0700e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-29.1, 32.5], effective rank 2 (90% var), 61 (99% var), spectral_radius=0.985
Mutation: lr: 2.0E-3 -> 3.0E-3, batch_size: 16 -> 8
Parent rule: highest UCB node (Node 79), exploit strategy reverting batch_size=8 and testing higher model lr to break 0.97 plateau
Observation: batch_size=8 reverted, lr=3.0E-3 tested. connectivity_R2=0.9732 remains at ~0.97 plateau (vs 0.946 at iter 79 with batch_size=16). final_loss=507 higher than iter 79's 393 but connectivity recovery improved. 7 consecutive converged results (iters 74-80). higher model lr (3.0E-3) did not break 0.97 plateau. test_pearson=0.504 stable. low_rank+Dale_law regime has robust 0.97 ceiling - may be inherent limitation of low_rank structure rather than hyperparameter tuning
Next: parent=80 (highest UCB=2.387), try coeff_W_L1=1.0E-6 (10x lower L1 regularization) to test if less regularization on W can push connectivity_R2 above 0.97 plateau

## Iter 81: converged
Node: id=81, parent=80
Mode/Strategy: exploit
Config: lr_W=5.0E-2, lr=3.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-6, batch_size=8, noise_model_level=0.5
Simulation: connectivity_type=low_rank, connectivity_rank=50, Dale_law=True, Dale_law_factor=0.6
Metrics: test_R2=0.9982, test_pearson=0.4993, connectivity_R2=0.9737, final_loss=4.9845e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-29.1, 32.5], effective rank 2 (90% var), 61 (99% var), spectral_radius=0.985
Mutation: coeff_W_L1: 1.0E-5 -> 1.0E-6
Parent rule: highest UCB node (Node 80), exploit strategy testing lower L1 regularization to break 0.97 plateau
Observation: coeff_W_L1=1.0E-6 did not break 0.97 plateau - connectivity_R2=0.9737 essentially same as iter 80's 0.9732. 8 consecutive converged results (iters 74-81) with connectivity_R2 stuck at ~0.97 despite exploring lr_W (1E-2 to 5E-2), lr (1E-3 to 3E-3), batch_size (8, 16), coeff_W_L1 (1E-5, 1E-6). test_pearson=0.499 stable at ~0.50. 0.97 plateau appears to be inherent structural limitation of low_rank+Dale_law regime
Next: parent=81 (highest UCB=2.474), try noise_model_level=1.0 (higher noise) to test if increased noise can push connectivity_R2 above 0.97 plateau since all other training parameters have been exhausted

## Iter 82: converged
Node: id=82, parent=81
Mode/Strategy: exploit
Config: lr_W=5.0E-2, lr=3.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-6, batch_size=8, noise_model_level=1.0
Simulation: connectivity_type=low_rank, connectivity_rank=50, Dale_law=True, Dale_law_factor=0.6
Metrics: test_R2=0.9992, test_pearson=0.0594, connectivity_R2=0.9990, final_loss=5.3398e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-42.7, 39.3], effective rank 43 (90% var), 87 (99% var), spectral_radius=0.985
Mutation: noise_model_level: 0.5 -> 1.0
Parent rule: highest UCB node (Node 81), exploit strategy testing higher noise to break 0.97 plateau
Observation: noise_model_level=1.0 achieved breakthrough - connectivity_R2=0.9990 breaks 0.97 plateau for first time in block 6! test_R2=0.9992 excellent. however test_pearson collapsed from 0.50 to 0.06. activity range increased significantly [-42.7, 39.3] vs previous [-29.1, 32.5]. effective rank jumped from 61 to 87 (99% var) indicating noise increases signal complexity. higher noise acts as strong regularizer enabling better W recovery but may interfere with temporal correlation
Next: parent=82 (highest UCB), try noise_model_level=0.75 to find balance between connectivity_R2>0.99 and test_pearson>0.3

## Iter 83: converged
Node: id=83, parent=82
Mode/Strategy: exploit
Config: lr_W=5.0E-2, lr=3.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-6, batch_size=8, noise_model_level=0.75
Simulation: connectivity_type=low_rank, connectivity_rank=50, Dale_law=True, Dale_law_factor=0.6
Metrics: test_R2=0.9983, test_pearson=0.0430, connectivity_R2=0.9992, final_loss=5.0225e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-40.8, 34.0], effective rank 26 (90% var), 79 (99% var), spectral_radius=0.985
Mutation: noise_model_level: 1.0 -> 0.75
Parent rule: highest UCB node (Node 82), exploit strategy testing intermediate noise to balance connectivity and temporal prediction
Observation: noise_model_level=0.75 maintains excellent connectivity_R2=0.9992 (same as iter 82's 0.999), but test_pearson=0.043 remains very low (slightly worse than iter 82's 0.059). effective rank reduced from 87 to 79 (99% var) with lower noise. confirms the noise-connectivity vs pearson trade-off: noise>0.5 enables connectivity_R2>0.99 breakthrough but sacrifices temporal correlation. best balanced result remains iter 74-81's ~0.97 connectivity with ~0.50 pearson. this is iteration 11/12 in block 6
Next: parent=83 (highest UCB=2.658), final iteration of block 6 - try lr_W=3.0E-2 with noise_model_level=0.75 to test if lower lr_W can recover test_pearson while maintaining connectivity

## Iter 84: converged
Node: id=84, parent=83
Mode/Strategy: exploit
Config: lr_W=3.0E-2, lr=3.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-6, batch_size=8, noise_model_level=0.75
Simulation: connectivity_type=low_rank, connectivity_rank=50, Dale_law=True, Dale_law_factor=0.6
Metrics: test_R2=0.9979, test_pearson=0.1152, connectivity_R2=0.9994, final_loss=4.4900e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-40.8, 34.0], effective rank 26 (90% var), 79 (99% var), spectral_radius=0.985
Mutation: lr_W: 5.0E-2 -> 3.0E-2
Parent rule: highest UCB node (Node 83), exploit strategy testing lower lr_W to improve pearson while maintaining connectivity
Observation: lr_W=3.0E-2 with noise=0.75 maintains excellent connectivity_R2=0.9994 (best in block 6!) while improving test_pearson from 0.043 (iter 83) to 0.115 (2.7x improvement). however pearson still far below noise=0.5 regime's ~0.50. this confirms fundamental trade-off: noise>0.5 enables connectivity_R2>0.99 but sacrifices pearson. block 6 established two operating regimes for low_rank+Dale_law: (1) noise=0.5 gives R2~0.97, pearson~0.50; (2) noise=0.75-1.0 gives R2>0.99 but pearson<0.12

### Block 6 Summary (iters 73-84)
Simulation: low_rank (rank=50), Dale_law=True, Dale_law_factor=0.6, noise=variable
Best R2: 0.9994 at iter 84 (connectivity), 0.513 pearson at iter 74 (best temporal)
Observation:
- low_rank+Dale_law without noise fails completely (iter 73: R2=0.0006), same as chaotic+Dale_law without noise in Block 4/5
- noise=0.5 enables convergence with ~0.97 connectivity_R2 plateau and ~0.50 pearson (iters 74-81)
- noise=0.75-1.0 breaks 0.97 plateau achieving R2>0.99 but collapses pearson to <0.12 (iters 82-84)
- fundamental trade-off discovered: higher noise improves connectivity recovery but degrades temporal prediction
- lr_W robust across 1E-2 to 5E-2 (5x range), batch_size=8 optimal, coeff_W_L1=1E-6 slightly better than 1E-5
Optimum training parameters:
- For best connectivity (R2>0.99): lr_W=3.0E-2, lr=3.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-6, batch_size=8, noise_model_level=0.75
- For best temporal (pearson~0.50): lr_W=1.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.5

---

Next: Block 7 starts at iter 85, explore connectivity_type="low_rank" with lower connectivity_rank=20 (vs 50) to test if lower rank structure is easier or harder to learn

## Iter 85: failed
--- NEW BLOCK ---
Simulation: connectivity_type=low_rank, connectivity_rank=20, Dale_law=True, Dale_law_factor=0.6, noise_model_level=0.0
Node: id=85, parent=root
Mode/Strategy: exploit
Config: lr_W=2.0E-3, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9918, test_pearson=0.9861, connectivity_R2=0.0006, final_loss=4.8046e+01
Activity: dynamics collapsed to flat steady-state across 100 neurons, range [-1.3, 20.9], effective rank 1 (90% var), 1 (99% var), spectral_radius=1.008
Mutation: baseline (first iteration of block), connectivity_rank reduced from 50 to 20
Parent rule: first iteration of block, parent=root
Observation: low_rank+Dale_law without noise fails again, same pattern as Block 6 iter 73. connectivity_R2=0.0006 despite spectral_radius=1.008 being reasonable. test_pearson=0.986 misleadingly high because predicting flat signal is trivial. confirms noise regularization is mandatory for Dale_law regimes regardless of connectivity_rank. Block 6 established noise=0.5 minimum required
Next: parent=85, add noise_model_level=0.5 and apply Block 6 optimal training parameters (lr_W=1.0E-2, lr=1.0E-3) to test if low_rank with rank=20 converges with noise

## Iter 86: converged
Node: id=86, parent=85
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.5
Simulation: connectivity_type=low_rank, connectivity_rank=20, Dale_law=True, Dale_law_factor=0.6
Metrics: test_R2=0.9986, test_pearson=-0.5399, connectivity_R2=0.9020, final_loss=4.0693e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-32.1, 30.4], effective rank 8 (90% var), 67 (99% var), spectral_radius=1.008
Mutation: noise_model_level: 0.0 -> 0.5, lr_W: 2.0E-3 -> 1.0E-2, lr: 1.0E-4 -> 1.0E-3
Parent rule: highest UCB node (Node 85 was only option), exploit strategy with noise=0.5 fix validated from Block 6
Observation: noise regularization enabled convergence for rank=20! connectivity_R2=0.902 after iter 85's failure (R2=0.0006). rank=20 achieves lower R2 than rank=50's ~0.97 (Block 6), suggesting lower rank is harder to learn. unusual negative test_pearson=-0.54 indicates anti-correlated temporal prediction - may need different lr settings. effective rank 8 (90%) vs rank=50's 2 shows rank=20 has more distributed variance. dynamics healthy with spectral_radius=1.008
Next: parent=86 (highest UCB=1.609), try lr_W=2.0E-2 to test if higher lr_W can improve connectivity_R2 toward 0.97+ like Block 6

## Iter 87: converged
Node: id=87, parent=86
Mode/Strategy: exploit
Config: lr_W=2.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.5
Simulation: connectivity_type=low_rank, connectivity_rank=20, Dale_law=True, Dale_law_factor=0.6
Metrics: test_R2=0.9991, test_pearson=-0.5362, connectivity_R2=0.9384, final_loss=4.3080e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-32.1, 30.4], effective rank 8 (90% var), 67 (99% var), spectral_radius=1.008
Mutation: lr_W: 1.0E-2 -> 2.0E-2
Parent rule: highest UCB node (Node 86), exploit strategy testing higher lr_W to improve connectivity_R2
Observation: lr_W=2.0E-2 improved connectivity_R2 from 0.902 (iter 86) to 0.938 (+4%). rank=20 regime shows consistent pattern with Block 6 rank=50: higher lr_W improves connectivity recovery. negative test_pearson=-0.54 persists (same as iter 86), possibly inherent to low_rank structure or noise interaction. 2 consecutive converged results (iters 86-87) in Block 7. rank=20 converging slower than rank=50 (~0.94 vs ~0.97 plateau)
Next: parent=87 (highest UCB=1.804), try lr_W=3.0E-2 to test if higher lr_W can push connectivity_R2 toward 0.97+ like Block 6 rank=50

## Iter 88: converged
Node: id=88, parent=87
Mode/Strategy: exploit
Config: lr_W=3.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.5
Simulation: connectivity_type=low_rank, connectivity_rank=20, Dale_law=True, Dale_law_factor=0.6
Metrics: test_R2=0.9995, test_pearson=-0.5347, connectivity_R2=0.9424, final_loss=4.5861e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-32.1, 30.4], effective rank 8 (90% var), 67 (99% var), spectral_radius=1.008
Mutation: lr_W: 2.0E-2 -> 3.0E-2
Parent rule: highest UCB node (Node 87), exploit strategy continuing lr_W exploration
Observation: lr_W=3.0E-2 improved connectivity_R2 from 0.938 (iter 87) to 0.942 (+0.4%), but improvement marginal. rank=20 regime plateauing around 0.94, slower convergence than rank=50 (~0.97). 3 consecutive converged results (iters 86-88) triggers failure-probe strategy. negative test_pearson=-0.53 persists.
Next: parent=88 (highest UCB=1.942), failure-probe with lr_W=6.0E-2 (2x current) to find upper stability boundary

## Iter 89: converged
Node: id=89, parent=88
Mode/Strategy: failure-probe
Config: lr_W=6.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.5
Simulation: connectivity_type=low_rank, connectivity_rank=20, Dale_law=True, Dale_law_factor=0.6
Metrics: test_R2=0.9986, test_pearson=-0.5325, connectivity_R2=0.9315, final_loss=5.4425e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-32.1, 30.4], effective rank 8 (90% var), 67 (99% var), spectral_radius=1.008
Mutation: lr_W: 3.0E-2 -> 6.0E-2 (2x increase, failure-probe)
Parent rule: highest UCB node (Node 88), failure-probe after 3 consecutive successes (iters 86-88) to find lr_W upper boundary
Observation: failure-probe with lr_W=6.0E-2 degraded connectivity_R2 from 0.942 (iter 88) to 0.932 (-1%), confirming lr_W=3.0E-2 is near optimal for rank=20. final_loss=544 higher than iter 88's 459 indicating training instability at extreme lr_W. negative test_pearson=-0.53 persists across all rank=20 iterations (86-89), appears inherent to this low_rank structure. 4 consecutive converged results (iters 86-89) but R2 plateaued at ~0.93-0.94, below rank=50's ~0.97 plateau
Next: parent=89 (highest UCB=2.050), try lr=2.0E-3 (2x current) to test if higher model lr can break 0.94 plateau

## Iter 90: converged
Node: id=90, parent=89
Mode/Strategy: exploit
Config: lr_W=3.0E-2, lr=2.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.5
Simulation: connectivity_type=low_rank, connectivity_rank=20, Dale_law=True, Dale_law_factor=0.6
Metrics: test_R2=0.9992, test_pearson=-0.5391, connectivity_R2=0.9584, final_loss=4.3907e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-32.1, 30.4], effective rank 8 (90% var), 67 (99% var), spectral_radius=1.008
Mutation: lr: 1.0E-3 -> 2.0E-3
Parent rule: highest UCB node (Node 89), exploit strategy testing higher model lr to break 0.94 plateau
Observation: lr=2.0E-3 broke the 0.94 plateau! connectivity_R2=0.958 (+2.6% from iter 89's 0.932). this matches Block 6 pattern where higher lr improved connectivity. negative test_pearson=-0.54 persists across all rank=20 iterations. rank=20 now approaching rank=50's ~0.97 plateau but not quite there yet. 5 consecutive converged results (iters 86-90)
Next: parent=90 (highest UCB=2.183), try lr=3.0E-3 to test if higher model lr can push connectivity_R2 toward 0.97+

## Iter 91: converged
Node: id=91, parent=90
Mode/Strategy: exploit
Config: lr_W=3.0E-2, lr=3.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.5
Simulation: connectivity_type=low_rank, connectivity_rank=20, Dale_law=True, Dale_law_factor=0.6
Metrics: test_R2=0.9993, test_pearson=-0.5400, connectivity_R2=0.9608, final_loss=4.5416e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-32.1, 30.4], effective rank 8 (90% var), 67 (99% var), spectral_radius=1.008
Mutation: lr: 2.0E-3 -> 3.0E-3
Parent rule: highest UCB node (Node 90), exploit strategy testing higher model lr to push toward 0.97+
Observation: lr=3.0E-3 marginally improved connectivity_R2 from 0.958 (iter 90) to 0.961 (+0.3%), continuing slow improvement trend. rank=20 regime plateauing at ~0.96, still below rank=50's ~0.97. 6 consecutive converged results (iters 86-91). negative test_pearson=-0.54 persists across all rank=20 iterations. higher lr settings (lr_W=3E-2, lr=3E-3) haven't broken through the 0.96 ceiling. may need to explore noise=0.75 like Block 6 to break plateau, though that sacrifices pearson
Next: parent=91 (highest UCB=2.284), try noise_model_level=0.75 to test if higher noise can break 0.96 plateau like Block 6's noise experiment

## Iter 92: converged
Node: id=92, parent=91
Mode/Strategy: exploit
Config: lr_W=3.0E-2, lr=3.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.75
Simulation: connectivity_type=low_rank, connectivity_rank=20, Dale_law=True, Dale_law_factor=0.6
Metrics: test_R2=0.9992, test_pearson=0.2008, connectivity_R2=0.9990, final_loss=4.6173e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-37.5, 36.8], effective rank 31 (90% var), 82 (99% var), spectral_radius=1.008
Mutation: noise_model_level: 0.5 -> 0.75
Parent rule: highest UCB node (Node 91), exploit strategy testing noise=0.75 to break 0.96 plateau
Observation: noise=0.75 dramatically improved connectivity_R2 from 0.961 (iter 91) to 0.999 (+3.8%)! this matches Block 6 pattern where noise=0.75 achieved near-perfect connectivity. test_pearson jumped from -0.54 to +0.20, breaking the negative correlation pattern. activity range expanded [-37.5, 36.8] vs [-32.1, 30.4] and effective rank increased 31 vs 8 (90% var). 7 consecutive converged results (iters 86-92). rank=20 with noise=0.75 now matches rank=50's best performance (~0.999). Block 7 optimal: lr_W=3E-2, lr=3E-3, noise=0.75 for low_rank with Dale_law
Next: parent=92 (highest UCB=2.413), failure-probe with lr_W=1.0E-1 (3x current) to find upper stability boundary after 7 consecutive successes

## Iter 93: converged
Node: id=93, parent=92
Mode/Strategy: failure-probe
Config: lr_W=1.0E-1, lr=3.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.75
Simulation: connectivity_type=low_rank, connectivity_rank=20, Dale_law=True, Dale_law_factor=0.6
Metrics: test_R2=0.9990, test_pearson=0.2035, connectivity_R2=0.9990, final_loss=6.0195e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-37.5, 36.8], effective rank 31 (90% var), 82 (99% var), spectral_radius=1.008
Mutation: lr_W: 3.0E-2 -> 1.0E-1 (3.3x increase, failure-probe)
Parent rule: highest UCB node (Node 92), failure-probe after 8 consecutive successes (iters 86-93) to find lr_W upper boundary
Observation: extreme lr_W=1.0E-1 maintains excellent connectivity_R2=0.999 (same as iter 92), demonstrating remarkable robustness. final_loss=602 higher than iter 92's 462 (+30%) indicating training less efficient but still convergent. test_pearson=0.204 vs 0.201 unchanged. rank=20 with noise=0.75 can tolerate 10x lr_W variation (1E-2 to 1E-1) without connectivity degradation. 8 consecutive converged results (iters 86-93). lr_W upper boundary not yet found - regime extremely stable
Next: parent=93 (highest UCB=2.499), try lr=5.0E-3 (1.67x current) to test if higher model lr improves temporal prediction while maintaining connectivity at R2>0.99
Next: parent=93 (highest UCB=2.499), try lr=5.0E-3 (1.67x current) to test if higher model lr improves temporal prediction while maintaining connectivity at R2>0.99

## Iter 94: converged
Node: id=94, parent=93
Mode/Strategy: exploit
Config: lr_W=1.0E-1, lr=5.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.75
Simulation: connectivity_type=low_rank, connectivity_rank=20, Dale_law=True, Dale_law_factor=0.6
Metrics: test_R2=0.9990, test_pearson=0.2032, connectivity_R2=0.9982, final_loss=7.7662e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-37.5, 36.8], effective rank 31 (90% var), 82 (99% var), spectral_radius=1.008
Mutation: lr: 3.0E-3 -> 5.0E-3
Parent rule: highest UCB node (Node 93), exploit strategy testing higher model lr to improve temporal prediction
Observation: lr=5.0E-3 maintained excellent connectivity_R2=0.998 (same as iter 93's 0.999), but test_pearson=0.203 unchanged and final_loss=777 increased (+29% from iter 93's 602). higher model lr did not improve temporal prediction in noise=0.75 regime. 9 consecutive converged results (iters 86-94). rank=20 with noise=0.75 shows consistent ~0.20 pearson ceiling regardless of lr settings. this is iteration 10/12 in Block 7
Next: parent=94 (highest UCB=2.579), try coeff_W_L1=1.0E-6 to test if lower L1 regularization can improve temporal prediction while maintaining connectivity

## Iter 95: converged
Node: id=95, parent=94
Mode/Strategy: exploit
Config: lr_W=1.0E-1, lr=5.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-6, batch_size=8, noise_model_level=0.75
Simulation: connectivity_type=low_rank, connectivity_rank=20, Dale_law=True, Dale_law_factor=0.6
Metrics: test_R2=0.9988, test_pearson=0.2080, connectivity_R2=0.9985, final_loss=7.6121e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-37.5, 36.8], effective rank 31 (90% var), 82 (99% var), spectral_radius=1.008
Mutation: coeff_W_L1: 1.0E-5 -> 1.0E-6
Parent rule: highest UCB node (Node 94), exploit strategy testing lower L1 regularization for improved temporal prediction
Observation: coeff_W_L1=1.0E-6 maintained excellent connectivity_R2=0.999, test_pearson=0.208 virtually unchanged from iter 94 (0.203). final_loss=762 similar to iter 94 777 (-2%). lower L1 regularization had minimal impact on temporal prediction in noise=0.75 regime. 10 consecutive converged results (iters 86-95). rank=20 with noise=0.75 shows consistent ~0.20 pearson ceiling regardless of L1 settings. this is iteration 11/12 in Block 7. pearson ceiling may be inherent to low_rank=20 structure rather than training parameters
Next: parent=95 (highest UCB=2.657), final iteration of Block 7 - robustness-test by re-running best config (iter 92: lr_W=3.0E-2, lr=3.0E-3, coeff_W_L1=1.0E-5) to verify reproducibility

## Iter 96: converged
Node: id=96, parent=95
Mode/Strategy: robustness-test
Config: lr_W=3.0E-2, lr=3.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.75
Simulation: connectivity_type=low_rank, connectivity_rank=20, Dale_law=True, Dale_law_factor=0.6
Metrics: test_R2=0.9994, test_pearson=0.2072, connectivity_R2=0.9992, final_loss=4.5968e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-37.5, 36.8], effective rank 31 (90% var), 82 (99% var), spectral_radius=1.008
Mutation: robustness-test - reverted to iter 92's config (lr_W=3.0E-2, lr=3.0E-3, coeff_W_L1=1.0E-5)
Parent rule: highest UCB node (Node 95), robustness-test to verify best config reproducibility as final Block 7 iteration
Observation: robustness-test confirms iter 92's optimal config reproduces well: connectivity_R2=0.9992 (vs 0.9990 at iter 92), test_pearson=0.207 (vs 0.201). 11 consecutive converged results (iters 86-96). rank=20 with noise=0.75 regime is highly reproducible. Block 7 completed with 11/12 converged iterations (iter 85 failed without noise).

### Block 7 Summary (iters 85-96)
Simulation: low_rank (rank=20), Dale_law=True, Dale_law_factor=0.6, noise=variable
Best R2: 0.9992 at iter 96 (connectivity), 0.207 pearson at iter 95-96 (best temporal)
Observation:
- low_rank+Dale_law without noise fails completely (iter 85: R2=0.0006), consistent with Block 6 pattern
- noise=0.5 enables convergence at ~0.94-0.96 connectivity_R2 plateau with negative test_pearson (-0.54) (iters 86-91)
- noise=0.75 breaks plateau achieving R2>0.999 and switches pearson from negative (-0.54) to positive (+0.20) (iters 92-96)
- rank=20 converges slower than rank=50 (Block 6) at noise=0.5, but matches rank=50 performance at noise=0.75
- lr_W extremely robust: tolerates 10x variation (1E-2 to 1E-1) without connectivity degradation at noise=0.75
Optimum training parameters:
- For best connectivity (R2>0.99): lr_W=3.0E-2, lr=3.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.75
- Temporal prediction ceiling at ~0.20 pearson regardless of lr/L1 settings, may be inherent to low_rank structure

---

## Iter 97: partial
--- NEW BLOCK ---
Simulation: connectivity_type=chaotic, Dale_law=True, Dale_law_factor=0.3 (30% excitatory, 70% inhibitory)
Node: id=97, parent=root
Mode/Strategy: exploit
Config: lr_W=2.0E-3, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.0
Metrics: test_R2=0.8352, test_pearson=0.7412, connectivity_R2=0.5063, final_loss=7.3241e+02
Activity: oscillatory dynamics across 100 neurons, range [-12.8, 13.3], effective rank 10 (90% var), 30 (99% var), spectral_radius=2.600
Mutation: baseline (first iteration of new block with chaotic + Dale_law + extreme E/I imbalance)
Parent rule: first iteration of Block 8, parent=root
Observation: chaotic+Dale_law with 30% excitatory (70% inhibitory) produces partial convergence at R2=0.506. spectral_radius=2.6 is significantly higher than balanced Dale_law (factor=0.6 had 1.32) but lower than extreme excitatory (factor=0.8 had 4.47). activity range [-12.8, 13.3] is narrower than chaotic without Dale_law (Block 1: [-26, 24]) and effective rank=30 is reasonable. test_R2=0.84 and pearson=0.74 show good temporal prediction despite lower connectivity recovery. this inhibitory-dominant regime achieves partial convergence without noise (vs noise=0.5 required for factor=0.6), suggesting inhibitory dominance may be more stable than excitatory dominance
Next: parent=97, increase lr_W to 7.0E-3 based on Block 1 finding that lr_W=7E-3 achieved R2>0.99 for chaotic connectivity

## Iter 98: converged
Node: id=98, parent=97
Mode/Strategy: exploit
Config: lr_W=7.0E-3, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.0
Metrics: test_R2=0.9867, test_pearson=0.9484, connectivity_R2=0.9312, final_loss=3.3261e+02
Activity: oscillatory dynamics across 100 neurons, range [-13.1, 13.3], effective rank 10 (90% var), 29 (99% var), spectral_radius=2.600
Mutation: lr_W: 2.0E-3 -> 7.0E-3
Parent rule: highest UCB node (Node 97 was only option), exploit strategy
Observation: lr_W=7.0E-3 dramatically improves connectivity_R2 from 0.506 to 0.931, confirming higher lr_W is critical for chaotic+Dale_law regime with 30% excitatory. test_R2=0.987 and pearson=0.948 are excellent. this inhibitory-dominant regime (factor=0.3) converges without noise, unlike excitatory-dominant (factor=0.6/0.7/0.8) which needed noise=0.5-0.75. spectral_radius=2.6 remains stable. effective rank dropped slightly from 30 to 29
Next: parent=98 (highest UCB=1.638), try lr_W=1.0E-2 to push connectivity_R2 above 0.95

## Iter 99: converged
Node: id=99, parent=98
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.0
Simulation: connectivity_type=chaotic, Dale_law=True, Dale_law_factor=0.3
Metrics: test_R2=0.9894, test_pearson=0.9586, connectivity_R2=0.9638, final_loss=3.0536e+02
Activity: oscillatory dynamics across 100 neurons, range [-12.0, 11.7], effective rank 10 (90% var), 30 (99% var), spectral_radius=2.600
Mutation: lr_W: 7.0E-3 -> 1.0E-2
Parent rule: highest UCB node (Node 98), exploit strategy continuing lr_W exploration
Observation: lr_W=1.0E-2 improved connectivity_R2 from 0.931 (iter 98) to 0.964 (+3.5%). test_pearson=0.959 slightly improved from 0.948. 2 consecutive converged results (iters 98-99) in Block 8. chaotic+Dale_law with 30% excitatory continues to converge well without noise. activity range slightly narrower [-12.0, 11.7] vs [-13.1, 13.3] at iter 98. final_loss=305 lower than iter 98's 333 (-8%), indicating more efficient training at higher lr_W
Next: parent=99 (highest UCB=1.830), try lr_W=1.5E-2 to test if higher lr_W can push connectivity_R2 toward 0.97+

## Iter 100: converged
Node: id=100, parent=99
Mode/Strategy: exploit
Config: lr_W=1.5E-2, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.0
Simulation: connectivity_type=chaotic, Dale_law=True, Dale_law_factor=0.3
Metrics: test_R2=0.8744, test_pearson=0.7828, connectivity_R2=0.9706, final_loss=2.9754e+02
Activity: oscillatory dynamics across 100 neurons, range [-11.1, 13.4], effective rank 10 (90% var), 30 (99% var), spectral_radius=2.600
Mutation: lr_W: 1.0E-2 -> 1.5E-2
Parent rule: highest UCB node (Node 99), exploit strategy continuing lr_W exploration
Observation: lr_W=1.5E-2 improved connectivity_R2 from 0.964 (iter 99) to 0.971 (+0.7%), but test_pearson dropped significantly from 0.959 to 0.783 (-18%) and test_R2 dropped from 0.989 to 0.874. 3 consecutive converged results (iters 98-100) in Block 8. higher lr_W benefits connectivity recovery but degrades temporal prediction quality. final_loss=298 lower than iter 99's 305 (-2%). chaotic+Dale_law with 30% excitatory (inhibitory-dominant) maintains convergence without noise, confirming this regime is more stable than excitatory-dominant (factor=0.6+)
Next: parent=100 (highest UCB=1.971), failure-probe with lr_W=5.0E-2 (3.3x current) to find lr_W upper stability boundary after 3 consecutive successes

## Iter 101: partial
Node: id=101, parent=100
Mode/Strategy: failure-probe
Config: lr_W=5.0E-2, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.0
Simulation: connectivity_type=chaotic, Dale_law=True, Dale_law_factor=0.3
Metrics: test_R2=0.8890, test_pearson=0.8000, connectivity_R2=0.8981, final_loss=4.7996e+02
Activity: oscillatory dynamics across 100 neurons, range [-12.7, 11.2], effective rank 11 (90% var), 31 (99% var), spectral_radius=2.600
Mutation: lr_W: 1.5E-2 -> 5.0E-2 (3.3x increase, failure-probe)
Parent rule: highest UCB node (Node 100), failure-probe to find lr_W upper stability boundary
Observation: lr_W=5.0E-2 found the upper boundary - connectivity_R2 dropped from 0.971 (iter 100) to 0.898 (-7.5%), falling below convergence threshold. test_pearson=0.800 slightly improved from iter 100's 0.783, but test_R2=0.889 lower than iter 100's 0.874. final_loss=480 higher than iter 100's 298 (+61%). confirms lr_W upper limit for chaotic+Dale_law with factor=0.3 is between 1.5E-2 (converged) and 5.0E-2 (partial). optimal lr_W appears to be around 1.0E-2 to 1.5E-2 for this regime
Next: parent=100 (highest UCB=1.716, R2=0.971), try lr_W=2.0E-2 (between 1.5E-2 and 5.0E-2) to refine lr_W upper boundary

## Iter 102: converged
Node: id=102, parent=101
Mode/Strategy: exploit
Config: lr_W=2.0E-2, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.0
Simulation: connectivity_type=chaotic, Dale_law=True, Dale_law_factor=0.3
Metrics: test_R2=0.9596, test_pearson=0.9029, connectivity_R2=0.9817, final_loss=3.1394e+02
Activity: oscillatory dynamics across 100 neurons, range [-12.2, 10.9], effective rank 11 (90% var), 31 (99% var), spectral_radius=2.600
Mutation: lr_W: 5.0E-2 -> 2.0E-2 (reverted from failure-probe to intermediate value)
Parent rule: highest UCB node (Node 101, but iter 101 was partial), exploit strategy to refine lr_W upper boundary
Observation: lr_W=2.0E-2 achieves strong convergence with R2=0.982, well above 0.9 threshold. test_pearson=0.903 recovered from iter 101's 0.800, and test_R2=0.960 improved from 0.889. confirms lr_W=2.0E-2 is within stable region for chaotic+Dale_law with factor=0.3 (inhibitory-dominant). final_loss=314 vs iter 100's 298 (+5%) is slightly higher but acceptable. lr_W boundary: 1.5E-2 to 2.0E-2 both converge (R2>0.97), 5.0E-2 fails (R2=0.898). 4 consecutive converged results (iters 98-100, 102) in Block 8, with one partial (iter 101 failure-probe)
Next: parent=102 (highest UCB=2.206), try lr=3.0E-4 (3x increase from baseline 1.0E-4) to test if higher model lr can improve temporal prediction (test_pearson) while maintaining connectivity_R2

## Iter 103: partial
Node: id=103, parent=102
Mode/Strategy: exploit
Config: lr_W=2.0E-2, lr=3.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.0
Simulation: connectivity_type=chaotic, Dale_law=True, Dale_law_factor=0.3
Metrics: test_R2=0.9303, test_pearson=0.8688, connectivity_R2=0.8320, final_loss=4.6611e+02
Activity: oscillatory dynamics across 100 neurons, range [-11.8, 13.7], effective rank 10 (90% var), 30 (99% var), spectral_radius=2.600
Mutation: lr: 1.0E-4 -> 3.0E-4
Parent rule: highest UCB node (Node 102), exploit strategy testing higher model lr for improved temporal prediction
Observation: lr=3.0E-4 degraded connectivity_R2 from 0.982 (iter 102) to 0.832 (-15%), falling below convergence threshold. test_pearson=0.869 and test_R2=0.930 remain reasonable but lower than iter 102's (0.903, 0.960). confirms model lr=1.0E-4 is optimal for chaotic+Dale_law with factor=0.3 (inhibitory-dominant) - higher lr destabilizes connectivity recovery. final_loss=466 higher than iter 102's 314 (+48%). this result establishes lr upper boundary at 1.0E-4 for this regime. Block 8 now has 4 converged (iters 98-100, 102), 2 partial (iters 97, 103), 1 to follow pattern from previous blocks
Next: parent=102 (highest UCB=1.864, R2=0.982), revert to optimal config (lr_W=2.0E-2, lr=1.0E-4) and try lr_W=1.0E-2 (lower) to test if reduced lr_W with baseline lr can achieve better balanced performance (high pearson + high R2)

## Iter 104: converged
Node: id=104, parent=103
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.0
Simulation: connectivity_type=chaotic, Dale_law=True, Dale_law_factor=0.3
Metrics: test_R2=0.9952, test_pearson=0.9846, connectivity_R2=0.9709, final_loss=3.0509e+02
Activity: oscillatory dynamics across 100 neurons, range [-11.3, 12.5], effective rank 9 (90% var), 30 (99% var), spectral_radius=2.600
Mutation: lr_W: 2.0E-2 -> 1.0E-2, lr: 3.0E-4 -> 1.0E-4
Parent rule: highest UCB node (Node 103), exploit strategy reverting lr to optimal 1.0E-4 and testing lower lr_W
Observation: lr_W=1.0E-2 with lr=1.0E-4 achieves excellent balanced performance - connectivity_R2=0.971 (converged) with best test_pearson=0.985 in Block 8 and test_R2=0.995. this matches iter 99's config and improves on its pearson (0.959 -> 0.985). confirms lr_W=1.0E-2 is optimal for chaotic+Dale_law with factor=0.3 - higher lr_W (1.5E-2 to 2.0E-2) improves connectivity slightly but degrades temporal prediction. 5 converged results (iters 98-100, 102, 104) in Block 8
Next: parent=104 (highest UCB=2.385), try batch_size=16 to test if larger batches can further improve connectivity_R2 toward 0.98+ while maintaining excellent temporal prediction

## Iter 105: converged
Node: id=105, parent=104
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=16, noise_model_level=0.0
Simulation: connectivity_type=chaotic, Dale_law=True, Dale_law_factor=0.3
Metrics: test_R2=0.8641, test_pearson=0.7887, connectivity_R2=0.9166, final_loss=2.3932e+02
Activity: oscillatory dynamics across 100 neurons, range [-11.1, 12.5], effective rank 10 (90% var), 30 (99% var), spectral_radius=2.600
Mutation: batch_size: 8 -> 16
Parent rule: highest UCB node (Node 104), exploit strategy testing larger batch size
Observation: batch_size=16 degraded performance compared to iter 104 (batch_size=8) - connectivity_R2 dropped from 0.971 to 0.917 (-5.5%), test_pearson dropped from 0.985 to 0.789 (-20%), test_R2 dropped from 0.995 to 0.864 (-13%). confirms batch_size=8 is optimal for chaotic+Dale_law with factor=0.3. final_loss=239 lower than iter 104's 305 (-22%) but other metrics worse. Block 8 now has 6 converged (iters 98-100, 102, 104, 105) with iter 104 showing best balanced performance
Next: parent=104 (R2=0.971, best balanced config), revert batch_size to 8 and try noise_model_level=0.5 to test if noise regularization can push connectivity_R2 above 0.98 for this inhibitory-dominant regime

## Iter 106: converged
Node: id=106, parent=105
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.5
Simulation: connectivity_type=chaotic, Dale_law=True, Dale_law_factor=0.3
Metrics: test_R2=0.9963, test_pearson=0.2210, connectivity_R2=0.9998, final_loss=2.1479e+02
Activity: sustained oscillatory dynamics across 100 neurons, range [-21.5, 18.9], effective rank 48 (90% var), 89 (99% var), spectral_radius=2.600
Mutation: batch_size: 16 -> 8, noise_model_level: 0.0 -> 0.5
Parent rule: highest UCB node (Node 105), exploit strategy testing noise regularization with reverted batch_size
Observation: noise_model_level=0.5 dramatically improved connectivity_R2 from 0.917 (iter 105) to 0.9998 - best connectivity in Block 8. however, test_pearson dropped from 0.789 (iter 105) to 0.221 (-72%), consistent with Block 6/7 trade-off pattern where noise degrades temporal prediction. effective rank increased from 30 to 89 (99% var) due to noise injection. activity range expanded [-21.5, 18.9] vs iter 105's [-11.1, 12.5]. confirms chaotic+Dale_law with factor=0.3 (inhibitory-dominant) follows same noise trade-off as factor=0.6 regimes. 7 converged results in Block 8 (iters 98-100, 102, 104-106)
Next: parent=106 (highest UCB=2.581), try noise_model_level=0.25 (between 0 and 0.5) to find intermediate noise level that balances connectivity_R2 and temporal prediction

## Iter 107: converged
Node: id=107, parent=106
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.25
Simulation: connectivity_type=chaotic, Dale_law=True, Dale_law_factor=0.3
Metrics: test_R2=0.9728, test_pearson=0.3000, connectivity_R2=0.9998, final_loss=1.8714e+02
Activity: oscillatory dynamics across 100 neurons, range [-14.6, 17.7], effective rank 29 (90% var), 78 (99% var), spectral_radius=2.600
Mutation: noise_model_level: 0.5 -> 0.25
Parent rule: highest UCB node (Node 106), exploit strategy testing intermediate noise level to balance connectivity and temporal prediction
Observation: noise_model_level=0.25 maintains excellent connectivity_R2=0.9998 (matches iter 106's peak), with test_pearson=0.300 slightly improved from iter 106's 0.221 (+36%) but still much lower than noise-free configs (iter 104: 0.985). test_R2=0.972 comparable to iter 106's 0.996. effective rank=78 (99% var) lower than iter 106's 89, confirming reduced noise decreases activity complexity. final_loss=187 lowest in Block 8 (-13% vs iter 106's 215). Block 8 result: noise helps connectivity recovery but always degrades temporal prediction for chaotic+Dale_law factor=0.3 regime. optimal trade-off depends on priority: connectivity → noise=0.25-0.5, temporal → no noise with lr_W=1.0E-2
Next: parent=107 (highest UCB=2.658), last iteration of Block 8, robustness-test by re-running optimal no-noise config (lr_W=1.0E-2, lr=1.0E-4, batch_size=8, noise=0) to verify iter 104's balanced performance is reproducible

## Iter 108: converged
Node: id=108, parent=107
Mode/Strategy: robustness-test
Config: lr_W=1.0E-2, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.0
Simulation: connectivity_type=chaotic, Dale_law=True, Dale_law_factor=0.3
Metrics: test_R2=0.9812, test_pearson=0.9294, connectivity_R2=0.9544, final_loss=2.8957e+02
Activity: oscillatory dynamics across 100 neurons, range [-11.0, 13.2], effective rank 10 (90% var), 30 (99% var), spectral_radius=2.600
Mutation: robustness-test - reverted to optimal no-noise config (noise_model_level: 0.25 -> 0.0)
Parent rule: highest UCB node (Node 107), robustness-test as final Block 8 iteration to verify no-noise balanced performance
Observation: robustness-test confirms no-noise config achieves excellent balanced performance: connectivity_R2=0.954 (converged) with test_pearson=0.929 (strong temporal prediction). compared to iter 104 (R2=0.971, pearson=0.985), this run shows slightly lower connectivity but still converged. compared to noise configs (iter 106-107: R2>0.999, pearson~0.22-0.30), no-noise preserves temporal prediction at cost of ~5% connectivity. Block 8 completed with 9 converged (iters 98-100, 102, 104-108), 1 partial (iter 97 baseline), 2 partial from probes (iters 101, 103)

### Block 8 Summary (iters 97-108)
Simulation: chaotic connectivity, Dale_law=True, Dale_law_factor=0.3 (30% excitatory, 70% inhibitory)
Best R2: 0.9998 at iter 106-107 (connectivity with noise), 0.985 pearson at iter 104 (best temporal without noise)
Observation:
- inhibitory-dominant regime (factor=0.3) converges without noise - baseline (iter 97) achieved partial R2=0.506, unlike excitatory-dominant (factor=0.6+) which failed completely without noise
- spectral_radius=2.6 is moderate: higher than balanced Dale_law (factor=0.6: 1.32) but lower than pathological excitatory (factor=0.8: 4.47)
- noise trade-off confirmed: noise=0.25-0.5 achieves R2>0.999 but degrades pearson from ~0.93-0.99 to ~0.22-0.30
- optimal lr_W range: 1.0E-2 to 2.0E-2 for no-noise regime, lr_W=5.0E-2 causes instability (iter 101 partial)
- model lr sensitive: lr=1.0E-4 optimal, lr=3.0E-4 degrades connectivity (iter 103 partial)
Optimum training parameters:
- For best connectivity (R2>0.999): lr_W=1.0E-2, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.25-0.5
- For best temporal (pearson~0.98): lr_W=1.0E-2, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.0

---

Next: Block 9 starts at iter 109, explore connectivity_type="chaotic" with Dale_law=True and Dale_law_factor=0.4 (40% excitatory, 60% inhibitory) to map the transition between inhibitory-dominant (factor=0.3, converges without noise) and balanced (factor=0.6, requires noise)

## Iter 109: partial
--- NEW BLOCK ---
Simulation: connectivity_type=chaotic, Dale_law=True, Dale_law_factor=0.4 (40% excitatory, 60% inhibitory), noise_model_level=0.0
Node: id=109, parent=root
Mode/Strategy: exploit
Config: lr_W=2.0E-3, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.0
Metrics: test_R2=0.7599, test_pearson=0.5952, connectivity_R2=0.3054, final_loss=6.9414e+02
Activity: oscillatory dynamics across 100 neurons, range [-12.4, 15.9], effective rank 4 (90% var), 17 (99% var), spectral_radius=1.389
Mutation: baseline (first iteration of new block), simulation changed to Dale_law_factor=0.4
Parent rule: first iteration of Block 9, parent=root
Observation: chaotic+Dale_law with 40% excitatory (60% inhibitory) produces partial convergence at R2=0.305. spectral_radius=1.389 is between factor=0.3 (2.6) and factor=0.6 (1.32), moderate instability. effective rank=17 (99% var) is lower than factor=0.3's 30, indicating simpler dynamics. baseline lr_W=2.0E-3 achieves partial R2=0.305, similar to factor=0.3 baseline (iter 97: R2=0.506) - both start partial without noise
Next: parent=109, increase lr_W to 1.0E-2 (optimal from Block 8 for factor=0.3) to test if higher lr_W can achieve convergence


## Iter 110: partial
Node: id=110, parent=109
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.0
Simulation: connectivity_type=chaotic, Dale_law=True, Dale_law_factor=0.4
Metrics: test_R2=0.9812, test_pearson=0.8999, connectivity_R2=0.7237, final_loss=2.4456e+02
Activity: oscillatory dynamics across 100 neurons, range [-12.4, 15.9], effective rank 4 (90% var), 17 (99% var), spectral_radius=1.389
Mutation: lr_W: 2.0E-3 -> 1.0E-2 (5x increase)
Parent rule: highest UCB node (Node 109), exploit strategy testing optimal lr_W from Block 8
Observation: lr_W=1.0E-2 improved connectivity_R2 from 0.305 (iter 109) to 0.724 (+137%) but still partial (below 0.9 threshold). test_R2=0.981 and test_pearson=0.900 show excellent temporal prediction. final_loss=245 lower than iter 109's 694 (-65%). factor=0.4 regime shows intermediate behavior between factor=0.3 (converged at R2=0.97 with same lr_W) and factor=0.6 (required noise). effective rank remains low (17 at 99% var), dynamics relatively simple. higher lr_W may push to convergence, or noise may be needed
Next: parent=110 (highest UCB=1.431), try lr_W=2.0E-2 (2x increase) to test if higher lr_W can push connectivity_R2 above 0.9 threshold

## Iter 111: partial
Node: id=111, parent=110
Mode/Strategy: exploit
Config: lr_W=2.0E-2, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.0
Simulation: connectivity_type=chaotic, Dale_law=True, Dale_law_factor=0.4
Metrics: test_R2=0.9838, test_pearson=0.9259, connectivity_R2=0.7824, final_loss=2.2019e+02
Activity: oscillatory dynamics across 100 neurons, range [-12.4, 15.9], effective rank 4 (90% var), 17 (99% var), spectral_radius=1.389
Mutation: lr_W: 1.0E-2 -> 2.0E-2 (2x increase)
Parent rule: highest UCB node (Node 110), exploit strategy testing higher lr_W to push toward convergence
Observation: lr_W=2.0E-2 improved connectivity_R2 from 0.724 (iter 110) to 0.782 (+8%) but still partial. test_pearson=0.926 improved from 0.900 (+3%). final_loss=220 decreased from 245 (-10%). factor=0.4 regime resists convergence without noise even with high lr_W - similar to factor=0.6 blocks that required noise. effective rank=17 (99% var) unchanged, dynamics remain relatively simple. higher lr_W shows diminishing returns (iter 109→110: +137% R2 gain, iter 110→111: +8% R2 gain). noise regularization may be required for this regime
Next: parent=111 (highest UCB=1.648), add noise_model_level=0.5 to test if noise can push connectivity_R2 above 0.9 threshold like it did for factor=0.3 and factor=0.6 regimes

## Iter 112: converged
Node: id=112, parent=111
Mode/Strategy: exploit
Config: lr_W=2.0E-2, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.5
Simulation: connectivity_type=chaotic, Dale_law=True, Dale_law_factor=0.4
Metrics: test_R2=0.8935, test_pearson=0.1716, connectivity_R2=0.9996, final_loss=2.7745e+02
Activity: oscillatory dynamics across 100 neurons, range [-18.2, 19.9], effective rank 48 (90% var), 89 (99% var), spectral_radius=1.389
Mutation: noise_model_level: 0.0 -> 0.5
Parent rule: highest UCB node (Node 111), exploit strategy adding noise regularization to achieve convergence
Observation: breakthrough! noise_model_level=0.5 achieved convergence with connectivity_R2=0.9996 (up from iter 111's partial 0.782). effective rank jumped from 17 to 89 (99% var), showing noise enriches dynamics. test_pearson=0.172 degraded from iter 111's excellent 0.926 - same noise trade-off pattern seen in factor=0.3 (Block 8) and factor=0.6 (Block 5). confirms Dale_law_factor=0.4 requires noise for connectivity recovery like factor=0.6, unlike factor=0.3 which converged without noise
Next: parent=112 (highest UCB=2.000), try noise_model_level=0.25 (intermediate) to test if lower noise can maintain convergence while improving temporal prediction (pearson)

## Iter 113: converged
Node: id=113, parent=112
Mode/Strategy: exploit
Config: lr_W=2.0E-2, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.25
Simulation: connectivity_type=chaotic, Dale_law=True, Dale_law_factor=0.4
Metrics: test_R2=0.8816, test_pearson=0.2626, connectivity_R2=0.9997, final_loss=2.1048e+02
Activity: oscillatory dynamics across 100 neurons, range [-14.1, 20.2], effective rank 24 (90% var), 74 (99% var), spectral_radius=1.389
Mutation: noise_model_level: 0.5 -> 0.25 (reduced noise)
Parent rule: highest UCB node (Node 112), exploit strategy testing lower noise to balance connectivity and temporal prediction
Observation: noise_model_level=0.25 maintains excellent connectivity_R2=0.9997 (matching iter 112's 0.9996). test_pearson=0.263 improved slightly from iter 112's 0.172 but still degraded compared to no-noise iters 110-111 (pearson ~0.9). effective rank dropped from 89 to 74 (99% var) with reduced noise. confirms factor=0.4 regime needs noise for connectivity recovery but noise degrades temporal prediction - similar trade-off pattern as factor=0.3 and factor=0.6
Next: parent=113 (highest UCB=2.118), try noise_model_level=0.1 to test minimum noise threshold for convergence in factor=0.4 regime

## Iter 114: converged
Node: id=114, parent=113
Mode/Strategy: exploit
Config: lr_W=2.0E-2, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.1
Simulation: connectivity_type=chaotic, Dale_law=True, Dale_law_factor=0.4
Metrics: test_R2=0.9099, test_pearson=0.5237, connectivity_R2=0.9887, final_loss=1.9043e+02
Activity: oscillatory dynamics across 100 neurons, range [-15.3, 19.6], effective rank 6 (90% var), 39 (99% var), spectral_radius=1.389
Mutation: noise_model_level: 0.25 -> 0.1 (reduced noise to find minimum threshold)
Parent rule: highest UCB node (Node 113), exploit strategy testing minimum noise threshold for convergence
Observation: noise_model_level=0.1 maintains convergence with connectivity_R2=0.989 (slightly below iter 113's 0.9997 but still excellent). test_pearson=0.524 improved significantly from iter 113's 0.263 (+99%), showing lower noise preserves more temporal correlation. effective rank=39 (99% var) decreased from iter 113's 74 as expected with reduced noise. this establishes noise=0.1 as viable minimum for factor=0.4 convergence - significant improvement over noise=0.25-0.5 for temporal prediction while maintaining connectivity recovery. 3 consecutive converged results (iters 112-114)
Next: parent=114 (highest UCB=2.213), try lr_W=1.0E-2 (lower) with noise=0.1 to test if reduced lr_W can improve temporal prediction (pearson) while maintaining convergence

## Iter 115: converged
Node: id=115, parent=114
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.1
Simulation: connectivity_type=chaotic, Dale_law=True, Dale_law_factor=0.4
Metrics: test_R2=0.9148, test_pearson=0.4942, connectivity_R2=0.9789, final_loss=1.6990e+02
Activity: oscillatory dynamics across 100 neurons, range [-15.3, 19.6], effective rank 6 (90% var), 39 (99% var), spectral_radius=1.389
Mutation: lr_W: 2.0E-2 -> 1.0E-2 (reduced lr_W)
Parent rule: highest UCB node (Node 114), exploit strategy testing reduced lr_W with minimal noise to improve temporal prediction
Observation: lr_W=1.0E-2 with noise=0.1 maintains excellent connectivity_R2=0.979 (slightly below iter 114's 0.989 but still converged). test_pearson=0.494 slightly decreased from iter 114's 0.524 (-5.7%), suggesting lr_W=2.0E-2 (iter 114) is better than lr_W=1.0E-2 for temporal prediction in factor=0.4 regime with noise=0.1. final_loss=170 lower than iter 114's 190 (-11%). 4 consecutive converged results (iters 112-115). noise=0.1 establishes viable operating point for factor=0.4: connectivity_R2~0.98-0.99 with pearson~0.49-0.52, better balanced than noise=0.25-0.5 regimes
Next: parent=115 (highest UCB=2.302), try lr=1.0E-3 (10x higher model lr) with noise=0.1 to test if higher model lr can improve temporal prediction while maintaining convergence - this worked in previous blocks for improving connectivity

## Iter 116: converged
Node: id=116, parent=115
Mode/Strategy: exploit
Config: lr_W=1.0E-2, lr=1.0E-3, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8, noise_model_level=0.1
Simulation: connectivity_type=chaotic, Dale_law=True, Dale_law_factor=0.4
Metrics: test_R2=0.8416, test_pearson=0.5771, connectivity_R2=0.9573, final_loss=2.4148e+02
Activity: oscillatory dynamics across 100 neurons, range [-15.3, 19.6], effective rank 6 (90% var), 39 (99% var), spectral_radius=1.389
Mutation: lr: 1.0E-4 -> 1.0E-3 (10x increase)
Parent rule: highest UCB node (Node 115), exploit strategy testing higher model lr to improve temporal prediction
Observation: lr=1.0E-3 with noise=0.1 maintains convergence with connectivity_R2=0.957 (slightly below iter 115's 0.979 but still converged). test_pearson=0.577 improved significantly from iter 115's 0.494 (+17%), confirming higher model lr improves temporal prediction. test_R2=0.842 lower than iter 115's 0.915. 5 consecutive converged results (iters 112-116). factor=0.4 regime with noise=0.1 shows best pearson yet (0.577) among converged results - better than noise=0.25-0.5 which capped at ~0.26
Next: parent=116 (highest UCB=2.372), try lr=2.0E-3 (2x current) to test if higher model lr further improves temporal prediction while maintaining connectivity_R2>0.9
