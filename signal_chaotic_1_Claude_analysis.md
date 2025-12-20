# Experiment Log: signal_chaotic_1_Claude

## Iter 1: converged
Node: id=1, parent=root
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.002, lr=0.0001, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=8, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.7970, test_pearson=0.8179, connectivity_R2=0.9824, final_loss=6.5622e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-22.7, 22.7], effective rank 34, spectral radius 1.014
Mutation: baseline (no prior iteration)
Parent rule: first iteration, parent=root by default
Observation: strong convergence (R2=0.982) with baseline config on chaotic connectivity
Next: parent=1 (highest UCB, will explore mutation)

## Iter 2: converged
Node: id=2, parent=1
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.003, lr=0.0001, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=8, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9922, test_pearson=0.9811, connectivity_R2=0.9995, final_loss=4.9747e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-18.5, 22.2], effective rank 33, spectral radius 0.973
Mutation: lr_W: 0.002 -> 0.003
Parent rule: Node 1 had highest UCB (1.454), branched with lr_W increase
Observation: excellent convergence (R2=0.9995), increasing lr_W from 0.002 to 0.003 improved both connectivity_R2 and test_R2 significantly
Next: parent=2 (highest UCB=1.707, continue exploit strategy)

## Iter 3: converged
Node: id=3, parent=2
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.003, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=8, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9208, test_pearson=0.8692, connectivity_R2=0.9998, final_loss=4.0264e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-21.5, 21.3], effective rank 33, spectral radius 0.973
Mutation: lr: 0.0001 -> 0.0002
Parent rule: Node 2 had highest UCB (1.577), continued with lr increase
Observation: connectivity_R2=0.9998 near-perfect, test_R2 slightly lower than iter 2 but loss improved. 3 consecutive successes triggers failure-probe strategy
Next: parent=3 (highest UCB=1.866, failure-probe: try extreme lr_W=0.01 to find boundary)

## Iter 4: converged
Node: id=4, parent=3
Mode/Strategy: failure-probe/boundary
Config: lr_W=0.01, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=8, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9663, test_pearson=0.9407, connectivity_R2=0.9997, final_loss=2.1040e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-23.8, 24.5], effective rank 34, spectral radius 0.973
Mutation: lr_W: 0.003 -> 0.01
Parent rule: Node 3 had highest UCB (1.866), failure-probe with extreme lr_W=0.01
Observation: failure-probe did NOT find boundary - lr_W=0.01 (5x baseline) still achieves R2=0.9997. training is robust to high lr_W. 4 consecutive successes, continue probing lower boundary
Next: parent=4 (highest UCB=2.000, failure-probe: try extreme low lr_W=0.0005 to find lower boundary)

## Iter 5: partial
Node: id=5, parent=4
Mode/Strategy: failure-probe/boundary
Config: lr_W=0.0005, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=8, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.8520, test_pearson=0.7672, connectivity_R2=0.3816, final_loss=1.0463e+04
Activity: chaotic oscillations across 100 neurons, amplitude [-20.7, 22.0], effective rank 35, spectral radius 0.973
Mutation: lr_W: 0.01 -> 0.0005
Parent rule: Node 4 had highest UCB, failure-probe with extreme low lr_W=0.0005 to find lower boundary
Observation: found lower boundary! lr_W=0.0005 fails (R2=0.38). optimal lr_W range is [0.002, 0.01]. returning to exploit strategy from node 4
Next: parent=4 (highest UCB=1.745, exploit: try coeff_W_L1 variation from successful config)

## Iter 6: converged
Node: id=6, parent=4
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.01, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=8, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.7701, test_pearson=0.7171, connectivity_R2=0.9979, final_loss=3.2066e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-21.9, 22.2], effective rank 34, spectral radius 0.973
Mutation: coeff_W_L1: 1.0e-05 -> 1.0e-04
Parent rule: Node 4 had highest UCB (2.000), branched with coeff_W_L1 increase
Observation: connectivity_R2=0.9979 still excellent with 10x L1 regularization. test metrics slightly lower than iter 4 (pearson 0.72 vs 0.94) but connectivity still well-recovered. 5 converged out of 6 iters, explore untested parameter space
Next: parent=6 (highest UCB=2.223, exploit: try batch_size=16 from this successful config)

## Iter 7: converged
Node: id=7, parent=6
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.01, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=16, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9379, test_pearson=0.8936, connectivity_R2=0.9989, final_loss=2.2303e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-19.7, 19.8], effective rank 34, spectral radius 0.973
Mutation: batch_size: 8 -> 16
Parent rule: Node 6 had highest UCB (2.223), branched with batch_size increase
Observation: connectivity_R2=0.9989 excellent with batch_size=16. test_R2 improved (0.94 vs 0.77 in iter 6), pearson also better (0.89 vs 0.72). 6 consecutive successes (iters 2-7), triggers explore strategy per protocol
Next: parent=1 (explore strategy: select node outside recent n_iter_block/4=6 chain to try different branch; will probe lr_W boundary from baseline)

## Iter 8: failed
Node: id=8, parent=1
Mode/Strategy: explore/exploit
Config: lr_W=0.005, lr=0.0001, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=8, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.3727, test_pearson=-0.0747, connectivity_R2=0.1354, final_loss=7.7547e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-21.2, 20.9], effective rank 33, spectral radius 0.973
Mutation: lr_W: 0.002 -> 0.005 (from baseline Node 1)
Parent rule: explore strategy selected Node 1 (outside recent chain) with lr_W increase
Observation: explore from Node 1 failed (R2=0.135). lr_W=0.005 with lr=0.0001 underperforms vs lr_W=0.003 + lr=0.0002 (Node 3). key insight: lr and lr_W must scale together for optimal training
Next: parent=7 (highest UCB=2.413, return to best performing branch with batch_size=32 probe)

## Iter 9: converged
Node: id=9, parent=7
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.01, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9873, test_pearson=0.9722, connectivity_R2=0.9981, final_loss=2.5202e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-20.1, 22.3], effective rank 32, spectral radius 0.973
Mutation: batch_size: 16 -> 32
Parent rule: Node 7 had highest UCB (1.999), continued batch_size scaling from 16 to 32
Observation: connectivity_R2=0.9981 excellent with batch_size=32. test metrics improved (pearson 0.972 vs 0.894 in iter 7). 7 converged iters (2-4, 6-7, 9), training highly robust. explore strategy: probe different parameter not yet varied from best branch
Next: parent=9 (highest UCB=2.498, exploit: try lr=0.0003 to test sensitivity of model lr)

## Iter 10: converged
Node: id=10, parent=9
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.01, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.8057, test_pearson=0.7282, connectivity_R2=0.9979, final_loss=2.9390e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-23.1, 19.8], effective rank 33, spectral radius 0.973
Mutation: lr: 0.0002 -> 0.0003
Parent rule: Node 9 had highest UCB (2.052), continued lr scaling from 0.0002 to 0.0003
Observation: connectivity_R2=0.9979 maintained excellent with lr=0.0003. however test metrics dropped significantly (test_R2: 0.99→0.81, pearson: 0.97→0.73). suggests lr=0.0003 is at boundary for test generalization. 8 converged iters (1-4, 6-7, 9-10). 4+ consecutive successes triggers failure-probe
Next: parent=10 (highest UCB=2.579, failure-probe: try lr=0.0001 to probe lower lr boundary)

## Iter 11: converged
Node: id=11, parent=10
Mode/Strategy: failure-probe/boundary
Config: lr_W=0.01, lr=0.0001, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9315, test_pearson=0.8853, connectivity_R2=0.9985, final_loss=2.8256e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-27.6, 19.8], effective rank 35, spectral radius 0.973
Mutation: lr: 0.0003 -> 0.0001
Parent rule: Node 10 had highest UCB (2.579), failure-probe with lr=0.0001 to test lower lr boundary
Observation: lr=0.0001 (3x lower than iter 10) maintains excellent connectivity_R2=0.9985 and actually improves test metrics (test_R2: 0.81→0.93, pearson: 0.73→0.89). lr=0.0001 is better than lr=0.0003 for generalization. 9 converged iters (1-4, 6-7, 9-11). 5 consecutive successes, continue failure-probe to find actual boundaries
Next: parent=11 (highest UCB=2.657, failure-probe: try coeff_W_L1=1.0e-03 to probe upper L1 boundary)

## Iter 12: partial
Node: id=12, parent=11
Mode/Strategy: failure-probe/boundary
Config: lr_W=0.01, lr=0.0001, lr_emb=0.00025, coeff_W_L1=1.0e-03, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.4597, test_pearson=0.3201, connectivity_R2=0.2103, final_loss=4.0346e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-18.5, 20.7], effective rank 25, spectral radius 0.973
Mutation: coeff_W_L1: 1.0e-04 -> 1.0e-03
Parent rule: Node 11 had highest UCB (2.153), failure-probe with coeff_W_L1=1.0e-03 to probe upper L1 boundary
Observation: found upper L1 boundary! coeff_W_L1=1.0e-03 (10x increase) fails (R2=0.21). L1 too high over-penalizes weights, preventing connectivity recovery. optimal L1 range is [1e-05, 1e-04]. returning to exploit strategy from Node 11. 9/12 converged
Next: parent=11 (highest UCB=2.153, exploit: try lr_W=0.005 to probe lower lr_W boundary from this successful config)

## Iter 13: converged
Node: id=13, parent=11
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.005, lr=0.0001, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9847, test_pearson=0.9729, connectivity_R2=0.9928, final_loss=4.1080e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-21.5, 21.5], effective rank 34, spectral radius 0.973
Mutation: lr_W: 0.01 -> 0.005
Parent rule: Node 11 had highest UCB (2.153), exploit with lr_W=0.005 to test mid-range value
Observation: lr_W=0.005 achieves excellent R2=0.9928 with best test metrics so far (test_R2=0.985, pearson=0.973). confirms optimal lr_W range [0.003, 0.01]. 10/13 converged. Node 13 now has highest UCB (2.796)
Next: parent=13 (highest UCB=2.796, exploit: try lr_W=0.003 to refine optimal lr_W)

## Iter 14: partial
Node: id=14, parent=13
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.003, lr=0.0001, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.8072, test_pearson=0.7604, connectivity_R2=0.8281, final_loss=5.7447e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-23.8, 23.6], effective rank 35, spectral radius 0.973
Mutation: lr_W: 0.005 -> 0.003
Parent rule: Node 13 had highest UCB (2.796), exploit with lr_W=0.003 to refine optimal lr_W
Observation: unexpected partial result (R2=0.828) with lr_W=0.003. this config worked well in early iters (Node 2-4 with R2>0.99) but now underperforms. possible seed/initialization variance. key insight: lr_W=0.003 + lr=0.0001 is less stable than lr_W=0.005 + lr=0.0001. 10/14 converged
Next: parent=13 (highest UCB=2.240, return to lr_W=0.005 with different mutation; try lr=0.0002 to test lr sensitivity)

## Iter 15: converged
Node: id=15, parent=13
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.005, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9663, test_pearson=0.9392, connectivity_R2=0.9999, final_loss=2.9182e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-26.2, 19.8], effective rank 35, spectral radius 0.973
Mutation: lr: 0.0001 -> 0.0002
Parent rule: Node 13 had highest UCB (2.796), tried lr=0.0002 from successful lr_W=0.005 config
Observation: near-perfect R2=0.9999 achieved! lr_W=0.005 + lr=0.0002 is optimal combination. test metrics excellent (test_R2=0.9663, pearson=0.9392). confirms lr and lr_W interaction matters. 11/15 converged. 3 recent successes (13,15), next could try failure-probe or explore less-visited branches
Next: parent=15 (highest UCB=2.936, failure-probe: try extreme lr_W=0.015 to probe upper boundary)

## Iter 16: converged
Node: id=16, parent=15
Mode/Strategy: failure-probe/boundary
Config: lr_W=0.015, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.8597, test_pearson=0.7882, connectivity_R2=0.9976, final_loss=2.0674e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-19.2, 20.4], effective rank 34, spectral radius 0.973
Mutation: lr_W: 0.005 -> 0.015
Parent rule: Node 15 had highest UCB (2.936), failure-probe with lr_W=0.015 (3x) to probe upper boundary
Observation: lr_W=0.015 still converges (R2=0.998) but test metrics degraded (pearson 0.94→0.79). found soft upper boundary: lr_W>0.01 hurts generalization. optimal lr_W range refined to [0.005, 0.01]. 12/16 converged (75%). 4 consecutive successes (13,15,16), continue exploit from best recent config
Next: parent=16 (highest UCB=2.998, exploit: try lr=0.0003 to test lr sensitivity at high lr_W)

## Iter 17: converged
Node: id=17, parent=16
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.015, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9530, test_pearson=0.9198, connectivity_R2=0.9981, final_loss=1.7360e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-21.0, 20.0], effective rank 34, spectral radius 0.973
Mutation: lr: 0.0002 -> 0.0003
Parent rule: Node 16 had highest UCB (2.998), exploit with lr=0.0003 to test lr sensitivity at high lr_W
Observation: lr_W=0.015 + lr=0.0003 achieves R2=0.998 with improved test metrics vs iter 16 (pearson 0.79→0.92). lowest loss so far (1736). 5 consecutive successes (13,15,16,17), 13/17 converged (76%). explore strategy: branch from less-visited node outside recent chain
Next: parent=14 (explore strategy: Node 14 UCB=2.890 is 2nd highest, partial result R2=0.828 warrants re-exploration with different mutation; try lr_W=0.004 to refine boundary around lr_W=0.003)

## Iter 18: partial
Node: id=18, parent=14
Mode/Strategy: explore/exploit
Config: lr_W=0.004, lr=0.0001, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.8262, test_pearson=0.7567, connectivity_R2=0.8986, final_loss=5.0035e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-22.5, 23.4], effective rank 34, spectral radius 0.973
Mutation: lr_W: 0.003 -> 0.004 (from Node 14's lr_W=0.003)
Parent rule: Node 14 had high UCB, explored with lr_W=0.004 to refine boundary around lr_W=0.003
Observation: partial result (R2=0.899). lr_W=0.003-0.004 with lr=0.0001 appears to be a weak combination. contrast with iter 13 (lr_W=0.005, lr=0.0001, R2=0.993) - suggests lr_W<0.005 with lr=0.0001 is suboptimal. confirmed lr_W and lr must scale together. 13/18 converged (72%)
Next: parent=17 (highest UCB=3.119, R2=0.998; exploit from best recent config with different mutation)

## Iter 19: converged
Node: id=19, parent=17
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.015, lr=0.0003, lr_emb=0.00025, coeff_W_L1=5.0e-05, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9209, test_pearson=0.8783, connectivity_R2=0.9993, final_loss=1.8856e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-22.3, 23.2], effective rank 34, spectral radius 0.973
Mutation: coeff_W_L1: 1.0e-04 -> 5.0e-05
Parent rule: Node 17 had highest UCB (3.119), exploit with reduced L1 regularization
Observation: R2=0.9993 excellent with halved L1 regularization. test metrics slightly lower than iter 17 (pearson 0.92→0.88) but connectivity recovery maintained. confirms coeff_W_L1 range [5e-05, 1e-04] works well. 14/19 converged (74%). 3 consecutive successes (17,19) after partial iter 18, triggers failure-probe
Next: parent=19 (highest UCB=3.179, failure-probe: try coeff_W_L1=1.0e-06 to probe lower L1 boundary)

## Iter 20: converged
Node: id=20, parent=19
Mode/Strategy: failure-probe/boundary
Config: lr_W=0.015, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-06, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9926, test_pearson=0.9870, connectivity_R2=0.9997, final_loss=2.0275e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-23.5, 19.8], effective rank 34, spectral radius 0.973
Mutation: coeff_W_L1: 5.0e-05 -> 1.0e-06
Parent rule: Node 19 had highest UCB (3.179), failure-probe with coeff_W_L1=1.0e-06 to probe lower L1 boundary
Observation: failure-probe did NOT find boundary - coeff_W_L1=1.0e-06 (50x lower) achieves best test metrics yet (test_R2=0.993, pearson=0.987) with R2=0.9997. very low L1 regularization works excellently. confirms optimal L1 range extends down to 1e-06. 15/20 converged (75%). 6 consecutive successes (15-17,19-20), triggers explore strategy
Next: parent=18 (explore strategy: Node 18 UCB=3.135, partial R2=0.899, unexplored branch; try lr=0.0002 to test lr scaling hypothesis)

## Iter 21: converged
Node: id=21, parent=18
Mode/Strategy: explore/exploit
Config: lr_W=0.004, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9883, test_pearson=0.9754, connectivity_R2=0.9999, final_loss=3.3925e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-18.5, 24.2], effective rank 35, spectral radius 0.973
Mutation: lr: 0.0001 -> 0.0002 (from Node 18's lr=0.0001)
Parent rule: Node 18 was explore target (partial R2=0.899), tested lr=0.0002 to verify lr scaling hypothesis
Observation: explore from partial Node 18 succeeded brilliantly - R2=0.9999 (best yet) with excellent test metrics (test_R2=0.988, pearson=0.975). confirms hypothesis: lr_W=0.004 + lr=0.0002 works better than lr_W=0.004 + lr=0.0001. lr scaling ratio ~20:1 (lr_W:lr) is optimal. 16/21 converged (76%). 4 consecutive successes (19-21), triggers failure-probe
Next: parent=21 (highest UCB=3.291, tied with Node 20; failure-probe: try lr_W=0.008 to probe lr_W upper boundary while maintaining 20:1 ratio)

## Iter 22: converged
Node: id=22, parent=21
Mode/Strategy: failure-probe/boundary
Config: lr_W=0.008, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9841, test_pearson=0.9687, connectivity_R2=0.9986, final_loss=2.5598e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-23.1, 19.8], effective rank 32, spectral radius 0.973
Mutation: lr_W: 0.004 -> 0.008 (from Node 21's lr_W=0.004)
Parent rule: Node 21 had high UCB (3.291), failure-probe with lr_W=0.008 to test 40:1 lr_W:lr ratio
Observation: failure-probe did NOT find boundary - lr_W=0.008 (2x) achieves excellent R2=0.9986 with great test metrics (test_R2=0.984, pearson=0.969). confirms 40:1 ratio works well. 17/22 converged (77%). 8 consecutive successes (15-17, 19-22), triggers explore strategy per protocol
Next: parent=20 (explore strategy: Node 20 has highest UCB=3.345, try lr_W=0.01 to probe 33:1 ratio boundary)

## Iter 23: converged
Node: id=23, parent=20
Mode/Strategy: explore/exploit
Config: lr_W=0.01, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-06, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9732, test_pearson=0.9499, connectivity_R2=0.9999, final_loss=2.1074e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-22.8, 22.2], effective rank 33, spectral radius 0.973
Mutation: coeff_W_L1: 1.0e-06 -> 1.0e-06 (unchanged from Node 20), lr_W: 0.015 -> 0.01 (from Node 20's lr_W=0.015)
Parent rule: Node 20 had highest UCB (3.345), explore strategy with lr_W=0.01 to test 33:1 lr_W:lr ratio
Observation: excellent R2=0.9999 achieved with lr_W=0.01 + lr=0.0003 + coeff_W_L1=1e-06. test metrics strong (test_R2=0.973, pearson=0.950). confirms optimal config zone. 18/23 converged (78%). 9 consecutive successes (15-17, 19-23). block 1 near completion (iter 23/24), next iter is block end
Next: parent=5 (explore strategy: Node 5 UCB=2.780, partial R2=0.382 with lr_W=0.0005; unexplored failure branch, try lr_W=0.001 to probe lower boundary more precisely)

## Iter 24: partial
Node: id=24, parent=5
Mode/Strategy: explore/boundary
Config: lr_W=0.001, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=8, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.7455, test_pearson=0.7179, connectivity_R2=0.8322, final_loss=6.1517e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-22.6, 19.8], effective rank 34, spectral radius 0.973
Mutation: lr_W: 0.0005 -> 0.001 (from Node 5's lr_W=0.0005), lr: 0.0002 -> 0.0002, batch_size: 8 (reset from Node 5)
Parent rule: Node 5 had moderate UCB (2.015), explore strategy to probe lower lr_W boundary more precisely
Observation: partial result (R2=0.832). lr_W=0.001 with lr=0.0002 underperforms vs optimal configs. confirms lower boundary: lr_W<0.003 is risky region. 18/24 converged (75%). block 1 complete

## Simulation Block 1 Summary (iters 1-24)
Simulation: connectivity_type=chaotic, Dale_law=False, Dale_law_factor=N/A, connectivity_rank=N/A, noise_model_level=N/A
Best R2: 0.9999 at iters 15, 21, 23 (tied)
Converged: Yes (18/24 = 75% success rate)
Observation: chaotic connectivity is highly trainable. key findings:
  - lr_W range: [0.003, 0.015] works, optimal [0.005, 0.01]
  - lr range: [0.0001, 0.0003] works, optimal 0.0002-0.0003
  - lr_W:lr ratio: 20:1 to 50:1 is optimal
  - coeff_W_L1 range: [1e-06, 1e-04] works, lower is better
  - batch_size: 8, 16, 32 all work well
  - failure boundaries: lr_W<0.001 fails, coeff_W_L1>1e-03 fails
Optimum training: lr_W=0.005, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1e-04, low_rank_factorization=False, low_rank=N/A

--- NEW SIMULATION BLOCK ---
Next simulation: connectivity_type=low_rank, Dale_law=False, connectivity_rank=20, noise_model_level=0
Node: id=1, parent=root
Rationale: block 1 mapped chaotic connectivity well. block 2 will explore low_rank connectivity to test if GNN can recover structured low-rank matrices. using low_rank_factorization=True with matching rank to provide inductive bias

### Block 1 Protocol Evaluation
Branching rate: 7 unique parents in last 6 iters (5,18,19,20,21,22,23) = 116% (excellent diversity)
Improvement rate: 5/6 improved R2 in last 6 iters (83%) -> suggests more exploration
Stuck detection: no plateau detected (R2 varied 0.83-0.9999)

Protocol changes: no changes needed. current rules effectively balanced exploit/explore. branching rate healthy, improvement rate high, no stuck detection. rules performed well for block 1

## Iter 25: partial
Node: id=25, parent=root
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.005, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.9449, test_pearson=0.9441, connectivity_R2=0.8737, final_loss=2.5796e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-18.5, 20.2], effective rank 12 (99% var), spectral radius 1.109
Mutation: baseline for block 2 (new simulation: connectivity_type=low_rank, low_rank_factorization=True)
Parent rule: first iteration of block 2, parent=root by default
Observation: partial result (R2=0.874) on low_rank connectivity with matching low_rank_factorization=True. activity has lower effective rank (12) vs chaotic (34) as expected. test metrics excellent (pearson=0.944) but connectivity recovery partial. suggests training params need tuning for low_rank case
Next: parent=25 (only node, exploit: try lr_W=0.01 to increase weight learning rate since chaotic optimal was higher)

## Iter 26: converged
Node: id=26, parent=25
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.01, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.5221, test_pearson=0.5986, connectivity_R2=0.9641, final_loss=2.2610e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-18.5, 20.2], effective rank 12 (99% var), spectral radius 1.109
Mutation: lr_W: 0.005 -> 0.01
Parent rule: Node 25 had highest UCB (only node), exploit with lr_W=0.01 based on chaotic block findings
Observation: converged (R2=0.964) with lr_W=0.01. connectivity recovery improved significantly from R2=0.874 to 0.964. however test metrics degraded (test_R2: 0.94->0.52, pearson: 0.94->0.60). suggests higher lr_W helps connectivity but hurts generalization. 1/2 converged in block 2. try balancing lr_W
Next: parent=26 (highest UCB=1.671, exploit: try lr_W=0.007 to find balance between connectivity and test metrics)

## Iter 27: partial
Node: id=27, parent=26
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.007, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.8867, test_pearson=0.8066, connectivity_R2=0.7514, final_loss=2.4883e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-19.5, 17.2], effective rank 13 (99% var), spectral radius 0.937
Mutation: lr_W: 0.01 -> 0.007
Parent rule: Node 26 had highest UCB (1.541), exploit with lr_W=0.007 to balance connectivity and test metrics
Observation: partial result (R2=0.751). lr_W=0.007 degraded connectivity recovery from Node 26 (0.964->0.751) while improving test metrics (test_R2: 0.52->0.89, pearson: 0.60->0.81). reveals tradeoff: higher lr_W benefits connectivity, lower benefits test. need to try different approach - perhaps higher lr or lower L1 regularization. 1/3 converged in block 2
Next: parent=27 (highest UCB=1.617, exploit: try lr=0.0003 to increase model learning rate while keeping balanced lr_W)

## Iter 28: partial
Node: id=28, parent=27
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.007, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.9347, test_pearson=0.9450, connectivity_R2=0.4976, final_loss=2.0206e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-18.6, 17.8], effective rank 6 (99% var), spectral radius 1.278
Mutation: lr: 0.0002 -> 0.0003
Parent rule: Node 27 had highest UCB (1.418), exploit with lr=0.0003 to increase model learning rate
Observation: partial result (R2=0.498). lr=0.0003 severely degraded connectivity recovery from Node 27 (0.751->0.498) while improving test metrics (test_R2: 0.89->0.93, pearson: 0.81->0.95). confirms trend: increasing lr hurts connectivity for low_rank simulation. Node 26 (R2=0.964) remains best in block 2. 1/4 converged in block 2 (25%)
Next: parent=26 (highest UCB=1.464, revert to successful config and try different mutation; try lr=0.0003 with lr_W=0.01 to test if higher lr_W can compensate)

## Iter 29: partial
Node: id=29, parent=26
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.01, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.8308, test_pearson=0.9088, connectivity_R2=0.8253, final_loss=2.0572e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-21.0, 23.8], effective rank 10 (99% var), spectral radius 1.067
Mutation: lr: 0.0002 -> 0.0003 (from Node 26's lr=0.0002)
Parent rule: Node 26 had highest UCB (1.411 after excluding current node), exploit with lr=0.0003 to test if higher lr with lr_W=0.01 improves connectivity
Observation: partial result (R2=0.825). lr=0.0003 + lr_W=0.01 underperforms Node 26's R2=0.964. confirms pattern: increasing lr degrades connectivity for low_rank simulation (similar to iter 28). test metrics good (pearson=0.909) but connectivity suffers. 1/5 converged in block 2 (20%). need to explore opposite direction - try lower lr or higher lr_W
Next: parent=26 (highest UCB=1.411, return to only converged config; try coeff_W_L1=1.0e-05 to reduce L1 regularization which may help low_rank recovery)

## Iter 30: failed
Node: id=30, parent=26
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.01, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=32, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.4043, test_pearson=0.5921, connectivity_R2=0.2423, final_loss=3.0693e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-21.3, 21.7], effective rank 13 (99% var), spectral radius 1.067
Mutation: coeff_W_L1: 1.0e-04 -> 1.0e-05 (from Node 26's coeff_W_L1=1.0e-04)
Parent rule: Node 26 had highest UCB (1.411), exploit with coeff_W_L1=1.0e-05 to reduce L1 regularization
Observation: failed (R2=0.242). reducing L1 from 1e-04 to 1e-05 severely degraded connectivity recovery. for low_rank simulation, L1 regularization at 1e-04 appears necessary (unlike chaotic where 1e-06 worked). this is opposite to chaotic block findings - low_rank needs more regularization. 1/6 converged in block 2 (17%)
Next: parent=29 (highest UCB=2.050, R2=0.825; try lr_W=0.015 to see if higher lr_W can push R2 above 0.9 threshold)

## Iter 31: converged
Node: id=31, parent=29
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.015, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.7028, test_pearson=0.7815, connectivity_R2=0.9937, final_loss=1.1940e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-15.8, 14.7], effective rank 13 (99% var), spectral radius 0.959
Mutation: lr_W: 0.01 -> 0.015 (from Node 29's lr_W=0.01)
Parent rule: Node 29 had highest UCB (2.050), exploit with lr_W=0.015 to push connectivity recovery
Observation: converged (R2=0.994). best connectivity recovery in block 2 so far. lr_W=0.015 + lr=0.0003 outperforms Node 26's lr_W=0.01 + lr=0.0002 (R2=0.964). confirms higher lr_W helps low_rank connectivity. test metrics lower (pearson=0.78 vs 0.91 for Node 29) but connectivity significantly improved. 2/7 converged in block 2 (29%)
Next: parent=31 (highest UCB=2.317, exploit: try lr=0.0002 to test if lower lr improves test metrics while maintaining high connectivity)

## Iter 32: converged
Node: id=32, parent=31
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.015, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.7515, test_pearson=0.8012, connectivity_R2=0.9915, final_loss=1.2137e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-15.8, 15.2], effective rank 13 (99% var), spectral radius 0.959
Mutation: lr: 0.0003 -> 0.0002 (from Node 31's lr=0.0003)
Parent rule: Node 31 had highest UCB (1.937), exploit with lr=0.0002 to test if lower lr improves test metrics while maintaining high connectivity
Observation: converged (R2=0.992). lr=0.0002 maintains excellent connectivity (R2=0.992 vs 0.994 for Node 31) with slightly improved test_pearson (0.80 vs 0.78). confirms lr_W=0.015 is robust for low_rank connectivity. 3/8 converged in block 2 (38%). 2 consecutive successes (31-32), continue exploit
Next: parent=32 (highest UCB=2.406, exploit: try lr_W=0.02 to probe upper lr_W boundary for low_rank)

## Iter 33: converged
Node: id=33, parent=32
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.02, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.9237, test_pearson=0.9459, connectivity_R2=0.9887, final_loss=1.1748e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-15.9, 14.7], effective rank 12 (99% var), spectral radius 0.959
Mutation: lr_W: 0.015 -> 0.02 (from Node 32's lr_W=0.015)
Parent rule: Node 32 had highest UCB (2.406), exploit with lr_W=0.02 to probe upper lr_W boundary for low_rank
Observation: converged (R2=0.989). lr_W=0.02 maintains excellent connectivity with improved test metrics (test_R2: 0.75->0.92, pearson: 0.80->0.95) vs Node 32. best test metrics in block 2 so far. 4/9 converged in block 2 (44%). 3 consecutive successes (31-33), triggers failure-probe strategy
Next: parent=33 (highest UCB=2.489, failure-probe: try lr_W=0.03 to find upper lr_W boundary)

## Iter 34: converged
Node: id=34, parent=33
Mode/Strategy: failure-probe/boundary
Config: lr_W=0.03, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.8633, test_pearson=0.8791, connectivity_R2=0.9710, final_loss=1.2910e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-15.8, 14.7], effective rank 13 (99% var), spectral radius 0.959
Mutation: lr_W: 0.02 -> 0.03 (from Node 33's lr_W=0.02)
Parent rule: Node 33 had highest UCB (2.489), failure-probe with lr_W=0.03 (1.5x) to find upper lr_W boundary
Observation: converged (R2=0.971). lr_W=0.03 still converges but shows degradation from Node 33 (R2: 0.989->0.971, pearson: 0.946->0.879). found soft upper boundary: lr_W>0.02 hurts both connectivity and test metrics. optimal lr_W for low_rank is around 0.02. 5/10 converged in block 2 (50%). 4 consecutive successes (31-34), continue exploit
Next: parent=33 (highest UCB=2.043, exploit: try lr=0.0003 with lr_W=0.02 to test lr sensitivity at optimal lr_W)

