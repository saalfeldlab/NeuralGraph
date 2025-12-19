# Experiment Log: signal_Claude

## Iter 1: converged
--- NEW SIMULATION BLOCK ---
Simulation: connectivity_type=chaotic, Dale_law=False, noise_model_level=0
Node: id=1, parent=root
Mode/Strategy: exploit
Config: lr_W=2.0E-3, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9987, test_pearson=0.9988, connectivity_R2=0.9989, final_loss=6.1873e+03
Activity: chaotic dynamics with effective rank 34 (99% var), range [-23.6, 23.5], good signal complexity
Mutation: baseline config (first iteration)
Parent rule: UCB file shows only node 1, selecting parent=1 for next iteration
Observation: excellent baseline performance - chaotic connectivity with default training params converges immediately
Next: parent=1

## Iter 2: converged
Node: id=2, parent=1
Mode/Strategy: exploit
Config: lr_W=4.0E-3, lr=1.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9136, test_pearson=0.8745, connectivity_R2=0.9987, final_loss=5.0314e+03
Activity: chaotic dynamics with effective rank 11-34, range [-19.6, 24.6], spectral radius 1.025
Mutation: lr_W: 2.0E-3 -> 4.0E-3
Parent rule: node 2 has highest UCB (1.706 vs 1.470), continue exploiting this branch
Observation: doubled lr_W maintains excellent connectivity_R2 but test_R2 dropped (0.9987->0.9136) - higher lr_W may cause overfitting to connectivity at cost of dynamics prediction
Next: parent=2

## Iter 3: converged
Node: id=3, parent=2
Mode/Strategy: failure-probe
Config: lr_W=4.0E-3, lr=2.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9387, test_pearson=0.9168, connectivity_R2=0.9999, final_loss=3.3992e+03
Activity: chaotic dynamics with effective rank 35 (99% var), range [-23.8, 17.8], spectral radius 1.025
Mutation: lr: 1.0E-4 -> 2.0E-4
Parent rule: node 3 has highest UCB (1.866); 3 consecutive successes triggers failure-probe strategy
Observation: doubling lr improved both test_R2 (0.9136->0.9387) and connectivity_R2 (0.9987->0.9999) with lower loss - this is the best config so far; now probing failure boundary with extreme lr_W
Next: parent=3

## Iter 4: converged
Node: id=4, parent=3
Mode/Strategy: failure-probe
Config: lr_W=4.0E-2, lr=2.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9772, test_pearson=0.9623, connectivity_R2=0.9981, final_loss=1.9010e+03
Activity: chaotic dynamics with effective rank 13-37, range [-19.6, 20.6], spectral radius 1.025
Mutation: lr_W: 4.0E-3 -> 4.0E-2 (10x increase for failure probe)
Parent rule: node 4 has highest UCB (1.998); 4 consecutive successes, continuing failure-probe to find boundary
Observation: 10x lr_W increase still converges with improved test_R2 (0.9387->0.9772) and much lower loss (3.4e+03->1.9e+03); connectivity_R2 slightly decreased but still excellent (0.9999->0.9981); probing higher lr_W to find failure boundary
Next: parent=4

## Iter 5: converged
Node: id=5, parent=4
Mode/Strategy: failure-probe
Config: lr_W=1.0E-1, lr=2.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9660, test_pearson=0.9535, connectivity_R2=0.9938, final_loss=2.2571e+03
Activity: chaotic dynamics with effective rank 12-35, range [-19.6, 19.3], spectral radius 1.025
Mutation: lr_W: 4.0E-2 -> 1.0E-1 (2.5x increase for failure probe)
Parent rule: node 5 has highest UCB (2.112); 5 consecutive successes, continuing failure-probe to find boundary
Observation: lr_W=0.1 still converges but shows slight regression - test_R2 dropped (0.9772->0.9660), connectivity_R2 dropped (0.9981->0.9938), loss increased (1.9e+03->2.3e+03); approaching failure boundary for lr_W; next probe lr_W=0.2 to confirm boundary
Next: parent=5

## Iter 6: converged
Node: id=6, parent=5
Mode/Strategy: failure-probe
Config: lr_W=2.0E-1, lr=2.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.6357, test_pearson=0.5863, connectivity_R2=0.9832, final_loss=2.7099e+03
Activity: chaotic dynamics with effective rank 11-34, range [-20.0, 20.4], spectral radius 1.025
Mutation: lr_W: 1.0E-1 -> 2.0E-1 (2x increase for failure probe)
Parent rule: node 6 has highest UCB (2.208); 6 consecutive successes triggers explore strategy
Observation: found failure boundary - lr_W=0.2 causes severe test_R2 degradation (0.9660->0.6357) while connectivity_R2 remains acceptable (0.9832); optimal lr_W region is 4.0E-2 to 1.0E-1; explore different dimension from node 4
Next: parent=4

## Iter 7: converged
Node: id=7, parent=4
Mode/Strategy: explore
Config: lr_W=4.0E-2, lr=4.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=8
Metrics: test_R2=0.9155, test_pearson=0.8806, connectivity_R2=0.9818, final_loss=1.8257e+03
Activity: chaotic dynamics with effective rank 11-33, range [-19.6, 21.4], spectral radius 1.025
Mutation: lr: 2.0E-4 -> 4.0E-4 (2x increase exploring lr dimension from node 4)
Parent rule: node 6 has highest UCB (2.306) but explore strategy selects node 4 (best config) to explore new dimension
Observation: doubling lr from node 4 baseline slightly decreased connectivity_R2 (0.998->0.982) and test_R2 (0.977->0.916); lr=2.0E-4 appears optimal; now explore batch_size or coeff_W_L1 from node 4
Next: parent=4

## Iter 8: converged
Node: id=8, parent=4
Mode/Strategy: explore
Config: lr_W=4.0E-2, lr=2.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=16
Metrics: test_R2=0.9845, test_pearson=0.9779, connectivity_R2=0.9912, final_loss=1.5116e+03
Activity: chaotic dynamics with effective rank 10-33, range [-19.6, 18.9], spectral radius 1.025
Mutation: batch_size: 8 -> 16 (2x increase exploring batch_size dimension from node 4)
Parent rule: node 8 has highest UCB (2.405); 8 consecutive successes, explore strategy - continue exploring new dimensions from best config
Observation: batch_size increase significantly improved all metrics - connectivity_R2 (0.998->0.991), test_R2 (0.977->0.985), loss (1.9e+03->1.5e+03); batch_size=16 is beneficial; explore coeff_W_L1 dimension next
Next: parent=8

## Iter 9: converged
Node: id=9, parent=8
Mode/Strategy: explore
Config: lr_W=4.0E-2, lr=2.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-4, batch_size=16
Metrics: test_R2=0.9157, test_pearson=0.8925, connectivity_R2=0.9827, final_loss=2.3136e+03
Activity: chaotic dynamics with effective rank 12-36, range [-20.0, 18.7], spectral radius 1.025
Mutation: coeff_W_L1: 1.0E-5 -> 1.0E-4 (10x increase exploring L1 regularization)
Parent rule: node 9 tied with node 6 for highest UCB (2.483); 9 consecutive successes, explore strategy
Observation: increased L1 regularization hurt performance - connectivity_R2 dropped (0.9912->0.9827), test_R2 dropped (0.9845->0.9157), loss increased (1.5e+03->2.3e+03); coeff_W_L1=1.0E-5 is better; try batch_size=32 from node 8
Next: parent=8

## Iter 10: converged
Node: id=10, parent=8
Mode/Strategy: explore
Config: lr_W=4.0E-2, lr=2.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=32
Metrics: test_R2=0.9830, test_pearson=0.9726, connectivity_R2=0.9994, final_loss=1.8448e+03
Activity: chaotic dynamics with effective rank 11-34, range [-22.1, 20.4], spectral radius 1.025
Mutation: batch_size: 16 -> 32 (2x increase exploring batch_size dimension from node 8)
Parent rule: node 10 has highest UCB (2.581); 10 consecutive successes, explore strategy
Observation: batch_size=32 achieved best connectivity_R2 (0.9994) but slightly lower test_R2 than batch_size=16 (0.9830 vs 0.9845); optimal batch_size appears to be 16-32; now explore lr_emb dimension from node 8 (best balanced config)
Next: parent=8

## Iter 11: converged
Node: id=11, parent=8
Mode/Strategy: explore
Config: lr_W=4.0E-2, lr=2.0E-4, lr_emb=5.0E-4, coeff_W_L1=1.0E-5, batch_size=16
Metrics: test_R2=0.9166, test_pearson=0.8691, connectivity_R2=0.9988, final_loss=1.5394e+03
Activity: chaotic dynamics with effective rank 12-34, range [-19.6, 20.5], spectral radius 1.025
Mutation: lr_emb: 2.5E-4 -> 5.0E-4 (2x increase exploring lr_emb dimension from node 8)
Parent rule: node 10 has highest UCB (2.658); 11 consecutive successes, explore strategy
Observation: doubled lr_emb hurt test_R2 (0.9845->0.9166) and test_pearson (0.978->0.869) while connectivity_R2 remained excellent (0.9988); lr_emb=2.5E-4 is optimal; explore lr dimension from node 10 (highest UCB with best connectivity_R2)
Next: parent=10

## Iter 12: converged
Node: id=12, parent=10
Mode/Strategy: explore
Config: lr_W=4.0E-2, lr=4.0E-4, lr_emb=2.5E-4, coeff_W_L1=1.0E-5, batch_size=32
Metrics: test_R2=0.9490, test_pearson=0.9249, connectivity_R2=0.9991, final_loss=1.5413e+03
Activity: chaotic dynamics with effective rank 12-34, range [-19.6, 20.8], spectral radius 1.025
Mutation: lr: 2.0E-4 -> 4.0E-4 (2x increase exploring lr dimension from node 10)
Parent rule: node 12 ties for highest UCB (2.731); 12 consecutive successes, explore strategy
Observation: doubled lr from node 10 slightly decreased test_R2 (0.9830->0.9490) and test_pearson (0.973->0.925) while maintaining excellent connectivity_R2 (0.9994->0.9991); lr=2.0E-4 remains optimal; explore lr_W dimension from node 12 with smaller step
Next: parent=12

