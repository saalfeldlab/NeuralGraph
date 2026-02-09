# Working Memory: signal_landscape (parallel)

## Knowledge Base (accumulated across all blocks)

### Regime Comparison Table
| Block | Regime | E/I | n_frames | n_neurons | n_types | noise | eff_rank | Best R² | Optimal lr_W | Optimal L1 | Degeneracy | Key finding |
| ----- | ------ | --- | -------- | --------- | ------- | ----- | -------- | ------- | ------------ | ---------- | ---------- | ----------- |
| 1 | chaotic | - | 10k | 100 | 1 | 0 | 35 | 1.000 | 4E-3 | 1E-5 | 0/12 | baseline easy; lr_W=4E-3 sweet spot; lr=1E-4 optimal |
| 2 | low_rank (r=20) | - | 10k | 100 | 1 | 0 | 12-13 | 0.993 | 3E-3 | 1E-6 | 1/12 | L1=1E-6 critical for dynamics; factorization hurts |
| 3 | chaotic+Dale | 50/50 | 10k | 100 | 1 | 0 | 12 | 0.986 | 4.5E-3 | 1E-6 | 4/12 | Dale reduces eff_rank 35->12; lr_W cliff at 5E-3 |
| 4 | chaotic+4types | - | 10k | 100 | 4 | 0 | 38 | 0.992 | 5E-3 | 1E-6 | 0/12 | dual-objective best at lr_W=5E-3; L1=1E-6 critical for embedding |
| 5 | chaotic+noise | - | 10k | 100 | 1 | 0.1-1.0 | 42-90 | 1.000 | 2-4E-3 | 1E-5 | 0/12 | 100% convergence; noise inflates eff_rank; inverse lr_W-noise |
| 6 | chaotic n=200 | - | 10k | 200 | 1 | 0 | 41-44 | 0.956 | 5E-3 | 1E-5 | 0/12 | harder than n=100 (67% vs 92%); convergence boundary ~2x higher |
| 7 | sparse 50% | - | 10k | 100 | 1 | 0 | 21 | 0.466 | 1E-2 | 1E-5 | **12/12** | universal degeneracy; subcritical rho=0.746; gap 0.53-0.82 |
| 8 | sparse 50%+noise | - | 10k | 100 | 1 | 0.5 | 91 | 0.490 | any | 1E-5 | 0/12 | NOT degenerate (pearson low too); structural data limit |
| 9 | chaotic n=300 | - | 10k | 300 | 1 | 0 | 44-47 | 0.890 | 1E-2 | 1E-5 | 2/12 | mild training-limited degeneracy; n_epochs=2 key |
| 10 | chaotic n=300 (v2) | - | 10k | 300 | 1 | 0 | 44-47 | 0.924 | 1E-2 | 1E-6 | 0/12 | more epochs resolved degeneracy; 25% conv rate |
| 11 | chaotic n=200 (v2) | - | 10k | 200 | 1 | 0 | 40-43 | 0.994 | 8E-3 | 1E-5 | 0/12 | 100% conv (12/12); lr_W=8E-3 optimal |
| 12 | chaotic n=600 | - | 10k | 600 | 1 | 0 | 50 | 0.626 | 1E-2 | 1E-5 | 0/12 | NOT degenerate (underfitting); training-capacity-limited |
| 13 | chaotic n=200+4types | - | 10k | 200 | 4 | 0 | 42-44 | 0.988 | 8E-3 | 1E-5 | 0/12 | FULL dual convergence; L1=1E-5 > L1=1E-6 at n=200 |
| 14 | chaotic n=200+recurrent | - | 10k | 200 | 1 | 0 | 42-44 | 0.993 | 8E-3 | 1E-5 | 0/4 | recurrent boosts conn+0.3% but dynamics-12.3%; 8/12 infra failures |
| 15 | **chaotic n=300 (30k)** | - | **30k** | 300 | 1 | 0 | **79-80** | **1.000** | 3E-3 | 1E-5 | **0/12** | **100% conv (12/12); n_frames transformative; all params non-critical** |
| 16 | **chaotic n=600 (30k)** | - | **30k** | 600 | 1 | 0 | **87** | **0.993** | 5E-3 | 1E-5 | **0/12** | **100% conv (12/12); 30k transforms n=600 (0%→100%); lr=1E-4 NOT catastrophic** |
| 17 | **sparse 50% (30k)** | - | **30k** | 100 | 1 | 0 | **13** | **0.436** | 5E-3 | 1E-5 | **12/12** | **0% conv; n_frames NOT helpful; eff_rank=13<21 at 10k; subcritical rho immune to data** |
| 18 | **chaotic n=1000 (30k)** | - | **30k** | 1000 | 1 | 0 | **144** | **0.745** | 1E-2 | 1E-5 | **0/12** | **0% conv; 30k insufficient (needs ~100k); lr=1E-4 Pareto-better; 8ep optimal at lr=1E-4** |
| 19 | **chaotic g=3** | - | 10k | 100 | 1 | 0 | **26** | **0.955** | 8E-3 | 1E-5 | **4/12** | **gain=3 reduces eff_rank 35→26; 42% conv; 2ep minimum; no lr_W cliff to 1.2E-2; training-limited like large n** |
| 20 | **chaotic g=3 n=200** | - | 10k | 200 | 1 | 0 | **31** | **0.489** | 1.2E-2 | 1E-5 | **12/12** | **0% conv; gain × n compounds severely; 6ep best; no lr_W cliff to 2E-2; batch=16 catastrophic (-21%); needs 30k frames** |
| 21 | **chaotic g=3 n=200 (30k)** | - | **30k** | 200 | 1 | 0 | **53-57** | **0.996** | 4E-3 | 1E-5 | **0/12** | **100% conv (12/12); 30k rescues g=3/n=200 (0%→100%); eff_rank +80%; no cliff to 3E-2; ALL params non-critical; batch=16 safe** |
| 22 | **chaotic fill=80%** | - | 10k | 100 | 1 | 0 | **36** | **0.802** | 8E-3 | 1E-5 | **0/12** | **0% conv; conn plateau 0.802; rho=0.985 near-critical; eff_rank=36 (same as 100%); complete param insensitivity; sharp transition from 50% (rho 0.746→0.985)** |
| 23 | **chaotic fill=80% (30k)** | - | **30k** | 100 | 1 | 0 | **48-49** | **0.802** | any | any | **0/12** | **0% conv; n_frames did NOT break plateau; conn=0.802 IDENTICAL to 10k; ABSOLUTE param insensitivity (12/12 at 0.802); conn_ceiling ≈ fill% is STRUCTURAL invariant** |
| 24 | **chaotic fill=90%** | - | 10k | 100 | 1 | 0 | **35-36** | **0.907** | any [3E-3, 1.5E-2] | any | **0/12** | **83% conv (10/12); conn plateau at 0.907≈fill%; rho=0.995; ABSOLUTE param insensitivity; fill=90% is TRANSITIONAL (right at R2>0.9 boundary)** |
| 25 | **chaotic g=1** | - | 10k | 100 | 1 | 0 | **5** | **0.007** | any [1E-3, 3E-2] | any | **12/12** | **0% conv; FIXED-POINT COLLAPSE; eff_rank=5; rho=1.065 (supercritical); conn~0.000; COMPLETE param insensitivity; two-phase ONLY marginal signal (0.007); MORE severe than sparse; hardest regime tested** |
| 26 | **chaotic g=1 (30k)** | - | **30k** | 100 | 1 | 0 | **1** | **0.018** | any [4E-3, 2E-2] | any | **12/12** | **0% conv; eff_rank DROPPED 5→1 at 30k; WORSE than 10k; n_frames IMMUNE; edge_diff=500 harmful; two-phase marginal (0.009); g=1 CONFIRMED UNSOLVABLE by n_frames** |
| 27 | **chaotic g=2** | - | 10k | 100 | 1 | 0 | **17** | **0.519** | **5E-4** | 1E-5 | **12/12** | **0% conv; INVERSE lr_W (5E-4 optimal, 100x lower than g=7); epoch scaling NOT diminishing; lr_W=5E-4/5ep ≈ 1E-3/12ep; eff_rank=17 between g=1(5) and g=3(26); needs 30k** |
| 28 | **chaotic g=2 (30k)** | - | **30k** | 100 | 1 | 0 | **16** | **0.997** | **5E-4** | 1E-5 | **5/12** | **42% conv; eff_rank=16 UNCHANGED from 10k; inverse lr_W PERSISTS (5E-4-7E-4; >=2E-3 catastrophic); epoch scaling NOT diminishing (8ep→0.997); Pareto: 5E-4/8ep** |
| 29 | **chaotic g=2 n=200 (30k)** | - | **30k** | 200 | 1 | 0 | **35-38** | **0.979** | **3E-4** | 1E-5 | **0/12** | **92% conv; eff_rank 35-38 (CONTRADICTS n=100's 16); inverse lr_W persists (optimal 3E-4, ceiling 1E-3); epoch scaling diminishing at 12ep; batch=16 safe** |
| 30 | **chaotic n=1000 (100k)** | - | **100k** | 1000 | 1 | 0 | **high** | **1.000** | **3E-3** | 1E-5 | **0/12** | **BREAKTHROUGH: 100% conv (12/12); 100k transforms n=1000; lr_W=3E-3 + batch=16/3ep Pareto-optimal (conn=1.000, test_R2=0.882, 304 min); lr_W×epoch interaction: low ep→low lr_W best; high ep→moderate lr_W best** |

### Established Principles
1. **lr tolerance scales with network size, eff_rank, AND n_frames**: lr=1E-4 optimal at n=100 eff_rank=35; lr=2E-4 safe at eff_rank>=42; lr=1E-4 CATASTROPHIC at n=600/10k (conn=0.000) BUT NOT at n=600/30k (conn=0.993, iter 192); n_frames rescues lr=1E-4 at large n. (evidence: blocks 1, 3, 5, 6, 9, 10, 12, 16)
2. **connectivity convergence boundary scales sub-linearly with n_neurons**: n=100→~1.5E-3, n=200→~3.5E-3, n=300→~7E-3, n=600→~3-4E-3 (NOT linear); lr_W=6E-3 gives 0.588 at n=600 (only -6% vs optimal 1E-2). (evidence: blocks 1, 6, 9, 12)
3. **L1=1E-6 effect is n-dependent and NON-MONOTONIC — overrides heterogeneous rule at n>=200**: HARMFUL at n<=200 (both n_types=1 and n_types=4); BENEFICIAL at n=300/10k/n_types=1; HARMFUL at n>=600; at n=100/4types L1=1E-6 critical, but at n=200/4types L1=1E-5 is BETTER. BUT at n=300/30k L1 is IRRELEVANT (both work). (evidence: blocks 2-13, 15)
4. **factorization=True hurts in low_rank regime**: direct W learning outperforms factorized W=W_L@W_R. (evidence: block 2)
5. **optimal lr_W depends on n_neurons, eff_rank, constraints, noise, n_types, sparsity, AND n_frames**: n=100 chaotic 4E-3; n=200 chaotic 8E-3; n=300/10k chaotic 1E-2; n=300/30k chaotic 3E-3; n=600/10k chaotic 1E-2; **n=600/30k chaotic 5E-3**; low_rank 3E-3; Dale 4-4.5E-3; heterogeneous 5E-3; noisy: inverse with noise level. At high n_frames, dynamics-optimal lr_W shifts LOWER while conn remains safe over wide range. (evidence: blocks 1-16)
6. **Dale_law creates sharp lr_W cliff at ~5E-3**: safe range [3.5E-3, 4.5E-3]. (evidence: block 3)
7. **Dale_law reduces eff_rank from 35 to 12**: E/I constraint concentrates variance. (evidence: block 3)
8. **batch_size=16 is detrimental at LOW n_frames**: at 10k, batch=16 hurts heterogeneous, Dale, and n>=300; BUT at 30k, batch=16 is SAFE even at n=300 (conn=0.999-1.000, iters 176, 178); n_frames determines batch sensitivity. (evidence: blocks 2-4, 8, 10, 11, 15)
9. **lr_emb coupled to lr_W for heterogeneous**: lr_emb/lr_W ratio ~0.2 safe; lr_emb=1E-3 at lr_W>=4E-3. (evidence: block 4)
10. **lr_W=5E-3 optimal for dual-objective in heterogeneous chaotic at n=100**. (evidence: block 4)
11. **heterogeneous networks increase eff_rank**: n_types=4 raises 35->38. (evidence: block 4)
12. **noise inflates eff_rank but only rescues dense connectivity, NOT sparse**: dense eff_rank 35→42-90 with 100% convergence; sparse eff_rank 21→91 but only +5% conn (0.466→0.489). (evidence: blocks 5, 8)
13. **inverse lr_W-noise relationship for dynamics**: higher noise needs lower lr_W. (evidence: block 5)
14. **rollout quality anti-correlates with noise**: noise=0.1 best rollout (kino_R2=0.405). (evidence: block 5)
15. **n scaling: eff_rank grows slowly with n at fixed n_frames, but DOUBLES with 3x n_frames**: n=100→35, n=200→43, n=300/10k→47, n=300/30k→80, n=600→50; at fixed n_frames ~log scaling; increasing n_frames is the dominant lever. (evidence: blocks 1, 6, 9, 12, 15)
16. **dynamics cliff depends on n, n_epochs, AND n_frames**: at 10k/1ep: n=100→8E-3, n=200→5.5E-3; at 10k/2ep: n=200 cliff>1.2E-2; at 30k: n=300 no cliff up to 2E-2. More data widens safe lr_W range even more than more epochs. (evidence: blocks 1, 6, 9, 11, 15)
17. **convergence rate depends on n AND n_frames (more than n_epochs)**: n=100/10k/1ep: 92%; n=200/10k/2ep: 100%; n=300/10k/3-4ep: 25%; **n=300/30k/1ep: 100%**; n=600/10k/10ep: 0%. n_frames >> n_epochs for convergence. (evidence: blocks 1, 6, 9, 10, 11, 12, 15)
18. **sparse connectivity drastically reduces eff_rank and makes dynamics subcritical**: 50% fill → eff_rank 35→21, spectral_radius 1.03→0.746; 0% convergence at 10k frames/2 epochs. (evidence: block 7)
19. **n_epochs is dominant in sparse-without-noise but irrelevant in sparse+noise**. (evidence: blocks 7, 8)
20. **sparse regime has no lr_W cliff up to 1.5E-2**: monotonic improvement in no-noise; complete insensitivity in noise. (evidence: blocks 7, 8)
21. **recurrent training catastrophic in noisy subcritical regime**: time_step=4 collapsed connectivity 0.489→0.054. (evidence: block 8)
22. **sparse 50% conn ~0.49 is a structural data limit at 10k frames**. (evidence: blocks 7-8)
23. **n=300 training param requirements are n_frames-dependent**: at 10k frames requires 3ep AND L1=1E-6; at 30k frames even 1ep converges and L1 irrelevant. n_frames dominates over all training params. (evidence: blocks 9-10, 15)
24. **lr tolerance narrows at high lr_W (at low n_frames)**: lr=3E-4 degrades at lr_W=1E-2 at n=300/10k; lr=2E-4 safe everywhere; at n=300/30k lr=3E-4 preserves dynamics but damages cluster_accuracy. (evidence: blocks 9, 15)
25. **n_epochs has diminishing returns — depends on gain AND lr_W, not just n**: n=300/10k diminishing; n=600/10k NOT diminishing; g=2/lr_W=1E-3 NOT diminishing (5ep→0.356, 12ep→0.519); low gain + low lr_W preserves epoch effectiveness. REVISED from "small n" rule. (evidence: blocks 10, 12, 27)
26. **lr_W=1.1E-2 is neutral at n=300/10k**: sweet spot exactly 1E-2 at 10k. (evidence: block 10)
27. **n=200 recipe: lr_W=8E-3, lr=2E-4, L1=1E-5, n_epochs=2-3**: 100% convergence. (evidence: block 11)
28. **n_epochs extends safe lr_W range**: more training smooths loss landscape. (evidence: blocks 6, 11)
29. **n=600 is severely training-capacity-limited at 10k frames**: 10ep best conn=0.626; gains ~4-8% per +2ep; lr=1E-4 catastrophic; lr_W=1E-2 optimal; L1=1E-5 better than 1E-6. (evidence: block 12)
30. **n=200/4types recipe: lr_W=8E-3, lr=2E-4, lr_emb=1E-3, L1=1E-5, batch=8, 3ep**: achieves full dual convergence (conn=0.988, cluster=1.000). (evidence: block 13)
31. **lr_emb ceiling at n=200/4types is 1E-3**: lr_emb=2E-3 overshoots; lr_emb/lr_W ratio must be <=0.125. (evidence: block 13)
32. **heterogeneous lr_W optimal scales with n like homogeneous**: n=100/4types→5E-3, n=200/4types→8E-3. (evidence: blocks 4, 13)
33. **non-monotonic lr_W at n=200/4types**: 8E-3 best, 1E-2 dip, 1.2E-2 partial recovery. (evidence: block 13)
34. **2ep sufficient for W-convergence at n=200/4types but 3ep needed for full dual**. (evidence: block 13)
35. **recurrent training (time_step=4) at supercritical rho creates conn-dynamics trade-off**: conn +0.3% but dynamics -12.3%. (evidence: block 14)
36. **recurrent warmup (start_ep>=1) shifts capacity from W to MLP**: dynamics +9.5% but conn -8.2%. (evidence: block 14)
37. **noise_recurrent_level ceiling is 0.01**: noise_rec=0.05 degrades conn to partial. (evidence: block 14)
38. **recurrent training is NOT catastrophic at supercritical rho>=1**: requires subcritical rho AND noise for catastrophe. (evidence: blocks 8, 14)
39. **n_frames is the DOMINANT lever for connectivity recovery at large n**: at n=300, 3x n_frames (10k→30k) boosts convergence rate 25%→100% and best conn 0.924→1.000; ALL training params become non-critical; eff_rank doubles (47→80). (evidence: block 15)
40. **at high n_frames, dynamics-optimal lr_W is LOWER than conn-optimal lr_W**: at n=300/30k, conn converges at lr_W=3E-3 to 2E-2 (insensitive), but dynamics best at lr_W=3E-3 (test_R2=0.990) vs lr_W=2E-2 (0.944). Lower lr_W gives MLP more capacity for dynamics when data abundance handles W. (evidence: block 15)
41. **aug_loop=20 preserves connectivity but costs ~6% dynamics at 30k frames**: conn=1.000 at both aug=20 and aug=40; training time 16 vs 42 min; use aug=40 for quality, aug=20 for speed. (evidence: block 15, iter 179)
42. **n=600/30k recipe: lr_W=5E-3, lr=2E-4, L1=1E-5, batch=16, 5ep**: conn=0.993, test_R2=0.966, kino_R2=0.964 (iter 189); lr_W=5E-3 Pareto-optimal at n=600/30k for BOTH conn AND dynamics (lower lr_W than 10k's 1E-2). (evidence: block 16)
43. **n_frames rescues ALL parameter catastrophes at large n — EXCEPT fill<1 connectivity**: lr=1E-4 catastrophic at n=600/10k but conn=0.993 at 30k; L1 sensitivity vanishes at 30k; epoch requirements drop 10ep→2ep; lr_W cliff eliminated. **EXCEPTION 1**: sparse 50% at n=100/30k: conn max 0.436 (subcritical rho immune to n_frames). **EXCEPTION 2**: fill=80% at n=100/30k: conn=0.802 IDENTICAL to 10k (conn_ceiling ≈ fill% is structural invariant); n_frames fails for ALL fill<1. (evidence: blocks 15, 16, 17, **23**)
44. **dynamics-optimal lr_W inversely scales with n_frames at fixed n_neurons**: n=600/10k→1E-2, n=600/30k→5E-3; n=300/10k→1E-2, n=300/30k→3E-3; more data means MLP has more gradient signal, so lower lr_W suffices and avoids overshooting. (evidence: blocks 15, 16)
45. **sparse 50% eff_rank is LOWER at 30k than at 10k (13 vs 21)**: eff_rank is determined by W structure (spectral radius), NOT data volume; subcritical rho=0.746 constrains the effective dimensionality regardless of n_frames. (evidence: block 17)
46. **sparse 50% at n=100 is structurally limited — complete parameter insensitivity**: conn range [0.213, 0.436] across 12 iters with ALL training params varied; two-phase + more epochs is the ONLY marginal improvement (+15%); subcritical spectral_radius (rho=0.746) is the true barrier. (evidence: blocks 7, 8, 17)
47. **two-phase training is the only positive signal in sparse regime**: n_epochs_init=2, first_coeff_L1=0, coeff_lin_phi_zero=1.0 gives +0.029 at 3ep and +0.057 at 5ep over non-two-phase; still insufficient for convergence. (evidence: block 17)
48. **n=1000/30k is insufficient for convergence — max conn=0.745**: 0% convergence (12/12 partial); user prior confirmed (needs ~100k frames); eff_rank=144; lr_W=1E-2 optimal; two-phase training used throughout. (evidence: block 18)
49. **lr=1E-4 is definitively Pareto-better at n=1000/30k**: at 8ep, lr=1E-4 gives conn=0.745 + test_R2=0.829 vs lr=2E-4's conn=0.734 + test_R2=0.588 at 5ep; lr=2E-4 OVERSHOOTS at 10ep (conn DECREASES to 0.716); lr=1E-4 required for high-epoch training at large n. (evidence: block 18)
50. **eff_rank scales superlinearly with n_neurons at 30k frames**: n=300→80, n=600→87, n=1000→144; the jump n=600→n=1000 is +65% while neurons increase +67%. (evidence: blocks 15, 16, 18)
51. **epoch scaling at n=1000/30k is lr-dependent**: at lr=1E-4, steady improvement 3ep→5ep→8ep (0.666→0.726→0.745); at lr=2E-4, REVERSAL at 10ep (0.734→0.716 = overtraining). Higher lr amplifies overtraining risk at large n. (evidence: block 18)
52. **dynamics stochastic variance increases with n**: at n=1000, identical configs give test_R2 range 0.725-0.829 (~14% spread); conn is reproducible (0.743 vs 0.745). (evidence: block 18)
53. **low gain (g=3) is an independent difficulty axis**: g=3 reduces eff_rank 35→26 (-26%) at n=100 while spectral_radius stays supercritical (1.065); universal degeneracy at 1ep (4/4, gaps 0.35-0.75); 2ep resolves degeneracy; 3ep optimal (conn=0.955); no lr_W cliff up to 1.2E-2 (unlike g=7 cliff at 8E-3); g=3/n=100 at 1ep ≈ g=7/n=600 in difficulty; batch=16 catastrophic (-42%); L1=1E-6 harmful. (evidence: block 19)
54. **gain modulates lr_W cliff position**: g=7/n=100 cliff at ~8E-3; g=3/n=100 no cliff up to 1.2E-2; g=3/n=200 no cliff up to 2E-2; lower gain shifts optimal lr_W higher and eliminates cliff — weaker interactions need more aggressive W learning. (evidence: blocks 1, 19, 20)
55. **g=3 recipe: lr_W=8E-3, lr=1E-4, L1=1E-5, batch=8, 3ep → conn=0.955**: at n=100/10k; n_epochs is dominant lever (1ep→0.636, 2ep→0.906, 3ep→0.955). (evidence: block 19)
56. **gain × n_neurons compounds difficulty severely**: g=3/n=200/10k max conn=0.489 at 6ep (0% conv) vs g=3/n=100/10k 0.955 at 3ep (42% conv) and g=7/n=200/10k 0.956 at 2ep (100% conv); eff_rank=31; universal degeneracy (12/12); epoch scaling diminishing at 4-6ep; coeff_edge_diff=500 marginal; batch=16 catastrophic (-21%); likely needs 30k frames. (evidence: block 20)
57. **g=3 eliminates lr_W cliff across n_neurons**: no cliff at n=100 up to 1.2E-2 and at n=200 up to 2E-2; lr_W and epochs are substitutable (lr_W=2E-2/3ep ≈ lr_W=1.2E-2/4ep); but lr_W saturates — epochs more effective for conn improvement. (evidence: blocks 19, 20)
58. **batch=16 catastrophic at low gain AT 10k only**: g=3/n=100/10k -42% (iter 224), g=3/n=200/10k -21% (iter 240); BUT at 30k batch=16 is SAFE (-0.4%, iter 251); n_frames overrides batch sensitivity at low gain. (evidence: blocks 19, 20, 21)
59. **30k frames rescues g=3/n=200 — gain is SOLVABLE by n_frames**: 0% conv at 10k → 100% at 30k (12/12); eff_rank 31→53-57 (+80%); Pareto: lr_W=4E-3, lr=2E-4, 2ep → conn=0.996, test_R2=0.999; ALL params non-critical; no lr_W cliff to 3E-2; confirms gain is NOT an independent unsolvable axis. (evidence: block 21)
60. **g=3/30k lr tolerance wider than g=7/30k**: lr=3E-4 safe at g=3/n=200/30k (conn=0.993, cluster=0.985) but damages cluster at g=7/n=300/30k (0.567, iter 180); lower gain widens parameter tolerance. (evidence: blocks 15, 21)
61. **g=3/n=200/30k recipe: lr_W=4E-3, lr=2E-4, L1=1E-5, batch=8, 2ep**: conn=0.996, test_R2=0.999, kino_R2=0.999 (iter 245); dynamics-optimal lr_W=3.5-4E-3 confirms inversely scaling with n_frames. (evidence: block 21)
62. **fill=80% creates conn plateau at ~0.802 at n=100/10k**: eff_rank=36 (same as 100% fill), rho=0.985 (near-critical); COMPLETE parameter insensitivity (lr_W 4E-3-2E-2, epochs 1-5, lr 1E-4-2E-4, L1 1E-5-1E-6); 0/12 degenerate; dynamics improve but conn stuck; NOT like sparse 50% (no degeneracy, near-critical). (evidence: block 22)
63. **filling_factor transition from 50% to 80% is SHARP**: rho 0.746→0.985; eff_rank 21→36; conn ceiling 0.49→0.80; critical filling boundary between 50-80% separates subcritical (unsolvable at 10k) from near-critical (structurally limited but potentially solvable with n_frames). (evidence: blocks 7, 17, 22)
64. **conn ceiling scales approximately linearly with filling_factor at 10k**: fill=50%→conn~0.49, fill=80%→conn~0.80, fill=100%→conn~1.00; conn_ceiling ≈ filling_factor at n=100/10k. (evidence: blocks 1, 7, 22)
65. **conn_ceiling ≈ filling_factor is a STRUCTURAL invariant across n_frames**: fill=80% at 10k → conn=0.802; fill=80% at 30k → conn=0.802 (IDENTICAL); 12/12 partial at 30k with ABSOLUTE parameter insensitivity (lr_W 2E-3-2E-2, epochs 1-5, L1 1E-6-1E-4, batch 8-16, two-phase, edge_diff 100-500); n_frames rescues dynamics/eff_rank but NOT structural conn limit; this extends to ALL fill<1 (including sparse 50%). (evidence: blocks 22, 23)
66. **fill=90% is TRANSITIONAL regime — conn_ceiling at convergence boundary**: conn=0.907 across 12/12 iters with ABSOLUTE parameter insensitivity (lr_W 3E-3-1.5E-2, L1 1E-5-1E-6, lr 1E-4-2E-4, batch 8-16, epochs 2-3, edge_diff 100-500); rho=0.995; eff_rank=35-36; 83% convergence rate (10/12 cross R2>0.9 threshold); conn_ceiling law now validated at 4 fill points: 50%→0.49, 80%→0.80, 90%→0.91, 100%→1.00; relationship is LINEAR (conn≈fill). (evidence: block 24)
67. **fill<1 has no lr_W cliff**: tested at fill=50% (up to 1.5E-2), fill=80% (up to 2E-2), fill=90% (up to 1.5E-2) — no cliff detected; the lr_W cliff (which exists at fill=100% g=7 at ~8E-3) disappears when connectivity is partial; conn insensitivity to lr_W increases as fill decreases. (evidence: blocks 7, 22, 23, 24)
68. **g=1 creates FIXED-POINT COLLAPSE at n=100/10k**: eff_rank=5 (vs g=3's 26, g=7's 35); rho=1.065 (supercritical, same as g=3); dynamics are stable fixed points (flat lines) despite supercritical W — tanh saturates weak-gain dynamics; 12/12 FAILED (0% conv); conn range [0.000, 0.007]; COMPLETE parameter insensitivity across lr_W [1E-3, 3E-2], epochs [1, 15], L1, edge_diff, two-phase, recurrent; MORE severe than sparse 50% (which has eff_rank=21 and conn~0.4-0.5). (evidence: block 25)
69. **two-phase training is the ONLY non-zero intervention at g=1/10k**: conn=0.007 vs 0.000-0.002 for all other configs; parallels sparse regime where two-phase was also the only marginal improvement. (evidence: block 25)
70. **spectral radius is INDEPENDENT of gain**: rho=1.065 at g=1, g=3, and g=7 (all at n=100 chaotic); rho depends on W structure, not on the gain parameter; BUT dynamics quality (eff_rank) depends STRONGLY on gain: g=1→5, g=3→26, g=7→35. (evidence: blocks 1, 19, 25)
71. **eff_rank is a NON-LINEAR function of gain**: g=7→35, g=3→26 (-26%), g=1→5 (-86%); the gain-eff_rank relationship is highly nonlinear — g=1 creates a catastrophic collapse in data dimensionality; there may be a critical gain threshold between g=1 and g=3 where dynamics transition from fixed-point to oscillatory. (evidence: blocks 1, 19, 25)
72. **epoch scaling effect depends on eff_rank**: at eff_rank=35 (g=7): 1ep optimal; at eff_rank=26 (g=3): 3ep needed; at eff_rank=5 (g=1): epochs 1-15 ALL give conn~0.000 — below some eff_rank threshold (~10?), no amount of training helps. (evidence: blocks 1, 19, 25)
73. **no lr_W cliff at g=1 up to 3E-2**: extends principle 54 (gain modulates lr_W cliff) to extreme; at g=1, lr_W is completely irrelevant [1E-3, 3E-2]; weakest gain eliminates cliff most aggressively but also eliminates all learning. (evidence: blocks 25, 26)
74. **g=1 fixed-point collapse is IMMUNE to n_frames**: 30k frames does NOT rescue g=1; eff_rank DROPS from 5 (10k) to 1 (30k); conn range [0.002, 0.018] (0% conv); ABSOLUTE parameter insensitivity (12/12 failed at 30k); more data lets fixed-point dynamics converge faster, REDUCING dimensionality; g=1 is CONFIRMED UNSOLVABLE by data alone. (evidence: blocks 25, 26)
75. **eff_rank can DECREASE with more data in collapsed regimes**: g=1 eff_rank 5→1 at 30k; sparse 50% eff_rank 21→13 at 30k; when dynamics converge to fixed points or subcritical decay, more frames capture less variance (dynamics converge more completely); eff_rank-n_frames correlation is POSITIVE for oscillatory regimes (g≥3) but NEGATIVE for collapsed regimes (g=1, sparse subcritical). (evidence: blocks 17, 26 vs 15, 16, 21)
76. **edge_diff=500 is HARMFUL at g=1**: conn=0.002 (worst in block 26) vs 0.009-0.018 at edge_diff=100; constraining MLP monotonicity does NOT redirect learning capacity to W in collapsed regimes; at g=1, all learning capacity goes to MLP regardless of constraints. (evidence: block 26)
77. **more training INCREASES degeneracy at g=1**: 10ep gives best dynamics (test_R2=0.998) but conn IDENTICAL to 3ep (0.009); aug=40/5ep gives best dynamics but worst conn (0.008); additional training capacity is absorbed exclusively by MLP; overtraining at g=1 does not overshoot W but WIDENS degeneracy gap. (evidence: block 26)
78. **g=2 requires INVERSE lr_W optimization**: lr_W=5E-4→conn=0.515, lr_W=1E-3→0.356-0.519, lr_W=2E-3→0.078-0.125, lr_W=4-12E-3→0.004; optimal lr_W is 100x LOWER than g=7 (4E-3); at eff_rank=17, standard lr_W overshoots W; lower lr_W substitutes for epochs (5E-4/5ep ≈ 1E-3/12ep). (evidence: block 27)
79. **g=2 eff_rank=17 — critical gain-eff_rank transition point**: g=1→5, g=2→17 (+240%), g=3→26, g=7→35; steepest slope between g=1 and g=2; g=2 is ABOVE fixed-point threshold (eff_rank>10 → learnable) but requires very low lr_W and many epochs; rho=1.065 (independent of gain, confirming principle 70). (evidence: block 27)
80. **epoch scaling depends on gain AND lr_W, not just eff_rank**: at g=2/lr_W=1E-3: 5ep→0.356, 8ep→0.397, 12ep→0.519 (NOT diminishing); CONTRADICTS principle 25 (diminishing at small n); low gain creates regime where each epoch provides more information to distinguish W from MLP — epochs remain effective precisely because learning rate is low enough to not overshoot. (evidence: blocks 1, 19, 27)
81. **optimal lr_W inversely scales with gain**: g=7→4E-3, g=3→8E-3, g=2→5E-4; at low gain, weaker W signals need slower learning to avoid MLP absorption; g=3 reverses the trend (higher than g=7) because g=3 benefits from more aggressive W exploration while still having sufficient eff_rank. (evidence: blocks 1, 19, 27)
82. **g=2 at 10k: 0% convergence, max conn=0.519**: eff_rank=17; 12/12 degenerate; dynamics always excellent (test_R2≥0.997 at lr_W≤1E-3); likely needs 30k frames; comparable difficulty to g=3/n=200/10k (max 0.489). (evidence: block 27)
83. **g=2/30k: 42% convergence, Pareto conn=0.997**: eff_rank=16 (UNCHANGED from 10k's 17 — 30k does NOT increase dimensionality at g=2); inverse lr_W PERSISTS (5E-4-7E-4 optimal; >=2E-3 catastrophic); epoch scaling NOT diminishing: 2ep→0.848, 3ep→0.943, 5ep→0.983, 8ep→0.997; minimum convergent: lr_W=7E-4/3ep; Pareto: lr_W=5E-4/8ep; lr=2E-4 helps at 2ep but neutral at 5+ep. (evidence: block 28)
84. **g=2 eff_rank at n=100 does NOT increase with n_frames BUT scales with n_neurons**: at n=100: 16 at 30k vs 17 at 10k (FLAT); at n=200: 35-38 (2.2x increase from n=100); CONTRADICTS original claim of fixed intrinsic dimensionality — the invariance was n=100-specific; at n=200, more neurons provide more independent dynamical modes even at low gain; eff_rank-n_frames correlation is NEGATIVE for g=1, FLAT for g=2/n=100, POSITIVE for g=2/n=200 and g>=3. (evidence: blocks 27, 28, 29)
85. **lr_W advantage vanishes at high epochs at g=2**: at 2ep lr_W=7E-4 > 5E-4 (0.865 vs 0.848); at 5ep 7E-4 ≈ 5E-4 (0.985 vs 0.983); epochs dominate over lr_W fine-tuning when sufficient training; same pattern as n_frames dominance at high n. (evidence: block 28)
86. **lr=2E-4 benefit is epoch-dependent at g=2/30k**: at 2ep lr=2E-4 helps +2.7% conn (0.871 vs 0.848); at 5ep lr=2E-4 NEUTRAL for conn (0.984 vs 0.983) but HURTS clustering (0.640 vs 0.730); use lr=1E-4 at 5+ep. (evidence: block 28)
87. **g=2 eff_rank scales with n_neurons — inverse lr_W threshold scales with n**: at n=100 eff_rank=16-17, at n=200 eff_rank=35-38 (2.2x); optimal lr_W: n=100→5E-4, n=200→3E-4; catastrophic ceiling: n=100→>=2E-3, n=200→>=1E-3 safe (ceiling scales UP with n); higher eff_rank from more neurons allows slightly higher lr_W. (evidence: blocks 27-29)
88. **g=2/n=200/30k recipe: lr_W=3E-4, lr=1E-4, L1=1E-5, batch=8, 10-12ep**: conn=0.976-0.979, cluster=1.000; 92% convergence rate; epoch scaling diminishing at 12ep (+0.3%); lr_W=1E-3 epoch-insensitive and cluster-harmful. (evidence: block 29)
89. **lr_W=1E-3 epoch scaling is NEGATIVE at g=2/n=200**: conn stagnates (0.942-0.943 at 8-10ep) while cluster DEGRADES (0.975→0.730); above optimal lr_W, more epochs overshoots W and damages embedding; epoch scaling only works within the correct lr_W range. (evidence: block 29)
90. **gain×n interaction is COMPENSATORY for eff_rank**: g=2/n=100→eff_rank=16, g=2/n=200→35-38; doubling n at fixed low gain RESTORES eff_rank to g=7/n=100 levels (35); more neurons provide more independent modes that compensate for gain-suppressed dynamics; this explains why g=2/n=200/30k (92% conv) is much easier than g=2/n=100/30k (42% conv). (evidence: blocks 27-29)
91. **100k frames TRANSFORMS n=1000**: 30k/8ep max conn=0.745 (0% conv) → 100k/1ep conn=0.998-0.999 (100% conv, 4/4); 3.3x more data compensates for 8x fewer epochs; n_frames >> n_epochs for large n CONFIRMED monumentally; conn essentially perfect at 1ep. (evidence: blocks 18, 30)
92. **100k frames optimal lr_W=3E-3 at n=1000**: confirms principle 44 (dynamics-optimal lr_W inversely scales with n_frames); pattern: 10k→1E-2, 30k→5E-3, 100k→3E-3; ~linear inverse relationship with sqrt(n_frames). (evidence: blocks 12, 16, 30)
93. **batch=16 efficiency gain at 100k/n=1000**: training time HALVES (102 vs 186 min per epoch) with <0.1% conn penalty (0.998 vs 0.999); at very high n_frames, batch=16 is a MAJOR efficiency lever; dynamics penalty small (test_R2=0.721 vs 0.825). (evidence: block 30)
94. **dynamics stochastic variance at n=1000/100k**: same config (lr_W=5E-3, 1ep) gives test_R2 range 0.728-0.825 (12% variance); conn identical (0.998); dynamics variance increases with n even at 100k; CONFIRMS principle 52 extended to 100k. (evidence: block 30)
95. **3ep is breakthrough for dynamics at n=1000/100k**: 1-2ep gives test_R2=0.72-0.80; 3ep gives test_R2=0.88; epoch scaling ~+10% at 2→3ep; conn SOLVED at 1ep (0.998+) but dynamics need 3ep. (evidence: block 30)
96. **batch=16/3ep Pareto-optimal at n=1000/100k**: conn=1.000, test_R2=0.882, kino_R2=0.870, training=304 min; batch=16 is 45% faster than batch=8 (304 vs 560 min) with BETTER conn (1.000 vs 0.999); the efficiency gain from batch=16 INCREASES with n_frames and n_neurons. (evidence: block 30)
97. **lr_W=5E-3 hurts dynamics even at 3ep at 100k**: lr_W=3E-3/3ep gives test_R2=0.882 vs lr_W=5E-3/3ep 0.787 (-10.8%); confirms dynamics-optimal lr_W is 2-3E-3 at 100k, NOT 5E-3; the optimal lr_W for dynamics is LOWER than for conn. (evidence: block 30)
98. **conn insensitive to lr_W at 100k/n=1000**: lr_W=[1E-3, 3E-3] ALL give conn=0.999; lr_W only affects dynamics at 100k; COMPLETE lr_W insensitivity for connectivity when data abundant. (evidence: block 30)
99. **lr_W×epoch interaction at 100k**: at low epochs (1ep), LOWER lr_W gives BETTER dynamics (1E-3→0.778 > 3E-3→0.750); at high epochs (3ep), MODERATE lr_W wins (3E-3→0.882); mechanism: low lr_W preserves MLP capacity when W not converged; moderate lr_W at sufficient epochs allows faster W convergence which releases MLP capacity. (evidence: block 30)

### Degeneracy Analysis (across all blocks)

Degeneracy = high test_pearson but low connectivity_R2 (gap > 0.3). The MLP compensates for wrong W.

| Block | Regime | Degenerate iters | Max gap | Mechanism |
|-------|--------|:----------------:|:-------:|-----------|
| 1 | Chaotic n=100 | 0/12 | 0.15 | Healthy |
| 2 | Low-rank n=100 | 1/12 (iter 17) | 0.45 | Stochastic failure at lr_W=5E-3 |
| 3 | Dale law n=100 | 4/12 (iters 28-30,32) | 0.53 | lr_W above Dale cliff (>=5E-3) |
| 4 | Heterogeneous n=100 | 0/12 | 0.11 | Healthy |
| 5 | Noise n=100 | 0/12 | N/A | Healthy (conn=1.000 always) |
| 6 | Chaotic n=200 | 0/12 | 0.26 | Healthy (borderline iter 62) |
| 7 | **Sparse 50% n=100** | **12/12** | **0.82** | **Universal degeneracy — subcritical rho=0.746** |
| 8 | Sparse+noise n=100 | 0/12 | N/A | Not degenerate (pearson too low) |
| 9 | Chaotic n=300 1ep | 2/12 (iters 98,99) | 0.38 | Training-limited |
| 10 | Chaotic n=300 2-4ep | 0/12 | 0.08 | Healthy |
| 11 | Chaotic n=200 2-3ep | 0/12 | 0.00 | Healthy |
| 12 | Chaotic n=600 | 0/12 | 0.26 | Not degenerate (underfitting) |
| 13 | n=200+4types | 0/12 | 0.05 | Healthy |
| 14 | n=200+recurrent | 0/4 | 0.07 | Healthy |
| 15 | **n=300 30k frames** | **0/12** | **-0.01** | **Healthy — abundant data eliminates degeneracy** |
| 16 | **n=600 30k frames** | **0/12** | **-0.22** | **Healthy — all negative gaps (pearson < conn)** |
| 17 | **sparse 50% n=100 30k** | **12/12** | **0.74** | **Universal degeneracy — subcritical rho=0.746; n_frames did NOT help** |
| 18 | **chaotic n=1000 30k** | **0/12** | **-0.08** | **Healthy — underfitting (conn > pearson); 30k insufficient** |
| 19 | **chaotic g=3 n=100** | **4/12** | **0.75** | **Training-limited degeneracy at 1ep; resolved at 2ep (gap<0.1)** |
| 20 | **chaotic g=3 n=200** | **12/12** | **0.67** | **Universal training-limited degeneracy; gap narrows with epochs (0.67→0.46) but never resolves at 10k** |
| 21 | **chaotic g=3 n=200 (30k)** | **0/12** | **-0.01** | **Healthy — 30k resolves all degeneracy from block 20; all gaps ≤0.01** |
| 22 | **chaotic fill=80%** | **0/12** | **0.29** | **Healthy — no degeneracy despite conn plateau at 0.802; max gap at insufficient lr_W** |
| 23 | **chaotic fill=80% (30k)** | **0/12** | **0.19** | **Healthy — no degeneracy; conn locked at 0.802; max gap=0.19 at lr_W=4E-3** |
| 24 | **chaotic fill=90%** | **0/12** | **0.08** | **Healthy — no degeneracy; conn locked at 0.907; max gap=0.08** |
| 25 | **chaotic g=1 n=100** | **12/12** | **1.000** | **Universal SEVERE degeneracy — FIXED-POINT COLLAPSE; eff_rank=5; conn=0.000-0.007; ALL gaps 0.99+** |
| 26 | **chaotic g=1 n=100 (30k)** | **12/12** | **0.990** | **Universal SEVERE degeneracy — eff_rank DROPPED to 1; conn=0.002-0.018; 30k WORSE (eff_rank 5→1)** |
| 27 | **chaotic g=2 n=100** | **12/12** | **0.603** | **Universal degeneracy — eff_rank=17; conn [0.004, 0.519]; inverse lr_W (5E-4 optimal); gaps 0.48-0.90; narrowing with conn improvement** |
| 28 | **chaotic g=2 n=100 (30k)** | **5/12** | **0.799** | **Degeneracy at 2ep with lr_W>=1E-3 or insufficient epochs; resolved at 5+ep with lr_W<=7E-4; 30k reduces degeneracy rate 100%→42%** |
| 29 | **chaotic g=2 n=200 (30k)** | **0/12** | **0.12** | **Healthy — no degeneracy; eff_rank=35-38 provides sufficient signal; all gaps < 0.12** |
| 30 | **chaotic n=1000 (100k)** | **0/12** | **-0.31** | **Healthy — ALL 12 iters healthy; negative gaps (conn > pearson); conn PERFECT (0.998-1.000); dynamics lag at 1ep but 3ep solves** |

**Total**: 88/360 degenerate iterations (24.4%), 24 sparse (blocks 7+17), 12 low-gain/n=200 (block 20), 24 g=1 fixed-point (blocks 25+26), 4 low-gain 1ep (block 19), 12 g=2/10k (block 27), 5 g=2/30k (block 28).

**Five degeneracy mechanisms:**
1. **Structural degeneracy** (Blocks 7, 17): subcritical spectral radius; cannot be fixed by training parameters or n_frames
2. **Training-limited degeneracy** (Blocks 3, 9, 20): fixable with more epochs, correct lr_W, or more n_frames; at g=3/n=200, gap narrows with epochs (0.67→0.46) but does not resolve at 10k
3. **Fixed-point collapse degeneracy** (Blocks 25, 26): eff_rank=5→1 at g=1; dynamics are fixed points (flat lines) despite supercritical rho; W contains NO recoverable information; 30k frames makes WORSE (eff_rank drops); IMMUNE to n_frames
4. **n_frames-amplified degeneracy** (Block 26): more training at g=1/30k makes MLP BETTER but W WORSE; gap increases with training (0.943→0.990); unique mechanism where data abundance widens degeneracy
5. **Low-gain lr_W-overshooting degeneracy** (Blocks 27, 28): at g=2/eff_rank=17, standard lr_W (4E-3+) causes universal degeneracy (conn=0.004); REDUCIBLE by lowering lr_W to 5E-4-7E-4 (conn up to 0.519 at 10k, 0.997 at 30k); at 30k with proper lr_W+epochs, fully resolved (gap 0.003 at 8ep); 30k reduces degeneracy rate 100%→42%

### Open Questions
- what happens with Dale_law + low_rank?
- does n_neuron_types>1 interact with low_rank?
- why does Dale regime behave differently from low_rank for L1 sensitivity despite same eff_rank=12?
- does noise=2.0 still converge?
- ~~can sparse 50% reach convergence with 30k frames?~~ **NO** — block 17: 0% conv, eff_rank=13 (LOWER than 10k), universal degeneracy
- what is the minimum filling_factor that maintains convergence? (tested 100% and 50%)
- can n=1000 converge? reference config uses 100k frames — **TESTING IN BLOCK 18**
- ~~at what n_neurons does lr=1E-4 become catastrophic?~~ RESOLVED at 30k: NOT catastrophic at n=600/30k
- ~~does n_frames dominance hold at n=600?~~ YES — block 16 confirmed (0%→100%)
- ~~what is n=600 eff_rank at 30k?~~ ANSWERED: 87
- ~~can sparse 50% benefit from two-phase training?~~ MARGINALLY — iter 201: +15% but still 0% convergence
- ~~what is sparse 50% eff_rank at 30k?~~ ANSWERED: 13 (LOWER than 21 at 10k)
- can sparse regime converge at n=1000/100k as reference config suggests?
- ~~what is n=1000 eff_rank at 30k? how does it scale?~~ ANSWERED: 144; superlinear scaling from n=600/30k's 87
- ~~does low gain (g=3) reduce eff_rank and create sparse-like difficulties?~~ ANSWERED: eff_rank 35→26, NOT subcritical (rho=1.065); NOT sparse-like — solvable with 2-3ep; independent difficulty axis
- at n=1000/100k, can convergence be achieved? (user prior says yes)
- does lr=1E-4 remain optimal at n=1000/100k or does 100k rescue lr=2E-4?
- ~~does g=3 + n=200 compound difficulty?~~ YES — block 20: 0% conv, max 0.489 at 6ep; gain × n compounds severely
- what is the minimum gain that maintains g=7-like easy convergence? (tested g=1, g=3, g=7; transition between g=1 and g=3)
- ~~can g=1/30k rescue fixed-point collapse like g=3/30k rescued g=3?~~ **NO** — block 26: eff_rank DROPS 5→1; 0% conv (12/12 failed); g=1 CONFIRMED UNSOLVABLE by n_frames
- ~~what gain produces the fixed-point→oscillatory transition?~~ ANSWERED: g=2 (eff_rank=17) is ABOVE threshold; transition between g=1 (eff_rank=5, unsolvable) and g=2 (eff_rank=17, partially solvable with low lr_W); critical eff_rank threshold ~10
- ~~does g=1 create subcritical behavior?~~ **NO** — g=1 has rho=1.065 (supercritical, same as g=3 and g=7); BUT creates FIXED-POINT COLLAPSE (eff_rank=5); rho is independent of gain
- ~~can 30k frames rescue g=3/n=200?~~ **YES** — block 21: 100% conv (12/12), conn=0.996; n_frames rescues low gain
- ~~does g=3/n=200 eff_rank increase at 30k like g=7/n=200→300 does?~~ **YES** — 31→53-57 (+80%)
- ~~does filling_factor=80% at n=100/10k converge?~~ **NO** — block 22: 0% conv, conn plateau at 0.802; rho=0.985; eff_rank=36; complete parameter insensitivity
- ~~what is the minimum filling_factor for convergence at 10k?~~ ANSWERED: fill=90% converges (83%, conn=0.907>0.90); fill=80% fails (0%, conn=0.802<0.90); critical threshold between 80-90%
- ~~can fill=80% converge at 30k frames?~~ **NO** — block 23: conn=0.802 IDENTICAL to 10k; n_frames does NOT rescue fill<1
- ~~is conn_ceiling ≈ filling_factor a general relationship?~~ **YES — STRUCTURAL INVARIANT** — holds at 10k AND 30k, ALL params; strongest law found
- what filling_factor produces the subcritical→near-critical transition? (between 50-80%)
- ~~can g=2/30k rescue conn like g=3/n=200/30k?~~ **YES** — block 28: 42% conv, max conn=0.997; BUT eff_rank=16 (NOT doubled as expected; unchanged from 10k's 17)
- ~~does inverse lr_W pattern at g=2 persist at 30k?~~ **YES** — inverse lr_W PERSISTS structurally; optimal 5E-4-7E-4; >=2E-3 catastrophic; 30k does NOT widen lr_W ceiling
- ~~does g=2/n=200/30k converge?~~ **YES** — block 29: 92% conv, best 0.979; MUCH easier than predicted; eff_rank=35-38 (gain×n compensatory for eff_rank)
- what is eff_rank at g=4 or g=5? can we map the gain-eff_rank transition more precisely?
- does self-excitation (s parameter) affect convergence?

### Simulation-GNN training landscape
- chaotic baseline (n=100, eff_rank=35) is easy: 92% convergence, wide lr_W range
- low_rank (eff_rank=12-13): harder at L1=1E-5, fully recoverable with L1=1E-6
- Dale_law (eff_rank=12): 67% convergence; lr_W cliff at 5E-3
- heterogeneous (eff_rank=38): 83% conn, 17% FULL dual convergence
- **noisy dense (eff_rank=42-90): EASIEST — 100% convergence; noise is data augmentation for connectivity**
- **n=200 chaotic (eff_rank=42): SOLVED with 2ep — 100% convergence (12/12); lr_W=8E-3 optimal**
- **n=300/10k chaotic (eff_rank=47): 25% convergence; training-param-sensitive**
- **n=300/30k chaotic (eff_rank=80): SOLVED — 100% convergence (12/12); all params non-critical; Pareto: lr_W=3E-3/3ep**
- **n=600/10k chaotic (eff_rank=50): 0% convergence at 10ep; training-capacity-limited**
- **n=600/30k chaotic (eff_rank=87): SOLVED — 100% convergence (12/12); Pareto: lr_W=5E-3/5ep → conn=0.993, test_R2=0.966**
- **sparse 50% (eff_rank=13-21): HARDEST — 0% convergence at BOTH 10k and 30k; subcritical (rho=0.746); eff_rank DROPS at 30k (13 vs 21); IMMUNE to n_frames**
- **KEY INSIGHT: n_frames is the DOMINANT lever — more impactful than n_epochs, L1, lr_W, or any training param**
- **KEY INSIGHT: at sufficient n_frames, ALL parameter catastrophes are rescued EXCEPT subcritical sparse**
- **KEY INSIGHT: at sufficient n_frames, training params become non-critical — the landscape flattens**
- **KEY INSIGHT: dynamics-optimal lr_W inversely scales with n_frames (more data → lower lr_W optimal)**
- **RESOLVED: n_frames does NOT rescue subcritical spectral_radius (sparse)** — block 17 confirmed; subcritical rho is the ONLY unsolvable difficulty axis with data alone
- eff_rank necessary but NOT sufficient for predicting difficulty — subcritical rho is the true barrier
- **network size (n_neurons) is a key difficulty factor independent of eff_rank, but SOLVABLE with n_frames**
- **three independent difficulty axes: (1) subcritical spectral radius (sparse — UNSOLVED), (2) parameter count scaling (large n — SOLVED by n_frames), (3) data abundance (n_frames — SOLVED by scaling)**
- **n=1000/30k: eff_rank=144, max conn=0.745 — 30k insufficient but 100k should work per user prior**
- **recurrent training (time_step=4) is a conn-dynamics trade-off at supercritical rho**: may be useful in conn-bottleneck regimes
- **low gain (g=3) at n=100 (eff_rank=26): 42% convergence; training-limited like large n; n_epochs dominant lever; no lr_W cliff; batch=16/L1=1E-6 catastrophic**
- **low gain (g=3) at n=200 (eff_rank=31): 0% convergence at 10k/6ep; gain × n compounds severely; epoch scaling diminishing; needs 30k frames or 10+ep**
- **gain is a 4th independent difficulty axis**: (1) subcritical rho (sparse — UNSOLVED), (2) n_neurons scaling (SOLVED by n_frames), (3) data abundance, (4) gain reduction (SOLVED by n_epochs at small n, POSSIBLY needs n_frames at large n)
- **three difficulty axes, all solvable EXCEPT subcritical rho**: gain (SOLVED by n_frames — block 21: g=3/n=200/30k 100% conv), n_neurons (SOLVED by n_frames), subcritical rho (UNSOLVED); gain is NOT an independent axis — it compounds with n but is equally rescued by n_frames
- **gain × n interaction is SUPER-ADDITIVE at fixed n_frames**: g=3 alone costs ~58% at n=100 (0.955 vs 1.000), n=200 alone costs ~4% (0.956 vs 1.000), but g=3+n=200 costs ~51% (0.489 vs 1.000); BUT all resolved at 30k (0.996)
- **n_frames is the UNIVERSAL solver**: rescues large n (blocks 15, 16), low gain (block 21), parameter catastrophes (lr=1E-4, L1, batch); ONLY exception is subcritical spectral radius (sparse 50%)
- **fill=80% (rho=0.985): intermediate regime**: eff_rank=36 (same as 100%), conn plateau at 0.80, 0% conv at 10k; NOT subcritical like 50% (no degeneracy); conn_ceiling ≈ filling_factor at 10k
- **filling_factor transition is SHARP between 50-80%**: rho jumps 0.746→0.985; the subcritical barrier lives between 50-80% fill
- **six difficulty axes**: (1) subcritical rho (sparse 50% — UNSOLVED), (2) n_neurons scaling (SOLVED by n_frames), (3) data abundance, (4) moderate gain reduction g=3 (SOLVED by n_frames), (5) partial connectivity (fill<1 conn capped at fill% — CONFIRMED UNSOLVED by n_frames; structural invariant), (6) extreme gain reduction g=1 (FIXED-POINT COLLAPSE — eff_rank=5, conn~0.000; status at 30k TBD)
- **g=1 is the HARDEST regime tested**: conn=0.000-0.018 at BOTH 10k AND 30k; eff_rank=5 (10k) → 1 (30k); 30k makes WORSE; n_frames IMMUNE; epoch/lr_W/L1/recurrent/edge_diff ALL useless; only two-phase gives marginal signal
- **eff_rank can DECREASE with more data**: g=1 (5→1 at 30k), sparse 50% (21→13 at 30k) — collapsed dynamics (fixed points, subcritical decay) have NEGATIVE eff_rank-n_frames correlation; this is the anti-pattern to the normal POSITIVE correlation (e.g., g=7 n=300: 47→80)
- **conn_ceiling ≈ fill% is the STRONGEST structural law**: holds across 10k AND 30k, ALL training params, fill=50%, 80%, AND 90%; validated at 4 fill points (LINEAR relationship); the missing connections create an unrecoverable information gap; GNN correctly learns the existing connections but cannot infer zeros
- **n_frames rescues dynamics and eff_rank at fill<1**: eff_rank improves (36→48-49 at fill=80%) and dynamics decouple (kino_R2 up to 0.999) but conn stays locked at fill%
- **fill=90% is transitional**: conn=0.907 right at R2>0.9 convergence threshold; 83% convergence rate; rho=0.995; eff_rank=35-36; ABSOLUTE parameter insensitivity like fill=80% but at higher conn level
- **g=2 at n=100 (eff_rank=16-17) requires INVERSE lr_W**: optimal lr_W=5E-4; at 10k 0% conv, at 30k 42% conv (Pareto 0.997 at 8ep)
- **g=2 at n=200 (eff_rank=35-38) MUCH easier**: 92% conv at 30k; optimal lr_W=3E-4; eff_rank jumps 2.2x from n=100→n=200; doubling n compensates for gain reduction
- **gain modulates optimal lr_W nonlinearly**: g=7→4E-3, g=3→8E-3, g=2/n=100→5E-4, g=2/n=200→3E-4; g=3 anomaly (higher than g=7) due to eff_rank=26; g=2 demands low lr_W but ceiling scales with n
- **g=2 inverse lr_W is STRUCTURAL but n-dependent**: lr_W ceiling n=100→1E-3, n=200→somewhere between 1E-3-2E-3; eff_rank determines the constraint; n=200 eff_rank=35-38 allows higher ceiling than n=100 eff_rank=16
- **n=1000/100k SOLVED**: 100% convergence (12/12); conn=0.998-1.000; confirms 100k as the n=1000 data requirement; Pareto: lr_W=3E-3, batch=16, 3ep → conn=1.000, test_R2=0.882
- **optimal lr_W inversely scales with n_frames (confirmed at 100k)**: 10k→1E-2, 30k→5E-3, 100k→3E-3 at n=1000; approximately sqrt inverse relationship
- **lr_W×epoch interaction at 100k/n=1000**: at 1ep, lower lr_W wins (1E-3→0.778 > 3E-3→0.750); at 3ep, moderate lr_W wins (3E-3→0.882); mechanism: low lr_W preserves MLP capacity; moderate lr_W at sufficient epochs releases MLP capacity after W converges
- **conn insensitive to lr_W at 100k**: ALL lr_W [1E-3, 3E-3] give conn=0.999; training params become non-critical when data abundant

---

## Previous Block Summary (Block 29)

Block 29 (chaotic, g=2, n=200, 1type, 30k frames): 11/12 converged (92%).
eff_rank=35-38 (MUCH higher than g=2/n=100's 16-17). Best: iter 345 (lr_W=3E-4, 12ep, conn=0.979, cluster=1.000).
Key: g=2/n=200/30k MUCH easier than predicted (92% vs expected <42%); eff_rank scales with n at g=2 (CONTRADICTS principle 84); inverse lr_W persists but ceiling scales with n (1E-3 safe at n=200); epoch scaling diminishing at 12ep (+0.3%); lr_W=1E-3 epoch-scaling NEGATIVE (cluster degrades); batch=16 safe; lr_W=3E-4 Pareto-optimal.

---

## Current Block (Block 30)

### Block Info
Simulation: connectivity_type=chaotic, Dale_law=False, n_neurons=1000, n_neuron_types=1, n_frames=100000, gain=7, noise_model_level=0, connectivity_filling_factor=1
Iterations: 349 to 360 (n_iter_block=12)

### Hypothesis
Test n=1000 with 100k frames — the user priority. Block 18 (n=1000/30k) gave max conn=0.745 (0% conv). User expects ~100k frames to work.
Predictions:
1. 100k frames should provide eff_rank >> 144 (block 18's value at 30k); possibly 300+
2. lr_W should shift lower per principle 44 (dynamics-optimal lr_W inversely scales with n_frames): 30k optimal was 1E-2, expect 100k optimal ~3-5E-3
3. Fewer epochs needed than 30k (principle 39: n_frames >> n_epochs)
4. lr=1E-4 should be safe (was Pareto-better at 30k, principle 49)
5. Convergence rate should be high if 100k is sufficient (100k/30k = 3.3x data, similar to 30k/10k which transformed n=300 and n=600)
6. If convergence achieved, this VALIDATES n_frames as universal solver for n scaling
Strategy: spread lr_W from 3E-3 to 1E-2; test 3-8 epochs; lr=1E-4; L1=1E-5; batch=8-16.

### Iterations This Block

## Batch 1 (iters 349-352): ALL CONVERGED (4/4)
- Iter 349: lr_W=5E-3, batch=8, 1ep → conn=0.998, test_R2=0.825
- Iter 350: lr_W=3E-3, batch=8, 1ep → conn=0.999 (BEST), test_R2=0.801
- Iter 351: lr_W=1E-2, batch=16, 1ep → conn=0.998, test_R2=0.721, time=102 min (HALF)
- Iter 352: lr_W=5E-3, batch=8, 1ep → conn=0.998, test_R2=0.728 (variance check)

**Key findings:**
- 100k frames BREAKTHROUGH: 100% convergence at 1ep (vs 30k/8ep 0% conv)
- lr_W=3E-3 optimal for conn (0.999)
- batch=16 HALVES training time with <0.1% conn penalty
- dynamics NOT converged at 1ep (needs more epochs)
- stochastic variance 12% at n=1000

## Batch 2 (iters 353-356): ALL CONVERGED (4/4)
- Iter 353: lr_W=3E-3, batch=8, 2ep → conn=0.999, test_R2=0.772, kino_R2=0.714, time=370 min
- Iter 354: lr_W=2E-3, batch=8, 2ep → conn=0.999, test_R2=0.794, kino_R2=0.757, time=374 min
- Iter 355: lr_W=3E-3, batch=16, 3ep → **conn=1.000, test_R2=0.882, kino_R2=0.870**, time=304 min **(BEST)**
- Iter 356: lr_W=5E-3, batch=8, 3ep → conn=0.999, test_R2=0.787, kino_R2=0.752, time=560 min

**Key findings:**
- **8/8 cumulative convergence** — 100% continues at n=1000/100k
- **Iter 355 BEST**: batch=16/3ep achieves conn=1.000 (PERFECT), test_R2=0.882, AND 45% faster
- lr_W=2E-3 beats 3E-3 at 2ep for dynamics (0.794 vs 0.772) — principle 44 confirmed
- lr_W=5E-3/3ep (0.787) << lr_W=3E-3/3ep (0.882) — principle 44 STRONGLY confirmed
- **batch=16 is MAJOR efficiency lever**: 45% faster (304 vs 560 min) with BETTER results
- **3ep is breakthrough for dynamics**: test_R2 0.77-0.79 at 2ep → 0.88 at 3ep
- conn SOLVED at 100k (0.998-1.000 always); dynamics now the challenge

## Batch 3 (iters 357-360): ALL CONVERGED (4/4)
- Iter 357: lr_W=3E-3, batch=16, 1ep → conn=0.999, test_R2=0.750, kino_R2=0.689, time=102 min
- Iter 358: lr_W=2E-3, batch=16, 1ep → conn=0.999, test_R2=0.756, kino_R2=0.705, time=101 min
- Iter 359: lr_W=1.5E-3, batch=16, 1ep → conn=0.999, test_R2=0.775, kino_R2=0.738, time=102 min
- Iter 360: lr_W=1E-3, batch=16, 1ep → conn=0.999, test_R2=0.778, kino_R2=0.724, time=101 min

**Key findings:**
- **12/12 cumulative convergence (100%)** — BLOCK COMPLETE
- **lr_W inversely scales with dynamics at 1ep**: 1E-3→0.778, 1.5E-3→0.775, 2E-3→0.756, 3E-3→0.750
- **Lowest lr_W (1E-3) gives BEST dynamics at 1ep** — BUT NOT at 3ep (principle 99 discovered)
- conn=0.999 across ALL lr_W [1E-3, 3E-3] — COMPLETE insensitivity (principle 98)
- lr_W×epoch interaction: at low epochs, lower lr_W wins; at high epochs (3ep), moderate lr_W (3E-3) wins

>>> BLOCK 30 COMPLETE <<<

---

## Previous Block Summary (Block 30)

Block 30 (chaotic, n=1000, 1type, 100k frames, gain=7): **12/12 CONVERGED (100%)** — BREAKTHROUGH!
Best: iter 355 (lr_W=3E-3, batch=16, 3ep) → conn=1.000, test_R2=0.882, kino_R2=0.870, time=304 min

**Key findings:**
- **100k frames TRANSFORMS n=1000**: 30k/8ep max conn=0.745 (0% conv) → 100k/1ep conn=0.998-0.999 (100% conv)
- **n_frames >> n_epochs CONFIRMED monumentally**: 3.3x more data compensates for 8x fewer epochs
- **batch=16 Pareto-optimal**: 45% faster than batch=8 with equal or better results (102 vs 186 min/ep)
- **lr_W×epoch interaction discovered**: at 1ep, lower lr_W better (1E-3→0.778 > 3E-3→0.750); at 3ep, moderate lr_W better (3E-3→0.882)
- **conn SOLVED at 100k**: 12/12 at 0.998-1.000; dynamics needs 3ep (0.75-0.78 at 1ep → 0.88 at 3ep)
- **conn insensitive to lr_W at 100k**: [1E-3, 3E-3] ALL give 0.999
- **optimal lr_W=3E-3 at 100k/n=1000**: confirms inverse sqrt(n_frames) scaling (10k→1E-2, 30k→5E-3, 100k→3E-3)

New principles: 98 (conn insensitive to lr_W at 100k), 99 (lr_W×epoch interaction)

---

## Current Block (Block 31)

### Block Info
Simulation: connectivity_type=chaotic, Dale_law=False, n_neurons=1000, n_neuron_types=4, n_frames=100000, gain=7, noise_model_level=0, connectivity_filling_factor=1
Iterations: 361 to 372 (n_iter_block=12)

### Hypothesis
Test dual-objective (connectivity + clustering) at n=1000/100k. Block 30 solved connectivity (100%); now test if adding type inference (n_types=4) changes the picture.
Predictions:
1. heterogeneous networks should increase eff_rank (principle 11: n_types=4 raises 35→38 at n=100)
2. lr_W=3-5E-3 should work (block 30's optimal range + block 4/13 heterogeneous adjustment)
3. lr_emb ceiling likely ~1E-3 at n=1000 (principle 31 at n=200; scale with n?)
4. batch=16 should remain efficient (100k dominates batch sensitivity, principle 8)
5. 2-3ep should suffice for dual convergence (principles 34, 95)
Strategy: spread lr_W from 3E-3 to 5E-3; test lr_emb=1E-3 vs 2E-3; batch=8 vs 16; 2-3 epochs.

### Batch 1 (iters 361-364) Strategy
| Slot | Role | lr_W | lr | lr_emb | L1 | batch | epochs | Parent | Rationale |
|------|------|------|----|--------|-----|-------|--------|--------|-----------|
| 0 | exploit | 3E-3 | 1E-4 | 1E-3 | 1E-5 | 16 | 2 | root | block 30 optimal lr_W + heterogeneous lr_emb |
| 1 | exploit | 5E-3 | 1E-4 | 1E-3 | 1E-5 | 16 | 2 | root | heterogeneous-optimal lr_W (block 4/13) |
| 2 | explore | 3E-3 | 1E-4 | 1E-3 | 1E-5 | 8 | 3 | root | batch=8 + 3ep reference |
| 3 | principle-test | 5E-3 | 1E-4 | 2E-3 | 1E-5 | 16 | 2 | root | testing principle 31: is lr_emb ceiling 1E-3 at n=1000? |
