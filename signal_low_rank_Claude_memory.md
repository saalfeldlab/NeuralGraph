# Working Memory: signal_low_rank (parallel)

## Knowledge Base (accumulated across all blocks)

### Regime Comparison Table
| Block | Regime | E/I | n_frames | n_neurons | n_types | noise | eff_rank | Best R² | Optimal lr_W | Optimal L1 | Degeneracy | Key finding |
| ----- | ------ | --- | -------- | --------- | ------- | ----- | -------- | ------- | ------------ | ---------- | ---------- | ----------- |

### Best Configurations Found

| Blk | Seed | lr_W | lr | L1 | edge_diff | n_ep_init | first_L1 | batch | n_epochs | conn_R2 | test_R2 | Finding |
| --- | ---- | ---- | -- | -- | --------- | --------- | -------- | ----- | -------- | ------- | ------- | ------- |
| 1 | 42 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 1.000 | 0.998 | best overall (seed=42) |
| 1 | 137 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 1.000 | 0.989 | best at seed=137 |
| 2 | 7 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 3 | 1.000 | 0.985 | 3 epochs closes gap |
| 3 | 99 | 5E-3 | 1E-4 | 1E-6 | 10000 | 2 | 0 | 8 | 2 | 1.000 | 0.992 | L1=1E-6 transforms seed=99 |
| 4 | 256 | 6E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 1.000 | 0.994 | lr_W=6E-3 transforms seed=256 |
| 4 | 314 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 1.000 | 0.990 | new seed, strong at standard recipe |
| 6 | 500 | 4E-3 | 1E-4 | 1E-6 | 15000 | 2 | 0 | 8 | 2 | 1.000 | 0.994 | edge_diff=15000 new seed=500 best |
| 7 | 2000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 3 | 1.000 | 0.991 | n_epochs=3 transforms seed=2000 |
| 12 | 1000 | 4E-3 | 1E-4 | 1E-6 | 10000 | 2 | 0 | 8 | 3 | 1.000 | 0.991 | BREAKTHROUGH — rescued from "unlearnable" |
| 12 | 8000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 3 | 1.000 | 0.999 | n_epochs=3 transforms seed=8000 |
| 11 | 4000 | 5E-3 | 1E-4 | 1E-6 | 20000 | 2 | 0 | 8 | 3 | 1.000 | 0.987 | L1=1E-6+edge_diff=20000+3ep triple combo |
| 10 | 6000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 3 | 1.000 | 0.991 | n_epochs=3 transforms seed=6000 |
| 11 | 7000 | 5E-3 | 1E-4 | 1E-6 | 20000 | 2 | 0 | 8 | 2 | 1.000 | 0.974 | edge_diff=20000+L1=1E-6 best for seed=7000 |
| 13 | 10000 | 6E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 1.000 | 0.980 | BEST STABLE — lr_W=6E-3+L1=1E-5 (range=0.022) |
| 9 | 5000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 1.000 | 0.953 | ABANDONED — fragile |
| 8 | 3000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 3 | 0.322 | 0.597 | UNLEARNABLE — 8 configs all failed |
| 6 | 9000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 0.326 | 0.571 | UNLEARNABLE — 7 configs all failed |
| 14 | 11000 | 5E-3 | 1E-4 | 1E-6 | 10000 | 2 | 0 | 8 | 3 | 1.000 | 0.977 | BEST — L1=1E-6+3ep combo, LOCKED |
| 15 | 13000 | 6E-3 | 1E-4 | 1E-6 | 10000 | 2 | 0 | 8 | 2 | 1.000 | 0.975 | NEW BEST — L1=1E-6+lr_W=6E-3 synergy (+0.040 from baseline) |
| 15 | 14000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 1.000 | 0.991 | standard recipe, top-tier seed |
| 13 | 12000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 0.309 | 0.527 | HARD SEED — V-recovery failure |
| 15 | 15000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 0.309 | 0.291 | UNLEARNABLE — 2 attempts, V-recovery failure |
| 15 | 16000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 0.335 | 0.292 | HARD SEED — V-recovery failure (V_R2=0.406) |
| 15 | 17000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 1.000 | 0.952 | new mid-tier seed, standard recipe |
| 15 | 18000 | 5E-3 | 1E-4 | 1E-6 | 10000 | 2 | 0 | 8 | 2 | 0.999 | 0.955 | L1=1E-6 helps (+0.006), LOCKED |
| 15 | 20000 | 5E-3 | 1E-4 | 1E-6 | 10000 | 2 | 0 | 8 | 2 | 1.000 | 0.993 | L1=1E-6 TRANSFORMS (+0.037) |
| 15 | 19000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 0.291 | 0.336 | HARD SEED — V-recovery failure |
| 16 | 21000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 1.000 | 0.999 | TOP-TIER new seed, standard recipe, LOCKED |
| 16 | 17000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 1.000 | 0.952 | LOCKED — no intervention helps |
| 16 | 22000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 1.000 | 0.987 | new upper mid-tier seed, standard recipe |
| 16 | 23000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 1.000 | 0.930 | new lower mid-tier seed, needs tuning |
| 16 | 24000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 1.000 | 0.680 | FRAGILE — NOT hard (V_R2=0.97), needs aggressive tuning |
| 16 | 25000 | 5E-3 | 1E-4 | 1E-6 | 10000 | 2 | 0 | 8 | 2 | 1.000 | 0.962 | L1=1E-6 transforms (+0.018), LOCKED |
| 17 | 24000 | 5E-3 | 1E-4 | 1E-6 | 10000 | 2 | 0 | 8 | 3 | 1.000 | 0.960 | MASSIVE RESCUE — L1=1E-6+3ep combo (+0.280 from 0.680) |
| 16 | 22000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 1.000 | 0.987 | LOCKED at standard — L1=1E-6 catastrophic (-0.112) |
| 17 | 23000 | 6E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 1.000 | 0.966 | NEW BEST — lr_W=6E-3 helps (+0.020 vs L1=1E-6), LOCKED |
| 17 | 26000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 1.000 | 0.966 | NEW upper mid-tier seed, candidate for L1=1E-6 |
| 17 | 27000 | 5E-3 | 1E-4 | 1E-6 | 10000 | 2 | 0 | 8 | 2 | 1.000 | 0.979 | L1=1E-6 TRANSFORMS (+0.082), NOW TOP-TIER, LOCKED |
| 17 | 26000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 1.000 | 0.966 | LOCKED at standard (L1=1E-6 neutral +0.002) |
| 17 | 28000 | - | - | - | - | - | - | - | - | 0.307 | 0.388 | HARD SEED — V-recovery failure (7th hard seed) |
| 18 | 29000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 1.000 | 0.991 | TOP-TIER new seed, standard recipe, LOCKED |
| 18 | 30000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 1.000 | 0.862 | LOWER MID-TIER — needs L1=1E-6+3ep rescue |
| 18 | 31000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 1.000 | 0.963 | UPPER MID-TIER — candidate for L1=1E-6 |
| 18 | 32000 | - | - | - | - | - | - | - | - | 0.338 | 0.408 | HARD SEED — V-recovery failure (8th hard seed) |
| 18 | 30000 | 5E-3 | 1E-4 | 1E-6 | 10000 | 2 | 0 | 8 | 3 | 1.000 | 0.955 | RESCUED (+0.093) — L1=1E-6+3ep combo, LOCKED |
| 18 | 31000 | 5E-3 | 1E-4 | 1E-6 | 10000 | 2 | 0 | 8 | 2 | 0.999 | 0.980 | L1=1E-6 BOOST (+0.017) — now top-tier, LOCKED |
| 18 | 33000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 1.000 | 0.909 | NEW lower mid-tier, needs L1=1E-6+3ep rescue |
| 18 | 34000 | - | - | - | - | - | - | - | - | 0.299 | 0.273 | HARD SEED — 9th hard (V_R2=0.382) |
| 18 | 33000 | 5E-3 | 1E-4 | 1E-6 | 10000 | 2 | 0 | 8 | 3 | 1.000 | 0.954 | RESCUED (+0.045) — L1=1E-6+3ep combo, LOCKED |
| 18 | 35000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 0.9999 | 0.964 | NEW UPPER MID-TIER — candidate for L1=1E-6 |
| 18 | 36000 | - | - | - | - | - | - | - | - | 0.304 | 0.440 | HARD SEED — 10th hard (V_R2=0.393) |
| 18 | 37000 | - | - | - | - | - | - | - | - | 0.312 | 0.331 | HARD SEED — 11th hard (V_R2=0.414) |
| 18 | 35000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 0.9999 | 0.964 | LOCKED at standard — L1=1E-6 hurts (-0.006) |
| 18 | 38000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 0.9999 | 0.900 | NEW LOWER MID-TIER — needs L1=1E-6+3ep rescue |
| 18 | 39000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 0.9998 | 0.951 | NEW UPPER MID-TIER — candidate for L1=1E-6 |
| 18 | 40000 | 5E-3 | 1E-4 | 1E-6 | 10000 | 2 | 0 | 8 | 3 | 0.9999 | 0.935 | RESCUED (+0.010) — L1=1E-6+3ep, LOCKED |
| 18 | 38000 | 5E-3 | 1E-4 | 1E-6 | 10000 | 2 | 0 | 8 | 3 | 0.9999 | 0.912 | RESCUED (+0.012) — L1=1E-6+3ep, LOCKED |
| 18 | 39000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 0.9998 | 0.951 | LOCKED at standard — L1=1E-6 hurts (-0.027) |
| n=200 | 18000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 0.989 | 0.972 | NEW n=200 LEARNABLE — candidate for L1=1E-6 |
| 19 | 41000 | 5E-3 | 1E-4 | 1E-6 | 10000 | 2 | 0 | 8 | 3 | 1.000 | 0.991 | RESCUED (+0.063) — L1=1E-6+3ep, NOW TOP-TIER, LOCKED |
| 19 | 42000 | - | - | - | - | - | - | - | - | 0.340 | 0.413 | HARD SEED — 12th hard (V_R2=0.415) |
| 19 | 43000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 0.9997 | 0.995 | NEW TOP-TIER — standard recipe, LOCKED |
| 19 | 44000 | - | - | - | - | - | - | - | - | 0.339 | 0.473 | HARD SEED — 13th hard (V_R2=0.405) |
| 20 | 45000 | - | - | - | - | - | - | - | - | 0.306 | 0.358 | HARD SEED — 14th hard (V_R2=0.392) |
| 20 | 46000 | - | - | - | - | - | - | - | - | 0.293 | 0.244 | HARD SEED — 15th hard (V_R2=0.377) |
| 20 | 47000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 0.9999 | 0.878 | NEW LOWER MID-TIER — needs rescue |
| 20 | 48000 | 5E-3 | 1E-4 | 1E-6 | 10000 | 2 | 0 | 8 | 3 | 0.9999 | 0.933 | MINOR RESCUE (+0.008), LOCKED |
| 20 | 47000 | 5E-3 | 1E-4 | 1E-6 | 10000 | 2 | 0 | 8 | 3 | 1.000 | 0.929 | RESCUED (+0.051), LOCKED |
| 20 | 49000 | - | - | - | - | - | - | - | - | 0.328 | 0.450 | HARD SEED — 16th (V_R2=0.401) |
| 20 | 50000 | 5E-3 | 1E-4 | 1E-6 | 10000 | 2 | 0 | 8 | 3 | 0.9999 | 0.995 | MASSIVE RESCUE (+0.170), NOW TOP-TIER, LOCKED |
| 20 | 51000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 0.9997 | 0.992 | NEW TOP-TIER seed, standard recipe, LOCKED |
| 20 | 52000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 0.9996 | 0.926 | NEW LOWER MID-TIER — needs L1=1E-6+3ep rescue |
| 20 | 53000 | 5E-3 | 1E-4 | 1E-6 | 10000 | 2 | 0 | 8 | 3 | 1.000 | 0.958 | RESCUED (+0.028) — L1=1E-6+3ep, LOCKED |
| 21 | 52000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 0.9996 | 0.926 | LOCKED at standard — L1=1E-6+3ep HURTS (-0.042) |
| 21 | 54000 | - | - | - | - | - | - | - | - | 0.343 | 0.419 | HARD SEED — 17th hard (V_R2=0.414) |
| 21 | 55000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 0.9998 | 0.972 | LOCKED at standard — L1=1E-6 hurts (-0.049) |
| 21 | 56000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 0.9999 | 0.932 | NEW UPPER MID-TIER — candidate for L1=1E-6+3ep |
| 21 | 57000 | 5E-3 | 1E-4 | 1E-6 | 10000 | 2 | 0 | 8 | 3 | 1.000 | 0.975 | MASSIVE RESCUE (+0.183) — L1=1E-6+3ep, LOCKED |
| 21 | 58000 | - | - | - | - | - | - | - | - | 0.348 | 0.350 | HARD SEED — 18th hard (V_R2=0.426) |
| 21 | 56000 | 5E-3 | 1E-4 | 1E-6 | 10000 | 2 | 0 | 8 | 3 | 0.9999 | 0.945 | marginal boost (+0.013), LOCKED |
| 21 | 59000 | - | - | - | - | - | - | - | - | 0.351 | 0.401 | HARD SEED — 19th hard (V_R2=0.421) |
| 21 | 60000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 0.9997 | 0.974 | NEW UPPER MID-TIER — standard recipe, LOCKED |
| 22 | 61000 | - | - | - | - | - | - | - | - | 0.358 | 0.319 | HARD SEED — 20th hard (V_R2=0.429) |
| 22 | 62000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 0.9999 | 0.866 | LOWER MID-TIER — needs L1=1E-6+3ep rescue |
| 22 | 63000 | - | - | - | - | - | - | - | - | 0.344 | 0.482 | HARD SEED — 21st hard (V_R2=0.424) |
| 22 | 64000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 0 | 8 | 2 | 0.970 | 0.709 | FRAGILE SEED — conn_R2=0.970, needs aggressive rescue |

### Established Principles

1. **lr_W is seed-dependent, not universally 5E-3** — seed=256: peak at 6E-3. seed=1000: 4E-3 transforms. seed=42: 6E-3 catastrophic. seed=14000: 6E-3 catastrophic (-0.156). seed=10000: 6E-3+L1=1E-5 gives 0.980. seed=13000: 6E-3 gives +0.023. seed=23000: 6E-3 helps (+0.020). seed=27000: 6E-3 neutral (+0.002). lr_W=5E-3 is optimal default but some seeds need 4E-3 or 6E-3. 38 data points
2. **L1 is seed-dependent** — L1=1E-6 helps seeds 99, 500, 7000, 4000, 8000, 1000, 10000, 11000, 13000, 20000, 23000, 24000, 25000, 27000, 38000, 40000, 41000, 53000, 57000 but HURTS seeds 2000, 137, 42, 5000, 14000, 22000, 35000, 39000, 52000, 55000. NEUTRAL at seeds 17000, 26000, 56000. L1=1E-5 is safe default; L1=1E-6 helps ~56% of tested seeds (19/35 helped, 10/35 hurt, 6/35 neutral). 56 data points
3. **lr=1E-4 is optimal** — lr=2E-4 degrades, lr=5E-5 catastrophic. 3 data points
4. **W recovery is trivially easy when lr_W+L1 are right** — most learnable seeds achieve conn_R2>=0.999. hard seeds (3000, 9000, 12000) stuck at conn_R2~0.31-0.37 — V-recovery failure. 92+ data points
5. **coeff_edge_diff is non-monotonic and seed-dependent** — 10000 safe default; 15000/20000 helps some mid-tier seeds. CATASTROPHIC at seeds 5000, 10000, 11000. 30 data points
6. **n_epochs effect is seed-dependent** — 3ep helps ~45% of seeds, hurts ~55%. 2ep is safe default. seed=13000 hurt by 3ep at BOTH L1=1E-5 (-0.041) AND L1=1E-6 (-0.094). seed=14000 hurt by 3ep (-0.068). seed=18000 hurt by 3ep (-0.187, severe). seed=23000 hurt by 3ep (-0.046). 33 data points
7. **seed is dominant factor BUT per-seed tuning closes gap** — 70 seeds tested: 21000(0.999)=8000(0.999) > 42(0.998) > 50000(0.995)=43000(0.995) > 256(0.994)=500(0.994) > 20000(0.993) > 99(0.992)=51000(0.992) > 14000(0.991)=1000(0.991)=6000(0.991)=2000(0.991)=41000(0.991)=29000(0.991) > 314(0.990) > 137(0.989) > 22000(0.987)=4000(0.987) > 7(0.985) > 31000(0.980)=10000(0.980) > 27000(0.979) > 11000(0.977) > 57000(0.975)=13000(0.975) > 60000(0.974)=7000(0.974) > 55000(0.972) > 23000(0.966)=26000(0.966) > 35000(0.964) > 25000(0.962) > 24000(0.960) > 53000(0.958) > 18000(0.955)=30000(0.955) > 33000(0.954) > 5000(0.953,ABAND) > 17000(0.952) > 39000(0.951) > 56000(0.945) > 40000(0.935) > 48000(0.933) > 47000(0.929) > 52000(0.926) > 38000(0.912) > 62000(0.866,NEW) > 64000(0.709,FRAGILE) >> 61000(0.319,HARD) > 63000(0.482,HARD) > 59000(0.401,HARD) > 44000(0.473,HARD) > 49000(0.450,HARD) > 36000(0.440,HARD) > 54000(0.419,HARD) > 42000(0.413,HARD) > 32000(0.408,HARD) > 28000(0.388,HARD) > 45000(0.358,HARD) > 58000(0.350,HARD) > 37000(0.331,HARD) > 19000(0.336,HARD) > 16000(0.292,HARD) > 15000(0.291,UNLEARN) > 34000(0.273,HARD) > 46000(0.244,HARD) > 3000(0.597,UNLEARN) > 9000(0.571,UNLEARN) > 12000(0.527,HARD). 264 data points
8. **lr_emb=5E-4 is fragile** — catastrophic at lr_W=4E-3. 1 data point
9. **batch_size=8 is the safe default** — batch=16 hurts at seeds 137(-0.091), 500(-0.125), 42(-0.069). universally harmful. 7 data points
10. **training_single_type has no effect** — 2 data points
11. **two-phase training structure matters** — n_epochs_init=3 with n_epochs=3 eliminates L1 phase and hurts. 5 data points
12. **L1 landscape is a cliff** — no useful intermediate L1 between 1E-6 and 1E-5. 4 data points
13. **rescue recipe lr_W=4E-3+L1=1E-6+3ep is seed=1000-specific** — failed at 3000, 9000, 5000. 4 data points
14. **seed=5000 ABANDONED** — only works at exact standard recipe (0.953). 8 data points
15. **recurrent training (time_step=4) hurts dynamics** — 1 data point
16. **edge_diff + n_epochs interaction** — higher edge_diff requires fewer epochs. 3 data points
17. **L1=1E-6+edge_diff=20000+3ep triple combo is seed-specific** — works for seed=4000, hurts seed=11000 (-0.011). 2 data points
18. **hard seeds share V-recovery failure** — seeds 3000, 9000, 12000, 15000, 16000, 19000, 28000, 32000, 34000, 36000, 37000, 42000, 44000, 45000, 46000, 49000, 54000, 58000, 59000, 61000, 63000 all show U_R2~0.94-0.97, V_R2~0.38-0.43, conn_R2~0.29-0.37. fundamentally different failure mode. 21 hard seeds identified out of 70 tested (30.0%). 36 data points
19. **seed=10000 has HIGH stochastic variance** — lr_W=6E-3+L1=1E-5 is MORE STABLE (range=0.022 vs L1=1E-6 range=0.052). 4 data points

### Open Questions

1. is there a universal "enhanced recipe" that beats standard for multiple seeds? answer: NO — per-seed tuning is required.
2. what makes seeds 3000/9000/12000/15000/16000/19000 fundamentally different? V-recovery failure. no known intervention.
3. seed=17000: RESOLVED — LOCKED at standard (0.952). no intervention helps.
4. seed=18000: RESOLVED — L1=1E-6 transforms to 0.955, LOCKED.
5. seed=20000: RESOLVED — L1=1E-6 transforms to 0.993, now top-tier.
6. seed=16000: declare unlearnable (1 attempt failed, V-recovery failure).
7. seed=12000: declare unlearnable or 1 more rescue attempt?
8. FINAL: 70 seeds tested, 49 learnable (70.0%), 21 hard (30.0%), 1 abandoned (5000).
9. seeds 62000 (0.866) and 64000 (0.709) identified as learnable but no time for rescue optimization.
10. EXPLORATION COMPLETE at 256 iterations.

---

## Previous Block Summary (Block 21)

block 21 (12 iters, 241-252): seed=57000 MASSIVELY RESCUED (+0.183 to 0.975) via L1=1E-6+3ep — largest rescue since seed=50000! seed=53000 RESCUED (+0.028 to 0.958). seed=60000 NEW UPPER MID-TIER at standard (0.974). seeds 54000, 58000, 59000 HARD (V-recovery failure). 5/12 converged (42% success). 66 seeds tested: 47 learnable (71.2%), 19 hard (28.8%).

## Block 22 Summary (FINAL)

block 22 (4 iters, 253-256): FINAL BATCH. 2/4 learnable (50%). seed=61000 HARD (V_R2=0.429). seed=62000 LOWER MID-TIER (0.866, needs rescue). seed=63000 HARD (V_R2=0.424). seed=64000 FRAGILE (0.709, conn_R2=0.970, needs aggressive rescue). no time for rescue optimization. EXPLORATION COMPLETE: 70 seeds tested, 49 learnable (70.0%), 21 hard (30.0%).

---

## n=100 Exploration Summary (Blocks 1-22, 256 iterations)

- 70 seeds tested: 49 learnable (70.0%), 21 hard (30.0%)
- hard seed rate: 30.0% — V-recovery failure (V_R2<0.43) is the definitive signature
- standard recipe (lr_W=5E-3, L1=1E-5, 2ep) works for ~72% of seeds
- L1=1E-6+3ep rescue combo transforms ~54% of lower mid-tier seeds
- per-seed tuning required — no universal enhanced recipe

---

## Current Block (Block 23) — n=200 Exploration

### Block Info

NEW REGIME: n_neurons=200, low_rank=20, n_frames=10000
Goal: Compare n=100 vs n=200 low_rank regimes
Continues from iteration 257

### Hypothesis

Based on landscape exploration principles:
1. n=200 should have higher eff_rank than n=100 (~24 vs ~12) — more independent modes
2. optimal lr_W may shift higher (n=200 chaotic was 8E-3 vs n=100's 4E-3)
3. hard seed rate may be LOWER due to more information in the data
4. L1=1E-6 may be less critical if eff_rank is higher

Starting with seeds 42, 137, 7, 99 (known from n=100 exploration) to enable direct comparison.

### Batch 1 (iters 257-260) Strategy

Test standard recipe at n=200 with known seeds to establish baseline.

| Slot | Seed | lr_W | lr | L1 | edge_diff | n_ep_init | batch | n_epochs | Rationale |
|------|------|------|----|-----|-----------|-----------|-------|----------|-----------|
| 0 | 42 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 8 | 2 | standard recipe, best n=100 seed |
| 1 | 137 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 8 | 2 | standard recipe |
| 2 | 7 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 8 | 2 | standard recipe |
| 3 | 99 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | 8 | 2 | standard recipe |

### Iterations This Block

## Iter 257: converged
Node: id=257, parent=root
Mode/Strategy: baseline
Config: seed=42, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.991, test_pearson=0.985, connectivity_R2=0.970, cluster_accuracy=1.000, final_loss=9.937E+02, kino_R2=0.990, kino_SSIM=0.968, kino_WD=0.116
Activity: n=200 low-rank, U_R2=0.988, V_R2=0.963
Mutation: n_neurons: 100 -> 200, standard recipe at known seed=42
Observation: CONVERGED but test_R2=0.991 < n=100's 0.998. n=200 is HARDER.

## Iter 258: failed
Node: id=258, parent=root
Mode/Strategy: baseline
Config: seed=137, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.450, test_pearson=0.056, connectivity_R2=0.055, cluster_accuracy=1.000, final_loss=3.494E+03
Activity: n=200 low-rank, U_R2=0.966, V_R2=0.105 — V-RECOVERY FAILURE!
Mutation: n_neurons: 100 -> 200, standard recipe at known seed=137
Observation: HARD at n=200! Was LEARNABLE at n=100 (0.989). CRITICAL FINDING.

## Iter 259: converged
Node: id=259, parent=root
Mode/Strategy: baseline
Config: seed=7, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.995, test_pearson=0.993, connectivity_R2=0.916, cluster_accuracy=1.000, final_loss=8.304E+02
Activity: n=200 low-rank, U_R2=0.986, V_R2=0.912
Mutation: n_neurons: 100 -> 200, standard recipe at known seed=7
Observation: CONVERGED. test_R2=0.995 vs n=100's 0.985 — BETTER at n=200!

## Iter 260: converged
Node: id=260, parent=root
Mode/Strategy: baseline
Config: seed=99, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.998, test_pearson=0.996, connectivity_R2=0.954, cluster_accuracy=1.000, final_loss=8.571E+02
Activity: n=200 low-rank, U_R2=0.987, V_R2=0.946
Mutation: n_neurons: 100 -> 200, standard recipe at known seed=99
Observation: CONVERGED. test_R2=0.998 vs n=100's 0.992 — BETTER at n=200!

### Emerging Observations

n=200 vs n=100 comparison (batch 1, 4 seeds):
1. **seed=137 FLIPPED**: learnable at n=100 (0.989) → HARD at n=200 (V_R2=0.105). CRITICAL.
2. **conn_R2 universally harder**: n=200 gives 0.916-0.970 vs n=100's 1.000
3. **test_R2 mixed**: seeds 7,99 BETTER at n=200; seed 42 WORSE
4. **V-recovery failure persists** at n=200 — same failure mode as n=100
5. **hard seed rate may be HIGHER** at n=200 (1/4=25% vs n=100's 30%)

### Batch 2 Results (iters 261-264)

## Iter 261: failed
Node: id=261, parent=258
Mode/Strategy: rescue
Config: seed=137, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-6, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=3, recurrent=F, time_step=1
Metrics: test_R2=0.538, test_pearson=0.049, connectivity_R2=0.072, cluster_accuracy=1.000, final_loss=2.712E+03, kino_R2=-13.13, kino_SSIM=0.452, kino_WD=4.385
Activity: n=200 low-rank, U_R2=0.968, V_R2=0.133
Mutation: L1: 1E-5 -> 1E-6, n_epochs: 2 -> 3 (rescue combo)
Observation: FAILED. L1=1E-6+3ep rescue combo DOES NOT WORK at n=200. seed=137 UNLEARNABLE at n=200.

## Iter 262: failed
Node: id=262, parent=257
Mode/Strategy: exploit
Config: seed=42, lr_W=6E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.582, test_pearson=0.274, connectivity_R2=0.014, cluster_accuracy=1.000, final_loss=2.124E+03, kino_R2=-2.78, kino_SSIM=0.479, kino_WD=3.608
Activity: n=200 low-rank, U_R2=0.966, V_R2=0.071
Mutation: lr_W: 5E-3 -> 6E-3
Observation: CATASTROPHIC. lr_W=6E-3 DESTROYS n=200 seed=42. n=200 is MORE SENSITIVE to lr_W.

## Iter 263: converged
Node: id=263, parent=259
Mode/Strategy: exploit
Config: seed=7, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-6, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.997, test_pearson=0.994, connectivity_R2=0.964, cluster_accuracy=1.000, final_loss=8.115E+02, kino_R2=0.996, kino_SSIM=0.986, kino_WD=0.057
Activity: n=200 low-rank, U_R2=0.988, V_R2=0.957
Mutation: L1: 1E-5 -> 1E-6
Observation: BOOST! L1=1E-6 improves n=200 seed=7: test_R2 +0.002, conn_R2 +0.048. L1=1E-6 WORKS at n=200.

## Iter 264: converged
Node: id=264, parent=root
Mode/Strategy: explore
Config: seed=256, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.985, test_pearson=0.976, connectivity_R2=0.933, cluster_accuracy=1.000, final_loss=7.528E+02, kino_R2=0.984, kino_SSIM=0.952, kino_WD=0.148
Activity: n=200 low-rank, U_R2=0.988, V_R2=0.926
Mutation: seed: new -> 256
Observation: CONVERGED. seed=256 at n=200 test_R2=0.985 < n=100's 0.994. n=200 systematically harder.

### Batch 3 Results (iters 265-268)

## Iter 265: converged
Node: id=265, parent=257
Mode/Strategy: exploit
Config: seed=42, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-6, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.997, test_pearson=0.995, connectivity_R2=0.970, cluster_accuracy=1.000, final_loss=8.692E+02, kino_R2=0.997, kino_SSIM=0.988, kino_WD=0.049
Activity: n=200 low-rank, U_R2=0.988, V_R2=0.961
Mutation: L1: 1E-5 -> 1E-6
Observation: BOOST! L1=1E-6 improves test_R2 0.991->0.997 (+0.006). conn_R2 unchanged. L1=1E-6 universally helpful.

## Iter 266: converged
Node: id=266, parent=260
Mode/Strategy: exploit
Config: seed=99, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-6, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=1.000, test_pearson=1.000, connectivity_R2=0.950, cluster_accuracy=1.000, final_loss=8.385E+02, kino_R2=1.000, kino_SSIM=0.999, kino_WD=0.021
Activity: n=200 low-rank, U_R2=0.987, V_R2=0.942
Mutation: L1: 1E-5 -> 1E-6
Observation: NEAR-PERFECT! test_R2=0.9997 (rounds to 1.000). conn_R2 dropped 0.954->0.950.

## Iter 267: converged
Node: id=267, parent=264
Mode/Strategy: exploit
Config: seed=256, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-6, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.997, test_pearson=0.995, connectivity_R2=0.966, cluster_accuracy=1.000, final_loss=8.116E+02, kino_R2=0.997, kino_SSIM=0.990, kino_WD=0.061
Activity: n=200 low-rank, U_R2=0.988, V_R2=0.959
Mutation: L1: 1E-5 -> 1E-6
Observation: MAJOR BOOST! test_R2 +0.012, conn_R2 +0.033, V_R2 +0.033. Largest improvement so far.

## Iter 268: converged
Node: id=268, parent=root
Mode/Strategy: explore
Config: seed=314, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.997, test_pearson=0.995, connectivity_R2=0.985, cluster_accuracy=1.000, final_loss=7.949E+02, kino_R2=0.997, kino_SSIM=0.989, kino_WD=0.074
Activity: n=200 low-rank, U_R2=0.988, V_R2=0.976
Mutation: seed: new -> 314
Observation: TOP-TIER at standard! BEST conn_R2 (0.985) and V_R2 (0.976) at n=200. Some seeds prefer L1=1E-5.

### Batch 4 Results (iters 269-272)

## Iter 269: partial
Node: id=269, parent=268
Mode/Strategy: exploit
Config: seed=314, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-6, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.920, test_pearson=0.897, connectivity_R2=0.935, cluster_accuracy=1.000, final_loss=7.978E+02, kino_R2=0.916, kino_SSIM=0.839, kino_WD=0.268
Activity: n=200 low-rank, U_R2=0.987, V_R2=0.929
Mutation: L1: 1E-5 -> 1E-6
Observation: L1=1E-6 HURTS seed=314! test_R2 0.997->0.920 (-0.077). seed=314 LOCKED at L1=1E-5.

## Iter 270: converged
Node: id=270, parent=root
Mode/Strategy: explore
Config: seed=500, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.996, test_pearson=0.994, connectivity_R2=0.948, cluster_accuracy=1.000, final_loss=8.152E+02, kino_R2=0.996, kino_SSIM=0.986, kino_WD=0.081
Activity: n=200 low-rank, U_R2=0.987, V_R2=0.941
Mutation: seed: new -> 500
Observation: CONVERGED. 7th learnable seed at n=200. candidate for L1=1E-6 test.

## Iter 271: failed
Node: id=271, parent=root
Mode/Strategy: explore
Config: seed=1000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.296, test_pearson=0.036, connectivity_R2=0.167, cluster_accuracy=1.000, final_loss=4.682E+03
Activity: n=200 low-rank, U_R2=0.971, V_R2=0.210 — V-RECOVERY FAILURE!
Mutation: seed: new -> 1000
Observation: HARD at n=200! Was RESCUED at n=100 (0.991). 2nd n=100 FLIP!

## Iter 272: converged
Node: id=272, parent=root
Mode/Strategy: explore
Config: seed=2000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.995, test_pearson=0.992, connectivity_R2=0.989, cluster_accuracy=1.000, final_loss=8.830E+02, kino_R2=0.995, kino_SSIM=0.982, kino_WD=0.079
Activity: n=200 low-rank, U_R2=0.988, V_R2=0.979
Mutation: seed: new -> 2000
Observation: BEST conn_R2 (0.989) and V_R2 (0.979) at n=200! candidate for L1=1E-6 test.

### Batch 5 Results (iters 273-276)

## Iter 273: converged
Node: id=273, parent=270
Mode/Strategy: exploit
Config: seed=500, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-6, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.9996, test_pearson=0.9991, connectivity_R2=0.958, cluster_accuracy=1.000, final_loss=8.154E+02, kino_R2=0.9995, kino_SSIM=0.998, kino_WD=0.032
Activity: n=200 low-rank, U_R2=0.987, V_R2=0.951
Mutation: L1: 1E-5 -> 1E-6
Observation: L1=1E-6 BOOSTS seed=500: test_R2 +0.004, conn_R2 +0.010. now TOP-TIER (0.9996)!

## Iter 274: converged
Node: id=274, parent=272
Mode/Strategy: exploit
Config: seed=2000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-6, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.9992, test_pearson=0.9989, connectivity_R2=0.975, cluster_accuracy=1.000, final_loss=8.665E+02, kino_R2=0.9992, kino_SSIM=0.996, kino_WD=0.033
Activity: n=200 low-rank, U_R2=0.988, V_R2=0.966
Mutation: L1: 1E-5 -> 1E-6
Observation: L1=1E-6 HURTS W-recovery! conn_R2 0.989->0.975 (-0.014). seed=2000 LOCKED at L1=1E-5.

## Iter 275: failed
Node: id=275, parent=root
Mode/Strategy: explore
Config: seed=3000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.342, test_pearson=-0.088, connectivity_R2=0.162, cluster_accuracy=1.000, final_loss=4.669E+03
Activity: n=200 low-rank, U_R2=0.970, V_R2=0.208 — V-RECOVERY FAILURE!
Mutation: seed: new -> 3000
Observation: HARD at n=200! Was UNLEARNABLE at n=100 (0.597) — SAME FAILURE MODE ACROSS n.

## Iter 276: converged
Node: id=276, parent=root
Mode/Strategy: explore
Config: seed=4000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.995, test_pearson=0.993, connectivity_R2=0.964, cluster_accuracy=1.000, final_loss=8.056E+02
Activity: n=200 low-rank, U_R2=0.988, V_R2=0.956
Mutation: seed: new -> 4000
Observation: NEW LEARNABLE at n=200! Was complex combo at n=100. Standard recipe works!

### Batch 6 Results (iters 277-280)

## Iter 277: converged
Node: id=277, parent=276
Mode/Strategy: exploit
Config: seed=4000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-6, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.980, test_pearson=0.968, connectivity_R2=0.954, cluster_accuracy=1.000, final_loss=1.258E+03, kino_R2=0.979, kino_SSIM=0.944, kino_WD=0.172
Activity: n=200 low-rank, U_R2=0.988, V_R2=0.948
Mutation: L1: 1E-5 -> 1E-6
Observation: L1=1E-6 HURTS seed=4000! test_R2 0.995->0.980 (-0.015), conn_R2 0.964->0.954 (-0.010). seed=4000 LOCKED at L1=1E-5.

## Iter 278: failed
Node: id=278, parent=root
Mode/Strategy: explore
Config: seed=5000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.326, test_pearson=0.025, connectivity_R2=0.035, cluster_accuracy=1.000, final_loss=3.335E+03
Activity: n=200 low-rank, U_R2=0.969, V_R2=0.095 — V-RECOVERY FAILURE!
Mutation: seed: new -> 5000 (was ABANDONED at n=100)
Observation: HARD at n=200! seed=5000 was ABANDONED at n=100 (0.953) — now UNLEARNABLE (V_R2=0.095).

## Iter 279: failed
Node: id=279, parent=root
Mode/Strategy: explore
Config: seed=6000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.454, test_pearson=0.152, connectivity_R2=0.023, cluster_accuracy=1.000, final_loss=3.216E+03
Activity: n=200 low-rank, U_R2=0.962, V_R2=0.086 — V-RECOVERY FAILURE!
Mutation: seed: new -> 6000 (was LEARNABLE at n=100, 0.991)
Observation: CRITICAL FLIP! seed=6000 was LEARNABLE at n=100 (0.991) → HARD at n=200 (V_R2=0.086). 3rd n=100 FLIP!

## Iter 280: converged
Node: id=280, parent=root
Mode/Strategy: explore
Config: seed=7000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.998, test_pearson=0.997, connectivity_R2=0.980, cluster_accuracy=1.000, final_loss=7.330E+02, kino_R2=0.998, kino_SSIM=0.992, kino_WD=0.048
Activity: n=200 low-rank, U_R2=0.988, V_R2=0.969
Mutation: seed: new -> 7000 (was mid-tier at n=100, 0.974)
Observation: TOP-TIER at n=200! test_R2=0.998, conn_R2=0.980. BEST W-recovery! seed=7000 IMPROVED from n=100.

### Batch 7 Results (iters 281-284)

## Iter 281: converged
Node: id=281, parent=280
Mode/Strategy: exploit
Config: seed=7000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-6, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.9988, test_pearson=0.9978, connectivity_R2=0.9722, cluster_accuracy=1.000, final_loss=7.645E+02, kino_R2=0.9987, kino_SSIM=0.9951, kino_WD=0.0401
Activity: n=200 low-rank, U_R2=0.988, V_R2=0.963
Mutation: L1: 1E-5 -> 1E-6
Observation: L1=1E-6 HURTS W-recovery! conn_R2 0.980->0.972 (-0.008). seed=7000 LOCKED at L1=1E-5.

## Iter 282: converged
Node: id=282, parent=root
Mode/Strategy: explore
Config: seed=8000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.9981, test_pearson=0.9967, connectivity_R2=0.9533, cluster_accuracy=1.000, final_loss=8.412E+02, kino_R2=0.9980, kino_SSIM=0.9916, kino_WD=0.0426
Activity: n=200 low-rank, U_R2=0.988, V_R2=0.946
Mutation: seed: new -> 8000
Observation: NEW TOP-TIER seed at n=200! test_R2=0.998. CONSISTENT with n=100 (0.999).

## Iter 283: failed
Node: id=283, parent=root
Mode/Strategy: explore
Config: seed=9000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.3126, test_pearson=-0.0200, connectivity_R2=0.0463, cluster_accuracy=1.000, final_loss=3.882E+03
Activity: n=200 low-rank, U_R2=0.965, V_R2=0.104 — V-RECOVERY FAILURE!
Mutation: seed: new -> 9000 (was UNLEARNABLE at n=100)
Observation: HARD at n=200! CONSISTENT failure mode across n (V_R2~0.10).

## Iter 284: converged
Node: id=284, parent=root
Mode/Strategy: explore
Config: seed=10000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.9958, test_pearson=0.9938, connectivity_R2=0.9648, cluster_accuracy=1.000, final_loss=8.478E+02, kino_R2=0.9957, kino_SSIM=0.9830, kino_WD=0.0781
Activity: n=200 low-rank, U_R2=0.988, V_R2=0.957
Mutation: seed: new -> 10000
Observation: NEW UPPER MID-TIER at n=200! test_R2=0.996. candidate for L1=1E-6 test.

### Batch 8 Results (iters 285-288)

## Iter 285: converged
Node: id=285, parent=282
Mode/Strategy: exploit
Config: seed=8000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-6, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.9950, test_pearson=0.9921, connectivity_R2=0.9255, cluster_accuracy=1.000, final_loss=8.239E+02, kino_R2=0.9948, kino_SSIM=0.9817, kino_WD=0.0833
Activity: n=200 low-rank, U_R2=0.987, V_R2=0.920
Mutation: L1: 1E-5 -> 1E-6
Observation: L1=1E-6 HURTS seed=8000! test_R2 -0.003, conn_R2 -0.028. seed=8000 LOCKED at L1=1E-5.

## Iter 286: converged
Node: id=286, parent=284
Mode/Strategy: exploit
Config: seed=10000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-6, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.9974, test_pearson=0.9950, connectivity_R2=0.9713, cluster_accuracy=1.000, final_loss=8.367E+02, kino_R2=0.9971, kino_SSIM=0.9900, kino_WD=0.0562
Activity: n=200 low-rank, U_R2=0.988, V_R2=0.962
Mutation: L1: 1E-5 -> 1E-6
Observation: L1=1E-6 BOOSTS seed=10000! test_R2 +0.002, conn_R2 +0.007. NOW TOP-TIER. LOCKED at L1=1E-6.

## Iter 287: converged
Node: id=287, parent=root
Mode/Strategy: explore
Config: seed=11000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.9992, test_pearson=0.9987, connectivity_R2=0.9680, cluster_accuracy=1.000, final_loss=8.130E+02, kino_R2=0.9992, kino_SSIM=0.9963, kino_WD=0.0303
Activity: n=200 low-rank, U_R2=0.988, V_R2=0.959
Mutation: seed: new -> 11000 (was MID-TIER at n=100, 0.977)
Observation: NEW TOP-TIER at n=200! test_R2=0.999. IMPROVED from n=100 (0.977 -> 0.999). candidate for L1=1E-6.

## Iter 288: failed
Node: id=288, parent=root
Mode/Strategy: explore
Config: seed=12000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.4198, test_pearson=-0.0076, connectivity_R2=0.1543, cluster_accuracy=1.000, final_loss=4.225E+03
Activity: n=200 low-rank, U_R2=0.971, V_R2=0.192 — V-RECOVERY FAILURE!
Mutation: seed: new -> 12000 (was HARD at n=100)
Observation: HARD at n=200! CONSISTENT with n=100. 7th hard seed at n=200.

### Batch 9 Results (iters 289-292)

## Iter 289: converged
Node: id=289, parent=287
Mode/Strategy: exploit
Config: seed=11000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-6, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.993, test_pearson=0.986, connectivity_R2=0.993, cluster_accuracy=1.000, final_loss=8.217E+02, kino_R2=0.992, kino_SSIM=0.976, kino_WD=0.091
Activity: n=200 low-rank, U_R2=0.988, V_R2=0.983
Mutation: L1: 1E-5 -> 1E-6
Observation: L1=1E-6 MIXED! test_R2 0.999->0.993 (-0.006), conn_R2 0.968->0.993 (+0.025). TRADEOFF. LOCKED at L1=1E-5 for dynamics.

## Iter 290: converged
Node: id=290, parent=root
Mode/Strategy: explore
Config: seed=13000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.996, test_pearson=0.993, connectivity_R2=0.921, cluster_accuracy=1.000, final_loss=9.414E+02, kino_R2=0.996, kino_SSIM=0.984, kino_WD=0.045
Activity: n=200 low-rank, U_R2=0.987, V_R2=0.918
Mutation: seed: new -> 13000
Observation: NEW UPPER MID-TIER at n=200! test_R2=0.996. was BEST L1=1E-6 combo at n=100 (0.975). candidate for L1=1E-6 test.

## Iter 291: converged
Node: id=291, parent=root
Mode/Strategy: explore
Config: seed=14000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.993, test_pearson=0.989, connectivity_R2=0.923, cluster_accuracy=1.000, final_loss=7.706E+02, kino_R2=0.993, kino_SSIM=0.976, kino_WD=0.118
Activity: n=200 low-rank, U_R2=0.987, V_R2=0.917
Mutation: seed: new -> 14000
Observation: NEW UPPER MID-TIER at n=200! test_R2=0.993. was TOP-TIER at n=100 (0.991). VERY CONSISTENT cross-n.

## Iter 292: failed
Node: id=292, parent=root
Mode/Strategy: explore
Config: seed=15000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.601, test_pearson=0.187, connectivity_R2=0.224, cluster_accuracy=1.000, final_loss=4.552E+03, kino_R2=-194.2, kino_SSIM=0.667, kino_WD=20.83
Activity: n=200 low-rank, U_R2=0.971, V_R2=0.263 — V-RECOVERY FAILURE!
Mutation: seed: new -> 15000 (was UNLEARNABLE at n=100)
Observation: HARD at n=200! CONSISTENT with n=100 (also V-recovery failure). 8th hard seed at n=200.

### Batch 10 Results (iters 293-296)

## Iter 293: converged
Node: id=293, parent=290
Mode/Strategy: exploit
Config: seed=13000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-6, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.967, test_pearson=0.949, connectivity_R2=0.977, cluster_accuracy=1.000, final_loss=9.056E+02, kino_R2=0.966, kino_SSIM=0.916, kino_WD=0.188
Activity: n=200 low-rank, U_R2=0.988, V_R2=0.968
Mutation: L1: 1E-5 -> 1E-6
Observation: L1=1E-6 MIXED! test_R2 -0.029, conn_R2 +0.056. seed=13000 LOCKED at L1=1E-5 for dynamics.

## Iter 294: converged
Node: id=294, parent=291
Mode/Strategy: exploit
Config: seed=14000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-6, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.988, test_pearson=0.982, connectivity_R2=0.892, cluster_accuracy=1.000, final_loss=7.965E+02, kino_R2=0.987, kino_SSIM=0.958, kino_WD=0.156
Activity: n=200 low-rank, U_R2=0.987, V_R2=0.889
Mutation: L1: 1E-5 -> 1E-6
Observation: L1=1E-6 HURTS! conn_R2 -0.031, test_R2 -0.005. seed=14000 LOCKED at L1=1E-5.

## Iter 295: failed
Node: id=295, parent=root
Mode/Strategy: explore
Config: seed=16000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.378, test_pearson=0.157, connectivity_R2=0.167, cluster_accuracy=1.000, final_loss=4.606E+03
Activity: n=200 low-rank, U_R2=0.973, V_R2=0.214 — V-RECOVERY FAILURE!
Mutation: seed: new -> 16000 (was HARD at n=100)
Observation: HARD at n=200! CONSISTENT with n=100. 9th hard seed at n=200.

## Iter 296: failed
Node: id=296, parent=root
Mode/Strategy: explore
Config: seed=17000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.378, test_pearson=0.131, connectivity_R2=0.052, cluster_accuracy=1.000, final_loss=4.532E+03
Activity: n=200 low-rank, U_R2=0.969, V_R2=0.108 — V-RECOVERY FAILURE!
Mutation: seed: new -> 17000 (was LEARNABLE at n=100, 0.952!)
Observation: CRITICAL FLIP! seed=17000 FLIPPED from LEARNABLE at n=100 → HARD at n=200. 4th n=100 FLIP!

### Batch 11 Results (iters 297-300)

## Iter 297: converged
Node: id=297, parent=root
Mode/Strategy: explore
Config: seed=18000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.972, test_pearson=0.956, connectivity_R2=0.989, cluster_accuracy=1.000, final_loss=8.831E+02, kino_R2=0.971, kino_SSIM=0.929, kino_WD=0.200
Activity: n=200 low-rank, U_R2=0.988, V_R2=0.979
Mutation: seed: new -> 18000
Observation: NEW LEARNABLE at n=200! BEST conn_R2=0.989. Candidate for L1=1E-6 test.

## Iter 298: failed
Node: id=298, parent=root
Mode/Strategy: explore
Config: seed=19000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.504, test_pearson=0.133, connectivity_R2=0.128, cluster_accuracy=1.000, final_loss=4.098E+03
Activity: n=200 low-rank, U_R2=0.972, V_R2=0.172 — V-RECOVERY FAILURE!
Mutation: seed: new -> 19000 (was HARD at n=100)
Observation: HARD at n=200! CONSISTENT hard across n.

## Iter 299: failed
Node: id=299, parent=root
Mode/Strategy: explore
Config: seed=20000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.490, test_pearson=0.350, connectivity_R2=0.037, cluster_accuracy=1.000, final_loss=3.326E+03
Activity: n=200 low-rank, U_R2=0.968, V_R2=0.096 — V-RECOVERY FAILURE!
Mutation: seed: new -> 20000 (was TOP-TIER at n=100, 0.993!)
Observation: CRITICAL FLIP! 5th n=100 FLIP!

## Iter 300: failed
Node: id=300, parent=root
Mode/Strategy: explore
Config: seed=21000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.591, test_pearson=0.256, connectivity_R2=0.010, cluster_accuracy=1.000, final_loss=3.320E+03
Activity: n=200 low-rank, U_R2=0.964, V_R2=0.071 — V-RECOVERY FAILURE!
Mutation: seed: new -> 21000 (was TOP-TIER at n=100, 0.999!)
Observation: CRITICAL FLIP! 6th n=100 FLIP! MOST DRAMATIC!

### Batch 12 Results (iters 301-304)

## Iter 301: converged
Node: id=301, parent=297
Mode/Strategy: exploit
Config: seed=18000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-6, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.993, test_pearson=0.989, connectivity_R2=0.929, cluster_accuracy=1.000, final_loss=8.441E+02, kino_R2=0.993, kino_SSIM=0.975, kino_WD=0.101
Activity: n=200 low-rank, U_R2=0.987, V_R2=0.925
Mutation: L1: 1E-5 -> 1E-6
Observation: L1=1E-6 MIXED! test_R2 +0.021 (0.972->0.993), conn_R2 -0.060 (0.989->0.929). LOCKED at L1=1E-5.

## Iter 302: converged
Node: id=302, parent=root
Mode/Strategy: explore
Config: seed=22000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.992, test_pearson=0.986, connectivity_R2=0.962, cluster_accuracy=1.000, final_loss=8.181E+02, kino_R2=0.991, kino_SSIM=0.971, kino_WD=0.106
Activity: n=200 low-rank, U_R2=0.987, V_R2=0.955
Mutation: seed: new -> 22000
Observation: NEW UPPER MID-TIER at n=200! test_R2=0.992, conn_R2=0.962. candidate for L1=1E-6.

## Iter 303: converged
Node: id=303, parent=root
Mode/Strategy: explore
Config: seed=23000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.995, test_pearson=0.992, connectivity_R2=0.957, cluster_accuracy=1.000, final_loss=7.584E+02, kino_R2=0.995, kino_SSIM=0.980, kino_WD=0.081
Activity: n=200 low-rank, U_R2=0.988, V_R2=0.948
Mutation: seed: new -> 23000
Observation: NEW TOP-TIER at n=200! test_R2=0.995. candidate for L1=1E-6 test.

## Iter 304: converged
Node: id=304, parent=root
Mode/Strategy: explore
Config: seed=24000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.996, test_pearson=0.994, connectivity_R2=0.985, cluster_accuracy=1.000, final_loss=7.993E+02, kino_R2=0.996, kino_SSIM=0.986, kino_WD=0.066
Activity: n=200 low-rank, U_R2=0.988, V_R2=0.975
Mutation: seed: new -> 24000
Observation: NEW TOP-TIER at n=200! BEST conn_R2=0.985, V_R2=0.975 at n=200!

### Batch 13 Results (iters 305-308)

## Iter 305: converged
Node: id=305, parent=304
Mode/Strategy: exploit
Config: seed=24000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-6, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.968, test_pearson=0.952, connectivity_R2=0.973, cluster_accuracy=1.000, final_loss=8.136E+02, kino_R2=0.967, kino_SSIM=0.921, kino_WD=0.222
Activity: n=200 low-rank, U_R2=0.988, V_R2=0.964
Mutation: L1: 1E-5 -> 1E-6
Observation: L1=1E-6 HURTS seed=24000! test_R2 0.996->0.968 (-0.028), conn_R2 0.985->0.973 (-0.012). seed=24000 LOCKED at L1=1E-5.

## Iter 306: converged
Node: id=306, parent=303
Mode/Strategy: exploit
Config: seed=23000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-6, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.994, test_pearson=0.990, connectivity_R2=0.963, cluster_accuracy=1.000, final_loss=7.632E+02, kino_R2=0.993, kino_SSIM=0.978, kino_WD=0.094
Activity: n=200 low-rank, U_R2=0.988, V_R2=0.954
Mutation: L1: 1E-5 -> 1E-6
Observation: L1=1E-6 NEUTRAL at seed=23000. test_R2 0.995->0.994 (-0.001), conn_R2 0.957->0.963 (+0.006). LOCKED at L1=1E-5 (marginal).

## Iter 307: converged
Node: id=307, parent=302
Mode/Strategy: exploit
Config: seed=22000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-6, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.980, test_pearson=0.968, connectivity_R2=0.955, cluster_accuracy=1.000, final_loss=8.140E+02, kino_R2=0.980, kino_SSIM=0.946, kino_WD=0.192
Activity: n=200 low-rank, U_R2=0.987, V_R2=0.948
Mutation: L1: 1E-5 -> 1E-6
Observation: L1=1E-6 HURTS seed=22000! test_R2 0.992->0.980 (-0.012), conn_R2 0.962->0.955 (-0.007). seed=22000 LOCKED at L1=1E-5.

## Iter 308: converged
Node: id=308, parent=root
Mode/Strategy: explore
Config: seed=25000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.993, test_pearson=0.989, connectivity_R2=0.962, cluster_accuracy=1.000, final_loss=7.509E+02, kino_R2=0.993, kino_SSIM=0.973, kino_WD=0.097
Activity: n=200 low-rank, U_R2=0.988, V_R2=0.953
Mutation: seed: new -> 25000
Observation: NEW UPPER MID-TIER at n=200! test_R2=0.993. candidate for L1=1E-6 test.

### Batch 14 Results (iters 309-312)

## Iter 309: converged
Node: id=309, parent=308
Mode/Strategy: exploit
Config: seed=25000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-6, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.996, test_pearson=0.993, connectivity_R2=0.980, cluster_accuracy=1.000, final_loss=7.430E+02, kino_R2=0.996, kino_SSIM=0.987, kino_WD=0.063
Activity: n=200 low-rank, U_R2=0.994, V_R2=0.971
Mutation: L1: 1E-5 -> 1E-6
Observation: L1=1E-6 BOOSTS seed=25000! test_R2 +0.003 (0.993->0.996), conn_R2 +0.018 (0.962->0.980). NOW TOP-TIER. LOCKED at L1=1E-6.

## Iter 310: converged
Node: id=310, parent=root
Mode/Strategy: explore
Config: seed=26000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.991, test_pearson=0.985, connectivity_R2=0.988, cluster_accuracy=1.000, final_loss=7.890E+02, kino_R2=0.991, kino_SSIM=0.970, kino_WD=0.121
Activity: n=200 low-rank, U_R2=0.994, V_R2=0.978
Mutation: seed: new -> 26000
Observation: NEW TOP-TIER at n=200! test_R2=0.991, BEST conn_R2=0.988 so far. candidate for L1=1E-6 test.

## Iter 311: converged
Node: id=311, parent=root
Mode/Strategy: explore
Config: seed=27000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.988, test_pearson=0.982, connectivity_R2=0.938, cluster_accuracy=1.000, final_loss=7.973E+02, kino_R2=0.988, kino_SSIM=0.965, kino_WD=0.129
Activity: n=200 low-rank, U_R2=0.994, V_R2=0.930
Mutation: seed: new -> 27000
Observation: NEW UPPER MID-TIER at n=200! test_R2=0.988, lower conn_R2=0.938. candidate for L1=1E-6 test.

## Iter 312: failed
Node: id=312, parent=root
Mode/Strategy: explore
Config: seed=28000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.467, test_pearson=0.236, connectivity_R2=0.046, cluster_accuracy=1.000, final_loss=3.342E+03, kino_R2=-9.14, kino_SSIM=0.448, kino_WD=4.671
Activity: n=200 low-rank, U_R2=0.985, V_R2=0.108 — V-RECOVERY FAILURE!
Mutation: seed: new -> 28000
Observation: HARD at n=200! 14th hard seed (V_R2=0.108). consistent V-recovery failure mode.

### Batch 15 Results (iters 313-316)

## Iter 313: converged
Node: id=313, parent=310
Mode/Strategy: exploit
Config: seed=26000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-6, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.9995, test_pearson=0.9990, connectivity_R2=0.9648, cluster_accuracy=1.000, final_loss=7.906E+02, kino_R2=0.9994, kino_SSIM=0.9976, kino_WD=0.0228
Activity: n=200 low-rank, U_R2=0.9875, V_R2=0.9564
Mutation: L1: 1E-5 -> 1E-6
Observation: L1=1E-6 TRADEOFF! test_R2 +0.008 (0.991->0.9995), conn_R2 -0.023 (0.988->0.965). dynamics improved, W-recovery degraded. LOCKED at L1=1E-6 for dynamics.

## Iter 314: converged
Node: id=314, parent=311
Mode/Strategy: exploit
Config: seed=27000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-6, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.9998, test_pearson=0.9996, connectivity_R2=0.9740, cluster_accuracy=1.000, final_loss=7.841E+02, kino_R2=0.9998, kino_SSIM=0.9990, kino_WD=0.0181
Activity: n=200 low-rank, U_R2=0.9875, V_R2=0.9647
Mutation: L1: 1E-5 -> 1E-6
Observation: L1=1E-6 BOOSTS BOTH! test_R2 +0.012 (0.988->0.9998), conn_R2 +0.036 (0.938->0.974). NOW TOP-TIER. LOCKED at L1=1E-6.

## Iter 315: converged
Node: id=315, parent=root
Mode/Strategy: explore
Config: seed=29000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.9926, test_pearson=0.9873, connectivity_R2=0.9859, cluster_accuracy=1.000, final_loss=8.766E+02, kino_R2=0.9925, kino_SSIM=0.9734, kino_WD=0.1163
Activity: n=200 low-rank, U_R2=0.9881, V_R2=0.9761
Mutation: seed: new -> 29000
Observation: NEW TOP-TIER at n=200! test_R2=0.993, BEST conn_R2=0.986 so far at n=200! candidate for L1=1E-6 test.

## Iter 316: converged
Node: id=316, parent=root
Mode/Strategy: explore
Config: seed=30000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.9810, test_pearson=0.9694, connectivity_R2=0.9698, cluster_accuracy=1.000, final_loss=1.056E+03, kino_R2=0.9804, kino_SSIM=0.9492, kino_WD=0.1850
Activity: n=200 low-rank, U_R2=0.9878, V_R2=0.9618
Mutation: seed: new -> 30000
Observation: NEW UPPER MID-TIER at n=200! test_R2=0.981, conn_R2=0.970. candidate for L1=1E-6 test.

### Batch 16 Results (iters 317-320)

## Iter 317: failed
Node: id=317, parent=315
Mode/Strategy: exploit
Config: seed=29000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-6, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.492, test_pearson=0.270, connectivity_R2=0.023, cluster_accuracy=1.000, final_loss=2.139E+03, kino_R2=-2.80, kino_SSIM=0.447, kino_WD=2.99
Activity: n=200 low-rank, U_R2=0.966, V_R2=0.080 — V-RECOVERY FAILURE!
Mutation: L1: 1E-5 -> 1E-6
Observation: CATASTROPHIC! L1=1E-6 DESTROYS seed=29000! Was TOP-TIER (0.993, conn_R2=0.986) → now FAILED (0.492). LOCKED at L1=1E-5.

## Iter 318: converged
Node: id=318, parent=316
Mode/Strategy: exploit
Config: seed=30000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-6, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.998, test_pearson=0.996, connectivity_R2=0.933, cluster_accuracy=1.000, final_loss=1.116E+03, kino_R2=0.998, kino_SSIM=0.992, kino_WD=0.038
Activity: n=200 low-rank, U_R2=0.987, V_R2=0.929
Mutation: L1: 1E-5 -> 1E-6
Observation: L1=1E-6 BOOSTS test_R2 +0.017 (0.981->0.998), conn_R2 -0.037 (0.970->0.933). NOW TOP-TIER. LOCKED at L1=1E-6.

## Iter 319: failed
Node: id=319, parent=root
Mode/Strategy: explore
Config: seed=31000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.460, test_pearson=0.320, connectivity_R2=0.030, cluster_accuracy=1.000, final_loss=3.211E+03, kino_R2=-3.98, kino_SSIM=0.384, kino_WD=3.63
Activity: n=200 low-rank, U_R2=0.966, V_R2=0.086 — V-RECOVERY FAILURE!
Mutation: seed: new -> 31000
Observation: HARD SEED! 15th hard at n=200 (V_R2=0.086). V-recovery failure.

## Iter 320: failed
Node: id=320, parent=root
Mode/Strategy: explore
Config: seed=32000, lr_W=5E-3, lr=1E-4, lr_emb=2.5E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs_init=2, first_coeff_L1=0, batch_size=8, n_epochs=2, recurrent=F, time_step=1
Metrics: test_R2=0.473, test_pearson=-0.058, connectivity_R2=0.047, cluster_accuracy=1.000, final_loss=3.911E+03, kino_R2=-40.06, kino_SSIM=0.428, kino_WD=11.95
Activity: n=200 low-rank, U_R2=0.965, V_R2=0.100 — V-RECOVERY FAILURE!
Mutation: seed: new -> 32000
Observation: HARD SEED! 16th hard at n=200 (V_R2=0.100). V-recovery failure.

### Batch 17 (iters 321-324) Strategy

| Slot | Seed | lr_W | lr | L1 | edge_diff | n_epochs | Rationale |
|------|------|------|----|-----|-----------|----------|-----------|
| 0 | 33000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | new seed exploration |
| 1 | 34000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | new seed exploration |
| 2 | 35000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | new seed exploration |
| 3 | 36000 | 5E-3 | 1E-4 | 1E-5 | 10000 | 2 | new seed exploration |

### Emerging Observations

n=200 vs n=100 comparison (64 data points across 16 batches, 39 seeds):
1. **seed=137 FLIPPED**: learnable at n=100 (0.989) → HARD at n=200 (V_R2=0.133). rescue failed.
2. **seed=1000 FLIPPED**: RESCUED at n=100 (0.991) → HARD at n=200 (V_R2=0.210). 2nd flip!
3. **seed=6000 FLIPPED**: LEARNABLE at n=100 (0.991) → HARD at n=200 (V_R2=0.086). 3rd flip!
4. **seed=17000 FLIPPED**: LEARNABLE at n=100 (0.952) → HARD at n=200 (V_R2=0.108). 4th flip!
5. **seed=20000 FLIPPED**: TOP-TIER at n=100 (0.993) → HARD at n=200 (V_R2=0.096). 5th flip!
6. **seed=21000 FLIPPED**: TOP-TIER at n=100 (0.999) → HARD at n=200 (V_R2=0.071). 6th flip! MOST DRAMATIC!
7. **seed=5000 WORSENED**: fragile at n=100 (0.953) → HARD at n=200 (V_R2=0.095).
8. **seed=3000 CONSISTENT**: UNLEARNABLE at n=100 (0.597) → HARD at n=200 (V_R2=0.208). same failure.
9. **seed=9000 CONSISTENT**: UNLEARNABLE at n=100 (0.571) → HARD at n=200 (V_R2=0.104). same failure.
10. **seed=12000 CONSISTENT**: HARD at n=100 (0.527) → HARD at n=200 (V_R2=0.192). same failure.
11. **seed=15000 CONSISTENT**: UNLEARNABLE at n=100 (0.291) → HARD at n=200 (V_R2=0.263). same failure.
12. **seed=16000 CONSISTENT**: HARD at n=100 (V_R2=0.406) → HARD at n=200 (V_R2=0.214). same failure.
13. **seed=7000 IMPROVED**: mid-tier at n=100 (0.974) → TOP-TIER at n=200 (0.998)! L1=1E-5 LOCKED.
14. **seed=11000 IMPROVED**: mid-tier at n=100 (0.977) → TOP-TIER at n=200 (0.999)! L1=1E-5 LOCKED.
15. **seed=8000 CONSISTENT**: TOP-TIER at n=100 (0.999) → TOP-TIER at n=200 (0.998)! L1=1E-5 LOCKED.
16. **seed=4000 EASIER at n=200**: needed complex combo at n=100 → standard recipe works at n=200!
17. **seed=13000 NOW LOCKED**: L1=1E-6 hurts dynamics (-0.029). LOCKED at L1=1E-5 (0.996).
18. **seed=14000 NOW LOCKED**: L1=1E-6 hurts (-0.031 conn_R2, -0.005 test_R2). LOCKED at L1=1E-5 (0.993).
19. **lr_W=6E-3 CATASTROPHIC at n=200**: destroyed seed=42. DO NOT use at n=200!
20. **L1=1E-6 is SEED-DEPENDENT at n=200**: helps 42, 99, 7, 256, 500, 10000, 25000, 26000, 27000, 30000 (10); HURTS 314, 2000, 4000, 7000, 8000, 11000, 13000, 14000, 18000, 22000, 24000, 29000 (12); NEUTRAL 23000 (1)
21. **hard seed rate at n=200**: 16/39 = 41.0% (vs n=100's 30%) — n=200 is MUCH HARDER
22. **seeds 22000, 23000, 24000 NOW LOCKED at L1=1E-5**: batch 13 confirmed all hurt by L1=1E-6!
23. **seed=25000 TRANSFORMED**: L1=1E-6 boosts +0.003 test_R2, +0.018 conn_R2, NOW TOP-TIER, LOCKED!
24. **seed=26000 L1=1E-6 TRADEOFF**: test_R2 +0.008 (now 0.9995), conn_R2 -0.023 (now 0.965). LOCKED at L1=1E-6 for dynamics!
25. **seed=27000 TRANSFORMED**: L1=1E-6 boosts BOTH: test_R2 +0.012 (now 0.9998), conn_R2 +0.036 (now 0.974). NOW TOP-TIER. LOCKED at L1=1E-6!
26. **seed=28000 HARD**: 14th hard seed (V_R2=0.108). consistent V-recovery failure.
27. **seed=29000 CATASTROPHE**: L1=1E-6 DESTROYS TOP-TIER seed (0.993->0.492)! MOST EXTREME case yet. LOCKED at L1=1E-5.
28. **seed=30000 TRANSFORMED**: L1=1E-6 boosts test_R2 +0.017 (0.981->0.998), NOW TOP-TIER, LOCKED at L1=1E-6.
29. **seeds 31000, 32000 HARD**: 15th+16th hard seeds (V_R2=0.086, 0.100). 41% hard rate at n=200.

n=200 regime CONFIRMED characteristics:
- lr_W ceiling LOWER (5E-3 optimal, 6E-3 catastrophic)
- W recovery harder but still excellent (conn_R2 0.92-0.993 typical)
- L1=1E-6 helps 43% of seeds (9/21), hurts 52% (11/21), neutral 5% (1/21) — L1=1E-5 is BETTER default at n=200!
- Standard recipe (L1=1E-5) optimal for seeds 314, 2000, 4000, 7000, 8000, 11000, 13000, 14000, 18000, 22000, 23000, 24000, 29000 — 13 seeds!
- L1=1E-6 optimal for seeds 42, 99, 7, 256, 500, 10000, 25000, 26000, 27000, 30000 — 10 seeds!
- n=100 "RESCUED" seeds can FLIP to HARD at n=200 (137, 1000, 6000, 17000)
- n=100 "MID-TIER" seeds can IMPROVE at n=200 (7000, 11000)
- Hard seeds (3000, 9000, 12000, 15000, 16000) remain hard at both n values

n=200 Best Configurations (37 seeds tested):
| Seed | L1 | test_R2 | conn_R2 | V_R2 | Status |
|------|-----|---------|---------|------|--------|
| 99 | 1E-6 | 1.000 | 0.950 | 0.942 | TOP-TIER, LOCKED |
| 27000 | 1E-6 | 0.9998 | 0.974 | 0.965 | TOP-TIER, LOCKED (L1=1E-6 transforms!) |
| 26000 | 1E-6 | 0.9995 | 0.965 | 0.956 | TOP-TIER, LOCKED (L1=1E-6 tradeoff — better dynamics) |
| 11000 | 1E-5 | 0.999 | 0.968 | 0.959 | TOP-TIER, LOCKED (L1=1E-6 hurts dynamics) |
| 7000 | 1E-5 | 0.998 | 0.980 | 0.969 | TOP-TIER, LOCKED (L1=1E-6 hurts) |
| 8000 | 1E-5 | 0.998 | 0.953 | 0.946 | TOP-TIER, LOCKED (L1=1E-6 hurts) |
| 500 | 1E-6 | 0.9996 | 0.958 | 0.951 | TOP-TIER, LOCKED |
| 10000 | 1E-6 | 0.997 | 0.971 | 0.962 | TOP-TIER, LOCKED |
| 42 | 1E-6 | 0.997 | 0.970 | 0.961 | TOP-TIER, LOCKED |
| 314 | 1E-5 | 0.997 | 0.985 | 0.976 | TOP-TIER, LOCKED (L1=1E-6 hurts) |
| 256 | 1E-6 | 0.997 | 0.966 | 0.959 | TOP-TIER, LOCKED |
| 7 | 1E-6 | 0.997 | 0.964 | 0.957 | TOP-TIER, LOCKED |
| 24000 | 1E-5 | 0.996 | 0.985 | 0.975 | TOP-TIER, LOCKED (L1=1E-6 hurts -0.028) |
| 25000 | 1E-6 | 0.996 | 0.980 | 0.971 | TOP-TIER, LOCKED (L1=1E-6 transforms!) |
| 13000 | 1E-5 | 0.996 | 0.921 | 0.918 | UPPER MID-TIER, LOCKED (L1=1E-6 hurts dynamics) |
| 23000 | 1E-5 | 0.995 | 0.957 | 0.948 | TOP-TIER, LOCKED (L1=1E-6 neutral) |
| 2000 | 1E-5 | 0.995 | 0.989 | 0.979 | MID-TIER, LOCKED (L1=1E-6 hurts) |
| 4000 | 1E-5 | 0.995 | 0.964 | 0.956 | MID-TIER, LOCKED (L1=1E-6 hurts) |
| 14000 | 1E-5 | 0.993 | 0.923 | 0.917 | UPPER MID-TIER, LOCKED (L1=1E-6 hurts) |
| 30000 | 1E-6 | 0.998 | 0.933 | 0.929 | TOP-TIER, LOCKED (L1=1E-6 transforms +0.017!) |
| 29000 | 1E-5 | 0.993 | 0.986 | 0.976 | TOP-TIER, LOCKED (L1=1E-6 CATASTROPHIC -0.501!) |
| 22000 | 1E-5 | 0.992 | 0.962 | 0.955 | UPPER MID-TIER, LOCKED (L1=1E-6 hurts -0.012) |
| 18000 | 1E-5 | 0.972 | 0.989 | 0.979 | LOCKED at L1=1E-5 (L1=1E-6 hurts conn_R2) |
| 31000 | - | - | - | 0.086 | HARD (15th hard) |
| 32000 | - | - | - | 0.100 | HARD (16th hard) |
| 28000 | - | - | - | 0.108 | HARD (14th hard) |
| 137 | - | - | - | 0.133 | HARD (n=100 FLIP) |
| 1000 | - | - | - | 0.210 | HARD (n=100 FLIP) |
| 3000 | - | - | - | 0.208 | HARD (same as n=100) |
| 5000 | - | - | - | 0.095 | HARD (worsened from n=100) |
| 6000 | - | - | - | 0.086 | HARD (n=100 FLIP — was 0.991!) |
| 9000 | - | - | - | 0.104 | HARD (same as n=100) |
| 12000 | - | - | - | 0.192 | HARD (same as n=100) |
| 15000 | - | - | - | 0.263 | HARD (same as n=100) |
| 16000 | - | - | - | 0.214 | HARD (same as n=100) |
| 17000 | - | - | - | 0.108 | HARD (n=100 FLIP — was 0.952!) |
| 19000 | - | - | - | 0.172 | HARD (same as n=100) |
| 20000 | - | - | - | 0.096 | HARD (n=100 FLIP — was 0.993!) |
| 21000 | - | - | - | 0.071 | HARD (n=100 FLIP — was 0.999!) |

---

## Previous Block Summary (Block 23)

n=200 exploration: 37 seeds tested, 22 learnable (59.5%), 15 hard (40.5%). Hard seed rate HIGHER at n=200 (40.5% vs n=100's 30.0%). L1=1E-6 helps ~50% of learnable seeds. Several n=100 learnable seeds FLIP to hard at n=200 (137, 1000, 6000, 17000, 20000, 21000). n=200 conn_R2 ranges 0.916-0.989 (vs n=100's 1.000).

---

## Current Block (Block 24) — LOCKED-SLOT N-SCALING

### Block Info

**NEW REGIME**: Each slot is LOCKED to a specific n_neurons value. All slots share n_frames=100000, data_augmentation_loop=50, n_epochs=1.

### Locked Slot Configuration

| Slot | n_neurons | Config file |
|------|-----------|-------------|
| 0 | 200 | signal_low_rank_Claude_00.yaml |
| 1 | 400 | signal_low_rank_Claude_01.yaml |
| 2 | 600 | signal_low_rank_Claude_02.yaml |
| 3 | 1000 | signal_low_rank_Claude_03.yaml |

### Locked Parameters (DO NOT CHANGE)

- `n_neurons`: per-slot (see table above)
- `n_frames`: 100000
- `data_augmentation_loop`: 50
- `n_epochs`: 1

### Exploration Strategy

Each slot runs an INDEPENDENT UCB tree, varying training params (lr_W, coeff_W_L1, coeff_edge_diff, seed, lr_emb, etc.) while keeping n_neurons fixed. This enables a systematic N-scaling comparison under identical training budgets.

### Key Prior Knowledge (from Blocks 1-23)

- At n=100: conn_R2≈1.000 for 70% of seeds; 30% are hard (V-recovery failure)
- At n=200: conn_R2 drops to 0.916-0.989; hard seed rate rises to 40.5%
- Standard recipe: lr_W=5E-3, lr=1E-4, L1=1E-5, edge_diff=10000, batch=8
- L1=1E-6 helps ~50% of seeds at both n=100 and n=200
- lr_W=6E-3 is CATASTROPHIC at n=200 (was beneficial for some n=100 seeds)
- n=200 is MORE SENSITIVE to hyperparameters than n=100
- n_epochs=3 helps some seeds but hurts others (seed-dependent)
- n_frames=100000 is 10x more than n=100 regime (10000) — may improve convergence

### Hypothesis

With n_frames=100000 (10x previous n=100/200 regime), all slots get much more training data. n=200 with 100000 frames may match n=100 performance. n=400/600/1000 are UNEXPLORED territory for low_rank. The locked-slot design enables clean N-scaling curves.

### Iterations This Block

## Iter 321: partial (n=200)
Node: id=321, parent=318
Mode/Strategy: exploit
Config: n_neurons=200, seed=33000, lr_W=5E-3, lr=1E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs=1
Metrics: test_R2=0.996, test_pearson=0.993, connectivity_R2=0.841, kino_R2=0.995
Activity: U_R2=0.986, V_R2=0.841
Mutation: seed: 30000 -> 33000
Observation: PARTIAL! conn_R2=0.841 below 0.9, excellent dynamics (0.996). candidate for L1=1E-6 test.
Next: parent=321

## Iter 322: converged (n=400) REVERSE PATTERN
Node: id=322, parent=root
Mode/Strategy: explore
Config: n_neurons=400, seed=34000, lr_W=5E-3, lr=1E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs=1
Metrics: test_R2=0.917, test_pearson=0.858, connectivity_R2=0.999, kino_R2=0.908
Activity: U_R2=0.994, V_R2=0.992
Mutation: seed: new -> 34000
Observation: REVERSE PATTERN! Excellent W-recovery (0.999, V_R2=0.992) but WEAK dynamics (0.917). n=400 needs hyperparameter tuning.
Next: parent=322

## Iter 323: converged (n=600) SAME PATTERN
Node: id=323, parent=root
Mode/Strategy: explore
Config: n_neurons=600, seed=35000, lr_W=5E-3, lr=1E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs=1
Metrics: test_R2=0.940, test_pearson=0.952, connectivity_R2=0.999, kino_R2=0.930
Activity: U_R2=0.995, V_R2=0.995
Mutation: seed: new -> 35000
Observation: SAME PATTERN as n=400! Excellent W-recovery (0.999, V_R2=0.995) but mediocre dynamics (0.940). larger n improves W-recovery.
Next: parent=323

## Iter 324: converged (n=1000) BEST SCALING
Node: id=324, parent=root
Mode/Strategy: explore
Config: n_neurons=1000, seed=36000, lr_W=5E-3, lr=1E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs=1
Metrics: test_R2=0.997, test_pearson=0.994, connectivity_R2=0.992, kino_R2=0.996
Activity: U_R2=0.997, V_R2=0.989
Mutation: seed: new -> 36000
Observation: BEST SCALING! Both excellent: test_R2=0.997, conn_R2=0.992. n=1000 may be EASIEST scale for low-rank!
Next: parent=324

---

>>> BLOCK 27 END <<<

### Emerging Observations — N-SCALING (Block 27, 4 data points)

CRITICAL N-SCALING FINDINGS (batch 17):
1. **W-RECOVERY IMPROVES WITH n**: conn_R2 = 0.841 (n=200) < 0.992 (n=1000) < 0.999 (n=400/600)
2. **DYNAMICS NON-MONOTONIC**: test_R2 = 0.917 (n=400) < 0.940 (n=600) < 0.996 (n=200) ~ 0.997 (n=1000)
3. **n=1000 IS EASIEST**: Both metrics excellent in ONE iteration!
4. **n=400/600 REVERSE PATTERN**: Excellent W, weak dynamics — need lr_W or lr tuning
5. **n=200 has PARTIAL W-recovery**: seed=33000 conn_R2=0.841 (may need L1=1E-6)

N-Scaling Summary Table:
| n | test_R2 | conn_R2 | V_R2 | Notes |
|---|---------|---------|------|-------|
| 200 | 0.996 | 0.841 | 0.841 | PARTIAL — needs tuning |
| 400 | 0.917 | 0.999 | 0.992 | REVERSE — W excellent, dynamics weak |
| 600 | 0.940 | 0.999 | 0.995 | REVERSE — same as n=400 |
| 1000 | 0.997 | 0.992 | 0.989 | BEST — both excellent! |

Hypotheses for next batch:
- n=200: L1=1E-6 may improve conn_R2
- n=400: lr_W=4E-3 or lr=5E-5 may improve dynamics (reduce MLP capacity)
- n=600: same as n=400 — try lower lr_W
- n=1000: already excellent, test new seed for robustness

---

### Batch 18 Results (iters 325-328)

## Iter 325: partial (n=200)
Node: id=325, parent=321
Mode/Strategy: exploit
Config: n_neurons=200, seed=33000, lr_W=5E-3, lr=1E-4, coeff_W_L1=1E-6, coeff_edge_diff=10000, n_epochs=1
Metrics: test_R2=0.940, test_pearson=0.911, connectivity_R2=0.881, cluster_accuracy=1.000, final_loss=3.190E+03, kino_R2=0.939, kino_SSIM=0.882, kino_WD=0.267
Activity: n=200, U_R2=0.986, V_R2=0.881
Mutation: L1: 1E-5 -> 1E-6
Observation: L1=1E-6 HURTS dynamics! test_R2 0.996->0.940 (-0.056), conn_R2 0.841->0.881 (+0.040). TRADEOFF but dynamics worse. seed=33000 LOCKED at L1=1E-5.
Next: parent=321

## Iter 326: failed (n=400)
Node: id=326, parent=322
Mode/Strategy: exploit
Config: n_neurons=400, seed=34000, lr_W=4E-3, lr=1E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs=1
Metrics: test_R2=0.578, test_pearson=0.503, connectivity_R2=0.613, cluster_accuracy=1.000, final_loss=1.399E+04, kino_R2=0.377, kino_SSIM=0.477, kino_WD=0.527
Activity: n=400, U_R2=0.985, V_R2=0.641
Mutation: lr_W: 5E-3 -> 4E-3
Observation: CATASTROPHIC! lr_W=4E-3 DESTROYS n=400! test_R2 0.917->0.578 (-0.339), conn_R2 0.999->0.613 (-0.386). n=400 needs HIGHER lr_W (try 6E-3).
Next: parent=322

## Iter 327: converged (n=600) BEST N=600
Node: id=327, parent=323
Mode/Strategy: exploit
Config: n_neurons=600, seed=35000, lr_W=4E-3, lr=1E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs=1
Metrics: test_R2=0.963, test_pearson=0.975, connectivity_R2=0.9995, cluster_accuracy=1.000, final_loss=2.908E+03, kino_R2=0.957, kino_SSIM=0.907, kino_WD=0.137
Activity: n=600, U_R2=0.995, V_R2=0.995
Mutation: lr_W: 5E-3 -> 4E-3
Observation: lr_W=4E-3 IMPROVES n=600! test_R2 0.940->0.963 (+0.023), conn_R2 unchanged at 0.9995. NOW EXCELLENT. LOCKED at lr_W=4E-3.
Next: parent=327

## Iter 328: failed (n=1000) HARD SEED
Node: id=328, parent=root
Mode/Strategy: explore
Config: n_neurons=1000, seed=37000, lr_W=5E-3, lr=1E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs=1
Metrics: test_R2=0.608, test_pearson=0.423, connectivity_R2=0.309, cluster_accuracy=1.000, final_loss=9.225E+03, kino_R2=0.223, kino_SSIM=0.531, kino_WD=0.666
Activity: n=1000, U_R2=0.992, V_R2=0.320
Mutation: seed: 36000 -> 37000
Observation: HARD SEED at n=1000! V_R2=0.320 — V-recovery failure. seed=37000 is UNLEARNABLE at n=1000.
Next: parent=324

---

### Batch 19 Results (iters 329-332)

## Iter 329: failed (n=200) HARD SEED
Node: id=329, parent=root
Mode/Strategy: explore
Config: n_neurons=200, seed=34000, lr_W=5E-3, lr=1E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs=1
Metrics: test_R2=0.478, test_pearson=0.231, connectivity_R2=0.027, kino_R2=-4.90
Activity: n=200, U_R2=0.966, V_R2=0.088 — V-RECOVERY FAILURE!
Mutation: seed: 33000 -> 34000
Observation: HARD SEED at n=200! V_R2=0.088. seed=34000 is UNLEARNABLE at n=200. 17th hard seed at n=200.
Next: parent=root

## Iter 330: converged (n=400) BREAKTHROUGH!
Node: id=330, parent=322
Mode/Strategy: exploit
Config: n_neurons=400, seed=34000, lr_W=6E-3, lr=1E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs=1
Metrics: test_R2=0.978, test_pearson=0.975, connectivity_R2=0.9995, kino_R2=0.977
Activity: n=400, U_R2=0.994, V_R2=0.993
Mutation: lr_W: 5E-3 -> 6E-3
Observation: lr_W=6E-3 TRANSFORMS n=400! test_R2 0.917->0.978 (+0.061). NOW EXCELLENT. LOCKED at lr_W=6E-3.
Next: parent=330

## Iter 331: converged (n=600) SEED DEGRADATION
Node: id=331, parent=327
Mode/Strategy: explore
Config: n_neurons=600, seed=36000, lr_W=4E-3, lr=1E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs=1
Metrics: test_R2=0.874, test_pearson=0.915, connectivity_R2=0.9994, kino_R2=0.827
Activity: n=600, U_R2=0.995, V_R2=0.995
Mutation: seed: 35000 -> 36000
Observation: seed=36000 at n=600 DEGRADES dynamics (0.963->0.874). lr_W=4E-3 is seed-dependent. try lr_W=5E-3.
Next: parent=327

## Iter 332: partial (n=1000) REVERSE PATTERN
Node: id=332, parent=324
Mode/Strategy: explore
Config: n_neurons=1000, seed=38000, lr_W=5E-3, lr=1E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs=1
Metrics: test_R2=0.742, test_pearson=0.640, connectivity_R2=0.994, kino_R2=0.628
Activity: n=1000, U_R2=0.997, V_R2=0.992
Mutation: seed: 37000 -> 38000
Observation: REVERSE PATTERN at n=1000! Excellent W-recovery (0.994) but weak dynamics (0.742). needs lr_W tuning.
Next: parent=332

---

## Iter 333: partial (n=200)
Node: id=333, parent=332
Mode/Strategy: exploit
Config: n_neurons=200, seed=35000, lr_W=5E-3, lr=1E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs=1
Metrics: test_R2=0.994, test_pearson=0.990, connectivity_R2=0.861, cluster_accuracy=1.000, final_loss=2.801E+03, kino_R2=0.994, kino_SSIM=0.976, kino_WD=0.063
Activity: n=200, U_R2=0.986, V_R2=0.859
Mutation: seed: 34000 -> 35000
Observation: PARTIAL! Excellent dynamics (0.994), conn_R2=0.861 below 0.9. seed=35000 is LEARNABLE. candidate for L1=1E-6 test.
Next: parent=333

## Iter 334: converged (n=400)
Node: id=334, parent=330
Mode/Strategy: exploit
Config: n_neurons=400, seed=35000, lr_W=6E-3, lr=1E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs=1
Metrics: test_R2=0.990, test_pearson=0.989, connectivity_R2=0.999, cluster_accuracy=1.000, final_loss=3.052E+03, kino_R2=0.989, kino_SSIM=0.964, kino_WD=0.103
Activity: n=400, U_R2=0.994, V_R2=0.993
Mutation: seed: 34000 -> 35000
Observation: lr_W=6E-3 TRANSFERS! seed=35000 at n=400 gives test_R2=0.990, conn_R2=0.999. CONFIRMED: lr_W=6E-3 is optimal for n=400.
Next: parent=334

## Iter 335: partial (n=600)
Node: id=335, parent=331
Mode/Strategy: exploit
Config: n_neurons=600, seed=36000, lr_W=5E-3, lr=1E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs=1
Metrics: test_R2=0.917, test_pearson=0.948, connectivity_R2=0.999, cluster_accuracy=1.000, final_loss=3.242E+03, kino_R2=0.899, kino_SSIM=0.852, kino_WD=0.183
Activity: n=600, U_R2=0.995, V_R2=0.994
Mutation: lr_W: 4E-3 -> 5E-3
Observation: lr_W=5E-3 IMPROVES n=600 seed=36000! test_R2 0.874->0.917 (+0.043). Still weak dynamics but better. Try lr_W=6E-3.
Next: parent=335

## Iter 336: partial (n=1000)
Node: id=336, parent=332
Mode/Strategy: exploit
Config: n_neurons=1000, seed=38000, lr_W=6E-3, lr=1E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs=1
Metrics: test_R2=0.814, test_pearson=0.767, connectivity_R2=0.992, cluster_accuracy=1.000, final_loss=2.754E+03, kino_R2=0.795, kino_SSIM=0.729, kino_WD=0.419
Activity: n=1000, U_R2=0.997, V_R2=0.990
Mutation: lr_W: 5E-3 -> 6E-3
Observation: lr_W=6E-3 IMPROVES n=1000 seed=38000! test_R2 0.742->0.814 (+0.072). Still REVERSE pattern. Try lr_W=7E-3 or new seed.
Next: parent=336

---

>>> BLOCK 28 END <<<

### Emerging Observations — N-SCALING (Block 28, 16 data points)

CRITICAL N-SCALING FINDINGS (batches 18-20):
1. **lr_W=6E-3 CONFIRMED for n=400**: seed=35000 at lr_W=6E-3 gives test_R2=0.990, conn_R2=0.999. ROBUST!
2. **lr_W scales with n**: n=200→5E-3, n=400→6E-3, n=600→5-6E-3, n=1000→6-7E-3 (HIGHER lr_W for larger n)
3. **n=600 seed=36000 needs MORE lr_W**: 4E-3→5E-3 improved 0.874→0.917 (+0.043), still weak. TRY 6E-3.
4. **n=1000 seed=38000 still REVERSE**: lr_W=6E-3 improved 0.742→0.814 (+0.072), needs 7E-3 or new seed.
5. **seed=35000 is LEARNABLE at all n**: n=200 (0.994), n=400 (0.990), n=600 (0.963). GOOD TEST SEED.
6. **REVERSE PATTERN** at larger n: excellent conn_R2 (0.99+) with weak dynamics — solved by HIGHER lr_W.

Updated N-Scaling Summary Table:
| n | Seed | lr_W | test_R2 | conn_R2 | V_R2 | Notes |
|---|------|------|---------|---------|------|-------|
| 200 | 33000 | 5E-3 | 0.996 | 0.841 | 0.841 | LOCKED at L1=1E-5 |
| 200 | 34000 | 5E-3 | 0.478 | 0.027 | 0.088 | HARD SEED |
| 200 | 35000 | 5E-3 | 0.994 | 0.861 | 0.859 | PARTIAL — candidate for L1=1E-6 |
| 400 | 34000 | 6E-3 | 0.978 | 0.9995 | 0.993 | LOCKED at lr_W=6E-3 |
| 400 | 35000 | 6E-3 | 0.990 | 0.999 | 0.993 | lr_W=6E-3 CONFIRMED ROBUST! |
| 600 | 35000 | 4E-3 | 0.963 | 0.9995 | 0.995 | LOCKED for seed=35000 |
| 600 | 36000 | 5E-3 | 0.917 | 0.999 | 0.994 | IMPROVED but still weak — TRY 6E-3 |
| 1000 | 36000 | 5E-3 | 0.997 | 0.992 | 0.989 | EXCELLENT |
| 1000 | 37000 | 5E-3 | 0.608 | 0.309 | 0.320 | HARD SEED |
| 1000 | 38000 | 6E-3 | 0.814 | 0.992 | 0.990 | IMPROVED but still REVERSE — TRY 7E-3 |

Optimal lr_W by n (UPDATED):
- n=200: 5E-3 (confirmed)
- n=400: **6E-3 CONFIRMED BEST** (2 seeds: 34000, 35000)
- n=600: 4E-3 for seed=35000, TRY 6E-3 for seed=36000
- n=1000: 5E-3 for seed=36000, TRY 7E-3 for seed=38000 (6E-3 partial)

### Batch 21 Results (iters 337-340)

## Iter 337: partial (n=200)
Node: id=337, parent=333
Mode/Strategy: exploit
Config: n_neurons=200, seed=35000, lr_W=5E-3, lr=1E-4, coeff_W_L1=1E-6, coeff_edge_diff=10000, n_epochs=1
Metrics: test_R2=0.950, test_pearson=0.923, connectivity_R2=0.873, cluster_accuracy=1.000, final_loss=2.948E+03, kino_R2=0.949, kino_SSIM=0.881, kino_WD=0.237
Activity: n=200, U_R2=0.986, V_R2=0.873
Mutation: L1: 1E-5 -> 1E-6
Observation: L1=1E-6 HURTS n=200 seed=35000! test_R2 0.994->0.950 (-0.044), conn_R2 +0.012. seed=35000 LOCKED at L1=1E-5.
Next: parent=333

## Iter 338: partial (n=400)
Node: id=338, parent=334
Mode/Strategy: exploit
Config: n_neurons=400, seed=36000, lr_W=6E-3, lr=1E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs=1
Metrics: test_R2=0.916, test_pearson=0.918, connectivity_R2=0.999, cluster_accuracy=1.000, final_loss=3.442E+03, kino_R2=0.907, kino_SSIM=0.803, kino_WD=0.257
Activity: n=400, U_R2=0.994, V_R2=0.993
Mutation: seed: 35000 -> 36000
Observation: REVERSE PATTERN! seed=36000 at n=400: excellent W (0.999) but weak dynamics (0.916). seed=36000 needs different tuning at n=400.
Next: parent=338

## Iter 339: partial (n=600)
Node: id=339, parent=335
Mode/Strategy: exploit
Config: n_neurons=600, seed=36000, lr_W=6E-3, lr=1E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs=1
Metrics: test_R2=0.868, test_pearson=0.921, connectivity_R2=0.999, cluster_accuracy=1.000, final_loss=3.240E+03, kino_R2=0.850, kino_SSIM=0.806, kino_WD=0.213
Activity: n=600, U_R2=0.995, V_R2=0.994
Mutation: lr_W: 5E-3 -> 6E-3
Observation: lr_W=6E-3 DEGRADES n=600! test_R2 0.917->0.868 (-0.049). n=600 seed=36000 optimal lr_W is LOWER, not higher. LOCKED at lr_W=5E-3.
Next: parent=335

## Iter 340: partial (n=1000)
Node: id=340, parent=336
Mode/Strategy: exploit
Config: n_neurons=1000, seed=38000, lr_W=7E-3, lr=1E-4, coeff_W_L1=1E-5, coeff_edge_diff=10000, n_epochs=1
Metrics: test_R2=0.836, test_pearson=0.806, connectivity_R2=0.990, cluster_accuracy=1.000, final_loss=2.805E+03, kino_R2=0.828, kino_SSIM=0.750, kino_WD=0.431
Activity: n=1000, U_R2=0.998, V_R2=0.988
Mutation: lr_W: 6E-3 -> 7E-3
Observation: lr_W=7E-3 marginal gain +0.022 (0.814->0.836). Still REVERSE pattern at n=1000 seed=38000. Try L1=1E-6 or new seed.
Next: parent=root

---

## Block 29 — Batch 22 (iters 341-344) Strategy

| Slot | n | Seed | lr_W | L1 | Mutation | Rationale |
|------|---|------|------|----|----------|-----------|
| 0 | 200 | 36000 | 5E-3 | 1E-5 | seed: 35000 -> 36000 | try new seed at n=200 |
| 1 | 400 | 36000 | 6E-3 | 1E-6 | L1: 1E-5 -> 1E-6 | test if L1=1E-6 improves dynamics |
| 2 | 600 | 37000 | 5E-3 | 1E-5 | seed: 36000 -> 37000 | try new seed at optimal lr_W |
| 3 | 1000 | 39000 | 5E-3 | 1E-5 | seed: 38000 -> 39000 | new seed at standard recipe |

### Emerging Observations — N-SCALING (Block 29, 20 data points)

CRITICAL N-SCALING FINDINGS (batch 21):
1. **L1=1E-6 HURTS n=200 seed=35000**: test_R2 -0.044. LOCKED at L1=1E-5.
2. **lr_W=6E-3 is SEED-DEPENDENT at n=400**: seed=36000 shows REVERSE pattern (W=0.999, dynamics=0.916)
3. **lr_W=6E-3 is WRONG for n=600**: test_R2 degraded 0.917->0.868. n=600 optimal is 4E-3 (seed=35000) or 5E-3 (seed=36000)
4. **n=1000 REVERSE pattern persists**: lr_W=7E-3 only marginal gain. seed=38000 may need L1=1E-6 or different approach
5. **REVERSE pattern is seed-specific**: some seeds at larger n have excellent W-recovery but stuck dynamics

Updated N-Scaling Summary Table:
| n | Seed | lr_W | L1 | test_R2 | conn_R2 | V_R2 | Notes |
|---|------|------|----|---------|---------|------|-------|
| 200 | 35000 | 5E-3 | 1E-5 | 0.994 | 0.861 | 0.859 | LOCKED at L1=1E-5 (L1=1E-6 hurts -0.044) |
| 400 | 35000 | 6E-3 | 1E-5 | 0.990 | 0.999 | 0.993 | BEST n=400 config |
| 400 | 36000 | 6E-3 | 1E-5 | 0.916 | 0.999 | 0.993 | REVERSE — candidate for L1=1E-6 |
| 600 | 35000 | 4E-3 | 1E-5 | 0.963 | 0.9995 | 0.995 | BEST n=600 config |
| 600 | 36000 | 5E-3 | 1E-5 | 0.917 | 0.999 | 0.994 | LOCKED at lr_W=5E-3 (6E-3 hurts -0.049) |
| 1000 | 36000 | 5E-3 | 1E-5 | 0.997 | 0.992 | 0.989 | BEST n=1000 config |
| 1000 | 38000 | 7E-3 | 1E-5 | 0.836 | 0.990 | 0.988 | REVERSE — try L1=1E-6 or new seed |

Optimal lr_W by n (FINAL):
- n=200: **5E-3 CONFIRMED** (6E-3 catastrophic)
- n=400: **6E-3 CONFIRMED** for seed=34000/35000, seed=36000 shows REVERSE
- n=600: **4E-3 for seed=35000**, 5E-3 for seed=36000, 6E-3 degrades
- n=1000: **5E-3 for seed=36000**, 6-7E-3 for seed=38000 (still REVERSE)

**CRITICAL: This section must ALWAYS be at the END of memory file.**
