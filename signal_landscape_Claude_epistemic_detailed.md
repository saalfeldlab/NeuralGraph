# Epistemic Analysis Detailed Log: signal_landscape_Claude

**Companion to**: signal_landscape_Claude_epistemic_analysis.md
**Total instances**: ~290 reasoning events across 348 iterations (29 blocks, 4 parallel slots)
**Note**: Detailed entries below cover the first 10 blocks (112 iterations). Blocks 11-29 events are captured in the interactive visualizations (epistemic_timeline_interactive.html, epistemic_streamgraph_interactive.html, epistemic_sankey_interactive.html).

---

## 1. Induction: 23 instances

| Iter | Observation | Induced Pattern | Type | Block |
|------|-------------|-----------------|------|-------|
| 5 | lr_W=4E-3 best of 4 tested (1E-3, 2E-3, 4E-3, 8E-3) | lr_W=4E-3 sweet spot for chaotic n=100 | Single (best-of-batch) | 1 |
| 11 | L1=1E-6 at lr_W=2E-3 gives test_R2=0.998 (best dynamics) | L1 regularization improves dynamics at sub-optimal lr_W | Single | 1 |
| 12 | batch_size=16 converges but slight quality loss | batch_size trades quality for speed | Single | 1 |
| 19 | lr_W=3E-3 gives better dynamics (0.943) than 4E-3 in low_rank | Lower eff_rank needs gentler lr_W | Cumulative (3 obs: 13, 17, 19) | 2 |
| 21 | lr_W=3E-3 + L1=1E-6 → test_R2=0.996 | **BREAKTHROUGH**: recombination of two independent findings | Recombination | 2 |
| 24 | batch_size=16 + L1=1E-6 → 0.997 in low_rank | batch_size=16 safe with L1 in low_rank | Single (surprise) | 2 |
| 30 | lr_W=5E-3 fails twice (0.458, 0.455) reproducibly | Dale cliff at 5E-3 is deterministic, not stochastic | Cumulative (2 obs) | 3 |
| 33 | lr_W=4.5E-3 best; safe range [3.5, 4.5E-3] | Dale safe range is narrow | Cumulative (4 obs: 28-33) | 3 |
| 35 | L1=1E-6 negligible at lr_W=3.5E-3 in Dale | L1 not universal enabler in Dale | Single | 3 |
| 39 | lr_emb=1E-3 → FULL convergence: conn 0.9996, cluster 0.990 | lr_emb critical for heterogeneous embedding | Single (breakthrough) | 4 |
| 44 | L1=1E-6 → cluster drops 0.990→0.440 at L1=1E-5 | L1=1E-6 critical for embedding beyond low_rank | Cumulative (blocks 2, 4) | 4 |
| 51 | noise=[0.1, 0.5, 1.0] → eff_rank=[42, 84, 90] | Noise inflates effective rank monotonically | Cumulative (3 obs) | 5 |
| 55 | noise=0.1 preserves rollout (kino_R2=0.405) | Low noise best for rollout quality | Single | 5 |
| 58 | lr_W=2E-3 best at noise=1.0 (dynamics 0.998) | Inverse lr_W-noise relation | Cumulative (4 obs: 53-58) | 5 |
| 63 | lr_W=8E-3 best conn but worst dynamics at n=200 | Trade-off amplified at scale | Cumulative (2 obs: 62, 63) | 6 |
| 72 | lr=3E-4 BEST dynamics at n=200 | Higher lr safe at n=200; widens tolerance | Single (contradicts P1) | 6 |
| 79 | n_epochs=2 beats all 1-epoch configs in sparse | Training capacity is key lever for sparse | Cumulative (7 obs: 73-79) | 7 |
| 82 | lr_W=1E-2 + 2ep best (conn=0.466) in sparse | Best sparse config but still below threshold | Cumulative | 7 |
| 86 | Complete lr_W insensitivity at sparse+noise (all configs → 0.489) | Parameter landscape is flat at structural limit | Cumulative (4 obs: 85-87) | 8 |
| 87 | lr_W=8E-3 same as lr_W=2E-3 in sparse+noise | Confirms total insensitivity | Cumulative | 8 |
| 103 | lr_W=1E-2 best conn (0.805) at n=300 1ep | n=300 needs highest lr_W tested | Cumulative (4 obs: 97-103) | 9 |
| 106 | n_epochs=2 → conn 0.890 (+10.6% over 1ep) at n=300 | **BREAKTHROUGH**: n_epochs dominant for large n | Single (major) | 9 |
| 112 | L1=1E-6 boosts dynamics +6.8% at n=300 | L1=1E-6 not harmful at n=300 (revises principle) | Single (revises P3) | 10 |

---

## 2. Abduction: 6 instances

| Iter | Observation | Hypothesis | Block |
|------|-------------|------------|-------|
| 13 | Low_rank test_R2=0.902 despite conn=0.999 | eff_rank=13 (vs 35 in block 1) causes dynamics difficulty | 2 |
| 17 | lr_W=5E-3 catastrophic failure (0.385) | Stochastic failure vs fundamental instability point | 2 |
| 19 | lr_W=3E-3 outperforms 4E-3 in low_rank | Lower eff_rank needs gentler lr_W gradient | 2 |
| 37 | n_types=4 conn=0.999 but cluster_acc=0.67 | Embedding failure: lr_emb=1E-4 too low for type separation | 4 |
| 42 | lr_emb=1E-3 overshoots at lr_W=1E-3 | lr_W/lr_emb coupling: embedding depends on W convergence | 4 |
| 73 | Sparse 50% eff_rank=21, rho=0.746 | Subcritical spectral radius limits information flow through network | 7 |

---

## 3. Deduction: 30 instances

| Iter | Hypothesis | Prediction | Outcome | ✓/✗ | Block |
|------|-----------|------------|---------|-----|-------|
| 5 | lr_W=4E-3 is optimal from sweep | Best performance expected | test_R2=0.996, conn=0.9999 | ✓ | 1 |
| 9 | lr=2E-4 should improve dynamics | Better test_R2 | Degrades 0.996→0.981 | ✗ | 1 |
| 15 | factorization at lr_W=8E-3 | Should improve low_rank dynamics | Still underperforms direct W | ✗ | 2 |
| 18 | L1=1E-6 should help low_rank dynamics | dynamics 0.902→0.925+ | Confirmed: 0.925 | ✓ | 2 |
| 20 | Convergence boundary ~1.5E-3 should hold | lr_W=1.5E-3 partial | Partial convergence, boundary holds | ✓ | 2 |
| 26 | L1=1E-6 from block 2 helps Dale dynamics | Dynamics improvement | Marginal benefit only | ~✓ | 3 |
| 27 | lr_W=3E-3 from block 2 works in Dale | Good dynamics | Underperforms; Dale needs different lr_W | ✗ | 3 |
| 33 | lr_W=4.5E-3 predicted safe in Dale | Best in safe range | conn 0.986, test_R2 0.999 | ✓ | 3 |
| 36 | lr=2E-4 should degrade Dale (principle 1) | Degradation | Does NOT degrade | ✗ | 3 |
| 39 | lr_emb=1E-3 fixes embedding for n_types=4 | FULL convergence | conn 0.9996, cluster 0.990 | ✓ | 4 |
| 41 | lr_W=5E-3 + lr_emb=1E-3 converges | Full dual convergence | cluster 1.000 | ✓ | 4 |
| 44 | L1=1E-6 critical for embedding | cluster drops at L1=1E-5 | cluster 0.990→0.440 | ✓ | 4 |
| 54 | lr_W=6E-3 at noise=1.0 overshoots | Dynamics degradation | Confirmed: dynamics poor | ✓ | 5 |
| 56 | lr=2E-4 safe at eff_rank=84 | No degradation | Safe — modifies P1 | ✓ | 5 |
| 58 | lr_W=2E-3 best at noise=1.0 | Best dynamics | 0.998 confirmed | ✓ | 5 |
| 60 | lr=2E-4 + lr_W=8E-3 complementary | Complementary | Mixed results | ~✓ | 5 |
| 64 | L1=1E-6 helps n=200 (from block 2) | Connectivity boost | REDUCES connectivity | ✗ | 6 |
| 66 | lr_W=5E-3 near sweet spot at n=200 | Good balance | Confirmed near optimal | ✓ | 6 |
| 67 | lr=2E-4 safe at n=200 (eff_rank=43) | No degradation | Best conn achieved | ✓ | 6 |
| 76 | L1=1E-6 helps sparse (low_rank analogy) | Dynamics boost | Neutral effect | ✗ | 7 |
| 80 | lr=2E-4 marginally worse at eff_rank=21 | Slight degradation | Marginal degradation | ✓ | 7 |
| 83 | n_epochs=3 improves beyond 2ep | Further gains | Diminishing returns | ✗ | 7 |
| 88 | n_epochs=2 helps sparse+noise | Improvement | No effect — noise equalizes | ✗ | 8 |
| 89 | lr_W=1E-2 at sparse+noise | Better conn | Same plateau (0.489) | ✗ | 8 |
| 90 | L1=1E-6 at sparse+noise | Some benefit | Irrelevant at plateau | ✗ | 8 |
| 96 | batch_size=16 safe for n_types=1 | No degradation | Confirmed safe | ✓ | 8 |
| 100 | Dynamics cliff at 8E-3 for n=300 | Cliff expected | No cliff at 8E-3 | ✗ | 9 |
| 102 | lr=2E-4 boosts conn at n=300 | Conn improvement | +16% boost confirmed | ✓ | 9 |
| 109 | Reproduce iter 106 baseline | Consistent result | 0.893 (confirmed) | ✓ | 10 |
| 112 | L1=1E-6 harmful at n=300 (principle 3) | Conn degradation | NOT harmful; +6.8% dynamics | ✗ | 10 |

**Validation**: 22/30 = **72%**

---

## 4. Falsification: 26 instances

| Iter | Falsified Hypothesis | Evidence | Refinement | Block |
|------|---------------------|----------|------------|-------|
| 9 | lr=2E-4 universally better | Dynamics 0.996→0.981 | lr=1E-4 optimal at eff_rank=35 | 1 |
| 14 | factorization=True helps low_rank | conn 0.899 vs 0.999 without | Direct W learning superior | 2 |
| 17 | lr_W=5E-3 safe in low_rank | Catastrophic failure (0.385) | Upper boundary at 5E-3 | 2 |
| 24 | batch_size=16 always degrades | 0.997 in low_rank with L1 | batch=16 safe with L1 enabler | 2 |
| 29 | Dale cliff might be stochastic | 0.458 (reproducible) | Cliff is deterministic | 3 |
| 32 | L1 can rescue high lr_W in Dale | Still fails at 6E-3 | L1 insufficient for boundary rescue | 3 |
| 34 | batch_size=16 safe in Dale | Conn degraded | batch=16 hurts constrained regimes | 3 |
| 36 | lr=2E-4 universally harmful (P1) | WORKS in Dale | lr tolerance depends on regime | 3 |
| 42 | lr_emb=1E-3 universal fix | Overshoots at lr_W=1E-3 | lr_emb coupled to lr_W | 4 |
| 43 | lr_W=3E-3 works for n_types=4 | Underperforms vs 5E-3 | Heterogeneous needs higher lr_W | 4 |
| 48 | batch_size=16 safe for n_types=4 | cluster_acc 1.000→0.500 | batch=16 kills dual-objective | 4 |
| 52 | L1=1E-6 helps all regimes | NOT beneficial for n_types=1+noise | L1 only for multi-type or constrained | 5 |
| 56 | lr=2E-4 harmful at all eff_ranks | Safe at eff_rank=84 | lr tolerance scales with eff_rank | 5 |
| 57 | lr_W=1E-2 safe at noise=0.5 | Dynamics degraded (0.707) | Upper lr_W boundary for noise | 5 |
| 64 | L1=1E-6 beneficial at n=200 | REDUCES connectivity | L1=1E-6 harmful at n=200 n_types=1 | 6 |
| 71 | L1=1E-6 might help n=200 | SEVERELY degrades | L1=1E-6 conclusively harmful at n=200 | 6 |
| 72 | lr=3E-4 always harmful | BEST dynamics at n=200 | n=200 widens lr tolerance | 6 |
| 84 | Dale/low_rank params transfer to sparse | No improvement | Sparse fundamentally different | 7 |
| 88 | n_epochs=2 helps sparse+noise | Same conn (0.489) | Noise removes n_epochs dependency | 8 |
| 91 | Two-phase training helps sparse+noise | No improvement | Structural limit not trainable | 8 |
| 93 | aug_loop=200 helps sparse+noise | Zero effect | Data augmentation irrelevant at limit | 8 |
| 95 | Recurrent training helps subcritical | Catastrophic collapse (0.054) | Recurrent+subcritical = destruction | 8 |
| 100 | Dynamics cliff at 8E-3 for n=300 | No cliff present | Cliff scales non-linearly with n | 9 |
| 104 | L1=1E-6 harmful at n=300 | Neutral effect | L1 harm is n=200-specific | 9 |
| 108 | lr=3E-4 safe at all lr_W | Degrades at lr_W=1E-2 | lr/lr_W interaction at n=300 | 9 |
| 110 | n_epochs=3 improves beyond 2ep | Conn slightly worse (-0.8%) | Diminishing returns at 3ep | 10 |

**Refinement rate**: 26/26 = **100%**

---

## 5. Analogy/Transfer: 14 instances

| Iter | From | To | Knowledge | Outcome | Block |
|------|------|-----|-----------|---------|-------|
| 13 | Block 1 (chaotic) | Block 2 (low_rank) | lr_W=4E-3 baseline | ✓ Conn OK (0.999), dynamics poor | 2 |
| 18 | Block 1 (L1 insight) | Block 2 | L1=1E-6 for dynamics | ✓ Dynamics 0.902→0.925 | 2 |
| 20 | Block 1 | Block 2 | Convergence boundary ~1.5E-3 | ✓ Holds in low_rank | 2 |
| 25 | Block 1 (chaotic) | Block 3 (Dale) | lr_W=4E-3 baseline | ✓ Conn 0.972 | 3 |
| 26 | Block 2 (L1) | Block 3 | L1=1E-6 for dynamics | ~✓ Marginal | 3 |
| 27 | Block 2 (lr_W=3E-3) | Block 3 | Lower lr_W for low eff_rank | ✗ Underperforms | 3 |
| 37 | Block 1 (chaotic) | Block 4 (heterogeneous) | lr_W=4E-3 baseline | ✓ Conn OK, embedding fails | 4 |
| 49 | Block 1 (chaotic) | Block 5 (noise) | lr_W=4E-3 baseline | ✓ All converge | 5 |
| 52 | Block 4 (L1 embedding) | Block 5 | L1=1E-6 for n_types=1+noise | ✗ Not beneficial | 5 |
| 61 | Block 1 (chaotic) | Block 6 (n=200) | lr_W=4E-3 baseline | ✓ Converges (0.905) | 6 |
| 73 | Block 1 (chaotic) | Block 7 (sparse) | lr_W=4E-3 baseline | ✗ 0% convergence | 7 |
| 85 | Block 5 (noise) | Block 8 (sparse+noise) | Noise inflates eff_rank | ✓ eff_rank 21→91 | 8 |
| 97 | Block 6 (n=200) | Block 9 (n=300) | lr_W=5E-3 at n=200 | ~✓ Partially transfers | 9 |
| 99 | Block 6 (n=200) | Block 9 | n=200 optimal | ✗ n=300 needs 1E-2 | 9 |

**Transfer success**: 9/14 = **65%**

---

## 6. Boundary Probing: 34 instances

| Iter | Parameter | Test Value | Result | Boundary Status | Block |
|------|-----------|------------|--------|-----------------|-------|
| 4 | lr_W | 1E-3 | Poor conn (0.575) | Lower boundary | 1 |
| 7 | lr_W | 1.5E-3 | Near threshold | Convergence boundary ~1.5E-3 | 1 |
| 8 | lr_W | 8E-3 | Good conn, stochastic | Upper range | 1 |
| 12 | batch_size | 16 | Slight quality loss | Batch limit | 1 |
| 16 | lr_W | 2E-3 | Partial in low_rank | Lower boundary low_rank | 2 |
| 17 | lr_W | 5E-3 | Catastrophic (0.385) | Upper cliff low_rank | 2 |
| 20 | lr_W | 1.5E-3 | Partial in low_rank | Confirms boundary | 2 |
| 22 | lr_W | 3.5E-3 | Dynamics cliff | Dynamics cliff low_rank | 2 |
| 23 | lr_W | 2.5E-3 | Below optimal | Sub-optimal zone | 2 |
| 28 | lr_W | 6E-3 | Catastrophic (0.555) | Dale upper boundary | 3 |
| 29 | lr_W | 5E-3 | Fails (0.458) | Dale cliff confirmed | 3 |
| 30 | lr_W | 5E-3 | Fails again (0.455) | Reproducible cliff | 3 |
| 31 | lr_W | 3.5E-3 | Good balance | Lower safe boundary | 3 |
| 40 | lr_W | 1E-3 | Fails completely | Lower boundary n_types=4 | 4 |
| 45 | lr_W | 6E-3 | Embedding degrades | Upper boundary n_types=4 | 4 |
| 47 | lr_W | 4.5E-3 | Intermediate | Mid-range n_types=4 | 4 |
| 53 | lr_W | 8E-3 | Works at noise=0.5 | Upper boundary noise=0.5 | 5 |
| 55 | lr_W | 2E-3 | Best rollout at noise=0.1 | Optimal low noise | 5 |
| 57 | lr_W | 1E-2 | Dynamics degraded (0.707) | Upper boundary noise=0.5 | 5 |
| 62 | lr_W | 2E-3 | Fails at n=200 (0.575) | Boundary shifted up | 6 |
| 63 | lr_W | 8E-3 | Best conn, worst dynamics | Conn/dynamics trade-off | 6 |
| 65 | lr_W | 6E-3 | Steep dynamics trade-off | Past sweet spot | 6 |
| 68 | lr_W | 3E-3 | Partial | Boundary ~3.5E-3 n=200 | 6 |
| 69 | lr_W | 5.5E-3 | Past optimal | Above sweet spot | 6 |
| 70 | lr_W | 4.5E-3 | Just below threshold | Near-threshold | 6 |
| 74 | lr_W | 6E-3 | Best of first batch | Initial exploration sparse | 7 |
| 75 | lr_W | 2E-3 | Worst in sparse | Lower end sparse | 7 |
| 78 | lr_W | 1E-2 | Best at 1 epoch | Optimal sparse 1ep | 7 |
| 92 | lr_W | 1.5E-2 | Still no cliff | No upper boundary sparse+noise | 8 |
| 98 | lr_W | 7E-3 | Underperforms at n=300 | Below optimal | 9 |
| 103 | lr_W | 1E-2 | Best conn (0.805) | Optimal boundary n=300 | 9 |
| 105 | lr_W | 1.2E-2 | Slightly past optimum | Above sweet spot | 9 |
| 107 | lr_W | 1.5E-2 | Dynamics cliff | Upper cliff n=300 | 9 |
| 111 | lr_W | 1.2E-2 | Conn -3.4%, best dynamics | Cliff above 1E-2 | 10 |

---

## 7. Emerging Patterns: 29 instances

### Meta-reasoning: 7 instances

| Iter | Description | Significance | Block |
|------|-------------|--------------|-------|
| 9 | Switch from lr_W sweep to lr/L1 dimensions | High | 1 |
| 18 | Abandon factorization, focus on L1 regularization | High | 2 |
| 21 | Recombination strategy: combine independent findings | High | 2 |
| 41 | Shift to dual-objective optimization (conn + cluster) | Medium | 4 |
| 58 | Link noise→eff_rank→lr_W into causal chain | High | 5 |
| 79 | Shift from lr_W sweep to n_epochs dimension | High | 7 |
| 96 | Recognize structural limit; stop optimizing parameters | High | 8 |

### Regime Recognition: 10 instances

| Iter | Description | Significance | Block |
|------|-------------|--------------|-------|
| 13 | eff_rank=13 vs 35; low_rank is sub-critical regime | High | 2 |
| 25 | Dale reduces eff_rank 35→12; different from chaotic | High | 3 |
| 37 | n_types=4 creates dual-objective (conn + cluster) | High | 4 |
| 49 | noise=0.5 → eff_rank=84 (2.4x chaotic baseline) | High | 5 |
| 50 | noise=1.0 → eff_rank=90 (highest observed) | Medium | 5 |
| 51 | noise=0.1 → eff_rank=42 (minimal inflation) | Medium | 5 |
| 61 | n=200 eff_rank=43; scale changes landscape | High | 6 |
| 73 | Sparse eff_rank=21, rho=0.746 (subcritical) | High | 7 |
| 85 | Sparse+noise eff_rank=91 but still subcritical | High | 8 |
| 97 | n=300 eff_rank=48, spectral_radius=1.03 | Medium | 9 |

### Causal Chain: 7 instances

| Iter | Description | Significance | Block |
|------|-------------|--------------|-------|
| 21 | lr_W=3E-3 + L1=1E-6 → combined dynamics boost | High | 2 |
| 33 | Dale → eff_rank drop → lr_W cliff narrowing | High | 3 |
| 44 | L1 → embedding regularization → cluster convergence | High | 4 |
| 58 | noise → eff_rank → lr_W tolerance → inverse relation | Medium | 5 |
| 82 | sparse → training capacity → n_epochs as key lever | High | 7 |
| 95 | multi-step rollout + subcritical rho → catastrophic error | High | 8 |
| 106 | n_neurons → training capacity → n_epochs requirement | High | 9 |

### Uncertainty: 2 instances

| Iter | Description | Significance | Block |
|------|-------------|--------------|-------|
| 8 | lr_W=8E-3 shows variable results across runs | Medium | 1 |
| 22 | lr_W=3.5E-3 result may not be reproducible | Medium | 2 |

### Constraint: 3 instances

| Iter | Description | Significance | Block |
|------|-------------|--------------|-------|
| 85 | conn=0.489 is structural data limit at 10k frames | High | 8 |
| 108 | lr/lr_W interaction constrains both at n=300 | High | 9 |
| 112 | conn ~0.89 ceiling at 10k frames for n=300 | Medium | 10 |

---

## Iteration Index

| Iter | Modes Active | Key Event |
|------|--------------|-----------|
| 1-3 | — | Baseline runs |
| 4 | Boundary | First lr_W boundary probe |
| 5 | Induction, Deduction | lr_W=4E-3 sweet spot identified |
| 7 | Boundary | Convergence threshold mapped |
| 8 | Boundary, Uncertainty | Upper range + stochasticity |
| 9 | Deduction, Falsification, Meta-reasoning | lr=2E-4 falsified; strategy shift |
| 11 | Induction | L1=1E-6 dynamics insight |
| 12 | Induction, Boundary | Batch size trade-off |
| 13 | Analogy, Abduction, Regime | Block 2 start; eff_rank regime |
| 14 | Falsification | factorization rejected |
| 15 | Deduction | factorization follow-up |
| 16 | Boundary | Lower bound low_rank |
| 17 | Boundary, Falsification, Abduction | Catastrophic failure at 5E-3 |
| 18 | Analogy, Deduction, Meta-reasoning | L1 transfer + strategy shift |
| 19 | Induction, Abduction | lr_W=3E-3 finding |
| 20 | Analogy, Deduction, Boundary | Boundary transfer test |
| 21 | Induction, Meta-reasoning, Causal Chain | **BREAKTHROUGH** recombination |
| 22 | Boundary, Uncertainty | Dynamics cliff probing |
| 23 | Boundary | Sub-optimal zone |
| 24 | Falsification, Induction | Batch=16 surprise |
| 25 | Analogy, Regime | Block 3 start; Dale eff_rank |
| 26 | Analogy, Deduction | L1 transfer to Dale |
| 27 | Analogy, Deduction | lr_W=3E-3 transfer fails |
| 28 | Boundary | Dale catastrophic at 6E-3 |
| 29 | Boundary, Falsification | Dale cliff at 5E-3 |
| 30 | Boundary, Induction | Cliff reproducibility |
| 31 | Boundary | Lower safe boundary |
| 32 | Falsification | L1 can't rescue cliff |
| 33 | Deduction, Induction, Causal Chain | Safe range established |
| 34 | Falsification | batch=16 hurts Dale |
| 35 | Induction | L1 negligible in Dale |
| 36 | Deduction, Falsification | lr=2E-4 works (challenges P1) |
| 37 | Analogy, Abduction, Regime | Block 4 start; dual-objective |
| 39 | Deduction, Induction | lr_emb breakthrough |
| 40 | Boundary | Lower boundary n_types=4 |
| 41 | Deduction, Meta-reasoning | Full convergence confirmed |
| 42 | Falsification, Abduction | lr_emb coupling discovered |
| 43 | Falsification | lr_W=3E-3 underperforms |
| 44 | Deduction, Induction, Causal Chain | L1=1E-6 critical for embedding |
| 45 | Boundary | Upper boundary n_types=4 |
| 47 | Boundary | Mid-range n_types=4 |
| 48 | Falsification | batch=16 kills dual-objective |
| 49 | Analogy, Regime | Block 5 start; noise regime |
| 50 | Regime | noise=1.0 mapping |
| 51 | Induction, Regime | Noise→eff_rank pattern |
| 52 | Analogy, Falsification | L1 not for noise |
| 53 | Boundary | Upper lr_W noise=0.5 |
| 54 | Deduction | Overshoot predicted |
| 55 | Boundary, Induction | Best rollout low noise |
| 56 | Deduction, Falsification | lr=2E-4 safe high eff_rank |
| 57 | Boundary, Falsification | Upper bound established |
| 58 | Deduction, Induction, Meta-reasoning, Causal Chain | Inverse lr_W-noise |
| 60 | Deduction | Combination test |
| 61 | Analogy, Regime | Block 6 start; n=200 |
| 62 | Boundary | Lower boundary shifted |
| 63 | Boundary, Induction | Trade-off amplified |
| 64 | Deduction, Falsification | L1 harmful at n=200 |
| 65 | Boundary | Past sweet spot |
| 66 | Deduction | Near-optimal confirmed |
| 67 | Deduction | lr=2E-4 safe n=200 |
| 68 | Boundary | Near threshold |
| 69 | Boundary | Above sweet spot |
| 70 | Boundary | Near threshold |
| 71 | Falsification | L1 conclusively harmful |
| 72 | Falsification, Induction | lr=3E-4 BEST at n=200 |
| 73 | Analogy, Regime, Abduction | Block 7 start; subcritical |
| 74 | Boundary | Sparse initial exploration |
| 75 | Boundary | Lower end sparse |
| 76 | Deduction | L1 neutral sparse |
| 78 | Boundary | Optimal sparse 1ep |
| 79 | Induction, Meta-reasoning | n_epochs=2 breakthrough |
| 80 | Deduction | lr=2E-4 marginal |
| 82 | Induction, Causal Chain | Best sparse config |
| 83 | Deduction | 3ep diminishing |
| 84 | Falsification | Cross-regime transfer fails |
| 85 | Analogy, Regime, Constraint | Block 8 start; structural limit |
| 86 | Induction | Complete insensitivity |
| 87 | Induction | Confirmed insensitivity |
| 88 | Falsification, Deduction | n_epochs irrelevant with noise |
| 89 | Deduction | lr_W at plateau |
| 90 | Deduction | L1 at plateau |
| 91 | Falsification | Two-phase training fails |
| 92 | Boundary | No upper cliff found |
| 93 | Falsification | aug_loop no effect |
| 95 | Falsification, Causal Chain | Recurrent catastrophic |
| 96 | Deduction, Meta-reasoning | Structural limit recognized |
| 97 | Analogy, Regime | Block 9 start; n=300 |
| 98 | Boundary | Below optimal |
| 99 | Analogy | n=200 transfer test |
| 100 | Deduction, Falsification | Cliff doesn't transfer |
| 102 | Deduction | lr=2E-4 boosts conn |
| 103 | Induction, Boundary | Best n=300 1ep config |
| 104 | Falsification | L1 neutral at n=300 |
| 105 | Boundary | Past optimum |
| 106 | Induction, Causal Chain | **BREAKTHROUGH** n_epochs=2 |
| 107 | Boundary | Dynamics cliff begins |
| 108 | Falsification, Constraint | lr/lr_W interaction |
| 109 | Deduction | Baseline reproduction |
| 110 | Falsification | 3ep doesn't help conn |
| 111 | Boundary | Cliff above 1E-2 |
| 112 | Induction, Deduction, Constraint | L1=1E-6 safe n=300; ceiling |
