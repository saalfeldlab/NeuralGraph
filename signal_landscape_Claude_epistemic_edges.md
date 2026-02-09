# Epistemic Analysis Edges: signal_landscape_Claude

**Companion to**: signal_landscape_Claude_epistemic_analysis.md, signal_landscape_Claude_epistemic_detailed.md
**Total edges**: ~150 causal relationships (35 detailed below for blocks 1-10; full set in plot_epistemic_interactive.py)

---

## Within-Block Edges

### Block 1 (Chaotic baseline, iters 1–12)

| From Iter | From Mode | To Iter | To Mode | Type | Description |
|-----------|-----------|---------|---------|------|-------------|
| 4 | Boundary | 5 | Deduction | leads_to | Lower boundary found → predicted 4E-3 sweet spot |
| 5 | Deduction | 8 | Induction | leads_to | Sweet spot confirmed → cumulative lr_W ordering pattern |
| 5 | Induction | 9 | Deduction | triggers | lr_W=4E-3 established → tested lr=2E-4 at optimal lr_W |
| 9 | Deduction | 9 | Falsification | leads_to | lr=2E-4 prediction tested → rejected (0.996→0.981) |
| 8 | Boundary | 11 | Induction | leads_to | Upper lr_W range mapped → explored L1 at low lr_W |
| 9 | Meta-reasoning | 11 | Induction | triggers | Shift from lr_W to secondary dims → L1=1E-6 finding |

### Block 2 (Low-rank, iters 13–24)

| From Iter | From Mode | To Iter | To Mode | Type | Description |
|-----------|-----------|---------|---------|------|-------------|
| 13 | Abduction | 18 | Deduction | triggers | eff_rank hypothesis → predicted L1=1E-6 should help |
| 13 | Regime | 19 | Abduction | triggers | Low eff_rank identified → hypothesized lower lr_W needed |
| 17 | Boundary | 19 | Induction | leads_to | lr_W=5E-3 failure → explored 3E-3 → found new optimum |
| 18 | Deduction | 21 | Induction | leads_to | L1=1E-6 validated → combined with lr_W=3E-3 for breakthrough |
| 19 | Induction | 21 | Induction | leads_to | lr_W=3E-3 best dynamics → combined with L1=1E-6 |
| 21 | Meta-reasoning | 21 | Causal | leads_to | Recombination strategy → causal chain constructed |
| 14 | Falsification | 24 | Induction | refines | factorization rejected → direct W learning with L1=1E-6 explored |
| 22 | Boundary | 23 | Boundary | leads_to | Sharp dynamics cliff at 3.5E-3 → probed 2.5E-3 lower bound |

### Block 3 (Dale's law, iters 25–36)

| From Iter | From Mode | To Iter | To Mode | Type | Description |
|-----------|-----------|---------|---------|------|-------------|
| 25 | Regime | 28 | Boundary | triggers | eff_rank=12 discovered → probed upper lr_W boundary |
| 28 | Boundary | 29 | Boundary | leads_to | lr_W=6E-3 fails → tested 5E-3 |
| 29 | Boundary | 30 | Induction | leads_to | lr_W=5E-3 first failure → second run confirms reproducibility |
| 30 | Induction | 33 | Deduction | leads_to | Cliff at 5E-3 established → predicted 4.5E-3 safe |
| 33 | Deduction | 33 | Induction | leads_to | 4.5E-3 validated → safe range [3.5E-3, 4.5E-3] mapped |
| 29 | Falsification | 32 | Falsification | triggers | lr_W=5E-3 fails with L1=1E-6 → tested if L1 rescues lr_W=6E-3 |
| 34 | Falsification | 36 | Deduction | leads_to | batch_size effect confirmed → tested lr=2E-4 principle |

### Block 4 (Heterogeneous, iters 37–48)

| From Iter | From Mode | To Iter | To Mode | Type | Description |
|-----------|-----------|---------|---------|------|-------------|
| 37 | Abduction | 39 | Deduction | triggers | lr_emb insufficient hypothesis → predicted lr_emb=1E-3 helps |
| 39 | Deduction | 41 | Deduction | leads_to | lr_emb=1E-3 validated → tested lr_W=5E-3 with it |
| 39 | Induction | 44 | Deduction | triggers | L1=1E-6 pattern → tested if L1 matters for embedding at high eff_rank |
| 42 | Falsification | 44 | Causal | triggers | lr_emb overshoot → investigated L1/embedding mechanism |
| 41 | Deduction | 48 | Falsification | leads_to | FULL convergence established → tested batch_size=16 at best config |

### Block 5 (Noise, iters 49–60)

| From Iter | From Mode | To Iter | To Mode | Type | Description |
|-----------|-----------|---------|---------|------|-------------|
| 49 | Regime | 53 | Boundary | triggers | eff_rank=84 observed → probed lr_W=8E-3 at high eff_rank |
| 51 | Induction | 55 | Boundary | leads_to | 100% convergence observed → probed lr_W=2E-3 boundary |
| 53 | Boundary | 57 | Boundary | leads_to | lr_W=8E-3 works → probed lr_W=1E-2 upper limit |
| 54 | Deduction | 58 | Deduction | leads_to | lr_W=6E-3 degrades at noise=1.0 → predicted lr_W=2E-3 best |
| 55 | Induction | 58 | Induction | leads_to | Rollout pattern → confirmed inverse lr_W-noise relation |

### Block 6 (Scale n=200, iters 61–64)

| From Iter | From Mode | To Iter | To Mode | Type | Description |
|-----------|-----------|---------|---------|------|-------------|
| 62 | Boundary | 63 | Boundary | leads_to | lr_W=2E-3 fails → tested lr_W=8E-3 upper range |

---

## Cross-Block Edges

| From Iter | From Mode | To Iter | To Mode | Type | Description |
|-----------|-----------|---------|---------|------|-------------|
| 11 | Induction | 18 | Deduction | triggers | L1=1E-6 dynamics insight (block 1) → transferred to low_rank (block 2) |
| 5 | Induction | 25 | Analogy | triggers | lr_W=4E-3 sweet spot (block 1) → transferred to Dale (block 3) |
| 21 | Induction | 26 | Analogy | triggers | L1=1E-6 breakthrough (block 2) → transferred to Dale (block 3) |
| 19 | Induction | 27 | Analogy | triggers | lr_W=3E-3 optimal in low_rank (block 2) → tested in Dale (block 3) |
| 5 | Induction | 37 | Analogy | triggers | lr_W=4E-3 baseline (block 1) → transferred to heterogeneous (block 4) |
| 5 | Induction | 49 | Analogy | triggers | lr_W=4E-3 baseline (block 1) → transferred to noise regime (block 5) |
| 44 | Induction | 52 | Analogy | triggers | L1=1E-6 for embedding (block 4) → tested at noise regime (block 5) |
| 9 | Falsification | 56 | Deduction | triggers | lr=1E-4 optimal (block 1) → tested at eff_rank=84 (block 5) |
| 5 | Induction | 61 | Analogy | triggers | lr_W=4E-3 baseline (block 1) → transferred to n=200 (block 6) |

---

## Edge Summary

| Category | Count |
|----------|-------|
| Within-block | 26 |
| Cross-block | 9 |
| **Total** | **35** |

| Edge Type | Count |
|-----------|-------|
| leads_to | 21 |
| triggers | 12 |
| refines | 1 |
| rejects | 1 |

---

## Causal Chain Highlights

### Chain 1: Low-rank Breakthrough (iters 11→18→19→21)
```
Block 1 iter 11 (L1=1E-6 best dynamics)
  →[triggers] Block 2 iter 18 (L1=1E-6 in low_rank: 0.925)
  →[leads_to] iter 19 (lr_W=3E-3 best dynamics: 0.943)
  →[leads_to] iter 21 (BREAKTHROUGH: 3E-3 + 1E-6 = 0.996)
```

### Chain 2: Dale Cliff Exploration (iters 25→28→29→30→33)
```
iter 25 (Dale eff_rank=12 discovered)
  →[triggers] iter 28 (lr_W=6E-3 fails: 0.555)
  →[leads_to] iter 29 (lr_W=5E-3 fails: 0.458)
  →[leads_to] iter 30 (reproduced: 0.455)
  →[leads_to] iter 33 (4.5E-3 safe: 0.986)
```

### Chain 3: Embedding Learning (iters 37→39→44)
```
iter 37 (embedding fails at lr_emb=2.5E-4)
  →[triggers] iter 39 (lr_emb=1E-3 → FULL convergence)
  →[triggers] iter 44 (L1=1E-6 critical for embedding)
```

### Chain 4: Noise Regime (iters 49→54→58)
```
iter 49 (noise=0.5: eff_rank=84, conn=1.000)
  →[triggers] iter 54 (lr_W=6E-3 degrades at noise=1.0)
  →[leads_to] iter 58 (lr_W=2E-3 best at noise=1.0: 0.998)
```
