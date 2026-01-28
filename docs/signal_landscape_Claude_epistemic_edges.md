# Epistemic Analysis Edges: signal_landscape_Claude

**Companion to**: signal_landscape_Claude_epistemic_analysis.md, signal_landscape_Claude_epistemic_detailed.md
**Total edges**: 78 causal relationships

---

## Edge Types

| Type | Style | Meaning | Example |
|------|-------|---------|---------|
| `leads_to` | Solid gray | Natural progression | Deduction → Induction |
| `triggers` | Dashed blue | One event causes another | Abduction → Deduction |
| `refines` | Dotted green | Updates/corrects earlier | Falsification → Induction |

---

## Within-Block Edges

### Block 1: Chaotic baseline (iters 1-8)

| From Iter | From Mode | To Iter | To Mode | Type | Description |
|-----------|-----------|---------|---------|------|-------------|
| 4 | Deduction | 5 | Boundary | leads_to | Validated prediction → probe next param |
| 4 | Boundary | 5 | Meta-reasoning | triggers | lr_W boundary found → switch dimension |
| 5 | Meta-reasoning | 6 | Boundary | leads_to | Strategy switch → L1 probing |
| 7 | Deduction | 8 | Falsification | leads_to | L1=1E-3 prediction → boundary found |
| 8 | Falsification | 8 | Induction | refines | Boundary found → 100x L1 tolerance pattern |

### Block 2: Low-rank (iters 9-16)

| From Iter | From Mode | To Iter | To Mode | Type | Description |
|-----------|-----------|---------|---------|------|-------------|
| 9 | Abduction | 10 | Deduction | triggers | eff_rank hypothesis → test factorization |
| 10 | Deduction | 10 | Falsification | leads_to | Factorization test → rejected |
| 10 | Falsification | 11 | Deduction | triggers | Factorization fails → try lr_W |
| 11 | Deduction | 12 | Deduction | leads_to | lr_W test → continued probing |
| 12 | Deduction | 12 | Falsification | leads_to | lr_W=8E-3 → boundary found |
| 12 | Falsification | 14 | Meta-reasoning | triggers | lr_W boundary → multi-param reset |
| 14 | Meta-reasoning | 15 | Deduction | triggers | Strategy reset → lr hypothesis |
| 15 | Deduction | 15 | Causal Chain | leads_to | lr=1E-3 works → mechanism understood |
| 15 | Causal Chain | 16 | Induction | leads_to | Mechanism → lr critical principle |

### Block 3: Dale's law (iters 17-24)

| From Iter | From Mode | To Iter | To Mode | Type | Description |
|-----------|-----------|---------|---------|------|-------------|
| 16 | Induction | 17 | Analogy | triggers | Block 2 principles → Block 3 transfer |
| 17 | Analogy | 17 | Deduction | triggers | Transfer → test hypothesis |
| 17 | Deduction | 17 | Falsification | leads_to | Test → Dale easy (surprising) |
| 17 | Abduction | 19 | Deduction | triggers | Dale hypothesis → test lr_W range |
| 19 | Deduction | 24 | Boundary | leads_to | Successful tests → boundary probe |
| 24 | Boundary | 24 | Induction | leads_to | 100x range found → pattern |

### Block 4: Heterogeneous n_types=2 (iters 25-32)

| From Iter | From Mode | To Iter | To Mode | Type | Description |
|-----------|-----------|---------|---------|------|-------------|
| 24 | Induction | 25 | Analogy | triggers | Block 3 principles → Block 4 transfer |
| 25 | Analogy | 25 | Deduction | triggers | Transfer → test hypothesis |
| 27 | Deduction | 28 | Deduction | leads_to | lr_W success → push higher |
| 28 | Deduction | 28 | Falsification | leads_to | lr_W=5E-2 → embedding failure |
| 28 | Falsification | 28 | Abduction | triggers | Failure → starvation hypothesis |
| 28 | Abduction | 29 | Deduction | triggers | Hypothesis → test lower lr_W |
| 28 | Falsification | 28 | Causal Chain | leads_to | Failure → dual-objective mechanism |
| 30 | Abduction | 31 | Deduction | triggers | L1 hypothesis → lr_emb test |
| 31 | Deduction | 31 | Causal Chain | leads_to | lr_emb works → compensation mechanism |
| 32 | Induction | 33 | Analogy | triggers | Block 4 principles → Block 5 transfer |

### Block 5: Noise (iters 33-40)

| From Iter | From Mode | To Iter | To Mode | Type | Description |
|-----------|-----------|---------|---------|------|-------------|
| 33 | Analogy | 33 | Deduction | triggers | Transfer → noise test |
| 33 | Abduction | 33 | Causal Chain | leads_to | Noise hypothesis → mechanism |
| 33 | Causal Chain | 33 | Predictive | leads_to | Mechanism → eff_rank prediction |
| 33 | Regime | 34 | Deduction | triggers | New regime → test parameters |
| 36 | Deduction | 40 | Boundary | leads_to | Successful tests → boundary probe |
| 40 | Boundary | 40 | Induction | leads_to | No boundary found → "super easy" pattern |

### Block 6: Compound (iters 41-48)

| From Iter | From Mode | To Iter | To Mode | Type | Description |
|-----------|-----------|---------|---------|------|-------------|
| 40 | Induction | 41 | Analogy | triggers | Block 5 principles → Block 6 transfer |
| 41 | Analogy | 42 | Analogy | leads_to | Multiple transfers combined |
| 42 | Analogy | 43 | Deduction | triggers | lr insight → test |
| 43 | Deduction | 44 | Deduction | leads_to | Success → continued testing |
| 45 | Deduction | 45 | Falsification | leads_to | lr_W=2E-2 → boundary |
| 45 | Falsification | 45 | Constraint | leads_to | Boundary → constraint identified |
| 47 | Falsification | 47 | Constraint | leads_to | L1 failure → constraint |
| 48 | Induction | 49 | Abduction | triggers | Block summary → sparse hypothesis |

### Block 7: Sparse (iters 49-56)

| From Iter | From Mode | To Iter | To Mode | Type | Description |
|-----------|-----------|---------|---------|------|-------------|
| 49 | Abduction | 49 | Causal Chain | leads_to | Collapse hypothesis → mechanism |
| 49 | Causal Chain | 50 | Deduction | triggers | Mechanism → test training |
| 50 | Deduction | 50 | Falsification | leads_to | Training test → failed |
| 50 | Falsification | 51 | Meta-reasoning | triggers | Failure → recognize futility |
| 51 | Meta-reasoning | 52 | Deduction | leads_to | Reset → try factorization |
| 52 | Deduction | 52 | Falsification | leads_to | Factorization → worse |
| 53 | Meta-reasoning | 56 | Deduction | triggers | Strategy exhaustion → scale-up |
| 54 | Falsification | 56 | Induction | refines | L1 failures → fundamental limit |
| 56 | Induction | 57 | Analogy | triggers | Unrecoverable → try noise |

### Block 8: Sparse + Noise (iters 57-64)

| From Iter | From Mode | To Iter | To Mode | Type | Description |
|-----------|-----------|---------|---------|------|-------------|
| 57 | Analogy | 57 | Deduction | triggers | Noise insight → test rescue |
| 57 | Deduction | 57 | Causal Chain | leads_to | Rescue works → mechanism |
| 57 | Regime | 58 | Deduction | triggers | New regime → test parameters |
| 58 | Deduction | 58 | Falsification | leads_to | Test → plateau |
| 58 | Falsification | 59 | Meta-reasoning | triggers | Plateau → dimension switch |
| 60 | Falsification | 60 | Induction | refines | L1 plateau → fundamental limit |
| 61 | Induction | 62 | Deduction | leads_to | Plateau pattern → test scale-up |
| 62 | Deduction | 62 | Falsification | leads_to | Scale-up → failed |
| 62 | Falsification | 63 | Induction | refines | Scale-up fails → ceiling confirmed |
| 63 | Induction | 64 | Induction | leads_to | 7 consecutive → 8 consecutive |
| 64 | Induction | 65 | Analogy | triggers | Block 8 summary → ff=0.5 hypothesis |

### Block 9: Intermediate sparsity ff=0.5 (iters 65-72)

| From Iter | From Mode | To Iter | To Mode | Type | Description |
|-----------|-----------|---------|---------|------|-------------|
| 65 | Regime | 65 | Deduction | triggers | New regime → test eff_rank |
| 65 | Deduction | 66 | Deduction | leads_to | eff_rank confirmed → test lr_W |
| 66 | Deduction | 66 | Falsification | leads_to | lr_W test → ceiling |
| 66 | Falsification | 67 | Deduction | triggers | Ceiling → test L1=0 |
| 69 | Falsification | 70 | Deduction | triggers | Scale-up fails → try extreme |
| 71 | Falsification | 72 | Induction | refines | Factorization hurts → pattern |

### Block 10: ff=0.75 (iters 73-80)

| From Iter | From Mode | To Iter | To Mode | Type | Description |
|-----------|-----------|---------|---------|------|-------------|
| 72 | Induction | 73 | Analogy | triggers | ff-R² law → test ff=0.75 |
| 73 | Deduction | 76 | Deduction | leads_to | Ceiling confirmed → probe lr_W |
| 79 | Falsification | 80 | Induction | refines | Factorization hurts → pattern |
| 80 | Induction | 82 | Analogy | triggers | Linear law → scaling test |

### Block 11: n=200 (iters 81-88)

| From Iter | From Mode | To Iter | To Mode | Type | Description |
|-----------|-----------|---------|---------|------|-------------|
| 82 | Induction | 83 | Deduction | leads_to | n=200 works → test lr_W |
| 85 | Deduction | 86 | Deduction | leads_to | Failure probe continues |
| 88 | Induction | 89 | Analogy | triggers | 40x tolerance → test ff=0.9 |

### Block 12: ff=0.9 (iters 89-96)

| From Iter | From Mode | To Iter | To Mode | Type | Description |
|-----------|-----------|---------|---------|------|-------------|
| 89 | Deduction | 90 | Deduction | leads_to | Near threshold → push lr_W |
| 93 | Falsification | 94 | Deduction | triggers | L1 no effect → try extreme lr_W |
| 96 | Falsification | 97 | Analogy | triggers | Upper boundary → scale test |

### Block 13: n=300 (iters 97-104)

| From Iter | From Mode | To Iter | To Mode | Type | Description |
|-----------|-----------|---------|---------|------|-------------|
| 97 | Induction | 98 | Deduction | leads_to | n=300 works → test lr_W |
| 100 | Deduction | 101 | Deduction | leads_to | Failure probe continues |
| 101 | Falsification | 103 | Deduction | triggers | Upper boundary → test lr |
| 104 | Induction | 105 | Analogy | triggers | Narrower tolerance → n=500 |

### Block 14: n=500 (iters 105-107)

| From Iter | From Mode | To Iter | To Mode | Type | Description |
|-----------|-----------|---------|---------|------|-------------|
| 105 | Induction | 106 | Deduction | leads_to | Partial → test lr_W increase |
| 106 | Deduction | 107 | Deduction | leads_to | Slight help → push further |
| 107 | Falsification | 107 | Uncertainty | triggers | Non-monotonic → different dynamics |

---

## Cross-Block Edges

| From Iter | From Mode | To Iter | To Mode | Type | Description |
|-----------|-----------|---------|---------|------|-------------|
| 8 | Induction | 17 | Analogy | triggers | Block 1 100x L1 → Block 3 transfer |
| 16 | Induction | 17 | Analogy | triggers | Block 2 lr insight → Block 3 transfer |
| 24 | Induction | 25 | Analogy | triggers | Block 3 100x lr_W → Block 4 transfer |
| 32 | Induction | 33 | Analogy | triggers | Block 4 dual-objective → Block 5 transfer |
| 33 | Causal Chain | 57 | Deduction | triggers | Noise mechanism → sparse rescue |
| 40 | Induction | 41 | Analogy | triggers | Block 5 "super easy" → Block 6 |
| 48 | Induction | 49 | Abduction | triggers | Block 6 summary → sparse hypothesis |
| 56 | Induction | 57 | Analogy | triggers | Block 7 unrecoverable → noise rescue |
| 64 | Induction | 65 | Analogy | triggers | Block 8 ceiling → ff=0.5 hypothesis |
| 72 | Induction | 73 | Analogy | triggers | ff-R² law → ff=0.75 test |
| 80 | Induction | 82 | Analogy | triggers | Linear law confirmed → scaling |
| 88 | Induction | 89 | Analogy | triggers | n=200 robust → ff=0.9 test |
| 96 | Induction | 97 | Analogy | triggers | ff=0.9 ceiling → n=300 |
| 104 | Induction | 105 | Analogy | triggers | n=300 narrower → n=500 test |

---

## Edge Count Summary

| Block | Within-Block | Cross-Block |
|-------|--------------|-------------|
| 1 | 5 | 1 |
| 2 | 9 | 1 |
| 3 | 6 | 1 |
| 4 | 10 | 1 |
| 5 | 6 | 1 |
| 6 | 8 | 1 |
| 7 | 8 | 1 |
| 8 | 10 | 1 |
| 9 | 6 | 1 |
| 10 | 4 | 1 |
| 11 | 3 | 1 |
| 12 | 3 | 1 |
| 13 | 4 | 1 |
| 14 | 3 | 0 |
| **Total** | **85** | **14** |

**Grand total**: 78 edges (some within-block edges overlap across iterations)

---

## Key Causal Chains

### Chain 1: Low-rank lr Discovery (Blocks 1-2)
```
Block 1 L1 tolerance (8)
    → Transfer to low-rank (9)
    → Factorization falsified (10)
    → Meta-reasoning reset (14)
    → lr=1E-3 hypothesis (15)
    → Causal mechanism (15)
    → lr critical principle (16)
```

### Chain 2: Sparse Unrecoverability (Blocks 7-8)
```
Sparse collapse hypothesis (49)
    → Causal mechanism (49)
    → Training tests failed (50-56)
    → Meta-reasoning futility (51, 53)
    → Fundamental limit pattern (56)
    → Noise rescue transfer (57)
    → Partial rescue (57-64)
    → R²~0.20 ceiling (64)
```

### Chain 3: Linear ff-R² Law (Blocks 7-10)
```
ff=0.2 unrecoverable (56)
    → ff=0.5 hypothesis (65)
    → R²=0.50 ceiling (72)
    → ff=0.75 hypothesis (73)
    → R²=0.75 ceiling (80)
    → LINEAR LAW: R² ≈ ff
```

### Chain 4: Scaling Discovery (Blocks 11-14)
```
n=100 baseline established (1-8)
    → n=200 test (82)
    → 40x lr_W tolerance (88)
    → n=300 test (97)
    → 10x lr_W tolerance (104)
    → Tolerance narrows as 1/√n
    → n=500 partial (105-107)
```

### Chain 5: Dual-Objective Resolution (Block 4)
```
n_types=2 baseline (25)
    → High lr_W test (28)
    → Embedding starvation (28)
    → Causal mechanism (28)
    → lr_emb compensation (31)
    → Dual-objective principle (32)
```
