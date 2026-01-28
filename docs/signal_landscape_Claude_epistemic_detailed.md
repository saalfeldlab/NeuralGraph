# Epistemic Analysis Detailed Log: signal_landscape_Claude

**Companion to**: signal_landscape_Claude_epistemic_analysis.md
**Total instances**: 299 reasoning events across 107 iterations

---

## 1. Induction: 48 instances

| Iter | Observation | Induced Pattern | Type | Block |
|------|-------------|-----------------|------|-------|
| 4 | lr_W 2E-3 to 5E-3 converge | 2.5x lr_W range works | Single | 1 |
| 8 | 6/8 converged with varied L1 | 100x L1 tolerance (1E-6 to 1E-4) | Cumulative (6 obs) | 1 |
| 11 | lr_W=8E-3 improves R² | Higher lr_W better for low_rank | Single | 2 |
| 16 | lr=1E-3 breakthrough | MLP lr critical for low eff_rank | Cumulative (4 obs) | 2 |
| 19 | Dale 3 consecutive converged | Dale regime robust | Cumulative (3 obs) | 3 |
| 24 | lr_W=5E-1 still works | 100x lr_W range for Dale | Cumulative (8 obs) | 3 |
| 27 | n_types=2 baseline converges | Heterogeneous robust | Single | 4 |
| 32 | Block 4 complete | Dual-objective needs lr_emb compensation | Cumulative (8 obs) | 4 |
| 33 | Noise eff_rank=83 | Noise increases effective rank | Single | 5 |
| 34 | Noise tolerates lr_W | Noise regime robust | Cumulative (2 obs) | 5 |
| 39 | 7/7 converged | Noise is "super easy mode" | Cumulative (7 obs) | 5 |
| 40 | No boundary found | Noise regime extremely tolerant | Cumulative (8 obs) | 5 |
| 43 | Both lr tuned work | Compound needs both lr and lr_W | Cumulative (3 obs) | 6 |
| 44 | lr_W upper higher than expected | Compound tolerates 1E-2 | Single | 6 |
| 48 | Block 6 complete | Compound difficulty confirmed | Cumulative (8 obs) | 6 |
| 51 | 3 failures in row | Sparse training unsuccessful | Cumulative (3 obs) | 7 |
| 56 | 8 failures | Sparse ff=0.2 UNRECOVERABLE | Cumulative (8 obs) | 7 |
| 57 | Noise rescues eff_rank | eff_rank 6→92 with noise | Single | 8 |
| 60 | Plateau at R²~0.20 | Noise rescues rank not R² | Cumulative (4 obs) | 8 |
| 61 | 5 consecutive partial | Plateau is stable | Cumulative (5 obs) | 8 |
| 63 | 7 consecutive partial | Plateau confirmed | Cumulative (7 obs) | 8 |
| 64 | 8 consecutive partial | Definitive plateau at 0.20 | Cumulative (8 obs) | 8 |
| 65 | ff=0.5 eff_rank=26 | Intermediate sparsity recoverable | Single | 9 |
| 67 | test_pearson recovered | L1=0 helps activity prediction | Single | 9 |
| 68 | 4 consecutive partial at 0.496 | ff=0.5 has ceiling | Cumulative (4 obs) | 9 |
| 69 | Scale-up didn't break plateau | Ceiling is fundamental | Cumulative (5 obs) | 9 |
| 70 | 6 consecutive partial | 10x lr_W range all fail | Cumulative (6 obs) | 9 |
| 72 | Block 9 complete | R² ≈ filling_factor | Cumulative (8 obs) | 9 |
| 74 | ff=0.75 R²=0.744 | Linear ff-R² continues | Single | 10 |
| 78 | 6 consecutive at 0.74-0.75 | ff=0.75 ceiling confirmed | Cumulative (6 obs) | 10 |
| 80 | Block 10 complete | LINEAR LAW: R² ≈ ff | Cumulative (8 obs) | 10 |
| 82 | n=200 converges easily | Model scales to 200 neurons | Single | 11 |
| 83 | lr_W=8E-3 works for n=200 | Same tolerance as n=100 | Cumulative (2 obs) | 11 |
| 86 | lr_W=5E-2 still converges | n=200 has wide tolerance | Cumulative (5 obs) | 11 |
| 88 | lr_W=2E-1 still converges | 40x lr_W range for n=200 | Cumulative (7 obs) | 11 |
| 90 | ff=0.9 R²=0.907 | Breaks linear ceiling slightly | Single | 12 |
| 91 | Plateau at 0.907 | ff=0.9 ceiling found | Cumulative (2 obs) | 12 |
| 93 | L1=0 no effect | Ceiling is fundamental | Cumulative (4 obs) | 12 |
| 96 | Block 12 complete | ff=0.9 ceiling at 0.907 | Cumulative (8 obs) | 12 |
| 97 | n=300 converges first try | Model scales to 300 | Single | 13 |
| 102 | 5 converged out of 6 | n=300 87.5% convergence | Cumulative (5 obs) | 13 |
| 104 | Block 13 complete | n=300 needs narrower lr_W | Cumulative (7 obs) | 13 |
| 105 | n=500 partial at baseline | n=500 harder | Single | 14 |
| 107 | Non-monotonic lr_W response | n=500 behaves differently | Cumulative (3 obs) | 14 |

---

## 2. Abduction: 32 instances

| Iter | Observation | Hypothesis | Block |
|------|-------------|------------|-------|
| 9 | R²=0.355 with baseline | eff_rank=11 causes difficulty | 2 |
| 10 | Factorization no help | Over-constrains model | 2 |
| 15 | lr=1E-3 breakthrough | MLP undertrained at lr=1E-4 | 2 |
| 17 | Dale reduces eff_rank | E/I constraint affects rank | 3 |
| 28 | lr_W=5E-2 breaks embedding | High lr_W starves embedding | 4 |
| 30 | L1 hurts embedding | Sparsity conflicts with clustering | 4 |
| 31 | lr_emb compensates | Separate embedding lr helps | 4 |
| 33 | Noise eff_rank=83 | Noise adds variance to signals | 5 |
| 41 | Compound hard | Low_rank + n_types multiply difficulty | 6 |
| 47 | L1=1E-4 fails | L1 tolerance narrow for compound | 6 |
| 49 | eff_rank=6 collapse | Sparse causes fixed-point collapse | 7 |
| 50 | Training can't fix | Problem is data not training | 7 |
| 52 | Constraints can't fix | Model architecture not issue | 7 |
| 54 | L1=0 overfits to noise | Without regularization, fits noise | 7 |
| 57 | Noise rescues training | Higher eff_rank enables learning | 8 |
| 66 | lr_W hurts test_pearson | Trade-off between W and MLP | 9 |
| 75 | Higher lr_W at ceiling | lr_W can push slightly above ff | 10 |
| 79 | Factorization hurts ff=0.75 | Over-constraints hurt sparse | 10 |
| 84 | Higher lr_W trades off | W vs MLP optimization conflict | 11 |
| 89 | ff=0.9 near threshold | 10% sparsity barely affects | 12 |
| 94 | Aggressive lr_W no effect | Ceiling is fundamental | 12 |
| 106 | n=500 needs more lr_W | Larger network needs adjustment | 14 |
| 107 | lr_W=2E-2 hurt | Optimal is around 1E-2 | 14 |

---

## 3. Deduction: 67 instances

| Iter | Hypothesis | Prediction | Outcome | ✓/✗ | Block |
|------|-----------|------------|---------|-----|-------|
| 4 | lr_W approaching boundary | R² will degrade at extreme | R²=0.922 (still OK) | ✓ | 1 |
| 7 | L1 extreme test | L1=1E-3 may hurt | R²=0.922 (still OK) | ✗ | 1 |
| 8 | L1=1E-3 will fail | Boundary exists | R²=0.762 (found!) | ✓ | 1 |
| 10 | Factorization helps low-rank | Will improve R² | R²=0.355 (hurt) | ✗ | 2 |
| 11 | Higher lr_W helps | Will improve low-rank | R²=0.355→0.351 (no) | ✗ | 2 |
| 12 | lr_W=8E-3 will help | Continued improvement | R²=0.351 (boundary) | ✗ | 2 |
| 15 | lr=1E-3 breakthrough | Will fix low-rank | R²=0.953 (YES!) | ✓ | 2 |
| 16 | lr=1E-3 confirmed | Reproducible | R²=0.953 (confirmed) | ✓ | 2 |
| 17 | Block 1-2 settings transfer | Will work for Dale | R²=0.989 (yes) | ✓ | 3 |
| 18 | lr_W=8E-3 works | Continues to work | R²=0.998 (yes) | ✓ | 3 |
| 19 | lr_W=2E-2 works | Wide tolerance | R²=0.999 (yes) | ✓ | 3 |
| 20 | lr_W=5E-2 works | Even wider | R²=0.999 (yes) | ✓ | 3 |
| 22 | lr_W=1E-1 works | Extreme test | R²=0.999 (yes!) | ✓ | 3 |
| 24 | lr_W=5E-1 works | Maximum test | R²=0.9998 (yes!) | ✓ | 3 |
| 25 | Blocks 2-3 transfer | Settings will work | R²=0.988 (yes) | ✓ | 4 |
| 27 | lr_W=1E-2 works | Wider lr_W | R²=0.992 (yes) | ✓ | 4 |
| 28 | lr_W=5E-2 works | Even wider | R²=0.85 (failed emb) | ✗ | 4 |
| 29 | lr_W=2E-2 restores | Lower lr_W fixes | R²=0.93 (partial) | ✓ | 4 |
| 31 | lr_emb compensation | Will restore both | R²=0.92, cluster=0.95 | ✓ | 4 |
| 32 | L1=5E-4 with lr_emb | Combined works | R²=0.987 (yes) | ✓ | 4 |
| 33 | Noise transfer | Block 3-4 settings work | R²=0.999 (yes) | ✓ | 5 |
| 34 | lr_W=1E-2 works | Noise tolerates | R²=0.999 (yes) | ✓ | 5 |
| 36 | lr_W=5E-2 works | Wide tolerance | R²=0.999 (yes) | ✓ | 5 |
| 38 | lr_W=1E-1 works | Extreme | R²=0.999 (yes) | ✓ | 5 |
| 40 | L1=1E-3 works | L1 robust | R²=0.9996 (yes) | ✓ | 5 |
| 43 | lr_W=8E-3, lr=1E-3 | Both tuned | R²=0.969 (yes) | ✓ | 6 |
| 44 | lr_W=1E-2 works | Slightly higher | R²=0.959 (yes) | ✓ | 6 |
| 45 | lr_W=2E-2 test | Upper boundary | R²=0.912 (boundary) | ✗ | 6 |
| 46 | lr_emb probe | Will help | Partial improvement | ✓ | 6 |
| 50 | lr=1E-3 test | Will fix sparse | R²=0.004 (no) | ✗ | 7 |
| 51 | lr_W boost test | Will help | R²=0.003 (no) | ✗ | 7 |
| 52 | Factorization test | May help sparse | R²=0.002 (worse) | ✗ | 7 |
| 56 | Scale-up test | 5x data helps | R²=0.08 (no) | ✗ | 7 |
| 57 | Noise rescue test | Will restore eff_rank | eff_rank=92 (yes) | ✓ | 8 |
| 62 | Scale-up test | Will break plateau | R²=0.20 (no) | ✗ | 8 |
| 65 | eff_rank hypothesis | ff=0.5 has higher rank | eff_rank=26 (yes) | ✓ | 9 |
| 66 | lr_W=1E-2 test | Will help | R²=0.496 (ceiling) | ✗ | 9 |
| 67 | L1=0 test | Removes constraint | test_pearson=0.97 | ✓ | 9 |
| 71 | Factorization test | May help ff=0.5 | R²=0.374 (hurt) | ✗ | 9 |
| 73 | ff=0.75 test | R² ≈ 0.75 | R²=0.744 (yes) | ✓ | 10 |
| 76 | lr_W increase | Push above ceiling | R²=0.750 (slightly) | ✓ | 10 |
| 79 | Factorization test | May help ff=0.75 | R²=0.55 (hurt) | ✗ | 10 |
| 80 | Revert factorization | Restore R² | R²=0.741 (yes) | ✓ | 10 |
| 82 | n=200 baseline | Will converge | R²=0.999 (yes) | ✓ | 11 |
| 83 | lr_W=8E-3 | Same tolerance | R²=0.998 (yes) | ✓ | 11 |
| 85 | lr_W=3E-2 | Failure probe | R²=0.984 (still OK) | ✓ | 11 |
| 86 | lr_W=5E-2 | Continue probe | R²=0.987 (still OK) | ✓ | 11 |
| 87 | lr_W=1E-1 | Aggressive | R²=0.975 (still OK) | ✓ | 11 |
| 88 | lr_W=2E-1 | Extreme | R²=0.978 (still OK!) | ✓ | 11 |
| 89 | ff=0.9 baseline | Near threshold | R²=0.898 (partial) | ✓ | 12 |
| 90 | lr_W=8E-3 | Push above | R²=0.907 (yes) | ✓ | 12 |
| 94 | lr_W=5E-2 | Aggressive | R²=0.907 (same) | ✓ | 12 |
| 95 | lr_W=1E-1 | More aggressive | R²=0.907 (same) | ✓ | 12 |
| 96 | lr_W=5E-1 | Extreme | R²=0.898 (broke) | ✓ | 12 |
| 97 | n=300 baseline | Will converge | R²=0.943 (yes) | ✓ | 13 |
| 98 | lr_W=1E-2 | Higher | R²=0.949 (yes) | ✓ | 13 |
| 99 | lr_W=2E-2 | Even higher | R²=0.945 (yes) | ✓ | 13 |
| 100 | lr_W=5E-2 | Failure probe | R²=0.931 (still OK) | ✓ | 13 |
| 101 | lr_W=2E-1 | Aggressive | R²=0.076 (broke!) | ✓ | 13 |
| 103 | lr=2E-3 | Higher MLP lr | R²=0.940 (yes) | ✓ | 13 |
| 105 | n=500 baseline | May struggle | R²=0.705 (partial) | ✓ | 14 |
| 106 | lr_W=1E-2 | Will help | R²=0.728 (slightly) | ✓ | 14 |
| 107 | lr_W=2E-2 | More aggressive | R²=0.644 (hurt!) | ✗ | 14 |

**Validation rate**: 46/67 = 69%

---

## 4. Falsification: 41 instances

| Iter | Falsified Hypothesis | Evidence | Refinement | Block |
|------|---------------------|----------|------------|-------|
| 8 | L1 always beneficial | R²=0.762 at L1=1E-3 | Upper bound 5E-4 | 1 |
| 10 | Factorization helps low-rank | R²=0.355 (same) | Over-constrains | 2 |
| 12 | Higher lr_W always helps | R²=0.351 (boundary) | lr_W has upper limit | 2 |
| 14 | L1+aug together help | R²=0.35 (no change) | Need different approach | 2 |
| 17 | Dale adds difficulty | R²=0.989 (easy!) | Dale robust | 3 |
| 28 | Higher lr_W always better | cluster_acc=0.48 | Embedding starved | 4 |
| 30 | L1 always helps sparsity | cluster_acc dropped | L1 hurts embedding | 4 |
| 45 | lr_W=2E-2 works compound | R²=0.912 (boundary) | Narrower than expected | 6 |
| 47 | L1=1E-4 works | R²=0.853 (failed) | L1 tolerance narrow | 6 |
| 48 | lr=5E-4 sufficient | R²=0.899 (partial) | Needs lr=1E-3 | 6 |
| 50 | Training fixes sparse | R²=0.004 | Data is problem | 7 |
| 51 | lr_W boost fixes | R²=0.003 | Still fails | 7 |
| 52 | Factorization helps sparse | R²=0.002 | Made worse | 7 |
| 54 | L1=0 helps sparse | R²=0.02 (worse) | Overfits to noise | 7 |
| 55 | L1 restore helps | R²=0.01 | Still fails | 7 |
| 56 | Scale-up fixes sparse | R²=0.08 | Fundamental limit | 7 |
| 58 | lr_W breaks plateau | R²=0.20 (same) | Plateau is robust | 8 |
| 60 | L1 breaks plateau | R²=0.20 (same) | Plateau is fundamental | 8 |
| 62 | Scale-up breaks plateau | R²=0.20 (same) | Can't break ceiling | 8 |
| 66 | lr_W=1E-2 breaks ff=0.5 | R²=0.49 (same) | Ceiling confirmed | 9 |
| 69 | Scale-up breaks ff=0.5 | R²=0.50 (same) | Fundamental limit | 9 |
| 71 | Factorization helps ff=0.5 | R²=0.37 (hurt) | Over-constrains | 9 |
| 76 | lr_W pushes above ff | R²=0.750 (marginal) | Ceiling is hard | 10 |
| 79 | Factorization helps ff=0.75 | R²=0.55 (hurt) | Consistently bad | 10 |
| 91 | lr_W pushes above ff=0.9 | R²=0.907 (same) | Ceiling stable | 12 |
| 93 | L1=0 breaks ff=0.9 | R²=0.907 (same) | No effect | 12 |
| 96 | lr_W=5E-1 works ff=0.9 | R²=0.898 (broke) | Upper boundary | 12 |
| 101 | lr_W=2E-1 works n=300 | R²=0.076 (broke) | Upper boundary | 13 |
| 107 | Higher lr_W helps n=500 | R²=0.644 (hurt) | Non-monotonic | 14 |

---

## 5. Analogy/Transfer: 18 instances

| Iter | From | To | Knowledge | Outcome | Block |
|------|------|-----|-----------|---------|-------|
| 17 | Block 1 | Block 3 | lr_W=4E-3 baseline | ✓ R²=0.989 | 3 |
| 25 | Blocks 2-3 | Block 4 | Combined settings | ✓ R²=0.988 | 4 |
| 31 | Block 2 | Block 4 | lr_emb insight | ✓ Works | 4 |
| 33 | Blocks 3-4 | Block 5 | Baseline settings | ✓ R²=0.999 | 5 |
| 41 | Blocks 2+4 | Block 6 | Compound difficulty | ✓ Predicted | 6 |
| 42 | Block 2 | Block 6 | lr=1E-3 insight | ✓ Critical | 6 |
| 43 | Block 4 | Block 6 | lr_emb insight | ✓ Helps | 6 |
| 50 | Block 2 | Block 7 | lr insight | ✗ Didn't help | 7 |
| 57 | Block 5 | Block 8 | Noise rescues | ✓ eff_rank=92 | 8 |
| 57 | Block 2 | Block 8 | lr insight | ✓ Applied | 8 |
| 65 | Block 8 | Block 9 | ff=0.5 hypothesis | ✓ Confirmed | 9 |
| 73 | Block 9 | Block 10 | Linear ff-R² law | ✓ R²=0.744 | 10 |
| 82 | Blocks 1-3 | Block 11 | Scale settings | ✓ Converges | 11 |
| 89 | Blocks 9-10 | Block 12 | Ceiling law | ✓ R²=0.907 | 12 |
| 97 | Blocks 1,11 | Block 13 | Scale settings | ✓ Converges | 13 |
| 103 | Block 13 | Block 14 | Higher lr helps | ✗ Partial only | 14 |
| 105 | Block 13 | Block 14 | Baseline transfer | ✗ Partial | 14 |
| 106 | Block 11 | Block 14 | lr_W increase | ✓ Slight help | 14 |

**Success rate**: 14/18 = 77%

---

## 6. Boundary Probing: 38 instances

| Iter | Parameter | Test Value | Result | Boundary Status | Block |
|------|-----------|------------|--------|-----------------|-------|
| 4 | lr_W | 1E-2 | R²=0.922 | Approaching upper | 1 |
| 5 | L1 | 5E-5 | R²=0.999 | Testing lower | 1 |
| 6 | L1 | 1E-4 | R²=0.999 | Tolerance confirmed | 1 |
| 7 | L1 | 1E-3 | R²=0.762 | Upper found | 1 |
| 12 | lr_W | 8E-3 | R²=0.351 | Upper for low_rank | 2 |
| 19 | lr_W | 2E-2 | R²=0.999 | Still OK | 3 |
| 20 | lr_W | 5E-2 | R²=0.999 | Still OK | 3 |
| 21 | L1 | 1E-3 | R²=0.999 | Still OK | 3 |
| 22 | lr_W | 1E-1 | R²=0.999 | Still OK | 3 |
| 23 | lr_W | 2E-1 | R²=0.999 | Still OK | 3 |
| 24 | lr_W | 5E-1 | R²=0.9998 | 100x range confirmed | 3 |
| 27 | lr_W | 1E-2 | R²=0.992 | Testing | 4 |
| 28 | lr_W | 5E-2 | cluster=0.48 | Embedding boundary | 4 |
| 29 | lr_W | 2E-2 | R²=0.93 | Narrowed | 4 |
| 36 | lr_W | 5E-2 | R²=0.999 | Still OK | 5 |
| 38 | lr_W | 1E-1 | R²=0.999 | Still OK | 5 |
| 39 | L1 | 5E-4 | R²=0.999 | Testing | 5 |
| 40 | L1 | 1E-3 | R²=0.9996 | No boundary found | 5 |
| 44 | lr_W | 1E-2 | R²=0.959 | Testing | 6 |
| 45 | lr_W | 2E-2 | R²=0.912 | Upper found | 6 |
| 47 | L1 | 1E-4 | R²=0.853 | Upper found | 6 |
| 59 | lr_W | varies | R²=0.20 | Plateau | 8 |
| 60 | L1 | varies | R²=0.20 | Plateau | 8 |
| 70 | lr_W | 5E-2 | R²=0.47 | Ceiling | 9 |
| 78 | lr_W | 5E-2 | R²=0.74 | Ceiling | 10 |
| 85 | lr_W | 3E-2 | R²=0.984 | Still OK | 11 |
| 86 | lr_W | 5E-2 | R²=0.987 | Still OK | 11 |
| 87 | lr_W | 1E-1 | R²=0.975 | Still OK | 11 |
| 88 | lr_W | 2E-1 | R²=0.978 | 40x range! | 11 |
| 91 | lr_W | 1.2E-2 | R²=0.907 | Ceiling | 12 |
| 92 | lr_W | 1.5E-2 | R²=0.907 | Ceiling | 12 |
| 94 | lr_W | 5E-2 | R²=0.907 | Ceiling | 12 |
| 95 | lr_W | 1E-1 | R²=0.907 | Ceiling | 12 |
| 96 | lr_W | 5E-1 | R²=0.898 | Upper found | 12 |
| 99 | lr_W | 2E-2 | R²=0.945 | Testing | 13 |
| 100 | lr_W | 5E-2 | R²=0.931 | Still OK | 13 |
| 101 | lr_W | 2E-1 | R²=0.076 | Upper found | 13 |
| 106 | lr_W | 1E-2 | R²=0.728 | Testing | 14 |

---

## 7. Emerging Patterns: 45 instances

| Iter | Pattern Type | Description | Significance | Block |
|------|--------------|-------------|--------------|-------|
| 5 | Meta-reasoning | Recognized lr_W exhausted, switch to L1 | Medium | 1 |
| 9 | Regime | Low_rank regime discovered (eff_rank=11) | High | 2 |
| 14 | Meta-reasoning | Multi-param violation, reset strategy | Medium | 2 |
| 15 | Causal Chain | lr mechanism: low lr → undertrained MLP | High | 2 |
| 17 | Regime | Dale regime identified | Medium | 3 |
| 28 | Causal Chain | High lr_W → starved embedding → poor clustering | High | 4 |
| 31 | Causal Chain | lr_emb compensation mechanism | Medium | 4 |
| 33 | Regime | Noise regime (eff_rank=83) | High | 5 |
| 33 | Predictive | eff_rank increase prediction | High | 5 |
| 33 | Causal Chain | Noise → variance → higher eff_rank | High | 5 |
| 40 | Regime | "Super easy mode" identified | High | 5 |
| 41 | Regime | Compound difficulty regime | High | 6 |
| 41 | Predictive | eff_rank prediction for compound | Medium | 6 |
| 12 | Constraint | low_rank lr_W constraint | High | 2 |
| 28 | Constraint | n_types lr_W constraint | High | 4 |
| 45 | Constraint | Compound lr_W constraint | High | 6 |
| 47 | Constraint | L1 constraint for compound | High | 6 |
| 48 | Constraint | lr requirement for compound | High | 6 |
| 49 | Regime | Sparse collapse regime (eff_rank=6) | High | 7 |
| 49 | Causal Chain | Sparse → collapse → untrainable | High | 7 |
| 49 | Predictive | eff_rank<8 prediction | High | 7 |
| 49 | Uncertainty | eff_rank variance in sparse | Medium | 7 |
| 51 | Meta-reasoning | Recognized training futility | High | 7 |
| 53 | Meta-reasoning | Strategy exhaustion, pivot to noise | High | 7 |
| 54 | Uncertainty | eff_rank dropped noted | Medium | 7 |
| 57 | Regime | Sparse+noise regime | High | 8 |
| 57 | Causal Chain | Noise rescues eff_rank mechanism | High | 8 |
| 57 | Predictive | eff_rank=92 prediction | High | 8 |
| 59 | Meta-reasoning | Dimension switch after plateau | Medium | 8 |
| 60 | Uncertainty | Plateau may be fundamental | High | 8 |
| 65 | Regime | ff=0.5 regime discovered | High | 9 |
| 67 | Predictive | Linear ff-R² relationship | High | 9 |
| 72 | Constraint | ff=0.5 ceiling constraint | High | 9 |
| 80 | Predictive | ff=0.75 → R²=0.75 confirmed | High | 10 |
| 88 | Regime | n=200 "extremely robust" | High | 11 |
| 96 | Constraint | ff=0.9 lr_W upper constraint | High | 12 |
| 101 | Constraint | n=300 lr_W boundary | High | 13 |
| 104 | Predictive | lr_W tolerance narrows with n | High | 13 |
| 107 | Uncertainty | n=500 non-monotonic behavior | High | 14 |

---

## Iteration Index

| Iter | Modes Active | Key Event |
|------|--------------|-----------|
| 1-3 | — | Baseline convergence |
| 4 | Induction, Deduction, Boundary | First boundary probe |
| 5 | Boundary, Meta-reasoning | Dimension switch |
| 6-7 | Boundary, Deduction | L1 exploration |
| 8 | Induction, Deduction, Falsification | Block 1 summary |
| 9 | Abduction, Regime | Low-rank discovered |
| 10 | Deduction, Falsification, Abduction | Factorization falsified |
| 11-14 | Deduction, Induction, Meta-reasoning | lr_W exploration |
| 15-16 | Deduction, Causal Chain, Abduction, Induction | lr breakthrough |
| 17 | Analogy, Deduction, Abduction, Regime, Falsification | Block 3 start |
| 18-24 | Deduction, Boundary, Induction | Dale exploration |
| 25-32 | Analogy, Deduction, Falsification, Boundary, Causal, Induction | Block 4 |
| 33-40 | Analogy, Deduction, Abduction, Regime, Causal, Predictive, Induction | Block 5 |
| 41-48 | Analogy, Abduction, Regime, Predictive, Deduction, Falsification, Constraint, Induction | Block 6 |
| 49-56 | Abduction, Regime, Causal, Predictive, Uncertainty, Deduction, Falsification, Meta, Induction | Block 7 |
| 57-64 | Analogy, Deduction, Causal, Regime, Predictive, Induction, Falsification, Meta, Uncertainty | Block 8 |
| 65-72 | Analogy, Regime, Deduction, Induction, Falsification, Predictive, Constraint | Block 9 |
| 73-80 | Deduction, Induction, Abduction, Falsification, Predictive | Block 10 |
| 81-88 | Induction, Deduction, Boundary, Regime | Block 11 |
| 89-96 | Induction, Deduction, Boundary, Falsification, Abduction, Constraint | Block 12 |
| 97-104 | Induction, Deduction, Boundary, Falsification, Constraint, Predictive | Block 13 |
| 105-107 | Induction, Abduction, Deduction, Uncertainty | Block 14 |
