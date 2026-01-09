# Epistemic Analysis Instructions

Framework from "Understanding: an experiment-LLM-memory experiment" (Allier & Saalfeld, 2026).

---

#### Background

The experiment-LLM-memory triad: **experiments** provide validation, **LLM** generates hypotheses, **memory** stores cumulative knowledge. Goal: quantify how the system acquires, tests, revises, and transfers knowledge.

---

#### Reasoning Modes

**1. Induction** (Observations → Pattern)
- Multiple observations → generalized rule
- Markers: "scales with", "optimal for", "consistently"
- *Exclude* patterns given as priors

**2. Abduction** (Observation → Hypothesis)
- Unexpected result → causal explanation
- Markers: "likely because", "suggests", "caused by"

**3. Deduction** (Hypothesis → Prediction)
- Hypothesis → testable prediction
- Markers: "if...then", "should", "expect"
- Track: validation rate = confirmed / total

**4. Falsification** (Prediction Failed → Refine)
- Prediction contradicted → hypothesis rejected/refined
- Markers: "rejected", "falsified", "does NOT"

**5. Analogy/Transfer** (Cross-Regime)
- Prior finding applied to new context
- Markers: "generalizes", "transfers", "based on Block N"

**6. Boundary Probing** (Limit-Finding)
- Sequential parameter changes → thresholds
- Markers: "boundary", "minimum", "limit"

---

#### Excluding Priors

**Exclude**: Parameter ranges, architecture properties, classification thresholds, training dynamics from protocol.

**Include**: Specific values discovered, relationships found, boundaries probed, cross-block generalizations.

---

#### Confidence Scoring

`confidence = min(100%, 30% + 5%×log2(n_confirmations+1) + 10%×log2(n_alt_rejected+1) + 15%×n_blocks)`

| Component | Weight | Basis |
|-----------|--------|-------|
| Base | 30% | Single observation (weak) |
| n_confirmations | +5%×log2(n+1) | Diminishing returns (10 tests → +17%) |
| n_alt_rejected | +10%×log2(n+1) | Popper's asymmetry (10 rejected → +35%) |
| n_blocks | +15% each | Cross-context strongest evidence |

*Note*: Logarithmic scaling prevents inflation at high iteration counts (2048+ iterations).

| Level | Score | Criteria |
|-------|-------|----------|
| Very High | 90-100% | ≥20 tests + ≥5 alt rejected + ≥3 blocks |
| High | 75-89% | ≥10 tests across ≥2 blocks OR ≥10 alt rejected |
| Medium | 60-74% | ≥5 tests OR 2 blocks |
| Low | <60% | <5 tests OR single block OR contradictory |

**Adjustments**: Cap 85% if variance observed. Reduce 15% if single regime. Note "needs testing" if <10 tests.

---

#### Evidence Strength (Popper, Lakatos)

| Type | Weight | Description |
|------|--------|-------------|
| Falsification | Highest | Alternative rejected |
| Boundary probing | High | Systematic limits |
| Cross-block | High | Generalization |
| Single confirmation | Medium | One test |
| Indirect inference | Low | Derived |

---

#### Procedure

1. Catalog priors from protocol
2. Parse logs chronologically, tag reasoning modes
3. Filter prior-derived conclusions
4. Calculate metrics (counts, validation rates)
5. Assess what was learned vs given

---

#### Output Format

**Header**

```markdown
# Epistemic Analysis: {experiment_name}

**Experiment**: {description} | **Iterations**: N (M blocks × K) | **Date**: YYYY-MM-DD
```

**Priors Excluded Table**

| Prior Category | Specific Priors Given |
|----------------|----------------------|
| Parameter ranges | lr: X to Y, ... |
| Architecture | Model descriptions from protocol |
| Classification | R² thresholds, success criteria |
| Training dynamics | Known relationships from protocol |

**Reasoning Modes Table**

| Mode | Count | Validation | First Appearance |
|------|-------|------------|------------------|
| Induction | N | N/A | Iter X (single), Y (cumulative) |
| Abduction | N | N/A | Iter X |
| Deduction | N | X% (Y/N) | Iter X |
| Falsification | N | 100% refinement | Iter X |
| Analogy/Transfer | N | X% (Y/N) | Iter X |
| Boundary Probing | N | N/A | Iter X |

**Detailed Mode Tables**

For each reasoning mode, provide a table with iteration-level detail:

```markdown
#### 1. Induction (N instances)

| Iter | Observation | Induced Pattern | Type |
|------|-------------|-----------------|------|
| X | What was observed | Pattern extracted | Single/Cumulative (N obs)/Cross-block |

#### 2. Abduction (N instances)

| Iter | Observation | Hypothesis |
|------|-------------|------------|
| X | Unexpected result | Causal explanation |

#### 3. Deduction (N instances) — X% validated

| Iter | Hypothesis | Prediction | Outcome | ✓/✗ |
|------|-----------|------------|---------|-----|
| X | If A then B | Expected result | Actual result | ✓/✗/~ |

#### 4. Falsification (N instances)

| Iter | Falsified Hypothesis | Result |
|------|---------------------|--------|
| X | What was rejected | **Rejected**: evidence |

#### 5. Analogy/Transfer (N instances) — X% success

| From | To | Knowledge | Outcome |
|------|-----|-----------|---------|
| Block/Regime | Block/Regime | What transferred | ✓/✗/Partial |

#### 6. Boundary Probing (N instances)

| Parameter | Range | Boundary Found | Iter |
|-----------|-------|----------------|------|
| param_name | X→Y | Threshold value | X-Y |
```

**Timeline Table**

| Iter | Milestone | Mode |
|------|-----------|------|
| X | First significant event | Mode type |

**Principles Table** (by confidence)

| # | Principle | Prior | Discovery | Evidence | Conf |
|---|-----------|-------|-----------|----------|------|
| 1 | Name | "text"/None | Description | N tests, M alt, B blocks | X% |

**Confidence Calculation**

| # | n_tests | n_alt | n_blocks | Score |
|---|---------|-------|----------|-------|
| 1 | N | M | B | 30+X+Y+Z=**N%** |

**Summary Paragraph**

Brief synthesis: reasoning progression, validation rates, key discoveries, major falsifications.

**Metrics Table**

| Metric | Value |
|--------|-------|
| Iterations | N |
| Blocks | M |
| Reasoning instances | N |
| Deduction validation | X% |
| Transfer success | X% |
| Principles discovered | N |

---

#### Discussion Caveat

Do NOT claim "emergent reasoning" or "transcends components" without ablation studies. Claims about component contributions require LLM-only / memory-ablated comparisons. Describe observations only.

---

#### Timeline Thresholds

| Capability | Typical |
|------------|---------|
| Single-shot | ~5 iter |
| Cumulative induction | ~12 iter |
| Falsification→principle | ~23 iter |
| Cross-domain transfer | ~25 iter |

---

**Reference**: Allier & Saalfeld (2026). Understanding: an experiment-LLM-memory experiment. Janelia/HHMI.
