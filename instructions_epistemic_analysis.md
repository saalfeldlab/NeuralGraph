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

#### Emerging Reasoning Patterns

Document novel reasoning behaviors not captured by the six standard modes. Look for:

**7. Meta-reasoning** (Reasoning about reasoning)
- Self-correction of strategy mid-block
- Recognizing when a search strategy is ineffective
- Markers: "strategy isn't working", "need different approach", "stuck"

**8. Uncertainty Quantification**
- Explicit acknowledgment of confidence levels
- Distinguishing robust vs stochastic findings
- Markers: "not reproducible", "high variance", "need more tests"

**9. Causal Chain Construction**
- Multi-step causal explanations linking observations
- Building mechanistic models beyond single hypotheses
- Markers: "because X, which causes Y, leading to Z"

**10. Constraint Propagation**
- Inferring parameter relationships from failures
- Deducing what must be true given what failed
- Markers: "since X failed, Y must be", "implies", "constrains"

**11. Regime Recognition**
- Identifying qualitatively different operating modes
- Recognizing phase transitions in parameter space
- Markers: "different regime", "phase transition", "fundamentally different"

**12. Predictive Modeling**
- Building quantitative relationships (not just qualitative)
- Predicting specific values, not just directions
- Markers: "expect R²≈X", "should need ~Y iterations", "scales as"

**Format for emerging patterns:**

```markdown
#### 7. Emerging Reasoning Patterns

| Iter | Pattern Type | Description | Significance |
|------|--------------|-------------|--------------|
| X | Meta-reasoning | Recognized lr_W search exhausted, switched to lr | Strategy adaptation |
| Y | Regime Recognition | Identified eff_rank=6 as qualitatively different | Phase boundary |
| Z | Uncertainty Quantification | Noted R²=0.886 not reproducible | Stochasticity awareness |
```

**Significance ratings:**
- **High**: Led to breakthrough or prevented wasted iterations
- **Medium**: Improved search efficiency
- **Low**: Interesting but no clear impact

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

Generate **three files**:

1. **`{experiment}_epistemic_analysis.md`** — Main summary with counts, key examples, principles
2. **`{experiment}_epistemic_detailed.md`** — Exhaustive list of every reasoning instance
3. **`{experiment}_epistemic_edges.md`** — Causal relationships between reasoning events

---

### File 1: Main Analysis (`_epistemic_analysis.md`)

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

**Key Examples Table** (3-5 representative instances per mode)

Show only the most significant examples in the main file. Full details go in the detailed file.

```markdown
### Key Examples

#### Induction (N instances total — see detailed file)
| Iter | Pattern | Significance |
|------|---------|--------------|
| X | Key pattern discovered | High/Medium |

#### Deduction (N instances, X% validated — see detailed file)
| Iter | Prediction | Outcome | ✓/✗ |
|------|------------|---------|-----|
| X | Key prediction | Result | ✓/✗ |

#### Falsification (N instances — see detailed file)
| Iter | Hypothesis Rejected | Impact |
|------|---------------------|--------|
| X | What was rejected | Led to principle N |
```

**Timeline Table**

| Iter | Milestone | Mode |
|------|-----------|------|
| X | First significant event | Mode type |

**Principles Table** (by confidence)

| # | Principle | Prior | Origin | Evidence | Conf |
|---|-----------|-------|-----------|----------|------|
| 1 | Name | "text"/None | Description | N tests, M alt, B blocks | X% |

**Confidence Calculation**

| # | n_tests | n_alt | n_blocks | Score |
|---|---------|-------|----------|-------|
| 1 | N | M | B | 30+X+Y+Z=**N%** |

**Summary Paragraph**

Brief synthesis: reasoning progression, validation rates, key findings, major falsifications.

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

### File 2: Detailed Log (`_epistemic_detailed.md`)

Exhaustive list of every reasoning instance for reproducibility and visualization.

**Header**

```markdown
# Epistemic Analysis Detailed Log: {experiment_name}

**Companion to**: {experiment}_epistemic_analysis.md
**Total instances**: N reasoning events across M iterations
```

**Exhaustive Mode Tables**

For each reasoning mode, list ALL instances:

```markdown
## 1. Induction: N instances

| Iter | Observation | Induced Pattern | Type | Block |
|------|-------------|-----------------|------|-------|
| 6 | lr_W 2E-3 to 4E-2 all converge | 10x robust range | Cumulative (5 obs) | 1 |
| 9 | 8 consecutive converged | Regime robustness | Cumulative (8 obs) | 1 |
...

## 2. Abduction: N instances

| Iter | Observation | Hypothesis | Block |
|------|-------------|------------|-------|
| 17 | Dale_law reduces R² | eff_rank reduction | 2 |
...

## 3. Deduction: N instances

| Iter | Hypothesis | Prediction | Outcome | ✓/✗ | Block |
|------|-----------|------------|---------|-----|-------|
| 4 | lr_W approaching boundary | R² will degrade | R²=0.922 | ✓ | 1 |
...

## 4. Falsification: N instances

| Iter | Falsified Hypothesis | Evidence | Refinement | Block |
|------|---------------------|----------|------------|-------|
| 10 | L1 always beneficial | R²=0.762 at L1=1E-3 | Upper bound 5E-4 | 1 |
...

## 5. Analogy/Transfer: N instances

| Iter | From | To | Knowledge | Outcome | Block |
|------|------|-----|-----------|---------|-------|
| 17 | Block 1 | Block 2 | lr_W=4E-3 baseline | ✗ Failed | 2 |
...

## 6. Boundary Probing: N instances

| Iter | Parameter | Test Value | Result | Boundary Status | Block |
|------|-----------|------------|--------|-----------------|-------|
| 4 | lr_W | 1E-2 | R²=0.922 | Approaching upper | 1 |
...

## 7. Emerging Patterns: N instances

| Iter | Pattern Type | Description | Significance | Block |
|------|--------------|-------------|--------------|-------|
| 6 | Meta-reasoning | Switch-dimension triggered | Medium | 1 |
...
```

**Cross-Reference Index**

```markdown
## Iteration Index

| Iter | Modes Active | Key Event |
|------|--------------|-----------|
| 1 | — | Baseline |
| 2 | Deduction | First convergence |
| 4 | Deduction, Boundary | First boundary probe |
...
```

This detailed file enables:
- Accurate counts for the main summary
- Data source for timeline visualizations
- Reproducible analysis

---

### File 3: Causal Edges (`_epistemic_edges.md`)

Document causal relationships between reasoning events for visualization.

**Header**

```markdown
# Epistemic Analysis Edges: {experiment_name}

**Companion to**: {experiment}_epistemic_analysis.md, {experiment}_epistemic_detailed.md
**Total edges**: N causal relationships
```

**Edge Types:**

| Type | Style | Meaning | Example |
|------|-------|---------|---------|
| `leads_to` | Solid gray | Natural progression | Deduction → Induction |
| `triggers` | Dashed blue | One event causes another | Abduction → Deduction |
| `refines` | Dotted green | Updates/corrects earlier | Falsification → Induction |
| `rejects` | Solid red, vertical | Falsification rejects hypothesis | Falsification → Abduction (backward) |

**Note on Falsification Edges:**
Falsification represents *negative feedback* that rejects a prior hypothesis. Unlike other edges that flow forward in time (cause → effect), Falsification edges should be drawn **vertically or pointing backward** to the hypothesis they reject. This visually distinguishes:
- Forward arrows: constructive reasoning (hypothesis → test → pattern)
- Backward/vertical arrows: destructive reasoning (test → rejection of prior hypothesis)

---

## CRITICAL: Edge Validity Rules

### Rule 1: Different Iterations Required
**Edges must ALWAYS connect different iterations (from_iter < to_iter).**

Rationale:
- A hypothesis (Abduction) cannot be tested in the same iteration it was formed
- Deduction (prediction) requires a subsequent experiment to validate/falsify
- Falsification requires observing experimental results, which takes at least one iteration

**INVALID examples:**
```
(17, 'Abduction', 17, 'Regime')      # ✗ Same iteration
(10, 'Deduction', 10, 'Falsification') # ✗ Same iteration
(6, 'Induction', 6, 'Meta-reasoning')  # ✗ Same iteration
```

**VALID examples:**
```
(17, 'Abduction', 21, 'Deduction')   # ✓ Hypothesis tested 4 iters later
(4, 'Deduction', 10, 'Falsification') # ✓ Prediction falsified 6 iters later
(6, 'Induction', 16, 'Induction')    # ✓ Pattern leads to block summary
```

### Rule 2: Source Node Must Exist
**The source (from_iter, from_mode) must have a corresponding event in the events list.**

Before creating edge `(X, 'Mode1', Y, 'Mode2')`, verify that `(X, 'Mode1', ...)` exists in events.

### Rule 3: Target Node Must Exist
**The target (to_iter, to_mode) must have a corresponding event in the events list.**

Before creating edge `(X, 'Mode1', Y, 'Mode2')`, verify that `(Y, 'Mode2', ...)` exists in events.

### Rule 4: Temporal Causality
**Edges must flow forward in time (from_iter < to_iter).**

Reasoning events cause future events, not past ones.

### Rule 5: Logical Causality
**The edge must represent a plausible causal relationship.**

Valid causal patterns:
| From Mode | To Mode | Rationale |
|-----------|---------|-----------|
| Abduction | Deduction | Hypothesis generates testable prediction |
| Deduction | Falsification | Failed prediction rejects hypothesis |
| Deduction | Induction | Validated predictions form pattern |
| Falsification | Induction | Rejection refines understanding |
| Falsification | Abduction | Failure prompts new hypothesis |
| Falsification | Boundary | Failure reveals parameter limit |
| Falsification | Causal | Understanding why something failed |
| Induction | Analogy/Transfer | Pattern applied to new regime |
| Induction | Induction | Cumulative patterns → block summary |
| Abduction | Uncertainty | Hypothesis reveals stochasticity |
| Boundary | Deduction | Boundary finding enables prediction |
| Boundary | Induction | Boundaries form pattern |
| Causal | Induction | Mechanistic model becomes principle |
| Predictive | Constraint | Quantitative model implies constraints |
| Regime | Deduction | New regime triggers new tests |

---

## Systematic Edge Identification

For each reasoning event at iteration X, ask:

1. **What triggered this?**
   - Look backward to find a PREVIOUS iteration (< X) that caused this event
   - The cause must exist as an event in the events list

2. **What does this enable?**
   - Look forward to find a FUTURE iteration (> X) that this event causes
   - The effect must exist as an event in the events list

3. **Verify both endpoints exist**
   - Check events list for source node
   - Check events list for target node

---

## Edge Format

```markdown
## Within-Block Edges

### Block 1 (Chaotic baseline)

| From Iter | From Mode | To Iter | To Mode | Type | Description |
|-----------|-----------|---------|---------|------|-------------|
| 4 | Deduction | 6 | Induction | leads_to | Validated prediction → pattern |
| 4 | Deduction | 10 | Falsification | leads_to | Prediction tested → boundary found |
| 6 | Induction | 16 | Induction | leads_to | Cumulative patterns → block summary |

### Block 2 (Dale_law)
...

## Cross-Block Edges

| From Iter | From Mode | To Iter | To Mode | Type | Description |
|-----------|-----------|---------|---------|------|-------------|
| 16 | Induction | 17 | Analogy/Transfer | triggers | Block 1 principles → Block 2 transfer |
| 22 | Induction | 33 | Analogy/Transfer | triggers | eff_rank ceiling → Block 3 transfer |
...
```

---

## Edge Count Guidelines

For a typical 16-iteration block, expect:
- 3-5 within-block edges (hypothesis → test → result chains)
- 1-2 cross-block edges (principle transfer)

For 8 blocks (~128 iterations), expect **25-40 edges total**.

Quality over quantity: only include edges where both endpoints exist and causality is clear.

---

### Timeline Visualization (`_epistemic_timeline.png`)

Generate a timeline plot with the following specifications:

**Layout:**
- X-axis: Iteration number (0 to max_iteration + 5)
- Y-axis: Reasoning modes (bottom to top):
  - Evidence gathering: Induction, Boundary
  - Hypothesis testing: Abduction, Deduction, Falsification
  - Meta-cognition: Analogy/Transfer, Meta-reasoning, Regime, Uncertainty
  - Advanced patterns: Causal, Predictive, Constraint

**Visual Elements:**
- **Nodes**: Scatter points colored by mode, sized by significance (High=150, Medium=80, Low=40)
- **Edges**: Arrows connecting nodes across iterations
  - `leads_to`: Solid gray line
  - `triggers`: Dashed blue line
  - `refines`: Dotted green line
- **Block backgrounds**: Alternating pastel colors for each block

**Styling:**
- No title
- No block labels
- Large axis labels (fontsize=24)
- Large tick labels (fontsize=16)
- Legend for modes, edge types, and node sizes
- Grid on x-axis only

**Color scheme:**
```python
COLORS = {
    'Induction': '#2ecc71',        # Green
    'Abduction': '#9b59b6',        # Purple
    'Deduction': '#3498db',        # Blue
    'Falsification': '#e74c3c',    # Red
    'Analogy/Transfer': '#f39c12', # Orange
    'Boundary': '#1abc9c',         # Teal
    'Meta-reasoning': '#e91e63',   # Pink
    'Regime': '#795548',           # Brown
    'Uncertainty': '#607d8b',      # Gray
    'Causal': '#00bcd4',           # Cyan
    'Predictive': '#8bc34a',       # Light green
    'Constraint': '#ff5722',       # Deep orange
}
```

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
