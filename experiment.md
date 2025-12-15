# Experiment Analysis Protocol

This document defines the criteria for analyzing neural activity patterns and guiding configuration changes.

## 1. Activity Pattern Classification

After running `python GNN_Main.py -o generate <config_name>`, examine `graphs_data/signal/<dataset_name>/activity.png`.

### 1.1 Steady State (Undesirable)
**Visual indicators:**
- All neuron traces appear as flat horizontal lines
- No variation in activity over time
- Activity stabilizes very quickly (within first few hundred frames)

**Diagnosis:** The network dynamics have collapsed to a fixed point. This typically happens when:
- `gain` (g parameter) is too low - insufficient amplification
- `Dale_law_factor` creates too much inhibition (too low, e.g., < 0.5)
- Initial connectivity is too weak

### 1.2 Chaotic Regime (Desirable)
**Visual indicators:**
- Neuron traces show irregular, aperiodic fluctuations
- Activity varies continuously over time without repeating patterns
- Different neurons show different activity patterns (heterogeneity)
- No obvious periodicity or convergence to fixed point

**Assessment metrics:**
- **Amplitude range:** Estimate the typical peak-to-peak amplitude of oscillations (e.g., "amplitude ~20 units")
- **Temporal structure:** Are fluctuations slow (spanning thousands of frames) or fast (hundreds of frames)?
- **Neuron diversity:** Do different neurons show different behaviors or are they all similar?

### 1.3 Oscillatory/Periodic (Intermediate)
**Visual indicators:**
- Regular, repeating patterns in activity
- Visible periodicity across neurons
- May show external input influence (yellow dashed line at top if present)

**Assessment:** Note the apparent period and whether oscillations are synchronized across neurons.

## 2. Analysis Template for analysis.md

Each experiment entry should follow this format:

```markdown
## Experiment: <config_filename>
**Date:** YYYY-MM-DD
**Dataset:** <dataset_name>

### Configuration Summary
- gain (g): <value>
- Dale_law_factor: <value> (if applicable)
- n_neurons: <value>
- connectivity_init: <range>
- Other relevant params: ...

### Activity Analysis
**Classification:** [Steady State | Chaotic | Oscillatory]

**Observations:**
- <Describe what you see in activity.png>
- <Note amplitude, temporal structure, neuron diversity>
- <Compare to previous experiments if applicable>

### Diagnosis
<Why does the network behave this way given the parameters?>

### Next Steps
<Specific parameter changes to try and rationale>
```

## 3. Parameter Adjustment Guidelines

### 3.1 From Steady State to Chaotic

| Problem | Solution | Rationale |
|---------|----------|-----------|
| Flat activity | Increase `gain` (g) | Higher gain amplifies network activity |
| Too much inhibition | Increase `Dale_law_factor` towards 0.6-0.7 | More excitatory neurons drive activity |
| Weak connectivity | Increase upper bound of `connectivity_init` | Stronger connections propagate activity |

**Typical adjustments:**
- If steady state with g=10: try g=12, g=15
- If steady state with Dale_law_factor=0.55: try 0.60, 0.65

### 3.2 From Chaotic to More Structured

| Problem | Solution | Rationale |
|---------|----------|-----------|
| Too chaotic/noisy | Decrease `gain` slightly | Reduce amplification |
| Activity exploding | Decrease `connectivity_init` range | Weaker connections stabilize |
| Need periodicity | Add external oscillatory input | Drive rhythmic behavior |

### 3.3 Dale's Law Experiments

The `Dale_law_factor` controls the fraction of excitatory neurons:
- **0.5:** Equal excitation/inhibition (balanced)
- **< 0.5:** More inhibitory neurons (tends toward steady state)
- **> 0.5:** More excitatory neurons (tends toward chaos/activity)

Typical biological range: 0.6-0.8 (more excitatory than inhibitory)

### 3.4 Gain Parameter Guide

The `gain` (g) in params controls the steepness of the activation function:
- **g < 5:** Weak nonlinearity, likely steady state
- **g = 7-10:** Moderate, good starting point for chaotic dynamics
- **g > 15:** Strong nonlinearity, may cause instability

## 4. Experiment Workflow

1. **Generate:** `python GNN_Main.py -o generate <config_name>`
2. **Analyze:** Examine `activity.png`, classify dynamics
3. **Document:** Update `analysis.md` with findings
4. **Iterate:** Modify config based on guidelines above
5. **Repeat:** Until desired dynamics achieved

## 5. Success Criteria

A good chaotic signal dataset should have:
- Sustained activity throughout simulation (no collapse to steady state)
- Irregular, non-periodic fluctuations
- Heterogeneous activity across neurons
- Activity bounded within reasonable range (not exploding)
- Rich temporal dynamics suitable for training GNN models
