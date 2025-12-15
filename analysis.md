# Experiment Analysis Log

## Experiment: signal_chaotic_Dale_1.yaml
**Date:** 2025-12-15
**Dataset:** signal_chaotic_Dale_1

### Configuration Summary
- gain (g): 10.0
- Dale_law_factor: 0.55
- n_neurons: 100
- n_neuron_types: 1
- connectivity_type: chaotic
- connectivity_init: [0, 0.1]
- phi: tanh
- params: [a=1.0, b=0.0, g=10.0, s=0.0, w=1.0, h=0.0]

### Activity Analysis
**Classification:** Steady State

**Observations:**
- All 100 neuron traces appear as flat horizontal lines
- No visible oscillations or fluctuations over the 10,000 frame simulation
- Activity converged to fixed values very quickly
- Each neuron settled at a different steady-state value (visible as different vertical positions)

### Diagnosis
The combination of:
1. **Dale_law_factor = 0.55** creates near-balanced excitation/inhibition (55% excitatory), which may be slightly too inhibitory to sustain chaotic activity
2. **gain = 10.0** should be sufficient for nonlinear dynamics, but combined with the balanced E/I ratio, the network stabilizes

The chaotic connectivity type alone is not sufficient to generate chaotic dynamics - the intrinsic parameters must also support sustained activity.

### Next Steps
Try increasing excitation to push the network away from steady state:

**Option A:** Increase Dale_law_factor
- Change `Dale_law_factor: 0.55` → `Dale_law_factor: 0.65`
- Rationale: More excitatory neurons (65%) should drive more activity

**Option B:** Increase gain
- Change gain in params from 10.0 → 12.0 or 15.0
- Rationale: Stronger nonlinearity may prevent collapse to fixed point

**Recommended:** Try Option A first (Dale_law_factor: 0.65), as this is the parameter under investigation.

---

## Experiment: signal_chaotic_1.yaml
**Date:** 2025-12-15
**Dataset:** signal_chaotic_1

### Configuration Summary
- gain (g): 7.0
- Dale_law_factor: not set (no Dale's law applied)
- n_neurons: 100
- n_neuron_types: 1
- connectivity_type: chaotic
- connectivity_init: [0, 0.1]
- phi: tanh
- params: [a=1.0, b=0.0, g=7.0, s=0.0, w=1.0, h=0.0]

### Activity Analysis
**Classification:** Steady State

**Observations:**
- All 100 neuron traces appear as flat horizontal lines throughout the 10,000 frame simulation
- No visible oscillations, fluctuations, or temporal dynamics
- Activity converged to fixed values very quickly (within the first few frames)
- Each neuron settled at a different steady-state value (visible as different vertical positions, ranging roughly 0-40)
- The y-axis scale shows values from 0 to ~40, indicating some heterogeneity in fixed point values

### Diagnosis
The network has collapsed to a fixed point attractor. The primary issue is:

1. **gain = 7.0 is too low** for chaotic dynamics with this connectivity structure
   - The gain parameter controls the steepness of the tanh activation
   - g=7 provides moderate nonlinearity but is insufficient to sustain activity
   - Combined with weak connectivity (init: [0, 0.1]), the network quickly stabilizes

2. **No Dale's law** - all connections are mixed sign, which can sometimes lead to cancellation effects

3. **Weak connectivity range [0, 0.1]** - initial weights are all small positive values, limiting the strength of interactions

### Next Steps
To push the network from steady state to chaotic regime, try the following changes:

**Option A (Recommended):** Increase gain
- Change gain in params from 7.0 → 10.0 or 12.0
- Rationale: Higher gain amplifies network activity and strengthens nonlinear effects

**Option B:** Increase connectivity strength
- Change `connectivity_init: [0, 0.1]` → `connectivity_init: [0, 0.15]` or `[0, 0.2]`
- Rationale: Stronger connections propagate activity more effectively

**Option C:** Add Dale's law with excitatory bias
- Add `Dale_law_factor: 0.65`
- Rationale: Ensures proper E/I structure with excitatory dominance

**Applied change:** Increasing gain from 7.0 to 10.0 as the first intervention.

---
