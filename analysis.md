# Experiment Log: signal_chaotic_1

## Iter 1: Steady State
Observation: all neuron traces are flat horizontal lines with no variation over 10000 frames
Change: gain: 10.0 -> 15.0

## Iter 2: Steady State
Observation: all neuron traces remain flat horizontal lines despite gain increase to 15.0
Change: connectivity_init: [0, 0.1] -> [0, 0.3]

## Iter 3: Steady State
Observation: all neuron traces remain flat horizontal lines despite increased connectivity to [0, 0.3]
Change: Dale_law: False -> True, Dale_law_factor: 0.65 (add more excitatory neurons)

## Iter 4: Steady State
Observation: all neuron traces remain flat horizontal lines despite Dale's law with 65% excitatory neurons and gain=15
Change: gain: 15.0 -> 20.0 (further increase gain to push toward chaos)

## Iter 5: Steady State
Observation: all neuron traces remain flat horizontal lines despite gain=20, Dale_law_factor=0.65, and connectivity [0, 0.3]
Change: connectivity_init: [0, 0.3] -> [0, 0.5] (stronger network coupling to propagate activity)

## Iter 6: Steady State
Observation: all neuron traces remain flat horizontal lines despite gain=20, connectivity [0, 0.5], and Dale_law_factor=0.65
Change: params s (self-recurrence): 0.0 -> 0.5 (add self-recurrence to sustain neuron activity)

## Iter 7: Steady State
Observation: all neuron traces remain flat horizontal lines despite spectral radius=1.955, gain=20, s=0.5, connectivity [0, 0.5]
Change: dpos_init: 0 -> 1 (add initial noise to neuron positions/states to perturb from fixed point)

## Iter 8: Steady State
Observation: all neuron traces remain flat horizontal lines despite initial noise dpos_init=1; activity decays too quickly
Change: params a (decay): 1.0 -> 0.1 (reduce decay rate to allow activity to persist longer)

## Iter 9: Steady State
Observation: all neuron traces remain flat horizontal lines despite decay=0.1; network still collapses to fixed point
Change: add external oscillatory input (input_type: oscillation, oscillation_amplitude: 1.0, oscillation_period: 100) to drive sustained activity

## Iter 10: Steady State
Observation: all neuron traces remain flat horizontal lines; external input visible but neurons unresponsive, oscillation frequency may be too high
Change: oscillation_frequency: 5.0 -> 0.5, oscillation_max_amplitude: 1.0 -> 5.0 (slower stronger input to drive activity)

## Iter 11: Steady State
Observation: all neuron traces remain flat horizontal lines; external input visible (yellow dashed line) but neurons still unresponsive despite slow strong oscillation
Change: add external_input_mode: "additive" (missing parameter to couple external input to neuron dynamics)
