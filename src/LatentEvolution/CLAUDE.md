# LatentEvolution

## goal

model real neural activity in a low-dimensional latent space. given N neurons, learn a latent representation of dimension L << N that captures dynamics and enables prediction. flyvis (simulated fly visual system) serves as a testbed to develop methods; zapbench (real zebrafish calcium imaging) is the main focus.

## core approach

### architecture (EED: encoder-evolver-decoder)

```
x(t) [N] ---> encoder ---> z(t) [L]
                                 |
                       evolver: z(t+1) = z(t) + f(z(t), ...)
                                 |
                         decoder: z(t+k) ---> x_hat(t+k) [N]
```

- **encoder/decoder**: symmetric MLPWithSkips. maps between neural activity and latent space.
- **evolver**: MLPWithSkips with Tanh. residual update, zero-initialized (starts as identity).
- optional inputs to evolver (e.g., stimulus encoding) depend on the application.

### acquisition modes

- **time aligned**: all neurons measured simultaneously at t=0, tu, 2tu, ...
- **staggered**: subset of neurons acquired each step, cycling so every neuron is measured once per observation window. this is the real-world case of interest.

### losses

- **RECON**: MSE(decoder(encoder(x)), x) — autoencoder reconstruction.
- **EVOLVE**: MSE at each step during rollout.
- **TV_LOSS**: L1 on evolver delta_z (stabilizes rollouts).
- **REG**: L1 weight regularization.

### training phases

1. reconstruction warmup: encoder/decoder only, evolver frozen
2. main training: all components jointly with multi-step rollout

## shared infrastructure

| file | role |
|------|------|
| `eed_model.py` | model architecture and config classes |
| `mlp.py` | MLP / MLPWithSkips |
| `checkpoint.py` | checkpoint save/load |
| `hparam_paths.py` | run directory / hparam path management |
| `diagnostics.py` | validation analysis, rollout evaluation, plotting |
| `interpolate_staggered.py` | interpolation utilities for staggered data |

### cluster runs

results: `/groups/saalfeld/home/kumarv4/repos/NeuralGraph/runs`. local `./runs` is ephemeral (testing only).

---

## flyvis (simulation testbed)

simulated fly visual system. feedforward network driven by video stimulus. useful for developing and validating methods before applying to real data.

### key characteristics

- N = 13741 neurons
- feedforward architecture, heavily driven by visual stimulus
- ground truth dynamics available (simulated)
- controlled staggered acquisition for method development

### flyvis-specific architecture

```
x(t) [N] ---> encoder ---> z(t) [L]
                                 |
stim(t) --> stim_encoder --> z_s(t) [64]
                                 |
                       evolver: z(t+1) = z(t) + f(z(t), z_s(t))
                                 |
                           (repeat tu times)
                                 |
                         decoder: z(t+tu) ---> x_hat(t+tu) [N]
```

- **stimulus encoder**: 3-layer MLP (1736 -> 64 dims), optionally pretrained as autoencoder.

### stimulus-only null model

baseline model that predicts neural activity from stimulus context alone, without encoder/decoder/latent space. tests whether stimulus is sufficient to explain activity (spoiler: it largely is, since flyvis is feedforward). implemented in `stimulus_only.py`.

```
stim[t-tu:t] --> stim_encoder (per frame) --> flatten --> predictor MLP --> x_hat(t) [N]
```

### training files

| file | role |
|------|------|
| `latent.py` | time-aligned training |
| `latent_stag_interp.py` | staggered training: interpolate activities, encode to latent, evolve+decode, loss on real measurements only |
| `latent_stag_z0_bank.py` | staggered with learned z0 bank. **deprecated, does not work** |
| `stimulus_only.py` | stimulus-only null model baseline |

### other flyvis files

| file | role |
|------|------|
| `training_config.py` | pydantic config dataclasses |
| `acquisition.py` | acquisition modes (all_time_points, time_aligned, staggered_random) |
| `pipeline_chunk_loader.py` | 3-stage parallel data loader (disk -> cpu -> gpu) |
| `stimulus_ae_model.py` | stimulus autoencoder pretraining |
| `stimulus_utils.py` | stimulus downsampling |
| `diagnostics_stag.py` | staggered-specific diagnostics |
| `benchmark_rollout.py` | rollout benchmarking |

### configs

yaml files: `latent_1step.yaml`, `latent_20step.yaml`, `latent_50step.yaml`, `latent_stag_20step.yaml`, `stimulus_only_20step.yaml`.

### key parameters

- `time_units` (tu): observation interval in time steps.
- `evolve_multiple_steps` (ems): tu-multiples to roll out during training. total evolution = tu × ems.
- `latent_dims`: bottleneck dimension (default 256).
- `acquisition_mode`: all_time_points | time_aligned | staggered_random.
- `stimulus_frequency`: ALL | NONE | TIME_UNITS_CONSTANT | TIME_UNITS_INTERPOLATE.
- `reconstruction_warmup_epochs`: epochs to pretrain encoder/decoder before enabling evolver.
- `pretrain_stimulus_ae`: pretrain stimulus encoder as autoencoder.
- `zero_init`: zero-initialize evolver output (starts as identity).
- `tv_reg_loss`: total variation on evolver updates.
- `unconnected_to_zero`: connectome-based augmentation loss.

### flyvis-specific losses

- **AUG_LOSS**: connectome augmentation (zeroing unconnected inputs shouldn't change output).

### usage

```bash
python latent.py <experiment_name> latent_20step.yaml [--overrides]
python latent_stag_interp.py <experiment_name> latent_stag_20step.yaml [--overrides]
python stimulus_only.py <experiment_name> stimulus_only_20step.yaml [--overrides]
```

### tensorboard plots

- `CrossVal/.*/multi_start_2000step_latent_rollout_mses_by_time`: long-term rollout MSE on held-out data. should not diverge within 2000 steps.
- `CrossVal/.*/short_rollout_mse_latent`: rollout MSE for first 250 steps. includes constant baseline; linear interpolation baseline when tu > 1.
- `CrossVal/.*/[best|worst].*rollout_latent_mse_var_scatter`: total variance vs unexplained variance per neuron, colored by cell type.
- `CrossVal/.*/[best|worst]_2000step_rollout_latent_traces`: predicted vs ground truth traces.

### experiment docs

- `experiments/flyvis_voltage_1step.md` - 1-step baseline
- `experiments/flyvis_voltage_Nsteps_aligned.md` - time-aligned multi-step
- `experiments/flyvis_voltage_Nsteps_staggered.md` - staggered acquisition
- `experiments/flyvis_calcium.md` - calcium signal
- `experiments/flyvis_stimulus_only_null.md` - stimulus-only null model

---

## zapbench (real zebrafish data) — main focus

real calcium imaging data from zebrafish. this is the primary application; methods developed on flyvis are adapted here.

### key differences from flyvis

| aspect | flyvis | zapbench |
|--------|--------|----------|
| data | simulated | real calcium imaging |
| stimulus | visual input via stim_encoder | none |
| network | feedforward, stimulus-driven | recurrent, spontaneous activity |
| timing | strict phase (neurons cycle through tu) | irregular (varies within frame) |
| parameters | tu × ems | fitting_window (bins) |
| metrics | per-step MSE | per-frame MSE (aggregate within original frames) |

### data characteristics

- **acquisition**: volumetric calcium imaging, 72 Z planes per frame
- **staggered timing**: neurons observed at different times within each frame (~914ms between successive observations of same neuron)
- **ephys timestamps**: `cell_ephys_index` zarr array stores 6kHz sample indices for each (frame, neuron)
- **observation density**: ~2.6% (sparse)

### data processing

1. load `cell_ephys_index` (sample indices at 6kHz)
2. convert to milliseconds: `acq_ms = cell_ephys_index * 1000 / sampling_freq_hz`
3. bin into 40ms intervals: `bin_idx = floor(acq_ms / bin_size_ms)`
4. store as sparse `(N, K)` tensors: `obs_times`, `obs_vals`
5. interpolate via searchsorted to fill all bins

### architecture

simplified EED without stimulus encoder:

```
z = encode(interpolated_activity[:, 0, :])
for t in range(fitting_window):
    x_pred = decode(z)
    loss += masked_mse(x_pred, activity[:, t, :], obs_mask[:, t, :])
    z = evolve(z)
```

- **context**: encode single bin to initialize latent state
- **evolution**: one step per bin (no stimulus input)
- **loss**: MSE only on observed values (frame_index >= 0)

### evaluation

- **rollout**: encode initial state, evolve+decode for full validation length
- **metrics**: MSE/MAE aggregated per original frame (not per bin)
- **baseline**: predict mean of first 4 frames per neuron (constant prediction)
- **stability**: rollout MSE should not diverge; plotted vs frame index

### key files

| file | role |
|------|------|
| `zapbench_train.py` | training loop |
| `zapbench_data.py` | sparse loading, binning, interpolation |
| `zapbench_eed.py` | EED model (encoder-evolver-decoder) |
| `zapbench_config.py` | config classes, CONDITIONS metadata |

### key parameters

- `bin_size_ms`: interpolation grid spacing (default 40ms)
- `fitting_window`: bins to predict during training (default 100, ~4s)
- `latent_dims`: latent space dimension (default 64)

### usage

```bash
python zapbench_train.py <expt_code> [--overrides]
python zapbench_train.py my_expt --train.fitting_window 50 --model.latent_dims 128
```

---

## development practices

- **keep CLAUDE.md up to date.** when making significant code changes (new files, changed architecture, deprecated approaches, new parameters), check if this file needs updates. do this before committing, before creating a PR, and after finishing a big feature.
- never commit to main; always use a feature branch. branch naming: `claude/<topic>`.
- conventional commits (`feat:`, `fix:`, `refactor:`). keep messages high-level: state motivation, note breaking changes. don't enumerate files.
- PRs: brief, high-level description only. e.g. "refactor to share the training loop between latent.py and latent_stag_interp.py" — not a detailed explanation of how.
- bug fixes: write a minimal reproducing test first (unittest), confirm it fails, then fix. not always feasible (some bugs need full training).
- **IMPORTANT: use `unittest`, not pytest.** run tests with `python -m unittest discover -s src/LatentEvolution -p '*_test.py'`. never use pytest. test file for `module.py` should be `module_test.py` in the same directory.
- **run `make test` before committing big changes.** unit tests take 1-2 minutes.
- conda env: `neural-graph-linux`.
- prefer vectorized/batched tensor ops over python loops.
- **no lazy imports.** all imports must be top-level. never use `import` inside a function or method. use `from __future__ import annotations` and `TYPE_CHECKING` blocks to break circular import issues when needed for type hints only.
- **circular import check on refactors.** when refactoring modules, verify there are no circular import issues by importing every `.py` module in this directory (e.g. `from LatentEvolution.<module> import ...` for each module) and confirming no `ImportError`.
