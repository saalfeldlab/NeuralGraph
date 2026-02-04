# LatentEvolution

## goal

model neural activity of N neurons (N=13741 for flyvis) in a latent space of dimension L << N (typically L=256). the network takes visual stimuli and past activity and predicts the next time step. initial focus: fly visual system (flyvis).

## key question

is fitting neural activity subsampled at interval `time_units` (tu) sufficient to recover t->t+1 dynamics? target: tu >= 20. stimulus is provided every time step regardless of tu.

## acquisition modes

- **time aligned**: all neurons measured simultaneously at t=0, tu, 2tu, ...
- **staggered**: subset of neurons acquired each step, cycling so every neuron is measured once per tu window. this is the case of real interest. see `experiments/flyvis_voltage_Nsteps_staggered.md` for diagrams.

## architecture (EED)

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

- **encoder/decoder**: symmetric MLPWithSkips. maps between neural activity and latent space.
- **evolver**: MLPWithSkips with Tanh. residual update, zero-initialized (starts as identity).
- **stimulus encoder**: 3-layer MLP (1736 -> 64 dims), optionally pretrained as autoencoder.

## training files

| file | role |
|------|------|
| `latent.py` | time-aligned training |
| `latent_stag_interp.py` | staggered training (working approach): interpolate staggered activities, encode to latent, evolve+decode, loss on real measurements only |
| `latent_stag_z0_bank.py` | staggered with learned z0 bank. **deprecated, does not work** |

## other files

| file | role |
|------|------|
| `eed_model.py` | model architecture and config classes |
| `mlp.py` | MLP / MLPWithSkips |
| `training_config.py` | pydantic config dataclasses |
| `acquisition.py` | acquisition modes (all_time_points, time_aligned, staggered_random) |
| `pipeline_chunk_loader.py` | 3-stage parallel data loader (disk -> cpu -> gpu) |
| `stimulus_ae_model.py` | stimulus autoencoder pretraining |
| `stimulus_utils.py` | stimulus downsampling |
| `diagnostics.py` | validation analysis, rollout evaluation, plotting |
| `diagnostics_stag.py` | staggered-specific diagnostics |
| `benchmark_rollout.py` | rollout benchmarking |
| `hparam_paths.py` | run directory / hparam path management |
| `checkpoint.py` | checkpoint save/load |
| `interpolate_staggered.py` | interpolation utilities for staggered data |

## configs

yaml files: `latent_1step.yaml`, `latent_20step.yaml`, `latent_50step.yaml`, `latent_stag_20step.yaml`. these set tu, evolve_multiple_steps, acquisition_mode, and model architecture.

## key parameters

- `time_units` (tu): observation interval in time steps.
- `evolve_multiple_steps` (ems): tu-multiples to roll out during training. total evolution = tu * ems.
- `latent_dims`: bottleneck dimension (default 256).
- `acquisition_mode`: all_time_points | time_aligned | staggered_random.
- `stimulus_frequency`: ALL | NONE | TIME_UNITS_CONSTANT | TIME_UNITS_INTERPOLATE.
- `reconstruction_warmup_epochs`: epochs to pretrain encoder/decoder before enabling evolver.
- `pretrain_stimulus_ae`: pretrain stimulus encoder as autoencoder.
- `zero_init`: zero-initialize evolver output (starts as identity).
- `tv_reg_loss`: total variation on evolver updates (stabilizes rollouts).
- `unconnected_to_zero`: connectome-based augmentation loss.

## losses

- **RECON**: MSE(decoder(encoder(x)), x).
- **EVOLVE**: MSE at each tu-multiple during rollout.
- **TV_LOSS**: L1 on evolver delta_z.
- **AUG_LOSS**: connectome augmentation (zeroing unconnected inputs shouldn't change output).
- **REG**: L1 weight regularization.

## training phases

1. stimulus autoencoder pretraining (optional)
2. reconstruction warmup: encoder/decoder only, evolver frozen
3. main training: all components jointly with multi-step rollout

## usage

```bash
python latent.py <experiment_name> latent_20step.yaml [--overrides]
python latent_stag_interp.py <experiment_name> latent_stag_20step.yaml [--overrides]
```

## cluster runs

results: `/groups/saalfeld/home/kumarv4/repos/NeuralGraph/runs`. local `./runs` is ephemeral (testing only).

experiments documented in `experiments/*.md`. bsub commands there specify run directories.

## tensorboard plots for evaluating a run

- `CrossVal/.*/multi_start_2000step_latent_rollout_mses_by_time`: long-term rollout MSE on held-out data (davis_2016_2017, optical_flow). averaged over starts and neurons. should not diverge within 2000 steps.
- `CrossVal/.*/time_aligned_mse_latent`: rollout MSE zoomed to training window (tu * ems). includes constant/linear interpolation baselines.
- `CrossVal/.*/[best|worst].*rollout_latent_mse_var_scatter`: total variance vs unexplained variance per neuron, colored by cell type.
- `CrossVal/.*/[best|worst]_2000step_rollout_latent_traces`: predicted vs ground truth traces for best/worst rollouts.

## experiment docs

- `experiments/flyvis_voltage_1step.md` - 1-step baseline
- `experiments/flyvis_voltage_Nsteps_aligned.md` - time-aligned multi-step
- `experiments/flyvis_voltage_Nsteps_staggered.md` - staggered acquisition
- `experiments/flyvis_calcium.md` - calcium signal

## development practices

- never commit to main; always use a feature branch. branch naming: `claude/<topic>`.
- conventional commits (`feat:`, `fix:`, `refactor:`). keep messages high-level: state motivation, note breaking changes. don't enumerate files.
- PRs: brief, high-level description only. e.g. "refactor to share the training loop between latent.py and latent_stag_interp.py" — not a detailed explanation of how.
- bug fixes: write a minimal reproducing test first (unittest), confirm it fails, then fix. not always feasible (some bugs need full training).
- use `unittest`, not pytest.
- conda env: `neural-graph-linux`.
- prefer vectorized/batched tensor ops over python loops.
- **no lazy imports.** all imports must be top-level. never use `import` inside a function or method. use `from __future__ import annotations` and `TYPE_CHECKING` blocks to break circular import issues when needed for type hints only.
- **circular import check on refactors.** when refactoring modules, verify there are no circular import issues by importing every `.py` module in this directory (e.g. `from LatentEvolution.<module> import ...` for each module) and confirming no `ImportError`.
