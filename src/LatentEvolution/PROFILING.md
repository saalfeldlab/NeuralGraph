# Performance Profiling Guide

This guide explains how to use PyTorch's profiler to generate Chrome trace files for analyzing GPU/CPU performance in the training loop.

## Overview

The profiling feature captures detailed performance metrics including:
- GPU kernel execution times
- CPU operations
- Memory usage (allocations/deallocations)
- Tensor shapes
- Data transfer between CPU/GPU
- CUDA stream information

## Enabling Profiling

To enable profiling, add a `profiling` section at the top level of your configuration YAML file (same level as `training`, `encoder_params`, etc.):

```yaml
# ... encoder_params, decoder_params, evolver_params, etc. ...

training:
  epochs: 11  # Minimum: wait + (warmup + active) * (repeat + 1) = 3 + (1+3)*2 = 11
  # ... other training config ...

profiling:
  wait: 3              # Number of epochs to skip before profiling starts
  warmup: 1            # Number of warmup epochs (not recorded)
  active: 3            # Number of epochs to actively profile
  repeat: 1            # Number of times to repeat the cycle
  record_shapes: true  # Record tensor shapes in traces
  profile_memory: true # Profile memory allocations
  with_stack: false    # Record source code stack traces (adds overhead)
```

### Configuration Parameters

- **`wait`**: Skip this many epochs before starting the profiler. Useful to skip compilation warmup. Default: 3 epochs.
- **`warmup`**: Number of epochs for profiler warmup (recorded but not saved). Default: 1 epoch.
- **`active`**: Number of epochs to actively profile and save traces for. Default: 3 epochs.
- **`repeat`**: How many times to repeat the warmup-active cycle. With repeat=1, the cycle runs twice (once initially, then repeats once). Default: 1.
- **`record_shapes`**: Record tensor shapes in the trace (helpful for debugging shape mismatches).
- **`profile_memory`**: Profile memory usage including allocations and deallocations.
- **`with_stack`**: Record Python stack traces (WARNING: adds significant overhead, use sparingly).

**Note**: Ensure `training.epochs` is at least `wait + (warmup + active) * (repeat + 1)`. For defaults (wait=3, warmup=1, active=3, repeat=1): 3 + (1+3)*2 = **11 epochs minimum**.

## Running with Profiling

You can enable profiling in two ways:

### Option 1: Uncomment in latent_default.yaml

Edit `latent_default.yaml` and uncomment the profiling section at the bottom.

### Option 2: Create a profiling-specific config

Create a new YAML file (e.g., `my_profiling_run.yaml`) with profiling enabled:

```yaml
# Copy settings from latent_default.yaml
latent_dims: 256
# ... other model params ...

training:
  epochs: 11  # Minimum: wait=3 + (warmup=1 + active=3) * (repeat+1=2) = 11
  diagnostics_freq_epochs: 0  # Disable diagnostics during profiling
  # ... other training params ...

profiling:
  wait: 3
  warmup: 1
  active: 3
  repeat: 1
  record_shapes: true
  profile_memory: true
  with_stack: false
```

### Option 3: Override via CLI

Override the profiling config directly from the command line. **Important**: You must include `profiling:profile-config` as a positional argument to enable profiling:

```bash
python src/LatentEvolution/latent.py my_profiling_run \
  --training.epochs 11 \
  profiling:profile-config \
  --profiling.wait 3 \
  --profiling.warmup 1 \
  --profiling.active 3 \
  --profiling.repeat 1 \
  --profiling.record-shapes \
  --profiling.profile-memory \
  --profiling.no-with-stack
```

To disable profiling (default), use `profiling:None` or omit the profiling argument entirely.

**Why `profiling:profile-config`?** Since `profiling` is an optional field (can be `None`), tyro treats it as a union type. You must explicitly select which variant to use: `profiling:profile-config` to enable it, or `profiling:None` to disable it.

**Tip**: Run `python src/LatentEvolution/latent.py my_run --help` to see all available profiling options and the correct syntax.

**Boolean flags**: Use `--profiling.record-shapes` to enable or `--profiling.no-record-shapes` to disable. Don't use `True`/`False`.

## Viewing the Results

The profiler saves trace files in the run directory under `profiler_traces/`:

```
runs/
└── my_profiling_run_2024-01-01_abc123/
    └── run_uuid/
        └── profiler_traces/
            ├── <timestamp>.pt.trace.json  # Chrome trace file
            └── ...
```

### Using Chrome Tracing (Recommended)

1. Open Chrome or Chromium browser
2. Navigate to `chrome://tracing`
3. Click "Load" button
4. Select the `.pt.trace.json` file
5. Explore the timeline:
   - Use WASD keys to navigate
   - Mouse wheel to zoom
   - Click on operations to see details

### Using TensorBoard

The traces are also compatible with TensorBoard:

```bash
tensorboard --logdir=runs/my_profiling_run_2024-01-01_abc123/run_uuid/profiler_traces
```

Then navigate to the "PROFILE" tab in TensorBoard.

## Performance Analysis Tips

### Identifying Bottlenecks

1. **Look for gaps**: Large gaps between operations indicate synchronization points or data transfer overhead.
2. **GPU utilization**: Check if GPU kernels are running continuously or if there are idle periods.
3. **Memory transfers**: Look for H2D (Host to Device) and D2H (Device to Host) transfers that might be blocking.
4. **Kernel duration**: Identify which operations take the longest.

### Common Issues

- **Low GPU utilization**: May indicate CPU bottleneck, small batch sizes, or excessive data transfers.
- **Memory spikes**: Look for unexpected allocations that might cause OOM errors.
- **Slow operations**: Identify if specific layers or operations dominate runtime.

### Optimization Strategies

Based on profiling results, you can:

1. **Increase batch size** if memory allows (better GPU utilization)
2. **Use mixed precision training** (FP16/BF16) to reduce memory and increase speed
3. **Pin memory** for faster CPU-GPU transfers
4. **Use torch.compile** with different modes
5. **Reduce unnecessary data transfers** between CPU and GPU
6. **Optimize data loading** (increase num_workers, use prefetching)

## Best Practices

1. **Disable torch.compile**: Use `--training.train_step train_step_nocompile` to see detailed operations.
2. **Profile 1 active epoch**: Use `--profiling.active 1` to keep trace files small (~50-100MB).
3. **Disable diagnostics**: Set `diagnostics_freq_epochs: 0` during profiling runs.
4. **Skip warmup**: Set `wait: 3` to skip the first few epochs (model warmup).
5. **Start simple**: Keep `with_stack: false` (default) to minimize overhead.
6. **Profile on target hardware**: Profile on the same GPU type you'll use for full training.

## Important: torch.compile and Profiling

**⚠️ torch.compile hides profiling details!**

By default, the training uses `train_step="train_step"` which is a torch.compile'd function. This appears as opaque compiled blocks in the profiler, hiding all the individual GPU operations.

**To see detailed operations, disable torch.compile for profiling:**

```bash
python src/LatentEvolution/latent.py my_profiling_run \
  --training.epochs 11 \
  --training.train_step train_step_nocompile \  # Use uncompiled version!
  profiling:profile-config \
  --profiling.wait 3 \
  --profiling.active 1
```

Or in YAML:
```yaml
training:
  train_step: train_step_nocompile  # Instead of train_step
  epochs: 11

profiling:
  wait: 3
  active: 1  # Just 1 epoch for detailed profiling
```

**Trade-off**: Uncompiled code is slower but shows every operation. Profile with `train_step_nocompile` to identify bottlenecks, then optimize, then use `train_step` for production.

## Platform Support

**Supported Devices**: CUDA (NVIDIA GPUs), CPU

**Not Supported**: MPS (Apple Silicon) - PyTorch profiler is unstable on MPS and will cause crashes. If you enable profiling on a Mac with Apple Silicon, the script will fail with an assertion error. To profile your code, use a machine with NVIDIA GPU or CPU.

## Troubleshooting

### Trace files are too large

- Reduce `active` epochs (1-2 is usually sufficient)
- Set `record_shapes: false` in YAML or use `--profiling.no-record-shapes` in CLI
- Set `with_stack: false` in YAML (this is the default)

### Profiling adds too much overhead

- Reduce `active` epochs
- Ensure `with_stack: false` (default, or use `--profiling.no-with-stack`)
- Use lower `repeat` count

### Missing operations in trace

- Ensure operations are inside the training loop
- Check that `profiler.step()` is called after each epoch

### Trace shows very little detail / appears trivial

This is usually because **torch.compile hides the details**:
- Compiled code appears as single opaque blocks
- Solution: Use `--training.train_step train_step_nocompile`
- You should see hundreds/thousands of GPU kernels per epoch when using uncompiled code
- In Chrome tracing, zoom in and look at the CUDA stream rows at the bottom

### Chrome trace viewer is blank

- **File too large**: 500MB+ files can hang Chrome. Reduce to 1 active epoch.
- **Wrong file**: Open the `.pt.trace.json` file (not `.gz`)
- **Use TensorBoard instead**: `tensorboard --logdir=runs/your_run/profiler_traces`

## Example Workflow

1. Run a short profiling session (with torch.compile disabled for detail):
   ```bash
   python src/LatentEvolution/latent.py profile_test \
     --training.epochs 11 \
     --training.train_step train_step_nocompile \
     profiling:profile-config \
     --profiling.wait 3 \
     --profiling.active 1 \
     --profiling.repeat 0 \
     --profiling.no-record-shapes
   ```

2. Load trace in Chrome at `chrome://tracing`

3. Identify bottlenecks (e.g., data loading, specific layers)

4. Apply optimizations to your config

5. Profile again to verify improvements

6. Run full training with optimized config (profiling disabled)
