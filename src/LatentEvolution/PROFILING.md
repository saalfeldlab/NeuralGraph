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
python latent.py my_profiling_run \
  --training.epochs 11 \
  profiling:profile-config \
  --profiling.wait 3 \
  --profiling.warmup 1 \
  --profiling.active 3 \
  --profiling.repeat 1 \
  --profiling.record_shapes True \
  --profiling.profile_memory True \
  --profiling.with_stack False
```

To disable profiling (default), use `profiling:None` or omit the profiling argument entirely.

**Why `profiling:profile-config`?** Since `profiling` is an optional field (can be `None`), tyro treats it as a union type. You must explicitly select which variant to use: `profiling:profile-config` to enable it, or `profiling:None` to disable it.

**Tip**: Run `python latent.py my_run --help` to see all available profiling options and the correct syntax.

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

1. **Profile short runs**: Use 11-15 epochs max to keep trace files manageable (minimum 11 with defaults).
2. **Disable diagnostics**: Set `diagnostics_freq_epochs: 0` during profiling runs.
3. **Skip warmup**: Set `wait: 3` to skip the first few epochs (torch.compile and model warmup).
4. **Start simple**: Begin with `with_stack: false` to minimize overhead.
5. **Profile on target hardware**: Profile on the same GPU type you'll use for full training.

## Troubleshooting

### Trace files are too large

- Reduce `active` epochs (1-2 is usually sufficient)
- Set `record_shapes: false`
- Set `with_stack: false`

### Profiling adds too much overhead

- Reduce `active` epochs
- Disable `with_stack`
- Use lower `repeat` count

### Missing operations in trace

- Ensure operations are inside the training loop
- Check that `profiler.step()` is called after each epoch

## Example Workflow

1. Run a short profiling session:
   ```bash
   python latent.py profile_test --training.epochs 11 profiling:profile-config --profiling.wait 3 --profiling.active 3
   ```

2. Load trace in Chrome at `chrome://tracing`

3. Identify bottlenecks (e.g., data loading, specific layers)

4. Apply optimizations to your config

5. Profile again to verify improvements

6. Run full training with optimized config (profiling disabled)
