# Chunked Streaming Integration Guide

This guide shows how to integrate `RandomChunkLoader` into `latent.py` to reduce GPU memory usage.

## Overview

**Current approach:**
- Load entire training dataset to GPU (~60 GB)
- Sample batches from GPU tensor
- Fast but memory-intensive

**Chunked streaming approach:**
- Load random 64K chunks from disk → CPU (background thread)
- Transfer chunks to GPU (async)
- Sample batches within current chunk
- Memory: ~4 GB (2 chunks) vs 60 GB (full dataset)

## Integration Steps

### 1. Import New Modules

```python
from LatentEvolution.chunk_loader import RandomChunkLoader
from LatentEvolution.chunk_streaming import (
    create_zarr_loader,
    sample_batch_within_chunk,
    calculate_chunk_params,
    ChunkLatencyStats,
)
```

### 2. Replace `load_dataset()` Function

**Before:**
```python
train_data, val_data, train_stim, val_stim, neuron_data = load_dataset(
    simulation_config=cfg.training.simulation_config,
    column_to_model=cfg.training.column_to_model,
    data_split=cfg.training.data_split,
    num_input_dims=cfg.stimulus_encoder_params.num_input_dims,
    device=device,
)
# train_data is on GPU, shape: (967680, 13741) ~53 GB
```

**After:**
```python
# Load metadata and validation data only (validation stays on GPU - small enough)
data_path = f"graphs_data/fly/{cfg.training.simulation_config}/x_list_0"
column_idx = FlyVisSim[cfg.training.column_to_model].value

# validation data (small, fits on GPU)
val_data = torch.from_numpy(
    load_column_slice(data_path, column_idx,
                     cfg.training.data_split.validation_start,
                     cfg.training.data_split.validation_end)
).to(device)

val_stim = torch.from_numpy(
    load_column_slice(data_path, FlyVisSim.STIMULUS.value,
                     cfg.training.data_split.validation_start,
                     cfg.training.data_split.validation_end,
                     neuron_limit=cfg.stimulus_encoder_params.num_input_dims)
).to(device)

# Load neuron metadata
metadata = load_metadata(data_path)
neuron_data = NeuronData.from_metadata(metadata)

# Create zarr loader for training data (NOT loaded to GPU yet)
train_total_timesteps = cfg.training.data_split.train_end - cfg.training.data_split.train_start

zarr_load_fn = create_zarr_loader(
    data_path=data_path,
    column_idx=column_idx,
    stim_column_idx=FlyVisSim.STIMULUS.value,
    num_stim_dims=cfg.stimulus_encoder_params.num_input_dims,
)

# Create chunk loader (streams data in background)
chunk_size = 65536  # 64K timesteps
chunk_loader = RandomChunkLoader(
    load_fn=lambda start, end: zarr_load_fn(
        cfg.training.data_split.train_start + start,
        cfg.training.data_split.train_start + end
    ),
    total_timesteps=train_total_timesteps,
    chunk_size=chunk_size,
    device=device,
    prefetch=2,  # buffer 2 chunks ahead
    seed=cfg.training.seed,
)
```

### 3. Calculate Chunk Parameters

```python
# Calculate chunking parameters
total_steps = cfg.training.time_units * cfg.training.evolve_multiple_steps
chunks_per_epoch, batches_per_chunk, batches_per_epoch = calculate_chunk_params(
    total_timesteps=train_total_timesteps,
    chunk_size=chunk_size,
    batch_size=cfg.training.batch_size,
    data_passes_per_epoch=cfg.training.data_passes_per_epoch,
)

print(f"Chunked streaming: {chunks_per_epoch} chunks/epoch, "
      f"{batches_per_chunk} batches/chunk, {batches_per_epoch} total batches/epoch")
```

### 4. Modify Training Loop

**Before:**
```python
# Old: single iterator over full dataset
batch_indices_iter = make_batches_random(
    train_data, train_stim, wmat_indices, wmat_indptr, cfg
)

for epoch in range(cfg.training.epochs):
    for _ in range(batches_per_epoch):
        batch_indices, selected, needed = next(batch_indices_iter)
        loss_tuple = train_step_fn(model, train_data, train_stim,
                                   batch_indices, selected, needed, cfg)
        loss_tuple[0].backward()
        optimizer.step()
```

**After:**
```python
# Initialize latency tracker
latency_stats = ChunkLatencyStats()

# Epoch loop with chunked streaming
for epoch in range(cfg.training.epochs):
    epoch_start = datetime.now()
    losses = LossComponents()

    # Start loading chunks for this epoch
    chunk_loader.start_epoch(num_chunks=chunks_per_epoch)

    # Iterate over chunks
    for chunk_idx in range(chunks_per_epoch):
        # Get next chunk (blocks until ready, overlaps with previous training)
        get_start = time.time()
        chunk_data, chunk_stim = chunk_loader.get_next_chunk()
        latency_stats.record_chunk_get(time.time() - get_start)

        if chunk_data is None:
            break  # end of epoch

        # Train on batches within this chunk
        for batch_in_chunk in range(batches_per_chunk):
            optimizer.zero_grad()

            # Sample batch within current chunk
            batch_indices, selected_neurons, needed_indices = sample_batch_within_chunk(
                chunk_data=chunk_data,
                chunk_stim=chunk_stim,
                wmat_indices=wmat_indices,
                wmat_indptr=wmat_indptr,
                batch_size=cfg.training.batch_size,
                total_steps=total_steps,
                num_neurons_to_zero=cfg.training.unconnected_to_zero.num_neurons,
                device=device,
            )

            # Training step (same as before, but uses chunk instead of full dataset)
            forward_start = time.time()
            loss_tuple = train_step_fn(
                model, chunk_data, chunk_stim,
                batch_indices, selected_neurons, needed_indices, cfg
            )
            forward_time = time.time() - forward_start

            backward_start = time.time()
            loss_tuple[0].backward()
            backward_time = time.time() - backward_start

            step_start = time.time()
            if cfg.training.grad_clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_max_norm)
            optimizer.step()
            step_time = time.time() - step_start

            losses.accumulate(*loss_tuple)

            # Sample timing every 10 batches to avoid overhead
            if batch_in_chunk % 10 == 0:
                latency_stats.record_batch_times(forward_time, backward_time, step_time)

    # Print latency stats every epoch
    if epoch % 10 == 0:
        latency_stats.print_summary()

    # ... rest of epoch logic (validation, checkpointing, etc.)
```

### 5. Cleanup

```python
# At end of training
chunk_loader.cleanup()
```

## Memory Savings

**Before (full dataset on GPU):**
- Training data: 967,680 × 13,741 × 4 bytes = **53 GB**
- Training stimulus: 967,680 × 1,736 × 4 bytes = **6.7 GB**
- **Total: ~60 GB**

**After (chunked streaming):**
- Current chunk: 65,536 × 13,741 × 4 bytes = **3.6 GB**
- Prefetch buffer (2 chunks): **7.2 GB**
- Validation data (stays on GPU): **0.8 GB**
- **Total: ~8 GB** (87% reduction)

## Performance Considerations

**Key metrics to monitor:**
- `chunk_get_mean_ms`: Should be <50ms if overlap is working (chunks prefetched)
- `chunk_get_max_ms`: First chunk will be slower (~200ms cold start)
- `batch_forward_mean_ms`: Should be similar to before
- Epoch duration: Expect 0-10% overhead if overlap is good

**Tuning parameters:**
- `chunk_size`: Larger = better amortization of overhead, but more memory
  - 65536 (64K) recommended for your dataset
- `prefetch`: Number of chunks to buffer
  - 1-2 recommended (more wastes CPU RAM)
- `batches_per_chunk`: Determined by chunk_size / batch_size
  - Your case: 65536 / 256 = 256 batches per chunk

## Expected Latencies

Based on unit tests with 100ms load + 100ms train:
- **chunk_get_time**: 0-10ms (after warmup)
- **speedup**: ~1.4-1.7x vs sequential (overlap working)

With real training (zarr load ~50-100ms, training ~20-50ms per batch × 256):
- **chunk_get_time**: Should be near-zero (background stays ahead)
- **overhead**: <5% if overlap is good
