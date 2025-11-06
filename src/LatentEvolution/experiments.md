# Experiments on flyvis data using latent evolution model

## Baseline experiment

We start with fly_N9_62_1 & model voltage at the simulated 20ms time step resolution. Note that
we are not adding in the stimulus.

### sweep batch size

```bash

for bs in 32 128 512 2048; do \
    bsub -J "batch${bs}" -n 12 -gpu "num=1" -q gpu_a100 -o batch${bs}.log python \
        src/LatentEvolution/latent.py batch_size_sweep \
        --training.batch-size $bs \
        --training.epochs 5000
    done
```

#### Results (Claude)

All 4 runs completed successfully.

| Batch Size | Train Time (min) | Train Loss | Val Loss | Test Loss |
| ---------- | ---------------- | ---------- | -------- | --------- |
| 32         | 114.0            | 0.040364   | 0.023608 | 0.022846  |
| 128        | 32.6             | 0.037480   | 0.058066 | 0.071266  |
| 512        | 14.0             | 0.037044   | 0.024020 | 0.023698  |
| 2048       | 9.4              | 0.041691   | 0.029996 | 0.029193  |

**Observations:**

- **Training Speed**: Larger batch sizes are significantly faster (2048 is 12× faster than batch size 32)
- **Model Quality**:
  - **Batch size 128 failed catastrophically** with test loss of 0.071266 (3× worse than others)
  - **Best performance**: Batch size 32 achieved lowest test loss (0.022846)
  - Batch sizes 32 and 512 perform similarly well (test loss ~0.023)
  - Batch size 2048 shows degraded performance (test loss 0.029193)
- The batch size 128 result appears to be a training failure rather than proper convergence

### sweep learning rate

```bash

for lr in 0.001 0.0001 0.00001 0.000001 ; do \
    bsub -J "lr${lr}" -n 12 -gpu "num=1" -q gpu_a100 -o lr${lr}.log python \
        src/LatentEvolution/latent.py learning_rate_sweep \
        --training.batch-size 128 \
        --training.epochs 5000 \
        --training.learning-rate $lr
    done
```

#### Results (Claude)

All 4 runs completed successfully.

| Learning Rate | Train Time (min) | Train Loss | Val Loss | Test Loss |
| ------------- | ---------------- | ---------- | -------- | --------- |
| 0.001         | 33.9             | 0.037480   | 0.058066 | 0.071266  |
| 0.0001        | 33.8             | 0.038113   | 0.024581 | 0.024795  |
| 0.00001       | 33.7             | 0.045904   | 0.030507 | 0.029725  |
| 0.000001      | 32.7             | 0.063929   | 0.039382 | 0.038898  |

**Observations:**

- **Training Speed**: All learning rates have nearly identical training times (~33 min) - learning rate does not impact training speed
- **Model Quality**:
  - **LR 0.0001 achieves best performance** with test loss of 0.024795
  - **LR 0.001 (default) failed catastrophically** with test loss of 0.071266 (3× worse)
  - LR 0.00001 shows moderate performance degradation (test loss 0.029725)
  - LR 0.000001 (too small) leads to poor convergence (test loss 0.038898)
- The catastrophic failure at LR 0.001 with batch size 128 suggests this specific configuration is problematic
- **Recommendation**: Use learning rate 0.0001 for best model quality

See [#25](https://github.com/saalfeldlab/NeuralGraph/pull/25) - many jobs failed due
to the use of `max-autotune` compilation. I don't yet understand why, but changing to
`reduce-overhead` which is less aggressive worked.

### sweep latent dimensions

Based on the above two experiments, let's set the default batch size to 32 and
the learning rate to 1e-4. tf32 didn't make a difference, let's turn it off
for now.

```bash

for ldim in 64 128 256 512; do \
    bsub -J "ldim${ldim}" -n 12 -gpu "num=1" -q gpu_a100 -o ldim${bs}.log python \
        src/LatentEvolution/latent.py latent_dim_sweep \
        --training.latent-dims $ldim \
        --training.epochs 10000
    done
```

## Performance benchmark experiments

Assess the performance impact of (GPU, compile?, tensor float32).

```bash

gpu_types=("gpu_l4" "gpu_a100" "gpu_h100" "gpu_h200")
train_steps=("train_step_nocompile" "train_step")
tf32_flags=("use-tf32-matmul" "no-use-tf32-matmul")

for gpu_type in "${gpu_types[@]}"
do
    for train_step in "${train_steps[@]}"
    do
        for tf32_flag in "${tf32_flags[@]}"
        do
            if [[ $gpu_type == "gpu_l4" ]]; then
                slots_per_gpu="8"
            else
                slots_per_gpu="12"
            fi
            name="${gpu_type}_${train_step}_${use_tf32_matmul}"
            bsub -J $name -n $slots_per_gpu \
                -gpu \"num=1\" -q $gpu_type -o ${name}.log python \
                src/LatentEvolution/latent.py gpu_type_sweep \
                --training.train-step $train_step \
                --training.${tf32_flag} \
                --training.epochs 5000 \
                --training.batch-size 256
        done
    done
done
```

### Results (Claude)

All 16 runs completed successfully. The experiment swept 3 parameters across 4 × 2 × 2 = 16 combinations:

- **GPU Type**: L4, A100, H100, H200
- **Compilation**: `train_step` (compiled) vs `train_step_nocompile`
- **TF32**: enabled vs disabled

| GPU Type   | Compilation   | TF32   | Train Time (min)   | Avg Epoch (s)   | GPU Mem (GB)   | Mem Used (GB)   | Mem %   | Train Loss   | Val Loss   | Test Loss   |
| ---------- | ------------- | ------ | ------------------ | --------------- | -------------- | --------------- | ------- | ------------ | ---------- | ----------- |
| **L4**     | Compiled      | ✓      | 73.9               | 0.83            | 22.5           | 7.6             | 33.9%   | 0.036807     | 0.022916   | 0.022492    |
| **L4**     | Compiled      | ✗      | 73.1               | 0.82            | 22.5           | 7.6             | 33.9%   | 0.037038     | 0.022594   | 0.022269    |
| **L4**     | No compile    | ✓      | 129.9              | 1.50            | 22.5           | 7.6             | 34.0%   | 0.036822     | 0.030394   | 0.029912    |
| **L4**     | No compile    | ✗      | 139.6              | 1.61            | 22.5           | 7.6             | 34.0%   | 0.037055     | 0.022462   | 0.022135    |
| ---------- | ------------- | ------ | ------------------ | --------------- | -------------- | --------------- | ------- | ------------ | ---------- | ----------- |
| **A100**   | Compiled      | ✓      | 19.4               | 0.20            | 80.0           | 7.9             | 9.9%    | 0.036817     | 0.026472   | 0.025493    |
| **A100**   | Compiled      | ✗      | 25.5               | 0.26            | 80.0           | 7.9             | 9.9%    | 0.037191     | 0.022161   | 0.021853    |
| **A100**   | No compile    | ✓      | 58.9               | 0.67            | 80.0           | 7.9             | 9.9%    | 0.037209     | 0.023222   | 0.023485    |
| **A100**   | No compile    | ✗      | 58.8               | 0.67            | 80.0           | 7.9             | 9.9%    | 0.037032     | 0.024684   | 0.024398    |
| ---------- | ------------- | ------ | ------------------ | --------------- | -------------- | --------------- | ------- | ------------ | ---------- | ----------- |
| **H100**   | Compiled      | ✓      | 34.7               | 0.37            | 79.6           | 8.0             | 10.0%   | 0.036632     | 0.030513   | 0.030244    |
| **H100**   | Compiled      | ✗      | 34.8               | 0.37            | 79.6           | 8.0             | 10.0%   | 0.036865     | 0.024154   | 0.023849    |
| **H100**   | No compile    | ✓      | 56.7               | 0.63            | 79.6           | 8.1             | 10.2%   | 0.037008     | 0.023195   | 0.023047    |
| **H100**   | No compile    | ✗      | 61.0               | 0.68            | 79.6           | 8.1             | 10.2%   | 0.037247     | 0.024638   | 0.024301    |
| ---------- | ------------- | ------ | ------------------ | --------------- | -------------- | --------------- | ------- | ------------ | ---------- | ----------- |
| **H200**   | Compiled      | ✓      | 19.0               | 0.20            | 140.4          | 8.0             | 5.7%    | 0.036632     | 0.030513   | 0.030244    |
| **H200**   | Compiled      | ✗      | 19.1               | 0.20            | 140.4          | 8.0             | 5.7%    | 0.036865     | 0.024154   | 0.023849    |
| **H200**   | No compile    | ✓      | 40.7               | 0.46            | 140.4          | 8.1             | 5.8%    | 0.037008     | 0.023195   | 0.023047    |
| **H200**   | No compile    | ✗      | 42.0               | 0.47            | 140.4          | 8.1             | 5.8%    | 0.037247     | 0.024638   | 0.024301    |

**Performance Observations:**

**Training Speed:**

- **Fastest**: H200 + Compiled + TF32 (19.0 min)
- **Slowest**: L4 + No compile + no TF32 (139.6 min, 7.3× slower)
- **Compilation speedup**: 1.5-2.4× across all GPUs
- **GPU Ranking**: H200 ≈ A100 (fastest) > H100 > L4 (slowest)
- TF32 has minimal impact on training speed (typically <5% difference)

**GPU Memory Usage:**

- All configurations use ~8 GB memory regardless of GPU type, compilation, or TF32 settings
- L4 (22.5 GB total) has highest utilization at 34%
- H200 (140.4 GB total) has lowest utilization at 5.7%
- Memory usage is not a limiting factor for this workload on any GPU
