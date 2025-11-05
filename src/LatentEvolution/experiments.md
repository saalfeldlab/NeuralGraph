# Experiments on flyvis data using latent evolution model

## Baseline experiment

We start with fly_N9_62_1 & model voltage at the simulated 20ms time step resolution. Note that
we are not adding in the stimulus.

### sweep batch size

```bash

for bs in 32 128 512 2048; do \
    bsub -J "batch${bs}" -n 12 -gpu "num=1" -q gpu_a100 -o batch${bs}.log python \
        src/LatentEvolution/latent.py \
        --training.batch-size $bs \
        --training.epochs 5000
    done
```

### sweep learning rate

```bash

for lr in 0.001 0.0001 0.00001 0.000001 ; do \
    bsub -J "lr${lr}" -n 12 -gpu "num=1" -q gpu_a100 -o lr${lr}.log python \
        src/LatentEvolution/latent.py \
        --training.batch-size 128 \
        --training.epochs 5000 \
        --training.learning-rate $lr
    done
```

See [#25](https://github.com/saalfeldlab/NeuralGraph/pull/25) - many jobs failed due to the use of `max-autotune` compilation. I don't yet understand why, but changing to `reduce-overhead` which is less aggressive worked.

## Performance benchmark experiments

Assess the performance impact of (GPU, compile?, tensor float32).

```bash

gpu_types=("gpu_l4" "gpu_a100" "gpu_h100" "gpu_h200")
train_steps=("train_step_nocompile" "train_step")
tf32_flag=("use-tf32-matmul" "no-use-tf32-matmul")

for gpu_type in "${gpu_types[@]}"
do
    for train_step in "${train_steps[@]}"
    do
        for use_tf32_matmul in "${use_tf32_matmuls[@]}"
        do
            if [[ $gpu_type == "gpu_l4" ]]; then
                slots_per_gpu="8"
            else
                slots_per_gpu="12"
            fi
            name="${gpu_type}_${train_step}_${use_tf32_matmul}"
            bsub -J $name -n $slots_per_gpu \
                -gpu \"num=1\" -q $gpu_type -o ${name}.log python \
                src/LatentEvolution/latent.py \
                --training.train-step $train_step \
                --training.${tf32_flag} \
                --training.epochs 5000 \
                --training.batch-size 256
        done
    done
done
```
