# Experiments on flyvis data using latent evolution model

## Baseline experiment

We start with fly_N9_62_1 & model voltage at the simulated 20ms time step resolution. Note that
we are not adding in the stimulus.

### sweep batch size

```bash

for bs in 32 128 512 2048; do \
    bsub -J "batch${bs}" -n 12 -gpu "num=1" -q gpu_a100 -o batch${bs}.log python \
        src/LatentEvolution/latent.py \
        --training.batch_size $bs \
        --training.epochs 5000
    done
```

### sweep learning rate

```bash

for lr in 0.001 0.0001 0.00001 0.000001 ; do \
    bsub -J "lr${lr}" -n 12 -gpu "num=1" -q gpu_a100 -o lr${lr}.log python \
        src/LatentEvolution/latent.py \
        --training.batch_size 128 \
        --training.epochs 5000 \
        --training.learning_rate $lr
    done
```

TODO:

- [ ] add stimulus
- [ ] make diagnostic plots
- [ ] are we doing better than predicting a constant
- [ ] sweep parameters
