# Calcium activity prediction at 20ms

Let's take the successful voltage 20ms model and just apply as is to flyvis calcium at the same
20ms timescale.

## baseline

```bash
bsub -J calcium -n 1 -gpu "num=1" -q gpu_a100 -o calcium.log python \
  src/LatentEvolution/latent.py calcium0 \
  --training.column-to-model CALCIUM
```

Results:

- training loss for t->t+1 evolution is ~ 2e-4. But this is actually larger than the constant
  model loss, which is 0.5e-4 so about 4x smaller.
- The rollout mse plots show a cross-over point: the t->t+2 predictions are matched with the
  baseline constant model, and then improve t->t+3, t->t+4 and so on.
- the latent space rollout mse is stable at ~ 1e-3 for about 300 steps and then starts to blow up.
  the activity space rollout is less stable and more or less immediately starts to climb to large
  values.

```bash
bsub -J calcium -n 1 -gpu "num=1" -q gpu_a100 -o calcium.log python \
  src/LatentEvolution/latent.py calcium0 \
  --training.column-to-model CALCIUM
```

## increase latent space dim

In the flyvis voltage experiments on network size we saw that reducing the latent dim had the
effect of keeping the latent rollout stable while the activity rollout exploded. So let's try
to increase the latent dimension for calcium and see if that has an impact.

```bash
for ldim in 256 384 512 1024; do \
  bsub -J "ldim${ldim}" -n 1 -gpu "num=1" -q gpu_a100 -o ldim${ldim}.log python \
    src/LatentEvolution/latent.py calcium_latent \
    --training.column-to-model CALCIUM \
    --latent-dims $ldim
done
```
