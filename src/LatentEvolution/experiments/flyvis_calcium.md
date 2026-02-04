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

Results:

- jobs failed with OOM. We should increase slots to 2.
- even at 1024d the latent roll out fails.

## generate tu20 baselines for aligned and staggered

An initial test for 30 epochs that just substituted voltage -> calcium with the
existing architecture "just worked" - roll out below the constant baseline, and
reasonable MSE that's competitive with the linear interpolation MSE.

So let's establish a baseline by running the existing method for 100 epochs.

```bash

bsub -J tu20_ca_aln -n 8 -W 8:00 -gpu "num=1" -q gpu_a100 -o tu20_ca_aln.log \
  python src/LatentEvolution/latent.py tu20_ca_baseline \
  calcium_align_20step.yaml --training.epochs 100

bsub -J tu20_ca_stag -n 8 -W 8:00 -gpu "num=1" -q gpu_a100 -o tu20_ca_stag.log \
  python src/LatentEvolution/latent_stag_interp.py tu20_ca_baseline \
  calcium_stag_20step.yaml --training.epochs 100
```
