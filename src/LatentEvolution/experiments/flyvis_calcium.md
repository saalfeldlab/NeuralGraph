# Baseline experiment

Let's take the successful voltage 20ms model and just apply as is to flyvis calcium.

```bash
bsub -J calcium -n 1 -gpu "num=1" -q gpu_a100 -o calcium.log python \
  src/LatentEvolution/latent.py calcium0 \
  --training.column-to-model CALCIUM
```
