# Test fitting with 100ms data

Use voltage data that is available every 100ms (5 time steps) and see if we can
properly model the 1-step connectome-constrained dynamics.

## Baseline

Run 1 time step & 5 time step with/without the connectome constraint applied.

```bash
bsub -J t1 -q gpu_a100 -gpu "num=1" -n 1 -o t1.log \
    python src/LatentEvolution/latent.py test_aug latent_1step.yaml
bsub -J t1_aug -q gpu_a100 -gpu "num=1" -n 1 -o t1_aug.log \
    python src/LatentEvolution/latent.py test_aug latent_1step.yaml \
    --training.unconnected-to-zero.num-neurons 100
bsub -J t5 -q gpu_a100 -gpu "num=1" -n 1 -o t5.log \
    python src/LatentEvolution/latent.py test_aug latent_5step.yaml
bsub -J t5_aug -q gpu_a100 -gpu "num=1" -n 1 -o t5_aug.log \
    python src/LatentEvolution/latent.py test_aug latent_5step.yaml \
    --training.unconnected-to-zero.num-neurons 100
```
