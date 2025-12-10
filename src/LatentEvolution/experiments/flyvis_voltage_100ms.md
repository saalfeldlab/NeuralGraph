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

See [#103](https://github.com/saalfeldlab/NeuralGraph/pull/103) for some images.

### t=1 results

The total variance vs unexplained variance scatter is changed (optical flow):

- cells types like T2 that have high raw variance get _worse_ with the connectome
  constraint.
- some improvements for Mi4 cell types that are at (0.6, 0.6) -> (0.6, 0.4) or so.

The latent roll out blows up at ~1500 iterations in the baseline, but interestingly,
remains stable with the connectome augmentation.

### t=5 results

Latent roll out starts to diverge even in the 100 step window.
Learning the t=5 update works, but we do not learn the t = 1, 2, 3, 4 updates.

Adding the augmentation perhaps makes things better here. Interestingly, the
activity roll out remains stable over 100 steps. Just not the latent roll out.

Next, optimize the augmentation weight as well as the number of neurons to sample
for the augmentation.

## Sweep weights

```bash
for n in 10 100; do
    for x in 0.01 0.1; do
        bsub -J ${x}_${n} -q gpu_a100 -gpu "num=1" -n 1 -o t5_${x}_${n}.log \
            python src/LatentEvolution/latent.py test_aug latent_5step.yaml \
            --training.unconnected-to-zero.num-neurons $n \
            --training.unconnected-to-zero.loss-coeff $x
```
