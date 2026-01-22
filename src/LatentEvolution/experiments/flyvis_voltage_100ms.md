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
            python src/LatentEvolution/latent.py aug5 latent_5step.yaml \
            --training.unconnected-to-zero.num-neurons $n \
            --training.unconnected-to-zero.loss-coeff $x
        done
    done
```

## Intermediate loss steps

Test applying evolution loss at intermediate steps (in addition to the final step).

```bash
# baseline (no intermediate loss, final step only)
bsub -J ils_none -q gpu_a100 -gpu "num=1" -n 1 -o ils_none.log \
    python src/LatentEvolution/latent.py test_ils latent_5step.yaml

# sweep intermediate steps
for t in 1 2 3 4; do
    bsub -J ils_t${t} -q gpu_a100 -gpu "num=1" -n 1 -o ils_t${t}.log \
        python src/LatentEvolution/latent.py test_ils latent_5step.yaml \
        --training.intermediate-loss-steps $t
done
```

The results are very interesting, because when we have an intermediate loss at
t=3 or t=4 we see stable roll outs. In fact, supplying data at dt=3 and dt=5 is
even better than the dt=1 baseline.

## Apply loss at multiples of dt

Inspired by the previous experiment, let's see what happens if we apply a loss at
both dt and 2dt.

```bash
# evolve to t+5 and t+10
bsub -J 2x -q gpu_a100 -gpu "num=1" -n 1 -o 2x.log \
    python src/LatentEvolution/latent.py multiple_steps latent_5step.yaml \
    --training.evolve-multiple-steps 2
# evolve to t+5 and t+10 and t+15
bsub -J 3x -q gpu_a100 -gpu "num=1" -n 1 -o 3x.log \
    python src/LatentEvolution/latent.py multiple_steps latent_5step.yaml \
    --training.evolve-multiple-steps 3
bsub -J 5x -q gpu_a100 -gpu "num=1" -n 1 -o 5x.log \
    python src/LatentEvolution/latent.py multiple_steps latent_5step.yaml \
    --training.evolve-multiple-steps 5
```

This experiment shows interesting results. When we apply loss at t+5 and t+10, we
see a dip in the losses at t+15, t+20 etc. Suggesting some kind of generalization.
With ems=3 (t+5, t+10, t+15) we see solid generalization to both intermediate time
points, like t+2, as well as beyond the training horizon t+15. But we see a blow up
in MSE beyond the 100 step window. With ems=5, the roll out on the validation data
with noise is just as good as t->t+1, which is very encouraging. However, we
observe overfitting, i.e., the model at intermediate training points shows
better generalization on the optical flow cross-validation dataset than the final
model. So, let's add more metrics and try to come up with a stopping criterion.

## Early stopping investigation

Run a series of experiments to investigate stability of long roll out, on
both the validation data (training dataset but on an unseen time range), and
on an entirely different dataset like optical flow. These experiments will
have some logging so we can decide.

```bash
bsub -J chk -q gpu_a100 -gpu "num=1" -n 2 -o chk.log \
    python src/LatentEvolution/latent.py chk latent_1step.yaml \
    --training.save-checkpoint-every-n-epochs 5

bsub -J 5x5 -q gpu_a100 -gpu "num=1" -n 2 -o 5x5.log \
    python src/LatentEvolution/latent.py multiple_steps latent_5step.yaml \
    --training.evolve-multiple-steps 5 \
    --training.save-checkpoint-every-n-epochs 5

bsub -J 10x5 -q gpu_a100 -gpu "num=1" -n 2 -o 10x5.log \
    python src/LatentEvolution/latent.py multiple_steps latent_5step.yaml \
    --training.time-units 10 \
    --training.evolve-multiple-steps 5 \
    --training.save-checkpoint-every-n-epochs 5

bsub -J 20x5 -q gpu_a100 -gpu "num=1" -n 2 -o 20x5.log \
    python src/LatentEvolution/latent.py multiple_steps latent_5step.yaml \
    --training.time-units 20 \
    --training.evolve-multiple-steps 5 \
    --training.save-checkpoint-every-n-epochs 5
```

Analyze intermediate epochs to understand cross-validation dataset performance.

```bash

for epoch in $(seq 25 25 100); do \
    for run_dir in $(find runs/multiple_steps_20260112_c6bf0ef/ -name config.yaml | xargs dirname); do \
        name=$(basename ${run_dir})_${epoch}
        bsub -n 1 -o ${name}.log -J $name -q gpu_a100 -gpu "num=1" \
            python src/LatentEvolution/post_run_analyze.py $run_dir --epoch $epoch
    done
done
```

Currently we define a training & validation dataset by taking different time
segments of the DAVIS dataset. It turns out that this isn't good enough to
define a valid early stopping point. If we focus on the `time_units=20` results,
we find that the optical flow roll out is stable for > 1000 time steps at epoch
25, but becomes unstable after. The current validation dataset roll out remains
stable. The 500-step rollout is quite fast, so we could run longer roll outs to
see if there's a signal there.

When we apply a training loss with `ems=n`, each data point is seen n times more
often and it would make sense to reduce the number of epochs by `n`.

We see the best results for `time_units=20`, but the roll outs are worse for
other cases, like `time_units=10`. Perhaps this is because the 20-step subsampling
ends up seeing a bigger window of data and is fundamentally more stable as a
result.

## Try time-units=50

The zapbench data timescale is 1s and represents a 50x time subsampling. Let's
give that a shot.

```bash

bsub -J 50x1 -q gpu_a100 -gpu "num=1" -n 2 -o 50x1.log \
    python src/LatentEvolution/latent.py 50x_init latent_5step.yaml \
    --training.time-units 50 \
    --training.evolve-multiple-steps 1 \
    --training.epochs 100 \
    --training.save-checkpoint-every-n-epochs 5

bsub -J 50x2 -q gpu_a100 -gpu "num=1" -n 2 -o 50x2.log \
    python src/LatentEvolution/latent.py 50x_init latent_5step.yaml \
    --training.time-units 50 \
    --training.evolve-multiple-steps 2 \
    --training.epochs 100 \
    --training.save-checkpoint-every-n-epochs 5

bsub -J 50x5 -q gpu_a100 -gpu "num=1" -n 2 -o 50x5.log \
    python src/LatentEvolution/latent.py 50x_init latent_5step.yaml \
    --training.time-units 50 \
    --training.evolve-multiple-steps 5 \
    --training.ems-warmup-epochs 10 \
    --training.grad_clip_max_norm 0 \
    --training.epochs 50 \
    --training.save-checkpoint-every-n-epochs 5

```

These initial experiments all fail to capture the correct `0<t<50` dynamics. Let's
first focus on getting tu=20 right and then move up to tu=50.

## tu20 baseline experiment

Let's just reproduce the tu=20 experiment that we ran earlier with ems=5. The key
thing we need from this network is good MSE over the intervening steps `0<t<20`, since
we can then feed this to the GNN. At the same time we want to make sure that we are
able to roll it out beyond the training window so we can be sure we have some power
to generalize.

```bash

# Baseline config for tu=20
bsub -J 20x5 -n 1 -q gpu_a100 -gpu "num=1" -o 20x5.log \
    python src/LatentEvolution/latent.py 20x_base latent_20step.yaml

# How does ems=4 do?
bsub -J 20x4 -n 1 -q gpu_a100 -gpu "num=1" -o 20x4.log \
    python src/LatentEvolution/latent.py 20x_base latent_20step.yaml \
    --training.evolve-multiple-steps 4

# No gradient clipping. This is what we tested previously.
bsub -J 20x5noclip -n 1 -q gpu_a100 -gpu "num=1" -o 20x5noclip.log \
    python src/LatentEvolution/latent.py 20x_base latent_20step.yaml \
    --training.grad-clip-max-norm 0.0

```

Gradient clipping was hurting results. Let's turn it off. We now have a working
baseline with tu=20 - `20x_base_20260114_d5a2309/614b03`. There are things to
investigate in the future.

- First, there are multiple epochs where the rollout
  diverges and then we recover generalization during training. 100 epochs seems
  like a magical point and is most definitely cherry picked. We should run training
  for longer and understand this phenotype.

- Second, the other experiments show a phenotype where at t=20n, we have low
  MSE ~ 1e-2, but then the error blows up quickly to ~ 1.0
  and then back down to 1e-2 at time point t=20(n+1). Why does this happen and is
  this some artifact of the training (like gradient clipping which we turned off)?

## Test code changes for stability

We test two ideas:

- use tanh activation for evolver
- initialize final layer to zero so that evolver starts off being a constant
- pre-train the encoder/decoder in a warmup phase
  We hope these changes improve the stability of training.

```bash

bsub -J "recon_warmup_test" -n 1 -gpu "num=1" -q gpu_a100 -o recon_warmup_test.log python \
      src/LatentEvolution/latent.py tu20_recon_warmup_test latent_20step.yaml

bsub -J "recon_warmup_seed" -n 1 -gpu "num=1" -q gpu_a100 -o recon_warmup_seed.log python \
      src/LatentEvolution/latent.py tu20_recon_warmup_test latent_20step.yaml \
      --training.seed 35235
```

The changes work really well. Two nice things happen:

- good generalization from epoch 0
- stable roll outs starting from epoch 0 itself
  Over training epochs we observe that the MSE reduces progressively. We also see that
  different random seeds are now comparable and the training is stable.

## establish new baseline with youtube dataset

We've downloaded a much larger dataset with ~ 990K unique frames. Let's train on this and
make sure our results hold up.

```bash
bsub -J youtube -n 1 -q gpu_a100 -gpu "num=1" -o youtube.log \
    python src/LatentEvolution/latent.py tu20_youtube_baseline latent_20step.yaml
```

## Model acquisition

We have been training tu=20 and predicting x(t+20) starting from x(t). But, in order to do so
we actually use all the data points. So during training the model sees all x(t), x(t+1), ...
It is very likely that since the model is able to represent intermediate time points we can
learn the correct dynamics. But we won't ever have access to the intermediate time points.

```bash

# time_aligned: observations at 0, 20, 40, ... for all neurons
bsub -J test_acq -q gpu_a100 -gpu "num=1" -n 8 -o acq_aligned.log \
    python src/LatentEvolution/latent.py test_acq latent_20step.yaml \
    --training.acquisition-mode.mode time_aligned \
    --training.data-passes-per-epoch 20

# staggered_random: each neuron at different phase
bsub -J test_acq -q gpu_a100 -gpu "num=1" -n 8 -o acq_stag.log \
    python src/LatentEvolution/latent.py test_acq latent_20step.yaml \
    --training.acquisition-mode.mode staggered_random \
    --training.acquisition-mode.seed 42 \
    --training.data-passes-per-epoch 20
```
