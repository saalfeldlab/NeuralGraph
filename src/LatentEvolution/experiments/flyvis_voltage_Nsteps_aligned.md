# Test fitting with 100ms data

Use voltage data that is available every 100ms (5 time steps) and see if we can
properly model the 1-step connectome-constrained dynamics.

Our goal is to model data that is acquired like this:

```
  time_aligned mode (tu=5)
  ============================
  All neurons observed simultaneously at regular intervals

  Time:     0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
           ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
  Neuron 1 │ X│  │  │  │  │ X│  │  │  │  │ X│  │  │  │  │ X│  │  │  │  │ X│
  Neuron 2 │ X│  │  │  │  │ X│  │  │  │  │ X│  │  │  │  │ X│  │  │  │  │ X│
  Neuron 3 │ X│  │  │  │  │ X│  │  │  │  │ X│  │  │  │  │ X│  │  │  │  │ X│
  Neuron 4 │ X│  │  │  │  │ X│  │  │  │  │ X│  │  │  │  │ X│  │  │  │  │ X│
  Neuron 5 │ X│  │  │  │  │ X│  │  │  │  │ X│  │  │  │  │ X│  │  │  │  │ X│
           └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
           ↑              ↑              ↑              ↑              ↑
         t=0            t=5           t=10           t=15           t=20
```

We use `time_units` or `tu` to refer to the interval between two activity recordings.
We are assuming that neurons are all measured instantaneously at each time. Of course,
this is not realistic. See experiments in `flyvis_voltage_Nsteps_staggered.md` that
describe a more realistic acquisition scenario.

To simplify, we begin with training a latent model to predict t -> t+5, but we allow
ourselves access to all the data points, t=0, 1, 2, ... etc. We first want to see if
the evolver can actually learn the correct t->t+1 update when only provided with
`(x(t), x(t+5))` in the loss function for each batch. Of course, because the network
has access to the intermediate points `x(t+1), ..., x(t+4)` the encoder and decoder
can build a good representation of these intermediate states that the evolver can
then properly arrange in time and find the right one step evolution.

Throughout we assume that the stimulus is provided at each time step. See some
experiments at the end that show that this is absolutely critical for us to learn
the right 1 step neural activity evolver in latent space.

The summary as of 1/27/2026 is that we can achieve this goal of learning a good
model for the t->t+1 evolution with `time_units=5` and even `time_units=20`.

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
bsub -J aligned -q gpu_a100 -gpu "num=1" -n 8 -o acq_aligned.log \
    python src/LatentEvolution/latent.py test_acq latent_20step.yaml \
    training.acquisition-mode:time-aligned-mode

# staggered_random: each neuron at different phase
bsub -J stag -q gpu_a100 -gpu "num=1" -n 8 -o acq_stag.log \
    python src/LatentEvolution/latent.py test_acq latent_20step.yaml \
    training.acquisition-mode:staggered-random-mode \
    --training.acquisition-mode.seed 42
```

We see really nice results - the time-aligned data works! We see low MSE at intervening points
and we see a long stable roll out. Let's make this the new default since this accurately
models data availability.

## Sweep ems

I am not sure if we really need ems=5 now. Let's try a few different values. Lower ems values
would be better since the compiler won't need to unroll large loops.

```bash
for ems in 1 2 3 5; do \
    bsub -J ems${ems} -q gpu_a100 -gpu "num=1" -n 8 -o ems${ems}.log \
        python src/LatentEvolution/latent.py ems_sweep latent_20step.yaml \
        --training.evolve-multiple-steps $ems
```

## Derivative experiment

Prior to changing the evolver (tanh activation, initialized to 0) and using
pretraining for reconstruction, we observed an instability in training in the
`tu20_seed_sweep_20260115_f6144bd` experiment. We want to revisit and see if
adding TV norm regularization can rescue that phenotype.

Understand which feature contributes to the stability. We changed many things at
the same time to get to a working point.

```bash
for zero_init in zero-init no-zero-init; do \
    for activation in Tanh ReLU; do \
        for warmup in 0 10; do \
        name="z${zero_init}_${activation}_w${warmup}"
        bsub -J $name -q gpu_a100 -gpu "num=1" -n 8 -o ${name}.log \
            python src/LatentEvolution/latent.py test_stability latent_20step.yaml \
            --evolver_params.${zero_init} \
            --evolver_params.activation $activation \
            --training.reconstruction-warmup-epochs $warmup \
            --training.seed 35235
        done
    done
done
```

Results:

- it turns out the evolver initialization is the key change. This removes the blow up
  and allows the system to learn the right update rule.
- warmup and activation don't seem to make a significant difference once the evolver
  is correctly initialized.
- warmup does two nice things
  - lower mse at intervening time steps
  - training after epoch 1 is already stable in terms of rollout

Understand if TV norm can bring stability to the training without pretraining for
reconstruction or the other features we added.

```bash

for tv in 0.0 0.00001 0.0001 0.001; do \
    bsub -J tv${tv} -q gpu_a100 -gpu "num=1" -n 8 -o tv${tv}.log \
        python src/LatentEvolution/latent.py tv_sweep latent_20step.yaml \
        --evolver_params.no-zero-init \
        --evolver_params.activation ReLU \
        --training.reconstruction-warmup-epochs 0 \
        --evolver_params.tv-reg-loss $tv \
        --training.seed 97651
done
```

Results:

- tv norm at 1e-3 successfully mitigates the artifact
- the mse at t->t+1 is closer to a constant than from the previous experiment. So
  while tv norm does help the training converge to a sensible evolution model, it
  does harm the t->t+1 mse.

## Stimulus downsampling

We want to avoid depending on the details of the stimulus provided since in general
it won't be known with such granularity. As a first step, we only provide the
stimulus every `tu` steps. At time step `n <= t < n + tu` we linearly interpolate between
the stimulus at time `n` and the one at time `n+tu`.

```bash
for mode in TIME_UNITS_INTERPOLATE TIME_UNITS_CONSTANT NONE; do \
    bsub -J stim_${mode} -q gpu_a100 -gpu "num=1" -n 8 -o stim_${mode}.log \
        python src/LatentEvolution/latent.py stim_freq_sweep latent_20step.yaml \
        --training.stimulus-frequency $mode
done
```

The results suggest that in the current setup we are critically reliant on the stimulus
being provided at every time step. When it is not, we see a blow up in the MSE when we
roll out past the training horizon. Even within the training horizon the error does not
fall below the linear interpolation baseline. And we are unable to learn the right
rule even at the loss time points 0, `tu`, `2tu`, ...

We will revisit this experiment later on.

## Attempt tu=50 again

Given the improvements we made to training stability, we will revisit the tu=50
experiment. Since tu=20 worked with ems=5 we will reason that perhaps showing the
model a rollout of 100 steps suffices. So maybe we can get away with ems=2 here?

```bash

bsub -J 50x2 -W 8:00 -q gpu_a100 -gpu "num=1" -n 2 -o 50x2.log \
    python src/LatentEvolution/latent.py 50x_redo latent_20step.yaml \
    --training.time-units 50 \
    --training.evolve-multiple-steps 2 \
    --training.epochs 100
```

`ems3` results are promising. The roll out does blow up but remains stable a little
outside the training window. Perhaps we should just try ems5 and see what happens.
The time aligned mse plot shows poor performance in the time window [0, tu]. Perhaps
the tv_norm loss will help here. But first, let's try ems5.

```bash
bsub -J 50x5 -W 8:00 -q gpu_a100 -gpu "num=1" -n 2 -o 50x5.log \
    python src/LatentEvolution/latent.py 50x_ems5 latent_20step.yaml \
    --training.time-units 50

bsub -J 50x3_tv -W 4:00 -q gpu_a100 -gpu "num=1" -n 2 -o 50x3_tv.log \
    python src/LatentEvolution/latent.py 50x3_tv latent_20step.yaml \
    --training.time-units 50 \
    --training.epochs 50 \
    --training.evolve-multiple-steps 3 \
    --evolver_params.tv-reg-loss 0.01

bsub -J 50x3_tv2 -W 4:00 -q gpu_a100 -gpu "num=1" -n 2 -o 50x3_tv2.log \
    python src/LatentEvolution/latent.py 50x3_tv latent_20step.yaml \
    --training.time-units 50 \
    --training.epochs 50 \
    --training.evolve-multiple-steps 3 \
    --evolver_params.tv-reg-loss 0.001
```
