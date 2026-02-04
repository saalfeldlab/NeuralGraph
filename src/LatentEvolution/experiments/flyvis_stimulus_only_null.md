# Stimulus-only null model

## motivation

the EED model uses both past neural activity (via the latent state) and stimulus to predict future activity. how much of its prediction accuracy comes from stimulus alone? a stimulus-only baseline answers this: if it performs well, the latent dynamics may just be memorizing stimulus-activity mappings rather than learning temporal dynamics.

## architecture

```
stim(t-tu+1), ..., stim(t)  [tu frames, each 1736 dims]
        |
  stimulus_encoder (frozen, pretrained as autoencoder)
        |
  z_s(t-tu+1), ..., z_s(t)  [tu frames, each 64 dims]
        |
  flatten  ->  [tu * 64 = 1280]
        |
  predictor MLP (MLPWithSkips, 2 hidden layers, 512 units)
        |
  x_hat(t)  [13741 neurons]
```

no encoder, decoder, or latent space. no autoregressive rollout during training -- each prediction is independent given the stimulus window. the stimulus encoder is shared with the EED model (same architecture, pretrained as autoencoder, then frozen).

## training

- loss: MSE between predicted and true activity at time t
- stimulus context window: tu=20 frames preceding t
- no multi-step rollout, no reconstruction loss, no augmentation loss
- script: `stimulus_only.py`, config: `stimulus_only_20step.yaml`

## how to evaluate

compare the 2000-step rollout MSE curves (tensorboard `CrossVal/.*/multi_start_2000step_stimulus_only_rollout_mses_by_time`) against the EED model and the constant baseline. if stimulus-only matches EED, the latent dynamics are not adding value. if it matches the constant baseline, stimulus alone is uninformative.

# Experiments

## baseline model

Run the baseline model without any optimizations. We have 18M parameters in the model, which is
comparable to the EED model, or maybe

```bash
bsub -J stim_null -n 8 -gpu "num=1" -q gpu_a100 -o stim_null.log python \
  src/LatentEvolution/stimulus_only.py stim_null stimulus_only_20step.yaml
```
