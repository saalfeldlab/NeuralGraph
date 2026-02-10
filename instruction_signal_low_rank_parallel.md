# Parallel Mode Addendum — Low-Rank Gain × Rank Exploration

This addendum applies when running in **parallel mode** (GNN_LLM_parallel.py). Follow all rules from the base instruction file, with these modifications.

## Batch Processing

- You receive **4 results per batch** and must propose **4 mutations**
- Each slot has its own config file, metrics log, and activity image
- Write **4 separate** `## Iter N:` log entries (one per slot/iteration)
- Each iteration gets its own Node id in the UCB tree

## Config Files

- Edit all 4 config files listed in the prompt: `{name}_00.yaml` through `{name}_03.yaml`
- Each config's `dataset` field is pre-set to route data to separate directories — **DO NOT change the `dataset` field**
- **n_neurons=100 is LOCKED** — DO NOT change it
- You MAY change `simulation:` parameters: `connectivity_rank`, `params` (gain only), and `seed`
- You MAY change all `training:` parameters and `claude:` where allowed

## Locked Parameters

The following parameters are **fixed across all slots** — DO NOT modify them:

| Parameter | Locked Value |
| --------- | ------------ |
| `n_neurons` | 100 |
| `n_frames` | 10000 |
| `data_augmentation_loop` | 200 |
| `n_epochs` | 2 |

## Slot Assignments — Initial (gain, rank) Regions

Each slot starts at a different (gain, rank) corner to maximize coverage. **Slots are NOT locked** — the LLM may reassign (gain, rank) across slots as the exploration progresses.

| Slot | Initial gain | Initial rank | Region |
| ---- | ------------ | ------------ | ------ |
| 0 | **7** | **20** | Known baseline (exploit) |
| 1 | **4** | **20** | Low gain, same rank (explore) |
| 2 | **10** | **20** | High gain, same rank (explore) |
| 3 | **7** | **10** | Same gain, low rank (explore) |

## Parallel UCB Strategy

When selecting parents for 4 simultaneous mutations, **diversify** your choices across the gain × rank landscape. Each slot maintains its own UCB subtree.

Suggested first-block strategy:
- Slot 0: Validate baseline recipe at gain=7, rank=20 with different seeds
- Slot 1-3: Map the gain/rank landscape — test whether the gain=7 recipe transfers

You may use exploit/explore/principle-test strategies within each slot independently.

## Start Call (first batch, no results yet)

When the prompt says `PARALLEL START`:
- Read the base config to understand the starting training parameters
- Set each slot's gain and rank to its initial assignment (see table above)
- Set `n_neurons=100`, `n_frames=10000`, `data_augmentation_loop=200`, `n_epochs=2` in all 4 slots
- Use the known-good recipe (lr_W=3E-3, L1=1E-6) as starting point for all slots
- Write the planned initial variations to the working memory file

## Logging Format

Same as base instructions, but you write 4 entries per batch:

```
## Iter N: [converged/partial/failed]
Node: id=N, parent=P
Mode/Strategy: [strategy]
Config: gain=G, rank=R, seed=S, lr_W=X, lr=Y, lr_emb=Z, coeff_W_L1=W, coeff_edge_diff=D, n_epochs_init=I, first_coeff_L1=F, batch_size=B
Metrics: test_R2=A, test_pearson=B, connectivity_R2=C, cluster_accuracy=D, final_loss=E, kino_R2=F, kino_SSIM=G, kino_WD=H
Activity: eff_rank=R, spectral_radius=S, [brief description]
Mutation: [param]: [old] -> [new]
Parent rule: [one line]
Observation: [one line]
Next: parent=P
```

**CRITICAL**: The `Mutation:` line is parsed by the UCB tree builder. Always include the exact parameter change.

**CRITICAL**: The `Next: parent=P` line selects the parent for the **next batch's** mutations. `P` must refer to a node from a **previous** batch or the current batch — but NEVER set `Next: parent=P` where P is `id+1` (circular reference).

Write all 4 entries before editing the 4 config files for the next batch.

## Block Boundaries

- At block boundaries, choose a new **(gain, rank) region** to explore in the next block
- Training parameters can differ across the 4 slots
- Slots can be reassigned to different (gain, rank) values

## Failed Slots

If a slot is marked `[FAILED]` in the prompt:
- Write a brief `## Iter N: failed` entry noting the failure
- Still propose a mutation for that slot's config in the next batch
- Do not draw conclusions from a single failure (may be stochastic)
