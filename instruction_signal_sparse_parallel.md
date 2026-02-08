# Parallel Mode Addendum — Sparse Exploration

This addendum applies when running in **parallel mode** (GNN_LLM_parallel.py). Follow all rules from the base instruction file, with these modifications.

## Batch Processing

- You receive **4 results per batch** and must propose **4 mutations**
- Each slot has its own config file, metrics log, and activity image
- Write **4 separate** `## Iter N:` log entries (one per slot/iteration)
- Each iteration gets its own Node id in the UCB tree

## Config Files

- Edit all 4 config files listed in the prompt: `{name}_00.yaml` through `{name}_03.yaml`
- Each config's `dataset` field is pre-set to route data to separate directories — **DO NOT change the `dataset` field**
- **DO NOT change `simulation:` parameters** — this is a fixed-regime exploration
- Modify `training:` parameters, `claude:` where allowed, **and GNN code** (see Step 5.2 in base instructions)

## Locked Parameters

The following parameters are **fixed across all slots** — DO NOT modify them:

| Parameter | Locked Value |
| --------- | ------------ |
| `n_frames` | 100000 |
| `data_augmentation_loop` | 50 |
| `n_epochs` | 2 |

## Slot Assignments — Locked n_neurons

Each slot is locked to a specific `n_neurons` value. **DO NOT change `n_neurons`** — only vary other training parameters within each slot.

| Slot | n_neurons | Exploration focus |
| ---- | --------- | ----------------- |
| 0 | **200** | Explore training params at small network scale |
| 1 | **400** | Explore training params at medium network scale |
| 2 | **600** | Explore training params at large network scale |
| 3 | **1000** | Explore training params at very large network scale |

## Parallel UCB Strategy

When selecting parents for 4 simultaneous mutations, **diversify** your choices across training parameters (lr_W, coeff_W_L1, batch_size, etc.) while keeping each slot's `n_neurons` fixed. Each slot maintains its own UCB subtree.

You may use exploit/explore/principle-test strategies within each slot independently.

**When conn_R2 plateaus across config sweeps**: apply a **code-modification** (see Step 5.2 in base instructions). Code changes apply to ALL 4 slots simultaneously (shared codebase). Use the 4 slots to test different config parameters around the same code change — one code change, four different parameter variations per batch.

## Start Call (first batch, no results yet)

When the prompt says `PARALLEL START`:
- Read the base config to understand the starting training parameters
- Set each slot's `n_neurons` to its locked value (200, 400, 600, 1000)
- Set `n_frames=100000`, `data_augmentation_loop=50`, `n_epochs=2` in all 4 slots
- Vary other training parameters (e.g., `coeff_W_L1`, `learning_rate_W_start`) across the 4 slots — L1 is the primary lever for sparse recovery
- All 4 slots share the same simulation parameters (DO NOT change them)
- Write the planned initial variations to the working memory file

## Logging Format

Same as base instructions, but you write 4 entries per batch:

```
## Iter N: [converged/partial/failed]
Node: id=N, parent=P
Mode/Strategy: [strategy]
Config: seed=S, lr_W=X, lr=Y, lr_emb=Z, coeff_W_L1=W, coeff_edge_diff=D, n_epochs_init=I, first_coeff_L1=F, batch_size=B, recurrent=[T/F], time_step=T
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

- At block boundaries, simulation stays the same (fixed-regime exploration)
- Training parameters can differ across the 4 slots
- Choose a new **parameter subspace** to explore in the next block

## Failed Slots

If a slot is marked `[FAILED]` in the prompt:
- Write a brief `## Iter N: failed` entry noting the failure
- Still propose a mutation for that slot's config in the next batch
- Do not draw conclusions from a single failure (may be stochastic)
