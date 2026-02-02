# Parallel Mode Addendum

This addendum applies when running in **parallel mode** (GNN_LLM_parallel.py). Follow all rules from the base instruction file, with these modifications.

## Batch Processing

- You receive **4 results per batch** and must propose **4 mutations**
- Each slot has its own config file, metrics log, and activity image
- Write **4 separate** `## Iter N:` log entries (one per slot/iteration)
- Each iteration gets its own Node id in the UCB tree

## Config Files

- Edit all 4 config files listed in the prompt: `{name}_00.yaml` through `{name}_03.yaml`
- Each config's `dataset` field is pre-set to route data to separate directories — **DO NOT change the `dataset` field**
- Only modify `training:` and `simulation:` parameters (and `claude:` where allowed)

## Parallel UCB Strategy

When selecting parents for 4 simultaneous mutations, **diversify** your choices:

| Slot | Role | Description |
| ---- | ---- | ----------- |
| 0 | **exploit** | Highest UCB node, conservative mutation |
| 1 | **exploit** | 2nd highest UCB node, or same parent different param |
| 2 | **explore** | Under-visited node, or new parameter dimension |
| 3 | **principle-test** | Randomly pick one Established Principle from `memory.md` and design an experiment that tests or challenges it (see below) |

You may deviate from this split based on context (e.g., all exploit if early in block, all boundary-probe if everything converges).

### Slot 3: Principle Testing

At each batch, slot 3 should be used to **validate or challenge** one of the Established Principles listed in the working memory (`{config}_memory.md`):

1. Read the "Established Principles" section in memory.md
2. **Randomly select one principle** (rotate through them across batches — do not repeat the same one consecutively)
3. Design a config that specifically tests this principle:
   - If the principle says "X works when Y", test it under a different condition
   - If the principle says "Z always fails", try to make Z succeed
   - If the principle gives a range, test at the boundary
4. In the log entry, write: `Mode/Strategy: principle-test`
5. In the Mutation line, include: `Testing principle: "[quoted principle text]"`
6. After results, update the principle's evidence level in memory.md:
   - Confirmed → keep in Established Principles
   - Contradicted → move to Open Questions with note

If there are no Established Principles yet (early in the experiment), use slot 3 as a **boundary-probe** instead.

## Start Call (first batch, no results yet)

When the prompt says `PARALLEL START`:
- Read the base config to understand the simulation regime
- Create 4 diverse initial training parameter variations
- Suggested spread: vary `learning_rate_W_start` across the range (e.g. 1E-3, 2E-3, 5E-3, 1E-2)
- All 4 slots share the same simulation parameters
- Write the planned initial variations to the working memory file

## Logging Format

Same as base instructions, but you write 4 entries per batch:

```
## Iter N: [converged/partial/failed]
Node: id=N, parent=P
Mode/Strategy: [strategy]
Config: lr_W=X, lr=Y, lr_emb=Z, coeff_W_L1=W, batch_size=B, low_rank_factorization=[T/F], low_rank=R, n_frames=NF, recurrent=[T/F], time_step=T
Metrics: test_R2=A, test_pearson=B, connectivity_R2=C, cluster_accuracy=D, final_loss=E
Activity: eff_rank=R, spectral_radius=S, [brief description]
Embedding: [visual observation when n_types>1]
Mutation: [param]: [old] -> [new]
Parent rule: [one line]
Observation: [one line]
Next: parent=P
```

**CRITICAL**: The `Mutation:` line is parsed by the UCB tree builder. Always include the exact parameter change (e.g., `Mutation: lr_W: 2E-3 -> 5E-3`). For principle-test slots, append the principle being tested (e.g., `Mutation: lr_W: 2E-3 -> 5E-2. Testing principle: "lr_W > 1E-2 fails for low_rank"`).

**CRITICAL**: The `Next: parent=P` line selects the parent for the **next batch's** mutations. `P` must refer to a node from a **previous** batch or the current batch — but NEVER set `Next: parent=P` where P is `id+1` (the next slot in the same batch). This would make a node its own parent and create a circular reference. For example, if the current batch is iterations 37-40, iteration 38 must NOT write `Next: parent=39` because that would make node 39 its own parent.

Write all 4 entries before editing the 4 config files for the next batch.

## Block Boundaries

- At block boundaries, all 4 configs must share the same `simulation:` parameters
- Training parameters can differ across the 4 slots
- Block end is detected when any iteration in the batch hits `n_iter_block`

## Failed Slots

If a slot is marked `[FAILED]` in the prompt:
- Write a brief `## Iter N: failed` entry noting the failure
- Still propose a mutation for that slot's config in the next batch
- Do not draw conclusions from a single failure (may be stochastic)
