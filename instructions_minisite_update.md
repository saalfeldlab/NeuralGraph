# instructions: minisite update

## overview

the minisite is a quarto project rendered locally (`quarto render`). source files are `.qmd` in the project root; output goes to `docs/` (gitignored). interactive html panels are in `assets/` (git-tracked).

## data sources

for each active exploration loop, read the working memory file to understand what changed since the last update:

| exploration | memory file | analysis log |
|---|---|---|
| landscape (main) | `signal_landscape_Claude_memory.md` | `signal_landscape_Claude_analysis.md` |
| low-rank case | `signal_low_rank_Claude_memory.md` | — |
| sparse case | `signal_sparse_Claude_memory.md` | — |

check the **regime comparison table** (top of memory) for new/updated blocks.
check the **established principles** section for new principles.
check the **previous block summary** and **current block** sections for the latest results.

## step-by-step update procedure

### 1. identify delta

compare the memory files against the current qmd content to find:
- new blocks (new rows in regime table)
- updated blocks (changed convergence rates, best conn values)
- new principles
- new case study iterations

### 2. update qmd files

update in this order:

1. **`results.qmd`** — main results page
   - performance table (block rows with convergence rates, best conn, degeneracy)
   - eff_rank table
   - degeneracy table
   - n_frames lever table
   - block insight callouts
   - iteration/block counts in header

2. **`index.qmd`** — front page
   - results summary paragraph (iteration count, block count, principle count, key findings)
   - case study summaries
   - exploration gallery description

3. **`epistemic-analysis.qmd`** — epistemic analysis page
   - status callout (iteration/block/event/principle counts)
   - outcomes section
   - add new block epistemic events section

4. **`case-low-rank.qmd`** — low-rank case study (if changed)
   - subtitle iteration/seed counts
   - per-seed best configs table
   - block progression table
   - ucb tree tabs (copy images from `log/Claude_exploration/instruction_signal_low_rank_parallel/exploration_tree/`)
   - seed difficulty spectrum
   - established principles

5. **`case-sparse.qmd`** — sparse case study (if changed)
   - subtitle iteration/code mod counts
   - code modification sections
   - summary table
   - ucb tree tabs (copy images from `log/Claude_exploration/instruction_signal_sparse_parallel/exploration_tree/`)

### 3. update plot scripts

these python scripts generate interactive html panels in `assets/`:

1. **`plot_epistemic_interactive.py`** — generates 3 html + 2 png files
   - add new block to `blocks` list: `(start_iter, end_iter, 'B{n}: label', 'regime', 'eff_rank')`
   - add events for the new block to `events` list: `(iter, 'Mode', 'Significance')`
   - add within-block edges to `edges` list
   - add cross-block edges (from previous block to new block)
   - update sankey title (iteration/block counts)
   - update x-axis range if needed

2. **`plot_landscape_partitioned.py`** — generates partitioned landscape html
   - add new block with `add_block(block_num, regime, n_frames, center_x=eff_rank, conns=[...], start_iter, n_neurons, gain, fill, extra, mutation)`
   - `conns` is sorted descending list of conn_R2 values from the block's iterations
   - `center_x` is the eff_rank value (used as x-axis position)

3. **`plot_landscape_interactive.py`** — generates landscape quadrant interactive html
   - same `add_block()` format as partitioned

4. **`plot_landscape_quadrant.py`** — generates static landscape png
   - uses `add_iters(eff_rank, [conn_values])` format

### 4. run plot scripts

```bash
python3 plot_epistemic_interactive.py
python3 plot_landscape_partitioned.py
python3 plot_landscape_interactive.py
python3 plot_landscape_quadrant.py
```

output files:
- `assets/epistemic_timeline_interactive.html`
- `assets/epistemic_streamgraph_interactive.html`
- `assets/epistemic_sankey_interactive.html`
- `assets/epistemic_timeline.png`
- `assets/epistemic_streamgraph.png`
- `assets/landscape_partitioned_interactive.html`
- `assets/landscape_quadrant_interactive.html`
- `assets/landscape_quadrant.png`

### 5. update epistemic markdown files

update header counts in:
- `signal_landscape_Claude_epistemic_analysis.md` — reasoning mode counts, metrics table, summary
- `signal_landscape_Claude_epistemic_detailed.md` — header counts
- `signal_landscape_Claude_epistemic_edges.md` — edge counts

### 6. copy exploration images (if needed)

for case studies with new ucb tree snapshots:
```bash
# low-rank
cp log/Claude_exploration/instruction_signal_low_rank_parallel/exploration_tree/ucb_tree_iter_*.png \
   docs/log/Claude_exploration/instruction_signal_low_rank_parallel/exploration_tree/

# sparse
cp log/Claude_exploration/instruction_signal_sparse_parallel/exploration_tree/ucb_tree_iter_*.png \
   docs/log/Claude_exploration/instruction_signal_sparse_parallel/exploration_tree/
```

### 7. render locally

```bash
quarto render
```

do NOT push `docs/` to git (it is gitignored).

## file structure reference

```
*.qmd              — quarto source files (git-tracked)
assets/            — interactive html panels and images (git-tracked)
docs/              — rendered output (gitignored, render locally)
plot_*.py          — visualization scripts (git-tracked)
signal_*_memory.md — llm working memory (gitignored)
signal_*_analysis.md — llm analysis logs (gitignored)
signal_*_epistemic_*.md — epistemic analysis data (gitignored)
```

## key conventions

- the exploration uses iteration numbers globally across all blocks
- blocks are numbered sequentially (1, 2, ..., 29, 30, ...)
- each block has 12 iterations (3 batches of 4 parallel slots)
- some blocks have gaps in iteration numbers (e.g., block 24 fill=90% at iters 277-288)
- plot scripts use their own iteration numbering that may not exactly match the analysis log
- when adding events to plot_epistemic_interactive.py, place them within the block's iter range
- epistemic events per block: ~4-6 events, capturing the key reasoning modes observed
- edges per block: ~3-5 within-block + 1 cross-block edge to the next block
