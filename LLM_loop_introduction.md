# LLM-Guided Experiment/Understanding Loop

> The system couples three components: an **Experiment** module that generates, trains, and evaluates models; an **LLM** that analyzes results and proposes parameter modifications; and a **Memory** that accumulates established principles across exploration blocks.

## 1. System Architecture

```
                  activity.png
                  analysis.log
                  ucb_scores.txt
+--------------+        |         +-----------------+       +--------------+
|  EXPERIMENT  |  ------+----->   |       LLM       |  <->  |    MEMORY    |
+--------------+                  | instructions.md |       |  memory.md   |
       ^                          +-----------------+       +--------------+
       |                                 |
       |        config.yaml              |
       +---------------------------------+
                                         |
                                         v
                                  +--------------+
                                  |     LOG      |
                                  | analysis.md  |
                                  +--------------+
```

### File Exchange Summary

| Direction      | File                   | Purpose                       |
| -------------- | ---------------------- | ----------------------------- |
| Exp -> LLM     | `activity.png`         | Neural activity visualization |
| Exp -> LLM     | `analysis.log`         | Metrics (R², eff_rank, loss)  |
| Exp -> LLM     | `ucb_scores.txt`       | UCB tree exploration scores   |
| LLM -> Exp     | `config/{task}.yaml`   | Updated hyperparameters       |
| LLM <-> Memory | `{task}_memory.md`     | Working memory (read/write)   |
| LLM -> Log     | `{task}_analysis.md`   | Append-only experiment log    |
| LLM -> Log     | `{task}_reasoning.log` | Claude reasoning trace        |

---

## 2. GNN_LLM.py Algorithm

### Initialization (iteration 1)

- Copy base config to `{task}_Claude.yaml`
- Clear UCB scores, analysis log, memory file
- Initialize memory with template structure

### Main Loop (for each iteration)

```
FOR iteration = 1 to n_iterations:

    ┌─────────────────────────────────────────────────────────┐
    │  EXPERIMENT PHASE                                       │
    ├─────────────────────────────────────────────────────────┤
    │  1. Reload config (pick up LLM changes)                 │
    │  2. data_generate() → simulate neural activity          │
    │  3. data_train()    → train GNN on activity             │
    │  4. data_test()     → evaluate connectivity recovery    │
    │  5. data_plot()     → generate visualizations           │
    │  6. Write metrics to analysis.log                       │
    └─────────────────────────────────────────────────────────┘
                              │
                              v
    ┌─────────────────────────────────────────────────────────┐
    │  UCB COMPUTATION                                        │
    ├─────────────────────────────────────────────────────────┤
    │  7. compute_ucb_scores() → update exploration tree      │
    │  8. plot_ucb_tree()      → visualize search progress    │
    └─────────────────────────────────────────────────────────┘
                              │
                              v
    ┌─────────────────────────────────────────────────────────┐
    │  LLM PHASE (Claude CLI)                                 │
    ├─────────────────────────────────────────────────────────┤
    │  9. Read: instruction.md, memory.md, analysis.log,      │
    │           ucb_scores.txt, activity.png                  │
    │ 10. Analyze results, select parent node (UCB)           │
    │ 11. Choose strategy (exploit/explore/boundary/...)      │
    │ 12. Edit config.yaml (mutate ONE parameter)             │
    │ 13. Update memory.md (current block progress)           │
    │ 14. Append to analysis.md (full log)                    │
    └─────────────────────────────────────────────────────────┘
                              │
                              v
    ┌─────────────────────────────────────────────────────────┐
    │  BLOCK BOUNDARY (every n_iter_block iterations)         │
    ├─────────────────────────────────────────────────────────┤
    │ 15. Clear UCB scores (new exploration tree)             │
    │ 16. LLM: Edit instruction.md (add/modify rules)         │
    │ 17. LLM: Choose next simulation regime                  │
    │ 18. LLM: Update Knowledge Base in memory.md             │
    │ 19. Save memory snapshot                                │
    └─────────────────────────────────────────────────────────┘

END FOR
```

### Scoring

The UCB score is based on **connectivity_R²**: the ability of the GNN to recover the true connectivity matrix W from simulated neural activity data. Higher R² (closer to 1.0) indicates better recovery of the ground-truth synaptic weights.

---

## 3. Instruction File Structure (`instruction_{task}.md`)

### Goal Definition

- Define the scientific objective (e.g., map simulation-GNN training landscape)
- Specify success criteria (connectivity_R2 > 0.9)

### Block and Iteration Structure

- **Iteration**: Single experiment cycle (generate → train → test → LLM analysis)
- **Block**: Group of `n_iter_block` iterations exploring one simulation regime
- Training parameters can change within a block
- Simulation parameters can only change at block boundaries

### Iteration Workflow (Steps 1-5)

1. **Read Working Memory** - Recall established principles, block progress
2. **Analyze Results** - Parse metrics, classify outcome (converged/partial/failed)
3. **Write Outputs** - Log iteration in analysis.md and memory.md
4. **Parent Selection** - Use UCB scores to select exploration node
5. **Edit Config** - Mutate ONE parameter based on strategy

### Block Workflow (End of Block)

1. **Edit Instructions** - Add/modify parent selection rules
2. **Choose Next Regime** - Select untested simulation configuration
3. **Update Memory** - Summarize block, update Knowledge Base

---

## Appendix: Claude CLI Call

```python
# call Claude CLI for analysis
print("\033[93mClaude analysis...\033[0m")

claude_prompt = f"""Iteration {iteration}/{n_iterations}
Block info: block {block_number}, iteration {iter_in_block}/{n_iter_block} within block
{">>> BLOCK END <<<" if is_block_end else ""}

Instructions (follow all instructions): {instruction_path}
Working memory: {memory_path}
Full log (append only): {analysis_path}
Activity image: {activity_path}
Metrics log: {analysis_log_path}
UCB scores: {ucb_path}
Current config: {config_path}"""

claude_cmd = [
    'claude',
    '-p', claude_prompt,
    '--output-format', 'text',
    '--max-turns', '100',
    '--allowedTools',
    'Read', 'Edit'
]

# run with real-time output streaming
process = subprocess.Popen(
    claude_cmd,
    cwd=root_dir,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)
```
