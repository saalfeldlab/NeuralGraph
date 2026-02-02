# Parent Rules Adaptation Analysis

Track modifications to UCB parent selection rules from `{config}_analysis.md` files.

---

## Output Format

```markdown
# Parent Rules Analysis: {experiment_name}

**Experiment**: {description} | **Blocks**: N | **Date**: YYYY-MM-DD

## Rule Modifications

| Block | Action | Rule | Trigger | Rationale |
|-------|--------|------|---------|-----------|
| 1 | ADD | boundary-skip | branching<20% | "3+ partial results probing same boundary → accept and explore elsewhere" |
| 2 | NONE | - | - | branching rate healthy (27%) |
| 3 | MODIFY | switch-dimension | 4+ same-param mutations | lowered threshold from 5 to 4 |

## Observations

- **Total modifications**: N rules added, M modified, K removed
- **Key insight**: [what triggered most changes, what worked]
- **Pattern**: [when/why rules were adapted]
```

---

## How to Extract

1. Search for `INSTRUCTIONS EDITED:` entries in block summaries
2. Record: which block, what action (ADD/MODIFY/REMOVE), rule name, trigger condition
3. Note brief rationale from the log

---

**Reference**: UCB exploration framework in NeuralGraph instruction files.
