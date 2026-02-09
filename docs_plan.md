# NeuralGraph Documentation Minisite Plan

## Goal
Create a Quarto-rendered documentation site for the LLM-in-the-loop GNN experiments, similar to MPM-pytorch's `docs/` structure.

---

## 1. Project Structure

```
NeuralGraph/
├── _quarto.yml                    # Quarto config (NEW)
├── docs/                          # Generated output (NEW)
│   ├── index.html
│   ├── *.html
│   ├── assets/                    # Images, videos
│   └── site_libs/                 # Quarto dependencies
│
├── index.qmd                      # Landing page (NEW)
├── architecture.qmd               # System architecture (NEW)
├── experiment-loop.qmd            # GNN_LLM.py walkthrough (NEW)
├── epistemic-analysis.qmd         # Reasoning metrics (NEW)
├── results.qmd                    # Signal landscape results (NEW)
├── gnn-model.qmd                  # GNN architecture from paper (NEW)
│
├── papers/                        # Source material (EXISTING)
│   ├── Graph_neural_networks_*.tex
│   ├── epistemic_metrics_appendix.tex
│   └── epistemic_timeline_figure.tex
├── signal_landscape_Claude_*.md   # Results (EXISTING)
├── instruction_signal_landscape.md # Instructions (EXISTING)
├── LLM_loop_introduction.md       # System intro (EXISTING)
└── GNN_LLM.py                     # Main script (EXISTING)
```

---

## 2. Quarto Configuration (`_quarto.yml`)

```yaml
project:
  type: website
  output-dir: docs
  render:
    - "*.qmd"

website:
  title: "NeuralGraph: LLM-in-the-Loop"
  navbar:
    left:
      - href: index.qmd
        text: Home
      - text: System
        menu:
          - architecture.qmd
          - experiment-loop.qmd
      - text: Analysis
        menu:
          - epistemic-analysis.qmd
          - results.qmd
      - href: gnn-model.qmd
        text: GNN Model
    right:
      - icon: github
        href: https://github.com/allierc/NeuralGraph

format:
  html:
    theme:
      light: flatly
      dark: darkly
    toc: true
    toc-depth: 3
    smooth-scroll: true
    code-fold: true
    code-tools: true
```

---

## 3. Page Content Plan

### 3.1 `index.qmd` - Landing Page

**Content:**
- Hero section: "Autonomous GNN Training via LLM-Guided Exploration"
- 30-second summary of the approach
- Key results teaser (best R², discovered principles)
- Video/GIF of epistemic timeline or simulation
- Quick links to sections

**Source Material:**
- Abstract from `papers/Graph_neural_networks_*.tex`
- `signal_landscape_Claude_epistemic.png`

---

### 3.2 `architecture.qmd` - System Architecture

**Content:**
- 3-way file exchange diagram (EXPERIMENT ↔ LLM ↔ MEMORY)
- File flow table (what files Claude reads/writes)
- UCB tree exploration visualization
- Config structure explanation

**Source Material:**
- `LLM_loop_introduction.md` (architecture diagram section)
- `instruction_signal_landscape.md` (file structure section)

---

### 3.3 `experiment-loop.qmd` - Experiment Loop Walkthrough

**Content:**
- Step-by-step iteration workflow
- Code walkthrough of `GNN_LLM.py` key sections
- Strategic decision rules table (19 strategies)
- Block boundary handling
- Auto-repair mechanism

**Source Material:**
- `GNN_LLM.py` (annotated code blocks)
- `instruction_signal_landscape.md` (workflow sections)

---

### 3.4 `epistemic-analysis.qmd` - Reasoning Analysis

**Content:**
- Definition of 12 epistemic modes (from appendix)
- Metrics computation (HTR, DA, TSR, etc.)
- Sankey diagram embed (interactive HTML)
- Timeline visualization
- Discovered principles table with confidence scores

**Source Material:**
- `papers/epistemic_metrics_appendix.tex`
- `signal_landscape_Claude_epistemic_analysis.md`
- `signal_landscape_Claude_epistemic_sankey.html` (iframe embed)
- `signal_landscape_Claude_epistemic.png`

---

### 3.5 `results.qmd` - Signal Landscape Results

**Content:**
- Block-by-block summary table
- Key findings:
  - eff_rank determines difficulty
  - Chaotic regime is "easy mode"
  - Low-rank requires lr boost
  - Sparse networks fundamentally unrecoverable
- Scaling experiments (n=100 → 300 → 500)
- Boundary exploration examples

**Source Material:**
- `signal_landscape_Claude_analysis.md`
- `signal_landscape_Claude_memory.md`

---

### 3.6 `gnn-model.qmd` - GNN Model Architecture

**Content:**
- Network dynamics equation: `du/dt = lin_phi(u,a) + W @ lin_edge(u,a)`
- Message passing architecture
- Connectivity matrix learning
- Effective rank definition
- Spectral radius properties

**Source Material:**
- `papers/Graph_neural_networks_*.tex` (Methods section)
- `instruction_signal_landscape.md` (Background section)

---

## 4. Assets to Create/Copy

| Asset | Source | Destination |
|-------|--------|-------------|
| Epistemic timeline | `signal_landscape_Claude_epistemic.png` | `docs/assets/` |
| Sankey diagram | `signal_landscape_Claude_epistemic_sankey.html` | `docs/assets/` |
| Activity videos | `log/Claude_exploration/*/activity/` | `docs/assets/videos/` |
| UCB tree plots | `log/Claude_exploration/*/tree/` | `docs/assets/trees/` |
| Architecture diagram | CREATE from `LLM_loop_introduction.md` | `docs/assets/` |

---

## 5. Implementation Steps

### Phase 1: Setup (Day 1)
1. [ ] Create `_quarto.yml` with navbar structure
2. [ ] Create `index.qmd` skeleton with frontmatter
3. [ ] Test `quarto render` locally
4. [ ] Setup GitHub Pages deployment

### Phase 2: Core Pages (Day 2-3)
5. [ ] Write `architecture.qmd` from `LLM_loop_introduction.md`
6. [ ] Write `experiment-loop.qmd` with `GNN_LLM.py` code blocks
7. [ ] Write `gnn-model.qmd` from paper methods section

### Phase 3: Analysis Pages (Day 4-5)
8. [ ] Write `epistemic-analysis.qmd` with Sankey embed
9. [ ] Write `results.qmd` with block summaries
10. [ ] Create/embed visualizations

### Phase 4: Polish (Day 6)
11. [ ] Add dark mode testing
12. [ ] Mobile responsiveness check
13. [ ] Search functionality verification
14. [ ] Final deploy to GitHub Pages

---

## 6. Design Guidelines (Flat Design)

Following MPM-pytorch's approach:
- **Theme**: Flatly (light) / Darkly (dark)
- **Typography**: Clean sans-serif headers
- **Tables**: Striped, hover-enabled
- **Code**: Foldable blocks with copy button
- **Math**: MathJax for equations
- **Callouts**: Use for important notes/warnings
- **Tab sets**: For multi-view content (e.g., different strategies)

---

## 7. GitHub Pages Deployment

Add to repository:
```yaml
# .github/workflows/quarto-publish.yml
on:
  push:
    branches: main
jobs:
  build-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: quarto-dev/quarto-actions/setup@v2
      - run: quarto render
      - uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs
```

---

## 8. Next Steps

1. **Approve this plan** or suggest modifications
2. **Start with `_quarto.yml`** and `index.qmd`
3. **Iteratively build pages** with content from existing materials
