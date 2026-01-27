#!/usr/bin/env python3
"""
Visualize epistemic reasoning timeline from signal_landscape_Claude experiment.
67 iterations across 9 blocks.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from collections import Counter
from matplotlib.gridspec import GridSpec
from matplotlib.path import Path
import matplotlib.patches as mpatches

# Define color scheme for reasoning modes
COLORS = {
    'Induction': '#2ecc71',        # Green
    'Abduction': '#9b59b6',        # Purple
    'Deduction': '#3498db',        # Blue
    'Falsification': '#e74c3c',    # Red
    'Analogy': '#f39c12',          # Orange
    'Boundary': '#1abc9c',         # Teal
    'Meta-reasoning': '#e91e63',   # Pink
    'Regime': '#795548',           # Brown
    'Uncertainty': '#607d8b',      # Gray
    'Causal Chain': '#00bcd4',     # Cyan
    'Predictive': '#8bc34a',       # Light green
    'Constraint': '#ff5722',       # Deep orange
}

DEFINITIONS = {
    'Induction': 'observations -> pattern',
    'Abduction': 'observation -> hypothesis',
    'Deduction': 'hypothesis -> prediction',
    'Falsification': 'prediction failed -> refine',
    'Analogy': 'cross-regime transfer',
    'Boundary': 'limit-finding',
    'Meta-reasoning': 'strategy adaptation',
    'Regime': 'phase identification',
    'Uncertainty': 'stochasticity awareness',
    'Causal Chain': 'multi-step causation',
    'Predictive': 'quantitative modeling',
    'Constraint': 'parameter relationships',
}

# Events from signal_landscape_Claude_epistemic_detailed.md
# 60 iterations, 8 blocks
events = [
    # Block 1: Chaotic baseline (iters 1-8)
    (4, 'Induction', 'Medium'),      # lr_W 5x range
    (4, 'Deduction', 'High'),        # lr_W boundary prediction
    (4, 'Boundary', 'Medium'),       # lr_W approaching
    (5, 'Boundary', 'Medium'),       # L1 probe
    (5, 'Meta-reasoning', 'Medium'), # dimension switch
    (6, 'Boundary', 'Medium'),       # L1 tolerance
    (7, 'Deduction', 'Medium'),      # L1 extreme test
    (7, 'Boundary', 'Medium'),       # L1 boundary
    (8, 'Induction', 'High'),        # Block summary: 100x L1 range
    (8, 'Deduction', 'Medium'),      # L1=1E-3 test
    (8, 'Falsification', 'High'),    # L1 boundary not found

    # Block 2: Low-rank (iters 9-16)
    (9, 'Abduction', 'High'),        # eff_rank hypothesis
    (9, 'Regime', 'High'),           # Low_rank regime discovered
    (10, 'Deduction', 'High'),       # Factorization test
    (10, 'Falsification', 'High'),   # Factorization hurts
    (10, 'Abduction', 'Medium'),     # Over-constrains model
    (11, 'Deduction', 'Medium'),     # Higher lr_W
    (11, 'Induction', 'Medium'),     # lr_W improvement
    (12, 'Deduction', 'High'),       # lr_W=8E-3 test
    (12, 'Falsification', 'High'),   # lr_W boundary found
    (12, 'Boundary', 'High'),        # lr_W upper limit
    (12, 'Constraint', 'High'),      # low_rank lr_W constraint
    (14, 'Meta-reasoning', 'Medium'), # Multi-param violation
    (14, 'Falsification', 'Medium'), # L1+aug didn't help
    (15, 'Deduction', 'High'),       # lr breakthrough
    (15, 'Causal Chain', 'High'),    # lr mechanism
    (15, 'Abduction', 'High'),       # MLP lr critical
    (16, 'Deduction', 'Medium'),     # lr=1E-3 confirmed
    (16, 'Induction', 'High'),       # Block summary

    # Block 3: Dale's law (iters 17-24)
    (17, 'Analogy', 'High'),         # Block 2 transfer
    (17, 'Deduction', 'High'),       # Settings transfer test
    (17, 'Abduction', 'High'),       # Dale reduces eff_rank
    (17, 'Regime', 'High'),          # Dale regime
    (17, 'Falsification', 'High'),   # Dale is easy
    (18, 'Deduction', 'High'),       # lr_W=8E-3 works
    (19, 'Deduction', 'High'),       # lr_W=2E-2 works
    (19, 'Boundary', 'Medium'),      # lr_W probe
    (19, 'Induction', 'Medium'),     # Dale robust
    (20, 'Deduction', 'High'),       # lr_W=5E-2 works
    (20, 'Boundary', 'Medium'),      # lr_W extreme
    (21, 'Boundary', 'Medium'),      # L1=1E-3 test
    (22, 'Deduction', 'High'),       # lr_W=1E-1 works
    (22, 'Boundary', 'Medium'),      # More extreme
    (23, 'Boundary', 'Medium'),      # lr_W=2E-1
    (24, 'Deduction', 'High'),       # lr_W=5E-1 works
    (24, 'Boundary', 'High'),        # 100x range found
    (24, 'Induction', 'High'),       # Block summary

    # Block 4: Heterogeneous n_types=2 (iters 25-32)
    (25, 'Analogy', 'High'),         # Blocks 2-3 transfer
    (25, 'Deduction', 'High'),       # Transfer test
    (27, 'Deduction', 'Medium'),     # lr_W=1E-2 works
    (27, 'Boundary', 'Medium'),      # lr_W probe
    (27, 'Induction', 'Medium'),     # Heterogeneous robust
    (28, 'Deduction', 'High'),       # lr_W=5E-2 test
    (28, 'Falsification', 'High'),   # Embedding boundary
    (28, 'Abduction', 'High'),       # lr_W starves embedding
    (28, 'Boundary', 'High'),        # lr_W boundary
    (28, 'Causal Chain', 'High'),    # Dual-objective conflict
    (28, 'Constraint', 'High'),      # n_types lr_W constraint
    (29, 'Deduction', 'High'),       # lr_W=2E-2 restores
    (29, 'Boundary', 'Medium'),      # Boundary narrowed
    (30, 'Abduction', 'Medium'),     # L1 hurts embedding
    (30, 'Falsification', 'Medium'), # L1 effect
    (31, 'Deduction', 'High'),       # lr_emb compensation
    (31, 'Abduction', 'Medium'),     # lr_emb compensates
    (31, 'Causal Chain', 'Medium'),  # Compensation mechanism
    (31, 'Analogy', 'Medium'),       # lr_emb insight
    (32, 'Deduction', 'High'),       # L1=5E-4 with lr_emb
    (32, 'Induction', 'High'),       # Block summary

    # Block 5: Noise (iters 33-40)
    (33, 'Analogy', 'High'),         # Block 3-4 transfer
    (33, 'Deduction', 'High'),       # Noise test
    (33, 'Abduction', 'High'),       # Noise adds variance
    (33, 'Regime', 'High'),          # Noise regime
    (33, 'Causal Chain', 'High'),    # Noise mechanism
    (33, 'Predictive', 'High'),      # eff_rank prediction
    (33, 'Induction', 'High'),       # Noise increases eff_rank
    (34, 'Deduction', 'High'),       # lr_W=1E-2 works
    (34, 'Induction', 'Medium'),     # Noise tolerates lr_W
    (36, 'Deduction', 'High'),       # lr_W=5E-2 works
    (36, 'Boundary', 'Medium'),      # lr_W probe
    (38, 'Deduction', 'High'),       # lr_W=1E-1 works
    (38, 'Boundary', 'Medium'),      # More extreme
    (39, 'Boundary', 'Medium'),      # L1=5E-4
    (39, 'Induction', 'Medium'),     # 7/7 converged
    (40, 'Deduction', 'High'),       # L1=1E-3 works
    (40, 'Boundary', 'High'),        # No boundary found
    (40, 'Induction', 'High'),       # Block summary: super easy
    (40, 'Regime', 'High'),          # Super easy regime

    # Block 6: Low_rank + n_types=2 (iters 41-48)
    (41, 'Analogy', 'Medium'),       # Block 2+4 transfer
    (41, 'Abduction', 'High'),       # Compound difficulty
    (41, 'Regime', 'High'),          # Compound regime
    (41, 'Predictive', 'Medium'),    # eff_rank prediction
    (42, 'Analogy', 'High'),         # lr insight transfer
    (43, 'Deduction', 'High'),       # lr_W=8E-3, lr=1E-3
    (43, 'Induction', 'Medium'),     # Both lr tuned
    (43, 'Analogy', 'High'),         # Combined solution
    (44, 'Deduction', 'High'),       # lr_W=1E-2 works
    (44, 'Boundary', 'Medium'),      # lr_W probe
    (44, 'Induction', 'Medium'),     # lr_W upper higher
    (45, 'Deduction', 'High'),       # lr_W=2E-2 test
    (45, 'Falsification', 'High'),   # lr_W boundary
    (45, 'Boundary', 'High'),        # lr_W upper found
    (45, 'Constraint', 'High'),      # Compound lr_W constraint
    (46, 'Deduction', 'Medium'),     # lr_emb probe
    (47, 'Falsification', 'High'),   # L1=1E-4 fails
    (47, 'Abduction', 'High'),       # L1 tolerance narrow
    (47, 'Boundary', 'High'),        # L1 boundary
    (47, 'Constraint', 'High'),      # L1 constraint
    (48, 'Falsification', 'High'),   # lr=5E-4 insufficient
    (48, 'Constraint', 'High'),      # lr requirement
    (48, 'Induction', 'High'),       # Block summary

    # Block 7: Sparse (iters 49-56)
    (49, 'Abduction', 'High'),       # Sparse collapse
    (49, 'Regime', 'High'),          # Sparse regime
    (49, 'Causal Chain', 'High'),    # Collapse mechanism
    (49, 'Predictive', 'High'),      # eff_rank<8 prediction
    (49, 'Uncertainty', 'Medium'),   # eff_rank varies
    (50, 'Deduction', 'High'),       # lr=1E-3 test
    (50, 'Falsification', 'High'),   # Training can't fix
    (50, 'Abduction', 'High'),       # Problem is data
    (50, 'Analogy', 'Medium'),       # Block 2 insight
    (51, 'Deduction', 'High'),       # lr_W boost test
    (51, 'Falsification', 'High'),   # Still fails
    (51, 'Meta-reasoning', 'High'),  # Recognized futility
    (51, 'Induction', 'Medium'),     # 3 failures
    (52, 'Deduction', 'High'),       # Factorization test
    (52, 'Falsification', 'High'),   # Made worse
    (52, 'Abduction', 'Medium'),     # Constraints can't fix
    (53, 'Meta-reasoning', 'High'),  # Strategy exhaustion
    (54, 'Falsification', 'High'),   # L1=0 worse
    (54, 'Abduction', 'Medium'),     # Overfits to noise
    (54, 'Uncertainty', 'Medium'),   # eff_rank dropped
    (55, 'Falsification', 'High'),   # L1 restore fails
    (56, 'Deduction', 'High'),       # Scale-up test
    (56, 'Falsification', 'High'),   # Still fails
    (56, 'Induction', 'High'),       # Block summary: unrecoverable

    # Block 8: Sparse + Noise (iters 57-64)
    (57, 'Analogy', 'High'),         # Block 5 noise insight
    (57, 'Deduction', 'High'),       # Noise rescue test
    (57, 'Causal Chain', 'High'),    # Rescue mechanism
    (57, 'Regime', 'High'),          # Rescued regime
    (57, 'Predictive', 'High'),      # eff_rank=92
    (57, 'Induction', 'High'),       # Noise rescues sparse
    (57, 'Analogy', 'Medium'),       # lr insight applied
    (58, 'Falsification', 'Medium'), # lr_W plateau
    (59, 'Meta-reasoning', 'Medium'), # Dimension switch
    (59, 'Boundary', 'Medium'),      # lr_W plateau
    (60, 'Falsification', 'Medium'), # L1 plateau
    (60, 'Boundary', 'Medium'),      # L1 plateau
    (60, 'Induction', 'Medium'),     # Plateau at R2~0.20
    (60, 'Uncertainty', 'Medium'),   # May be fundamental limit
    (61, 'Induction', 'High'),       # 5 consecutive partial plateau
    (62, 'Deduction', 'High'),       # Scale-up test
    (62, 'Falsification', 'High'),   # Scale-up failed
    (63, 'Induction', 'High'),       # 7 consecutive partial
    (64, 'Induction', 'High'),       # 8 consecutive partial - definitive

    # Block 9: Intermediate sparsity ff=0.5 (iters 65-67)
    (65, 'Analogy', 'High'),         # Block 8 failure → ff=0.5 hypothesis
    (65, 'Regime', 'High'),          # ff=0.5 regime discovered
    (65, 'Deduction', 'High'),       # eff_rank hypothesis validated
    (65, 'Induction', 'High'),       # eff_rank=26 vs ff=0.2 eff_rank=6
    (66, 'Deduction', 'High'),       # lr_W=1E-2 test
    (66, 'Falsification', 'High'),   # lr_W hurt test_pearson
    (67, 'Deduction', 'High'),       # L1=0 test
    (67, 'Induction', 'Medium'),     # test_pearson recovered but W stuck
]

# Causal edges from signal_landscape_Claude_epistemic_edges.md
edges = [
    # Block 1
    (4, 'Deduction', 8, 'Induction', 'leads_to'),
    (4, 'Boundary', 5, 'Meta-reasoning', 'triggers'),
    (7, 'Deduction', 8, 'Falsification', 'leads_to'),
    (8, 'Induction', 9, 'Analogy', 'triggers'),

    # Block 2
    (9, 'Abduction', 10, 'Deduction', 'triggers'),
    (10, 'Deduction', 10, 'Falsification', 'leads_to'),
    (10, 'Falsification', 11, 'Induction', 'refines'),
    (11, 'Deduction', 12, 'Deduction', 'leads_to'),
    (12, 'Deduction', 12, 'Falsification', 'leads_to'),
    (12, 'Falsification', 12, 'Boundary', 'leads_to'),
    (12, 'Boundary', 15, 'Deduction', 'triggers'),
    (15, 'Deduction', 15, 'Causal Chain', 'leads_to'),
    (15, 'Causal Chain', 16, 'Induction', 'leads_to'),
    (16, 'Induction', 17, 'Analogy', 'triggers'),

    # Block 3
    (17, 'Analogy', 17, 'Deduction', 'triggers'),
    (17, 'Deduction', 17, 'Falsification', 'leads_to'),
    (17, 'Abduction', 19, 'Deduction', 'triggers'),
    (19, 'Deduction', 24, 'Boundary', 'leads_to'),
    (24, 'Boundary', 24, 'Induction', 'leads_to'),
    (24, 'Induction', 25, 'Analogy', 'triggers'),

    # Block 4
    (25, 'Analogy', 25, 'Deduction', 'triggers'),
    (27, 'Deduction', 28, 'Deduction', 'leads_to'),
    (28, 'Deduction', 28, 'Falsification', 'leads_to'),
    (28, 'Falsification', 28, 'Abduction', 'triggers'),
    (28, 'Abduction', 29, 'Deduction', 'triggers'),
    (28, 'Falsification', 28, 'Boundary', 'leads_to'),
    (30, 'Abduction', 31, 'Deduction', 'triggers'),
    (31, 'Deduction', 31, 'Causal Chain', 'leads_to'),
    (32, 'Induction', 33, 'Analogy', 'triggers'),

    # Block 5
    (33, 'Analogy', 33, 'Deduction', 'triggers'),
    (33, 'Abduction', 33, 'Causal Chain', 'leads_to'),
    (33, 'Regime', 34, 'Deduction', 'triggers'),
    (36, 'Deduction', 40, 'Boundary', 'leads_to'),
    (40, 'Boundary', 40, 'Induction', 'leads_to'),
    (40, 'Induction', 41, 'Analogy', 'triggers'),

    # Block 6
    (41, 'Analogy', 42, 'Analogy', 'triggers'),
    (42, 'Analogy', 43, 'Deduction', 'triggers'),
    (43, 'Deduction', 44, 'Deduction', 'leads_to'),
    (45, 'Deduction', 45, 'Falsification', 'leads_to'),
    (45, 'Falsification', 45, 'Constraint', 'leads_to'),
    (47, 'Falsification', 47, 'Constraint', 'leads_to'),
    (48, 'Induction', 49, 'Analogy', 'triggers'),

    # Block 7
    (49, 'Abduction', 49, 'Causal Chain', 'leads_to'),
    (49, 'Causal Chain', 50, 'Deduction', 'triggers'),
    (50, 'Deduction', 50, 'Falsification', 'leads_to'),
    (50, 'Falsification', 51, 'Meta-reasoning', 'triggers'),
    (51, 'Meta-reasoning', 52, 'Deduction', 'leads_to'),
    (52, 'Deduction', 52, 'Falsification', 'leads_to'),
    (54, 'Falsification', 56, 'Induction', 'refines'),
    (56, 'Induction', 57, 'Analogy', 'triggers'),

    # Block 8
    (57, 'Analogy', 57, 'Deduction', 'triggers'),
    (57, 'Deduction', 57, 'Causal Chain', 'leads_to'),
    (57, 'Regime', 58, 'Deduction', 'triggers'),
    (58, 'Deduction', 58, 'Falsification', 'leads_to'),
    (58, 'Falsification', 59, 'Meta-reasoning', 'triggers'),
    (60, 'Falsification', 60, 'Induction', 'refines'),
    (61, 'Induction', 62, 'Deduction', 'leads_to'),
    (62, 'Deduction', 62, 'Falsification', 'leads_to'),
    (62, 'Falsification', 63, 'Induction', 'refines'),
    (63, 'Induction', 64, 'Induction', 'leads_to'),
    (64, 'Induction', 65, 'Analogy', 'triggers'),

    # Block 9
    (65, 'Regime', 65, 'Deduction', 'triggers'),
    (65, 'Deduction', 66, 'Deduction', 'leads_to'),
    (66, 'Deduction', 66, 'Falsification', 'leads_to'),
    (66, 'Falsification', 67, 'Deduction', 'triggers'),

    # Cross-block
    (33, 'Causal Chain', 57, 'Deduction', 'triggers'),
    (64, 'Induction', 65, 'Analogy', 'triggers'),
]

# Block boundaries
blocks = [
    (1, 8, 'Block 1', {'regime': 'chaotic', 'eff_rank': '34-35'}),
    (9, 16, 'Block 2', {'regime': 'low_rank', 'eff_rank': '11'}),
    (17, 24, 'Block 3', {'regime': 'Dale', 'eff_rank': '23'}),
    (25, 32, 'Block 4', {'regime': 'n_types=2', 'eff_rank': '32-34'}),
    (33, 40, 'Block 5', {'regime': 'noise', 'eff_rank': '83'}),
    (41, 48, 'Block 6', {'regime': 'compound', 'eff_rank': '9-11'}),
    (49, 56, 'Block 7', {'regime': 'sparse', 'eff_rank': '4-6'}),
    (57, 64, 'Block 8', {'regime': 'sparse+noise', 'eff_rank': '92'}),
    (65, 67, 'Block 9', {'regime': 'ff=0.5', 'eff_rank': '26'}),
]


def create_2x2_panels():
    """Create 3-row panel figure for signal_landscape_Claude.

    Layout:
    - Row 1: Scatterplot (left), Legend (right)
    - Row 2: Streamgraph (left), Summary (right)
    - Row 3: Sankey diagram (full width, from Sankey.png)
    """

    fig = plt.figure(figsize=(18, 18), constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig, width_ratios=[2.5, 0.8], height_ratios=[1, 1, 1.2])

    modes = [
        'Induction', 'Boundary', 'Abduction', 'Deduction', 'Falsification',
        'Analogy', 'Meta-reasoning', 'Regime', 'Uncertainty',
        'Constraint', 'Predictive', 'Causal Chain',
    ]
    mode_to_y = {mode: i for i, mode in enumerate(modes)}

    # Common x-axis limits for alignment
    x_min, x_max = 0, 72

    # ============ TOP LEFT: Scatterplot ============
    ax1 = fig.add_subplot(gs[0, 0])

    for block_idx, (start, end, label, info) in enumerate(blocks):
        color = '#f0f0f0' if block_idx % 2 == 1 else 'white'
        ax1.axvspan(start - 0.5, end + 0.5, alpha=1.0, color=color)

    for iteration, mode, significance in events:
        if mode not in mode_to_y:
            continue
        y = mode_to_y[mode]
        color = COLORS.get(mode, '#333333')
        size = {'High': 150, 'Medium': 80, 'Low': 40}.get(significance, 80)
        ax1.scatter(iteration, y, c=color, s=size, alpha=0.9,
                   edgecolors='none', zorder=3)

    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(-0.5, len(modes) - 0.5)
    ax1.set_yticks([])
    ax1.set_xticks([])
    ax1.grid(True, axis='x', alpha=0.2, linestyle='--')
    # Remove box
    for spine in ax1.spines.values():
        spine.set_visible(False)

    # ============ TOP RIGHT: Legend Panel ============
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('Legend', fontsize=16, loc='left')
    ax2.axis('off')

    y_pos = 0.95
    y_step = 0.075

    for mode in modes:
        color = COLORS.get(mode, '#333333')
        definition = DEFINITIONS.get(mode, '')
        ax2.scatter([0.08], [y_pos], c=color, s=200, edgecolors='none')
        ax2.text(0.16, y_pos, f"{mode}: {definition}", fontsize=13, va='center')
        y_pos -= y_step

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    # ============ BOTTOM LEFT: Streamgraph ============
    ax3 = fig.add_subplot(gs[1, 0])

    max_iter = max(e[0] for e in events)
    bin_width = 4
    n_bins = (max_iter // bin_width) + 2  # Extra bin for proper coverage
    # Center bins on iteration ranges: bin 0 covers iters 1-4, centered at 2.5
    iterations = np.arange(bin_width / 2, (n_bins + 0.5) * bin_width, bin_width)

    data = np.zeros((len(modes), n_bins))
    for iteration, mode, significance in events:
        if mode in modes:
            mode_idx = modes.index(mode)
            # Bin index: iter 1-4 -> bin 0, iter 5-8 -> bin 1, etc.
            bin_idx = (iteration - 1) // bin_width
            if 0 <= bin_idx < n_bins:
                weight = {'High': 2.0, 'Medium': 1.0, 'Low': 0.5}.get(significance, 1.0)
                data[mode_idx, bin_idx] += weight

    sigma = 1.0
    data_smooth = np.array([gaussian_filter1d(row, sigma) for row in data])

    n_layers = len(modes)
    baseline = np.zeros(n_bins)
    for i in range(n_layers):
        baseline -= (n_layers - i - 0.5) * data_smooth[i]
    baseline /= n_layers

    y_stack = np.vstack([baseline, baseline + np.cumsum(data_smooth, axis=0)])

    for i, mode in enumerate(modes):
        color = COLORS.get(mode, '#333333')
        ax3.fill_between(iterations, y_stack[i], y_stack[i + 1],
                        color=color, alpha=0.8,
                        edgecolor='white', linewidth=0.3)

    # Align x-axis with scatterplot
    ax3.set_xlim(x_min, x_max)
    ax3.set_yticks([])
    ax3.set_facecolor('#f8f8f8')
    # Remove box except bottom x-axis line
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_visible(False)

    # ============ BOTTOM RIGHT: Summary Panel ============
    ax_summary = fig.add_subplot(gs[1, 1])
    ax_summary.set_title('Summary', fontsize=16, loc='left')
    ax_summary.axis('off')

    mode_counts = Counter(e[1] for e in events)
    total_events = len(events)
    total_edges = len(edges)

    summary_text = f"""signal_landscape_Claude

{total_events} reasoning instances
67 iterations, 9 blocks
12 principles discovered

Deduction validation: 69%
Transfer success: 77%

Mode counts:
  Induction: {mode_counts.get('Induction', 0)}
  Abduction: {mode_counts.get('Abduction', 0)}
  Deduction: {mode_counts.get('Deduction', 0)}
  Falsification: {mode_counts.get('Falsification', 0)}
  Analogy: {mode_counts.get('Analogy', 0)}
  Boundary: {mode_counts.get('Boundary', 0)}

{total_edges} causal edges
5 key causal chains"""

    ax_summary.text(0.05, 0.95, summary_text, fontsize=13, va='top', ha='left',
                   linespacing=1.4)
    ax_summary.set_xlim(0, 1)
    ax_summary.set_ylim(0, 1)

    # ============ ROW 3: Sankey Diagram (first column) ============
    ax_sankey = fig.add_subplot(gs[2, 0])
    ax_sankey.axis('off')

    # Load and display Sankey.png
    import os
    sankey_path = os.path.join(os.path.dirname(__file__) or '.', 'Sankey.png')
    if os.path.exists(sankey_path):
        sankey_img = plt.imread(sankey_path)
        ax_sankey.imshow(sankey_img, aspect='equal')
    else:
        ax_sankey.text(0.5, 0.5, 'Sankey.png not found', fontsize=14,
                      ha='center', va='center', transform=ax_sankey.transAxes)

    # ============ ROW 3: Caption (second column) ============
    ax_caption = fig.add_subplot(gs[2, 1])
    ax_caption.set_title('Claude summary', fontsize=16, loc='left')
    ax_caption.axis('off')

    caption_text = ("Epistemic flow of LLM-guided scientific discovery. "
                   "Deduction acts as the central hub—nearly all reasoning modes "
                   "converge through it before branching outward. The dominant "
                   "Deduction→Falsification pathway demonstrates genuine hypothesis "
                   "testing rather than mere pattern matching: predictions are "
                   "generated, tested, and refined through iterative loops back to "
                   "induction and abduction. Late-stage emergence of boundary-finding "
                   "and meta-reasoning indicate progressive deepening.")

    # Use textwrap for proper formatting
    import textwrap
    wrapped_text = textwrap.fill(caption_text, width=35)
    ax_caption.text(0.0, 0.88, wrapped_text, fontsize=11, va='top', ha='left',
                   linespacing=1.4)
    ax_caption.set_xlim(0, 1)
    ax_caption.set_ylim(0, 1)

    plt.savefig('signal_landscape_Claude_epistemic.png', dpi=150, bbox_inches='tight',
                pad_inches=0.3)
    print("Saved: signal_landscape_Claude_epistemic.png")
    plt.close()


def create_sankey_html():
    """Create interactive Sankey diagram using Plotly with legend box."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Plotly not installed")
        return

    modes = list(COLORS.keys())
    mode_to_idx = {mode: i for i, mode in enumerate(modes)}

    transitions = Counter()
    for from_iter, from_mode, to_iter, to_mode, edge_type in edges:
        transitions[(from_mode, to_mode)] += 1

    source = []
    target = []
    value = []
    link_colors = []

    for (from_mode, to_mode), count in transitions.items():
        if from_mode in mode_to_idx and to_mode in mode_to_idx:
            source.append(mode_to_idx[from_mode])
            target.append(mode_to_idx[to_mode])
            value.append(count)
            color = COLORS.get(from_mode, '#888888')
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            link_colors.append(f'rgba({r},{g},{b},0.4)')

    node_colors = [COLORS.get(mode, '#888888') for mode in modes]
    # Use only mode names on nodes
    node_labels = modes

    fig = go.Figure(data=[go.Sankey(
        textfont=dict(size=11),
        node=dict(
            pad=20,
            thickness=25,
            line=dict(color='black', width=0.5),
            label=node_labels,
            color=node_colors,
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors,
        ),
        arrangement='snap',
    )])

    fig.update_layout(
        title_text="Epistemic Reasoning Flow: signal_landscape_Claude",
        font_size=12,
        width=1400,
        height=800,
        margin=dict(l=50, r=200, t=50, b=50),
    )

    fig.write_html('signal_landscape_Claude_epistemic_sankey.html')
    print("Saved: signal_landscape_Claude_epistemic_sankey.html")


if __name__ == '__main__':
    print("Creating signal_landscape_Claude epistemic visualizations...")
    print(f"Total events: {len(events)}")
    print(f"Total edges: {len(edges)}")

    counts = Counter(e[1] for e in events)
    for mode, count in sorted(counts.items()):
        print(f"  {mode}: {count}")

    print("\n--- 2x2 Panels ---")
    create_2x2_panels()

    print("\n--- Sankey HTML ---")
    create_sankey_html()

    print("\nDone!")
