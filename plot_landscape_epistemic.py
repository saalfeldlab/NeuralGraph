#!/usr/bin/env python3
"""
Visualize epistemic reasoning timeline from signal_landscape_Claude experiment.
107 iterations across 14 blocks.
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
# 107 iterations, 14 blocks
events = [
    # Block 1: Chaotic baseline (iters 1-8)
    (4, 'Induction', 'Medium'),
    (4, 'Deduction', 'High'),
    (4, 'Boundary', 'Medium'),
    (5, 'Boundary', 'Medium'),
    (5, 'Meta-reasoning', 'Medium'),
    (6, 'Boundary', 'Medium'),
    (7, 'Deduction', 'Medium'),
    (7, 'Boundary', 'Medium'),
    (8, 'Induction', 'High'),
    (8, 'Deduction', 'Medium'),
    (8, 'Falsification', 'High'),

    # Block 2: Low-rank (iters 9-16)
    (9, 'Abduction', 'High'),
    (9, 'Regime', 'High'),
    (10, 'Deduction', 'High'),
    (10, 'Falsification', 'High'),
    (10, 'Abduction', 'Medium'),
    (11, 'Deduction', 'Medium'),
    (11, 'Induction', 'Medium'),
    (12, 'Deduction', 'High'),
    (12, 'Falsification', 'High'),
    (12, 'Boundary', 'High'),
    (12, 'Constraint', 'High'),
    (14, 'Meta-reasoning', 'Medium'),
    (14, 'Falsification', 'Medium'),
    (15, 'Deduction', 'High'),
    (15, 'Causal Chain', 'High'),
    (15, 'Abduction', 'High'),
    (16, 'Deduction', 'Medium'),
    (16, 'Induction', 'High'),

    # Block 3: Dale's law (iters 17-24)
    (17, 'Analogy', 'High'),
    (17, 'Deduction', 'High'),
    (17, 'Abduction', 'High'),
    (17, 'Regime', 'High'),
    (17, 'Falsification', 'High'),
    (18, 'Deduction', 'High'),
    (19, 'Deduction', 'High'),
    (19, 'Boundary', 'Medium'),
    (19, 'Induction', 'Medium'),
    (20, 'Deduction', 'High'),
    (20, 'Boundary', 'Medium'),
    (21, 'Boundary', 'Medium'),
    (22, 'Deduction', 'High'),
    (22, 'Boundary', 'Medium'),
    (23, 'Boundary', 'Medium'),
    (24, 'Deduction', 'High'),
    (24, 'Boundary', 'High'),
    (24, 'Induction', 'High'),

    # Block 4: Heterogeneous n_types=2 (iters 25-32)
    (25, 'Analogy', 'High'),
    (25, 'Deduction', 'High'),
    (27, 'Deduction', 'Medium'),
    (27, 'Boundary', 'Medium'),
    (27, 'Induction', 'Medium'),
    (28, 'Deduction', 'High'),
    (28, 'Falsification', 'High'),
    (28, 'Abduction', 'High'),
    (28, 'Boundary', 'High'),
    (28, 'Causal Chain', 'High'),
    (28, 'Constraint', 'High'),
    (29, 'Deduction', 'High'),
    (29, 'Boundary', 'Medium'),
    (30, 'Abduction', 'Medium'),
    (30, 'Falsification', 'Medium'),
    (31, 'Deduction', 'High'),
    (31, 'Abduction', 'Medium'),
    (31, 'Causal Chain', 'Medium'),
    (31, 'Analogy', 'Medium'),
    (32, 'Deduction', 'High'),
    (32, 'Induction', 'High'),

    # Block 5: Noise (iters 33-40)
    (33, 'Analogy', 'High'),
    (33, 'Deduction', 'High'),
    (33, 'Abduction', 'High'),
    (33, 'Regime', 'High'),
    (33, 'Causal Chain', 'High'),
    (33, 'Predictive', 'High'),
    (33, 'Induction', 'High'),
    (34, 'Deduction', 'High'),
    (34, 'Induction', 'Medium'),
    (36, 'Deduction', 'High'),
    (36, 'Boundary', 'Medium'),
    (38, 'Deduction', 'High'),
    (38, 'Boundary', 'Medium'),
    (39, 'Boundary', 'Medium'),
    (39, 'Induction', 'Medium'),
    (40, 'Deduction', 'High'),
    (40, 'Boundary', 'High'),
    (40, 'Induction', 'High'),
    (40, 'Regime', 'High'),

    # Block 6: Low_rank + n_types=2 (iters 41-48)
    (41, 'Analogy', 'Medium'),
    (41, 'Abduction', 'High'),
    (41, 'Regime', 'High'),
    (41, 'Predictive', 'Medium'),
    (42, 'Analogy', 'High'),
    (43, 'Deduction', 'High'),
    (43, 'Induction', 'Medium'),
    (43, 'Analogy', 'High'),
    (44, 'Deduction', 'High'),
    (44, 'Boundary', 'Medium'),
    (44, 'Induction', 'Medium'),
    (45, 'Deduction', 'High'),
    (45, 'Falsification', 'High'),
    (45, 'Boundary', 'High'),
    (45, 'Constraint', 'High'),
    (46, 'Deduction', 'Medium'),
    (47, 'Falsification', 'High'),
    (47, 'Abduction', 'High'),
    (47, 'Boundary', 'High'),
    (47, 'Constraint', 'High'),
    (48, 'Falsification', 'High'),
    (48, 'Constraint', 'High'),
    (48, 'Induction', 'High'),

    # Block 7: Sparse (iters 49-56)
    (49, 'Abduction', 'High'),
    (49, 'Regime', 'High'),
    (49, 'Causal Chain', 'High'),
    (49, 'Predictive', 'High'),
    (49, 'Uncertainty', 'Medium'),
    (50, 'Deduction', 'High'),
    (50, 'Falsification', 'High'),
    (50, 'Abduction', 'High'),
    (50, 'Analogy', 'Medium'),
    (51, 'Deduction', 'High'),
    (51, 'Falsification', 'High'),
    (51, 'Meta-reasoning', 'High'),
    (51, 'Induction', 'Medium'),
    (52, 'Deduction', 'High'),
    (52, 'Falsification', 'High'),
    (52, 'Abduction', 'Medium'),
    (53, 'Meta-reasoning', 'High'),
    (54, 'Falsification', 'High'),
    (54, 'Abduction', 'Medium'),
    (54, 'Uncertainty', 'Medium'),
    (55, 'Falsification', 'High'),
    (56, 'Deduction', 'High'),
    (56, 'Falsification', 'High'),
    (56, 'Induction', 'High'),

    # Block 8: Sparse + Noise (iters 57-64)
    (57, 'Analogy', 'High'),
    (57, 'Deduction', 'High'),
    (57, 'Causal Chain', 'High'),
    (57, 'Regime', 'High'),
    (57, 'Predictive', 'High'),
    (57, 'Induction', 'High'),
    (57, 'Analogy', 'Medium'),
    (58, 'Falsification', 'Medium'),
    (59, 'Meta-reasoning', 'Medium'),
    (59, 'Boundary', 'Medium'),
    (60, 'Falsification', 'Medium'),
    (60, 'Boundary', 'Medium'),
    (60, 'Induction', 'Medium'),
    (60, 'Uncertainty', 'Medium'),
    (61, 'Induction', 'High'),
    (62, 'Deduction', 'High'),
    (62, 'Falsification', 'High'),
    (63, 'Induction', 'High'),
    (64, 'Induction', 'High'),

    # Block 9: Intermediate sparsity ff=0.5 (iters 65-72)
    (65, 'Analogy', 'High'),
    (65, 'Regime', 'High'),
    (65, 'Deduction', 'High'),
    (65, 'Induction', 'High'),
    (66, 'Deduction', 'High'),
    (66, 'Falsification', 'High'),
    (67, 'Deduction', 'High'),
    (67, 'Induction', 'Medium'),
    (68, 'Induction', 'Medium'),
    (69, 'Induction', 'High'),
    (69, 'Falsification', 'High'),
    (70, 'Deduction', 'High'),
    (70, 'Induction', 'High'),
    (71, 'Deduction', 'High'),
    (71, 'Falsification', 'High'),
    (72, 'Induction', 'High'),
    (72, 'Constraint', 'High'),

    # Block 10: ff=0.75 (iters 73-80)
    (73, 'Analogy', 'High'),
    (73, 'Deduction', 'High'),
    (74, 'Induction', 'Medium'),
    (76, 'Deduction', 'High'),
    (76, 'Falsification', 'Medium'),
    (78, 'Induction', 'High'),
    (79, 'Deduction', 'High'),
    (79, 'Falsification', 'High'),
    (80, 'Deduction', 'High'),
    (80, 'Induction', 'High'),
    (80, 'Predictive', 'High'),

    # Block 11: n=200 (iters 81-88)
    (82, 'Analogy', 'High'),
    (82, 'Induction', 'High'),
    (83, 'Deduction', 'High'),
    (83, 'Induction', 'Medium'),
    (85, 'Deduction', 'High'),
    (86, 'Deduction', 'High'),
    (86, 'Induction', 'High'),
    (87, 'Deduction', 'High'),
    (88, 'Deduction', 'High'),
    (88, 'Induction', 'High'),
    (88, 'Regime', 'High'),

    # Block 12: ff=0.9 (iters 89-96)
    (89, 'Analogy', 'High'),
    (89, 'Deduction', 'High'),
    (90, 'Induction', 'High'),
    (90, 'Deduction', 'High'),
    (91, 'Induction', 'Medium'),
    (91, 'Boundary', 'Medium'),
    (92, 'Boundary', 'Medium'),
    (93, 'Falsification', 'High'),
    (93, 'Induction', 'High'),
    (94, 'Deduction', 'High'),
    (94, 'Boundary', 'High'),
    (95, 'Deduction', 'High'),
    (95, 'Boundary', 'High'),
    (96, 'Deduction', 'High'),
    (96, 'Falsification', 'High'),
    (96, 'Induction', 'High'),
    (96, 'Constraint', 'High'),

    # Block 13: n=300 (iters 97-104)
    (97, 'Analogy', 'High'),
    (97, 'Induction', 'High'),
    (98, 'Deduction', 'High'),
    (99, 'Deduction', 'High'),
    (99, 'Boundary', 'Medium'),
    (100, 'Deduction', 'High'),
    (100, 'Boundary', 'High'),
    (101, 'Deduction', 'High'),
    (101, 'Falsification', 'High'),
    (101, 'Constraint', 'High'),
    (102, 'Induction', 'High'),
    (103, 'Deduction', 'High'),
    (104, 'Induction', 'High'),
    (104, 'Predictive', 'High'),

    # Block 14: n=500 (iters 105-107)
    (105, 'Induction', 'High'),
    (105, 'Abduction', 'High'),
    (106, 'Deduction', 'High'),
    (106, 'Abduction', 'Medium'),
    (107, 'Deduction', 'High'),
    (107, 'Falsification', 'High'),
    (107, 'Induction', 'High'),
    (107, 'Uncertainty', 'High'),
]

# Causal edges from signal_landscape_Claude_epistemic_edges.md
edges = [
    # Block 1
    (4, 'Deduction', 5, 'Boundary', 'leads_to'),
    (4, 'Boundary', 5, 'Meta-reasoning', 'triggers'),
    (7, 'Deduction', 8, 'Falsification', 'leads_to'),
    (8, 'Falsification', 8, 'Induction', 'refines'),

    # Block 2
    (9, 'Abduction', 10, 'Deduction', 'triggers'),
    (10, 'Deduction', 10, 'Falsification', 'leads_to'),
    (10, 'Falsification', 11, 'Deduction', 'triggers'),
    (12, 'Falsification', 14, 'Meta-reasoning', 'triggers'),
    (14, 'Meta-reasoning', 15, 'Deduction', 'triggers'),
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
    (28, 'Falsification', 28, 'Causal Chain', 'leads_to'),
    (30, 'Abduction', 31, 'Deduction', 'triggers'),
    (31, 'Deduction', 31, 'Causal Chain', 'leads_to'),
    (32, 'Induction', 33, 'Analogy', 'triggers'),

    # Block 5
    (33, 'Analogy', 33, 'Deduction', 'triggers'),
    (33, 'Abduction', 33, 'Causal Chain', 'leads_to'),
    (33, 'Causal Chain', 33, 'Predictive', 'leads_to'),
    (33, 'Regime', 34, 'Deduction', 'triggers'),
    (36, 'Deduction', 40, 'Boundary', 'leads_to'),
    (40, 'Boundary', 40, 'Induction', 'leads_to'),
    (40, 'Induction', 41, 'Analogy', 'triggers'),

    # Block 6
    (41, 'Analogy', 42, 'Analogy', 'leads_to'),
    (42, 'Analogy', 43, 'Deduction', 'triggers'),
    (43, 'Deduction', 44, 'Deduction', 'leads_to'),
    (45, 'Deduction', 45, 'Falsification', 'leads_to'),
    (45, 'Falsification', 45, 'Constraint', 'leads_to'),
    (47, 'Falsification', 47, 'Constraint', 'leads_to'),
    (48, 'Induction', 49, 'Abduction', 'triggers'),

    # Block 7
    (49, 'Abduction', 49, 'Causal Chain', 'leads_to'),
    (49, 'Causal Chain', 50, 'Deduction', 'triggers'),
    (50, 'Deduction', 50, 'Falsification', 'leads_to'),
    (50, 'Falsification', 51, 'Meta-reasoning', 'triggers'),
    (51, 'Meta-reasoning', 52, 'Deduction', 'leads_to'),
    (52, 'Deduction', 52, 'Falsification', 'leads_to'),
    (53, 'Meta-reasoning', 56, 'Deduction', 'triggers'),
    (54, 'Falsification', 56, 'Induction', 'refines'),
    (56, 'Induction', 57, 'Analogy', 'triggers'),

    # Block 8
    (57, 'Analogy', 57, 'Deduction', 'triggers'),
    (57, 'Deduction', 57, 'Causal Chain', 'leads_to'),
    (57, 'Regime', 58, 'Falsification', 'triggers'),
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
    (69, 'Falsification', 70, 'Deduction', 'triggers'),
    (71, 'Falsification', 72, 'Induction', 'refines'),
    (72, 'Induction', 73, 'Analogy', 'triggers'),

    # Block 10
    (73, 'Deduction', 76, 'Deduction', 'leads_to'),
    (79, 'Falsification', 80, 'Induction', 'refines'),
    (80, 'Induction', 82, 'Analogy', 'triggers'),

    # Block 11
    (82, 'Induction', 83, 'Deduction', 'leads_to'),
    (85, 'Deduction', 86, 'Deduction', 'leads_to'),
    (88, 'Induction', 89, 'Analogy', 'triggers'),

    # Block 12
    (89, 'Deduction', 90, 'Deduction', 'leads_to'),
    (93, 'Falsification', 94, 'Deduction', 'triggers'),
    (96, 'Falsification', 97, 'Analogy', 'triggers'),

    # Block 13
    (97, 'Induction', 98, 'Deduction', 'leads_to'),
    (100, 'Deduction', 101, 'Deduction', 'leads_to'),
    (101, 'Falsification', 103, 'Deduction', 'triggers'),
    (104, 'Induction', 105, 'Abduction', 'triggers'),

    # Block 14
    (105, 'Induction', 106, 'Deduction', 'leads_to'),
    (106, 'Deduction', 107, 'Deduction', 'leads_to'),
    (107, 'Falsification', 107, 'Uncertainty', 'triggers'),

    # Cross-block
    (33, 'Causal Chain', 57, 'Deduction', 'triggers'),
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
    (65, 72, 'Block 9', {'regime': 'ff=0.5', 'eff_rank': '20-26'}),
    (73, 80, 'Block 10', {'regime': 'ff=0.75', 'eff_rank': '25-33'}),
    (81, 88, 'Block 11', {'regime': 'n=200', 'eff_rank': '42-44'}),
    (89, 96, 'Block 12', {'regime': 'ff=0.9', 'eff_rank': '34-35'}),
    (97, 104, 'Block 13', {'regime': 'n=300', 'eff_rank': '44-46'}),
    (105, 107, 'Block 14', {'regime': 'n=500', 'eff_rank': '45-50'}),
]


def create_2x2_panels():
    """Create 3-row panel figure for signal_landscape_Claude."""

    fig = plt.figure(figsize=(20, 20), constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig, width_ratios=[2.5, 0.8], height_ratios=[1, 1, 1.2])

    modes = [
        'Induction', 'Boundary', 'Abduction', 'Deduction', 'Falsification',
        'Analogy', 'Meta-reasoning', 'Regime', 'Uncertainty',
        'Constraint', 'Predictive', 'Causal Chain',
    ]
    mode_to_y = {mode: i for i, mode in enumerate(modes)}

    x_min, x_max = 0, 112

    # ============ TOP LEFT: Scatterplot ============
    ax1 = fig.add_subplot(gs[0, 0])

    for block_idx, (start, end, label, info) in enumerate(blocks):
        color = '#f5f5f5' if block_idx % 2 == 1 else 'white'
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
    n_bins = (max_iter // bin_width) + 2
    iterations = np.arange(bin_width / 2, (n_bins + 0.5) * bin_width, bin_width)

    data = np.zeros((len(modes), n_bins))
    for iteration, mode, significance in events:
        if mode in modes:
            mode_idx = modes.index(mode)
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

    ax3.set_xlim(x_min, x_max)
    ax3.set_yticks([])
    ax3.set_facecolor('#fafafa')
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
107 iterations, 14 blocks
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

    # ============ ROW 3: Sankey Diagram ============
    ax_sankey = fig.add_subplot(gs[2, 0])
    ax_sankey.axis('off')

    import os
    sankey_path = os.path.join(os.path.dirname(__file__) or '.', 'Sankey.png')
    if os.path.exists(sankey_path):
        sankey_img = plt.imread(sankey_path)
        ax_sankey.imshow(sankey_img, aspect='equal')
    else:
        ax_sankey.text(0.5, 0.5, 'Sankey.png not found', fontsize=14,
                      ha='center', va='center', transform=ax_sankey.transAxes)

    # ============ ROW 3: Caption ============
    ax_caption = fig.add_subplot(gs[2, 1])
    ax_caption.set_title('Claude summary', fontsize=16, loc='left')
    ax_caption.axis('off')

    caption_text = ("Epistemic flow of LLM-guided scientific discovery. "
                   "Deduction acts as the central hub-nearly all reasoning modes "
                   "converge through it before branching outward. The dominant "
                   "Deduction->Falsification pathway demonstrates genuine hypothesis "
                   "testing rather than mere pattern matching.")

    import textwrap
    wrapped_text = textwrap.fill(caption_text, width=35)
    ax_caption.text(0.0, 0.88, wrapped_text, fontsize=11, va='top', ha='left',
                   linespacing=1.4)
    ax_caption.set_xlim(0, 1)
    ax_caption.set_ylim(0, 1)

    plt.savefig('signal_landscape_Claude_epistemic.png', dpi=150, bbox_inches='tight',
                pad_inches=0.3, facecolor='white')
    print("Saved: signal_landscape_Claude_epistemic.png")
    plt.close()


def create_sankey():
    """Create flow diagram showing transitions between reasoning modes using networkx."""
    import networkx as nx

    # Count transitions between modes
    transitions = Counter()
    for from_iter, from_mode, to_iter, to_mode, edge_type in edges:
        transitions[(from_mode, to_mode)] += 1

    # Create directed graph
    G = nx.DiGraph()

    # Add nodes with colors
    for mode in COLORS:
        G.add_node(mode)

    # Add edges with weights
    for (from_mode, to_mode), count in transitions.items():
        if from_mode in COLORS and to_mode in COLORS:
            G.add_edge(from_mode, to_mode, weight=count)

    # Remove isolated nodes
    isolated = [n for n in G.nodes() if G.degree(n) == 0]
    G.remove_nodes_from(isolated)

    fig, ax = plt.subplots(figsize=(14, 10))

    # Use shell layout with Deduction at center
    # Group by role: sources (more outflow), sinks (more inflow), hubs (both)
    outflow = {n: sum(d['weight'] for _, _, d in G.out_edges(n, data=True)) for n in G.nodes()}
    inflow = {n: sum(d['weight'] for _, _, d in G.in_edges(n, data=True)) for n in G.nodes()}

    # Custom positions - arrange in layers
    pos = {}
    nodes = list(G.nodes())

    # Layer 1 (left): mainly sources - Induction, Boundary, Abduction
    # Layer 2 (center): hub - Deduction
    # Layer 3 (right): mainly sinks - Falsification, Analogy, others

    layers = {
        0: ['Induction', 'Boundary', 'Regime'],
        1: ['Abduction', 'Deduction', 'Causal Chain'],
        2: ['Falsification', 'Analogy', 'Meta-reasoning'],
        3: ['Predictive', 'Constraint', 'Uncertainty'],
    }

    for layer_idx, layer_nodes in layers.items():
        x = layer_idx * 2.5
        active_nodes = [n for n in layer_nodes if n in G.nodes()]
        n_nodes = len(active_nodes)
        for i, node in enumerate(active_nodes):
            y = (i - (n_nodes - 1) / 2) * 1.8
            pos[node] = (x, y)

    # Draw edges with varying width based on weight
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1

    for (u, v, d) in G.edges(data=True):
        weight = d['weight']
        width = 1 + (weight / max_weight) * 8
        alpha = 0.3 + (weight / max_weight) * 0.5
        color = COLORS.get(u, '#888888')

        # Draw curved edge
        rad = 0.2 if pos[u][0] != pos[v][0] else 0.4
        ax.annotate('', xy=pos[v], xytext=pos[u],
                   arrowprops=dict(arrowstyle='->', color=color, alpha=alpha,
                                 lw=width, connectionstyle=f'arc3,rad={rad}'))

    # Draw nodes
    node_colors = [COLORS.get(n, '#888888') for n in G.nodes()]
    node_sizes = [800 + (outflow.get(n, 0) + inflow.get(n, 0)) * 80 for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                          alpha=0.9, ax=ax, edgecolors='white', linewidths=2)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold', ax=ax)

    # Add edge weight labels for major flows
    for (u, v, d) in G.edges(data=True):
        if d['weight'] >= 3:
            mid_x = (pos[u][0] + pos[v][0]) / 2
            mid_y = (pos[u][1] + pos[v][1]) / 2 + 0.15
            ax.text(mid_x, mid_y, str(d['weight']), fontsize=9, ha='center',
                   color='#333', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    ax.set_xlim(-1, 9)
    ax.set_ylim(-4, 4)
    ax.axis('off')
    ax.set_facecolor('white')

    plt.tight_layout()
    plt.savefig('Sankey.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("Saved: Sankey.png")
    plt.close()


def create_timeline():
    """Create timeline visualization."""
    fig, ax = plt.subplots(figsize=(24, 10))

    modes = [
        'Induction', 'Boundary',
        'Abduction', 'Deduction', 'Falsification',
        'Analogy', 'Meta-reasoning', 'Regime', 'Uncertainty',
        'Causal Chain', 'Predictive', 'Constraint',
    ]
    mode_to_y = {mode: i for i, mode in enumerate(modes)}

    for block_idx, (start, end, label, info) in enumerate(blocks):
        color = '#f8f8f8' if block_idx % 2 == 1 else 'white'
        ax.axvspan(start - 0.5, end + 0.5, alpha=1.0, color=color, zorder=0)

    for iteration, mode, significance in events:
        if mode not in mode_to_y:
            continue
        y = mode_to_y[mode]
        color = COLORS.get(mode, '#333333')
        size = {'High': 150, 'Medium': 80, 'Low': 40}.get(significance, 80)
        ax.scatter(iteration, y, c=color, s=size, alpha=0.9, edgecolors='white', linewidth=0.5, zorder=3)

    ax.set_xlim(0, 112)
    ax.set_ylim(-0.5, len(modes) - 0.5)
    ax.set_yticks(range(len(modes)))
    ax.set_yticklabels(modes, fontsize=16)
    ax.set_xlabel('Iteration', fontsize=24)
    ax.tick_params(axis='x', labelsize=16)
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    legend_elements = [plt.scatter([], [], c=COLORS[m], s=100, label=m) for m in modes]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12, ncol=3)

    plt.tight_layout()
    plt.savefig('signal_landscape_Claude_epistemic_timeline.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("Saved: signal_landscape_Claude_epistemic_timeline.png")
    plt.close()


def create_streamgraph():
    """Create standalone streamgraph (stacked area chart) visualization."""
    fig, ax = plt.subplots(figsize=(16, 6))

    modes = [
        'Induction', 'Boundary', 'Abduction', 'Deduction', 'Falsification',
        'Analogy', 'Meta-reasoning', 'Regime', 'Uncertainty',
        'Constraint', 'Predictive', 'Causal Chain',
    ]

    max_iter = max(e[0] for e in events)
    bin_width = 4
    n_bins = (max_iter // bin_width) + 2
    iterations = np.arange(bin_width / 2, (n_bins + 0.5) * bin_width, bin_width)

    data = np.zeros((len(modes), n_bins))
    for iteration, mode, significance in events:
        if mode in modes:
            mode_idx = modes.index(mode)
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
        ax.fill_between(iterations, y_stack[i], y_stack[i + 1],
                        color=color, alpha=0.8, label=mode,
                        edgecolor='white', linewidth=0.3)

    # Add block boundaries
    for block_idx, (start, end, label, info) in enumerate(blocks):
        ax.axvline(x=end + 0.5, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)

    ax.set_xlim(0, 112)
    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_ylabel('Reasoning Activity', fontsize=14)
    ax.set_facecolor('#fafafa')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend outside
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10, frameon=False)

    plt.tight_layout()
    plt.savefig('signal_landscape_Claude_epistemic_streamgraph.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("Saved: signal_landscape_Claude_epistemic_streamgraph.png")
    plt.close()


if __name__ == '__main__':
    print("Creating signal_landscape_Claude epistemic visualizations...")
    print(f"Total events: {len(events)}")
    print(f"Total edges: {len(edges)}")

    counts = Counter(e[1] for e in events)
    for mode, count in sorted(counts.items()):
        print(f"  {mode}: {count}")

    print("\n--- Timeline ---")
    create_timeline()

    print("\n--- Streamgraph ---")
    create_streamgraph()

    print("\n--- 2x2 Panels ---")
    create_2x2_panels()

    print("\n--- Sankey ---")
    create_sankey()

    print("\nDone!")
