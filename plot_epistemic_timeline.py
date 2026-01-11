#!/usr/bin/env python3
"""
Visualize epistemic reasoning timeline from signal_chaotic_1_Claude experiment.
Color-coded by reasoning mode type.
Based on fresh analysis of reasoning logs (2026-01-10).
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Define color scheme for reasoning modes
COLORS = {
    'Induction': '#2ecc71',        # Green - pattern extraction
    'Abduction': '#9b59b6',        # Purple - causal hypothesis
    'Deduction': '#3498db',        # Blue - prediction testing
    'Falsification': '#e74c3c',    # Red - hypothesis rejection
    'Analogy/Transfer': '#f39c12', # Orange - cross-regime
    'Boundary': '#1abc9c',         # Teal - limit finding
    'Meta-reasoning': '#e91e63',   # Pink - strategy adaptation
    'Regime': '#795548',           # Brown - phase identification
    'Uncertainty': '#607d8b',      # Gray - stochasticity awareness
    'Causal': '#00bcd4',           # Cyan - causal chain
    'Predictive': '#8bc34a',       # Light green - predictive modeling
    'Constraint': '#ff5722',       # Deep orange - constraint propagation
}

# Complete events from fresh epistemic analysis
# Format: (iteration, mode, significance)
events = [
    # === Induction: 18 instances ===
    (6, 'Induction', 'High'),
    (9, 'Induction', 'Medium'),
    (16, 'Induction', 'High'),
    (22, 'Induction', 'High'),
    (32, 'Induction', 'High'),
    (40, 'Induction', 'High'),
    (48, 'Induction', 'High'),
    (53, 'Induction', 'High'),
    (64, 'Induction', 'High'),
    (72, 'Induction', 'High'),
    (80, 'Induction', 'High'),
    (88, 'Induction', 'High'),
    (96, 'Induction', 'High'),
    (104, 'Induction', 'Medium'),
    (112, 'Induction', 'High'),
    (127, 'Induction', 'High'),
    (143, 'Induction', 'High'),
    (159, 'Induction', 'High'),

    # === Abduction: 14 instances ===
    (17, 'Abduction', 'High'),
    (21, 'Abduction', 'High'),
    (33, 'Abduction', 'High'),
    (36, 'Abduction', 'Medium'),
    (41, 'Abduction', 'High'),
    (49, 'Abduction', 'High'),
    (65, 'Abduction', 'High'),
    (82, 'Abduction', 'High'),
    (97, 'Abduction', 'Medium'),
    (113, 'Abduction', 'High'),
    (129, 'Abduction', 'High'),
    (145, 'Abduction', 'High'),
    (149, 'Abduction', 'Medium'),
    (154, 'Abduction', 'Medium'),

    # === Deduction: 24 instances ===
    (4, 'Deduction', 'High'),
    (5, 'Deduction', 'High'),
    (10, 'Deduction', 'High'),
    (13, 'Deduction', 'High'),
    (19, 'Deduction', 'High'),
    (24, 'Deduction', 'High'),
    (25, 'Deduction', 'Medium'),
    (34, 'Deduction', 'High'),
    (35, 'Deduction', 'High'),
    (36, 'Deduction', 'Medium'),
    (40, 'Deduction', 'Medium'),
    (42, 'Deduction', 'Medium'),
    (53, 'Deduction', 'High'),
    (70, 'Deduction', 'High'),
    (81, 'Deduction', 'High'),
    (97, 'Deduction', 'High'),
    (102, 'Deduction', 'High'),
    (125, 'Deduction', 'High'),
    (137, 'Deduction', 'High'),
    (138, 'Deduction', 'Medium'),
    (149, 'Deduction', 'High'),
    (156, 'Deduction', 'High'),
    (159, 'Deduction', 'High'),

    # === Falsification: 19 instances ===
    (5, 'Falsification', 'High'),
    (10, 'Falsification', 'High'),
    (13, 'Falsification', 'High'),
    (19, 'Falsification', 'High'),
    (25, 'Falsification', 'High'),
    (30, 'Falsification', 'Medium'),
    (35, 'Falsification', 'High'),
    (41, 'Falsification', 'High'),
    (42, 'Falsification', 'High'),
    (49, 'Falsification', 'High'),
    (57, 'Falsification', 'Medium'),
    (66, 'Falsification', 'High'),
    (82, 'Falsification', 'High'),
    (109, 'Falsification', 'Medium'),
    (113, 'Falsification', 'High'),
    (125, 'Falsification', 'High'),
    (127, 'Falsification', 'High'),
    (138, 'Falsification', 'High'),
    (145, 'Falsification', 'High'),
    (150, 'Falsification', 'Medium'),

    # === Analogy/Transfer: 16 instances ===
    (17, 'Analogy/Transfer', 'Medium'),
    (33, 'Analogy/Transfer', 'Medium'),
    (34, 'Analogy/Transfer', 'High'),
    (42, 'Analogy/Transfer', 'High'),
    (49, 'Analogy/Transfer', 'High'),
    (53, 'Analogy/Transfer', 'Medium'),
    (65, 'Analogy/Transfer', 'High'),
    (70, 'Analogy/Transfer', 'High'),
    (81, 'Analogy/Transfer', 'High'),
    (97, 'Analogy/Transfer', 'High'),
    (104, 'Analogy/Transfer', 'High'),
    (113, 'Analogy/Transfer', 'Medium'),
    (129, 'Analogy/Transfer', 'High'),
    (137, 'Analogy/Transfer', 'High'),
    (145, 'Analogy/Transfer', 'High'),
    (149, 'Analogy/Transfer', 'Medium'),

    # === Boundary Probing: 16 instances ===
    (4, 'Boundary', 'High'),
    (5, 'Boundary', 'High'),
    (10, 'Boundary', 'Medium'),
    (14, 'Boundary', 'Medium'),
    (17, 'Boundary', 'High'),
    (26, 'Boundary', 'Medium'),
    (30, 'Boundary', 'Medium'),
    (33, 'Boundary', 'High'),
    (41, 'Boundary', 'Medium'),
    (53, 'Boundary', 'High'),
    (65, 'Boundary', 'High'),
    (78, 'Boundary', 'Medium'),
    (98, 'Boundary', 'High'),
    (106, 'Boundary', 'High'),
    (125, 'Boundary', 'High'),
    (154, 'Boundary', 'High'),

    # === Emerging Patterns ===
    # Meta-reasoning: 6 instances
    (6, 'Meta-reasoning', 'Medium'),
    (8, 'Meta-reasoning', 'Medium'),
    (26, 'Meta-reasoning', 'Medium'),
    (57, 'Meta-reasoning', 'High'),
    (64, 'Meta-reasoning', 'High'),
    (104, 'Meta-reasoning', 'Medium'),

    # Regime Recognition: 8 instances
    (22, 'Regime', 'High'),
    (33, 'Regime', 'High'),
    (48, 'Regime', 'High'),
    (97, 'Regime', 'High'),
    (112, 'Regime', 'High'),
    (113, 'Regime', 'High'),
    (145, 'Regime', 'High'),

    # Uncertainty: 5 instances
    (41, 'Uncertainty', 'High'),
    (42, 'Uncertainty', 'High'),
    (82, 'Uncertainty', 'High'),
    (97, 'Uncertainty', 'Medium'),
    (113, 'Uncertainty', 'High'),

    # Causal: 7 instances
    (48, 'Causal', 'High'),
    (53, 'Causal', 'High'),
    (127, 'Causal', 'High'),
    (135, 'Causal', 'High'),
    (149, 'Causal', 'High'),

    # Predictive: 3 instances
    (88, 'Predictive', 'High'),
    (96, 'Predictive', 'High'),
    (159, 'Predictive', 'High'),

    # Constraint: 1 instance
    (104, 'Constraint', 'High'),
]

# Causal edges between reasoning events
# RULE: All edges satisfy from_iter < to_iter (temporal causality)
# Format: (from_iter, from_mode, to_iter, to_mode, edge_type)
edges = [
    # === Block 1 (Chaotic baseline) ===
    (4, 'Deduction', 6, 'Induction', 'leads_to'),
    (5, 'Falsification', 6, 'Induction', 'leads_to'),
    (6, 'Induction', 8, 'Meta-reasoning', 'leads_to'),
    (6, 'Induction', 16, 'Induction', 'leads_to'),
    (10, 'Falsification', 13, 'Deduction', 'leads_to'),

    # === Block 2 (Low_rank=20) ===
    (17, 'Abduction', 21, 'Abduction', 'triggers'),
    (19, 'Falsification', 22, 'Induction', 'leads_to'),
    (22, 'Induction', 24, 'Deduction', 'leads_to'),
    (22, 'Regime', 26, 'Boundary', 'leads_to'),
    (25, 'Falsification', 26, 'Meta-reasoning', 'leads_to'),
    (26, 'Boundary', 32, 'Induction', 'leads_to'),

    # === Block 3 (Dale_law) ===
    (33, 'Abduction', 35, 'Deduction', 'triggers'),
    (33, 'Regime', 36, 'Abduction', 'leads_to'),
    (35, 'Falsification', 36, 'Deduction', 'leads_to'),
    (40, 'Induction', 41, 'Abduction', 'leads_to'),
    (41, 'Uncertainty', 42, 'Falsification', 'leads_to'),
    (42, 'Falsification', 48, 'Induction', 'leads_to'),
    (42, 'Uncertainty', 48, 'Causal', 'leads_to'),

    # === Block 4 (Low_rank=50) ===
    (49, 'Abduction', 53, 'Deduction', 'triggers'),
    (53, 'Deduction', 57, 'Meta-reasoning', 'leads_to'),
    (53, 'Causal', 64, 'Induction', 'leads_to'),
    (57, 'Meta-reasoning', 64, 'Meta-reasoning', 'leads_to'),

    # === Block 5 (Double constraint) ===
    (65, 'Abduction', 70, 'Deduction', 'triggers'),
    (66, 'Falsification', 70, 'Deduction', 'leads_to'),
    (70, 'Deduction', 72, 'Induction', 'leads_to'),
    (72, 'Induction', 80, 'Induction', 'leads_to'),

    # === Block 6 (Low_rank=50+Dale) ===
    (82, 'Abduction', 88, 'Induction', 'triggers'),
    (82, 'Uncertainty', 88, 'Predictive', 'leads_to'),
    (88, 'Predictive', 96, 'Induction', 'leads_to'),

    # === Block 7 (Dale_law re-test) ===
    (97, 'Regime', 102, 'Deduction', 'triggers'),
    (97, 'Deduction', 104, 'Induction', 'leads_to'),
    (102, 'Deduction', 106, 'Boundary', 'leads_to'),
    (106, 'Boundary', 112, 'Induction', 'leads_to'),

    # === Block 8 (Low_rank=10) ===
    (113, 'Abduction', 125, 'Falsification', 'triggers'),
    (113, 'Regime', 125, 'Boundary', 'leads_to'),
    (125, 'Falsification', 127, 'Causal', 'leads_to'),

    # === Block 9 (Low_rank=10+Dale) ===
    (129, 'Abduction', 135, 'Causal', 'triggers'),
    (135, 'Causal', 137, 'Deduction', 'leads_to'),
    (137, 'Deduction', 138, 'Falsification', 'leads_to'),
    (138, 'Falsification', 143, 'Induction', 'leads_to'),

    # === Block 10 (n_neurons=1000) ===
    (145, 'Falsification', 149, 'Deduction', 'leads_to'),
    (145, 'Regime', 149, 'Causal', 'leads_to'),
    (149, 'Deduction', 154, 'Boundary', 'leads_to'),
    (149, 'Causal', 159, 'Induction', 'leads_to'),
    (154, 'Boundary', 159, 'Induction', 'leads_to'),

    # === Cross-Block Edges (Knowledge Transfer) ===
    (16, 'Induction', 17, 'Analogy/Transfer', 'triggers'),
    (22, 'Induction', 33, 'Analogy/Transfer', 'triggers'),
    (32, 'Induction', 34, 'Analogy/Transfer', 'triggers'),
    (48, 'Causal', 53, 'Deduction', 'triggers'),
    (64, 'Induction', 65, 'Analogy/Transfer', 'triggers'),
    (80, 'Induction', 81, 'Analogy/Transfer', 'triggers'),
    (88, 'Predictive', 97, 'Analogy/Transfer', 'triggers'),
    (96, 'Induction', 129, 'Analogy/Transfer', 'triggers'),
    (112, 'Induction', 145, 'Analogy/Transfer', 'triggers'),
]

# Block boundaries
blocks = [
    (1, 16, 'Block 1: Chaotic'),
    (17, 32, 'Block 2: Low_rank=20'),
    (33, 48, 'Block 3: Dale_law'),
    (49, 64, 'Block 4: Low_rank=50'),
    (65, 80, 'Block 5: Double constraint'),
    (81, 96, 'Block 6: Low_rank=50+Dale'),
    (97, 112, 'Block 7: Dale_law re-test'),
    (113, 128, 'Block 8: Low_rank=10'),
    (129, 144, 'Block 9: Low_rank=10+Dale'),
    (145, 160, 'Block 10: n_neurons=1000'),
]

def create_timeline():
    fig, ax = plt.subplots(figsize=(24, 14))

    # Map modes to y-positions (group by category)
    # Order: Deduction above Abduction (hypothesis generates prediction)
    modes = [
        # Evidence gathering
        'Induction', 'Boundary',
        # Hypothesis testing (Deduction above Abduction)
        'Abduction', 'Deduction', 'Falsification',
        # Meta-cognition
        'Analogy/Transfer', 'Meta-reasoning', 'Regime', 'Uncertainty',
        # Advanced patterns
        'Causal', 'Predictive', 'Constraint',
    ]
    mode_to_y = {mode: i for i, mode in enumerate(modes)}

    # Draw block backgrounds (no labels)
    for start, end, label in blocks:
        block_idx = blocks.index((start, end, label))
        color = plt.cm.Pastel1(block_idx / len(blocks))
        ax.axvspan(start - 0.5, end + 0.5, alpha=0.3, color=color)

    # Draw edges with arrowheads FIRST (so they're behind nodes)
    from matplotlib.patches import FancyArrowPatch
    edge_styles = {
        'leads_to': {'linestyle': '-', 'color': '#555555', 'alpha': 0.6},
        'triggers': {'linestyle': '--', 'color': '#2196F3', 'alpha': 0.7},
        'refines': {'linestyle': ':', 'color': '#4CAF50', 'alpha': 0.6},
    }

    for from_iter, from_mode, to_iter, to_mode, edge_type in edges:
        if from_mode not in mode_to_y or to_mode not in mode_to_y:
            continue

        x1, y1 = from_iter, mode_to_y[from_mode]
        x2, y2 = to_iter, mode_to_y[to_mode]

        style = edge_styles.get(edge_type, edge_styles['leads_to'])

        # Draw arrow with straight line
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='-|>',
            mutation_scale=12,
            linestyle=style['linestyle'],
            color=style['color'],
            alpha=style['alpha'],
            linewidth=1.5,
            zorder=1
        )
        ax.add_patch(arrow)

    # Plot events (nodes)
    for iteration, mode, significance in events:
        if mode not in mode_to_y:
            continue
        y = mode_to_y[mode]
        color = COLORS.get(mode, '#333333')

        # Size based on significance
        size = {'High': 150, 'Medium': 80, 'Low': 40}.get(significance, 80)

        ax.scatter(iteration, y, c=color, s=size, alpha=0.9,
                   edgecolors='black', linewidths=0.5, zorder=3)

    # Add category labels on right side
    ax.axhline(y=1.5, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axhline(y=4.5, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axhline(y=8.5, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

    # Styling
    ax.set_xlim(0, 165)
    ax.set_ylim(-0.5, len(modes) + 0.5)
    ax.set_yticks(range(len(modes)))
    ax.set_yticklabels(modes, fontsize=16)
    ax.set_xlabel('Iteration', fontsize=24)
    ax.set_ylabel('Reasoning Mode', fontsize=24)
    ax.tick_params(axis='x', labelsize=16)

    # Add grid
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')

    # Legend for colors (modes)
    legend_patches = [mpatches.Patch(color=COLORS[mode], label=mode)
                      for mode in modes if mode in COLORS]
    legend1 = ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.01, 1),
                        title='Reasoning Modes', fontsize=9)
    ax.add_artist(legend1)

    # Legend for edges
    from matplotlib.lines import Line2D
    edge_legend = [
        Line2D([0], [0], color='#555555', linestyle='-', linewidth=2, label='leads to'),
        Line2D([0], [0], color='#2196F3', linestyle='--', linewidth=2, label='triggers'),
        Line2D([0], [0], color='#4CAF50', linestyle=':', linewidth=2, label='refines'),
    ]
    legend2 = ax.legend(handles=edge_legend, loc='upper left', bbox_to_anchor=(1.01, 0.55),
                        title='Edge Types', fontsize=9)
    ax.add_artist(legend2)

    # Legend for node sizes
    size_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=12, label='High significance'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=8, label='Medium significance'),
    ]
    ax.legend(handles=size_legend, loc='upper left', bbox_to_anchor=(1.01, 0.35),
              title='Node Size', fontsize=9)

    plt.tight_layout()
    plt.savefig('signal_chaotic_1_Claude_epistemic_timeline.png', dpi=150, bbox_inches='tight')
    print("Saved: signal_chaotic_1_Claude_epistemic_timeline.png")
    plt.close()


if __name__ == '__main__':
    print("Creating epistemic timeline visualization...")
    print(f"Total events: {len(events)}")
    print(f"Total edges: {len(edges)}")

    # Count by mode
    from collections import Counter
    counts = Counter(e[1] for e in events)
    for mode, count in sorted(counts.items()):
        print(f"  {mode}: {count}")

    create_timeline()
    print("\nDone!")
