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

# Events from epistemic_detailed.md - MUST MATCH EXACTLY
# Format: (iteration, mode, significance)
events = [
    # === Induction: 13 instances (from detailed.md) ===
    (6, 'Induction', 'High'),    # lr_W robust range
    (9, 'Induction', 'Medium'),  # 8 consecutive converged
    (16, 'Induction', 'High'),   # Block 1 summary
    (22, 'Induction', 'High'),   # Dale_law lr_W pattern
    (32, 'Induction', 'High'),   # Block 2 summary
    (48, 'Induction', 'High'),   # Low_rank learnable
    (64, 'Induction', 'High'),   # n_frames affects eff_rank
    (80, 'Induction', 'High'),   # Block 5 summary
    (96, 'Induction', 'High'),   # n_frames threshold
    (112, 'Induction', 'High'),  # Block 7 summary
    (128, 'Induction', 'High'),  # Dale re-test confirms
    (143, 'Induction', 'High'),  # low_rank=10 unlearnable
    (148, 'Induction', 'Medium'), # n_neurons=1000 failing

    # === Abduction: 10 instances (from detailed.md) ===
    (1, 'Abduction', 'High'),    # eff_rank hypothesis
    (17, 'Abduction', 'High'),   # Dale_law reduces eff_rank
    (33, 'Abduction', 'High'),   # connectivity rank constrains
    (49, 'Abduction', 'High'),   # eff_rank doesn't scale
    (65, 'Abduction', 'High'),   # Data quantity affects eff_rank
    (81, 'Abduction', 'High'),   # Below minimum threshold
    (97, 'Abduction', 'Medium'), # n_frames threshold hypothesis
    (113, 'Abduction', 'High'),  # eff_rank=4 unlearnable
    (129, 'Abduction', 'High'),  # Double constraint
    (145, 'Abduction', 'High'),  # Scaling laws differ

    # === Deduction: 17 instances (from detailed.md) ===
    (4, 'Deduction', 'High'),    # lr_W boundary prediction
    (5, 'Deduction', 'High'),    # lr_W=2E-2 prediction
    (8, 'Deduction', 'High'),    # L1=1E-4 prediction
    (13, 'Deduction', 'High'),   # factorization prediction
    (19, 'Deduction', 'High'),   # Higher lr_W for Dale
    (21, 'Deduction', 'High'),   # lr_W=5E-2 converge
    (34, 'Deduction', 'High'),   # Block 1 params transfer
    (37, 'Deduction', 'High'),   # factorization helps
    (53, 'Deduction', 'High'),   # n_frames boost eff_rank
    (70, 'Deduction', 'High'),   # n_frames=5000 quality
    (86, 'Deduction', 'High'),   # n_frames=2500 converge
    (97, 'Deduction', 'High'),   # n_frames=20000 solves
    (104, 'Deduction', 'High'),  # n_frames=7500 threshold
    (117, 'Deduction', 'High'),  # Dale re-test
    (121, 'Deduction', 'High'),  # low_rank=10 fail
    (137, 'Deduction', 'High'),  # n_frames=30000 help
    (148, 'Deduction', 'Medium'), # n_neurons needs 100K

    # === Falsification: 20 instances (from detailed.md) ===
    (5, 'Falsification', 'High'),    # lr_W boundary falsified
    (10, 'Falsification', 'High'),   # L1 upper bound
    (13, 'Falsification', 'High'),   # factorization fails
    (19, 'Falsification', 'High'),   # Block 1 lr_W doesn't work
    (25, 'Falsification', 'High'),   # factorization still fails
    (30, 'Falsification', 'Medium'), # L1=1E-4 fails for Dale
    (35, 'Falsification', 'High'),   # Block 1 params fail
    (41, 'Falsification', 'High'),   # L1=1E-7 fails
    (42, 'Falsification', 'High'),   # lr_W=0.5 fails
    (49, 'Falsification', 'High'),   # low_rank=50 harder
    (57, 'Falsification', 'Medium'), # connectivity_rank doesn't help
    (66, 'Falsification', 'High'),   # n_frames=5000 not universal
    (82, 'Falsification', 'High'),   # n_frames=2500 catastrophic
    (109, 'Falsification', 'Medium'), # n_frames=7500 not universal
    (113, 'Falsification', 'High'),  # lr_W doesn't help low_rank=10
    (125, 'Falsification', 'High'),  # n_frames=20000 doesn't help
    (127, 'Falsification', 'High'),  # low_rank=10 unlearnable
    (138, 'Falsification', 'High'),  # factorization limited
    (145, 'Falsification', 'High'),  # n_neurons scaling fails

    # === Analogy/Transfer: 10 instances (from detailed.md) ===
    (17, 'Analogy/Transfer', 'Medium'),  # Block 1→2 failed
    (33, 'Analogy/Transfer', 'Medium'),  # Block 1→3 failed
    (34, 'Analogy/Transfer', 'High'),    # Block 2→3 partial
    (49, 'Analogy/Transfer', 'High'),    # Block 3→4 failed
    (65, 'Analogy/Transfer', 'High'),    # Block 1-4→5 success
    (81, 'Analogy/Transfer', 'High'),    # Block 5→6 failed
    (97, 'Analogy/Transfer', 'High'),    # Block 6→7 success
    (113, 'Analogy/Transfer', 'Medium'), # Block 3→8 failed
    (129, 'Analogy/Transfer', 'High'),   # Block 2+3→9 partial
    (145, 'Analogy/Transfer', 'High'),   # All→11 failing

    # === Boundary: 12 instances (from detailed.md) ===
    (4, 'Boundary', 'High'),    # lr_W approaching
    (6, 'Boundary', 'High'),    # lr_W still working
    (10, 'Boundary', 'High'),   # L1 upper bound
    (14, 'Boundary', 'Medium'), # L1 refined
    (24, 'Boundary', 'High'),   # Dale lr_W upper
    (26, 'Boundary', 'Medium'), # Dale L1 sensitive
    (42, 'Boundary', 'High'),   # low_rank lr_W upper
    (57, 'Boundary', 'High'),   # connectivity_rank
    (82, 'Boundary', 'High'),   # n_frames below
    (104, 'Boundary', 'High'),  # n_frames near threshold
    (121, 'Boundary', 'High'),  # low_rank=10 unlearnable
    (126, 'Boundary', 'High'),  # hard barrier

    # === Meta-reasoning: 6 instances (from detailed.md) ===
    (6, 'Meta-reasoning', 'Medium'),   # Strategy switch
    (26, 'Meta-reasoning', 'Medium'),  # Strategy reset
    (57, 'Meta-reasoning', 'High'),    # Dimension re-evaluation
    (97, 'Meta-reasoning', 'High'),    # Insight synthesis
    (137, 'Meta-reasoning', 'High'),   # Barrier recognition
    (148, 'Meta-reasoning', 'High'),   # Scale recognition

    # === Regime: 7 instances (from detailed.md) ===
    (17, 'Regime', 'High'),   # Dale_law different
    (33, 'Regime', 'High'),   # low_rank constraints
    (65, 'Regime', 'Medium'), # Data regime
    (81, 'Regime', 'High'),   # Failure regime
    (113, 'Regime', 'High'),  # Hard barrier
    (129, 'Regime', 'High'),  # Combined regime
    (145, 'Regime', 'High'),  # Scale regime

    # === Uncertainty: 5 instances (from detailed.md) ===
    (41, 'Uncertainty', 'High'),   # R² varies
    (42, 'Uncertainty', 'High'),   # Same params differ
    (82, 'Uncertainty', 'High'),   # Threshold uncertain
    (97, 'Uncertainty', 'Medium'), # Dale variance
    (113, 'Uncertainty', 'High'),  # low_rank=10 unpredictable

    # === Causal: 5 instances (from detailed.md) ===
    (48, 'Causal', 'High'),   # connectivity_rank → eff_rank → R²
    (64, 'Causal', 'High'),   # n_frames → eff_rank → lr_W
    (96, 'Causal', 'High'),   # n_frames=2500 → unlearnable
    (127, 'Causal', 'High'),  # low_rank=10 → eff_rank=4
    (135, 'Causal', 'Medium'), # Dale → spectral → eff_rank

    # === Predictive: 3 instances (from detailed.md) ===
    (88, 'Predictive', 'High'),   # eff_rank≥20 rule
    (96, 'Predictive', 'High'),   # n_frames threshold
    (144, 'Predictive', 'High'),  # low_rank ≥ 3x rule

    # === Constraint: 1 instance (from detailed.md) ===
    (104, 'Constraint', 'High'),  # n_frames formula
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

    # === Block 10 (low_rank+Dale factorization) ===
    (138, 'Falsification', 144, 'Predictive', 'leads_to'),
    (144, 'Predictive', 145, 'Analogy/Transfer', 'leads_to'),

    # === Block 11 (n_neurons=1000) ===
    (145, 'Falsification', 148, 'Deduction', 'leads_to'),
    (145, 'Regime', 148, 'Meta-reasoning', 'leads_to'),

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
    (17, 32, 'Block 2: Dale_law'),
    (33, 48, 'Block 3: Low_rank=20'),
    (49, 64, 'Block 4: Low_rank=50'),
    (65, 80, 'Block 5: n_frames=5000'),
    (81, 96, 'Block 6: n_frames=2500'),
    (97, 112, 'Block 7: n_frames=7500'),
    (113, 128, 'Block 8: Dale_law re-test'),
    (129, 144, 'Block 9-10: low_rank+Dale'),
    (145, 149, 'Block 11: n_neurons=1000'),  # In progress, 149 completed
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
        # Advanced patterns (complexity: Constraint < Predictive < Causal)
        'Constraint', 'Predictive', 'Causal',
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
        'rejects': {'linestyle': '-', 'color': '#e74c3c', 'alpha': 0.8},  # Red for rejection
    }

    for from_iter, from_mode, to_iter, to_mode, edge_type in edges:
        if from_mode not in mode_to_y or to_mode not in mode_to_y:
            continue

        x1, y1 = from_iter, mode_to_y[from_mode]
        x2, y2 = to_iter, mode_to_y[to_mode]

        style = edge_styles.get(edge_type, edge_styles['leads_to'])

        # Special handling for edges TO Falsification: these should point backward
        # Falsification rejects the prior hypothesis, so we draw arrow from Falsification
        # back to the hypothesis it rejects (reversed direction, red color)
        if to_mode == 'Falsification':
            # Draw backward arrow: from the Falsification node (x2,y2) back to source hypothesis (x1,y1)
            # Arrow points LEFT (backward in time) with arrowhead at the hypothesis end
            arrow = FancyArrowPatch(
                (x2, y2), (x1, y1),  # Start at Falsification, end at hypothesis
                arrowstyle='<|-',    # Arrowhead at START (pointing backward)
                mutation_scale=12,
                linestyle='-',
                color='#e74c3c',  # Red for rejection
                alpha=0.8,
                linewidth=2.0,
                zorder=1
            )
        else:
            # Normal forward arrow
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

    # Plot events (nodes) with iteration labels
    for iteration, mode, significance in events:
        if mode not in mode_to_y:
            continue
        y = mode_to_y[mode]
        color = COLORS.get(mode, '#333333')

        # Size based on significance
        size = {'High': 150, 'Medium': 80, 'Low': 40}.get(significance, 80)

        ax.scatter(iteration, y, c=color, s=size, alpha=0.9,
                   edgecolors='black', linewidths=0.5, zorder=3)

        # Add iteration number label (small font, offset slightly)
        ax.annotate(str(iteration), (iteration, y),
                    xytext=(0, -12), textcoords='offset points',
                    fontsize=6, ha='center', va='top', color='#333333',
                    zorder=4)

    # Add category labels on right side
    ax.axhline(y=1.5, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axhline(y=4.5, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axhline(y=8.5, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

    # Styling
    ax.set_xlim(0, 155)  # Max iter 149 + margin
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
