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
# Adding edges one by one from epistemic_edges.md
edges = [
    # === Block 1 (Chaotic baseline) - from epistemic_edges.md ===
    (1, 'Abduction', 4, 'Deduction', 'triggers'),      # eff_rank hypothesis → lr_W test
    (4, 'Deduction', 5, 'Falsification', 'leads_to'),  # Boundary prediction → falsified
    (5, 'Falsification', 6, 'Induction', 'leads_to'),  # Falsified boundary → robust range pattern
    (6, 'Boundary', 10, 'Falsification', 'leads_to'),  # lr_W probing → L1 boundary found
    (10, 'Falsification', 13, 'Deduction', 'leads_to'), # L1 boundary → factorization test
    (6, 'Induction', 16, 'Induction', 'leads_to'),     # Cumulative patterns → block summary

    # === Block 2 (Dale_law) - from epistemic_edges.md ===
    (17, 'Abduction', 19, 'Deduction', 'triggers'),     # eff_rank reduction → lr_W scaling hypothesis
    (19, 'Falsification', 22, 'Induction', 'leads_to'), # Failed transfer → new lr_W pattern
    (19, 'Deduction', 21, 'Deduction', 'leads_to'),     # Sequential lr_W tests
    (22, 'Induction', 24, 'Boundary', 'leads_to'),      # Pattern → boundary probing
    (25, 'Falsification', 26, 'Meta-reasoning', 'leads_to'), # factorization failure → strategy reassessment

    # === Block 3 (Low_rank=20) - from epistemic_edges.md ===
    (33, 'Abduction', 34, 'Deduction', 'triggers'),      # connectivity constraint hypothesis → test
    (35, 'Falsification', 37, 'Deduction', 'leads_to'),  # Failed transfer → try factorization
    (37, 'Deduction', 41, 'Uncertainty', 'leads_to'),    # Tests reveal stochasticity
    (41, 'Uncertainty', 42, 'Falsification', 'leads_to'), # Uncertainty → more boundary tests
    (42, 'Falsification', 48, 'Induction', 'leads_to'),  # Boundary failures → causal understanding
    (42, 'Falsification', 48, 'Causal', 'leads_to'),     # Failures → mechanistic model

    # === Block 4 (Low_rank=50) - from epistemic_edges.md ===
    (49, 'Abduction', 53, 'Deduction', 'triggers'),       # Scaling hypothesis → test
    (49, 'Falsification', 57, 'Meta-reasoning', 'leads_to'), # Scaling failed → reframe problem
    (53, 'Deduction', 57, 'Falsification', 'leads_to'),   # Prediction tested → falsified
    (57, 'Meta-reasoning', 64, 'Causal', 'leads_to'),     # Reframing → new causal model

    # === Block 5 (n_frames=5000) - from epistemic_edges.md ===
    (65, 'Abduction', 70, 'Deduction', 'triggers'),       # Data hypothesis → test
    (66, 'Falsification', 70, 'Deduction', 'leads_to'),   # Partial failure → refined test
    (70, 'Deduction', 80, 'Induction', 'leads_to'),       # Tests → block pattern

    # === Block 6 (n_frames=2500) - from epistemic_edges.md ===
    (81, 'Abduction', 86, 'Deduction', 'triggers'),       # Threshold hypothesis → test
    (82, 'Falsification', 88, 'Predictive', 'leads_to'),  # Failure → quantitative rule
    (88, 'Predictive', 96, 'Causal', 'leads_to'),         # Rule → mechanistic understanding
    (88, 'Predictive', 96, 'Induction', 'leads_to'),      # Rule → generalized pattern

    # === Block 7 (n_frames=7500) - from epistemic_edges.md ===
    (97, 'Abduction', 104, 'Deduction', 'triggers'),      # Threshold hypothesis → test
    (97, 'Deduction', 104, 'Constraint', 'leads_to'),     # Tests → constraint formulation
    (104, 'Constraint', 109, 'Falsification', 'leads_to'), # Constraint tested → boundary found
    (109, 'Falsification', 112, 'Induction', 'leads_to'), # Boundary → pattern

    # === Block 8 (Dale_law re-test) - from epistemic_edges.md ===
    (113, 'Abduction', 117, 'Deduction', 'triggers'),     # low_rank=10 hypothesis → test
    (113, 'Abduction', 125, 'Falsification', 'triggers'), # Hypothesis eventually falsified
    (117, 'Deduction', 121, 'Deduction', 'leads_to'),     # Sequential tests
    (121, 'Deduction', 125, 'Falsification', 'leads_to'), # Prediction → falsification
    (125, 'Falsification', 127, 'Causal', 'leads_to'),    # Failure → causal understanding

    # === Block 9-10 (low_rank+Dale) - from epistemic_edges.md ===
    (129, 'Abduction', 135, 'Causal', 'triggers'),        # Combined constraint → spectral model
    (135, 'Causal', 137, 'Deduction', 'leads_to'),        # Model → prediction
    (137, 'Deduction', 138, 'Falsification', 'leads_to'), # Test → failure
    (138, 'Falsification', 143, 'Induction', 'leads_to'), # Failures → pattern
    (138, 'Falsification', 144, 'Predictive', 'leads_to'), # Failures → design rule
    (144, 'Predictive', 145, 'Analogy/Transfer', 'leads_to'), # Rule → transfer attempt

    # === Block 11 (n_neurons=1000) - from epistemic_edges.md ===
    (145, 'Falsification', 148, 'Deduction', 'leads_to'), # Transfer failed → new hypothesis
    (145, 'Regime', 148, 'Meta-reasoning', 'leads_to'),   # Scale regime → strategy reassessment

    # === Cross-Block Edges (Knowledge Transfer) - from epistemic_edges.md ===
    (16, 'Induction', 17, 'Analogy/Transfer', 'triggers'),  # Block 1 principles → Block 2 transfer
    (22, 'Induction', 33, 'Analogy/Transfer', 'triggers'),  # Dale_law pattern → Block 3 transfer
    (32, 'Induction', 34, 'Analogy/Transfer', 'triggers'),  # Block 2 summary → Block 3 application
    (48, 'Causal', 53, 'Deduction', 'triggers'),            # Causal model → Block 4 prediction
    (64, 'Induction', 65, 'Analogy/Transfer', 'triggers'),  # n_frames insight → Block 5
    (80, 'Induction', 81, 'Analogy/Transfer', 'triggers'),  # Block 5 → Block 6 transfer
    (88, 'Predictive', 97, 'Analogy/Transfer', 'triggers'), # eff_rank rule → Block 7
    (96, 'Induction', 129, 'Analogy/Transfer', 'triggers'), # n_frames pattern → Block 9
    (112, 'Induction', 145, 'Analogy/Transfer', 'triggers'), # Block 7 → Block 11 transfer
]

# Block boundaries with detailed info
# Format: (start, end, label, info_dict)
blocks = [
    (1, 16, 'Block 1', {'regime': 'chaotic', 'E/I': '-', 'n_frames': 10000, 'n_neurons': 100, 'conn_rank': 'full', 'eff_rank': '31-35'}),
    (17, 32, 'Block 2', {'regime': 'chaotic', 'E/I': '0.5', 'n_frames': 10000, 'n_neurons': 100, 'conn_rank': 'full', 'eff_rank': '10'}),
    (33, 48, 'Block 3', {'regime': 'low_rank', 'E/I': '-', 'n_frames': 10000, 'n_neurons': 100, 'conn_rank': '20', 'eff_rank': '6'}),
    (49, 64, 'Block 4', {'regime': 'low_rank', 'E/I': '-', 'n_frames': 10000, 'n_neurons': 100, 'conn_rank': '50', 'eff_rank': '7'}),
    (65, 80, 'Block 5', {'regime': 'chaotic', 'E/I': '-', 'n_frames': 5000, 'n_neurons': 100, 'conn_rank': 'full', 'eff_rank': '20'}),
    (81, 96, 'Block 6', {'regime': 'chaotic', 'E/I': '-', 'n_frames': 2500, 'n_neurons': 100, 'conn_rank': 'full', 'eff_rank': '6'}),
    (97, 112, 'Block 7', {'regime': 'chaotic', 'E/I': '-', 'n_frames': 7500, 'n_neurons': 100, 'conn_rank': 'full', 'eff_rank': '25-29'}),
    (113, 128, 'Block 8', {'regime': 'chaotic', 'E/I': '0.5', 'n_frames': 10000, 'n_neurons': 100, 'conn_rank': 'full', 'eff_rank': '10-27'}),
    (129, 144, 'Block 9-10', {'regime': 'low_rank', 'E/I': '0.5', 'n_frames': 10000, 'n_neurons': 100, 'conn_rank': '20', 'eff_rank': '16'}),
    (145, 149, 'Block 11', {'regime': 'chaotic', 'E/I': '-', 'n_frames': 10000, 'n_neurons': 1000, 'conn_rank': 'full', 'eff_rank': '51-52'}),
]

def create_timeline():
    _, ax = plt.subplots(figsize=(24, 14))

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

    # Draw block backgrounds (alternating white and light gray) and add labels
    for block_idx, (start, end, label, info) in enumerate(blocks):
        color = '#f0f0f0' if block_idx % 2 == 1 else 'white'
        ax.axvspan(start - 0.5, end + 0.5, alpha=1.0, color=color)

        # Add block label at top with multiple lines of info
        left_x = start + 0.3
        top_y = len(modes) - 0.3

        # Format block info as multi-line text
        block_text = f"{label}\n{info['regime']}\nE/I={info['E/I']}\nT={info['n_frames']}\nN={info['n_neurons']}\nr={info['conn_rank']}\neff={info['eff_rank']}"
        ax.text(left_x, top_y, block_text, ha='left', va='bottom', fontsize=13,
                linespacing=0.85)

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

        # Size based on significance (increased)
        size = {'High': 300, 'Medium': 180, 'Low': 100}.get(significance, 180)

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
    ax.set_ylim(-0.5, len(modes) + 2.5)  # Extra space for block labels at top
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
    plt.savefig('signal_chaotic_1_Claude_epistemic_timeline.png', dpi=150, bbox_inches='tight',
                pad_inches=0.5)
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
