#!/usr/bin/env python3
"""
Visualize epistemic reasoning timeline from signal_landscape_Claude experiment.
112 iterations across 10 blocks (parallel run).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from collections import Counter
from matplotlib.gridspec import GridSpec

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
# 112 iterations, 10 blocks (parallel run)
events = [
    # Block 1: Chaotic baseline (iters 1-12)
    (4, 'Boundary', 'Medium'),       # lr_W=1E-3 lower boundary mapped
    (5, 'Induction', 'High'),        # lr_W=4E-3 sweet spot identified
    (5, 'Deduction', 'High'),        # predicted optimal confirmed
    (7, 'Boundary', 'Medium'),       # lr_W=1.5E-3 convergence threshold
    (8, 'Boundary', 'Medium'),       # lr_W=8E-3 upper range explored
    (8, 'Uncertainty', 'Medium'),    # stochastic variation at lr_W=8E-3
    (9, 'Deduction', 'High'),        # lr=2E-4 predicted to help
    (9, 'Falsification', 'High'),    # lr=2E-4 degrades dynamics
    (9, 'Meta-reasoning', 'High'),   # switch from lr_W to other dimensions
    (11, 'Induction', 'Medium'),     # L1=1E-6 best dynamics at low lr_W
    (12, 'Induction', 'Medium'),     # batch_size=16 quality trade identified
    (12, 'Boundary', 'Medium'),      # batch_size limit probed

    # Block 2: Low-rank (iters 13-24)
    (13, 'Analogy', 'High'),         # transfer lr_W=4E-3 from block 1
    (13, 'Abduction', 'High'),       # eff_rank=13 hypothesis for dynamics gap
    (13, 'Regime', 'High'),          # eff_rank=13 vs 35, sub-critical (0.952)
    (14, 'Falsification', 'High'),   # factorization=True hurts
    (15, 'Deduction', 'Medium'),     # factorization at 8E-3 still underperforms
    (16, 'Boundary', 'Medium'),      # lr_W=2E-3 lower boundary in low_rank
    (17, 'Boundary', 'High'),        # lr_W=5E-3 catastrophic failure (0.385)
    (17, 'Falsification', 'High'),   # lr_W=5E-3 without factorization fails
    (17, 'Abduction', 'Medium'),     # stochastic failure vs unstable point?
    (18, 'Analogy', 'High'),         # L1 transfer from block 1
    (18, 'Deduction', 'High'),       # L1=1E-6 should help dynamics
    (18, 'Meta-reasoning', 'High'),  # switch from factorization to L1
    (19, 'Induction', 'High'),       # lr_W=3E-3 BETTER dynamics (surprise)
    (19, 'Abduction', 'Medium'),     # lower eff_rank needs gentler lr_W
    (20, 'Analogy', 'Medium'),       # convergence boundary transfer test
    (20, 'Deduction', 'Medium'),     # boundary should hold in low_rank
    (20, 'Boundary', 'Medium'),      # lr_W=1.5E-3 partial in low_rank
    (21, 'Induction', 'High'),       # BREAKTHROUGH: 3E-3 + L1=1E-6 -> 0.996
    (21, 'Meta-reasoning', 'High'),  # recombination strategy
    (21, 'Causal Chain', 'High'),    # lr_W + L1 combined effect
    (22, 'Boundary', 'Medium'),      # lr_W=3.5E-3 dynamics cliff
    (22, 'Uncertainty', 'Medium'),   # stochastic variation
    (23, 'Boundary', 'Medium'),      # lr_W=2.5E-3 below optimal zone
    (24, 'Falsification', 'Medium'), # batch=16 surprise: 0.997 (challenges)
    (24, 'Induction', 'Medium'),     # batch=16 + L1=1E-6 synergy

    # Block 3: Dale's law (iters 25-36)
    (25, 'Analogy', 'High'),         # transfer from block 1
    (25, 'Regime', 'High'),          # Dale reduces eff_rank 35->12
    (26, 'Analogy', 'Medium'),       # L1 transfer from block 2
    (26, 'Deduction', 'Medium'),     # L1=1E-6 marginal in Dale
    (27, 'Analogy', 'Medium'),       # lr_W=3E-3 from block 2
    (27, 'Deduction', 'Medium'),     # 3E-3 underperforms in Dale
    (28, 'Boundary', 'High'),        # lr_W=6E-3 catastrophic (0.555)
    (29, 'Boundary', 'High'),        # lr_W=5E-3 fails (0.458)
    (29, 'Falsification', 'High'),   # cliff is tighter than expected
    (30, 'Boundary', 'High'),        # lr_W=5E-3 second run (0.455) reproducible
    (30, 'Induction', 'High'),       # cliff is reproducible, not stochastic
    (31, 'Boundary', 'Medium'),      # lr_W=3.5E-3 good balance
    (32, 'Falsification', 'Medium'), # L1 can't rescue 6E-3
    (33, 'Deduction', 'High'),       # 4.5E-3 predicted safe -> confirmed
    (33, 'Induction', 'High'),       # safe range expanded to [3.5, 4.5E-3]
    (33, 'Causal Chain', 'High'),    # Dale -> eff_rank -> lr_W cliff
    (34, 'Falsification', 'Medium'), # batch=16 hurts Dale connectivity
    (35, 'Induction', 'Medium'),     # L1 negligible at lr_W=3.5E-3 in Dale
    (36, 'Deduction', 'Medium'),     # lr=2E-4 tested in Dale
    (36, 'Falsification', 'High'),   # lr=2E-4 WORKS -> challenges principle 1

    # Block 4: Heterogeneous n_types=4 (iters 37-48)
    (37, 'Analogy', 'High'),         # transfer from block 1
    (37, 'Abduction', 'High'),       # embedding failure hypothesis
    (37, 'Regime', 'High'),          # n_types=4 dual-objective
    (39, 'Deduction', 'High'),       # lr_emb=1E-3 predicted to fix embedding
    (39, 'Induction', 'High'),       # FULL convergence: conn 0.9996
    (40, 'Boundary', 'Medium'),      # lr_W=1E-3 fails completely
    (41, 'Deduction', 'High'),       # lr_W=5E-3 predicted to work
    (41, 'Meta-reasoning', 'Medium'), # dual-objective strategy
    (42, 'Falsification', 'High'),   # lr_emb=1E-3 overshoots at low lr_W
    (42, 'Abduction', 'Medium'),     # lr_W/lr_emb coupling hypothesis
    (43, 'Falsification', 'Medium'), # lr_W=3E-3 underperforms for n_types=4
    (44, 'Deduction', 'High'),       # L1=1E-6 critical for embedding
    (44, 'Induction', 'High'),       # L1=1E-6 extends beyond low_rank
    (44, 'Causal Chain', 'High'),    # L1 -> embedding mechanism
    (45, 'Boundary', 'Medium'),      # lr_W=6E-3 degrades embedding
    (47, 'Boundary', 'Medium'),      # lr_W=4.5E-3 intermediate
    (48, 'Falsification', 'High'),   # batch=16 degrades heterogeneous

    # Block 5: Noise (iters 49-60)
    (49, 'Analogy', 'High'),         # transfer from block 1
    (49, 'Regime', 'High'),          # noise=0.5 eff_rank=84
    (50, 'Regime', 'Medium'),        # noise=1.0 eff_rank=90
    (51, 'Induction', 'High'),       # noise increases eff_rank pattern
    (51, 'Regime', 'Medium'),        # noise=0.1 eff_rank=42
    (52, 'Analogy', 'Medium'),       # L1 principle test with noise
    (52, 'Falsification', 'Medium'), # L1=1E-6 NOT beneficial n_types=1+noise
    (53, 'Boundary', 'Medium'),      # lr_W=8E-3 works at noise=0.5
    (54, 'Deduction', 'High'),       # lr_W=6E-3 at noise=1.0 overshoots
    (55, 'Boundary', 'Medium'),      # lr_W=2E-3 at noise=0.1, best rollout
    (55, 'Induction', 'High'),       # low noise preserves rollout
    (56, 'Deduction', 'High'),       # lr=2E-4 safe at eff_rank=84
    (56, 'Falsification', 'High'),   # contradicts principle 1 at high eff_rank
    (57, 'Boundary', 'High'),        # lr_W=1E-2 degrades dynamics severely
    (57, 'Falsification', 'High'),   # upper lr_W boundary established
    (58, 'Deduction', 'High'),       # lr_W=2E-3 best at noise=1.0
    (58, 'Induction', 'High'),       # inverse lr_W-noise relation
    (58, 'Meta-reasoning', 'High'),  # noise->eff_rank->lr_W insight
    (58, 'Causal Chain', 'Medium'),  # noise -> eff_rank -> lr_W tolerance
    (60, 'Deduction', 'Medium'),     # lr=2E-4 + lr_W=8E-3 combination

    # Block 6: Scale n=200 (iters 61-72)
    (61, 'Analogy', 'High'),         # transfer from block 1
    (61, 'Regime', 'High'),          # n=200 eff_rank=43
    (62, 'Boundary', 'High'),        # lr_W=2E-3 fails at n=200 (0.575)
    (63, 'Boundary', 'High'),        # lr_W=8E-3 best conn but worst dynamics
    (63, 'Induction', 'Medium'),     # trade-off amplified at n=200
    (64, 'Deduction', 'Medium'),     # L1=1E-6 test at n=200
    (64, 'Falsification', 'Medium'), # L1=1E-6 REDUCES connectivity at n=200
    (65, 'Boundary', 'Medium'),      # lr_W=6E-3 steep dynamics trade-off
    (66, 'Deduction', 'Medium'),     # lr_W=5E-3 confirmed near sweet spot
    (67, 'Deduction', 'High'),       # lr=2E-4 safe at n=200, best conn
    (68, 'Boundary', 'High'),        # lr_W=3E-3 partial, boundary at ~3.5E-3
    (69, 'Boundary', 'Medium'),      # lr_W=5.5E-3 past optimal
    (70, 'Boundary', 'Medium'),      # lr_W=4.5E-3 just below threshold
    (71, 'Falsification', 'High'),   # L1=1E-6 SEVERELY degrades at n=200
    (72, 'Falsification', 'High'),   # lr=3E-4 works at n=200 (contradicts P1)
    (72, 'Induction', 'High'),       # lr=3E-4 BEST dynamics; n=200 widens lr

    # Block 7: Sparse 50% (iters 73-84)
    (73, 'Analogy', 'High'),         # transfer from block 1
    (73, 'Regime', 'High'),          # eff_rank=21, subcritical rho=0.746
    (73, 'Abduction', 'High'),       # subcritical spectral radius hypothesis
    (74, 'Boundary', 'Medium'),      # lr_W=6E-3 best of first batch
    (75, 'Boundary', 'Medium'),      # lr_W=2E-3 worst
    (76, 'Deduction', 'Medium'),     # L1=1E-6 neutral in sparse
    (78, 'Boundary', 'Medium'),      # lr_W=1E-2 best at 1 epoch
    (79, 'Induction', 'High'),       # n_epochs=2 beats all 1-epoch configs
    (79, 'Meta-reasoning', 'High'),  # shift from lr_W to n_epochs dimension
    (80, 'Deduction', 'Medium'),     # lr=2E-4 marginally worse at eff_rank=21
    (82, 'Induction', 'High'),       # lr_W=1E-2 + 2ep best (0.466)
    (82, 'Causal Chain', 'High'),    # training capacity key bottleneck
    (83, 'Deduction', 'Medium'),     # 3 epochs diminishing returns
    (84, 'Falsification', 'Medium'), # Dale/low_rank params don't transfer

    # Block 8: Sparse + Noise (iters 85-96)
    (85, 'Analogy', 'High'),         # transfer noise principle from block 5
    (85, 'Regime', 'High'),          # eff_rank 21->91 but still subcritical
    (85, 'Constraint', 'High'),      # conn=0.489 structural data limit
    (86, 'Induction', 'High'),       # complete lr_W insensitivity
    (87, 'Induction', 'High'),       # confirmed: lr_W=8E-3 same as 2E-3
    (88, 'Falsification', 'High'),   # n_epochs=1 = 2, noise removes dependency
    (88, 'Deduction', 'Medium'),     # tests principle about n_epochs in sparse
    (89, 'Deduction', 'Medium'),     # lr_W=1E-2 same plateau
    (90, 'Deduction', 'Medium'),     # L1=1E-6 irrelevant at plateau
    (91, 'Falsification', 'High'),   # two-phase training fails
    (92, 'Boundary', 'Medium'),      # lr_W=1.5E-2 still no cliff
    (93, 'Falsification', 'High'),   # aug_loop=200 zero effect
    (95, 'Falsification', 'High'),   # recurrent training catastrophic
    (95, 'Causal Chain', 'High'),    # multi-step rollout + subcritical = fail
    (96, 'Deduction', 'Medium'),     # batch=16 safe n_types=1 (confirmed)
    (96, 'Meta-reasoning', 'High'),  # recognize structural limit

    # Block 9: n=300 (iters 97-108)
    (97, 'Analogy', 'High'),         # transfer from block 6 n=200
    (97, 'Regime', 'Medium'),        # eff_rank=48, spectral_radius=1.03
    (98, 'Boundary', 'Medium'),      # lr_W=7E-3 underperforms
    (99, 'Analogy', 'Medium'),       # n=200 optimal (lr_W=5E-3) transfer test
    (100, 'Deduction', 'Medium'),    # principle test: dynamics cliff at 8E-3?
    (100, 'Falsification', 'High'),  # cliff NOT at 8E-3 for n=300
    (102, 'Deduction', 'High'),      # lr=2E-4 at lr_W=8E-3 boosts conn +16%
    (103, 'Induction', 'High'),      # lr_W=1E-2 best conn (0.805)
    (103, 'Boundary', 'High'),       # upper lr_W range for n=300
    (104, 'Falsification', 'High'),  # L1=1E-6 NOT harmful at n=300
    (105, 'Boundary', 'Medium'),     # lr_W=1.2E-2 slightly past optimum
    (106, 'Induction', 'High'),      # n_epochs=2 BREAKTHROUGH (+10.6%)
    (106, 'Causal Chain', 'High'),   # n -> training-capacity -> convergence
    (107, 'Boundary', 'Medium'),     # lr_W=1.5E-2 dynamics cliff begins
    (108, 'Falsification', 'High'),  # lr=3E-4 degrades at high lr_W
    (108, 'Constraint', 'High'),     # lr/lr_W interaction at n=300

    # Block 10: n=300, n_epochs=2 baseline (iters 109-112)
    (109, 'Deduction', 'Medium'),    # reproduce baseline (0.893 confirmed)
    (110, 'Falsification', 'High'),  # 3ep doesn't help conn (slightly worse)
    (111, 'Boundary', 'High'),       # lr_W=1.2E-2 confirms conn cliff >1E-2
    (112, 'Induction', 'High'),      # L1=1E-6 boosts dynamics +6.8%
    (112, 'Deduction', 'High'),      # principle refined: harmful only n<=200
    (112, 'Constraint', 'Medium'),   # conn ceiling ~0.89 at 10k frames
]

# Causal edges from signal_landscape_Claude_epistemic_edges.md
edges = [
    # Block 1
    (4, 'Boundary', 5, 'Deduction', 'leads_to'),
    (5, 'Deduction', 8, 'Induction', 'leads_to'),
    (5, 'Induction', 9, 'Deduction', 'triggers'),
    (8, 'Boundary', 11, 'Induction', 'leads_to'),
    (9, 'Meta-reasoning', 11, 'Induction', 'triggers'),

    # Block 2
    (13, 'Abduction', 18, 'Deduction', 'triggers'),
    (13, 'Regime', 19, 'Abduction', 'triggers'),
    (17, 'Boundary', 19, 'Induction', 'leads_to'),
    (18, 'Deduction', 21, 'Induction', 'leads_to'),
    (19, 'Induction', 21, 'Induction', 'leads_to'),
    (14, 'Falsification', 24, 'Induction', 'refines'),
    (22, 'Boundary', 23, 'Boundary', 'leads_to'),

    # Block 3
    (25, 'Regime', 28, 'Boundary', 'triggers'),
    (28, 'Boundary', 29, 'Boundary', 'leads_to'),
    (29, 'Boundary', 30, 'Induction', 'leads_to'),
    (30, 'Induction', 33, 'Deduction', 'leads_to'),
    (29, 'Falsification', 32, 'Falsification', 'triggers'),
    (34, 'Falsification', 36, 'Deduction', 'leads_to'),

    # Block 4
    (37, 'Abduction', 39, 'Deduction', 'triggers'),
    (39, 'Deduction', 41, 'Deduction', 'leads_to'),
    (39, 'Induction', 44, 'Deduction', 'triggers'),
    (42, 'Falsification', 44, 'Causal Chain', 'triggers'),
    (41, 'Deduction', 48, 'Falsification', 'leads_to'),

    # Block 5
    (49, 'Regime', 53, 'Boundary', 'triggers'),
    (51, 'Induction', 55, 'Boundary', 'leads_to'),
    (53, 'Boundary', 57, 'Boundary', 'leads_to'),
    (54, 'Deduction', 58, 'Deduction', 'leads_to'),
    (55, 'Induction', 58, 'Induction', 'leads_to'),

    # Block 6
    (62, 'Boundary', 63, 'Boundary', 'leads_to'),
    (63, 'Induction', 66, 'Deduction', 'leads_to'),
    (64, 'Falsification', 71, 'Falsification', 'leads_to'),
    (67, 'Deduction', 72, 'Falsification', 'triggers'),

    # Block 7
    (73, 'Regime', 79, 'Meta-reasoning', 'triggers'),
    (74, 'Boundary', 78, 'Boundary', 'leads_to'),
    (79, 'Induction', 82, 'Induction', 'leads_to'),
    (82, 'Causal Chain', 83, 'Deduction', 'leads_to'),

    # Block 8
    (85, 'Regime', 88, 'Falsification', 'triggers'),
    (86, 'Induction', 93, 'Falsification', 'leads_to'),
    (88, 'Falsification', 95, 'Falsification', 'triggers'),
    (85, 'Constraint', 96, 'Meta-reasoning', 'leads_to'),

    # Block 9
    (97, 'Analogy', 103, 'Induction', 'leads_to'),
    (100, 'Falsification', 103, 'Boundary', 'triggers'),
    (103, 'Induction', 106, 'Induction', 'leads_to'),
    (106, 'Causal Chain', 108, 'Constraint', 'leads_to'),

    # Block 10
    (109, 'Deduction', 110, 'Falsification', 'leads_to'),
    (106, 'Induction', 112, 'Deduction', 'triggers'),

    # Cross-block
    (11, 'Induction', 18, 'Deduction', 'triggers'),        # B1->B2: L1 insight
    (5, 'Induction', 25, 'Analogy', 'triggers'),           # B1->B3: lr_W transfer
    (21, 'Induction', 26, 'Analogy', 'triggers'),          # B2->B3: L1 transfer
    (19, 'Induction', 27, 'Analogy', 'triggers'),          # B2->B3: lr_W=3E-3
    (5, 'Induction', 37, 'Analogy', 'triggers'),           # B1->B4: lr_W transfer
    (5, 'Induction', 49, 'Analogy', 'triggers'),           # B1->B5: lr_W transfer
    (44, 'Induction', 52, 'Analogy', 'triggers'),          # B4->B5: L1 embedding
    (9, 'Falsification', 56, 'Deduction', 'triggers'),     # B1->B5: lr tolerance
    (5, 'Induction', 61, 'Analogy', 'triggers'),           # B1->B6: lr_W transfer
    (5, 'Induction', 73, 'Analogy', 'triggers'),           # B1->B7: lr_W transfer
    (51, 'Induction', 85, 'Analogy', 'triggers'),          # B5->B8: noise principle
    (67, 'Deduction', 97, 'Analogy', 'triggers'),          # B6->B9: n=200->n=300
    (79, 'Induction', 106, 'Induction', 'triggers'),       # B7->B9: epochs insight
    (82, 'Causal Chain', 106, 'Causal Chain', 'triggers'),  # B7->B9: training cap.
]

# Block boundaries
blocks = [
    (1, 12, 'Block 1', {'regime': 'chaotic', 'eff_rank': '35'}),
    (13, 24, 'Block 2', {'regime': 'low_rank', 'eff_rank': '12-14'}),
    (25, 36, 'Block 3', {'regime': 'Dale', 'eff_rank': '12'}),
    (37, 48, 'Block 4', {'regime': 'n_types=4', 'eff_rank': '35-38'}),
    (49, 60, 'Block 5', {'regime': 'noise', 'eff_rank': '42-90'}),
    (61, 72, 'Block 6', {'regime': 'n=200', 'eff_rank': '41-44'}),
    (73, 84, 'Block 7', {'regime': 'sparse 50%', 'eff_rank': '21'}),
    (85, 96, 'Block 8', {'regime': 'sparse+noise', 'eff_rank': '91'}),
    (97, 108, 'Block 9', {'regime': 'n=300', 'eff_rank': '44-47'}),
    (109, 112, 'Block 10', {'regime': 'n=300 2ep', 'eff_rank': '44-47'}),
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

    x_min, x_max = 0, 117

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

    summary_text = f"""signal_landscape_Claude (parallel)

{total_events} reasoning instances
112 iterations, 10 blocks
24 principles discovered

Deduction validation: 72%
Transfer success: 65%

Mode counts:
  Induction: {mode_counts.get('Induction', 0)}
  Abduction: {mode_counts.get('Abduction', 0)}
  Deduction: {mode_counts.get('Deduction', 0)}
  Falsification: {mode_counts.get('Falsification', 0)}
  Analogy: {mode_counts.get('Analogy', 0)}
  Boundary: {mode_counts.get('Boundary', 0)}

{total_edges} causal edges
7 key causal chains"""

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

    caption_text = ("Epistemic flow of LLM-guided scientific exploration across "
                   "10 exploration blocks. Deduction acts as the central hub---"
                   "nearly all reasoning modes converge through it. The dominant "
                   "Deduction->Falsification pathway demonstrates genuine hypothesis "
                   "testing. Blocks 7-8 (sparse) show increased Constraint and "
                   "Meta-reasoning as the system recognizes structural limits.")

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
    outflow = {n: sum(d['weight'] for _, _, d in G.out_edges(n, data=True)) for n in G.nodes()}
    inflow = {n: sum(d['weight'] for _, _, d in G.in_edges(n, data=True)) for n in G.nodes()}

    # Custom positions - arrange in layers
    layers = {
        0: ['Induction', 'Boundary', 'Regime'],
        1: ['Abduction', 'Deduction', 'Causal Chain'],
        2: ['Falsification', 'Analogy', 'Meta-reasoning', 'Constraint'],
        3: ['Predictive', 'Uncertainty'],
    }

    pos = {}
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
    fig, ax = plt.subplots(figsize=(28, 10))

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

    ax.set_xlim(0, 117)
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
    fig, ax = plt.subplots(figsize=(20, 6))

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

    ax.set_xlim(0, 117)
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
