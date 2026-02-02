#!/usr/bin/env python3
"""
Generate interactive Sankey HTML diagram for epistemic analysis.
Uses Plotly for rich interactivity (hover, drag, zoom).
188 iterations, 16 blocks.
"""

import plotly.graph_objects as go
from collections import Counter

# Color scheme matching plot_landscape_epistemic.py
COLORS = {
    'Induction': '#2ecc71',
    'Abduction': '#9b59b6',
    'Deduction': '#3498db',
    'Falsification': '#e74c3c',
    'Analogy': '#f39c12',
    'Boundary': '#1abc9c',
    'Meta-reasoning': '#e91e63',
    'Regime': '#795548',
    'Uncertainty': '#607d8b',
    'Causal Chain': '#00bcd4',
    'Predictive': '#8bc34a',
    'Constraint': '#ff5722',
}

DEFINITIONS = {
    'Induction': 'observations → pattern',
    'Abduction': 'observation → hypothesis',
    'Deduction': 'hypothesis → prediction',
    'Falsification': 'prediction failed → refine',
    'Analogy': 'cross-regime transfer',
    'Boundary': 'limit-finding',
    'Meta-reasoning': 'strategy adaptation',
    'Regime': 'phase identification',
    'Uncertainty': 'stochasticity awareness',
    'Causal Chain': 'multi-step causation',
    'Predictive': 'quantitative modeling',
    'Constraint': 'parameter relationships',
}

# Import edges from plot_landscape_epistemic.py data
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

    # Block 11
    (117, 'Analogy', 126, 'Induction', 'leads_to'),
    (126, 'Induction', 128, 'Falsification', 'triggers'),

    # Block 12
    (129, 'Analogy', 135, 'Constraint', 'leads_to'),
    (135, 'Constraint', 137, 'Induction', 'triggers'),
    (137, 'Causal Chain', 140, 'Deduction', 'leads_to'),

    # Block 13
    (145, 'Analogy', 149, 'Induction', 'leads_to'),
    (149, 'Induction', 150, 'Falsification', 'triggers'),
    (150, 'Falsification', 152, 'Falsification', 'triggers'),

    # Block 14
    (165, 'Deduction', 166, 'Deduction', 'leads_to'),
    (166, 'Causal Chain', 167, 'Boundary', 'triggers'),
    (167, 'Boundary', 168, 'Falsification', 'leads_to'),

    # Block 15
    (169, 'Analogy', 170, 'Falsification', 'leads_to'),
    (170, 'Falsification', 172, 'Falsification', 'triggers'),
    (172, 'Meta-reasoning', 175, 'Induction', 'leads_to'),
    (175, 'Induction', 176, 'Falsification', 'triggers'),
    (177, 'Induction', 178, 'Boundary', 'triggers'),

    # Block 16
    (181, 'Analogy', 183, 'Induction', 'leads_to'),
    (183, 'Causal Chain', 184, 'Deduction', 'leads_to'),
    (184, 'Deduction', 185, 'Falsification', 'triggers'),
    (185, 'Falsification', 186, 'Induction', 'leads_to'),
    (186, 'Induction', 188, 'Falsification', 'triggers'),

    # Cross-block
    (11, 'Induction', 18, 'Deduction', 'triggers'),
    (5, 'Induction', 25, 'Analogy', 'triggers'),
    (21, 'Induction', 26, 'Analogy', 'triggers'),
    (19, 'Induction', 27, 'Analogy', 'triggers'),
    (5, 'Induction', 37, 'Analogy', 'triggers'),
    (5, 'Induction', 49, 'Analogy', 'triggers'),
    (44, 'Induction', 52, 'Analogy', 'triggers'),
    (9, 'Falsification', 56, 'Deduction', 'triggers'),
    (5, 'Induction', 61, 'Analogy', 'triggers'),
    (5, 'Induction', 73, 'Analogy', 'triggers'),
    (51, 'Induction', 85, 'Analogy', 'triggers'),
    (67, 'Deduction', 97, 'Analogy', 'triggers'),
    (79, 'Induction', 106, 'Induction', 'triggers'),
    (82, 'Causal Chain', 106, 'Causal Chain', 'triggers'),
    (67, 'Deduction', 117, 'Analogy', 'triggers'),    # Block 6 → Block 11
    (106, 'Induction', 117, 'Analogy', 'triggers'),   # Block 9 epochs → Block 11
    (97, 'Analogy', 129, 'Analogy', 'triggers'),      # Block 9 n=300 → Block 12
    (39, 'Deduction', 145, 'Analogy', 'triggers'),    # Block 4 hetero → Block 13
    (126, 'Induction', 145, 'Analogy', 'triggers'),   # Block 11 n=200 → Block 13
    (126, 'Induction', 165, 'Deduction', 'triggers'),  # Block 11 → Block 14
    (112, 'Induction', 169, 'Analogy', 'triggers'),   # Block 10 → Block 15
    (137, 'Induction', 181, 'Analogy', 'triggers'),   # Block 12 → Block 16
    (175, 'Induction', 183, 'Induction', 'triggers'),  # Block 15 → Block 16
]

# Events data for mode counts
events = [
    # Block 1
    (4, 'Boundary', 'Medium'), (5, 'Induction', 'High'), (5, 'Deduction', 'High'),
    (7, 'Boundary', 'Medium'), (8, 'Boundary', 'Medium'), (8, 'Uncertainty', 'Medium'),
    (9, 'Deduction', 'High'), (9, 'Falsification', 'High'), (9, 'Meta-reasoning', 'High'),
    (11, 'Induction', 'Medium'), (12, 'Induction', 'Medium'), (12, 'Boundary', 'Medium'),

    # Block 2
    (13, 'Analogy', 'High'), (13, 'Abduction', 'High'), (13, 'Regime', 'High'),
    (14, 'Falsification', 'High'), (15, 'Deduction', 'Medium'), (16, 'Boundary', 'Medium'),
    (17, 'Boundary', 'High'), (17, 'Falsification', 'High'), (17, 'Abduction', 'Medium'),
    (18, 'Analogy', 'High'), (18, 'Deduction', 'High'), (18, 'Meta-reasoning', 'High'),
    (19, 'Induction', 'High'), (19, 'Abduction', 'Medium'), (20, 'Analogy', 'Medium'),
    (20, 'Deduction', 'Medium'), (20, 'Boundary', 'Medium'), (21, 'Induction', 'High'),
    (21, 'Meta-reasoning', 'High'), (21, 'Causal Chain', 'High'),
    (22, 'Boundary', 'Medium'), (22, 'Uncertainty', 'Medium'),
    (23, 'Boundary', 'Medium'), (24, 'Falsification', 'Medium'), (24, 'Induction', 'Medium'),

    # Block 3
    (25, 'Analogy', 'High'), (25, 'Regime', 'High'), (26, 'Analogy', 'Medium'),
    (26, 'Deduction', 'Medium'), (27, 'Analogy', 'Medium'), (27, 'Deduction', 'Medium'),
    (28, 'Boundary', 'High'), (29, 'Boundary', 'High'), (29, 'Falsification', 'High'),
    (30, 'Boundary', 'High'), (30, 'Induction', 'High'),
    (31, 'Boundary', 'Medium'), (32, 'Falsification', 'Medium'),
    (33, 'Deduction', 'High'), (33, 'Induction', 'High'), (33, 'Causal Chain', 'High'),
    (34, 'Falsification', 'Medium'), (35, 'Induction', 'Medium'),
    (36, 'Deduction', 'Medium'), (36, 'Falsification', 'High'),

    # Block 4
    (37, 'Analogy', 'High'), (37, 'Abduction', 'High'), (37, 'Regime', 'High'),
    (39, 'Deduction', 'High'), (39, 'Induction', 'High'),
    (40, 'Boundary', 'Medium'), (41, 'Deduction', 'High'), (41, 'Meta-reasoning', 'Medium'),
    (42, 'Falsification', 'High'), (42, 'Abduction', 'Medium'),
    (43, 'Falsification', 'Medium'), (44, 'Deduction', 'High'),
    (44, 'Induction', 'High'), (44, 'Causal Chain', 'High'),
    (45, 'Boundary', 'Medium'), (47, 'Boundary', 'Medium'), (48, 'Falsification', 'High'),

    # Block 5
    (49, 'Analogy', 'High'), (49, 'Regime', 'High'), (50, 'Regime', 'Medium'),
    (51, 'Induction', 'High'), (51, 'Regime', 'Medium'), (52, 'Analogy', 'Medium'),
    (52, 'Falsification', 'Medium'), (53, 'Boundary', 'Medium'),
    (54, 'Deduction', 'High'), (55, 'Boundary', 'Medium'), (55, 'Induction', 'High'),
    (56, 'Deduction', 'High'), (56, 'Falsification', 'High'),
    (57, 'Boundary', 'High'), (57, 'Falsification', 'High'),
    (58, 'Deduction', 'High'), (58, 'Induction', 'High'),
    (58, 'Meta-reasoning', 'High'), (58, 'Causal Chain', 'Medium'),
    (60, 'Deduction', 'Medium'),

    # Block 6
    (61, 'Analogy', 'High'), (61, 'Regime', 'High'),
    (62, 'Boundary', 'High'), (63, 'Boundary', 'High'), (63, 'Induction', 'Medium'),
    (64, 'Deduction', 'Medium'), (64, 'Falsification', 'Medium'),
    (65, 'Boundary', 'Medium'), (66, 'Deduction', 'Medium'),
    (67, 'Deduction', 'High'), (68, 'Boundary', 'High'),
    (69, 'Boundary', 'Medium'), (70, 'Boundary', 'Medium'),
    (71, 'Falsification', 'High'), (72, 'Falsification', 'High'), (72, 'Induction', 'High'),

    # Block 7
    (73, 'Analogy', 'High'), (73, 'Regime', 'High'), (73, 'Abduction', 'High'),
    (74, 'Boundary', 'Medium'), (75, 'Boundary', 'Medium'),
    (76, 'Deduction', 'Medium'), (78, 'Boundary', 'Medium'),
    (79, 'Induction', 'High'), (79, 'Meta-reasoning', 'High'),
    (80, 'Deduction', 'Medium'), (82, 'Induction', 'High'), (82, 'Causal Chain', 'High'),
    (83, 'Deduction', 'Medium'), (84, 'Falsification', 'Medium'),

    # Block 8
    (85, 'Analogy', 'High'), (85, 'Regime', 'High'), (85, 'Constraint', 'High'),
    (86, 'Induction', 'High'), (87, 'Induction', 'High'),
    (88, 'Falsification', 'High'), (88, 'Deduction', 'Medium'),
    (89, 'Deduction', 'Medium'), (90, 'Deduction', 'Medium'),
    (91, 'Falsification', 'High'), (92, 'Boundary', 'Medium'),
    (93, 'Falsification', 'High'), (95, 'Falsification', 'High'),
    (95, 'Causal Chain', 'High'), (96, 'Deduction', 'Medium'),
    (96, 'Meta-reasoning', 'High'),

    # Block 9
    (97, 'Analogy', 'High'), (97, 'Regime', 'Medium'),
    (98, 'Boundary', 'Medium'), (99, 'Analogy', 'Medium'),
    (100, 'Deduction', 'Medium'), (100, 'Falsification', 'High'),
    (102, 'Deduction', 'High'), (103, 'Induction', 'High'), (103, 'Boundary', 'High'),
    (104, 'Falsification', 'High'), (105, 'Boundary', 'Medium'),
    (106, 'Induction', 'High'), (106, 'Causal Chain', 'High'),
    (107, 'Boundary', 'Medium'), (108, 'Falsification', 'High'), (108, 'Constraint', 'High'),

    # Block 10
    (109, 'Deduction', 'Medium'), (110, 'Falsification', 'High'),
    (111, 'Boundary', 'High'), (112, 'Induction', 'High'),
    (112, 'Deduction', 'High'), (112, 'Constraint', 'Medium'),

    # Block 11 --- n=200 Solved
    (117, 'Analogy', 'High'), (118, 'Analogy', 'High'),
    (119, 'Analogy', 'Medium'), (120, 'Analogy', 'Medium'),
    (121, 'Boundary', 'Medium'), (122, 'Deduction', 'Medium'),
    (123, 'Boundary', 'Medium'), (124, 'Induction', 'Medium'),
    (126, 'Induction', 'High'), (126, 'Deduction', 'High'),
    (127, 'Boundary', 'Medium'),
    (128, 'Falsification', 'High'),

    # Block 12 --- n=600 (10k)
    (129, 'Analogy', 'High'), (130, 'Analogy', 'High'),
    (131, 'Analogy', 'Medium'), (132, 'Analogy', 'Medium'),
    (133, 'Boundary', 'Medium'), (134, 'Deduction', 'Medium'),
    (135, 'Constraint', 'High'), (135, 'Falsification', 'High'),
    (136, 'Boundary', 'Medium'),
    (137, 'Induction', 'High'), (137, 'Causal Chain', 'High'),
    (138, 'Induction', 'Medium'), (139, 'Boundary', 'Medium'),
    (140, 'Deduction', 'Medium'),

    # Block 13 --- n=200 + 4 types
    (145, 'Analogy', 'High'), (146, 'Analogy', 'High'),
    (147, 'Analogy', 'Medium'), (148, 'Analogy', 'Medium'),
    (149, 'Induction', 'High'), (149, 'Causal Chain', 'High'),
    (150, 'Falsification', 'High'),
    (151, 'Boundary', 'Medium'),
    (152, 'Falsification', 'High'),
    (153, 'Deduction', 'Medium'), (154, 'Boundary', 'Medium'),
    (155, 'Induction', 'Medium'), (156, 'Deduction', 'Medium'),

    # Block 14 --- Recurrent Training
    (165, 'Deduction', 'High'),
    (166, 'Deduction', 'High'), (166, 'Causal Chain', 'High'),
    (167, 'Boundary', 'High'), (167, 'Induction', 'Medium'),
    (168, 'Falsification', 'High'),

    # Block 15 --- n=300 at 30k frames
    (169, 'Analogy', 'High'), (169, 'Regime', 'High'),
    (170, 'Falsification', 'High'),
    (171, 'Analogy', 'High'),
    (172, 'Falsification', 'High'), (172, 'Meta-reasoning', 'High'),
    (173, 'Boundary', 'Medium'), (174, 'Deduction', 'Medium'),
    (175, 'Induction', 'High'),
    (176, 'Falsification', 'High'),
    (177, 'Induction', 'High'), (177, 'Causal Chain', 'Medium'),
    (178, 'Boundary', 'High'),
    (179, 'Deduction', 'Medium'), (180, 'Induction', 'Medium'),

    # Block 16 --- n=600 at 30k frames
    (181, 'Analogy', 'High'),
    (182, 'Boundary', 'Medium'),
    (183, 'Induction', 'High'), (183, 'Causal Chain', 'High'),
    (184, 'Deduction', 'High'),
    (185, 'Falsification', 'High'),
    (186, 'Induction', 'High'),
    (187, 'Boundary', 'Medium'),
    (188, 'Falsification', 'High'), (188, 'Induction', 'Medium'),
]


def create_sankey_html():
    """Create interactive Plotly Sankey diagram."""

    # Count transitions between modes
    transitions = Counter()
    for from_iter, from_mode, to_iter, to_mode, edge_type in edges:
        transitions[(from_mode, to_mode)] += 1

    # Count mode occurrences for node sizing info
    mode_counts = Counter(e[1] for e in events)

    # Build node list (only modes that appear in edges)
    all_modes_in_edges = set()
    for (from_mode, to_mode) in transitions:
        all_modes_in_edges.add(from_mode)
        all_modes_in_edges.add(to_mode)

    # Order modes by flow volume (most connected first)
    mode_flow = Counter()
    for (from_mode, to_mode), count in transitions.items():
        mode_flow[from_mode] += count
        mode_flow[to_mode] += count

    node_labels = sorted(all_modes_in_edges, key=lambda m: -mode_flow[m])
    node_idx = {mode: i for i, mode in enumerate(node_labels)}

    # Node colors and hover text
    node_colors = [COLORS.get(m, '#888') for m in node_labels]
    node_hover = [
        f"<b>{m}</b><br>{DEFINITIONS.get(m, '')}<br>"
        f"Events: {mode_counts.get(m, 0)}<br>"
        f"Connections: {mode_flow[m]}"
        for m in node_labels
    ]

    # Build links
    sources = []
    targets = []
    values = []
    link_colors = []
    link_labels = []

    for (from_mode, to_mode), count in sorted(transitions.items(), key=lambda x: -x[1]):
        sources.append(node_idx[from_mode])
        targets.append(node_idx[to_mode])
        values.append(count)

        # Link color: semi-transparent version of source color
        base_color = COLORS.get(from_mode, '#888')
        # Convert hex to rgba
        r = int(base_color[1:3], 16)
        g = int(base_color[3:5], 16)
        b = int(base_color[5:7], 16)
        link_colors.append(f'rgba({r},{g},{b},0.4)')

        link_labels.append(f'{from_mode} → {to_mode}: {count}')

    n_events = len(events)
    n_edges = len(edges)
    n_blocks = 16

    fig = go.Figure(data=[go.Sankey(
        arrangement='snap',
        node=dict(
            pad=20,
            thickness=25,
            line=dict(color='white', width=2),
            label=node_labels,
            color=node_colors,
            customdata=node_hover,
            hovertemplate='%{customdata}<extra></extra>',
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            label=link_labels,
            hovertemplate='%{label}<extra></extra>',
        ),
    )])

    fig.update_layout(
        title=dict(
            text='Epistemic Flow: LLM Reasoning Mode Transitions<br>'
                 f'<sub>signal_landscape_Claude — 188 iterations, {n_blocks} blocks, '
                 f'{n_events} events, {n_edges} edges</sub>',
            font=dict(size=20),
        ),
        font=dict(size=14, family='Inter, system-ui, sans-serif'),
        width=1200,
        height=800,
        margin=dict(l=30, r=30, t=80, b=30),
        paper_bgcolor='white',
    )

    fig.write_html(
        'signal_landscape_Claude_epistemic_sankey.html',
        include_plotlyjs=True,
        full_html=True,
        config={
            'displayModeBar': True,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'epistemic_sankey',
                'height': 800,
                'width': 1200,
                'scale': 2,
            },
        },
    )
    print(f"Saved: signal_landscape_Claude_epistemic_sankey.html")
    print(f"  {n_events} events, {n_edges} edges, {n_blocks} blocks")


if __name__ == '__main__':
    create_sankey_html()
