#!/usr/bin/env python3
"""
Generate interactive Plotly HTML visualizations for epistemic analysis.
336 iterations across 28 blocks (parallel run).

Generates:
  - assets/epistemic_timeline_interactive.html   (scatter timeline)
  - assets/epistemic_streamgraph_interactive.html (stacked area)
  - assets/epistemic_sankey_interactive.html      (Sankey flow diagram)
"""

import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
from scipy.ndimage import gaussian_filter1d

# ── Color scheme ──────────────────────────────────────────────

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

MODES = [
    'Induction', 'Boundary', 'Abduction', 'Deduction', 'Falsification',
    'Analogy', 'Meta-reasoning', 'Regime', 'Uncertainty',
    'Causal Chain', 'Predictive', 'Constraint',
]

# ── Block definitions ─────────────────────────────────────────

blocks = [
    (1, 12, 'B1: chaotic', 'chaotic', '35'),
    (13, 24, 'B2: low_rank', 'low_rank', '12-14'),
    (25, 36, 'B3: Dale', 'Dale', '12'),
    (37, 48, 'B4: n_types=4', 'n_types=4', '35-38'),
    (49, 60, 'B5: noise', 'noise', '42-90'),
    (61, 72, 'B6: n=200', 'n=200', '41-44'),
    (73, 84, 'B7: sparse 50%', 'sparse 50%', '21'),
    (85, 96, 'B8: sparse+noise', 'sparse+noise', '91'),
    (97, 108, 'B9: n=300', 'n=300', '44-47'),
    (109, 112, 'B10: n=300 2ep', 'n=300 2ep', '44-47'),
    (117, 128, 'B11: n=200 solved', 'n=200 solved', '43'),
    (129, 140, 'B12: n=600 (10k)', 'n=600 (10k)', '87'),
    (145, 156, 'B13: n=200 4types', 'n=200 4types', '43-44'),
    (165, 168, 'B14: recurrent', 'recurrent', '35'),
    (169, 180, 'B15: n=300 (30k)', 'n=300 (30k)', '48'),
    (181, 188, 'B16: n=600 (30k)', 'n=600 (30k)', '87'),
    (193, 204, 'B17: sparse (30k)', 'sparse 50% (30k)', '13'),
    (205, 216, 'B18: n=1000 (30k)', 'n=1000 (30k)', '144'),
    (217, 228, 'B19: g=3 n=100', 'g=3 n=100 (10k)', '26'),
    (229, 240, 'B20: g=3 n=200', 'g=3 n=200 (10k)', '31'),
    (241, 252, 'B21: g=3/200 30k', 'g=3 n=200 (30k)', '53-57'),
    (253, 264, 'B22: fill=80%', 'fill=80% (10k)', '36'),
    (265, 276, 'B23: fill=80% 30k', 'fill=80% (30k)', '48-49'),
    (289, 300, 'B24: fill=90%', 'fill=90% (10k)', '35-36'),
    (301, 312, 'B25: g=1 (10k)', 'g=1 (10k)', '5'),
    (313, 324, 'B26: g=1 (30k)', 'g=1 (30k)', '1'),
    (325, 336, 'B27: g=2 (10k)', 'g=2 (10k)', '17'),
    (337, 348, 'B28: g=2 (30k)', 'g=2 (30k)', '16'),
]

# ── Events: (iteration, mode, significance) ──────────────────

events = [
    # Block 1: Chaotic baseline (iters 1-12)
    (4, 'Boundary', 'Medium'), (5, 'Induction', 'High'), (5, 'Deduction', 'High'),
    (7, 'Boundary', 'Medium'), (8, 'Boundary', 'Medium'), (8, 'Uncertainty', 'Medium'),
    (9, 'Deduction', 'High'), (9, 'Falsification', 'High'), (9, 'Meta-reasoning', 'High'),
    (11, 'Induction', 'Medium'), (12, 'Induction', 'Medium'), (12, 'Boundary', 'Medium'),

    # Block 2: Low-rank (iters 13-24)
    (13, 'Analogy', 'High'), (13, 'Abduction', 'High'), (13, 'Regime', 'High'),
    (14, 'Falsification', 'High'), (15, 'Deduction', 'Medium'), (16, 'Boundary', 'Medium'),
    (17, 'Boundary', 'High'), (17, 'Falsification', 'High'), (17, 'Abduction', 'Medium'),
    (18, 'Analogy', 'High'), (18, 'Deduction', 'High'), (18, 'Meta-reasoning', 'High'),
    (19, 'Induction', 'High'), (19, 'Abduction', 'Medium'), (20, 'Analogy', 'Medium'),
    (20, 'Deduction', 'Medium'), (20, 'Boundary', 'Medium'), (21, 'Induction', 'High'),
    (21, 'Meta-reasoning', 'High'), (21, 'Causal Chain', 'High'),
    (22, 'Boundary', 'Medium'), (22, 'Uncertainty', 'Medium'),
    (23, 'Boundary', 'Medium'), (24, 'Falsification', 'Medium'), (24, 'Induction', 'Medium'),

    # Block 3: Dale's law (iters 25-36)
    (25, 'Analogy', 'High'), (25, 'Regime', 'High'), (26, 'Analogy', 'Medium'),
    (26, 'Deduction', 'Medium'), (27, 'Analogy', 'Medium'), (27, 'Deduction', 'Medium'),
    (28, 'Boundary', 'High'), (29, 'Boundary', 'High'), (29, 'Falsification', 'High'),
    (30, 'Boundary', 'High'), (30, 'Induction', 'High'),
    (31, 'Boundary', 'Medium'), (32, 'Falsification', 'Medium'),
    (33, 'Deduction', 'High'), (33, 'Induction', 'High'), (33, 'Causal Chain', 'High'),
    (34, 'Falsification', 'Medium'), (35, 'Induction', 'Medium'),
    (36, 'Deduction', 'Medium'), (36, 'Falsification', 'High'),

    # Block 4: Heterogeneous n_types=4 (iters 37-48)
    (37, 'Analogy', 'High'), (37, 'Abduction', 'High'), (37, 'Regime', 'High'),
    (39, 'Deduction', 'High'), (39, 'Induction', 'High'),
    (40, 'Boundary', 'Medium'), (41, 'Deduction', 'High'), (41, 'Meta-reasoning', 'Medium'),
    (42, 'Falsification', 'High'), (42, 'Abduction', 'Medium'),
    (43, 'Falsification', 'Medium'), (44, 'Deduction', 'High'),
    (44, 'Induction', 'High'), (44, 'Causal Chain', 'High'),
    (45, 'Boundary', 'Medium'), (47, 'Boundary', 'Medium'), (48, 'Falsification', 'High'),

    # Block 5: Noise (iters 49-60)
    (49, 'Analogy', 'High'), (49, 'Regime', 'High'), (50, 'Regime', 'Medium'),
    (51, 'Induction', 'High'), (51, 'Regime', 'Medium'), (52, 'Analogy', 'Medium'),
    (52, 'Falsification', 'Medium'), (53, 'Boundary', 'Medium'),
    (54, 'Deduction', 'High'), (55, 'Boundary', 'Medium'), (55, 'Induction', 'High'),
    (56, 'Deduction', 'High'), (56, 'Falsification', 'High'),
    (57, 'Boundary', 'High'), (57, 'Falsification', 'High'),
    (58, 'Deduction', 'High'), (58, 'Induction', 'High'),
    (58, 'Meta-reasoning', 'High'), (58, 'Causal Chain', 'Medium'),
    (60, 'Deduction', 'Medium'),

    # Block 6: Scale n=200 (iters 61-72)
    (61, 'Analogy', 'High'), (61, 'Regime', 'High'),
    (62, 'Boundary', 'High'), (63, 'Boundary', 'High'), (63, 'Induction', 'Medium'),
    (64, 'Deduction', 'Medium'), (64, 'Falsification', 'Medium'),
    (65, 'Boundary', 'Medium'), (66, 'Deduction', 'Medium'),
    (67, 'Deduction', 'High'), (68, 'Boundary', 'High'),
    (69, 'Boundary', 'Medium'), (70, 'Boundary', 'Medium'),
    (71, 'Falsification', 'High'), (72, 'Falsification', 'High'), (72, 'Induction', 'High'),

    # Block 7: Sparse 50% (iters 73-84)
    (73, 'Analogy', 'High'), (73, 'Regime', 'High'), (73, 'Abduction', 'High'),
    (74, 'Boundary', 'Medium'), (75, 'Boundary', 'Medium'),
    (76, 'Deduction', 'Medium'), (78, 'Boundary', 'Medium'),
    (79, 'Induction', 'High'), (79, 'Meta-reasoning', 'High'),
    (80, 'Deduction', 'Medium'), (82, 'Induction', 'High'), (82, 'Causal Chain', 'High'),
    (83, 'Deduction', 'Medium'), (84, 'Falsification', 'Medium'),

    # Block 8: Sparse + Noise (iters 85-96)
    (85, 'Analogy', 'High'), (85, 'Regime', 'High'), (85, 'Constraint', 'High'),
    (86, 'Induction', 'High'), (87, 'Induction', 'High'),
    (88, 'Falsification', 'High'), (88, 'Deduction', 'Medium'),
    (89, 'Deduction', 'Medium'), (90, 'Deduction', 'Medium'),
    (91, 'Falsification', 'High'), (92, 'Boundary', 'Medium'),
    (93, 'Falsification', 'High'), (95, 'Falsification', 'High'),
    (95, 'Causal Chain', 'High'), (96, 'Deduction', 'Medium'),
    (96, 'Meta-reasoning', 'High'),

    # Block 9: n=300 (iters 97-108)
    (97, 'Analogy', 'High'), (97, 'Regime', 'Medium'),
    (98, 'Boundary', 'Medium'), (99, 'Analogy', 'Medium'),
    (100, 'Deduction', 'Medium'), (100, 'Falsification', 'High'),
    (102, 'Deduction', 'High'), (103, 'Induction', 'High'), (103, 'Boundary', 'High'),
    (104, 'Falsification', 'High'), (105, 'Boundary', 'Medium'),
    (106, 'Induction', 'High'), (106, 'Causal Chain', 'High'),
    (107, 'Boundary', 'Medium'), (108, 'Falsification', 'High'), (108, 'Constraint', 'High'),

    # Block 10: n=300 2ep (iters 109-112)
    (109, 'Deduction', 'Medium'), (110, 'Falsification', 'High'),
    (111, 'Boundary', 'High'), (112, 'Induction', 'High'),
    (112, 'Deduction', 'High'), (112, 'Constraint', 'Medium'),

    # Block 11: n=200 Solved (iters 117-128)
    (117, 'Analogy', 'High'), (118, 'Analogy', 'High'),
    (119, 'Analogy', 'Medium'), (120, 'Analogy', 'Medium'),
    (121, 'Boundary', 'Medium'), (122, 'Deduction', 'Medium'),
    (123, 'Boundary', 'Medium'), (124, 'Induction', 'Medium'),
    (126, 'Induction', 'High'), (126, 'Deduction', 'High'),
    (127, 'Boundary', 'Medium'),
    (128, 'Falsification', 'High'),

    # Block 12: n=600 (10k) (iters 129-140)
    (129, 'Analogy', 'High'), (130, 'Analogy', 'High'),
    (131, 'Analogy', 'Medium'), (132, 'Analogy', 'Medium'),
    (133, 'Boundary', 'Medium'), (134, 'Deduction', 'Medium'),
    (135, 'Constraint', 'High'), (135, 'Falsification', 'High'),
    (136, 'Boundary', 'Medium'),
    (137, 'Induction', 'High'), (137, 'Causal Chain', 'High'),
    (138, 'Induction', 'Medium'), (139, 'Boundary', 'Medium'),
    (140, 'Deduction', 'Medium'),

    # Block 13: n=200 + 4 types (iters 145-156)
    (145, 'Analogy', 'High'), (146, 'Analogy', 'High'),
    (147, 'Analogy', 'Medium'), (148, 'Analogy', 'Medium'),
    (149, 'Induction', 'High'), (149, 'Causal Chain', 'High'),
    (150, 'Falsification', 'High'),
    (151, 'Boundary', 'Medium'),
    (152, 'Falsification', 'High'),
    (153, 'Deduction', 'Medium'), (154, 'Boundary', 'Medium'),
    (155, 'Induction', 'Medium'), (156, 'Deduction', 'Medium'),

    # Block 14: Recurrent Training (iters 165-168)
    (165, 'Deduction', 'High'),
    (166, 'Deduction', 'High'), (166, 'Causal Chain', 'High'),
    (167, 'Boundary', 'High'), (167, 'Induction', 'Medium'),
    (168, 'Falsification', 'High'),

    # Block 15: n=300 at 30k frames (iters 169-180)
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

    # Block 16: n=600 at 30k frames (iters 181-188)
    (181, 'Analogy', 'High'),
    (182, 'Boundary', 'Medium'),
    (183, 'Induction', 'High'), (183, 'Causal Chain', 'High'),
    (184, 'Deduction', 'High'),
    (185, 'Falsification', 'High'),
    (186, 'Induction', 'High'),
    (187, 'Boundary', 'Medium'),
    (188, 'Falsification', 'High'), (188, 'Induction', 'Medium'),

    # Block 17: Sparse 50% at 30k (iters 193-204)
    (193, 'Analogy', 'High'),
    (195, 'Falsification', 'High'),
    (197, 'Constraint', 'High'),
    (201, 'Induction', 'Medium'),
    (204, 'Meta-reasoning', 'High'),

    # Block 18: n=1000 at 30k (iters 205-216)
    (205, 'Boundary', 'High'),
    (209, 'Induction', 'High'),
    (212, 'Constraint', 'High'),
    (214, 'Induction', 'Medium'),
    (216, 'Causal Chain', 'High'),

    # Block 19: g=3 n=100 (iters 217-228)
    (217, 'Regime', 'High'),
    (219, 'Induction', 'High'),
    (221, 'Boundary', 'High'),
    (224, 'Falsification', 'High'),
    (228, 'Induction', 'High'),

    # Block 20: g=3 n=200 (iters 229-240)
    (229, 'Deduction', 'High'),
    (233, 'Constraint', 'High'),
    (236, 'Induction', 'Medium'),
    (240, 'Falsification', 'High'),

    # Block 21: g=3/n=200 at 30k (iters 241-252)
    (241, 'Analogy', 'High'),
    (245, 'Induction', 'High'),
    (248, 'Induction', 'High'),
    (251, 'Falsification', 'High'),
    (252, 'Induction', 'High'),

    # Block 22: fill=80% (iters 253-264)
    (253, 'Regime', 'High'),
    (255, 'Constraint', 'High'),
    (259, 'Induction', 'High'),
    (261, 'Induction', 'Medium'),
    (264, 'Deduction', 'High'),

    # Block 23: fill=80% at 30k (iters 265-276)
    (265, 'Falsification', 'High'),
    (266, 'Constraint', 'High'),
    (267, 'Induction', 'Medium'),
    (268, 'Constraint', 'High'),

    # Block 24: fill=90% (iters 289-300)
    (289, 'Regime', 'High'),
    (291, 'Induction', 'High'),
    (295, 'Constraint', 'High'),
    (298, 'Deduction', 'High'),

    # Block 25: g=1 (10k) (iters 301-312)
    (301, 'Regime', 'High'),
    (303, 'Constraint', 'High'),
    (307, 'Boundary', 'High'),
    (310, 'Induction', 'Medium'),

    # Block 26: g=1 (30k) (iters 313-324)
    (313, 'Falsification', 'High'),
    (317, 'Constraint', 'High'),
    (319, 'Causal Chain', 'High'),
    (322, 'Meta-reasoning', 'High'),

    # Block 27: g=2 (10k) (iters 325-336)
    (325, 'Regime', 'High'),
    (327, 'Induction', 'High'),
    (331, 'Causal Chain', 'High'),
    (335, 'Deduction', 'High'),

    # Block 28: g=2 (30k) (iters 337-348)
    (337, 'Analogy', 'High'),
    (341, 'Induction', 'High'),
    (343, 'Deduction', 'High'),
    (347, 'Causal Chain', 'High'),
]

# ── Causal edges ──────────────────────────────────────────────

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

    # Block 17
    (193, 'Analogy', 195, 'Falsification', 'leads_to'),
    (195, 'Falsification', 197, 'Constraint', 'triggers'),
    (197, 'Constraint', 201, 'Induction', 'leads_to'),
    (201, 'Induction', 204, 'Meta-reasoning', 'leads_to'),

    # Block 18
    (205, 'Boundary', 209, 'Induction', 'leads_to'),
    (209, 'Induction', 212, 'Constraint', 'leads_to'),
    (212, 'Constraint', 214, 'Induction', 'triggers'),
    (214, 'Induction', 216, 'Causal Chain', 'leads_to'),

    # Block 19
    (217, 'Regime', 219, 'Induction', 'triggers'),
    (219, 'Induction', 221, 'Boundary', 'leads_to'),
    (221, 'Boundary', 224, 'Falsification', 'triggers'),
    (224, 'Falsification', 228, 'Induction', 'leads_to'),

    # Block 20
    (229, 'Deduction', 233, 'Constraint', 'leads_to'),
    (233, 'Constraint', 236, 'Induction', 'triggers'),
    (236, 'Induction', 240, 'Falsification', 'triggers'),

    # Block 21
    (241, 'Analogy', 245, 'Induction', 'leads_to'),
    (245, 'Induction', 248, 'Induction', 'leads_to'),
    (248, 'Induction', 251, 'Falsification', 'triggers'),
    (251, 'Falsification', 252, 'Induction', 'leads_to'),

    # Block 22
    (253, 'Regime', 255, 'Constraint', 'triggers'),
    (255, 'Constraint', 259, 'Induction', 'leads_to'),
    (259, 'Induction', 261, 'Induction', 'leads_to'),
    (261, 'Induction', 264, 'Deduction', 'leads_to'),

    # Block 23
    (265, 'Falsification', 266, 'Constraint', 'triggers'),
    (266, 'Constraint', 267, 'Induction', 'leads_to'),
    (267, 'Induction', 268, 'Constraint', 'leads_to'),

    # Block 24
    (289, 'Regime', 291, 'Induction', 'triggers'),
    (291, 'Induction', 295, 'Constraint', 'leads_to'),
    (295, 'Constraint', 298, 'Deduction', 'leads_to'),

    # Block 25
    (301, 'Regime', 303, 'Constraint', 'triggers'),
    (303, 'Constraint', 307, 'Boundary', 'leads_to'),
    (307, 'Boundary', 310, 'Induction', 'leads_to'),

    # Block 26
    (313, 'Falsification', 317, 'Constraint', 'triggers'),
    (317, 'Constraint', 319, 'Causal Chain', 'leads_to'),
    (319, 'Causal Chain', 322, 'Meta-reasoning', 'leads_to'),

    # Block 27
    (325, 'Regime', 327, 'Induction', 'triggers'),
    (327, 'Induction', 331, 'Causal Chain', 'leads_to'),
    (331, 'Causal Chain', 335, 'Deduction', 'leads_to'),

    # Block 28
    (337, 'Analogy', 341, 'Induction', 'leads_to'),
    (341, 'Induction', 343, 'Deduction', 'leads_to'),
    (343, 'Deduction', 347, 'Causal Chain', 'leads_to'),

    # ── Cross-block edges ─────────────────────────────────────
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
    (67, 'Deduction', 117, 'Analogy', 'triggers'),
    (106, 'Induction', 117, 'Analogy', 'triggers'),
    (97, 'Analogy', 129, 'Analogy', 'triggers'),
    (39, 'Deduction', 145, 'Analogy', 'triggers'),
    (126, 'Induction', 145, 'Analogy', 'triggers'),
    (126, 'Induction', 165, 'Deduction', 'triggers'),
    (112, 'Induction', 169, 'Analogy', 'triggers'),
    (137, 'Induction', 181, 'Analogy', 'triggers'),
    (175, 'Induction', 183, 'Induction', 'triggers'),
    # Blocks 17-28 cross-block
    (186, 'Induction', 193, 'Analogy', 'triggers'),
    (197, 'Constraint', 205, 'Boundary', 'triggers'),
    (228, 'Induction', 229, 'Deduction', 'triggers'),
    (233, 'Constraint', 241, 'Analogy', 'triggers'),
    (197, 'Constraint', 253, 'Regime', 'triggers'),
    (264, 'Deduction', 265, 'Falsification', 'triggers'),
    (268, 'Constraint', 289, 'Regime', 'triggers'),
    (217, 'Regime', 301, 'Regime', 'triggers'),
    (303, 'Constraint', 313, 'Falsification', 'triggers'),
    (322, 'Meta-reasoning', 325, 'Regime', 'triggers'),
    (335, 'Deduction', 337, 'Analogy', 'triggers'),
]


# ── Helpers ───────────────────────────────────────────────────

def _block_for_iter(it):
    """Return block label for a given iteration."""
    for start, end, label, *_ in blocks:
        if start <= it <= end:
            return label
    return ''


def _hex_to_rgba(hex_color, alpha=0.4):
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f'rgba({r},{g},{b},{alpha})'


# ── Timeline ──────────────────────────────────────────────────

def create_timeline_html(outpath='assets/epistemic_timeline_interactive.html'):
    """Interactive scatter timeline of reasoning events."""

    mode_to_y = {mode: i for i, mode in enumerate(MODES)}

    fig = go.Figure()

    # Block background shading via shapes
    shapes = []
    annotations = []
    for idx, (start, end, label, regime, eff_rank) in enumerate(blocks):
        color = 'rgba(245,245,245,0.7)' if idx % 2 == 1 else 'rgba(255,255,255,0.0)'
        shapes.append(dict(
            type='rect', xref='x', yref='paper',
            x0=start - 0.5, x1=end + 0.5, y0=0, y1=1,
            fillcolor=color, line_width=0, layer='below',
        ))
        mid = (start + end) / 2
        annotations.append(dict(
            x=mid, y=len(MODES) - 0.3, yref='y',
            text=f'<b>{label}</b>', showarrow=False,
            font=dict(size=8, color='#999'),
        ))

    # One trace per mode for legend control
    for mode in MODES:
        pts = [(it, sig) for it, m, sig in events if m == mode]
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [mode_to_y[mode]] * len(pts)
        sizes = [18 if s == 'High' else 10 for _, s in pts]
        hover = [
            f'<b>{mode}</b> ({sig})<br>'
            f'Iter {it}<br>'
            f'{_block_for_iter(it)}<br>'
            f'{DEFINITIONS.get(mode, "")}'
            for it, sig in pts
        ]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='markers',
            marker=dict(size=sizes, color=COLORS[mode], opacity=0.85,
                        line=dict(width=0)),
            name=f'{mode}',
            text=hover, hovertemplate='%{text}<extra></extra>',
            legendgroup=mode,
        ))

    fig.update_layout(
        xaxis=dict(title='Iteration', range=[0, 355], dtick=24,
                    gridcolor='rgba(0,0,0,0.08)'),
        yaxis=dict(
            tickvals=list(range(len(MODES))),
            ticktext=MODES,
            range=[-0.5, len(MODES) - 0.3],
            gridcolor='rgba(0,0,0,0.05)',
        ),
        shapes=shapes,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12, family='Inter, system-ui, sans-serif'),
        width=1400, height=600,
        margin=dict(l=120, r=30, t=40, b=50),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        hovermode='closest',
    )

    fig.write_html(outpath, include_plotlyjs=True, full_html=True,
                   config={'displayModeBar': True, 'toImageButtonOptions': {
                       'format': 'png', 'filename': 'epistemic_timeline',
                       'height': 600, 'width': 1400, 'scale': 2}})
    print(f'Saved: {outpath}')


# ── Streamgraph ───────────────────────────────────────────────

def create_streamgraph_html(outpath='assets/epistemic_streamgraph_interactive.html'):
    """Interactive stacked-area (streamgraph) of reasoning modes."""

    max_iter = max(e[0] for e in events)
    bin_width = 6
    n_bins = (max_iter // bin_width) + 2
    bin_centers = np.arange(bin_width / 2, (n_bins + 0.5) * bin_width, bin_width)

    data = np.zeros((len(MODES), n_bins))
    for iteration, mode, significance in events:
        if mode in MODES:
            mode_idx = MODES.index(mode)
            bin_idx = (iteration - 1) // bin_width
            if 0 <= bin_idx < n_bins:
                weight = {'High': 2.0, 'Medium': 1.0, 'Low': 0.5}.get(significance, 1.0)
                data[mode_idx, bin_idx] += weight

    sigma = 1.2
    data_smooth = np.array([gaussian_filter1d(row, sigma) for row in data])

    # Compute wiggle baseline (ThemeRiver)
    n_layers = len(MODES)
    baseline = np.zeros(n_bins)
    for i in range(n_layers):
        baseline -= (n_layers - i - 0.5) * data_smooth[i]
    baseline /= n_layers

    y_stack = np.vstack([baseline, baseline + np.cumsum(data_smooth, axis=0)])

    fig = go.Figure()

    # Add invisible baseline trace first
    fig.add_trace(go.Scatter(
        x=bin_centers.tolist(), y=y_stack[0].tolist(),
        mode='lines', line=dict(width=0, color='rgba(0,0,0,0)'),
        showlegend=False, hoverinfo='skip',
    ))

    # Add areas bottom-to-top, all filling to previous trace
    for i, mode in enumerate(MODES):
        fig.add_trace(go.Scatter(
            x=bin_centers.tolist(), y=y_stack[i + 1].tolist(),
            mode='lines', line=dict(width=0.5, color='white'),
            fillcolor=_hex_to_rgba(COLORS[mode], 0.8),
            fill='tonexty',
            name=mode, legendgroup=mode,
            hovertemplate=f'<b>{mode}</b><br>Iter ~%{{x:.0f}}<br>Weight: %{{y:.1f}}<extra></extra>',
        ))

    # Block boundaries
    for idx, (start, end, label, *_) in enumerate(blocks):
        fig.add_vline(x=end + 0.5, line_width=0.5, line_dash='dot',
                      line_color='rgba(0,0,0,0.15)')

    fig.update_layout(
        xaxis=dict(title='Iteration', range=[0, 355], gridcolor='rgba(0,0,0,0.08)'),
        yaxis=dict(title='Reasoning Activity', gridcolor='rgba(0,0,0,0.05)'),
        plot_bgcolor='#fafafa',
        paper_bgcolor='white',
        font=dict(size=12, family='Inter, system-ui, sans-serif'),
        width=1400, height=450,
        margin=dict(l=60, r=30, t=40, b=50),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        hovermode='x unified',
    )

    fig.write_html(outpath, include_plotlyjs=True, full_html=True,
                   config={'displayModeBar': True, 'toImageButtonOptions': {
                       'format': 'png', 'filename': 'epistemic_streamgraph',
                       'height': 450, 'width': 1400, 'scale': 2}})
    print(f'Saved: {outpath}')


# ── Sankey ────────────────────────────────────────────────────

def create_sankey_html(outpath='assets/epistemic_sankey_interactive.html'):
    """Interactive Plotly Sankey diagram of reasoning mode transitions."""

    transitions = Counter()
    for from_iter, from_mode, to_iter, to_mode, edge_type in edges:
        transitions[(from_mode, to_mode)] += 1

    mode_counts = Counter(e[1] for e in events)

    all_modes_in_edges = set()
    for (from_mode, to_mode) in transitions:
        all_modes_in_edges.add(from_mode)
        all_modes_in_edges.add(to_mode)

    mode_flow = Counter()
    for (from_mode, to_mode), count in transitions.items():
        mode_flow[from_mode] += count
        mode_flow[to_mode] += count

    node_labels = sorted(all_modes_in_edges, key=lambda m: -mode_flow[m])
    node_idx = {mode: i for i, mode in enumerate(node_labels)}

    node_colors = [COLORS.get(m, '#888') for m in node_labels]
    node_hover = [
        f"<b>{m}</b><br>{DEFINITIONS.get(m, '')}<br>"
        f"Events: {mode_counts.get(m, 0)}<br>"
        f"Connections: {mode_flow[m]}"
        for m in node_labels
    ]

    sources, targets, values, link_colors, link_labels = [], [], [], [], []

    for (from_mode, to_mode), count in sorted(transitions.items(), key=lambda x: -x[1]):
        sources.append(node_idx[from_mode])
        targets.append(node_idx[to_mode])
        values.append(count)
        link_colors.append(_hex_to_rgba(COLORS.get(from_mode, '#888'), 0.4))
        link_labels.append(f'{from_mode} → {to_mode}: {count}')

    fig = go.Figure(data=[go.Sankey(
        arrangement='snap',
        node=dict(
            pad=20, thickness=25,
            line=dict(color='white', width=2),
            label=node_labels, color=node_colors,
            customdata=node_hover,
            hovertemplate='%{customdata}<extra></extra>',
        ),
        link=dict(
            source=sources, target=targets, value=values,
            color=link_colors, label=link_labels,
            hovertemplate='%{label}<extra></extra>',
        ),
    )])

    fig.update_layout(
        title=dict(
            text='Epistemic Flow: Reasoning Mode Transitions<br>'
                 f'<sub>336 iterations · 28 blocks · {len(events)} events · '
                 f'{len(edges)} edges</sub>',
            font=dict(size=20),
        ),
        font=dict(size=14, family='Inter, system-ui, sans-serif'),
        width=1200, height=800,
        margin=dict(l=30, r=30, t=80, b=30),
        paper_bgcolor='white',
    )

    fig.write_html(outpath, include_plotlyjs=True, full_html=True,
                   config={'displayModeBar': True, 'toImageButtonOptions': {
                       'format': 'png', 'filename': 'epistemic_sankey',
                       'height': 800, 'width': 1200, 'scale': 2}})
    print(f'Saved: {outpath}')


# ── Static fallback PNGs ─────────────────────────────────────

def create_static_pngs():
    """Generate static PNG fallbacks using matplotlib."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    mode_to_y = {mode: i for i, mode in enumerate(MODES)}

    # Timeline
    fig, ax = plt.subplots(figsize=(28, 8))
    for idx, (start, end, label, regime, eff_rank) in enumerate(blocks):
        color = '#f8f8f8' if idx % 2 == 1 else 'white'
        ax.axvspan(start - 0.5, end + 0.5, alpha=1.0, color=color, zorder=0)
    for iteration, mode, significance in events:
        if mode not in mode_to_y:
            continue
        y = mode_to_y[mode]
        color = COLORS.get(mode, '#333333')
        size = {'High': 150, 'Medium': 80, 'Low': 40}.get(significance, 80)
        ax.scatter(iteration, y, c=color, s=size, alpha=0.9,
                   edgecolors='white', linewidth=0.5, zorder=3)
    ax.set_xlim(0, 355)
    ax.set_ylim(-0.5, len(MODES) - 0.5)
    ax.set_yticks(range(len(MODES)))
    ax.set_yticklabels(MODES, fontsize=14)
    ax.set_xlabel('Iteration', fontsize=18)
    ax.tick_params(axis='x', labelsize=14)
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    legend_elements = [plt.scatter([], [], c=COLORS[m], s=100, label=m) for m in MODES]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, ncol=4)
    plt.tight_layout()
    plt.savefig('assets/epistemic_timeline.png', dpi=150, bbox_inches='tight', facecolor='white')
    print('Saved: assets/epistemic_timeline.png')
    plt.close()

    # Streamgraph
    fig, ax = plt.subplots(figsize=(28, 6))
    max_iter = max(e[0] for e in events)
    bin_width = 6
    n_bins = (max_iter // bin_width) + 2
    iterations = np.arange(bin_width / 2, (n_bins + 0.5) * bin_width, bin_width)
    data = np.zeros((len(MODES), n_bins))
    for iteration, mode, significance in events:
        if mode in MODES:
            mode_idx = MODES.index(mode)
            bin_idx = (iteration - 1) // bin_width
            if 0 <= bin_idx < n_bins:
                weight = {'High': 2.0, 'Medium': 1.0, 'Low': 0.5}.get(significance, 1.0)
                data[mode_idx, bin_idx] += weight
    sigma = 1.2
    data_smooth = np.array([gaussian_filter1d(row, sigma) for row in data])
    n_layers = len(MODES)
    baseline = np.zeros(n_bins)
    for i in range(n_layers):
        baseline -= (n_layers - i - 0.5) * data_smooth[i]
    baseline /= n_layers
    y_stack = np.vstack([baseline, baseline + np.cumsum(data_smooth, axis=0)])
    for i, mode in enumerate(MODES):
        color = COLORS.get(mode, '#333333')
        ax.fill_between(iterations, y_stack[i], y_stack[i + 1],
                         color=color, alpha=0.8, label=mode,
                         edgecolor='white', linewidth=0.3)
    for idx, (start, end, label, *_) in enumerate(blocks):
        ax.axvline(x=end + 0.5, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    ax.set_xlim(0, 355)
    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_ylabel('Reasoning Activity', fontsize=14)
    ax.set_facecolor('#fafafa')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10, frameon=False)
    plt.tight_layout()
    plt.savefig('assets/epistemic_streamgraph.png', dpi=150, bbox_inches='tight', facecolor='white')
    print('Saved: assets/epistemic_streamgraph.png')
    plt.close()


# ── Main ──────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs('assets', exist_ok=True)
    n_events = len(events)
    n_edges = len(edges)
    mode_counts = Counter(e[1] for e in events)

    print(f'Total events: {n_events}')
    print(f'Total edges: {n_edges}')
    for mode, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
        print(f'  {mode}: {count}')

    print('\n--- Interactive Timeline ---')
    create_timeline_html()

    print('\n--- Interactive Streamgraph ---')
    create_streamgraph_html()

    print('\n--- Interactive Sankey ---')
    create_sankey_html()

    print('\n--- Static PNGs ---')
    create_static_pngs()

    print('\nDone!')
