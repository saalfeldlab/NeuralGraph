#!/usr/bin/env python3
"""
Visualize epistemic reasoning timeline from signal_chaotic_1_Claude experiment.
Color-coded by reasoning mode type.
Based on epistemic_detailed.md and epistemic_edges.md (2026-01-18).
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Define color scheme for reasoning modes
COLORS = {
    'Induction': '#2ecc71',        # Green - pattern extraction
    'Abduction': '#9b59b6',        # Purple - causal hypothesis
    'Deduction': '#3498db',        # Blue - prediction testing
    'Falsification': '#e74c3c',    # Red - hypothesis rejection
    'Analogy': '#f39c12',          # Orange - cross-regime transfer
    'Boundary': '#1abc9c',         # Teal - limit finding
    'Meta-reasoning': '#e91e63',   # Pink - strategy adaptation
    'Regime': '#795548',           # Brown - phase identification
    'Uncertainty': '#607d8b',      # Gray - stochasticity awareness
    'Causal Chain': '#00bcd4',     # Cyan - causal chain
    'Predictive': '#8bc34a',       # Light green - predictive modeling
    'Constraint': '#ff5722',       # Deep orange - constraint propagation
}

# Definitions for each reasoning mode (for legend)
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

# Events from epistemic_detailed.md
# Format: (iteration, mode, significance)
events = [
    # === Induction: 47 instances ===
    (6, 'Induction', 'Medium'),    # 5 consecutive converged
    (9, 'Induction', 'Medium'),    # 8 consecutive converged
    (16, 'Induction', 'High'),     # Block 1 summary (4 patterns)
    (32, 'Induction', 'High'),     # Block 2 summary (3 patterns)
    (48, 'Induction', 'High'),     # Block 3 summary (3 patterns)
    (64, 'Induction', 'High'),     # Block 4 summary (2 patterns)
    (80, 'Induction', 'High'),     # Block 5 summary (2 patterns)
    (96, 'Induction', 'High'),     # Block 6 summary (2 patterns)
    (111, 'Induction', 'High'),    # Block 7 summary (2 patterns)
    (127, 'Induction', 'High'),    # Block 8 summary (3 patterns)
    (144, 'Induction', 'High'),    # Block 10 summary (2 patterns)
    (160, 'Induction', 'High'),    # Block 11 summary (3 patterns)
    (176, 'Induction', 'High'),    # Block 12 summary (3 patterns)
    (192, 'Induction', 'High'),    # Block 13 summary (2 patterns)
    (208, 'Induction', 'High'),    # Block 14 summary (2 patterns)
    (224, 'Induction', 'High'),    # Block 15 summary (4 patterns)
    (240, 'Induction', 'High'),    # Block 16 summary (4 patterns)
    (255, 'Induction', 'High'),    # Block 17 summary (4 patterns)

    # === Abduction: 31 instances ===
    (17, 'Abduction', 'High'),     # eff_rank hypothesis
    (33, 'Abduction', 'High'),     # connectivity structure hypothesis
    (34, 'Abduction', 'Medium'),   # Higher lr_W needed
    (41, 'Abduction', 'Medium'),   # Initialization stochasticity
    (42, 'Abduction', 'High'),     # Results not reproducible
    (49, 'Abduction', 'High'),     # rank ≠ eff_rank
    (53, 'Abduction', 'Medium'),   # W structure matching
    (65, 'Abduction', 'High'),     # n_frames → complexity
    (81, 'Abduction', 'High'),     # Data minimum exists
    (97, 'Abduction', 'Medium'),   # Stochastic intermediate
    (112, 'Abduction', 'Medium'),  # High eff_rank possible
    (113, 'Abduction', 'Medium'),  # eff_rank stochastic
    (128, 'Abduction', 'Medium'),  # Combined constraints
    (141, 'Abduction', 'Medium'),  # Over-parameterization helps
    (145, 'Abduction', 'High'),    # Scale requires different params
    (149, 'Abduction', 'High'),    # MLP training bottleneck
    (163, 'Abduction', 'Medium'),  # Combined improvement
    (171, 'Abduction', 'Medium'),  # Very low L1 helps
    (177, 'Abduction', 'High'),    # E/I destabilizes large networks
    (178, 'Abduction', 'Medium'),  # Stochastic initialization
    (187, 'Abduction', 'Medium'),  # Higher lr_W helps
    (193, 'Abduction', 'Medium'),  # Connectivity dominates
    (198, 'Abduction', 'Medium'),  # MLP lr critical
    (203, 'Abduction', 'High'),    # HIGH stochasticity
    (225, 'Abduction', 'High'),    # Data quantity scales eff_rank
    (232, 'Abduction', 'Medium'),  # lr_W floor exists
    (241, 'Abduction', 'Medium'),  # High data stabilizes
    (243, 'Abduction', 'Medium'),  # Lower lr_W optimal
    (246, 'Abduction', 'Medium'),  # L1 affects generalization

    # === Deduction: 58 instances ===
    (4, 'Deduction', 'High'),      # lr_W boundary prediction ✓
    (5, 'Deduction', 'Medium'),    # lr_W extreme test ✗
    (6, 'Deduction', 'Medium'),    # lr_W=4E-2 ✗
    (8, 'Deduction', 'Medium'),    # L1=1E-4 ✓
    (9, 'Deduction', 'Medium'),    # L1=5E-4 ✓
    (10, 'Deduction', 'High'),     # L1=1E-3 extreme ✓
    (13, 'Deduction', 'High'),     # factorization ✗
    (18, 'Deduction', 'High'),     # Higher lr_W Dale ✓
    (20, 'Deduction', 'Medium'),   # Continue lr_W ✓
    (21, 'Deduction', 'High'),     # lr_W=5E-2 ✓
    (24, 'Deduction', 'High'),     # lr_W upper ✓
    (25, 'Deduction', 'Medium'),   # factorization Dale ✗
    (34, 'Deduction', 'High'),     # 20x lr_W ✓
    (36, 'Deduction', 'Medium'),   # factorization low_rank ✗
    (40, 'Deduction', 'High'),     # lr_W=9E-2 ✓
    (42, 'Deduction', 'High'),     # Robustness test ✗
    (49, 'Deduction', 'High'),     # rank=50 ✗
    (53, 'Deduction', 'High'),     # factorization rank=50 ✓
    (56, 'Deduction', 'Medium'),   # low_rank=80 ✓
    (65, 'Deduction', 'High'),     # Block 1 at 5k ✗
    (74, 'Deduction', 'High'),     # lr_W=2E-2 ✓
    (77, 'Deduction', 'Medium'),   # lr_W=6E-2 ✓
    (81, 'Deduction', 'High'),     # n_frames=2500 ✗
    (91, 'Deduction', 'Medium'),   # factorization 2500 ✗
    (98, 'Deduction', 'High'),     # lr_W=3E-2 7500 ✓
    (101, 'Deduction', 'Medium'),  # lr_W=1E-1 ✓
    (102, 'Deduction', 'High'),    # lr_W=1.5E-1 ✓
    (112, 'Deduction', 'High'),    # Block 2 optimized ✓
    (122, 'Deduction', 'Medium'),  # eff_rank=10 optimize ✓
    (127, 'Deduction', 'High'),    # L1=1E-6 ✓
    (128, 'Deduction', 'Medium'),  # low_rank+Dale ✗
    (141, 'Deduction', 'High'),    # low_rank=30 ✓
    (142, 'Deduction', 'Medium'),  # low_rank=40 ✓
    (143, 'Deduction', 'Medium'),  # low_rank=50 ✓
    (144, 'Deduction', 'High'),    # low_rank=60 ✓
    (145, 'Deduction', 'High'),    # Block 1 n=1000 ✗
    (149, 'Deduction', 'High'),    # Higher lr ✓
    (156, 'Deduction', 'High'),    # lr_W=8E-3 ✓
    (161, 'Deduction', 'High'),    # 2x training ✗
    (171, 'Deduction', 'High'),    # L1=1E-7 ✓
    (175, 'Deduction', 'Medium'),  # lr=5E-3 ✓
    (178, 'Deduction', 'Medium'),  # Dynamics recover ✓
    (187, 'Deduction', 'High'),    # lr_W=1.5E-1 ✓
    (198, 'Deduction', 'High'),    # lr=1E-2 breakthrough ✓
    (203, 'Deduction', 'High'),    # Robustness test ✗
    (209, 'Deduction', 'High'),    # Block 1 at 50k ✓
    (218, 'Deduction', 'Medium'),  # lr_W lower ✓
    (219, 'Deduction', 'Medium'),  # lr_W=5E-4 ✓
    (220, 'Deduction', 'Medium'),  # lr_W=2.5E-4 ✓
    (225, 'Deduction', 'High'),    # n=1000+50k breakthrough ✓
    (226, 'Deduction', 'Medium'),  # L1=1E-5 ✗
    (232, 'Deduction', 'Medium'),  # lr_W=2E-3 ✓
    (241, 'Deduction', 'High'),    # Dale+50k ✓
    (243, 'Deduction', 'High'),    # lr_W=1E-2 ✓
    (244, 'Deduction', 'Medium'),  # lr_W=5E-4 ✓
    (251, 'Deduction', 'High'),    # Recombine optimal ✓
    (253, 'Deduction', 'High'),    # Robustness test ✓
    (255, 'Deduction', 'High'),    # Final optimization ✓

    # === Falsification: 34 instances ===
    (10, 'Falsification', 'High'),    # L1 upper bound
    (13, 'Falsification', 'High'),    # factorization fails
    (25, 'Falsification', 'High'),    # factorization Dale fails
    (26, 'Falsification', 'High'),    # L1=1E-4 fails Dale
    (35, 'Falsification', 'High'),    # lr_W upper low_rank
    (36, 'Falsification', 'Medium'),  # factorization low_rank
    (39, 'Falsification', 'Medium'),  # lr critical
    (42, 'Falsification', 'High'),    # Stochasticity dominates
    (49, 'Falsification', 'High'),    # rank ≠ eff_rank
    (91, 'Falsification', 'Medium'),  # factorization 2500
    (102, 'Falsification', 'High'),   # lr_W upper 7500
    (107, 'Falsification', 'Medium'), # L1 boundary
    (108, 'Falsification', 'Medium'), # L1 confirms
    (161, 'Falsification', 'High'),   # More training doesn't help
    (162, 'Falsification', 'High'),   # factorization n=1000
    (166, 'Falsification', 'Medium'), # lr_W upper n=1000
    (167, 'Falsification', 'High'),   # lr upper catastrophic
    (169, 'Falsification', 'Medium'), # L1 boundary n=1000
    (177, 'Falsification', 'High'),   # E/I collapse
    (183, 'Falsification', 'Medium'), # lr upper Dale n=1000
    (185, 'Falsification', 'High'),   # factorization any n=1000
    (188, 'Falsification', 'Medium'), # lr_W upper Dale n=1000
    (195, 'Falsification', 'High'),   # factorization low_rank n=1000
    (199, 'Falsification', 'Medium'), # lr upper low_rank n=1000
    (200, 'Falsification', 'Medium'), # lr_W upper low_rank n=1000
    (201, 'Falsification', 'Medium'), # lr intermediate
    (202, 'Falsification', 'Medium'), # lr_W lower boundary
    (203, 'Falsification', 'High'),   # HIGH stochasticity
    (206, 'Falsification', 'Medium'), # lr lower boundary
    (226, 'Falsification', 'High'),   # L1=1E-5 too strong
    (232, 'Falsification', 'Medium'), # lr_W lower boundary
    (236, 'Falsification', 'Medium'), # lr upper boundary
    (239, 'Falsification', 'Medium'), # L1=1E-9 degrades
    (244, 'Falsification', 'High'),   # lr_W lower Dale 50k

    # === Analogy/Transfer: 19 instances ===
    (17, 'Analogy', 'High'),     # Block 1→2 failed
    (33, 'Analogy', 'High'),     # Block 1→3 failed
    (49, 'Analogy', 'Medium'),   # Block 3→4 partial
    (65, 'Analogy', 'Medium'),   # Block 1→5 failed
    (81, 'Analogy', 'High'),     # Block 5→6 failed
    (97, 'Analogy', 'Medium'),   # Block 1→7 partial
    (112, 'Analogy', 'High'),    # Block 2→8 breakthrough!
    (128, 'Analogy', 'Medium'),  # Block 8→9 partial
    (129, 'Analogy', 'Medium'),  # Block 8→10 valid
    (145, 'Analogy', 'High'),    # Block 1→11 failed
    (161, 'Analogy', 'Medium'),  # Block 11→12 held
    (177, 'Analogy', 'Medium'),  # Block 12→13 partial
    (193, 'Analogy', 'Medium'),  # Block 12→14 starting point
    (209, 'Analogy', 'High'),    # Block 1→15 perfect transfer!
    (225, 'Analogy', 'High'),    # Block 12→16 breakthrough
    (241, 'Analogy', 'High'),    # Block 16→17 success
    (242, 'Analogy', 'Medium'),  # Direct transfer better
    (251, 'Analogy', 'Medium'),  # Combined best
    (253, 'Analogy', 'Medium'),  # Robustness validated

    # === Boundary: 42 instances ===
    (4, 'Boundary', 'Medium'),     # lr_W approaching
    (5, 'Boundary', 'Medium'),     # lr_W=2E-2
    (6, 'Boundary', 'Medium'),     # lr_W=4E-2
    (9, 'Boundary', 'Medium'),     # L1=5E-4
    (10, 'Boundary', 'High'),      # L1=1E-3 above
    (14, 'Boundary', 'Medium'),    # L1=7E-4
    (15, 'Boundary', 'Medium'),    # L1=6E-4
    (16, 'Boundary', 'High'),      # L1 upper precise
    (24, 'Boundary', 'High'),      # Dale lr_W upper
    (30, 'Boundary', 'Medium'),    # Dale lr lower
    (35, 'Boundary', 'High'),      # low_rank lr_W upper
    (38, 'Boundary', 'Medium'),    # Confirms upper
    (51, 'Boundary', 'Medium'),    # rank=50 no convergence
    (77, 'Boundary', 'High'),      # 5k frames lr_W upper
    (82, 'Boundary', 'High'),      # n_frames=2500 below minimum
    (101, 'Boundary', 'Medium'),   # 7500 lr_W=1E-1
    (102, 'Boundary', 'High'),     # 7500 lr_W upper
    (107, 'Boundary', 'Medium'),   # 7500 L1 near
    (108, 'Boundary', 'Medium'),   # 7500 L1 above
    (110, 'Boundary', 'Medium'),   # 7500 lr_W refine
    (111, 'Boundary', 'High'),     # 7500 lr_W upper precise
    (146, 'Boundary', 'Medium'),   # n=1000 lr_W=4E-2
    (151, 'Boundary', 'Medium'),   # n=1000 lr_W=2E-2
    (154, 'Boundary', 'Medium'),   # n=1000 lr upper
    (166, 'Boundary', 'High'),     # n=1000 2x lr_W upper
    (167, 'Boundary', 'High'),     # n=1000 lr catastrophic
    (188, 'Boundary', 'Medium'),   # Dale n=1000 lr_W upper
    (199, 'Boundary', 'Medium'),   # low_rank n=1000 lr upper
    (201, 'Boundary', 'Medium'),   # low_rank n=1000 lr boundary
    (202, 'Boundary', 'Medium'),   # low_rank n=1000 lr_W lower
    (206, 'Boundary', 'Medium'),   # low_rank n=1000 lr lower
    (213, 'Boundary', 'Medium'),   # 50k lr_W=2E-1
    (214, 'Boundary', 'Medium'),   # 50k lr_W=5E-1
    (215, 'Boundary', 'Medium'),   # 50k lr_W=1.0
    (216, 'Boundary', 'High'),     # 50k lr_W upper
    (218, 'Boundary', 'Medium'),   # 50k lr_W=1E-3
    (219, 'Boundary', 'Medium'),   # 50k lr_W=5E-4
    (220, 'Boundary', 'High'),     # 50k lr_W lower (8000x range!)
    (232, 'Boundary', 'High'),     # 50k n=1000 lr_W lower
    (236, 'Boundary', 'Medium'),   # 50k n=1000 lr upper
    (244, 'Boundary', 'High'),     # Dale 50k n=1000 lr_W lower
    (248, 'Boundary', 'Medium'),   # Dale 50k n=1000 L1 upper

    # === Meta-reasoning: 6 instances ===
    (6, 'Meta-reasoning', 'Medium'),    # Switch-dimension after lr_W mutations
    (26, 'Meta-reasoning', 'Medium'),   # Strategy reassessment
    (57, 'Meta-reasoning', 'Medium'),   # Dimension re-evaluation
    (97, 'Meta-reasoning', 'High'),     # Insight synthesis
    (137, 'Meta-reasoning', 'High'),    # Barrier recognition
    (149, 'Meta-reasoning', 'High'),    # Recognized lr_W exhausted

    # === Regime Recognition: 9 instances ===
    (17, 'Regime', 'High'),    # Dale_law different
    (33, 'Regime', 'High'),    # low_rank constraints
    (42, 'Regime', 'High'),    # eff_rank=6 qualitatively different
    (65, 'Regime', 'Medium'),  # Data regime
    (91, 'Regime', 'High'),    # n_frames=2500 qualitatively different
    (113, 'Regime', 'High'),   # Hard barrier
    (160, 'Regime', 'High'),   # n=1000 fundamentally different
    (177, 'Regime', 'High'),   # E/I collapse mode
    (225, 'Regime', 'High'),   # n=1000+high data qualitatively different

    # === Uncertainty: 7 instances ===
    (42, 'Uncertainty', 'High'),    # R²=0.886 not reproducible
    (97, 'Uncertainty', 'Medium'),  # Stochastic in intermediate
    (112, 'Uncertainty', 'High'),   # eff_rank=27 vs usual 10
    (113, 'Uncertainty', 'Medium'), # eff_rank stochastic
    (177, 'Uncertainty', 'Medium'), # Collapse stochastic
    (192, 'Uncertainty', 'High'),   # High stochasticity
    (203, 'Uncertainty', 'High'),   # Same config: 0.244 vs 0.093

    # === Causal: 5 instances ===
    (48, 'Causal Chain', 'High'),    # low_rank→eff_rank=6→R²<0.4
    (64, 'Causal Chain', 'High'),    # rank≠eff_rank
    (96, 'Causal Chain', 'High'),    # n_frames→eff_rank→R²
    (127, 'Causal Chain', 'Medium'), # eff_rank=10→R²=0.945
    (224, 'Causal Chain', 'High'),   # high n_frames→high eff_rank→high R²

    # === Predictive: 8 instances ===
    (96, 'Predictive', 'High'),    # Minimum n_frames ≈ 5000
    (127, 'Predictive', 'High'),   # eff_rank=10 ceiling ~0.945
    (141, 'Predictive', 'High'),   # low_rank +10 → +0.07 R²
    (144, 'Predictive', 'High'),   # low_rank ≥ 3x connectivity_rank
    (216, 'Predictive', 'High'),   # 8000x lr_W range
    (224, 'Predictive', 'High'),   # 8000x lr_W range with test_R2
    (225, 'Predictive', 'High'),   # n_frames=50000 breaks ceiling
    (255, 'Predictive', 'Medium'), # Dale+high data optimal params

    # === Constraint: 5 instances ===
    (64, 'Constraint', 'High'),    # rank≠eff_rank
    (144, 'Constraint', 'High'),   # low_rank ≥ 3x rule
    (160, 'Constraint', 'High'),   # lr/lr_W ratio inverted
    (240, 'Constraint', 'Medium'), # L1 affects generalization
    (232, 'Constraint', 'Medium'), # Lower boundary for lr_W
]

# Causal edges from epistemic_edges.md - 68 total
# Format: (from_iter, from_mode, to_iter, to_mode, edge_type)
edges = [
    # === Block 1 (Chaotic baseline, iters 1-16) ===
    (4, 'Deduction', 6, 'Induction', 'leads_to'),       # Validated prediction → pattern
    (4, 'Boundary', 10, 'Falsification', 'leads_to'),   # Boundary approach → failure found
    (6, 'Induction', 9, 'Induction', 'leads_to'),       # 5 obs → 8 obs cumulative
    (9, 'Induction', 10, 'Deduction', 'triggers'),      # Robustness observed → test L1
    (10, 'Falsification', 16, 'Induction', 'refines'),  # L1 boundary → established principle
    (13, 'Falsification', 16, 'Induction', 'refines'),  # Factorization failure → principle

    # === Block 2 (Dale_law=True, iters 17-32) ===
    (17, 'Abduction', 18, 'Deduction', 'triggers'),     # eff_rank hypothesis → test higher lr_W
    (17, 'Analogy', 17, 'Abduction', 'triggers'),       # Transfer failure → hypothesis formation
    (18, 'Deduction', 21, 'Deduction', 'leads_to'),     # Partial success → continued testing
    (21, 'Deduction', 24, 'Boundary', 'leads_to'),      # Convergence → probe upper bound
    (24, 'Boundary', 26, 'Falsification', 'leads_to'),  # Boundary found → test other dimension
    (25, 'Falsification', 32, 'Induction', 'refines'),  # Factorization fails Dale → principle
    (26, 'Falsification', 32, 'Induction', 'refines'),  # L1 sensitivity → principle

    # === Block 3 (low_rank, iters 33-48) ===
    (33, 'Abduction', 34, 'Deduction', 'triggers'),     # eff_rank=6 → test higher lr_W
    (33, 'Analogy', 33, 'Abduction', 'triggers'),       # Transfer failure → new hypothesis
    (36, 'Falsification', 42, 'Deduction', 'triggers'), # Factorization failed → try other approach
    (40, 'Deduction', 42, 'Falsification', 'leads_to'), # Best result → robustness test
    (42, 'Falsification', 42, 'Uncertainty', 'triggers'), # Failed test → stochasticity recognized
    (42, 'Uncertainty', 48, 'Induction', 'leads_to'),   # Stochasticity → eff_rank principle
    (42, 'Regime', 48, 'Causal Chain', 'triggers'),           # eff_rank=6 regime → causal model

    # === Block 4 (rank=50, iters 49-64) ===
    (49, 'Deduction', 49, 'Falsification', 'triggers'), # Prediction failed → reject hypothesis
    (49, 'Falsification', 64, 'Constraint', 'leads_to'), # rank≠eff_rank → constraint principle
    (53, 'Deduction', 56, 'Deduction', 'leads_to'),     # Factorization helped → continue

    # === Block 5 (5k frames, iters 65-80) ===
    (65, 'Deduction', 65, 'Abduction', 'triggers'),     # Failure → hypothesis about n_frames
    (65, 'Abduction', 74, 'Deduction', 'triggers'),     # Hypothesis → test higher lr_W
    (74, 'Deduction', 77, 'Boundary', 'leads_to'),      # Convergence → probe boundary
    (77, 'Boundary', 80, 'Induction', 'leads_to'),      # Boundary found → block summary

    # === Block 6 (2500 frames, iters 81-96) ===
    (81, 'Deduction', 82, 'Falsification', 'triggers'), # Partial → failure on next iter
    (82, 'Falsification', 91, 'Regime', 'triggers'),    # Repeated eff_rank=6 → regime recognition
    (91, 'Regime', 96, 'Induction', 'leads_to'),        # Below threshold recognized → principle
    (91, 'Falsification', 96, 'Causal Chain', 'leads_to'),    # Factorization fails → causal understanding

    # === Block 7 (7500 frames, iters 97-111) ===
    (98, 'Deduction', 101, 'Boundary', 'triggers'),     # Convergence → probe upper bound
    (101, 'Boundary', 102, 'Falsification', 'leads_to'), # Extreme test → boundary found
    (102, 'Falsification', 111, 'Induction', 'refines'), # Upper bound → refined principle

    # === Block 8 (Dale_law retry, iters 112-127) ===
    (112, 'Analogy', 112, 'Deduction', 'triggers'),     # Transfer attempt → breakthrough
    (112, 'Deduction', 113, 'Uncertainty', 'triggers'), # R²=0.995 → but eff_rank stochastic
    (113, 'Uncertainty', 127, 'Predictive', 'leads_to'), # Stochasticity → eff_rank=10 prediction
    (127, 'Induction', 127, 'Predictive', 'triggers'),  # Best result → ceiling prediction

    # === Block 10 (low_rank+Dale continuation, iters 129-144) ===
    (141, 'Deduction', 141, 'Predictive', 'triggers'),  # +0.07 per +10 → quantitative model
    (141, 'Predictive', 144, 'Deduction', 'leads_to'),  # Prediction → test continues
    (144, 'Deduction', 144, 'Constraint', 'triggers'),  # Convergence → 3x rule discovered

    # === Block 11 (n=1000, iters 145-160) ===
    (145, 'Abduction', 149, 'Deduction', 'triggers'),   # MLP bottleneck → test lr
    (149, 'Deduction', 149, 'Meta-reasoning', 'triggers'), # Switch-dimension successful
    (149, 'Meta-reasoning', 156, 'Deduction', 'leads_to'), # Strategy change → optimization
    (156, 'Deduction', 160, 'Regime', 'leads_to'),      # Best result → n=1000 regime recognized
    (160, 'Regime', 160, 'Constraint', 'triggers'),     # New regime → inverted ratio

    # === Block 12 (n=1000 2x training, iters 161-176) ===
    (161, 'Falsification', 171, 'Deduction', 'triggers'), # Training not bottleneck → try L1
    (171, 'Deduction', 175, 'Deduction', 'leads_to'),   # L1 helped → continue
    (175, 'Induction', 176, 'Induction', 'leads_to'),   # Best config → block summary

    # === Block 13 (n=1000+Dale, iters 177-192) ===
    (177, 'Abduction', 177, 'Regime', 'triggers'),      # Collapse → new regime mode
    (178, 'Deduction', 187, 'Deduction', 'leads_to'),   # Recovery → continued exploration
    (187, 'Deduction', 192, 'Uncertainty', 'triggers'), # Breakthrough → but stochastic

    # === Block 14 (low_rank n=1000, iters 193-208) ===
    (198, 'Deduction', 203, 'Falsification', 'leads_to'), # Best result → robustness test
    (203, 'Falsification', 203, 'Uncertainty', 'triggers'), # Test failed → HIGH stochasticity
    (203, 'Uncertainty', 208, 'Induction', 'leads_to'), # Recognized → block principle

    # === Block 15 (50k frames n=100, iters 209-224) ===
    (209, 'Analogy', 209, 'Deduction', 'triggers'),     # Perfect transfer → validate
    (216, 'Boundary', 220, 'Boundary', 'leads_to'),     # Upper found → probe lower
    (220, 'Boundary', 224, 'Predictive', 'leads_to'),   # 8000x range → extraordinary robustness

    # === Block 16 (50k frames n=1000, iters 225-240) ===
    (225, 'Deduction', 225, 'Predictive', 'triggers'),  # Breakthrough → validated prediction
    (225, 'Predictive', 226, 'Falsification', 'leads_to'), # Success → test L1 boundary
    (226, 'Falsification', 240, 'Induction', 'refines'), # L1 boundary → n=1000 principle
    (232, 'Boundary', 240, 'Constraint', 'leads_to'),   # Lower bound → constraint established

    # === Block 17 (Dale+50k n=1000, iters 241-255) ===
    (241, 'Analogy', 243, 'Deduction', 'triggers'),     # Transfer success → optimize
    (243, 'Deduction', 244, 'Boundary', 'leads_to'),    # Best result → probe boundary
    (244, 'Boundary', 246, 'Deduction', 'triggers'),    # Lower found → try other dimension
    (251, 'Deduction', 253, 'Deduction', 'triggers'),   # Recombine → robustness test
    (253, 'Deduction', 255, 'Induction', 'leads_to'),   # Reproducible → final principle

    # === Cross-Block Edges ===
    (16, 'Induction', 17, 'Analogy', 'triggers'),       # Block 1 principles → Block 2 transfer
    (16, 'Induction', 33, 'Analogy', 'triggers'),       # Block 1 principles → Block 3 transfer
    (32, 'Induction', 49, 'Analogy', 'triggers'),       # Dale_law findings → rank test
    (32, 'Induction', 112, 'Analogy', 'triggers'),      # Block 2 findings → Block 8 retry
    (48, 'Induction', 65, 'Abduction', 'triggers'),     # eff_rank→R² → n_frames hypothesis
    (48, 'Causal Chain', 96, 'Causal Chain', 'refines'),            # eff_rank ceiling → minimum n_frames
    (64, 'Constraint', 144, 'Constraint', 'refines'),   # rank≠eff_rank → 3x rule
    (80, 'Induction', 97, 'Deduction', 'triggers'),     # n_frames relationship → intermediate test
    (96, 'Regime', 112, 'Deduction', 'triggers'),       # Data threshold → high data retry
    (127, 'Predictive', 225, 'Predictive', 'triggers'), # eff_rank ceiling → break with more data
    (144, 'Constraint', 162, 'Falsification', 'triggers'), # 3x rule → test at n=1000
    (160, 'Regime', 225, 'Predictive', 'triggers'),     # n=1000 ceiling → high data prediction
    (192, 'Induction', 241, 'Analogy', 'triggers'),     # n=1000+Dale findings → high data test
    (224, 'Induction', 225, 'Analogy', 'triggers'),     # Block 15 findings → Block 16 transfer
    (240, 'Induction', 241, 'Analogy', 'triggers'),     # Block 16 findings → Block 17 transfer
]

# Block boundaries with detailed info (17 blocks)
blocks = [
    (1, 16, 'Block 1', {'regime': 'chaotic', 'E/I': '-', 'n_frames': 10000, 'n_neurons': 100, 'eff_rank': '31-35'}),
    (17, 32, 'Block 2', {'regime': 'chaotic', 'E/I': '0.5', 'n_frames': 10000, 'n_neurons': 100, 'eff_rank': '10'}),
    (33, 48, 'Block 3', {'regime': 'low_rank', 'E/I': '-', 'n_frames': 10000, 'n_neurons': 100, 'eff_rank': '6'}),
    (49, 64, 'Block 4', {'regime': 'low_rank', 'E/I': '-', 'n_frames': 10000, 'n_neurons': 100, 'eff_rank': '7'}),
    (65, 80, 'Block 5', {'regime': 'chaotic', 'E/I': '-', 'n_frames': 5000, 'n_neurons': 100, 'eff_rank': '20'}),
    (81, 96, 'Block 6', {'regime': 'chaotic', 'E/I': '-', 'n_frames': 2500, 'n_neurons': 100, 'eff_rank': '6'}),
    (97, 111, 'Block 7', {'regime': 'chaotic', 'E/I': '-', 'n_frames': 7500, 'n_neurons': 100, 'eff_rank': '25-29'}),
    (112, 127, 'Block 8', {'regime': 'chaotic', 'E/I': '0.5', 'n_frames': 10000, 'n_neurons': 100, 'eff_rank': '10-27'}),
    (128, 128, 'Block 9', {'regime': 'low_rank', 'E/I': '0.5', 'n_frames': 10000, 'n_neurons': 100, 'eff_rank': '16'}),
    (129, 144, 'Block 10', {'regime': 'low_rank', 'E/I': '0.5', 'n_frames': 10000, 'n_neurons': 100, 'eff_rank': '16'}),
    (145, 160, 'Block 11', {'regime': 'chaotic', 'E/I': '-', 'n_frames': 10000, 'n_neurons': 1000, 'eff_rank': '52-54'}),
    (161, 176, 'Block 12', {'regime': 'chaotic', 'E/I': '-', 'n_frames': 10000, 'n_neurons': 1000, 'eff_rank': '51-53'}),
    (177, 192, 'Block 13', {'regime': 'chaotic', 'E/I': '0.5', 'n_frames': 10000, 'n_neurons': 1000, 'eff_rank': '35-38'}),
    (193, 208, 'Block 14', {'regime': 'low_rank', 'E/I': '-', 'n_frames': 10000, 'n_neurons': 1000, 'eff_rank': '7-14'}),
    (209, 224, 'Block 15', {'regime': 'chaotic', 'E/I': '-', 'n_frames': 50000, 'n_neurons': 100, 'eff_rank': '49-54'}),
    (225, 240, 'Block 16', {'regime': 'chaotic', 'E/I': '-', 'n_frames': 50000, 'n_neurons': 1000, 'eff_rank': '94'}),
    (241, 255, 'Block 17', {'regime': 'chaotic', 'E/I': '0.5', 'n_frames': 50000, 'n_neurons': 1000, 'eff_rank': '80-83'}),
]


def create_timeline():
    _, ax = plt.subplots(figsize=(32, 16))

    # Map modes to y-positions (group by category)
    modes = [
        # Evidence gathering
        'Induction', 'Boundary',
        # Hypothesis testing
        'Abduction', 'Deduction', 'Falsification',
        # Meta-cognition
        'Analogy', 'Meta-reasoning', 'Regime', 'Uncertainty',
        # Advanced patterns
        'Constraint', 'Predictive', 'Causal Chain',
    ]
    mode_to_y = {mode: i for i, mode in enumerate(modes)}

    # Draw block backgrounds (alternating colors)
    for block_idx, (start, end, label, info) in enumerate(blocks):
        color = '#f0f0f0' if block_idx % 2 == 1 else 'white'
        ax.axvspan(start - 0.5, end + 0.5, alpha=1.0, color=color)

        # Add block label at top
        left_x = start + 0.3
        top_y = len(modes) - 0.3

        block_text = f"{label}\n{info['regime']}\nE/I={info['E/I']}\nT={info['n_frames']}\nN={info['n_neurons']}\neff={info['eff_rank']}"
        ax.text(left_x, top_y, block_text, ha='left', va='bottom', fontsize=9,
                linespacing=0.85)

    # Plot events (nodes) with iteration labels
    for iteration, mode, significance in events:
        if mode not in mode_to_y:
            continue
        y = mode_to_y[mode]
        color = COLORS.get(mode, '#333333')

        size = {'High': 200, 'Medium': 120, 'Low': 70}.get(significance, 120)

        ax.scatter(iteration, y, c=color, s=size, alpha=0.9,
                   edgecolors='black', linewidths=0.4, zorder=3)

        # Add iteration number label
        ax.annotate(str(iteration), (iteration, y),
                    xytext=(0, -10), textcoords='offset points',
                    fontsize=5, ha='center', va='top', color='#333333',
                    zorder=4)

    # Add category dividers
    ax.axhline(y=1.5, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axhline(y=4.5, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axhline(y=8.5, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

    # Styling
    ax.set_xlim(0, 260)
    ax.set_ylim(-0.5, len(modes) + 2.5)
    ax.set_yticks(range(len(modes)))
    # Y-tick labels with definitions in parenthesis
    ytick_labels = [f"{mode} ({DEFINITIONS.get(mode, '')})" for mode in modes]
    ax.set_yticklabels(ytick_labels, fontsize=11)
    ax.set_xlabel('Iteration', fontsize=20)
    ax.set_ylabel('Reasoning Mode', fontsize=20)
    ax.tick_params(axis='x', labelsize=12)

    ax.grid(True, axis='x', alpha=0.3, linestyle='--')

    # Legend for node sizes
    from matplotlib.lines import Line2D
    size_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=10, label='High significance'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=7, label='Medium significance'),
    ]
    ax.legend(handles=size_legend, loc='upper left', bbox_to_anchor=(1.01, 0.55),
              title='Node Size', fontsize=8)

    plt.tight_layout()
    plt.savefig('signal_chaotic_1_Claude_epistemic_timeline.png', dpi=150, bbox_inches='tight',
                pad_inches=0.5)
    print("Saved: signal_chaotic_1_Claude_epistemic_timeline.png")
    plt.close()


def create_streamgraph():
    """
    Create a streamgraph visualization showing reasoning mode activity over iterations.
    Streamgraph is a stacked area chart with a wiggle baseline for aesthetic flow.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    # Define modes in order (for stacking)
    modes = [
        'Induction', 'Boundary', 'Abduction', 'Deduction', 'Falsification',
        'Analogy', 'Meta-reasoning', 'Regime', 'Uncertainty',
        'Constraint', 'Predictive', 'Causal Chain',
    ]

    # Create iteration bins
    max_iter = max(e[0] for e in events)
    bin_width = 8  # Aggregate events into bins
    n_bins = (max_iter // bin_width) + 1
    iterations = np.arange(0, n_bins * bin_width, bin_width)

    # Count events per mode per bin
    data = np.zeros((len(modes), n_bins))
    for iteration, mode, significance in events:
        if mode in modes:
            mode_idx = modes.index(mode)
            bin_idx = iteration // bin_width
            if bin_idx < n_bins:
                # Weight by significance
                weight = {'High': 2.0, 'Medium': 1.0, 'Low': 0.5}.get(significance, 1.0)
                data[mode_idx, bin_idx] += weight

    # Smooth the data for aesthetic flow
    sigma = 1.0
    data_smooth = np.array([gaussian_filter1d(row, sigma) for row in data])

    # Compute streamgraph baseline (wiggle/ThemeRiver)
    # Baseline offset to minimize weighted wiggle
    n_layers = len(modes)
    baseline = np.zeros(n_bins)
    for i in range(n_layers):
        baseline -= (n_layers - i - 0.5) * data_smooth[i]
    baseline /= n_layers

    # Stack the layers
    y_stack = np.vstack([baseline, baseline + np.cumsum(data_smooth, axis=0)])

    # Create figure
    fig, ax = plt.subplots(figsize=(24, 10))

    # Plot each layer
    for i, mode in enumerate(modes):
        color = COLORS.get(mode, '#333333')
        ax.fill_between(iterations, y_stack[i], y_stack[i + 1],
                        color=color, alpha=0.8, label=f"{mode} ({DEFINITIONS.get(mode, '')})",
                        edgecolor='white', linewidth=0.3)

    # Add block boundaries as vertical lines
    for start, end, label, info in blocks:
        ax.axvline(x=start, color='gray', linewidth=0.5, linestyle='--', alpha=0.3)

    # Styling
    ax.set_xlim(0, max_iter + 5)
    ax.set_xlabel('Iteration', fontsize=18)
    ax.set_ylabel('Reasoning Activity (weighted)', fontsize=18)
    ax.tick_params(axis='both', labelsize=12)

    # Legend with definitions
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=9,
              title='Reasoning Modes', title_fontsize=11)

    ax.set_facecolor('#f8f8f8')
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig('signal_chaotic_1_Claude_epistemic_streamgraph.png', dpi=150, bbox_inches='tight',
                pad_inches=0.5)
    print("Saved: signal_chaotic_1_Claude_epistemic_streamgraph.png")
    plt.close()


def create_sankey_diagram():
    """
    Create a Sankey diagram showing causal flows between reasoning modes.
    Uses Plotly for interactive visualization.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly not installed. Install with: pip install plotly")
        return

    from collections import Counter

    # Count transitions between modes from edges
    transitions = Counter()
    for from_iter, from_mode, to_iter, to_mode, edge_type in edges:
        transitions[(from_mode, to_mode)] += 1

    # Define modes (nodes)
    modes = list(COLORS.keys())
    mode_to_idx = {mode: i for i, mode in enumerate(modes)}

    # Build source, target, value lists for Sankey
    source = []
    target = []
    value = []
    link_colors = []

    for (from_mode, to_mode), count in transitions.items():
        if from_mode in mode_to_idx and to_mode in mode_to_idx:
            source.append(mode_to_idx[from_mode])
            target.append(mode_to_idx[to_mode])
            value.append(count)
            # Use source mode color with transparency
            color = COLORS.get(from_mode, '#888888')
            # Convert hex to rgba
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            link_colors.append(f'rgba({r},{g},{b},0.4)')

    # Node colors
    node_colors = [COLORS.get(mode, '#888888') for mode in modes]

    # Node labels with definitions
    node_labels = [f"{mode}<br>({DEFINITIONS.get(mode, '')})" for mode in modes]

    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='black', width=0.5),
            label=node_labels,
            color=node_colors,
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors,
        )
    )])

    fig.update_layout(
        title_text="Epistemic Reasoning Flow: signal_chaotic_1_Claude",
        font_size=12,
        width=1200,
        height=800,
    )

    # Save as HTML (interactive) and PNG (static)
    fig.write_html('signal_chaotic_1_Claude_epistemic_sankey.html')
    print("Saved: signal_chaotic_1_Claude_epistemic_sankey.html")

    try:
        fig.write_image('signal_chaotic_1_Claude_epistemic_sankey.png', scale=2)
        print("Saved: signal_chaotic_1_Claude_epistemic_sankey.png")
    except Exception as e:
        print(f"Could not save PNG (kaleido may be needed): {e}")


def create_2x2_panels():
    """
    Create a 2x3 panel figure:
    - Top left: Scatterplot (timeline)
    - Top middle: Legend with colors and definitions
    - Top right: Summary statistics
    - Bottom left: Streamgraph
    - Bottom right: Sankey Diagram (reference to HTML)

    Only top middle and top right panels have text; others show just colors.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    from collections import Counter
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(28, 16))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[2, 1, 1], height_ratios=[1, 1])

    # Define modes in order
    modes = [
        'Induction', 'Boundary', 'Abduction', 'Deduction', 'Falsification',
        'Analogy', 'Meta-reasoning', 'Regime', 'Uncertainty',
        'Constraint', 'Predictive', 'Causal Chain',
    ]
    mode_to_y = {mode: i for i, mode in enumerate(modes)}

    # ============ TOP LEFT: Scatterplot ============
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Scatterplot', fontsize=18)

    # Draw block backgrounds
    for block_idx, (start, end, label, info) in enumerate(blocks):
        color = '#f0f0f0' if block_idx % 2 == 1 else 'white'
        ax1.axvspan(start - 0.5, end + 0.5, alpha=1.0, color=color)

    # Plot events
    for iteration, mode, significance in events:
        if mode not in mode_to_y:
            continue
        y = mode_to_y[mode]
        color = COLORS.get(mode, '#333333')
        size = {'High': 150, 'Medium': 80, 'Low': 40}.get(significance, 80)
        ax1.scatter(iteration, y, c=color, s=size, alpha=0.9,
                   edgecolors='black', linewidths=0.3, zorder=3)

    # Minimal styling (no labels)
    ax1.set_xlim(0, 260)
    ax1.set_ylim(-0.5, len(modes) - 0.5)
    ax1.set_yticks([])
    ax1.set_xticks([])
    ax1.grid(True, axis='x', alpha=0.2, linestyle='--')

    # ============ TOP MIDDLE: Legend Panel ============
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('Legend', fontsize=18)
    ax2.axis('off')

    # Create legend entries
    y_pos = 0.95
    y_step = 0.07

    for mode in modes:
        color = COLORS.get(mode, '#333333')
        definition = DEFINITIONS.get(mode, '')

        # Color circle
        ax2.scatter([0.08], [y_pos], c=color, s=250, edgecolors='black', linewidths=1)
        # Mode name and definition
        ax2.text(0.18, y_pos, f"{mode}", fontsize=11, va='center')
        ax2.text(0.18, y_pos - 0.028, f"({definition})", fontsize=9, va='center', color='#666666')

        y_pos -= y_step

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    # ============ TOP RIGHT: Summary Panel ============
    ax_summary = fig.add_subplot(gs[0, 2])
    ax_summary.set_title('Summary', fontsize=18)
    ax_summary.axis('off')

    # Count statistics from events
    mode_counts = Counter(e[1] for e in events)
    total_events = len(events)
    total_edges = len(edges)

    # Calculate deduction validation rate (from the data)
    deduction_validated = 22  # from analysis
    deduction_total = 31
    deduction_rate = f"{100 * deduction_validated / deduction_total:.0f}%"

    # Transfer success rate
    transfer_success = 9
    transfer_total = 12
    transfer_rate = f"{100 * transfer_success / transfer_total:.0f}%"

    summary_text = f"""Epistemic Analysis

{total_events} reasoning instances
60 iterations, 8 blocks
10 principles discovered

Deduction validation: {deduction_rate}
Transfer success: {transfer_rate}

Mode counts:
  Induction: {mode_counts.get('Induction', 0)}
  Abduction: {mode_counts.get('Abduction', 0)}
  Deduction: {mode_counts.get('Deduction', 0)}
  Falsification: {mode_counts.get('Falsification', 0)}
  Analogy: {mode_counts.get('Analogy', 0)}
  Boundary: {mode_counts.get('Boundary', 0)}

{total_edges} causal edges
4 key causal chains

See HTML for interactive Sankey"""

    ax_summary.text(0.05, 0.95, summary_text, fontsize=10, va='top', ha='left',
                   family='monospace', linespacing=1.4)
    ax_summary.set_xlim(0, 1)
    ax_summary.set_ylim(0, 1)

    # ============ BOTTOM LEFT: Streamgraph ============
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title('Streamgraph', fontsize=18)

    # Create iteration bins
    max_iter = max(e[0] for e in events)
    bin_width = 8
    n_bins = (max_iter // bin_width) + 1
    iterations = np.arange(0, n_bins * bin_width, bin_width)

    # Count events per mode per bin
    data = np.zeros((len(modes), n_bins))
    for iteration, mode, significance in events:
        if mode in modes:
            mode_idx = modes.index(mode)
            bin_idx = iteration // bin_width
            if bin_idx < n_bins:
                weight = {'High': 2.0, 'Medium': 1.0, 'Low': 0.5}.get(significance, 1.0)
                data[mode_idx, bin_idx] += weight

    # Smooth the data
    sigma = 1.0
    data_smooth = np.array([gaussian_filter1d(row, sigma) for row in data])

    # Compute streamgraph baseline (wiggle)
    n_layers = len(modes)
    baseline = np.zeros(n_bins)
    for i in range(n_layers):
        baseline -= (n_layers - i - 0.5) * data_smooth[i]
    baseline /= n_layers

    # Stack the layers
    y_stack = np.vstack([baseline, baseline + np.cumsum(data_smooth, axis=0)])

    # Plot each layer (no labels)
    for i, mode in enumerate(modes):
        color = COLORS.get(mode, '#333333')
        ax3.fill_between(iterations, y_stack[i], y_stack[i + 1],
                        color=color, alpha=0.8,
                        edgecolor='white', linewidth=0.3)

    # Minimal styling
    ax3.set_xlim(0, max_iter + 5)
    ax3.set_yticks([])
    ax3.set_xticks([])
    ax3.set_facecolor('#f8f8f8')

    # ============ BOTTOM RIGHT: Sankey-like flow (spans 2 columns) ============
    ax4 = fig.add_subplot(gs[1, 1:])
    ax4.set_title('Sankey Diagram (see HTML for interactive version)', fontsize=18)

    # Count transitions between modes from edges
    transitions = Counter()
    for from_iter, from_mode, to_iter, to_mode, edge_type in edges:
        transitions[(from_mode, to_mode)] += 1

    # Position modes in a circle
    n_modes = len(modes)
    angles = np.linspace(0, 2 * np.pi, n_modes, endpoint=False)
    radius = 0.35
    center = (0.5, 0.5)

    # Draw nodes
    for i, mode in enumerate(modes):
        x = center[0] + radius * np.cos(angles[i])
        y = center[1] + radius * np.sin(angles[i])
        color = COLORS.get(mode, '#333333')
        ax4.scatter([x], [y], c=color, s=800, edgecolors='black', linewidths=1.5, zorder=5)

    # Draw connections (curved lines)
    from matplotlib.path import Path

    for (from_mode, to_mode), count in transitions.items():
        if from_mode in modes and to_mode in modes:
            from_idx = modes.index(from_mode)
            to_idx = modes.index(to_mode)

            x1 = center[0] + radius * np.cos(angles[from_idx])
            y1 = center[1] + radius * np.sin(angles[from_idx])
            x2 = center[0] + radius * np.cos(angles[to_idx])
            y2 = center[1] + radius * np.sin(angles[to_idx])

            # Control point for curve (towards center)
            cx = center[0] + 0.1 * (x1 + x2 - 2 * center[0])
            cy = center[1] + 0.1 * (y1 + y2 - 2 * center[1])

            color = COLORS.get(from_mode, '#888888')
            alpha = min(0.3 + count * 0.1, 0.8)
            linewidth = 1 + count * 0.5

            # Draw quadratic bezier curve
            verts = [(x1, y1), (cx, cy), (x2, y2)]
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            path = Path(verts, codes)
            patch = mpatches.PathPatch(path, facecolor='none', edgecolor=color,
                                       alpha=alpha, linewidth=linewidth)
            ax4.add_patch(patch)

    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('signal_chaotic_1_Claude_epistemic_2x2_panels.png', dpi=150, bbox_inches='tight',
                pad_inches=0.3)
    print("Saved: signal_chaotic_1_Claude_epistemic_2x2_panels.png")
    plt.close()


if __name__ == '__main__':
    print("Creating epistemic timeline visualizations...")
    print(f"Total events: {len(events)}")
    print(f"Total edges: {len(edges)}")

    from collections import Counter
    counts = Counter(e[1] for e in events)
    for mode, count in sorted(counts.items()):
        print(f"  {mode}: {count}")

    print("\n--- Timeline ---")
    create_timeline()

    print("\n--- Streamgraph ---")
    create_streamgraph()

    print("\n--- Sankey Diagram ---")
    create_sankey_diagram()

    print("\n--- 2x2 Panels ---")
    create_2x2_panels()

    print("\nDone!")
