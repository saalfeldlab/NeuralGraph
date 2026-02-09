#!/usr/bin/env python3
"""
Interactive scatter chart — regime landscape.
336 iterations, 28 blocks.
Plotly: zoom/pan, hover tooltips with iter/regime/training info.
"""

import plotly.graph_objects as go
import numpy as np

np.random.seed(42)

# --- Data with full metadata per point ---
points = []


def add_block(block, regime, n_frames_str, center_x, conns, start_iter,
              n_neurons, gain=7, fill=100, jitter=1.5, extra=''):
    """Add one block of iterations with metadata."""
    for i, c in enumerate(conns):
        x = center_x + np.random.uniform(-jitter, jitter)
        points.append(dict(
            x=x, y=c,
            block=block, regime=regime,
            n_frames=n_frames_str,
            iter=start_iter + i,
            n_neurons=n_neurons,
            gain=gain, fill=fill,
            extra=extra,
        ))


# Block 1: Chaotic n=100
add_block(1, 'Chaotic baseline', '10k', 35, [
    0.9999, 0.998, 0.996, 0.996, 0.993, 0.990,
    0.985, 0.981, 0.970, 0.960, 0.940, 0.385
], start_iter=1, n_neurons=100,
    extra='lr_W sweep 1E-3→1E-2; optimal 4E-3')

# Block 2: Low-rank n=100
add_block(2, 'Low-rank (rank=20)', '10k', 12, [
    0.9997, 0.997, 0.996, 0.990, 0.980, 0.950,
    0.925, 0.902, 0.899, 0.580, 0.500, 0.420
], start_iter=13, n_neurons=100,
    extra='L1=1E-6 critical; lr_W=3E-3 optimal')

# Block 3: Dale n=100
add_block(3, 'Dale law (E/I)', '10k', 12, [
    0.986, 0.974, 0.972, 0.958, 0.940, 0.920,
    0.900, 0.880, 0.555, 0.458, 0.455, 0.420
], start_iter=25, n_neurons=100,
    extra='lr_W cliff at 5E-3; safe [3.5E-3, 4.5E-3]')

# Block 4: Hetero n=100
add_block(4, 'Heterogeneous (4 types)', '10k', 38, [
    0.9996, 0.998, 0.992, 0.990, 0.985, 0.970,
    0.950, 0.930, 0.900, 0.870, 0.670, 0.500
], start_iter=37, n_neurons=100,
    extra='dual-objective; lr_emb=1E-3 required')

# Block 5: Noise n=100 — spread across eff_rank range
for i, c in enumerate([1.000, 1.000, 1.000, 1.000, 0.998, 0.998,
                        0.997, 0.996, 0.995, 0.990, 0.985, 0.980]):
    x = np.random.uniform(42, 90)
    points.append(dict(
        x=x, y=c, block=5, regime='Noise (σ=0.1–1.0)',
        n_frames='10k', iter=49 + i, n_neurons=100,
        gain=7, fill=100,
        extra='100% convergence; inverse lr_W-noise',
    ))

# Block 6: n=200
add_block(6, 'Scale n=200', '10k', 43, [
    0.956, 0.950, 0.940, 0.930, 0.920, 0.910,
    0.905, 0.890, 0.800, 0.750, 0.650, 0.575
], start_iter=61, n_neurons=200,
    extra='lr_W=5E-3; boundary shifts up with n')

# Block 7: Sparse n=100
add_block(7, 'Sparse 50%', '10k', 21, [
    0.466, 0.450, 0.440, 0.430, 0.423, 0.410,
    0.400, 0.390, 0.380, 0.350, 0.320, 0.310
], start_iter=73, n_neurons=100, fill=50,
    extra='ρ=0.746 subcritical; 0% conv; n_epochs key')

# Block 8: Sparse+Noise n=100
add_block(8, 'Sparse 50% + Noise', '10k', 91, [
    0.490, 0.489, 0.489, 0.489, 0.489, 0.485,
    0.480, 0.475, 0.470, 0.460, 0.300, 0.054
], start_iter=85, n_neurons=100, fill=50,
    extra='structural data limit 0.489; param insensitive')

# Block 9: n=300 1-2ep
add_block(9, 'n=300 (1–2 ep)', '10k', 47, [
    0.890, 0.870, 0.850, 0.830, 0.810, 0.805,
    0.780, 0.750, 0.730, 0.720, 0.710, 0.699
], start_iter=97, n_neurons=300,
    extra='n_epochs=2 breakthrough; lr_W=1E-2')

# Block 10: n=300 2-4ep (8 iters)
add_block(10, 'n=300 (2–4 ep)', '10k', 47, [
    0.924, 0.920, 0.910, 0.897, 0.893, 0.886, 0.870, 0.850
], start_iter=109, n_neurons=300,
    extra='conn ceiling ~0.92 at 10k frames')

# Block 11: n=200 v2
add_block(11, 'n=200 solved (2–3 ep)', '10k', 43, [
    0.994, 0.993, 0.990, 0.988, 0.985, 0.980,
    0.975, 0.970, 0.965, 0.960, 0.950, 0.935
], start_iter=117, n_neurons=200,
    extra='100% conv (12/12); lr_W=8E-3')

# Block 12: n=600
add_block(12, 'n=600 (10k)', '10k', 50, [
    0.626, 0.600, 0.580, 0.560, 0.554, 0.540,
    0.520, 0.500, 0.480, 0.450, 0.350, 0.000
], start_iter=129, n_neurons=600,
    extra='training-capacity-limited; lr=1E-4 catastrophic')

# Block 13: n=200 + 4 types (16 iters)
add_block(13, 'n=200 + 4 types', '10k', 42, [
    0.991, 0.988, 0.985, 0.980, 0.975, 0.960, 0.948, 0.940,
    0.932, 0.920, 0.910, 0.908, 0.890, 0.870, 0.850, 0.830
], start_iter=141, n_neurons=200,
    extra='full dual conv; conn=0.988, cluster=1.000')

# Block 14: Recurrent n=200 (4 completed)
add_block(14, 'Recurrent n=200', '10k', 42, [
    0.993, 0.990, 0.912, 0.772
], start_iter=165, n_neurons=200,
    extra='conn-dynamics trade-off; 8/12 infra failures')

# Block 15: n=300 at 30k
add_block(15, 'n=300 (30k) — SOLVED', '30k', 80, [
    1.000, 1.000, 0.999, 0.999, 0.999, 0.999,
    0.999, 0.999, 0.999, 0.999, 0.999, 0.999
], start_iter=169, n_neurons=300,
    extra='100% conv; all params non-critical')

# Block 16: n=600 at 30k (8 iters)
add_block(16, 'n=600 (30k) — SOLVED', '30k', 87, [
    0.992, 0.976, 0.973, 0.967, 0.960, 0.950, 0.940, 0.933
], start_iter=181, n_neurons=600,
    extra='100% conv; lr_W=5E-3 optimal')

# Block 17: Sparse 50% at 30k
add_block(17, 'Sparse 50% (30k) — FAILED', '30k', 13, [
    0.436, 0.420, 0.410, 0.400, 0.390, 0.380,
    0.370, 0.350, 0.330, 0.300, 0.260, 0.213
], start_iter=193, n_neurons=100, fill=50,
    extra='n_frames NOT rescue; eff_rank DROPS 21→13')

# Block 18: n=1000 at 30k
add_block(18, 'n=1000 (30k)', '30k', 144, [
    0.745, 0.743, 0.734, 0.726, 0.720, 0.716,
    0.710, 0.700, 0.690, 0.680, 0.666, 0.640
], start_iter=205, n_neurons=1000, jitter=3,
    extra='30k insufficient; needs ~100k; lr=1E-4 Pareto')

# Block 19: g=3 n=100
add_block(19, 'g=3 n=100', '10k', 26, [
    0.955, 0.940, 0.920, 0.906, 0.880, 0.850,
    0.820, 0.790, 0.750, 0.636, 0.600, 0.550
], start_iter=217, n_neurons=100, gain=3,
    extra='gain as difficulty axis; n_epochs dominant')

# Block 20: g=3 n=200/10k
add_block(20, 'g=3 n=200 (10k)', '10k', 31, [
    0.489, 0.480, 0.470, 0.460, 0.450, 0.440,
    0.420, 0.400, 0.380, 0.360, 0.340, 0.300
], start_iter=229, n_neurons=200, gain=3,
    extra='gain×n compounds; 0% conv; needs 30k')

# Block 21: g=3 n=200 at 30k
add_block(21, 'g=3 n=200 (30k) — SOLVED', '30k', 55, [
    0.996, 0.995, 0.994, 0.993, 0.992, 0.990,
    0.988, 0.985, 0.982, 0.980, 0.975, 0.970
], start_iter=241, n_neurons=200, gain=3,
    extra='100% conv; gain SOLVABLE by n_frames')

# Block 22: fill=80% at 10k
add_block(22, 'fill=80% (10k)', '10k', 36, [
    0.802, 0.802, 0.802, 0.802, 0.802, 0.802,
    0.802, 0.802, 0.801, 0.801, 0.800, 0.800
], start_iter=253, n_neurons=100, fill=80,
    extra='conn plateau ≈ fill%; ρ=0.985; param insensitive')

# Block 23: fill=80% at 30k (12 iters — all locked at ~0.802)
add_block(23, 'fill=80% (30k) — FAILED', '30k', 49, [
    0.802, 0.802, 0.802, 0.802, 0.802, 0.802,
    0.802, 0.802, 0.802, 0.802, 0.802, 0.802
], start_iter=265, n_neurons=100, fill=80,
    extra='30k NOT rescue; conn locked at fill%; 12/12 at 0.802')

# Block 24: fill=90% at 10k
add_block(24, 'fill=90% (10k)', '10k', 36, [
    0.907, 0.907, 0.907, 0.907, 0.907, 0.907,
    0.907, 0.907, 0.906, 0.906, 0.905, 0.905
], start_iter=289, n_neurons=100, fill=90,
    extra='conn plateau ≈ fill%; transitional; param insensitive')

# Block 25: g=1 at 10k (fixed-point collapse)
add_block(25, 'g=1 (10k) — FAILED', '10k', 5, [
    0.007, 0.005, 0.004, 0.003, 0.003, 0.002,
    0.002, 0.002, 0.001, 0.001, 0.000, 0.000
], start_iter=301, n_neurons=100, gain=1,
    extra='fixed-point collapse; eff_rank=5; hardest regime')

# Block 26: g=1 at 30k (confirmed unsolvable)
add_block(26, 'g=1 (30k) — FAILED', '30k', 3, [
    0.018, 0.015, 0.012, 0.010, 0.008, 0.007,
    0.006, 0.005, 0.004, 0.003, 0.002, 0.002
], start_iter=313, n_neurons=100, gain=1,
    extra='eff_rank DROPS 5→1; 30k makes g=1 WORSE')

# Block 27: g=2 at 10k
add_block(27, 'g=2 (10k)', '10k', 17, [
    0.519, 0.397, 0.356, 0.300, 0.250, 0.200,
    0.150, 0.100, 0.050, 0.020, 0.010, 0.004
], start_iter=325, n_neurons=100, gain=2,
    extra='inverse lr_W=5E-4; epoch scaling not diminishing')

# Block 28: g=2 (30k)
add_block(28, 'g=2 (30k)', '30k', 16, [
    0.997, 0.983, 0.943, 0.871, 0.848, 0.640,
    0.500, 0.300, 0.125, 0.050, 0.010, 0.001
], start_iter=337, n_neurons=100, gain=2,
    extra='42% conv; inverse lr_W persists; eff_rank flat at 16')

# Block 29: g=2 n=200 (30k)
add_block(29, 'g=2 n=200 (30k)', '30k', 37, [
    0.979, 0.976, 0.972, 0.963, 0.962, 0.955,
    0.953, 0.944, 0.943, 0.942, 0.913, 0.877
], start_iter=349, n_neurons=200, gain=2,
    extra='92% conv; eff_rank=35-38; inverse lr_W=3E-4; contradicts n=100')


# --- Build Plotly figure ---
pts_10k = [p for p in points if p['n_frames'] == '10k']
pts_30k = [p for p in points if p['n_frames'] == '30k']


def make_hover(pts):
    return [
        (f"<b>Iter {p['iter']}</b> — Block {p['block']}<br>"
         f"<b>{p['regime']}</b><br>"
         f"n={p['n_neurons']}, {p['n_frames']} frames, "
         f"g={p['gain']}, fill={p['fill']}%<br>"
         f"eff_rank ≈ {p['x']:.0f} · conn R² = {p['y']:.3f}<br>"
         f"<i>{p['extra']}</i>")
        for p in pts
    ]


fig = go.Figure()

# 10k trace
fig.add_trace(go.Scatter(
    x=[p['x'] for p in pts_10k],
    y=[p['y'] for p in pts_10k],
    mode='markers',
    marker=dict(size=7, color='#546e7a', opacity=0.65),
    name='10k frames',
    text=make_hover(pts_10k),
    hoverinfo='text',
))

# 30k trace
fig.add_trace(go.Scatter(
    x=[p['x'] for p in pts_30k],
    y=[p['y'] for p in pts_30k],
    mode='markers',
    marker=dict(size=9, color='#2e7d32', opacity=0.80),
    name='30k frames',
    text=make_hover(pts_30k),
    hoverinfo='text',
))

# --- Migration arrows ---
# Green: rescued by ×3 data
arrow_style_green = dict(
    arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor='#2e7d32',
    showarrow=True, xref='x', yref='y', axref='x', ayref='y',
)
arrow_style_red = dict(
    arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor='#c62828',
    showarrow=True, xref='x', yref='y', axref='x', ayref='y',
)

# n=300: (~47, ~0.90) → (~80, ~1.000)
fig.add_annotation(x=78, y=0.997, ax=49, ay=0.90, **arrow_style_green)
# n=600: (~50, ~0.55) → (~87, ~0.965)
fig.add_annotation(x=85, y=0.960, ax=52, ay=0.55, **arrow_style_green)
# g=3/n=200: (~31, ~0.49) → (~55, ~0.99)
fig.add_annotation(x=53, y=0.985, ax=33, ay=0.49, **arrow_style_green)
# sparse 50%: (~21, ~0.41) → (~13, ~0.37)
fig.add_annotation(x=14, y=0.37, ax=20, ay=0.41, **arrow_style_red)
# fill=80%: (~36, ~0.80) → (~49, ~0.80)
fig.add_annotation(x=47, y=0.802, ax=38, ay=0.802, **arrow_style_red)
# g=1/10k: (~5, ~0.007) → g=1/30k: (~3, ~0.018) — NOT rescued
fig.add_annotation(x=3, y=0.018, ax=5, ay=0.007, **arrow_style_red)
# g=2/10k: (~17, ~0.52) → g=2/30k: (~16, ~0.997) — partially rescued
fig.add_annotation(x=16, y=0.99, ax=17, ay=0.52, **arrow_style_green)


# --- Dummy traces for arrow legend entries ---
fig.add_trace(go.Scatter(
    x=[None], y=[None], mode='lines',
    line=dict(color='#2e7d32', width=2),
    name='rescued by ×3 data',
))
fig.add_trace(go.Scatter(
    x=[None], y=[None], mode='lines',
    line=dict(color='#c62828', width=2),
    name='NOT rescued',
))

# --- Layout ---
fig.update_layout(
    title=dict(text='Regime Landscape', font=dict(size=18)),
    xaxis=dict(
        title='effective rank',
        range=[0, 155],
        dtick=10,
        gridcolor='rgba(0,0,0,0.08)',
        gridwidth=1,
    ),
    yaxis=dict(
        title='connectivity R²',
        range=[-0.03, 1.06],
        tickvals=[0, 0.25, 0.5, 0.75, 1],
        ticktext=['0%', '25%', '50%', '75%', '100%'],
        gridcolor='rgba(0,0,0,0.08)',
        gridwidth=1,
    ),
    template='plotly_white',
    legend=dict(
        x=0.01, y=0.01, xanchor='left', yanchor='bottom',
        bgcolor='rgba(255,255,255,0.85)',
        bordercolor='#eee', borderwidth=1,
        font=dict(size=12),
    ),
    width=950, height=680,
    hoverlabel=dict(
        bgcolor='white', font_size=12,
        font_family='monospace',
    ),
    margin=dict(l=60, r=30, t=50, b=60),
    plot_bgcolor='white',
)

fig.write_html(
    'assets/landscape_quadrant_interactive.html',
    include_plotlyjs='cdn',
    full_html=True,
    config={
        'displayModeBar': True,
        'modeBarButtonsToAdd': ['zoomIn2d', 'zoomOut2d'],
        'displaylogo': False,
        'scrollZoom': True,
    },
)
print("Saved: assets/landscape_quadrant_interactive.html")
