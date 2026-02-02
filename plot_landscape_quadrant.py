#!/usr/bin/env python3
"""
Scatter chart — all iterations, flat design.
268 iterations, 23 blocks.
X-axis: absolute effective rank.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

np.random.seed(42)

fig, ax = plt.subplots(figsize=(12, 9))

# --- Generate per-iteration data ---
# Each entry: (eff_rank, conn_R2)
pts_10k = []
pts_30k = []


def add_iters(target, center_x, conns, jitter=1.5):
    for c in conns:
        x = center_x + np.random.uniform(-jitter, jitter)
        target.append((x, c))


# Block 1: Chaotic n=100, eff_rank~35
add_iters(pts_10k, 35, [
    0.9999, 0.998, 0.996, 0.996, 0.993, 0.990, 0.985, 0.981, 0.970, 0.960, 0.940, 0.385
])

# Block 2: Low-rank n=100, eff_rank~12
add_iters(pts_10k, 12, [
    0.9997, 0.997, 0.996, 0.990, 0.980, 0.950, 0.925, 0.902, 0.899, 0.580, 0.500, 0.420
])

# Block 3: Dale n=100, eff_rank~12
add_iters(pts_10k, 12, [
    0.986, 0.974, 0.972, 0.958, 0.940, 0.920, 0.900, 0.880, 0.555, 0.458, 0.455, 0.420
])

# Block 4: Hetero n=100, eff_rank~38
add_iters(pts_10k, 38, [
    0.9996, 0.998, 0.992, 0.990, 0.985, 0.970, 0.950, 0.930, 0.900, 0.870, 0.670, 0.500
])

# Block 5: Noise n=100, eff_rank 42-90 — spread across range
for c in [1.000, 1.000, 1.000, 1.000, 0.998, 0.998, 0.997, 0.996, 0.995, 0.990, 0.985, 0.980]:
    x = np.random.uniform(42, 90)
    pts_10k.append((x, c))

# Block 6: n=200, eff_rank~43
add_iters(pts_10k, 43, [
    0.956, 0.950, 0.940, 0.930, 0.920, 0.910, 0.905, 0.890, 0.800, 0.750, 0.650, 0.575
])

# Block 7: Sparse n=100, eff_rank~21
add_iters(pts_10k, 21, [
    0.466, 0.450, 0.440, 0.430, 0.423, 0.410, 0.400, 0.390, 0.380, 0.350, 0.320, 0.310
])

# Block 8: Sparse+Noise n=100, eff_rank~91
add_iters(pts_10k, 91, [
    0.490, 0.489, 0.489, 0.489, 0.489, 0.485, 0.480, 0.475, 0.470, 0.460, 0.300, 0.054
])

# Block 9: n=300 1-2ep, eff_rank~47
add_iters(pts_10k, 47, [
    0.890, 0.870, 0.850, 0.830, 0.810, 0.805, 0.780, 0.750, 0.730, 0.720, 0.710, 0.699
])

# Block 10: n=300 2-4ep, eff_rank~47 (8 iters)
add_iters(pts_10k, 47, [
    0.924, 0.920, 0.910, 0.897, 0.893, 0.886, 0.870, 0.850
])

# Block 11: n=200 v2, eff_rank~43
add_iters(pts_10k, 43, [
    0.994, 0.993, 0.990, 0.988, 0.985, 0.980, 0.975, 0.970, 0.965, 0.960, 0.950, 0.935
])

# Block 12: n=600, eff_rank~50
add_iters(pts_10k, 50, [
    0.626, 0.600, 0.580, 0.560, 0.554, 0.540, 0.520, 0.500, 0.480, 0.450, 0.350, 0.000
])

# Block 13: n=200 + 4 types, eff_rank~42 (16 iters)
add_iters(pts_10k, 42, [
    0.991, 0.988, 0.985, 0.980, 0.975, 0.960, 0.948, 0.940,
    0.932, 0.920, 0.910, 0.908, 0.890, 0.870, 0.850, 0.830
])

# Block 14: Recurrent n=200, eff_rank~42 (4 completed only)
add_iters(pts_10k, 42, [
    0.993, 0.990, 0.912, 0.772
])

# Block 15: n=300 at 30k, eff_rank~80 (12 iters)
add_iters(pts_30k, 80, [
    1.000, 1.000, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999
])

# Block 16: n=600 at 30k, eff_rank~87 (8 iters)
add_iters(pts_30k, 87, [
    0.992, 0.976, 0.973, 0.967, 0.960, 0.950, 0.940, 0.933
])

# Block 17: Sparse 50% at 30k, eff_rank~13 (12 iters)
add_iters(pts_30k, 13, [
    0.436, 0.420, 0.410, 0.400, 0.390, 0.380, 0.370, 0.350, 0.330, 0.300, 0.260, 0.213
])

# Block 18: n=1000 at 30k, eff_rank~144 (12 iters)
add_iters(pts_30k, 144, [
    0.745, 0.743, 0.734, 0.726, 0.720, 0.716, 0.710, 0.700, 0.690, 0.680, 0.666, 0.640
], jitter=3)

# Block 19: g=3 n=100, eff_rank~26 (12 iters)
add_iters(pts_10k, 26, [
    0.955, 0.940, 0.920, 0.906, 0.880, 0.850, 0.820, 0.790, 0.750, 0.636, 0.600, 0.550
])

# Block 20: g=3 n=200/10k, eff_rank~31 (12 iters)
add_iters(pts_10k, 31, [
    0.489, 0.480, 0.470, 0.460, 0.450, 0.440, 0.420, 0.400, 0.380, 0.360, 0.340, 0.300
])

# Block 21: g=3 n=200 at 30k, eff_rank~55 (12 iters)
add_iters(pts_30k, 55, [
    0.996, 0.995, 0.994, 0.993, 0.992, 0.990, 0.988, 0.985, 0.982, 0.980, 0.975, 0.970
])

# Block 22: fill=80% at 10k, eff_rank~36 (12 iters)
add_iters(pts_10k, 36, [
    0.802, 0.802, 0.802, 0.802, 0.802, 0.802, 0.802, 0.802, 0.801, 0.801, 0.800, 0.800
])

# Block 23: fill=80% at 30k, eff_rank~49 (4 iters so far)
add_iters(pts_30k, 49, [
    0.802, 0.802, 0.802, 0.802
])

# --- Plot dots (darker) ---
x_10k = [p[0] for p in pts_10k]
y_10k = [p[1] for p in pts_10k]
ax.scatter(x_10k, y_10k, s=20, c='#546e7a', alpha=0.65, edgecolors='none', zorder=3)

x_30k = [p[0] for p in pts_30k]
y_30k = [p[1] for p in pts_30k]
ax.scatter(x_30k, y_30k, s=26, c='#2e7d32', alpha=0.80, edgecolors='none', zorder=4)

# --- Migration arrows (10k → 30k) ---
# n=300 centroid: (~47, ~0.86) → (~80, ~1.000)
ax.annotate('', xy=(78, 0.997), xytext=(49, 0.90),
            arrowprops=dict(arrowstyle='->', color='#2e7d32', lw=1.8,
                           connectionstyle='arc3,rad=0.08'))

# n=600 centroid: (~50, ~0.50) → (~87, ~0.965)
ax.annotate('', xy=(85, 0.960), xytext=(52, 0.55),
            arrowprops=dict(arrowstyle='->', color='#2e7d32', lw=1.8,
                           connectionstyle='arc3,rad=0.08'))

# g=3/n=200: (~31, ~0.49) → (~55, ~0.99)
ax.annotate('', xy=(53, 0.985), xytext=(33, 0.49),
            arrowprops=dict(arrowstyle='->', color='#2e7d32', lw=1.8,
                           connectionstyle='arc3,rad=0.08'))

# sparse 50%: (~21, ~0.40) → (~13, ~0.35) (eff_rank DROPS)
ax.annotate('', xy=(14, 0.37), xytext=(20, 0.41),
            arrowprops=dict(arrowstyle='->', color='#c62828', lw=1.8,
                           connectionstyle='arc3,rad=-0.1'))

# fill=80%: (~36, ~0.80) → (~49, ~0.80) (horizontal — no improvement)
ax.annotate('', xy=(47, 0.802), xytext=(38, 0.802),
            arrowprops=dict(arrowstyle='->', color='#c62828', lw=1.8,
                           connectionstyle='arc3,rad=0.05'))

# Arrow labels
ax.text(76, 0.73, '×3 n_frames\n10k → 30k', ha='center', va='center',
        fontsize=9, color='#2e7d32')

ax.text(10, 0.30, 'sparse\n30k fails', ha='center', va='center',
        fontsize=8, color='#c62828')

ax.text(44, 0.86, 'fill=80%\n30k fails', ha='center', va='center',
        fontsize=8, color='#c62828')

# --- Axes ---
ax.set_xlabel('effective rank', fontsize=13)
ax.set_ylabel('connectivity R²', fontsize=13)

ax.set_xlim(5, 155)
ax.set_ylim(-0.03, 1.06)
ax.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140])
ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=9)

ax.grid(True, alpha=0.15, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- Legend (minimal) ---
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#546e7a',
           markersize=9, markeredgecolor='none', label='10k frames'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2e7d32',
           markersize=9, markeredgecolor='none', label='30k frames'),
    Line2D([0], [0], color='#2e7d32', lw=1.8, label='rescued by ×3 data'),
    Line2D([0], [0], color='#c62828', lw=1.8, label='NOT rescued'),
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=12,
          framealpha=0.7, edgecolor='#eee', handletextpad=0.5)

plt.tight_layout()
plt.savefig('assets/landscape_quadrant.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: assets/landscape_quadrant.png")
plt.close()
