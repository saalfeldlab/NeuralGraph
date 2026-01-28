#!/usr/bin/env python3
"""
Generate quadrant chart for landscape exploration results.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data: (name, eff_rank_normalized, recovery, label_offset)
# eff_rank_normalized: 0-1 scale where 0=low rank, 1=high rank
# recovery: 0-1 scale where 0=failed, 1=perfect
regimes = [
    ('Chaotic', 0.75, 0.98, (5, 5)),
    ('Dale', 0.70, 0.98, (5, -12)),
    ('Noise', 0.95, 0.98, (5, 5)),
    ('n=200', 0.75, 0.88, (-35, 5)),
    ('n=300', 0.78, 0.85, (5, 5)),
    ('n=500', 0.80, 0.80, (5, -12)),
    ('ff=0.75', 0.65, 0.93, (5, 5)),
    ('ff=0.9', 0.78, 0.95, (-40, 5)),
    ('n_types=2', 0.75, 0.55, (5, 5)),
    ('ff=0.5', 0.45, 0.50, (5, 5)),
    ('Low-rank', 0.25, 0.40, (5, 5)),
    ('Low-rank+types', 0.25, 0.37, (5, -12)),
    ('Sparse', 0.12, 0.08, (5, 5)),
    ('Sparse+Noise', 0.95, 0.20, (-75, -12)),
]

fig, ax = plt.subplots(figsize=(10, 8))

# Draw colored quadrants
ax.fill_between([0, 0.5], [0.5, 0.5], [1, 1], color='#fff3cd', alpha=0.5, label='Challenging')
ax.fill_between([0.5, 1], [0.5, 0.5], [1, 1], color='#d4edda', alpha=0.5, label='Easy Mode')
ax.fill_between([0, 0.5], [0, 0], [0.5, 0.5], color='#f8d7da', alpha=0.5, label='Unrecoverable')
ax.fill_between([0.5, 1], [0, 0], [0.5, 0.5], color='#cce5ff', alpha=0.5, label='Achievable')

# Draw quadrant lines
ax.axhline(y=0.5, color='gray', linestyle='-', linewidth=1, alpha=0.5)
ax.axvline(x=0.5, color='gray', linestyle='-', linewidth=1, alpha=0.5)

# Quadrant labels
ax.text(0.25, 0.92, 'Challenging', ha='center', va='center', fontsize=14, color='#856404')
ax.text(0.75, 0.92, 'Easy Mode', ha='center', va='center', fontsize=14, color='#155724')
ax.text(0.25, 0.08, 'Unrecoverable', ha='center', va='center', fontsize=14, color='#721c24')
ax.text(0.75, 0.08, 'Achievable', ha='center', va='center', fontsize=14, color='#004085')

# Plot points
for name, x, y, offset in regimes:
    # Color based on quadrant
    if x < 0.5 and y < 0.5:
        color = '#dc3545'  # red - unrecoverable
    elif x < 0.5 and y >= 0.5:
        color = '#ffc107'  # yellow - challenging
    elif x >= 0.5 and y < 0.5:
        color = '#007bff'  # blue - achievable
    else:
        color = '#28a745'  # green - easy

    ax.scatter(x, y, s=120, c=color, edgecolors='white', linewidths=1.5, zorder=5)
    ax.annotate(name, (x, y), xytext=offset, textcoords='offset points',
                fontsize=10, ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))

# Labels and title
ax.set_xlabel('Effective Rank', fontsize=14)
ax.set_ylabel('Connectivity Recovery (R²)', fontsize=14)
ax.set_title('Regime Difficulty vs Recovery', fontsize=16, pad=15)

# Set limits with padding
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)

# Custom ticks
ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax.set_xticklabels(['Low', '', 'Medium', '', 'High'])
ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])

# Grid
ax.grid(True, alpha=0.3, linestyle='--')

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('landscape_quadrant.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: landscape_quadrant.png")
plt.close()
