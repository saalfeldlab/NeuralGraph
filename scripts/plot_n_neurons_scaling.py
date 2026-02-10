"""
Flat-design plot: influence of n_neurons on low-rank recovery metrics.
Generates assets/n_neurons_scaling.png for case-low-rank.qmd.
"""
import matplotlib.pyplot as plt
import numpy as np

# Data: best per scale from dedicated exploration
n = np.array([100, 200, 400, 600, 1000])
conn_R2 = np.array([1.000, 0.861, 0.999, 0.9995, 0.992])
roll_R2 = np.array([0.999, 0.996, 0.990, 0.963, 0.997])
V_R2 = np.array([0.97, 0.859, 0.993, 0.995, 0.989])
hard_rate = np.array([30.0, 40.5, np.nan, np.nan, np.nan])  # only measured at 100, 200

# Flat design palette (muted, no edge colors)
c_conn = '#2563eb'   # blue
c_roll = '#16a34a'   # green
c_V = '#dc2626'      # red

fig, ax1 = plt.subplots(figsize=(7, 4))

# Clean style
ax1.set_facecolor('white')
fig.patch.set_facecolor('white')
for spine in ax1.spines.values():
    spine.set_color('#d1d5db')
    spine.set_linewidth(0.8)

# Plot metrics
ms = 4
lw = 1.0
ax1.plot(n, conn_R2, '-o', color=c_conn, markersize=ms, linewidth=lw,
         markeredgecolor='none', label='connectivity R²', zorder=3)
ax1.plot(n, roll_R2, '-s', color=c_roll, markersize=ms, linewidth=lw,
         markeredgecolor='none', label='rollout R²', zorder=3)
ax1.plot(n, V_R2, '-^', color=c_V, markersize=ms, linewidth=lw,
         markeredgecolor='none', label='V R²', zorder=3)

# Axes
ax1.set_xlabel('n_neurons', fontsize=9, fontfamily='sans-serif')
ax1.set_ylabel('R²  (best per scale)', fontsize=9, fontfamily='sans-serif')
ax1.set_xscale('log')
ax1.set_xticks(n)
ax1.set_xticklabels([str(x) for x in n], fontsize=8)
ax1.minorticks_off()
ax1.set_ylim(0, 1.05)
ax1.set_xlim(80, 1200)

# Light grid
ax1.grid(axis='y', color='#e5e7eb', linewidth=0.4, zorder=0)
ax1.grid(axis='x', color='#e5e7eb', linewidth=0.4, zorder=0)
ax1.tick_params(colors='#374151', labelsize=8)

# Annotate the n=200 dip
ax1.annotate('n=200 dip', xy=(200, 0.861), xytext=(280, 0.82),
             fontsize=7, color='#6b7280', fontstyle='italic',
             arrowprops=dict(arrowstyle='->', color='#9ca3af', lw=0.6))

# Legend
ax1.legend(loc='lower right', frameon=True, framealpha=0.9,
           edgecolor='#e5e7eb', fontsize=8, handlelength=1.5)

plt.tight_layout()
plt.savefig('/workspace/NeuralGraph/assets/n_neurons_scaling.png', dpi=200, bbox_inches='tight')
print('Saved assets/n_neurons_scaling.png')
