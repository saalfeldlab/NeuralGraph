"""
Mode-based rollout analysis for iter_021.
Project GT and predicted kinographs onto the true U basis to separate
spatial mode recovery (U) from temporal transition accuracy (V).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import torch

log_dir = '/workspace/NeuralGraph/log/signal/signal_iter_021'
results_dir = f'{log_dir}/results'

# Load kinographs: (n_neurons, n_frames)
gt = np.load(f'{results_dir}/kinograph_gt.npy')
pred = np.load(f'{results_dir}/kinograph_pred.npy')
n_neurons, n_frames = gt.shape
print(f'Kinograph: {n_neurons} neurons x {n_frames} frames')

# Load true connectivity and extract U_true
conn = torch.load(f'{log_dir}/connectivity.pt', map_location='cpu', weights_only=False)
# conn is the adjacency/W matrix — extract true U via SVD
if isinstance(conn, dict):
    W_true = conn.get('W', conn.get('adjacency', None))
    if W_true is None:
        W_true = list(conn.values())[0]
else:
    W_true = conn
W_true = np.array(W_true).squeeze()
print(f'W_true shape: {W_true.shape}')

# SVD to get true U (n_neurons x rank) and V (rank x n_neurons)
U_full, S_full, Vt_full = np.linalg.svd(W_true, full_matrices=False)
rank = 20
U_true = U_full[:, :rank]  # (100, 20)
V_true = Vt_full[:rank, :]  # (20, 100)
S_true = S_full[:rank]
print(f'Top-{rank} singular values: {S_true[:5].round(2)}...')

# --- Analysis 1: Project kinographs onto U basis ---
# mode_activations = U_true.T @ kinograph  ->  (rank, n_frames)
modes_gt = U_true.T @ gt      # (20, 10000)
modes_pred = U_true.T @ pred

# Per-mode temporal correlation
mode_corrs = []
for k in range(rank):
    r, _ = pearsonr(modes_gt[k], modes_pred[k])
    mode_corrs.append(r)
mode_corrs = np.array(mode_corrs)
print(f'\nPer-mode temporal correlation (GT vs pred projected onto U_true):')
for k in range(rank):
    print(f'  Mode {k:2d} (σ={S_true[k]:.2f}): r={mode_corrs[k]:.3f}')

# --- Analysis 2: Spatial correlation per frame ---
spatial_corrs = np.array([pearsonr(gt[:, t], pred[:, t])[0] for t in range(n_frames)])

# --- Analysis 3: Residual after removing U-projected component ---
# If U modes explain the prediction well, the residual should be small
pred_in_U = U_true @ modes_pred   # reconstruct pred from its U-projection
residual_norm = np.linalg.norm(pred - pred_in_U, axis=0)
gt_in_U = U_true @ modes_gt
gt_residual_norm = np.linalg.norm(gt - gt_in_U, axis=0)

# --- Plot ---
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('white')
gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)

# Row 1: Mode activations for top 4 modes
for i in range(4):
    ax = fig.add_subplot(gs[0, 0] if i < 2 else gs[0, 1])
    if i == 0 or i == 2:
        ax2 = ax
    break

# Better layout: top row = mode activations, middle = spatial corr, bottom = mode corr bar
ax_modes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
ax_spatial = fig.add_subplot(gs[1, :])
ax_bar = fig.add_subplot(gs[2, 0])
ax_scatter = fig.add_subplot(gs[2, 1])
ax_kino_gt = fig.add_subplot(gs[3, 0])
ax_kino_pred = fig.add_subplot(gs[3, 1])

# Top-left: mode 0 activation
t = np.arange(n_frames)
for idx, ax in enumerate(ax_modes):
    k = idx  # mode 0 and mode 1
    ax.plot(t[::10], modes_gt[k, ::10], color='#2563eb', linewidth=0.5, alpha=0.8, label='GT')
    ax.plot(t[::10], modes_pred[k, ::10], color='#dc2626', linewidth=0.5, alpha=0.8, label='pred')
    ax.set_title(f'Mode {k} activation (σ={S_true[k]:.1f}, r={mode_corrs[k]:.3f})', fontsize=9)
    ax.set_xlabel('frames', fontsize=8)
    ax.legend(fontsize=7, loc='upper right')
    ax.tick_params(labelsize=7)

# Middle: spatial correlation over time
ax_spatial.plot(t[::5], spatial_corrs[::5], color='#374151', linewidth=0.4)
ax_spatial.axhline(y=np.mean(spatial_corrs), color='#dc2626', linewidth=0.8, linestyle='--',
                   label=f'mean = {np.mean(spatial_corrs):.3f}')
ax_spatial.set_title('Cross-neuron correlation per frame (GT vs pred)', fontsize=9)
ax_spatial.set_xlabel('frames', fontsize=8)
ax_spatial.set_ylabel('Pearson r', fontsize=8)
ax_spatial.set_ylim(-0.5, 1.0)
ax_spatial.legend(fontsize=8)
ax_spatial.tick_params(labelsize=7)

# Bar chart: per-mode correlation
colors = ['#2563eb' if c > 0.5 else '#f59e0b' if c > 0 else '#dc2626' for c in mode_corrs]
ax_bar.bar(range(rank), mode_corrs, color=colors, edgecolor='none', width=0.7)
ax_bar.set_xlabel('SVD mode index', fontsize=8)
ax_bar.set_ylabel('temporal r', fontsize=8)
ax_bar.set_title(f'Per-mode temporal correlation (mean={mode_corrs.mean():.3f})', fontsize=9)
ax_bar.axhline(y=0, color='#9ca3af', linewidth=0.5)
ax_bar.tick_params(labelsize=7)

# Scatter: mode energy in GT vs pred
energy_gt = np.var(modes_gt, axis=1)
energy_pred = np.var(modes_pred, axis=1)
ax_scatter.scatter(energy_gt, energy_pred, c='#2563eb',
                   edgecolors='#555555', linewidths=0.5, s=40)
for k in range(rank):
    ax_scatter.annotate(str(k), (energy_gt[k], energy_pred[k]), fontsize=6, ha='center', va='bottom')
ax_scatter.plot([0, energy_gt.max()], [0, energy_gt.max()], '--', color='#9ca3af', linewidth=0.5)
ax_scatter.set_xlabel('GT mode variance', fontsize=8)
ax_scatter.set_ylabel('Pred mode variance', fontsize=8)
ax_scatter.set_title('Mode power: GT vs pred', fontsize=9)
ax_scatter.tick_params(labelsize=7)

# Bottom: kinographs sorted by U-mode 0 loading
sort_idx = np.argsort(U_true[:, 0])
vmin = min(gt.min(), pred.min())
vmax = max(gt.max(), pred.max())
ax_kino_gt.imshow(gt[sort_idx], aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
ax_kino_gt.set_title('GT (neurons sorted by mode-0 loading)', fontsize=9)
ax_kino_gt.set_ylabel('neurons', fontsize=8)
ax_kino_gt.set_xlabel('frames', fontsize=8)
ax_kino_gt.tick_params(labelsize=7)

ax_kino_pred.imshow(pred[sort_idx], aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
ax_kino_pred.set_title('Pred (neurons sorted by mode-0 loading)', fontsize=9)
ax_kino_pred.set_ylabel('neurons', fontsize=8)
ax_kino_pred.set_xlabel('frames', fontsize=8)
ax_kino_pred.tick_params(labelsize=7)

plt.savefig(f'{results_dir}/kinograph_mode_analysis.png', dpi=200, bbox_inches='tight')
print(f'\nSaved kinograph_mode_analysis.png')

# Summary stats
print(f'\n=== SUMMARY ===')
print(f'Spatial correlation (per-frame):  mean={spatial_corrs.mean():.3f}, '
      f'std={spatial_corrs.std():.3f}, min={spatial_corrs.min():.3f}')
print(f'Mode temporal correlation:        mean={mode_corrs.mean():.3f}, '
      f'top-5 mean={mode_corrs[:5].mean():.3f}')
print(f'Mode energy ratio (pred/gt):      {energy_pred.sum()/energy_gt.sum():.3f}')
