"""
Sorted kinograph comparison for iter_021.
Reorder neurons to test whether GNN rollout is a row-permutation of ground truth.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment

log_dir = '/workspace/NeuralGraph/log/signal/signal_iter_021/results'

# Load kinographs: shape (n_neurons, n_frames)
gt = np.load(f'{log_dir}/kinograph_gt.npy')
pred = np.load(f'{log_dir}/kinograph_pred.npy')
print(f'gt shape: {gt.shape}, pred shape: {pred.shape}')

n_neurons, n_frames = gt.shape

# --- Method 1: Sort both by mean activity (same ordering for both) ---
gt_mean = gt.mean(axis=1)
sort_idx_gt = np.argsort(gt_mean)

# --- Method 2: Hungarian matching - find best neuron permutation ---
# Build cost matrix: negative correlation between each gt neuron and each pred neuron
print('Computing neuron-to-neuron correlation matrix...')
corr_matrix = np.zeros((n_neurons, n_neurons))
for i in range(n_neurons):
    for j in range(n_neurons):
        corr_matrix[i, j], _ = pearsonr(gt[i], pred[j])

# Hungarian: maximize correlation (minimize negative correlation)
row_ind, col_ind = linear_sum_assignment(-corr_matrix)
# col_ind[i] = which pred neuron best matches gt neuron i
pred_matched = pred[col_ind]

# Compute R² before and after matching
def compute_r2(a, b):
    ss_res = np.sum((a - b)**2)
    ss_tot = np.sum((a - a.mean())**2)
    return 1 - ss_res / ss_tot

r2_original = compute_r2(gt, pred)
r2_matched = compute_r2(gt, pred_matched)

# Mean per-neuron correlation before/after
corr_orig = np.mean([pearsonr(gt[i], pred[i])[0] for i in range(n_neurons)])
corr_matched = np.mean([pearsonr(gt[i], pred_matched[i])[0] for i in range(n_neurons)])
print(f'Original:  R²={r2_original:.4f}, mean neuron corr={corr_orig:.4f}')
print(f'Matched:   R²={r2_matched:.4f}, mean neuron corr={corr_matched:.4f}')

# --- Method 3: Sort both gt and pred by gt mean activity ---
gt_sorted = gt[sort_idx_gt]
pred_sorted = pred[sort_idx_gt]

# --- Plot ---
vmin = min(gt.min(), pred.min())
vmax = max(gt.max(), pred.max())

fig, axes = plt.subplots(2, 3, figsize=(14, 6))
fig.patch.set_facecolor('white')

titles = [
    'ground truth', 'GNN rollout', 'residual',
    'GT (sorted by mean)', 'GNN (same sorting)', 'GNN (Hungarian matched)',
]
data = [
    gt, pred, gt - pred,
    gt_sorted, pred_sorted, pred_matched[sort_idx_gt],
]

for ax, title, d in zip(axes.flat, titles, data):
    if 'residual' in title:
        vlim = max(abs(d.min()), abs(d.max()))
        im = ax.imshow(d, aspect='auto', cmap='RdBu_r', vmin=-vlim, vmax=vlim,
                       interpolation='none')
    else:
        im = ax.imshow(d, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax,
                       interpolation='none')
    ax.set_title(title, fontsize=9)
    ax.set_ylabel('neurons', fontsize=8)
    ax.set_xlabel('frames', fontsize=8)
    ax.tick_params(labelsize=7)

fig.suptitle(
    f'iter_021 rollout — original R²={r2_original:.3f} | '
    f'Hungarian-matched R²={r2_matched:.3f}\n'
    f'mean neuron corr: {corr_orig:.3f} → {corr_matched:.3f}',
    fontsize=10
)
plt.tight_layout()
plt.savefig(f'{log_dir}/kinograph_sorted_comparison.png', dpi=200, bbox_inches='tight')
print(f'Saved kinograph_sorted_comparison.png')

# --- Also show the permutation mapping ---
print(f'\nHungarian permutation stats:')
identity = np.arange(n_neurons)
n_fixed = np.sum(col_ind == identity)
print(f'  Neurons mapped to themselves: {n_fixed}/{n_neurons}')
displacement = np.abs(col_ind - identity)
print(f'  Mean displacement: {displacement.mean():.1f} neurons')
print(f'  Max displacement: {displacement.max()} neurons')

# Diagonal correlation stats
diag_corrs = np.array([corr_matrix[i, i] for i in range(n_neurons)])
matched_corrs = np.array([corr_matrix[i, col_ind[i]] for i in range(n_neurons)])
print(f'  Mean diagonal (identity) correlation: {diag_corrs.mean():.4f}')
print(f'  Mean matched correlation: {matched_corrs.mean():.4f}')
