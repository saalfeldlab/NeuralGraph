import numpy as np

import matplotlib.pyplot as plt

from GNN_PlotFigure import compare_experiments


def plot_gnn_histograms(runs, save_path, title, bins=50):
    metrics = {
        'weights R²': [r['r2_mean'] for r in runs],
        'tau R²': [r['tau_r2_mean'] for r in runs],
        'V_rest R²': [r['vrest_r2_mean'] for r in runs],
        'clustering accuracy': [r['acc_mean'] for r in runs],
    }
    colors = {
        'weights R²': 'tab:blue',
        'tau R²': 'tab:green',
        'V_rest R²': 'tab:orange',
        'clustering accuracy': 'tab:red',
    }

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))

    for ax, (label, values) in zip(axes.flat, metrics.items()):
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]

        ax.hist(vals, bins=bins, color=colors[label])
        ax.set_xlabel(label)
        ax.set_ylabel('count')

        if vals.size:
            mn = float(np.min(vals))
            mx = float(np.max(vals))
            med = float(np.median(vals))
            sd = float(np.std(vals))

            stats_txt = f"min: {mn:.3g}\nmax: {mx:.3g}\nmedian: {med:.3g}\nstd: {sd:.3g}"
            ax.text(
                0.98,
                0.98,
                stats_txt,
                transform=ax.transAxes,
                ha='right',
                va='top',
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'),
            )
            ax.axvline(med, color='k', linestyle='--', alpha=0.7)
        ax.set_xlim(0, 1)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
    fig.savefig(save_path)


if __name__ == "__main__":

    without_noise = compare_experiments([f'fly_N9_22_10__mid_{i:03d}' for i in range(50)], varied_parameter=None)
    plot_gnn_histograms(without_noise['gnn'], 'histograms_without_noise.png', 'Without noise')

    with_noise = compare_experiments([f'fly_N9_44_24__mid_{i:03d}' for i in range(50)], varied_parameter=None)
    plot_gnn_histograms(with_noise['gnn'], 'histograms_with_noise.png', 'With noise')