import numpy as np
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from tqdm import trange
from itertools import product, combinations
from math import log
from tqdm import tqdm
from collections import defaultdict
import os
from matplotlib import pyplot as plt

def analyze_ising_model(x_list, delta_t, log_dir, logger, mc):
    """
    Perform comprehensive Ising model analysis including energy distribution,
    coupling analysis, information ratio estimation, and triplet KL analysis.

    Args:
        x_list: List of input data arrays
        log_dir: Directory for saving results
        logger: Logger object for recording results
        mc: Matplotlib color for plots

    Returns:
        dict: Dictionary containing all analysis results
    """

    # Load or compute Ising fit
    if os.path.exists(f"./{log_dir}/results/E.npy"):
        print(f'loading existing sparse Ising analysis...')
        E = np.load(f"./{log_dir}/results/E.npy")
        s = np.load(f"./{log_dir}/results/s.npy")
        h = np.load(f"./{log_dir}/results/h.npy")
        J = np.load(f"./{log_dir}/results/J.npy", allow_pickle=True)
    else:
        print(f'Computing sparse Ising analysis...')
        energy_stride = 1
        s, h, J, E = sparse_ising_fit_fast(
            x=x_list[0],
            voltage_col=3,
            top_k=51,
            block_size=2000,
            energy_stride=energy_stride
        )
        np.save(f"./{log_dir}/results/E.npy", E)
        np.save(f"./{log_dir}/results/s.npy", s)
        np.save(f"./{log_dir}/results/h.npy", h)
        np.save(f"./{log_dir}/results/J.npy", J)

    # Energy statistics
    E_mean = np.mean(E)
    E_std = np.std(E)
    hist, _ = np.histogram(E, bins=100, density=True)
    E_entropy = -np.sum(hist * np.log(hist + 1e-12))

    # Coupling statistics
    J_vals = []
    for Ji in J:
        if isinstance(Ji, dict):
            J_vals.extend(Ji.values())
        else:
            J_vals.extend(np.asarray(Ji).ravel())
    J_vals = np.asarray(J_vals, dtype=np.float32)
    J_vals = J_vals[np.isfinite(J_vals)]

    J_mean = np.mean(J_vals)
    J_std = np.std(J_vals)
    J_sign_ratio = (J_vals > 0).mean()
    th = np.percentile(np.abs(J_vals), 90.0)
    J_frac_strong = (np.abs(J_vals) > th).mean()

    # Create visualization
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    # Panel 1: Energy histogram
    axs[0].hist(E, bins=100, color='salmon', edgecolor=mc, density=True)
    axs[0].set_xlabel("Energy", fontsize=24)
    axs[0].set_ylabel("Density", fontsize=24)
    axs[0].tick_params(axis='both', which='major', labelsize=12)
    axs[0].text(0.05, 0.95,
                f'Mean: {E_mean:.2f}\nStd: {E_std:.2f}\nEntropy: {E_entropy:.2f}',
                transform=axs[0].transAxes,
                fontsize=18, verticalalignment='top')
    # Panel 2: Couplings histogram
    axs[1].hist(J_vals, bins=100, color='skyblue', edgecolor=mc, density=True)
    axs[1].set_xlabel(r"Coupling strength $J_{ij}$", fontsize=24)
    axs[1].set_ylabel("Density", fontsize=24)
    axs[1].tick_params(axis='both', which='major', labelsize=12)
    axs[1].text(0.05, 0.95,
                f'Mean: {J_mean:.2f}\nStd: {J_std:.2f}\nSign ratio: {J_sign_ratio:.2f}\nFrac strong: {J_frac_strong:.2f}',
                transform=axs[1].transAxes,
                fontsize=18, verticalalignment='top')
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/Ising_panels.png", dpi=150)
    plt.close(fig)

    # Information ratio analysis
    n_subsets = 1000
    N = 10
    bin_size = 1
    voltage_col = 3
    seed = 0
    rng = np.random.RandomState(seed)

    print('computing information ratio estimates...')
    results = []
    for i in trange(n_subsets, desc="processing subsets"):
        idx = np.sort(rng.choice(x_list[0].shape[1], size=N, replace=False))

        s_subset = rebin_voltage_subset(
            x_list[0], idx,
            bin_size=bin_size,
            voltage_col=voltage_col,
            agg="mean",
            threshold="mean"
        )

        point = compute_info_ratio_estimator(
            s_subset,
            logbase=2.0,
            alpha_joint=1e-3,
            alpha_marg=0.5,
            delta_t = delta_t,
            enforce_monotone=True
        )
        results.append(point)

    # Extract results
    INs = np.array([r.I_N for r in results])
    I2s = np.array([r.I2 for r in results])
    ratios = np.array([r.ratio for r in results])
    non_monotonic = np.array([r.count_non_monotonic for r in results])

    # Plot like Schneidman Figure 2a
    # Collect all patterns from all subsets
    all_observed = []
    all_predicted_pairwise = []
    all_predicted_independent = []

    for r in results:
        all_observed.extend(r.observed_rates)
        all_predicted_pairwise.extend(r.predicted_rates_pairwise)
        all_predicted_independent.extend(r.predicted_rates_independent)

    # Create 2x3 subplot layout (removed panel d, added independent model to panel c)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Panel A: Pattern rates scatter plot (like Figure 2a)
    ax1.loglog(all_observed, all_predicted_pairwise, 'ro', alpha=0.1, markersize=0.5,
               label='pairwise model')
    ax1.loglog(all_observed, all_predicted_independent, 'go', alpha=0.1, markersize=0.5,
               label='independent model')
    ax1.plot([1e-4, 1e1], [1e-4, 1e1], 'w-', linewidth=2)
    ax1.set_xlim(1e-4, 1e1)
    ax1.set_ylim(1e-4, 1e1)
    ax1.set_xlabel('observed rate', fontsize=18)
    ax1.set_ylabel('predicted rate', fontsize=18)
    # ax1.legend(fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    js_pairwise = []
    js_independent = []

    for r in results:
        obs = r.observed_rates + 1e-12  # avoid log(0)
        pred_pair = r.predicted_rates_pairwise + 1e-12
        pred_indep = r.predicted_rates_independent + 1e-12

        # Jensen-Shannon divergence: JS(P,Q) = 0.5*[KL(P,M) + KL(Q,M)] where M=(P+Q)/2
        def js_divergence(p, q):
            m = 0.5 * (p + q)
            return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))

        js_pairwise.append(js_divergence(obs, pred_pair))
        js_independent.append(js_divergence(obs, pred_indep))

    ax2.hist(js_independent, bins=50, alpha=0.7, color='green', label='independent model', density=True)
    ax2.hist(js_pairwise, bins=50, alpha=0.7, color='red', label='pairwise model', density=True)
    ax2.set_xlabel('jensen-shannon divergence', fontsize=18)
    ax2.set_ylabel('probability density', fontsize=18)
    ax2.set_xscale('log')
    ax2.legend(fontsize=16)
    ax2.set_xlim(1e-2, 2e1)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax3.scatter(INs, ratios, c='red', alpha=0.6, s=20, edgecolors='none', label='pairwise model')
    ax3.set_xlabel(r'multi-information $I_N$ (bits)', fontsize=18)
    ax3.set_ylabel(r'$I^{(2)}/I_N$', fontsize=18)
    ax3.set_ylim(0, 1.1)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0.9, color='black', linestyle='--', alpha=0.5)
    ax3.tick_params(axis='both', which='major', labelsize=16)
    ax3.set_xlim(0, 150)
    ax3.set_ylim(0, 1.1)
    ax4.set_visible(False)

    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/Ising_rates.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    results_dict = {
        'I_N': INs,
        'I2': I2s,
        'ratio': ratios,
        'count_non_monotonic': non_monotonic,
        'H_true': np.array([r.H_true for r in results]),
        'H_indep': np.array([r.H_indep for r in results]),
        'H_pair': np.array([r.H_pair for r in results]),
        'predicted_rates_pairwise': np.array([r.predicted_rates_pairwise for r in results]),
        'predicted_rates_independent': np.array([r.predicted_rates_independent for r in results]),
        'observed_rates': np.array([r.observed_rates for r in results])
    }

    np.savez_compressed(f"{log_dir}/results/info_ratio_results.npz", **results_dict)

    # Log results
    print(f"non monotonic ratio {non_monotonic.sum()} out of {n_subsets}")

    q25_IN, q75_IN = np.nanpercentile(INs, [25, 75])
    q25_I2, q75_I2 = np.nanpercentile(I2s, [25, 75])
    q25_ratio, q75_ratio = np.nanpercentile(ratios, [25, 75])

    print(f"I_N:    median={np.nanmedian(INs):.3f},   IQR=[{q25_IN:.3f}, {q75_IN:.3f}],   std={np.nanstd(INs):.3f}")
    print(f"I2:     median={np.nanmedian(I2s):.3f},   IQR=[{q25_I2:.3f}, {q75_I2:.3f}],   std={np.nanstd(I2s):.3f}")
    print(f"ratio:  median={np.nanmedian(ratios):.3f},    IQR=[{q25_ratio:.3f}, {q75_ratio:.3f}],   std={np.nanstd(ratios):.3f}")

    logger.info(f"non monotonic ratio {non_monotonic.sum() / n_subsets:.2f}")
    logger.info(
        f"I_N:    median={np.nanmedian(INs):.3f},   IQR=[{q25_IN:.3f}, {q75_IN:.3f}],   std={np.nanstd(INs):.3f}")
    logger.info(
        f"I2:     median={np.nanmedian(I2s):.3f},   IQR=[{q25_I2:.3f}, {q75_I2:.3f}],   std={np.nanstd(I2s):.3f}")
    logger.info(
        f"ratio:  median={np.nanmedian(ratios):.3f},   IQR=[{q25_ratio:.3f}, {q75_ratio:.3f}],   std={np.nanstd(ratios):.3f}")

    # Triplet KL analysis
    print('computing triplet KL analysis...')
    kl_results = triplet_residuals_full(
        S_data_pm1=s,
        h_full=h,
        J_sparse=J,
        n_model=200_000,
        seed=0,
        n_per_stratum=1000
    )

    for name, stats in kl_results.items():
        print(f"Triplet KL [{name}]: median={stats['median']:.4f}, "
              f"IQR=[{stats['q1']:.4f}, {stats['q3']:.4f}], n={stats['n']}")
        logger.info(f"Triplet KL [{name}]: median={stats['median']:.4f}, "
                    f"IQR=[{stats['q1']:.4f}, {stats['q3']:.4f}], n={stats['n']}")

    np.save(f"{log_dir}/results/triplet_KL_full.npy", kl_results)

    if False:
        # Exact triplet KL analysis
        triplet_results, summary = triplet_residuals_from_sparseJ(
            S_pm1=s,
            J_sparse=J,
            n_per_stratum=1000,
            neighborhood_size=10,
            tau_present=0.0,
            k_neighbors_each=4,
            rng_seed=0
        )

        print("Triplet KL (nats) summaries:")
        for k in ["triangle", "wedge", "one_edge", "no_edge", "all"]:
            print(k, summary[k])
            logger.info(f"Triplet KL exact [{k}]: {summary[k]}")

    return {
        'energy_stats': {'mean': E_mean, 'std': E_std, 'entropy': E_entropy},
        'coupling_stats': {'mean': J_mean, 'std': J_std, 'sign_ratio': J_sign_ratio, 'frac_strong': J_frac_strong},
        'info_ratio_results': results_dict,
        'triplet_kl_results': kl_results,
        # 'triplet_kl_exact': {'results': triplet_results, 'summary': summary},
        'ising_params': {'s': s, 'h': h, 'J': J, 'E': E}
    }




def sparse_ising_fit_fast(x, voltage_col=3, top_k=50, block_size=2000, dtype=np.float32, energy_stride=10):
    """
    Fast, blockwise sparse Ising approximation (correlation-based).
    Also computes Ising energy for every `energy_stride` frames.

    Returns
    -------
    s : np.ndarray (int8)
        binarized states {-1,+1}, shape [n_frames, n_neurons]
    h : np.ndarray (float32)
        bias terms (mean-field approx), shape [n_neurons]
    J : list of dict
        sparse couplings: J[i] is dict {j: value, ...} (top_k entries)
    E : np.ndarray (float32)
        energy per frame (subsampled), shape [n_frames//energy_stride]
    """
    n_frames, n_neurons, _ = x.shape
    voltage = x[:, :, voltage_col]

    # 1) Binarize at per-neuron mean -> {-1,+1}
    mean_v = voltage.mean(axis=0)
    s = np.where(voltage > mean_v, 1, -1).astype(np.int8)

    # 2) mean magnetization and biases
    m = s.mean(axis=0).astype(dtype)
    m = np.clip(m, -0.999, 0.999)
    h = np.arctanh(m).astype(dtype)

    # 3) Blockwise correlation scan, keeping top_k couplings safely
    J = [dict() for _ in range(n_neurons)]
    blocks = list(range(0, n_neurons, block_size))

    # Temporary storage for top-k values and indices
    topk_vals = np.full((n_neurons, top_k), -np.inf, dtype=dtype)
    topk_idx = np.full((n_neurons, top_k), -1, dtype=np.int32)

    for bj in trange(len(blocks), desc="blocks (j)"):
        j0, j1 = blocks[bj], min(n_neurons, blocks[bj] + block_size)
        sj = s[:, j0:j1].astype(dtype)

        for bi in range(0, n_neurons, block_size):
            i0, i1 = bi, min(n_neurons, bi + block_size)
            si = s[:, i0:i1].astype(dtype)

            # compute block correlation
            C_block = (si.T @ sj) / n_frames

            for local_i in range(i1 - i0):
                global_i = i0 + local_i
                vals = np.ascontiguousarray(C_block[local_i]).ravel()  # flatten safely
                cur_vals = topk_vals[global_i]
                cur_idx = topk_idx[global_i]

                # concatenate previous top-k and current block
                new_vals = np.concatenate([cur_vals, vals])
                new_idx = np.concatenate([cur_idx, np.arange(j0, j1, dtype=np.int32)])

                # filter out non-finite values before selecting top-k
                mask = np.isfinite(new_vals)
                filtered_vals = new_vals[mask]
                filtered_idx = new_idx[mask]

                # select top-k by absolute value
                if len(filtered_vals) > 0:
                    order = np.argsort(np.abs(filtered_vals))[-top_k:]
                    topk_vals[global_i] = filtered_vals[order]
                    topk_idx[global_i] = filtered_idx[order]

    # fill sparse coupling dictionaries
    for i in trange(n_neurons, desc="fill J"):
        vals, idxs = topk_vals[i], topk_idx[i]
        for v, j in zip(vals, idxs):
            if j != i and np.isfinite(v):
                J[i][int(j)] = float(v)

    # Convert J to edge list (upper triangle only)
    rows, cols, vals = [], [], []
    for i, Ji in enumerate(J):
        for j, Jij in Ji.items():
            if j > i:
                rows.append(i)
                cols.append(j)
                vals.append(Jij)

    rows = np.array(rows, dtype=np.int32)
    cols = np.array(cols, dtype=np.int32)
    vals = np.array(vals, dtype=np.float32)

    # Parameters
    chunk_size = 5000  # adjust based on RAM
    n_frames = s.shape[0]
    E = np.zeros(n_frames, dtype=np.float32)

    # Vectorized energy calculation in chunks with tqdm
    for start in trange(0, n_frames, chunk_size, desc="energy chunks"):
        end = min(start + chunk_size, n_frames)
        s_chunk = s[start:end].astype(np.float32)  # shape (chunk_size, n_neurons)

        # Field term
        field_E = -(s_chunk @ h)

        # Coupling term (vectorized over chunk)
        coupling_E = -0.5 * np.sum(vals[None, :] * s_chunk[:, rows] * s_chunk[:, cols], axis=1)

        E[start:end] = field_E + coupling_E

    E = E / 1000 # arbitrary

    return s, h, J, E




@dataclass
class InfoRatioResult:
    H_true: float
    H_indep: float
    H_pair: float
    I_N: float
    I2: float
    ratio: float
    count_non_monotonic: float
    observed_rates: np.ndarray
    predicted_rates_pairwise: np.ndarray  # P2 model (Ising)
    predicted_rates_independent: np.ndarray  # P1 model
# ========== Rebin utilities ==========
def aggregate_time(v: np.ndarray, bin_size: int, agg: str = "mean") -> np.ndarray:
    T, K = v.shape
    if bin_size <= 1:
        return v
    T_trim = (T // bin_size) * bin_size
    v = v[:T_trim]
    v = v.reshape(T_trim // bin_size, bin_size, K)
    if agg == "median":
        return np.median(v, axis=1)
    else:
        return v.mean(axis=1)
def rebin_voltage_subset( x: np.ndarray, idx: np.ndarray, bin_size: int = 1, voltage_col: int = 3, agg: str = "mean", threshold: str = "mean",
) -> np.ndarray:
    v = x[:, idx, voltage_col].astype(np.float32, copy=False)
    v_b = aggregate_time(v, bin_size=bin_size, agg=agg)
    thr = np.median(v_b, axis=0) if threshold == "median" else v_b.mean(axis=0)
    return np.where(v_b > thr[None, :], 1, -1).astype(np.int8)
def compute_info_ratio_estimator(S: np.ndarray,logbase: float = 2.0,alpha_joint: float = 1e-3,alpha_marg: float = 0.5,enforce_monotone: bool = False,ratio_eps: float = 1e-6,delta_t: float = 0.02,
) -> InfoRatioResult:
    """
    Compute information ratio and pattern rates for both pairwise and independent models.

    Parameters
    ----------
    S : np.ndarray, shape (T, N)
        Binary activity patterns in {-1, +1}

    Returns
    -------
    InfoRatioResult with pattern rates:
        - observed_rates: empirical probability of each 2^N pattern
        - predicted_rates_pairwise: Ising model (P2) probability of each pattern
        - predicted_rates_independent: Independent model (P1) probability of each pattern
    """
    H_true = entropy_true_pseudocount(S, logbase, alpha_joint)
    H_ind = entropy_indep_bernoulli_jeffreys(S, logbase, alpha_marg)

    h, J, H_pair = entropy_exact_ising(S)

    if (H_pair > H_ind) | (H_true > H_pair):
        count_non_monotonic = 1
    else:
        count_non_monotonic = 0

    if enforce_monotone:
        H_pair = min(H_pair, H_ind)
        H_true = min(H_true, H_pair)

    I_N = H_ind - H_true
    I2 = H_ind - H_pair
    ratio = I2 / I_N if I_N > ratio_eps else float('nan')

    # Pattern rate computation (always done)
    T, N = S.shape

    # Convert {-1,+1} to {0,1} for pattern indexing
    S_binary = (S + 1) // 2  # shape (T, N)

    # Compute pattern indices (each pattern -> integer 0 to 2^N-1)
    powers = 2 ** np.arange(N)  # [1, 2, 4, 8, ...]
    pattern_indices = S_binary @ powers  # shape (T,)

    # Count empirical pattern occurrences
    pattern_counts = np.bincount(pattern_indices, minlength=2 ** N)
    observed_rates = pattern_counts.astype(float) / T

    # Compute individual neuron firing probabilities for independent model
    p_i = np.mean(S_binary, axis=0)  # shape (N,)

    # Compute model predictions for all 2^N patterns
    predicted_rates_pairwise = np.zeros(2 ** N)
    predicted_rates_independent = np.zeros(2 ** N)

    # Generate all possible binary patterns
    for pattern_idx in range(2 ** N):
        # Convert pattern index back to binary array
        binary_pattern = np.array([(pattern_idx >> i) & 1 for i in range(N)])

        # PAIRWISE MODEL (P2): Ising model
        sigma = 2 * binary_pattern - 1  # Convert to {-1, +1}
        field_energy = -np.sum(h * sigma)
        coupling_energy = -0.5 * np.sum(J * np.outer(sigma, sigma))
        energy = field_energy + coupling_energy
        predicted_rates_pairwise[pattern_idx] = np.exp(-energy)

        # INDEPENDENT MODEL (P1): Product of individual probabilities
        prob_independent = 1.0
        for i in range(N):
            if binary_pattern[i] == 1:
                prob_independent *= p_i[i]
            else:
                prob_independent *= (1.0 - p_i[i])
        predicted_rates_independent[pattern_idx] = prob_independent

    # Normalize pairwise model to get probabilities
    Z_pairwise = np.sum(predicted_rates_pairwise)
    predicted_rates_pairwise = predicted_rates_pairwise / Z_pairwise

    # Independent model already normalized (probabilities sum to 1)

    return InfoRatioResult(
        H_true=H_true,
        H_indep=H_ind,
        H_pair=H_pair,
        I_N=I_N / delta_t,
        I2=I2 / delta_t,
        ratio=ratio,
        count_non_monotonic=count_non_monotonic,
        observed_rates=observed_rates / delta_t,  # NOW in s^-1
        predicted_rates_pairwise=predicted_rates_pairwise / delta_t,  # NOW in s^-1
        predicted_rates_independent=predicted_rates_independent / delta_t
    )
# ========== Entropy estimators ==========
def plugin_entropy(p: np.ndarray, logbase: float = math.e) -> float:
    p = p[p > 0]
    H = -np.sum(p * np.log(p))
    return H / math.log(logbase)
def miller_madow_correction(k: int, n: int, logbase: float = math.e) -> float:
    if n <= 0: return 0.0
    return ((k - 1) / (2.0 * n)) / math.log(logbase)
def empirical_entropy_true(S: np.ndarray, logbase: float = math.e) -> float:
    T, N = S.shape
    X = ((S + 1) // 2).astype(np.int8)
    keys = X @ (1 << np.arange(N, dtype=np.int64))
    counts = np.bincount(keys, minlength=1 << N).astype(np.float64)
    p = counts[counts > 0] / T
    H = plugin_entropy(p, logbase)
    H += miller_madow_correction(len(p), T, logbase)
    return H
def entropy_true_pseudocount(S: np.ndarray, logbase: float, alpha: float) -> float:
    T, N = S.shape
    X = ((S + 1) // 2).astype(np.int8)
    keys = X @ (1 << np.arange(N, dtype=np.int64))
    counts = np.bincount(keys, minlength=1 << N).astype(np.float64) + alpha
    p = counts / counts.sum()
    H = plugin_entropy(p, logbase)
    H += miller_madow_correction(np.count_nonzero(p), int(counts.sum()), logbase)
    return H
def entropy_indep_bernoulli_jeffreys(S: np.ndarray, logbase: float, alpha: float) -> float:
    T, N = S.shape
    X = ((S + 1) // 2).astype(np.int8)
    H = 0.0
    for i in range(N):
        k1 = X[:, i].sum()
        p1 = (k1 + alpha) / (T + 2 * alpha)
        p0 = 1.0 - p1
        h = 0.0
        if p1 > 0: h -= p1 * math.log(p1)
        if p0 > 0: h -= p0 * math.log(p0)
        H += h / math.log(logbase)
        H += miller_madow_correction(2, T, logbase)
    return H
def entropy_exact_ising(S, max_iter=200, lr=1.0, lam=0.0, tol=1e-6, logbase=2.0, verbose=False):
    """
    Maximum likelihood fit of pairwise Ising with exact enumeration.
    S in {-1,+1}^{T x N}. Returns h, J, H_pair.
    """
    S = S.astype(np.int8)
    # Ensure {-1,+1}
    uniq = np.unique(S)
    if set(uniq.tolist()) == {0,1}:
        S = 2*S - 1
    elif not set(uniq.tolist()) <= {-1,1}:
        raise ValueError("S must be binary in {-1,+1} or {0,1}.")

    T, N = S.shape
    m_data = S.mean(axis=0)                                 # ⟨s_i⟩_data
    C_data = (S[:, :, None] * S[:, None, :]).mean(axis=0)   # ⟨s_i s_j⟩_data
    np.fill_diagonal(C_data, 1.0)

    # Initialize (PL result or zeros). Start from PL helps, but zeros also work for N=10.
    h = np.zeros(N, dtype=np.float64)
    J = np.zeros((N, N), dtype=np.float64)

    X = enumerate_states_pm1(N)

    def neg_loglike_and_grad(h, J):
        # ℓ(θ) = ∑_t [ h·s + 1/2 s^T J s ] - T * log Z(θ)
        # We return -ℓ for minimization.
        p, _, _ = model_probs(h, J, X)
        m_mod = p @ X
        C_mod = np.tensordot(p, X[:, :, None] * X[:, None, :], axes=(0,0))
        np.fill_diagonal(C_mod, 1.0)
        # Gradients of -ℓ (so we can do standard gradient DESCENT on f = -ℓ)
        # For ascent on ℓ, use the negative of these.
        g_h = -(m_data - m_mod) + lam * h
        g_J = -(C_data - C_mod) + lam * J
        np.fill_diagonal(g_J, 0.0)
        # Negative log-likelihood:
        # -ℓ = -T*( h·⟨s⟩_data + 1/2 tr(J C_data) ) + T*logZ + reg
        # The constant “−T * H_emp” is irrelevant, so omitted.
        # We don’t compute f precisely here; we only need gradients for optimization & monitoring.
        return g_h, g_J

    # Backtracking line search on ℓ (maximize ℓ)
    prev_ll = -np.inf
    for it in range(1, max_iter+1):
        # Compute model stats and log-likelihood for monitoring
        p, _, logZ = model_probs(h, J, X)
        m_mod = p @ X
        C_mod = np.tensordot(p, X[:, :, None] * X[:, None, :], axes=(0,0))
        np.fill_diagonal(C_mod, 1.0)

        ll = T * (m_data @ h + 0.5 * np.sum(J * C_data) - logZ) - 0.5*lam*(h@h + (J*J).sum())
        # Gradient for ascent on ℓ:
        g_h, g_J = neg_loglike_and_grad(h, J)
        g_h *= -1.0
        g_J *= -1.0

        step = lr
        # Backtracking to ensure ll increases
        for _ in range(30):
            h_new = h + step * g_h
            J_new = J + step * g_J
            J_new = 0.5 * (J_new + J_new.T)
            np.fill_diagonal(J_new, 0.0)
            p_new, _, logZ_new = model_probs(h_new, J_new, X)
            ll_new = T * (m_data @ h_new + 0.5 * np.sum(J_new * C_data) - logZ_new) - 0.5*lam*(h_new@h_new + (J_new*J_new).sum())
            if ll_new >= ll:
                break
            step *= 0.5

        h, J, ll = h_new, J_new, ll_new

        # Convergence checks
        grad_norm = np.sqrt((g_h @ g_h) + (g_J*g_J).sum())
        if verbose and (it % 10 == 0 or it == 1):
            print(f"[it {it:3d}] ll/T = {ll/T:.6f}  step={step:.3e}  ||grad||={grad_norm:.3e}")

        if abs(ll - prev_ll) / (1 + abs(prev_ll)) < tol and grad_norm < 1e-5:
            break
        prev_ll = ll

    H_pair = entropy_from_model(h, J, logbase=logbase)
    return h, J, H_pair
def gibbs_sample_ising(h, J, T_samples=100000, burn=20000, seed=None):
    rng = np.random.default_rng(seed)
    N = h.shape[0]
    s = rng.choice([-1, 1], size=N)
    samples = []
    for t in range(burn + T_samples):
        i = rng.integers(0, N)
        local = h[i] + J[i] @ s - J[i, i] * s[i]
        p_plus = 1.0 / (1.0 + np.exp(-2.0 * local))
        s[i] = 1 if rng.random() < p_plus else -1
        if t >= burn:
            samples.append(s.copy())
    return np.array(samples, dtype=np.int8)
def enumerate_states_pm1(N):
    # All 2^N states in {-1,+1}^N
    X = np.array(list(product([-1, 1], repeat=N)), dtype=np.int8)
    return X  # shape [2^N, N]
def energy(h, J, X):
    # E(s) = - h·s - 1/2 s^T J s  (assumes J symmetric, diag(J)=0)
    # X: [M, N], h: [N], J: [N, N]
    lin = X @ h
    quad = np.einsum('bi,ij,bj->b', X, J, X)  # s^T J s
    return -(lin + 0.5 * quad)
def log_partition(E):
    # log Z = logsumexp(-E) if E is defined as +energy; here E is already energy
    # We defined p ∝ exp(-E), so we need logsumexp(-E)
    m = (-E).max()
    return m + np.log(np.exp((-E - m)).sum())
def model_probs(h, J, X):
    E = energy(h, J, X)
    logZ = log_partition(E)
    logp = -E - logZ
    p = np.exp(logp)
    return p, E, logZ
def model_moments_exact(h, J):
    N = len(h)
    X = enumerate_states_pm1(N)
    p, _, _ = model_probs(h, J, X)
    m = p @ X                          # ⟨s_i⟩
    C = (X[:, :, None] * X[:, None, :])  # [M, N, N]
    C = np.tensordot(p, C, axes=(0,0))   # ⟨s_i s_j⟩
    return m, C
def entropy_from_model(h, J, logbase=2.0):
    X = enumerate_states_pm1(len(h))
    p, E, logZ = model_probs(h, J, X)
    # H = -sum p log p
    H_nats = -(p * (np.log(p + 1e-300))).sum()
    return H_nats / log(logbase)




def gibbs_sample_sparse_ising(h, J_sparse, T_samples=200_000, seed=0, burn=0, thin=1, init=None):
    """
    h: (N,) float
    J_sparse: list of dicts; J_sparse[i][j] = J_ij (symmetric implied)
    returns S: (T_eff, N) in {-1,+1}
    """
    rng = np.random.default_rng(seed)
    N = len(h)
    s = rng.choice([-1, 1], size=N) if init is None else init.copy()
    samples = []
    total_steps = burn + T_samples*thin
    for t in range(total_steps):
        i = rng.integers(0, N)
        # local field from sparse neighbors
        local = h[i]
        for j, Jij in J_sparse[i].items():
            local += Jij * s[j]
        p_plus = 1.0 / (1.0 + np.exp(-2.0 * local))
        s[i] = 1 if rng.random() < p_plus else -1
        if t >= burn and ((t - burn) % thin == 0):
            samples.append(s.copy())
    return np.array(samples, dtype=np.int8)
def sample_triplets_stratified(J_sparse, n_per_stratum=5000, seed=0):
    rng = np.random.default_rng(seed)
    N = len(J_sparse)
    edges = set()
    for i, nbrs in enumerate(J_sparse):
        for j in nbrs.keys():
            if j > i:
                edges.add((i, j))
    adj = defaultdict(set)
    for i, j in edges:
        adj[i].add(j); adj[j].add(i)

    triplets = {"triangle": [], "wedge": [], "one_edge": [], "no_edge": []}
    nodes = np.arange(N)
    while len(triplets["triangle"]) < n_per_stratum:
        i = rng.integers(0, N)
        if len(adj[i]) < 2: continue
        j, k = rng.choice(list(adj[i]), size=2, replace=False)
        if j in adj[k]:
            tri = tuple(sorted((i, j, k)))
            triplets["triangle"].append(tri)
    while len(triplets["wedge"]) < n_per_stratum:
        i = rng.integers(0, N)
        if len(adj[i]) < 2: continue
        j, k = rng.choice(list(adj[i]), size=2, replace=False)
        if j not in adj[k]:
            tri = tuple(sorted((i, j, k)))
            triplets["wedge"].append(tri)
    while len(triplets["one_edge"]) < n_per_stratum or len(triplets["no_edge"]) < n_per_stratum:
        i, j, k = np.sort(rng.choice(N, size=3, replace=False))
        e = ((i, j) in edges) + ((i, k) in edges) + ((j, k) in edges)
        if e == 1 and len(triplets["one_edge"]) < n_per_stratum:
            triplets["one_edge"].append((i, j, k))
        elif e == 0 and len(triplets["no_edge"]) < n_per_stratum:
            triplets["no_edge"].append((i, j, k))

    return triplets  # dict of lists of 3-tuples
def triplet_KL_full_population(S_data_pm1, S_model_pm1, triplets, eps=1e-6, units="bits"):
    """
    S_data_pm1, S_model_pm1: (T, N) in {-1,+1}
    triplets: list of (i,j,k)
    returns per-triplet KLs (np.array)
    """
    logbase = np.log(2.0) if units == "bits" else 1.0
    T_data = S_data_pm1.shape[0]; T_mod = S_model_pm1.shape[0]
    kl = np.empty(len(triplets), dtype=np.float64)

    w = np.array([4, 2, 1], dtype=np.int32)
    for idx, (i, j, k) in enumerate(triplets):
        # empirical
        Xd = (S_data_pm1[:, [i, j, k]] + 1) // 2
        pd = np.bincount((Xd @ w), minlength=8).astype(float); pd += eps; pd /= pd.sum()
        # model
        Xm = (S_model_pm1[:, [i, j, k]] + 1) // 2
        pm = np.bincount((Xm @ w), minlength=8).astype(float); pm += eps; pm /= pm.sum()
        # KL(data || model)
        kl[idx] = np.sum(pd * (np.log(pd) - np.log(pm))) / logbase
    return kl
def triplet_residuals_full(S_data_pm1, h_full, J_sparse, n_model=200_000, seed=0, n_per_stratum=5000, units="bits"):
    # single model sample pool
    S_model_pm1 = gibbs_sample_sparse_ising(h_full, J_sparse, T_samples=n_model, seed=seed, burn=0)
    # stratified triplets
    strata = sample_triplets_stratified(J_sparse, n_per_stratum=n_per_stratum, seed=seed)

    out = {}
    for name, trips in strata.items():
        kls = triplet_KL_full_population(S_data_pm1, S_model_pm1, trips, units=units)
        q1, med, q3 = np.percentile(kls, [25, 50, 75])
        out[name] = dict(median=float(med), q1=float(q1), q3=float(q3), n=len(kls))
    # optionally also aggregate all strata together
    all_trips = sum(strata.values(), [])
    kls_all = triplet_KL_full_population(S_data_pm1, S_model_pm1, all_trips, units=units)
    q1, med, q3 = np.percentile(kls_all, [25, 50, 75])
    out["all"] = dict(median=float(med), q1=float(q1), q3=float(q3), n=len(kls_all))
    return out










def sample_stratified_triplets_from_sparse_J(J_sparse, n_per_stratum=1000, tau=0.0, rng_seed=0):

    rng = np.random.default_rng(rng_seed)
    N = len(J_sparse)
    # Convert sparse J to symmetric (mode="avg")
    J_sym = [defaultdict(float) for _ in range(N)]
    for i in range(N):
        for j, v in J_sparse[i].items():
            if i == j: continue
            if i < j and i in J_sparse[j]:
                v = 0.5 * (v + J_sparse[j][i])  # average if both exist
            J_sym[i][j] = v
            J_sym[j][i] = v

    # Helper functions (inlined)
    def edge_present(i, j):
        return (j in J_sym[i]) and (abs(J_sym[i][j]) > tau)

    def triplet_pattern(i, j, k):
        bij = int(edge_present(i, j))
        bik = int(edge_present(i, k))
        bjk = int(edge_present(j, k))
        return bij + bik + bjk  # return edge count

    # Initialize buckets
    buckets = {'triangle': [], 'wedge': [], 'one_edge': [], 'no_edge': []}

    # Bias sampling toward high-degree nodes for efficiency
    deg = [len(J_sym[i]) for i in range(N)]
    candidates = np.argsort(deg)[::-1]

    # Sample until all strata filled
    max_trials = 200000
    trials = 0
    while any(len(buckets[k]) < n_per_stratum for k in buckets) and trials < max_trials:
        trials += 1

        # Choose seed node (biased toward high degree)
        u = candidates[min(int(rng.exponential(scale=100)), len(candidates) - 1)]

        # Pick two others (preferably neighbors)
        neigh = list(J_sym[u].keys())
        if len(neigh) >= 2:
            v, w = rng.choice(neigh, size=2, replace=False)
        else:
            v, w = rng.integers(0, N, size=2)

        i, j, k = sorted({int(u), int(v), int(w)})
        if len({i, j, k}) < 3: continue

        # Classify and add to appropriate bucket
        m = triplet_pattern(i, j, k)
        if m == 3 and len(buckets['triangle']) < n_per_stratum:
            buckets['triangle'].append((i, j, k))
        elif m == 2 and len(buckets['wedge']) < n_per_stratum:
            buckets['wedge'].append((i, j, k))
        elif m == 1 and len(buckets['one_edge']) < n_per_stratum:
            buckets['one_edge'].append((i, j, k))
        elif m == 0 and len(buckets['no_edge']) < n_per_stratum:
            buckets['no_edge'].append((i, j, k))

    return buckets, J_sym

def _neighbors_by_absJ(J_sym, nodes, k_each=3, exclude=None):
    """
    Collect top-|J| neighbors around a set of 'nodes'.
    Returns a ranked list of candidate neighbor indices (no duplicates, excludes 'exclude').
    """
    if exclude is None:
        exclude = set(nodes)
    else:
        exclude = set(exclude) | set(nodes)
    scores = defaultdict(float)
    for u in nodes:
        for v, w in J_sym[u].items():
            if v in exclude:
                continue
            scores[v] = max(scores[v], abs(w))  # keep the strongest link seen
    # sort by descending strength
    ranked = sorted(scores.keys(), key=lambda v: scores[v], reverse=True)
    return ranked
def _empirical_triplet_marginal(S_sub_10, idx_triplet_10):
    """
    S_sub_10: [T,10] in {-1,+1}; idx_triplet_10: tuple of 3 positions in 0..9
    Returns p_data(8,) over states ordered lexicographically for (si,sj,sk) in {-1,+1}^3 with -1 < +1.
    """
    T = S_sub_10.shape[0]
    i, j, k = idx_triplet_10
    # map {-1,+1} -> {0,1}
    X = ((S_sub_10[:, [i, j, k]] + 1) // 2).astype(np.int8)  # [T,3]
    keys = X @ np.array([4, 2, 1], dtype=np.int8)            # 3-bit code: 4*si+2*sj+1*sk
    counts = np.bincount(keys, minlength=8).astype(np.float64)
    p = counts / counts.sum()
    return p
def _model_triplet_marginal_from_exact(h10, J10, triplet_idx):
    """
    Use exact 10-neuron model to get P(s) over 2^10 states, then marginalize to (i,j,k).
    """
    X = enumerate_states_pm1(10)  # [1024,10]  # :contentReference[oaicite:4]{index=4}
    p, _, _ = model_probs(h10, J10, X)        # :contentReference[oaicite:5]{index=5}
    i, j, k = triplet_idx
    # extract 3 bits for each 10-bit state, map {-1,+1}->{0,1} and then to 0..7 keys
    bits = ((X[:, [i, j, k]] + 1) // 2).astype(np.int8)
    keys = bits @ np.array([4, 2, 1], dtype=np.int8)
    p3 = np.zeros(8, dtype=np.float64)
    np.add.at(p3, keys, p)  # sum probabilities of 10-state that share same 3-state
    return p3
def _kl(p, q, eps=1e-12, logbase=np.e):
    """KL(p||q) with safety eps; returns nats if logbase=e, bits if logbase=2."""
    p = np.clip(p, eps, 1.0); p /= p.sum()
    q = np.clip(q, eps, 1.0); q /= q.sum()
    log = np.log if logbase == np.e else (lambda x: np.log(x)/np.log(logbase))
    return np.sum(p * (log(p) - log(q)))
def triplet_residuals_from_sparseJ(
    S_pm1,                 # ndarray [T,N] in {-1,+1}
    J_sparse,              # list[dict], from sparse_ising_fit_fast  # :contentReference[oaicite:6]{index=6}
    n_per_stratum=500,
    neighborhood_size=10,
    tau_present=0.0,
    k_neighbors_each=4,
    rng_seed=0,
):

    rng = np.random.default_rng(rng_seed)
    N = S_pm1.shape[1]
    assert S_pm1.dtype in (np.int8, np.int16, np.int32), "S must be in {-1,+1}"

    strata, J_sym = sample_stratified_triplets_from_sparse_J(J_sparse, n_per_stratum=1000, tau=0.0, rng_seed=0)

    # loop & compute KL per triplet via local exact 10-neuron fit
    results = {k: [] for k in strata}
    for cat, triplets in strata.items():
        for (i, j, k) in tqdm(triplets, desc=f"Processing {cat}", leave=False):
            # --- build 10-neuron neighborhood anchored on (i,j,k)
            neigh_ranked = _neighbors_by_absJ(J_sym, nodes=[i,j,k], k_each=k_neighbors_each, exclude=[i,j,k])
            take = [i, j, k] + neigh_ranked[:max(0, neighborhood_size-3)]
            take = take[:neighborhood_size]
            if len(take) < neighborhood_size:
                # pad with random non-duplicates if neighborhood too small
                pool = np.setdiff1d(np.arange(N), np.array(take))
                add = rng.choice(pool, size=neighborhood_size-len(take), replace=False).tolist()
                take += add
            take = np.array(take, dtype=int)
            # positions of (i,j,k) within the 10
            triplet_pos = (int(np.where(take==i)[0][0]),
                           int(np.where(take==j)[0][0]),
                           int(np.where(take==k)[0][0]))

            # --- slice data and fit exact 10-neuron Ising
            S10 = S_pm1[:, take]                    # [T,10]
            h10, J10, _Hpair = entropy_exact_ising(S10, logbase=np.e, verbose=False)  # :contentReference[oaicite:7]{index=7}

            # --- compute empirical & model triplet marginals
            p_data = _empirical_triplet_marginal(S10, triplet_pos)               # (8,)
            p_mod  = _model_triplet_marginal_from_exact(h10, J10, triplet_pos)   # (8,)

            # --- KL (nats)
            dkl = _kl(p_data, p_mod, eps=1e-12, logbase=np.e)
            results[cat].append(dkl)

    # 3) summarize
    def _summary(vec):
        v = np.asarray(vec, dtype=float)
        if v.size == 0:
            return dict(n=0, median=np.nan, iqr=[np.nan, np.nan], std=np.nan)
        q1, q3 = np.percentile(v, [25, 75])
        return dict(n=v.size, median=float(np.median(v)), iqr=[float(q1), float(q3)], std=float(v.std(ddof=0)))

    summary = {k: _summary(v) for k, v in results.items()}
    # overall (concatenate all strata)
    all_vals = np.concatenate([np.asarray(v, dtype=float) for v in results.values()]) if results else np.array([])
    summary["all"] = _summary(all_vals)

    return results, summary
