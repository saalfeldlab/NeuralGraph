
# ising_analysis.py

import numpy as np
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from tqdm import trange

# ========== Rebin utilities ==========
def _aggregate_time(v: np.ndarray, bin_size: int, agg: str = "mean") -> np.ndarray:
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

def rebin_voltage_subset(
    x: np.ndarray,
    idx: np.ndarray,
    bin_size: int = 1,
    voltage_col: int = 3,
    agg: str = "mean",
    threshold: str = "mean",
) -> np.ndarray:
    v = x[:, idx, voltage_col].astype(np.float32, copy=False)
    v_b = _aggregate_time(v, bin_size=bin_size, agg=agg)
    thr = np.median(v_b, axis=0) if threshold == "median" else v_b.mean(axis=0)
    return np.where(v_b > thr[None, :], 1, -1).astype(np.int8)

def rebin_binary_subset(
    S: np.ndarray,
    bin_size: int = 1,
    rule: str = "majority",
) -> np.ndarray:
    T, K = S.shape
    if bin_size <= 1:
        return S
    T_trim = (T // bin_size) * bin_size
    Sb = S[:T_trim].reshape(T_trim // bin_size, bin_size, K)
    if rule == "any_pos":
        out = (Sb.max(axis=1) > 0).astype(np.int8)
    elif rule == "all_pos":
        out = (Sb.min(axis=1) > 0).astype(np.int8)
    else:
        sums = Sb.sum(axis=1)
        out = np.where(sums >= 0, 1, -1).astype(np.int8)
    out[out == 0] = -1
    return out

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

def independent_entropy(S: np.ndarray, logbase: float = math.e) -> float:
    T, N = S.shape
    H = 0.0
    for i in range(N):
        p1 = (S[:, i] == 1).mean()
        p = np.array([p1, 1 - p1])
        H += plugin_entropy(p, logbase)
        H += miller_madow_correction(np.count_nonzero(p), T, logbase)
    return H

def _entropy_true_pseudocount(S: np.ndarray, logbase: float, alpha: float) -> float:
    T, N = S.shape
    X = ((S + 1) // 2).astype(np.int8)
    keys = X @ (1 << np.arange(N, dtype=np.int64))
    counts = np.bincount(keys, minlength=1 << N).astype(np.float64) + alpha
    p = counts / counts.sum()
    H = plugin_entropy(p, logbase)
    H += miller_madow_correction(np.count_nonzero(p), int(counts.sum()), logbase)
    return H

def _entropy_indep_bernoulli_jeffreys(S: np.ndarray, logbase: float, alpha: float) -> float:
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

# ========== Ising model fitting and sampling ==========
def fit_ising_pseudolikelihood(S, max_iter=200, lr=0.2, lam=1e-4, tol=1e-6):
    T, N = S.shape
    S = S.astype(np.float64)
    h = np.zeros(N); J = np.zeros((N, N))
    for _ in range(max_iter):
        field = (S @ J.T) + h
        tanh_field = np.tanh(field)
        grad_h = (S - tanh_field).sum(0) - lam * h
        residual = S - tanh_field
        grad_J = (residual.T @ S) - lam * J
        np.fill_diagonal(grad_J, 0.0)
        h_new = h + lr * grad_h / T
        J_new = J + lr * grad_J / T
        J_new = 0.5 * (J_new + J_new.T)
        np.fill_diagonal(J_new, 0.0)
        if np.max(np.abs(h_new - h)) + np.max(np.abs(J_new - J)) < tol:
            break
        h, J = h_new, J_new
    return h, J

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

def model_entropy_via_sampling(h, J, n_samples=200000, burn=20000, thin=1, logbase=math.e, seed=None):
    S_model = gibbs_sample_ising(h, J, T_samples=n_samples, burn=burn, seed=seed)
    return empirical_entropy_true(S_model, logbase=logbase)

# ========== Info ratio data class ==========
@dataclass
class InfoRatioResult:
    H_true: float
    H_indep: float
    H_pair: float
    I_N: float
    I2: float
    ratio: float
    mean_match: float
    corr_match: float

# ========== API: Debiased ==========
def compute_info_ratio_debiased(
    S: np.ndarray,
    logbase: float = 2.0,
    alpha_joint: float = 1e-3,
    alpha_marg: float = 0.5,
    pl_max_iter: int = 400,
    pl_lr: float = 0.2,
    pl_lam: float = 1e-4,
    gibbs_samples: int = 200000,
    gibbs_burn: int = 20000,
    seed: Optional[int] = None,
    enforce_monotone: bool = True,
    ratio_eps: float = 1e-6,
) -> Tuple[InfoRatioResult, Tuple[float, float]]:
    H_true = _entropy_true_pseudocount(S, logbase, alpha_joint)
    H_ind = _entropy_indep_bernoulli_jeffreys(S, logbase, alpha_marg)
    h, J = fit_ising_pseudolikelihood(S, max_iter=pl_max_iter, lr=pl_lr, lam=pl_lam)
    H_pair = model_entropy_via_sampling(h, J, n_samples=gibbs_samples, burn=gibbs_burn, logbase=logbase, seed=seed)
    if enforce_monotone:
        H_pair = min(H_pair, H_ind)
        H_true = min(H_true, H_pair)
    I_N = H_ind - H_true
    I2 = H_ind - H_pair
    ratio = I2 / I_N if I_N > ratio_eps else float('nan')
    return InfoRatioResult(H_true, H_ind, H_pair, I_N, I2, ratio, 0.0, 0.0), (float('nan'), float('nan'))


# ========== Triplet KL Residuals ==========

def convert_J_sparse_to_dense(J_sparse, N):
    J_dense = np.zeros((N, N), dtype=np.float32)
    for i, Ji in enumerate(J_sparse):
        for j, val in Ji.items():
            J_dense[i, j] = val
            J_dense[j, i] = val  # enforce symmetry
    return J_dense

def triplet_residual_KL(S_data, h, J, n_model=200_000, seed=0, n_triplets=200):
    """
    Estimate how well the pairwise model predicts 3-neuron distributions.
    Computes KL divergence between empirical and model triplet marginals.
    Returns median and IQR of KLs across sampled triplets.
    """
    rng = np.random.default_rng(seed)
    N = S_data.shape[1]
    triplets = [tuple(sorted(rng.choice(N, 3, replace=False))) for _ in range(n_triplets)]

    S_mod = gibbs_sample_ising(h, J, T_samples=n_model, burn=20000, seed=seed + 1)

    kl_list = []
    for (i, j, k) in triplets:
        # empirical triplet
        Xd = ((S_data[:, [i, j, k]] + 1) // 2)
        idx_d = Xd @ np.array([4, 2, 1])
        pd = np.bincount(idx_d, minlength=8).astype(float) + 1e-6
        pd /= pd.sum()

        # model triplet
        Xm = ((S_mod[:, [i, j, k]] + 1) // 2)
        idx_m = Xm @ np.array([4, 2, 1])
        pm = np.bincount(idx_m, minlength=8).astype(float) + 1e-6
        pm /= pm.sum()

        # KL divergence
        kl = np.sum(pd * (np.log(pd) - np.log(pm)))
        kl_list.append(kl)

    kl_array = np.array(kl_list)
    return float(np.median(kl_array)), float(np.percentile(kl_array, 25)), float(np.percentile(kl_array, 75))


# ========== Visualization Tools ==========

import matplotlib.pyplot as plt

def plot_ratio_histogram(ratios, title="I2 / IN Ratio Distribution", bins=20):
    ratios = np.array(ratios)
    ratios = ratios[np.isfinite(ratios)]
    plt.figure(figsize=(6,4))
    plt.hist(ratios, bins=bins, color='steelblue', edgecolor='k', alpha=0.7)
    plt.xlabel("I^(2) / I_N")
    plt.ylabel("Count")
    plt.title(title)
    plt.axvline(1.0, color='red', linestyle='--', label='Max valid ratio')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_triplet_KL_hist(kls, title="Triplet KL Divergence (nats)", bins=30):
    kls = np.array(kls)
    kls = kls[np.isfinite(kls)]
    plt.figure(figsize=(6,4))
    plt.hist(kls, bins=bins, color='darkorange', edgecolor='k', alpha=0.7)
    plt.xlabel("Triplet KL (nats)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_IN_vs_tripletKL(INs, triplet_KLs, title="I_N vs. Triplet KL"):
    INs = np.array(INs)
    triplet_KLs = np.array(triplet_KLs)
    mask = np.isfinite(INs) & np.isfinite(triplet_KLs)
    INs = INs[mask]
    triplet_KLs = triplet_KLs[mask]
    plt.figure(figsize=(6,5))
    plt.scatter(INs, triplet_KLs, color='purple', alpha=0.6)
    plt.xlabel("I_N (bits)")
    plt.ylabel("Triplet KL (nats)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
