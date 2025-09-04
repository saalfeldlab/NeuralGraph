
# ising_analysis.py

import numpy as np
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from tqdm import trange
from itertools import product
from math import log

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
    flag_non_monotonic: float


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
        kl = np.sum(pd * (np.log(pd) - np.log(pm))) / np.log(2)  # in bits
        kl_list.append(kl)

    kl_array = np.array(kl_list)
    return float(np.median(kl_array)), float(np.percentile(kl_array, 25)), float(np.percentile(kl_array, 75))


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
    enforce_monotone: bool = False,
    ratio_eps: float = 1e-6,
) -> Tuple[InfoRatioResult, Tuple[float, float]]:

    H_true = entropy_true_pseudocount(S, logbase, alpha_joint)
    H_ind = entropy_indep_bernoulli_jeffreys(S, logbase, alpha_marg)

    h, J, H_pair = entropy_exact_ising(S)

    if (H_pair > H_ind) | (H_true > H_pair):
        # print(f"Warning: Non-monotonic entropies detected: H_true={H_true} < H_pair={H_pair} < H_ind={H_ind}")
        flag_non_monotonic = 1
    else:
        flag_non_monotonic = 0
    if enforce_monotone:
        H_pair = min(H_pair, H_ind)
        H_true = min(H_true, H_pair)
    I_N = H_ind - H_true
    I2 = H_ind - H_pair
    ratio = I2 / I_N if I_N > ratio_eps else float('nan')

    return InfoRatioResult(H_true, H_ind, H_pair, I_N, I2, ratio, 0.0, 0.0, flag_non_monotonic), (float('nan'), float('nan'))
