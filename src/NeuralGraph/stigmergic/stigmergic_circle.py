#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stigmergic shape formation on a graph via random walks + pheromones.

Default: forms a CIRCLE by rewarding (i) tangential edges around a target radius
and (ii) edges whose midpoints sit in a narrow annulus.

Optional: set --shape square to get the original square behavior.

Usage:
  python stigmergic_shape.py --shape circle --steps 20000 --plot_every 1000

Dependencies: numpy, networkx, matplotlib
"""

import argparse
import math
import random
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# --------------------------- configuration ---------------------------

@dataclass
class Config:
    # graph
    n_nodes: int = 500
    avg_degree: float = 4.0
    use_circular_layout_init: bool = False  # can help ring formation converge faster

    # walks
    n_walkers: int = 200
    walk_len: int = 10

    # pheromones
    evap_rate: float = 0.01  # 0..1 (fraction evaporated per outer step)
    deposit_base: float = 1.0
    deposit_shape_bonus: float = 1.0  # scales shape-based per-edge deposit bonus
    noise_eps: float = 1e-6           # for numerical stability

    # temperature (softmax over edge utilities)
    init_temp: float = 1.5
    min_temp: float = 0.1
    max_temp: float = 3.0
    temp_lr: float = 0.1              # step size of thermostat update
    score_target: float = 0.80        # thermostat drives score toward this
    score_momentum: float = 0.8

    # shape score weights (circle)
    w_tangent: float = 0.6
    w_ring: float = 0.4
    sigma_ring_ratio: float = 0.08     # annulus thickness ~ ratio of bbox size

    # shape score weights (square)
    w_axis: float = 0.6
    w_perim: float = 0.4
    sigma_perim_ratio: float = 0.06

    # run
    steps: int = 20000
    plot_every: int = 1000
    top_frac_to_plot: float = 0.04     # plot strongest ~4% of directed edges
    seed: int = 42


# --------------------------- utilities ---------------------------

def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)


def softmax(x: np.ndarray, temp: float) -> np.ndarray:
    x = x / max(temp, 1e-8)
    x = x - np.max(x)  # stabilize
    p = np.exp(x)
    s = p.sum()
    if s <= 0:
        return np.ones_like(p) / p.size
    return p / s


def build_random_graph(cfg: Config) -> nx.DiGraph:
    n = cfg.n_nodes
    # Start from undirected sparse graph, then make it directed by doubling edges
    p = min(1.0, cfg.avg_degree / max(1, n - 1))
    G_und = nx.gnp_random_graph(n, p, seed=cfg.seed, directed=False)
    G = nx.DiGraph()
    G.add_nodes_from(G_und.nodes)
    for u, v in G_und.edges:
        G.add_edge(u, v)
        G.add_edge(v, u)
    return G


def initial_positions(G: nx.DiGraph, cfg: Config) -> np.ndarray:
    if cfg.use_circular_layout_init:
        pos = nx.circular_layout(G)
    else:
        # spring layout tends to give a nice spread; seed for reproducibility
        pos = nx.spring_layout(G, seed=cfg.seed, k=1.5 / math.sqrt(G.number_of_nodes()))
    xy = np.array([pos[i] for i in range(G.number_of_nodes())], dtype=float)
    # standardize scale
    xy = xy - xy.mean(0, keepdims=True)
    m = np.abs(xy).max()
    if m > 0:
        xy = xy / m
    return xy


def graph_arrays(G: nx.DiGraph) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Return edge arrays and adjacency list for fast sampling."""
    edges = list(G.edges())
    src = np.array([u for u, v in edges], dtype=int)
    dst = np.array([v for u, v in edges], dtype=int)

    # adjacency: for each node, list of outgoing edge indices
    out_edges = {u: [] for u in G.nodes()}
    for idx, (u, v) in enumerate(edges):
        out_edges[u].append(idx)
    return src, dst, out_edges


# --------------------------- shape scores ---------------------------

def circle_score(xy: np.ndarray,
                 src: np.ndarray,
                 dst: np.ndarray,
                 q: np.ndarray,
                 cfg: Config):
    """
    CIRCLE score:
      S_tan  — edges oriented tangentially to circle at their midpoints
      S_ring — edge midpoints near target radius (annulus)
      S      — combined score
      w_edge — per-edge annulus weight for deposit bonus
    """
    eps = 1e-9
    # geometry
    vi = xy[src]              # (E,2)
    vj = xy[dst]              # (E,2)
    dv = vj - vi              # (E,2)
    L = np.linalg.norm(dv, axis=1) + eps
    mid = 0.5 * (vi + vj)     # (E,2)

    # circle center & radius
    c = xy.mean(axis=0, keepdims=True)              # (1,2)
    r_mid = np.linalg.norm(mid - c, axis=1)         # (E,)
    r_nodes = np.linalg.norm(xy - c, axis=1)        # (N,)
    r_target = np.median(r_nodes)

    # annulus width
    bbox = (xy.max(0) - xy.min(0)).sum() + eps
    sigma = cfg.sigma_ring_ratio * bbox

    # tangential alignment
    radial = (mid - c) / (r_mid[:, None] + eps)             # unit radial
    tangent = np.stack([-radial[:, 1], radial[:, 0]], 1)    # rotate 90°
    dir_u = dv / L[:, None]
    cos_t = np.abs((dir_u * tangent).sum(1))                # |cos|
    S_tan = (q * cos_t).sum() / (q.sum() + eps)

    # ring proximity
    w_edge = np.exp(-((r_mid - r_target) ** 2) / (sigma ** 2))
    S_ring = (q * w_edge).sum() / (q.sum() + eps)

    S = cfg.w_tangent * S_tan + cfg.w_ring * S_ring
    return float(S), float(S_tan), float(S_ring), w_edge


def square_score(xy: np.ndarray,
                 src: np.ndarray,
                 dst: np.ndarray,
                 q: np.ndarray,
                 cfg: Config):
    """
    SQUARE score (kept for convenience):
      favors axis-aligned edges and edges whose midpoints lie near the perimeter
      of the bounding box (i.e., 'square' ring).
    """
    eps = 1e-9
    vi = xy[src]
    vj = xy[dst]
    dv = vj - vi
    L = np.linalg.norm(dv, axis=1) + eps
    dir_u = dv / L[:, None]
    # axis alignment (prefer edges aligned with x or y)
    axis_align = np.maximum(np.abs(dir_u[:, 0]), np.abs(dir_u[:, 1]))

    # perimeter proximity (distance of midpoint to box edges)
    mid = 0.5 * (vi + vj)
    xmin, ymin = xy.min(0)
    xmax, ymax = xy.max(0)
    dx = np.minimum(mid[:, 0] - xmin, xmax - mid[:, 0])
    dy = np.minimum(mid[:, 1] - ymin, ymax - mid[:, 1])
    d_perim = np.minimum(dx, dy)
    # convert "near edge" to a Gaussian bump
    bbox = (xmax - xmin) + (ymax - ymin)
    sigma = cfg.sigma_perim_ratio * (bbox + eps)
    w_edge = np.exp(-(d_perim ** 2) / (sigma ** 2))

    S_axis = (q * axis_align).sum() / (q.sum() + eps)
    S_perim = (q * w_edge).sum() / (q.sum() + eps)
    S = cfg.w_axis * S_axis + cfg.w_perim * S_perim
    return float(S), float(S_axis), float(S_perim), w_edge


# --------------------------- stigmergic engine ---------------------------

def run_stigmergy(shape: str, cfg: Config):
    set_seed(cfg.seed)
    G = build_random_graph(cfg)
    xy = initial_positions(G, cfg)
    src, dst, out_edges = graph_arrays(G)
    E = src.shape[0]

    # pheromone and utilities
    q = np.zeros(E, dtype=float)  # pheromone per directed edge
    W = np.zeros(E, dtype=float)  # base utility (can be learned or static; keep zero)
    temp = cfg.init_temp

    # thermostat memory for smoother updates
    ema_score = cfg.score_target

    # precompute degrees for walk starts
    nodes = list(G.nodes())
    deg = np.array([G.out_degree(n) for n in nodes], dtype=int)
    valid_start_nodes = [n for n, d in zip(nodes, deg) if d > 0]
    if not valid_start_nodes:
        raise RuntimeError("Graph has no nodes with outgoing edges.")

    # choose score function
    score_fn = circle_score if shape == "circle" else square_score

    # plotting helper

    def plot_state(step: int):
        # uses: xy, src, dst, q, cfg, temp, ema_score, shape
        E = src.shape[0]  # total number of directed edges
        if E == 0:
            # Nothing to draw; show nodes only
            plt.figure(figsize=(8, 8))
            plt.scatter(xy[:, 0], xy[:, 1], s=5, c='k', alpha=0.6)
            plt.title(f"stigmergic '{shape}' — step {step} (no edges)")
            plt.axis('equal'); plt.axis('off'); plt.tight_layout(); plt.show()
            return

        # choose top-k edges robustly
        frac = max(0.0, float(cfg.top_frac_to_plot))
        k = int(round(frac * E))
        k = min(max(k, 1), E)  # clamp to [1, E]

        # sanitize q to avoid NaN/Inf issues
        qsafe = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)

        # fast top-k selection, then sort for prettier lines
        idx = np.argpartition(qsafe, -k)[-k:]
        idx = idx[np.argsort(qsafe[idx])]

        # line widths with safe zero-variance handling
        qk = qsafe[idx]
        qptp = np.ptp(qk)   # instead of qk.ptp()
        if qptp <= 0:
            lw = np.full_like(qk, 2.0, dtype=float)
        else:
            lw = 1.0 + 4.0 * (qk - qk.min()) / (qptp + 1e-9)


        # draw
        plt.figure(figsize=(8, 8))
        plt.scatter(xy[:, 0], xy[:, 1], s=5, c='k', alpha=0.6)
        for jj, e in enumerate(idx):
            i = src[e]; jn = dst[e]
            plt.plot([xy[i, 0], xy[jn, 0]], [xy[i, 1], xy[jn, 1]],
                    '-', alpha=0.6, linewidth=float(lw[jj]), color='#1f77b4')
        plt.title(f"stigmergic '{shape}' — step {step}, temp={temp:.2f}, score={ema_score:.3f}")
        plt.axis('equal'); plt.axis('off'); plt.tight_layout(); plt.show()


    # main loop
    for step in range(1, cfg.steps + 1):
        # evaporate
        q *= (1.0 - cfg.evap_rate)

        # circle or square scoring (for thermostat & bonus)
        S, S_a, S_b, w_edge = score_fn(xy, src, dst, q + cfg.noise_eps, cfg)
        # EMA of score for smoother temperature control
        ema_score = cfg.score_momentum * ema_score + (1 - cfg.score_momentum) * S

        # temperature update to steer score toward target
        temp_grad = (ema_score - cfg.score_target)  # if score > target => too rigid => heat up (increase temp)
        temp = np.clip(temp + cfg.temp_lr * temp_grad, cfg.min_temp, cfg.max_temp)

        # random walks & deposits
        for _ in range(cfg.n_walkers):
            node = random.choice(valid_start_nodes)
            for _ in range(cfg.walk_len):
                oe = out_edges[node]
                if not oe:
                    break
                # utilities: W + lambda*q (lambda=1 here)
                util = W[oe] + q[oe]
                # softmax over edges at current temp
                probs = softmax(util, temp)
                eidx = np.random.choice(oe, p=probs)

                # deposit (base + shape bonus at this edge)
                bonus = (1.0 + cfg.deposit_shape_bonus * w_edge[eidx])
                q[eidx] += cfg.deposit_base * bonus

                # advance walker
                node = dst[eidx]

        # plotting
        if (cfg.plot_every > 0) and (step % cfg.plot_every == 0 or step == 1):
            plot_state(step)

    # final plot
    plot_state(cfg.steps)


# --------------------------- CLI ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Stigmergic shape formation (circle/square).")
    parser.add_argument("--shape", type=str, default="circle", choices=["circle", "square"],
                        help="Target shape bias for the stigmergic process.")
    parser.add_argument("--nodes", type=int, default=500)
    parser.add_argument("--avg_deg", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--plot_every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--circular_init", action="store_true",
                        help="Initialize positions on a circle (helps ring formation).")
    args = parser.parse_args()

    cfg = Config(
        n_nodes=args.nodes,
        avg_degree=args.avg_deg,
        steps=args.steps,
        plot_every=args.plot_every,
        seed=args.seed,
        use_circular_layout_init=args.circular_init
    )

    # stronger circle bias
    cfg.w_tangent = 0.8        # was 0.6
    cfg.w_ring = 0.2           # was 0.4
    cfg.sigma_ring_ratio = 0.05  # was 0.08 (narrower annulus)

    # faster consolidation
    cfg.evap_rate = 0.02       # evaporate a bit faster
    cfg.deposit_shape_bonus = 2.0  


    run_stigmergy(args.shape, cfg)


if __name__ == "__main__":
    main()
