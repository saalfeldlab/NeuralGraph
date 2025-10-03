#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# ===================== Config =====================
N = 100
p_edge = 0.03
T = 1000                # episodes (frames)
path_len = 20
lam, alpha, eta, q_max = 2.0, 0.2, 0.05, 3.0
seed = 7

# Non-stationarity (to keep map evolving)
block_len = 200
hub_jitter_radius = 4
eps_noise = 0.15
wash_windows = [(180,199),(380,399),(580,599),(780,799)]

# Viz / IO
save_dir = "src/stimergic/Fig"
dpi = 150
topK_draw = 200
node_size = 30          # smaller dots
base_w = 0.2
scale_w = 6.0

os.makedirs(save_dir, exist_ok=True)
rng = np.random.default_rng(seed)

# ===================== Build graph =====================
G = nx.DiGraph()
G.add_nodes_from(range(N))
for i in range(N):
    for j in range(N):
        if i != j and rng.random() < p_edge:
            G.add_edge(i, j)
for i in range(N):
    if G.out_degree(i) == 0:
        j = rng.integers(0, N-1)
        if j >= i: j += 1
        G.add_edge(i, j)

edges = list(G.edges())
E = len(edges)
src = np.array([u for u,v in edges])
dst = np.array([v for u,v in edges])
out_idx = {i: np.where(src==i)[0] for i in range(N)}

# Fixed layout
pos = nx.spring_layout(G, seed=seed, k=1.5/np.sqrt(N))

# ===================== State =====================
W = np.zeros(E, dtype=float)
q = np.zeros(E, dtype=float)
current_hub = int(rng.integers(0, N))

def in_wash(ep):
    return any(a <= ep <= b for a,b in wash_windows)

def draw_frame(ep, q_vec):
    order = np.argsort(-q_vec)
    keep = set(order[:min(topK_draw, len(order))])

    plt.figure(figsize=(8,6))
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color="black")

    # background edges
    nx.draw_networkx_edges(G, pos, edgelist=edges,
                           width=base_w, edge_color="black",
                           alpha=0.05, arrows=False)

    if keep:
        for ei in keep:
            u,v = edges[ei]
            width = base_w + scale_w*(q_vec[ei]/q_max if q_max>0 else 0.0)
            alpha_edge = 0.2 + 0.8*(q_vec[ei]/q_max if q_max>0 else 0.0)
            nx.draw_networkx_edges(G, pos, edgelist=[(u,v)],
                                   width=float(width),
                                   edge_color="black",
                                   alpha=float(alpha_edge),
                                   arrows=False)

    plt.axis("off")
    out_path = os.path.join(save_dir, f"ep_{ep:04d}.png")
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close()

# ===================== Run =====================
for t in range(1, T+1):
    # shift hub every block
    if (t-1) % block_len == 0 and t > 1:
        current_hub = int(rng.integers(0, N))

    # jitter start
    node = current_hub
    for _ in range(rng.integers(0, hub_jitter_radius+1)):
        idx = out_idx[node]
        if idx.size == 0: break
        e_idx = rng.choice(idx)
        node = dst[e_idx]
    start_node = node

    # path
    used_edges = []
    node = start_node
    for _ in range(path_len):
        idx = out_idx[node]
        if idx.size == 0: break
        logits = W[idx] + lam*q[idx] + eps_noise*rng.normal(size=idx.size)
        z = logits - logits.max()
        p = np.exp(z)/np.exp(z).sum()
        e_local = rng.choice(np.arange(idx.size), p=p)
        e_idx = idx[e_local]
        used_edges.append(e_idx)
        node = dst[e_idx]

    # update traces
    q *= (1.0 - eta)
    if not in_wash(t):
        for e in used_edges: q[e] += alpha
    q = np.clip(q, 0, q_max)

    draw_frame(t, q)
    if t % 100 == 0 or t == 1:
        print(f"Saved frame {t}/{T}")

print(f"Done. {T} frames in '{save_dir}/'. Load them in ImageJ to make a video.")
