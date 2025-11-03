#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stigmergic traversal with square-shape score and temperature control (no gradients).

Visualization
- Only top-K edges and their incident nodes are emphasized.
- All other edges and nodes are drawn very faint.
- Edges are drawn first, then nodes so nodes are visible on top.
- Frames saved only every `save_every` episodes; output folder is cleared first.
"""

import os, glob
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# ===================== Config =====================
N = 100
p_edge = 0.03
T_episodes = 10000
path_len = 20

lam = 2.0
alpha = 0.20
eta = 0.05
q_max = 3.0

# Temperature control
T_temp = 1.0
T_min, T_max = 0.3, 2.0
score_update_K = 10
kappa = 0.1

# Square score weights
w_ori = 0.6
w_perim = 0.4
sigma_perim_ratio = 0.08

eps_noise = 0.1

use_shape_deposit = True
beta_deposit = 0.2

# Visualization / IO
save_dir = "Fig"
dpi = 150
save_every = 100
topK_draw = 200
node_size_fg = 30
node_size_bg = 18
base_w = 0.2
scale_w = 6.0
alpha_bg_edges = 0.02
alpha_bg_nodes = 0.05
seed = 7

# ===================== Prep output folder =====================
os.makedirs(save_dir, exist_ok=True)
for f in glob.glob(os.path.join(save_dir, "ep_*.png")):
    try: os.remove(f)
    except: pass

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
src = np.array([u for u, v in edges], dtype=int)
dst = np.array([v for u, v in edges], dtype=int)
out_idx = {i: np.where(src == i)[0] for i in range(N)}

pos = nx.spring_layout(G, seed=seed, k=1.5/np.sqrt(N))
xy = np.array([pos[i] for i in range(N)])

# ===================== State =====================
W = np.zeros(E, dtype=float)
q = np.zeros(E, dtype=float)
S_prev_for_update = None

# ===================== Square score =====================
def compute_square_score(q_vec):
    eps = 1e-8
    q_sum = float(q_vec.sum()) + eps
    vi = xy[src]; vj = xy[dst]
    dv = vj - vi
    L = np.linalg.norm(dv, axis=1) + eps
    ux2 = (dv[:,0]/L)**2
    uy2 = (dv[:,1]/L)**2
    s_ori_edge = np.maximum(ux2, uy2)
    S_ori = float((q_vec * s_ori_edge).sum()/q_sum)
    xmin,ymin = xy.min(axis=0); xmax,ymax = xy.max(axis=0)
    box_size = max(xmax-xmin, ymax-ymin) + eps
    sigma = sigma_perim_ratio * box_size
    mid = 0.5*(vi+vj)
    dx = np.minimum(np.abs(mid[:,0]-xmin), np.abs(xmax-mid[:,0]))
    dy = np.minimum(np.abs(mid[:,1]-ymin), np.abs(ymax-mid[:,1]))
    d = np.minimum(dx,dy)
    w_perim_edge = np.exp(-(d**2)/(sigma**2))
    S_perim = float((q_vec * w_perim_edge).sum()/q_sum)
    return w_ori*S_ori + w_perim*S_perim, S_ori, S_perim, w_perim_edge

# ===================== Viz =====================
def draw_frame(ep, q_vec):
    order = np.argsort(-q_vec)
    keep = set(order[:min(topK_draw,len(order))])
    nodes_fg = set()
    for ei in keep:
        u,v = edges[ei]
        nodes_fg.add(u); nodes_fg.add(v)
    [n for n in G.nodes if n not in nodes_fg]
    edges_bg = [e for i,e in enumerate(edges) if i not in keep]

    plt.figure(figsize=(8,6))
    # background edges
    if edges_bg:
        nx.draw_networkx_edges(G,pos,edgelist=edges_bg,width=base_w,
                               edge_color="black",alpha=alpha_bg_edges,arrows=False)
    # top edges
    for ei in keep:
        u,v = edges[ei]
        strength = q_vec[ei]/(q_max if q_max>0 else 1.0)
        width = base_w + scale_w*strength
        alpha_edge = 0.25 + 0.75*strength
        nx.draw_networkx_edges(G,pos,edgelist=[(u,v)],width=float(width),
                               edge_color="black",alpha=float(alpha_edge),arrows=False)
    # # background nodes
    # if nodes_bg:
    #     nx.draw_networkx_nodes(G,pos,nodelist=nodes_bg,node_size=node_size_bg,
    #                            node_color="black",alpha=alpha_bg_nodes)
    # # foreground nodes
    # if nodes_fg:
    #     nx.draw_networkx_nodes(G,pos,nodelist=list(nodes_fg),node_size=node_size_fg,
    #                            node_color="black",alpha=1.0)

    plt.axis("off")
    out_path = os.path.join(save_dir,f"ep_{ep:04d}.png")
    plt.savefig(out_path,dpi=dpi,bbox_inches="tight",pad_inches=0.05)
    plt.close()

# ===================== Run =====================
for t in range(1,T_episodes+1):
    used_edges=[]
    node=int(rng.integers(0,N))
    for _ in range(path_len):
        idx = out_idx[node]
        if idx.size==0: break
        logits=(W[idx]+lam*q[idx])/max(T_temp,1e-6)
        if eps_noise>0:
            logits=logits+eps_noise*rng.normal(size=idx.size)
        z=logits-logits.max()
        p=np.exp(z)/np.exp(z).sum()
        e_loc=rng.choice(np.arange(idx.size),p=p)
        e_idx=idx[e_loc]
        used_edges.append(e_idx)
        node=dst[e_idx]
    # score + thermostat
    S,S_ori,S_perim,w_perim_edge=compute_square_score(q)
    if (t%score_update_K)==0:
        if S_prev_for_update is None: S_prev_for_update=S
        dS=S-S_prev_for_update
        T_temp=np.clip(T_temp*np.exp(-kappa*dS),T_min,T_max)
        S_prev_for_update=S
    # stigmergy
    q*=(1.0-eta)
    if use_shape_deposit:
        for e in used_edges:
            bonus=(1.0+beta_deposit*w_perim_edge[e])
            q[e]+=alpha*bonus
    else:
        for e in used_edges:
            q[e]+=alpha
    q=np.clip(q,0.0,q_max)
    # save frame
    if (t%save_every)==0:
        draw_frame(t,q)
        print(f"Saved ep_{t:04d}.png | T={T_temp:.3f} | S={S:.3f} (ori={S_ori:.3f}, perim={S_perim:.3f})")

print(f"Done. Frames (every {save_every}) saved in '{save_dir}/'.")
