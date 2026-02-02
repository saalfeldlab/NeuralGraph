import numpy as np, torch
for i in range(4):
    s = f'signal_landscape_Claude_0{i}'
    x = np.load(f'graphs_data/signal/{s}/x_list_0.npy')
    a = x[:, :, 0]
    U, S, V = np.linalg.svd(a, full_matrices=False)
    c = np.cumsum(S**2) / np.sum(S**2)
    r99 = int(np.searchsorted(c, 0.99) + 1)
    W = torch.load(f'graphs_data/signal/{s}/connectivity.pt', map_location='cpu')
    if isinstance(W, dict):
        W = list(W.values())[0]
    W = W.numpy() if hasattr(W, 'numpy') else W
    e = np.linalg.eigvals(W)
    sr = float(np.max(np.abs(e)))
    print(f'slot{i}: r99={r99} sr={sr:.3f}')
