import torch, numpy as np
for slot in ['00','01','02','03']:
    path = 'graphs_data/signal/signal_landscape_Claude_%s/x_list_0.npy' % slot
    x = np.load(path)
    activity = x[:, :, -1]
    U, S, Vt = np.linalg.svd(activity, full_matrices=False)
    cumvar = np.cumsum(S**2) / np.sum(S**2)
    eff_rank_99 = int(np.searchsorted(cumvar, 0.99)) + 1
    print('Slot %s: eff_rank_99=%d shape=%s' % (slot, eff_rank_99, str(x.shape)))
    c = torch.load('graphs_data/signal/signal_landscape_Claude_%s/connectivity.pt' % slot, map_location='cpu', weights_only=True)
    rho = float(np.max(np.abs(np.linalg.eigvals(c.numpy()))))
    print('  rho=%.3f' % rho)
