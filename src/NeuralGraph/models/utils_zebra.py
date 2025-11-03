def plot_field_comparison(x, model, k, n_frames, ones, output_path, step, plot_batch_size=700):
    """
    Original viz + YZ (continuous) in a new rightmost column.

    Top row (2×4):  [data] [NNR field – discrete] [NNR field – continuous] [blank]
    Bottom (2×4):   [data XZ]  [NNR XZ - discrete] [NNR XZ - continuous] [NNR YZ - continuous]

    Returns:
        field_discrete (torch.Tensor)
    """
    # ---------- imports ----------
    import numpy as np
    import pandas as pd
    import torch
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import datashader as ds
    import datashader.transfer_functions as tf

    # ---------- fixed LUT ----------
    vmin, vmax = 0.048, 0.251

    # ---------- fonts & spacing ----------
    TITLE_FZ  = 16
    LABEL_FZ  = 14
    TICK_FZ   = 8
    TITLE_PAD = 8
    LABEL_PAD = 8  # extra distance from axes to avoid overlaps

    # ---------- tensors & fields ----------
    x = torch.tensor(x, dtype=torch.float32, device=ones.device)

    # discrete (batched inference)
    in_features = torch.cat((x[:, 1:4]/model.NNR_f_xy_period, k * model.delta_t / model.NNR_f_T_period * ones), 1)

    field_discrete_list = []
    with torch.no_grad():
        for start in range(0, in_features.shape[0], plot_batch_size):
            end = min(start + plot_batch_size, in_features.shape[0])
            batch = in_features[start:end]
            field_discrete_list.append(model.NNR_f(batch)**2)
    field_discrete = torch.cat(field_discrete_list, dim=0)

    # continuous via fresh per-step, per-neuron jitter (matches original look)
    base  = x.unsqueeze(0).repeat(step, 1, 1)          # [step, N, F]
    noise = torch.randn_like(base) * 0.002             # fresh noise everywhere
    x_pert = base + noise
    x_flat = x_pert.reshape(-1, x.shape[1])
    in_feat_cont = torch.cat((x_flat[:, 1:4]/model.NNR_f_xy_period, k * model.delta_t / model.NNR_f_T_period * ones.repeat(step, 1)), 1)

    field_cont_list = []
    with torch.no_grad():
        for start in range(0, in_feat_cont.shape[0], plot_batch_size):
            end = min(start + plot_batch_size, in_feat_cont.shape[0])
            batch = in_feat_cont[start:end]
            field_cont_list.append(model.NNR_f(batch)**2)
    field_cont = torch.cat(field_cont_list, dim=0)
    field_cont = field_cont.reshape(step, -1)

    # to CPU
    x_cpu = x.cpu().numpy()
    field_discrete_cpu = field_discrete.cpu().numpy().squeeze()
    x_pert_cpu = x_pert.cpu().numpy()
    field_cont_cpu = field_cont.cpu().numpy()

    # flattened continuous for plots
    x_cont_flat = x_pert_cpu.reshape(-1, x_pert_cpu.shape[2])
    field_cont_flat = field_cont_cpu.flatten()

    # ---------- extents ----------
    X_MIN, X_MAX = 0.0, 0.8
    Y_MIN, Y_MAX = 0.0, 0.51
    Z_MIN, Z_MAX = 0.0, 0.285
    spanX, spanY, spanZ = X_MAX - X_MIN, Y_MAX - Y_MIN, Z_MAX - Z_MIN

    # proportional layout
    z_over_y     = spanZ / (spanY + 1e-12)      # bottom row shorter
    yz_col_ratio = spanY / (spanX + 1e-12)      # YZ column narrower than XY/XZ

    # ---------- canvases ----------
    canvas_width  = 800
    canvas_height = 600

    canvas_xy = ds.Canvas(plot_width=canvas_width, plot_height=canvas_height,
                          x_range=(X_MIN, X_MAX), y_range=(Y_MIN, Y_MAX))
    canvas_xz = ds.Canvas(plot_width=canvas_width,
                          plot_height=int(canvas_height * z_over_y),
                          x_range=(X_MIN, X_MAX), y_range=(Z_MIN, Z_MAX))
    canvas_yz = ds.Canvas(plot_width=int(canvas_width * yz_col_ratio),
                          plot_height=int(canvas_height * z_over_y),
                          x_range=(Y_MIN, Y_MAX), y_range=(Z_MIN, Z_MAX))

    # ---------- figure & gridspec ----------
    fig = plt.figure(figsize=(20, 7))
    gs = fig.add_gridspec(
        2, 4,
        height_ratios=[1.0, z_over_y],
        width_ratios=[1.0, 1.0, 1.0, yz_col_ratio],
        hspace=0.28,
        wspace=0.20  # more horizontal space between panels
    )

    # ---------------- TOP ROW ----------------
    # 1) data
    ax1 = fig.add_subplot(gs[0, 0])
    df1 = pd.DataFrame({'x': x_cpu[:, 1], 'y': Y_MAX - x_cpu[:, 2], 'value': x_cpu[:, 6]})
    img1 = tf.shade(canvas_xy.points(df1, 'x', 'y', ds.mean('value')),
                    cmap=cm.plasma, how='linear', span=[vmin, vmax]).to_pil()
    ax1.imshow(np.array(img1), extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], origin='lower', aspect='auto')
    ax1.set_title('data', fontsize=TITLE_FZ, pad=TITLE_PAD)
    ax1.set_xlabel('X', fontsize=LABEL_FZ, labelpad=LABEL_PAD)
    ax1.set_ylabel('Y', fontsize=LABEL_FZ, labelpad=LABEL_PAD)
    ax1.tick_params(labelsize=TICK_FZ)

    # condition text (unchanged)
    condition_names = {0:"gain",1:"dots",2:"flash",3:"taxis",4:"turning",5:"position",6:"open loop",7:"rotation",8:"dark",-1:"none"}
    try:
        condition_idx = int(x_cpu[0, 7])
        ax1.text(0.05, 0.95, f"{condition_names.get(condition_idx,'unknown')}\nframe: {k}",
                 transform=ax1.transAxes, fontsize=12, va='top', alpha=0.9)
    except Exception:
        pass

    # 2) NNR field – discrete
    ax2 = fig.add_subplot(gs[0, 1])
    df2 = pd.DataFrame({'x': x_cpu[:, 1], 'y': Y_MAX - x_cpu[:, 2], 'value': field_discrete_cpu})
    img2 = tf.shade(canvas_xy.points(df2, 'x', 'y', ds.mean('value')),
                    cmap=cm.plasma, how='linear', span=[vmin, vmax]).to_pil()
    ax2.imshow(np.array(img2), extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], origin='lower', aspect='auto')
    ax2.set_title('NNR field – discrete', fontsize=TITLE_FZ, pad=TITLE_PAD)
    ax2.set_xlabel('X', fontsize=LABEL_FZ, labelpad=LABEL_PAD)
    ax2.set_ylabel('Y', fontsize=LABEL_FZ, labelpad=LABEL_PAD)
    ax2.tick_params(labelsize=TICK_FZ)

    # 3) NNR field – continuous
    ax3 = fig.add_subplot(gs[0, 2])
    df3 = pd.DataFrame({'x': x_cont_flat[:, 1], 'y': Y_MAX - x_cont_flat[:, 2], 'value': field_cont_flat})
    img3 = tf.shade(canvas_xy.points(df3, 'x', 'y', ds.mean('value')),
                    cmap=cm.plasma, how='linear', span=[vmin, vmax]).to_pil()
    ax3.imshow(np.array(img3), extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], origin='lower', aspect='auto')
    ax3.set_title('NNR field – continuous', fontsize=TITLE_FZ, pad=TITLE_PAD)
    ax3.set_xlabel('X', fontsize=LABEL_FZ, labelpad=LABEL_PAD)
    ax3.set_ylabel('Y', fontsize=LABEL_FZ, labelpad=LABEL_PAD)
    ax3.tick_params(labelsize=TICK_FZ)

    # 4) top-right blank
    ax_blank = fig.add_subplot(gs[0, 3]); ax_blank.axis('off')

    # ---------------- BOTTOM ROW (no titles) ----------------
    # 5) data XZ
    ax4 = fig.add_subplot(gs[1, 0])
    df4 = pd.DataFrame({'x': x_cpu[:, 1], 'z': Z_MAX - x_cpu[:, 3], 'value': x_cpu[:, 6]})
    img4 = tf.shade(canvas_xz.points(df4, 'x', 'z', ds.mean('value')),
                    cmap=cm.plasma, how='linear', span=[vmin, vmax]).to_pil()
    ax4.imshow(np.array(img4), extent=[X_MIN, X_MAX, Z_MIN, Z_MAX], origin='lower', aspect='auto')
    ax4.set_xlabel('X', fontsize=LABEL_FZ, labelpad=LABEL_PAD)
    ax4.set_ylabel('Z', fontsize=LABEL_FZ, labelpad=LABEL_PAD)
    ax4.tick_params(labelsize=TICK_FZ)

    # 6) NNR field XZ - discrete
    ax5 = fig.add_subplot(gs[1, 1])
    df5 = pd.DataFrame({'x': x_cpu[:, 1], 'z': Z_MAX - x_cpu[:, 3], 'value': field_discrete_cpu})
    img5 = tf.shade(canvas_xz.points(df5, 'x', 'z', ds.mean('value')),
                    cmap=cm.plasma, how='linear', span=[vmin, vmax]).to_pil()
    ax5.imshow(np.array(img5), extent=[X_MIN, X_MAX, Z_MIN, Z_MAX], origin='lower', aspect='auto')
    ax5.set_xlabel('X', fontsize=LABEL_FZ, labelpad=LABEL_PAD)
    ax5.set_ylabel('Z', fontsize=LABEL_FZ, labelpad=LABEL_PAD)
    ax5.tick_params(labelsize=TICK_FZ)

    # 7) NNR field XZ - continuous
    ax6 = fig.add_subplot(gs[1, 2])
    df6 = pd.DataFrame({'x': x_cont_flat[:, 1], 'z': Z_MAX - x_cont_flat[:, 3], 'value': field_cont_flat})
    img6 = tf.shade(canvas_xz.points(df6, 'x', 'z', ds.mean('value')),
                    cmap=cm.plasma, how='linear', span=[vmin, vmax]).to_pil()
    ax6.imshow(np.array(img6), extent=[X_MIN, X_MAX, Z_MIN, Z_MAX], origin='lower', aspect='auto')
    ax6.set_xlabel('X', fontsize=LABEL_FZ, labelpad=LABEL_PAD)
    ax6.set_ylabel('Z', fontsize=LABEL_FZ, labelpad=LABEL_PAD)
    ax6.tick_params(labelsize=TICK_FZ)

    # 8) NNR YZ - continuous (flipped horizontally)
    ax7 = fig.add_subplot(gs[1, 3])
    df7 = pd.DataFrame({'y': x_cont_flat[:, 2], 'z': Z_MAX - x_cont_flat[:, 3], 'value': field_cont_flat})
    img7 = tf.shade(canvas_yz.points(df7, 'y', 'z', ds.mean('value')),
                    cmap=cm.plasma, how='linear', span=[vmin, vmax]).to_pil()
    ax7.imshow(np.array(img7), extent=[Y_MIN, Y_MAX, Z_MIN, Z_MAX], origin='lower', aspect='auto')
    ax7.invert_xaxis()
    ax7.set_xlabel('Y', fontsize=LABEL_FZ, labelpad=LABEL_PAD)
    ax7.set_ylabel('Z', fontsize=LABEL_FZ, labelpad=LABEL_PAD)
    ax7.tick_params(labelsize=TICK_FZ)

    # save
    plt.savefig(output_path, dpi=80)
    plt.close(fig)

    return field_discrete

def plot_field_comparison_continuous_slices(
    x, model, k, n_frames, ones, output_path,
    voxel_size=0.001,
    z_slices=(0.10, 0.15),
    y_slices=(0.20, 0.30),
    mask_points_per_neuron=32,
    mask_jitter_sigma=0.002,
    slice_half_thickness=0.004,
    mask_min_count=1,
    rng_seed=1234,
    vmin=0.0, vmax=0.75, dpi=300,
):
    """
    Continuous neural field slices with jitter-based data masks.

    Layout (2×2):
        Top row:    [XY @ Z=0.10]   [XY @ Z=0.20]
        Bottom row: [XZ @ Y=0.20]   [XZ @ Y=0.30]
    """
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from matplotlib import cm

    # ---------- font and style ----------
    TITLE_FZ  = 13
    LABEL_FZ  = 11
    TICK_FZ   = 7
    TITLE_PAD = 6
    LABEL_PAD = 6

    device = ones.device
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    pos = x[:, 1:4]  # (N,3)
    x_cpu = x.detach().cpu().numpy()

    # ---------- spatial extents ----------
    X_MIN, X_MAX = 0.0, 0.8
    Y_MIN, Y_MAX = 0.0, 0.51
    Z_MIN, Z_MAX = 0.0, 0.285
    spanX, spanY, spanZ = X_MAX - X_MIN, Y_MAX - Y_MIN, Z_MAX - Z_MIN
    z_over_y = spanZ / (spanY + 1e-12)

    # ---------- regular grids ----------
    def lin(lo, hi, step):
        n = max(2, int(np.ceil((hi - lo) / step)) + 1)
        return torch.linspace(lo, hi, n, device=device)
    Xg = lin(X_MIN, X_MAX, voxel_size)
    Yg = lin(Y_MIN, Y_MAX, voxel_size)
    Zg = lin(Z_MIN, Z_MAX, voxel_size)
    Nx, Ny, Nz = len(Xg), len(Yg), len(Zg)

    # ---------- model inference ----------
    def infer_field(points_3d):
        t_feat = (k * model.delta_t) / model.NNR_f_T_period
        feats = torch.cat(
            (points_3d / model.NNR_f_xy_period,
             torch.full((points_3d.shape[0], 1), float(t_feat), device=device, dtype=points_3d.dtype)),
            dim=1
        )
        with torch.inference_mode():
            vals = (model.NNR_f(feats)**2).squeeze(-1)
        return vals

    def render_xy(z0):
        XX, YY = torch.meshgrid(Xg, Yg, indexing="ij")
        ZZ = torch.full_like(XX, float(z0))
        pts = torch.stack([XX, YY, ZZ], dim=-1).reshape(-1, 3)
        vals = infer_field(pts)
        return vals.detach().cpu().numpy().reshape(XX.shape)

    def render_xz(y0):
        XX, ZZ = torch.meshgrid(Xg, Zg, indexing="ij")
        YY = torch.full_like(XX, float(y0))
        pts = torch.stack([XX, YY, ZZ], dim=-1).reshape(-1, 3)
        vals = infer_field(pts)
        return vals.detach().cpu().numpy().reshape(XX.shape)

    # ---------- jitter-based masks ----------
    g = torch.Generator(device=device)
    g.manual_seed(rng_seed)

    def raster_mask_xy(z0):
        N = pos.shape[0]
        jitter = torch.randn((N, mask_points_per_neuron, 3), generator=g, device=device) * mask_jitter_sigma
        pts = (pos[:, None, :] + jitter).reshape(-1, 3)
        keep = (pts[:, 2] >= (z0 - slice_half_thickness)) & (pts[:, 2] <= (z0 + slice_half_thickness))
        pts = pts[keep]
        if pts.numel() == 0:
            return np.zeros((Nx, Ny), dtype=np.uint8)
        ix = torch.clamp(((pts[:, 0]-X_MIN)/spanX*(Nx-1)).round().long(), 0, Nx-1)
        iy = torch.clamp(((pts[:, 1]-Y_MIN)/spanY*(Ny-1)).round().long(), 0, Ny-1)
        mask = torch.zeros((Nx, Ny), device=device, dtype=torch.int32)
        mask.index_put_((ix, iy), torch.ones_like(ix, dtype=torch.int32), accumulate=True)
        return (mask >= mask_min_count).cpu().numpy().astype(np.uint8)

    def raster_mask_xz(y0):
        N = pos.shape[0]
        jitter = torch.randn((N, mask_points_per_neuron, 3), generator=g, device=device) * mask_jitter_sigma
        pts = (pos[:, None, :] + jitter).reshape(-1, 3)
        keep = (pts[:, 1] >= (y0 - slice_half_thickness)) & (pts[:, 1] <= (y0 + slice_half_thickness))
        pts = pts[keep]
        if pts.numel() == 0:
            return np.zeros((Nx, Nz), dtype=np.uint8)
        ix = torch.clamp(((pts[:, 0]-X_MIN)/spanX*(Nx-1)).round().long(), 0, Nx-1)
        iz = torch.clamp(((pts[:, 2]-Z_MIN)/spanZ*(Nz-1)).round().long(), 0, Nz-1)
        mask = torch.zeros((Nx, Nz), device=device, dtype=torch.int32)
        mask.index_put_((ix, iz), torch.ones_like(ix, dtype=torch.int32), accumulate=True)
        return (mask >= mask_min_count).cpu().numpy().astype(np.uint8)

    # ---------- evaluate fields ----------
    xy_zL = render_xy(z_slices[0])   # left (Z=0.10)
    xy_zR = render_xy(z_slices[1])   # right (Z=0.20)
    xz_yL = render_xz(y_slices[0])   # bottom-left (Y=0.20)
    xz_yR = render_xz(y_slices[1])   # bottom-right (Y=0.30)

    # ---------- build and apply masks ----------
    m_xy_L = raster_mask_xy(z_slices[0])
    m_xy_R = raster_mask_xy(z_slices[1])
    m_xz_L = raster_mask_xz(y_slices[0])
    m_xz_R = raster_mask_xz(y_slices[1])

    xy_zL[m_xy_L == 0] = np.nan
    xy_zR[m_xy_R == 0] = np.nan
    xz_yL[m_xz_L == 0] = np.nan
    xz_yR[m_xz_R == 0] = np.nan

    # ---------- figure ----------
    fig = plt.figure(figsize=(15, 7), facecolor='black')
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1.0, z_over_y],
        width_ratios=[1.0, 1.0],
        hspace=0.28,
        wspace=0.20
    )

    cmap = cm.plasma.copy()
    cmap.set_bad(color='black')  # mask → black background

    # --- Top: XY Z=0.10 left ---
    ax1 = fig.add_subplot(gs[0, 0], facecolor='black')
    ax1.imshow(xy_zL.T, extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], origin='lower',
               aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.set_title(f'XY @ Z={z_slices[0]:.2f}', fontsize=TITLE_FZ, pad=TITLE_PAD, color='white')
    ax1.set_xlabel('X', fontsize=LABEL_FZ, labelpad=LABEL_PAD, color='white')
    ax1.set_ylabel('Y', fontsize=LABEL_FZ, labelpad=LABEL_PAD, color='white')
    ax1.tick_params(labelsize=TICK_FZ, colors='white')

    # --- Top: XY Z=0.20 right ---
    ax2 = fig.add_subplot(gs[0, 1], facecolor='black')
    ax2.imshow(xy_zR.T, extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], origin='lower',
               aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax2.set_title(f'XY @ Z={z_slices[1]:.2f}', fontsize=TITLE_FZ, pad=TITLE_PAD, color='white')
    ax2.set_xlabel('X', fontsize=LABEL_FZ, labelpad=LABEL_PAD, color='white')
    ax2.set_ylabel('Y', fontsize=LABEL_FZ, labelpad=LABEL_PAD, color='white')
    ax2.tick_params(labelsize=TICK_FZ, colors='white')

    # --- Bottom: XZ Y=0.20 left ---
    ax3 = fig.add_subplot(gs[1, 0], facecolor='black')
    ax3.imshow(xz_yL.T, extent=[X_MIN, X_MAX, Z_MIN, Z_MAX], origin='lower',
               aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax3.set_title(f'XZ @ Y={y_slices[0]:.2f}', fontsize=TITLE_FZ, pad=TITLE_PAD, color='white')
    ax3.set_xlabel('X', fontsize=LABEL_FZ, labelpad=LABEL_PAD, color='white')
    ax3.set_ylabel('Z', fontsize=LABEL_FZ, labelpad=LABEL_PAD, color='white')
    ax3.tick_params(labelsize=TICK_FZ, colors='white')

    # --- Bottom: XZ Y=0.30 right ---
    ax4 = fig.add_subplot(gs[1, 1], facecolor='black')
    ax4.imshow(xz_yR.T, extent=[X_MIN, X_MAX, Z_MIN, Z_MAX], origin='lower',
               aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax4.set_title(f'XZ @ Y={y_slices[1]:.2f}', fontsize=TITLE_FZ, pad=TITLE_PAD, color='white')
    ax4.set_xlabel('X', fontsize=LABEL_FZ, labelpad=LABEL_PAD, color='white')
    ax4.set_ylabel('Z', fontsize=LABEL_FZ, labelpad=LABEL_PAD, color='white')
    ax4.tick_params(labelsize=TICK_FZ, colors='white')

    # optional condition label
    condition_names = {
        0:"gain",1:"dots",2:"flash",3:"taxis",4:"turning",
        5:"position",6:"open loop",7:"rotation",8:"dark",-1:"none"
    }
    try:
        cond_idx = int(x_cpu[0, 7])
        ax1.text(0.03, 0.96, f"{condition_names.get(cond_idx,'unknown')}\nframe: {k}",
                 transform=ax1.transAxes, fontsize=9, va='top', color='white', alpha=0.8)
    except Exception:
        pass

    plt.savefig(output_path, dpi=dpi, facecolor='black')
    plt.close(fig)

def plot_field_comparison_discrete_slices(
    x, model, k, n_frames, ones, output_path,
    # which slices
    z_slices=(0.10, 0.20),    
    y_slices=(0.20, 0.30),    
    slice_half_thickness=0.004,
    # viz
    dot_size=3.0,
    vmin=0.0, vmax=0.75, dpi=300,
    # flips for consistency with continuous viz
    flip_top_y=True,      # invert Y-axis on XY panels (for dorsal-ventral convention)
    flip_bottom_x=False,  
    flip_bottom_z=False,  
):
    """
    Discrete, slice-based comparison at neuron locations (no grids, no jitter):
      Top row:    [model XY @ Z=z_slices[0]]   [model XY @ Z=z_slices[1]]
      Bottom row: [model XZ @ Y=y_slices[0]]   [model XZ @ Y=y_slices[1]]

    Notes:
      - Uses the same feature scaling as your other functions:
        [x,y,z]/NNR_f_xy_period and t = k*Δt/NNR_f_T_period.
      - Black background; smaller fonts to match your latest style.
      - Dot color = model value (NNR_f(x,t)^2). No interpolation.
    """
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from matplotlib import cm

    # ---------- style ----------
    TITLE_FZ  = 13
    LABEL_FZ  = 11
    TICK_FZ   = 7
    TITLE_PAD = 6
    LABEL_PAD = 6
    cmap = cm.plasma

    # ---------- device & tensors ----------
    device = ones.device
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    pos = x[:, 1:4]                 # (N,3) X,Y,Z
    x_cpu = x.detach().cpu().numpy()

    # ---------- extents (match your constants) ----------
    X_MIN, X_MAX = 0.0, 0.8
    Y_MIN, Y_MAX = 0.0, 0.51
    Z_MIN, Z_MAX = 0.0, 0.285
    _spanX, spanY, spanZ = X_MAX - X_MIN, Y_MAX - Y_MIN, Z_MAX - Z_MIN
    z_over_y = spanZ / (spanY + 1e-12)

    # ---------- model inference once (discrete at all points) ----------
    t_feat = (k * model.delta_t) / model.NNR_f_T_period
    feats = torch.cat(
        (pos / model.NNR_f_xy_period,
         torch.full((pos.shape[0], 1), float(t_feat), device=device, dtype=pos.dtype)),
        dim=1
    )
    with torch.inference_mode():
        field_discrete = (model.NNR_f(feats)**2).squeeze(-1)   # (N,)

    # move to numpy
    x_np = pos[:, 0].detach().cpu().numpy()
    y_np = pos[:, 1].detach().cpu().numpy()
    z_np = pos[:, 2].detach().cpu().numpy()
    val_np = field_discrete.detach().cpu().numpy()

    # Apply same flips you’ve been using for visuals:
    # XY views: flip Y vertically; XZ views: flip Z vertically.
    y_plot = (Y_MAX - y_np)
    z_plot = (Z_MAX - z_np)

    # ---------- slice selectors ----------
    def sel_xy_at_z(z0):
        return np.where(np.abs(z_np - z0) <= slice_half_thickness)[0]

    def sel_xz_at_y(y0):
        return np.where(np.abs(y_np - y0) <= slice_half_thickness)[0]

    inds_xy_L = sel_xy_at_z(z_slices[0])   # XY left (Z=0.10)
    inds_xy_R = sel_xy_at_z(z_slices[1])   # XY right (Z=0.20)
    inds_xz_L = sel_xz_at_y(y_slices[0])   # XZ left (Y=0.20)
    inds_xz_R = sel_xz_at_y(y_slices[1])   # XZ right (Y=0.30)

    # ---------- figure ----------
    fig = plt.figure(figsize=(15, 7), facecolor='black')
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1.0, z_over_y],
        width_ratios=[1.0, 1.0],
        hspace=0.28,
        wspace=0.20
    )


    ax1 = fig.add_subplot(gs[0, 0], facecolor='black')
    if inds_xy_L.size > 0:
        # Use y_np or y_plot depending on flip
        y_coords = y_plot if flip_top_y else y_np
        ax1.scatter(x_np[inds_xy_L], y_coords[inds_xy_L], c=val_np[inds_xy_L],
                    s=dot_size, cmap=cmap, vmin=vmin, vmax=vmax,
                    edgecolors='none', linewidths=0)
    ax1.set_xlim(X_MIN, X_MAX)
    ax1.set_ylim(Y_MIN, Y_MAX)
    if flip_top_y:
        ax1.invert_yaxis()  # Flip Y axis for dorsal view
    # ... [rest of ax1 setup] ...

    # --- TOP RIGHT: XY @ Z=z_slices[1] ---
    ax2 = fig.add_subplot(gs[0, 1], facecolor='black')
    if inds_xy_R.size > 0:
        y_coords = y_plot if flip_top_y else y_np
        ax2.scatter(x_np[inds_xy_R], y_coords[inds_xy_R], c=val_np[inds_xy_R],
                    s=dot_size, cmap=cmap, vmin=vmin, vmax=vmax,
                    edgecolors='none', linewidths=0)
    ax2.set_xlim(X_MIN, X_MAX)
    ax2.set_ylim(Y_MIN, Y_MAX)
    if flip_top_y:
        ax2.invert_yaxis()

    # --- BOTTOM LEFT: XZ @ Y=y_slices[0] ---
    ax3 = fig.add_subplot(gs[1, 0], facecolor='black')
    if inds_xz_L.size > 0:
        ax3.scatter(x_np[inds_xz_L], z_plot[inds_xz_L], c=val_np[inds_xz_L],
                    s=dot_size, cmap=cmap, vmin=vmin, vmax=vmax,
                    edgecolors='none', linewidths=0)
    ax3.set_xlim(X_MIN, X_MAX); ax3.set_ylim(Z_MIN, Z_MAX)
    ax3.set_title(f'XZ @ Y={y_slices[0]:.2f}', fontsize=TITLE_FZ, pad=TITLE_PAD, color='white')
    ax3.set_xlabel('X', fontsize=LABEL_FZ, labelpad=LABEL_PAD, color='white')
    ax3.set_ylabel('Z', fontsize=LABEL_FZ, labelpad=LABEL_PAD, color='white')
    ax3.tick_params(labelsize=TICK_FZ, colors='white')
    if flip_bottom_x: ax3.invert_xaxis()
    if flip_bottom_z: ax3.invert_yaxis()  # flips Z (vertical axis here)

    # --- BOTTOM RIGHT: XZ @ Y=y_slices[1] ---
    ax4 = fig.add_subplot(gs[1, 1], facecolor='black')
    if inds_xz_R.size > 0:
        ax4.scatter(x_np[inds_xz_R], z_plot[inds_xz_R], c=val_np[inds_xz_R],
                    s=dot_size, cmap=cmap, vmin=vmin, vmax=vmax,
                    edgecolors='none', linewidths=0)
    ax4.set_xlim(X_MIN, X_MAX); ax4.set_ylim(Z_MIN, Z_MAX)
    ax4.set_title(f'XZ @ Y={y_slices[1]:.2f}', fontsize=TITLE_FZ, pad=TITLE_PAD, color='white')
    ax4.set_xlabel('X', fontsize=LABEL_FZ, labelpad=LABEL_PAD, color='white')
    ax4.set_ylabel('Z', fontsize=LABEL_FZ, labelpad=LABEL_PAD, color='white')
    ax4.tick_params(labelsize=TICK_FZ, colors='white')
    if flip_bottom_x: ax4.invert_xaxis()
    if flip_bottom_z: ax4.invert_yaxis()

    # optional condition label
    condition_names = {
        0:"gain",1:"dots",2:"flash",3:"taxis",4:"turning",
        5:"position",6:"open loop",7:"rotation",8:"dark",-1:"none"
    }
    try:
        cond_idx = int(x_cpu[0, 7])
        ax1.text(0.03, 0.96, f"{condition_names.get(cond_idx,'unknown')}\nframe: {k}",
                 transform=ax1.transAxes, fontsize=9, va='top', color='white', alpha=0.8)
    except Exception:
        pass

    plt.savefig(output_path, dpi=dpi, facecolor='black')
    plt.close(fig)

    # return for metrics/logging consistency
    return field_discrete.unsqueeze(-1)  # (N,1)

def plot_field_discrete_xy_slices_grid(
    x, model, k, n_frames, ones, output_path,
    z_start=0.10,           # first slice depth
    z_step=0.004,           # step between slices
    n_cols=5, n_rows=4,     # 5×4 = 20 panels
    slice_half_thickness=0.002,  # include points within ±this around each Z
    dot_size=2.0,           # scatter size (matplotlib "s")
    vmin=0.0, vmax=0.75,    # color limits
    dpi=300,
):
    """
    5×4 grid of discrete XY slices (model only), no interpolation/jitter.
    Panels are ordered left→right, top→bottom with increasing Z:
        Z = z_start + idx*z_step, idx = 0..(n_rows*n_cols-1)

    Returns:
        field_discrete (torch.Tensor)  # (N,1) on device
    """
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from matplotlib import cm

    # --- styling (small fonts, black bg) ---
    LABEL_FZ  = 9
    TICK_FZ   = 6
    LABEL_PAD = 4
    cmap = cm.plasma

    device = ones.device
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    pos = x[:, 1:4]  # (N,3): X,Y,Z

    # --- extents (match your constants) ---
    X_MIN, X_MAX = 0.0, 0.8
    Y_MIN, Y_MAX = 0.0, 0.51
    _Z_MIN, _Z_MAX = 0.0, 0.285

    # precompute model at discrete positions (once)
    t_feat = (k * model.delta_t) / model.NNR_f_T_period
    feats = torch.cat(
        (pos / model.NNR_f_xy_period,
         torch.full((pos.shape[0], 1), float(t_feat), device=device, dtype=pos.dtype)),
        dim=1
    )
    with torch.inference_mode():
        field_discrete = (model.NNR_f(feats)**2).unsqueeze(-1)  # (N,1)

    # numpy arrays for plotting / selection
    x_np = pos[:, 0].detach().cpu().numpy()
    y_np = pos[:, 1].detach().cpu().numpy()
    z_np = pos[:, 2].detach().cpu().numpy()
    val_np = field_discrete.squeeze(-1).detach().cpu().numpy()

    # flip Y to keep your previous visual convention
    y_plot = (Y_MAX - y_np)

    # generate Z slice values (20 total by default)
    n_panels = n_cols * n_rows
    z_vals = z_start + np.arange(n_panels) * z_step

    # figure sized to keep panels readable
    # each panel roughly 3.0×2.2 inches → 5×4 grid ≈ 15×8.8 in
    fig_w = 15.0
    fig_h = 8.8
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='black')
    gs = fig.add_gridspec(
        n_rows, n_cols,
        hspace=0.20,
        wspace=0.15
    )

    # draw panels
    for r in range(n_rows):
        for c in range(n_cols):
            idx = r * n_cols + c
            z0 = z_vals[idx]

            ax = fig.add_subplot(gs[r, c], facecolor='black')
            # indices within the slice band
            sel = np.where(np.abs(z_np - z0) <= slice_half_thickness)[0]

            if sel.size > 0:
                ax.scatter(
                    x_np[sel], y_plot[sel], c=val_np[sel],
                    s=dot_size, cmap=cmap, vmin=vmin, vmax=vmax,
                    edgecolors='none', linewidths=0
                )

            ax.set_xlim(X_MIN, X_MAX)
            ax.set_ylim(Y_MIN, Y_MAX)

            # title with Z value
            # ax.set_title(f"XY @ Z={z0:.3f}", fontsize=TITLE_FZ, pad=TITLE_PAD, color='white')

            ax.text(0.05, 0.90, f"Z={z0:.3f}",
                    transform=ax.transAxes, fontsize=8, color='white')
            # show labels only on outer edges to reduce clutter
            if r == n_rows - 1 and c == 0:
                ax.set_xlabel('X', fontsize=LABEL_FZ, labelpad=LABEL_PAD, color='white')
                ax.set_ylabel('Y', fontsize=LABEL_FZ, labelpad=LABEL_PAD, color='white')
            else:
                ax.set_axis_off()
                ax.set_xticklabels([])
                ax.set_yticklabels([])

            ax.tick_params(labelsize=TICK_FZ, colors='white')
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, facecolor='black')
    plt.close(fig)

    return field_discrete
