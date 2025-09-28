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
