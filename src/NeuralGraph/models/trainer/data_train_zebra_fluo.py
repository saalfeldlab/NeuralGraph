"""Extracted from graph_trainer.py"""
from ._imports import *

def data_train_zebra_fluo(config, erase, best_model, device):
    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    dimension = simulation_config.dimension
    n_epochs = train_config.n_epochs
    n_neurons = simulation_config.n_neurons
    n_input_neurons = simulation_config.n_input_neurons
    n_neuron_types = simulation_config.n_neuron_types
    calcium_type = simulation_config.calcium_type
    delta_t = simulation_config.delta_t

    dataset_name = config.dataset
    n_runs = train_config.n_runs
    n_frames = simulation_config.n_frames

    data_augmentation_loop = train_config.data_augmentation_loop
    recurrent_training = train_config.recurrent_training
    recurrent_loop = train_config.recurrent_loop
    batch_size = train_config.batch_size
    batch_ratio = train_config.batch_ratio
    training_NNR_start_epoch = train_config.training_NNR_start_epoch
    time_window = train_config.time_window
    plot_batch_size = config.plotting.plot_batch_size

    coeff_NNR_f = train_config.coeff_NNR_f

    if config.training.seed != 42:
        torch.random.fork_rng(devices=device)
        torch.random.manual_seed(config.training.seed)

    cmap = CustomColorMap(config=config)
    plt.style.use('dark_background')

    log_dir, logger = create_log_dir(config, erase)

    dy_um = config.zarr.dy_um
    dx_um = config.zarr.dx_um
    dz_um = config.zarr.dz_um  # light sheet step

    STORE = config.zarr.store_fluo
    FRAME = 3736

    ds = open_gcs_zarr(STORE)
    vol_xyz = ds[..., FRAME].read().result()

    ground_truth = torch.tensor(vol_xyz, device=device, dtype=torch.float32) / 512
    ground_truth = ground_truth.permute(1,0,2)

    # down sample
    factor = 4
    gt = ground_truth.unsqueeze(0).unsqueeze(0)
    gt_down = F.interpolate(
        gt,
        size=(gt.shape[2] // factor, gt.shape[3] // factor, gt.shape[4]),
        mode='trilinear',
        align_corners=False
    )
    ground_truth = gt_down.squeeze(0).squeeze(0)
    nx, ny, nz = ground_truth.shape
    print(f'\nshape: {ground_truth.shape}')

    side_length = max(nx, ny)
    iy, ix, iz = torch.meshgrid(
        torch.arange(ny, device=device),
        torch.arange(nx, device=device),
        torch.arange(nz, device=device),
        indexing='ij'  # valid values: 'ij' or 'xy'
    )
    model_input = torch.stack([iy.reshape(-1), ix.reshape(-1), iz.reshape(-1)], dim=1).to(dtype=torch.float32)  # [nx*ny*nz, 3]
    model_input[:,0] = model_input[:,0].float() / (side_length - 1)
    model_input[:,1] = model_input[:,1].float() / (side_length - 1)
    model_input[:,2] = model_input[:,2].float() / nz
    model_input = model_input.cuda()

    ground_truth = ground_truth.reshape([nx*ny*nz, 1]).cuda()

    img_siren = Siren(in_features=3, out_features=1, hidden_features=1024, hidden_layers=3, outermost_linear=True, first_omega_0=512., hidden_omega_0=512.)
    img_siren.cuda()
    total_steps = 3000  # Since the whole image is our dataset, this just means 500 gradient descent steps.
    steps_til_summary = 1000
    optim = torch.optim.Adam(lr=1e-5, params=img_siren.parameters())

    batch_ratio = 0.01

    loss_list = []
    for step in trange(total_steps+1):

        sample_ids = np.random.choice(model_input.shape[0], int(model_input.shape[0]*batch_ratio), replace=False)
        model_input_batch = model_input[sample_ids]
        ground_truth_batch = ground_truth[sample_ids]
        loss = ((img_siren(model_input_batch) - ground_truth_batch) ** 2).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()

        loss_list.append(loss.item())

    if (step % steps_til_summary == 0) and (step>0):
        print("Step %d, Total loss %0.6f" % (step, loss))

        z_idx = 20

        z_vals = torch.unique(model_input[:, 2], sorted=True)
        z_val  = z_vals[z_idx]
        plane_mask = torch.isclose(model_input[:, 2], z_val, atol=1e-6, rtol=0.0)
        plane_in = model_input[plane_mask]

        with torch.no_grad():
            model_output = img_siren(plane_in)

        gt = to_numpy(ground_truth).reshape(nx, ny, nz)   # both expected (nx, ny, nz)
        gt_xy = gt[:, :, z_idx].astype(np.float32)
        pd_xy = to_numpy(model_output).reshape(nx, ny)

        vmin, vmax = np.percentile(gt_xy, [1, 99.9]);
        vmin = float(vmin) if np.isfinite(vmin) else float(min(gt_xy.min(), pd_xy.min()))
        vmax = float(vmax) if np.isfinite(vmax) and vmax>vmin else float(max(gt_xy.max(), pd_xy.max()))
        rmse = float(np.sqrt(np.mean((pd_xy - gt_xy) ** 2)))
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        im0 = ax[0].imshow(gt_xy, cmap="gray", vmin=vmin, vmax=vmax); ax[0].set_title(f"GT z={z_idx}"); ax[0].axis("off")
        im1 = ax[1].imshow(pd_xy, cmap="gray", vmin=vmin, vmax=vmax); ax[1].set_title(f"Pred z={z_idx}  RMSE={rmse:.4f}"); ax[1].axis("off")
        fig.colorbar(im1, ax=ax.ravel().tolist(), fraction=0.046, pad=0.04, label="intensity")
        plt.show()

    viewer = napari.Viewer()
    viewer.add_image(gt, name='ground_truth', scale=[dy_um*factor, dx_um*factor, dz_um])
    viewer.add_image(pd, name='pred_img', scale=[dy_um*factor, dx_um*factor, dz_um])
    viewer.dims.ndisplay = 3
    napari.run()

