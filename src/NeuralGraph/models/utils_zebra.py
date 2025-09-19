import torch
import numpy as np
import matplotlib.pyplot as plt
import datashader as ds
import datashader.transfer_functions as tf
from datashader import utils
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize

def plot_field_comparison(x, model, k, n_frames, ones, output_path, step):
    """
    Plots true field, NNR field, and noisy NNR field using datashader for fast rendering.
    
    Args:
        x: torch.Tensor, shape [n_neurons, features]
        model: model with NNR_f method
        n_frames: int, total number of frames
        ones: torch.Tensor, shape [n_neurons, 1]
        output_path: str, output file path
        step: int, number of perturbations for continuous plot
    """
    x = torch.tensor(x, dtype=torch.float32, device=ones.device)
    
    vmin = 0.048
    vmax = 0.451
    
    # Compute field once for discrete plots
    in_features = torch.cat((x[:,1:4], k/n_frames * ones), 1)
    field_discrete = model.NNR_f(in_features)**2

    # Create all perturbations at once
    x_perturbed = x.unsqueeze(0).repeat(step, 1, 1)  # [step, n_neurons, features]
    noise = torch.randn_like(x_perturbed) * 0.01
    x_perturbed = x_perturbed + noise
    
    # Batch process all perturbations
    x_perturbed_flat = x_perturbed.reshape(-1, x.shape[1])  # [step * n_neurons, features]
    in_features_perturbed = torch.cat((
        x_perturbed_flat[:, 1:4], 
        k/n_frames * ones.repeat(step, 1)
    ), 1)
    
    with torch.no_grad():  # Save memory
        field_continuous = model.NNR_f(in_features_perturbed)**2
    field_continuous = field_continuous.reshape(step, -1)  # [step, n_neurons]
    
    # Move to CPU for plotting
    x_cpu = x.cpu().numpy()
    field_discrete_cpu = field_discrete.cpu().numpy().squeeze()
    x_perturbed_cpu = x_perturbed.cpu().numpy()
    field_continuous_cpu = field_continuous.cpu().numpy()
    

    # Setup figure
    fig = plt.figure(figsize=(20, 12))
    
    # Define canvas size for datashader
    canvas_width = 800
    canvas_height = 600
    
    # --- ROW 1: XY Views ---


    condition_names = {
        0: "gain",
        1: "dots",
        2: "flash",
        3: "taxis",
        4: "turning",
        5: "position",
        6: "open loop",
        7: "rotation",
        8: "dark",
        -1: "none"  # no condition / padding
    }
    
    # Plot 1: Data XY
    ax1 = plt.subplot(2, 3, 1)
    df1 = pd.DataFrame({
        'x': x_cpu[:, 1],
        'y': 1.35 - x_cpu[:, 2],
        'value': x_cpu[:, 6]
    })
    canvas_xy = ds.Canvas(plot_width=canvas_width, plot_height=canvas_height,
                          x_range=(0, 2), y_range=(0, 1.35))
    agg1 = canvas_xy.points(df1, 'x', 'y', ds.mean('value'))
    img1 = tf.shade(agg1, cmap=cm.plasma, how='linear', span=[vmin, vmax])
    img1_np = np.array(img1.to_pil())
    ax1.imshow(img1_np, extent=[0, 2, 0, 1.35], aspect='auto', origin='lower')
    ax1.set_title('data XY', fontsize=20)
    ax1.set_xlabel('X', fontsize=10)
    ax1.set_ylabel('Y', fontsize=10)
    
    # Add condition text if available

    condition_idx = int(x_cpu[0, 7])
    condition_text = condition_names.get(condition_idx, "unknown")
    ax1.text(0.05, 0.95, f"{condition_text}\nframe: {k}", transform=ax1.transAxes,fontsize=14, verticalalignment='top', alpha=0.9)

    # Plot 2: NNR field XY - discrete
    ax2 = plt.subplot(2, 3, 2)
    df2 = pd.DataFrame({
        'x': x_cpu[:, 1],
        'y': 1.35 - x_cpu[:, 2],
        'value': field_discrete_cpu
    })
    agg2 = canvas_xy.points(df2, 'x', 'y', ds.mean('value'))
    img2 = tf.shade(agg2, cmap=cm.plasma, how='linear', span=[vmin, vmax])
    img2_np = np.array(img2.to_pil())
    ax2.imshow(img2_np, extent=[0, 2, 0, 1.35], aspect='auto', origin='lower')
    ax2.set_title('NNR field XY - discrete', fontsize=20)
    ax2.set_xlabel('X', fontsize=10)
    ax2.set_ylabel('Y', fontsize=10)
    
    # Plot 3: NNR field XY - continuous
    ax3 = plt.subplot(2, 3, 3)
    # Flatten all perturbations
    x_cont_flat = x_perturbed_cpu.reshape(-1, x_perturbed_cpu.shape[2])
    field_cont_flat = field_continuous_cpu.flatten()
    
    df3 = pd.DataFrame({
        'x': x_cont_flat[:, 1],
        'y': 1.35 - x_cont_flat[:, 2],
        'value': field_cont_flat
    })
    agg3 = canvas_xy.points(df3, 'x', 'y', ds.mean('value'))
    img3 = tf.shade(agg3, cmap=cm.plasma, how='linear', span=[vmin, vmax])
    img3_np = np.array(img3.to_pil())
    ax3.imshow(img3_np, extent=[0, 2, 0, 1.35], aspect='auto', origin='lower')
    ax3.set_title('NNR field XY - continuous', fontsize=20)
    ax3.set_xlabel('X', fontsize=10)
    ax3.set_ylabel('Y', fontsize=10)
    
    # --- ROW 2: XZ Views ---
    
    # Canvas for XZ view
    canvas_xz = ds.Canvas(plot_width=canvas_width, plot_height=int(canvas_height*0.5),
                          x_range=(0, 2), y_range=(0, 0.7))
    
    # Plot 4: Data XZ
    ax4 = plt.subplot(4, 3, 7)
    df4 = pd.DataFrame({
        'x': x_cpu[:, 1],
        'z': 0.7 - x_cpu[:, 3],
        'value': x_cpu[:, 6]
    })
    agg4 = canvas_xz.points(df4, 'x', 'z', ds.mean('value'))
    img4 = tf.shade(agg4, cmap=cm.plasma, how='linear', span=[vmin, vmax])
    img4_np = np.array(img4.to_pil())
    ax4.imshow(img4_np, extent=[0, 2, 0, 0.7], aspect='auto', origin='lower')
    ax4.set_title('data XZ', fontsize=20)
    ax4.set_xlabel('X', fontsize=10)
    ax4.set_ylabel('Z', fontsize=10)
    
    # Plot 5: NNR field XZ - discrete
    ax5 = plt.subplot(4, 3, 8)
    df5 = pd.DataFrame({
        'x': x_cpu[:, 1],
        'z': 0.7 - x_cpu[:, 3],
        'value': field_discrete_cpu
    })
    agg5 = canvas_xz.points(df5, 'x', 'z', ds.mean('value'))
    img5 = tf.shade(agg5, cmap=cm.plasma, how='linear', span=[vmin, vmax])
    img5_np = np.array(img5.to_pil())
    ax5.imshow(img5_np, extent=[0, 2, 0, 0.7], aspect='auto', origin='lower')
    ax5.set_title('NNR field XZ - discrete', fontsize=20)
    ax5.set_xlabel('X', fontsize=10)
    ax5.set_ylabel('Z', fontsize=10)
    
    # Plot 6: NNR field XZ - continuous
    ax6 = plt.subplot(4, 3, 9)
    df6 = pd.DataFrame({
        'x': x_cont_flat[:, 1],
        'z': 0.7 - x_cont_flat[:, 3],
        'value': field_cont_flat
    })
    agg6 = canvas_xz.points(df6, 'x', 'z', ds.mean('value'))
    img6 = tf.shade(agg6, cmap=cm.plasma, how='linear', span=[vmin, vmax])
    img6_np = np.array(img6.to_pil())
    ax6.imshow(img6_np, extent=[0, 2, 0, 0.7], aspect='auto', origin='lower')
    ax6.set_title('NNR field XZ - continuous', fontsize=20)
    ax6.set_xlabel('X', fontsize=10)
    ax6.set_ylabel('Z', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=80)
    plt.close()

    return field_discrete



# pip install pyvista numpy matplotlib
import numpy as np
import torch
import pyvista as pv
import matplotlib.pyplot as plt

def show_continuous_field_3d(
    x: torch.Tensor,                  # [N, features], positions in x[:,1:4]
    model,                            # has .NNR_f(in_features) -> [M,1]
    k: int,                           # current frame index
    n_frames: int,                    # total frames
    ones: torch.Tensor,               # [N,1] (used only for device/time scalar)
    grid_shape=(128, 96, 64),         # (nx, ny, nz) sampling resolution
    bounds=None,                      # ((xmin,xmax),(ymin,ymax),(zmin,zmax)); default = data bounds
    overlay_points=True,              # draw neuron positions as faint points
    point_size=2.0,
    jitter_sigma=0.0,                 # optional coord jitter (in data units) before sampling
    cmap="plasma",
    opacity="sigmoid",                # PyVista presets: 'linear','sigmoid','sigmoid_5','sigmoid_50', etc.
    clim=None,                        # (vmin, vmax); if None, computed from volume percentiles
    batch_size=1_000_000,             # how many grid points per model forward (tune for your GPU/CPU)
    background="black",
    show_axes=True,
):
    """
    Render the continuous NNR field f(x,y,z,t=k/n_frames) as an interactive 3D volume.

    - Samples model.NNR_f on a regular grid covering the neuron cloud (or custom bounds).
    - One batched forward (chunked) builds a dense volume for fast GPU volume rendering.
    - Optionally overlays neuron positions as points for anatomical context.

    Controls: rotate/zoom with mouse; press 'o' in the scalar bar to toggle log; use toolbar.
    """

    # ---- prep positions / bounds ----
    device = ones.device
    x = x.to(device, dtype=torch.float32)
    pos = x[:, 1:4]  # [N,3]

    if bounds is None:
        mins = pos.min(dim=0).values.detach().cpu().numpy()
        maxs = pos.max(dim=0).values.detach().cpu().numpy()
        # small padding so outer voxels are visible
        pad = 0.02 * (maxs - mins + 1e-12)
        mins, maxs = mins - pad, maxs + pad
    else:
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
        mins = np.array([xmin, ymin, zmin], dtype=np.float32)
        maxs = np.array([xmax, ymax, zmax], dtype=np.float32)

    nx, ny, nz = map(int, grid_shape)
    xs = torch.linspace(float(mins[0]), float(maxs[0]), nx, device=device)
    ys = torch.linspace(float(mins[1]), float(maxs[1]), ny, device=device)
    zs = torch.linspace(float(mins[2]), float(maxs[2]), nz, device=device)

    # ---- build grid coordinates (x,y,z) ----
    # We generate in blocks along z to keep memory low.
    t_scalar = (k / float(n_frames))
    # A single scalar time for all grid points
    tcol = torch.full((1, 1), t_scalar, device=device, dtype=torch.float32)

    vol = torch.empty((nx, ny, nz), device=device, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        for iz in range(nz):
            zslice = zs[iz].repeat(nx * ny)
            xv, yv = torch.meshgrid(xs, ys, indexing="ij")  # (nx,ny)
            xflat = xv.reshape(-1)
            yflat = yv.reshape(-1)

            if jitter_sigma > 0:
                xflat = xflat + torch.randn_like(xflat) * jitter_sigma
                yflat = yflat + torch.randn_like(yflat) * jitter_sigma
                zslice = zslice + torch.randn_like(zslice) * jitter_sigma

            # Assemble features [x,y,z,t]
            feats = torch.stack([xflat, yflat, zslice], dim=1)   # (nx*ny, 3)
            tvec = tcol.expand(feats.shape[0], -1)              # (nx*ny, 1)
            in_features = torch.cat([feats, tvec], dim=1)       # (nx*ny, 4)

            # Chunked forward to avoid OOM
            out = []
            for start in range(0, in_features.shape[0], batch_size):
                stop = min(start + batch_size, in_features.shape[0])
                yhat = model.NNR_f(in_features[start:stop]).pow(2)  # (m,1)
                out.append(yhat.squeeze(1))
            plane = torch.cat(out, dim=0).reshape(nx, ny)  # (nx,ny)

            vol[:, :, iz] = plane

    # Move to NumPy for VTK
    vol_np = vol.detach().cpu().numpy()

    # ---- color limits ----
    if clim is None:
        vmin, vmax = np.percentile(vol_np, [2, 98])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = float(np.nanmin(vol_np)), float(np.nanmax(vol_np))
    else:
        vmin, vmax = clim

    # ---- build UniformGrid for volume rendering ----
    # VTK expects spacing and origin in (x,y,z); data layout is Fortran-ordered.
    dx = (maxs[0] - mins[0]) / max(nx - 1, 1)
    dy = (maxs[1] - mins[1]) / max(ny - 1, 1)
    dz = (maxs[2] - mins[2]) / max(nz - 1, 1)

    grid = pv.UniformGrid() if hasattr(pv, "UniformGrid") else pv.ImageData()
    grid.dimensions = np.array([nx, ny, nz])  # number of points on each axis
    grid.origin = (float(mins[0]), float(mins[1]), float(mins[2]))
    grid.spacing = (float(dx), float(dy), float(dz))
    # Ensure Fortran order when attaching scalars
    grid.point_data["field"] = np.ascontiguousarray(vol_np.ravel(order="F"))

    # ---- Plotter ----
    plotter = pv.Plotter(window_size=(1200, 900))
    plotter.set_background(background)
    plasma = plt.get_cmap(cmap)

    # Volume

    from pyvista import opacity_transfer_function

    tf = opacity_transfer_function('sigmoid_10', 256) * 0.1

    plotter.add_volume(
        grid,
        cmap=plasma,
        opacity=tf,      # try 'sigmoid_5' to emphasize high values
        clim=(vmin, vmax),
        shade=False,          # shading off often looks better for scalar fields
        scalar_bar_args=dict(title="NNR field", label_font_size=10, title_font_size=12),
    )

    # Optional neuron overlay for context (thin and faint)
    if overlay_points:
        pts = pos.detach().cpu().numpy()
        cloud = pv.PolyData(pts)
        plotter.add_points(
            cloud,
            color="white",
            point_size=point_size,
            render_points_as_spheres=True,
            opacity=0.08,
        )

    if show_axes:
        plotter.show_axes()
    plotter.show_grid(color="gray")

    # Nice default view
    plotter.view_isometric()
    try:
        plotter.enable_anti_aliasing()
    except Exception:
        pass

    # For headless servers:
    # pv.start_xvfb()  # uncomment if needed before calling this function

    # Launch interactive window
    plotter.show()

