#!/usr/bin/env python3
# Downstream Tasks Module for instantngp_kidney.py
# Implements advanced feature extraction from trained neural fields

import torch
import os
from tqdm import trange

def extract_gradient_field(model, xyz, device, batch_size=2**20):
    """
    Extract gradients (∇F_θ(x,y,z)) from the trained neural field.
    Returns gradient vectors for all coordinates.
    """
    total_coords = xyz.shape[0]
    gradient_field = torch.zeros((total_coords, 3), device=device)
    
    num_batches = (total_coords + batch_size - 1) // batch_size
    
    for batch_idx in trange(num_batches, desc="Extracting gradients", ncols=150):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_coords)
        batch_coords = xyz[start_idx:end_idx].clone().detach().requires_grad_(True)
        
        # Forward pass
        density = model(batch_coords).sum()  # Sum to create a scalar for backward
        
        # Compute gradient
        gradient = torch.autograd.grad(
            outputs=density,
            inputs=batch_coords,
            create_graph=False,
            retain_graph=False)[0]
        
        gradient_field[start_idx:end_idx] = gradient
    
    return gradient_field

def calculate_surface_normals(gradient_field):
    """
    Calculate surface normals as n̂(x) = ∇Φ(x)/|∇Φ(x)|
    """
    # Compute gradient magnitude (add small epsilon to avoid division by zero)
    gradient_magnitude = torch.norm(gradient_field, dim=1, keepdim=True) + 1e-8
    
    # Normalize to get surface normals
    normals = gradient_field / gradient_magnitude
    
    return normals, gradient_magnitude

def calculate_divergence(gradient_field, xyz, depth, height, width, device):
    """
    Calculate divergence (∇⋅E(x)) which represents density distribution
    The divergence is calculated by reshaping gradient field to 3D volume
    and computing spatial derivatives
    """
    # Reshape gradient components to 3D volumes
    gx = torch.zeros((depth, height, width), device=device)
    gy = torch.zeros((depth, height, width), device=device)
    gz = torch.zeros((depth, height, width), device=device)
    
    # Convert flat indices back to 3D indices
    flat_indices = torch.arange(0, xyz.shape[0], device=device)
    z_indices = flat_indices // (height * width)
    y_indices = (flat_indices % (height * width)) // width
    x_indices = flat_indices % width
    
    # Place gradient components into 3D volumes
    gx[z_indices, y_indices, x_indices] = gradient_field[:, 0]
    gy[z_indices, y_indices, x_indices] = gradient_field[:, 1]
    gz[z_indices, y_indices, x_indices] = gradient_field[:, 2]
    
    # Calculate divergence using finite differences
    # For x component: partial derivative with respect to x
    dx = torch.zeros_like(gx)
    dx[:, :, 1:-1] = (gx[:, :, 2:] - gx[:, :, :-2]) / 2.0
    dx[:, :, 0] = gx[:, :, 1] - gx[:, :, 0]
    dx[:, :, -1] = gx[:, :, -1] - gx[:, :, -2]
    
    # For y component: partial derivative with respect to y
    dy = torch.zeros_like(gy)
    dy[:, 1:-1, :] = (gy[:, 2:, :] - gy[:, :-2, :]) / 2.0
    dy[:, 0, :] = gy[:, 1, :] - gy[:, 0, :]
    dy[:, -1, :] = gy[:, -1, :] - gy[:, -2, :]
    
    # For z component: partial derivative with respect to z
    dz = torch.zeros_like(gz)
    dz[1:-1, :, :] = (gz[2:, :, :] - gz[:-2, :, :]) / 2.0
    dz[0, :, :] = gz[1, :, :] - gz[0, :, :]
    dz[-1, :, :] = gz[-1, :, :] - gz[-2, :, :]
    
    # Divergence is the sum of the partial derivatives
    divergence = dx + dy + dz
    
    return divergence

def compute_hessian_eigenvalues(model, xyz, device, batch_size=2**19, epsilon=1e-4):
    """
    Compute principal curvatures using finite differences to approximate Hessian eigenvalues
    (avoids double backward pass which tiny-cuda-nn doesn't support)
    """
    total_coords = xyz.shape[0]
    curvature_field = torch.zeros((total_coords, 3), device=device)
    
    model.eval()
    
    num_batches = (total_coords + batch_size - 1) // batch_size
    
    for batch_idx in trange(num_batches, desc="Computing curvatures", ncols=150):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_coords)
        batch_coords = xyz[start_idx:end_idx]
        batch_size_actual = batch_coords.shape[0]
        
        # Process in smaller sub-batches for memory efficiency
        sub_batch_size = 16000
        num_sub_batches = (batch_size_actual + sub_batch_size - 1) // sub_batch_size
        
        for sub_batch_idx in range(num_sub_batches):
            sub_start = sub_batch_idx * sub_batch_size
            sub_end = min(sub_start + sub_batch_size, batch_size_actual)
            sub_coords = batch_coords[sub_start:sub_end]
            
            # Approximate Hessian eigenvalues using finite differences
            curvatures = approximate_curvature_finite_diff(model, sub_coords, device, epsilon)
            curvature_field[start_idx + sub_start:start_idx + sub_end] = curvatures

    return curvature_field

def approximate_curvature_finite_diff(model, coords, device, epsilon=1e-4):
    """
    Approximate principal curvatures using finite differences on the gradient field
    """
    n_points = coords.shape[0]
    curvatures = torch.zeros((n_points, 3), device=device)
    
    # Approximate second derivatives using finite differences
    for dim in range(3):
        # Create perturbed coordinates
        coords_plus = coords.clone()
        coords_minus = coords.clone()
        coords_plus[:, dim] += epsilon
        coords_minus[:, dim] -= epsilon
        
        # Clamp to valid range [0, 1]
        coords_plus = torch.clamp(coords_plus, 0, 1)
        coords_minus = torch.clamp(coords_minus, 0, 1)
        
        # Get gradients at perturbed points
        coords_plus_grad = coords_plus.clone().requires_grad_(True)
        coords_minus_grad = coords_minus.clone().requires_grad_(True)
        
        with torch.enable_grad():
            output_plus = model(coords_plus_grad)
            output_minus = model(coords_minus_grad)
            
            grad_plus = torch.autograd.grad(
                outputs=output_plus.sum(),
                inputs=coords_plus_grad,
                create_graph=False,
                retain_graph=False)[0]
            
            grad_minus = torch.autograd.grad(
                outputs=output_minus.sum(),
                inputs=coords_minus_grad,
                create_graph=False,
                retain_graph=False)[0]
            
            # Approximate second derivative
            second_deriv = (grad_plus - grad_minus) / (2 * epsilon)
            
            # Use magnitude of second derivative as curvature approximation
            curvatures[:, dim] = torch.norm(second_deriv, dim=1)
    
    return curvatures

def run_feature_extraction(model, xyz, depth, height, width, device, 
                          output_dir="features", subsample_factor=1.0, gradients_only=True):
    """
    Extract gradient field and other features from a trained neural field model
    
    Args:
        model: Trained neural field model
        xyz: Coordinate grid tensor
        depth, height, width: Volume dimensions  
        device: CUDA/CPU device
        output_dir: Directory to save features
        subsample_factor: Factor to subsample coordinates (default: no subsampling)
        gradients_only: If True, only extract gradients (faster)
    
    Returns:
        output_dir: Directory where features were saved
    """
    print("Starting feature extraction pipeline...")
    print(f"Volume dimensions: {depth}×{height}×{width}")
    print(f"Curvature subsample factor: {subsample_factor}")
    
    print("Starting feature extraction with volume reconstruction...")
    print("First: Reconstructed volume visualization in napari")
    print("Second: Surface normals (RGB) visualization in napari")
    print("Third: Mean & Gaussian curvature visualization in napari")
    features_dir = os.path.join(output_dir, "features") if "features" not in output_dir else output_dir
    os.makedirs(features_dir, exist_ok=True)
    
    # First reconstruct the volume from the neural field
    # Use half precision to match model output
    reconstructed_volume = torch.zeros((depth, height, width), device=device, dtype=torch.half)
    
    # Evaluate the model on all coordinates to get the reconstructed volume
    with torch.no_grad():
        num_batches = (xyz.shape[0] + 2**20 - 1) // 2**20
        for batch_idx in trange(num_batches, desc="Reconstructing volume", ncols=150):
            start_idx = batch_idx * 2**20
            end_idx = min(start_idx + 2**20, xyz.shape[0])
            batch_coords = xyz[start_idx:end_idx]
            
            # Get model predictions
            batch_output = model(batch_coords).squeeze()
            
            # Convert flat indices back to 3D indices
            batch_flat_indices = torch.arange(start_idx, end_idx, device=device)
            z_indices = batch_flat_indices // (height * width)
            y_indices = (batch_flat_indices % (height * width)) // width
            x_indices = batch_flat_indices % width
            
            # Place predictions into 3D volume
            reconstructed_volume[z_indices, y_indices, x_indices] = batch_output
    
    # Show reconstructed volume in napari first
    try:
        import napari
        
        # Convert to numpy for napari
        volume_np = reconstructed_volume.cpu().numpy()
        
        # Create viewer with simple, stable configuration
        viewer = napari.Viewer()
        
        # Calculate volume statistics for better contrast
        vol_min = volume_np.min()
        vol_max = volume_np.max()
        vol_std = volume_np.std()
        vol_mean = volume_np.mean()
        
        # Set contrast limits to show structure with transparency
        # Use mean ± 2*std to focus on interesting regions
        contrast_min = max(vol_min, vol_mean - 2 * vol_std)
        contrast_max = min(vol_max, vol_mean + 2 * vol_std)
        
        # Add reconstructed volume
        viewer.add_image(
            volume_np,
            name='Reconstructed Volume',
            colormap='viridis',
            contrast_limits=[contrast_min, contrast_max]
        )
        
        # Set 3D display mode
        viewer.dims.ndisplay = 3
        
        # Start napari event loop (blocks until window is closed)
        napari.run()
    except ImportError:
        pass
    except Exception as e:
        pass

    # Extract gradient field
    gradient_field = extract_gradient_field(model, xyz, device)
    
    # Calculate surface normals and gradient magnitude
    normals, gradient_magnitude = calculate_surface_normals(gradient_field)
    
    # Create 3D volumes from flattened fields for napari visualization
    flat_indices = torch.arange(0, xyz.shape[0], device=device)
    z_indices = flat_indices // (height * width)
    y_indices = (flat_indices % (height * width)) // width
    x_indices = flat_indices % width
    
    # Create color representations of normals (RGB mapping)
    normals_3d = torch.zeros((depth, height, width, 3), device=device)
    # Scale from [-1,1] to [0,1] for RGB
    normals_rgb = (normals + 1) / 2
    normals_3d[z_indices, y_indices, x_indices] = normals_rgb
    
    # Show surface normals in napari as RGB volume
    try:
        import napari
        
        # Convert to numpy for napari
        normals_np = normals_3d.cpu().numpy()
        
        # Create viewer with simple, stable configuration
        viewer = napari.Viewer()
        
        # Add normals as RGB image (no colormap needed)
        viewer.add_image(
            normals_np,
            name='Surface Normals (RGB)',
            rgb=True,  # Interpret as RGB data
            contrast_limits=[0, 1]  # RGB values are in [0,1]
        )
        
        # Set 3D display mode
        viewer.dims.ndisplay = 3
        
        # Start napari event loop (blocks until window is closed)
        napari.run()
    except ImportError:
        pass
    except Exception as e:
        pass
    
    
    # Calculate principal curvatures (sampling a subset for efficiency)
    # Subsample for efficiency
    subsample_z = torch.arange(0, depth, int(subsample_factor), dtype=torch.long, device=device)
    subsample_y = torch.arange(0, height, int(subsample_factor), dtype=torch.long, device=device)
    subsample_x = torch.arange(0, width, int(subsample_factor), dtype=torch.long, device=device)
    
    ss_z, ss_y, ss_x = torch.meshgrid(subsample_z, subsample_y, subsample_x, indexing='ij')
    subsample_indices = ss_z.flatten() * (height * width) + ss_y.flatten() * width + ss_x.flatten()
    subsample_indices = subsample_indices[subsample_indices < xyz.shape[0]].long()
    
    subsample_xyz = xyz[subsample_indices]
    
    # Compute Hessian eigenvalues on the subsampled points
    curvature_field = compute_hessian_eigenvalues(model, subsample_xyz, device)
    
    # Save middle slice visualizations of all features
    middle_slice = depth // 2
    
    # Reconstruct gradient magnitude 3D volume
    gradient_magnitude_3d = torch.zeros((depth, height, width), device=device)
    gradient_magnitude_3d[z_indices, y_indices, x_indices] = gradient_magnitude.squeeze(1)
    

    # Reconstruct and save principal curvatures for the subsampled volume
    if curvature_field.shape[0] > 0:
        # Create 3D volume with subsampling taken into account
        subsampled_depth = len(subsample_z)
        subsampled_height = len(subsample_y)
        subsampled_width = len(subsample_x)
        
        # Mean curvature (average of first two eigenvalues)
        mean_curvature = (curvature_field[:, 0] + curvature_field[:, 1]) / 2
        mean_curvature_3d = mean_curvature.reshape(subsampled_depth, subsampled_height, subsampled_width)
        
        # Gaussian curvature (product of eigenvalues)
        gaussian_curvature = curvature_field[:, 0] * curvature_field[:, 1] * curvature_field[:, 2]
        gaussian_curvature_3d = gaussian_curvature.reshape(subsampled_depth, subsampled_height, subsampled_width)
        
        # Show curvature results in napari
        try:
            import napari
            
            # Convert to numpy for napari
            mean_curvature_np = mean_curvature_3d.cpu().numpy()
            gaussian_curvature_np = gaussian_curvature_3d.cpu().numpy()
            
            # Create viewer with simple, stable configuration
            viewer = napari.Viewer()
            
            # Calculate curvature statistics for better contrast
            mean_curv_min = mean_curvature_np.min()
            mean_curv_max = mean_curvature_np.max()
            mean_curv_std = mean_curvature_np.std()
            mean_curv_mean = mean_curvature_np.mean()
            
            gauss_curv_min = gaussian_curvature_np.min()
            gauss_curv_max = gaussian_curvature_np.max()
            gauss_curv_std = gaussian_curvature_np.std()
            gauss_curv_mean = gaussian_curvature_np.mean()
            
            # Set contrast limits for mean curvature
            mean_contrast_min = max(mean_curv_min, mean_curv_mean - 2 * mean_curv_std)
            mean_contrast_max = min(mean_curv_max, mean_curv_mean + 2 * mean_curv_std)
            
            # Set contrast limits for Gaussian curvature  
            gauss_contrast_min = max(gauss_curv_min, gauss_curv_mean - 2 * gauss_curv_std)
            gauss_contrast_max = min(gauss_curv_max, gauss_curv_mean + 2 * gauss_curv_std)
            
            # Add mean curvature
            viewer.add_image(
                mean_curvature_np,
                name='Mean Curvature',
                colormap='plasma',
                contrast_limits=[mean_contrast_min, mean_contrast_max]
            )
            
            # Add Gaussian curvature
            viewer.add_image(
                gaussian_curvature_np,
                name='Gaussian Curvature',
                colormap='turbo',
                contrast_limits=[gauss_contrast_min, gauss_contrast_max]
            )
            
            # Set 3D display mode
            viewer.dims.ndisplay = 3
            
            print("\nCurvature statistics:")
            print(f"Mean curvature: min={mean_curv_min:.4f}, max={mean_curv_max:.4f}, mean={mean_curv_mean:.4f}")
            print(f"Gaussian curvature: min={gauss_curv_min:.4f}, max={gauss_curv_max:.4f}, mean={gauss_curv_mean:.4f}")
            
            # Start napari event loop (blocks until window is closed)
            napari.run()
        except ImportError:
            pass
        except Exception as e:
            pass

    return features_dir