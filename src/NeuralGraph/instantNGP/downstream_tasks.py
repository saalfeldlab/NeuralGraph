#!/usr/bin/env python3
# Downstream Tasks Module for instantngp_kidney.py
# Implements advanced feature extraction from trained neural fields

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os
import tifffile

def extract_gradient_field(model, xyz, device, batch_size=2**16):
    """
    Extract gradients (∇F_θ(x,y,z)) from the trained neural field.
    Returns gradient vectors for all coordinates.
    """
    total_coords = xyz.shape[0]
    gradient_field = torch.zeros((total_coords, 3), device=device)
    
    for start_idx in range(0, total_coords, batch_size):
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
        
        if (start_idx // batch_size) % 10 == 0:
            progress = 100.0 * end_idx / total_coords
            print(f"  Gradient field computation: {progress:.1f}%")
    
    print("Gradient field extraction completed")
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

def compute_hessian_eigenvalues(model, xyz, device, batch_size=10000, epsilon=1e-4):
    """
    Compute principal curvatures using finite differences to approximate Hessian eigenvalues
    (avoids double backward pass which tiny-cuda-nn doesn't support)
    """
    print("Computing principal curvatures (finite difference approximation)...")
    
    total_coords = xyz.shape[0]
    curvature_field = torch.zeros((total_coords, 3), device=device)
    
    model.eval()
    
    for start_idx in range(0, total_coords, batch_size):
        end_idx = min(start_idx + batch_size, total_coords)
        batch_coords = xyz[start_idx:end_idx]
        batch_size_actual = batch_coords.shape[0]
        
        # Process in smaller sub-batches for memory efficiency
        sub_batch_size = 1000
        for sub_start in range(0, batch_size_actual, sub_batch_size):
            sub_end = min(sub_start + sub_batch_size, batch_size_actual)
            sub_coords = batch_coords[sub_start:sub_end]
            
            # Approximate Hessian eigenvalues using finite differences
            curvatures = approximate_curvature_finite_diff(model, sub_coords, device, epsilon)
            curvature_field[start_idx + sub_start:start_idx + sub_end] = curvatures
        
        if (start_idx // batch_size) % 5 == 0:
            progress = 100.0 * end_idx / total_coords
            print(f"  Curvature computation: {progress:.1f}%")
    
    print("Curvature calculation completed")
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

def run_feature_extraction(model, xyz, depth, height, width, device, output_dir, subsample_factor=4):
    """
    Run all feature extraction methods and save results
    
    Args:
        subsample_factor: Subsampling factor for curvature computation (default: 4)
    """
    print("Starting feature extraction pipeline...")
    print(f"Volume dimensions: {depth}×{height}×{width}")
    print(f"Curvature subsample factor: {subsample_factor}")
    
    features_dir = os.path.join(output_dir, "features") if "features" not in output_dir else output_dir
    os.makedirs(features_dir, exist_ok=True)
    
    # Extract gradient field
    print("Extracting gradient field...")
    gradient_field = extract_gradient_field(model, xyz, device)
    
    # Calculate surface normals and gradient magnitude
    print("Calculating surface normals...")
    normals, gradient_magnitude = calculate_surface_normals(gradient_field)
    
    # Calculate divergence for density distribution
    print("Computing density distribution (divergence)...")
    divergence = calculate_divergence(gradient_field, xyz, depth, height, width, device)
    
    # Calculate principal curvatures (sampling a subset for efficiency)
    print("Computing principal curvatures (Hessian eigenvalues)...")
    # Subsample for efficiency
    subsample_z = torch.arange(0, depth, subsample_factor)
    subsample_y = torch.arange(0, height, subsample_factor)
    subsample_x = torch.arange(0, width, subsample_factor)
    
    ss_z, ss_y, ss_x = torch.meshgrid(subsample_z, subsample_y, subsample_x, indexing='ij')
    subsample_indices = ss_z.flatten() * (height * width) + ss_y.flatten() * width + ss_x.flatten()
    subsample_indices = subsample_indices[subsample_indices < xyz.shape[0]]
    
    subsample_xyz = xyz[subsample_indices]
    
    # Compute Hessian eigenvalues on the subsampled points
    curvature_field = compute_hessian_eigenvalues(model, subsample_xyz, device)
    
    # Save middle slice visualizations of all features
    middle_slice = depth // 2
    
    # Reconstruct 3D volumes from flattened fields
    gradient_magnitude_3d = torch.zeros((depth, height, width), device=device)
    flat_indices = torch.arange(0, xyz.shape[0], device=device)
    z_indices = flat_indices // (height * width)
    y_indices = (flat_indices % (height * width)) // width
    x_indices = flat_indices % width
    
    # Place gradient magnitude into 3D volume
    gradient_magnitude_3d[z_indices, y_indices, x_indices] = gradient_magnitude.squeeze(1)
    
    # Create color representations of normals (RGB mapping)
    normals_3d = torch.zeros((depth, height, width, 3), device=device)
    # Scale from [-1,1] to [0,1] for RGB
    normals_rgb = (normals + 1) / 2
    normals_3d[z_indices, y_indices, x_indices] = normals_rgb
    
    # Save middle slices
    print("Saving feature visualizations...")
    
    # Save gradient magnitude slice
    gradient_mag_slice = gradient_magnitude_3d[middle_slice].cpu().numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(gradient_mag_slice, cmap='inferno')
    plt.colorbar(label='Gradient Magnitude')
    plt.title('Gradient Magnitude (Edge Detection)')
    plt.savefig(os.path.join(features_dir, "gradient_magnitude.png"), dpi=150)
    plt.close()
    
    # Save normal map slice (RGB)
    normals_slice = normals_3d[middle_slice].cpu().numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(normals_slice)
    plt.title('Surface Normals (RGB Mapping)')
    plt.savefig(os.path.join(features_dir, "surface_normals.png"), dpi=150)
    plt.close()
    
    # Save divergence slice
    divergence_slice = divergence[middle_slice].cpu().numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(divergence_slice, cmap='viridis')
    plt.colorbar(label='Divergence')
    plt.title('Density Distribution (Divergence ∇⋅E)')
    plt.savefig(os.path.join(features_dir, "divergence.png"), dpi=150)
    plt.close()
    
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
        
        # Middle slice of subsampled volume
        subsampled_middle = subsampled_depth // 2
        
        # Save mean curvature
        mean_curv_slice = mean_curvature_3d[subsampled_middle].cpu().numpy()
        plt.figure(figsize=(10, 10))
        plt.imshow(mean_curv_slice, cmap='coolwarm')
        plt.colorbar(label='Mean Curvature')
        plt.title('Mean Curvature (H)')
        plt.savefig(os.path.join(features_dir, "mean_curvature.png"), dpi=150)
        plt.close()
        
        # Save Gaussian curvature
        gaussian_curv_slice = gaussian_curvature_3d[subsampled_middle].cpu().numpy()
        plt.figure(figsize=(10, 10))
        plt.imshow(gaussian_curv_slice, cmap='coolwarm', norm=Normalize(vmin=-0.01, vmax=0.01))
        plt.colorbar(label='Gaussian Curvature')
        plt.title('Gaussian Curvature (K)')
        plt.savefig(os.path.join(features_dir, "gaussian_curvature.png"), dpi=150)
        plt.close()
    
    # Save full 3D volumes as TIFF
    print("Saving 3D feature volumes...")
    
    # Save gradient magnitude volume
    gradient_mag_volume = gradient_magnitude_3d.cpu().numpy()
    # Normalize to [0, 65535] for uint16
    gradient_mag_norm = np.clip(gradient_mag_volume / np.max(gradient_mag_volume) * 65535, 0, 65535).astype(np.uint16)
    tifffile.imwrite(os.path.join(features_dir, "gradient_magnitude_volume.tif"), gradient_mag_norm)
    
    # Save divergence volume
    divergence_volume = divergence.cpu().numpy()
    # Center around zero and scale to [0, 65535]
    divergence_min = np.min(divergence_volume)
    divergence_max = np.max(divergence_volume)
    div_range = max(abs(divergence_min), abs(divergence_max))
    if div_range > 0:
        divergence_norm = ((divergence_volume / div_range) * 0.5 + 0.5) * 65535
    else:
        divergence_norm = np.zeros_like(divergence_volume)
    divergence_norm = np.clip(divergence_norm, 0, 65535).astype(np.uint16)
    tifffile.imwrite(os.path.join(features_dir, "divergence_volume.tif"), divergence_norm)
    
    # Create a feature summary file
    with open(os.path.join(features_dir, "feature_summary.txt"), 'w') as f:
        f.write("Feature Extraction Summary\n")
        f.write("========================\n\n")
        f.write(f"Gradient magnitude range: [{gradient_mag_volume.min():.6f}, {gradient_mag_volume.max():.6f}]\n")
        f.write(f"Divergence range: [{divergence_volume.min():.6f}, {divergence_volume.max():.6f}]\n")
        if curvature_field.shape[0] > 0:
            mean_curv_volume = mean_curvature_3d.cpu().numpy()
            gaussian_curv_volume = gaussian_curvature_3d.cpu().numpy()
            f.write(f"Mean curvature range: [{mean_curv_volume.min():.6f}, {mean_curv_volume.max():.6f}]\n")
            f.write(f"Gaussian curvature range: [{gaussian_curv_volume.min():.6f}, {gaussian_curv_volume.max():.6f}]\n")
        f.write("\n")
        f.write("Files generated:\n")
        f.write("- gradient_magnitude.png: Edge detection visualization (middle slice)\n")
        f.write("- surface_normals.png: Surface orientation visualization (middle slice)\n")
        f.write("- divergence.png: Density distribution visualization (middle slice)\n")
        f.write("- gradient_magnitude_volume.tif: Full 3D edge detection volume\n")
        f.write("- divergence_volume.tif: Full 3D density distribution volume\n")
        if curvature_field.shape[0] > 0:
            f.write("- mean_curvature.png: Mean curvature visualization (subsampled middle slice)\n")
            f.write("- gaussian_curvature.png: Gaussian curvature visualization (subsampled middle slice)\n")
    
    print(f"Feature extraction complete. Results saved to {features_dir}")
    return features_dir