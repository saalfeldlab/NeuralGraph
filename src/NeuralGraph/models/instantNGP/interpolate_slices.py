#!/usr/bin/env python3
"""
Interpolate slices from trained InstantNGP model
Creates rotational slices perpendicular to XY plane and saves as TIFF
Usage: python interpolate_slices.py [model_path] [--num_angles 36] [--slice_resolution 512]
"""

import argparse
import numpy as np
import os
import torch
import tinycudann as tcnn
import tifffile
from tqdm import trange
import math

def load_trained_model(model_path, device):
    """Load a trained InstantNGP model from checkpoint"""
    print(f"Loading trained model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    volume_shape = checkpoint['volume_shape']
    final_psnr = checkpoint['final_psnr']
    
    print(f"Model info: PSNR={final_psnr:.2f} dB, Volume shape={volume_shape}")
    
    # Recreate the model architecture
    model = tcnn.NetworkWithInputEncoding(
        n_input_dims=3,
        n_output_dims=1,
        encoding_config=config["encoding"],
        network_config=config["network"]
    ).to(device)
    
    # Load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config, volume_shape, final_psnr

def create_rotated_plane_coordinates(angle_deg, slice_resolution=512, plane_size=1.0):
    """
    Create coordinates for a plane perpendicular to XY, rotated around Z axis
    
    Args:
        angle_deg: Rotation angle in degrees around Z axis
        slice_resolution: Resolution for both Y and Z axes (square slice)
        plane_size: Size of the plane in normalized coordinates [0,1]
    
    Returns:
        coords: Tensor of shape (slice_resolution * slice_resolution, 3) with XYZ coordinates
    """
    # Create a grid in the YZ plane (perpendicular to X)
    y_range = np.linspace(-plane_size/2, plane_size/2, slice_resolution)
    z_range = np.linspace(-plane_size/2, plane_size/2, slice_resolution)
    
    # Create meshgrid
    Y, Z = np.meshgrid(y_range, z_range, indexing='ij')
    
    # Start with X=0 (middle of the volume), Y and Z varying
    X = np.zeros_like(Y)
    
    # Stack to get (N, 3) coordinates
    coords_yz = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    
    # Convert angle to radians
    angle_rad = math.radians(angle_deg)
    
    # Create rotation matrix around Z axis
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    # Rotation matrix around Z axis
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1]
    ])
    
    # Apply rotation
    coords_rotated = coords_yz @ rotation_matrix.T
    
    # Translate to center of volume [0,1] and ensure we're within bounds
    coords_rotated += 0.5
    coords_rotated = np.clip(coords_rotated, 0, 1)
    
    return torch.from_numpy(coords_rotated).float()

def interpolate_slice(model, coords, device, batch_size=2**16):
    """
    Interpolate values at given coordinates using the trained model
    
    Args:
        model: Trained InstantNGP model
        coords: Coordinates tensor (N, 3)
        device: CUDA/CPU device
        batch_size: Batch size for inference
    
    Returns:
        values: Interpolated values at the coordinates
    """
    total_coords = coords.shape[0]
    values = torch.zeros(total_coords, device=device, dtype=torch.half)
    coords = coords.to(device)
    
    num_batches = (total_coords + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_coords)
            batch_coords = coords[start_idx:end_idx]
            
            # Get model predictions
            batch_output = model(batch_coords).squeeze()
            values[start_idx:end_idx] = batch_output
    
    return values.cpu().numpy()

def create_rotational_slices(model, device, num_angles=36, slice_resolution=512,
                           plane_size=0.8, output_path="rotational_slices.tif"):
    """
    Create a series of rotated slices and save as TIFF stack
    
    Args:
        model: Trained InstantNGP model
        device: CUDA/CPU device
        num_angles: Number of rotation angles
        slice_resolution: Resolution for both Y and Z axes (square slice)
        plane_size: Size of the plane relative to volume
        output_path: Output TIFF file path
    """
    print(f"Creating {num_angles} rotational slices at {slice_resolution}x{slice_resolution} resolution")
    
    # Calculate angles for full rotation
    angles = np.linspace(0, 360, num_angles, endpoint=False)
    
    # Initialize stack to store all slices
    slices = np.zeros((num_angles, slice_resolution, slice_resolution), dtype=np.float32)
    
    for i, angle in enumerate(trange(len(angles), desc="Generating slices", ncols=150)):
        angle_deg = angles[i]
        
        # Create coordinates for this rotated plane
        coords = create_rotated_plane_coordinates(
            angle_deg, slice_resolution, plane_size
        )
        
        # Interpolate values at these coordinates
        values = interpolate_slice(model, coords, device)
        
        # Reshape to 2D slice
        slice_2d = values.reshape(slice_resolution, slice_resolution)
        slices[i] = slice_2d
        
        if i % 10 == 0:
            print(f"  Slice {i+1}/{num_angles}: angle={angle_deg:.1f}°, "
                  f"value range=[{slice_2d.min():.4f}, {slice_2d.max():.4f}]")
    
    # Calculate statistics for the entire stack
    stack_min = slices.min()
    stack_max = slices.max()
    stack_mean = slices.mean()
    stack_std = slices.std()
    
    print("\nStack statistics:")
    print(f"  Shape: {slices.shape}")
    print(f"  Value range: [{stack_min:.4f}, {stack_max:.4f}]")
    print(f"  Mean: {stack_mean:.4f}, Std: {stack_std:.4f}")
    
    # Normalize to uint16 range for better TIFF compatibility
    slices_normalized = ((slices - stack_min) / (stack_max - stack_min) * 65535).astype(np.uint16)
    
    # Save as TIFF stack
    print(f"\\nSaving TIFF stack to: {output_path}")
    tifffile.imwrite(output_path, slices_normalized, 
                     imagej=True,  # Enable ImageJ compatibility
                     metadata={'axes': 'ZYX',  # Time, Y, X for ImageJ
                              'unit': 'pixel',
                              'spacing': 1.0,
                              'loop': True})  # Enable looping in ImageJ
    
    print(f"✅ Successfully saved {num_angles} slices to {output_path}")
    print(f"   Each slice: {slice_resolution}x{slice_resolution} pixels (square)")
    print("   Data type: uint16 (0-65535 range)")
    print(f"   Total file size: ~{os.path.getsize(output_path) / (1024*1024):.1f} MB")
    
    return slices, output_path

def main():
    # Get the script directory to make default path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model_path = os.path.join(script_dir, "instantngp_outputs", "trained_model.pth")
    
    parser = argparse.ArgumentParser(description="Create rotational slices from trained InstantNGP model")
    parser.add_argument("model_path", nargs='?', default=default_model_path,
                       help=f"Path to trained model (.pth file) (default: {default_model_path})")
    parser.add_argument("--num_angles", type=int, default=180,
                       help="Number of rotation angles (default: 180)")
    parser.add_argument("--slice_resolution", type=int, default=512,
                       help="Resolution for both Y and Z axes (square slice) (default: 512)")
    parser.add_argument("--plane_size", type=float, default=0.8,
                       help="Size of the plane relative to volume (default: 0.8)")
    parser.add_argument("--output_path", default="kidney_rotation_hires_180.tif",
                       help="Output TIFF file path (default: kidney_rotation_hires_180.tif)")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model file not found: {args.model_path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script directory: {script_dir}")
        return
    
    # Load the trained model
    model, config, volume_shape, final_psnr = load_trained_model(args.model_path, device)
    
    # Create output directory if needed
    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(output_dir, exist_ok=True)
    
    # Create rotational slices
    try:
        slices, output_path = create_rotational_slices(
            model, device,
            num_angles=args.num_angles,
            slice_resolution=args.slice_resolution,
            plane_size=args.plane_size,
            output_path=args.output_path
        )
    
        
    except Exception as e:
        print(f"❌ Slice interpolation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()