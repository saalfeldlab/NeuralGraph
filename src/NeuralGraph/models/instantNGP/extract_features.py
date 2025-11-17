#!/usr/bin/env python3
"""
Standalone feature extraction script for trained InstantNGP models
Usage: python extract_features.py [model_path] [--subsample_factor 10] [--output_dir path]
Default model_path: instantngp_outputs/trained_model.pth
"""

import argparse
import os
import torch
import tinycudann as tcnn
from downstream_tasks import run_feature_extraction

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

def create_coordinate_grid(depth, height, width, device):
    """Create 3D coordinate grid for the volume"""
    print(f"Creating coordinate grid: {depth}×{height}×{width}")
    
    # Create normalized coordinates [0, 1]
    z_coords = torch.linspace(0.5/depth, 1 - 0.5/depth, depth, device=device)
    y_coords = torch.linspace(0.5/height, 1 - 0.5/height, height, device=device)  
    x_coords = torch.linspace(0.5/width, 1 - 0.5/width, width, device=device)
    
    # Create meshgrid and flatten
    zv, yv, xv = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
    xyz = torch.stack([xv.flatten(), yv.flatten(), zv.flatten()], dim=1)
    
    return xyz

def main():
    # Get the script directory to make default path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model_path = os.path.join(script_dir, "instantngp_outputs", "trained_model.pth")
    
    parser = argparse.ArgumentParser(description="Extract features from trained InstantNGP model")
    parser.add_argument("model_path", nargs='?', default=default_model_path,
                       help=f"Path to trained model (.pth file) (default: {default_model_path})")
    parser.add_argument("--subsample_factor", type=float, default=1.0, 
                       help="Subsampling factor for feature extraction (default: 1.0)")
    parser.add_argument("--output_dir", default="", 
                       help="Output directory (default: same as model directory)")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        return
    
    # Load the trained model
    model, config, volume_shape, final_psnr = load_trained_model(args.model_path, device)
    depth, height, width = volume_shape
    
    # Create coordinate grid
    xyz = create_coordinate_grid(depth, height, width, device)
    
    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(args.model_path)
    
    # Create features subdirectory
    features_output_dir = os.path.join(output_dir, "features")
    os.makedirs(features_output_dir, exist_ok=True)
    

    
    # Run feature extraction (gradients only)
    try:
        features_dir = run_feature_extraction(
            model, xyz, depth, height, width, device, 
            features_output_dir, subsample_factor=args.subsample_factor,
            gradients_only=True
        )

    except Exception as e:
        print(f"❌ feature extraction failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()