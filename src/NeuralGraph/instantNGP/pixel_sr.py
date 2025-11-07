import numpy as np
import torch
import matplotlib.pyplot as plt
import tifffile as tiff
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os
from tqdm import tqdm, trange

def generate_shifted_downsampled_images(hr_image, downscale_factor, num_shifts):
    """
    Generate a set of shifted and downsampled images from a high-resolution image.
    """
    h, w = hr_image.shape
    
    # Create storage for low-res images and shifts
    lr_images = []
    shifts = []
    
    # Generate subpixel shifts in a grid pattern for even coverage
    grid_size = int(np.ceil(np.sqrt(num_shifts)))
    # Increase the shift step size for wider coverage
    step = min(1.0, 1.5 / grid_size)
    
    shift_idx = 0
    for i in trange(grid_size, desc="Generating shifted images", ncols=80):
        for j in range(grid_size):
            if shift_idx >= num_shifts:
                break
                
            # Subpixel shift in range [0, 1) for both x and y, with increased step
            shift_y = (i * step) % 1.0
            shift_x = (j * step) % 1.0
            shifts.append([shift_y, shift_x])
            
            # Apply shift and downsample
            grid_y, grid_x = torch.meshgrid(
                torch.arange(0, h, downscale_factor) + shift_y,
                torch.arange(0, w, downscale_factor) + shift_x,
                indexing='ij'
            )
            
            # Ensure we stay within image boundaries
            grid_y = torch.clamp(grid_y, 0, h-1.001)
            grid_x = torch.clamp(grid_x, 0, w-1.001)
            
            # Interpolate values at shifted positions
            y0, x0 = grid_y.floor().long(), grid_x.floor().long()
            y1, x1 = y0 + 1, x0 + 1
            
            # Ensure we don't go out of bounds
            y1 = torch.clamp(y1, 0, h-1)
            x1 = torch.clamp(x1, 0, w-1)
            
            wy = grid_y - y0.float()
            wx = grid_x - x0.float()
            
            # Bilinear interpolation formula
            val = (hr_image[y0, x0] * (1-wy) * (1-wx) + 
                   hr_image[y0, x1] * (1-wy) * wx + 
                   hr_image[y1, x0] * wy * (1-wx) + 
                   hr_image[y1, x1] * wy * wx)
            
            lr_images.append(val)
            shift_idx += 1
    
    return lr_images, shifts[:num_shifts]

def classical_pixel_sr(lr_images, shifts, downscale_factor):
    """
    Perform classical pixel super-resolution on a set of low-resolution images.
    """
    # Determine the high-resolution size
    lr_h, lr_w = lr_images[0].shape
    hr_h, hr_w = lr_h * downscale_factor, lr_w * downscale_factor
    
    # Initialize high-res grid with zeros
    hr_image = torch.zeros((hr_h, hr_w), dtype=torch.float32)
    contribution_count = torch.zeros((hr_h, hr_w), dtype=torch.float32)
    
    # For each shifted low-res image
    for idx, (lr_img, shift) in enumerate(tqdm(zip(lr_images, shifts), total=len(lr_images), 
                                              desc="Processing images", ncols=80)):
        shift_y, shift_x = shift
        
        # Calculate the starting position in the high-res grid
        start_y = int(round(shift_y * downscale_factor))
        start_x = int(round(shift_x * downscale_factor))
        
        # Place the values in the high-res grid with the right stride
        for i in range(lr_h):
            for j in range(lr_w):
                hr_y = (start_y + i * downscale_factor) % hr_h
                hr_x = (start_x + j * downscale_factor) % hr_w
                
                hr_image[hr_y, hr_x] += lr_img[i, j]
                contribution_count[hr_y, hr_x] += 1
    
    # Average values where we have multiple contributions
    valid_mask = contribution_count > 0
    hr_image[valid_mask] /= contribution_count[valid_mask]
    
    # Simple interpolation for missing pixels
    if not torch.all(valid_mask):
        missing_mask = ~valid_mask
        
        # Use a simple iterative diffusion to fill missing values
        for iter_idx in trange(20, desc="Filling missing values", ncols=80):
            # Create a new tensor with current HR image
            new_hr = hr_image.clone()
            
            # For each missing pixel, average surrounding valid pixels
            kernel = torch.ones((3, 3), dtype=torch.float32)
            kernel[1, 1] = 0  # Don't include the center pixel
            
            # Apply convolution to get weighted sum of neighbors
            from torch.nn.functional import conv2d
            
            # Add padding and reshape for conv2d
            padded = torch.nn.functional.pad(hr_image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate')
            neighborhood_sum = conv2d(padded, kernel.unsqueeze(0).unsqueeze(0), padding=0)
            
            # Also count valid neighbors with another convolution
            valid_padded = torch.nn.functional.pad(valid_mask.float().unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='constant')
            valid_count = conv2d(valid_padded, kernel.unsqueeze(0).unsqueeze(0), padding=0)
            
            # Compute average of valid neighbors where we have some
            has_valid_neighbors = valid_count.squeeze() > 0
            missing_with_neighbors = missing_mask & has_valid_neighbors.bool()
            
            # Update missing pixels with average of valid neighbors
            if torch.any(missing_with_neighbors):
                new_hr[missing_with_neighbors] = (neighborhood_sum.squeeze()[missing_with_neighbors] / 
                                                  valid_count.squeeze()[missing_with_neighbors])
            
            # Update image and valid mask
            hr_image = new_hr
            newly_valid = missing_with_neighbors
            valid_mask = valid_mask | newly_valid
            missing_mask = ~valid_mask
            
            # Stop if we've filled all missing pixels
            if not torch.any(missing_mask):
                break
    
    return hr_image

def main():
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sr_outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the boat image
    # Load the boat image from the current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    hr_image_path = os.path.join(script_dir, 'pics_boat.tif')
    hr_image = tiff.imread(hr_image_path)
    hr_image = torch.tensor(hr_image, dtype=torch.float32)
    
    # Normalize to [0, 1] for easier processing
    hr_image = hr_image / hr_image.max()
    
    # Parameters
    downscale_factor = 4
    num_shifts = 256
    
    print(f"High-resolution image shape: {hr_image.shape}")
    print(f"Downscale factor: {downscale_factor}")
    print(f"Number of shifted images: {num_shifts}")
    
    # Generate low-resolution shifted images
    lr_images, shifts = generate_shifted_downsampled_images(hr_image, downscale_factor, num_shifts)
    
    # Take a single downsampled image for comparison
    single_lr = lr_images[0]
    single_lr_upscaled = torch.nn.functional.interpolate(
        single_lr.unsqueeze(0).unsqueeze(0), 
        scale_factor=downscale_factor, 
        mode='bilinear'
    ).squeeze()
    
    # Apply classical pixel super-resolution
    sr_result = classical_pixel_sr(lr_images, shifts, downscale_factor)
    
    # Calculate metrics
    data_range = 1.0  # Images are normalized to [0,1]
    psnr_bilinear = psnr(hr_image.numpy(), single_lr_upscaled.numpy())
    psnr_sr = psnr(hr_image.numpy(), sr_result.numpy())
    ssim_bilinear = ssim(hr_image.numpy(), single_lr_upscaled.numpy(), data_range=data_range)
    ssim_sr = ssim(hr_image.numpy(), sr_result.numpy(), data_range=data_range)
    
    # Print metrics
    print("Metrics vs. ground truth:")
    print(f"Single LR upscaled - PSNR: {psnr_bilinear:.2f} dB, SSIM: {ssim_bilinear:.4f}")
    print(f"SR result - PSNR: {psnr_sr:.2f} dB, SSIM: {ssim_sr:.4f}")
    print(f"Improvement - PSNR: {psnr_sr - psnr_bilinear:.2f} dB, SSIM: {ssim_sr - ssim_bilinear:.4f}")
    
    # Create visualization images - Just the comparison figure
    plt.figure(figsize=(15, 10))

    plt.suptitle(f'Pixel Super-Resolution Comparison\nDownsampling factor: {downscale_factor}, Number of images: {num_shifts}', fontsize=16)

    plt.subplot(2, 2, 1)
    plt.imshow(hr_image, cmap='gray')
    plt.title('Original High-Resolution Image')

    plt.subplot(2, 2, 2)
    plt.imshow(single_lr, cmap='gray')
    plt.title('Single Low-Resolution Image')

    plt.subplot(2, 2, 3)
    plt.imshow(single_lr_upscaled, cmap='gray')
    plt.title(f'Bilinear Upsampling\nPSNR: {psnr_bilinear:.2f} dB')

    plt.subplot(2, 2, 4)
    plt.imshow(sr_result, cmap='gray')
    plt.title(f'Pixel Super-Resolution Result\nPSNR: {psnr_sr:.2f} dB')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{output_dir}/pixel_sr_comparison.png', dpi=300)
    
    print(f"Comparison image saved to '{output_dir}/pixel_sr_comparison.png'")

if __name__ == "__main__":
    main()