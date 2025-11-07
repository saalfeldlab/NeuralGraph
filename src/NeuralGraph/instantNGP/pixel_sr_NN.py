import numpy as np
import torch
import matplotlib.pyplot as plt
import tifffile as tiff
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os
from tqdm import tqdm, trange

# Re-use your existing pixel_sr.py functions for the classical approach
# Just import them or include them here
from pixel_sr import generate_shifted_downsampled_images, classical_pixel_sr

class SRRefineNet(torch.nn.Module):
    """Neural network for refining the classical SR result"""
    def __init__(self, channels=64):
        super(SRRefineNet, self).__init__()
        # Input refinement
        self.conv_in = torch.nn.Conv2d(1, channels, kernel_size=3, padding=1)
        
        # Residual blocks
        self.res_blocks = torch.nn.ModuleList([
            ResidualBlock(channels) for _ in range(5)
        ])
        
        # Global skip connection - important to preserve original details
        self.global_skip = torch.nn.Sequential(
            torch.nn.Conv2d(1, channels, kernel_size=1),
            torch.nn.Conv2d(channels, channels, kernel_size=1),
        )
        
        # Output projection
        self.conv_out = torch.nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        
        # Edge enhancement
        self.edge_enhance = torch.nn.Sequential(
            torch.nn.Conv2d(1, channels, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        # Store input for skip connection
        residual = x
        
        # Feature extraction
        feat = torch.nn.functional.relu(self.conv_in(x))
        
        # Residual blocks
        for res_block in self.res_blocks:
            feat = res_block(feat)
        
        # Global skip connection
        global_res = self.global_skip(residual)
        
        # Add global residual
        feat = feat + global_res
        
        # Output projection
        out = self.conv_out(feat)
        
        # Edge enhancement
        edge = self.edge_enhance(residual)
        
        # Add edge details to output
        return out + edge * 0.1 + residual


class ResidualBlock(torch.nn.Module):
    """Residual block with two convolutions and a skip connection"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual
        return self.relu(out)


def neural_sr_refinement(classical_sr_result, original_hr, downscale_factor=4, device="cuda"):
    """
    Refine classical SR result using a neural network.
    
    Args:
        classical_sr_result: Tensor from classical SR algorithm
        original_hr: Original high-res image (for supervised training and evaluation)
        downscale_factor: The downsampling factor
        device: Device to use for computation
    
    Returns:
        Refined SR image
    """
    if device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Prepare input data
    classical_sr_result = classical_sr_result.to(device)
    original_hr = original_hr.to(device)
    
    # Add batch and channel dimensions
    input_sr = classical_sr_result.unsqueeze(0).unsqueeze(0)
    target_hr = original_hr.unsqueeze(0).unsqueeze(0)
    
    # Create model
    model = SRRefineNet().to(device)
    
    # Optimizer with learning rate scheduling
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50
    )
    
    # Loss history for plotting
    loss_history = []
    psnr_history = []
    best_psnr = 0
    best_model = None
    
    # Create downsampled versions for cycle consistency
    target_lr = torch.nn.functional.interpolate(
        target_hr, scale_factor=1/downscale_factor, mode='bicubic'
    )
    
    # Training loop with progress bar
    num_epochs = 2000
    video_frames = []
    for epoch in trange(num_epochs, desc="Training neural refinement", ncols=100):
        # Forward pass
        refined = model(input_sr)
        
        # Create a downsampled version of our refined output
        refined_lr = torch.nn.functional.interpolate(
            refined, scale_factor=1/downscale_factor, mode='bicubic'
        )
        
        # Calculate losses:
        # 1. Supervised loss (if we have original HR)
        l_supervised = torch.nn.functional.l1_loss(refined, target_hr)
        
        # 2. Cycle consistency loss
        l_cycle = torch.nn.functional.l1_loss(refined_lr, target_lr)
        
        # 3. Edge preservation loss - using gradients
        refined_grad_x = torch.abs(refined[:,:,1:,:] - refined[:,:,:-1,:])
        refined_grad_y = torch.abs(refined[:,:,:,1:] - refined[:,:,:,:-1])
        target_grad_x = torch.abs(target_hr[:,:,1:,:] - target_hr[:,:,:-1,:])
        target_grad_y = torch.abs(target_hr[:,:,:,1:] - target_hr[:,:,:,:-1])
        
        l_edge = torch.nn.functional.l1_loss(refined_grad_x, target_grad_x) + \
                torch.nn.functional.l1_loss(refined_grad_y, target_grad_y)
        
        # Combined loss with weights
        loss = l_supervised + 0.1 * l_cycle + 0.5 * l_edge
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update scheduler
        scheduler.step(loss)
        
        # Calculate PSNR
        with torch.no_grad():
            current_psnr = psnr(
                target_hr.squeeze().cpu().numpy(),
                refined.squeeze().cpu().numpy()
            )
            
            # Save best model
            if current_psnr > best_psnr:
                best_psnr = current_psnr
                best_model = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'psnr': best_psnr
                }
        
        # Record metrics
        loss_history.append(loss.item())
        psnr_history.append(current_psnr)
        
        # Save frame for video every 20 epochs
        if (epoch + 1) % 20 == 0 or epoch == num_epochs - 1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(loss_history)
            ax.set_title(f'Epoch {epoch+1} Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            fig.tight_layout()
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            video_frames.append(frame)
            plt.close(fig)
        if (epoch + 1) % 50 == 0:
            tqdm.write(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, PSNR: {current_psnr:.2f} dB")
    
    # Load best model
    model.load_state_dict(best_model['state_dict'])
    
    # Generate final result
    with torch.no_grad():
        refined = model(input_sr)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(psnr_history)
    plt.title('PSNR (dB)')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    
    # Create output directory if it doesn't exist
    output_dir = 'sr_outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_history.png')
    
    # Return the final refined output and video frames
    return refined.squeeze().cpu(), best_psnr, video_frames


def main():
    """Main function to run the neural SR refinement experiment"""
    # Create output directory
    output_dir = 'sr_outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the boat image
    hr_image = tiff.imread('pics_boat.tif')
    hr_image = torch.tensor(hr_image, dtype=torch.float32)
    
    # Normalize to [0, 1] for easier processing
    hr_image = hr_image / hr_image.max()
    
    # Parameters
    downscale_factor = 4  # Using 4x as you mentioned
    num_shifts = 128      # Use 128 images for neural super-resolution
    
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
        mode='bicubic'
    ).squeeze()
    
    # Apply classical pixel super-resolution
    print("Running classical SR algorithm...")
    classical_sr_result = classical_pixel_sr(lr_images, shifts, downscale_factor)
    
    # First training run
    print("Applying neural network refinement (Run 1)...")
    neural_sr_result1, neural_psnr1, video_frames1 = neural_sr_refinement(
        classical_sr_result, hr_image, downscale_factor
    )

    # Second training run (start from previous result)
    print("Applying neural network refinement (Run 2)...")
    neural_sr_result2, neural_psnr2, video_frames2 = neural_sr_refinement(
        neural_sr_result1, hr_image, downscale_factor
    )

    # Combine video frames from both runs
    all_video_frames = video_frames1 + video_frames2
    video_path = os.path.join(output_dir, 'neural_sr_training_progress.mp4')
    print(f"Saving training progress video to: {video_path}")
    import imageio
    imageio.mimsave(video_path, all_video_frames, fps=10)
    print(f"Progress video saved to '{video_path}'")
    
    # Calculate metrics
    data_range = 1.0  # Images are normalized to [0,1]
    psnr_bicubic = psnr(hr_image.numpy(), single_lr_upscaled.numpy())
    psnr_classical = psnr(hr_image.numpy(), classical_sr_result.numpy())
    ssim_bicubic = ssim(hr_image.numpy(), single_lr_upscaled.numpy(), data_range=data_range)
    ssim_classical = ssim(hr_image.numpy(), classical_sr_result.numpy(), data_range=data_range)
    ssim_neural = ssim(hr_image.numpy(), neural_sr_result2.numpy(), data_range=data_range)
    
    # Print metrics
    print("\nMetrics vs. ground truth:")
    print(f"Bicubic upscaling - PSNR: {psnr_bicubic:.2f} dB, SSIM: {ssim_bicubic:.4f}")
    print(f"Classical SR - PSNR: {psnr_classical:.2f} dB, SSIM: {ssim_classical:.4f}")
    print(f"Neural SR - PSNR: {neural_psnr2:.2f} dB, SSIM: {ssim_neural:.4f}")
    print(f"Improvement over Classical - PSNR: {neural_psnr2 - psnr_classical:.2f} dB, SSIM: {ssim_neural - ssim_classical:.4f}")
    
    # Create visualization images - Extended comparison
    plt.figure(figsize=(15, 15))
    
    # First row
    plt.subplot(3, 2, 1)
    plt.imshow(hr_image, cmap='gray')
    plt.title('Original High-Resolution Image')
    
    plt.subplot(3, 2, 2)
    plt.imshow(single_lr, cmap='gray')  # Added downsampled image
    plt.title(f'Low-Resolution Image (1/{downscale_factor}x)')
    
    # Second row
    plt.subplot(3, 2, 3)
    plt.imshow(single_lr_upscaled, cmap='gray')
    plt.title(f'Bicubic Upsampling\nPSNR: {psnr_bicubic:.2f} dB')
    
    plt.subplot(3, 2, 4)
    plt.imshow(classical_sr_result, cmap='gray')
    plt.title(f'Classical Pixel SR\nPSNR: {psnr_classical:.2f} dB')
    
    # Third row
    plt.subplot(3, 2, 5)
    plt.imshow(neural_sr_result2, cmap='gray')
    plt.title(f'Neural SR Refinement\nPSNR: {neural_psnr2:.2f} dB')
    
    # Error map for neural SR
    plt.subplot(3, 2, 6)
    error_map = np.abs(hr_image.numpy() - neural_sr_result2.numpy())
    plt.imshow(error_map, cmap='hot', vmax=0.1)
    plt.colorbar()
    plt.title('Neural SR Error Map')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/neural_sr_comparison.png', dpi=300)
    
    # Save zoomed comparison of a detail region
    # Find a good region with details (e.g., around the mast of the boat)
    region_y, region_x = 100, 150  # Adjust this based on your image
    region_size = 64
    
    plt.figure(figsize=(15, 15))
    
    # First row - originals
    plt.subplot(3, 2, 1)
    plt.imshow(hr_image, cmap='gray')
    plt.title('Original HR (Full Image)')
    plt.gca().add_patch(plt.Rectangle((region_x, region_y), region_size, region_size, 
                                     edgecolor='red', facecolor='none', linewidth=2))
    
    plt.subplot(3, 2, 2)
    plt.imshow(hr_image[region_y:region_y+region_size, region_x:region_x+region_size], cmap='gray')
    plt.title('Original HR (Detail)')
    
    # Second row - comparison methods
    plt.subplot(3, 2, 3)
    plt.imshow(single_lr_upscaled[region_y:region_y+region_size, region_x:region_x+region_size], cmap='gray')
    plt.title(f'Bicubic Upscaling (Detail)\nPSNR: {psnr_bicubic:.2f} dB')
    
    plt.subplot(3, 2, 4)
    plt.imshow(classical_sr_result[region_y:region_y+region_size, region_x:region_x+region_size], cmap='gray')
    plt.title(f'Classical SR (Detail)\nPSNR: {psnr_classical:.2f} dB')
    
    # Third row - our method and error map
    plt.subplot(3, 2, 5)
    plt.imshow(neural_sr_result2[region_y:region_y+region_size, region_x:region_x+region_size], cmap='gray')
    plt.title(f'Neural SR (Detail)\nPSNR: {neural_psnr2:.2f} dB')
    
    plt.subplot(3, 2, 6)
    error_detail = np.abs(hr_image[region_y:region_y+region_size, region_x:region_x+region_size].numpy() - 
                         neural_sr_result2[region_y:region_y+region_size, region_x:region_x+region_size].numpy())
    plt.imshow(error_detail, cmap='hot', vmax=0.1)
    plt.colorbar()
    plt.title('Neural SR Error (Detail)')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/neural_sr_detail_comparison.png', dpi=300)
    
    # Create horizontal strip comparison
    strip_y = 128  # Adjust this for a good horizontal slice
    strip_height = 32
    
    plt.figure(figsize=(16, 10))
    
    # Original HR
    plt.subplot(5, 1, 1)
    plt.imshow(hr_image[strip_y:strip_y+strip_height, :], cmap='gray')
    plt.title('Original HR')
    plt.axis('off')
    
    # Low-res upscaled
    plt.subplot(5, 1, 2)
    plt.imshow(single_lr_upscaled[strip_y:strip_y+strip_height, :], cmap='gray')
    plt.title(f'Bicubic Upsampling (PSNR: {psnr_bicubic:.2f} dB)')
    plt.axis('off')
    
    # Classical SR
    plt.subplot(5, 1, 3)
    plt.imshow(classical_sr_result[strip_y:strip_y+strip_height, :], cmap='gray')
    plt.title(f'Classical SR (PSNR: {psnr_classical:.2f} dB)')
    plt.axis('off')
    
    # Neural SR
    plt.subplot(5, 1, 4)
    plt.imshow(neural_sr_result2[strip_y:strip_y+strip_height, :], cmap='gray')
    plt.title(f'Neural SR (PSNR: {neural_psnr2:.2f} dB)')
    plt.axis('off')
    
    # Error map
    plt.subplot(5, 1, 5)
    strip_error = np.abs(hr_image[strip_y:strip_y+strip_height, :].numpy() - 
                        neural_sr_result2[strip_y:strip_y+strip_height, :].numpy())
    plt.imshow(strip_error, cmap='hot', vmax=0.1)
    plt.colorbar()
    plt.title('Neural SR Error Map')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/neural_sr_strip_comparison.png', dpi=300)
    
    # Save all results for further analysis
    tiff.imwrite(f'{output_dir}/neural_sr_result.tif', neural_sr_result2.numpy())
    
    print(f"All comparison images saved to '{output_dir}'")

if __name__ == "__main__":
    main()