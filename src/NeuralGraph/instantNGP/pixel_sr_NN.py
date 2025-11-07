def plot_comparison_panel(target_hr, input_sr, bicubic_up, classical_sr, neural_sr, loss_history, psnr_history, epoch=None, dark_style=True):
    if dark_style:
        plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 10))
    # Panel 1: Original HR
    plt.subplot(2, 3, 1)
    plt.imshow(target_hr, cmap='gray')
    plt.title('Original High-Resolution')
    plt.axis('off')
    # Panel 2: Low-Resolution
    plt.subplot(2, 3, 2)
    plt.imshow(input_sr, cmap='gray')
    plt.title('Low-Resolution')
    plt.axis('off')
    # Panel 3: Bilinear Upsampling (upsample low-res by x4 and show PSNR)
    ax3 = plt.subplot(2, 3, 3)
    # Compute PSNR between bilinear upsample and target_hr
    try:
        # Ensure both arrays are float32 and same shape
        arr_hr = np.asarray(target_hr, dtype=np.float32)
        arr_bilinear = np.asarray(bicubic_up, dtype=np.float32)
        if arr_hr.shape == arr_bilinear.shape:
            psnr_bilinear = psnr(arr_hr, arr_bilinear)
            title_bilinear = f'Bilinear Upsampling\nPSNR: {psnr_bilinear:.2f} dB'
        else:
            title_bilinear = 'Bilinear Upsampling\nPSNR: N/A'
    except Exception:
        title_bilinear = 'Bilinear Upsampling\nPSNR: N/A'
    ax3.imshow(arr_bilinear, cmap='gray')
    ax3.set_title(title_bilinear)
    ax3.axis('off')
    # Panel 4: Classical Pixel SR
    plt.subplot(2, 3, 4)
    plt.imshow(classical_sr, cmap='gray')
    plt.title('Classical Pixel SR')
    plt.axis('off')
    # Panel 5: Neural SR
    plt.subplot(2, 3, 5)
    plt.imshow(neural_sr, cmap='gray')
    plt.title('Neural SR')
    plt.axis('off')
    # Panel 6: Loss and PSNR history
    ax = plt.subplot(2, 3, 6)
    loss_val = loss_history[-1] if len(loss_history) > 0 else None
    psnr_val = psnr_history[-1] if len(psnr_history) > 0 else None
    loss_label = f'Loss (final: {loss_val:.4f})' if loss_val is not None else 'Loss'
    psnr_label = f'PSNR (final: {psnr_val:.2f} dB)' if psnr_val is not None else 'PSNR'
    ax.plot(loss_history, label=loss_label)
    ax.plot(psnr_history, label=psnr_label)
    ax.set_title('Training History')
    ax.set_xlabel('Epoch')
    ax.legend()
    # Remove all ticks
    ax.set_xticks([])
    ax.set_yticks([])
    if epoch is not None:
        plt.suptitle(f'Epoch {epoch}')
    plt.tight_layout()
    return fig
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
        
        # Save 3x2 comparison panel for video every 20 epochs using the refactored function
        if (epoch + 1) % 20 == 0 or epoch == num_epochs - 1:
            with torch.no_grad():
                current_refined = model(input_sr)
            bilinear_up = torch.nn.functional.interpolate(
                input_sr, scale_factor=downscale_factor, mode='bilinear'
            ).squeeze().cpu().numpy()
            fig = plot_comparison_panel(
                target_hr.squeeze().cpu().numpy(),
                input_sr.squeeze().cpu().numpy(),
                bilinear_up,
                input_sr.squeeze().cpu().numpy(),  # classical_sr placeholder
                current_refined.squeeze().cpu().numpy(),
                loss_history,
                psnr_history,
                epoch=epoch+1,
                dark_style=True
            )
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
    
    # Return the final refined output, video frames, and histories
    return refined.squeeze().cpu(), best_psnr, video_frames, loss_history, psnr_history


def main():
    output_dir = 'sr_outputs'
    os.makedirs(output_dir, exist_ok=True)
    # Parameters
    downscale_factor = 4  # Using 4x as you mentioned
    num_shifts = 128      # Use 128 images for neural super-resolution
    # Load the boat image
    hr_image = tiff.imread('pics_boat.tif')
    hr_image = torch.tensor(hr_image, dtype=torch.float32)
    hr_image = hr_image / hr_image.max()
    # Generate low-resolution shifted images
    lr_images, shifts = generate_shifted_downsampled_images(hr_image, downscale_factor, num_shifts)

    # Apply classical pixel super-resolution
    print("Running classical SR algorithm...")
    classical_sr_result = classical_pixel_sr(lr_images, shifts, downscale_factor)
    # ...existing code before training...

    # Apply neural refinement
    print("Applying neural network refinement...")
    neural_sr_result, neural_psnr, video_frames, loss_history, psnr_history = neural_sr_refinement(
        classical_sr_result, hr_image, downscale_factor
    )

    # Save training progress video (pixel_NSTM.py strategy)
    video_path = os.path.join(output_dir, 'neural_sr_training_progress.mp4')
    video_frames_uint8 = []
    for frame in video_frames:
        # If frame is RGBA, drop alpha
        if frame.ndim == 3 and frame.shape[-1] == 4:
            frame = frame[..., :3]
        # If grayscale, convert to RGB
        if frame.ndim == 2:
            frame = np.stack([frame]*3, axis=-1)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        video_frames_uint8.append(frame)
    if len(video_frames_uint8) == 0:
        print("Warning: No frames to save in video!")
    else:
        print(f"First frame shape: {video_frames_uint8[0].shape}, dtype: {video_frames_uint8[0].dtype}")
        print(f"Saving video to: {video_path}")
        import imageio.v2 as imageio
        imageio.mimsave(video_path, video_frames_uint8, fps=10)
    """Main function to run the Neural SR experiment"""
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
        mode='bilinear'
    ).squeeze()
    
    # Apply classical pixel super-resolution
    print("Running classical SR algorithm...")
    classical_sr_result = classical_pixel_sr(lr_images, shifts, downscale_factor)
    
    # Single training run with double epochs
    print("Applying neural network refinement...")
    neural_sr_result, neural_psnr, video_frames, loss_history, psnr_history = neural_sr_refinement(
        classical_sr_result, hr_image, downscale_factor
    )

    # Save training progress video (MP4)
    video_path = os.path.join(output_dir, 'neural_sr_training_progress.mp4')
    print(f"Saving training progress video to: {video_path}")
    import imageio
    # Convert frames to uint8 RGB for MP4
    video_frames_uint8 = []
    for frame in video_frames:
        # If frame is RGBA, drop alpha
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        # If grayscale, convert to RGB
        if frame.ndim == 2:
            frame = np.stack([frame]*3, axis=-1)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        video_frames_uint8.append(frame)
    import imageio
    imageio.mimsave(video_path, video_frames_uint8, fps=10)
    print(f"Progress video saved to '{video_path}'")
    
    # Calculate metrics
    data_range = 1.0  # Images are normalized to [0,1]
    psnr_bilinear = psnr(hr_image.numpy(), single_lr_upscaled.numpy())
    psnr_classical = psnr(hr_image.numpy(), classical_sr_result.numpy())
    ssim_bilinear = ssim(hr_image.numpy(), single_lr_upscaled.numpy(), data_range=data_range)
    ssim_classical = ssim(hr_image.numpy(), classical_sr_result.numpy(), data_range=data_range)
    ssim_neural = ssim(hr_image.numpy(), neural_sr_result.numpy(), data_range=data_range)
    
    # Print metrics
    print("\nMetrics vs. ground truth:")
    print(f"Bilinear upscaling - PSNR: {psnr_bilinear:.2f} dB, SSIM: {ssim_bilinear:.4f}")
    print(f"Classical SR - PSNR: {psnr_classical:.2f} dB, SSIM: {ssim_classical:.4f}")
    print(f"Neural SR - PSNR: {neural_psnr:.2f} dB, SSIM: {ssim_neural:.4f}")
    print(f"Improvement over Classical - PSNR: {neural_psnr - psnr_classical:.2f} dB, SSIM: {ssim_neural - ssim_classical:.4f}")
    
    # Create visualization images - Extended comparison using the refactored function
    bilinear_up = torch.nn.functional.interpolate(
        single_lr.unsqueeze(0).unsqueeze(0), scale_factor=downscale_factor, mode='bilinear'
    ).squeeze().cpu().numpy()
    fig = plot_comparison_panel(
        hr_image.cpu().numpy(),
        single_lr.cpu().numpy(),
        bilinear_up,
        classical_sr_result.cpu().numpy(),
        neural_sr_result.cpu().numpy(),
        loss_history,
        psnr_history,
        epoch=None,
        dark_style=True
    )
    fig.savefig(f'{output_dir}/neural_sr_comparison.png', dpi=300)
    plt.close(fig)
    
    # Save zoomed comparison of a detail region using the same template as sr_comparison
    region_y, region_x = 100, 150  # Adjust this based on your image
    region_size = 64

    # Crop detail regions for each method, ensuring low-res is shown and no black panel
    hr_detail = hr_image[region_y:region_y+region_size, region_x:region_x+region_size].cpu().numpy()
    # For low-res, crop from the original HR then downsample to match detail size
    lr_detail_full = hr_image[region_y:region_y+region_size, region_x:region_x+region_size].unsqueeze(0).unsqueeze(0)
    lr_detail = torch.nn.functional.interpolate(lr_detail_full, scale_factor=1/downscale_factor, mode='bilinear').squeeze().cpu().numpy()
    # For upsampled, classical, neural, crop as before
    bilinear_detail = single_lr_upscaled.cpu().numpy()[region_y:region_y+region_size, region_x:region_x+region_size]
    classical_detail = classical_sr_result.cpu().numpy()[region_y:region_y+region_size, region_x:region_x+region_size]
    neural_detail = neural_sr_result.cpu().numpy()[region_y:region_y+region_size, region_x:region_x+region_size]

    # Use the same panel logic for both PNG and MP4
    fig_detail = plot_comparison_panel(
        hr_detail,
        lr_detail,
        bilinear_detail,
        classical_detail,
        neural_detail,
        loss_history,
        psnr_history,
        epoch=None,
        dark_style=True
    )
    fig_detail.savefig(f'{output_dir}/neural_sr_detail_comparison.png', dpi=300)
    plt.close(fig_detail)
    
    
    # Save all results for further analysis
    tiff.imwrite(f'{output_dir}/neural_sr_result.tif', neural_sr_result.numpy())
    
    print(f"All comparison images saved to '{output_dir}'")


if __name__ == "__main__":
    main()