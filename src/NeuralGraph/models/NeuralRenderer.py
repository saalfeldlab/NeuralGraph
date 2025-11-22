"""
NeuralRenderer.py - Differentiable Neural Network for Rendering Sparse Points to Activity Values

Architecture: Gaussian Splatting + U-Net Refinement
- Handles variable number of input neurons (N can change per frame)
- Input: (N, 3) where each row is (x, y, activity)
- Output: Activity values that can be sampled at any (x, y) coordinate

NEW DESIGN: Instead of outputting a full rendered image, this outputs activity VALUES
that can be sampled at arbitrary (x,y) coordinates. This is compatible with NSTM training
where fixed_scene(x,y) × activity_value(x,y) is the correct mathematical operation.

Author: Claude Code
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianSplatting(nn.Module):
    """Differentiable Gaussian splatting layer with learnable position scale

    Converts sparse point data (N, 3) to activity values that can be sampled at (x,y).

    Single learnable parameter:
    - position_scale (init 0.9): scales positions toward center to prevent border artifacts

    Handles variable N - loops over however many points are provided.
    """

    def __init__(self, resolution=512, sigma=0.02):
        """
        Args:
            resolution: Output image resolution (square) - used for grid generation
            sigma: Base Gaussian kernel width in normalized coordinates [0, 1]
        """
        super().__init__()
        self.resolution = resolution
        self.base_sigma = sigma

        # Learnable position scale to bring dots closer to center
        # Scale positions around center (0.5, 0.5): pos_scaled = 0.5 + scale * (pos - 0.5)
        # scale=1.0 means no change, scale<1.0 brings dots closer to center
        self.position_scale = nn.Parameter(torch.tensor(0.9))

        # Pre-compute coordinate grid (normalized to [0, 1])
        y = torch.linspace(0, 1, resolution)
        x = torch.linspace(0, 1, resolution)
        yv, xv = torch.meshgrid(y, x, indexing='ij')

        # Register as buffer so it moves with model to device
        self.register_buffer('grid_x', xv)  # (H, W)
        self.register_buffer('grid_y', yv)  # (H, W)

    def forward(self, positions, activities):
        """
        Args:
            positions: (N, 2) tensor of (x, y) coordinates in [0, 1]
            activities: (N,) or (N, 1) tensor of activity values

        Returns:
            splatted_image: (H, W) tensor - sum of activity value contributions
        """
        # Handle batch dimension if present
        if activities.dim() == 2:
            activities = activities.squeeze(-1)  # (N, 1) -> (N,)

        N = positions.shape[0]
        H, W = self.resolution, self.resolution

        # Apply learnable position scaling around center (0.5, 0.5)
        # This brings positions closer to center if position_scale < 1.0
        positions_scaled = 0.5 + self.position_scale * (positions - 0.5)

        # Flip y-axis to match matplotlib coordinate system (y increases upward)
        # positions_scaled[:, 1] is y-coordinate, flip it: y_flipped = 1 - y
        positions_scaled = torch.stack([
            positions_scaled[:, 0],           # x stays the same
            1.0 - positions_scaled[:, 1]      # y flipped
        ], dim=1)

        # Initialize canvas
        canvas = torch.zeros(H, W, device=positions.device, dtype=positions.dtype)

        # Splat each neuron onto canvas
        for i in range(N):
            x_i = positions_scaled[i, 0]  # scalar
            y_i = positions_scaled[i, 1]  # scalar
            act_i = activities[i]  # scalar

            # Compute squared distance from this neuron to all pixels
            # grid_x, grid_y are (H, W)
            dist_sq = (self.grid_x - x_i) ** 2 + (self.grid_y - y_i) ** 2

            # Gaussian kernel: exp(-dist^2 / (2*sigma^2))
            gaussian = torch.exp(-dist_sq / (2 * self.base_sigma ** 2))

            # Weight by activity and accumulate
            canvas += act_i * gaussian

        return canvas  # (H, W)


class UNetRefinement(nn.Module):
    """U-Net for refining splatted images

    Takes rough Gaussian-splatted image and refines it to match
    target rendering style (e.g., matplotlib scatter plots).
    """

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # Encoder (downsampling path)
        self.enc1 = self.conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = self.conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder (upsampling path with skip connections)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)  # 512 from upconv + 512 from skip

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)  # 256 from upconv + 256 from skip

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)  # 128 from upconv + 128 from skip

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)  # 64 from upconv + 64 from skip

        # Final output layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """Two conv layers with ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) or (C, H, W) or (H, W) input tensor

        Returns:
            out: Same shape as input
        """
        # Handle different input shapes
        input_shape = x.shape
        needs_batch = x.dim() == 2  # (H, W)
        needs_channel = x.dim() == 3  # (C, H, W)

        if needs_batch:
            x = x.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
        elif needs_channel:
            x = x.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)

        # Encoder
        enc1 = self.enc1(x)  # 512x512x64
        enc2 = self.enc2(self.pool1(enc1))  # 256x256x128
        enc3 = self.enc3(self.pool2(enc2))  # 128x128x256
        enc4 = self.enc4(self.pool3(enc3))  # 64x64x512

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))  # 32x32x1024

        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)  # 64x64x512
        dec4 = torch.cat([dec4, enc4], dim=1)  # 64x64x1024
        dec4 = self.dec4(dec4)  # 64x64x512

        dec3 = self.upconv3(dec4)  # 128x128x256
        dec3 = torch.cat([dec3, enc3], dim=1)  # 128x128x512
        dec3 = self.dec3(dec3)  # 128x128x256

        dec2 = self.upconv2(dec3)  # 256x256x128
        dec2 = torch.cat([dec2, enc2], dim=1)  # 256x256x256
        dec2 = self.dec2(dec2)  # 256x256x128

        dec1 = self.upconv1(dec2)  # 512x512x64
        dec1 = torch.cat([dec1, enc1], dim=1)  # 512x512x128
        dec1 = self.dec1(dec1)  # 512x512x64

        # Output
        out = self.out_conv(dec1)  # 512x512x1

        # Restore original shape
        if needs_batch:
            out = out.squeeze(0).squeeze(0)  # (1, 1, H, W) -> (H, W)
        elif needs_channel:
            out = out.squeeze(0)  # (1, C, H, W) -> (C, H, W)

        return out


class NeuralRenderer(nn.Module):
    """Complete neural renderer: Gaussian Splatting + Optional U-Net Refinement

    Outputs activity VALUES (not rendered images) that can be sampled at (x,y) coordinates.
    This design is compatible with NSTM training where fixed_scene(x,y) × activity_value(x,y).

    Handles variable number of neurons per frame.

    Example:
        renderer = NeuralRenderer(resolution=512, sigma=0.02)

        # Frame 1: 100 neurons
        positions1 = torch.rand(100, 2)  # Random positions
        activities1 = torch.rand(100)     # Random activities
        activity_values1 = renderer(positions1, activities1)  # (512, 512) activity values

        # Frame 2: 75 neurons (different N!)
        positions2 = torch.rand(75, 2)
        activities2 = torch.rand(75)
        activity_values2 = renderer(positions2, activities2)  # (512, 512) activity values

        # For NSTM: sample at arbitrary (x,y) coordinates
        sampled = F.grid_sample(activity_values1.unsqueeze(0).unsqueeze(0), grid, ...)
    """

    def __init__(self, resolution=512, sigma=0.02, use_unet=True):
        """
        Args:
            resolution: Output image resolution (square)
            sigma: Base Gaussian kernel width in normalized coordinates
            use_unet: If True, use U-Net refinement. If False, use raw splatting only.
        """
        super().__init__()

        self.resolution = resolution
        self.use_unet = use_unet

        # Gaussian splatting layer with four learnable parameters
        self.splatting = GaussianSplatting(
            resolution=resolution,
            sigma=sigma
        )

        # U-Net refinement (optional)
        if use_unet:
            self.unet = UNetRefinement(in_channels=1, out_channels=1)
        else:
            self.unet = None

    def forward(self, positions, activities):
        """
        Args:
            positions: (N, 2) tensor of (x, y) coordinates in [0, 1]
                       N can vary between calls!
            activities: (N,) or (N, 1) tensor of activity values

        Returns:
            activity_values: (H, W) tensor - can be sampled at any (x,y)
        """
        # Step 1: Gaussian splatting with learnable parameters (variable N)
        splatted = self.splatting(positions, activities)  # (H, W)

        # Step 2: U-Net refinement (optional)
        if self.use_unet:
            # U-Net expects (B, C, H, W), handles conversion internally
            refined = self.unet(splatted)  # (H, W)
            return refined
        else:
            return splatted

    def get_num_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_learnable_params(self):
        """Return current value of the learnable position_scale parameter"""
        return {
            'position_scale': self.splatting.position_scale.item()
        }


# Utility functions for backward compatibility with matplotlib rendering

def render_activity_matplotlib(positions, activities, resolution=512,
                               vmin=0, vmax=20, device='cuda'):
    """Fallback: render using matplotlib scatter (original method)

    Args:
        positions: (N, 2) numpy array or torch tensor of xy positions in [-0.5, 0.5]
        activities: (N,) numpy array or torch tensor of activity values
        resolution: Output image resolution
        vmin, vmax: Activity value range for colormap
        device: torch device (for compatibility, matplotlib runs on CPU)

    Returns:
        activity_image: (resolution, resolution) numpy array (float32)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.ndimage import zoom

    # Convert to numpy if needed
    if isinstance(positions, torch.Tensor):
        positions = positions.cpu().numpy()
    if isinstance(activities, torch.Tensor):
        activities = activities.cpu().numpy()

    # Create figure
    fig = plt.figure(figsize=(resolution/80, resolution/80), dpi=80)
    plt.scatter(
        positions[:, 0],
        positions[:, 1],
        s=700,
        c=activities,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax
    )
    plt.axis("off")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    # Render to canvas and extract grayscale
    fig.canvas.draw()
    img_rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img_rgba = img_rgba.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img_rgba = img_rgba[:, :, :3]  # RGBA to RGB

    # Convert RGB to grayscale
    img_gray = np.dot(img_rgba[...,:3], [0.2989, 0.5870, 0.1140])

    # Resize to exactly desired resolution if needed
    if img_gray.shape != (resolution, resolution):
        zoom_factors = (resolution / img_gray.shape[0], resolution / img_gray.shape[1])
        img_gray = zoom(img_gray, zoom_factors, order=1)

    plt.close(fig)

    return img_gray.astype(np.float32)


if __name__ == "__main__":
    # Test the neural renderer
    print("testing NeuralRenderer...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create renderer
    renderer = NeuralRenderer(
        resolution=512,
        sigma=0.02,
        use_unet=True
    ).to(device)

    print(f"total parameters: {renderer.get_num_parameters():,}")
    print(f"learnable params: {renderer.get_learnable_params()}")

    # Test with variable N
    print("\ntest 1: 100 neurons")
    positions1 = torch.rand(100, 2, device=device)
    activities1 = torch.rand(100, device=device) * 20
    activity_values1 = renderer(positions1, activities1)
    print(f"  output shape: {activity_values1.shape}")
    print(f"  output range: [{activity_values1.min():.3f}, {activity_values1.max():.3f}]")

    print("\ntest 2: 50 neurons (different N!)")
    positions2 = torch.rand(50, 2, device=device)
    activities2 = torch.rand(50, device=device) * 20
    activity_values2 = renderer(positions2, activities2)
    print(f"  output shape: {activity_values2.shape}")
    print(f"  output range: [{activity_values2.min():.3f}, {activity_values2.max():.3f}]")

    print("\ntest 3: splatting only (no U-Net)")
    renderer_simple = NeuralRenderer(
        resolution=512,
        sigma=0.02,
        use_unet=False
    ).to(device)
    activity_values3 = renderer_simple(positions1, activities1)
    print(f"  output shape: {activity_values3.shape}")
    print(f"  output range: [{activity_values3.min():.3f}, {activity_values3.max():.3f}]")
    print(f"  learnable params: {renderer_simple.get_learnable_params()}")

    print("\nall tests passed!")
