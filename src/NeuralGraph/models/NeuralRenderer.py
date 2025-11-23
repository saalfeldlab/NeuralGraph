"""
NeuralRenderer.py - Differentiable Neural Renderer with Gaussian Splatting + MLP

NEW ARCHITECTURE:
- Gaussian Splatting: Outputs continuous field with spatial derivatives [f, ∂f/∂x, ∂f/∂y]
- MLP: Refines field to match matplotlib rendering style

Design:
1. Takes neuron positions (N, 2) and activities (N,) from SIREN
2. Gaussian splatting evaluates at query points (M, 2) → (M, 3) features
3. MLP processes features → (M, 1) refined activities (squared output)

Learnable Parameters:
- Affine transform (6 params): a, b, c, d, tx, ty
- Gaussian kernel (2 params): σ (threshold), β (sharpness)
- MLP weights (~4k params)

Author: Claude Code
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GaussianSplatting(nn.Module):
    """Continuous Gaussian splatting with sigmoid kernel and spatial derivatives

    Outputs field value and gradients at query points for MLP refinement.

    Learnable parameters:
    - Affine transform (6): a, b, c, d, tx, ty
    - Kernel params (2): σ (threshold distance), β (sharpness)
    """

    def __init__(self, resolution=512, sigma_init=0.02, beta_init=50.0):
        """
        Args:
            resolution: Default grid resolution (for grid generation)
            sigma_init: Initial σ (threshold distance in sigmoid kernel)
            beta_init: Initial β (sharpness parameter)
        """
        super().__init__()
        self.resolution = resolution
        self.eps = 1e-8  # For numerical stability in distance computation

        # Learnable affine transformation (centered at 0.5, 0.5)
        # Simplified: x' = ax * (x - 0.5) + tx + 0.5,  y' = ay * (y - 0.5) + ty + 0.5
        self.affine_ax = nn.Parameter(torch.tensor(0.9))
        self.affine_ay = nn.Parameter(torch.tensor(0.9))
        self.affine_tx = nn.Parameter(torch.tensor(0.0))
        self.affine_ty = nn.Parameter(torch.tensor(0.0))

        # Learnable kernel parameters
        self.sigma = nn.Parameter(torch.tensor(sigma_init))  # Threshold distance
        self.beta = nn.Parameter(torch.tensor(beta_init))    # Sharpness

    def apply_affine(self, positions):
        """Apply affine transformation to positions

        Args:
            positions: (N, 2) tensor in [0, 1]

        Returns:
            transformed: (N, 2) tensor
        """
        # Center around (0.5, 0.5)
        centered = positions - 0.5

        # Apply simplified transform (independent scaling per axis)
        x_centered = centered[:, 0]
        y_centered = centered[:, 1]

        x_transformed = self.affine_ax * x_centered + self.affine_tx
        y_transformed = self.affine_ay * y_centered + self.affine_ty

        # Translate back and stack
        transformed = torch.stack([
            x_transformed + 0.5,
            y_transformed + 0.5
        ], dim=1)

        return transformed

    def forward(self, positions, activities, query_points):
        """Evaluate Gaussian splatting field and gradients at query points

        Args:
            positions: (N, 2) neuron positions in [0, 1]
            activities: (N,) or (N, 1) neuron activities
            query_points: (M, 2) query coordinates in [0, 1]

        Returns:
            features: (M, 3) tensor with [field_value, ∂f/∂x, ∂f/∂y]
        """
        # Handle batch dimension in activities
        if activities.dim() == 2:
            activities = activities.squeeze(-1)  # (N, 1) -> (N,)

        N = positions.shape[0]
        M = query_points.shape[0]
        device = positions.device

        # Apply affine transformation to neuron positions
        positions_transformed = self.apply_affine(positions)

        # Initialize outputs
        field_values = torch.zeros(M, device=device, dtype=positions.dtype)
        grad_x = torch.zeros(M, device=device, dtype=positions.dtype)
        grad_y = torch.zeros(M, device=device, dtype=positions.dtype)

        # For each neuron, compute contribution to all query points
        for i in range(N):
            # Neuron position and activity
            x_i = positions_transformed[i, 0]  # scalar
            y_i = positions_transformed[i, 1]  # scalar
            a_i = activities[i]                 # scalar

            # Distance from neuron to all query points
            dx = query_points[:, 0] - x_i  # (M,)
            dy = query_points[:, 1] - y_i  # (M,)
            r = torch.sqrt(dx**2 + dy**2 + self.eps**2)  # (M,) with numerical stability

            # Sigmoid kernel: s(r) = sigmoid(-β × (r - σ))
            s = torch.sigmoid(-self.beta * (r - self.sigma))  # (M,)

            # Sigmoid derivative: s'(r) = -β × s × (1 - s)
            s_prime = -self.beta * s * (1.0 - s)  # (M,)

            # Field contribution
            field_values += a_i * s  # (M,)

            # Gradient contributions: ∂f/∂x = a_i × s'(r) × ∂r/∂x
            # ∂r/∂x = dx / r,  ∂r/∂y = dy / r
            grad_x += a_i * s_prime * (dx / r)  # (M,)
            grad_y += a_i * s_prime * (dy / r)  # (M,)

        # Stack into feature tensor
        features = torch.stack([field_values, grad_x, grad_y], dim=1)  # (M, 3)

        return features

    def get_learnable_params(self):
        """Return current values of learnable parameters"""
        return {
            'affine_ax': self.affine_ax.item(),
            'affine_ay': self.affine_ay.item(),
            'affine_tx': self.affine_tx.item(),
            'affine_ty': self.affine_ty.item(),
            'sigma': self.sigma.item(),
            'beta': self.beta.item()
        }


class MLPRefinement(nn.Module):
    """3-layer MLP for refining Gaussian splatting features

    Takes [field_value, ∂f/∂x, ∂f/∂y] and outputs refined activity.
    No output activation - can output any value (positive or negative).
    """

    def __init__(self, hidden_dim=64):
        """
        Args:
            hidden_dim: Width of hidden layers
        """
        super().__init__()

        self.fc1 = nn.Linear(3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        """
        Args:
            features: (M, 3) tensor with [field_value, ∂f/∂x, ∂f/∂y]

        Returns:
            output: (M,) tensor of refined activities
        """
        x = F.relu(self.fc1(features))  # (M, 64)
        x = F.relu(self.fc2(x))          # (M, 64)
        x = self.fc3(x)                  # (M, 1)

        # No output activation
        output = x.squeeze(-1)           # (M,)

        return output


class NeuralRenderer(nn.Module):
    """Complete neural renderer: Gaussian Splatting + MLP Refinement

    Architecture:
    1. Gaussian Splatting: positions, activities → field features [f, ∂f/∂x, ∂f/∂y]
    2. MLP: features → refined activity (no output activation)

    Can evaluate at arbitrary query points (continuous representation).
    For training, queries on regular grid to compare with matplotlib target.

    Example:
        renderer = NeuralRenderer(resolution=512)

        # Neuron data
        positions = torch.rand(100, 2)   # N=100 neurons
        activities = torch.rand(100)      # Activities from SIREN

        # Query on grid
        query_grid = create_grid(512)     # (512*512, 2)
        output = renderer(positions, activities, query_grid)  # (512*512,)

        # Reshape to image
        output_image = output.reshape(512, 512)
    """

    def __init__(self, resolution=512, sigma_init=0.02, beta_init=50.0, hidden_dim=64):
        """
        Args:
            resolution: Default grid resolution
            sigma_init: Initial σ for Gaussian splatting
            beta_init: Initial β for Gaussian splatting
            hidden_dim: MLP hidden layer width
        """
        super().__init__()

        self.resolution = resolution

        # Gaussian splatting with learnable affine + kernel params
        self.splatting = GaussianSplatting(
            resolution=resolution,
            sigma_init=sigma_init,
            beta_init=beta_init
        )

        # MLP refinement
        self.mlp = MLPRefinement(hidden_dim=hidden_dim)

    def forward(self, positions, activities, query_points=None):
        """
        Args:
            positions: (N, 2) neuron positions in [0, 1]
            activities: (N,) or (N, 1) neuron activities
            query_points: (M, 2) query coordinates, or None to use default grid

        Returns:
            output: (M,) refined activities at query points
        """
        # If no query points provided, use regular grid
        if query_points is None:
            query_points = self.create_grid(self.resolution, device=positions.device)

        # Step 1: Gaussian splatting → features
        features = self.splatting(positions, activities, query_points)  # (M, 3)

        # Step 2: MLP refinement → output
        output = self.mlp(features)  # (M,)

        return output

    def forward_splatting_only(self, positions, activities, query_points=None):
        """Get Gaussian splatting output WITHOUT MLP refinement

        Useful for visualization/comparison.

        Returns:
            field_values: (M,) - just the field values, no gradients
        """
        if query_points is None:
            query_points = self.create_grid(self.resolution, device=positions.device)

        features = self.splatting(positions, activities, query_points)  # (M, 3)
        field_values = features[:, 0]  # Extract just field values

        return field_values

    def create_grid(self, resolution, device='cuda'):
        """Create regular grid of query points

        Args:
            resolution: Grid size (square)
            device: torch device

        Returns:
            grid: (resolution^2, 2) tensor of (x, y) coordinates in [0, 1]
        """
        y = torch.linspace(0, 1, resolution, device=device)
        x = torch.linspace(0, 1, resolution, device=device)
        yv, xv = torch.meshgrid(y, x, indexing='ij')

        grid = torch.stack([xv.flatten(), yv.flatten()], dim=1)  # (H*W, 2)

        return grid

    def get_num_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_learnable_params(self):
        """Return current values of key learnable parameters"""
        splatting_params = self.splatting.get_learnable_params()

        return {
            **splatting_params,
            'total_params': self.get_num_parameters()
        }

