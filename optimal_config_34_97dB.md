# Optimal Configuration for 34.97 dB PSNR Result

This document contains the complete configuration that achieved **34.97 dB PSNR** with kidney_90.tif using InstantNGP.

## Key Breakthrough: Learning Rate Optimization
- **Critical Setting**: Learning rate = **1e-4** (down from 0.01)
- This was the primary factor enabling high PSNR performance

## Volume Dataset
- **File**: kidney_90.tif
- **Dimensions**: 90×1024×1024 voxels
- **Aspect Ratio**: z:y:x = 0.088:1:1 (highly anisotropic)
- **Total Voxels**: 94,371,840

## Neural Network Configuration (config_hash_3d.json)

```json
{
	"loss": {
		"otype": "RelativeL2"
	},
	"optimizer": {
		"otype": "Adam",
		"learning_rate": 1e-4,
		"beta1": 0.9,
		"beta2": 0.999,
		"epsilon": 1e-15,
		"l2_reg": 1e-6
	},
	"encoding": {
		"otype": "HashGrid",
		"n_levels": 22,
		"n_features_per_level": 2,
		"log2_hashmap_size": 21,
		"base_resolution": 16,
		"per_level_scale": 1.5,
		"fixed_point_pos": false
	},
	"network": {
		"otype": "FullyFusedMLP",
		"activation": "ReLU",
		"output_activation": "Sigmoid",
		"n_neurons": 64,
		"n_hidden_layers": 3
	}
}
```

## Coordinate Handling: Anisotropic Scaling

For highly anisotropic volumes like kidney_90.tif, anisotropic coordinate scaling is beneficial:

```python
# Calculate anisotropic scaling factors based on volume dimensions
max_dim = max(depth, height, width)
scale_z = depth / max_dim
scale_y = height / max_dim  
scale_x = width / max_dim

# Create coordinate meshgrid with anisotropic scaling
z_coords = torch.linspace(0.5*scale_z/depth, scale_z-0.5*scale_z/depth, depth, device=device)
y_coords = torch.linspace(0.5*scale_y/height, scale_y-0.5*scale_y/height, height, device=device)
x_coords = torch.linspace(0.5*scale_x/width, scale_x-0.5*scale_x/width, width, device=device)

zv, yv, xv = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
xyz = torch.stack((zv.flatten(), yv.flatten(), xv.flatten())).t()
```

For kidney_90.tif, this results in scaling factors: z=0.088, y=1.000, x=1.000

## Volume Class with TracerWarning Fix

```python
class Volume(torch.nn.Module):
    def __init__(self, filename, device):
        super(Volume, self).__init__()
        self.data = read_volume(filename)
        self.shape = self.data.shape
        
        # Ensure consistent shape: (depth, height, width)
        if len(self.shape) == 3:
            if self.shape[2] < self.shape[0]:
                self.data = np.transpose(self.data, (2, 0, 1))
                self.shape = self.data.shape
                print(f"Reordered volume to: {self.shape} (depth, height, width)")
        
        self.data = torch.from_numpy(self.data).float().to(device)
        
        # Pre-compute scaling tensor to avoid TracerWarning
        depth, height, width = self.shape
        self.scale_tensor = torch.tensor([depth-1, height-1, width-1], device=device, dtype=torch.float32)

    def forward(self, coords):
        """Trilinear interpolation for 3D volume sampling"""
        with torch.no_grad():
            # coords is Nx3 with values in [0,1]
            # Scale to volume dimensions using pre-computed tensor
            depth, height, width = self.shape
            scaled_coords = coords * self.scale_tensor
            
            # ... rest of trilinear interpolation code ...
```

## Training Parameters
- **Batch Size**: 1,048,576 samples (2^20)
- **Optimizer**: Adam with lr=1e-4
- **Total Parameters**: 72,259,824 (72.3M)
- **Training Efficiency**: ~76.3%
- **Final PSNR**: 34.97 dB

## Performance Results Summary
- **kidney_90.tif** (90×1024×1024, anisotropic): **34.97 dB** ✅
- **kidney_512.tif** (288×512×512, less anisotropic): 
  - With anisotropic scaling: 14.61 dB
  - Without anisotropic scaling: 26.72 dB ✅

## Key Insights
1. **Learning Rate is Critical**: 1e-4 vs 0.01 makes the difference between success and failure
2. **Coordinate Handling Depends on Volume Geometry**: 
   - Highly anisotropic volumes benefit from anisotropic scaling
   - More balanced volumes work better with isotropic [0,1]³ coordinates
3. **TracerWarning Fix**: Pre-compute scaling tensors in __init__ to avoid tensor creation in traced functions

## Command to Reproduce 34.97 dB Result
```bash
cd /groups/saalfeld/home/allierc/Py/NeuralGraph/src/NeuralGraph/instantNGP
python instantngp_kidney.py kidney_90.tif
```

## Files Required
- `instantngp_kidney.py` (main script with anisotropic coordinate handling)
- `config_hash_3d.json` (network configuration with lr=1e-4)
- `kidney_90.tif` (test volume)