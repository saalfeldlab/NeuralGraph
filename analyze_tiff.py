import numpy as np
from tifffile import imread

# Read the saved TIFF
img = imread('./log/signal/signal_N11_2_1/activity/frame_000000.tif')

print(f"Image dtype: {img.dtype}")
print(f"Image shape: {img.shape}")
print(f"Image range: [{img.min():.2f}, {img.max():.2f}]")
print(f"Image mean: {img.mean():.2f}")
print(f"Image std: {img.std():.2f}")

# Sample some specific pixel values
print("\nSample pixel values:")
print(f"  Center pixel: {img[256, 256]:.2f}")
print(f"  Top-left corner: {img[0, 0]:.2f}")

# Check histogram of values
hist, bin_edges = np.histogram(img, bins=10)
print("\nHistogram (10 bins):")
for i in range(len(hist)):
    print(f"  [{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f}]: {hist[i]} pixels")

# Check unique values around 0, 30, 100
print("\nPixels near key values:")
print(f"  Pixels == 0: {np.sum(img == 0.0)}")
print(f"  Pixels in [0-5]: {np.sum((img >= 0) & (img <= 5))}")
print(f"  Pixels in [25-35]: {np.sum((img >= 25) & (img <= 35))}")
print(f"  Pixels in [95-100]: {np.sum((img >= 95) & (img <= 100))}")

# Now check the actual raw activity data
print("\n" + "="*50)
print("Raw neuron activity data:")
x = np.load('graphs_data/signal/x_list_0.npy')
print(f"Data shape: {x.shape}")
print(f"Activity column (index 6) for frame 0 range: [{x[0, :, 6].min():.2f}, {x[0, :, 6].max():.2f}]")
print(f"Activity values for first frame (n_neurons={x.shape[1]}):")
print(f"  Min: {x[0, :, 6].min():.2f}")
print(f"  Max: {x[0, :, 6].max():.2f}")
print(f"  Mean: {x[0, :, 6].mean():.2f}")
print(f"  Sample activities (first 10 neurons): {x[0, :10, 6]}")
