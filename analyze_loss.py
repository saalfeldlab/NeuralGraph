import torch
import numpy as np
import sys

# Load the loss file
loss_file = "log/fly/fly_N9_62_5_15/loss.pt"
loss_data = torch.load(loss_file, map_location='cpu')

print("="*80)
print(f"📊 Loss Analysis for fly_N9_62_5_15")
print("="*80)

# Convert to numpy for easier analysis
if isinstance(loss_data, torch.Tensor):
    loss_np = loss_data.numpy()
elif isinstance(loss_data, list):
    loss_np = np.array(loss_data)
else:
    print(f"Loss data type: {type(loss_data)}")
    loss_np = np.array(loss_data)

print(f"\nTotal epochs: {len(loss_np)}")
print(f"\nLoss Statistics:")
print(f"  Min:    {loss_np.min():.6f}")
print(f"  Max:    {loss_np.max():.6f}")
print(f"  Mean:   {loss_np.mean():.6f}")
print(f"  Median: {np.median(loss_np):.6f}")
print(f"  Std:    {loss_np.std():.6f}")

# Check for negative values
negative_mask = loss_np < 0
n_negative = negative_mask.sum()
print(f"\n🔍 Negative Loss Analysis:")
print(f"  Number of negative values: {n_negative} / {len(loss_np)} ({100*n_negative/len(loss_np):.1f}%)")

if n_negative > 0:
    print(f"  First negative at epoch: {np.where(negative_mask)[0][0]}")
    print(f"  Most negative value: {loss_np[negative_mask].min():.6f}")
    print(f"\n  Epochs with negative loss:")
    negative_epochs = np.where(negative_mask)[0]
    for epoch in negative_epochs[:20]:  # Show first 20
        print(f"    Epoch {epoch}: {loss_np[epoch]:.6f}")
    if len(negative_epochs) > 20:
        print(f"    ... and {len(negative_epochs) - 20} more")

# Show first 10 and last 10 epochs
print(f"\n📈 First 10 epochs:")
for i in range(min(10, len(loss_np))):
    print(f"  Epoch {i}: {loss_np[i]:.6f}")

print(f"\n📈 Last 10 epochs:")
for i in range(max(0, len(loss_np)-10), len(loss_np)):
    print(f"  Epoch {i}: {loss_np[i]:.6f}")

