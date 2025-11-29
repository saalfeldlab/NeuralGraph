import torch

# Load the first saved model (epoch 0)
model_path = '/workspace/NeuralGraph/log/signal/signal_N11_1_8_2/models/best_model_with_1_graphs_0_0.pt'

print(f"Loading model from: {model_path}")
checkpoint = torch.load(model_path, map_location='cpu')

print("\nKeys in checkpoint:")
for key in checkpoint.keys():
    print(f"  {key}")

if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    print("\nKeys in model_state_dict:")
    for key in state_dict.keys():
        print(f"  {key}: {state_dict[key].shape}")

    if 'a' in state_dict:
        print("\n" + "="*60)
        print("model.a values:")
        print("="*60)
        print(f"Shape: {state_dict['a'].shape}")
        print(f"Values:\n{state_dict['a']}")
        print(f"\nRequires grad (if available): {state_dict['a'].requires_grad if hasattr(state_dict['a'], 'requires_grad') else 'N/A'}")
    else:
        print("\nWarning: 'a' not found in model_state_dict")
else:
    print("\nWarning: 'model_state_dict' not found in checkpoint")
