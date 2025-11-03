#!/usr/bin/env python3
import argparse
import json
import numpy as np
import os
import torch
import tinycudann as tcnn
from PIL import Image as PILImage
import itertools
from tqdm import trange

def calculate_psnr(img1, img2, max_val=1.0):
    mse = np.mean((img1 - img2) ** 2)
    return float('inf') if mse == 0 else 20 * np.log10(max_val / np.sqrt(mse))

def read_video(filename):
    img = PILImage.open(filename)
    frames = []
    try:
        for i in range(img.n_frames):
            img.seek(i)
            frames.append(np.array(img.convert('RGB')).astype(np.float32) / 255.0)
    except EOFError:
        pass
    return np.stack(frames)

class Video(torch.nn.Module):
    def __init__(self, data, device):
        super().__init__()
        self.data = torch.from_numpy(data).float().to(device)
        self.shape = self.data.shape
    
    def forward(self, xyt):
        with torch.no_grad():
            T, H, W, C = self.shape
            coords = xyt * torch.tensor([W, H, T], device=xyt.device).float()
            idx = coords.long()
            lerp = coords - idx.float()
            
            x0, y0, t0 = idx[:, 0].clamp(0, W-1), idx[:, 1].clamp(0, H-1), idx[:, 2].clamp(0, T-1)
            x1, y1, t1 = (x0+1).clamp(max=W-1), (y0+1).clamp(max=H-1), (t0+1).clamp(max=T-1)
            wx, wy, wt = lerp[:, 0:1], lerp[:, 1:2], lerp[:, 2:3]
            
            return (
                self.data[t0, y0, x0] * (1-wx) * (1-wy) * (1-wt) +
                self.data[t0, y0, x1] * wx * (1-wy) * (1-wt) +
                self.data[t0, y1, x0] * (1-wx) * wy * (1-wt) +
                self.data[t0, y1, x1] * wx * wy * (1-wt) +
                self.data[t1, y0, x0] * (1-wx) * (1-wy) * wt +
                self.data[t1, y0, x1] * wx * (1-wy) * wt +
                self.data[t1, y1, x0] * (1-wx) * wy * wt +
                self.data[t1, y1, x1] * wx * wy * wt
            )

def train_config(video_data, config, n_steps, batch_size, device):
    T, H, W, C = video_data.shape
    video = Video(video_data, device)
    
    model = tcnn.NetworkWithInputEncoding(
        n_input_dims=3, n_output_dims=C,
        encoding_config=config["encoding"],
        network_config=config["network"]
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for _ in trange(n_steps, ncols=150):
        batch = torch.rand([batch_size, 3], device=device)
        targets = video(batch)
        output = model(batch)
        loss = ((output - targets)**2 / (output.detach()**2 + 0.01)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    t_coords = torch.linspace(0.5/T, 1-0.5/T, T, device=device)
    y_coords = torch.linspace(0.5/H, 1-0.5/H, H, device=device)
    x_coords = torch.linspace(0.5/W, 1-0.5/W, W, device=device)
    
    result_frames = []
    with torch.no_grad():
        for t in t_coords:
            yv, xv = torch.meshgrid(y_coords, x_coords, indexing='ij')
            tv = torch.full_like(xv, t)
            xyt = torch.stack([xv.flatten(), yv.flatten(), tv.flatten()], dim=1)
            frame = model(xyt).reshape(H, W, C).clamp(0, 1).cpu().numpy()
            result_frames.append(frame)
    
    psnr = calculate_psnr(video_data, np.stack(result_frames))
    return psnr, loss.item(), n_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="dolphins.tif")
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=2**20)
    parser.add_argument("--output", default="optimization_results.json")
    args = parser.parse_args()
    
    device = torch.device("cuda")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(script_dir, args.video)
    output_path = os.path.join(script_dir, args.output)
    
    video_data = read_video(video_path)
    
    # 32 configs: 4×2×4 = 32
    n_levels_opts = [16, 20, 24, 28]
    log2_hash_opts = [15, 19, 23, 27]
    n_neurons_opts = [64, 128]
    
    results = []
    total = len(list(itertools.product(n_levels_opts, log2_hash_opts, n_neurons_opts)))
    print(f"testing {total} configurations...")
    
    for n_lv, log2_h, n_neu in itertools.product(n_levels_opts, log2_hash_opts, n_neurons_opts):
        config = {
            "encoding": {
                "otype": "HashGrid",
                "n_levels": n_lv,
                "n_features_per_level": 2,
                "log2_hashmap_size": log2_h,
                "base_resolution": 16,
                "per_level_scale": 1.5
            },
            "network": {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": n_neu,
                "n_hidden_layers": 2
            }
        }
        
        psnr, loss, n_params = train_config(video_data, config, args.n_steps, args.batch_size, device)

        result = {
            "n_levels": n_lv, "log2_hash": log2_h, "n_neurons": n_neu,
            "psnr": float(psnr), "loss": float(loss), "n_params": n_params
        }
        results.append(result)
        print(f"PSNR: {psnr:.2f} dB | L={n_lv} H={log2_h} N={n_neu} | params: {n_params:,}")
    
    results.sort(key=lambda x: x["psnr"], reverse=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBest: PSNR={results[0]['psnr']:.2f} dB")
    print(json.dumps(results[0], indent=2))