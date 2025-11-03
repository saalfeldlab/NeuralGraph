#!/usr/bin/env python3
import argparse
import commentjson as json
import numpy as np
import os
import torch
import tinycudann as tcnn
from tqdm import trange
from PIL import Image as PILImage

def calculate_psnr(img1, img2, max_val=1.0):
    """Calculate PSNR between two images or video sequences"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    return psnr

def count_parameters(model):
    """Count the number of learnable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def read_video(filename):
    """Read multi-frame TIFF as (T, H, W, C) array in [0,1]"""
    img = PILImage.open(filename)
    frames = []
    try:
        for i in range(img.n_frames):
            img.seek(i)
            frames.append(np.array(img.convert('RGB')).astype(np.float32) / 255.0)
    except EOFError:
        pass
    return np.stack(frames)

def write_video(filename, frames):
    """Write (T, H, W, C) array to multi-frame TIFF"""
    frames = np.clip(frames * 255.0, 0, 255).astype(np.uint8)
    imgs = [PILImage.fromarray(f) for f in frames]
    imgs[0].save(filename, save_all=True, append_images=imgs[1:])

class Video(torch.nn.Module):
    def __init__(self, filename, device):
        super().__init__()
        self.data = torch.from_numpy(read_video(filename)).float().to(device)
        self.shape = self.data.shape  # (T, H, W, C)
    
    def forward(self, xyt):
        """xyt: (N, 3) with values in [0,1] for (x, y, t)"""
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", default="dolphins.tif")
    parser.add_argument("config", nargs="?", default="config_hash_video composite.json")
    parser.add_argument("n_steps", nargs="?", type=int, default=31000)
    args = parser.parse_args()
    
    device = torch.device("cuda")
    
    # Get script directory and construct paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, args.config)
    video_path = os.path.join(script_dir, args.video)
    result_path = os.path.join(script_dir, "result.tif")
    
    with open(config_path) as f:
        config = json.load(f)
    
    video = Video(video_path, device)
    T, H, W, C = video.shape
    
    model = tcnn.NetworkWithInputEncoding(
        n_input_dims=3, n_output_dims=C,
        encoding_config=config["encoding"],
        network_config=config["network"]
    ).to(device)
    
    # Print model information
    n_params = count_parameters(model)
    print (f'{args.config}')
    print(f"model has {n_params:,} learnable parameters")
    print(f"video shape: {video.shape} (T={T}, H={H}, W={W}, C={C}) = {T*H*W*C:,} values")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    batch_size = 2**20


    
    for i in trange(args.n_steps, ncols=80):
        batch = torch.rand([batch_size, 3], device=device)
        targets = video(batch)
        output = model(batch)
        loss = ((output - targets)**2 / (output.detach()**2 + 0.01)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10000 == 0:

            # Save result
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
            
            # Save result in current working directory
            result_stack = np.stack(result_frames)
            write_video(result_path, result_stack)
    
            # Calculate PSNR between ground truth and reconstruction
            ground_truth = video.data.cpu().numpy()
            psnr_db = calculate_psnr(ground_truth, result_stack)
            print(f"step {i+1}/{args.n_steps}, loss: {loss.item():.6f}, PSNR: {psnr_db:.2f} dB")