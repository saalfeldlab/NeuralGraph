#!/usr/bin/env python3
import argparse
import commentjson as json
import numpy as np
import torch
import tinycudann as tcnn
from tqdm import trange
from PIL import Image as PILImage

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
    parser.add_argument("config", nargs="?", default="config_hash_video.json")
    parser.add_argument("n_steps", nargs="?", type=int, default=10000)
    args = parser.parse_args()
    
    device = torch.device("cuda")
    with open(args.config) as f:
        config = json.load(f)
    
    video = Video(args.video, device)
    T, H, W, C = video.shape
    
    model = tcnn.NetworkWithInputEncoding(
        n_input_dims=3, n_output_dims=C,
        encoding_config=config["encoding"],
        network_config=config["network"]
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    batch_size = 2**20
    
    for i in trange(args.n_steps):
        batch = torch.rand([batch_size, 3], device=device)
        targets = video(batch)
        output = model(batch)
        loss = ((output - targets)**2 / (output.detach()**2 + 0.01)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
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
    
    write_video("result.tif", np.stack(result_frames))
    print(f"Final loss: {loss.item():.6f}")