import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_field_comparison(x, model, k, n_frames, ones, output_path, step):
    """
    Plots true field, NNR field, and noisy NNR field for a given batch.
    Args:
        x: torch.Tensor, shape [n_neurons, features]
        model: model with NNR_f method
        n_frames: int, total number of frames
        ones: torch.Tensor, shape [n_neurons, 1]
        log_dir: str, output directory
        epoch: int, current epoch
        N: int, current iteration
    """
    x = torch.tensor(x, dtype=torch.float32, device=ones.device)
    in_features = torch.cat((x[:,1:4], k/n_frames * ones), 1)
    field = model.NNR_f(in_features)**2
    vmin = 0.048
    vmax = 0.451
    # loss = (field - x[:, 6:7]).norm(2)
    # print("loss: {:.6f}".format(loss.item()))

    fig = plt.figure(figsize=(17, 10))
    plt.subplot(2, 2, 1)
    plt.scatter(x[:, 1].cpu().numpy(), x[:, 2].cpu().numpy(), c=x[:, 6].cpu().numpy().squeeze(), s=0.5, cmap='plasma', vmin=vmin, vmax=vmax)
    plt.title('true field', fontsize=24)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim([0, 2])
    plt.ylim([0, 1.35])
    plt.subplot(2, 2, 2)
    in_features = torch.cat((x[:,1:4], k/n_frames * ones), 1)
    field = model.NNR_f(in_features)**2
    plt.scatter(x[:, 1].cpu().numpy(), x[:, 2].cpu().numpy(), c=field.cpu().numpy().squeeze(), s=0.5, cmap='plasma', vmin=vmin, vmax=vmax)
    plt.title('NNR field', fontsize=24)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim([0, 2])
    plt.ylim([0, 1.35])
    plt.subplot(2, 2, 4)
    for loop in range(step):
        x_ = x.clone().detach()
        x_ = x_ + torch.randn(x_.shape, device=x_.device) * 0.05
        in_features = torch.cat((x_[:,1:4], k/n_frames * ones), 1)
        field = model.NNR_f(in_features)**2
        plt.scatter(x_[:, 1].cpu().numpy(), x_[:, 2].cpu().numpy(), c=field.cpu().numpy().squeeze(), s=0.25, alpha=0.5, edgecolors='None', cmap='plasma', vmin=vmin, vmax=vmax)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim([0, 2])
    plt.ylim([0, 1.35])
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()