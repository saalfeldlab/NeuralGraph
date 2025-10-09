# SIREN network
# Code adapted from the following GitHub repository:
# https://github.com/vsitzmann/siren?tab=readme-ov-file
import os

import numpy as np
import torch
import torch.nn as nn

# from NeuralGraph.generators.utils import get_time_series
import matplotlib
from matplotlib import pyplot as plt
from tifffile import imread, imwrite as imsave
from tqdm import trange
from tqdm import tqdm
# from torch.utils.data import DataLoader, Dataset
# from PIL import Image
# import skimage
# from torchvision.transforms import Resize, Compose, ToTensor, Normalize



class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        # coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output

class small_Siren(nn.Module):
    def __init__(self, in_features=1, hidden_features=128, hidden_layers=3, out_features=1, outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30):
        super().__init__()

        layers = [SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0)]
        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))
        layers.append(nn.Linear(hidden_features, out_features))  # final linear layer
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Siren_Network(nn.Module):
    def __init__(self, image_width, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30., device='cuda:0'):
        super().__init__()

        self.device = device 
        self.image_width = image_width

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)

        self.net = self.net.to(device)

        self.coords = None

    @property
    def values(self):
        # Call forward method
        output, coords = self.__call__()
        return output.squeeze().reshape(self.image_width, self.image_width)
    
    def coordinate_grid(self, n_points):
        coords = np.linspace(0, 1, n_points, endpoint=False)
        xy_grid = np.stack(np.meshgrid(coords, coords), -1)
        xy_grid = torch.tensor(xy_grid).unsqueeze(0).permute(0, 3, 1, 2).float().contiguous().to(self.device)
        return xy_grid
    
    def get_mgrid(self, sidelen, dim=2, enlarge=False):
        '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
        sidelen: int
        dim: int'''
        if enlarge:
            # tensors = tuple(dim * [torch.linspace(-0.2, 1.2, steps=sidelen*20)])
            tensors = tuple(dim * [torch.linspace(0, 1, steps=sidelen*20)])
        else:
            tensors = tuple(dim * [torch.linspace(0, 1, steps=sidelen)])

        mgrid = torch.stack(torch.meshgrid(*tensors,indexing="ij"), dim=-1)
        mgrid = mgrid.reshape(-1, dim)
        return mgrid

    def forward(self, coords=None, time=None, enlarge=False):

        if coords is None:
            coords = self.get_mgrid(self.image_width, dim=2, enlarge=enlarge).to(self.device)
            if time != None:
               coords = torch.cat((coords, time * torch.ones_like(coords[:, 0:1])), 1)

        # coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output

        output = self.net(coords)
        return output


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(0, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


from torch.utils.data import Dataset
class ImageFitting(Dataset):
    def __init__(self, sidelength):
        import skimage
        from PIL import Image
        from torchvision.transforms import Resize, Compose, ToTensor, Normalize
        img = Image.fromarray(skimage.data.camera())
        transform = Compose([
            Resize(sidelength),
            ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])
        self.img = transform(img)
        coords = get_mgrid(sidelength, dim=2)
        self.coords = coords

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.coords, self.img.view(-1, 1)
    
if __name__ == '__main__':


    device = 'cuda:0'
    try:
        matplotlib.use("Qt5Agg")
    except:
        pass
    
    from torch.utils.data import DataLoader


    img_siren = Siren(in_features=2, out_features=1, hidden_features=512,
                      hidden_layers=3, outermost_linear=True, first_omega_0=512., hidden_omega_0=512.)
    img_siren.cuda()

    total_steps = 1000  # Since the whole image is our dataset, this just means 500 gradient descent steps.
    steps_til_summary = 1000

    optim = torch.optim.Adam(lr=1e-5, params=img_siren.parameters())

    import skimage
    from PIL import Image
    from torchvision.transforms import Resize, Compose, ToTensor, Normalize

    sidelength = 256
    img = Image.fromarray(skimage.data.camera())
    transform = Compose([
            ToTensor(),
            Resize(sidelength)
        ])
    ground_truth = transform(img)
    # ground_truth = torch.cat((ground_truth, ground_truth), dim=1)  # (3, H, W)
    
    _, nx, ny= ground_truth.shape

    print(f'shape {ground_truth.shape}')

    side_legnth = max(nx,ny)
    iy, ix = torch.meshgrid(
        torch.arange(ny, device=device),
        torch.arange(nx, device=device),
        indexing='ij'
    )
    model_input = torch.stack([iy.reshape(-1), ix.reshape(-1)], dim=1).to(torch.int32) / (side_legnth - 1) # (N,2)
    model_input = model_input.cuda()

    ground_truth = ground_truth.view(-1, 1).cuda()

    
    loss_list = []
    for step in trange(total_steps):
        model_output = img_siren(model_input)
        loss = ((model_output - ground_truth) ** 2).mean()

        # TV loss
        # grad = gradient(model_output, model_input)
        # grad_norm = torch.sqrt(torch.sum(grad ** 2, dim=-1, keepdim=True) + 1e-10)
        # tv_loss = grad_norm.mean()
        # loss += 1e-5 * tv_loss

        optim.zero_grad()
        loss.backward()
        optim.step()

        loss_list.append(loss.item())

    print("Step %d, Total loss %0.6f" % (step, loss))
    # img_grad = gradient(model_output, coords)
    # img_laplacian = laplace(model_output, coords)


    fig, axes = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'wspace': 0.3, 'hspace': 0})
    axes[0].plot(loss_list, color='k')
    axes[0].set_xlabel('step')
    axes[0].set_ylabel('MSE Loss')
    pred_img = model_output.cpu().detach().numpy().reshape(nx, ny)
    axes[1].imshow(pred_img, cmap='gray')
    error = np.linalg.norm(ground_truth.cpu().detach().numpy().reshape(nx, ny) - pred_img, ord=2)
    axes[1].text(0.1, 0.95, f'L2 error: {error:.4f}', transform=axes[1].transAxes, fontsize=12, va='top', alpha=0.9)
    plt.tight_layout(pad=1.0)
    plt.show()              



    # factor = 160

    # x_upsampled = np.linspace(0, 1, 256*factor)
    # y_upsampled = np.ones(256*factor)*(100/256)  
    # coords_upsampled = np.stack([x_upsampled, y_upsampled], axis=1)  # shape (2560, 2)
    # coords_upsampled_torch = torch.tensor(coords_upsampled, dtype=torch.float32, device=device)
    # pred_upsampled = img_siren(coords_upsampled_torch).cpu().detach().numpy().squeeze()
    # x_upsampled = np.linspace(0, 256, 256*factor)

    # gt_img = ground_truth.cpu().detach().numpy().reshape(nx, ny)
    # pred_img = model_output.cpu().detach().numpy().reshape(nx, ny)

    # fig, axes = plt.subplots(1, 4, figsize=(18, 4), gridspec_kw={'wspace': 0.3, 'hspace': 0})
    # axes[0].plot(loss_list, color='k')
    # axes[0].set_xlabel('step')
    # axes[0].set_ylabel('MSE Loss')
    # axes[1].imshow(pred_img, cmap='gray')
    # error = np.linalg.norm(gt_img - pred_img, ord=2)
    # axes[1].text(0.1, 0.95, f'L2 error: {error:.4f}', transform=axes[1].transAxes, fontsize=12, va='top', alpha=0.9)
    # axes[2].plot(gt_img[:,100], color='green', label='true')
    # axes[2].scatter(x_upsampled,pred_upsampled, color='black', label='pred', s=1)
    # axes[3].plot(gt_img[:,100], color='green', label='true')
    # axes[3].scatter(x_upsampled,pred_upsampled, color='black', label='pred', s=1)
    # axes[3].set_xlim(200, 220)
    # axes[3].set_ylim(0, 0.4)
    # plt.tight_layout(pad=1.0)
    # plt.show()
    # plt.close()

