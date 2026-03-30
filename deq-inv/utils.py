import os
import torch
import deepinv as dinv
import numpy as np 
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SimpleImageFolder(Dataset):
    """A custom dataset to load images from a folder with no subdirectories."""

    def __init__(self, root, transform=None):
        """
        Args:
            root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root
        self.transform = transform
        # Get a list of all image file paths and filter out non-image files
        self.image_paths = sorted([
            os.path.join(root, f) for f in os.listdir(root)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
        ])

    def __len__(self):
        """Returns the total number of images."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Gets the image at a given index, applies the transform, and returns it.
        Note: We don't return a label here because there isn't one.
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB') # Ensure image is in RGB format

        if self.transform:
            image = self.transform(image)

        return image
        
        
        
        
        
def custom_init(option='zero'):
    """
    A factory that returns an initialization function based on the selected option.

    :param str option: The initialization strategy. Can be one of 'adjoint', 'zero', or 'y'.
    :return: A function that takes (y, physics) as input and returns the initialized tensors.
    """
    if option == 'adjoint':
        # Return a function that performs adjoint initialization
        def init_fn(y: torch.Tensor, physics: dinv.physics.Physics):
            x_init = physics.A_adjoint(y)
            z_init = physics.A_adjoint(y)
            return {"est": (x_init, z_init)}
        return init_fn
        
    elif option == 'dagger':
        # Return a function that performs adjoint initialization
        def init_fn(y: torch.Tensor, physics: dinv.physics.Physics):
            x_init = physics.A_dagger(y)
            z_init = physics.A_dagger(y)
            return {"est": (x_init, z_init)}
        return init_fn

    elif option == 'zero':
        # Return a function that performs zero initialization
        def init_fn(y: torch.Tensor, physics: dinv.physics.Physics):
            x_init = torch.zeros_like(y)
            z_init = torch.zeros_like(y)
            return {"est": (x_init, z_init)}
        return init_fn

    elif option == 'y':
        # Return a function that initializes with y
        def init_fn(y: torch.Tensor, physics: dinv.physics.Physics):
            x_init = torch.clone(y)
            z_init = torch.clone(y)
            return {"est": (x_init, z_init)}
        return init_fn
        
    else:
        # Raise an error for an unknown option
        raise ValueError(f"Unknown option '{option}'. Please choose from 'adjoint', 'zero', or 'y'.")







def get_singular_vector_by_freq(physics: dinv.physics.Physics, freq: float):
    """
    Constructs a singular vector based on the geometric distance of its frequency.
    """
    if not isinstance(physics, dinv.physics.BlurFFT):
        raise TypeError("This function only supports deepinv.physics.BlurFFT objects.")
    if not 0.0 <= freq <= 1.0:
        raise ValueError("freq must be between 0.0 and 1.0.")

    n_channels, H, W = physics.img_size
    device = physics.filter.device

    u_freqs = np.fft.fftfreq(H)
    v_freqs = np.fft.rfftfreq(W)
    v_grid, u_grid = np.meshgrid(v_freqs, u_freqs)
    dist_sq = u_grid**2 + v_grid**2

    sorted_indices = np.argsort(dist_sq.flatten())
    target_rank = int((len(sorted_indices) - 1) * freq)
    target_flat_idx = sorted_indices[target_rank]
    u, v = np.unravel_index(target_flat_idx, dist_sq.shape)

    f_shape_real = (H, W // 2 + 1, 2)
    one_hot_f = torch.zeros(f_shape_real, dtype=torch.float, device=device)
    one_hot_f[u, v, 0] = 1.0

    singular_vector_1ch = physics.V(one_hot_f.unsqueeze(0).unsqueeze(0))
    delta_x = singular_vector_1ch.repeat(1, n_channels, 1, 1)

    return delta_x


def get_singular_vector_by_value(physics: dinv.physics.Physics, percentile: float):
    """
    Constructs a singular vector based on the percentile of singular VALUES.
    
    Work not very well (tested percentile = 0.01, 0.1, 0.2, 0.3, 0.5, 0.8. diff stays at 1e-5)
    """
    if not isinstance(physics, dinv.physics.BlurFFT):
        raise TypeError("This function only supports deepinv.physics.BlurFFT objects.")
    if not 0.0 <= percentile <= 1.0:
        raise ValueError("percentile must be between 0.0 and 1.0.")

    n_channels, H, W = physics.img_size
    device = physics.filter.device

    s_vals = physics.mask[0, 0, ..., 0].cpu().numpy()
    sorted_indices = np.argsort(s_vals.flatten())
    target_rank = int((len(sorted_indices) - 1) * percentile)
    target_flat_idx = sorted_indices[target_rank]
    u, v = np.unravel_index(target_flat_idx, s_vals.shape)

    f_shape_real = (H, W // 2 + 1, 2)
    one_hot_f = torch.zeros(f_shape_real, dtype=torch.float, device=device)
    one_hot_f[u, v, 0] = 1.0

    singular_vector_1ch = physics.V(one_hot_f.unsqueeze(0).unsqueeze(0))
    delta_x = singular_vector_1ch.repeat(1, n_channels, 1, 1)

    return delta_x