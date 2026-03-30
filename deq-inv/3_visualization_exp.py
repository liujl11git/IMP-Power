"""
visualize ground truth, observation, and recovered img. (explicit model)
"""

import torch
from torchvision.utils import save_image

import deepinv as dinv
from deepinv.models import DRUNet
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.utils.demo import load_degradation
from deepinv.unfolded import unfolded_builder

from pathlib import Path
import numpy as np 
import argparse
from utils import custom_init
import glob
import os 

parser = argparse.ArgumentParser()
parser.add_argument('--eps', type=float, default=0.01)
parser.add_argument('--maxiters', type=int, default=20)
parser.add_argument('--freq', type=float, default=1.0)
parser.add_argument('--optim', type=str, default='PGD', choices=['PGD','HQS'])
parser.add_argument('--init', type=str, default='y', choices=['adjoint','dagger','zero','y'])
parser.add_argument('--idx', type=int, default=0)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--save_name', type=str, default='PGD_iter_10_init_y')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--load_path', default=None)
parser.add_argument('--datapth', type=str, default=None)
args = parser.parse_args()

# Setup paths.
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "measurements"
RESULTS_DIR = BASE_DIR / "results"
DEG_DIR = BASE_DIR / "degradations"
CKPT_DIR = BASE_DIR / "ckpts"

# Setup seeds and devices.
torch.manual_seed(0)
device = torch.device('cuda:'+args.gpu if torch.cuda.is_available() else 'cpu')
num_workers = 4 if torch.cuda.is_available() else 0

# Set up parameters.
img_size = 128
noise_level_img = 0.03  # Gaussian Noise standard deviation for the degradation
n_channels = 3  # 3 for color images, 1 for gray-scale images
n_images_max = 3  # Maximal number of images to restore from the input dataset
if args.optim == 'HQS':
    max_iter = args.maxiters
    stepsize = [0.0] * max_iter
    sigma_denoiser = [0.03] * max_iter
elif args.optim == 'PGD': 
    max_iter = args.maxiters
    stepsize = [0.0] * max_iter  
    sigma_denoiser = [0.03] * max_iter 
else: 
    max_iter = args.maxiters
    stepsize = [0.0] * max_iter  
    sigma_denoiser = [0.03] * max_iter 
early_stop = False  
epochs = args.epochs
lr = args.lr
save_name = args.save_name
idx = args.idx

# Generate a motion blur operator.
kernel_index = 1  # which kernel to chose among the 8 motion kernels from 'Levin09.mat'
kernel_torch = load_degradation("Levin09.npy", DEG_DIR / "kernels", index=kernel_index)
kernel_torch = kernel_torch.float().unsqueeze(0).unsqueeze(0)  

# Setup physics.
p = dinv.physics.BlurFFT(
    img_size=(n_channels, img_size, img_size),
    filter=kernel_torch,
    device=device,
    noise_model=dinv.physics.GaussianNoise(sigma=noise_level_img),
)

# Setup dataset path.
if args.datapth is None:
    dinv_dataset_path = 'measurements/CBSD500/deblur/dinv_dataset0.h5'
else:
    dinv_dataset_path = args.datapth

# Setup datasets.
test_dataset = dinv.datasets.HDF5Dataset(path=dinv_dataset_path, split='test') 

# wrap all the restoration parameters in a 'params_algo' dictionary
params_algo = {  
    "stepsize": stepsize,
    "g_param": sigma_denoiser,
}

# define which parameters from 'params_algo' are trainable
trainable_params = [ 
    # "stepsize",
    "g_param",
]  

# Define the unfolded trainable model.
prior = [
    PnP(denoiser=DRUNet(pretrained="download", device=device))
    for i in range(max_iter)
]

model = unfolded_builder(
    iteration=args.optim,
    params_algo=params_algo.copy(),
    trainable_params=trainable_params,
    data_fidelity=L2(),
    max_iter=max_iter,
    prior=prior,
    # max_iter_backward=20,
    # jacobian_free=False,
    custom_init=custom_init(args.init),
    device=device,
)

# load model ckpts.
def find_checkpoint(base_path):
  pattern = os.path.join(str(CKPT_DIR), base_path, '*', 'ckp_best.pth.tar')
  results = glob.glob(pattern)
  return results[0] if results else None

full_path = find_checkpoint(args.load_path)

if full_path is not None:
    checkpoint = torch.load(full_path, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    print(f'Model ckpts loaded from {full_path}')
else:
    print(f'No valid checkpoint, use pretrained model.')

# Setup testing.
model.eval()
x0, y0 = test_dataset[idx]
x0 = x0.unsqueeze(0).to(device)
y0 = y0.unsqueeze(0).to(device)

with torch.no_grad():
    x0_net = model(y0, physics=p, x_gt=x0)

psnr_0 = dinv.metric.PSNR()(x0_net, x0).item()

print(f'PSNR: {psnr_0} dB')

print("Visualizing and saving images...")

save_dir = Path("visualizations")
save_dir.mkdir(parents=True, exist_ok=True)

x0_clamped = torch.clamp(x0, 0, 1)
y0_clamped = torch.clamp(y0, 0, 1)
x0_net_clamped = torch.clamp(x0_net, 0, 1)

save_image(x0_clamped, save_dir / f"id_{idx}_truth.png")
save_image(y0_clamped, save_dir / f"id_{idx}_blured.png")
save_image(x0_net_clamped, save_dir / f"id_{idx}_recon_unet.png")

print(f"Images saved to '{save_dir.resolve()}'")
    
print('Model size:', sum(p.numel() for p in model.parameters() if p.requires_grad))

