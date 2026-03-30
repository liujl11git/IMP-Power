# This script trains implicit models (PGD/HQS structures)

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import deepinv as dinv
from deepinv.models import DRUNet
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.optim.optimizers import optim_builder
from deepinv.training import test
from deepinv.utils.demo import load_dataset, load_degradation
from deepinv.unfolded import DEQ_builder

from pathlib import Path
import numpy as np 
import argparse
from utils import custom_init

parser = argparse.ArgumentParser()
parser.add_argument('--maxiters', type=int, default=100)
parser.add_argument('--optim', type=str, default='HQS', choices=['PGD','HQS'])
parser.add_argument('--init', type=str, default='y', choices=['adjoint','dagger','zero','y'])
parser.add_argument('--batch_size', type=int, default=3)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--save_name', type=str, default='PGD_iter_100_init_y')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--datapth', type=str, default=None)
args = parser.parse_args()

# Set up paths.
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "measurements"
RESULTS_DIR = BASE_DIR / "results"
DEG_DIR = BASE_DIR / "degradations"
CKPT_DIR = BASE_DIR / "ckpts"

# Set up seed and devices
torch.manual_seed(0)
device = torch.device('cuda:'+args.gpu if torch.cuda.is_available() else 'cpu')
num_workers = 4 if torch.cuda.is_available() else 0

# Set up parameters.
img_size = 128
noise_level_img = 0.03  # Gaussian Noise standard deviation for the degradation
n_channels = 3  # 3 for color images, 1 for gray-scale images
n_images_max = 3  # Maximal number of images to restore from the input dataset
batch_size = args.batch_size  # batch size for testing. As the number of iterations is fixed, we can use batch_size > 1
if args.optim == 'HQS':
    stepsize = [4.0]  
    sigma_denoiser = [0.03]  
    max_iter = args.maxiters
elif args.optim == 'PGD':
    stepsize = [1.0]  
    sigma_denoiser = [0.03]  
    max_iter = args.maxiters
else:
    stepsize = [1.0]  
    sigma_denoiser = [0.03]  
    max_iter = args.maxiters
epochs = args.epochs
lr = args.lr
save_name = args.save_name

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
train_dataset = dinv.datasets.HDF5Dataset(path=dinv_dataset_path, split='train') 
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    num_workers=num_workers, 
    shuffle=True,
)
val_dataset = dinv.datasets.HDF5Dataset(path=dinv_dataset_path, split='val') 
val_dataloader = DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    num_workers=num_workers, 
    shuffle=False,
)
test_dataset = dinv.datasets.HDF5Dataset(path=dinv_dataset_path, split='test') 
test_dataloader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    num_workers=num_workers, 
    shuffle=False,
)

# wrap all the restoration parameters in a 'params_algo' dictionary
params_algo = {  
    "stepsize": stepsize,
    "g_param": sigma_denoiser,
}

# define which parameters from 'params_algo' are trainable
trainable_params = [ 
    "stepsize",
    "g_param",
]  

# Define the unfolded trainable model.
prior = PnP(denoiser=DRUNet(pretrained="download", device=device))

model = DEQ_builder(
    iteration=args.optim,
    params_algo=params_algo.copy(),
    trainable_params=trainable_params,
    data_fidelity=L2(),
    max_iter=max_iter,
    prior=prior,
    max_iter_backward=20,
    jacobian_free=False,
    custom_init=custom_init(args.init),
    device=device,
)

# Setup trainer.
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=lr, 
    weight_decay=1e-8
)
trainer = dinv.Trainer(
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=val_dataloader,
    optimizer=optimizer,
    scheduler=None,
    losses=[dinv.loss.SupLoss(metric=dinv.metric.MSE())],
    epochs=epochs,
    physics=p,
    # metrics=[dinv.metric.PSNR(), dinv.metric.LPIPS(device=device)],
    device=device,
    save_path=str(CKPT_DIR / save_name), 
    verbose=True,
    show_progress_bar=True, 
    no_learning_method='y',
    # online_measurements = True,
    wandb_vis = False,
)

trainer.train()

# Test model.
model = trainer.load_best_model()  # load model with best validation PSNR
trainer.test(test_dataloader)


