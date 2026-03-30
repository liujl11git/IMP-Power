"""
Compute (empirical) Lipschitz number of y_t(x)
"""

import deepinv as dinv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from deepinv.models import DRUNet
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.utils.demo import load_degradation
import numpy as np 
import argparse
from utils import custom_init, get_singular_vector_by_freq
from deepinv.unfolded import DEQ_builder
import glob
import os 

parser = argparse.ArgumentParser()
parser.add_argument('--eps', type=float, default=0.01)
parser.add_argument('--maxiters', type=int, default=100)
parser.add_argument('--optim', type=str, default='PGD', choices=['PGD','HQS'])
parser.add_argument('--init', type=str, default='zero', choices=['adjoint','dagger','zero','y'])
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--load_path', default=None)
parser.add_argument('--datapth', type=str, default=None)
args = parser.parse_args()
freqs = [0.1, 0.3, 0.5, 0.7, 0.9]

# Setup paths.
BASE_DIR = Path(".")
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
batch_size = args.batch_size
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

if args.load_path is None:
    raise ValueError("--load_path is required to locate the trained checkpoint.")

# Setup datasets.
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

custom_metrics = {
    'x_history': lambda metric, prev_x, current_x: current_x
}

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
    custom_metrics=custom_metrics,
    device=device,
)

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
trainer = dinv.Trainer(
    model=model,
    train_dataloader=test_dataloader,
    eval_dataloader=test_dataloader,
    optimizer=None,
    physics=p,
    metrics=[dinv.metric.PSNR(), dinv.metric.LPIPS(device=device)],
    device=device,
    save_path=None, #save_folder / f"iter_{max_iter}",
    verbose=True,
    plot_convergence_metrics=True,
    plot_images=True,
    no_learning_method='y',
    online_measurements = True,
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
    print(f"Loaded checkpoint: {full_path}")
else:
    raise FileNotFoundError(f"No checkpoint found under ckpts/{args.load_path}/")

# Setup testing.
model.eval()

trainer.setup_train(train=False)

print("Starting aggregated L_t / PSNR(t) computation.")
print(f"Model: {args.optim}")
print(f"Dataset: {dinv_dataset_path}")
print(f"Device: {device}")
print(f"Test samples: {len(test_dataset)}")
print(f"Batch size: {batch_size}")
print(f"Frequencies: {freqs}")
print(f"Max iterations: {max_iter}")
print("For each frequency, the script evaluates the original observation y0 and the perturbed observation y1 = y0 + delta_y,")
print("then aggregates:")
print("- L_t as the max over all samples and frequencies")
print("- PSNR(t) as mean +- std over all samples, frequencies, and both y0/y1 reconstructions")

all_metric_diffs = []
all_metric_psnrs = []

for freq in freqs:
    freq_diff_count = len(all_metric_diffs)
    freq_psnr_count = len(all_metric_psnrs)
    print(f"\n=== Frequency {freq} ===")
    for batch_idx, (x0, y0) in enumerate(test_dataloader):
        
        print(f"--- Freq {freq}: Batch {batch_idx + 1}/{len(test_dataloader)}, Size: {x0.shape[0]} ---")

        # Move the current batch of data to the correct device
        x0 = x0.to(device)
        y0 = y0.to(device)
        
        delta_x = args.eps * get_singular_vector_by_freq(p, freq)
        delta_y = p.A(delta_x)
        
        diff_input = torch.linalg.norm(delta_y).cpu().item() 
        
        repeats = (x0.shape[0], 1, 1, 1)
        delta_x = delta_x.repeat(repeats)
        delta_y = delta_y.repeat(repeats)
        
        y1 = y0 + delta_y
        x1 = x0 + delta_x
        
        trainer.model_inference(y=y0, physics=p, x=x0, train=False)
        
        conv_metrics0 = trainer.conv_metrics.copy()
        trainer.conv_metrics = None
        
        trainer.model_inference(y=y1, physics=p, x=x1, train=False)
        
        numiters = min(len(trainer.conv_metrics['psnr'][0]), len(trainer.conv_metrics['residual'][0]))
        
        for ss in range(x0.shape[0]):
            metric_diff = []
            metric_psnr0 = []
            metric_psnr1 = []
            for ii in range(numiters):
                x0_net = conv_metrics0['x_history'][ss][ii]
                x1_net = trainer.conv_metrics['x_history'][ss][ii]
                diff = torch.linalg.norm(x0_net-x1_net).cpu().item()
                psnr_0 = conv_metrics0['psnr'][ss][ii]
                psnr_1 = trainer.conv_metrics['psnr'][ss][ii]
                
                metric_diff.append(diff/diff_input)
                metric_psnr0.append(psnr_0)
                metric_psnr1.append(psnr_1)
            
            all_metric_diffs.append(metric_diff)
            all_metric_psnrs.append(metric_psnr0)
            all_metric_psnrs.append(metric_psnr1)

    print(
        f"Finished freq {freq}: "
        f"{len(all_metric_diffs) - freq_diff_count} Lipschitz rows, "
        f"{len(all_metric_psnrs) - freq_psnr_count} PSNR rows collected."
    )

all_metric_diffs = np.asarray(all_metric_diffs, dtype=float)
all_metric_psnrs = np.asarray(all_metric_psnrs, dtype=float)

L_t = np.max(all_metric_diffs, axis=0)
PSNR_mean = np.mean(all_metric_psnrs, axis=0)
PSNR_std = np.std(all_metric_psnrs, axis=0)

print("=== Aggregated curves over all frequencies and test samples ===")
print(f"Lipschitz rows: {all_metric_diffs.shape}, PSNR rows: {all_metric_psnrs.shape}")
print("Columns below mean:")
print("- k: iteration index starting from 1")
print("- L_k: max normalized perturbation amplification over all samples and frequencies")
print("- PSNR_mean: mean PSNR over all samples, frequencies, and both y0/y1 trajectories")
print("- PSNR_std: standard deviation of that PSNR")
print("k,L_k,PSNR_mean,PSNR_std")
for k, (l_k, psnr_mean_k, psnr_std_k) in enumerate(zip(L_t, PSNR_mean, PSNR_std), start=1):
    print(f"{k},{l_k:.10f},{psnr_mean_k:.10f},{psnr_std_k:.10f}")
