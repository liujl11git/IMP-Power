"""
Create the CBSD500 deblurring HDF5 dataset used by training and testing.
"""
from pathlib import Path
import deepinv as dinv
import torch
from torchvision import transforms
from deepinv.utils.demo import load_degradation, load_dataset
from utils import SimpleImageFolder

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "measurements"
RESULTS_DIR = BASE_DIR / "results"
DEG_DIR = BASE_DIR / "degradations"
CKPT_DIR = BASE_DIR / "ckpts"
DATASETS_DIR = BASE_DIR / "datasets"

torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_workers = 4 if torch.cuda.is_available() else 0

# Set up paths.
dataset_name = "CBSD500"
operation = "deblur"
measurement_dir = DATA_DIR / dataset_name / operation

# Set up parameters.
img_size = 128
noise_level_img = 0.03  # Gaussian Noise standard deviation for the degradation
n_channels = 3  # 3 for color images, 1 for gray-scale images
n_images_max = 3  # Maximal number of images to restore from the input dataset

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

# Bootstrap the CBSD500 download into ./datasets if it is not already present.
# We reuse the same transform pipeline as the commented block after the data is present.
bootstrap_transform = transforms.Compose(
    [
        transforms.RandomCrop(img_size),
        transforms.ToTensor(),
    ]
)
load_dataset(dataset_name, transform=bootstrap_transform, data_dir=DATASETS_DIR)

# Setup data folders.
transform = transforms.Compose(
    [
        transforms.RandomCrop(img_size),
        transforms.ToTensor(),
    ]
)

train_base_dataset = SimpleImageFolder(
    root=DATASETS_DIR / "CBSD500" / "BSR" / "BSDS500" / "data" / "images" / "train",
    transform=transform,
)
test_base_dataset = SimpleImageFolder(
    root=DATASETS_DIR / "CBSD500" / "BSR" / "BSDS500" / "data" / "images" / "test",
    transform=transform,
)
val_base_dataset = SimpleImageFolder(
    root=DATASETS_DIR / "CBSD500" / "BSR" / "BSDS500" / "data" / "images" / "val",
    transform=transform,
)

print(len(train_base_dataset), len(test_base_dataset), len(val_base_dataset))

# Generate a dataset in a HDF5 folder in "{dir}/dinv_dataset0.h5" and load it.
dinv_dataset_path = dinv.datasets.generate_dataset(
    train_dataset=train_base_dataset,
    test_dataset=test_base_dataset,
    val_dataset=val_base_dataset,
    physics=p,
    device=device,
    save_dir=measurement_dir,
    # train_datapoints=n_images_max,
    num_workers=num_workers,
)

print(f"Saved dataset to {dinv_dataset_path}")
