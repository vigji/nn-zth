# %%
import os
import sys
import einops.layers
import torch as t
from torch import nn, optim
import einops
from einops.layers.torch import Rearrange
import torch
from tqdm import tqdm
from dataclasses import dataclass, field
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from typing import Optional, Tuple, Literal, Union
import plotly.express as px
import torchinfo
import time
import wandb
from PIL import Image
import pandas as pd
from pathlib import Path
from datasets import load_dataset

from utils_p2 import print_param_count
import tests as tests
import solutions as solutions
from plotly_utils import imshow

from solutions_p2 import (
    Linear,
    ReLU,
    Sequential,
    BatchNorm2d,
)
from solutions_bonus_p2 import (
    pad1d,
    pad2d,
    conv1d_minimal,
    conv2d_minimal,
    Conv2d,
    Pair,
    IntOrPair,
    force_pair,
)

device = t.device(
    "mps"
    if t.backends.mps.is_available()
    else "cuda"
    if t.cuda.is_available()
    else "cpu"
)

section_dir = Path(__file__).parent
celeb_data_dir = section_dir / "data/celeba"
celeb_image_dir = celeb_data_dir / "img_align_celeba"
# %%
os.makedirs(celeb_image_dir, exist_ok=True)

if len(list(celeb_image_dir.glob("*.jpg"))) > 0:
    print("Dataset already loaded.")
else:
    dataset = load_dataset("nielsr/CelebA-faces")
    print("Dataset loaded.")

    for idx, item in tqdm(
        enumerate(dataset["train"]),
        total=len(dataset["train"]),
        desc="Saving individual images...",
    ):
        # The image is already a JpegImageFile, so we can directly save it
        item["image"].save(celeb_image_dir / f"{idx:06}.jpg")

    print("All images have been saved.")


def get_dataset(dataset: Literal["MNIST", "CELEB"], train: bool = True) -> Dataset:
    assert dataset in ["MNIST", "CELEB"]

    if dataset == "CELEB":
        image_size = 64
        assert train, "CelebA dataset only has a training set"
        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        trainset = datasets.ImageFolder(
            root=exercises_dir / "part5_gans_and_vaes/data/celeba", transform=transform
        )

    elif dataset == "MNIST":
        img_size = 28
        transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        trainset = datasets.MNIST(
            root=exercises_dir / "part5_gans_and_vaes/data",
            transform=transform,
            download=True,
        )

    return trainset


def display_data(x: t.Tensor, nrows: int, title: str):
    """Displays a batch of data, using plotly."""
    # Reshape into the right shape for plotting (make it 2D if image is monochrome)
    y = einops.rearrange(x, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=nrows).squeeze()
    # Normalize, in the 0-1 range
    y = (y - y.min()) / (y.max() - y.min())
    # Display data
    imshow(
        y,
        binary_string=(y.ndim == 2),
        height=50 * (nrows + 5),
        title=title + f"<br>single input shape = {x[0].shape}",
    )


# Load in MNIST, get first batch from dataloader, and display
trainset_mnist = get_dataset("MNIST")
x = next(iter(DataLoader(trainset_mnist, batch_size=64)))[0]
display_data(x, nrows=8, title="MNIST data")

# Load in CelebA, get first batch from dataloader, and display
trainset_celeb = get_dataset("CELEB")
x = next(iter(DataLoader(trainset_celeb, batch_size=64)))[0]
display_data(x, nrows=8, title="CelebA data")

# %%

testset = get_dataset("MNIST", train=False)
HOLDOUT_DATA_DICT = dict()
for data, target in DataLoader(testset, batch_size=1):
    if target.item() not in HOLDOUT_DATA_DICT:
        HOLDOUT_DATA_DICT[target.item()] = data.squeeze()
        if len(HOLDOUT_DATA_DICT) == 10: break
HOLDOUT_DATA = t.stack([HOLDOUT_DATA_DICT[i] for i in range(10)]).to(dtype=t.float, device=device).unsqueeze(1)

display_data(HOLDOUT_DATA, nrows=1, title="MNIST holdout data")
# %%

# Implementing an autoencoder

class Autoencoder(nn.Module):

    def __init__(self, latent_dim_size: int, hidden_dim_size: int):
        super().__init__()
        # Your code here
        # Input 28 x 28
        self.latent_dim_size = latent_dim_size
        self.hidden_dim_size = hidden_dim_size
        
        self.encoder = nn.Sequential(
                # Conv kernel 4x4, stride 2, padding 1, channels -> 16
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=1, bias=False),
                nn.ReLU(),
                # Conv kernel 4x4, stride 2, padding 1, channels 16 -> 32
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
                nn.ReLU(),
                # reshape to linear, then Linear layer with hidden_dim_size
                einops.layers.torch.Rearrange('b c h w -> b (c h w)'),
                nn.Linear(32*7*7, hidden_dim_size),
                nn.ReLU(),
                # Linear layer with latent_dim_size:
                nn.Linear(hidden_dim_size, latent_dim_size),
                # ReLU?
                nn.ReLU())
                # END OF ENCODER
        self.decoder = nn.Sequential(
                # Linear layer with hidden_dim_size
                nn.Linear(latent_dim_size, hidden_dim_size),
                nn.ReLU(),
                # Linear layer with 32*7*7
                nn.Linear(hidden_dim_size, 32*7*7),
                nn.ReLU(),
                # reshape to 32x7x7
                einops.layers.torch.Rearrange('b (c h w) -> b c h w', c=32, h=7, w=7),
                # ConvTranspose kernel 4x4, stride 2, padding 1, channels 32 -> 16
                nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1, bias=False),
                nn.ReLU(),
                # ConvTranspose kernel 4x4, stride 2, padding 1, channels 16 -> 1
                nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False),    
        )


    def forward(self, x: t.Tensor) -> t.Tensor:
        # Your code here
        encoded = self.layers(x)
        decoded = self.decoder(encoded)
        return decoded
    
import solutions
soln_Autoencoder = solutions.Autoencoder(latent_dim_size=5, hidden_dim_size=128)
my_Autoencoder = Autoencoder(latent_dim_size=5, hidden_dim_size=128)

print_param_count(my_Autoencoder, soln_Autoencoder)
# %%
