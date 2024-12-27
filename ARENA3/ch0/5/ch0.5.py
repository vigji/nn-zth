# %%
import os
from pyexpat import model
import sys
from altair import layer
from click import progressbar
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
    else "cuda" if t.cuda.is_available() else "cpu"
)

section_dir = Path(__file__).parent
data_dir = section_dir / "data"
data_dir.mkdir(exist_ok=True, parents=True)

celeb_data_dir = data_dir / "celeba"
celeb_image_dir = celeb_data_dir / "img_align_celeba"
celeb_image_dir.mkdir(exist_ok=True, parents=True)
# %%

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
            root=celeb_data_dir,
            transform=transform,
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
            root=data_dir,
            transform=transform,
            download=True,
            train=train,
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
        if len(HOLDOUT_DATA_DICT) == 10:
            break
HOLDOUT_DATA = (
    t.stack([HOLDOUT_DATA_DICT[i] for i in range(10)])
    .to(dtype=t.float, device=device)
    .unsqueeze(1)
)

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
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            # Conv kernel 4x4, stride 2, padding 1, channels 16 -> 32
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            # reshape to linear, then Linear layer with hidden_dim_size
            einops.layers.torch.Rearrange("b c h w -> b (c h w)"),
            nn.Linear(32 * 7 * 7, hidden_dim_size),
            nn.ReLU(),
            # Linear layer with latent_dim_size:
            nn.Linear(hidden_dim_size, latent_dim_size),
        )
        # END OF ENCODER
        self.decoder = nn.Sequential(
            # Linear layer with hidden_dim_size
            nn.Linear(latent_dim_size, hidden_dim_size),
            nn.ReLU(),
            # Linear layer with 32*7*7
            nn.Linear(hidden_dim_size, 32 * 7 * 7),
            nn.ReLU(),
            # reshape to 32x7x7
            einops.layers.torch.Rearrange("b (c h w) -> b c h w", c=32, h=7, w=7),
            # ConvTranspose kernel 4x4, stride 2, padding 1, channels 32 -> 16
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            # ConvTranspose kernel 4x4, stride 2, padding 1, channels 16 -> 1
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        # Your code here
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


import solutions

soln_Autoencoder = solutions.Autoencoder(latent_dim_size=5, hidden_dim_size=128)
my_Autoencoder = Autoencoder(latent_dim_size=5, hidden_dim_size=128)

print_param_count(my_Autoencoder, soln_Autoencoder)
#


# Let's start the actual training:
@dataclass
class AutoencoderArgs:
    # architecture
    latent_dim_size: int = 5
    hidden_dim_size: int = 128

    # data / training
    dataset: Literal["MNIST", "CELEB"] = "MNIST"
    batch_size: int = 512
    epochs: int = 10
    lr: float = 1e-3
    betas: tuple[float, float] = (0.5, 0.999)

    # logging
    use_wandb: bool = False
    wandb_project: Optional[str] = "day5-ae-mnist"
    wandb_name: Optional[str] = None


class AutoencoderTrainer:
    def __init__(self, args):
        self.args = args
        self.training_dataset = get_dataset(args.dataset, train=True)

        self.dataset_loader = DataLoader(
            self.training_dataset, batch_size=args.batch_size, shuffle=True
        )

        self.model = Autoencoder(
            hidden_dim_size=args.hidden_dim_size, latent_dim_size=args.latent_dim_size
        ).to(device)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=args.lr, betas=args.betas
        )

        self.loss_fun = nn.MSELoss()

    def train_step(self, x):
        pred = self.model(x)
        loss = self.loss_fun(x, pred)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    @torch.inference_mode()
    def evaluate(self):
        preds = self.model(HOLDOUT_DATA)

        if self.args.use_wandb:
            wandb_images = [
                wandb.Image(preds[i, 0, :, :]) for i in range(preds.shape[0])
            ]

            wandb.log({"Images": wandb_images}, step=self.step)
        else:
            display_data(preds, nrows=1, title="MNIST holdout data")

    def train(self):
        if self.args.use_wandb:
            wandb.init(project=args.wandb_project, name=args.wandb_name)
            wandb.watch(self.model)

        self.step = 0

        progress_bar = tqdm(self.dataset_loader, total=len(self.dataset_loader))
        for epoch in range(self.args.epochs):
            for batch_x, _ in progress_bar:
                loss = self.train_step(batch_x.to(device))

                self.step += 1
                progress_bar.set_description(
                    f"ep {epoch}; loss={loss:.4f}, examples={self.step}"
                )

                if self.args.use_wandb:
                    wandb.log(dict(loss=loss), step=self.step)
            self.evaluate()

        if self.args.use_wandb:
            wandb.finish()


args = AutoencoderArgs()
trainer = AutoencoderTrainer(args)
trainer.train()

# %%
tens = torch.randn([512, 1, 28, 28])
# %%
model = Autoencoder(
    hidden_dim_size=args.hidden_dim_size, latent_dim_size=args.latent_dim_size
).to("mps")
pred = model.encoder(HOLDOUT_DATA)
pred.shape
# %%
# Let's try to visualize the effects of latent dimensions


@torch.inference_mode()
def visualize_embeddings(model, n_pts=11, range_span=3):
    # we will visualize the first two dims:
    grid_latent = torch.zeros((n_pts**2, model.latent_dim_size)).to(device)

    xspan = torch.linspace(-range_span, range_span, n_pts)

    grid_latent[:, 0] = einops.repeat(xspan, "dim1 -> (dim1 dim2)", dim2=n_pts)
    grid_latent[:, 1] = einops.repeat(xspan, "dim1 -> (dim2 dim1)", dim2=n_pts)

    output = model.decoder(grid_latent)
    reshaped_to_plot = einops.rearrange(
        output, "(d1 d2) 1 h w -> (d1 h) (d2 w)", d1=n_pts, d2=n_pts
    )

    px.imshow(reshaped_to_plot.to("cpu")).show()

    return grid_latent


grid_latent = visualize_embeddings(trainer.model)

# %%


@t.inference_mode()
def visualise_input(
    model: Autoencoder,
    dataset: Dataset,
) -> None:
    """
    Visualises (in the form of a scatter plot) the input data in the latent space, along the first two dims.
    """
    # First get the model images' latent vectors, along first 2 dims
    imgs = t.stack([batch for batch, label in dataset]).to(device)
    latent_vectors = model.encoder(imgs)
    if latent_vectors.ndim == 3:
        latent_vectors = latent_vectors[0]  # useful for VAEs later
    latent_vectors = latent_vectors[:, :2].cpu().numpy()
    labels = [str(label) for img, label in dataset]

    # Make a dataframe for scatter (px.scatter is more convenient to use when supplied with a dataframe)
    df = pd.DataFrame(
        {"dim1": latent_vectors[:, 0], "dim2": latent_vectors[:, 1], "label": labels}
    )
    df = df.sort_values(by="label")
    fig = px.scatter(df, x="dim1", y="dim2", color="label")
    fig.update_layout(
        height=700,
        width=700,
        title="Scatter plot of latent space dims",
        legend_title="Digit",
    )
    data_range = df["dim1"].max() - df["dim1"].min()

    # Add images to the scatter plot (optional)
    output_on_data_to_plot = model.encoder(HOLDOUT_DATA.to(device))[:, :2].cpu()
    if output_on_data_to_plot.ndim == 3:
        output_on_data_to_plot = output_on_data_to_plot[0]  # useful for VAEs; see later
    data_translated = (HOLDOUT_DATA.cpu().numpy() * 0.3081) + 0.1307
    data_translated = (255 * data_translated).astype(np.uint8).squeeze()
    for i in range(10):
        x, y = output_on_data_to_plot[i]
        fig.add_layout_image(
            source=Image.fromarray(data_translated[i]).convert("L"),
            xref="x",
            yref="y",
            x=x,
            y=y,
            xanchor="right",
            yanchor="top",
            sizex=data_range / 15,
            sizey=data_range / 15,
        )
    fig.show()


small_dataset = Subset(get_dataset("MNIST"), indices=range(0, 5000))
visualise_input(trainer.model, small_dataset)
# %%
# What about variational autoencoders?


class VAE(nn.Module):
    def __init__(self, latent_dim_size: int, hidden_dim_size: int):
        super().__init__()
        # Your code here
        # Input 28 x 28
        self.latent_dim_size = latent_dim_size
        self.hidden_dim_size = hidden_dim_size

        self.encoder = nn.Sequential(
            # Conv kernel 4x4, stride 2, padding 1, channels -> 16
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            # Conv kernel 4x4, stride 2, padding 1, channels 16 -> 32
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            # reshape to linear, then Linear layer with hidden_dim_size
            einops.layers.torch.Rearrange("b c h w -> b (c h w)"),
            nn.Linear(32 * 7 * 7, hidden_dim_size),
            nn.ReLU(),
            # Linear layer with latent_dim_size:
            nn.Linear(hidden_dim_size, latent_dim_size * 2),
        )
        # END OF ENCODER
        self.decoder = nn.Sequential(
            # Linear layer with hidden_dim_size
            nn.Linear(latent_dim_size, hidden_dim_size),
            nn.ReLU(),
            # Linear layer with 32*7*7
            nn.Linear(hidden_dim_size, 32 * 7 * 7),
            nn.ReLU(),
            # reshape to 32x7x7
            einops.layers.torch.Rearrange("b (c h w) -> b c h w", c=32, h=7, w=7),
            # ConvTranspose kernel 4x4, stride 2, padding 1, channels 32 -> 16
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            # ConvTranspose kernel 4x4, stride 2, padding 1, channels 16 -> 1
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
        )

    def sample_latent_vector(self, x: t.Tensor) -> t.Tensor:
        latent_activation = self.encoder(x)
        mean = latent_activation[:, : self.latent_dim_size]
        logsigma = latent_activation[:, self.latent_dim_size :]
        sigma = torch.exp(logsigma)
        z = mean + sigma * torch.randn_like(sigma)
        return z, mean, logsigma

    def forward(self, x: t.Tensor) -> t.Tensor:
        # Your code here
        latent, mean, logsigma = self.sample_latent_vector(x)
        decoded = self.decoder(latent)

        return decoded, mean, logsigma


args = VAEArgs(latent_dim_size=10, hidden_dim_size=100, use_wandb=False)
model = VAE(latent_dim_size=args.latent_dim_size, hidden_dim_size=args.hidden_dim_size)

trainset_mnist = get_dataset("MNIST")
x = next(iter(DataLoader(trainset_mnist, batch_size=512)))[0]
# print(torchinfo.summary(model, input_data=x))
# print(x.shape)
latent, _, _ = model.sample_latent_vector(x)
dec = model.decoder(latent)
# latent.shape
torchinfo.summary(model, input_data=x)
# %%


@dataclass
class VAEArgs(AutoencoderArgs):
    wandb_project: Optional[str] = "day5-vae-mnist"
    beta_kl: float = 0.1


class VAETrainer:
    def __init__(self, args: VAEArgs):
        self.args = args
        self.trainset = get_dataset(args.dataset)
        self.trainloader = DataLoader(
            self.trainset, batch_size=args.batch_size, shuffle=True
        )
        self.model = VAE(
            latent_dim_size=args.latent_dim_size,
            hidden_dim_size=args.hidden_dim_size,
        ).to(device)
        self.optimizer = t.optim.Adam(
            self.model.parameters(), lr=args.lr, betas=args.betas
        )
        self.loss_fun = nn.MSELoss()

    def training_step(self, img: t.Tensor, label: t.Tensor):
        """
        Performs a training step on the batch of images in `img`. Returns the loss.
        """
        img_pred, mean, logsigma = self.model(img)
        sigma = torch.exp(logsigma)
        # print(img_pred.shape, img.shape)
        kl_loss = (((sigma**2 + mean**2 - 1) / 2 - logsigma)).mean() * self.args.beta_kl
        loss = self.loss_fun(img, img_pred) + kl_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, mean, sigma, kl_loss

    @torch.inference_mode()
    def evaluate(self):
        preds, _, _ = self.model(HOLDOUT_DATA)

        if self.args.use_wandb:
            wandb_images = [
                wandb.Image(preds[i, 0, :, :]) for i in range(preds.shape[0])
            ]

            wandb.log({"Images": wandb_images}, step=self.step)
        else:
            display_data(preds, nrows=1, title="MNIST holdout data")

    def train(self) -> None:
        """
        Performs a full training run, optionally logging to wandb.
        """
        self.step = 0
        if self.args.use_wandb:
            wandb.init(project=self.args.wandb_project, name=self.args.wandb_name)

        for epoch in range(self.args.epochs):
            progress_bar = tqdm(self.trainloader, total=int(len(self.trainloader)))

            for img, label in progress_bar:
                loss, mean, sigma, kl_loss = self.training_step(img.to(device), label)

                if self.args.use_wandb:
                    wandb.log(
                        dict(loss=loss, mean=mean, sigma=sigma, kl_loss=kl_loss),
                        step=self.step,
                    )

                progress_bar.set_description(
                    f"{epoch=}, {loss=:.4f}, examples_seen={self.step}"
                )
                self.step += 1
            self.evaluate()

        if self.args.use_wandb:
            wandb.finish()


args = VAEArgs(latent_dim_size=10, hidden_dim_size=100, use_wandb=True, epochs=20)
trainer = VAETrainer(args)
torchinfo.summary(trainer.model, input_data=x.to(device))
# trainer.model.sample_latent_vector(x)
trainer.train()
# %%
small_dataset = Subset(get_dataset("MNIST"), indices=range(0, 5000))
visualise_input(trainer.model, small_dataset)
# %%
grid_latent = visualize_embeddings(trainer.model)
# %%
from sklearn.decomposition import PCA


@t.inference_mode()
def get_pca_components(
    model: Autoencoder,
    dataset: Dataset,
) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    """
    Gets the first 2 dimensions in latent space, from the data.

    Returns:
        pca_vectors: shape (2, latent_dim_size)
            the first 2 principal component vectors in latent space
        principal_components: shape (batch_size, 2)
            components of data along the first 2 dimensions
    """
    # Unpack the (small) dataset into a single batch
    imgs = t.stack([batch[0] for batch in dataset]).to(device)
    labels = t.tensor([batch[1] for batch in dataset])

    # Get the latent vectors
    latent_vectors = model.encoder(imgs.to(device)).cpu().numpy()
    if latent_vectors.ndim == 3:
        latent_vectors = latent_vectors[0]  # useful for VAEs; see later

    # Perform PCA, to get the principle component directions (& projections of data in these directions)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(latent_vectors)
    pca_vectors = pca.components_
    return (
        t.from_numpy(pca_vectors).float(),
        t.from_numpy(principal_components).float(),
    )


pca_vectors, principal_components = get_pca_components(trainer.model, small_dataset)

# Constructing latent dim data by making two of the dimensions vary independently in the interpolation range
interpolation_range = (-3, 3)
n_points = 11
x = t.linspace(*interpolation_range, n_points)
grid_latent = t.stack(
    [
        einops.repeat(x, "dim1 -> dim1 dim2", dim2=n_points),
        einops.repeat(x, "dim2 -> dim1 dim2", dim1=n_points),
    ],
    dim=-1,
)
# Map grid to the basis of the PCA components
grid_latent = grid_latent @ pca_vectors


@t.inference_mode()
def visualise_input(
    model: Autoencoder,
    dataset: Dataset,
) -> None:
    """
    Visualises (in the form of a scatter plot) the input data in the latent space, along the first two latent dims.
    """
    # First get the model images' latent vectors, along first 2 dims
    imgs = t.stack([batch for batch, label in dataset]).to(device)
    latent_vectors = model.encoder(imgs)
    if latent_vectors.ndim == 3:
        latent_vectors = latent_vectors[0]  # useful for VAEs later
    latent_vectors = latent_vectors[:, :2].cpu().numpy()
    labels = [str(label) for img, label in dataset]

    # Make a dataframe for scatter (px.scatter is more convenient to use when supplied with a dataframe)
    df = pd.DataFrame(
        {"dim1": latent_vectors[:, 0], "dim2": latent_vectors[:, 1], "label": labels}
    )
    df = df.sort_values(by="label")
    fig = px.scatter(df, x="dim1", y="dim2", color="label")
    fig.update_layout(
        height=700,
        width=700,
        title="Scatter plot of latent space dims",
        legend_title="Digit",
    )
    data_range = df["dim1"].max() - df["dim1"].min()

    # Add images to the scatter plot (optional)
    output_on_data_to_plot = model.encoder(HOLDOUT_DATA.to(device))
    if output_on_data_to_plot.ndim == 3:
        output_on_data_to_plot = output_on_data_to_plot[0]  # useful for VAEs later
    output_on_data_to_plot = output_on_data_to_plot[:, :2].cpu()
    data_translated = (HOLDOUT_DATA.cpu().numpy() * 0.3081) + 0.1307
    data_translated = (255 * data_translated).astype(np.uint8).squeeze()
    for i in range(10):
        x, y = output_on_data_to_plot[i]
        fig.add_layout_image(
            source=Image.fromarray(data_translated[i]).convert("L"),
            xref="x",
            yref="y",
            x=x,
            y=y,
            xanchor="right",
            yanchor="top",
            sizex=data_range / 15,
            sizey=data_range / 15,
        )
    fig.show()


visualise_input(trainer.model, small_dataset)
# %%
# GANs

class Tanh(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return (t.exp(x) - t.exp(-x)) / (t.exp(x) + t.exp(-x))


tests.test_Tanh(Tanh)


class LeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.max(x, x * self.negative_slope)

    def extra_repr(self) -> str:
        return f"LeakyReLU; negative slope: {self.negative_slope}"

tests.test_LeakyReLU(LeakyReLU)


class Sigmoid(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return 1 / (1 + t.exp(-x))

tests.test_Sigmoid(Sigmoid)
# %%
# Implement the network:

class Generator(nn.Module):

    def __init__(
        self,
        latent_dim_size: int = 100,
        img_size: int = 64,
        img_channels: int = 3,
        hidden_channels: list[int] = [128, 256, 512],
    ):
        '''
        Implements the generator architecture from the DCGAN paper (the diagram at the top
        of page 4). We assume the size of the activations doubles at each layer (so image
        size has to be divisible by 2 ** len(hidden_channels)).

        Args:
            latent_dim_size:
                the size of the latent dimension, i.e. the input to the generator
            img_size:
                the size of the image, i.e. the output of the generator
            img_channels:
                the number of channels in the image (3 for RGB, 1 for grayscale)
            hidden_channels:
                the number of channels in the hidden layers of the generator (starting from
                the smallest / closest to the generated images, and working backwards to the 
                latent vector).

        '''
        n_layers = len(hidden_channels)
        assert img_size % (2 ** n_layers) == 0, "activation size must double at each layer"

        super().__init__()

        c = hidden_channels[-1]
        h = (img_size // 2 ** n_layers)
        w = h

        n_out_features = c * h * w
        self.linear = nn.Linear(in_features=latent_dim_size,
                                out_features=n_out_features, 
                                bias=False)
        
        self.rearranging_pattern = f"b ({c} {h} {w}) -> b {c} {h} {w}"
        
        layers = [BatchNorm2d(num_features=c),
                  ReLU()]
        
        reversed_channels = hidden_channels[::-1]
        prev_n_channels = reversed_channels[0]
        for i, out_hidden_channels in enumerate(reversed_channels[1:]):
            step_n = n_layers - i - 1
            # n_out_features = out_hidden_channels[-1] * (img_size // 2 ** step_n) * (img_size // 2 ** step_n)

            layers.append(nn.ConvTranspose2d(
                                            in_channels=prev_n_channels,
                                            out_channels=out_hidden_channels,
                                            kernel_size=4,
                                            stride=2,
                                            padding=1,
                                            bias=False,
                                        ))
            layers.append(nn.BatchNorm2d(num_features=out_hidden_channels))
            layers.append(ReLU())
            prev_n_channels = out_hidden_channels
        # layers.append(nn.View())

        layers.append(nn.ConvTranspose2d(
                                            in_channels=prev_n_channels,
                                            out_channels=3,
                                            kernel_size=4,
                                            stride=2,
                                            padding=1,
                                            bias=False,
                                        ))
        layers.append(Tanh())

        self.network = nn.Sequential(*layers)


    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.linear(x)
        x = einops.rearrange(x, self.rearranging_pattern)
        return self.network(x)

print_param_count(Generator(), solutions.DCGAN().netG)


# %%
class Discriminator(nn.Module):

    def __init__(
        self,
        img_size: int = 64,
        img_channels: int = 3,
        hidden_channels: list[int] = [128, 256, 512],
    ):
        '''
        Implements the discriminator architecture from the DCGAN paper (the mirror image of
        the diagram at the top of page 4). We assume the size of the activations doubles at
        each layer (so image size has to be divisible by 2 ** len(hidden_channels)).

        Args:
            img_size:
                the size of the image, i.e. the input of the discriminator
            img_channels:
                the number of channels in the image (3 for RGB, 1 for grayscale)
            hidden_channels:
                the number of channels in the hidden layers of the discriminator (starting from
                the smallest / closest to the input image, and working forwards to the probability
                output).
        '''
        n_layers = len(hidden_channels)
        assert img_size % (2 ** n_layers) == 0, "activation size must double at each layer"

        super().__init__()

        # self.sequence = nn.Sequential([
            # Conv kernel 4x4, stride 2, padding 1, channels -> 16

        # activation_functions = [nn.ReLU, nn.ReLU, Tanh]
        layers = []
        in_channels = img_channels
        batch_norms = [False, True, True]
        for n_hidden_channels, batch_norm in zip(hidden_channels, 
                                                          batch_norms):
            layers.append(nn.Conv2d(
                in_channels=in_channels,
                out_channels=n_hidden_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ))
            if batch_norm:
                layers.append(nn.BatchNorm2d(num_features=n_hidden_channels))
            layers.append(LeakyReLU())

            in_channels = n_hidden_channels
        
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_features=n_hidden_channels * (img_size // 2 ** n_layers) * (img_size // 2 ** n_layers), 
                                out_features=1, bias=False))
        
        self.network = nn.Sequential(*layers)

        pass

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.network(x)

print_param_count(Discriminator(), solutions.DCGAN().netD)

# %%

class DCGAN(nn.Module):
    netD: Discriminator
    netG: Generator

    def __init__(
        self,
        latent_dim_size: int = 100,
        img_size: int = 64,
        img_channels: int = 3,
        hidden_channels: list[int] = [128, 256, 512],
    ):
        '''
        Implements the DCGAN architecture from the DCGAN paper (i.e. a combined generator
        and discriminator).
        '''
        super().__init__()
        pass