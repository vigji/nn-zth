# %%
import os
from pyexpat import model
import sys
from altair import layer
from annotated_types import Ge
from click import progressbar
import einops.layers
import torch as t
from torch import nn, optim
import einops
from einops.layers.torch import Rearrange
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


if __name__ == '__main__':
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
# ============================
# GANs
# ============================


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


class GeneratorMine(nn.Module):

    def __init__(
        self,
        latent_dim_size: int = 100,
        img_size: int = 64,
        img_channels: int = 3,
        hidden_channels: list[int] = [128, 256, 512],
    ):
        """
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

        """
        n_layers = len(hidden_channels)
        assert (
            img_size % (2**n_layers) == 0
        ), "activation size must double at each layer"

        super().__init__()

        c = hidden_channels[-1]
        h = img_size // 2**n_layers
        w = h

        n_out_features = c * h * w
        self.linear_and_rearrange = nn.Sequential(
            *[
                nn.Linear(
                    in_features=latent_dim_size, out_features=n_out_features, bias=False
                ),
                einops.layers.torch.Rearrange("b (c w h) -> b c w h", c=c, h=h, w=w),
                BatchNorm2d(num_features=c),
                ReLU(),
            ]
        )

        layers = []

        reversed_channels = hidden_channels[::-1]
        prev_n_channels = reversed_channels[0]

        for out_hidden_channels in reversed_channels[1:]:

            layers.append(
                nn.ConvTranspose2d(
                    in_channels=prev_n_channels,
                    out_channels=out_hidden_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(num_features=out_hidden_channels))
            layers.append(ReLU())
            prev_n_channels = out_hidden_channels

        layers.append(
            nn.ConvTranspose2d(
                in_channels=prev_n_channels,
                out_channels=img_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
        )
        layers.append(Tanh())

        self.network = nn.Sequential(*layers)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.linear_and_rearrange(x)
        return self.network(x)


print_param_count(Generator(), solutions.DCGAN().netG)


# %%
class DiscriminatorMine(nn.Module):

    def __init__(
        self,
        img_size: int = 64,
        img_channels: int = 3,
        hidden_channels: list[int] = [128, 256, 512],
    ):
        """
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
        """
        n_layers = len(hidden_channels)
        assert (
            img_size % (2**n_layers) == 0
        ), "activation size must double at each layer"

        super().__init__()

        # self.sequence = nn.Sequential([
        # Conv kernel 4x4, stride 2, padding 1, channels -> 16

        # activation_functions = [nn.ReLU, nn.ReLU, Tanh]
        layers = []
        in_channels = img_channels
        batch_norms = [False, True, True]
        for n_hidden_channels, batch_norm in zip(hidden_channels, batch_norms):
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=n_hidden_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            if batch_norm:
                layers.append(nn.BatchNorm2d(num_features=n_hidden_channels))
            layers.append(LeakyReLU())

            in_channels = n_hidden_channels

        layers.append(nn.Flatten())
        layers.append(
            nn.Linear(
                in_features=n_hidden_channels
                * (img_size // 2**n_layers)
                * (img_size // 2**n_layers),
                out_features=1,
                bias=False,
            )
        )

        self.network = nn.Sequential(*layers)

        pass

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.network(x)


print_param_count(Discriminator(), solutions.DCGAN().netD)

# %%


class DCGANMine(nn.Module):
    netD: Discriminator
    netG: Generator

    def __init__(
        self,
        latent_dim_size: int = 100,
        img_size: int = 64,
        img_channels: int = 3,
        hidden_channels: list[int] = [128, 256, 512],
    ):
        """
        Implements the DCGAN architecture from the DCGAN paper (i.e. a combined generator
        and discriminator).
        """
        super().__init__()

        self.netD = Discriminator(
            img_size=img_size,
            img_channels=img_channels,
            hidden_channels=hidden_channels,
        )
        self.netG = Generator(
            latent_dim_size=latent_dim_size,
            img_channels=img_channels,
            img_size=img_size,
            hidden_channels=hidden_channels,
        )

        solutions.initialize_weights(self)


model = DCGAN().to(device)
x = t.randn(3, 100).to(device)
print(torchinfo.summary(model.netG, input_data=x), end="\n\n")
print(torchinfo.summary(model.netD, input_data=model.netG(x)))


#%%

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
                the number of channels in the hidden layers of the generator (starting closest
                to the middle of the DCGAN and going outward, i.e. in chronological order for
                the generator)
        '''
        n_layers = len(hidden_channels)
        assert img_size % (2 ** n_layers) == 0, "activation size must double at each layer"

        super().__init__()

        # Reverse hidden channels, so they're in chronological order
        hidden_channels = hidden_channels[::-1]

        self.latent_dim_size = latent_dim_size
        self.img_size = img_size
        self.img_channels = img_channels
        # Reverse them, so they're in chronological order for generator
        self.hidden_channels = hidden_channels

        # Define the first layer, i.e. latent dim -> (512, 4, 4) and reshape
        first_height = img_size // (2 ** n_layers)
        first_size = hidden_channels[0] * (first_height ** 2)
        self.project_and_reshape = Sequential(
            Linear(latent_dim_size, first_size, bias=False),
            Rearrange("b (ic h w) -> b ic h w", h=first_height, w=first_height),
            BatchNorm2d(hidden_channels[0]),
            ReLU(),
        )

        # Equivalent, but using conv rather than linear:
        # self.project_and_reshape = Sequential(
        #     Rearrange("b ic -> b ic 1 1"),
        #     solutions.ConvTranspose2d(latent_dim_size, hidden_channels[0], first_height, 1, 0),
        #     BatchNorm2d(hidden_channels[0]),
        #     ReLU(),
        # )

        # Get list of input & output channels for the convolutional blocks
        in_channels = hidden_channels
        out_channels = hidden_channels[1:] + [img_channels]

        # Define all the convolutional blocks (conv_transposed -> batchnorm -> activation)
        conv_layer_list = []
        for i, (c_in, c_out) in enumerate(zip(in_channels, out_channels)):
            conv_layer = [
                solutions.ConvTranspose2d(c_in, c_out, 4, 2, 1),
                ReLU() if i < n_layers - 1 else Tanh()
            ]
            if i < n_layers - 1:
                conv_layer.insert(1, BatchNorm2d(c_out))
            conv_layer_list.append(Sequential(*conv_layer))

        self.hidden_layers = Sequential(*conv_layer_list)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.project_and_reshape(x)
        x = self.hidden_layers(x)
        return x


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
                the number of channels in the hidden layers of the discriminator (starting
                closest to the middle of the DCGAN and going outward, i.e. in reverse-
                chronological order for the discriminator)
        '''
        n_layers = len(hidden_channels)
        assert img_size % (2 ** n_layers) == 0, "activation size must double at each layer"

        super().__init__()

        self.img_size = img_size
        self.img_channels = img_channels
        self.hidden_channels = hidden_channels

        # Get list of input & output channels for the convolutional blocks
        in_channels = [img_channels] + hidden_channels[:-1]
        out_channels = hidden_channels

        # Define all the convolutional blocks (conv_transposed -> batchnorm -> activation)
        conv_layer_list = []
        for i, (c_in, c_out) in enumerate(zip(in_channels, out_channels)):
            conv_layer = [
                Conv2d(c_in, c_out, 4, 2, 1),
                LeakyReLU(0.2),
            ]
            if i > 0:
                conv_layer.insert(1, BatchNorm2d(c_out))
            conv_layer_list.append(Sequential(*conv_layer))

        self.hidden_layers = Sequential(*conv_layer_list)

        # Define the last layer, i.e. reshape and (512, 4, 4) -> real/fake classification
        final_height = img_size // (2 ** n_layers)
        final_size = hidden_channels[-1] * (final_height ** 2)
        self.classifier = Sequential(
            Rearrange("b c h w -> b (c h w)"),
            Linear(final_size, 1, bias=False),
            Sigmoid(),
        )
        # Equivalent, but using conv rather than linear:
        # self.classifier = Sequential(
        #     Conv2d(out_channels[-1], 1, final_height, 1, 0),
        #     Rearrange("b c h w -> b (c h w)"),
        #     Sigmoid(),
        # )

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.hidden_layers(x)
        x = self.classifier(x)
        return x.squeeze() # remove dummy out_channels dimension


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
        super().__init__()
        self.latent_dim_size = latent_dim_size
        self.img_size = img_size
        self.img_channels = img_channels
        self.img_channels = img_channels
        self.hidden_channels = hidden_channels
        self.netD = Discriminator(img_size, img_channels, hidden_channels)
        self.netG = Generator(latent_dim_size, img_size, img_channels, hidden_channels)
        initialize_weights(self) # see next section for this

# %%
def initialize_weights_mine(model: nn.Module) -> None:
    """
    Initializes weights according to the DCGAN paper, by modifying model weights in place.
    """
    for module in model.modules():
        if any(
            [
                isinstance(module, m)
                for m in [
                    Conv2d,
                    nn.Conv2d,
                    nn.ConvTranspose2d,
                    solutions.ConvTranspose2d,
                    nn.Linear,
                    Linear,
                ]
            ]
        ):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif any([isinstance(module, m) for m in [BatchNorm2d, nn.BatchNorm2d]]):
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0.0)

def initialize_weights(model: nn.Module) -> None:
    '''
    Initializes weights according to the DCGAN paper (details at the end of
    page 3), by modifying the weights of the model in place.
    '''
    for (name, module) in model.named_modules():
        if any([
            isinstance(module, Module)
            for Module in [solutions.ConvTranspose2d, Conv2d, Linear]
        ]):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif isinstance(module, BatchNorm2d):
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0.0)

tests.test_initialize_weights(
    initialize_weights, solutions.ConvTranspose2d, Conv2d, Linear, BatchNorm2d
)
    # else:
    #     print("Not initializing ", module)
# %%

@dataclass
class DCGANArgs():
    '''
    Class for the arguments to the DCGAN (training and architecture).
    Note, we use field(defaultfactory(...)) when our default value is a mutable object.
    '''
    # architecture
    latent_dim_size: int = 100
    # Trick to initialize without pointing to always the same list!
    hidden_channels: list[int] = field(default_factory=lambda: [128, 256, 512])

    # data & training
    dataset: Literal["MNIST", "CELEB"] = "CELEB"
    batch_size: int = 8
    epochs: int = 1
    lr: float = 0.0002
    betas: tuple[float, float] = (0.5, 0.999)
    clip_grad_norm: float | None = 1.0

    # logging
    use_wandb: bool = False
    wandb_project: str | None = "day5-gan"
    wandb_name: str | None = None

args = DCGANArgs(
    dataset="MNIST",
    hidden_channels=[8, 16],
    epochs=10,
    batch_size=128,
)

# %%


class DCGANTrainer:
    def __init__(self, args: DCGANArgs):
        self.args = args

        self.trainset = get_dataset(self.args.dataset)
        self.trainloader = DataLoader(self.trainset, batch_size=args.batch_size, 
                                      shuffle=True)

        batch, img_channels, img_height, img_width = next(iter(self.trainloader))[0].shape
        assert img_height == img_width

        self.model = DCGAN(
            args.latent_dim_size,
            img_height,
            img_channels,
            args.hidden_channels,
        ).to(device).train()

        self.optG = t.optim.Adam(self.model.netG.parameters(), lr=args.lr, betas=args.betas)
        self.optD = t.optim.Adam(self.model.netD.parameters(), lr=args.lr, betas=args.betas)


    def training_step_discriminator(self, img_real: t.Tensor, img_fake: t.Tensor) -> t.Tensor:
        '''
        Generates a real and fake image, and performs a gradient step on the discriminator 
        to maximize log(D(x)) + log(1-D(G(z))).
        '''
        
        self.optD.zero_grad()

        d_g_z = self.model.netD(img_fake)
        d_x = self.model.netD(img_real)

        mean_log_d_g_z = t.mean(t.log(1 - d_g_z), axis=0)
        mean_log_g_x = t.mean(t.log(d_x), axis=0)

        loss = - (mean_log_d_g_z + mean_log_g_x)

        loss.backward()

        nn.utils.clip_grad_norm_(self.model.netD.parameters(), self.args.clip_grad_norm)

        self.optD.step()

        return loss.item()


    def training_step_generator(self, img_fake: t.Tensor) -> t.Tensor:
        '''
        Performs a gradient step on the generator to maximize log(D(G(z))).
        '''
        self.optG.zero_grad()

        D_G_z = self.model.netD(img_fake)

        # Calculating loss with clamping behaviour:
        # labels_real = t.ones_like(D_G_z)
        # loss = nn.BCELoss()(D_G_z, labels_real)
        loss = - (t.log(D_G_z).mean())

        loss.backward()
        
        nn.utils.clip_grad_norm_(self.model.netG.parameters(), self.args.clip_grad_norm)

        self.optG.step()

        return loss.item()


    @t.inference_mode()
    def evaluate(self) -> None:
        '''
        Performs evaluation by generating 8 instances of random noise and passing them through
        the generator, then either logging the results to Weights & Biases or displaying them inline.
        '''
        pass


    def train(self) -> None:
        '''
        Performs a full training run, while optionally logging to Weights & Biases.
        '''
        self.step = 0
        if self.args.use_wandb:
            # print("logging")

            wandb.init(project=self.args.wandb_project, name=self.args.wandb_name)

        for epoch in range(self.args.epochs):

            progress_bar = tqdm(self.trainloader, total=len(self.trainloader))

            for (img_real, label) in progress_bar:
                # Generate random noise & fake image
                noise = t.randn(self.args.batch_size, self.args.latent_dim_size).to(device)
                img_real = img_real.to(device)
                img_fake = self.model.netG(noise)

                # Training steps
                lossD = self.training_step_discriminator(img_real, img_fake.detach())
                lossG = self.training_step_generator(img_fake)

                # Log data
                if self.args.use_wandb:
                    # print("logging")
                    wandb.log(dict(lossD=lossD, lossG=lossG), step=self.step)

                # Update progress bar
                self.step += img_real.shape[0]
                progress_bar.set_description(f"{epoch=}, lossD={lossD:.4f}, lossG={lossG:.4f}, examples_seen={self.step}")

            # Evaluate model on the same batch of random data
            self.evaluate()

        if self.args.use_wandb:
            # print("logging")

            wandb.finish()


# Arguments for MNIST
args = DCGANArgs(
    dataset="MNIST",
    hidden_channels=[8, 16],
    epochs=10,
    batch_size=128,
    use_wandb=True
)
trainer = DCGANTrainer(args)
trainer.train()

# %%

# Arguments for CelebA
args = DCGANArgs(
    dataset="CELEB",
    hidden_channels=[128, 256, 512],
    batch_size=32, # if you get cuda errors, bring this down!
    epochs=5,
)
trainer = DCGANTrainer(args)
trainer.train()

# %%
s = t.Tensor([1])
s.item()
# %%
args
# %%
