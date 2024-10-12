# %%
import os
import functools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
import math
from unicodedata import numeric

import einops
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display
from jaxtyping import Float, Int
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm.notebook import tqdm

import tests as tests
from utils import print_param_count
from plotly_utils import line

device = t.device(
    "mps"
    if t.backends.mps.is_available()
    else "cuda"
    if t.cuda.is_available()
    else "cpu"
)
# %%
# Implement ReLU:


class ReLU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, val: Tensor) -> Tensor:
        return t.maximum(val, t.tensor(0.0))


tests.test_relu(ReLU)

# %%
# Implement Linear:


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        """
        A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        """
        super().__init__()
        kai_he_fact = 1 / (in_features**1 / 2)

        self.weight = nn.Parameter(
            t.rand(out_features, in_features) * kai_he_fact * 2 - kai_he_fact
        )

        self.bias = (
            nn.Parameter(t.rand(out_features) * kai_he_fact * 2 - kai_he_fact)
            if bias
            else None
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (*, in_features)
        Return: shape (*, out_features)
        """
        if len(x.shape) > 1:
            out = einops.einsum(self.weight, x, "i j, b j -> b i")

        if self.bias is not None:
            out += self.bias
        return out

    def extra_repr(self) -> str:
        return f"weights={self.weights}, biases={self.biases}, ..."


tests.test_linear_parameters(Linear, bias=False)
tests.test_linear_parameters(Linear, bias=True)
tests.test_linear_forward(Linear, bias=False)
tests.test_linear_forward(Linear, bias=True)

# %%
# Flatten


class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: t.Tensor) -> t.Tensor:
        """
        Flatten out dimensions from start_dim to end_dim, inclusive of both.
        """
        end_dim = (
            self.end_dim + 1
            if self.end_dim > 0
            else (len(input.shape) + self.end_dim + 1)
        )

        new_shape = [
            -1,
        ] * len(input.shape)
        new_shape[: self.start_dim] = input.shape[: self.start_dim]
        new_shape[end_dim:] = input.shape[end_dim:]

        for i in range(len(new_shape) - 1, 0, -1):
            if new_shape[i - 1] == -1 and new_shape[i] == -1:
                new_shape.pop(i)

        return input.reshape(new_shape)

    def extra_repr(self) -> str:
        return f"start_dim={self.start_dim}, end_dim={self.end_dim}, ..."


tests.test_flatten(Flatten)


# %%
## Implement MLP
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()

        n_inputs = 28**2
        n_hidden_units = 100
        n_outputs = 10

        self.flattener = Flatten(start_dim=1, end_dim=-1)
        self.linear1 = Linear(n_inputs, n_hidden_units, bias=True)
        self.linear2 = Linear(n_hidden_units, n_outputs, bias=True)

        self.relu = ReLU()

    def forward(self, x: t.Tensor) -> t.Tensor:
        flat_x = self.flattener(x)
        hidden_activations = self.relu(self.linear1(flat_x))
        final_activations = self.linear2(hidden_activations)

        return final_activations


tests.test_mlp_module(SimpleMLP)
tests.test_mlp_forward(SimpleMLP)
# %%

# Part 2: Training neural networks

MNIST_TRANSFORM = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


def get_mnist(subset: int = 1):
    """Returns MNIST training data, sampled by the frequency given in `subset`."""
    mnist_trainset = datasets.MNIST(
        root="./data", train=True, download=True, transform=MNIST_TRANSFORM
    )
    mnist_testset = datasets.MNIST(
        root="./data", train=False, download=True, transform=MNIST_TRANSFORM
    )

    if subset > 1:
        mnist_trainset = Subset(
            mnist_trainset, indices=range(0, len(mnist_trainset), subset)
        )
        mnist_testset = Subset(
            mnist_testset, indices=range(0, len(mnist_testset), subset)
        )

    return mnist_trainset, mnist_testset


mnist_trainset, mnist_testset = get_mnist()
mnist_trainloader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)
mnist_testloader = DataLoader(mnist_testset, batch_size=64, shuffle=False)
# %%
mnist_trainloader


# %%
# Training loop: add test accuracy
@dataclass
class SimpleMLPTrainingArgs:
    """
    Defining this class implicitly creates an __init__ method, which sets arguments as
    given below, e.g. self.batch_size = 64. Any of these arguments can also be overridden
    when you create an instance, e.g. args = SimpleMLPTrainingArgs(batch_size=128).
    """

    batch_size: int = 64
    epochs: int = 3
    learning_rate: float = 1e-3
    subset: int = 10


def train(args: SimpleMLPTrainingArgs):
    """
    Trains the model, using training parameters from the `args` object.
    """
    model = SimpleMLP().to(device)

    mnist_trainset, mnist_testset = get_mnist(subset=args.subset)
    mnist_trainloader = DataLoader(
        mnist_trainset, batch_size=args.batch_size, shuffle=True
    )
    mnist_testloader = DataLoader(
        mnist_testset, batch_size=args.batch_size, shuffle=False
    )

    optimizer = t.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_list = []
    accuracies_list_test = []

    for epoch in tqdm(range(args.epochs)):
        for imgs, labels in mnist_trainloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_list.append(loss.item())

        with t.inference_mode():
            all_preds = []
            for imgs, labels in mnist_testloader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                logits = model(imgs)

                all_preds.append(
                    (logits.argmax(axis=1) == labels).sum() / labels.shape[0]
                )
                # accuracy = (logits.argmax(axis=1) == labels) / len(labels)
                # accuracies_list_test.append(accuracy)
            accuracies_list_test.append(sum(all_preds) / len(all_preds))

    line(
        loss_list,
        yaxis_range=[0, max(loss_list) + 0.1],
        x=t.linspace(0, args.epochs, len(loss_list)),
        labels={"x": "Num epochs", "y": "Cross entropy loss"},
        title="SimpleMLP training on MNIST",
        width=700,
    )
    line(
        accuracies_list_test,
        # yaxis_range=[0, max(loss_list) + 0.1],
        x=t.linspace(0, args.epochs, len(accuracies_list_test)),
        labels={"x": "Num epochs", "y": "Epoch accuracy"},
        title="SimpleMLP training on MNIST",
        width=700,
    )

    return np.array([t.cpu() for t in accuracies_list_test])


args = SimpleMLPTrainingArgs()
# Play with averaging:
# accuracies_list_test_list = []
# for i in range(30):
#     accuracies_list_test_list.append(train(args))
# accuracies_list_test_arr = np.array([t for t in accuracies_list_test_list])
# accuracies_list_test_list_mn = np.nanmean(accuracies_list_test_arr, 0)
# line(
#     accuracies_list_test_list_mn,
#     # yaxis_range=[0, max(loss_list) + 0.1],
#     x=t.linspace(0, args.epochs, len(accuracies_list_test_list_mn)),
#     labels={"x": "Num epochs", "y": "Epoch accuracy"},
#     title="SimpleMLP training on MNIST",
#     width=700,
# )
# %%
# Convolutions


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        """
        Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.
        """
        # Xavier initialization, from PyTorch implementation:
        groups = 1  # we assume 1 group
        k = groups / (in_channels * kernel_size**2)
        super().__init__()
        self.weight = nn.Parameter(
            t.rand((out_channels, in_channels // groups, kernel_size, kernel_size))
            * 2
            * math.sqrt(k)
            - math.sqrt(k)
        )
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Apply the functional conv2d, which you can import."""
        return t.nn.functional.conv2d(
            x, self.weight, stride=self.stride, padding=self.padding
        )

    def extra_repr(self) -> str:
        return f"Conv2d layer (kernek_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


# %%


tests.test_conv2d_module(Conv2d)
m = Conv2d(in_channels=24, out_channels=12, kernel_size=3, stride=2, padding=1)
print(f"Manually verify that this is an informative repr: {m}")
# %%


class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: int, stride: int | None = None, padding: int = 1):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Call the functional version of max_pool2d."""
        return t.nn.functional.max_pool2d(
            x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding
        )

    def extra_repr(self) -> str:
        """Add additional information to the string representation of this class."""
        return f"MaxPool2d layer (kernek_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


tests.test_maxpool2d_module(MaxPool2d)
m = MaxPool2d(kernel_size=3, stride=2, padding=1)
print(f"Manually verify that this is an informative repr: {m}")
# %% ResNets
from collections import OrderedDict


class Sequential(nn.Module):
    _modules: dict[str, nn.Module]

    def __init__(self, *modules: nn.Module):
        super().__init__()
        for index, mod in enumerate(modules):
            self._modules[str(index)] = mod

    def __getitem__(self, index: int) -> nn.Module:
        index %= len(self._modules)  # deal with negative indices
        return self._modules[str(index)]

    def __setitem__(self, index: int, module: nn.Module) -> None:
        index %= len(self._modules)  # deal with negative indices
        self._modules[str(index)] = module

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Chain each module together, with the output from one feeding into the next one."""
        for mod in self._modules.values():
            x = mod(x)
        return x


# %%


class BatchNorm2d(nn.Module):
    # The type hints below aren't functional, they're just for documentation
    running_mean: Float[Tensor, "num_features"]
    running_var: Float[Tensor, "num_features"]
    num_batches_tracked: Int[Tensor, ""]  # This is how we denote a scalar tensor

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        """
        Like nn.BatchNorm2d with track_running_stats=True and affine=True.

        Name the learnable affine parameters `weight` and `bias` in that order.
        """
        super().__init__()

        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features

        self.weight = nn.Parameter(t.ones(num_features))
        self.bias = nn.Parameter(t.zeros(num_features))

        self.register_buffer("running_mean", t.zeros((num_features)))
        self.register_buffer("running_var", t.ones((num_features)))
        self.register_buffer("num_batches_tracked", t.zeros(()))

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        Normalize each channel.

        Compute the variance using `torch.var(x, unbiased=False)`
        Hint: you may also find it helpful to use the argument `keepdim`.

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        """
        if self.training:
            self.num_batches_tracked += 1

            var = t.var(x, dim=(0, 2, 3), unbiased=True, keepdim=True)
            mean = t.mean(x, dim=(0, 2, 3), keepdim=True)

            running_var = (1 - self.momentum) * self.running_var[
                t.newaxis, :, t.newaxis, t.newaxis
            ] + self.momentum * var

            running_mean = (1 - self.momentum) * self.running_mean[
                t.newaxis, :, t.newaxis, t.newaxis
            ] + self.momentum * mean

            self.running_var = t.squeeze(running_var)
            self.running_mean = t.squeeze(running_mean)

        else:
            mean = einops.rearrange(self.running_mean, "ch -> 1 ch 1 1")
            var = einops.rearrange(self.running_var, "ch -> 1 ch 1 1")

        weight = einops.rearrange(self.weight, "ch -> 1 ch 1 1")
        bias = einops.rearrange(self.bias, "ch -> 1 ch 1 1")

        return ((x - mean) / t.sqrt(var + self.eps)) * weight + bias

    def extra_repr(self) -> str:
        return f"BarchNorm2d (weight={self.weight}; bias={self.bias})"


# %%
tests.test_batchnorm2d_module(BatchNorm2d)
tests.test_batchnorm2d_forward(BatchNorm2d)
tests.test_batchnorm2d_running_mean(BatchNorm2d)


# %%
# AveragePool


class AveragePool(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        """
        super().__init__()

        return t.mean(x, dim=(2, 3))


# %% ResNet implementation


class ResidualBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        """
        A single residual block with optional downsampling.

        For compatibility with the pretrained model, declare the left side branch first using a `Sequential`.

        If first_stride is > 1, this means the optional (conv + bn) should be present on the right branch. Declare it second using another `Sequential`.
        """
        super().__init__()

        self.left = Sequential(
            Conv2d(
                in_channels=in_feats,
                out_channels=out_feats,
                stride=first_stride,
                kernel_size=3,
                padding=1,
            ),
            BatchNorm2d(num_features=out_feats),
            ReLU(),
            Conv2d(
                in_channels=out_feats,
                out_channels=out_feats,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            BatchNorm2d(num_features=out_feats),
        )
        if first_stride == 1 and in_feats == out_feats:
            self.right = Sequential(nn.Identity())
        else:
            self.right = Sequential(
                Conv2d(
                    stride=first_stride,
                    in_channels=in_feats,
                    out_channels=out_feats,
                    kernel_size=1,
                ),
                BatchNorm2d(num_features=out_feats),
            )

        self.relu = ReLU()

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / stride, width / stride)

        If no downsampling block is present, the addition should just add the left branch's output to the input.
        """

        return self.relu(self.left(x) + self.right(x))


# %%
class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
        """An n_blocks-long sequence of ResidualBlock where only the first block uses the provided stride."""
        super().__init__()

        block_list = [
            ResidualBlock(
                in_feats=in_feats, out_feats=out_feats, first_stride=first_stride
            )
        ]
        for _ in range(n_blocks - 1):
            block_list.append(
                ResidualBlock(in_feats=out_feats, out_feats=out_feats, first_stride=1)
            )

        self.blocks = Sequential(*block_list)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        """

        return self.blocks(x)


# %%


class ResNet34(nn.Module):
    def __init__(
        self,
        n_blocks_per_group=[3, 4, 6, 3],
        out_features_per_group=[64, 128, 256, 512],
        first_strides_per_group=[1, 2, 2, 2],
        n_classes=1000,
    ):
        super().__init__()
        img_ch = 3
        in_feats0 = 64
        kernel_size0 = 7

        self.n_blocks_per_group = n_blocks_per_group
        self.out_features_per_group = out_features_per_group
        self.first_strides_per_group = first_strides_per_group
        self.n_classes = n_classes

        self.in_layers = Sequential(
            Conv2d(
                in_channels=img_ch,
                out_channels=in_feats0,
                kernel_size=kernel_size0,
                padding=3,
                stride=2,
            ),
            BatchNorm2d(num_features=64),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2),
        )

        block_list = []
        in_feats = in_feats0
        for n_blocks, out_features, first_stride in zip(
            n_blocks_per_group, out_features_per_group, first_strides_per_group
        ):
            block_list.append(
                BlockGroup(
                    n_blocks=n_blocks,
                    in_feats=in_feats,
                    out_feats=out_features,
                    first_stride=first_stride,
                )
            )
            in_feats = out_features
        self.residual_layers = Sequential(*block_list)

        self.out_layers = Sequential(
            AveragePool(),
            Linear(in_features=out_features_per_group[-1], out_features=n_classes),
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, channels, height, width)
        Return: shape (batch, n_classes)
        """
        x = self.in_layers(x)
        x = self.residual_layers(x)
        x = self.out_layers(x)

        return x


my_resnet = ResNet34()

# %%
# Try to copy weights:


def copy_weights(
    my_resnet: ResNet34, pretrained_resnet: models.resnet.ResNet
) -> ResNet34:
    """Copy over the weights of `pretrained_resnet` to your resnet."""

    # Get the state dictionaries for each model, check they have the same number of parameters & buffers
    mydict = my_resnet.state_dict()
    pretraineddict = pretrained_resnet.state_dict()
    assert len(mydict) == len(pretraineddict), "Mismatching state dictionaries."

    # Define a dictionary mapping the names of your parameters / buffers to their values in the pretrained model
    state_dict_to_load = {
        mykey: pretrainedvalue
        for (mykey, myvalue), (pretrainedkey, pretrainedvalue) in zip(
            mydict.items(), pretraineddict.items()
        )
    }

    # Load in this dictionary to your model
    my_resnet.load_state_dict(state_dict_to_load)

    return my_resnet


pretrained_resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
my_resnet = copy_weights(my_resnet, pretrained_resnet)

print_param_count(pretrained_resnet, my_resnet)
# %%

IMAGE_FILENAMES = [
    "chimpanzee.jpg",
    "golden_retriever.jpg",
    "platypus.jpg",
    "frogs.jpg",
    "fireworks.jpg",
    "astronaut.jpg",
    "iguana.jpg",
    "volcano.jpg",
    "goofy.jpg",
    "dragonfly.jpg",
]

IMAGE_FOLDER = Path(__file__).parent / "resnet_inputs"

images = [Image.open(IMAGE_FOLDER / filename) for filename in IMAGE_FILENAMES]
# %%
images[0]
# %%
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGENET_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

prepared_images = t.stack([IMAGENET_TRANSFORM(img) for img in images], dim=0)

assert prepared_images.shape == (len(images), 3, IMAGE_SIZE, IMAGE_SIZE)

# %%

# %%
# Predict sample dataset


def predict(model, images: t.Tensor) -> t.Tensor:
    """
    Returns the predicted class for each image (as a 1D array of ints).
    """
    predictions = my_resnet(prepared_images)
    return predictions.argmax(1)
    # predicted_labels = [imagenet_labels[i] for i in best_predictions]


# %%
with open(Path(__file__).parent / "imagenet_labels.json") as f:
    imagenet_labels = list(json.load(f).values())
imagenet_labels
# %%
# Check your predictions match those of the pretrained model
my_predictions = predict(my_resnet, prepared_images)
pretrained_predictions = predict(pretrained_resnet, prepared_images)
assert all(my_predictions == pretrained_predictions)
print("All predictions match!")

# Print out your predictions, next to the corresponding images
for img, label in zip(images, my_predictions):
    print(f"Class {label}: {imagenet_labels[label]}")
    display(img)
    print()


# %%
# Aside: hooks
class NanModule(nn.Module):
    """
    Define a module that always returns NaNs (we will use hooks to identify this error).
    """

    def forward(self, x):
        return t.full_like(x, float("nan"))


model = nn.Sequential(nn.Identity(), NanModule(), nn.Identity())


def hook_check_for_nan_output(
    module: nn.Module, input: tuple[t.Tensor], output: t.Tensor
) -> None:
    """
    Hook function which detects when the output of a layer is NaN.
    """
    if t.isnan(output).any():
        raise ValueError(f"NaN output from {module}")


def add_hook(module: nn.Module) -> None:
    """
    Register our hook function in a module.

    Use model.apply(add_hook) to recursively apply the hook to model and all submodules.
    """
    module.register_forward_hook(hook_check_for_nan_output)


def remove_hooks(module: nn.Module) -> None:
    """
    Remove all hooks from module.

    Use module.apply(remove_hooks) to do this recursively.
    """
    module._backward_hooks.clear()
    module._forward_hooks.clear()
    module._forward_pre_hooks.clear()


model = model.apply(add_hook)
input = t.randn(3)

try:
    output = model(input)
except ValueError as e:
    print(e)

model = model.apply(remove_hooks)
# %%
