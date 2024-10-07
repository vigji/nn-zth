# %%
import os
import functools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

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

MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def get_mnist(subset: int = 1):
    '''Returns MNIST training data, sampled by the frequency given in `subset`.'''
    mnist_trainset = datasets.MNIST(root="./data", train=True, download=True, transform=MNIST_TRANSFORM)
    mnist_testset = datasets.MNIST(root="./data", train=False, download=True, transform=MNIST_TRANSFORM)

    if subset > 1:
        mnist_trainset = Subset(mnist_trainset, indices=range(0, len(mnist_trainset), subset))
        mnist_testset = Subset(mnist_testset, indices=range(0, len(mnist_testset), subset))

    return mnist_trainset, mnist_testset


mnist_trainset, mnist_testset = get_mnist()
mnist_trainloader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)
mnist_testloader = DataLoader(mnist_testset, batch_size=64, shuffle=False)
# %%
mnist_trainloader
# %%
# Training loop: add test accuracy
@dataclass
class SimpleMLPTrainingArgs():
    '''
    Defining this class implicitly creates an __init__ method, which sets arguments as 
    given below, e.g. self.batch_size = 64. Any of these arguments can also be overridden
    when you create an instance, e.g. args = SimpleMLPTrainingArgs(batch_size=128).
    '''
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 1e-3
    subset: int = 10


def train(args: SimpleMLPTrainingArgs):
    '''
    Trains the model, using training parameters from the `args` object.
    '''
    model = SimpleMLP().to(device)

    mnist_trainset, mnist_testset = get_mnist(subset=args.subset)
    mnist_trainloader = DataLoader(mnist_trainset, batch_size=args.batch_size, shuffle=True)
    mnist_testloader = DataLoader(mnist_testset, batch_size=args.batch_size, shuffle=False)
    
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
                
                all_preds.append((logits.argmax(axis=1) == labels).sum() / labels.shape[0])
                # accuracy = (logits.argmax(axis=1) == labels) / len(labels)
                #accuracies_list_test.append(accuracy)
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


args = SimpleMLPTrainingArgs()
train(args)
# %%
accuracies_list_test.shape
# %%
