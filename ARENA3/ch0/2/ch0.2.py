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
        print(self.weight.shape, x.shape)
        if len(x.shape > 1):
            out = einops.einsum(self.weight, x, "i j, b j -> b i")
        # else:
        #     out = einops.einsum(self.weight, x, "i j, b j -> b i")
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
