# %%
import os
import functools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import einops as ein
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

        self.biases = (
            nn.Parameter(t.rand(out_features) * kai_he_fact * 2 - kai_he_fact)
            if bias
            else None
        )

        self.weights = nn.Parameter(
            t.rand(out_features, in_features) * kai_he_fact * 2 - kai_he_fact
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (*, in_features)
        Return: shape (*, out_features)
        """
        out = t.einsum(self.weights, x, "ij, j -> i")
        if self.biases:
            out += self.bias
        return out

    def extra_repr(self) -> str:
        pass


tests.test_linear_parameters(Linear, bias=False)
tests.test_linear_parameters(Linear, bias=True)
tests.test_linear_forward(Linear, bias=False)
tests.test_linear_forward(Linear, bias=True)

# %%
in_features = 9
kaiming_he_factor = 1 / (in_features**1 / 2)
print(kaiming_he_factor)
biases = t.rand(1000) * kaiming_he_factor * 2 - kaiming_he_factor
biases.min(), biases.max()
# %%
