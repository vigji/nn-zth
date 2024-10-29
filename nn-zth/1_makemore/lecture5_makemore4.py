# %%
# Let's improve our generator in the direction of Bangio et al 2006
from matplotlib import cm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import einops as ein
import numpy as np

names_file = Path(__file__).parent / "names.txt"
words = open(names_file).read().splitlines()
PAD_CH = "."
device = "cpu"


n_possible_chars = 27
# N = torch.zeros((possible_chars, possible_chars), dtype=torch.int32)
chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi[PAD_CH] = 0
itos = {s: i for i, s in stoi.items()}

import random

random.seed(42)
random.shuffle(words)

X, Y = [], []
for word in words:
    context = [PAD_CH] * block_size
    for ch in word + PAD_CH:
        iy = stoi[ch]
        Y.append(iy)
        X.append([stoi[cch] for cch in context])

    context = context[1:] + [ch]

X = torch.tensor(X)
Y = torch.tensor(Y)

# %%

class Linear:
    def __init__(self, fan_in, fan_out, biases=True, init_gain=1, generator=None):
        initialization_gain = init_gain / (fan_in**0.5)
        self.weight = torch.randn(
            (fan_in, fan_out), generator=generator
        )  # * initialization_gain  #/ (fan_in ** 0.5)

        self.bias = torch.zeros(fan_out) if biases else None

    def __call__(self, X):
        self.out = X @ self.weight + self.bias
        return self.out

    def parameters(self):
        params = [
            self.weight,
        ]
        if self.bias is not None:
            params += [
                self.bias,
            ]

        return params


class BatchNorm1:
    def __init__(self, dim, eps=1e-5, momentum=0.1, gamma_coef=1) -> None:
        self.eps = eps
        self.momentum = momentum
        self.training = True

        self.gamma = torch.ones((1, dim)) * gamma_coef
        self.beta = torch.zeros((1, dim))

        self.running_mean = torch.zeros((1, dim))
        self.running_std = torch.zeros((1, dim))

    def parameters(self):
        return [self.gamma, self.beta]

    def __call__(self, X: torch.Any) -> torch.Any:
        if self.training:
            mean = X.mean(dim=0, keepdim=True)
            std = X.std(dim=0, keepdim=True)

            # context to avoid torch to keep track of those not needing backprop:
            with torch.no_grad():
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * mean
                self.running_std = (
                    1 - self.momentum
                ) * self.running_std + self.momentum * std
        else:
            mean = self.running_mean
            std = self.running_std

        # just for tracking purposes:
        self.out = self.gamma * (X - mean) / std + self.beta
        return self.out


class Tanh:
    def parameters(self) -> None:
        return []

    def __call__(self, X) -> torch.Any:
        self.out = F.tanh(X)
        return self.out


# %%
# let's now refactoring the network:
n_embd = 10
n_hidden = 100