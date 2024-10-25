# %%
# Let's improve our generator in the direction of Bangio et al 2006
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

block_size = 3

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


dataset_size = X.shape[0]
n_train = int(dataset_size * 0.8)
n_val = int(dataset_size * 0.1)
n_test = int(dataset_size * 0.1)

Xtr, Ytr = X[:n_train], Y[:n_train]
Xdev, Ydev = X[n_train : n_train + n_val], Y[n_train : n_train + n_val].to(
    device
)
Xte, Yte = X[n_train + n_val :], Y[n_train + n_val :]
# %%
# Util to compare to Torch autograd:
def cmp(s, t, dt):
    exact = torch.all(t.grad == dt).item()
    approx = torch.allclose(t.grad, dt).item()
    maxdiff = torch.max(torch.abs(t.grad - dt)).item()
    print(f"{s:15s} | Exact: {str(exact):5s} | Approx: {str(approx):5s} | maxdiff: {str(maxdiff):5s}")

# %%
# Let's start to implement the backprop ourselves!
n_hidden = 64
n_dims_embedding = 10
batch_size = 32

n_inputs = block_size * n_dims_embedding

g = torch.Generator().manual_seed(2147483647)
C = torch.randn(n_possible_chars, n_dims_embedding, generator=g)
W1 = torch.randn(n_inputs, n_hidden, generator=g)* (5/3)/(n_inputs ** .5)
b1 = torch.randn(n_hidden, generator=g) * 0.1
W2 = torch.randn(n_hidden, n_possible_chars, generator=g) * 0.1
b2 = torch.randn(n_possible_chars, generator=g) * 0.1
# %%
