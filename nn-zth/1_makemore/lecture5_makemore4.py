# %%
from audioop import bias
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

Xtr, Ytr = X[:n_train].to(device), Y[:n_train].to(device)
Xdev, Ydev = X[n_train : n_train + n_val].to(device), Y[n_train : n_train + n_val].to(
    device
)
Xte, Yte = X[n_train + n_val :].to(device), Y[n_train + n_val :].to(device)

Y
# %%

class Linear:
    def __init__(self, fan_in, fan_out, biases=True, init_gain=1):
        initialization_gain = init_gain / (fan_in**0.5)
        self.weight = torch.randn(
            (fan_in, fan_out)
        )  * initialization_gain

        self.bias = torch.zeros(fan_out) if biases else None

    def __call__(self, X):
        self.out = X @ self.weight
        if self.bias is not None:
            self.out += self.bias
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


class Embedding:
    def __init__(self, n_embeddings, n_possible_chars=27) -> None:
        self.weight = torch.randn(n_possible_chars, n_dims_embedding).to(device)

    def parameters(self) -> None:
        return [self.weight]
    
    def __call__(self, Xidx) -> torch.Any:
        return self.weight[Xidx]


class Flatten:
    def __call__(self, X) -> torch.Any:
        self.out = X.view(X.shape[0], -1)
        return self.out


# %%
torch.manual_seed(42)
# let's now refactoring the network:
n_hidden = 200
batch_size = 32
n_dims_embedding = 10
block_size = 3

n_hidden_layers = 4
tanh_gain = 5 / 3
layers = [
    Linear(n_dims_embedding * block_size, n_hidden, init_gain=tanh_gain, biases=False),
    BatchNorm1(n_hidden), Tanh(), Linear(n_hidden, n_possible_chars),
]

parameters = [
    C,
] + [p for layer in layers for p in layer.parameters()]
print(sum([p.nelement() for p in parameters]))
for param in parameters:
    param.requires_grad = True

# %%
lrs = torch.cat([torch.ones(150000) * 1, torch.ones(50000) * 0.01])

ud = []

train_loss, val_loss = [], []
for i, lr in tqdm(list(enumerate(lrs))):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), )

    Xb, Yb = Xtr[ix], Ytr[ix]

    # forward pass:
    x = C[Xb].view(batch_size, -1)
    for j, layer in enumerate(layers):
        x = layer(x)

    loss = F.cross_entropy(x, Yb)

    for layer in layers:
        layer.out.retain_grad()

    for param in parameters:
        param.grad = None
    loss.backward()
    train_loss.append(loss.log10().item())

    for param in parameters:
        param.data += -lr * param.grad

    if i % 10000 == 0:
        print(
            f"{i:7d} / {len(lrs)}: loss {loss.item():.4f}, loss val {loss.item():.4f}, "
        )

    # with torch.no_grad():
    #     ud.append(
    #         [(lr * p.grad.std() / p.data.std()).log10().item() for p in parameters]
    #     )

    # if i > 1000:
    #     break
# %%
plt.figure()
plt.plot(torch.tensor(train_loss).view(-1, 1000).mean(dim=1))
# %%
x
# %%
