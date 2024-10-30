# %%
# Implementing WaveNet!
from regex import D
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

block_size = 8

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
        self.weight = torch.randn(n_possible_chars, n_embeddings).to(device)
        self.out = None

    def parameters(self) -> None:
        return [self.weight]
    
    def __call__(self, Xidx) -> torch.Any:
        self.out = self.weight[Xidx]
        return self.out


class Flatten:
    def __call__(self, X) -> torch.Any:
        self.out = X.view(X.shape[0], -1)
        return self.out
    
    def parameters(self) -> None:
        return []

class Sequential:
    def __init__(self, layers) -> None:
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]



# %%
torch.manual_seed(42)

# let's now refactoring the network:
n_hidden = 200
batch_size = 32
n_dims_embedding = 10

# tanh_gain = 5 / 3
layers = Sequential([Embedding(n_dims_embedding, n_possible_chars), 
                     Flatten(block_size),
                     Linear(n_dims_embedding * block_size, n_hidden, 
                            biases=False),  # init_gain=tanh_gain, 
                     BatchNorm1(n_hidden), 
                     Tanh(), 
                     Linear(n_hidden, n_possible_chars),
])
with torch.no_grad():
    layers.layers[-1].weight *= 0.1

parameters = layers.parameters()
print(sum([p.nelement() for p in parameters]))
for param in parameters:
    param.requires_grad = True

# batch of just 4:
ix = torch.randint(0, Xtr.shape[0], (4,), )
Xb, Yb = Xtr[ix], Ytr[ix]
logits = layers(Xb)
for layer in layers.layers:
    print(layer.__class__.__name__, layer.out.shape)

# %%
lrs = torch.cat([torch.ones(150000) * 0.1, torch.ones(50000) * 0.01])

ud = []

train_loss, val_loss = [], []
for i, lr in tqdm(list(enumerate(lrs))):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), )
    Xb, Yb = Xtr[ix], Ytr[ix]
    logits = layers(Xb)

    loss = F.cross_entropy(logits, Yb)

    for param in parameters:
        param.grad = None
    loss.backward()
    train_loss.append(loss.log10().item())

    for param in parameters:
        param.data += -lr * param.grad

    if i % 10000 == 0:
        print(
            f"{i:7d} / {len(lrs)}: loss {loss.item():.4f}, "
        )

    # if i > 10000:
    #     break

 # %%
plt.figure()
# plt.plot(torch.tensor(train_loss))  #.view(-1, 10).mean(dim=1))

plt.plot(torch.tensor(train_loss).view(-1, 1000).mean(dim=1))
# %%
# Evaluate:
for layer in layers.layers:
    layer.training = False

@torch.no_grad()
def test_loss(split):
    Xb, Yb = dict(dev=(Xdev, Ydev), test=(Xte, Yte),
                  train=(Xtr, Ytr))[split]
    logits = layers(Xb)
    loss = F.cross_entropy(logits, Yb)

    print(f"{split} -> loss: {loss.item():.4f}")

test_loss("train")
test_loss("test")

# Generate syntetic names:
n_to_produce = 50


for _ in range(n_to_produce):
    start_from = torch.zeros(block_size, dtype=int)
    chars = []
    k = 0
    next_draw = torch.ones(1)
    while next_draw.item() != 0:
        X = start_from
        logits = layers(X.unsqueeze(0))
        counts = logits.exp()
        prob = counts / torch.sum(counts, dim=1, keepdim=True)
        next_draw = torch.multinomial(prob, 1, replacement=True)

        # next_draw = logits.argmax()
        chars.append(itos[next_draw.item()])
        start_from = torch.cat([start_from[1:], torch.tensor([next_draw])])
        k += 1
        if k > 10:
            break
    print("".join(chars))


# %%
# we will create joint embeddings for doublets of characters. 
# To do so, we combine the embeddings two-by-two:
x = torch.randn((batch_size, block_size, n_dims_embedding))
torch.cat([x[:, ::2, :], x[:, 1::2, :]], dim=-1).shape
# this is the same as
x.view(batch_size, -1, n_dims_embedding*2).shape
# we will update flatten to accept an argument for doing this 
# (NB: this will be different from pytorch  flatten!!)
class Flatten:
    def __init__(self, n=1) -> None:
        self.n = n  # if equal to block_size, this will be equal to prev behavior
    
    def __call__(self, X) -> torch.Any:
        B, L, D = X.shape 
        self.out = X.view(B, L // self.n, D*self.n)
        if self.out.shape[1] == 1:
            self.out = self.out.squeeze(1)
        
        return self.out
    
    def parameters(self) -> None:
        return []
# %%
# Before we can use this new version, we have to fix the batchnorm, as it is currently taking the mean only over the first dimension:
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
        dims_to_avg = tuple(range(len(X.shape) - 1))
        if self.training:
            mean = X.mean(dim=dims_to_avg, keepdim=True)
            std = X.std(dim=dims_to_avg, keepdim=True)

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


# Let's now use the new Flatten layer:
torch.manual_seed(42)

n_hidden = 68
batch_size = 32
n_dims_embedding = 10

new_binning = 2
layers = Sequential([Embedding(n_dims_embedding, n_possible_chars), 
                     Flatten(new_binning), Linear(n_dims_embedding * new_binning, n_hidden, biases=False), BatchNorm1(n_hidden), Tanh(), 
                     Flatten(new_binning), Linear(n_hidden * new_binning, n_hidden, biases=False), BatchNorm1(n_hidden), Tanh(), 
                     Flatten(new_binning), Linear(n_hidden * new_binning, n_hidden, biases=False), BatchNorm1(n_hidden), Tanh(), 
                     Linear(n_hidden, n_possible_chars),
])
with torch.no_grad():
    layers.layers[-1].weight *= 0.1

parameters = layers.parameters()
print(sum([p.nelement() for p in parameters]))
for param in parameters:
    param.requires_grad = True

# batch of just 4:
ix = torch.randint(0, Xtr.shape[0], (4,), )
Xb, Yb = Xtr[ix], Ytr[ix]
logits = layers(Xb)
for layer in layers.layers:
    print(layer.__class__.__name__, layer.out.shape)

 # %%
lrs = torch.cat([torch.ones(150000) * 0.1, torch.ones(50000) * 0.01])

ud = []

train_loss, val_loss = [], []
for i, lr in tqdm(list(enumerate(lrs))):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), )
    Xb, Yb = Xtr[ix], Ytr[ix]
    logits = layers(Xb)

    loss = F.cross_entropy(logits, Yb)

    for param in parameters:
        param.grad = None
    loss.backward()
    train_loss.append(loss.log10().item())

    for param in parameters:
        param.data += -lr * param.grad

    if i % 10000 == 0:
        print(
            f"{i:7d} / {len(lrs)}: loss {loss.item():.4f}, "
        )

    # if i > 10000:
    #     break

# %%
test_loss("train")
test_loss("test")

plt.figure()
 
plt.plot(torch.tensor(train_loss).view(-1, 1000).mean(dim=1))
# %%
