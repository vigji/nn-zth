# %%
# Let's improve our generator in the direction of Bangio et al 2006

from audioop import bias
from networkx import density
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import einops as ein
import numpy as np
%matplotlib widget

names_file = Path(__file__).parent / "names.txt"
words = open(names_file).read().splitlines()
PAD_CH = '.'
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
Xdev, Ydev = X[n_train:n_train+n_val].to(device), Y[n_train:n_train+n_val].to(device)
Xte, Yte = X[n_train+n_val:].to(device), Y[n_train+n_val:].to(device)
# %%
n_hidden = 200
n_dims_embedding = 10
batch_size = 32

n_steps1 = 100000
n_steps2 = 100000
n_steps3 = 0
lrs = torch.cat([torch.ones(n_steps1) *0.1, torch.ones(n_steps2) *0.01, torch.ones(n_steps3) *0.001])

n_inputs = block_size * n_dims_embedding

g = torch.Generator().manual_seed(2147483647)
C = torch.randn(n_possible_chars, n_dims_embedding, generator=g).to(device)
W1 = torch.randn(n_inputs, n_hidden, generator=g).to(device)
b1 = torch.randn(n_hidden, generator=g).to(device)
W2 = torch.randn(n_hidden, n_possible_chars, generator=g).to(device)
b2 = torch.randn(n_possible_chars, generator=g).to(device)

parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True



train_result = []
val_result = []
test_every = 100
for i, lr in enumerate(tqdm(lrs)):
    # select minibatch:
    idx = torch.randint(0, Xtr.shape[0], (batch_size,)).to(device)

    # Forward pass:
    emb = C[Xtr[idx]]
    h = torch.tanh(emb.view(-1, n_inputs) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[idx])
    train_result.append(loss.log10().item())
    
    # backward pass:
    for p in parameters:
        p.grad = None

    loss.backward()
    for p in parameters:
        p.data += -lr * p.grad

    # performance on val:
    if i % test_every == 0:
        with torch.no_grad():
            # Forward pass:
            emb = C[Xdev]
            h = torch.tanh(emb.view(-1, n_inputs) @ W1 + b1)
            logits = h @ W2 + b2
            loss_val = F.cross_entropy(logits, Ydev)

            val_result.extend([loss_val.log10().item(),]* test_every)

    if i % 10000 == 0:
        print(f"{i:7d} / {len(lrs)}: loss {loss.item():.4f}, loss val {loss_val.item():.4f}, ")
# %%
plt.figure()
plt.plot([t.item() for t in train_result])
# %%
# What is the initial loss that we expect?
# without knowledge, chars should have hom prob of 1/27 each
p = 1/27

# This is the loss that we would like to see, and it is actually much lower than ours
-torch.tensor(p).log()

# Issue is that at initialization network produces prob distributions that are very peaked on wrong
# chars, basically it is very wrong and confident about it.

# %%
# 4d example:
logits = torch.tensor([0., 1., 5., 0.])
probs = torch.softmax((logits), dim=0)
loss = -probs[2].log()
probs, loss

# So, we want the logits to be quite similar to each other at initialization.
# in our case, what happens above if we break early? The logits create quite extrem values.
# Options can be set biases to 0, or weights at low values.

# %%
g = torch.Generator().manual_seed(2147483647)
C = torch.randn(n_possible_chars, n_dims_embedding, generator=g).to(device)
W1 = torch.randn(n_inputs, n_hidden, generator=g).to(device)
b1 = torch.randn(n_hidden, generator=g).to(device)
W2 = torch.randn(n_hidden, n_possible_chars, generator=g).to(device) * 0.01
b2 = torch.randn(n_possible_chars, generator=g).to(device) * 0

parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True



train_result = []
val_result = []
test_every = 100
for i, lr in enumerate(tqdm(lrs)):
    # select minibatch:
    idx = torch.randint(0, Xtr.shape[0], (batch_size,)).to(device)

    # Forward pass:
    emb = C[Xtr[idx]]
    h = torch.tanh(emb.view(-1, n_inputs) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[idx])
    train_result.append(loss.log10().item())
    
    # backward pass:
    for p in parameters:
        p.grad = None

    loss.backward()
    for p in parameters:
        p.data += -lr * p.grad

    # performance on val:
    if i % test_every == 0:
        with torch.no_grad():
            # Forward pass:
            emb = C[Xdev]
            h = torch.tanh(emb.view(-1, n_inputs) @ W1 + b1)
            logits = h @ W2 + b2
            loss_val = F.cross_entropy(logits, Ydev)

            val_result.extend([loss_val.log10().item(),]* test_every)

    if i % 10000 == 0:
        print(f"{i:7d} / {len(lrs)}: loss {loss.item():.4f}, loss val {loss_val.item():.4f}, ")
# %%
# This removed a bit of the easy descent at the beginning.

# What about activations in the hidden layer? Issue is we have lot of values -1 or 1.
# This is because the pre-activations are quite broad, numbers are taking extreme values.
# This is killing the gradient!!!

plt.figure()
plt.hist(h.flatten().detach().numpy(), 50)

# %%
# Proper normalization for tanh: 5/(3 * sqrt(fan_in))

g = torch.Generator().manual_seed(2147483647)
C = torch.randn(n_possible_chars, n_dims_embedding, generator=g).to(device)
W1 = torch.randn(n_inputs, n_hidden, generator=g).to(device) * (5/(3*(n_inputs**0.5)))
b1 = torch.randn(n_hidden, generator=g).to(device) * 0.01
W2 = torch.randn(n_hidden, n_possible_chars, generator=g).to(device) * 0.01
b2 = torch.randn(n_possible_chars, generator=g).to(device) * 0

parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True



train_result = []
val_result = []
test_every = 100
for i, lr in enumerate(tqdm(lrs)):
    # select minibatch:
    idx = torch.randint(0, Xtr.shape[0], (batch_size,)).to(device)

    # Forward pass:
    emb = C[Xtr[idx]]
    h = torch.tanh(emb.view(-1, n_inputs) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[idx])
    train_result.append(loss.log10().item())
    
    # backward pass:
    for p in parameters:
        p.grad = None

    loss.backward()
    for p in parameters:
        p.data += -lr * p.grad

    # performance on val:
    if i % test_every == 0:
        with torch.no_grad():
            # Forward pass:
            emb = C[Xdev]
            h = torch.tanh(emb.view(-1, n_inputs) @ W1 + b1)
            logits = h @ W2 + b2
            loss_val = F.cross_entropy(logits, Ydev)

            val_result.extend([loss_val.log10().item(),]* test_every)

    if i % 10000 == 0:
        print(f"{i:7d} / {len(lrs)}: loss {loss.item():.4f}, loss val {loss_val.item():.4f}, ")
# %%
plt.figure()
plt.plot(train_result)
plt.plot(val_result)

# %%
# BATCH NORMALIZATION
# a trick that just made everything way simpler
# why don't we just normalize what we feed into our nonlinearities?
# (in this case, we can remove biase of the layer)
# Fior the evaluation, to be able to run network on single examples and not batches,
# we will just use a rolling mean and std

g = torch.Generator().manual_seed(2147483647)
C = torch.randn(n_possible_chars, n_dims_embedding, generator=g).to(device)
W1 = torch.randn(n_inputs, n_hidden, generator=g).to(device) * (5/(3*(n_inputs**0.5)))
W2 = torch.randn(n_hidden, n_possible_chars, generator=g).to(device) * 0.01
b2 = torch.randn(n_possible_chars, generator=g).to(device) * 0

bn_gain = torch.ones((1, n_hidden))
bn_bias = torch.zeros((1, n_hidden))
bn_mn_running = torch.ones((1, n_hidden))
bn_sd_running = torch.ones((1, n_hidden))

parameters = [C, W1, W2, b2, bn_gain, bn_bias]
for p in parameters:
    p.requires_grad = True

n_steps1 = 200000
n_steps2 = 200000
n_steps3 = 0
lrs = torch.cat([torch.ones(n_steps1) *0.1, torch.ones(n_steps2) *0.01, torch.ones(n_steps3) *0.001])

# We can use a large momentum if we use big batches
momentum = 0.001
train_result = []
val_result = []
test_every = 100
for i, lr in enumerate(tqdm(lrs)):
    # select minibatch:
    idx = torch.randint(0, Xtr.shape[0], (batch_size,)).to(device)

    # Forward pass:
    emb = C[Xtr[idx]]
    preh = emb.view(-1, n_inputs) @ W1
    mn = preh.mean(dim=0, keepdim=True)
    sd = preh.std(dim=0, keepdim=True)

    with torch.no_grad():
        bn_mn_running = (1-momentum) * bn_mn_running + momentum * mn
        bn_sd_running = (1-momentum) * bn_sd_running + momentum * sd


    preh = bn_gain * (preh - mn) / sd + bn_bias
    
    h = torch.tanh(preh)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[idx])
    train_result.append(loss.log10().item())

    # backward pass:
    for p in parameters:
        p.grad = None

    loss.backward()

    for p in parameters:
        p.data += -lr * p.grad

    # performance on val:
    if i % test_every == 0:
        with torch.no_grad():
            # Forward pass:
            emb = C[Xdev]
            preh = emb.view(-1, n_inputs) @ W1 + b1
            preh = bn_gain * (preh - bn_mn_running) / bn_sd_running + bn_bias
            
            h = torch.tanh(preh)
            logits = h @ W2 + b2
            loss_val = F.cross_entropy(logits, Ydev)

            val_result.extend([loss_val.log10().item(),]* test_every)

    if i % 10000 == 0:
        print(f"{i:7d} / {len(lrs)}: loss {loss.item():.4f}, loss val {loss_val.item():.4f}, ")
# %%
plt.figure()
plt.plot(train_result)
plt.plot(val_result)
# %%

# Let's pytorchify everything!

class Linear:
    def __init__(self, fan_in, fan_out, biases=True, init_gain=1, generator=None):
        initialization_gain = init_gain / (fan_in ** 0.5)
        self.weight = torch.randn((fan_in, fan_out), generator=generator) * initialization_gain  #/ (fan_in ** 0.5)
        

        self.bias = torch.zeros(fan_out) if biases else None

    def __call__(self, X):
        self.out = X @ self.weight + self.bias
        return self.out
    
    def parameters(self):
        params = [self.weight, ]
        if self.bias is not None:
            params += [self.bias, ]

        return params


class BatchNorm1:
    def __init__(self, dim, eps=1e-5, momentum=0.1) -> None:
        self.eps = eps
        self.momentum = momentum
        self.training = True
        
        self.gamma = torch.ones((1, dim))
        self.beta = torch.zeros((1, dim))

        self.running_mean = torch.zeros((1, dim))
        self.running_std = torch.zeros((1, dim))

    def parameters(self):
        return [self.running_mean, self.running_std]
    
    def __call__(self, X: torch.Any) -> torch.Any:
        if self.training:
            mean = X.mean(dim=0, keepdim=True)
            std = X.std(dim=0, keepdim=True)

            # context to avoid torch to keep track of those not needing backprop:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_std = (1 - self.momentum) * self.running_std + self.momentum * std
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

g = torch.Generator().manual_seed(2147483647)
C = torch.randn(n_possible_chars, n_dims_embedding, generator=g).to(device)

n_hidden_layers = 4
tanh_gain = 5 / 3
layers = [Linear(n_dims_embedding * block_size, n_hidden, generator=g, init_gain=tanh_gain), Tanh(), ]

for _ in range(n_hidden_layers):
    new_layer = [Linear(n_hidden, n_hidden, generator=g, init_gain=tanh_gain), Tanh(),]
    layers += new_layer

layers += [Linear(n_hidden, n_possible_chars, generator=g, init_gain=0.1),]  # arbitrarily less confident

parameters = [C,] + [p for layer in layers for p in layer.parameters()]
print(sum([p.data.sum() for p in parameters]))
for param in parameters:
    param.requires_grad = True
# %%


batch_size = 32
n_steps1 = 100000
n_steps2 = 100000
lrs = torch.cat([torch.ones(n_steps1) *0.1, torch.ones(n_steps2) *0.01])

train_loss, val_loss = [], []
for i, lr in enumerate(lrs):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)

    Xb, Yb = Xtr[ix], Ytr[ix]

    # forward pass:
    x = C[Xb].view(-1, n_inputs)
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
        print(f"{i:7d} / {len(lrs)}: loss {loss.item():.4f}, loss val {loss_val.item():.4f}, ")

    break

# %%
xbins = np.linspace(-1., 1., 50)
plt.figure()
for l in layers:
    if isinstance(l, Tanh):
        # print(l.out.flatten())
        t = l.out
        print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
        counts, _ = torch.histogram(l.out.flatten(), torch.tensor(xbins, dtype=l.out.dtype), density=True)
        print()
        plt.plot((xbins[1:] + xbins[:-1])/2, counts.detach())
plt.show()
# %%
xbins = np.linspace(-1., 1., 50)
plt.figure()
for l in layers:
    if isinstance(l, Tanh):
        # print(l.out.flatten())
        t = l.out
        print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
        counts, _ = torch.histogram(l.out.flatten(), torch.tensor(xbins, dtype=l.out.dtype), density=True)
        print()
        plt.plot((xbins[1:] + xbins[:-1])/2, counts.detach())
plt.show()
# %%
layers[0].weight.shape, x.shape
# %%
l.out.detach().numpy().shape
# %%
b
# %%
