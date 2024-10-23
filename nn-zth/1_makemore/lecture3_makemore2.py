# %%
# Let's improve our generator in the direction of Bangio et al 2006

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import einops as ein
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
g = torch.Generator().manual_seed(2147483647)
C = torch.randn(n_possible_chars, n_dims_embedding, generator=g).to(device)
W1 = torch.randn(n_inputs, n_hidden, generator=g).to(device) * 0.1
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