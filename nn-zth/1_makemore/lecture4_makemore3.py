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
    approx = torch.allclose(t.grad, dt)# .item()
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

bnorm_gain = torch.randn((1, n_hidden), generator=g) * 0.1 + 1
bnorm_bias = torch.randn((1, n_hidden), generator=g) * 0.1

parameters = [C, W1, b1, W2, b2, bnorm_gain, bnorm_bias]
print(sum(p.numel() for p in parameters))
for p in parameters:
    p.requires_grad = True


# single batch
batch_size = 32
ix = torch.randint(0, n_train, (batch_size,), generator=g)
Xb, Yb = Xtr[ix], Ytr[ix]

# Step by step forward pass:

emb = C[Xb]

# Layer 1
embcat = emb.view(batch_size, -1)
h_preact = embcat @ W1 + b1
# batch norm:
mean_preact = (1 / batch_size) * h_preact.sum(dim=0, keepdim=True)
bn_diff = h_preact - mean_preact
bn_diffsq = bn_diff ** 2
bn_var = (1 / (batch_size - 1)) * bn_diffsq.sum(dim=0, keepdim=True)
bn_std_inv = (bn_var + 1e-5)**-.5

bnormed = bn_diff * bn_std_inv
normed_preact = bnorm_gain * bnormed + bnorm_bias

h = torch.tanh(normed_preact)

# Layer 2
logits = h @ W2 + b2

# before softmax, subtract the maximum for each batch for numerical stability:
logits_max = logits.max(dim=1, keepdim=True).values
logits_norm = logits - logits_max

# softmax
counts = torch.exp(logits_norm)
counts_sum = counts.sum(dim=1, keepdim=True)
counts_sum_inv = (counts_sum) ** -1
probs = counts_sum_inv * counts

log_probs = probs.log()
loss = -torch.mean(log_probs[np.arange(batch_size), Yb])
print(loss)

for p in parameters:
    p.grad = None
# remember all grads:
for t in [loss, log_probs, probs, counts_sum_inv, counts_sum, counts, logits_norm, logits,
          logits_max, h, b2, W2, normed_preact, bnormed, bn_std_inv, bn_var,
          bn_diffsq, bn_diff, mean_preact, h_preact, b1, W1, embcat, emb]:
    t.retain_grad()
loss.backward()

# dlog_probs: 0 for all values not contributing to loss calculation
dlog_probs = torch.zeros_like(log_probs)
dlog_probs[np.arange(batch_size), Yb] = - 1. / batch_size
cmp("log_probs", log_probs, dlog_probs)

dprobs = 1 / probs * dlog_probs
cmp("probs", probs, dprobs)

# Here we sum as with broadcasting each row will contribute multiple times to final loss!
dcounts_sum_inv = (counts * dprobs).sum(dim=1, keepdim=True)
cmp("counts_sum_inv", counts_sum_inv, dcounts_sum_inv)

dcounts = counts_sum_inv * dprobs  # check later as it is used elsewhere

dcounts_sum = - 1 / (counts_sum ** 2) * dcounts_sum_inv
cmp("counts_sum", counts_sum, dcounts_sum)

dcounts += torch.ones_like(counts) * dcounts_sum
cmp("logits_exp", counts, dcounts)

dlogits_norm = torch.exp(logits_norm) * dcounts
cmp("logits_norm", logits_norm, dlogits_norm)

dlogits = dlogits_norm

dlogits_max = - dlogits.sum(dim=1, keepdim=True)
cmp("logits_max", logits_max, dlogits_max)

# Contribution of max computation:
dlogits += F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * dlogits_max

cmp("logits", logits, dlogits)

dW2 = (dlogits.T @ h).T
cmp("W2", W2, dW2)

dh = (W2 @ dlogits.T).T
cmp("h", h, dh)

db2 = dlogits.sum(dim=0, keepdim=True)
cmp("b2", b2, db2)

dnormed_preact = (1 - torch.tanh(normed_preact)**2) * dh
cmp("normed_preact", normed_preact, dnormed_preact)

dbnormed = bnorm_gain * dnormed_preact
cmp("bnormed", bnormed, dbnormed)

dbnorm_gain = (bnormed * dnormed_preact).sum(0, keepdim=True)
cmp("bnorm_gain", bnorm_gain, dbnorm_gain)

dbnorm_bias = dnormed_preact.sum(0, keepdim=True)
cmp("bnorm_bias", bnorm_bias, dbnorm_bias)

# ###
# embcat = emb.view(batch_size, -1)
# h_preact = embcat @ W1 + b1
# # batch norm:
# mean_preact = (1 / batch_size) * h_preact.sum(dim=0, keepdim=True)
# bn_diff = h_preact - mean_preact
# bn_diffsq = bn_diff ** 2
# bn_var = (1 / (batch_size - 1)) * bn_diffsq.sum(dim=0, keepdim=True)
# bn_std_inv = (bn_var + 1e-5)**-.5
# ###

dbn_diff = bn_std_inv * dbnormed
dbn_std_inv = (bn_diff * dbnormed).sum(0, keepdim=True)
dbn_var = -0.5*((bn_var + 1e-5) ** (-3/2)) * dbn_std_inv

dbn_diffsq = (1 / (batch_size - 1) * torch.ones_like(bn_diffsq)) * dbn_var
dbn_diff += 2 * bn_diff * dbn_diffsq
dh_preact = 1 * dbn_diff
dmean_preact = -dbn_diff.sum(dim=0)
dh_preact += (1 / (batch_size) * torch.ones_like(h_preact)) * dmean_preact

dW1 = (dh_preact.T @ embcat).T
dembcat = (W1 @ dh_preact.T).T
db1 = dh_preact.sum(dim=0, keepdim=True)
demb = ein.rearrange(dembcat, "b (e d) -> b e d", e=3, d=10)

dC = torch.zeros_like(C)
for i in range(Xb.shape[0]):
    for j in range(Xb.shape[1]):
        idx = Xb[i, j]
        dC[idx] += demb[i, j]

cmp("dbn_diff", bn_diff, dbn_diff)
cmp("dbn_std_inv", bn_std_inv, dbn_std_inv)
cmp("dbn_var", bn_var, dbn_var)
cmp("dbn_diffsq", bn_diffsq, dbn_diffsq)
cmp("mean_preact", mean_preact, dmean_preact)
cmp("h_preact", h_preact, dh_preact)
cmp("b1", b1, db1)
cmp("W1", W1, dW1)
cmp("embcat", embcat, dembcat)
cmp("emb", emb, demb)
cmp("C", C, dC)




# dlogits_exp_sum = - (1 / logits_exp_sum ** 2) * ddenom
# cmp("logits_exp_sum", logits_exp_sum, dlogits_exp_sum)

# dlogits_exp = dlogits_exp_sum * n_possible_chars
# cmp("logits_exp", logits_exp, dlogits_exp)
dembcat.shape

    # %%
dembcat.shape
 # %%
emb.shape
# %%
W2.sum(dim=0, keepdim=True).shape
# %%
bn_diffsq.grad
# %%
