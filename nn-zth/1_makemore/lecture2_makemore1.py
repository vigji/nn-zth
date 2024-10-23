# %%
# Let's improve our generator in the direction of Bangio et al 2006

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
import einops as ein
%matplotlib widget

names_file = Path(__file__).parent / "names.txt"
words = open(names_file).read().splitlines()

n_possible_chars = 27
# N = torch.zeros((possible_chars, possible_chars), dtype=torch.int32)
chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {s: i for i, s in stoi.items()}
# %%
print(itos)
# %%
block_size = 3  # size of the context window
PAD_CH = "."

X, Y = [], []

for word in words[:5]:
    # word = ["."] + list(word) + ["."]
    context = [PAD_CH] * block_size
    
    for ch in word + PAD_CH:
        iy = stoi[ch]
        Y.append(iy)
        X.append([stoi[cch] for cch in context])

        context = context[1:] + [ch]

X = torch.tensor(X)
Y = torch.tensor(Y)

# %%
# Build our embedding space
n_dims_embedding = 2
C = torch.randn(n_possible_chars, n_dims_embedding)
C[5]  # indexing works as matrix multiplication of one-hot vector for element @ C
# We can even index with multi-d index array!
C[X].shape  # (n_blocks, context_length) -> (n_blocks, context_length, n_dims_embedding)

# So this will be embedding:
emb = C[X] 

# Next step in the network is a fully connected layer of n_hidden neurons
n_inputs = block_size * n_dims_embedding
n_hidden = 100

W1 = torch.randn(n_inputs, n_hidden)  # weights
b1 = torch.randn(n_hidden) # biases

# %%
torch.cat([emb[:, i, :] for i in range(emb.shape[1])], dim=1) @ W1
# or alternatively
torch.cat(torch.unbind(emb, 1), 1) @ W1
# or alternatively
ein.rearrange(emb, "b ct dim -> b (ct dim)")  @ W1
# or alternatively, this is actually the most efficient:


# So, hidden activations will be:
h = torch.tanh(emb.view(-1, n_inputs) @ W1 + b1)
h
# %%
# The next layer will be fully connected to all possible 27 chars:
W2 = torch.randn(n_hidden, n_possible_chars)
b2 = torch.randn(n_possible_chars)

# not neurons but directly output, no non-linearity here:
logits = h @ W2 + b2
logits
# %%
# Softmax computation:
counts = logits.exp()
prob = counts / torch.sum(counts, dim=1, keepdim=True)

prob[torch.arange(32), Y]  # extracts probabilities for all Ys with current state of mapping

loss = -prob[torch.arange(32), Y].mean().log()
loss

# All this is equivalent to:
loss = F.cross_entropy(logits, Y)
# cross_entropy is not only more efficient than our manual implementation, but it is also
# numerical better behaved: it uses internal normalizations to avoid nan/inf when computing
# exponentiations with very small/large exponent
# %%
# Let's refactor everything now:
g = torch.Generator().manual_seed(2147483647)
n_hidden = 100

C = torch.randn(n_possible_chars, n_dims_embedding, generator=g)
W1 = torch.randn(n_inputs, n_hidden, generator=g)
b1 = torch.randn(n_hidden, generator=g)
W2 = torch.randn(n_hidden, n_possible_chars, generator=g)
b2 = torch.randn(n_possible_chars, generator=g)

parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True

lr = 0.1
for _ in range(1000):
    # Forward pass:
    emb = C[X]
    h = torch.tanh(emb.view(-1, n_inputs) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y)
    
    # backward pass:
    for p in parameters:
        p.grad = None

    loss.backward()

    for p in parameters:
        p.data += -lr * p.grad

print(loss.item())

# %%
# How come we do not get to zero=