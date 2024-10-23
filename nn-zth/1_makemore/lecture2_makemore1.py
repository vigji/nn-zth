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

possible_chars = 27
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
C = torch.randn(possible_chars, n_dims_embedding)
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
emb.view(-1, n_inputs) @ W1



# %%
