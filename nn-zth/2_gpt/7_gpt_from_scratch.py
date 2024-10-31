# %%
from regex import B
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
import einops

with open("tiny_squotilancia.txt", "r") as f:
    text = f.read()

print(len(text))
# %%
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(chars)
print(vocab_size)
# %%

stoi = {ch: i for i, ch in enumerate(chars)}  # string to int
itos = {i: ch for i, ch in enumerate(chars)}  # int to string

encode = lambda x: [stoi[ch] for ch in x]
decode = lambda x: "".join([itos[i] for i in x])

print(encode("ciao"), decode(encode("ciao")))
 # %%
perc_train = 0.9
n_split = int(len(text) * perc_train)

encoded_data = torch.tensor(encode(text), dtype=torch.long)
train_data = encoded_data[:n_split]
val_data = encoded_data[n_split:]
print(train_data.shape, val_data.shape)
print(train_data[:10])
# %%
block_size = 8
X = train_data[:block_size]
Y = train_data[1:block_size+1]
for t in range(block_size):
    context = X[:t]
    target = Y[t]
# %%
torch.manual_seed(1337)
batch_size = 4

def get_batch(split, batch_size=4):
    if split == "train":
        data = train_data
    elif split == "val":
        data = val_data
    else:
        raise ValueError("split must be 'train' or 'val'")
    
    start_idx = torch.randint(0, data.size(0) - block_size, (batch_size,))
    X = torch.stack([data[idx:idx + block_size] for idx in start_idx])
    Y = torch.stack([data[idx+1:idx + block_size+1] for idx in start_idx])
    return X, Y

xb, yb = get_batch("train", batch_size=4)
print(xb.shape, yb.shape)
# %%
import torch.nn as nn
from torch.nn import functional as F

class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, context, target):
        logits = self.token_embedding_table(context)

        logits = einops.rearrange(logits, 'b t c -> (b t) c')
        print(logits.shape)

        loss = F.cross_entropy(logits, target)
        

        return logits
    
m = BigramModel(vocab_size)
m(xb, yb).shape
# %%
