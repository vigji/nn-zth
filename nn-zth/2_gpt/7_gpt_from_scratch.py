# %%
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt

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
