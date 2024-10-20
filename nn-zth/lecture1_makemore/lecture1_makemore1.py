# %%
from pathlib import Path

names_file = Path(__file__).parent / "names.txt"
words = open(names_file).read().splitlines()
# %%
b = {}
for word in words:
    word = ["<S>"] + list(word) + ["<E>"]
    for c1, c2 in zip(word, word[1:]):
        bigram = (c1, c2)
        b[bigram] = b.get(bigram, 0) + 1


# %%
sorted(b.items(), key=lambda kv: -kv[1])
# %%
import torch as t

possible_chars = 28
N = t.zeros((28, 28), dtype=t.int32)

chars = sorted(list(set("".join(words))))
stoi = {s:i for i, s in enumerate(chars)}
stoi["<S>"] = 26
stoi["<E>"] = 27
itos = {s:i for i, s in stoi.items()}

b = {}
for word in words:
    word = ["<S>"] + list(word) + ["<E>"]
    for c1, c2 in zip(word, word[1:]):
        # bigram = (c1, c2)
        N[stoi[c1], stoi[c2]] += 1
        # b[bigram] = b.get(bigram, 0) + 1

# %%
from matplotlib import pyplot as plt
# %%
plt.figure(figsize=(16, 16))
plt.imshow(N, cmap="Blues")

for i in range(28):
    for j in range(28):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")

# %%
p = N[0, :].float()
p = p / p.sum()
# %%
g = t.Generator().manual_seed(2147483547)
p = t.rand(3, generator=g)
p = p/ p.sum()
p
# %%
torch.multinomial(p, num_samples=200, replacement=True, generator=g)