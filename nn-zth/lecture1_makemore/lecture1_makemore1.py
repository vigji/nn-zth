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
from numpy import dtype
from sklearn.covariance import log_likelihood
import torch as t

possible_chars = 27
N = t.zeros((possible_chars, possible_chars), dtype=t.int32)

chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
# stoi["<E>"] = 27
itos = {s: i for i, s in stoi.items()}

b = {}
for word in words:
    word = ["."] + list(word) + ["."]
    for c1, c2 in zip(word, word[1:]):
        # bigram = (c1, c2)
        N[stoi[c1], stoi[c2]] += 1
        # b[bigram] = b.get(bigram, 0) + 1

# %%
from matplotlib import pyplot as plt

# %%
plt.figure(figsize=(16, 16))
plt.imshow(N, cmap="Blues")

for i in range(possible_chars):
    for j in range(possible_chars):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")

# %%
N[0]
# %%
g = t.Generator().manual_seed(2147483647)
p = t.rand(3, generator=g)
p = p / p.sum()
p
# %%

# %%
p = N[0, :].float()
p = p / p.sum()
g = t.Generator().manual_seed(2147483647)
ix = t.multinomial(
    p, num_samples=100, replacement=True, generator=g
)  # index for the first letter
# for the second letter, we want to go to given row in the matrix

# To do this:
P = N.float() / N.sum(dim=1, keepdim=True)
# in practice, we do the following to smooth the model and have no 0 probabilities, which are ugly discontinuities
# wen we compute the log likelihood later:
N_norm = N + 1  # can be higher numbers for stronger regularization
P = N_norm.float() / N_norm.sum(dim=1, keepdim=True)
for _ in range(100):
    ix = 0
    sequence = ""
    while True:
        # p = N[ix, :].float()
        # p = p / p.sum()
        # g = t.Generator().manual_seed(2147483647)
        ix = t.multinomial(
            P[ix, :], num_samples=1, replacement=True, generator=g
        ).item()
        sequence += itos[ix]

        if ix == 0:
            break
    print(sequence)

# %%
# This kind of work better than completely random sequence, but still not great.


for w in words[:2]:
    for c1, c2 in zip(w, w[1:]):
        ix1, ix2 = stoi[c1], stoi[c2]
        prob = P[ix1, ix2]
        log_prob = t.log(prob)
        print(f"{c1}{c2}: {prob:.4f}  ({log_prob:.4f})")

# log(a*b*c) = log(a) + log(b) + log(c)
# %%
neg_log_likelihood = 0.0
n = 1
for w in ["sfkljslkjeughr"]:  # test this works for nonsense after regularization above
    for c1, c2 in zip(w, w[1:]):
        ix1, ix2 = stoi[c1], stoi[c2]
        prob = P[ix1, ix2]
        log_prob = t.log(prob)
        print(f"{c1}{c2}: {prob:.4f}  ({log_prob:.4f})")
        neg_log_likelihood -= log_prob
        n += 1

print(f"nll: {neg_log_likelihood}")
print(f"avg. nll: {neg_log_likelihood / n}")
# Neg log likelihood: loss function, we minimize to have the best possible model
# %%
# We will update parameters of our model to minimize this loss function.

# First, we create a training set:
xs, ys = [], []
# xs are the inputs, ys the outputs we expect based on the dataset

for w in words[:1]:  # test this works for nonsense after regularization above
    w = ["."] + list(w) + ["."]
    for c1, c2 in zip(w, w[1:]):
        ix1, ix2 = stoi[c1], stoi[c2]
        xs.append(ix1)
        ys.append(ix2)
        print(c1, c2)

xs = t.tensor(xs)  # uppercase Tensor defaults to float
ys = t.tensor(ys)
# %%
# for the neural network, we'll encode those index values as one-hot vectors
import torch.nn.functional as F

# By default one-hot is int. Cast to float to work with NNs
xenc = F.one_hot(xs, num_classes=possible_chars).float()
# %%
plt.figure()
plt.imshow(xenc)
# %%
# Now let's build our network!
n_neurons = 27
W = t.randn((27, n_neurons))  # one empty dim
(xenc @ W).shape  # output batch_size, number of neurons
# %%
