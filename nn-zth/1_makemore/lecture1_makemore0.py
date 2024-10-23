# %%
from pathlib import Path
import re
from tqdm import tqdm
from dataclasses import dataclass

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

# By default one-hot is int. Cast to float for future operations for NNs
xenc = F.one_hot(xs, num_classes=possible_chars).float()
yenc = F.one_hot(ys, num_classes=possible_chars).float()
# %%
plt.figure()
plt.imshow(xenc)
# %%
# Now let's build our network!
# minimal network, one single layer, n_neurons -> n_neurons 
# (input mapped, output will directly correspond to our prediction)
n_neurons_i = 27
n_neurons_o = 27
W = t.randn((n_neurons_i, n_neurons_o))  # one empty dim
(xenc @ W).shape  # output batch_size, number of neurons

(xenc @ W)[3, 13] == (xenc[3] * W[:, 13]).sum()  # same result
# %%
# We want the output of the matrix multiplication to be converted in a probability
# distribution over the possible characters. Such distribution has to sum to 1.
# our network is giving us log counts which we'll exponentiate element-wise:
logits = xenc @ W  # log counts
counts = logits.exp()  # equivalent to counts, bounded positive
probs = counts / counts.sum(1, keepdim=True)
probs.shape, yenc.shape # [0].sum()
# loss = -(probs * yenc).sum()
loss = -(probs[:, ys]).sum()
loss
# %%

batch_size = 5
# First, we create a training set:
xs_all, ys_all = [], []
for w in words:  # test this works for nonsense after regularization above
    w = ["."] + list(w) + ["."]
    for c1, c2 in zip(w, w[1:]):
        ix1, ix2 = stoi[c1], stoi[c2]
        xs_all.append(ix1)
        ys_all.append(ix2)
        print(c1, c2)

xs_all = t.tensor(xs_all)  # uppercase Tensor defaults to float
ys_all = t.tensor(ys_all) 
# %%
# My code before rest of the lecture:
# n_epochs = 3
train_fraction = 0.8
# batch_size = 100

@dataclass
class TrainingParams:
    epochs: int = 20
    learning_rate: float = 0.1
    batch_size: int = 1280

params = TrainingParams()

dataset = t.utils.data.TensorDataset(xs_all, ys_all)
train_size = int(train_fraction * len(xs_all))
test_size = len(xs_all) - train_size
train_dataset, test_dataset = t.utils.data.random_split(dataset, [train_size, test_size])


train_loader = t.utils.data.DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
test_loader = t.utils.data.DataLoader(test_dataset, batch_size=params.batch_size, shuffle=True)

n_hidden = 100
# multiple layers version, otherwise t.nn.Linear(in_features=n_neurons_i, out_features=n_neurons_o, bias=False)
linear_l1 = t.nn.Linear(in_features=n_neurons_i, out_features=n_hidden, bias=False)
linear_l2 = t.nn.Linear(in_features=n_hidden, out_features=n_neurons_o, bias=False)
relu = t.nn.ReLU()
optimizer = t.optim.SGD(list(linear_l1.parameters()) + list(linear_l2.parameters()), lr=params.learning_rate)

for _ in range(params.epochs):
    train_accuracy = 0
    for xs, ys in (pbar := tqdm(train_loader)):
        xenc = F.one_hot(xs, num_classes=possible_chars).float()
        # multiple layers version, otherwise ys_logits = linear_l1(xenc)
        ys_logits = relu(linear_l2(relu(linear_l1(xenc))))

        # compute manually cross-entropy loss:
        counts = ys_logits.exp()  # equivalent to counts, bounded positive
        probs = counts / counts.sum(dim=1, keepdim=True)
        loss = -t.log((probs[range(len(ys)), ys]).sum() / len(xs))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ys_predictions = probs.argmax(dim=1)
        train_accuracy += (ys_predictions == ys).sum().item()
        pbar.set_description(f"{loss:.4f}")

    train_accuracy /= train_size

    test_accuracy = 0
    with t.no_grad():
        # forward pass
        accuracy = 0
        for xs, ys in test_loader:
            xenc = F.one_hot(xs, num_classes=possible_chars).float()
            ys_logits = linear_l(xenc)
            counts = ys_logits.exp()  # equivalent to counts, bounded positive
            probs = counts / counts.sum(dim=1, keepdim=True)
            
            ys_predictions = probs.argmax(dim=1)
            test_accuracy += (ys_predictions == ys).sum().item()
    test_accuracy /= test_size
    print(train_accuracy, test_accuracy)

# %%
plt.figure()
plt.imshow(linear_l.weight.detach().numpy())
# %%
-(probs[:, ys]).sum() / len(xs)
# %%
network = t.nn.Module([linear_l1, linear_l2])
# %%
# First, we create a training set:
xs, ys = [], []
# xs are the inputs, ys the outputs we expect based on the dataset

for w in words: #[:1]:  # test this works for nonsense after regularization above
    w = ["."] + list(w) + ["."]
    for c1, c2 in zip(w, w[1:]):
        ix1, ix2 = stoi[c1], stoi[c2]
        xs.append(ix1)
        ys.append(ix2)
        # print(c1, c2)

xs = t.tensor(xs)  # uppercase Tensor defaults to float
ys = t.tensor(ys)

# simpler implementation in the lecture:
n_neurons_i = 27
n_neurons_o = 27
W = t.randn((n_neurons_i, n_neurons_o), generator=g, requires_grad=True)  # one empty dim

for _ in range(100):
    xenc = F.one_hot(xs, num_classes=possible_chars).float()
    # multiple layers version, otherwise ys_logits = linear_l1(xenc)
    ys_logits = xenc @ W

    # compute manually cross-entropy loss:
    counts = ys_logits.exp()  # equivalent to counts, bounded positive
    probs = counts / counts.sum(dim=1, keepdim=True)
    # loss = -t.log((probs[t.arange(len(ys)), ys]).sum() / len(xs))
    # adding an optional regularization
    loss = -(probs[t.arange(len(ys)), ys]).log().mean() + 0.01*(W**2).mean()  
    loss
    # backward pass
    W.grad = None  # zero the gradient
    loss.backward()
    W.data += -50 * W.grad
    print(loss.item())

# %%
loss.backward()
# we can aim at having a similar loss to when we were just using
# probability of bigrams, but now we can complexify the neural net
# %%
f, axs = plt.subplots(1,2)
axs[0].imshow(N, cmap="Blues")
axs[1].imshow(W.exp().detach().numpy(), cmap="Blues")

# %%
# Generate some words
