# %%
print(itos)
# %%
block_size = 3  # size of the context window
PAD_CH = "."

X, Y = [], []

for word in words:
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
b1 = torch.randn(n_hidden)  # biases

# %%
torch.cat([emb[:, i, :] for i in range(emb.shape[1])], dim=1) @ W1
# or alternatively
torch.cat(torch.unbind(emb, 1), 1) @ W1
# or alternatively
ein.rearrange(emb, "b ct dim -> b (ct dim)") @ W1
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

prob[
    torch.arange(32), Y
]  # extracts probabilities for all Ys with current state of mapping

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
for _ in range(100):
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
# How come we do not get to zero?
# Issue is we have initial characters of each word that cannot really be predicted from
# the prev chars '...'

# %%
# Let's do stuff in batches now:
lr = 0.1
batch_size = 32

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

for _ in range(100):
    # select minibatch:
    idx = torch.randint(0, X.shape[0], (batch_size,))

    # Forward pass:
    emb = C[X[idx]]
    h = torch.tanh(emb.view(-1, n_inputs) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y[idx])

    # backward pass:
    for p in parameters:
        p.grad = None

    loss.backward()

    for p in parameters:
        p.data += -lr * p.grad

print(loss.item())

emb = C[X]
h = torch.tanh(emb.view(-1, n_inputs) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Y)
print("Total loss: ", loss.item())
# %%
# How can we decide the learning rate?
lre = torch.linspace(-3, 0, 200)
lrs = 10**lre

g = torch.Generator().manual_seed(2147483647)
n_hidden = 100


results = []
for lr in tqdm(lrs):
    C = torch.randn(n_possible_chars, n_dims_embedding, generator=g)
    W1 = torch.randn(n_inputs, n_hidden, generator=g)
    b1 = torch.randn(n_hidden, generator=g)
    W2 = torch.randn(n_hidden, n_possible_chars, generator=g)
    b2 = torch.randn(n_possible_chars, generator=g)

    parameters = [C, W1, b1, W2, b2]
    for p in parameters:
        p.requires_grad = True

    result = []
    for _ in range(100):
        # select minibatch:
        idx = torch.randint(0, X.shape[0], (batch_size,))

        # Forward pass:
        emb = C[X[idx]]
        h = torch.tanh(emb.view(-1, n_inputs) @ W1 + b1)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, Y[idx])

        # backward pass:
        for p in parameters:
            p.grad = None

        loss.backward()
        result.append(loss.item())

        for p in parameters:
            p.data += -lr * p.grad

    results.append(result)

print(loss.item())
# %%
import numpy as np

plt.figure()
plt.plot(lre, np.array(results)[:, -1])
plt.show()
# %%
# Optimum around 0.1. We can do the following: some iterations (10k?) at 0.1, and then "cool down" at 0.01
n_steps1 = 30000
n_steps2 = 20000
n_steps3 = 20000
lrs = torch.cat(
    [
        torch.ones(n_steps1) * 0.1,
        torch.ones(n_steps2) * 0.01,
        torch.ones(n_steps3) * 0.001,
    ]
)


C = torch.randn(n_possible_chars, n_dims_embedding, generator=g)
W1 = torch.randn(n_inputs, n_hidden, generator=g)
b1 = torch.randn(n_hidden, generator=g)
W2 = torch.randn(n_hidden, n_possible_chars, generator=g)
b2 = torch.randn(n_possible_chars, generator=g)

parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True

result = []
for lr in tqdm(lrs):
    # select minibatch:
    idx = torch.randint(0, X.shape[0], (batch_size,))

    # Forward pass:
    emb = C[X[idx]]
    h = torch.tanh(emb.view(-1, n_inputs) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y[idx])

    # backward pass:
    for p in parameters:
        p.grad = None

    loss.backward()
    result.append(loss.item())

    for p in parameters:
        p.data += -lr * p.grad

# %%
smoothed = ein.reduce(torch.tensor(result), "(n 100) -> n", "mean")
plt.figure()
plt.plot(smoothed)
plt.figure()
plt.scatter(C[:, 0].detach().numpy(), C[:, 1].detach().numpy())
# %%
# to do things properly, we should split test, dev/validation (hyperparameters optimization), and test split
# Usually, 80%, 10%, 10%
import random

random.seed(42)
dataset_size = len("".join(words))
idxs = list(range(dataset_size))
random.shuffle(idxs)

dataset_size = X.shape[0]
n_train = int(dataset_size * 0.8)
n_val = int(dataset_size * 0.1)
n_test = int(dataset_size * 0.1)
print(n_train, n_val, n_test)

# %%
# all our params in one place:
device = "cpu"

block_size = 5
n_hidden = 2000
n_dims_embedding = 10
batch_size = 256

n_inputs = block_size * n_dims_embedding

# learning schedule:
n_steps1 = 200000
n_steps2 = 100000
n_steps3 = 50000
lrs = torch.cat(
    [
        torch.ones(n_steps1) * 0.1,
        torch.ones(n_steps2) * 0.01,
        torch.ones(n_steps3) * 0.001,
    ]
)

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
shuffledXs = X[idxs, :]
shuffledYs = Y[idxs]

Xtr, Ytr = shuffledXs[:n_train].to(device), shuffledYs[:n_train].to(device)
Xdev, Ydev = shuffledXs[n_train : n_train + n_val].to(device), shuffledYs[
    n_train : n_train + n_val
].to(device)
Xte, Yte = shuffledXs[n_train + n_val :].to(device), shuffledYs[n_train + n_val :].to(
    device
)
print(Xtr.shape, Xdev.shape, Xte.shape)
# %%

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
    train_result.append(loss)

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
            loss = F.cross_entropy(logits, Ydev)

            val_result.extend(
                [
                    loss.item(),
                ]
                * test_every
            )
# %%
Ytr.shape
# %%
smoothed_test = ein.reduce(torch.tensor(train_result), "(n 100) -> n", "mean")
smoothed_val = ein.reduce(torch.tensor(val_result), "(n 100) -> n", "mean")

plt.figure()
plt.plot(smoothed_test)
plt.plot(smoothed_val)
plt.title(smoothed_val[-1])
# if we are not overfitting our model is pretty small
# %%
logits.shape, emb.shape, h.shape
# %%

# Generate syntetic names:
n_to_produce = 50


for _ in range(n_to_produce):
    start_from = torch.zeros(block_size, dtype=int)
    chars = []
    k = 0
    next_draw = torch.ones(1)
    while next_draw.item() != 0:
        X = start_from
        emb = C[X]
        h = torch.tanh(emb.view(-1, n_inputs) @ W1 + b1)
        logits = h @ W2 + b2
        counts = logits.exp()
        prob = counts / torch.sum(counts, dim=1, keepdim=True)
        next_draw = torch.multinomial(prob, 1, replacement=True)

        # next_draw = logits.argmax()
        chars.append(itos[next_draw.item()])
        start_from = torch.cat([start_from[1:], torch.tensor([next_draw])])
        k += 1
        if k > 10:
            break
    print("".join(chars))

# %%
start_from
# %%
next_draw
# %%
