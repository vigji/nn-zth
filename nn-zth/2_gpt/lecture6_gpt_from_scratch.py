# %%
from regex import B
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
import einops

with open("tiny_squotilancia.txt", "r") as f:
    text = f.read()

device = "mps"

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
Y = train_data[1 : block_size + 1]
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
    X = torch.stack([data[idx : idx + block_size] for idx in start_idx]).to(device)
    Y = torch.stack([data[idx + 1 : idx + block_size + 1] for idx in start_idx]).to(
        device
    )
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

    def forward(self, context, target=None):
        logits = self.token_embedding_table(context)
        # print(logits.shape)

        if target is not None:
            target = einops.rearrange(target, "b t -> (b t)")
            logits = einops.rearrange(logits, "b t c -> (b t) c")
            loss = F.cross_entropy(logits, target)
        else:
            loss = None

        return logits, loss

    @torch.no_grad
    def generate(self, context, max_n_tokens):
        for _ in range(max_n_tokens):
            logits, _ = self(context, None)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)

            context = torch.cat([context, next_token], dim=1)

        return context


@torch.no_grad
def eval_loss(model, loss_iters=300):
    loss_dict = {}
    model.eval()
    for split in "train", "val":
        losses = torch.zeros(loss_iters)
        for i in range(loss_iters):
            xb, yb = get_batch(split=split)
            _, loss = model(xb, yb)

            losses[i] = loss
        loss_dict[split] = losses.mean().item()

    model.train()
    return loss_dict


def auto_generate(model, points=100):
    starting_point = torch.zeros((1, 1), dtype=torch.long).to(device)
    pred = model.generate(starting_point, points)
    print(decode(pred[0].tolist()))


# %%
batch_size = 32
# training loop:
n_batches = 100000
lr = 1e-3


m = BigramModel(vocab_size).to(device)
logits, loss = m(xb, yb)
# print(logits.shape)
# print(loss)
auto_generate(m)

# optimizer:
optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)

print(eval_loss(m))


for i in tqdm(range(n_batches)):
    xs, ys = get_batch("train", batch_size=batch_size)

    logits, loss = m(xs, ys)
    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    optimizer.step()
print(eval_loss(m))
# %%
auto_generate(m, points=1000)
# %%
# The trick of self attention!
torch.manual_seed(1337)
B, T, C = 4, 8, 2
x = torch.randn((B, T, C))

# For every batch, for every T, let's try to calculate average up to current token

xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, : t + 1, :]  # shape t, C
        xbow[b, t] = xprev.mean(dim=0)

# To see how to do this efficienly with matmult, look at this:
a = torch.ones((3, 3))
b = torch.randint(0, 10, (3, 2)).float()
c = a @ b  # this has columns sum in every entry of each column
print(a)
print(b)
print(c)
print("--------")
# and now a trick:
a = torch.tril(a)
c = a @ b  # this now becomes a cumulative sum for each row!

print(a)
print(b)
print(c)
print("--------")
# for the rolling average we can simply:
c /= a.sum(dim=1, keepdim=True)
print(c)

# %%
# Now back to our original matrix multiplication:
torch.manual_seed(1337)
B, T, C = 4, 8, 2
x = torch.randn((B, T, C))

weight = torch.tril(torch.ones((T, T)))
weight /= weight.sum(dim=1, keepdim=True)
xbow2 = weight @ x  # this will broacast the batch dimension B

print(torch.allclose(xbow, xbow2))

# %%
# What we will actually do with self attention is to use weights and normalize them
# using softmax:
xbow2[0, :, :], x[0, :, :]

# %%
tril = torch.tril(torch.ones((T, T), dtype=bool))
weight = torch.masked_fill(torch.zeros((T, T)), ~tril, float("-inf"))
weight_sm = torch.softmax(weight, dim=1)
xbow3 = weight_sm @ x  # this will broacast the batch dimension B

print(torch.allclose(xbow, xbow3))
# %%
weight_sm
# %%
# self attention:
torch.manual_seed(1337)
B, T, C = 4, 8, 32
x = torch.randn((B, T, C))

head_size = 16
query = nn.Linear(C, head_size, bias=False)
key = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)


# in self-attention all those are computed from x. in cross attention v and k comes from
# a separate input
q = query(x)  # broadcasting what i look for
k = key(x)  # broadcasting what i have
v = value(x)  # actual passed values
q.shape, k.shape
weights = q @ einops.rearrange(k, "b t c -> b c t")

# to avoi passing too peaky distributions inside softmax, we first normalize here:
weights *= head_size**-0.5

# trianular masking happens in a decoder head, encoder heads do not have it and all tokens
# can look at all other tokens.
tril = torch.tril(torch.ones((T, T), dtype=bool))
weight = torch.masked_fill(torch.zeros((T, T)), ~tril, float("-inf"))
weight = torch.softmax(weight, dim=1)
out = weight @ v  # this will broacast the batch dimension B


# %%

# %%
# Moving toward transformer models:

import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # tril is not really a parameter of the mode:
        self.register_buffer("tril", torch.tril(torch.ones((block_size, block_size), dtype=bool)))

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)  # broadcasting what i look for
        k = self.key(x)  # broadcasting what i have
        v = self.value(x)  # actual passed values
        # to avoid passing too peaky distributions inside softmax, we first normalize here:
        weight = q @ einops.rearrange(k, "b t c -> b c t") * C**-0.5

        # trianular masking happens in a decoder head, encoder heads do not have it and all tokens
        # can look at all other tokens.
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weight = torch.softmax(weight, dim=-1)
        out = weight @ v 
        return  out  # this will broacast the batch dimension B


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embs, block_size=8, head_size=16):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embs)
        self.positional_embedding_table = nn.Embedding(block_size, n_embs)
        self.sa_head = Head(n_embs)
        self.lm_head = nn.Linear(n_embs, vocab_size)

    def forward(self, context, target=None):
        B, T = context.shape
        embs = self.token_embedding_table(context)
        pos_embs = self.positional_embedding_table(torch.arange(T, device=device))
        x = embs + pos_embs
        x = self.sa_head(x)
        logits = self.lm_head(x)

        if target is None:
            loss = None
        else:
            target = einops.rearrange(target, "b t -> (b t)")
            logits = einops.rearrange(logits, "b t c -> (b t) c")
            loss = F.cross_entropy(logits, target)

        return logits, loss

    @torch.no_grad
    def generate(self, context, max_n_tokens):
        for i in range(max_n_tokens):
            context_crop = context[:, -self.block_size:]
            

            logits, _ = self(context_crop, None)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)

            context = torch.cat([context, next_token], dim=1)

        return context
    

def auto_generate(model, points=100):
    starting_point = torch.zeros((1, 8), dtype=torch.long).to(device)
    pred = model.generate(starting_point, points)
    print(decode(pred[0].tolist()))


# adjusting hyperparams:
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
eval_iters = 200
n_embd = 32
head_size = 16


torch.manual_seed(1337)

m = BigramLanguageModel(vocab_size=vocab_size, n_embs=n_embd, 
                        block_size=block_size, head_size=head_size).to(device)
logits, loss = m(xb, yb)
# print(logits.shape)
# print(loss)
auto_generate(m)

# optimizer:
optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)

print(eval_loss(m))


for i in tqdm(range(max_iters)):
    xs, ys = get_batch("train", batch_size=batch_size)

    logits, loss = m(xs, ys)
    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    optimizer.step()
print(eval_loss(m))

# %%
# What about multiple heads? We just get a bunch of heads to process things in parallel
# and concatenate their output:

class MultipleHead(nn.Module):
    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        self.multi_heads = nn.ModuleList([Head(head_size=head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([head(x) for head in self.multi_heads], dim=-1)
    

class MultiheadBigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embs, block_size=8, head_size=16):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embs)
        self.positional_embedding_table = nn.Embedding(block_size, n_embs)
        self.sa_head = MultipleHead(4, n_embs // 4)
        self.lm_head = nn.Linear(n_embs, vocab_size)

    def forward(self, context, target=None):
        B, T = context.shape
        embs = self.token_embedding_table(context)
        pos_embs = self.positional_embedding_table(torch.arange(T, device=device))
        x = embs + pos_embs
        x = self.sa_head(x)
        logits = self.lm_head(x)

        if target is None:
            loss = None
        else:
            target = einops.rearrange(target, "b t -> (b t)")
            logits = einops.rearrange(logits, "b t c -> (b t) c")
            loss = F.cross_entropy(logits, target)

        return logits, loss

    @torch.no_grad
    def generate(self, context, max_n_tokens):
        for i in range(max_n_tokens):
            context_crop = context[:, -self.block_size:]
            

            logits, _ = self(context_crop, None)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)

            context = torch.cat([context, next_token], dim=1)

        return context
    
torch.manual_seed(1337)

m = MultiheadBigramLanguageModel(vocab_size=vocab_size, n_embs=n_embd, 
                        block_size=block_size, head_size=head_size).to(device)
logits, loss = m(xb, yb)
# print(logits.shape)
# print(loss)
auto_generate(m)

# optimizer:
optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)

print(eval_loss(m))


for i in tqdm(range(max_iters)):
    xs, ys = get_batch("train", batch_size=batch_size)

    logits, loss = m(xs, ys)
    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    optimizer.step()
print(eval_loss(m))


        
# %%
# Add a feedforward layer, so that the network can think about the results of the head:

class FeedForward(nn.Module):
    def __init__(self, fan_in, fan_out) -> None:
        super().__init__()
        self.linear = nn.Linear(fan_in, fan_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.linear(x))

        return out
    
class FFMultiheadBigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embs, block_size=8, head_size=16):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embs)
        self.positional_embedding_table = nn.Embedding(block_size, n_embs)
        self.sa_head = MultipleHead(4, n_embs // 4)
        self.feedforward = FeedForward(n_embs, n_embs)
        self.lm_head = nn.Linear(n_embs, vocab_size)

    def forward(self, context, target=None):
        B, T = context.shape
        embs = self.token_embedding_table(context)
        pos_embs = self.positional_embedding_table(torch.arange(T, device=device))
        x = embs + pos_embs
        x = self.sa_head(x)
        # x = self.lm_head(x)
        x = self.feedforward(x)
        logits = self.lm_head(x)

        if target is None:
            loss = None
        else:
            target = einops.rearrange(target, "b t -> (b t)")
            logits = einops.rearrange(logits, "b t c -> (b t) c")
            loss = F.cross_entropy(logits, target)

        return logits, loss

    @torch.no_grad
    def generate(self, context, max_n_tokens):
        for i in range(max_n_tokens):
            context_crop = context[:, -self.block_size:]
            

            logits, _ = self(context_crop, None)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)

            context = torch.cat([context, next_token], dim=1)

        return context
    
torch.manual_seed(1337)

m = FFMultiheadBigramLanguageModel(vocab_size=vocab_size, n_embs=n_embd, 
                        block_size=block_size, head_size=head_size).to(device)
logits, loss = m(xb, yb)
# print(logits.shape)
# print(loss)
auto_generate(m)

# optimizer:
optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)

print(eval_loss(m))


for i in tqdm(range(max_iters)):
    xs, ys = get_batch("train", batch_size=batch_size)

    logits, loss = m(xs, ys)
    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    optimizer.step()
print(eval_loss(m))
# %%
