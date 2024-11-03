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
    X = torch.stack([data[idx:idx + block_size] for idx in start_idx]).to(device)
    Y = torch.stack([data[idx+1:idx + block_size+1] for idx in start_idx]).to(device)
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
            target = einops.rearrange(target, 'b t -> (b t)')
            logits = einops.rearrange(logits, 'b t c -> (b t) c')
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
lr=1e-3


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
        xprev = x[b, :t+1, :] # shape t, C
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
weight = torch.masked_fill(torch.zeros((T, T)), ~tril, float('-inf'))
weight_sm = torch.softmax(weight, dim=1)
xbow3 = weight_sm @ x  # this will broacast the batch dimension B

print(torch.allclose(xbow, xbow3))
# %%
weight_sm
# %%
