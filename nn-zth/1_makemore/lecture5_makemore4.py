# %%
# Let's improve our generator in the direction of Bangio et al 2006
from matplotlib import cm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import einops as ein
import numpy as np

names_file = Path(__file__).parent / "names.txt"
words = open(names_file).read().splitlines()
PAD_CH = "."
device = "cpu"


n_possible_chars = 27
# N = torch.zeros((possible_chars, possible_chars), dtype=torch.int32)
chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi[PAD_CH] = 0
itos = {s: i for i, s in stoi.items()}

import random

random.seed(42)
random.shuffle(words)

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
