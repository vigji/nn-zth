# %%
import functools
import sys
from pathlib import Path
import token
from typing import Callable

import circuitsvis as cv
import einops
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm
from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
    utils,
)
from transformer_lens.hook_points import HookPoint

device = t.device(
    "mps"
    if t.backends.mps.is_available()
    else "cuda"
    if t.cuda.is_available()
    else "cpu"
)

# Make sure exercises are in the path
chapter = "chapter1_transformer_interp"
section = "part2_intro_to_mech_interp"

import tests
from plotly_utils import (
    hist,
    imshow,
    plot_comp_scores,
    plot_logit_attribution,
    plot_loss_difference,
)

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

MAIN = __name__ == "__main__"
# %%
cfg = HookedTransformerConfig(
    d_model=768,
    d_head=64,
    n_heads=12,
    n_layers=2,
    n_ctx=2048,
    d_vocab=50278,
    attention_dir="causal",
    attn_only=True,  # defaults to False
    tokenizer_name="EleutherAI/gpt-neox-20b",
    seed=398,
    use_attn_result=True,
    normalization_type=None,  # defaults to "LN", i.e. layernorm with weights & biases
    positional_embedding_type="shortformer",
)

from huggingface_hub import hf_hub_download

REPO_ID = "callummcdougall/attn_only_2L_half"
FILENAME = "attn_only_2L_half.pth"

weights_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

model = HookedTransformer(cfg)
pretrained_weights = t.load(weights_path, map_location=device, weights_only=True)
model.load_state_dict(pretrained_weights)

A = t.randn(5, 2)
B = t.randn(2, 5)
AB = A @ B
AB_factor = FactoredMatrix(A, B)
print("Norms:")
print(AB.norm())
print(AB_factor.norm())

print(
    f"Right dimension: {AB_factor.rdim}, Left dimension: {AB_factor.ldim}, Hidden dimension: {AB_factor.mdim}"
)
# %%

print("Eigenvalues:")
print(t.linalg.eig(AB).eigenvalues)
print(AB_factor.eigenvalues)

print("\nSingular Values:")
print(t.linalg.svd(AB).S)
print(AB_factor.S)

print("\nFull SVD:")
print(AB_factor.svd())
# %%

C = t.randn(5, 300)
ABC = AB @ C
ABC_factor = AB_factor @ C

print(f"Unfactored: shape={ABC.shape}, norm={ABC.norm()}")
print(f"Factored: shape={ABC_factor.shape}, norm={ABC_factor.norm()}")
print(
    f"\nRight dimension: {ABC_factor.rdim}, "
    f"Left dimension: {ABC_factor.ldim}, "
    f"Hidden dimension: {ABC_factor.mdim}"
)

AB_unfactored = AB_factor.AB
t.testing.assert_close(AB_unfactored, AB)
# %%
head_index = 4
layer = 1

# YOUR CODE HERE - compute the `full_OV_circuit` object
W_V = model.get_parameter(f"blocks.{layer}.attn.W_V")[head_index, :, :]
W_O = model.get_parameter(f"blocks.{layer}.attn.W_O")[head_index, :, :]
W_VO = FactoredMatrix(W_V, W_O)
# W_EOV = FactoredMatrix(model.W_E, W_OV)
# W_EOVU = FactoredMatrix(W_EOV, model.W_U)

full_OV_circuit = model.W_E @ W_VO @ model.W_U   # @ model.W_E
tests.test_full_OV_circuit(full_OV_circuit, model, layer, head_index)
# %%

# %%
imshow(
    full_OV_circuit_sample,
    labels={"x": "Logits on output token", "y": "Input token"},
    title="Full OV circuit for copying head",
    width=700,
    height=600,
)
# %%
def top_1_acc(full_OV_circuit: FactoredMatrix, batch_size: int = 1000) -> float:
    """
    Return the fraction of the time that the maximum value is on the circuit diagonal.
    """
    matching = 0
    n = full_OV_circuit.shape[0] 
    for i in range(full_OV_circuit.shape[0] // batch_size):  # indices in t.split(t.arange(n, dtype=int, device=device), batch_size):
        # batch_size = 10
        indices = t.arange(batch_size, dtype=int).to(device) + i*batch_size  # t.randint(0, model.cfg.d_vocab, (10,))
        full_OV_circuit_sample = full_OV_circuit[indices, :].AB

        matching += (t.argmax(full_OV_circuit_sample, dim=1) == indices).sum().item()

    return matching / full_OV_circuit.shape[0]


print(f"Fraction of the time that the best logit is on the diagonal: {top_1_acc(full_OV_circuit):.4f}")
# %%
h1, h2 = 4, 10
layer = 1
W_V = model.get_parameter(f"blocks.{layer}.attn.W_V")[[h1, h2], :, :]
W_O = model.get_parameter(f"blocks.{layer}.attn.W_O")[[h1, h2], :, :]
W_V.shape, W_O.shape
W_V = einops.rearrange(W_V, "n d h -> d (n h)")
W_O = einops.rearrange(W_O, "n h d -> (n h) d")

# %%
W_VO4_10 = FactoredMatrix(W_V, W_O)
full_OV_circuit = model.W_E @ W_VO4_10 @ model.W_U
print(f"Fraction of the time that the best logit is on the diagonal: {top_1_acc(full_OV_circuit):.4f}")
# %%
layer = 0
head_index = 7

# Compute full QK matrix (for positional embeddings)
W_pos = model.W_pos
W_QK = model.W_Q[layer, head_index] @ model.W_K[layer, head_index].T
pos_by_pos_scores = W_pos @ W_QK @ W_pos.T

# Mask, scale and softmax the scores
mask = t.tril(t.ones_like(pos_by_pos_scores)).bool()
pos_by_pos_pattern = t.where(mask, pos_by_pos_scores / model.cfg.d_head**0.5, -1.0e6).softmax(-1)

# Plot the results
print(f"Avg lower-diagonal value: {pos_by_pos_pattern.diag(-1).mean():.4f}")
imshow(
    utils.to_numpy(pos_by_pos_pattern[:200, :200]),
    labels={"x": "Key", "y": "Query"},
    title="Attention patterns for prev-token QK circuit, first 100 indices",
    width=700,
    height=600,
)
# %%
