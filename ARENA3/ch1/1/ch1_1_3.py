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

def run_and_cache_model_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch_size: int = 1
) -> tuple[Tensor, Tensor, ActivationCache]:
    """
    Generates a sequence of repeated random tokens, and runs the model on it, returning (tokens, logits, cache). This
    function should use the `generate_repeated_tokens` function above

    Outputs are:
        rep_tokens: [batch_size, 1+2*seq_len]
        rep_logits: [batch_size, 1+2*seq_len, d_vocab]
        rep_cache: The cache of the model run on rep_tokens
    """
    tokens_seq = generate_repeated_tokens(model, seq_len, batch_size)

    logits, cache = model.run_with_cache(tokens_seq)

    return tokens_seq, logits, cache

# %%
def generate_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch_size: int = 1
) -> Int[Tensor, "batch_size full_seq_len"]:
    """
    Generates a sequence of repeated random tokens

    Outputs are:
        rep_tokens: [batch_size, 1+2*seq_len]
    """
    prefix = (t.ones(batch_size, 1) * model.tokenizer.bos_token_id).long()

    random_seq = t.randint(0, model.cfg.d_vocab, (batch_size, seq_len))

    return t.concat([prefix, random_seq, random_seq], axis=-1)

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


def decompose_qk_input(cache: ActivationCache) -> Float[Tensor, "n_heads+2 posn d_model"]:
    """
    Retrieves all the input tensors to the first attention layer, and concatenates them along the 0th dim.

    The [i, 0, 0]th element is y_i (from notation above). The sum of these tensors along the 0th dim should
    be the input to the first attention layer.
    """
    return t.concat([rep_cache["hook_embed"].unsqueeze(0),
                    rep_cache["hook_pos_embed"].unsqueeze(0),
                    einops.rearrange(rep_cache["blocks.0.attn.hook_result"], "posn n_h d -> n_h posn d")])


def decompose_q(
    decomposed_qk_input: Float[Tensor, "n_heads+2 posn d_model"],
    ind_head_index: int,
    model: HookedTransformer,
) -> Float[Tensor, "n_heads+2 posn d_head"]:
    """
    Computes the tensor of query vectors for each decomposed QK input.

    The [i, :, :]th element is y_i @ W_Q (so the sum along axis 0 is just the q-values).
    """

    return einops.einsum(decomposed_qk_input, 
                         model.get_parameter(f"blocks.1.attn.W_Q")[ind_head_index, :, :],
                         "n posn d_model, d_model d_head -> n posn d_head")


def decompose_k(
    decomposed_qk_input: Float[Tensor, "n_heads+2 posn d_model"],
    ind_head_index: int,
    model: HookedTransformer,
) -> Float[Tensor, "n_heads+2 posn d_head"]:
    """
    Computes the tensor of key vectors for each decomposed QK input.

    The [i, :, :]th element is y_i @ W_K(so the sum along axis 0 is just the k-values)
    """
    return einops.einsum(decomposed_qk_input, 
                         model.get_parameter(f"blocks.1.attn.W_K")[ind_head_index, :, :],
                         "n posn d_model, d_model d_head -> n posn d_head")


# Recompute rep tokens/logits/cache, if we haven't already
seq_len = 50
batch_size = 1
(rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(model, seq_len, batch_size)
rep_cache.remove_batch_dim()

ind_head_index = 4

# First we get decomposed q and k input, and check they're what we expect
decomposed_qk_input = decompose_qk_input(rep_cache)
decomposed_q = decompose_q(decomposed_qk_input, ind_head_index, model)
decomposed_k = decompose_k(decomposed_qk_input, ind_head_index, model)
t.testing.assert_close(
    decomposed_qk_input.sum(0), rep_cache["resid_pre", 1] + rep_cache["pos_embed"], rtol=0.01, atol=1e-05
)
t.testing.assert_close(decomposed_q.sum(0), rep_cache["q", 1][:, ind_head_index], rtol=0.01, atol=0.001)
t.testing.assert_close(decomposed_k.sum(0), rep_cache["k", 1][:, ind_head_index], rtol=0.01, atol=0.01)

# Second, we plot our results
component_labels = ["Embed", "PosEmbed"] + [f"0.{h}" for h in range(model.cfg.n_heads)]
for decomposed_input, name in [(decomposed_q, "query"), (decomposed_k, "key")]:
    imshow(
        utils.to_numpy(decomposed_input.pow(2).sum([-1])),
        labels={"x": "Position", "y": "Component"},
        title=f"Norms of components of {name}",
        y=component_labels,
        width=800,
        height=400,
    )
# %%
def decompose_attn_scores(
    decomposed_q: Float[Tensor, "q_comp q_pos d_model"],
    decomposed_k: Float[Tensor, "k_comp k_pos d_model"],
) -> Float[Tensor, "q_comp k_comp q_pos k_pos"]:
    """
    Output is decomposed_scores with shape [query_component, key_component, query_pos, key_pos]

    The [i, j, 0, 0]th element is y_i @ W_QK @ y_j^T (so the sum along both first axes are the attention scores)
    """
    return einops.einsum(decomposed_q, decomposed_k, 
                        "q_comp q_pos d_model, k_comp k_pos d_model -> q_comp k_comp q_pos k_pos")


tests.test_decompose_attn_scores(decompose_attn_scores, decomposed_q, decomposed_k)
# %%
# First plot: attention score contribution from (query_component, key_component) = (Embed, L0H7), you can replace this
# with any other pair and see that the values are generally much smaller, i.e. this pair dominates the attention score
# calculation
decomposed_scores = decompose_attn_scores(decomposed_q, decomposed_k)

q_label = "Embed"
k_label = "0.6"
decomposed_scores_from_pair = decomposed_scores[component_labels.index(q_label), component_labels.index(k_label)]

imshow(
    utils.to_numpy(t.tril(decomposed_scores_from_pair)),
    title=f"Attention score contributions from query = {q_label}, key = {k_label}<br>(by query & key sequence positions)",
    width=700,
)


# Second plot: std dev over query and key positions, shown by component. This shows us that the other pairs of
# (query_component, key_component) are much less important, without us having to look at each one individually like we
# did in the first plot!
decomposed_stds = einops.reduce(
    decomposed_scores, "query_decomp key_decomp query_pos key_pos -> query_decomp key_decomp", t.std
)
imshow(
    utils.to_numpy(decomposed_stds),
    labels={"x": "Key Component", "y": "Query Component"},
    title="Std dev of attn score contributions across sequence positions<br>(by query & key component)",
    x=component_labels,
    y=component_labels,
    width=700,
)
# %%
def find_K_comp_full_circuit(
    model: HookedTransformer, prev_token_head_index: int, ind_head_index: int
) -> FactoredMatrix:
    """
    Returns a (vocab, vocab)-size FactoredMatrix, with the first dimension being the query side (direct from token
    embeddings) and the second dimension being the key side (going via the previous token head).
    """
    # W_OV = model.get_parameter(f"blocks.0.attn.W_K")[prev_token_head_index, :, :]
    # W_QK = model.get_parameter(f"blocks.1.attn.W_K")[ind_head_index, :, :]

    W_E = model.get_parameter("W_E")
    W_O = model.get_parameter("blocks.0.attn.W_O")[prev_token_head_index, :, :]
    W_V = model.get_parameter("blocks.0.attn.W_V")[prev_token_head_index, :, :]
    W_K = model.get_parameter("blocks.1.attn.W_K")[ind_head_index, :, :]
    W_Q = model.get_parameter("blocks.1.attn.W_Q")[ind_head_index, : ,:]

    Q = W_E @ W_Q
    K = W_E @ W_V @ W_O @ W_K

    return FactoredMatrix(Q, K.T)





prev_token_head_index = 7
ind_head_index = 10
K_comp_circuit = find_K_comp_full_circuit(model, prev_token_head_index, ind_head_index)

tests.test_find_K_comp_full_circuit(find_K_comp_full_circuit, model)

print(f"Fraction of tokens where the highest activating key is the same token: {top_1_acc(K_comp_circuit.T):.4f}")
g# %%
model.get_parameter("blocks.0.attn.W_O").shape
# %%
