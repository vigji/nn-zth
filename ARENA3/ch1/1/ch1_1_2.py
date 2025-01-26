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
# import eindex # import eindex
from einindex import index as eindex
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

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# Make sure exercises are in the path
chapter = "chapter1_transformer_interp"
section = "part2_intro_to_mech_interp"

import tests as tests
from plotly_utils import hist, imshow, plot_comp_scores, plot_logit_attribution, plot_loss_difference

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

MAIN = __name__ == "__main__"
# %%
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
# %%

from huggingface_hub import hf_hub_download

REPO_ID = "callummcdougall/attn_only_2L_half"
FILENAME = "attn_only_2L_half.pth"

weights_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

model = HookedTransformer(cfg)
pretrained_weights = t.load(weights_path, map_location=device, weights_only=True)
model.load_state_dict(pretrained_weights)
# %%

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

seq_len = 50
batch_size = 1
generate_repeated_tokens(model, seq_len, batch_size)
# %%
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


def get_log_probs(
    logits: Float[Tensor, "batch posn d_vocab"], tokens: Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:
    logprobs = logits.log_softmax(dim=-1)
    # We want to get logprobs[b, s, tokens[b, s+1]], in eindex syntax this looks like:
    # correct_logprobs = eindex(logprobs, tokens, "b s [b s+1]")
    batch_size, seq_length = tokens.shape
    batch_indices = t.arange(batch_size).unsqueeze(1)  # Shape: [batch_size, 1]
    sequence_indices = t.arange(seq_length - 1)  # Shape: [seq_length - 1]
    selected_tokens = tokens[:, 1:]  # Adjust tokens for s+1 indexing tokens[b, s+1]
    # Gather logprobs[b, s, tokens[b, s+1]]
    result = logprobs[batch_indices, sequence_indices, selected_tokens]


    return result


seq_len = 50
batch_size = 1
(rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(model, seq_len, batch_size)
rep_cache.remove_batch_dim()
rep_str = model.to_str_tokens(rep_tokens)
model.reset_hooks()
log_probs = get_log_probs(rep_logits, rep_tokens).squeeze()


def hook_function(
    attn_pattern: Float[Tensor, "batch heads seq_len seq_len"],
    hook: HookPoint
): # -> Tensor["batch", "heads", "seq_len", "seq_len"]:

    # modify attn_pattern (can be inplace)
    return attn_pattern

loss = model.run_with_hooks(
    rep_tokens,
    return_type="loss",
    fwd_hooks=[
        ('blocks.1.attn.hook_pattern', hook_function)
    ]
)
# %%
seq_len = 50
batch_size = 10
rep_tokens_10 = generate_repeated_tokens(model, seq_len, batch_size)

# We make a tensor to store the induction score for each head.
# We put it on the model's device to avoid needing to move things between the GPU and CPU, which can be slow.
induction_score_store = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)


def induction_score_hook(pattern: Float[Tensor, "batch head_index dest_pos source_pos"], hook: HookPoint):
    """
    Calculates the induction score, and stores it in the [layer, head] position of the `induction_score_store` tensor.
    """
    i_layer = int(hook.name.split(".")[1])
    seq_length = (pattern.shape[-1] - 1) // 2
    scores = pattern.diagonal(-(seq_length-1), dim1=2, dim2=3).mean((0, 2))
    induction_score_store[i_layer, :] = scores

    return pattern


# We make a boolean filter on activation names, that's true only on attention pattern names
pattern_hook_names_filter = lambda name: name.endswith("pattern")

# Run with hooks (this is where we write to the `induction_score_store` tensor`)
model.run_with_hooks(
    rep_tokens_10,
    return_type=None,  # For efficiency, we don't need to calculate the logits
    fwd_hooks=[(pattern_hook_names_filter, induction_score_hook)],
)

# Plot the induction scores for each head in each layer
imshow(
    induction_score_store,
    labels={"x": "Head", "y": "Layer"},
    title="Induction Score by Head",
    text_auto=".2f",
    width=900,
    height=350,
)
# %%
gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")

induction_score_store = t.zeros((gpt2_small.cfg.n_layers, gpt2_small.cfg.n_heads), device=model.cfg.device)

def induction_score_hook(pattern: Float[Tensor, "batch head_index dest_pos source_pos"], hook: HookPoint):
    """
    Calculates the induction score, and stores it in the [layer, head] position of the `induction_score_store` tensor.
    """
    i_layer = int(hook.name.split(".")[1])
    seq_length = (pattern.shape[-1] - 1) // 2
    scores = pattern.diagonal(-(seq_length - 1), dim1=2, dim2=3).mean((0, 2))
    induction_score_store[i_layer, :] = scores

    return pattern

# Run with hooks (this is where we write to the `induction_score_store` tensor`)
gpt2_small.run_with_hooks(
    rep_tokens_10,
    return_type=None,  # For efficiency, we don't need to calculate the logits
    fwd_hooks=[(pattern_hook_names_filter, induction_score_hook)],
)
# Plot the induction scores for each head in each layer
imshow(
    induction_score_store,
    labels={"x": "Head", "y": "Layer"},
    title="Induction Score by Head",
    text_auto=".2f",
    width=900,
    height=350,
    zmin=-1,
    zmax=1
)
# %%

def visualize_pattern_hook(
    pattern: Float[Tensor, "batch head_index dest_pos source_pos"],
    hook: HookPoint,
):
    print("Layer: ", hook.layer())
    display(cv.attention.attention_patterns(tokens=gpt2_small.to_str_tokens(rep_tokens[0]), attention=pattern.mean(0)))

# %%
# Run with hooks (this is where we write to the `induction_score_store` tensor`)
gpt2_small.run_with_hooks(
    rep_tokens_10,
    return_type=None,  # For efficiency, we don't need to calculate the logits
    fwd_hooks=[(pattern_hook_names_filter, visualize_pattern_hook)],
)

# %%
text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
logits, cache = model.run_with_cache(text, remove_batch_dim=True)
str_tokens = model.to_str_tokens(text)
tokens = model.to_tokens(text)
# %%

def logit_attribution(
    embed: Float[Tensor, "seq d_model"],
    l1_results: Float[Tensor, "seq nheads d_model"],
    l2_results: Float[Tensor, "seq nheads d_model"],
    W_U: Float[Tensor, "d_model d_vocab"],
    tokens: Int[Tensor, "seq"],
) -> Float[Tensor, "seq-1 n_components"]:
    """
    Inputs:
        embed: the embeddings of the tokens (i.e. token + position embeddings)
        l1_results: the outputs of the attention heads at layer 1 (with head as one of the dimensions)
        l2_results: the outputs of the attention heads at layer 2 (with head as one of the dimensions)
        W_U: the unembedding matrix
        tokens: the token ids of the sequence

    Returns:
        Tensor of shape (seq_len-1, n_components)
        represents the concatenation (along dim=-1) of logit attributions from:
            the direct path (seq-1, 1)
            layer 0 logits (seq-1, n_heads)
            layer 1 logits (seq-1, n_heads)
        so n_components = 1 + 2*n_heads
    """
    W_U_correct_tokens = W_U[:, tokens[1:]]
    print(tokens.shape, W_U_correct_tokens.shape, embed.shape, l1_results.shape)
    return t.concat([(einops.einsum(W_U_correct_tokens, embed[:-1, :], "d b, b d -> b").unsqueeze(-1)),
                      einops.einsum(W_U_correct_tokens, l1_results[:-1, :], "d b, b h d -> b h"),
                      einops.einsum(W_U_correct_tokens, l2_results[:-1, :], "d b, b h d -> b h"),
                    ], axis=-1)

with t.inference_mode():
    embed = cache["embed"]
    l1_results = cache["result", 0]
    l2_results = cache["result", 1]
    logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens[0])
    # Uses fancy indexing to get a len(tokens[0])-1 length tensor, where the kth entry is the predicted logit for the correct k+1th token
    correct_token_logits = logits[0, t.arange(len(tokens[0]) - 1), tokens[0, 1:]]
    t.testing.assert_close(logit_attr.sum(1), correct_token_logits, atol=1e-3, rtol=0)
    print("Tests passed!")
# %%
embed = cache["embed"]
l1_results = cache["result", 0]
l2_results = cache["result", 1]
logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens.squeeze())

plot_logit_attribution(model, logit_attr, tokens, title="Logit attribution (demo prompt)")
# %%
embed = rep_cache["embed"]
l1_results = rep_cache["result", 0]
l2_results = rep_cache["result", 1]
logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, rep_tokens.squeeze())

plot_logit_attribution(model, logit_attr, rep_tokens, title="Logit attribution (repeat prompt)")
# %%
