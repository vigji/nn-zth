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
