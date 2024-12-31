# %%
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from aiohttp import DataQueue
import circuitsvis as cv
import datasets
import einops
import einops.layers
import einops.layers.torch
from matplotlib import axes
import numpy as np
import torch as t
import torch
import torch.nn as nn
import wandb
from IPython.display import display
from jaxtyping import Float, Int
from rich import print as rprint
from rich.table import Table
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utils import gelu_new, tokenize_and_concatenate
from transformers import PreTrainedTokenizerFast
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast

# Make sure exercises are in the path
chapter = r"chapter1_transformer_interp"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part1_transformer_from_scratch"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

import solutions as solutions
import tests as tests

device = t.device('mps' if t.backends.mps.is_available() else 'cuda' if t.cuda.is_available() else 'cpu')

MAIN = __name__ == '__main__'

reference_gpt2 = HookedTransformer.from_pretrained(
    "gpt2-small",
    fold_ln=False,
    center_unembed=False,
    center_writing_weights=False,
    device=device
)

sorted_vocab = sorted(list(reference_gpt2.tokenizer.vocab.items()), key=lambda n: n[1])
print(sorted_vocab[:20])
print()
print(sorted_vocab[250:270])
print()
print(sorted_vocab[990:1010])
print()
print(sorted_vocab[-20:])

# %%
lengths = dict.fromkeys(range(3, 8), "")

for tok, idx in sorted_vocab:
    if not lengths.get(len(tok), True):
        lengths[len(tok)] = tok

print(lengths)

# %%
print("Different tokens for same word:")
print(reference_gpt2.to_str_tokens("Ralph"))
print(reference_gpt2.to_str_tokens(" Ralph"))
print(reference_gpt2.to_str_tokens(" ralph"))
print(reference_gpt2.to_str_tokens("ralph"))

print("Aritmetics:")
print(print(reference_gpt2.to_str_tokens("56873+3184623=123456789-1000000000")))

# %%
print("Sequence: ")
reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
tokens = reference_gpt2.to_tokens(reference_text).to(device)
print(tokens)
print(tokens.shape)
print(reference_gpt2.to_str_tokens(tokens))

# %%
print("Prediction of logits: ")
logits, cache = reference_gpt2.run_with_cache(tokens, device=device)
print(logits.shape, logits[:5])

# %%
print("Probabilities:")
probs = logits.softmax(dim=-1)
print(probs.shape, probs[:5])

# %%
print("Predictions at each step: ")
most_likely_next_tokens = reference_gpt2.tokenizer.batch_decode(logits.argmax(dim=-1)[0])

print(list(zip(reference_gpt2.to_str_tokens(tokens), most_likely_next_tokens)))


# %%
next_token = logits[0, -1].argmax(dim=-1)
next_char = reference_gpt2.to_string(next_token)
print(repr(next_char))

print(f"Sequence so far: {reference_gpt2.to_string(tokens)[0]!r}")

for i in range(20):
    print(f"{tokens.shape[-1]+1}th char = {next_char!r}")
    # Define new input sequence, by appending the previously generated token
    tokens = t.cat([tokens, next_token[None, None]], dim=-1)
    # Pass our new sequence through the model, to get new output
    logits = reference_gpt2(tokens)
    # Get the predicted token at the end of our sequence
    next_token = logits[0, -1].argmax(dim=-1)
    # Decode and print the result
    next_char = reference_gpt2.to_string(next_token)

#%%
batch = 1
position = 35
d_model = 768
n_heads = 12
n_layers = 12
d_mlp = 4 * d_model
d_head = d_model // n_heads

# %%
for activation_name, activation in cache.items():
    # Only print for first layer
    #if ".0." in activation_name or "blocks" not in activation_name:
        print(f"{activation_name:30} {tuple(activation.shape)}")
        
# %%
param_count = 0
for name, param in reference_gpt2.named_parameters():
    # Only print for first layer
    param_count += param.numel()
    if ".0." in name or "blocks" not in name:
        print(f"{name:18} {tuple(param.shape)}")

param_count
# %%

# As a reference - note there's a lot of stuff we don't care about in here, to do with library internals or other architectures
print(reference_gpt2.cfg)
# %%
@dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12


cfg = Config()
print(cfg)

# %%
def rand_float_test(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    random_input = t.randn(shape).to(device)
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    if isinstance(output, tuple): output = output[0]
    print("Output shape:", output.shape, "\n")

def rand_int_test(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    random_input = t.randint(100, 1000, shape).to(device)
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    if isinstance(output, tuple): output = output[0]
    print("Output shape:", output.shape, "\n")

def load_gpt2_test(cls, gpt2_layer, input):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    layer.load_state_dict(gpt2_layer.state_dict(), strict=False)
    print("Input shape:", input.shape)
    output = layer(input)
    if isinstance(output, tuple): output = output[0]
    print("Output shape:", output.shape)
    try: reference_output = gpt2_layer(input)
    except: reference_output = gpt2_layer(input, input, input)
    print("Reference output shape:", reference_output.shape, "\n")
    comparison = t.isclose(output, reference_output, atol=1e-4, rtol=1e-3)
    print(f"{comparison.sum()/comparison.numel():.2%} of the values are correct\n")

# %%
###################
# LayerNorm
###################

class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(self, residual: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        if self.cfg.debug:
            print("residual shape: ", residual.shape)

        means = t.mean(residual, axis=-1, keepdim=True)
        vars = t.var(residual, axis=-1, keepdim=True, unbiased=False)

        normalized = (residual - means) / t.sqrt(vars + self.cfg.layer_norm_eps)
        scaled = normalized * self.w + self.b

        return scaled

rand_float_test(LayerNorm, [2, 4, 768])
load_gpt2_test(LayerNorm, reference_gpt2.ln_final, cache["resid_post", 11])
zero_input = t.zeros_like(cache["resid_post", 11]).to(device)
load_gpt2_test(LayerNorm, reference_gpt2.ln_final, zero_input)

# %%
a = t.randint(0, 50, (2, 4))
W = t.randn(50, 42)
# einops.einsum(a, W, "b s v, v d -> b s d").shape
# from einops.layers.torch import 
# einops.layers.torch
W[a, :].shape
# %%
######################
# Embedding
######################

class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
        if self.cfg.debug:
            print("Input shape: ", tokens.shape)
        return self.W_E[tokens, :]


rand_int_test(Embed, [2, 4])
load_gpt2_test(Embed, reference_gpt2.embed, tokens)
# %%
#########################
# Positional embedding
#########################

class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
        if self.cfg.debug:
            print("Input shape: ", tokens.shape)

        indices = t.stack([t.arange(tokens.shape[1]) for _ in range(tokens.shape[0])])

        return self.W_pos[indices, :]


rand_int_test(PosEmbed, [2, 4])
load_gpt2_test(PosEmbed, reference_gpt2.pos_embed, tokens)

# %%
###############################
# Full attention implementation
###############################


class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.register_buffer("IGNORE", t.tensor(float("-inf"), 
                                                dtype=t.float32, 
                                                device=device))
        
    def forward(self, normalized_resid_pre: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        # Calculate query, key and value vectors
        q = (
            einops.einsum(
                normalized_resid_pre, self.W_Q, "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head"
            )
            + self.b_Q
        )
        k = (
            einops.einsum(
                normalized_resid_pre, self.W_K, "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head"
            )
            + self.b_K
        )
        v = (
            einops.einsum(
                normalized_resid_pre, self.W_V, "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head"
            )
            + self.b_V
        )
        # print(q[0, 0, :2, :2])
        # print(k[0, 0, :2, :2])
        # print(v[0, 0, :2, :2])

        # Calculate attention scores, then scale and mask, and apply softmax to get probabilities
        attn_scores = einops.einsum(
            q, k, "batch posn_Q nheads d_head, batch posn_K nheads d_head -> batch nheads posn_Q posn_K"
        )
        attn_scores_masked = self.apply_causal_mask(attn_scores / self.cfg.d_head**0.5)
        attn_pattern = attn_scores_masked.softmax(-1)
        # print(attn_pattern[0, 0, :3, :3])

        # Take weighted sum of value vectors, according to attention probabilities
        # print(v[0, 0, :3, :3])
        # print(attn_pattern[0, 0, :3, :3])
        z = einops.einsum(
            v, attn_pattern, "batch posn_K nheads d_head, batch nheads posn_Q posn_K -> batch posn_Q nheads d_head"
        )
        print(z[0, 0, :3, :3])

        # Calculate output (by applying matrix W_O and summing over heads, then adding bias b_O)
        attn_out = (
            einops.einsum(z, self.W_O, "batch posn_Q nheads d_head, nheads d_head d_model -> batch posn_Q d_model")
            + self.b_O
        )

        return attn_out

    def forward_mine(self, normalized_resid_pre: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        # Compute Q, K and V activations:
        q_activations = einops.einsum(normalized_resid_pre, self.W_Q, 
                                 "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head") + self.b_Q
        k_activations = einops.einsum(normalized_resid_pre, self.W_K, 
                                 "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head") + self.b_K
        v_activations = einops.einsum(normalized_resid_pre, self.W_V, 
                                 "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head") + self.b_V
        # print(q_activations[0, 0, :2, :2])
        # print(k_activations[0, 0, :2, :2])
        # print(v_activations[0, 0, :2, :2])

        # Get attention matrix:
        attention_logits = einops.einsum(q_activations, k_activations, 
                                 "batch posq n_heads d_head, batch posk n_heads d_head -> batch n_heads posq posk")

        normed_masked_attention_logits = self.apply_causal_mask(
            attention_logits / t.sqrt(t.tensor(cfg.d_head, dtype=t.float32, device=device)))
        attention_p = t.softmax(normed_masked_attention_logits, dim=3)
        # print(attention_p[0, 0, :3, :3])

        # from sol:
        # attn_scores_masked = self.apply_causal_mask(attention_logits / self.cfg.d_head**0.5)
        # attention_p = attn_scores_masked.softmax(-1)

        # Apply and project to output:
        # print(v_activations[0, 0, :3, :3])
        # print(attention_p[0, 0, :3, :3])

        z = einops.einsum(
            v_activations, attention_p,
            "batch posk n_heads d_head, batch n_heads posq posk -> batch posq n_heads d_head")
        print(z[0, 0, :3, :3])
        
        result = einops.einsum(z, self.W_O, 
                               "batch posq n_heads d_head, n_heads d_head d_model -> batch posq d_model") + self.b_O

        # sum_over_heads = einops.einsum(result,
        #                       "batch posq n_heads d_model -> batch posq d_model")
        
        return result 
        

    def apply_causal_mask(
        self, attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"]
    ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        '''
        Applies a causal mask to attention scores, and returns masked scores.
        '''
        _, _, n_q, n_k = attn_scores.shape

        all_ones = t.ones(n_q, n_k, device=device, dtype=bool)
        mask = t.triu(all_ones, diagonal=1)

        # masked fill works with broadcasting
        return t.masked_fill(attn_scores, mask, self.IGNORE)


# %%
