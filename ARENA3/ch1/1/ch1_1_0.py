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
from einx import einx
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

import tests as tests
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
gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")
# %%
gpt2_small.cfg
# %%
model_description_text = """## Loading Models

HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of 
them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. Each model is loaded into 
the consistent HookedTransformer architecture, designed to be clean, consistent and 
interpretability-friendly.

For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. 
To try the model the model out, let's find the loss on this paragraph!"""

loss = gpt2_small(model_description_text, return_type="loss")
print("Model loss:", loss)

# %%
print(gpt2_small.to_str_tokens("gpt2"))
print(gpt2_small.to_str_tokens(["gpt2", "gpt2"]))
print(gpt2_small.to_tokens("gpt2"))
print(gpt2_small.to_string([50256, 70, 457, 17]))
# %%

logits: Tensor = gpt2_small(model_description_text, return_type="logits")
prediction = logits.argmax(dim=-1).squeeze()[:-1]

# YOUR CODE HERE - get the model's prediction on the text
# %%
tokenized = gpt2_small.to_tokens(model_description_text).squeeze(0)
# percentage right:
right = t.sum(prediction == tokenized[1:]) / len(prediction)
print("Perc. right: ", right)
# %%
for i in range(40):
    print("\n\n---")
    print(
        gpt2_small.to_string(tokenized[:i]), "...", gpt2_small.to_string(prediction[i])
    )
# %%
tokenized.shape
# %%
tokenized
# %%
gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
gpt2_tokens = gpt2_small.to_tokens(gpt2_text)
gpt2_logits, gpt2_cache = gpt2_small.run_with_cache(gpt2_tokens, remove_batch_dim=True)

print(type(gpt2_logits), type(gpt2_cache))

# %%
attn_patterns_from_shorthand = gpt2_cache["pattern", 0]
attn_patterns_from_full_name = gpt2_cache["blocks.0.attn.hook_pattern"]

t.testing.assert_close(attn_patterns_from_shorthand, attn_patterns_from_full_name)
# %%
layer0_pattern_from_cache = gpt2_cache["pattern", 0]

# YOUR CODE HERE - define `layer0_pattern_from_q_and_k` manually, by manually performing the steps of the attention calculation (dot product, masking, scaling, softmax)
q = gpt2_cache["q", 0]
k = gpt2_cache["k", 0]

q.shape, k.shape
attn_logits = einops.einsum(q, k, "seq_q n_h d, seq_k n_h d -> n_h seq_q seq_k")

upper_t = t.triu(t.ones((q.shape[0], k.shape[0]), dtype=bool), diagonal=1).to(device)
attn_logits.masked_fill_(upper_t, -1e9)
attn_logits_norm = attn_logits / (q.shape[2] ** 0.5)

layer0_pattern_from_q_and_k = attn_logits_norm.softmax(-1)
t.testing.assert_close(layer0_pattern_from_cache, layer0_pattern_from_q_and_k)

# %%

print(type(gpt2_cache))
attention_pattern = gpt2_cache["pattern", 0]
print(attention_pattern.shape)
gpt2_str_tokens = gpt2_small.to_str_tokens(gpt2_text)

print("Layer 0 Head Attention Patterns:")
display(
    cv.attention.attention_patterns(
        tokens=gpt2_str_tokens,
        attention=attention_pattern,
        # attention_head_names=[f"L0H{i}" for i in range(12)],
    )
)

# %%
neuron_activations_for_all_layers = t.stack(
    [gpt2_cache["post", layer] for layer in range(gpt2_small.cfg.n_layers)], dim=1
)
# shape = (seq_pos, layers, neurons)

cv.activations.text_neuron_activations(
    tokens=gpt2_str_tokens, activations=neuron_activations_for_all_layers
)
# %%
neuron_activations_for_all_layers_rearranged = utils.to_numpy(
    einops.rearrange(
        neuron_activations_for_all_layers, "seq layers neurons -> 1 layers seq neurons"
    )
)

cv.topk_tokens.topk_tokens(
    # Some weird indexing required here ¯\_(ツ)_/¯
    tokens=[gpt2_str_tokens],
    activations=neuron_activations_for_all_layers_rearranged,
    max_k=7,
    first_dimension_name="Layer",
    third_dimension_name="Neuron",
    first_dimension_labels=list(range(12)),
)
# %%
