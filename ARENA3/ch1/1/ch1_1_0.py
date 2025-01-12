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

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# Make sure exercises are in the path
chapter = "chapter1_transformer_interp"
section = "part2_intro_to_mech_interp"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import tests as tests
from plotly_utils import hist, imshow, plot_comp_scores, plot_logit_attribution, plot_loss_difference

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

logits: Tensor = gpt2_small(model_description_text, return_type="logits")
prediction = logits.argmax(dim=-1).squeeze()[:-1]

# YOUR CODE HERE - get the model's prediction on the text
# %%
tokens = gpt2_small.to_tokens(model_description_text).squeeze()[1:]

(prediction == tokens).sum() / (len(prediction))
# %%
for i in range(len(tokens)):
    print("\n\n====")
    print(gpt2_small.to_string(tokens[:i]) + "..." + gpt2_small.to_string(prediction[i]))
# %%
gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
gpt2_tokens = gpt2_small.to_tokens(gpt2_text)
gpt2_logits, gpt2_cache = gpt2_small.run_with_cache(gpt2_tokens, remove_batch_dim=True)

print(type(gpt2_logits), type(gpt2_cache))
# %%
layer0_pattern_from_cache = gpt2_cache["pattern", 0]

# YOUR CODE HERE - define `layer0_pattern_from_q_and_k` manually, by manually performing the steps of the attention calculation (dot product, masking, scaling, softmax)
t.testing.assert_close(layer0_pattern_from_cache, layer0_pattern_from_q_and_k)
print("Tests passed!")