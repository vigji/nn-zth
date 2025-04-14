# %%
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import einops
import numpy as np
import pandas as pd
import plotly.express as px
import torch as t
from IPython.display import HTML, display
from jaxtyping import Float
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
from tqdm.auto import tqdm

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# Make sure exercises are in the path
chapter = "chapter1_transformer_interp"
section = "part31_superposition_and_saes"
#vroot_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
#exercises_dir = root_dir / chapter / "exercises"
# section_dir = exercises_dir / section
#if str(exercises_dir) not in sys.path:
#    sys.path.append(str(exercises_dir))

import pt21_tests as tests
import pt21_utils as utils
from plotly_utils import imshow, line

MAIN = __name__ == "__main__"
# %%
from toy_model import ToyModel

class NeuronModel(ToyModel):
    def forward(self, features: Float[Tensor, "... inst feats"]) -> Float[Tensor, "... inst feats"]:
        h_input = einops.einsum(
            self.W, features, "... inst d_hid d_feat, ... inst d_feat -> ... inst d_hid")
        nonlin_h = t.relu(h_input)
        h_out = einops.einsum(
            self.W, nonlin_h, "... inst d_hid d_feat, ... inst d_hid -> ... inst d_feat")
        return t.relu(h_out + self.b_final)


tests.test_neuron_model(NeuronModel)

# %%
