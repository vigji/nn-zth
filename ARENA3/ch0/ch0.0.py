# %%
# Einops and einsums
import einops
import torch as t
from pathlib import Path
import numpy as np
from utils import display_array_as_img

asset_path = Path(__file__).parent / "assets"
numbers_file = asset_path / "numbers.npy"
numbers = np.load(numbers_file)

# %%
numbers.shape
# %%
