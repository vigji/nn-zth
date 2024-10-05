# %%
# Einops and einsums
import einops as ein
import torch as t
from pathlib import Path
import numpy as np
from utils import display_array_as_img

asset_path = Path(__file__).parent / "assets"
numbers_file = asset_path / "numbers.npy"
numbers = np.load(numbers_file)

# %%
numbers.shape  # b, c, h, w
# %%
display_array_as_img(numbers[0])
# %%
arr1 = ein.rearrange(numbers, "b c h w -> c h (b w)")
display_array_as_img(arr1)
# %%
arr2 = ein.repeat(numbers[0], "c h w -> c (2 h) w")
display_array_as_img(arr2)

# %%
arr3 = ein.repeat(numbers[:2], "b c h w -> c (b h) (2 w)")
display_array_as_img(arr3)

# %%
arr4 = ein.repeat(numbers[0], "c h w -> c (h 2) w")
display_array_as_img(arr4)
# %%
arr5 = ein.repeat(numbers[0], "c h w -> h (c w)")
display_array_as_img(arr5)
# %%
arr6 = ein.rearrange(numbers, "(b1 b2) c h w -> c (b1 h) (b2 w)", b1=2, b2=3)
display_array_as_img(arr6)
# %%
arr7 = ein.reduce(numbers, "b c h w -> h (b w)", "max")
display_array_as_img(arr7)
# %%
arr8 = ein.reduce(numbers, "b c h w -> h w", "min")
display_array_as_img(arr8)
# %%
arr9 = ein.rearrange(numbers[1], "c h w -> c w h")
display_array_as_img(arr9)
# %%
arr10 = ein.reduce(numbers, "(b1 b2) c (h 2) (w 2) -> c (b1 h) (b2 w)", "max", b1=2, b2=3)
display_array_as_img(arr10)
# %%
