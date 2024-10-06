# %%
import os
import sys
from pathlib import Path

from scipy import linalg

import einops
import plotly.express as px
import torch as t
from IPython.display import display
from ipywidgets import interact
from jaxtyping import Bool, Float, jaxtyped
from torch import Tensor
from typeguard import typechecked as typechecker
import numpy as np
from utils import *
import tests
import einops as ein


def make_rays_1d(num_pixels: int, y_limit: float) -> t.Tensor:
    '''
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    '''
    # SOLUTION
    rays = t.zeros((num_pixels, 2, 3), dtype=t.float32)
    t.linspace(-y_limit, y_limit, num_pixels, out=rays[:, 1, 1])
    rays[:, 1, 0] = 1
    return rays

rays1d = make_rays_1d(9, 10.0)
fig = render_lines_with_plotly(rays1d)

# %%
fig = setup_widget_fig_ray()
display(fig)

@interact
def response(seed=(0, 10, 1), v=(-2.0, 2.0, 0.01)):
    t.manual_seed(seed)
    L_1, L_2 = t.rand(2, 2)
    P = lambda v: L_1 + v * (L_2 - L_1)
    x, y = zip(P(-2), P(2))
    with fig.batch_update(): 
        fig.data[0].update({"x": x, "y": y}) 
        fig.data[1].update({"x": [L_1[0], L_2[0]], "y": [L_1[1], L_2[1]]}) 
        fig.data[2].update({"x": [P(v)[0]], "y": [P(v)[1]]})
# %%

segments = t.tensor([
    [[1.0, -12.0, 0.0], [1, -6.0, 0.0]], 
    [[0.5, 0.1, 0.0], [0.5, 1.15, 0.0]], 
    [[2, 12.0, 0.0], [2, 21.0, 0.0]]
])

fig = render_lines_with_plotly(rays1d, segments)
display(fig)

# %%
# My own solver:

def intersect_ray_1d(ray: t.Tensor, segment: t.Tensor) -> bool:
    '''
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    '''
    ray = ray[..., :2]
    segment = segment[..., :2]
    O, D = ray
    L1, L2 = segment

    A = t.stack([D, L1 - L2]).T
    B = L1 - O

    try:
        intersection = t.linalg.solve(A, B)
    except:
        return False

    u, v = intersection[0].item(), intersection[1].item()
    return  u >= 0 and 0 <= v <= 1

# YEEE
tests.test_intersect_ray_1d(intersect_ray_1d)
tests.test_intersect_ray_1d_special_case(intersect_ray_1d)
# %%
ray = rays1d[3]
ray = ray[..., :2]
O, D = ray
D
# %%

from jaxtyping import Float, Int, Bool, Shaped, jaxtyped
from typeguard import typechecked as typechecker
from torch import Tensor

@jaxtyped(typechecker=typechecker)
def my_concat(x: Float[Tensor, "a1 b"], y: Float[Tensor, "a2 b"]) -> Float[Tensor, "a1+a2 b"]:
    return t.concat([x, y], dim=0)

x = t.ones(3, 2)
print(x)
y = t.randn(4, 2)
z = my_concat(x, y)
# %%
@jaxtyped(typechecker=typechecker)
def intersect_ray_1d(ray: Float[Tensor, "2 3"], segment: Float[Tensor, "2 3"]) -> bool:
    '''
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    '''
    ray = ray[..., :2]
    segment = segment[..., :2]
    O, D = ray
    L1, L2 = segment

    A = t.stack([D, L1 - L2]).T
    B = L1 - O

    try:
        intersection = t.linalg.solve(A, B)
    except:
        return False

    u, v = intersection[0].item(), intersection[1].item()
    return  u >= 0 and 0 <= v <= 1

tests.test_intersect_ray_1d(intersect_ray_1d)
tests.test_intersect_ray_1d_special_case(intersect_ray_1d)

# %%
# Batched operations
x = t.randn(2, 3)
ein.repeat(x, "a b -> a b c", c=12)

# %%
def intersect_rays_1d(rays: Float[Tensor, "nrays 2 3"], 
                      segments: Float[Tensor, "nsegments 2 3"]) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return True if it intersects any segment.
    '''
    n_rays, n_segments = rays.shape[0], segments.shape[0]
    rays = rays[..., :2]
    segments = segments[..., :2]

    repeated_rays = ein.repeat(rays, "b p d -> (n b) p d", n=n_segments)
    repeated_segments = ein.repeat(segments, "b p d -> (b n) p d", n=n_rays)

    O = repeated_rays[:, :1]
    D = repeated_rays[:, -1:]
    L1 = repeated_segments[:, :1]
    L2 = repeated_segments[:, -1:]

    A = t.concat((D, L1 - L2), dim=1)
    A = ein.rearrange(A, "b p d -> b d p")
    B = (L1 - O)[:, 0, :]
    all_solutions = t.linalg.solve(A, B)
    all_valid = (all_solutions[:, 0] > 0) & (all_solutions[:, 1] > 0) & (all_solutions[:, 1] <= 1)
    return ein.reduce(all_valid, "(n_seg n_rays) -> n_rays", "any", n_seg=n_segments)


tests.test_intersect_rays_1d(intersect_rays_1d)
# tests.test_intersect_rays_1d_special_case(intersect_rays_1d)

# %%
ray = rays1d[0]
segment = segments[0]
ray = ray[..., :2]
segment = segment[..., :2]
O, D = ray
L1, L2 = segment

A = t.stack([D, L1 - L2]).T
B = L1 - O
print(A.shape, B.shape)
t.linalg.solve(A, B)
# %%
rays = rays1d
segments = segments
n_rays, n_segments = rays.shape[0], segments.shape[0]
rays = rays[..., :2]
segments = segments[..., :2]

repeated_rays = ein.repeat(rays, "b p d -> (n b) p d", n=n_segments)
repeated_segments = ein.repeat(segments, "b p d -> (b n) p d", n=n_rays)

O = repeated_rays[:, :1]
D = repeated_rays[:, -1:]
L1 = repeated_segments[:, :1]
L2 = repeated_segments[:, -1:]

A = t.concat((D, L1 - L2), dim=1)
A = ein.rearrange(A, "b p d -> b d p")
B = (L1 - O)[:, 0, :]
all_solutions = t.linalg.solve(A, B)
all_valid = (all_solutions[:, 0] >= 0) & (all_solutions[:, 1] >= 0) & (all_solutions[:, 1] >= 1)
ein.reduce(all_valid, "(n_seg n_rays) -> n_seg", "any", n_rays=n_rays)

# %%
repeated_rays.shape, repeated_segments.shape
# %%

# %%
