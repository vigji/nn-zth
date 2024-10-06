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
    """
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
    """
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

segments = t.tensor(
    [
        [[1.0, -12.0, 0.0], [1, -6.0, 0.0]],
        [[0.5, 0.1, 0.0], [0.5, 1.15, 0.0]],
        [[2, 12.0, 0.0], [2, 21.0, 0.0]],
    ]
)

fig = render_lines_with_plotly(rays1d, segments)
display(fig)

# %%
# My own solver:


def intersect_ray_1d(ray: t.Tensor, segment: t.Tensor) -> bool:
    """
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    """
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
    return u >= 0 and 0 <= v <= 1


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
def my_concat(
    x: Float[Tensor, "a1 b"], y: Float[Tensor, "a2 b"]
) -> Float[Tensor, "a1+a2 b"]:
    return t.concat([x, y], dim=0)


x = t.ones(3, 2)
print(x)
y = t.randn(4, 2)
z = my_concat(x, y)


# %%
@jaxtyped(typechecker=typechecker)
def intersect_ray_1d(ray: Float[Tensor, "2 3"], segment: Float[Tensor, "2 3"]) -> bool:
    """
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    """
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
    return u >= 0 and 0 <= v <= 1


tests.test_intersect_ray_1d(intersect_ray_1d)
tests.test_intersect_ray_1d_special_case(intersect_ray_1d)

# %%
# Batched operations
x = t.randn(2, 3)
ein.repeat(x, "a b -> a b c", c=12)


# %%
def intersect_rays_1d(
    rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]
) -> Bool[Tensor, "nrays"]:
    """
    For each ray, return True if it intersects any segment.
    """
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

    valid_mats = t.linalg.det(A).abs() > 10e-12
    all_solutions = t.linalg.solve(A[valid_mats], B[valid_mats])
    all_intersecting = (
        (all_solutions[:, 0] > 0)
        & (all_solutions[:, 1] > 0)
        & (all_solutions[:, 1] <= 1)
    )

    final_solution = t.full((n_rays * n_segments,), False)
    final_solution[valid_mats] = all_intersecting
    rays_intersecting = ein.reduce(
        final_solution, "(n_seg n_rays) -> n_rays", "any", n_seg=n_segments
    )
    return rays_intersecting


tests.test_intersect_rays_1d(intersect_rays_1d)
tests.test_intersect_rays_1d_special_case(intersect_rays_1d)

# %%


def make_rays_2d(
    num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float
) -> Float[t.Tensor, "nrays 2 3"]:
    """
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    """
    # SOLUTION
    rays = t.zeros((num_pixels_y * num_pixels_z, 2, 3), dtype=t.float32)
    y_values = t.linspace(-y_limit, y_limit, num_pixels_y)
    z_values = t.linspace(-z_limit, z_limit, num_pixels_z)
    rays[:, 1, 1] = ein.repeat(y_values, "n_y -> (n_z n_y)", n_z=num_pixels_z)
    rays[:, 1, 2] = ein.repeat(z_values, "n_z -> (n_z n_y)", n_y=num_pixels_y)

    rays[:, 1, 0] = 1
    return rays


# ein.repeat(t.arange(3), "n -> (3 n)")
rays_2d = make_rays_2d(10, 10, 0.5, 0.3)
render_lines_with_plotly(rays_2d)

# %%
# Part 3: Triangles
one_triangle = t.tensor([[0, 0, 0], [3, 0.5, 0], [2, 3, 0]])
A, B, C = one_triangle
x, y, z = one_triangle.T

fig = setup_widget_fig_triangle(x, y, z)


@interact(u=(-0.5, 1.5, 0.01), v=(-0.5, 1.5, 0.01))
def response(u=0.0, v=0.0):
    P = A + u * (B - A) + v * (C - A)
    fig.data[2].update({"x": [P[0]], "y": [P[1]]})


display(fig)
# %%
Point = Float[Tensor, "points=3"]
from typeguard import typechecked


@jaxtyped(typechecker=typechecked)
def triangle_ray_intersects(A: Point, B: Point, C: Point, O: Point, D: Point) -> bool:
    """
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the ray and the triangle intersect.
    """

    Amat = t.stack([-D, B - A, C - A], dim=1)
    Bmat = O - A

    try:
        intersection = t.linalg.solve(Amat, Bmat)
    except:
        return False

    result = (intersection[1:].sum() <= 1) and (intersection >= 0).all()
    return result.item()


tests.test_triangle_ray_intersects(triangle_ray_intersects)

# %%
A = t.tensor([0.5, -2.0, -2.0])
B = t.tensor([0.5, -2.0, 2.0])
C = t.tensor([0.5, 2.0, 2.0])
O = t.zeros(3)
D = t.zeros(3)
D[0] = 1
triangle_ray_intersects(A, B, C, O, D)
# %%


@jaxtyped(typechecker=typechecked)
def raytrace_triangle(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"],
) -> Bool[Tensor, "nrays"]:
    """
    For each ray, return True if the triangle intersects that ray.
    """
    n_rays = rays.shape[0]

    O = rays[:, :1, :]
    D = rays[:, -1:, :]

    A = einops.repeat(triangle[0, :], "d -> b 1 d", b=n_rays)
    B = einops.repeat(triangle[1, :], "d -> b 1 d", b=n_rays)
    C = einops.repeat(triangle[2, :], "d -> b 1 d", b=n_rays)

    Amat = t.concat([-D, B - A, C - A], dim=1)
    Amat = ein.rearrange(Amat, "b p d -> b d p")

    Bmat = O - A
    solution_pts = t.linalg.solve(Amat, Bmat[:, 0, :])

    return (t.sum(solution_pts[:, 1:], dim=1) <= 1) & t.all(
        solution_pts[:, 1:] > 0, dim=1
    )


A = t.tensor([1, 0.0, -0.5])
B = t.tensor([1, -0.5, 0.0])
C = t.tensor([1, 0.5, 0.5])
num_pixels_y = num_pixels_z = 20
y_limit = z_limit = 0.5

# Plot triangle & rays
test_triangle = t.stack([A, B, C], dim=0)
rays2d = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
triangle_lines = t.stack([A, B, C, A, B, C], dim=0).reshape(-1, 2, 3)
render_lines_with_plotly(rays2d, triangle_lines)

# Calculate and display intersections
intersects = raytrace_triangle(rays2d, test_triangle)

# Calculate and display intersections
from matplotlib import pyplot as plt

img = intersects.reshape(num_pixels_y, num_pixels_z).int()
plt.imshow(img, origin="lower")

# %%
# Multiple triangles:

with open("pikachu.pt", "rb") as f:
    triangles = t.load(f)
# %%


def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"],
) -> Float[Tensor, "nrays"]:
    """
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    """
    n_rays, n_triangles = rays.shape[0], triangles.shape[0]

    repeated_rays = ein.repeat(rays, "b p d -> (n b) 1 p d", n=n_triangles)
    repeated_triangles = ein.repeat(triangles, "b p d -> (b n) 1 p d", n=n_rays)

    O = repeated_rays[:, :, 0, :]
    D = repeated_rays[:, :, 0, :]

    A = repeated_triangles[
        :, :, 0, :
    ]  # einops.repeat(triangle[0, :], "d -> b 1 d", b=n_rays)
    B = repeated_triangles[
        :, :, 1, :
    ]  # einops.repeat(triangle[1, :], "d -> b 1 d", b=n_rays)
    C = repeated_triangles[
        :, :, 2, :
    ]  # einops.repeat(triangle[2, :], "d -> b 1 d", b=n_rays)

    Amat = t.concat([-D, B - A, C - A], dim=1)
    Amat = ein.rearrange(Amat, "b p d -> b d p")

    is_singular = t.linalg.det(Amat).abs() < 10e-12
    print(is_singular.sum())

    Amat[is_singular] = t.eye(3)

    Bmat = O - A
    solution_pts = t.linalg.solve(Amat, Bmat[:, 0, :])
    # print(solution_pts.shape, solution_pts[0])
    print(solution_pts[:, 0].unique())
    intersecting_sols = (
        (t.sum(solution_pts[:, 1:], dim=1) <= 1)
        & t.all(solution_pts[:, 1:] > 0, dim=1)
        & ~is_singular
    )

    print(solution_pts[:, 0].unique())

    final_solution = t.full((n_rays * n_triangles,), t.inf)
    final_solution[intersecting_sols] = solution_pts[intersecting_sols, 0]
    print(final_solution.unique())
    # final_solution[valid_mats][intersecting_sols] = solution_pts[intersecting_sols, 0]
    # print(final_solution.shape, solution_pts[:150, 0])

    # final_solution[valid_mats][all_intersecting] = t.inf

    rays_intersecting = ein.reduce(
        final_solution, "(n_tr n_rays) -> n_rays", "min", n_tr=n_triangles
    )
    # print(rays_intersecting.shape, rays_intersecting)
    return rays_intersecting


def raytrace_mesh_sol(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"],
) -> Float[Tensor, "nrays"]:
    """
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    """
    # SOLUTION
    NR = rays.size(0)
    NT = triangles.size(0)

    # Each triangle is [[Ax, Ay, Az], [Bx, By, Bz], [Cx, Cy, Cz]]
    triangles = einops.repeat(triangles, "NT pts dims -> pts NR NT dims", NR=NR)
    A, B, C = triangles
    assert A.shape == (NR, NT, 3)

    # Each ray is [[Ox, Oy, Oz], [Dx, Dy, Dz]]
    rays = einops.repeat(rays, "NR pts dims -> pts NR NT dims", NT=NT)
    O, D = rays
    assert O.shape == (NR, NT, 3)

    # Define matrix on left hand side of equation
    mat: Float[Tensor, "NR NT 3 3"] = t.stack([-D, B - A, C - A], dim=-1)
    # Get boolean of where matrix is singular, and replace it with the identity in these positions
    dets: Float[Tensor, "NR NT"] = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)
    print(is_singular.sum())

    # Define vector on the right hand side of equation
    vec: Float[Tensor, "NR NT 3"] = O - A

    # Solve eqns (note, s is the distance along ray)
    sol: Float[Tensor, "NR NT 3"] = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(-1)

    # Get boolean of intersects, and use it to set distance to infinity wherever there is no intersection
    intersects = (u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular
    s[~intersects] = float("inf")  # t.inf

    # Get the minimum distance (over all triangles) for each ray
    return s.min(dim=-1).values


num_pixels_y = 120
num_pixels_z = 120
y_limit = z_limit = 1

rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
rays[:, 0] = t.tensor([-2, 0.0, 0.0])
dists = raytrace_mesh(rays, triangles)
# render_lines_with_plotly(rays[::1000])
# dists.unique()

# %%
intersects = t.isfinite(dists).view(num_pixels_y, num_pixels_z)
dists_square = dists.view(num_pixels_y, num_pixels_z)
img = t.stack([intersects, dists_square], dim=0)
# %%
fig = px.imshow(
    img, facet_col=0, origin="lower", color_continuous_scale="magma", width=1000
)
fig.update_layout(coloraxis_showscale=False)
for i, text in enumerate(["Intersects", "Distance"]):
    fig.layout.annotations[i]["text"] = text
fig.show()
# %%
dists
# %%
