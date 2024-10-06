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
arr10 = ein.reduce(
    numbers, "(b1 b2) c (h 2) (w 2) -> c (b1 h) (b2 w)", "max", b1=2, b2=3
)
display_array_as_img(arr10)
# %%


def assert_all_equal(actual: t.Tensor, expected: t.Tensor) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    assert (actual == expected).all(), f"Value mismatch, got: {actual}"
    print("Passed!")


def assert_all_close(
    actual: t.Tensor, expected: t.Tensor, rtol=1e-05, atol=0.0001
) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    assert t.allclose(actual, expected, rtol=rtol, atol=atol)
    print("Passed!")


# %%


def rearrange_1() -> t.Tensor:
    """Return the following tensor using only torch.arange and einops.rearrange:

    [[3, 4],
     [5, 6],
     [7, 8]]
    """
    arr = t.arange(3, 9)
    return ein.rearrange(arr, "(a b) -> a b", b=2)


expected = t.tensor([[3, 4], [5, 6], [7, 8]])
expected

# %%
assert_all_equal(rearrange_1(), expected)

# %%


def temperatures_average(temps: t.Tensor) -> t.Tensor:
    """Return the average temperature for each week.

    temps: a 1D temperature containing temperatures for each day.
    Length will be a multiple of 7 and the first 7 days are for the first week, second 7 days for the second week, etc.

    You can do this with a single call to reduce.
    """
    assert len(temps) % 7 == 0

    return ein.reduce(temps, "(w 7) -> w", "mean")


temps = t.Tensor(
    [71, 72, 70, 75, 71, 72, 70, 68, 65, 60, 68, 60, 55, 59, 75, 80, 85, 80, 78, 72, 83]
)
expected = t.tensor([71.5714, 62.1429, 79.0])
assert_all_close(temperatures_average(temps), expected)
# %%


def temperatures_differences(temps: t.Tensor) -> t.Tensor:
    """For each day, subtract the average for the week the day belongs to.

    temps: as above
    """
    assert len(temps) % 7 == 0
    diff = ein.rearrange(temps, "(w d) -> w d", d=7) - ein.reduce(
        temps, "(w 7) -> w 1", "mean"
    )
    return ein.rearrange(diff, "w d -> (w d)")


expected = t.tensor(
    [
        -0.5714,
        0.4286,
        -1.5714,
        3.4286,
        -0.5714,
        0.4286,
        -1.5714,
        5.8571,
        2.8571,
        -2.1429,
        5.8571,
        -2.1429,
        -7.1429,
        -3.1429,
        -4.0,
        1.0,
        6.0,
        1.0,
        -1.0,
        -7.0,
        4.0,
    ]
)
actual = temperatures_differences(temps)
assert_all_close(actual, expected)


# %%
def temperatures_normalized(temps: t.Tensor) -> t.Tensor:
    """For each day, subtract the weekly average and divide by the weekly standard deviation.

    temps: as above

    Pass torch.std to reduce.
    """

    means = ein.repeat(temperatures_average(temps), "w -> (w 7)")
    stds = ein.repeat(ein.reduce(temps, "(w 7) -> w", t.std), "w -> (w 7)")

    return (temps - means) / stds


expected = t.tensor(
    [
        -0.3326,
        0.2494,
        -0.9146,
        1.9954,
        -0.3326,
        0.2494,
        -0.9146,
        1.1839,
        0.5775,
        -0.4331,
        1.1839,
        -0.4331,
        -1.4438,
        -0.6353,
        -0.8944,
        0.2236,
        1.3416,
        0.2236,
        -0.2236,
        -1.5652,
        0.8944,
    ]
)
actual = temperatures_normalized(temps)
assert_all_close(actual, expected)


# %%
def identity_matrix(n: int) -> t.Tensor:
    """Return the identity matrix of size nxn.

    Don't use torch.eye or similar.

    Hint: you can do it with arange, rearrange, and ==.
    Bonus: find a different way to do it.
    """
    assert n >= 0
    return ein.rearrange(
        (t.arange(0, n * n) % (n + 1) == 0).int(), "(i j) -> i j", i=n, j=n
    )


assert_all_equal(identity_matrix(3), t.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
assert_all_equal(identity_matrix(0), t.zeros((0, 0)))


# %%
def classifier_accuracy(scores: t.Tensor, true_classes: t.Tensor) -> t.Tensor:
    """Return the fraction of inputs for which the maximum score corresponds to the true class for that input.

    scores: shape (batch, n_classes). A higher score[b, i] means that the classifier thinks class i is more likely.
    true_classes: shape (batch, ). true_classes[b] is an integer from [0...n_classes).

    Use torch.argmax.
    """
    assert true_classes.max() < scores.shape[1]
    prediction = t.argmax(scores, dim=1)
    return t.mean((prediction == true_classes).float())


scores = t.tensor([[0.75, 0.5, 0.25], [0.1, 0.5, 0.4], [0.1, 0.7, 0.2]])
true_classes = t.tensor([0, 1, 0])
expected = 2.0 / 3.0
assert classifier_accuracy(scores, true_classes) == expected


# %%
def total_price_indexing(prices: t.Tensor, items: t.Tensor) -> float:
    """Given prices for each kind of item and a tensor of items purchased, return the total price.

    prices: shape (k, ). prices[i] is the price of the ith item.
    items: shape (n, ). A 1D tensor where each value is an item index from [0..k).

    Use integer array indexing. The below document describes this for NumPy but it's the same in PyTorch:

    https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing
    """
    assert items.max() < prices.shape[0]
    return prices[items].sum()


prices = t.tensor([0.5, 1, 1.5, 2, 2.5])
items = t.tensor([0, 0, 1, 1, 4, 3, 2])
assert total_price_indexing(prices, items) == 9.0


# %%
# %%
def total_price_gather(prices: t.Tensor, items: t.Tensor) -> float:
    """Compute the same as total_price_indexing, but use torch.gather."""
    assert items.max() < prices.shape[0]
    return t.gather(prices, 0, items).sum()


prices = t.tensor([0.5, 1, 1.5, 2, 2.5])
items = t.tensor([0, 0, 1, 1, 4, 3, 2])
assert total_price_gather(prices, items) == 9.0


# %%
def integer_array_indexing(matrix: t.Tensor, coords: t.Tensor) -> t.Tensor:
    """Return the values at each coordinate using integer array indexing.

    For details on integer array indexing, see:
    https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing

    matrix: shape (d_0, d_1, ..., d_n)
    coords: shape (batch, n)

    Return: (batch, )
    """
    return matrix[tuple(coords.T)]


mat_2d = t.arange(15).view(3, 5)
coords_2d = t.tensor([[0, 1], [0, 4], [1, 4]])
actual = integer_array_indexing(mat_2d, coords_2d)
assert_all_equal(actual, t.tensor([1, 4, 9]))
mat_3d = t.arange(2 * 3 * 4).view((2, 3, 4))
coords_3d = t.tensor([[0, 0, 0], [0, 1, 1], [0, 2, 2], [1, 0, 3], [1, 2, 0]])
actual = integer_array_indexing(mat_3d, coords_3d)
assert_all_equal(actual, t.tensor([0, 5, 10, 15, 20]))
# %%
mat_2d[tuple(coords_2d.T)]
# %%

# ein.einsum

import tests


def einsum_trace(mat: np.ndarray):
    """
    Returns the same as `np.trace`.
    """
    return ein.einsum(mat, "i i -> ")


def einsum_mv(mat: np.ndarray, vec: np.ndarray):
    """
    Returns the same as `np.matmul`, when `mat` is a 2D array and `vec` is 1D.
    """
    return ein.einsum(mat, vec, "i j, j -> i")


def einsum_mm(mat1: np.ndarray, mat2: np.ndarray):
    """
    Returns the same as `np.matmul`, when `mat1` and `mat2` are both 2D arrays.
    """
    return ein.einsum(mat1, mat2, "i j, j k -> i k")


def einsum_inner(vec1: np.ndarray, vec2: np.ndarray):
    """
    Returns the same as `np.inner`.
    """
    return ein.einsum(vec1, vec2, "i, i -> ")


def einsum_outer(vec1: np.ndarray, vec2: np.ndarray):
    """
    Returns the same as `np.outer`.
    """
    return ein.einsum(vec1, vec2, "i, j -> i j")


tests.test_einsum_trace(einsum_trace)
tests.test_einsum_mv(einsum_mv)
tests.test_einsum_mm(einsum_mm)
tests.test_einsum_inner(einsum_inner)
tests.test_einsum_outer(einsum_outer)
# %%
mat = np.array([[1, 2, 3], [11, 22, 33], [111, 222, 333]])

# %%
np.matmul(mat, np.array([1, 2, 3]))
# %%

ein.einsum(mat, mat, "i j, j k -> i k")
# %%
np.matmul(mat, mat)
# %%
np.outer(np.array([1, 2, 3]), np.array([1, 2, 3]))
# %%
