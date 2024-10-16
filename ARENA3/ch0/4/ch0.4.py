# Where we basically implement pytorch from scratch.

# %%
from ast import Param, arg
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, TypeAlias

import numpy as np
from regex import R
import torch as t
from torch.utils.data import DataLoader
from tqdm import tqdm

Arr = np.ndarray
grad_tracking_enabled = True

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part4_backprop"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import tests as tests
from utils import visualize, get_mnist
from plotly_utils import line

# %%
# 1. Introduction


def log_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    """Backwards function for f(x) = log(x)

    grad_out: Gradient of some loss wrt out
    out: the output of np.log(x).
    x: the input of np.log.

    Return: gradient of the given loss wrt x

    derivative = 1/x
    """

    return grad_out / x


tests.test_log_back(log_back)
# %%
# Unbroadcasting


def unbroadcast(broadcasted: Arr, original: Arr) -> Arr:
    """
    Sum 'broadcasted' until it has the shape of 'original'.

    broadcasted: An array that was formerly of the same shape of 'original' and was expanded by broadcasting rules.
    """
    # SOLUTION

    # Step 1: sum and remove prepended dims, so both arrays have same number of dims
    n_dims_to_sum = len(broadcasted.shape) - len(original.shape)
    broadcasted = broadcasted.sum(axis=tuple(range(n_dims_to_sum)))

    # Step 2: sum over dims which were originally 1 (but don't remove them)
    dims_to_sum = tuple(
        [
            i
            for i, (o, b) in enumerate(zip(original.shape, broadcasted.shape))
            if o == 1 and b > 1
        ]
    )
    broadcasted = broadcasted.sum(axis=dims_to_sum, keepdims=True)

    return broadcasted


tests.test_unbroadcast(unbroadcast)  # %%


# %%
def multiply_back0(grad_out: Arr, out: Arr, x: Arr, y: Arr | float) -> Arr:
    """Backwards function for x * y wrt argument 0 aka x."""
    if not isinstance(y, Arr):
        y = np.array(y)

    return unbroadcast(y * grad_out, x)


def multiply_back1(grad_out: Arr, out: Arr, x: Arr | float, y: Arr) -> Arr:
    """Backwards function for x * y wrt argument 1 aka y."""
    if not isinstance(x, Arr):
        x = np.array(x)

    return unbroadcast(x * grad_out, y)


tests.test_multiply_back(multiply_back0, multiply_back1)
tests.test_multiply_back_float(multiply_back0, multiply_back1)


# %%
def forward_and_back(a: Arr, b: Arr, c: Arr) -> tuple[Arr, Arr, Arr]:
    """
    Calculates the output of the computational graph above (g),
    then backpropogates the gradients and returns dg/da, dg/db, and dg/dc
    """
    d = a * b
    e = np.log(c)

    f = d * e
    g = np.log(f)

    final_grad = np.ones_like(g)
    dgdf = log_back(final_grad, g, f)
    dgdd = multiply_back0(dgdf, f, d, e)
    dgde = multiply_back1(dgdf, f, d, e)

    dgda = multiply_back0(dgdd, d, a, b)
    dgdb = multiply_back1(dgdd, d, a, b)

    dgdc = log_back(dgde, e, c)

    return dgda, dgdb, dgdc


tests.test_forward_and_back(forward_and_back)
# %%
# 2. Autograd


@dataclass(frozen=True)
class Recipe:
    """Extra information necessary to run backpropagation. You don't need to modify this."""

    func: Callable
    "The 'inner' NumPy function that does the actual forward computation."
    "Note, we call it 'inner' to distinguish it from the wrapper we'll create for it later on."

    args: tuple
    "The input arguments passed to func."
    "For instance, if func was np.sum then args would be a length-1 tuple containing the tensor to be summed."

    kwargs: dict[str, Any]
    "Keyword arguments passed to func."
    "For instance, if func was np.sum then kwargs might contain 'dim' and 'keepdims'."

    parents: dict[int, "Tensor"]
    "Map from positional argument index to the Tensor at that position, in order to be able to pass gradients back along the computational graph."


class BackwardFuncLookup:
    def __init__(self) -> None:
        self.linking_dict = dict()

    def add_back_func(
        self, forward_fn: Callable, arg_position: int, back_fn: Callable
    ) -> None:
        self.linking_dict[(forward_fn, arg_position)] = back_fn

    def get_back_func(self, forward_fn: Callable, arg_position: int) -> Callable:
        return self.linking_dict[(forward_fn, arg_position)]


BACK_FUNCS = BackwardFuncLookup()
BACK_FUNCS.add_back_func(np.log, 0, log_back)
BACK_FUNCS.add_back_func(np.multiply, 0, multiply_back0)
BACK_FUNCS.add_back_func(np.multiply, 1, multiply_back1)

assert BACK_FUNCS.get_back_func(np.log, 0) == log_back
assert BACK_FUNCS.get_back_func(np.multiply, 0) == multiply_back0
assert BACK_FUNCS.get_back_func(np.multiply, 1) == multiply_back1

print("Tests passed - BackwardFuncLookup class is working as expected!")
# %%

Arr = np.ndarray


class Tensor:
    """
    A drop-in replacement for torch.Tensor supporting a subset of features.
    """

    # array: Arr
    "The underlying array. Can be shared between multiple Tensors."
    # requires_grad: bool
    "If True, calling functions or methods on this tensor will track relevant data for backprop."
    # grad #: Tensor | None
    "Backpropagation will accumulate gradients into this field."
    # recipe: Recipe | None
    "Extra information necessary to run backpropagation."

    def __init__(self, array: Arr | list, requires_grad=False):
        self.array = array if isinstance(array, Arr) else np.array(array)
        if self.array.dtype == np.float64:
            self.array = self.array.astype(np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self.recipe = None
        "If not None, this tensor's array was created via recipe.func(*recipe.args, **recipe.kwargs)."

    def __neg__(self) -> "Tensor":
        return negative(self)

    def __add__(self, other) -> "Tensor":
        return add(self, other)

    def __radd__(self, other) -> "Tensor":
        return add(other, self)

    def __sub__(self, other) -> "Tensor":
        return subtract(self, other)

    def __rsub__(self, other):
        return subtract(other, self)

    def __mul__(self, other) -> "Tensor":
        return multiply(self, other)

    def __rmul__(self, other) -> "Tensor":
        return multiply(other, self)

    def __truediv__(self, other) -> "Tensor":
        return true_divide(self, other)

    def __rtruediv__(self, other) -> "Tensor":
        return true_divide(other, self)

    def __matmul__(self, other) -> "Tensor":
        return matmul(self, other)

    def __rmatmul__(self, other) -> "Tensor":
        return matmul(other, self)

    def __eq__(self, other) -> "Tensor":
        return eq(self, other)

    def __repr__(self) -> str:
        return f"Tensor({repr(self.array)}, requires_grad={self.requires_grad})"

    def __len__(self) -> int:
        if self.array.ndim == 0:
            raise TypeError
        return self.array.shape[0]

    def __hash__(self) -> int:
        return id(self)

    def __getitem__(self, index) -> "Tensor":
        return getitem(self, index)

    def add_(self, other: "Tensor", alpha: float = 1.0) -> "Tensor":
        add_(self, other, alpha=alpha)
        return self

    @property
    def T(self) -> "Tensor":
        return permute(self, axes=(-1, -2))

    def item(self):
        return self.array.item()

    def sum(self, dim=None, keepdim=False):
        return sum(self, dim=dim, keepdim=keepdim)

    def log(self):
        return log(self)

    def exp(self):
        return exp(self)

    def reshape(self, new_shape):
        return reshape(self, new_shape)

    def expand(self, new_shape):
        return expand(self, new_shape)

    def permute(self, dims):
        return permute(self, dims)

    def maximum(self, other):
        return maximum(self, other)

    def relu(self):
        return relu(self)

    def argmax(self, dim=None, keepdim=False):
        return argmax(self, dim=dim, keepdim=keepdim)

    def uniform_(self, low: float, high: float) -> "Tensor":
        self.array[:] = np.random.uniform(low, high, self.array.shape)
        return self

    def backward(self, end_grad: "Arr | Tensor | None" = None) -> None:
        if isinstance(end_grad, Arr):
            end_grad = Tensor(end_grad)
        return backprop(self, end_grad)

    def size(self, dim: int | None = None):
        if dim is None:
            return self.shape
        return self.shape[dim]

    @property
    def shape(self):
        return self.array.shape

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def is_leaf(self):
        """Same as https://pytorch.org/docs/stable/generated/torch.Tensor.is_leaf.html"""
        if self.requires_grad and self.recipe and self.recipe.parents:
            return False
        return True

    def __bool__(self):
        if np.array(self.shape).prod() != 1:
            raise RuntimeError(
                "bool value of Tensor with more than one value is ambiguous"
            )
        return bool(self.item())


def empty(*shape: int) -> Tensor:
    """Like torch.empty."""
    return Tensor(np.empty(shape))


def zeros(*shape: int) -> Tensor:
    """Like torch.zeros."""
    return Tensor(np.zeros(shape))


def arange(start: int, end: int, step=1) -> Tensor:
    """Like torch.arange(start, end)."""
    return Tensor(np.arange(start, end, step=step))


def tensor(array: Arr, requires_grad=False) -> Tensor:
    """Like torch.tensor."""
    return Tensor(array, requires_grad=requires_grad)


# Implementing custom functions:


def log_forward(x: Tensor) -> Tensor:
    """Performs np.log on a Tensor object."""

    log_val = np.log(x.array)

    recipe = Recipe(args=(x.array,), func=np.log, kwargs=dict(), parents={0: x})

    out = Tensor(log_val, requires_grad=grad_tracking_enabled and x.requires_grad)

    if out.requires_grad:
        out.recipe = recipe

    return out


# %%

log = log_forward
tests.test_log(Tensor, log_forward)
tests.test_log_no_grad(Tensor, log_forward)
a = Tensor([1], requires_grad=True)
grad_tracking_enabled = False
b = log_forward(a)
grad_tracking_enabled = True
assert not b.requires_grad, "should not require grad if grad tracking globally disabled"
assert b.recipe is None, "should not create recipe if grad tracking globally disabled"
# %%


def multiply_forward(a: Tensor | list, b: Tensor | list) -> Tensor:
    """Performs np.multiply on a Tensor object."""
    assert isinstance(a, Tensor) or isinstance(b, Tensor)
    print("here", a, b)

    if isinstance(a, Tensor) and not isinstance(b, Tensor):
        tensor_requires_grad = a.requires_grad
    elif isinstance(b, Tensor) and not isinstance(a, Tensor):
        tensor_requires_grad = b.requires_grad
    else:
        tensor_requires_grad = a.requires_grad or b.requires_grad
    requires_grad = grad_tracking_enabled and tensor_requires_grad

    arg0 = a.array if isinstance(a, Tensor) else a
    arg1 = b.array if isinstance(b, Tensor) else b
    out_val = arg0 * arg1
    parents = {idx: arr for idx, arr in enumerate([a, b]) if isinstance(arr, Tensor)}
    args = (arg0, arg1)

    out = Tensor(out_val, requires_grad=requires_grad)
    if requires_grad:
        recipe = Recipe(args=args, func=np.multiply, kwargs=dict(), parents=parents)
        out.recipe = recipe

    return out


multiply = multiply_forward
tests.test_multiply(Tensor, multiply_forward)
tests.test_multiply_no_grad(Tensor, multiply_forward)
tests.test_multiply_float(Tensor, multiply_forward)
a = Tensor([2], requires_grad=True)
b = Tensor([3], requires_grad=True)
grad_tracking_enabled = False
b = multiply_forward(a, b)
grad_tracking_enabled = True
assert not b.requires_grad, "should not require grad if grad tracking globally disabled"
assert b.recipe is None, "should not create recipe if grad tracking globally disabled"
# %%


def wrap_forward_fn_mine(numpy_func: Callable, is_differentiable=True) -> Callable:
    """
    numpy_func: Callable
        takes any number of positional arguments, some of which may be NumPy arrays, and
        any number of keyword arguments which we aren't allowing to be NumPy arrays at
        present. It returns a single NumPy array.

    is_differentiable:
        if True, numpy_func is differentiable with respect to some input argument, so we
        may need to track information in a Recipe. If False, we definitely don't need to
        track information.

    Return: Callable
        It has the same signature as numpy_func, except wherever there was a NumPy array,
        this has a Tensor instead.
    """

    def tensor_func(*args: Any, **kwargs: Any) -> Tensor:
        numpy_args = []
        parents = {}
        args_require_grad = False
        for i, arg in enumerate(args):
            if isinstance(arg, Tensor):
                new_arg = arg.array
                parents[i] = arg
                args_require_grad = args_require_grad or arg.requires_grad
            else:
                new_arg = arg
            numpy_args.append(new_arg)
        numpy_args = tuple(numpy_args)
        requires_grad = (
            is_differentiable and grad_tracking_enabled
        ) and args_require_grad

        out_val = numpy_func(*numpy_args, **kwargs)
        out_tensor = Tensor(out_val, requires_grad=requires_grad)

        if requires_grad:
            recipe = Recipe(
                args=numpy_args, func=numpy_func, kwargs=kwargs, parents=parents
            )
            out_tensor.recipe = recipe

        return out_tensor

    return tensor_func


def wrap_forward_fn(numpy_func: Callable, is_differentiable=True) -> Callable:
    """
    numpy_func: Callable
        takes any number of positional arguments, some of which may be NumPy arrays, and
        any number of keyword arguments which we aren't allowing to be NumPy arrays at
        present. It returns a single NumPy array.

    is_differentiable:
        if True, numpy_func is differentiable with respect to some input argument, so we
        may need to track information in a Recipe. If False, we definitely don't need to
        track information.

    Return: Callable
        It has the same signature as numpy_func, except wherever there was a NumPy array,
        this has a Tensor instead.
    """

    def tensor_func(*args: Any, **kwargs: Any) -> Tensor:
        # SOLUTION

        # Get all function arguments as non-tensors (i.e. either ints or arrays)
        arg_arrays = tuple([(a.array if isinstance(a, Tensor) else a) for a in args])

        # Calculate the output (which is a numpy array)
        out_arr = numpy_func(*arg_arrays, **kwargs)

        # Find whether the tensor requires grad (need to check if ANY of the inputs do)
        requires_grad = (
            grad_tracking_enabled
            and is_differentiable
            and any([isinstance(a, Tensor) and a.requires_grad for a in args])
        )

        # Create the output tensor from the underlying data and the requires_grad flag
        out = Tensor(out_arr, requires_grad)

        # If requires_grad, then create a recipe
        if requires_grad:
            parents = {idx: a for idx, a in enumerate(args) if isinstance(a, Tensor)}
            out.recipe = Recipe(numpy_func, arg_arrays, kwargs, parents)

        return out

    return tensor_func


def _sum(x: Arr, dim=None, keepdim=False) -> Arr:
    # need to be careful with sum, because kwargs have different names in torch and numpy
    return np.sum(x, axis=dim, keepdims=keepdim)


log = wrap_forward_fn(np.log)
multiply = wrap_forward_fn(np.multiply)
eq = wrap_forward_fn(np.equal, is_differentiable=False)
sum = wrap_forward_fn(_sum)

tests.test_log(Tensor, log)
tests.test_log_no_grad(Tensor, log)
tests.test_multiply(Tensor, multiply)
tests.test_multiply_no_grad(Tensor, multiply)
tests.test_multiply_float(Tensor, multiply)
tests.test_sum(Tensor)
# %%


class Node:
    def __init__(self, *children):
        self.children = list(children)


def get_children(node: Node) -> [Node]:
    return node.children


def topological_sort(node: Node, get_children: Callable) -> [Node]:
    """
    Return a list of node's descendants in reverse topological order from future to past (i.e. `node` should be last).

    Should raise an error if the graph with `node` as root is not in fact acyclic.
    """
    # SOLUTION

    result: [
        Node
    ] = []  # stores the list of nodes to be returned (in reverse topological order)
    perm: set[
        Node
    ] = set()  # same as `result`, but as a set (faster to check for membership)
    temp: set[
        Node
    ] = set()  # keeps track of previously visited nodes (to detect cyclicity)

    def visit(cur: Node):
        """
        Recursive function which visits all the children of the current node, and appends them all
        to `result` in the order they were found.
        """
        if cur in perm:
            return
        if cur in temp:
            raise ValueError("Not a DAG!")
        temp.add(cur)

        for next in get_children(cur):
            visit(next)

        result.append(cur)
        perm.add(cur)
        temp.remove(cur)

    visit(node)
    return result


# %%
tests.test_topological_sort_linked_list(topological_sort)
tests.test_topological_sort_branching(topological_sort)
tests.test_topological_sort_rejoining(topological_sort)
tests.test_topological_sort_cyclic(topological_sort)


# %%
def sorted_computational_graph(tensor: Tensor) -> [Tensor]:
    """
    For a given tensor, return a list of Tensors that make up the nodes of the given Tensor's computational graph,
    in reverse topological order (i.e. `tensor` should be first).
    """

    def _get_children(tensor):
        if not tensor.recipe:
            return []
        return list(tensor.recipe.parents.values())

    return topological_sort(tensor, _get_children)[::-1]


a = Tensor([1], requires_grad=True)
b = Tensor([2], requires_grad=True)
c = Tensor([3], requires_grad=True)
d = a * b
e = c.log()
f = d * e
g = f.log()
name_lookup = {a: "a", b: "b", c: "c", d: "d", e: "e", f: "f", g: "g"}

print([name_lookup[t] for t in sorted_computational_graph(g)])
# %%
# Finally backprop!


def backprop_mine(end_node: Tensor, end_grad: Tensor | None = None) -> None:
    """Accumulates gradients in the grad field of each leaf node.

    tensor.backward() is equivalent to backprop(tensor).

    end_node:
        The rightmost node in the computation graph.
        If it contains more than one element, end_grad must be provided.
    end_grad:
        A tensor of the same shape as end_node.
        Set to 1 if not specified and end_node has only one element.
    """
    if end_node.ndim <= 1:
        assert end_grad is not None
    if end_grad is None:
        end_grad = 1
    if isinstance(end_grad, Tensor):
        end_grad = end_grad.array

    recipe = end_node.recipe

    if recipe:
        for parent_i, parent in recipe.parents.items():
            back_fun = BACK_FUNCS.get_back_func(recipe.func, parent_i)
            new_grad = Tensor(back_fun(end_grad, end_node, *recipe.args))

            if (
                parent.recipe is None and parent.requires_grad
            ):  # leaf nodes requiring gradient:
                if parent.grad is not None:
                    grad_val = parent.grad.array + (
                        new_grad if not isinstance(new_grad, Tensor) else new_grad.array
                    )
                    parent.grad = Tensor(grad_val)
                else:
                    parent.grad = new_grad

            backprop_mine(parent, end_grad=new_grad)


def backprop(end_node: Tensor, end_grad: Tensor | None = None) -> None:
    """Accumulates gradients in the grad field of each leaf node.

    tensor.backward() is equivalent to backprop(tensor).

    end_node:
        The rightmost node in the computation graph.
        If it contains more than one element, end_grad must be provided.
    end_grad:
        A tensor of the same shape as end_node.
        Set to 1 if not specified and end_node has only one element.
    """
    # SOLUTION

    # Get value of end_grad_arr
    end_grad_arr = np.ones_like(end_node.array) if end_grad is None else end_grad.array

    # Create dict to store gradients
    grads: dict[Tensor, Arr] = {end_node: end_grad_arr}

    # Iterate through the computational graph, using your sorting function
    for node in sorted_computational_graph(end_node):
        # Get the outgradient from the grads dict
        outgrad = grads.pop(node)
        # We only store the gradients if this node is a leaf & requires_grad is true
        if node.is_leaf and node.requires_grad:
            # Add the gradient to this node's grad (need to deal with special case grad=None)
            if node.grad is None:
                node.grad = Tensor(outgrad)
            else:
                node.grad.array += outgrad

        # If node has no parents, then the backtracking through the computational
        # graph ends here
        if node.recipe is None or node.recipe.parents is None:
            continue

        # If node has a recipe, then we iterate through parents (which is a dict of {arg_posn: tensor})
        for argnum, parent in node.recipe.parents.items():
            # Get the backward function corresponding to the function that created this node
            back_fn = BACK_FUNCS.get_back_func(node.recipe.func, argnum)

            # Use this backward function to calculate the gradient
            in_grad = back_fn(
                outgrad, node.array, *node.recipe.args, **node.recipe.kwargs
            )

            # Add the gradient to this node in the dictionary `grads`
            # Note that we only set node.grad (from the grads dict) in the code block above
            if parent not in grads:
                grads[parent] = in_grad
            else:
                grads[parent] += in_grad


a = 2
b = Tensor([1, 2, 3], requires_grad=True)
c = 3
d = a * b
e = b * c
f = d * e
f.backward(end_grad=np.array([1.0, 1.0, 1.0]))
assert f.grad is None
assert b.grad is not None
# print(b.grad.array)
assert np.allclose(
    b.grad.array, np.array([12.0, 24.0, 36.0])
), "Multiple nodes may have the same parent."

# %%
tests.test_backprop(Tensor)
tests.test_backprop_branching(Tensor)
tests.test_backprop_requires_grad_false(Tensor)
tests.test_backprop_float_arg(Tensor)
tests.test_backprop_shared_parent(Tensor)
# %%
# Non differentiable:


def _argmax(x: Arr, dim=None, keepdim=False):
    """Like torch.argmax."""
    return np.expand_dims(np.argmax(x, axis=dim), axis=([] if dim is None else dim))


argmax = wrap_forward_fn(_argmax, is_differentiable=False)

a = Tensor([1.0, 0.0, 3.0, 4.0], requires_grad=True)
b = a.argmax()
assert not b.requires_grad
assert b.recipe is None
assert b.item() == 3
# %%


def negative_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    """Backward function for f(x) = -x elementwise."""
    return -grad_out


negative = wrap_forward_fn(np.negative)
BACK_FUNCS.add_back_func(np.negative, 0, negative_back)

tests.test_negative_back(Tensor)
# %%


def exp_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    return grad_out * np.exp(grad_out)


exp = wrap_forward_fn(np.exp)
BACK_FUNCS.add_back_func(np.exp, 0, exp_back)

tests.test_exp_back(Tensor)


# %%
def reshape_back(grad_out: Arr, out: Arr, x: Arr, new_shape: tuple) -> Arr:
    return grad_out.reshape(x.shape)


reshape = wrap_forward_fn(np.reshape)
BACK_FUNCS.add_back_func(np.reshape, 0, reshape_back)

tests.test_reshape_back(Tensor)


# %%
def invert_transposition(axes: tuple) -> tuple:
    """
    axes: tuple indicating a transition

    Returns: inverse of this transposition, i.e. the array `axes_inv` s.t. we have:
        np.transpose(np.transpose(x, axes), axes_inv) == x

    Some examples:
        (1, 0)    --> (1, 0)     # this is reversing a simple 2-element transposition
        (0, 2, 1) --> (0, 2, 1)  # also a 2-element transposition
        (1, 2, 0) --> (2, 0, 1)  # this is reversing the order of a 3-cycle
    """
    return [axes[a] for a in axes]


def permute_back(grad_out: Arr, out: Arr, x: Arr, axes: tuple) -> Arr:
    return np.transpose(grad_out, invert_transposition(axes))


BACK_FUNCS.add_back_func(np.transpose, 0, permute_back)
permute = wrap_forward_fn(np.transpose)

tests.test_permute_back(Tensor)


# %%
def expand_back(grad_out: Arr, out: Arr, x: Arr, new_shape: tuple) -> Arr:
    return unbroadcast(grad_out, x)


def _expand(x: Arr, new_shape) -> Arr:
    """
    Like torch.expand, calling np.broadcast_to internally.

    Note torch.expand supports -1 for a dimension size meaning "don't change the size".
    np.broadcast_to does not natively support this.
    """
    n_dims_x = len(x.shape)
    n_dims_target = len(new_shape)
    n_new_dims = n_dims_target - n_dims_x
    new_shape = tuple(
        [x.shape[i - n_new_dims] if s == -1 else s for i, s in enumerate(new_shape)]
    )

    return np.broadcast_to(x, new_shape)


expand = wrap_forward_fn(_expand)
BACK_FUNCS.add_back_func(_expand, 0, expand_back)

tests.test_expand(Tensor)
tests.test_expand_negative_length(Tensor)
# %%


def sum_back(grad_out: Arr, out: Arr, x: Arr, dim=None, keepdim=False):
    """Basic idea: repeat grad_out over the dims along which x was summed"""
    # SOLUTION

    # If grad_out is a scalar, we need to make it a tensor (so we can expand it later)
    if not isinstance(grad_out, Arr):
        grad_out = np.array(grad_out)

    # If dim=None, this means we summed over all axes, and we want to repeat back to input shape
    if dim is None:
        dim = list(range(x.ndim))

    # If keepdim=False, then we need to add back in dims, so grad_out and x have same number of dims
    if not keepdim:
        grad_out = np.expand_dims(grad_out, dim)

    # Finally, we repeat grad_out along the dims over which x was summed
    return np.broadcast_to(grad_out, x.shape)


def _sum(x: Arr, dim=None, keepdim=False) -> Arr:
    """Like torch.sum, calling np.sum internally."""
    return np.sum(x, axis=dim, keepdims=keepdim)


sum = wrap_forward_fn(_sum)
BACK_FUNCS.add_back_func(_sum, 0, sum_back)

tests.test_sum_keepdim_false(Tensor)
tests.test_sum_keepdim_true(Tensor)
tests.test_sum_dim_none(Tensor)
tests.test_sum_nonscalar_grad_out(Tensor)
# %%

Index = int | tuple[int, ...] | tuple[Arr] | tuple[Tensor]


def coerce_index(index: Index) -> int | tuple[int, ...] | tuple[Arr]:
    """
    If index is of type signature `tuple[Tensor]`, converts it to `tuple[Arr]`.
    """
    if isinstance(index, tuple):
        return tuple([(i.array if isinstance(i, Tensor) else i) for i in index])

    return index


tests.test_coerce_index(coerce_index, Tensor)


# %%
def _getitem(x: Arr, index: Index) -> Arr:
    """Like x[index] when x is a torch.Tensor."""
    if isinstance(x, Tensor):
        x = x.array
    return x[coerce_index(index)]


def getitem_back(grad_out: Arr, out: Arr, x: Arr, index: Index):
    """
    Backwards function for _getitem.

    Hint: use np.add.at(a, indices, b)
    This function works just like a[indices] += b, except that it allows for repeated indices.
    """
    base = np.zeros(x.shape)
    np.add.at(base, coerce_index(index), grad_out)

    return base


getitem = wrap_forward_fn(_getitem)
BACK_FUNCS.add_back_func(_getitem, 0, getitem_back)

# %%
tests.test_getitem_int(Tensor)
tests.test_getitem_tuple(Tensor)
tests.test_getitem_integer_array(Tensor)
tests.test_getitem_integer_tensor(Tensor)

# %%
add = wrap_forward_fn(np.add)
subtract = wrap_forward_fn(np.subtract)
true_divide = wrap_forward_fn(np.true_divide)

BACK_FUNCS.add_back_func(
    np.add, 0, lambda grad_out, out, x, y: unbroadcast(grad_out, x)
)
BACK_FUNCS.add_back_func(
    np.add, 1, lambda grad_out, out, x, y: unbroadcast(grad_out, y)
)
BACK_FUNCS.add_back_func(
    np.subtract, 0, lambda grad_out, out, x, y: unbroadcast(grad_out, x)
)
BACK_FUNCS.add_back_func(
    np.subtract, 1, lambda grad_out, out, x, y: -unbroadcast(grad_out, y)
)
BACK_FUNCS.add_back_func(
    np.multiply, 0, lambda grad_out, out, x, y: unbroadcast(x * grad_out, x)
)
BACK_FUNCS.add_back_func(
    np.multiply, 1, lambda grad_out, out, x, y: unbroadcast(y * grad_out, y)
)
BACK_FUNCS.add_back_func(
    np.true_divide, 0, lambda grad_out, out, x, y: unbroadcast(grad_out / y, x)
)
BACK_FUNCS.add_back_func(
    np.true_divide,
    1,
    lambda grad_out, out, x, y: unbroadcast(grad_out * (-x / y**2), y),
)

# subtract_back = lambda x: add(x)
# multiply_back = lambda x: true_divide(x)
# true_divide_back = lambda x: multiply(x)

# Your code here - add to the BACK_FUNCS object

tests.test_add_broadcasted(Tensor)
tests.test_subtract_broadcasted(Tensor)
tests.test_truedivide_broadcasted(Tensor)
# %%


def add_(x: Tensor, other: Tensor, alpha: float = 1.0) -> Tensor:
    """Like torch.add_. Compute x += other * alpha in-place and return tensor."""
    np.add(x.array, other.array * alpha, out=x.array)
    return x


def safe_example():
    """This example should work properly."""
    a = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
    a.add_(b)
    c = a * b
    c.sum().backward()
    assert a.grad is not None and np.allclose(a.grad.array, [2.0, 3.0, 4.0, 5.0])
    assert b.grad is not None and np.allclose(b.grad.array, [2.0, 4.0, 6.0, 8.0])


def unsafe_example():
    """This example is expected to compute the wrong gradients."""
    a = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
    c = a * b
    a.add_(b)
    c.sum().backward()
    if a.grad is not None and np.allclose(a.grad.array, [2.0, 3.0, 4.0, 5.0]):
        print("Grad wrt a is OK!")
    else:
        print("Grad wrt a is WRONG!")
    if b.grad is not None and np.allclose(b.grad.array, [0.0, 1.0, 2.0, 3.0]):
        print("Grad wrt b is OK!")
    else:
        print("Grad wrt b is WRONG!")


safe_example()
unsafe_example()
# %%
a = Tensor([0, 1, 2, 3], requires_grad=True)
(a * 2).sum().backward()
b = Tensor([0, 1, 2, 3], requires_grad=True)
(2 * b).sum().backward()
assert a.grad is not None
assert b.grad is not None
assert np.allclose(a.grad.array, b.grad.array)


# %%
def maximum_back0(grad_out: Arr, out: Arr, x: Arr, y: Arr):
    """Backwards function for max(x, y) wrt x."""
    # SOLUTION
    bool_sum = (x > y) + 0.5 * (x == y)
    return unbroadcast(grad_out * bool_sum, x)


def maximum_back1(grad_out: Arr, out: Arr, x: Arr, y: Arr):
    """Backwards function for max(x, y) wrt y."""
    # SOLUTION
    bool_sum = (x < y) + 0.5 * (x == y)
    return unbroadcast(grad_out * bool_sum, y)


print(maximum_back0(1, 0, np.array([0, 1]), np.array([-1, 2])))
maximum = wrap_forward_fn(np.maximum)

BACK_FUNCS.add_back_func(np.maximum, 0, maximum_back0)
BACK_FUNCS.add_back_func(np.maximum, 1, maximum_back1)

tests.test_maximum(Tensor)
tests.test_maximum_broadcasted(Tensor)

# %%


def relu(x: Tensor) -> Tensor:
    """Like torch.nn.function.relu(x, inplace=False)."""
    return maximum(x, 0.0)


tests.test_relu(Tensor)


# %%
def _matmul2d(x: Arr, y: Arr) -> Arr:
    """Matrix multiply restricted to the case where both inputs are exactly 2D."""
    return x @ y


def matmul2d_back0(grad_out: Arr, out: Arr, x: Arr, y: Arr) -> Arr:
    # SOLUTION
    return grad_out @ y.T


def matmul2d_back1(grad_out: Arr, out: Arr, x: Arr, y: Arr) -> Arr:
    # SOLUTION
    return x.T @ grad_out


matmul = wrap_forward_fn(_matmul2d)
BACK_FUNCS.add_back_func(_matmul2d, 0, matmul2d_back0)
BACK_FUNCS.add_back_func(_matmul2d, 1, matmul2d_back1)

tests.test_matmul2d(Tensor)


# %%
class Parameter(Tensor):
    def __init__(self, tensor: Tensor, requires_grad=True):
        """Share the array with the provided tensor."""
        return super().__init__(tensor.array, requires_grad=requires_grad)

    def __repr__(self):
        return f"Parameter containing:\n{super().__repr__()}"


x = Tensor([1.0, 2.0, 3.0])
p = Parameter(x)
print(repr(p))
assert p.requires_grad
assert p.array is x.array
assert (
    repr(p)
    == "Parameter containing:\nTensor(array([1., 2., 3.], dtype=float32), requires_grad=True)"
)
x.add_(Tensor(np.array(2.0)))
assert np.allclose(
    p.array, np.array([3.0, 4.0, 5.0])
), "in-place modifications to the original tensor should affect the parameter"
# %%


class Module:
    _modules: dict[str, "Module"]
    _parameters: dict[str, Parameter]

    def __init__(self):
        self._modules = {}
        self._parameters = {}

    def modules(self):
        """Return the direct child modules of this module."""
        return self.__dict__["_modules"].values()

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """
        Return an iterator over Module parameters.

        recurse: if True, the iterator includes parameters of submodules, recursively.
        """
        yield from self._parameters.values()

        if recurse:
            for module in self._modules.values():
                yield from module.parameters(recurse=recurse)

    def __setattr__(self, key: str, val: Any) -> None:
        """
        If val is a Parameter or Module, store it in the appropriate _parameters or _modules dict.
        Otherwise, call __setattr__ from the superclass.
        """
        if isinstance(val, Module):
            self._modules[key] = val
        elif isinstance(val, Parameter):
            self._parameters[key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key: str) -> "Parameter | Module":
        """
        If key is in _parameters or _modules, return the corresponding value.
        Otherwise, raise KeyError.
        """
        # if not key in self._modules.keys() or key in self._parameters.keys() or s
        #      raise KeyError("not in params nor in modules!")

        if key in self._modules.keys():
            return self._modules[key]
        if key in self._parameters.keys():
            return self._parameters[key]

        return self.__dict__[key]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self):
        raise NotImplementedError("Subclasses must implement forward!")

    def __repr__(self):
        def _indent(s_, numSpaces):
            return re.sub("\n", "\n" + (" " * numSpaces), s_)

        lines = [
            f"({key}): {_indent(repr(module), 2)}"
            for key, module in self._modules.items()
        ]
        return "".join(
            [
                self.__class__.__name__ + "(",
                "\n  " + "\n  ".join(lines) + "\n" if lines else "",
                ")",
            ]
        )


class TestInnerModule(Module):
    def __init__(self):
        super().__init__()
        self.param1 = Parameter(Tensor([1.0]))
        self.param2 = Parameter(Tensor([2.0]))


class TestModule(Module):
    def __init__(self):
        super().__init__()
        self.inner = TestInnerModule()
        self.param3 = Parameter(Tensor([3.0]))


mod = TestModule()
assert list(mod.modules()) == [mod.inner]
assert list(mod.parameters()) == [
    mod.param3,
    mod.inner.param1,
    mod.inner.param2,
], "parameters should come before submodule parameters"
print("Manually verify that the repr looks reasonable:")
print(mod)
# %%


class Linear(Module):
    weight: Parameter
    bias: Parameter | None

    def __init__(self, in_features: int, out_features: int, bias=True):
        """
        A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        """
        super().__init__()
        kai_he_fact = 1 / (in_features**1 / 2)

        self.weight = Parameter(
            Tensor(
                np.random.rand(out_features, in_features) * kai_he_fact * 2
                - kai_he_fact
            )
        )

        self.bias = (
            Parameter(
                Tensor(np.random.rand(out_features) * kai_he_fact * 2 - kai_he_fact)
            )
            if bias
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (*, in_features)
        Return: shape (*, out_features)
        """
        if len(x.shape) > 1:
            out = matmul(x, self.weight.T)  # , x, "i j, b j -> b i")

        if self.bias is not None:
            out += self.bias
        return out

    def extra_repr(self) -> str:
        # note, we need to use `self.bias is not None`, because `self.bias` is either a tensor or None, not bool
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


linear = Linear(3, 4)
assert isinstance(linear.weight, Tensor)
assert linear.weight.requires_grad

input = Tensor([[1.0, 2.0, 3.0]])
output = linear(input)
assert output.requires_grad

expected_output = input @ linear.weight.T + linear.bias
np.testing.assert_allclose(output.array, expected_output.array)

print("All tests for `Linear` passed!")
# %%


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return relu(x)


# %%
class MLP(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(28 * 28, 64)
        self.linear2 = Linear(64, 64)
        self.relu1 = ReLU()
        self.relu2 = ReLU()
        self.output = Linear(64, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = x.reshape((x.shape[0], 28 * 28))
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        x = self.output(x)
        return x


# %%
# %%
X = Tensor(np.random.randint(0, 10, (3, 4)))
Y = [0, 1, 2]
print(X)
X[arange(0, 3), Y]
# %%
# Cross-entropy


def cross_entropy(logits: Tensor, true_labels: Tensor) -> Tensor:
    """Like torch.nn.functional.cross_entropy with reduction='none'.

    logits: shape (batch, classes)
    true_labels: shape (batch,). Each element is the index of the correct label in the logits.

    Return: shape (batch, ) containing the per-example loss.
    """
    print(logits, true_labels)

    num = exp(logits[arange(0, logits.shape[0]), true_labels])
    print(num)
    num = num
    den = exp(logits).sum(1)
    return -log(num / den)


tests.test_cross_entropy(Tensor, cross_entropy)


# %%
class NoGrad:
    """Context manager that disables grad inside the block. Like torch.no_grad."""

    was_enabled: bool

    def __enter__(self):
        """
        Method which is called whenever the context manager is entered, i.e. at the
        start of the `with NoGrad():` block.
        """
        global grad_tracking_enabled

        self.was_enabled = grad_tracking_enabled

        grad_tracking_enabled = False

    def __exit__(self, type, value, traceback):
        """
        Method which is called whenever we exit the context manager.
        """
        global grad_tracking_enabled

        grad_tracking_enabled = self.was_enabled


# %%
train_loader, test_loader = get_mnist()
visualize(train_loader)


# %%
class SGD:
    def __init__(self, params: Iterable[Parameter], lr: float):
        """Vanilla SGD with no additional features."""
        self.params = list(params)
        self.lr = lr
        self.b = [None for _ in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        with NoGrad():
            for i, p in enumerate(self.params):
                assert isinstance(p.grad, Tensor)
                p.add_(p.grad, -self.lr)


def train(
    model: MLP,
    train_loader: DataLoader,
    optimizer: SGD,
    epoch: int,
    train_loss_list: list | None = None,
):
    print(f"Epoch: {epoch}")
    progress_bar = tqdm(enumerate(train_loader))
    for batch_idx, (data, target) in progress_bar:
        data = Tensor(data.numpy())
        target = Tensor(target.numpy())
        optimizer.zero_grad()
        output = model(data)
        loss = cross_entropy(output, target).sum() / len(output)
        loss.backward()
        progress_bar.set_description(f"Train set: Avg loss: {loss.item():.3f}")
        optimizer.step()
        if train_loss_list is not None:
            train_loss_list.append(loss.item())


def test(model: MLP, test_loader: DataLoader, test_loss_list: list | None = None):
    test_loss = 0
    correct = 0
    with NoGrad():
        for data, target in test_loader:
            data = Tensor(data.numpy())
            target = Tensor(target.numpy())
            output: Tensor = model(data)
            test_loss += cross_entropy(output, target).sum().item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += (pred == target.reshape(pred.shape)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(
        f"Test set:  Avg loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({correct / len(test_loader.dataset):.1%})"
    )
    if test_loss_list is not None:
        test_loss_list.append(test_loss)


# %%

num_epochs = 5
model = MLP()
start = time.time()
train_loss_list = []
test_loss_list = []
optimizer = SGD(model.parameters(), 0.01)
for epoch in range(num_epochs):
    train(model, train_loader, optimizer, epoch, train_loss_list)
    test(model, test_loader, test_loss_list)
    optimizer.step()
print(f"\nCompleted in {time.time() - start: .2f}s")

line(
    train_loss_list,
    yaxis_range=[0, max(train_loss_list) + 0.1],
    labels={"x": "Batches seen", "y": "Cross entropy loss"},
    title="ConvNet training on MNIST",
    width=800,
    hovermode="x unified",
    template="ggplot2",  # alternative aesthetic for your plots (-:
)
# %%
