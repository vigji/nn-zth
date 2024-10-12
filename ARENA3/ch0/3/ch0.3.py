# %%
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch as t
import torch.nn.functional as F
import wandb
from IPython.core.display import HTML
from IPython.display import display
from torch import Tensor, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from tqdm.notebook import tqdm

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"

from plotly_utils import bar, imshow, plot_train_loss_and_test_accuracy_from_trainer
from solutions_ch02 import IMAGENET_TRANSFORM, ResNet34
from solutions_bonus import get_resnet_for_feature_extraction
from utils import plot_fn, plot_fn_with_points
import tests as tests

device = t.device(
    "mps"
    if t.backends.mps.is_available()
    else "cuda"
    if t.cuda.is_available()
    else "cpu"
)

# %%


def pathological_curve_loss(x: t.Tensor, y: t.Tensor):
    # Example of a pathological curvature. There are many more possible, feel free to experiment here!
    x_loss = t.tanh(x) ** 2 + 0.01 * t.abs(x)
    y_loss = t.sigmoid(y)
    return x_loss + y_loss


plot_fn(pathological_curve_loss)
# %%
xy = t.tensor([2.5, 2.5], requires_grad=True)
optimizer = t.optim.SGD([xy])

pts_list = []
for step in range(10):
    optimizer.zero_grad()
    loss = pathological_curve_loss(xy[0], xy[1])

    loss.backward()
    optimizer.step()
    print(
        f"Step {step+1}: x = {xy[0].item():.4f}, y = {xy[1].item():.4f}, loss = {loss.item():.4f}"
    )

    new_xy = xy.detach().clone()
    pts_list.append(new_xy)
pts_list


# %%
def opt_fn_with_sgd(
    fn: Callable, xy: t.Tensor, lr=0.001, momentum=0.98, n_iters: int = 100
):
    """
    Optimize the a given function starting from the specified point.

    xy: shape (2,). The (x, y) starting point.
    n_iters: number of steps.
    lr, momentum: parameters passed to the torch.optim.SGD optimizer.

    Return: (n_iters, 2). The (x,y) BEFORE each step. So out[0] is the starting point.
    """
    # SOLUTION
    assert xy.requires_grad

    xys = t.zeros((n_iters, 2))

    # YOUR CODE HERE: run optimization, and populate `xys` with the coordinates before each step
    optimizer = optim.SGD([xy], lr=lr, momentum=momentum)

    for i in range(n_iters):
        xys[i] = xy.detach()
        out = fn(xy[0], xy[1])
        out.backward()
        optimizer.step()
        optimizer.zero_grad()

    return xys


points = []

optimizer_list = [
    (optim.SGD, {"lr": 0.1, "momentum": 0.0}),
    (optim.SGD, {"lr": 0.02, "momentum": 0.99}),
]

for optimizer_class, params in optimizer_list:
    xy = t.tensor([2.5, 2.5], requires_grad=True)
    xys = opt_fn_with_sgd(
        pathological_curve_loss,
        xy=xy,
        lr=params["lr"],
        momentum=params["momentum"],
        n_iters=99,
    )

    print(xys[-1, :])
    points.append((xys, optimizer_class, params))
plot_fn_with_points(pathological_curve_loss, points=points)


# %%
class SGD:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        """Implements SGD with momentum.

        Like the PyTorch version, but assume nesterov=False, maximize=False, and dampening=0
            https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD

        """
        self.lr = lr
        self.mu = momentum
        self.lmda = weight_decay

        params = list(
            params
        )  # turn params into a list (because it might be a generator)
        self.params = params

        self.gs = [t.zeros_like(param) for param in params]

        self.t = 0

    def zero_grad(self) -> None:
        """Zeros all gradients of the parameters in `self.params`."""
        for param in self.params:
            param.grad = None

    @t.inference_mode()  # otherwise opreations will add stuff to the gradient graph
    def step(self) -> None:
        """Performs a single optimization step of the SGD algorithm."""
        for i, (prev_grad, param) in enumerate(zip(self.gs, self.params)):
            new_g = param.grad

            if self.lmda > 0:
                new_g = new_g + self.lmda * param

            if self.mu > 0 and self.t > 0:
                new_g = self.mu * prev_grad + new_g

            self.gs[i] = new_g

            self.params[i] -= self.lr * new_g

        self.t += 1

    def __repr__(self) -> str:
        return f"SGD(lr={self.lr}, momentum={self.mu}, weight_decay={self.lmda})"


tests.test_sgd(SGD)


# %%
# RMSprop
class RMSprop:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-08,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
    ):
        """Implements RMSprop.

        Like the PyTorch version, but assumes centered=False
            https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html

        """
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.lmda = weight_decay
        self.mu = momentum

        params = list(
            params
        )  # turn params into a list (because it might be a generator)
        self.params = params

        self.vs = [t.zeros_like(param) for param in params]
        self.bs = [t.zeros_like(param) for param in params]

    def zero_grad(self) -> None:
        """Zeros all gradients of the parameters in `self.params`."""
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        for i, (prev_v, prev_b, param) in enumerate(zip(self.vs, self.bs, self.params)):
            new_grad = param.grad

            if self.lmda != 0:
                new_grad = new_grad + self.lmda * param

            v = self.alpha * prev_v + (1 - self.alpha) * new_grad**2
            self.vs[i] = v

            if self.mu > 0:
                b = self.mu * prev_b + new_grad / (t.sqrt(v) + self.eps)
                self.bs[i] = b
                self.params[i] -= self.lr * b
            else:
                self.params[i] -= self.lr * new_grad / (t.sqrt(v) + self.eps)

    def __repr__(self) -> str:
        return f"RMSprop(lr={self.lr}, eps={self.eps}, momentum={self.mu}, weight_decay={self.lmda}, alpha={self.alpha})"


tests.test_rmsprop(RMSprop)
# %%


class Adam:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
    ):
        """Implements Adam.

        Like the PyTorch version, but assumes amsgrad=False and maximize=False
            https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
        """
        params = list(
            params
        )  # turn params into a list (because it might be a generator)
        self.params = params
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.lmda = weight_decay
        self.t = 1

        self.vs = [t.zeros_like(param) for param in params]
        self.ms = [t.zeros_like(param) for param in params]

    def zero_grad(self) -> None:
        """Zeros all gradients of the parameters in `self.params`."""
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        for i, (prev_v, prev_m, param) in enumerate(zip(self.vs, self.ms, self.params)):
            new_grad = param.grad

            if self.lmda != 0:
                new_grad = new_grad + self.lmda * param

            m = self.beta1 * prev_m + (1 - self.beta1) * new_grad
            v = self.beta2 * prev_v + (1 - self.beta2) * new_grad.pow(2)
            self.ms[i] = m
            self.vs[i] = v

            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)

            self.params[i] -= self.lr * m_hat / (t.sqrt(v_hat) + self.eps)

        self.t += 1

    def __repr__(self) -> str:
        return f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps}, weight_decay={self.lmda})"


tests.test_adam(Adam)


# %%
class AdamW:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
    ):
        """Implements Adam.

        Like the PyTorch version, but assumes amsgrad=False and maximize=False
            https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
        """
        params = list(
            params
        )  # turn params into a list (because it might be a generator)
        self.params = params
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.lmda = weight_decay
        self.t = 1

        self.vs = [t.zeros_like(param) for param in params]
        self.ms = [t.zeros_like(param) for param in params]

    def zero_grad(self) -> None:
        """Zeros all gradients of the parameters in `self.params`."""
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        for i, (prev_v, prev_m, param) in enumerate(zip(self.vs, self.ms, self.params)):
            new_grad = param.grad

            self.params[i] -= self.lr * self.lmda * param

            m = self.beta1 * prev_m + (1 - self.beta1) * new_grad
            v = self.beta2 * prev_v + (1 - self.beta2) * new_grad.pow(2)
            self.ms[i] = m
            self.vs[i] = v

            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)

            self.params[i] -= self.lr * m_hat / (t.sqrt(v_hat) + self.eps)

        self.t += 1

    def __repr__(self) -> str:
        return f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps}, weight_decay={self.lmda})"


tests.test_adamw(AdamW)
# %%
