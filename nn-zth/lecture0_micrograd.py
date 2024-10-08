## Lecture 0: Micrograd
# Following material from: https://www.youtube.com/watch?v=VMj-3S1tku0

# %%

import math
import numpy as np
import matplotlib.pyplot as plt


# %%
def f(x):
    return 3 * x**2 - 4 * x + 5


# %%
f(3.0)
# %%
xs = np.arange(-5, 5, 0.25)
ys = f(xs)
plt.plot(xs, ys)
# %%
h = 0.000001
x = 2 / 3  # minimum
(f(x + h) - f(x)) / h

# %%
h = 0.00001
a = 2.0
b = -3.0
c = 10.0
d = a * b + c
print(d)

d1 = a * b + c
a += h
d2 = a * b + c
print(f"d1: {d1}, d2: {d2}, Numerical derivative: {(d2 - d1)/h}")


# %%
class Value:
    def __init__(self, val, _cildren=(), _op="", label="") -> None:
        self.data = val
        self._op = _op
        self.label = label
        self.grad = 0
        self._backward = lambda: None

        self._prev = set(_cildren)

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):
        return self + (other * -1)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * (other**-1)

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            self.grad += (
                out.data * out.grad
            )  # out is the output of the exp function above

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only support int/float powers"
        out = Value(self.data**other, (self,), f"{self.data}**{other}")

        def _backward():
            self.grad = other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    # %%


a = Value(2.0, label="a")
b = Value(-3, label="b")
c = Value(10.0, label="c")

e = a * b
e.label = "e"
d = e + c
d.label = "d"

f = Value(-2.0, label="f")
L = f * d
L.label = "L"
# %%
d._prev
# %%
from graphviz import Digraph


def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)

    return nodes, edges


def draw_dot(root):
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})

    nodes, edges = trace(root)

    for node in nodes:
        dot.node(
            name=str(id(node)), label=f"{node.label}: {node.data}; grad: {node.grad}"
        )

        if node._op:
            dot.node(name=str(id(node)) + node._op, label=f"{node._op}")

            dot.edge(str(id(node)) + node._op, str(id(node)))

    for src, dst in edges:
        dot.edge(str(id(src)), str(id(dst)) + (dst._op or ""))

    return dot


draw_dot(L)


# %%
def lol():
    h = 0.00001

    a = Value(2.0, label="a")
    b = Value(-3, label="b")
    c = Value(10.0, label="c")

    e = a * b
    e.label = "e"
    d = e + c
    d.label = "d"

    f = Value(-2.0, label="f")
    L = f * d
    L.label = "L"
    L1 = L.data

    a = Value(2.0 + h, label="a")
    e = a * b
    e.label = "e"
    d = e + c
    d.label = "d"
    L = f * d
    L.label = "L"
    L2 = L.data

    print(f"Numerical derivative: {(L2 - L1)/h}")


lol()


# %%
## Implementing a simple perceptron:

# Activation functions: tanh
x = np.arange(-5, 5, 0.25)
plt.plot(x, np.tanh(x))
plt.grid(True)
# %%
# Initialize using Value 2 inputs, 2 weights, 1 bias and label them:
x1, x2 = Value(2.0, label="x1"), Value(0.0, label="x2")
w1, w2 = Value(-3.0, label="w1"), Value(1.0, label="w2")
b = Value(6.8813735870195432, label="b")

x1w1 = x1 * w1
x1w1.label = "x1*w1"
x2w2 = x2 * w2
x2w2.label = "x2*w2"
x1w1x2w2 = x1w1 + x2w2
x1w1x2w2.label = "x1*w1 + x2*w2"
n = x1w1x2w2 + b
n.label = "n"
o = n.tanh()
o.label = "o"

# %%
draw_dot(o)


# %%
# How can we implement the backpropagation automatically?
# Let's add a method to the Value class (see addition above)
o.grad = 1.0
o._backward()
n._backward()
x1w1x2w2._backward()
x2w2._backward()
x1w1._backward()
x1._backward()
w1._backward()

draw_dot(o)

# %%
# How to call the backward method?
# We need to first topologically sort the graph, to make sure
# all dependencies flow "left-to-right".

# See class for an implementation:


# So now:
o.backward()
draw_dot(o)
# %%
# Let's try to break down tanh into smaller functions:
# Initialize using Value 2 inputs, 2 weights, 1 bias and label them:
x1, x2 = Value(2.0, label="x1"), Value(0.0, label="x2")
w1, w2 = Value(-3.0, label="w1"), Value(1.0, label="w2")
b = Value(6.8813735870195432, label="b")

x1w1 = x1 * w1
x1w1.label = "x1*w1"
x2w2 = x2 * w2
x2w2.label = "x2*w2"
x1w1x2w2 = x1w1 + x2w2
x1w1x2w2.label = "x1*w1 + x2*w2"
n = x1w1x2w2 + b
n.label = "n"
e = (n * 2).exp()
e.label = "e"
o = (e - 1) / (e + 1)
o.label = "o"
b.grad = 1.0
b._backward()

draw_dot(o)
# %%

# Real world: we work with PyTorch tensors!

import torch

# By default, torch tensors are assumed to be leaf nodes, which would need no gradients.
# We need to set requires_grad = True to tell PyTorch that we want to compute gradients.
x1 = torch.Tensor([2.0]).double()
x1.requires_grad = True
x2 = torch.Tensor([0.0]).double()
x2.requires_grad = True
w1 = torch.Tensor([-3.0]).double()
w1.requires_grad = True
w2 = torch.Tensor([1.0]).double()
w2.requires_grad = True
b = torch.Tensor([6.8813735870195432]).double()
b.requires_grad = True
n = x1 * w1 + x2 * w2 + b
o = torch.tanh(n)

print(o.data.item())
o.backward()

print("---")
print("x2", x2.grad.item())
print("w2", w2.grad.item())
print("x1", x1.grad.item())
print("w1", w1.grad.item())


# %%
torch.Tensor([2.0]).dtype
# %%

## Neural Networks
# Let's build a simple neural network, using a simplified version
# of the PyTorch API.

# %%


# %%
import random


class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        assert len(x) == len(
            self.w
        ), "Length of x must be equal to the number of weights"
        # w * x is a dot product
        # we implement w * x+ b
        out = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        out = out.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


x = [2.0, 3.0]
n = Neuron(2)
n(x)

# %%


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        result = [n(x) for n in self.neurons]
        return result[0] if len(result) == 1 else result

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


# %%
x = [2.0, 3.0, -1.0]
n = Layer(3, 3)

n(x)

# %%


class MLP:
    def __init__(self, nin, nouts):
        layer_sizes = [nin] + nouts  # sizes for each layer
        self.layers = [
            Layer(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(nouts))
        ]

    def __call__(self, x):
        # This implements the forward pass
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# %%
x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])
n(x)
n.parameters()

# %%
# 4 inputs and 4 targets for the neural net:
xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]

y_predictions = [n(x) for x in xs]
y_predictions
# %%
# Compute the loss first:
loss = sum((y_pred - y_true) ** 2 for y_pred, y_true in zip(y_predictions, ys))
# loss
loss
# %%
loss.backward()  # magic is happening here!
# %%

n.layers[0].neurons[0].w[0].grad
# %%
n.layers[0].neurons[0].w[0].data
# %%
# This is a single step of gradient descent:
for p in n.parameters():
    p.data += -0.01 * p.grad  # negative sign

# %%
n.layers[0].neurons[0].w[0].data

# %%
# Now we can do a full training loop:
n = MLP(3, [4, 4, 1])
for k in range(3000):
    # forward:pass
    y_predictions = [n(x) for x in xs]

    # compute loss
    loss = sum((y_pred - y) ** 2 for y_pred, y in zip(y_predictions, ys))
    print(f"k: {k}, loss: {loss.data}")

    for p in n.parameters():
        p.grad = 0

    # backward:
    loss.backward()

    # This is a single step of gradient descent:
    for p in n.parameters():
        p.data += -0.01 * p.grad  # negative sign

# %%
