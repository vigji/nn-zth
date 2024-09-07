# %%
import math
from cycler import V
import numpy as np
import matplotlib.pyplot as plt



# %%
def f(x):
    return 3*x**2 - 4*x + 5

# %%
f(3.)
# %%
xs = np.arange(-5, 5, 0.25)
ys = f(xs)
plt.plot(xs, ys)
# %%
h = 0.000001
x = 2/3  # minimum
(f(x + h) - f(x))/h

# %%
h = 0.00001
a = 2.
b = -3.
c = 10.
d = a*b + c
print(d)

d1 = a*b + c
a += h
d2 = a*b + c
print(f"d1: {d1}, d2: {d2}, Numerical derivative: {(d2 - d1)/h}")

# %%
class Value: 
    def __init__(self, val, _cildren=(), _op="", label="") -> None:
        self.data = val
        self._op = _op
        self.label=label

        self._prev = set(_cildren)

    def __repr__(self) -> str:
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        out = Value(self.data + other.data, 
                    (self, other), 
                    "+")
        return out
    
    def __mul__(self, other):
        out = Value(self.data * other.data, 
                    (self, other), 
                    "*")
        return out
    
    # %%
a = Value(2., label="a")
b = Value(-3, label="b")
c = Value(10., label="c") 

e = a*b; e.label = "e"
d = e + c; d.label = "d"

f = Value(-2., label="f")
L = f * d; L.label = "L"
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
        dot.node(name=str(id(node)), label=f"{node.label}: {node.data}")

        if node._op:
            dot.node(name=str(id(node)) + node._op, label=f"{node._op}")

            dot.edge(str(id(node)) + node._op, str(id(node)))

        
    for src, dst in edges:
        dot.edge(str(id(src)), str(id(dst)) + (dst._op or ""))

    return dot

draw_dot(L)
    
# %%


