import torch
from torch import nn

l = nn.Linear(2, 5)
v = torch.Tensor([1.0, 2.0])

print(l(v))

seq = nn.Sequential(
    nn.Linear(2, 5), nn.ReLU(), nn.Linear(5, 3), nn.ReLU(), nn.Softmax(dim=1)
)

v2 = torch.randn(7, 2)
print(seq(v2))
print(seq(v2).shape)

for val in seq.parameters():
    print(val)
