from turtle import forward
import torch
from torch import nn


class MyModule(nn.Module):
    def __init__(self, n_inputs, n_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pipe = nn.Sequential(
            nn.Linear(n_inputs, 5),
            nn.ReLU(),
            nn.Linear(5, n_classes),
            nn.ReLU(),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.pipe(x)


n_in = 3
seq = MyModule(n_inputs=n_in, n_classes=8)
v2 = torch.randn(7, n_in)
print(seq(v2))
print(seq(v2).shape)

# for val in seq.parameters():
#   print(val)
