import torch
from torch import nn as nn


class MLP(nn.Module):

    def __init__(self, input_shape, num_classes, hidden_dim=1000):
        super().__init__()
        self.linear = nn.Linear(input_shape[0] * input_shape[1] * input_shape[2], hidden_dim)
        self.logits = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = torch.relu(x)
        x = self.logits(x)
        return x
