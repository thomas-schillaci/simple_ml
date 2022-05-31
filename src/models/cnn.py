from math import log2

import torch
from torch import nn as nn


class CNNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.layer_norm = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv(x)
        x = torch.relu(x)
        x = self.layer_norm(x)
        return x


class CNN(nn.Module):

    def __init__(
            self,
            input_shape,
            num_classes,
            embedding_filters_log=2,
            dropout=0.2,
            hidden_dim=100
    ):
        super().__init__()
        self.embedding = nn.Conv2d(
            input_shape[0],
            2 ** embedding_filters_log,
            kernel_size=1,
            bias=False
        )
        num_stages = min(
            7,
            min(
                int(log2(input_shape[-1]) - log2(embedding_filters_log)),
                int(log2(input_shape[-2]) - log2(embedding_filters_log))
            )
        )
        self.stages = []
        for i in range(num_stages):
            self.stages.append(CNNBlock(
                2 ** (i + embedding_filters_log),
                2 ** (i + embedding_filters_log + 1),
                dropout
            ))
        self.stages = nn.ModuleList(self.stages)
        out_filters = 2 ** (num_stages + embedding_filters_log)
        dim_x = input_shape[-2] // (2 ** num_stages)
        dim_y = input_shape[-1] // (2 ** num_stages)
        self.linear = nn.Linear(out_filters * dim_x * dim_y, hidden_dim)
        self.logits = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        for stage in self.stages:
            x = stage(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = self.logits(x)
        return x