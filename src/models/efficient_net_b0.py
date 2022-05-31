import torch
from torch import nn as nn

from models.efficient_net import ENBlock


class EfficientNetB0(nn.Module):

    def __init__(self, input_shape, num_classes, hidden_size=1280):
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(32)
        blocks = [
            ENBlock(32, 16, 0),
            ENBlock(16, 24, 0.2 / 16, stride=2),
            ENBlock(24, 24, 0.2 * 2 / 16),
            ENBlock(24, 40, 0.2 * 3 / 16, kernel_size=5, stride=2),
            ENBlock(40, 40, 0.2 * 4 / 16, kernel_size=5),
            ENBlock(40, 80, 0.2 * 5 / 16, stride=2),
            ENBlock(80, 80, 0.2 * 6 / 16),
            ENBlock(80, 80, 0.2 * 7 / 16),
            ENBlock(80, 112, 0.2 * 8 / 16, kernel_size=5),
            ENBlock(112, 112, 0.2 * 9 / 16, kernel_size=5),
            ENBlock(112, 112, 0.2 * 10 / 16, kernel_size=5),
            ENBlock(112, 192, 0.2 * 11 / 16, kernel_size=5, stride=2),
            ENBlock(192, 192, 0.2 * 12 / 16, kernel_size=5),
            ENBlock(192, 192, 0.2 * 13 / 16, kernel_size=5),
            ENBlock(192, 192, 0.2 * 14 / 16, kernel_size=5),
            ENBlock(192, 320, 0.2 * 15 / 16)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.conv2 = nn.Conv2d(320, hidden_size, kernel_size=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x = torch.nn.functional.interpolate(x, scale_factor=3)

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = torch.relu(x)

        for block in self.blocks:
            x = block(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = torch.relu(x)
        n, c = x.shape[:2]
        x = torch.mean(x.view((n, c, -1)), dim=2)
        x = self.dropout(x)
        x = self.linear(x)
        return x