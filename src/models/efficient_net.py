import torch
from torch import nn as nn


class ENBlock(nn.Module):

    def __init__(self, channel_in, channel_out, dropout, kernel_size=3, stride=1):
        super().__init__()
        channel_hidden = channel_in * 6
        channel_se = channel_in // 4
        padding = 0
        if kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2

        self.conv1 = nn.Conv2d(channel_in, channel_hidden, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(channel_hidden)
        self.conv2 = nn.Conv2d(
            channel_hidden,
            channel_hidden,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            groups=channel_hidden,
            bias=False
        )
        self.batch_norm2 = nn.BatchNorm2d(channel_hidden)
        self.conv3 = nn.Conv2d(channel_hidden, channel_se, kernel_size=1)
        self.conv4 = nn.Conv2d(channel_se, channel_hidden, kernel_size=1)
        self.conv5 = nn.Conv2d(channel_hidden, channel_out, kernel_size=1, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(channel_out)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        y = self.conv1(x)
        y = self.batch_norm1(y)
        y = torch.relu(y)
        y = self.conv2(y)
        y = self.batch_norm2(y)
        y = torch.relu(y)
        n, c = y.shape[:2]
        s = torch.mean(y.view((n, c, -1)), dim=2)
        s = s.view((n, -1, 1, 1))
        s = self.conv3(s)
        s = torch.relu(s)
        s = self.conv4(s)
        s = torch.sigmoid(s)
        y = y * s
        y = self.conv5(y)
        y = self.batch_norm3(y)

        if x.shape == y.shape:
            x = self.dropout(x + y)
        else:
            x = y

        return x


class EfficientNet(nn.Module):

    def __init__(self, input_shape, num_classes, hidden_size=1280):
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(32)
        blocks = [
            ENBlock(32, 16, 0),
            ENBlock(16, 24, 0.2 / 12, stride=2),
            ENBlock(24, 24, 0.2 * 2 / 12),
            ENBlock(24, 40, 0.2 * 3 / 12, kernel_size=5, stride=2),
            ENBlock(40, 40, 0.2 * 4 / 12, kernel_size=5),
            ENBlock(40, 80, 0.2 * 5 / 12, stride=2),
            ENBlock(80, 80, 0.2 * 6 / 12),
            ENBlock(80, 80, 0.2 * 7 / 12),
            ENBlock(80, 112, 0.2 * 8 / 12, kernel_size=5),
            ENBlock(112, 112, 0.2 * 9 / 12, kernel_size=5),
            ENBlock(112, 112, 0.2 * 10 / 12, kernel_size=5),
            ENBlock(112, 192, 0.2 * 11 / 12)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.conv2 = nn.Conv2d(192, hidden_size, kernel_size=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
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
