import argparse
import os

import matplotlib.pyplot as plot
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm


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


class CNN(nn.Module):

    def __init__(self, input_shape, num_classes, hidden_dim=100):
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 8, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.linear = nn.Linear(32 * (input_shape[1] // 8) * (input_shape[2] // 8), hidden_dim)
        self.logits = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = self.logits(x)
        return x


class Block(nn.Module):

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
            Block(32, 16, 0),
            Block(16, 24, 0.2 / 12, stride=2),
            Block(24, 24, 0.2 * 2 / 12),
            Block(24, 40, 0.2 * 3 / 12, kernel_size=5, stride=2),
            Block(40, 40, 0.2 * 4 / 12, kernel_size=5),
            Block(40, 80, 0.2 * 5 / 12, stride=2),
            Block(80, 80, 0.2 * 6 / 12),
            Block(80, 80, 0.2 * 7 / 12),
            Block(80, 112, 0.2 * 8 / 12, kernel_size=5),
            Block(112, 112, 0.2 * 9 / 12, kernel_size=5),
            Block(112, 112, 0.2 * 10 / 12, kernel_size=5),
            Block(112, 192, 0.2 * 11 / 12)
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


class EfficientNetB0(nn.Module):

    def __init__(self, input_shape, num_classes, hidden_size=1280):
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(32)
        blocks = [
            Block(32, 16, 0),
            Block(16, 24, 0.2 / 16, stride=2),
            Block(24, 24, 0.2 * 2 / 16),
            Block(24, 40, 0.2 * 3 / 16, kernel_size=5, stride=2),
            Block(40, 40, 0.2 * 4 / 16, kernel_size=5),
            Block(40, 80, 0.2 * 5 / 16, stride=2),
            Block(80, 80, 0.2 * 6 / 16),
            Block(80, 80, 0.2 * 7 / 16),
            Block(80, 112, 0.2 * 8 / 16, kernel_size=5),
            Block(112, 112, 0.2 * 9 / 16, kernel_size=5),
            Block(112, 112, 0.2 * 10 / 16, kernel_size=5),
            Block(112, 192, 0.2 * 11 / 16, kernel_size=5, stride=2),
            Block(192, 192, 0.2 * 12 / 16, kernel_size=5),
            Block(192, 192, 0.2 * 13 / 16, kernel_size=5),
            Block(192, 192, 0.2 * 14 / 16, kernel_size=5),
            Block(192, 320, 0.2 * 15 / 16)
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


def get_loaders(config):
    dataset = getattr(torchvision.datasets, config.dataset)

    train_data = dataset(config.path, train=True, download=True).data
    if not isinstance(train_data, torch.Tensor):
        train_data = torch.tensor(train_data)
    mean = torch.mean(train_data.data.float() / 255)
    std = torch.std(train_data.data.float() / 255)

    train_data = dataset(
        config.path,
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((mean,), (std,))
        ])
    )
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, pin_memory=True)

    test_data = dataset(
        config.path,
        train=False,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((mean,), (std,))
        ])
    )
    test_loader = DataLoader(test_data, batch_size=config.batch_size, pin_memory=True)

    return train_loader, test_loader


def forward_batch(config, x, y, model, criterion, optimizer=None):
    x = x.to(config.device)
    y = y.to(config.device)
    y_pred = model(x)

    loss = criterion(y_pred, y)
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss = float(loss)
    accuracy = int(torch.sum(torch.argmax(y_pred, 1) == y)) / config.batch_size

    return loss, accuracy


def show_curves(losses, test_losses, accuracies, test_accuracies):
    filter_window = 100
    if len(losses) > 10 * filter_window:
        filtered_loss = np.convolve(losses, np.ones(filter_window) / filter_window, mode='valid')
        plot.semilogy(losses, color='tab:blue', alpha=0.5, label='_nolegend_')
        plot.semilogy(np.linspace(0, len(losses) - 1, len(filtered_loss)), filtered_loss, color='tab:blue')
    else:
        plot.semilogy(losses)
    if test_losses:
        plot.semilogy(np.linspace(0, len(losses) - 1, len(test_losses)), test_losses, color='tab:orange')

    plot.title('Training loss')
    plot.xlabel('Epoch')
    plot.ylabel('Loss')
    legend = ['loss']
    if test_losses:
        legend.append('test_loss')
    plot.legend(legend)
    x_ticks = np.arange(config.epochs + 1)
    x_locs = np.linspace(0, len(losses) - 1, len(x_ticks))
    plot.xticks(x_locs, x_ticks)

    plot.show()

    plot.plot(accuracies)
    plot.plot(np.linspace(0, len(accuracies) - 1, len(test_accuracies)), test_accuracies)

    plot.title('Training accuracies')
    plot.xlabel('Epoch')
    plot.ylabel('Accuracy')
    plot.legend(['accuracy', 'test_accuracy'])
    x_ticks = np.arange(config.epochs + 1)
    x_locs = np.linspace(0, len(accuracies) - 1, len(x_ticks))
    plot.xticks(x_locs, x_ticks)

    plot.show()


def visualize_predictions(config, loader, model):
    classes = loader.dataset.classes
    loader = iter(loader)

    for i in range(5):
        for j in range(5):
            x, y = next(loader)
            x = x.to(config.device)
            y = y.to(config.device)
            y_pred = torch.argmax(model(x)[0], 0)
            y_pred = classes[int(y_pred)]
            x = torch.permute(x[0], (1, 2, 0)).cpu()
            x = torch.clip((x + 1) / 2, 0, 1)
            y = classes[int(y[0])]

            plot.subplot(5, 5, i * 5 + j + 1)
            plot.title(y_pred, color='green' if y_pred == y else 'red')
            plot.axis('off')
            plot.imshow(x)

    plot.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model', type=str, default='EfficientNet')
    parser.add_argument('--path', type=str, default='files')
    parser.add_argument('--test-every', type=int, default=1)
    config = parser.parse_args()

    args = vars(config)
    max_len = 0
    for arg in args:
        max_len = max(max_len, len(arg))
    for arg in args:
        value = str(getattr(config, arg))
        display = '{:<%i}: {}' % (max_len + 1)
        print(display.format(arg, value))

    plot.style.use('seaborn-darkgrid')

    train_loader, test_loader = get_loaders(config)
    input_shape = next(iter(train_loader))[0][0].shape
    if len(input_shape) == 2:
        input_shape = (1, *input_shape)
    num_classes = len(train_loader.dataset.classes)

    if config.model == 'MLP':
        model = MLP(input_shape, num_classes).to(config.device)
    elif config.model == 'CNN':
        model = CNN(input_shape, num_classes).to(config.device)
    elif config.model == 'EfficientNet':
        model = EfficientNet(input_shape, num_classes).to(config.device)
    else:
        raise NotImplemented(f'Model {config.model} not implemented.')

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    metrics = {}
    losses = []
    test_losses = [float('nan')]
    accuracies = []
    test_accuracies = [float('nan')]

    for epoch in range(config.epochs):
        iterator = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.epochs}')

        mean_accuracy = 0
        for i, (x, y) in enumerate(iterator):
            loss, accuracy = forward_batch(config, x, y, model, criterion, optimizer=optimizer)
            losses.append(loss)
            metrics['loss'] = format(float(loss), '.3e')
            mean_accuracy = (mean_accuracy * i + accuracy) / (i + 1)
            accuracy = 100 * float(mean_accuracy)
            accuracies.append(accuracy)
            accuracy = format(accuracy, '.3f')
            metrics['accuracy'] = f'{accuracy}%'
            iterator.set_postfix(metrics)

        if epoch % config.test_every == 0:
            model.eval()
            test_loss = 0
            test_accuracy = 0

            iterator = tqdm(test_loader, desc=f'Testing')
            for i, (x, y) in enumerate(iterator):
                loss, accuracy = forward_batch(config, x, y, model, criterion)
                test_loss += float(loss)
                test_accuracy += float(accuracy)

                if i == len(test_loader) - 1:
                    test_loss = test_loss / len(test_loader)
                    test_losses.append(test_loss)
                    test_loss = format(test_loss, '.3e')
                    metrics['test_loss'] = test_loss
                    test_accuracy = 100 * test_accuracy / len(test_loader)
                    test_accuracies.append(test_accuracy)
                    test_accuracy = format(test_accuracy, '.3f')
                    test_accuracy = f'{test_accuracy}%'
                    metrics['test_accuracy'] = test_accuracy
                    iterator.set_postfix({'test_loss': test_loss, 'test_accuracy': test_accuracy})

            model.train()

    show_curves(losses, test_losses, accuracies, test_accuracies)
    model.eval()
    visualize_predictions(config, test_loader, model)
