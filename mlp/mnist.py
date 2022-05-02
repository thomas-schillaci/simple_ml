import argparse

import matplotlib.pyplot as plot
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm


class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(28 * 28, 1000)
        self.linear2 = nn.Linear(1000, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        # x = torch.softmax(x, -1)
        return x


def setup(config):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            'files/',
            train=True,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))  # TODO check me
            ])),
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            'files/',
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=config.batch_size,
        pin_memory=True
    )

    model = MLP().to(config.device)

    return train_loader, test_loader, model


def train(config, data, val_data, model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    metrics = {}
    losses = []
    val_losses = [float('nan')]

    for epoch in range(config.epochs):
        trange = tqdm(total=len(data), desc=f'Epoch {epoch + 1}/{config.epochs}')
        for x, y in data:
            x = x.to(config.device)
            y = y.to(config.device)
            x_pred = model(x)

            loss = criterion(x_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = float(loss)
            losses.append(loss)
            loss = format(loss, '.3e')
            metrics['loss'] = loss
            trange.set_postfix_str(metrics)
            trange.update()

        if epoch % config.eval_every == 0:
            val_loss = eval(config, val_data, model, criterion)
            val_losses.append(val_loss)
            val_loss = format(val_loss, '.3e')
            metrics['val_loss'] = val_loss
            trange.set_postfix_str(metrics)
            trange.update()

    plot.semilogy(losses)
    plot.semilogy(np.linspace(0, len(losses) - 1, len(val_losses)), val_losses)

    plot.title('Training losses')
    plot.xlabel('Epoch')
    plot.ylabel('Loss')
    plot.legend(['loss', 'val_loss'])
    x_ticks = np.arange(config.epochs + 1)
    x_locs = np.linspace(0, len(losses) - 1, len(x_ticks))
    plot.xticks(x_locs, x_ticks)

    plot.show()


def eval(config, data, model, criterion):
    model.eval()

    val_loss = 0
    for x, y in data:
        x = x.to(config.device)
        y = y.to(config.device)
        x_pred = model(x)

        loss = criterion(x_pred, y)
        val_loss += float(loss)

    val_loss /= len(data)

    model.train()

    return val_loss


def test(config, data, model):
    pass


model = torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--eval-every', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
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

    train_data, test_data, model = setup(config)
    train(config, train_data, test_data, model)
    test(config, test_data, model)
