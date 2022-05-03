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
        self.linear = nn.Linear(28 * 28, 1000)
        self.logits = nn.Linear(1000, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = torch.relu(x)
        x = self.logits(x)
        return x


def get_data(config):
    train_data = torch.utils.data.DataLoader(
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

    test_data = torch.utils.data.DataLoader(
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

    return train_data, test_data


def forward_batch(config, x, y, model, criterion, optimizer=None):
    x = x.to(config.device)
    y = y.to(config.device)
    x_pred = model(x)

    loss = criterion(x_pred, y)
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss = float(loss)
    accuracy = int(torch.sum(torch.argmax(x_pred, 1) == y)) / config.batch_size

    return loss, accuracy


def show_curves(losses, val_losses, accuracies, val_accuracies):
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

    plot.plot(accuracies)
    plot.plot(np.linspace(0, len(accuracies) - 1, len(val_accuracies)), val_accuracies)

    plot.title('Training accuracies')
    plot.xlabel('Epoch')
    plot.ylabel('Accuracy')
    plot.legend(['accuracy', 'val_accuracy'])
    x_ticks = np.arange(config.epochs + 1)
    x_locs = np.linspace(0, len(accuracies) - 1, len(x_ticks))
    plot.xticks(x_locs, x_ticks)

    plot.show()


def visualize_predictions(config, data, model):
    data = iter(data)
    for i in range(5):
        for j in range(5):
            index = np.random.randint(config.batch_size)
            x, y = next(data)
            x = x.to(config.device)
            y = y.to(config.device)
            x_pred = int(torch.argmax(model(x)[index], 0))
            x = 1 - x[index, 0].cpu()
            y = int(y[index])

            plot.subplot(5, 5, i * 5 + j + 1)
            plot.title(x_pred, color='green' if x_pred == y else 'red')
            plot.axis('off')
            plot.imshow(x)

    plot.show()


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

    train_data, test_data = get_data(config)
    model = MLP().to(config.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    metrics = {}
    losses = []
    val_losses = [float('nan')]
    accuracies = []
    val_accuracies = [float('nan')]

    for epoch in range(config.epochs):
        iterator = tqdm(train_data, desc=f'Epoch {epoch + 1}/{config.epochs}')

        for x, y in iterator:
            loss, accuracy = forward_batch(config, x, y, model, criterion, optimizer=optimizer)
            losses.append(loss)
            metrics['loss'] = format(float(loss), '.3e')
            accuracy = 100 * float(accuracy)
            accuracies.append(accuracy)
            accuracy = format(accuracy, '.3f')
            metrics['accuracy'] = f'{accuracy}%'
            iterator.set_postfix(metrics)

        if epoch % config.eval_every == 0:
            model.eval()
            val_loss = 0
            val_accuracy = 0

            for x, y in test_data:
                loss, accuracy = forward_batch(config, x, y, model, criterion)
                val_loss += float(loss)
                val_accuracy += float(accuracy)

            val_loss = val_loss / len(test_data)
            val_accuracy = 100 * val_accuracy / len(test_data)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            metrics['val_loss'] = format(val_loss, '.3e')
            val_accuracy = format(val_accuracy, '.3f')
            metrics['val_accuracy'] = f'{val_accuracy}%'
            iterator.set_postfix(metrics)  # FIXME
            model.train()

    show_curves(losses, val_losses, accuracies, val_accuracies)
    model.eval()
    visualize_predictions(config, test_data, model)
