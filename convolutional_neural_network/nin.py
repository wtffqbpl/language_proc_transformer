#! coding: utf-8

import unittest
from unittest.mock import patch
from io import StringIO
import torch
import torch.nn as nn
from torch.utils import data
import torchvision
from torchvision import transforms
import torchinfo
from utils.accumulator import Accumulator


class NiNBlock(nn.Module):
    def __init__(self, out_channels, kernel_size, stride, padding):
        super(NiNBlock, self).__init__()

        self.net = nn.Sequential(
            nn.LazyConv2d(out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.LazyConv2d(out_channels, kernel_size=1),
            nn.ReLU(),
            nn.LazyConv2d(out_channels, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class NiN(nn.Module):
    def __init__(self):
        super(NiN, self).__init__()

        self.net = nn.Sequential(
            NiNBlock(out_channels=96, kernel_size=11, stride=4, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            NiNBlock(out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            NiNBlock(out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            NiNBlock(out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, x):
        return self.net(x)


def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]

    if resize is not None:
        trans.insert(0, transforms.Resize(size=resize))
    trans = transforms.Compose(trans)

    mnist_train = torchvision.datasets.FashionMNIST(
        root='../data', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root='../data', train=False, transform=trans, download=True)

    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4))


def accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)

    cmp = y_hat.type(y.dtype) == y
    return float(cmp.to(y.dtype).sum())


def evaluate(model, data_iter):
    if isinstance(model, nn.Module):
        model.eval()

    metric = Accumulator(2)
    with torch.no_grad():
        for x, y in data_iter:
            metric.add(accuracy(model(x), y), torch.numel(y))

    return metric[0] / metric[1]


def init_model(m):
    # initialize weights
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.LazyConv2d) or \
            isinstance(m, nn.Linear) or isinstance(m, nn.LazyLinear):
        nn.init.xavier_uniform_(m.weight)

    # initialize bias
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.zeros_(m.bias)


def train(model, data_iter, test_iter, optimizer, loss_fn, num_epochs, device):
    for epoch in range(num_epochs):
        loss = None
        for i, (x, y) in enumerate(data_iter):
            # Copy data to the specified device
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], ',
                      f'Step [{i + 1}/{len(data_iter)}], ',
                      f'Loss: {loss.item()}')

        print(f'Epoch [{epoch + 1}/{num_epochs}], ',
              f'Loss: {loss.item()}',
              f'Accuracy: {evaluate(model, test_iter)}')


class IntegrationTest(unittest.TestCase):
    def test_nin_model_shape(self):
        model = NiN()

        output = None
        with patch('sys.stdout', new_callable=StringIO) as log:
            torchinfo.summary(model, input_size=(1, 1, 224, 224))
            output = log.getvalue().strip()

        print(output)

        expected_output = """
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
NiN                                      [1, 10]                   --
├─Sequential: 1-1                        [1, 10]                   --
│    └─NiNBlock: 2-1                     [1, 96, 54, 54]           --
│    │    └─Sequential: 3-1              [1, 96, 54, 54]           30,336
│    └─MaxPool2d: 2-2                    [1, 96, 26, 26]           --
│    └─NiNBlock: 2-3                     [1, 256, 26, 26]          --
│    │    └─Sequential: 3-2              [1, 256, 26, 26]          746,240
│    └─MaxPool2d: 2-4                    [1, 256, 12, 12]          --
│    └─NiNBlock: 2-5                     [1, 384, 12, 12]          --
│    │    └─Sequential: 3-3              [1, 384, 12, 12]          1,180,800
│    └─MaxPool2d: 2-6                    [1, 384, 5, 5]            --
│    └─NiNBlock: 2-7                     [1, 10, 5, 5]             --
│    │    └─Sequential: 3-4              [1, 10, 5, 5]             34,790
│    └─AdaptiveAvgPool2d: 2-8            [1, 10, 1, 1]             --
│    └─Flatten: 2-9                      [1, 10]                   --
==========================================================================================
Total params: 1,992,166
Trainable params: 1,992,166
Non-trainable params: 0
Total mult-adds (M): 763.82
==========================================================================================
Input size (MB): 0.20
Forward/backward pass size (MB): 12.20
Params size (MB): 7.97
Estimated Total Size (MB): 20.37
==========================================================================================
        """

        self.assertEqual(expected_output.strip(), output)

    def test_nin_model(self):
        # Hyperparameters
        batch_size = 128
        learning_rate = 0.1
        num_epochs = 10

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = NiN()
        model.to(device)

        # Dummy initialization for LazyCon2d
        _ = model(torch.randn(1, 1, 224, 224).to(device))

        # Apply initialization method to the model
        model.apply(init_model)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)

        train(model, train_iter, test_iter, optimizer, loss_fn, num_epochs, device)

        self.assertTrue(True)


if __name__ == "__main__":
    pass
