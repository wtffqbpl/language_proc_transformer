#! coding: utf-8

import unittest
from unittest.mock import patch
from io import StringIO
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import torchinfo
import sys
from pathlib import Path
sys.path.append((str(Path(__file__).resolve().parent.parent)))
from utils.accumulator import Accumulator


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Residual, self).__init__()

        self.conv1 = nn.LazyConv2d(out_channels=out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.LazyConv2d(out_channels=out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.LazyConv2d(out_channels, kernel_size=1, stride=stride))

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y = y + self.shortcut(x)
        return y


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        b1 = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        b2 = nn.Sequential(*self.resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*self.resnet_block(64, 128, 2))
        b4 = nn.Sequential(*self.resnet_block(128, 256, 2))
        b5 = nn.Sequential(*self.resnet_block(256, 512, 2))

        self.net = nn.Sequential(
            b1, b2, b3, b4, b5,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LazyLinear(10))

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def resnet_block(in_channels, num_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(in_channels, num_channels, stride=2))
            else:
                blk.append(Residual(num_channels, num_channels))
        return blk


# one of the challenges one encounters in the design of ResNet is the trade-off
# between nonlinear and dimensionality within a given block. That is, we could
# add more non-linearity by increasing the number of layers, or by increasing
# the width of the convolutions. An alternative strategy is to increase the
# number of channels that can carry information between blocks. Unfortunately,
# the latter comes with a quadratic penalty since the computational cost of
# ingesting c_i channels and emitting c_o channels is O(c_i * c_o).
# We can take some inspiration from the Inception block which has information
# flowing through the block in separate groups. Applying the idea of multiple
# independent groups to the Resnet block led to the design of ResNeXt.
# Different from the smorgasbord of the transformations in Inception, ResNeXt
# adopts the same transformation in all branches, thus minizing the need for
# manual tuning of each branch.
# The only challenge in this design is that no information is exchanged between
# the g groups. The ResNeXt block amends this in two ways:
#  1. The grouped convolution with a 3x3 kernel is sandwiched in between two
#     1x1 convolutions.
#  2. The second one serves double duty in changing the number of channels back.


class ResNeXtBlock(nn.Module):
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 cardinality=32,
                 base_width=4) -> None:
        super(ResNeXtBlock, self).__init__()

        self.cardinality = cardinality
        width = int(out_channels * (base_width / 64.)) * cardinality

        self.conv1 = nn.LazyConv2d(width, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.LazyConv2d(width, kernel_size=3, stride=stride, padding=1, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.LazyConv2d(out_channels * self.expansion, kernel_size=1, stride=1)
        self.bn3= nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.LazyConv2d(out_channels * self.expansion, kernel_size=1, stride=stride),
                nn.LazyBatchNorm2d())

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        y = y + self.shortcut(x)
        return self.relu(y)


class ResNeXt(nn.Module):
    def __init__(self, block, layers, num_classes=1000, cardinality=32, base_width=4) -> None:
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.base_width = base_width
        self.in_channels = 64

        layer0 = nn.Sequential(
            nn.LazyConv2d(self.in_channels, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        layer1 = self.resnext_block(block, 64, layers[0])
        layer2 = self.resnext_block(block, 128, layers[1], stride=2)
        layer3 = self.resnext_block(block, 256, layers[2], stride=2)
        layer4 = self.resnext_block(block, 512, layers[3], stride=2)

        self.net = nn.Sequential(
            layer0, layer1, layer2, layer3, layer4,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LazyLinear(num_classes))

    def forward(self, x):
        return self.net(x)

    def resnext_block(self, block, out_channels, num_residuals, stride=1):
        layers = [
            block(self.in_channels, out_channels, stride, self.cardinality, self.base_width)
        ]

        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_residuals):
            layers.append(
                block(self.in_channels, out_channels, 1, self.cardinality, self.base_width)
            )

        return nn.Sequential(*layers)


def accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    if y_hat.ndim > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)

    cmp = y_hat.type(y.dtype) == y
    return float(cmp.to(y.dtype).sum().item())


def evaluate(model: nn.Module,
             data_iter: data.DataLoader,
             device: torch.device = torch.device('cpu')) -> float:
    model.eval()

    metric = Accumulator(2)

    with torch.no_grad():
        for x, y in data_iter:
            if not x.device == device or not y.device == device:
                x, y = x.to(device), y.to(device)
            metric.add(accuracy(model(x), y), torch.numel(y))

    return metric[0] / metric[1]


def load_fashion_mnist(batch_size: int, resize=None) -> (data.DataLoader, data.DataLoader):
    trans = [transforms.ToTensor()]

    if resize is not None:
        trans.insert(0, transforms.Resize(resize))

    trans = transforms.Compose(trans)

    mnist_data = torchvision.datasets.FashionMNIST(
        root='../data', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root='../data', train=False, transform=trans, download=True)

    return (data.DataLoader(mnist_data, batch_size, shuffle=True, num_workers=4),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4))


def init_model(model: nn.Module):
    if isinstance(model, nn.LazyConv2d) or isinstance(model, nn.LazyLinear) or\
            isinstance(model, nn.Conv2d) or isinstance(model, nn.Linear):
        torch.nn.init.xavier_uniform_(model.weight)

    if hasattr(model, 'bias') and model.bias is not None:
        torch.nn.init.zeros_(model.bias)


def train(model: nn.Module,
          data_iter: data.DataLoader, test_iter: data.DataLoader,
          lr: float, num_epochs: int,
          device: torch.device = torch.device('cpu')) -> None:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(data_iter):
            model.train()
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f'epoch {epoch+1}, ',
                      f'iter [{i+1}/{len(data_iter)}], ',
                      f'loss {loss.item():.4f}')

        print(f'epoch {epoch+1}, ',
              f'training acc {evaluate(model, data_iter, device):.4f}, '
              f'test acc {evaluate(model, test_iter, device):.4f}')


class IntegrationTest(unittest.TestCase):
    def test_residual_block(self):
        blk = Residual(3, 3)
        x = torch.randn(size=(4, 3, 6, 6))
        y = blk(x)

        print(y.shape)

        self.assertEqual(torch.Size([4, 3, 6, 6]), y.shape)

        blk = Residual(3, 6, stride=2)
        y = blk(x)
        print(y.shape)
        self.assertEqual(torch.Size([4, 6, 3, 3]), y.shape)

    def test_resnet_shape(self):
        model = ResNet()

        act_output = None
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            torchinfo.summary(model, input_size=(4, 3, 224, 224))
            act_output = mock_stdout.getvalue().strip()

        print(act_output)

        expect_output = """
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
ResNet                                   [4, 10]                   --
├─Sequential: 1-1                        [4, 10]                   --
│    └─Sequential: 2-1                   [4, 64, 56, 56]           --
│    │    └─Conv2d: 3-1                  [4, 64, 112, 112]         9,472
│    │    └─BatchNorm2d: 3-2             [4, 64, 112, 112]         128
│    │    └─ReLU: 3-3                    [4, 64, 112, 112]         --
│    │    └─MaxPool2d: 3-4               [4, 64, 56, 56]           --
│    └─Sequential: 2-2                   [4, 64, 56, 56]           --
│    │    └─Residual: 3-5                [4, 64, 56, 56]           74,112
│    │    └─Residual: 3-6                [4, 64, 56, 56]           74,112
│    └─Sequential: 2-3                   [4, 128, 28, 28]          --
│    │    └─Residual: 3-7                [4, 128, 28, 28]          230,272
│    │    └─Residual: 3-8                [4, 128, 28, 28]          295,680
│    └─Sequential: 2-4                   [4, 256, 14, 14]          --
│    │    └─Residual: 3-9                [4, 256, 14, 14]          919,296
│    │    └─Residual: 3-10               [4, 256, 14, 14]          1,181,184
│    └─Sequential: 2-5                   [4, 512, 7, 7]            --
│    │    └─Residual: 3-11               [4, 512, 7, 7]            3,673,600
│    │    └─Residual: 3-12               [4, 512, 7, 7]            4,721,664
│    └─AdaptiveAvgPool2d: 2-6            [4, 512, 1, 1]            --
│    └─Flatten: 2-7                      [4, 512]                  --
│    └─Linear: 2-8                       [4, 10]                   5,130
==========================================================================================
Total params: 11,184,650
Trainable params: 11,184,650
Non-trainable params: 0
Total mult-adds (G): 7.26
==========================================================================================
Input size (MB): 2.41
Forward/backward pass size (MB): 153.34
Params size (MB): 44.74
Estimated Total Size (MB): 200.49
==========================================================================================
        """

        self.assertEqual(expect_output.strip(), act_output)

    def test_resnet(self):
        # hyperparameters
        batch_size = 256
        learning_rate = 0.05
        num_epochs = 10

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        resnet18 = ResNet().to(device)

        # Dummy initialization for LazyConv2d and LazyLinear
        _ = resnet18(torch.randn(size=(1, 1, 96, 96)).to(device))

        data_iter, test_iter = load_fashion_mnist(batch_size, resize=96)

        train(resnet18, data_iter, test_iter, learning_rate, num_epochs, device)

        self.assertTrue(True)

    def test_resnext18(self):
        # hyperparameters
        batch_size, learning_rate, num_epochs = 128, 0.01, 10

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        num_classes = 10
        # resnext50 = ResNeXt(ResNeXtBlock, [3, 4, 6, 3], num_classes=num_classes).to(device)
        resnext18 = ResNeXt(ResNeXtBlock, [2, 2, 2, 2], num_classes=num_classes).to(device)

        print(resnext18)

        data_iter, test_iter = load_fashion_mnist(batch_size, resize=96)

        train(resnext18, data_iter, test_iter, learning_rate, num_epochs, device)

        self.assertTrue(True)


if __name__ == '__main__':
    pass
