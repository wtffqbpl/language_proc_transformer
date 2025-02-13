#! coding: utf-8

import unittest
from unittest.mock import patch
from io import StringIO
import torch
import torch.nn as nn
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import torchinfo
from utils.accumulator import Accumulator
import utils.dlf as dlf


class InceptionBlock(nn.Module):
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(InceptionBlock, self).__init__()

        # Path 1: Single 1x1 convolution layer
        self.p1 = nn.Sequential(nn.LazyConv2d(c1, kernel_size=1), nn.ReLU())

        # Path 2: 1x1 convolution layer and 3x3 convolution layer
        self.p2 = nn.Sequential(nn.LazyConv2d(c2[0], kernel_size=1),
                                nn.LazyConv2d(c2[1], kernel_size=3, padding=1),
                                nn.ReLU())

        # Path 3: 1x1 convolution layer and 5x5 convolution layer
        self.p3 = nn.Sequential(nn.LazyConv2d(c3[0], kernel_size=1),
                                nn.LazyConv2d(c3[1], kernel_size=5, padding=2),
                                nn.ReLU())

        # Path 4: The 3x3 MaxPooling layer and 1x1 convolution layer
        self.p4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                nn.LazyConv2d(c4, kernel_size=1),
                                nn.ReLU())

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        p4 = self.p4(x)

        # Concat four parallel layers on the channel dimension
        return torch.cat([p1, p2, p3, p4], dim=1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()

        self.net = nn.Sequential(
            # Module 1: 64 channels & 7x7 kernel size convolution layer
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Module 2:
            nn.LazyConv2d(64, kernel_size=1), nn.ReLU(),
            nn.LazyConv2d(192, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Module 3: Two Inception layers
            InceptionBlock(64, (96, 128), (16, 32), 32),
            InceptionBlock(128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Module 4: Five Inception layers
            InceptionBlock(192, (96, 208), (16, 48), 64),
            InceptionBlock(160, (112, 224), (24, 64), 64),
            InceptionBlock(128, (128, 256), (24, 64), 64),
            InceptionBlock(112, (144, 288), (32, 64), 64),
            InceptionBlock(256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Module 5: Two Inception layers
            InceptionBlock(256, (160, 320), (32, 128), 128),
            InceptionBlock(384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LazyLinear(10)
        )

    def forward(self, x):
        return self.net(x)


class InceptionBlockBN(nn.Module):
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(InceptionBlockBN, self).__init__()

        # Path 1: Single 1x1 convolution layer
        self.p1 = nn.Sequential(nn.LazyConv2d(c1, kernel_size=1),
                                nn.BatchNorm2d(c1),
                                nn.ReLU())

        # Path 2: 1x1 convolution layer and 3x3 convolution layer
        self.p2 = nn.Sequential(nn.LazyConv2d(c2[0], kernel_size=1),
                                nn.LazyConv2d(c2[1], kernel_size=3, padding=1),
                                nn.BatchNorm2d(c2[1]),
                                nn.ReLU())

        # Path 3: 1x1 convolution layer and 5x5 convolution layer
        self.p3 = nn.Sequential(nn.LazyConv2d(c3[0], kernel_size=1),
                                nn.LazyConv2d(c3[1], kernel_size=5, padding=2),
                                nn.BatchNorm2d(c3[1]),
                                nn.ReLU())

        # Path 4: The 3x3 MaxPooling layer and 1x1 convolution layer
        self.p4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                nn.LazyConv2d(c4, kernel_size=1),
                                nn.BatchNorm2d(c4),
                                nn.ReLU())

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        p4 = self.p4(x)

        # Concat four parallel layers on the channel dimension
        return torch.cat([p1, p2, p3, p4], dim=1)


class GoogLeNetBN(nn.Module):
    def __init__(self):
        super(GoogLeNetBN, self).__init__()

        self.net = nn.Sequential(
            # Module 1: 64 channels & 7x7 kernel size convolution layer
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Module 2:
            nn.LazyConv2d(64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.LazyConv2d(192, kernel_size=3, padding=1), nn.BatchNorm2d(192), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Module 3: Two Inception layers
            InceptionBlockBN(64, (96, 128), (16, 32), 32),
            InceptionBlockBN(128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Module 4: Five Inception layers
            InceptionBlockBN(192, (96, 208), (16, 48), 64),
            InceptionBlockBN(160, (112, 224), (24, 64), 64),
            InceptionBlockBN(128, (128, 256), (24, 64), 64),
            InceptionBlockBN(112, (144, 288), (32, 64), 64),
            InceptionBlockBN(256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Module 5: Two Inception layers
            InceptionBlockBN(256, (160, 320), (32, 128), 128),
            InceptionBlockBN(384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LazyLinear(10)
        )

    def forward(self, x):
        return self.net(x)


def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]

    if resize is not None:
        trans.insert(0, transforms.Resize(size=resize))
    trans = transforms.Compose(trans)

    mnist_data = torchvision.datasets.FashionMNIST(
        root='../data', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root='../data', train=False, transform=trans, download=True)

    return (data.DataLoader(mnist_data, batch_size, shuffle=True, num_workers=4),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4))


def accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    if (len(y_hat.shape) > 1) and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)

    cmp = y_hat.type(y.dtype) == y
    return float(cmp.to(y.dtype).sum())


def evaluate(model, data_iter):
    if isinstance(model, torch.nn.Module):
        model.eval()

    metric = Accumulator(2)
    with torch.no_grad():
        for x, y in data_iter:
            metric.add(accuracy(model(x), y), torch.numel(y))

    return metric[0] / metric[1]


def init_model(model: torch.nn.Module):
    if isinstance(model, torch.nn.LazyConv2d) or isinstance(model, torch.nn.Conv2d) or\
            isinstance(model, torch.nn.LazyLinear) or isinstance(model, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(model.weight)

    if hasattr(model, 'bias') and model.bias is not None:
        torch.nn.init.zeros_(model.bias)


def train(model: nn.Module,
          data_iter: data.DataLoader, test_iter: data.DataLoader,
          num_epochs: int, lr: float,
          device: torch.device=torch.device('cpu')):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

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

            if i % 100 == 0:
                print(f"Epoch: [{epoch+1}/{num_epochs}], ",
                      f"step: [{i+1}/{len(data_iter)}], ",
                      f"loss: {loss.item():.4f}")

        print(f"Epoch: [{epoch+1}/{num_epochs}], ",
              f"loss: {loss.item():.4f}, ",
              f"training acc: {evaluate(model, data_iter):.4f}",
              f"test acc: {evaluate(model, test_iter):.4f}")


class IntegrationTest(unittest.TestCase):
    def test_googlenet_shape(self):
        model = GoogLeNet()

        act_output = None
        with patch('sys.stdout', new_callable=StringIO) as log:
            torchinfo.summary(model, input_size=(1, 1, 96, 96))
            act_output = log.getvalue().strip()

        print(act_output)

        expected_output = """
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
GoogLeNet                                [1, 10]                   --
├─Sequential: 1-1                        [1, 10]                   --
│    └─Conv2d: 2-1                       [1, 64, 48, 48]           3,200
│    └─ReLU: 2-2                         [1, 64, 48, 48]           --
│    └─MaxPool2d: 2-3                    [1, 64, 24, 24]           --
│    └─Conv2d: 2-4                       [1, 64, 24, 24]           4,160
│    └─ReLU: 2-5                         [1, 64, 24, 24]           --
│    └─Conv2d: 2-6                       [1, 192, 24, 24]          110,784
│    └─ReLU: 2-7                         [1, 192, 24, 24]          --
│    └─MaxPool2d: 2-8                    [1, 192, 12, 12]          --
│    └─InceptionBlock: 2-9               [1, 256, 12, 12]          --
│    │    └─Sequential: 3-1              [1, 64, 12, 12]           12,352
│    │    └─Sequential: 3-2              [1, 128, 12, 12]          129,248
│    │    └─Sequential: 3-3              [1, 32, 12, 12]           15,920
│    │    └─Sequential: 3-4              [1, 32, 12, 12]           6,176
│    └─InceptionBlock: 2-10              [1, 480, 12, 12]          --
│    │    └─Sequential: 3-5              [1, 128, 12, 12]          32,896
│    │    └─Sequential: 3-6              [1, 192, 12, 12]          254,272
│    │    └─Sequential: 3-7              [1, 96, 12, 12]           85,120
│    │    └─Sequential: 3-8              [1, 64, 12, 12]           16,448
│    └─MaxPool2d: 2-11                   [1, 480, 6, 6]            --
│    └─InceptionBlock: 2-12              [1, 512, 6, 6]            --
│    │    └─Sequential: 3-9              [1, 192, 6, 6]            92,352
│    │    └─Sequential: 3-10             [1, 208, 6, 6]            226,096
│    │    └─Sequential: 3-11             [1, 48, 6, 6]             26,944
│    │    └─Sequential: 3-12             [1, 64, 6, 6]             30,784
│    └─InceptionBlock: 2-13              [1, 512, 6, 6]            --
│    │    └─Sequential: 3-13             [1, 160, 6, 6]            82,080
│    │    └─Sequential: 3-14             [1, 224, 6, 6]            283,472
│    │    └─Sequential: 3-15             [1, 64, 6, 6]             50,776
│    │    └─Sequential: 3-16             [1, 64, 6, 6]             32,832
│    └─InceptionBlock: 2-14              [1, 512, 6, 6]            --
│    │    └─Sequential: 3-17             [1, 128, 6, 6]            65,664
│    │    └─Sequential: 3-18             [1, 256, 6, 6]            360,832
│    │    └─Sequential: 3-19             [1, 64, 6, 6]             50,776
│    │    └─Sequential: 3-20             [1, 64, 6, 6]             32,832
│    └─InceptionBlock: 2-15              [1, 528, 6, 6]            --
│    │    └─Sequential: 3-21             [1, 112, 6, 6]            57,456
│    │    └─Sequential: 3-22             [1, 288, 6, 6]            447,408
│    │    └─Sequential: 3-23             [1, 64, 6, 6]             67,680
│    │    └─Sequential: 3-24             [1, 64, 6, 6]             32,832
│    └─InceptionBlock: 2-16              [1, 832, 6, 6]            --
│    │    └─Sequential: 3-25             [1, 256, 6, 6]            135,424
│    │    └─Sequential: 3-26             [1, 320, 6, 6]            545,760
│    │    └─Sequential: 3-27             [1, 128, 6, 6]            119,456
│    │    └─Sequential: 3-28             [1, 128, 6, 6]            67,712
│    └─MaxPool2d: 2-17                   [1, 832, 3, 3]            --
│    └─InceptionBlock: 2-18              [1, 832, 3, 3]            --
│    │    └─Sequential: 3-29             [1, 256, 3, 3]            213,248
│    │    └─Sequential: 3-30             [1, 320, 3, 3]            594,400
│    │    └─Sequential: 3-31             [1, 128, 3, 3]            129,184
│    │    └─Sequential: 3-32             [1, 128, 3, 3]            106,624
│    └─InceptionBlock: 2-19              [1, 1024, 3, 3]           --
│    │    └─Sequential: 3-33             [1, 384, 3, 3]            319,872
│    │    └─Sequential: 3-34             [1, 384, 3, 3]            823,872
│    │    └─Sequential: 3-35             [1, 128, 3, 3]            193,712
│    │    └─Sequential: 3-36             [1, 128, 3, 3]            106,624
│    └─AdaptiveAvgPool2d: 2-20           [1, 1024, 1, 1]           --
│    └─Flatten: 2-21                     [1, 1024]                 --
│    └─Linear: 2-22                      [1, 10]                   10,250
==========================================================================================
Total params: 5,977,530
Trainable params: 5,977,530
Non-trainable params: 0
Total mult-adds (M): 276.66
==========================================================================================
Input size (MB): 0.04
Forward/backward pass size (MB): 4.74
Params size (MB): 23.91
Estimated Total Size (MB): 28.69
==========================================================================================
        """

        self.assertEqual(expected_output.strip(), act_output)

    def test_googlenet_model(self):
        # hyperparameters
        batch_size = 128
        learning_rate = 0.01
        num_epochs = 10

        device = dlf.devices()[0]

        data_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)

        model = GoogLeNet()
        model.to(device=device)

        # Dummy initialization for LazyConv2d
        _ = model(torch.randn((1, 1, 96, 96)).to(device))

        # Apply initialization method to the model
        model.apply(init_model)

        train(model, data_iter, test_iter, num_epochs, learning_rate, device)

        self.assertTrue(True)

    def test_googlenet_with_batch_normalization(self):
        # hyperparameters
        batch_size = 128
        learning_rate = 0.5
        num_epochs = 10

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        data_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)

        model = GoogLeNetBN()
        model.to(device=device)

        # Dummy initialization method to the model
        model.apply(init_model)

        train(model, data_iter, test_iter, num_epochs, learning_rate, device)

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main(verbosity=True)
