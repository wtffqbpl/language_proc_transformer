#! coding: utf-8

import unittest
from unittest.mock import patch
from io import StringIO
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.accumulator import Accumulator


# ResNet significantly changed the view of how to parameterize the function in deep
# networks. DenseNet (dense convolutional network) is to some extent the logical
# extensions of this. DenseNet is characterized by both the connectivity pattern
# where each layer connects to all the preceding layers and the concatenation
# operation (rather than the addition operator in ResNet) to preserve and reuse
# features from earlier layers. To understand how to arrive at it.
# Recall the Taylor expansion for functions. The key point is that it decomposes
# a function into terms of increasingly higher order. In a similar vein, ResNet
# decomposes functions into a simple linear term and a more complex nonlinear
# one.
# The key difference between ResNet and DenseNet is that in the latter case
# outputs are concatenated rather than added. As a result, we perform a mapping
# from x to its values after applying an increasingly complex sequence.
# in the end, all these functions are combined in MLP to reduce the number of
# features again. In terms of implementation, this is quite simple: rather than
# adding terms, we concatenate them.
# The main components that comprise a DenseNet are dense blocks and transition
# layers. The former define how the inputs and outputs are concatenated, while
# the latter control the number of channels so that it is not too large.
#
# DenseNet uses the modified "batch normalization, activation, and convolution"
# structure of ResNet.


class DenseBlock(nn.Module):

    def __init__(self, num_convs, out_channels):
        super(DenseBlock, self).__init__()

        layers = []
        for i in range(num_convs):
            layers.append(self.conv_block(out_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        for blk in self.net:
            y = blk(x)
            # Concatenate input and output of each block along the channel axis.
            x = torch.cat((x, y), dim=1)
        return x

    @staticmethod
    def conv_block(num_channels):
        return nn.Sequential(
            nn.LazyBatchNorm2d(), nn.ReLU(inplace=True),
            nn.LazyConv2d(out_channels=num_channels, kernel_size=3, padding=1))


class TransitionBlock(nn.Module):
    def __init__(self, num_channels):
        super(TransitionBlock, self).__init__()

        self.net = self.transition_block(num_channels)

    def forward(self, x):
        return self.net(x)
    
    @staticmethod
    def transition_block(num_channels):
        return nn.Sequential(
            nn.LazyBatchNorm2d(), nn.ReLU(inplace=True),
            nn.LazyConv2d(out_channels=num_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    def __init__(self, num_channels, growth_rate, num_convs_in_dense_blocks):
        super(DenseNet, self).__init__()

        self.net = self.dense_net(num_channels, growth_rate, num_convs_in_dense_blocks)

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def dense_net(num_channels, growth_rate, num_convs_in_dense_blocks):
        net = nn.Sequential(
            nn.LazyConv2d(out_channels=num_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_channels), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            net.add_module(
                f"DenseBlock_{i}", DenseBlock(num_convs, growth_rate))
            # This is the number of output channels in the previous dense block
            num_channels += num_convs * growth_rate
            # A transition layer that haves the number of channels is added between the dense blocks
            if i != len(num_convs_in_dense_blocks) - 1:
                num_channels //= 2
                net.add_module(f"TransitionBlock_{i}", TransitionBlock(num_channels))
        net.add_module("BN", nn.BatchNorm2d(num_channels))
        net.add_module("relu", nn.ReLU(inplace=True))
        net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1, 1)))
        net.add_module("fc", nn.Sequential(nn.Flatten(), nn.Linear(num_channels, 10)))

        return net


def accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
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
            metric.add(accuracy(model(x), y), y.numel())

    return metric[0] / metric[1]


def init_model(model: nn.Module):
    if isinstance(model, nn.LazyConv2d) or isinstance(model, nn.Conv2d) or \
        isinstance(model, nn.LazyLinear) or isinstance(model, nn.Linear):
        nn.init.xavier_uniform_(model.weight)

    if hasattr(model, 'bias') and model.bias is not None:
        nn.init.zeros_(model.bias)


def load_fashion_data(batch_size: int, resize=None) -> (data.DataLoader, data.DataLoader):
    trans = [transforms.ToTensor()]

    if resize is not None:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)

    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4))


def train(model: nn.Module,
          data_iter: data.DataLoader, test_iter: data.DataLoader,
          lr: float, num_epochs: int, device: torch.device = torch.device('cpu')):

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
              f'train acc {evaluate(model, data_iter, device):.4f}, ',
              f'test acc {evaluate(model, test_iter, device):.4f}')


class IntegrationTest(unittest.TestCase):
    def test_dense_shape(self):
        blk = DenseBlock(2, 10)
        x = torch.randn(size=(4, 3, 8, 8))
        y = blk(x)

        self.assertEqual(torch.Size([4, 23, 8, 8]), y.shape)

    def test_transition_shape(self):
        x = torch.randn(size=(4, 3, 8, 8))
        dense_blk = DenseBlock(2, 10)
        transition_blk = TransitionBlock.transition_block(10)
        y = transition_blk(dense_blk(x))

        self.assertEqual(torch.Size([4, 10, 4, 4]), y.shape)

    def test_densenet(self):
        # hyperparameters
        num_channels, growth_rate = 64, 32
        num_convs_in_dense_blocks = [4, 4, 4, 4]
        batch_size, learning_rate, num_epochs = 256, 0.1, 10

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = DenseNet(num_channels, growth_rate, num_convs_in_dense_blocks).to(device)

        # Dummy initialization
        _ = model(torch.zeros((1, 1, 96, 96)).to(device))

        # Apply initialization function
        model.apply(init_model)

        print(model)

        # Load the Fashion-MNIST dataset
        data_iter, test_iter = load_fashion_data(batch_size, resize=96)

        train(model, data_iter, test_iter, learning_rate, num_epochs, device)

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main(verbosity=True)

