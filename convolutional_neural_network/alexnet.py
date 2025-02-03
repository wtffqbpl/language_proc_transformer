#! coding: utf-8

import unittest
from unittest.mock import patch
from io import StringIO
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import transforms
import torchinfo


def load_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]

    if resize:
        trans.insert(0, transforms.Resize(size=resize))
    trans = transforms.Compose(trans)

    mnist_train = torchvision.datasets.FashionMNIST(
        root='../data', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root='../data', train=False, transform=trans, download=True)

    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4))


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.LazyLinear or \
            type(m) == nn.Conv2d or type(m) == nn.LazyConv2d:
        nn.init.xavier_uniform_(m.weight)


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.net = nn.Sequential(
            nn.LazyConv2d(out_channels=96, kernel_size=11, stride=4, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(out_channels=256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(out_channels=384, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(out_channels=384, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(out_channels=256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.LazyLinear(4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.LazyLinear(4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.LazyLinear(10)
        )

    def forward(self, x):
        return self.net(x)

    def get(self):
        return self.net


class AlexNetBN(nn.Module):
    def __init__(self):
        super(AlexNetBN, self).__init__()

        self.net = nn.Sequential(
            nn.LazyConv2d(out_channels=96, kernel_size=11, stride=4, padding=2), nn.BatchNorm2d(96), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(out_channels=256, kernel_size=5, padding=2), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(out_channels=384, kernel_size=3, padding=1), nn.BatchNorm2d(384), nn.ReLU(),
            nn.LazyConv2d(out_channels=384, kernel_size=3, padding=1), nn.BatchNorm2d(384), nn.ReLU(),
            nn.LazyConv2d(out_channels=256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.LazyLinear(4096), nn.LayerNorm(4096), nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.LazyLinear(4096), nn.LayerNorm(4096), nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.LazyLinear(10)
        )

    def forward(self, x):
        return self.net(x)


def evaluate(model: nn.Module, data_iter: data.DataLoader, device: torch.device) -> float:
    model.eval()

    correct, total = 0, 0

    with torch.no_grad():
        for x, y in data_iter:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            _, predicted = torch.max(y_hat, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    return correct / total


def train(net: nn.Module,
          train_iter: data.DataLoader, test_iter: data.DataLoader,
          lr: float, num_epochs: int,
          device=torch.device('cpu')):

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # Training
    for epoch in range(num_epochs):
        net.train()
        for i, (x, y) in enumerate(train_iter):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            y_hat = net(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            if i == 0 or ((i + 1) % 100 == 0):
                print(f'Epoch: {epoch}, ',
                      f'iteration: [{i+1}/{len(train_iter)}], ',
                      f'loss: {loss.item():.4f}')

        # Evaluation
        print(f'Epoch: {epoch+1}, ',
              f'training accuracy: {evaluate(net, train_iter, device):.4f}',
              f'accuracy: {evaluate(net, test_iter, device):.4f}')


class IntegrationTest(unittest.TestCase):
    def test_alexnet_output_shape(self):
        net = AlexNet()

        act_output = None
        with patch('sys.stdout', new_callable=StringIO) as log:
            torchinfo.summary(net, input_size=(1, 3, 224, 224))
            act_output = log.getvalue().strip()
        
        print(act_output)

        expect_output = """
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
AlexNet                                  [1, 10]                   --
├─Sequential: 1-1                        [1, 10]                   --
│    └─Conv2d: 2-1                       [1, 96, 55, 55]           34,944
│    └─ReLU: 2-2                         [1, 96, 55, 55]           --
│    └─MaxPool2d: 2-3                    [1, 96, 27, 27]           --
│    └─Conv2d: 2-4                       [1, 256, 27, 27]          614,656
│    └─ReLU: 2-5                         [1, 256, 27, 27]          --
│    └─MaxPool2d: 2-6                    [1, 256, 13, 13]          --
│    └─Conv2d: 2-7                       [1, 384, 13, 13]          885,120
│    └─ReLU: 2-8                         [1, 384, 13, 13]          --
│    └─Conv2d: 2-9                       [1, 384, 13, 13]          1,327,488
│    └─ReLU: 2-10                        [1, 384, 13, 13]          --
│    └─Conv2d: 2-11                      [1, 256, 13, 13]          884,992
│    └─ReLU: 2-12                        [1, 256, 13, 13]          --
│    └─MaxPool2d: 2-13                   [1, 256, 6, 6]            --
│    └─Flatten: 2-14                     [1, 9216]                 --
│    └─Linear: 2-15                      [1, 4096]                 37,752,832
│    └─ReLU: 2-16                        [1, 4096]                 --
│    └─Dropout: 2-17                     [1, 4096]                 --
│    └─Linear: 2-18                      [1, 4096]                 16,781,312
│    └─ReLU: 2-19                        [1, 4096]                 --
│    └─Dropout: 2-20                     [1, 4096]                 --
│    └─Linear: 2-21                      [1, 10]                   40,970
==========================================================================================
Total params: 58,322,314
Trainable params: 58,322,314
Non-trainable params: 0
Total mult-adds (G): 1.13
==========================================================================================
Input size (MB): 0.60
Forward/backward pass size (MB): 5.27
Params size (MB): 233.29
Estimated Total Size (MB): 239.16
==========================================================================================
        """

        # Check output shapes in each layer for the AlexNet model.
        self.assertEqual(expect_output.strip(), act_output)

        # Check the final output shape for the AlexNet model.
        y = net(torch.randn(1, 3, 224, 224))
        self.assertEqual(y.shape, torch.Size([1, 10]))

    def test_alexnet(self):
        # hyperparameters
        batch_size = 64
        learning_rate = 0.001
        num_epochs = 10

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        net = AlexNet()
        net.to(device)

        # Dummy initialization
        _ = net(torch.randn(1, 1, 224, 224).to(device))
        net.apply(init_weights)

        # load data
        train_iter, test_iter = load_fashion_mnist(batch_size, resize=224)

        # training
        train(net, train_iter, test_iter, learning_rate, num_epochs, device)

        self.assertTrue(True)

    def test_alexnetbn_shape(self):
        # define Alexnet (with batch normalization) model.
        model = AlexNetBN()

        act_output = None
        with patch('sys.stdout', new_callable=StringIO) as log:
            torchinfo.summary(model, input_size=(1, 3, 224, 224))
            act_output = log.getvalue().strip()

        print(act_output)

        expect_output = """
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
AlexNetBN                                [1, 10]                   --
├─Sequential: 1-1                        [1, 10]                   --
│    └─Conv2d: 2-1                       [1, 96, 55, 55]           34,944
│    └─BatchNorm2d: 2-2                  [1, 96, 55, 55]           192
│    └─ReLU: 2-3                         [1, 96, 55, 55]           --
│    └─MaxPool2d: 2-4                    [1, 96, 27, 27]           --
│    └─Conv2d: 2-5                       [1, 256, 27, 27]          614,656
│    └─BatchNorm2d: 2-6                  [1, 256, 27, 27]          512
│    └─ReLU: 2-7                         [1, 256, 27, 27]          --
│    └─MaxPool2d: 2-8                    [1, 256, 13, 13]          --
│    └─Conv2d: 2-9                       [1, 384, 13, 13]          885,120
│    └─BatchNorm2d: 2-10                 [1, 384, 13, 13]          768
│    └─ReLU: 2-11                        [1, 384, 13, 13]          --
│    └─Conv2d: 2-12                      [1, 384, 13, 13]          1,327,488
│    └─BatchNorm2d: 2-13                 [1, 384, 13, 13]          768
│    └─ReLU: 2-14                        [1, 384, 13, 13]          --
│    └─Conv2d: 2-15                      [1, 256, 13, 13]          884,992
│    └─BatchNorm2d: 2-16                 [1, 256, 13, 13]          512
│    └─ReLU: 2-17                        [1, 256, 13, 13]          --
│    └─MaxPool2d: 2-18                   [1, 256, 6, 6]            --
│    └─Flatten: 2-19                     [1, 9216]                 --
│    └─Linear: 2-20                      [1, 4096]                 37,752,832
│    └─LayerNorm: 2-21                   [1, 4096]                 8,192
│    └─ReLU: 2-22                        [1, 4096]                 --
│    └─Linear: 2-23                      [1, 4096]                 16,781,312
│    └─LayerNorm: 2-24                   [1, 4096]                 8,192
│    └─ReLU: 2-25                        [1, 4096]                 --
│    └─Linear: 2-26                      [1, 10]                   40,970
==========================================================================================
Total params: 58,341,450
Trainable params: 58,341,450
Non-trainable params: 0
Total mult-adds (G): 1.13
==========================================================================================
Input size (MB): 0.60
Forward/backward pass size (MB): 10.53
Params size (MB): 233.37
Estimated Total Size (MB): 244.50
==========================================================================================
        """

        self.assertEqual(expect_output.strip(), act_output)

    def test_alexnetbn(self):
        # hyperparameters
        batch_size = 64
        learning_rate = 0.05
        num_epochs = 10
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        net = AlexNetBN()
        net.to(device)
        
        # Dummy initialization
        _ = net(torch.randn(1, 1, 224, 224).to(device))

        # Apply initialization function
        net.apply(init_weights)

        # load training and testing data
        train_iter, test_iter = load_fashion_mnist(batch_size, resize=224)

        # training
        train(net, train_iter, test_iter, learning_rate, num_epochs, device)

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main(verbosity=True)
    pass
