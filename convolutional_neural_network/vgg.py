#! coding: utf-8

import unittest
from unittest.mock import patch
import sys
from io import StringIO
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchinfo
import torchvision
from torchvision import transforms
import torchsummary


class VggBlock(nn.Module):
    def __init__(self, num_convs, out_channels):
        super(VggBlock, self).__init__()

        layers = []
        for _ in range(num_convs):
            layers.append(nn.LazyConv2d(out_channels=out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Vgg(nn.Module):
    def __init__(self, ratio=1):
        super(Vgg, self).__init__()
        conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

        layers = []

        for num_convs, out_channels in conv_arch:
            layers.append(VggBlock(num_convs, out_channels // ratio))

        self.net = nn.Sequential(*layers, nn.Flatten(),
                                 nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.5),
                                 nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.5),
                                 nn.LazyLinear(10))

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


def init_model(m):
    # initialize weights
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)

    # initialize biases
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.zeros_(m.bias)


def train(device=torch.device('cpu')):
    # hyperparameters
    batch_size = 64
    learning_rate = 0.05
    num_epochs = 10

    # define model
    net = Vgg(4)  # set ratio to 4 since fashion mnist images are small
    net.to(device)

    # Dummy initialization for LazyConv2d
    _ = net(torch.rand(1, 1, 224, 224).to(device))

    # apply initialization method to the model
    net.apply(init_model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # load data and resize fashion mnist images to 224x224
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)

    # training
    for epoch in range(num_epochs):
        net.train()

        for i, (x, y) in enumerate(train_iter):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            y_hat = net(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f'epoch {epoch+1}, iteration {i}, loss {loss.item()}')

        # evaluation
        net.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for x, y in test_iter:
                x, y = x.to(device), y.to(device)
                y_hat = net(x)
                _, predicted = torch.max(y_hat, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        print(f'epoch {epoch+1}, accuracy {correct/total:.2f}')


class IntegrationTest(unittest.TestCase):
    def test_vgg_arch(self):
        model = Vgg()
        x = torch.rand(1, 1, 224, 224)

        for layer in model.net:
            x = layer(x)
            print(layer.__class__.__name__, 'output shape:\t', x.shape)

        self.assertEqual(x.shape, torch.Size([1, 10]))

    def test_vgg_arch_with_summary(self):
        model = Vgg()

        # Testing
        output = None
        with patch('sys.stdout', new_callable=StringIO) as log:
            torchsummary.summary(model, (1, 224, 224))
            output = log.getvalue().strip()

        print(output)

        expected_output = """
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 224, 224]             640
              ReLU-2         [-1, 64, 224, 224]               0
         MaxPool2d-3         [-1, 64, 112, 112]               0
          VggBlock-4         [-1, 64, 112, 112]               0
            Conv2d-5        [-1, 128, 112, 112]          73,856
              ReLU-6        [-1, 128, 112, 112]               0
         MaxPool2d-7          [-1, 128, 56, 56]               0
          VggBlock-8          [-1, 128, 56, 56]               0
            Conv2d-9          [-1, 256, 56, 56]         295,168
             ReLU-10          [-1, 256, 56, 56]               0
           Conv2d-11          [-1, 256, 56, 56]         590,080
             ReLU-12          [-1, 256, 56, 56]               0
        MaxPool2d-13          [-1, 256, 28, 28]               0
         VggBlock-14          [-1, 256, 28, 28]               0
           Conv2d-15          [-1, 512, 28, 28]       1,180,160
             ReLU-16          [-1, 512, 28, 28]               0
           Conv2d-17          [-1, 512, 28, 28]       2,359,808
             ReLU-18          [-1, 512, 28, 28]               0
        MaxPool2d-19          [-1, 512, 14, 14]               0
         VggBlock-20          [-1, 512, 14, 14]               0
           Conv2d-21          [-1, 512, 14, 14]       2,359,808
             ReLU-22          [-1, 512, 14, 14]               0
           Conv2d-23          [-1, 512, 14, 14]       2,359,808
             ReLU-24          [-1, 512, 14, 14]               0
        MaxPool2d-25            [-1, 512, 7, 7]               0
         VggBlock-26            [-1, 512, 7, 7]               0
          Flatten-27                [-1, 25088]               0
           Linear-28                 [-1, 4096]     102,764,544
             ReLU-29                 [-1, 4096]               0
          Dropout-30                 [-1, 4096]               0
           Linear-31                 [-1, 4096]      16,781,312
             ReLU-32                 [-1, 4096]               0
          Dropout-33                 [-1, 4096]               0
           Linear-34                   [-1, 10]          40,970
================================================================
Total params: 128,806,154
Trainable params: 128,806,154
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.19
Forward/backward pass size (MB): 137.04
Params size (MB): 491.36
Estimated Total Size (MB): 628.59
----------------------------------------------------------------
        """
        self.assertEqual(expected_output.strip(), output)

    def test_vgg_arch_with_torchinfo(self):
        model = Vgg()
        torchinfo.summary(model, (1, 1, 224, 224))
        self.assertTrue(True)

    def test_vgg_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        train(device)

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main(verbosity=True)
    pass
