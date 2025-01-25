#! coding: utf-8

import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import transforms


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

        # self.net.apply(init_weights)

    def forward(self, x):
        return self.net(x)

    def get(self):
        return self.net


def train(device):
    # hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10

    net = AlexNet()
    net.to(device)

    # To initialize the network.
    x = torch.randn(1, 1, 224, 224).to(device)
    _ = net(x)

    net.get().apply(init_weights)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # Load data and resize to fashion-mnist images to 224 * 224
    train_iter, test_iter = load_fashion_mnist(batch_size, resize=224)

    # Training
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
                print(f'Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}')

        # Evaluation
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_iter:
                x, y = x.to(device), y.to(device)
                y_hat = net(x)
                _, predicted = torch.max(y_hat, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        print(f'Epoch: {epoch}, Accuracy: {correct / total}')


class IntegrationTest(unittest.TestCase):
    def test_alexnet_output_shape(self):
        net = AlexNet()

        # print net architecture
        x = torch.randn(1, 3, 224, 224)
        for layer in net.get():
            x = layer(x)
            print(layer.__class__.__name__, 'output shape:\t', x.shape)

        y = net(torch.randn(1, 3, 224, 224))
        self.assertEqual(y.shape, torch.Size([1, 10]))

    def test_alexnet(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train(device)
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main(verbosity=True)
    pass
