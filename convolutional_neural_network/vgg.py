#! coding: utf-8


import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import transforms


class VggBlock(nn.Module):
    def __init__(self, out_channels, num_convs):
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
    def __init__(self):
        super(Vgg, self).__init__()
        conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

        layers = []

        for num_convs, out_channels in conv_arch:
            layers.append(VggBlock(out_channels, num_convs))

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
    net = Vgg()
    net.to(device)

    # Dummy initialization for LazyConv2d
    _ = net(torch.rand(1, 1, 224, 224))

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

    def test_vgg_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        train(device)

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main(verbosity=True)
    pass
