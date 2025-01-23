#! coding: utf-8


import unittest
import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision
from torchvision import transforms
from utils.accumulator import Accumulator
from utils.timer import Timer


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


def accuracy(y_hat: torch.Tensor, y: torch.Tensor):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)

    return float((y_hat.type(y.dtype) == y).sum())


def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()

        if not device:
            device = next(iter(net.parameters())).device

    metric = Accumulator(2)
    with torch.no_grad():
        for x, y in data_iter:
            if isinstance(x, list):
                # BERT 微调所需
                x = [x_.to(device) for x_ in x]
            else:
                x = x.to(device)
            y = y.to(device)
            metric.add(accuracy(net(x), y), y.numel())

    return metric[0] / metric[1]


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.net = nn.Sequential(
            # Convolutional Neural Layers
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            # Multilayer Perceptron Layers
            nn.Linear(16 * 5 * 5, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
            # No need to use softmax activation function since we are using CrossEntropyLoss as loss function which
            # includes log-softmax activation function.
        )

    def forward(self, x):
        return self.net(x)


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()

        # Define LeNet model
        self.net = nn.Sequential(
            # Convolutional block
            # output_shape = ((28 + 2 * 2 - (5 - 1) - 1) / 1 + 1) * ((28 + 2 * 2 - (5 - 1) - 1) / 1 + 1) = 28 * 28
            # batch_size = 1, channels = 6
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            # output_shape floor((28 - 2) / 2 + 1) * floor((28 - 2) / 2 + 1) = 14 * 14
            nn.AvgPool2d(kernel_size=2, stride=2),

            # output_shape = floor((14 - (5 - 1) - 1) / 1 + 1) * floor((14 - (5 - 1) - 1) / 1 + 1) = 10 * 10
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Sigmoid(),
            # output_shape = floor((10 - 2) / 2 + 1) * floor((10 - 2) / 2 + 1) = 5 * 5
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.LazyLinear(120), nn.Sigmoid(),
            nn.LazyLinear(84), nn.Sigmoid(),
            nn.LazyLinear(10)
        )

    def forward(self, x):
        pass

    def get(self):
        return self.net


def train(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on ', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    timer = Timer()
    num_batches = len(train_iter)

    for epoch in range(num_epochs):
        metric = Accumulator(3)
        net.train()

        train_l, train_acc = 0., 0.
        for i, (x, y) in enumerate(train_iter):
            timer.start()

            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            y_hat = net(x)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()

            with torch.no_grad():
                metric.add(l * x.shape[0], accuracy(y_hat, y), x.shape[0])

            timer.stop()

            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]

            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                print(f'epoch={(epoch + i + 1) / num_batches}, ',
                      f'training loss: {train_l:.3f}, ',
                      f'training accuracy: {train_acc:.3f}')

        test_acc = evaluate_accuracy_gpu(net, test_iter)
        print(f'training loss: {train_l:.3f}, ',
              f'training accuracy: {train_acc:.3f}, ',
              f'testing accuracy: {test_acc:.3f}')
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples / sec ',
              f'on {str(device)}')


class IntegrationTest(unittest.TestCase):

    def test_lenet_model(self):
        model = LeNet()

        """
        x = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)

        for layer in model.get():
            x_ = layer(x)
            print(layer.__class__.__name__, 'output shape: \t', x_.shape)
        """

        batch_size = 256
        train_iter, test_iter = load_fashion_mnist(batch_size=batch_size)

        lr, num_epochs = 0.9, 10
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train(model.get(), train_iter, test_iter, num_epochs, lr, device)

        self.assertTrue(True)

    def test_lenet_model_standard(self):
        batch_size = 64
        learning_rate = 0.001
        num_epochs = 10

        # Transformations for the training and testing data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Load the MNIST dataset
        train_dataset = torchvision.datasets.FashionMNIST(
            root='../data', train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.FashionMNIST(
            root='../data', train=False, transform=transform, download=True)

        train_loader = data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = data.DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        model = LeNet5()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for i, (images, labels) in enumerate(train_loader):
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # backward pass and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if (i + 1) % 100 == 0:
                    print(f'Epoch [{epoch + 1} / {num_epochs}], ',
                          f'Step [{i + 1} / {len(train_loader)}], ',
                          f'Loss: {loss.item():.4f}')
            print(f'Epoch [{epoch + 1} / {num_epochs}], ',
                  f'Loss: {running_loss / len(train_loader):.4f}')

        # Evaluation loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')

        # Save the model checkpoint
        torch.save(model.state_dict(), 'lenet5_mnist.pth')


if __name__ == '__main__':
    unittest.main(verbosity=2)
    pass
