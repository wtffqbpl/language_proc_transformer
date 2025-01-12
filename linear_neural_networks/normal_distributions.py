#! coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import transforms
import math
import numpy as np
import random
import unittest

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from utils.timer import Timer
from utils.accumulator import Accumulator
from utils.animator import Animator


def normal(x, mu, sigma):
    return 1 / math.sqrt(2 * math.pi * sigma ** 2) * np.exp(-0.5 * (x - mu) ** 2 / sigma ** 2)


def test_print():
    x = np.arange(-7, 7, 0.01)
    params = [(0, 1), (0, 2), (3, 1), ]
    for mu, sigma in params:
        plt.plot(x, normal(x, mu, sigma), label=f'mu={mu}, sigma={sigma}')

    plt.legend()
    plt.show()


class LinearNeuralNetworkNaive:
    def __init__(self):
        pass

    def synthetic_data(self, w, b, num_examples):
        """
        Generate y = Xw + b + noise
        :param w:
        :param b:
        :param num_examples:
        :return:
        """
        X = torch.normal(0, 1, (num_examples, len(w)))
        y = torch.matmul(X, w) + b
        y += torch.normal(0, 0.01, y.shape)
        return X, y.reshape((-1, 1))

    def data_iter(self, batch_size, features, labels):
        num_examples = len(features)
        indices = list(range(num_examples))
        random.shuffle(indices)

        for i in range(0, num_examples, batch_size):
            batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
            yield features[batch_indices], labels[batch_indices]

    def linreg(self, x, w, b):
        return torch.matmul(x, w) + b

    def squared_loss(self, y_hat, y):
        return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

    def sgd(self, params, lr, batch_size):
        with torch.no_grad():
            for param in params:
                param -= lr * param.grad / batch_size
                param.grad.zero_()

    def forward(self):
        # 定义真实值
        true_w = torch.tensor([2, -3.4])
        true_b = 4.2

        # 构造数据集
        features, labels = self.synthetic_data(true_w, true_b, 1000)

        print("features: ", features[0], '\nlabel: ', labels[0])

        # 定义 hyperparameters
        batch_size = 10
        lr = 0.03
        num_epochs = 3
        net = self.linreg
        loss = self.squared_loss

        # 随机初始化输入值
        w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
        b = torch.zeros(1, requires_grad=True)

        # 开始迭代
        for epoch in range(num_epochs):
            for X, y in self.data_iter(batch_size, features, labels):
                # 1. 使用 X, w, b 来计算y^hat, 并计算与真实值y之间的loss
                l = loss(net(X, w, b), y)
                # 利用backward 计算梯度
                l.sum().backward()
                # 使用参数的梯度更新参数
                self.sgd([w, b], lr, batch_size)

            # torch.no_grad() --- Context manager that disables gradient calculation.
            with torch.no_grad():
                train_l = loss(net(features, w, b), labels)
                print(f"epoch {epoch + 1}, loss {float(train_l.mean()):f}")

        print(f'w loss: {true_w - w.reshape(true_w.shape)}')
        print(f'b loss: {true_b - b}')

        return True


class LinearNeuralNetworkSimple:
    def __init__(self):
        self._batch_size = 10
        self._true_w = torch.tensor([2, -3.4])
        self._true_b = 4.2
        self._features, self._labels = self._synthetic_data(self._true_w, self._true_b, 1000)

    def _synthetic_data(self, w, b, num_samples):
        x = torch.normal(0, 1, (num_samples, len(w)))
        y = torch.matmul(x, w) + b
        y += torch.normal(0, 0.01, y.shape)
        return x, y.reshape((-1, 1))

    def _load_array(self, data_array, batch_size, is_train=True):
        """ Construct a PyTorch data iterator """
        dataset = data.TensorDataset(*data_array)
        return data.DataLoader(dataset, batch_size, shuffle=is_train)

    def train(self):
        # Load data
        data_iter = self._load_array((self._features, self._labels), self._batch_size)

        # print(next(iter(data_iter)))

        # Define model
        net = nn.Sequential(nn.Linear(2, 1))
        net[0].weight.data.normal_(0, 0.01)
        net[0].bias.data.fill_(0)

        # Define loss function
        # Creates a criterion that measures the mean squared error (squared L2 norm)
        # between each element in the input x and target y.
        loss = nn.MSELoss()

        # Define optimizer
        # Implements stochastic gradient descent (optionally with momentum).
        trainer = optim.SGD(net.parameters(), lr=0.03)

        # Training
        num_epochs = 3
        for epoch in range(num_epochs):
            for x, y in data_iter:
                l = loss(net(x), y)
                trainer.zero_grad()
                l.backward()
                trainer.step()
            l = loss(net(self._features), self._labels)
            print(f'epoch {epoch + 1}, loss {l:f}')

        w = net[0].weight.data
        print('w loss: ', self._true_w - w.reshape(self._true_w.shape))
        b = net[0].bias.data
        print('bias loss: ', self._true_b - b)

        return True


class MNISTSimple:
    def __init__(self):
        pass

    def test(self):
        trans = transforms.ToTensor()
        self._mnist_train = torchvision.datasets.FashionMNIST(
            root='../data', train=True, transform=trans, download=True)
        self._mnist_test = torchvision.datasets.FashionMNIST(
            root='../data', train=False, transform=trans, download=True)
        print("len(mnist_train): ", len(self._mnist_train))
        print('len(mnist_test): ', len(self._mnist_test))
        print(type(self._mnist_train))
        print(self._mnist_train.classes)

        # Show images
        # def get_fashion_labels(labels):
        #     return [self._mnist_train.classes[int(i)] for i in labels]

        # x, y = next(iter(data.DataLoader(self._mnist_train, batch_size=18)))
        # self.show_images(x.reshape(18, 28, 28), 2, 9, titles=get_fashion_labels(y))

        batch_size = 256

        self._train_iter = data.DataLoader(
            self._mnist_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self._get_dataloader_workers())

        timer = Timer()
        for x, y in self._train_iter:
            continue
        print(f'{timer.stop():.2f} sec')

        train_iter, test_iter = self.data_iter(32, resize=64)
        for x, y in train_iter:
            print(x.shape, x.dtype, y.shape, y.dtype)
            break

        return True

    def _accuracy(self, y_hat: torch.Tensor, y: torch.Tensor) -> float:
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = y_hat.argmax(dim=1)

        cmp = y_hat.type(y.dtype) == y
        return float(cmp.to(y.dtype).sum())

    def evaluate_accuracy(self, net, data_iter):
        if isinstance(net, torch.nn.Module):
            net.eval()

        # 这里的第二个 accumulator 其实是用来统计个数的
        metric = Accumulator(2)
        with torch.no_grad():
            for x, y in data_iter:
                metric.add(self._accuracy(net(x), y), y.numel())

        return metric[0] / metric[1]

    def train_test(self):
        # hyperparameters
        num_inputs, num_outputs = 784, 10
        lr = 0.1
        w = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
        b = torch.zeros(num_outputs, requires_grad=True)

        def softmax(x):
            x_exp = torch.exp(x)
            partition = x_exp.sum(1, keepdim=True)
            return x_exp / partition

        def net(x: torch.Tensor):
            return softmax(torch.matmul(x.reshape(-1, w.shape[0]), w) + b)

        def cross_entropy(y_hat, y):
            return -torch.log(y_hat[range(len(y_hat)), y])

        def updater(bs):
            def sgd(params):
                with torch.no_grad():
                    for param in params:
                        param -= lr * param.grad / bs
                        param.grad.zero_()

            return sgd([w, b])

        num_epoch = 10
        batch_size = 256
        train_iter, test_iter = self.data_iter(batch_size)

        self.train(net, train_iter, test_iter, cross_entropy, num_epoch, updater)

        # x = torch.normal(0, 1, (2, 5))
        # x_prob = softmax(x)
        # print(x_prob, x_prob.sum(1))

        # y = torch.tensor([0, 2])
        # y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
        # print(y_hat[[0, 1], y])

        # print(cross_entropy(y_hat, y))

        # print("accuracy: ", self._accuracy(y_hat, y) / len(y))

        return True

    def train_epoch(self, net, train_iter, loss, updater):
        # Set model to training state
        if isinstance(net, torch.nn.Module):
            net.train()

        # Training
        metric = Accumulator(3)
        for x, y, in train_iter:
            y_hat = net(x)
            l = loss(y_hat, y)

            if isinstance(updater, torch.optim.Optimizer):
                updater.zero_grad()
                l.mean().backward()
                updater.step()
            else:
                l.sum().backward()
                updater(x.shape[0])
            metric.add(float(l.sum()), self._accuracy(y_hat, y), y.numel())

        # Return training loss and accuracy
        return metric[0] / metric[2], metric[1] / metric[2]

    def train(self, net, train_iter, test_iter, loss, num_epochs, updater):
        train_loss, train_acc, test_acc = 0.0, 0.0, 0.0

        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(net, train_iter, loss, updater)
            test_acc = self.evaluate_accuracy(net, test_iter)

            print(f'The {epoch + 1} training: train_loss={train_loss}, train_acc={train_acc}, test_acc={test_acc}')

        assert train_loss < 0.5, train_loss
        assert 1 >= train_acc > 0.7, train_acc
        assert 1 >= test_acc > 0.7, test_acc

    def _get_dataloader_workers(self):
        # Using 4 threads for data loading
        return 4

    def data_iter(self, batch_size, resize=None):
        # Download Fashion-MNIST dataset, and load it into memory
        trans = [transforms.ToTensor()]
        if resize:
            trans.insert(0, transforms.Resize(resize))
        trans = transforms.Compose(trans)

        mnist_train = torchvision.datasets.FashionMNIST(
            root='../data', train=True, transform=trans, download=True)
        mnist_test = torchvision.datasets.FashionMNIST(
            root='../data', train=False, transform=trans, download=True)

        return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                                num_workers=self._get_dataloader_workers()),
                data.DataLoader(mnist_test, batch_size, shuffle=False,
                                num_workers=self._get_dataloader_workers()))

    def softmax_training_simple(self):
        # Hyperparameters
        num_inputs, num_outputs = 784, 10
        batch_size = 256
        lr = 0.1
        num_epochs = 10

        train_iter, test_iter = self.data_iter(batch_size)

        net = torch.nn.Sequential(nn.Flatten(), nn.Linear(num_inputs, num_outputs))

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, std=0.01)

        net.apply(init_weights)
        loss = nn.CrossEntropyLoss(reduction='none')
        trainer = torch.optim.SGD(net.parameters(), lr=lr)

        self.train(net, train_iter, test_iter, loss, num_epochs, trainer)

        return True

    def show_images(self, imgs, num_rows, num_cols, titles=None, scale=1.5):
        # Plot image list
        figsize = (num_cols * scale, num_rows * scale)
        _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        axes = axes.flatten()

        for i, (ax, img) in enumerate(zip(axes, imgs)):
            if torch.is_tensor(img):
                # Image tensor
                ax.imshow(img.numpy())
            else:
                # PIL image
                ax.imshow(img)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            if titles:
                ax.set_title(titles[i])

        plt.show()


class IntegrationTest(unittest.TestCase):

    def test_naive_model(self):
        model = LinearNeuralNetworkNaive()
        res = model.forward()
        self.assertTrue(res)

    def test_simple_pytorch_model(self):
        simple_torch_model = LinearNeuralNetworkSimple()
        # 一般训练的步骤
        #  1. 通过调用 net(x) 生成预测并计算损失 l (前向传播)
        #  2. 通过反向传播计算梯度
        #  3. 通过调用优化器来更新模型参数
        res = simple_torch_model.train()

        self.assertTrue(res)

    def test_mnist(self):
        mnist_model = MNISTSimple()
        # res = mnist_model.test()
        # res = mnist_model.train_test()
        res = mnist_model.softmax_training_simple()

        self.assertTrue(res)

    def test_mnist_raw(self):
        mnist_model = MNISTSimple()
        res = mnist_model.train_test()

        self.assertTrue(res)



if __name__ == '__main__':
    unittest.main(verbosity=True)