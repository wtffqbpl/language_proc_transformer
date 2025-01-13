#! coding: utf-8

import math
import unittest
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
import torchvision
from torchvision import transforms
from utils.plot import plot
from utils.accumulator import Accumulator

# Multiplayer Perceptron 在输出层和输入层之间增加一个或者多个全连接隐藏层，并通过
# 激活函数转换隐藏层的输出。
# 常用的激活函数包括ReLU函数、sigmoid函数、tanh函数。


class MultiplayerPerceptronModel:
    def __init__(self):
        self._batch_size = 256
        self._dataloader_workers = 4
        self._num_inputs = 784
        self._num_outputs = 10
        self._num_hiddens = 256
        self._num_epochs = 10
        self._lr = 0.1

        w1 = nn.Parameter(torch.randn(
            self._num_inputs, self._num_hiddens, requires_grad=True) * 0.01)
        b1 = nn.Parameter(torch.zeros(self._num_hiddens, requires_grad=True))

        w2 = nn.Parameter(torch.randn(
            self._num_hiddens, self._num_outputs, requires_grad=True) * 0.01)
        b2 = nn.Parameter(torch.zeros(self._num_outputs, requires_grad=True))

        self.params = [w1, b1, w2, b2]

    def train_simple_wrapper(self):
        net = self.net
        train_iter, test_iter = self.data_iter(self._batch_size)
        loss = self.loss()
        updater = self.updater(self.params)

        self.train(net, train_iter, test_iter, loss, self._num_epochs, updater)

        return True

    def train_pytorch_wrapper(self):
        # 对于相同的分类问题，多层感知机的实现与softmax回归的实现想通，区别是多层感知机的实现
        # 里增加了带有激活函数的隐藏层。
        net = nn.Sequential(nn.Flatten(),
                            nn.Linear(self._num_inputs, self._num_hiddens),
                            nn.ReLU(),
                            nn.Linear(self._num_hiddens, self._num_outputs))

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, std=0.01)

        net.apply(init_weights)

        loss = nn.CrossEntropyLoss(reduction='none')
        trainer = torch.optim.SGD(net.parameters(), lr=self._lr)

        train_iter, test_iter = self.data_iter(self._batch_size)

        self.train(net, train_iter, test_iter, loss, self._num_epochs, trainer)

        return True

    def train_polynomial_fitting(self):
        # y = 5 + 1.2x - 3.4* x^2 / 2! + 5.6 * x^3 / 3! + epsilon (where epsilon is N(0, 0.1^2)
        max_degree = 20  # The max degree of polynomial
        n_train, n_test = 100, 100  # training dataset and the size of dataset
        true_w = np.zeros(max_degree)
        true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

        features = np.random.normal(size=(n_train + n_test, 1))
        np.random.shuffle(features)
        poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
        for i in range(max_degree):
            poly_features[:, i] /= math.gamma(i + 1)  # gamma(n) = (n - 1)!

        # The dimension of labels: (n_train + n_test, )
        labels = np.dot(poly_features, true_w)
        labels += np.random.normal(scale=0.1, size=labels.shape)

        # numpy ndarray to torch.tensor
        true_w, features, poly_features, labels = [
            torch.tensor(x, dtype=torch.float32) for x in [true_w, features, poly_features, labels]]

        print(features[:2], poly_features[:2, :], labels[:2])

        # Normal fitting
        weight = self.polynomial_fitting(poly_features[:n_train, :4],
                                         poly_features[n_train:, :4],
                                         labels[:n_train],
                                         labels[n_train:])
        print(f'weight: {weight}')

        # under-fitting
        weight = self.polynomial_fitting(poly_features[:n_train, :2],
                                         poly_features[n_train:, :2],
                                         labels[:n_train],
                                         labels[n_train:])
        print(f'weight: {weight}')

        # over-fitting
        weight = self.polynomial_fitting(poly_features[:n_train, :],
                                         poly_features[n_train:, :],
                                         labels[:n_train],
                                         labels[n_train:],
                                         num_epochs=1500)
        print(f'weight: {weight}')

        return True

    def polynomial_fitting(self, train_features, test_features, train_labels, test_labels,
                           num_epochs=400):
        loss = nn.MSELoss(reduction='none')
        input_shape = train_features.shape[-1]
        net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
        batch_size = min(10, train_labels.shape[0])
        train_iter = self.load_array((train_features, train_labels.reshape(-1, 1)),
                                     batch_size)
        test_iter = self.load_array((test_features, test_labels.reshape(-1, 1)),
                                    batch_size, is_train=False)
        trainer = torch.optim.SGD(net.parameters(), lr=0.01)

        for epoch in range(num_epochs):
            self.train_epoch(net, train_iter, loss, trainer)
            if epoch == 0 or (epoch + 1) % 20 == 0:
                print(f"{epoch + 1} iter: training_loss: {self.evaluate_loss(net, train_iter, loss)}, ",
                      f"test_loss: {self.evaluate_loss(net, test_iter, loss)}")

        # Weight
        return net[0].weight.detach().numpy()

    def load_array(self, data_arrays, batch_size, is_train=True):
        dataset = data.TensorDataset(*data_arrays)
        return data.DataLoader(dataset, batch_size, shuffle=is_train)

    def evaluate_loss(self, net, data_iter, loss):
        metric = Accumulator(2)
        for x, y in data_iter:
            out = net(x)
            y = y.reshape(out.shape)
            l = loss(out, y)
            metric.add(l.sum(), l.numel())
        return metric[0] / metric[1]

    def train(self, net, train_iter, test_iter, loss, num_epoch, updater):
        train_loss, train_acc, test_acc = 0.0, 0.0, 0.0

        for epoch in range(num_epoch):
            train_loss, train_acc = self.train_epoch(net, train_iter, loss, updater)
            test_acc = self.evaluate_accuracy(net, test_iter)

            print(f"The {epoch + 1} training: train_loss={train_loss}, train_acc={train_acc}, test_acc={test_acc}")

        assert train_loss < 0.5, train_loss
        assert 1 >= train_acc > 0.7, train_acc
        assert 1 >= train_acc > 0.7, test_acc

    def train_epoch(self, net, train_iter, loss, updater):
        # Set model to training state
        if isinstance(net, torch.nn.Module):
            net.train()

        # Training
        metric = Accumulator(3)
        for x, y in train_iter:
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

    def _accuracy(self, y_hat: torch.Tensor, y: torch.Tensor) -> float:
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = y_hat.argmax(dim=1)

        cmp = y_hat.type(y.dtype) == y
        return float(cmp.to(y.dtype).sum())

    def evaluate_accuracy(self, net, data_iter):
        if isinstance(net, torch.nn.Module):
            net.eval()

        # 这里的第二个accumulator 其实是用来统计个数的
        metric = Accumulator(2)
        with torch.no_grad():
            for x, y in data_iter:
                metric.add(self._accuracy(net(x), y), y.numel())

        return metric[0] / metric[1]

    def relu(self, x):
        a = torch.zeros_like(x)
        return torch.max(x, a)

    def net(self, x):
        w1, b1, w2, b2 = self.params
        x = x.reshape(-1, self._num_inputs)
        h = self.relu(x @ w1 + b1)  # @ 代表矩阵乘法
        return h @ w2 + b2

    def loss(self):
        return nn.CrossEntropyLoss(reduction='none')

    def updater(self, params):
        return torch.optim.SGD(params, lr=self._lr)

    def data_iter(self, batch_size, resize=None):
        trans = [transforms.ToTensor()]

        if resize:
            trans.insert(0, transforms.Resize(resize))
        trans = transforms.Compose(trans)

        mnist_train = torchvision.datasets.FashionMNIST(
            root='../data', train=True, transform=trans, download=True)
        mnist_test = torchvision.datasets.FashionMNIST(
            root='../data', train=True, transform=trans, download=True)

        return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                                num_workers=self._dataloader_workers),
                data.DataLoader(mnist_test, batch_size, shuffle=False,
                                num_workers=self._dataloader_workers))


class HighDimensionalPolynomialRegression:
    def __init__(self):
        self.n_train = 20
        self.n_test = 100
        self.num_inputs = 200
        self.batch_size = 5
        self.true_w = torch.ones((self.num_inputs, 1)) * 0.01
        self.true_b = 0.05

        # Training dataset
        train_data = self.synthetic_data(self.true_w, self.true_b, self.n_train)
        self.train_iter = self.load_array(train_data, self.batch_size)

        # Inference dataset
        test_data = self.synthetic_data(self.true_w, self.true_b, self.n_test)
        self.test_iter = self.load_array(test_data, self.batch_size, is_train=False)

    def train(self, lambd):
        w, b = self.init_params()
        net = lambda X: self.linreg(X, w, b)
        loss = self.squared_loss
        num_epochs, lr = 100, 0.003

        for epoch in range(num_epochs):
            for x, y in self.train_iter:
                # Add L2 norm penalty
                l = loss(net(x), y) + lambd * self.l2_penalty(w)
                l.sum().backward()
                self.sgd([w, b], lr, self.batch_size)

            if (epoch + 1) % 5 == 0:
                print(f'train_loss: {self.evaluate_loss(net, self.train_iter, loss)}',
                      f'test_loss: {self.evaluate_loss(net, self.test_iter, loss)}')

        # print('The W L2 norm: ', torch.norm(w).item())
        return torch.norm(w).item()

    def train_concise(self, wd):
        net = nn.Sequential(nn.Linear(self.num_inputs, 1))

        for param in net.parameters():
            param.data.normal_()

        loss = nn.MSELoss(reduction='none')
        num_epochs, lr = 100, 0.003

        trainer = torch.optim.SGD([
            {'params': net[0].weight, 'weight_decay': wd},
            {'params': net[0].bias}], lr=lr)

        for epoch in range(num_epochs):
            for x, y in self.train_iter:
                trainer.zero_grad()
                l = loss(net(x), y)
                l.mean().backward()
                trainer.step()
                if (epoch + 1) % 5 == 0:
                    print(f'Epoch = {epoch + 1}: train_loss = {self.evaluate_loss(net, self.train_iter, loss)}, ',
                          f'test_loss = {self.evaluate_loss(net, self.test_iter, loss)}')

        return net[0].weight.norm().item()

    def init_params(self):
        w = torch.normal(0, 1, size=(self.num_inputs, 1), requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        return [w, b]

    def evaluate_loss(self, net, data_iter, loss):
        metric = Accumulator(2)

        for x, y in data_iter:
            out = net(x)
            y = y.reshape(out.shape)
            l = loss(out, y)
            metric.add(l.sum(), l.numel())

        return metric[0] / metric[1]

    def sgd(self, params, lr, batch_size):
        with torch.no_grad():
            for param in params:
                param -= lr * param.grad / batch_size
                param.grad.zero_()

    def l2_penalty(self, w):
        return torch.sum(w.pow(2)) / 2

    def linreg(self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
        return torch.matmul(x, w) + b

    def squared_loss(self, y_hat, y):
        return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

    def synthetic_data(self, w, b, num_examples):
        x = torch.normal(0, 1, (num_examples, len(w)))
        y = torch.matmul(x, w) + b
        y += torch.normal(0, 0.01, y.shape)
        return x, y.reshape(-1, 1)

    def load_array(self, data_array, batch_size, is_train=True):
        dataset = data.TensorDataset(*data_array)
        return data.DataLoader(dataset, batch_size, shuffle=is_train)


class IntegrationTest(unittest.TestCase):

    def test_relu_function(self):
        x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
        # 使用relu 的原因是，它的求导表现特别好: 要么让参数消失，要么让参数通过。
        # 这使得优化表现更好，并且ReLU 缓解了以往神经网络的梯度消失的问题。
        # ReLU function has some variations, such as the parameterized ReLU.
        # This variation adds a linear item, so some negative values are
        # allowed even if the parameter is negative.
        # pReLU(x) = max(0, x) + alpha * min(0, x)
        y = torch.relu(x)
        plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))

        y.backward(torch.ones_like(x), retain_graph=True)
        plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
        # TODO: no need to check the result for this testcase
        self.assertTrue(True)

    def test_sigmoid_funcion(self):
        # The sigmoid function compresses any variables in (-inf, inf) range
        # into (0, 1) range.
        # sigmoid(x) = 1.0 / (1.0 + exp(-x))
        # 在最早的神经网络中，科学家们感兴趣的是对 "激活" 或 "不激活" 的生物神经元进行建模。
        # 当人们逐渐关注与基于梯度的学习时，sigmoid 函数是一个自然的选择，因为它是一个平滑
        # 的、可微的阈值单元的近似函数。当我们想要将输出视为二元分类问题的概率时，sigmoid
        # 仍然被广泛用作输出单元上的激活函数（sigmoid可视为softmax的特例）。然而sigmoid
        # 在隐藏层中已经较少使用，大部分时候被更简单、更容易训练的ReLU所替代。

        # The sigmoid function
        x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
        y = torch.sigmoid(x)
        plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))

        # Compute gradients
        # The differentiation of the sigmoid function is:
        # d(sigmoid(x))/dx = exp(-x) / (1 + exp(-x))^2 = sigmoid(x) * (1 - sigmoid(x))
        y.backward(torch.ones_like(x), retain_graph=True)
        plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))

        self.assertTrue(True)

    def test_tanh_funcion(self):
        # 与sigmoid 函数类似，tanh(双曲正切) 函数也能将其输入压缩转换到区间(-1, 1)上。 tanh函数
        # 如下:
        # tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x))
        x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
        y = torch.tanh(x)
        plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))

        # The differentiation of the tanh(x) function is:
        # d(tanh(x)) / d(x) = 1 - tanh^2(x)
        y.backward(torch.ones_like(x), retain_graph=True)
        plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
        self.assertTrue(True)

    def test_multiplayer_perceptron(self):
        model = MultiplayerPerceptronModel()

        res = model.train_simple_wrapper()

        self.assertTrue(res)

    def test_multiplayer_perceptron_pytorch(self):
        model = MultiplayerPerceptronModel()

        res = model.train_pytorch_wrapper()

        self.assertTrue(res)

    def test_polynomial_fitting(self):
        model = MultiplayerPerceptronModel()

        res = model.train_polynomial_fitting()

        self.assertTrue(res)

    def test_weight_decay(self):
        model = HighDimensionalPolynomialRegression()
        norm = model.train(lambd=0)
        print('lambda = 0: The W L2 norm: ', norm)

        norm = model.train(lambd=3)
        print('lambda = 3: The W L2 norm: ', norm)

        self.assertTrue(True)

    def test_weight_decay_concise(self):
        model = HighDimensionalPolynomialRegression()

        norm = model.train_concise(wd=0)
        print('lambda = 0: The W L2 norm: ', norm)

        norm = model.train_concise(wd=3)
        print('lambda = 0: The W L2 norm: ', norm)

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main(verbosity=True)
