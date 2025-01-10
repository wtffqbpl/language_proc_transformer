#! coding: utf-8

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import math
import numpy as np
import random

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


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

    def linreg(self, X, w, b):
        return torch.matmul(X, w) + b

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


if __name__ == '__main__':
    model = LinearNeuralNetworkNaive()
    model.forward()
