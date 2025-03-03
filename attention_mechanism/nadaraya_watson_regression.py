#! coding: utf-8

import unittest
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.plot import show_heatmaps, plot


class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super(NWKernelRegression, self).__init__(**kwargs)
        self.w = nn.Parameter(torch.rand(size=(1,), requires_grad=True))

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor):
        # The shape of the queries and attention_weights is (num_queries, num_key_values_pairs)
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        attention_weights = nn.functional.softmax(-((queries - keys) * self.w)**2 / 2, dim=1)

        # The shape of the value is (num_queries, num_key_values_pairs)
        return torch.bmm(attention_weights.unsqueeze(1), values.unsqueeze(-1)).reshape(-1)


class IntegrationTest(unittest.TestCase):
    def test_heatmap(self):
        attention_weights = torch.eye(10).reshape(1, 1, 10, 10)
        show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
        plt.show()
        self.assertTrue(True)

    def test_nadaraya_watson_regression(self):
        # Generate datasets
        n_train = 50
        x_train, _ = torch.sort(torch.rand(n_train) * 5)

        def f(x):
            return 2 * torch.sin(x) + x ** 0.8

        # The output for the training dataset
        y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))

        # testing samples
        x_test = torch.arange(0, 5, 0.1)

        y_truth = f(x_test)
        n_test = x_test.numel()

        def plot_kernel_reg(y_hat):
            plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'], xlim=[0, 5], ylim=[-1, 5])
            plt.plot(x_train, y_train, 'o', alpha=0.5)

        y_hat = torch.repeat_interleave(y_train.mean(), n_test)
        plot_kernel_reg(y_hat)
        plt.show()

        # Each line has the same test values
        x_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
        # attention_weights.shape = (n_test, n_train)
        attention_weights = nn.functional.softmax(-(x_repeat - x_train)**2 / 2, dim=1)
        y_hat = torch.matmul(attention_weights, y_train)
        plot_kernel_reg(y_hat)
        plt.show()

        show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
                      xlabel='Sorted training inputs',
                      ylabel='Sorted testing inputs')
        plt.show()

        self.assertTrue(True)

    def test_nw_training(self):
        n_train = 50
        x_train, _ = torch.sort(torch.rand(n_train) * 5)

        def f(x):
            return 2 * torch.sin(x) + x ** 0.8

        y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))
        # The x_tile.shape = (n_train, n_train)
        x_tile = x_train.repeat((n_train, 1))
        # The y_tile.shape = (n_train, n_train)
        y_tile = y_train.repeat((n_train, 1))
        # The keys.shape = (n_train, n_train - 1)
        # 这一行的意思是，pos = (1 - torch.eye(n_train)).type(torch.bool) 生成一个除了对角线位置以外，
        # 其他位置都是True，
        # x_tile[pos] 的作用是，将为True位置的元素挑出来，False位置的元素不需要，
        # 因此结合上面pos的生成过程来看，keys 将x_tile 中的除对角线元素以外的元素挑出来
        # 因此keys.shape = (n_train, n_train - 1)
        keys = x_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
        # The values.shape = (n_train, n_train - 1)
        values = y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

        num_epochs, learning_rate = 5, 0.5
        net = NWKernelRegression()
        loss_fn = nn.MSELoss(reduction='none')
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = loss_fn(net(x_train, keys, values), y_train)
            loss.sum().backward()
            optimizer.step()

            print(f'Epoch: {epoch+1}, ', f'loss: {float(loss.sum()):.6f}')

        x_test = torch.arange(0, 5, 0.1)
        y_truth = f(x_test)
        n_test = x_test.numel()

        # inference
        keys = x_train.repeat((n_test, 1))
        values = y_train.repeat((n_test, 1))
        y_hat = net(x_test, keys, values).unsqueeze(1).detach()

        def plot_kernel_reg(y_hat):
            plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'], xlim=[0, 5], ylim=[-1, 5])
            plt.plot(x_train, y_train, 'o', alpha=0.5)

        plot_kernel_reg(y_hat)
        plt.show()

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main(verbosity=True)
