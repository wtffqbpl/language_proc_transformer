#! coding: utf-8


import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F


def corr2d(x, k):
    h, w = k.shape
    y = torch.zeros((x.shape[0] - h + 1, x.shape[1] - w + 1))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i, j] = (x[i:i + h, j:j + w] * k).sum()
    return y


class Conv2D(nn.Module):
    # A convolutional layer cross-correlates the input and kernel and adds a scalar bias to
    # produce an output. The two parameters of a convolutional layer are the kernel and the
    # scalar bias. When training models based on convolutional layers, we typically
    # initialize the kernels randomly, just as we would with a fully connected layer.

    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


class Conv2DWithLearning:
    def __init__(self, kernel_size, y):
        self.kernel_size = kernel_size
        self.conv2d = nn.Conv2d(1, 1, kernel_size=self.kernel_size, bias=False)
        self.y = y
        self.lr = 3e-2

        print(self.conv2d.bias)

        # [batch_size, channel, height, width]
    def train(self, x):
        y_hat = self.conv2d(x)
        l = (y_hat - self.y) ** 2
        self.conv2d.zero_grad()
        l.sum().backward()

        self.conv2d.weight.data[:] -= self.lr * self.conv2d.weight.grad

        return l

    def get(self):
        return self.conv2d.weight.data.reshape(self.kernel_size)


class IntegrationTest(unittest.TestCase):

    def test_object_edge_detection(self):
        shape = (6, 8)
        x = torch.ones(shape)
        x[:, 2:6] = 0
        print(x)

        # Define a kernel
        k = torch.tensor([[1.0, -1.0]])

        y = corr2d(x, k)
        print(y)
        # The 1 represents the boundary from black to white, and the -1 represents
        # the boundary from white to black.
        # tensor([[0., 1., 0., 0., 0., -1., 0.],
        #         [0., 1., 0., 0., 0., -1., 0.],
        #         [0., 1., 0., 0., 0., -1., 0.],
        #         [0., 1., 0., 0., 0., -1., 0.],
        #         [0., 1., 0., 0., 0., -1., 0.],
        #         [0., 1., 0., 0., 0., -1., 0.]])

        # And this convolution function only could detect the vertical boundaries, and
        # the horizontal boundaries cannot be detected using this convolution function.
        y2 = corr2d(x.t(), k)

        self.assertTrue(True)

    def test_conv2d_with_learning(self):
        kernel_size = (1, 2)
        shape = (6, 8)

        x = torch.ones(shape)
        x[:, 2:6] = 0

        k = torch.tensor([[1.0, -1.0]])
        y = corr2d(x, k)

        conv2d_shape = (1, 1, 6, 8)
        x = x.reshape(conv2d_shape)
        print(y.shape)
        y = y.reshape((1, 1, 6, 7))

        net = Conv2DWithLearning(kernel_size, y)

        for epoch in range(100):
            l = net.train(x)

            if (epoch + 1) % 2 == 0:
                print(f'epoch {epoch + 1}, loss {l.sum():.3f}')

        print(net.get())

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main(verbosity=True)
