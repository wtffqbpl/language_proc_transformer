#! coding: utf-8

import unittest
import math
import numpy as np
import torch
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.plot import plot
import utils.dlf as dlf


def f(x):
    return x * torch.cos(np.pi * x)


def g(x):
    return f(x) + 0.2 * torch.cos(5 * np.pi * x)


def annotate(text, xy, xytext):
    plt.gca().annotate(text, xy=xy, xytext=xytext, arrowprops=dict(arrowstyle='->'))


def show_trace(results, f):
    n = max(abs(min(results)), abs(max(results)))
    f_line = torch.arange(-n, n, 0.01)
    plot([f_line, results], [[f(x) for x in f_line], [f(x) for x in results]],
         'x', 'f(x)', fmts=['-', '-o'])
    plt.show()


def train_2d(trainer, steps=20, f_grad=None):
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
        print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results


def show_trace_2d(f, results):
    plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = torch.meshgrid(torch.arange(-5.5, 1.0, 0.1),
                            torch.arange(-3.0, 1.0, 0.1))
    plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


class GradientDescentCase:
    def __init__(self):
        self.eta = 0.1

    def f(self, x):
        return x ** 2

    def f_grad(self, x):
        return 2 * x

    def gd_impl(self, eta, f_grad):
        x = 10.0
        results = [x]
        for i in range(10):
            x -= eta * f_grad(x)
            results.append(float(x))
        print(f'epoch 10, x: {x:f}')
        return results

    def gradient_descent_1d(self, learning_rate=0.2):
        results = self.gd_impl(learning_rate, self.f_grad)
        show_trace(results, self.f)

    def f_2d(self, x1, x2):
        return x1 ** 2 + 2 * x2 ** 2

    def f_2d_grad(self, x1, x2):
        return 2 * x1, 4 * x2

    def gd_2d(self, x1, x2, s1, s2, f_grad):
        g1, g2 = f_grad(x1, x2)
        return x1 - self.eta * g1, x2 - self.eta * g2, 0, 0

    def gradient_descent_2d(self, eta):
        self.eta = eta

        show_trace_2d(self.f_2d, train_2d(self.gd_2d, f_grad=self.f_2d_grad))


class SGDCases:
    def __init__(self, eta, lr_fn):
        self.lr_fn = lr_fn
        self.eta = eta
        pass

    def f(self, x1, x2):
        return x1 ** 2 + 2 * x2**2

    def f_grad(self, x1, x2):
        return 2 * x1, 4 * x2

    def sgd(self, x1, x2, s1, s2, grad_fn):
        g1, g2 = grad_fn(x1, x2)

        # Simulate gradients with noise
        g1 += torch.normal(0.0, 1, (1, ))
        g2 += torch.normal(0.0, 1, (1,))
        eta_t = self.eta * self.lr_fn()
        return (x1 - eta_t * g1).item(), (x2 - eta_t * g2).item(), 0, 0


t = 1
k = 1


class IntegrationTest(unittest.TestCase):
    def test_risky(self):
        x = torch.arange(0.5, 1.5, 0.01)
        plot(x, [f(x), g(x)], 'x', 'risk', figsize=(5.5, 3.5))
        annotate('min of\nempirical risk', (1.0, -1.2), (0.5, -1.1))
        annotate('min of risk', (1.1, -1.05), (0.95, -0.5))
        plt.show()

        self.assertTrue(True)

    def test_show_local_maximum(self):
        x = torch.arange(-1.0, 2.0, 0.01)
        plot(x, [f(x), ], 'x', 'f(x)')
        annotate('local minimum', (-0.3, -0.25), (-0.77, -1.0))
        annotate('global minimum', (1.1, -0.95), (0.6, 0.8))
        plt.show()

        self.assertTrue(True)

    def test_show_saddle_point(self):
        x = torch.arange(-2.0, 2.0, 0.01)
        plot(x, [x ** 3], 'x', 'f(x)')
        annotate('saddle point', (0, -0.2), (-0.52, 5.0))
        plt.show()

        self.assertTrue(True)

    def test_gradient_descent(self):
        gd_case = GradientDescentCase()

        gd_case.gradient_descent_1d(0.2)

        # The learning rate `eta` can be set by the algorithm designer.
        # If we use a learning rate that is too small, it will cause x to
        # update very slowly, requiring more iterations to get a better solution.
        gd_case.gradient_descent_1d(0.05)

        # Conversely, if we use an excessively high learning rate `eta * f^prime(x)`
        # might be too large for the first-order Taylor expansion formula.
        # That is, the term O(eta^2 * fprime^2(x)) might become significant.
        # In this case, we cannot guarantee that the iteration of x will be able
        # to lower the value of f(x).
        gd_case.gradient_descent_1d(1.1)

        self.assertTrue(True)

    def test_gradient_descent_2d(self):
        eta = 0.1
        gd_case = GradientDescentCase()
        gd_case.gradient_descent_2d(eta)

        self.assertTrue(True)

    def test_sgd(self):
        def constant_lr():
            return 1

        eta = 0.1
        sgd_handle = SGDCases(eta, constant_lr)
        show_trace_2d(sgd_handle.f, train_2d(sgd_handle.sgd, steps=50, f_grad=sgd_handle.f_grad))
        plt.show()

        def exponential_lr():
            global t
            t += 1
            return math.exp(-0.1 * t)

        sgd_handle = SGDCases(eta, exponential_lr)
        show_trace_2d(sgd_handle.f, train_2d(sgd_handle.sgd, steps=50, f_grad=sgd_handle.f_grad))
        plt.show()

        # polynomial learning rate
        def polynomial_lr():
            global k
            k += 1
            return (1 + 0.1 * k) ** (-0.5)

        sgd_handle = SGDCases(eta, polynomial_lr)
        show_trace_2d(sgd_handle.f, train_2d(sgd_handle.sgd, steps=50, f_grad=sgd_handle.f_grad))

        self.assertTrue(True)


# Mini-batch SGD

dlf.DATA_HUB['airfoil'] = (dlf.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')


def get_data(batch_size=10, n=1500):
    data = np.getfromtxt(dlf.download('airfoil'), dtype=np.float32, dtlimiter='\t')
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
    data_iter = dlf.load_array((data[:n, :-1], data[:n, -1]), batch_size, is_train=True)
    return data_iter, data.shape[1] - 1


def sgd(params, states, hyperparams):
    for p in params:
        p.data.sub_(hyperparams['lr'] * p.grad)
        p.grad.data.zero_()


def train(trainer_fn, states, hyperparams, data_iter, feature_dim, num_epochs=2):
    # Init model
    w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1), requires_grad=True)
    b = torch.zeros((1), requires_grad=True)

    # net, loss = lambda X: dlf.li
    pass


if __name__ == '__main__':
    unittest.main(verbosity=True)
