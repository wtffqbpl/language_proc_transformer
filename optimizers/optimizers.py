#! coding: utf-8

import unittest
import numpy as np
import torch
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.plot import plot


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


if __name__ == '__main__':
    unittest.main(verbosity=True)
