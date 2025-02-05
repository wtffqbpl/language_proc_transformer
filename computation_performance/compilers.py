#! coding: utf-8


import unittest
from unittest.mock import patch
import torch
import torch.nn as nn
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.benchmark import Benchmark


# Symbolic Programming is a programming paradigm that uses symbols to represent values.
# it usually performs only once the process has been fully defined. This strategy is
# used by multiple deep learning frameworks, including Theano and TensorFlow. It usually
# involves the following steps:
#   1. Define the operations to be executed.
#   2. Compile the operations into an executable program.
#   3. Provide the required inputs and call the compiled program for execution.
# This allows for a significant amount of optimization.
# Symbolic programming is more efficient and easier to port. Symbolic programming makes
# it easier to optimize the code during compilation, while also having the ability
# to port the program into a format independent of Python. This allows the program
# to be run in a non-python environment, thus any potential performance issues
# related to the Python interpreter.
# PyTorch is based on imperative programming and uses dynamic computation graphs.
# In an effort to leverage the portability and efficiency of symbolic programming,
# developers considered whether it would be possible to combine the benefits of
# both programming paradigms. This led to a torchscript that lets users develop
# and debug using pure imperative programming, while having the ability to
# convert most programs into symbolic programs to be run when product-level
# computing performance and deployment are required.
#
# * Deep learning frameworks may decouple the Python frontend from an execution
#   backend. This allows for fast asynchronous insertion of commands into the
#   backend and associated parallelism.
# * Asynchrony leads to a rather responsive frontend. However, use caution not
#   to overfill the task queue since it may lead to excessive memory consumption.
#   It is recommended to synchronize for each minibatch to keep frontend and
#   backend approximately synchronized.
# * Chip vendors offer sophisticated performance analysis tools to obtain a
#   much more fine-grained insight into the efficiency of deep learning.


def get_net():
    net = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 2))
    return net


class IntegrationTest(unittest.TestCase):

    def test_sequential_hybridizing(self):
        x = torch.randn(size=(1, 512))
        net = get_net()
        net_jit = torch.jit.script(net)

        y = net(x)
        y_jit = net_jit(x)

        print(y)
        print(y_jit)

        self.assertEqual(torch.Size([1, 2]), y.shape)
        self.assertEqual(torch.Size([1, 2]), y_jit.shape)
        self.assertTrue(torch.allclose(y, y_jit))

        # Performance test
        with Benchmark('No torchscript'):
            for _ in range(1000):
                y = net(x)

        with Benchmark('Torchscript'):
            for _ in range(1000):
                y = net_jit(x)

        # Save model
        net_jit.save('model.pt')


if __name__ == '__main__':
    unittest.main(verbosity=True)
