#! coding: utf-8

import unittest
import torch
import torch.nn as nn

# ModuleList
# class torch.nn.ModuleList(modules=None)
# Holds submodules in a list.
# ModuleList can be indexed like a regular Python list, but modules it contains are properly
# registered, and will be visible by all Module methods.
# Parameters
# modules(iterable, optional) --- an iterable of modules to add
# Methods
#  * append(module) --- Append a given module to the end of the list.
#  * extend(module) --- Append module from a python iterable to the end of the list.
#  * insert(index, module) --- Insert a given module before a given index in the list.


# Examples
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linear_list = nn.ModuleList([nn.Linear(10, 10) for _ in range(10)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linear_list):
            x = self.linear_list[i // 2](x) + l(x)
        return x


class IntegrationTest(unittest.TestCase):
    def test_module_list(self):
        model = MyModule()

        feature_dim, num_size = 10, 10
        res = model(torch.randn(size=(feature_dim, num_size), dtype=torch.float32))

        self.assertEqual(torch.Size([feature_dim, num_size]), res.shape)


if __name__ == "__main__":
    unittest.main(verbosity=True)
