#! coding: utf-8


import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from pprint import pprint


class NestedLayerParamInfo:

    def block1(self):
        return nn.Sequential(
            nn.LazyLinear(8),
            nn.ReLU(),
            nn.LazyLinear(4),
            nn.ReLU()
        )

    def block2(self):
        net = nn.Sequential()
        for i in range(4):
            net.add_module(f'block_{i}', self.block1())
        return net


class CenteredLayer(nn.Module):
    def __init__(self):
        super(CenteredLayer, self).__init__()

    def forward(self, x):
        return x - x.mean()


class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super(MyLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))

    def forward(self, x):
        linear = torch.matmul(x, self.weight.data) + self.bias.data
        return F.relu(linear)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.LazyLinear(256)
        self.output = nn.LazyLinear(10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))


class IntegrationTest(unittest.TestCase):

    def test_basic(self):
        net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(), nn.LazyLinear(1))
        x = torch.rand(size=(2, 4))
        y = net(x)
        print(net[2].state_dict())
        print(type(net[2].bias))
        print(net[2].bias)
        print(net[2].bias.data)
        print(net[2].weight.grad is None)

        # Transverse all the parameters
        print('The input layer parameters:')
        print(*[(name, param.shape) for name, param in net[0].named_parameters()])
        print('All parameters:')
        print(*[(name, param.shape) for name, param in net.named_parameters()])

        self.assertEqual(y.shape, (2, 1))

    def test_nested_layer_param_info(self):
        net = nn.Sequential(NestedLayerParamInfo().block2(), nn.LazyLinear(1))
        x = torch.rand(size=(2, 4))
        y = net(x)
        print('All parameters:')
        pprint([(name, param.shape) for name, param in net.named_parameters()])
        self.assertEqual(y.shape, (2, 1))

    def test_customized_layer(self):
        net = CenteredLayer()
        x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
        y = net(x)
        print(y)
        self.assertEqual(y.shape, (5,))

    def test_parameterized_layer(self):
        net = MyLinear(5, 3)
        print(net.weight)

        self.assertEqual(net.weight.shape, (5, 3))

    def test_load_save_layer(self):
        x = torch.arange(4)
        torch.save(x, 'x-file')

        x2 = torch.load('x-file', weights_only=True)

        self.assertEqual(x2.shape, (4, ))
        self.assertTrue(torch.allclose(x2, torch.arange(4)))

        y = torch.zeros(4)
        torch.save([x, y], 'x-file')
        x2, y2 = torch.load('x-file', weights_only=True)

        self.assertEqual(y2.shape, y.shape)

        self.assertTrue(torch.allclose(y, y2))

        mydict = {'x': x, 'y': y}
        torch.save(mydict, 'mydict')

        mydict2 = torch.load('mydict', weights_only=True)

        for key1, key2 in zip(mydict, mydict2):
            self.assertEqual(key1, key2)
            self.assertTrue(torch.allclose(mydict[key1], mydict2[key2]))

    def test_save_load_models(self):
        net = MLP()
        x = torch.randn(size=(2, 20))
        y = net(x)
        # save model parameters to file
        torch.save(net.state_dict(), 'mlp.params')

        # Restore model
        clone = MLP()
        clone.load_state_dict(torch.load('mlp.params', weights_only=True))
        clone.eval()

        y_clone = clone(x)
        self.assertTrue(torch.allclose(y, y_clone))

    def test_gpu_usages(self):
        print(torch.cuda.device_count())
        self.assertTrue(torch.cuda.device_count() >= 0)


if __name__ == '__main__':
    unittest.main()
