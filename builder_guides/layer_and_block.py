#! coding: utf-8

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.dlf as dlf


# Perhaps the easiest way to develop intuition about how a module works is to implement
# one ourselves. Before we do that, we briefly summarize the basic functionality that
# each module must provide:
#   1. Ingest input data as arguments to its forward propagation method.
#   2. Generate an output by having the forward propagation method return a value.
#      Note that the output may have a different shape from the input. For example, the
#      first fully connected layer in our model above ingests an input of arbitrary
#      dimension but returns an output of dimension 256.
#   3. Calculate the gradient of its output with respect to its input, which can be
#      accessed via its backpropagation method. Typically, this happens automatically.
#   4. Store and provide access to those parameters necessary for executing the forward
#      propagation computation.
#   5. Initialize model parameters as needed.


class MLP(nn.Module):
    def __init__(self):
        # 调用MLP的父类Module的构造函数执行必要的初始化
        # 这样，在类实例化时也是可以指定其他函数参数，例如模型参数params
        super(MLP, self).__init__()
        self.hidden = nn.LazyLinear(256)  # Hidden Layer
        self.out = nn.LazyLinear(10)  # Output Layer

    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        # 注意，这里我们使用ReLU函数作为激活函数，其在nn.functional模块中定义
        return self.out(F.relu(self.hidden(x)))


class Factory(nn.Module):
    def __init__(self, k):
        super(Factory, self).__init__()
        modules = list()

        for i in range(k):
            modules.append(MLP())
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


def build_blocks(n:int):
    modules = nn.Sequential()

    for i in range(n):
        modules.add_module(str(i), MLP())
    return modules


class MySequential(nn.Module):
    def __init__(self, *args):
        super(MySequential, self).__init__()

        for idx, module in enumerate(args):
            # 这里 module是Module子类的一个实例，我们把它们保存在Module类的成员
            self._modules[str(idx)] = module

    def forward(self, x):
        for block in self._modules.values():
            x = block(x)
        return x


class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super(FixedHiddenMLP, self).__init__()
        # 不计算梯度的随机权重参数，因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.LazyLinear(20)

    def forward(self, x):
        x = self.linear(x)
        # 使用创建的常量参数以及relu和mm函数
        x = F.relu(torch.mm(x, self.rand_weight) + 1)
        # 复用全脸基层这相当于两个全连接层共享参数
        x = self.linear(x)
        # 控制流
        while x.abs().sum() > 1:
            x /= 2
        return x.sum()


class NestMLP(nn.Module):
    def __init__(self):
        super(NestMLP, self).__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.LazyLinear(32),
            nn.ReLU()
        )
        self.linear = nn.LazyLinear(16)

    def forward(self, x):
        return self.linear(self.net(x))


class IntegrationTest(unittest.TestCase):
    def test_mlp(self):
        net = MLP()
        X = torch.rand(2, 20)
        y = net(X)
        print(y)
        self.assertEqual(y.shape, (2, 10))

    def test_my_sequential(self):
        net = MySequential(
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.LazyLinear(10)
        )

        x = torch.rand(2, 20)
        y = net(x)
        print(y)
        self.assertEqual(y.shape, (2, 10))

    def test_fixed_hidden_mlp_layer(self):
        net = FixedHiddenMLP()
        print(net)
        X = torch.rand(2, 20)
        y = net(X)
        print(y)
        self.assertEqual(y.shape, torch.Size([]))

    def test_nest_mlp(self):
        nest_net = NestMLP()
        print(nest_net)
        X = torch.rand(2, 20)
        y = nest_net(X)
        print(y)
        self.assertEqual(y.shape, (2, 16))

    def test_factory_creation(self):
        factory = Factory(3)
        print(factory)
        X = torch.rand(2, 20)
        y = factory(X)
        print(y)
        self.assertEqual(y.shape, (2, 10))

    def test_build_blocks(self):
        model = build_blocks(3)
        print(model)
        x = torch.rand(2, 20)
        y = model(x)
        print(y)
        self.assertEqual(y.shape, (2, 10))


if __name__ == '__main__':
    unittest.main()
    pass
