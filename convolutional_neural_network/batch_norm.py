#! coding: utf-8


import unittest
from unittest.mock import patch
from io import StringIO
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchinfo
from utils.accumulator import Accumulator


def batch_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, moving_mean: torch.Tensor,
               moving_var: torch.Tensor, eps: float, momentum: float):
    # Deciding whether the current batch normalization is training mode or inference mode with
    # is_grad_enabled() method.
    if not torch.is_grad_enabled():
        # If the current batch normalization is inference mode, then we should use the passing moving_mean
        # and moving_var parameters directly.
        x_hat = (x - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(x.shape) in (2, 4)

        if x.ndim == 2:
            # For the dense layer
            mean = x.mean(dim=0)
            var = ((x - mean) ** 2).mean(dim=0)
        elif x.ndim == 4:
            # For the convolution layer, we compute the batch normalization on the channels dim
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = ((x - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        else:
            raise Exception("Cannot process this branch")
        # For the training mode, we compute the batch normalization with the computed mean and var.
        x_hat = (x - mean) / torch.sqrt(var + eps)

        # Update the moving_mean and moving_var
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var

    # Scaling and shifting
    y = gamma * x_hat + beta

    return y, moving_mean.data, moving_var.data


class BatchNorm(nn.Module):
    # num_features: The output number of the dense layers, or the output number of the convolution layers
    # num_dims: 2 --- Indicates the dense layer, 4 --- indicates the convolution layer
    def __init__(self, num_features, num_dims):
        super(BatchNorm, self).__init__()

        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)

        # The learning parameters about the batch normalization.
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

        # The non-model parameters
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, x):
        # If x is not on the memory, let's copy the moving_mean and moving_var to the memory location
        # of the x located.
        if self.moving_mean.device != x.device:
            self.moving_mean = self.moving_mean.to(x.device)
            self.moving_var = self.moving_var.to(x.device)

        # Update the mean and variance
        y, self.moving_mean, self.moving_var = batch_norm(
            x, self.gamma, self.beta, self.moving_mean, self.moving_var, eps=1e-5, momentum=0.9)

        return y


class Reshape(nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)


class LeNetBN(nn.Module):
    def __init__(self):
        super(LeNetBN, self).__init__()

        self.net = nn.Sequential(
            Reshape(),
            nn.LazyConv2d(6, kernel_size=5), BatchNorm(num_features=6, num_dims=4), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5), BatchNorm(num_features=16, num_dims=4), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
            nn.LazyLinear(120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
            nn.LazyLinear(84), BatchNorm(num_features=84, num_dims=2), nn.Sigmoid(),
            nn.LazyLinear(10))

    def forward(self, x):
        return self.net(x)


def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]

    if resize is not None:
        trans.insert(0, transforms.Resize(size=resize))

    trans = transforms.Compose(trans)

    mnist_data = torchvision.datasets.FashionMNIST(
        root='../data', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root='../data', train=False, transform=trans, download=True)

    return (data.DataLoader(mnist_data, batch_size, shuffle=True, num_workers=4),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4))


def accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    if y_hat.ndim > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)

    cmp = y_hat.type(y.dtype) == y
    return float(cmp.to(y.dtype).sum().item())


def evaluate(model, data_iter, device=torch.device('cpu')):
    if isinstance(model, torch.nn.Module):
        model.eval()

    metric = Accumulator(2)

    with torch.no_grad():
        for x, y in data_iter:
            x, y = x.to(device), y.to(device)
            metric.add(accuracy(model(x), y), torch.numel(y))

    return metric[0] / metric[1]


def init_model(model: torch.nn.Module):
    if isinstance(model, torch.nn.LazyConv2d) or isinstance(model, torch.nn.Conv2d) or \
        isinstance(model, torch.nn.LazyLinear) or isinstance(model, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(model.weight)

    if hasattr(model, 'bias') and model.bias is not None:
        torch.nn.init.zeros_(model.bias)


def train(model: nn.Module,
          data_iter: data.DataLoader, test_iter: data.DataLoader,
          num_epochs: int, lr: float,
          device: torch.device=torch.device('cpu')):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        loss = None

        for i, (x, y) in enumerate(data_iter):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f'Epoch: [{epoch+1}/{num_epochs}], ',
                      f'Step: [{i+1}/{len(data_iter)}], ',
                      f'Loss: {loss.item():.4f}')
        print(f'Epoch: [{epoch+1}/{num_epochs}], ',
              f'Loss: {loss.item():.4f}',
              f'Training Accuracy: {evaluate(model, data_iter):.4f}',
              f'Test Accuracy: {evaluate(model, test_iter):.4f}')
    pass


class IntegrationTest(unittest.TestCase):

    def test_lenet_shape(self):
        model = LeNetBN()

        act_output = None
        with patch('sys.stdout', new_callable=StringIO) as log:
            torchinfo.summary(model, input_size=(1, 1, 28, 28))
            act_output = log.getvalue().strip()

        print(act_output)

        expected_output = """
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
LeNetBN                                  [1, 16, 8, 10]            --
├─Sequential: 1-1                        [1, 16, 8, 10]            --
│    └─Reshape: 2-1                      [1, 1, 28, 28]            --
│    └─Conv2d: 2-2                       [1, 6, 24, 24]            156
│    └─BatchNorm: 2-3                    [1, 6, 24, 24]            12
│    └─Sigmoid: 2-4                      [1, 6, 24, 24]            --
│    └─AvgPool2d: 2-5                    [1, 6, 12, 12]            --
│    └─Conv2d: 2-6                       [1, 16, 8, 8]             2,416
│    └─BatchNorm: 2-7                    [1, 16, 8, 8]             32
│    └─Sigmoid: 2-8                      [1, 16, 8, 8]             --
│    └─Linear: 2-9                       [1, 16, 8, 120]           1,080
│    └─BatchNorm: 2-10                   [1, 16, 8, 120]           240
│    └─Sigmoid: 2-11                     [1, 16, 8, 120]           --
│    └─Linear: 2-12                      [1, 16, 8, 84]            10,164
│    └─BatchNorm: 2-13                   [1, 16, 8, 84]            168
│    └─Sigmoid: 2-14                     [1, 16, 8, 84]            --
│    └─Linear: 2-15                      [1, 16, 8, 10]            850
==========================================================================================
Total params: 15,118
Trainable params: 15,118
Non-trainable params: 0
Total mult-adds (M): 0.26
==========================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.50
Params size (MB): 0.06
Estimated Total Size (MB): 0.56
==========================================================================================
        """
        self.assertEqual(expected_output.strip(), act_output)

    def test_lenetbn(self):
        # hyperparameters
        batch_size = 128
        learning_rate = 1.0
        num_epochs = 10

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        data_iter, test_iter = load_data_fashion_mnist(batch_size)

        model = LeNetBN()
        model.to(device=device)

        # Dummy initialization for LazyConv2d and LazyLinear
        _ = model(torch.randn(size=(1, 1, 28, 28)).to(device=device))

        # Apply initialization method to the model
        model.apply(init_model)

        train(model, data_iter, test_iter, num_epochs, learning_rate, device)

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main(verbosity=True)
