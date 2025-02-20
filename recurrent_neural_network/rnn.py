#! coding: utf-8

import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from utils.plot import plot
from utils.accumulator import Accumulator
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import utils.dlf as dlf


def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def evaluate_loss(net, data_iter, loss):
    metric = Accumulator(2)
    for x, y in data_iter:
        out = net(x)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]


class RNNSimple:

    def __init__(self):
        pass

    def get_net(self):
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)

        net = nn.Sequential(
            nn.Linear(4, 10),
            nn.ReLU(),
            nn.Linear(10, 1))
        net.apply(init_weights)
        return net

    def get_loss(self):
        return nn.MSELoss(reduction='none')

    def train(self, net, train_iter, loss, epochs, lr):
        trainer = torch.optim.Adam(net.parameters(), lr)
        for epoch in range(epochs):
            for x, y in train_iter:
                trainer.zero_grad()
                l = loss(net(x), y)
                l.sum().backward()
                trainer.step()
            print(f'epoch {epoch + 1}, ',
                  f'loss: {evaluate_loss(net, train_iter, loss):f}')


class IntegrationTest(unittest.TestCase):
    def test_basic(self):
        T = 1000
        time = torch.arange(1, T + 1, dtype=torch.float32)
        x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
        plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
        self.assertTrue(True)

    def test_rnn_simple(self):
        T = 1000
        time = torch.arange(1, T + 1, dtype=torch.float32)
        x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
        tau = 4
        features = torch.zeros((T - tau, tau))
        for i in range(tau):
            features[:, i] = time[i: T - tau + i]
        labels = x[tau:].reshape((-1, 1))

        batch_size, n_train = 16, 600
        train_iter = load_array((features[:n_train], labels[:n_train]),
                                batch_size, is_train=True)

        model = RNNSimple()
        net, loss = model.get_net(), model.get_loss()
        model.train(net, train_iter, loss, 50, 0.01)

        # one-step ahead prediction
        onestep_preds = net(features)
        plot([time, time[tau:]],
             [x.detach().numpy(), onestep_preds.detach().numpy()], 'time',
             'x', legend=['data', '1-step preds'], xlim=[1, 1000],
             figsize=(6, 3))

        self.assertTrue(True)

    def test_rnn_module(self):
        device = dlf.devices()[0]
        print(device)
        # x.shape = (batch_size, num_steps)
        # w_xh.shape = (batch_size, num_hiddens)
        # h.shape = (batch_size, num_hiddens)
        # w_hh.shape = (num_hiddens, num_hiddens)
        batch_size, num_steps, num_hiddens = 3, 1, 4
        x = torch.normal(0, 1, (batch_size, num_steps)).to(device=device)
        w_xh = torch.normal(0, 1, (num_steps, num_hiddens)).to(device=device)
        h = torch.normal(0, 1, (batch_size, num_hiddens)).to(device=device)
        w_hh = torch.normal(0, 1, (num_hiddens, num_hiddens)).to(device=device)
        res = torch.matmul(x, w_xh) + torch.matmul(h, w_hh)
        self.assertEqual(torch.Size([batch_size, num_hiddens]), res.shape)
        print(res)

        res = torch.matmul(torch.cat((x, h), dim=1), torch.cat((w_xh, w_hh), dim=0))
        print(res)

        self.assertEqual(torch.Size([batch_size, num_hiddens]), res.shape)


if __name__ == "__main__":
    unittest.main(verbosity=True)
