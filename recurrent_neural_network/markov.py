#! coding: utf-8

import unittest
import os
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.plot import plot
from utils.accumulator import Accumulator
import utils.dlf as dlf


class SequenceDataset(Dataset):
    def __init__(self, num_samples):
        super(SequenceDataset, self).__init__()
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        pass


def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
    pass


def init_weights(m: nn.Module):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


class MarkovNet(nn.Module):
    def __init__(self, tau=4):
        super(MarkovNet, self).__init__()
        self.net = nn.Sequential(nn.Linear(tau, 10),
                                 nn.ReLU(),
                                 nn.Linear(10, 1))

    def forward(self, x):
        return self.net(x)


def evaluate_loss(net, data_iter, loss_fn, device) -> float:
    metric = Accumulator(2)
    for x, y in data_iter:
        x, y = x.to(device=device), y.to(device=device)
        out = net(x)
        y = y.reshape(out.shape)
        l = loss_fn(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]


def train(model: nn.Module, train_iter: DataLoader,
          loss_fn, optimizer, epochs: int, lr: float, device: torch.device):
    model.to(device=device)

    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_iter):
            x, y = x.to(device=device), y.to(device=device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.sum().backward()
            optimizer.step()
        print(f'epoch: {epoch+1}, ',
              f'loss: {evaluate_loss(model, train_iter, loss_fn, device):.4f}')


class IntegrationTest(unittest.TestCase):
    def setUp(self):
        self.markov_model_path = os.path.join(str(Path(__file__).resolve().parent), 'markvo_net.pth')

        self.tau = 4
        self.n_train = 600
        self.num_samples = 1000
        self.time = torch.arange(1, self.num_samples + 1, dtype=torch.float32)
        self.x = torch.sin(0.01 * self.time) + torch.normal(0, 0.2, (self.num_samples,))

        self.features = torch.zeros((self.num_samples - self.tau, self.tau))
        for i in range(self.tau):
            self.features[:, i] = self.x[i: self.num_samples - self.tau + i]
        self.labels = self.x[self.tau:].reshape((-1, 1))

        plot(self.time, [self.x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))

    def test_markov_model_training(self):

        # hyperparameters
        batch_size = 600
        epochs, learning_rate = 10, 0.01
        train_iter = load_array((self.features[:self.n_train], self.labels[:self.n_train]),
                                batch_size, is_train=True)

        model = MarkovNet(tau=self.tau)
        model.apply(init_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = torch.nn.MSELoss(reduction='none')
        device = dlf.devices()[0]

        train(model, train_iter, loss_fn, optimizer, epochs, learning_rate, device)

        torch.save(model, self.markov_model_path)

        self.assertTrue(True)

    def test_markov_net_inference(self):
        device = dlf.devices()[0]

        if not os.path.exists(self.markov_model_path):
            self.test_markov_model_training()

        model = torch.load(self.markov_model_path, weights_only=False)
        model.to(device=device)

        onestep_preds = model(self.features.to(device=device))

        # Plot 1-step results
        plot([self.time, self.time[self.tau:]],
             [self.x.detach().numpy(), onestep_preds.detach().cpu().numpy()], 'time',
             'x', legend=['data', '1-step preds'], xlim=[1, 1000], figsize=(6, 3))

        multistep_preds = torch.zeros(self.num_samples, device=device)
        multistep_preds[: self.n_train + self.tau] = self.x[: self.n_train + self.tau]
        for i in range(self.n_train + self.tau, self.num_samples):
            multistep_preds[i] = model(multistep_preds[i - self.tau:i].reshape((1, -1)))

        # Plot multi-step results
        plot([self.time, self.time[self.tau:], self.time[self.n_train + self.tau:]],
             [self.x.detach().numpy(), onestep_preds.detach().cpu().numpy(),
              multistep_preds[self.n_train + self.tau:].detach().cpu().numpy()],
             'time', 'x', legend=['data', '1-step preds', 'multistep preds'],
             xlim=[1, 1000], figsize=(6, 3))

        def k_step_pred(k):
            features = []
            for i in range(self.tau):
                features.append(self.x[i: i + self.num_samples - self.tau - k + 1].to(device=device))

            # The (i+tau)-th element stores the (i+1)-step-ahead predictions
            for i in range(k):
                preds = model(torch.stack(features[i: i+self.tau], 1))
                features.append(preds.reshape(-1))
            return features[self.tau:]

        steps = (1, 4, 16, 64)
        preds = k_step_pred(steps[-1])
        plot(self.time[self.tau + steps[-1]-1:],
             [preds[k - 1].detach().cpu().numpy() for k in steps],
             'time', 'x', legend=[f'{i}-step preds' for i in steps],
             xlim=[5, 1000], figsize=(6, 3))

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main(verbosity=True)
