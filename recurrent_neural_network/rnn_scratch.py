#! coding: utf-8

import unittest
from unittest.mock import patch
from io import StringIO
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchinfo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from recurrent_neural_network.rnn_utils import load_data_time_machine,\
    RNNModelScratch, train, RNNModel, inference, RNNModelWithTorch
import utils.dlf as dlf


def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # Hidden parameters
    w_xh = normal((num_inputs, num_hiddens))
    w_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)

    # Output parameters
    w_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    # Attach gradients
    params = [w_xh, w_hh, b_h, w_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_rnn_state(batch_size, num_hiddens, device):
    return torch.zeros((batch_size, num_hiddens), device=device),


def rnn(inputs, state, params):
    # The inputs shape is (num_steps, batch_size, vocab_size)
    w_xh, w_hh, b_h, w_hq, b_q = params
    h, = state
    outputs = []
    # Shape of `X`: (`batch_size`, `vocab_size`)
    for x in inputs:
        h = torch.tanh(torch.mm(x, w_xh) + torch.mm(h, w_hh) + b_h)
        y = torch.mm(h, w_hq) + b_q
        outputs.append(y)
    return torch.cat(outputs, dim=0), (h,)


# Generate random dataset, and the regression target is the sum of the sequence
def generate_data(num_samples, seq_length, input_size):
    # Generate the random dataset, the shape is [num_samples, seq_length, input_size]
    x = torch.randn(num_samples, seq_length, input_size)
    # The y.shape = [num_samples, 1]
    y = x.sum(dim=1).sum(dim=1, keepdim=True)
    return x, y


def load_pesudo_dataset(batch_size, num_samples, seq_length, input_size):

    train_x, train_y = generate_data(num_samples, seq_length, input_size)
    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    return train_iter


class IntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size, self.num_steps = 32, 35
        self.train_iter, self.vocab = load_data_time_machine(self.batch_size, self.num_steps)

    def test_onehot_simple(self):
        def onehot(x, n_class, dtype=torch.float32):
            result = torch.zeros(x.shape[0], n_class, dtype=dtype)
            result.scatter_(1, x.long().view(-1, 1), 1)
            return result

        x = torch.tensor([0, 2])
        x_onehot = onehot(x, 4)
        self.assertTrue(torch.equal(x_onehot, torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0]])))

        one_hot_encoding = F.one_hot(torch.tensor([0, 2]), len(self.vocab))

        print(one_hot_encoding)
        expected_output = torch.tensor([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.assertTrue(torch.equal(expected_output, one_hot_encoding))

        x = torch.arange(10).reshape((2, 5))
        x_onehot_encoding = F.one_hot(x.T, 28)

        self.assertEqual(torch.Size([5, 2, 28]), x_onehot_encoding.shape)

    def test_rnn_model_shape(self):
        num_hiddens = 512
        device = dlf.devices()[0]
        net = RNNModelScratch(len(self.vocab), num_hiddens, device,
                              get_params_fn=get_params,
                              init_state_fn=init_rnn_state,
                              forward_fn=rnn)

        x = torch.arange(10, device=device).reshape((2, 5))

        state = net.begin_state(x.shape[0], device)
        y, new_state = net(x.to(device=device), state)

        print(y.shape)
        self.assertEqual(torch.Size([10, 28]), y.shape)

        print(len(new_state))
        self.assertEqual(1, len(new_state))

        print(new_state[0].shape)
        self.assertEqual(torch.Size([2, 512]), new_state[0].shape)

    def test_inference(self):
        num_hiddens = 512
        device = dlf.devices()[0]
        net = RNNModelScratch(len(self.vocab), num_hiddens, device,
                              get_params_fn=get_params,
                              init_state_fn=init_rnn_state,
                              forward_fn=rnn)

        res = inference('time traveller', 10, net, self.vocab, dlf.devices()[0])

        print(res)

    def test_training(self):
        num_hiddens, num_epochs, lr = 512, 500, 1

        loss_fn = torch.nn.CrossEntropyLoss()
        device = dlf.devices()[0]
        net = RNNModelScratch(len(self.vocab), num_hiddens, device,
                              get_params_fn=get_params,
                              init_state_fn=init_rnn_state,
                              forward_fn=rnn)

        # Sequential sampling
        train(net, self.train_iter, self.vocab, loss_fn, lr, num_epochs, device)

        # Random sampling
        train(net, self.train_iter, self.vocab, loss_fn, lr, num_epochs, device,
              use_random_iter=True)

    def test_rnn_model(self):
        input_size = 10
        hidden_size = 20
        output_size = 1
        num_layers = 1
        num_epochs = 10
        batch_size = 16
        seq_length = 5
        learning_rate = 0.01
        num_samples = 1000

        train_iter = load_pesudo_dataset(batch_size, num_samples, seq_length, input_size)
        model = RNNModel(input_size, hidden_size, output_size, num_layers)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

        # Training
        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for i, (x, y) in enumerate(train_iter):
                optimizer.zero_grad()
                output = model(x)
                loss = loss_fn(output, y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            print(f'epoch: [{epoch+1}/{num_epochs}], ',
                  f'loss: {epoch_loss/len(train_iter):.4f}')

        # inference
        model.eval()
        x, y = generate_data(5, seq_length, input_size)
        with torch.no_grad():
            predictions = model(x)
            print('predictions: ', predictions)
            print('actual results: ', y)


# Custom dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def load_data_iters(batch_size, x_train, x_test, y_train, y_test):
    train_dataset = TimeSeriesDataset(x_train, y_train)
    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TimeSeriesDataset(x_test, y_test)
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_iter, test_iter


class RNNForecast(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers = 1, output_size=1):
        super(RNNForecast, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # RNN layer. The input shape: [batch, seq_length, input_size]
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer. The output shape: [batch, seq_length, output_size]
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # The x shape is [batch, seq_length]
        # Add the channel dimension
        x = x.unsqueeze(-1)
        # The RNN output
        out, _ = self.rnn(x)
        # The last time step output
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze()


def forcast_train(model: nn.Module, train_iter, loss_fn, optimizer, num_epochs, device):
    model.to(device=device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for x, y in train_iter:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        total_loss /= len(train_iter.dataset)
        if (epoch + 1) % 10 == 0:
            print(f'epoch: {epoch+1}, loss: {total_loss:.4f}')


def forecast_inference(model: nn.Module, test_iter, device):
    model.to(device=device)

    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for x, y in test_iter:
            x, y = x.to(device), y.to(device)
            output = model(x)
            preds.append(output.detach().cpu().numpy())
            actuals.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)
    test_loss = np.mean((preds - actuals) ** 2)
    print(f'test loss: {test_loss:.4f}')

    plt.figure(figsize=(10, 5))
    plt.plot(actuals, label='Actual')
    plt.plot(preds, label='Predicted')
    plt.title("RNN Forecast on Airline Passengers (Normalized)")
    plt.legend()
    plt.show()


class PassangersTest(unittest.TestCase):
    def test_passagers_forecast(self):
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
        df = pd.read_csv(url, parse_dates=['Month'])
        print(df.head())

        # Just using the number of passengers as the target.
        data = df['Passengers'].values.astype(np.float32)

        # Normalize the data (var=0, mean=0)
        data_mean = np.mean(data)
        data_std = np.std(data)
        data_normalized = (data - data_mean) / data_std

        # Generate time sequence: using the previous seq_length to predict the next value
        seq_length = 12

        def create_sequence(data_, seq_length_):
            xs, ys = [], []

            for i in range(len(data_) - seq_length_):
                x = data_[i:i+seq_length_]
                y = data_[i+seq_length_]
                xs.append(x)
                ys.append(y)
            return np.array(xs), np.array(ys)

        x, y = create_sequence(data_normalized, seq_length)
        print("x.shape: ", x.shape)  # (num_samples, seq_length)
        print("y.shape: ", y.shape)  # (num_samples, )

        # Split the dataset into training and validation
        train_size = int(len(x) * 0.8)
        x_train, y_valid = x[:train_size], y[:train_size]
        x_test, y_test = x[train_size:], y[train_size:]

        batch_size, learning_rate = 16, 0.01
        device = dlf.devices()[0]

        train_iter, test_iter = load_data_iters(batch_size, x_train, x_test, y_valid, y_test)
        model = RNNForecast()
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        forcast_train(model, train_iter, loss_fn, optimizer, 100, device)

        forecast_inference(model, test_iter, device)


class RNNTorchAPITest(unittest.TestCase):
    def test_rnn_torch_api(self):
        batch_size, num_steps = 32, 35
        train_iter, vocab = load_data_time_machine(batch_size, num_steps)

        num_hiddens = 256
        rnn_layer = nn.RNN(input_size=len(vocab), hidden_size=num_hiddens)

        state = torch.zeros((1, batch_size, num_hiddens))
        print(state.shape)

        x = torch.rand(size=(num_steps, batch_size, len(vocab)))
        y, state_new = rnn_layer(x, state)
        print(y.shape)
        print(state_new.shape)

        device = dlf.devices()[0]
        net = RNNModelWithTorch(rnn_layer, vocab_size=len(vocab))
        net = net.to(device=device)

        act_output = None
        with patch('sys.stdout', new=StringIO()) as fake_out:
            act_output = torchinfo.summary(net, input_size=(batch_size, num_steps), device=device)
        print(act_output)

        # Inference before training
        print('Inference before training')
        inference('time traveller', 10, net, vocab, device)

        # Training
        num_epochs, learning_rate = 500, 1
        train(net, train_iter, vocab, nn.CrossEntropyLoss(), learning_rate, num_epochs, device)

        # Inference after training
        print('Inference after training')
        inference('time traveller', 10, net, vocab, device)


if __name__ == "__main__":
    unittest.main(verbosity=True)
