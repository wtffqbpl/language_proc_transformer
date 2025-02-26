#! coding: utf-8

import unittest
import torch
import torch.nn as nn
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import utils.dlf as dlf
from recurrent_neural_network.rnn_utils import load_data_time_machine, RNNModelScratch, RNNModelWithTorch, train


def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    w_xi, w_hi, b_i = three()  # Input gate parameters
    w_xf, w_hf, b_f = three()  # Forget gate parameters
    w_xo, w_ho, b_o = three()  # Output gate parameters
    w_xc, w_hc, b_c = three()  # Candidate hidden state parameters

    # Output layer parameters
    w_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    # Attach gradients
    params = [w_xi, w_hi, b_i, w_xf, w_hf, b_f, w_xo, w_ho, b_o, w_xc, w_hc, b_c, w_hq, b_q]
    for param in params:
        param.requires_grad_(True)

    return params


def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))


def lstm(inputs, state, params):
    [w_xi, w_hi, b_i, w_xf, w_hf, b_f, w_xo, w_ho, b_o, w_xc, w_hc, b_c, w_hq, b_q] = params
    (h, c) = state
    outputs = []

    for x in inputs:
        i = torch.sigmoid(torch.mm(x, w_xi) + torch.mm(h, w_hi) + b_i)
        f = torch.sigmoid(torch.mm(x, w_xf) + torch.mm(h, w_hf) + b_f)
        o = torch.sigmoid(torch.mm(x, w_xo) + torch.mm(h, w_ho) + b_o)
        c_tilda = torch.tanh(torch.mm(x, w_xc) + torch.mm(h, w_hc) + b_c)
        c = f * c + i * c_tilda
        h = o * torch.tanh(c)
        y = torch.mm(h, w_hq) + b_q
        outputs.append(y)
    return torch.cat(outputs, dim=0), (h, c)


class IntegrationTest(unittest.TestCase):

    def test_lstm(self):
        batch_size, num_steps = 32, 35
        train_iter, vocab = load_data_time_machine(batch_size, num_steps, use_random_iter=True)

        vocab_size, num_hiddens = len(vocab), 256
        device = dlf.devices()[0]
        num_epochs, learning_rate = 500, 1

        model = RNNModelScratch(len(vocab), num_hiddens, device, get_lstm_params, init_lstm_state, lstm)

        loss_fn = torch.nn.CrossEntropyLoss()
        train(model, train_iter, vocab, loss_fn, lr=learning_rate, num_epochs=num_epochs, device=device)

        self.assertTrue(True)

    def test_lstm_pytorch_api(self):
        batch_size, num_steps = 32, 35
        train_iter, vocab = load_data_time_machine(batch_size, num_steps, use_random_iter=True)

        vocab_size, num_hiddens = len(vocab), 256
        device = dlf.devices()[0]
        num_epochs, learning_rate = 600, 1
        num_inputs = vocab_size

        lstm_layer = nn.LSTM(num_inputs, num_hiddens)

        model = RNNModelWithTorch(lstm_layer, vocab_size).to(device=device)
        loss_fn = torch.nn.CrossEntropyLoss()
        train(model, train_iter, vocab, loss_fn, lr=learning_rate, num_epochs=num_epochs, device=device)

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main(verbosity=True)
