#! coding: utf-8

import unittest
import torch
import torch.nn as nn
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from recurrent_neural_network.rnn_utils import load_data_time_machine, RNNModelScratch, RNNModelWithTorch, train
import utils.dlf as dlf


def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    w_xz, w_hz, b_z = three()  # Update gate parameters
    w_xr, w_hr, b_r = three()  # Reset gate parameters
    w_xh, w_hh, b_h = three()  # Candidate hidden state parameters

    w_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    # Attach gradients
    params = [w_xz, w_hz, b_z, w_xr, w_hr, b_r, w_xh, w_hh, b_h, w_hq, b_q]
    for param in params:
        param.requires_grad_(True)

    return params


def init_gru_state(batch_size, num_hiddens, device):
    return torch.zeros((batch_size, num_hiddens), device=device),


def gru(inputs, state, params):
    w_xz, w_hz, b_z, w_xr, w_hr, b_r, w_xh, w_hh, b_h, w_hq, b_q = params
    H, = state

    outputs = []
    for x in inputs:
        z = torch.sigmoid(torch.mm(x, w_xz) + torch.mm(H, w_hz) + b_z)
        r = torch.sigmoid(torch.mm(x, w_xr) + torch.mm(H, w_hr) + b_r)
        h_tilda = torch.tanh(torch.mm(x, w_xh) + torch.mm(r * H, w_hh) + b_h)
        h = z * H + (1 - z) * h_tilda
        y = torch.mm(h, w_hq) + b_q
        outputs.append(y)

    return torch.cat(outputs, dim=0), (H,)


class IntegrationTest(unittest.TestCase):

    def test_gru_model(self):
        device = dlf.devices()[0]
        batch_size, num_steps = 32, 35
        train_iter, vocab = load_data_time_machine(batch_size, num_steps, use_random_iter=True)

        vocab_size, num_hiddens = len(vocab), 256

        num_epochs, learning_rate = 500, 1
        model = RNNModelScratch(len(vocab), num_hiddens, device, get_params, init_gru_state, gru)
        loss_fn = torch.nn.CrossEntropyLoss()

        train(model, train_iter, vocab, loss_fn, learning_rate, num_epochs, device, use_random_iter=True)

        self.assertTrue(True)

    def test_gru_pytorch_api(self):
        device = dlf.devices()[0]
        batch_size, num_steps = 32, 35
        train_iter, vocab = load_data_time_machine(batch_size, num_steps, use_random_iter=True)

        vocab_size, num_hiddens = len(vocab), 256

        num_epochs, learning_rate = 500, 1
        input_size = output_size = vocab_size
        num_layers, output_size, hidden_size = 1, 1, 20

        num_inputs = vocab_size
        gru_layer = nn.GRU(num_inputs, num_hiddens)
        model = RNNModelWithTorch(gru_layer, vocab_size).to(device=device)
        loss_fn = torch.nn.CrossEntropyLoss()

        train(model, train_iter, vocab, loss_fn, learning_rate, num_epochs, device)


if __name__ == '__main__':
    unittest.main()
