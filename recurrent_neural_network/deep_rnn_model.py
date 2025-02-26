#! coding: utf-8

import unittest
import torch
import torch.nn as nn
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from recurrent_neural_network.rnn_utils import load_data_time_machine, RNNModelScratch, RNNModelWithTorch, train
import utils.dlf as dlf


class IntegrationTest(unittest.TestCase):
    def test_deep_rnn_model(self):
        batch_size, num_steps = 32, 35
        train_iter, vocab = load_data_time_machine(batch_size, num_steps)

        vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
        num_inputs = vocab_size
        device = dlf.devices()[0]
        lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
        model = RNNModelWithTorch(lstm_layer, vocab_size).to(device)
        loss_fn = torch.nn.CrossEntropyLoss()

        num_epochs, lr = 500, 1
        train(model, train_iter, vocab, loss_fn, lr, num_epochs, device)

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main(verbosity=True)
