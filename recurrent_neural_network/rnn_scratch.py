#! coding: utf-8

import unittest
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from recurrent_neural_network.rnn_utils import load_data_time_machine, Vocab
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


class RNNModelScratch:
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params_fn, init_state_fn, forward_fn):
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.params = get_params_fn(vocab_size, num_hiddens, device)
        self.init_state = init_state_fn
        self.forward_fn = forward_fn

    def __call__(self, x, state):
        x = F.one_hot(x.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(x, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


def inference(prefix, num_preds, net, vocab, device):
    """Generate new characters following the `prefix` """
    state = net.begin_state(batch_size=1, devic=device)
    outputs = [vocab[prefix[0]]]

    get_inputs = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))

    for y in prefix[1:]:  # Warm-up period
        _, state = net(get_inputs(), state)
        outputs.append(vocab[y])

    for _ in range(num_preds):  # Predict `num_preds` steps
        y, state = net(get_inputs(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


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


if __name__ == "__main__":
    unittest.main(verbosity=True)
