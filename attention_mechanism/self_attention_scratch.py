#! coding: utf-8

import unittest
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from .attention_utils import DotProductAttention
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.plot import plot, show_heatmaps
from .attention_utils import PositionalEncoding


class IntegrationTest(unittest.TestCase):
    def test_sin_cos_positional_encoding(self):
        encoding_dim, num_steps = 32, 60
        pos_encoding = PositionalEncoding(encoding_dim, 0)
        pos_encoding.eval()

        x = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
        p = pos_encoding.p[:, :x.shape[1], :]
        plot(torch.arange(num_steps), p[0, :, 6:10].T, xlabel='Row (position)',
             figsize=(6, 2.5), legend=['Col %d' % d for d in torch.arange(6, 10)])
        plt.show()

        p = pos_encoding.p[0, 0:60, :].unsqueeze(0).unsqueeze(0)
        show_heatmaps(p, xlabel='Column (encoding dimension)',
                      ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
        plt.show()

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main(verbosity=True)
