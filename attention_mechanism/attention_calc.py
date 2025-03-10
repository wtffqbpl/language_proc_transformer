#! coding: utf-8

import unittest
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.plot import show_heatmaps
from attention_mechanism.attention_utils import (
    masked_softmax,
    DotProductAttention,
    AdditiveAttention,
    GeneralAttention)


class IntegrationTest(unittest.TestCase):
    def test_masked_softmax(self):
        res = masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3]))
        print(res)

        self.assertEqual(torch.Size([4, 4]), res.shape)

    def test_additive_attention(self):
        queries, keys = torch.normal(0, 1, size=(2, 1, 20)), torch.ones(size=(2, 10, 2))
        values = torch.arange(40, dtype=torch.float32).reshape((1, 10, 4)).repeat(2, 1, 1)
        valid_lens = torch.tensor([2, 6])

        attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
        attention.eval()

        res = attention(queries, keys, values, valid_lens)
        
        self.assertEqual(torch.Size([2, 1, 4]),res.shape)

        print(res)

        show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                      xlabel='Keys', ylabel='Queries', figsize=(4, 4))
        plt.show()

    def test_scaled_dot_product_attention(self):
        keys = torch.ones(size=(2, 10, 2))
        values = torch.arange(40, dtype=torch.float32).reshape((1, 10, 4)).repeat(2, 1, 1)
        queries = torch.normal(0, 1, (2, 1, 2))
        valid_lens = torch.tensor([2, 6])

        attention = DotProductAttention(dropout=0.5)
        attention.eval()
        res = attention(queries, keys, values, valid_lens)

        print(res)

        show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                      xlabel='Keys', ylabel='Queries', figsize=(4, 4))
        plt.show()

        self.assertTrue(True)

    def test_general_attention_scores(self):
        batch_size, seq_len = 2, 5
        query_dim, key_dim, value_dim = 16, 20, 32

        queries = torch.rand(batch_size, query_dim)
        keys = torch.rand(batch_size, seq_len, key_dim)
        values = torch.rand(batch_size, seq_len, value_dim)

        general_attention = GeneralAttention(query_dim, key_dim)
        context, attn_weights = general_attention(queries, keys, values)

        print('General attention context shape: ', context.shape)
        print('Attention weights shape: ', attn_weights.shape)

        show_heatmaps(context.reshape((1, 1, batch_size, value_dim)),
                      xlabel='Keys', ylabel='Queries', figsize=(4, 4))
        plt.show()
        
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main(verbosity=True)
