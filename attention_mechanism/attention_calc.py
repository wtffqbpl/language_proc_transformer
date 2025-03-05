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


def sequence_mask(x, valid_len, value: float = 0):
    max_len = x.size(1)
    mask = torch.arange(max_len, dtype=torch.float32,
                        device=x.device)[None, :] < valid_len[:, None]
    x[~mask] = value
    return x


def masked_softmax(x: torch.Tensor, valid_lens):
    """ Mask the last dimension before computing the softmax"""
    # X is 3D tensor, valid_lens: 1D or 2D tensor
    if valid_lens is None:
        return nn.functional.softmax(x, dim=-1)
    else:
        shape = x.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        x = sequence_mask(x.reshape(-1, shape[-1]), valid_lens, value=-1e6)

        return nn.functional.softmax(x.reshape(shape), dim=-1)


class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.w_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.w_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None

    def forward(self,
                queries: torch.Tensor,
                keys: torch.Tensor,
                values: torch.Tensor,
                valid_lens: torch.Tensor):
        queries, keys = self.w_q(queries), self.w_k(keys)

        # expand dimensions. The tensors' shapes after expansion are:
        # queries.shape = (batch_size, num_queries, 1, num_hiddens)
        # key.shape = (batch_size, 1, num_key_value_pairs, num_hiddens)
        # Using broadcasting mechanism to elementwise-addition.
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)

        # There is only one output of the self.w_v, so remove the last dim
        # scores.shape = (batch_size, num_queries, num_key_value_pairs)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)

        # The values.shape = (batch_size, num_key_value_pairs,
        return torch.bmm(self.dropout(self.attention_weights), values)


class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None

    def forward(self,
                queries: torch.Tensor,
                keys: torch.Tensor,
                values: torch.Tensor,
                valid_lens: torch.Tensor = None):
        # queries.shape: (batch_size, num_queries, d)
        # keys.shape: (batch_size, num_key_value_pairs, d)
        # values.shape: (batch_size, num_key_value_pairs, value_dimension)
        # valid_lens.shape: (batch_size, ) or (batch_size, num_queries)
        d = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.transpose(1, 2)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


# TODO:General Attention 的核心思想是先对 key 做一个线性变换，再与 query 直接做点积计算得分，其公式为：
# score = Q^T * W * K
class GeneralAttention(nn.Module):
    def __init__(self, num_queries, num_keys, **kwargs):
        super(GeneralAttention, self).__init__(**kwargs)

        self.w = nn.Linear(num_keys, num_queries, bias=False)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask=None):
        if queries.dim() == 2:
            queries = queries.unsqueeze(1)  # (batch_size, 1, num_queries)

        transformed_keys = self.w(keys)  # (batch_size, seq_len, num_queries)
        # compute scores
        scores = torch.bmm(queries, transformed_keys.transpose(1, 2)).squeeze(1)  # (batch_size, seq_len)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, value=-1e9)
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), values).unsqueeze(1)
        return context, attn_weights


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
