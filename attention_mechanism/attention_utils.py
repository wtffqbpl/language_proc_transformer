#! coding: utf-8

import torch
import torch.nn as nn
import math


def sequence_mask(x: torch.Tensor, valid_len: torch.Tensor, value: float = 0):
    max_len = x.size(1)
    mask = torch.arange(max_len, dtype=torch.float32, device=x.device)[None, :] < valid_len[:, None]
    x[~mask] = value
    return x


def masked_softmax(x: torch.Tensor, valid_lens):
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
        # values.shape: (batch_size, num_key_value_pairs, valid_dimension)
        # valid_lens.shape: (batch_size, ) or (batch_size, num_queries)
        d = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.transpose(1, 2)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


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
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)

        scores = self.w_v(features).squeeze(-1)
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
            # The queries.shape: (batch_size, 1, num_queries)
            queries = queries.unsqueeze(1)

        transformed_keys = self.w(keys)  # (batch_size, seq_len, num_queries)
        # compute scores. The scores.shape: (batch_size, seq_len)
        scores = torch.bmm(queries, transformed_keys.transpose(1, 2)).squeeze(1)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, value=-1e9)
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), values).unsqueeze(1)
        return context, attn_weights


def transpose_qkv(x: torch.Tensor, num_heads: int):
    # x.shape: (batch_size, num_key_value_pairs, num_hiddens)
    # The output shape: (batch_size, num_key_value_pairs, num_heads, num_hiddens / num_heads)
    x = x.reshape(x.shape[0], x.shape[1], num_heads, -1)

    # The output.shape: (batch_size, num_heads, num_key_value_pairs, num_hiddens / num_heads)
    x = x.permute(0, 2, 1, 3)

    # output.shape: (batch_size * num_heads, num_key_value_pairs, num_hiddens / num_heads)
    return x.reshape(-1, x.shape[2], x.shape[3])


def transpose_output(x: torch.Tensor, num_heads):
    # Reverse the transpose_qkv operations
    x = x.reshape(-1, num_heads, x.shape[1], x.shape[2])
    x = x.permute(0, 2, 1, 3)

    return x.reshape(x.shape[0], x.shape[1], -1)


class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)

        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)

        self.w_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.w_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.w_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.w_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self,
                queries: torch.Tensor,
                keys: torch.Tensor,
                values: torch.Tensor,
                valid_lens: torch.Tensor):
        # The shape of queries, keys and values is (batch_size, num_key_value_pairs, num_hiddens)
        # valid_lens.shape: (batch_size, ) or (batch_size, num_queries)
        # The shape of queries, keys and values after transformation is:
        # (batch_size * num_heads, num_key_value_pairs, num_hiddens / num_heads)
        queries = transpose_qkv(self.w_q(queries), self.num_heads)
        keys = transpose_qkv(self.w_k(keys), self.num_heads)
        values = transpose_qkv(self.w_v(values), self.num_heads)

        if valid_lens is not None:
            # Repeat on the first dim
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        # output.shape: (batch_size * num_heads, num_queries, num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat.shape: (batch_size, num_queries, num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.w_o(output_concat)
