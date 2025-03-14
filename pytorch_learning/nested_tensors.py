#! coding: utf-8

import unittest
import numpy as np
import timeit
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import utils.dlf as dlf


torch.manual_seed(1)
np.random.seed(1)


# PyTorch Nested Tensor Tutorials
# https://pytorch.org/tutorials/prototype/nestedtensor.html


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            e_q: int,
            e_k: int,
            e_v: int,
            e_total: int,
            num_heads: int,
            dropout_p: float = 0.0,
            bias=True,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}

        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dropout_p = dropout_p

        self._qkv_same_embed_dim = e_q == e_k and e_q == e_v

        if self._qkv_same_embed_dim:
            self.packed_proj = nn.Linear(e_q, e_total * 3, bias=bias, **factory_kwargs)
        else:
            self.query_proj = nn.Linear(e_q, e_total, **factory_kwargs)
            self.key_proj = nn.Linear(e_k, e_total, **factory_kwargs)
            self.value_proj = nn.Linear(e_v, e_total, **factory_kwargs)
        e_out = e_q
        self.out_proj = nn.Linear(e_total, e_out, **factory_kwargs)
        assert e_total % num_heads == 0, "Embedding dim is not divisible by num_heads"
        self.e_head = e_total // num_heads
        self.bias = bias

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_mask=None,
            is_causal=False,
    ) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection
        :param query: (torch.Tensor) query of shape (N, L_t E_q)
        :param key: (torch.Tensor) key of shape (N, L_s, E_k)
        :param value: (torch.Tensor) value of shape (N, L_s, E_v)
        :param attn_mask: (torch.Tensor, optional) attention mask of shape (N, L_q, L_kv) to pass to SDPA.
                            Default is None
        :param is_causal: (bool, optional) Whether to apply causal mask. Default: False
        :return:
            attn_output (torch.Tensor): output of shape (N, L_t, E_q)
        """
        # Step 1: Apply input projection
        if self._qkv_same_embed_dim:
            if query is key and key is value:
                result = self.packed_proj(query)
                query, key, value = torch.chunk(result, 3, dim=-1)
            else:
                q_weight, k_weight, v_weight = torch.chunk(self.packed_proj.weight, 3, dim=0)
                if self.bias:
                    q_bias, k_bias, v_bias = torch.chunk(self.packed_proj.bias, 3, dim=0)
                else:
                    q_bias, k_bias, v_bias = None, None, None
                query, key, value = (
                    F.linear(query, q_weight, q_bias),
                    F.linear(key, k_weight, k_bias),
                    F.linear(value, v_weight, v_bias)
                )
        else:
            query = self.query_proj(query)
            key = self.key_proj(key)
            value = self.value_proj(value)

        # Step 2: Split heads and prepare for SDPA
        # reshape query, key, value to separate for SDPA
        # (N, L_t, E_total) -> (N, L_t, num_heads, e_heads) -> (N, num_heads, L_t, E_heads)
        query = query.unflatten(-1, [self.num_heads, self.e_head]).transpose(1, 2)

        # (N, L_s, E_total) -> (N, L_s, num_heads, E_head) -> (N, num_heads, L_s, E_head)
        key = key.unflatten(-1, [self.num_heads, self.e_head]).transpose(1, 2)

        # (N, L_s, E_total) -> (N, L_s, num_heads, E_head) -> (N, num_heads, L_s, E_head)
        value = value.unflatten(-1, [self.num_heads, self.e_head]).transpose(1, 2)

        # Step 3: Run SDPA
        # (N, num_heads, L_t, E_head)
        attn_output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout_p, is_causal=is_causal)

        # (N, num_heads, L_t, e_heads) -> (N, L_t, num_heads, E_heads) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 4: Apply output projection
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output


def zipf_sentence_lengths(alpha: float, batch_size: int) -> torch.Tensor:
    # Generate fake corpus by unigram Zipf distribution
    # from wikitext-2 corpus, we get rank "." = 3, "!" = 386, "?" = 858
    sentence_lengths = np.empty(batch_size, dtype=int)
    for ibatch in range(batch_size):
        sentence_lengths[ibatch] = 1
        word = np.random.zipf(alpha)
        while word != 3 and word != 386 and word != 858:
            sentence_lengths[ibatch] += 1
            word = np.random.zipf(alpha)
    return torch.tensor(sentence_lengths)


# Generate a batch of semi-realistic data using Zipf distribution for sentence lengths
# in the form of nested tensors with the jagged layout.
def gen_batch(n, e_q, e_k, e_v, device, dtype=torch.float32, query_seq_len_1=False):
    # Generate semi-realistic data using Zipf distribution for sentence lengths
    sentence_lengths = zipf_sentence_lengths(alpha=1.2, batch_size=n)

    # Note: the torch.jagged layout is a nested tensor layout that supports a single ragged.
    # dimension and works with torch.compile. The batch items each have shape (B, S*, D)
    # where B = batch size, S* = ragged sequence length, and D = embedding dimension

    if query_seq_len_1:
        query = torch.nested.nested_tensor(
            [torch.randn(1, e_q, dtype=dtype, device=device) for l in sentence_lengths],
            layout=torch.jagged)
    else:
        query = torch.nested.nested_tensor([
            torch.randn(l.item(), e_q, device=device)
            for l in sentence_lengths], layout=torch.jagged)

    key = torch.nested.nested_tensor([
        torch.randn(s.item(), e_k, device=device)
        for s in sentence_lengths], layout=torch.jagged)

    value = torch.nested.nested_tensor([
        torch.randn(s.item(), e_v, device=device)
        for s in sentence_lengths], layout=torch.jagged)

    return query, key, value, sentence_lengths


# Generate padded forms of query, key, value for comparison
def jagged_to_padded(jt, padding_val):
    return torch.nested.to_padded_tensor(
        torch.nested.nested_tensor(list(jt.unbind())),
        padding_val)


# Check correctness and performance
def benchmark(func, *args, **kwargs):
    begin = timeit.default_timer()
    output = func(*args, **kwargs)
    end = timeit.default_timer()
    return output, end - begin


class IntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.device = dlf.devices('cpu')[0]

    def test_nested_tensor_simple(self):
        # From the Python frontend, a nested tensor can be created from a list of tensors. We denote
        # nt[i] as the ith tensor component of a nested-tensor.

        nt = torch.nested.nested_tensor([torch.arange(12).reshape(2, 6),
                                         torch.arange(start=18, end=18 + 18).reshape(3, 6)],
                                        dtype=torch.float32, device=self.device)

        # By padding every underlying tensor to the same shape, a nestedtensor can be
        # converted to a regular tensor
        padded_out_tensor = torch.nested.to_padded_tensor(nt, padding=0.0)
        print(f'{padded_out_tensor=}')

        # All tensors posses an attribute for determining if they are nested.
        print(f'nt is nested: {nt.is_nested}')
        print(f'padded_out_tensor is nested: {padded_out_tensor.is_nested}')
        self.assertTrue(nt.is_nested)
        self.assertFalse(padded_out_tensor.is_nested)

        # It is common to construct nestedtensors from batches of irregularly shaped tensors. i.e. dimension 0
        # is assumed to be the batch dimension. Indexing dimension 0 gives back the first underlying
        # tensor component.
        print('First underlying tensor component: ', nt[0], sep='\n')
        print('last column of 2nd underlying tensor component: ', nt[1, :, -1], sep='\n')

        # An important note is that slicing in dimension 0 has not been supported yet. Which means it not
        # currently possible to construct a view that combines the underlying tensor components.

    def test_nested_tensor_operations(self):
        padded_stentences_for_softmax = torch.tensor([[1.0, 2.0, float('-inf')],
                                                      [3.0, 4.0, 5.0]])
        print(F.softmax(padded_stentences_for_softmax, -1))

        nested_sentences = torch.nested.nested_tensor(([torch.tensor([1.0, 2.0]),
                                                        torch.tensor([3.0, 4.0, 5.0])]))
        print(f'{nested_sentences=}')
        print(F.softmax(nested_sentences, -1))

        self.assertTrue(True)

    def test_nested_tensor_multi_head_attention(self):
        n = 512
        e_q, e_k, e_v, e_total = 512, 512, 512, 512
        e_out = e_q
        num_heads = 8
        dropout_p = 0.0
        device = dlf.devices('cpu')[0]

        query, key, value, sentence_lengths = gen_batch(n, e_q, e_k, e_v, device)

        padded_query, padded_key, padded_value = (
            jagged_to_padded(t, 0.0) for t in (query, key, value))

        # Construct the model
        mha = MultiHeadAttention(e_q, e_k, e_v, e_total, num_heads, dropout_p).to(device=device)

        # Check performance
        output_nested, time_nested = benchmark(mha, query, key, value)
        output_padded, time_padded = benchmark(mha, padded_query, padded_key, padded_value)

        # Padding-specific step: remove output projection bias from padded entries for fair comparison
        for i, entry_length in enumerate(sentence_lengths):
            output_padded[i, entry_length:] = 0.0

        print('=== without torch.compile ===')
        print('nested and padded calculations differ by',
              (jagged_to_padded(output_nested, 0.0) - output_padded).abs().max().item())
        print('nested tensor multi-head attention takes', time_nested, 'seconds')
        print('padded tensor multi-head attention takes', time_padded, 'seconds')

        # warm up compile first...
        compiled_mha = torch.compile(mha)
        compiled_mha(query, key, value)
        # ... now benchmark
        compiled_output_nested, compiled_time_nested = benchmark(
            compiled_mha, query, key, value)

        # warm up compile first
        compiled_mha(padded_query, padded_key, padded_value)
        # ...now benchmark
        compiled_output_padded, compiled_time_padded = benchmark(
            compiled_mha, padded_query, padded_key, padded_value)

        # padding-specific step: remove output projection bias from padded entries
        # for fair comparison
        for i, entry_length in enumerate(sentence_lengths):
            compiled_output_padded[i, entry_length:] = 0.0

        print('=== with torch.compile ===')
        print('nested and padded calculations differ by',
              (jagged_to_padded(compiled_output_nested, 0.0) - compiled_output_padded).abs().max().item())

        print('nested tensor multi-head attention takes', compiled_time_nested, 'seconds')
        print('padded tensor multi-head attention takes', compiled_time_padded, 'seconds')

        print(f'Nested speedup: {compiled_time_padded / compiled_time_nested:.3f}')

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main(verbosity=True)
