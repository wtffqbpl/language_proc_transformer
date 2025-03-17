#! coding: utf-8

import unittest
import torch
import torch.nn as nn
from .attention_utils import EncoderBlock


def get_tokens_and_segments(tokens_a, tokens_b=None):
    """ Get tokens of the BERT input sequence and their segment IDs. """
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0 and 1 are marking segment A and B, respectively
    segments = [0] * (len(tokens_a) + 2)  # with two extra tokens: <cls> and <sep>
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)

    return tokens, segments


class BERTEncoder(nn.Module):
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()

        for i in range(num_layers):
            self.blks.add_module(f'{i}', EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input,
                ffn_num_hiddens, num_heads, dropout, use_bias=True))
        # In BERT, positional embeddings are learnable, thus we create parameter
        # of positional embeddings that are long enough
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))

    def forward(self, tokens: torch.Tensor, segments: torch.Tensor, valid_lens: torch.Tensor):
        # Shape of `x` remains unchanged in the following code snippet:
        # (batch_size, max_sequence_length, num_hiddens)
        x = self.token_embedding(tokens) + self.segment_embedding(segments)
        x = x + self.pos_embedding.data[:, :x.shape[1], :]

        for blk in self.blks:
            x = blk(x, valid_lens)
        return x


# masked Language Modeling
# To encode context bidirectionally for representing each token, BERT randomly masks tokens and
# uses tokens from the bidirectional context to predict the masked tokens in a self-su
class MaskLM(nn.Module):
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, x, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = x.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # Let
        return x
        pass


class IntegrationTest(unittest.TestCase):
    def test_bert_encoder(self):
        # Suppose that the vocabulary size is 10000. To demonstrate forward inference
        # of BERTEncoder, let's create an instance of it and initialize its parameters.
        vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
        norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
        encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                              ffn_num_hiddens, num_heads, num_layers, dropout)

        tokens = torch.randint(0, vocab_size, (2, 8))
        segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
        encoded_x = encoder(tokens, segments, None)
        self.assertEqual(torch.Size([2, 8, num_hiddens]), encoded_x.shape)

        # Depress warning
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main(verbosity=True)
