#! coding: utf-8

import unittest
import os
from abc import ABC
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import utils.dlf as dlf
from utils.plot import show_heatmaps
from recurrent_neural_network.rnn_utils import (
    Decoder, EncoderDecoder, Seq2SeqEncoder,
    load_data_nmt, train_seq2seq, predict_seq2seq, bleu)


class AttentionDecoder(Decoder, ABC):
    def __init__(self, **kwargs):
        super( AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError


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


def transpose_qkv(x, num_heads):
    # x.shape: (batch_size, num_key_value_pairs, num_hiddens)
    # The output shape: (batch_size, num_key_value_pairs, num_heads, num_hiddens / num_heads)
    x = x.reshape(x.shape[0], x.shape[1], num_heads, -1)

    # The output x.shape: (batch_size, num_heads, num_key_value_pairs, num_hiddens / num_heads)
    x = x.permute(0, 2, 1, 3)

    # The final output shape: (batch_size * num_heads, num_key_value_pairs, num_hiddens / num_heads)
    return x.reshape(-1, x.shape[2], x.shape[3])


def transpose_output(x, num_heads):
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

    def forward(self, queries, keys, values, valid_lens):
        # The shapes of the queries, keys, values are (batch_size, num_key_query_pairs, num_hiddens)
        # The shape of the valid_lens: (batch_size, ) or (batch_size, num_queries)
        # The shape of the queries, keys, values after transformation:
        # (batch_size * num_heads, num_key_query_pairs, num_hiddens / num_heads)
        queries = transpose_qkv(self.w_q(queries), self.num_heads)
        keys = transpose_qkv(self.w_k(keys), self.num_heads)
        values = transpose_qkv(self.w_v(values), self.num_heads)

        if valid_lens is not None:
            # Repeat the first
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        # output.shape: (batch_size * num_heads, num_queries, num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat.shape: (batch_size, num_queries, num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.w_o(output_concat)


class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout: float = 0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = AdditiveAttention(num_hiddens, num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)
        self._attention_weights = []

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # output.shape: (batch_size, num_steps, num_hiddens)
        # hidden_state.shape: (num_layers, batch_size, num_hiddens)
        outputs, hidden_state = enc_outputs
        return outputs.permute(1, 0, 2), hidden_state, enc_valid_lens

    def forward(self, x, state):
        # enc_outputs.shape: (batch_size, num_steps, num_hiddens)
        # hidden_state.shape: (num_layers, batch_size, num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # x.shape: (num_steps, batch_size, embed_size
        x_tokens = self.embedding(x).permute(1, 0, 2)
        outputs = []

        for x in x_tokens:
            # query.shape: (batch_size, 1, num_hiddens)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # context.shape: (batch_size, 1, num_hiddens)
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)
            # Concatenate at the feature dimension
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # reshape the x to (1, batch_size, embed_size + num_hiddens)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)

        # After dense layer, the outputs.shape is (num_steps, batch_size, vocab_size)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights


class IntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model_path = os.path.join(str(Path(__file__).resolve().parent), 'seq2seq_with_attention.pth')
        pass

    def test_attention_decoder(self):
        encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
        encoder.eval()
        decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
        decoder.eval()

        x = torch.zeros((4, 7), dtype=torch.long)  # (batch_size, num_steps)
        state = decoder.init_state(encoder(x), None)
        output, state = decoder(x, state)

        print(output.shape)
        print(len(state), state[0].shape)
        print(len(state[1]), state[1][0].shape)

        self.assertTrue(True)

    def test_training(self):
        embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
        batch_size, num_steps = 64, 10
        lr, num_epochs = 0.005, 250
        device = dlf.devices()[0]

        train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
        encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)

        decoder = Seq2SeqAttentionDecoder(vocab_size=len(tgt_vocab),
                                          embed_size=embed_size,
                                          num_hiddens=num_hiddens,
                                          num_layers=num_layers,
                                          dropout=dropout)

        net = EncoderDecoder(encoder, decoder)

        train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

        torch.save(net, self.model_path)

        self.assertTrue(True)

    def test_inference(self):
        if not os.path.exists(self.model_path):
            self.test_training()

        batch_size, num_steps = 64, 10
        device = dlf.devices()[0]

        _, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)

        net = torch.load(self.model_path, weights_only=False).to(device)

        engs = ['go .', "i lost .", "he\'s calm .", "i\'m home ."]
        fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']

        dec_attention_weight_seq = None  # Depressing warnings only
        for eng, fra in zip(engs, fras):
            translation, dec_attention_weight_seq = predict_seq2seq(
                net, eng, src_vocab, tgt_vocab, num_steps, device, True)
            print(f'{eng} => {translation}, ', f'bleu {bleu(translation, fra, k=2):.3f}')

        attention_weights = torch.cat(
            [step[0][0][0] for step in dec_attention_weight_seq],
            dim=0).reshape((1, 1, -1, num_steps))

        show_heatmaps(attention_weights[:, :, :, :len(engs[-1].split()) + 1].cpu(),
                      xlabel='Key positions', ylabel='Query positions')
        plt.show()

    def test_multi_head_attention(self):
        num_hiddens, num_heads = 100, 5

        attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                       num_hiddens, num_heads, 0.5)
        attention.eval()

        print(attention)

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main(verbosity=True)

