#! coding: utf-8

import unittest
from abc import ABC
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from .attention_utils import (
    MultiHeadAttention,
    Encoder,
    EncoderDecoder,
    PositionalEncoding,
    AttentionDecoder)
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import utils.dlf as dlf
from utils.plot import show_heatmaps
from recurrent_neural_network.rnn_utils import (
    load_data_nmt,
    train_seq2seq,
    predict_seq2seq,
    bleu)


class PositionWiseFFN(nn.Module):
    """Feed-forward network based position"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, x):
        return self.dense2(self.relu(self.dense1(x)))


class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, x, y):
        return self.ln(self.dropout(y) + x)


class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape,
                 ffn_num_input, ffn_num_hiddens, num_heads, dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.add_norm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.add_norm2 = AddNorm(norm_shape, dropout)

    def forward(self, x: torch.Tensor, valid_lens: torch.Tensor):
        y = self.add_norm1(x, self.attention(x, x, x, valid_lens))
        return self.add_norm2(y, self.ffn(y))


class TransformerEncoder(Encoder, ABC):
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        # For visualization only.
        self.attention_weights = None

        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()

        for i in range(num_layers):
            self.blks.add_module(
                'block' + str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))

    def forward(self, x: torch.Tensor, valid_lens: torch.Tensor, *args):
        x = self.pos_encoding(self.embedding(x) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            x = blk(x, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return x


class DecoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape,
                 ffn_num_input, ffn_num_hiddens, num_heads, dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.add_norm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.add_norm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.add_norm3 = AddNorm(norm_shape, dropout)

    def forward(self, x: torch.Tensor, state: torch.Tensor):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # All the output tokens would be processed at the same time during training.
        # So we initialize state[2][self.i] is None
        # The output tokens would be decoded sequentially during inference process.
        # So the state[2][self.i] contains (0~i) time step decoded output info.
        if state[2][self.i] is None:
            key_values = x
        else:
            key_values = torch.cat((state[2][self.i], x), dim=1)

        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = x.shape
            # dec_valid_lens.shape: (batch_size, num_steps)
            # Each line contains [1, 2, ..., num_steps]
            dec_valid_lens = torch.arange(1, num_steps + 1, device=x.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        # Self-attention
        x2 = self.attention1(x, key_values, key_values, dec_valid_lens)
        y = self.add_norm1(x, x2)

        # Encoder-Decoder + attention
        # The enc_outputs.shape: (batch_size, num_steps, num_hiddens)
        y2 = self.attention2(y, enc_outputs, enc_outputs, enc_valid_lens)
        z = self.add_norm2(y, y2)
        return self.add_norm3(z, self.ffn(z)), state


class TransformerDecoder(AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        # For visualization only.
        self._attention_weights = None

        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()

        for i in range(num_layers):
            self.blks.add_module(
                "block" + str(i),
                DecoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape,
                             ffn_num_input, ffn_num_hiddens, num_heads, dropout, i))
            self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs: torch.Tensor, enc_valid_lens: torch.Tensor, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, x: torch.Tensor, state):
        x = self.pos_encoding(self.embedding(x) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            x, state = blk(x, state)

            # Decoder self-attention weights
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            # Encoder-Decoder self-attention weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights

        return self.dense(x), state

    @property
    def attention_weights(self):
        return self._attention_weights


class IntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.transformer_model_path = os.path.join(
            str(Path(__file__).resolve().parent), 'transformer_model.pth')

    def test_position_wise_ffn(self):
        ffn = PositionWiseFFN(4, 4, 8)
        ffn.eval()

        res = ffn(torch.ones((2, 3, 4)))[0]
        self.assertEqual(torch.Size([3, 8]), res.shape)

    def test_transformer_encoder_block(self):
        x = torch.ones((2, 100, 24))
        valid_lens = torch.tensor([3, 2])
        encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
        encoder_blk.eval()
        res = encoder_blk(x, valid_lens)

        self.assertEqual(torch.Size([2, 100, 24]), res.shape)

    def test_transformer_encoder(self):
        valid_lens = torch.tensor([3, 2])
        encoder = TransformerEncoder(200, 24, 24, 24, 24, [100, 24], 24, 48, 8, 2, 0.5)
        encoder.eval()

        res = encoder(torch.ones((2, 100), dtype=torch.long), valid_lens)

        self.assertEqual(torch.Size([2, 100, 24]), res.shape)

    def test_transformer_decoder_block(self):
        x = torch.ones((2, 100, 24))
        valid_lens = torch.tensor([3, 2])
        encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)

        decoder_blk = DecoderBlock(24, 24, 24, 24, [100, 24],
                                   24, 48, 8, 0.5, 0)
        decoder_blk.eval()
        x = torch.ones((2, 100, 24))
        state = [encoder_blk(x, valid_lens), valid_lens, [None]]
        res = decoder_blk(x, state)[0]

        self.assertEqual(torch.Size([2, 100, 24]), res.shape)

    def test_transformer_training(self):
        num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
        lr, num_epochs = 0.005, 500
        device = dlf.devices()[0]
        ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
        key_size, query_size, value_size = 32, 32, 32
        norm_shape = [32]

        train_iter, src_vocab, target_vocab = load_data_nmt(batch_size, num_steps)
        encoder = TransformerEncoder(
            len(src_vocab), key_size, query_size, value_size, num_hiddens, norm_shape,
            ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout)
        decoder = TransformerDecoder(
            len(target_vocab), key_size, query_size, value_size, num_hiddens, norm_shape,
            ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout)

        net = EncoderDecoder(encoder, decoder)
        train_seq2seq(net, train_iter, lr, num_epochs, target_vocab, device)

        torch.save(net, self.transformer_model_path)

    def test_transformer_inference(self):
        if not os.path.exists(self.transformer_model_path):
            self.test_transformer_training()

        num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
        num_heads = 4
        device = dlf.devices()[0]

        train_iter, src_vocab, target_vocab = load_data_nmt(batch_size, num_steps)

        net = torch.load(self.transformer_model_path, weights_only=False).to(device)

        engs = ['go .', "i lost .", "he\'s calm .", "i\'m home ."]
        fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']

        translation, dec_attention_weight_seq = None, None
        for eng, fra in zip(engs, fras):
            translation, dec_attention_weight_seq = predict_seq2seq(
                net, eng, src_vocab, target_vocab, num_steps, device, True)
            print(f'{eng} => {translation}, ',
                  f'bleu {bleu(translation, fra, k=2):.3f}')

        enc_attention_weights = torch.cat(
            net.encoder.attention_weights, dim=0).reshape((num_layers, num_heads, -1, num_steps))
        print(enc_attention_weights.shape)
        show_heatmaps(
            enc_attention_weights.cpu(), xlabel='Key positions',
            ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
            figsize=(7, 4))
        plt.show()

        dec_attention_weight_2d = [head[0].tolist()
                                   for step in dec_attention_weight_seq
                                   for attn in step for blk in attn for head in blk]
        dec_attention_weights_filled = torch.tensor(
            pd.DataFrame(dec_attention_weight_2d).fillna(0.0).values)
        dec_attention_weights = dec_attention_weights_filled.reshape((-1, 2, num_layers, num_heads, num_steps))
        dec_self_attention_weights, dec_inter_attention_weights = dec_attention_weights.permute(1, 2, 3, 0, 4)

        show_heatmaps(dec_self_attention_weights[:, :, :, :len(translation.split()) + 1],
                      xlabel='Key positions', ylabel='Query positions',
                      titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 4))
        plt.show()

        show_heatmaps(
            dec_inter_attention_weights, xlabel='Key positions',
            ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
            figsize=(7, 4))
        plt.show()


if __name__ == '__main__':
    unittest.main(verbosity=True)
