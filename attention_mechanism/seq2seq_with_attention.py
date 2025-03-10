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
from attention_mechanism.attention_utils import (
    masked_softmax,
    transpose_qkv,
    transpose_output,
    DotProductAttention,
    AdditiveAttention,
    MultiHeadAttention)

from recurrent_neural_network.rnn_utils import (
    Decoder, EncoderDecoder, Seq2SeqEncoder,
    load_data_nmt, train_seq2seq, predict_seq2seq, bleu)


class AttentionDecoder(Decoder, ABC):
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError


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

