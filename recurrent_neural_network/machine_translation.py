#! coding: utf-8

import unittest
import os
import math
import collections
import torch
import torch.nn as nn
from pprint import pprint
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import utils.dlf as dlf
from utils.timer import Timer
from utils.accumulator import Accumulator
from recurrent_neural_network.rnn_utils import (
    Vocab, EncoderDecoder, Seq2SeqEncoder, Seq2SeqDecoder, MaskedSoftmaxCELoss,
    truncate_pad, preprocess_nmt, tokenize_nmt,
    load_data_nmt, read_data_nmt,
    train_seq2seq, predict_seq2seq,
    sequence_mask, bleu)


def show_list_len_pair_histogram(list1, list2, list1_name, list2_name, x_label, y_label):
    # _, _, patches = plt.hist([[len(l) for l in list1], [len(l) for l in list2]], bins=range(0, 100, 10), stacked=True)
    _, _, patches = plt.hist([[len(l) for l in list1], [len(l) for l in list2]])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    for patch in patches[1].patches:
        patch.set_hatch('/')

    plt.legend([list1_name, list2_name])
    plt.show()


class IntegrationTest(unittest.TestCase):
    def test_read_data(self):
        raw_text = read_data_nmt()
        print(raw_text[:75])

        text = preprocess_nmt(raw_text)
        print(text[:80])

        source, target = tokenize_nmt(text)
        for src, tgt in zip(source[:6], target[:6]):
            print(src, tgt)

        show_list_len_pair_histogram(source, target,
                                     'source', 'target',
                                     'number of tokens per sentence', 'count')

        src_vocab = Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
        print(f'len(src_vocab)={len(src_vocab)}')

        print(truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>']))

        train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)

        for x, x_valid_len, y, y_valid_len in train_iter:
            print('x:', x, '\nvalid lengths for x:', x_valid_len)
            print('y:', y, '\nvalid lengths for y:', y_valid_len)
            break

    def test_encoder(self):
        vocab_size, embed_size, num_hiddens, num_layers = 10, 8, 16, 2
        encoder = Seq2SeqEncoder(vocab_size=vocab_size,
                                 embed_size=embed_size,
                                 num_hiddens=num_hiddens,
                                 num_layers=num_layers)
        encoder.eval()
        batch_size = 4
        x = torch.zeros(size=(4, 7), dtype=torch.long)
        output, state = encoder(x)

        self.assertEqual(torch.Size([7, batch_size, num_hiddens]), output.shape)
        self.assertEqual(torch.Size([num_layers, batch_size, num_hiddens]), state.shape)

    def test_decoder(self):
        vocab_size, embed_size, num_hiddens, num_layers = 10, 8, 16, 2
        encoder = Seq2SeqEncoder(vocab_size, embed_size, num_hiddens, num_layers)
        decoder = Seq2SeqDecoder(vocab_size, embed_size, num_hiddens, num_layers)

        batch_size, num_steps = 4, 7
        x = torch.zeros(size=(batch_size, num_steps), dtype=torch.long)
        state = decoder.init_state(encoder(x))
        dec_outputs, state = decoder(x, state)

        self.assertEqual(torch.Size([batch_size, num_steps, vocab_size]), dec_outputs.shape)
        self.assertEqual(torch.Size([num_steps, batch_size, num_hiddens]), state[0].shape)
        self.assertEqual(torch.Size([num_layers, batch_size, num_hiddens]), state[1].shape)

    def test_sequence_mask(self):
        x = torch.tensor([[1, 2, 3], [4, 5, 6]])
        indices = torch.tensor([1, 2])

        x = sequence_mask(x, indices)

        x_target = torch.tensor([[1, 0, 0], [4, 5, 0]])
        self.assertTrue(torch.equal(x_target, x))

        x = torch.ones(size=(2, 3, 4))
        x_out = sequence_mask(x, indices, value=-1)

    def test_cross_entropy_with_mask(self):
        loss = MaskedSoftmaxCELoss()
        ret = loss(
            torch.ones(3, 4, 10),
            torch.ones((3, 4), dtype=torch.long),
            torch.tensor([4, 2, 0])
        )
        print(ret)
        self.assertTrue(True)

    def test_training(self):
        embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
        batch_size, num_steps = 64, 10
        lr, num_epochs = 0.005, 300
        device = dlf.devices()[0]

        train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
        encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
        decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
        net = EncoderDecoder(encoder, decoder)

        train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

        engs = ['go .', "i lost .", "he\'s calm .", "i\'m home ."]
        fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']

        for eng, fra in zip(engs, fras):
            translation, attention_weight_seq = predict_seq2seq(
                net, eng, src_vocab, tgt_vocab, num_steps, device)
            print(f'{eng} => {translation}, ',
                  f'bleu {bleu(translation, fra, k=2):.3f}')


if __name__ == "__main__":
    unittest.main(verbosity=True)

