#! coding: utf-8

import unittest
import os
import torch
import torch.nn as nn
from pprint import pprint
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import utils.dlf as dlf
from recurrent_neural_network.rnn_utils import Vocab


dlf.DATA_HUB['fra-eng'] = (dlf.DATA_URL + 'fra-eng.zip', '94646ad1522d915e7b0f9296181140edcf86a4f5')


def read_data_nmt():
    data_dir = dlf.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r', encoding='utf-8') as f:
        return f.read()


def preprocess_nmt(text):
    """Preprocess the English-French dataset."""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # Replace non-breaking space with space, and convert uppercase letters to lowercase ones
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()

    # Insert space between words and punctuation marks
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char for i, char in enumerate(text)]
    return ''.join(out)


def tokenize_nmt(text, num_examples=None):
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


def show_list_len_pair_histogram(list1, list2, list1_name, list2_name, x_label, y_label):
    # _, _, patches = plt.hist([[len(l) for l in list1], [len(l) for l in list2]], bins=range(0, 100, 10), stacked=True)
    _, _, patches = plt.hist([[len(l) for l in list1], [len(l) for l in list2]])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    for patch in patches[1].patches:
        patch.set_hatch('/')

    plt.legend([list1_name, list2_name])
    plt.show()


def truncate_pad(line, num_steps, padding_token):
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))


def build_array_nmt(lines, vocab, num_steps):
    # Convert lines into word indices
    lines = [vocab[l] for l in lines]
    # Add end-of-sequence character
    lines = [l + [vocab['<eos>']] for l in lines]
    # Pad or truncate the sequence to num_steps
    lines = [truncate_pad(l, num_steps, vocab['<pad>']) for l in lines]
    array = torch.tensor(lines)
    # Compute the valid length
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len


def load_data_nmt(batch_size, num_steps, num_examples=600):
    # Return the iterator and the vocabularies of the English and French data sets
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = dlf.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab


class Encoder(nn.Module):
    """ Encoder-Decoder architecture for neural machine translation."""
    
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, x, *args):
        raise NotImplementedError


class Decoder(nn.Module):
    """ Encoder-Decoder architecture for neural machine translation."""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, x, *args):
        raise NotImplementedError


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_x, dec_x, *args):
        enc_outputs = self.encoder(enc_x, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_x, dec_state)


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


if __name__ == "__main__":
    unittest.main(verbosity=True)

