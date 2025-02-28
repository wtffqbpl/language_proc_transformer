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


class Seq2SeqEncoder(Encoder):
    """ The encoder for the sequence-to-sequence model"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, x, *args):
        # The output x.shape = [batch_size, num_steps, embed_size]
        x = self.embedding(x)
        # In RNN model, the first dim should be the time step
        x = x.permute(1, 0, 2)
        output, state = self.rnn(x)
        # The output.shape = [num_steps, batch_size, num_hiddens]
        # The state[0].shape = [num_layers, batch_size, num_hiddens]
        return output, state


class Seq2SeqDecoder(Decoder):
    """ The decoder for the sequence-to-sequence model"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_all_outputs, *args):
        return enc_all_outputs

    def forward(self, x, state):
        # The input x.shape = [batch_size, num_steps]
        # The output x.shape = [num_steps, batch_size, embed_size]
        embs = self.embedding(x.t().to(torch.int32))
        enc_output, hidden_state = state
        # context.shape = [batch_size, num_hiddens]
        context = enc_output[-1]
        # Broadcast context to [num_steps, batch_size, embed_size]
        context = context.repeat(embs.shape[0], 1, 1)
        # Concat at the feature dimension
        embs_and_context = torch.cat((embs, context), -1)
        outputs, hidden_state = self.rnn(embs_and_context, hidden_state)
        outputs = self.dense(outputs).swapaxes(0, 1)
        # outputs.shape = [batch_size, num_steps, vocab_size]
        # hidden_sate.shape = [num_layers, batch_size, num_hiddens]
        return outputs, [enc_output, hidden_state]


def sequence_mask(x, valid_len, value=0):
    max_len = x.size(1)
    mask = torch.arange(max_len, dtype=torch.float32, device=x.device)
    mask = mask[None, :] < valid_len[:, None]
    x[~mask] = value
    return x


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """ Cross entropy with mask and softmax"""
    # The pred.shape = [batch_size, num_steps, vocab_size]
    # the label.shape = [batch_size, num_steps]
    # valid_len.shape = [batch_size]
    def forward(self, pred: torch.Tensor, label: torch.Tensor, valid_len) -> torch.Tensor:
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super().forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


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


if __name__ == "__main__":
    unittest.main(verbosity=True)

