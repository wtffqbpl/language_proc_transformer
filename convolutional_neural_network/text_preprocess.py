#! coding: utf-8

import unittest
import collections
import random
import re
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import utils.dlf as dlf


# Typical preprocessing pipelines execute the following steps:
#  1. Load text as strings into memory
#  2. Split the strings into tokens (e.g., words or characters)
#  3. Build a vocabulary dictionary to associate each vocabulary element with a numerical index.
#  4. Convert the text into sequences of numerical indices.


class TimeMachine:
    def __init__(self):
        dlf.DATA_HUB['time_machine'] = (
            dlf.DATA_URL + 'timemachine.txt',
            '090b5e7e70c295757f55df93cb0a180b9691891a')

    def download(self):
        with open(dlf.download('time_machine'), 'r') as f:
            contents = f.readlines()
        return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in contents]


class Tokenizer:
    def __init__(self, token: str = 'word') -> None:
        expected_tokens = ['word', 'char']
        assert token in expected_tokens
        self._token = token

    def tokenize(self, lines):
        if self._token == 'word':
            return [line.split() for line in lines]
        elif self._token == 'char':
            return [list(line) for line in lines]
        else:
            print("Error: unexpected token: %s" % self._token)
            raise KeyError


# Construct a vocabulary for our dataset, converting the sequence of strings into
# a list of numerical indices. Note that we have not lost any information and can
# easily convert our dataset back to its original (string) representation.
class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []

        # Flatten a 2D list if needed
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # Count token frequencies
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        # The list of unique tokens
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

        for token, freq in self.token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']


def load_corpus_time_machine(max_tokens=-1):
    lines = TimeMachine().download()
    tokens = Tokenizer(token='char').tokenize(lines)
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


class IntegrationTest(unittest.TestCase):
    def test_simple(self):
        time_machine = TimeMachine()
        lines = time_machine.download()
        print(f'# 文本总行数: {len(lines)}')
        print(lines[0])
        print(lines[10])
        self.assertEqual(3221, len(lines))

    def test_tokenizer(self):
        lines = TimeMachine().download()
        tokens = Tokenizer().tokenize(lines)
        for i in range(11):
            print(tokens[i])

        self.assertTrue(True)

    def test_vocab(self):
        lines = TimeMachine().download()
        tokens = Tokenizer().tokenize(lines)
        vocab = Vocab(tokens)
        indices = vocab[tokens[:10]]
        for i in [0, 10]:
            print('indices: ', tokens[i])
            print('words: ', vocab[tokens[i]])

        self.assertTrue(True)

    def test_time_machine(self):
        corpus, vocab = load_corpus_time_machine()
        print(len(corpus), len(vocab))

        self.assertEqual(170580, len(corpus))
        self.assertEqual(28, len(vocab))


if __name__ == "__main__":
    unittest.main(verbosity=True)
