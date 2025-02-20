#! coding: utf-8

import collections
import random
import re
import torch
import torch.nn as nn
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
                # Use break since tokens are ordered. If one is below threshold, the rest will be too low.
                break

            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                # We add the token to the dictionary and update the index sequentially, so the
                # current token index is always `len(self.idx_to_token) - 1`.
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            # If `tokens` is not a valid key, return the index for the unknown token.
            # The dict().get() method can be used to specify a default value (the second parameters).
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


def seq_data_iter_random(corpus, batch_size, num_steps):
    """
    Use random sampling method to generate a batch of sequences
    :param corpus:
    :param batch_size:
    :param num_steps:
    :return:
    """
    # Crop a random starting sequence, the random sequence length is [0, num_steps - 1]
    corpus = corpus[random.randint(0, num_steps - 1):]
    # We should consider the label, so we minus 1
    num_subseqs = (len(corpus) - 1) // num_steps

    # Calculate the starting indices for each subsequence
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # In random sampling, the subsequences from two adjacent random minibatches are
    # not necessarily adjacent on the original sequence.
    random.shuffle(initial_indices)

    def data(pos):
        # Return a sequence of length `num_steps` starting from `pos`
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # Here, `initial_indices` contains randomized starting indices for subsequences
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        x = [data(j) for j in initial_indices_per_batch]
        y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(x), torch.tensor(y)


def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """
    Use sequential sampling method to generate a batch of sequences
    :param corpus:
    :param batch_size:
    :param num_steps:
    :return:
    """
    # Start from a random offset to avoid starting at the same position
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    x = torch.tensor(corpus[offset: offset + num_tokens])
    y = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])

    x = x.reshape(batch_size, -1)
    y = y.reshape(batch_size, -1)

    num_batches = x.shape[1] // num_steps
    for i in range(0, num_batches * num_steps, num_steps):
        x_batch = x[:, i: i + num_steps]
        y_batch = y[:, i: i + num_steps]
        yield x_batch, y_batch


class SeqDataLoader:
    """
    Load sequence data in mini-batches
    """
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    """ Load the time machine dataset """
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


def grad_clipping(net, theta) -> None:
    """ Clipping gradients """
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
