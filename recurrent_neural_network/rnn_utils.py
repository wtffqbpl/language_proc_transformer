#! coding: utf-8

import collections
import math
import random
import re
import torch
import torch.nn as nn
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import utils.dlf as dlf
from utils.accumulator import Accumulator
from utils.timer import Timer


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


class RNNModelScratch:
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params_fn, init_state_fn, forward_fn):
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.params = get_params_fn(vocab_size, num_hiddens, device)
        self.init_state = init_state_fn
        self.forward_fn = forward_fn

    def __call__(self, x, state):
        x = torch.nn.functional.one_hot(x.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(x, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


def inference(prefix, num_preds, net, vocab, device):
    """Generate new characters following the `prefix` """
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]

    get_inputs = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))

    for y in prefix[1:]:  # Warm-up period
        _, state = net(get_inputs(), state)
        outputs.append(vocab[y])

    for _ in range(num_preds):  # Predict `num_preds` steps
        y, state = net(get_inputs(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


def train_epoch(net, train_iter, loss_fn, updater, device, use_random_iter):
    state, timer = None, Timer()
    metric = Accumulator(2)  # training loss, num tokens

    for X, Y in train_iter:
        if state is None or use_random_iter:
            # Initialize `state` when either it is the first iteration or using random sampling
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # `state` is a tensor for `nn.GRU`
                state.detach_()
            else:
                # state is a tuple for `nn.LSTM` and `nn.RNN`
                for s in state:
                    s.detach_()

        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss_fn(y_hat, y.long()).mean()

        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())

    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


def train(net, train_iter, vocab, loss_fn, lr, num_epochs, device, use_random_iter=False):
    """ Train model"""
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: dlf.sgd(net.params, lr, batch_size)

    predict = lambda prefix: inference(prefix, 50, net, vocab, device)

    ppl, speed = None, None
    # training and validation
    for epoch in range(num_epochs):
        ppl, speed = train_epoch(net, train_iter, loss_fn, updater, device, use_random_iter)

        if (epoch + 1) % 10 == 0:
            print(f'epoch {epoch + 1}, ', f'perplexity {ppl:.1f}, ',
                  f'speed {speed:.1f} tokens/sec, {str(device)}, ',
                  predict('time traveller'))

    print(f'perplexity {ppl:.1f}, ', f'speed {speed:.1f} tokens/sec, {str(device)}')
    print(predict("time traveller "))
    print(predict('traveller'))
