#! coding: utf-8

import os
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
                 dropout: float = 0, **kwargs):
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
                 dropout: float = 0, **kwargs):
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


class RNNModelWithTorch(nn.Module):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModelWithTorch, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size

        # If RNN is bidirectional, num_directions should be 2, else it should be 1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, vocab_size)

    def forward(self, inputs, state):
        x = torch.nn.functional.one_hot(inputs.T.long(), self.vocab_size).type(torch.float32)
        y, state = self.rnn(x, state)

        # The shape of `y` is (num_steps, batch_size, num_hiddens).
        # Here we only use the output of the last time step
        output = self.linear(y.reshape(-1, y.shape[-1]))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # `nn.GRU` takes a tensor as hidden state
            return torch.zeros(
                (self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens),
                device=device)
        else:
            # `nn.LSTM` takes a tuple of hidden states
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNModel, self).__init__()

        # Using nn.RNN to construct recurrent neural network.
        # batch_first=True means that the input shape is [batch, seq_length, feature_dim]
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # The dense layer, mapping the RNN output to the corresponding target output dims
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # The x.shape = [batch_size, seq_length, input_size]
        out, hidden = self.rnn(x)
        # Use the result of the last time step, and use the dense layer to get the last output
        out = self.fc(out[:, -1, :])
        return out


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


def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if 'weight' in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()

    metric = Accumulator(2)
    timer = Timer()
    for epoch in range(num_epochs):
        for batch in data_iter:
            optimizer.zero_grad()
            x, x_valid_len, y, y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, y[:, :-1]], 1)  # Forcing learning
            y_hat, _ = net(x, dec_input, x_valid_len)
            l = loss(y_hat, y, y_valid_len)
            l.sum().backward()
            grad_clipping(net, 1)
            num_tokens = y_valid_len.sum()
            optimizer.step()

            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}, loss {metric[0]/metric[1]:.4f}')
    print(f'loss {metric[0] / metric[1]:.4f}, {metric[1] / timer.stop():.1f} ',
          f'tokens/sec on {str(device)}')


def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """ The prediction of a sequence of sequences"""
    # Set the net to evaluation mode
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # Add batch_size dimension
    enc_x = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_x, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # Add batch_size dimension
    dec_x = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        y, dec_state = net.decoder(dec_x, dec_state)
        # We use the max possible vocab as the next step input for the decoder
        dec_x = y.argmax(dim=2)
        pred = dec_x.squeeze(dim=0).type(torch.int32).item()
        # Save the attention
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # Once the `<eos>` tag has been predicted, the output sequence should be stopped.
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)

    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


def bleu(pred_seq, label_seq, k):
    """ Compute the BLEU """
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, int(1 - len_label / len_pred)))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score
