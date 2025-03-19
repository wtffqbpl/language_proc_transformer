#! coding: utf-8

import unittest
import torch
import torch.nn as nn
from .attention_utils import EncoderBlock
import random
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import utils.dlf as dlf
from recurrent_neural_network.rnn_utils import Vocab


def get_tokens_and_segments(tokens_a, tokens_b=None):
    """ Get tokens of the BERT input sequence and their segment IDs. """
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0 and 1 are marking segment A and B, respectively
    segments = [0] * (len(tokens_a) + 2)  # with two extra tokens: <cls> and <sep>
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)

    return tokens, segments


class BERTEncoder(nn.Module):
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()

        for i in range(num_layers):
            self.blks.add_module(f'{i}', EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input,
                ffn_num_hiddens, num_heads, dropout, use_bias=True))
        # In BERT, positional embeddings are learnable, thus we create parameter
        # of positional embeddings that are long enough
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))

    def forward(self, tokens: torch.Tensor, segments: torch.Tensor, valid_lens: torch.Tensor):
        # Shape of `x` remains unchanged in the following code snippet:
        # (batch_size, max_sequence_length, num_hiddens)
        x = self.token_embedding(tokens) + self.segment_embedding(segments)
        x = x + self.pos_embedding.data[:, :x.shape[1], :]

        for blk in self.blks:
            x = blk(x, valid_lens)
        return x


# masked Language Modeling
# To encode context bidirectionally for representing each token, BERT randomly masks tokens and
# uses tokens from the bidirectional context to predict the masked tokens in a self-su
class MaskLM(nn.Module):
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, x, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = x.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # Suppose that `batch_size` = 2, `num_pred_positions` = 3, then
        # `batch_idx` is `torch.tensor([0, 0, 0, 1, 1, 1])`
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_x = x[batch_idx, pred_positions]
        masked_x = masked_x.reshape((batch_size, num_pred_positions, -1))
        mlm_y_hat = self.mlp(masked_x)
        return mlm_y_hat


# Although masked language modeling is able to encode bidirectional context for representing
# words, it does not explicitly model the logical relationship between text pairs. To help
# understand the relationship between two text sequences, BERT considers a binary classification
# task, next sentence prediction, in its pretraining. When generating sentence pairs for
# pretraining, for half of the time they are indeed consecutive sentences with the label "True";
# while for the other half of the time the second sentence is randomly sampled from the corpus
# with the label "False".
# The NextSentencePred class uses a one-hidden-layer MLP to predict whether the second sentence
# is the next sentence of the first in the BERT input sequence. Due to self-attention in the
# Transformer encoder, the BERT representation of the special token "<cls>" encodes both the
# two sentences from the input. Hence, the output layer (self.output) of the MLP classifier
# takes `x` as input, where `x` is the output of the MLP hidden layer whose input is the
# encoded `<cls>` token.
class NextSentencePred(nn.Module):
    """ The next sentence prediction task of BERT. """
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, x):
        # The x.shape: (batch_size, num_hiddens)
        return self.output(x)
    pass


class BERTModel(nn.Module):
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768, nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                                   ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                                   dropout, max_len=max_len, key_size=key_size, query_size=query_size,
                                   value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_x = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_y_hat = self.mlm(encoded_x, pred_positions)
        else:
            mlm_y_hat = None
        # The hidden layer of the MLP classifier for next sentence prediction.
        # 0 is the index of the `<cls>` token.
        nsp_y_hat = self.nsp(self.hidden(encoded_x[:, 0, :]))
        return encoded_x, mlm_y_hat, nsp_y_hat


dlf.DATA_HUB['wikitext-2'] = (
    'https://s3.amazonaws.com/research.metamind.io/wikitext/'
    'wikitext-2-v1.zip', '3c914d17d80b1459be871a5039ac23e752a53cbe')


def read_wiki(data_dir):
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # Uppercase letters are converted to lowercase ones
    paragraphs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs


def get_next_sentence(sentence, next_sentence, paragraphs):
    if random.random() < 0.5:
        is_next = True
    else:
        # paragraphs is a list of lists of lists
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next


# The following function generates training examples for next sentence prediction from
# the input paragraph by invoking the get_next_sentence function. Here paragraph is a
# list of sentences, where each sentence is a list of tokens. The argument max_len
# specifies the maximum length of a BERT input sentence during pretraining.
def get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs)
        # Consider 1 '<cls>' token and 2 '<sep>' tokens
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph


# Generate training examples for the masked language modeling task from a BERT input sequence.
def replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds, vocab):
    # For the input of a masked language model, make a new copy of tokens and
    # replace some of them by '<mask>' or random tokens.
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    # Shuffle for getting 15% random tokens for prediction in the masked language
    # modeling task
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) > num_mlm_preds:
            break
        masked_token = None

        # 80% of the time: replace the word with the '<mask>' token
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            if random.random() < 0.5:
                # 10% of the time: keep the word unchanged
                masked_token = tokens[mlm_pred_position]
            else:
                # 10% of the time: replace the word with a random word
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels


def get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_positions = []
    # tokens is a list of strings
    for i, token in enumerate(tokens):
        # Special tokens are not predicted in the masked language modeling task
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    # 15% of random tokens are predicted in the masked language modeling
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels, key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]


def pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens, = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []

    for token_ids, pred_positions, mlm_pred_label_ids, segments, is_next in examples:
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (
            max_len - len(segments)), dtype=torch.long))
        # valid_lens excludes count of <pad> tokens
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype=torch.long))

        # Predictions of padded tokens will be filtered out in the loss via
        # multiplication of 0 weights
        all_mlm_weights.append(
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)), dtype=torch.float32))
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)


def tokenize(lines, token='word'):
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print("Error: unexpected token: %s" % token)
        raise KeyError


class WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        paragraphs = [tokenize(paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs for sentence in paragraph]
        self.vocab = Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        examples = []
        for paragraph in paragraphs:
            examples.append(get_nsp_data_from_paragraph(paragraph, paragraphs, self.vocab, max_len))

        examples = [(get_mlm_data_from_tokens(tokens, self.vocab) + (segments, is_next))
                    for tokens, segments, is_next in examples]

        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = pad_bert_inputs(examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)


def load_data_wiki(batch_size, max_len):
    """ Load WikiText-2 dataset """
    num_workers = 4
    # data_dir = dlf.download_extract('wikitext-2', 'wikitext-2')
    data_dir = os.path.join(str(Path(__file__).resolve().parent))
    paragraphs = read_wiki(data_dir)
    train_set = WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers)

    return train_iter, train_set.vocab


class IntegrationTest(unittest.TestCase):
    def test_bert_encoder(self):
        # Suppose that the vocabulary size is 10000. To demonstrate forward inference
        # of BERTEncoder, let's create an instance of it and initialize its parameters.
        vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
        norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
        encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                              ffn_num_hiddens, num_heads, num_layers, dropout)

        tokens = torch.randint(0, vocab_size, (2, 8))
        segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
        encoded_x = encoder(tokens, segments, None)
        self.assertEqual(torch.Size([2, 8, num_hiddens]), encoded_x.shape)

        # Test MaskLM
        mlm = MaskLM(vocab_size, num_hiddens)
        mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])
        mlm_y_hat = mlm(encoded_x, mlm_positions)

        print(mlm_positions.shape)

        mlm_y = torch.tensor([[7, 8, 9], [10, 20, 30]])
        loss = nn.CrossEntropyLoss(reduction='none')
        mlm_l = loss(mlm_y_hat.reshape((-1, vocab_size)), mlm_y.reshape(-1))
        print(mlm_l.shape)

        # PyTorch by default will not flatten the tensor, if flatten=True, all but the first
        # axis of input data are collapsed together.
        encoded_x = torch.flatten(encoded_x, start_dim=1)
        # input_shape for NSP: (batch_size, num_hiddens)
        nsp = NextSentencePred(encoded_x.shape[-1])
        nsp_y_hat = nsp(encoded_x)
        print(nsp_y_hat.shape)

        # The cross-entropy loss of the 2 binary classifications can also be computed.
        nsp_y = torch.tensor([0, 1])
        nsp_l = loss(nsp_y_hat, nsp_y)
        print(nsp_l.shape)

        # Depress warning
        self.assertTrue(True)

    def test_wiki_dataset(self):
        batch_size, max_len = 512, 64
        train_iter, vocab = load_data_wiki(batch_size, max_len)

        for (tokens_x, segments_x, valid_lens_x, pred_positions_x, mlm_weights_x,
             mlm_y, nsp_y) in train_iter:
            print(tokens_x.shape, segments_x.shape, valid_lens_x.shape,
                  pred_positions_x.shape, mlm_weights_x.shape, mlm_y.shape,
                  nsp_y.shape)

        self.assertTrue(True)
        pass


if __name__ == '__main__':
    unittest.main(verbosity=True)
