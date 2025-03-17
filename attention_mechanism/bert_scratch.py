#! coding: utf-8

import unittest
import torch
import torch.nn as nn
from .attention_utils import EncoderBlock


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


if __name__ == '__main__':
    unittest.main(verbosity=True)
