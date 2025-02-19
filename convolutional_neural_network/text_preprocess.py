#! coding: utf-8
import unittest
import torch
import matplotlib.pyplot as plt
from pprint import pprint
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.plot import plot
from recurrent_neural_network.rnn_utils import (TimeMachine, Tokenizer, Vocab,
                                                load_corpus_time_machine,
                                                seq_data_iter_random,
                                                seq_data_iter_sequential)


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

        # Print the ten most frequent words in the vocabulary
        print(list(vocab.token_to_idx.items())[:10])
        pprint(vocab.token_freqs[:10])

        for i in [0, 10]:
            print('indices: ', tokens[i])
            print('words: ', vocab[tokens[i]])

        self.assertTrue(True)

    def test_time_machine(self):
        corpus, vocab = load_corpus_time_machine()
        print(len(corpus), len(vocab))

        self.assertEqual(170580, len(corpus))
        self.assertEqual(28, len(vocab))

    def test_random_sequence(self):
        my_seq = list(range(35))
        for x, y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
            print('X: ', x, '\nY:', y)
            t_x, t_y = torch.tensor(x), torch.tensor(y)
            self.assertTrue(torch.equal(t_x + 1, t_y))

    def test_sequential_sequence(self):
        my_seq = list(range(35))
        x_prev, y_prev = None, None
        batch_size, num_steps = 2, 5
        for x, y in seq_data_iter_sequential(my_seq, batch_size=batch_size, num_steps=num_steps):
            print('X: ', x, '\nY:', y)
            t_x, t_y = torch.tensor(x), torch.tensor(y)
            self.assertTrue(torch.equal(t_x + 1, t_y))

            if x_prev is not None:
                self.assertTrue(torch.equal(x_prev + num_steps, t_x))

            x_prev, y_prev = t_x, y_prev


class MarkovNGramModel(unittest.TestCase):
    # Such words that are common but not particularly descriptive are often
    # called stop words and, in previous generations of text classifiers based
    # on so-called bag-of-words representations, they were most often filtered
    # out. However, they carry meaning and it is not necessary to filer them
    # out when working with modern RNN- and Transformer-based neural models.
    #
    # Word frequency tends to follow a power law distribution (specifically
    # the ZipFian) as we go down the ranks.
    # This phenomenon is captured by Zipf's law, which states that the frequency
    # n_i of the i_th most frequent word is:
    #     n_i === 1 / i^alpha
    # which is equivalent to
    #     log(n_i) = -alpha * log(i) + c
    # where alpha is the exponent that characterizes the distribution and c is
    # a constant. This should already give us pause for thought if we want to
    # model words by counting statistics. After all, we will significantly
    # overestimate the frequency of the tail, also known as the infrequent
    # words. But what about the other word combinations, such as two consecutive
    # words (bigrams), three consecutive words (trigrams), and beyond.

    def setUp(self):
        self.tokens = Tokenizer().tokenize(TimeMachine().download())

        self.corpus = [token for line in self.tokens for token in line]
        self.vocab = Vocab(self.corpus)

    def test_unigram(self):
        expected_output = [
            ('the', 2261), ('i', 1267), ('and', 1245), ('of', 1155), ('a', 816),
            ('to', 695), ('was', 552), ('in', 541), ('that', 443), ('my', 440)
        ]
        self.assertEqual(expected_output, self.vocab.token_freqs[:10])

        freqs = [freq for token, freq in self.vocab.token_freqs]
        plot(freqs, xlabel='token: x', ylabel='frequency: n(x)', xscale='log', yscale='log')

        plt.show()

    def test_bigram(self):
        bigram_tokens = [pair for pair in zip(self.corpus[:-1], self.corpus[1:])]
        bigram_vocab = Vocab(bigram_tokens)

        expected_output = [
            (('of', 'the'), 309),
            (('in', 'the'), 169),
            (('i', 'had'), 130),
            (('i', 'was'), 112),
            (('and', 'the'), 109),
            (('the', 'time'), 102),
            (('it', 'was'), 99),
            (('to', 'the'), 85),
            (('as', 'i'), 78),
            (('of', 'a'), 73)]

        pprint(bigram_vocab.token_freqs[:10])
        self.assertEqual(expected_output, bigram_vocab.token_freqs[:10])

    def test_trigram(self):
        trigram_tokens = [triple for triple in zip(
            self.corpus[:-2], self.corpus[1:-1], self.corpus[2:])]
        trigram_vocab = Vocab(trigram_tokens)

        exptected_output = [
            (('the', 'time', 'traveller'), 59),
            (('the', 'time', 'machine'), 30),
            (('the', 'medical', 'man'), 24),
            (('it', 'seemed', 'to'), 16),
            (('it', 'was', 'a'), 15),
            (('here', 'and', 'there'), 15),
            (('seemed', 'to', 'me'), 14),
            (('i', 'did', 'not'), 14),
            (('i', 'saw', 'the'), 13),
            (('i', 'began', 'to'), 13)]

        print(trigram_vocab.token_freqs[:10])

    def test_markov_ngram_model(self):

        # bigram model
        bigram_tokens = [pair for pair in zip(self.corpus[:-1], self.corpus[1:])]
        bigram_vocab = Vocab(bigram_tokens)

        # trigram model
        trigram_tokens = [triple for triple in zip(self.corpus[:-2], self.corpus[1:-1], self.corpus[2:])]
        trigram_vocab = Vocab(trigram_tokens)

        unigram_freqs = [freq for token, freq in self.vocab.token_freqs]
        bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
        trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]

        plot([unigram_freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
             ylabel='frequency: n(x)', xscale='log', yscale='log',
             legend=['unigram', 'bigram', 'trigram'])
        # This figure is quite exciting.
        #   1. First, beyond unigram words, sequences of words also appear to
        #      be following Zipf's law, albeit with a smaller exponent alpha,
        #      depending on the sequence length.
        #   2. Second, the number of distinct n-grams is not that large. This
        #      gives us hope that there is quite a lot of structure in language.
        #   3. Third, many n-grams occur very rarely. This makes certain methods
        #      unsuitable for language modeling and motivates the use of deep
        #      learning models.

        # To preprocess text, we usually:
        #   1. split text into tokens;
        #   2. build a vocabulary to map token strings to numerical indices;
        #   3. convert text data into token indices for models to manipulate.
        # In practice, the frequency of words tends to follow Zipf's law. This
        # is true not just for individual words (unigrams), but also for n-grams.


if __name__ == "__main__":
    unittest.main(verbosity=True)
