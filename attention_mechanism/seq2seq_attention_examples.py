#! coding: utf-8

import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import random
import spacy
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import utils.dlf as dlf


# define special tokens
pad_token = '<pad>'
sos_token = '<sos>'
eos_token = '<eos>'
unk_token = '<unk>'


# Define iterator for construct vocab
def yield_tokens(data_iter, tokenizer, index):
    for item in data_iter:
        yield tokenizer(item[index])



class TranslationEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers, dropout, **kwargs):
        super(TranslationEncoder, self).__init__(**kwargs)

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src.shape: (src_len, batch_size)
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)

        return outputs, hidden, cell


class TranslationAttention(nn.Module):
    def __init__(self, hidden_dim, **kwargs):
        super(TranslationAttention, self).__init__(**kwargs)

        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor):
        # hidden.shape: (batch_size, hidden_dim)
        # encoder_outputs: (src_len, batch_size, hidden_dim)
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # (batch_size, src_len, hidden_dim)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # (batch_size, src_len, hidden_dim)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)  # (batch_size, src_len)
        return torch.softmax(attention, dim=1)


class TranslationDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, num_layers, dropout, attention, **kwargs):
        super(TranslationDecoder, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim + hidden_dim, hidden_dim, num_layers, dropout=dropout)
        self.fc_out = nn.Linear(emb_dim + hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor, encoder_outputs):
        # inputs: (batch_size)
        inputs = inputs.unsqueeze(0)  # (1, batch_size)
        embedded = self.dropout(self.embedding(inputs))
        # Compute attention weights
        attn = self.attention(hidden[-1], encoder_outputs)  # (batch_size, src_len)
        attn = attn.unsqueeze(1)  # (batch_size, 1, src_len)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # (batch_size, src_len, hidden_dim)
        weighted = torch.bmm(attn, encoder_outputs)  # (batch_size, 1, hidden_dim)
        weighted = weighted.permute(1, 0, 2)  # (1, batch_size, hidden_dim)

        # Concat embeddings and scaled attention weights
        rnn_input = torch.cat((embedded, weighted), dim=2)  # (1, batch_size, emb_dim + hidden_dim)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # output.shape: (1, batch_size, hidden_dim)
        embedded = embedded.squeeze(0)  # (batch_size, emb_dim)
        output = output.squeeze(0)  # (batch_size, hidden_dim)
        weighted = weighted.squeeze(0)  # (batch_size, hidden_dim)
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        return prediction, hidden, cell


class TranslationSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(TranslationSeq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src: torch.Tensor, target: torch.Tensor, teacher_forcing_ratio=0.5):
        # src: (src_len, batch_size), target: (target_len, batch_size)
        target_len, batch_size = target.shape[0], target.shape[1]
        target_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)

        encoder_outputs, hidden, cell = self.encoder(src)

        # The first input is <sos> for the decoder
        inputs = target[0, :]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(inputs, hidden, cell, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            inputs = target[t] if teacher_force else top1
        return outputs


# Training
def train_epoch(model, train_iter, optimizer, loss_fn, clip, device):
    model.train()
    epoch_loss = 0

    for src, target in train_iter:
        src, target = src.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(src, target)
        # output.shape; (target_len, batch_size, output_dim)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        target = target[1:].view(-1)
        loss = loss_fn(output, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(train_iter)


def evaluate_epoch(model, data_iter, loss_fn, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for src, target in data_iter:
            src, target = src.to(device), target.to(device)
            output = model(src, target, teacher_forcing_ratio=0)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            target = target[1:].view(-1)
            loss = loss_fn(output, target)
            epoch_loss += loss.item()

    return epoch_loss / len(data_iter)


class IntegrationTest(unittest.TestCase):
    def test_translate_german_to_english(self):
        spacy_de = spacy.load('de_core_news_sm')
        spacy_en = spacy.load('en_core_web_sm')

        # Define tokenizers
        token_de = lambda text: [tok.text for tok in spacy_de.tokenizer(text)]
        token_en = lambda text: [tok.text for tok in spacy_en.tokenizer(text)]

        specials = [pad_token, sos_token, eos_token, unk_token]

        # Transfer training dataset to a list since the iterator of the Multi30k could only iterate once.
        train_data = list(Multi30k(split='train', language_pair=('de', 'en')))

        # Construct source language(germany) vocab and target language(english) vocab.
        vocab_src = build_vocab_from_iterator(yield_tokens(train_data, token_de, index=0),
                                              specials=specials,
                                              min_freq=2)
        vocab_src.set_default_index(vocab_src[unk_token])

        vocab_target = build_vocab_from_iterator(yield_tokens(train_data, token_en, index=1),
                                                 specials=specials,
                                                 min_freq=2)
        vocab_target.set_default_index(vocab_target[unk_token])

        # Define collate_fn for padding each sentence in each batch
        def collate_fn(batch):
            src_batch, target_batch = [], []

            for src, target in batch:
                # Tokenize and add sos & eos tokens for each sentence
                src_tokens = [sos_token] + token_de(src) + [eos_token]
                target_tokens = [sos_token] + token_en(target) + [eos_token]

                # Mappings
                src_indices = [vocab_src[token] for token in src_tokens]
                target_indices = [vocab_target[token] for token in target_tokens]
                src_batch.append(torch.tensor(src_indices, dtype=torch.long))
                target_batch.append(torch.tensor(target_indices, dtype=torch.long))

            # Generate [seq_len, batch_size] tensor, and pad pad_token when the
            # sentence is not long enough. The seq_len is the longest sentence length.
            src_batch = pad_sequence(src_batch, padding_value=vocab_src[pad_token])
            target_batch = pad_sequence(target_batch, padding_value=vocab_target[pad_token])
            return src_batch, target_batch


        batch_size = 128

        train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        valid_data = list(Multi30k(split='valid', language_pair=('de', 'en')))
        valid_iter = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        # test_data = list(Multi30k(split='test', language_pair=('de', 'en')))
        # test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        device = dlf.devices()[0]

        input_dim = len(vocab_src)
        output_dim = len(vocab_target)
        enc_emb_dim = 256
        dec_emb_dim = 256
        hidden_dim = 512
        num_layers = 2
        enc_dropout = 0.5
        dec_dropout = 0.5

        attn = TranslationAttention(hidden_dim)
        encoder = TranslationEncoder(input_dim, enc_emb_dim, hidden_dim, num_layers, enc_dropout)
        decoder = TranslationDecoder(output_dim, dec_emb_dim, hidden_dim, num_layers, dec_dropout, attn)

        model = TranslationSeq2Seq(encoder, decoder, device).to(device)

        def init_weights(m: nn.Module):
            for name, param in m.named_parameters():
                nn.init.normal_(param.data, mean=0, std=0.01)

        model.apply(init_weights)

        optimizer = optim.Adam(model.parameters())
        target_pad_index = vocab_target[pad_token]
        loss_fn = nn.CrossEntropyLoss(ignore_index=target_pad_index)

        num_epochs = 10
        clip = 1

        for epoch in range(num_epochs):
            train_loss = train_epoch(model, train_iter, optimizer, loss_fn, clip, device)
            valid_loss = evaluate_epoch(model, valid_iter, loss_fn, device)
            print(f"Epoch: {epoch+1} | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}")

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main(verbosity=True)
