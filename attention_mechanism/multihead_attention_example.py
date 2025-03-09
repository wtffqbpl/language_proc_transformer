#! coding: utf-8

import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import random
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import utils.dlf as dlf


# Use fixed random seed for replicate test
seed = 42
torch.manual_seed(seed)

tokens = ['<pad>', '<sos>', '<eos>'] + [str(i) for i in range(10)]
vocab = {token: idx for idx, token in enumerate(tokens)}
inv_vocab = {idx: token for token, idx in vocab.items()}
vocab_size = len(vocab)


# Toy datasets
toy_data = [
    (['1', '2', '3', '4'], ['4', '3', '2', '1']),
    (['3', '5', '7'], ['7', '5', '3']),
    (['9', '8', '7', '6', '5'], ['5', '6', '7', '8', '9']),
    (['2', '2', '2'], ['2', '2', '2']),
    (['0', '1', '0'], ['0', '1', '0']),
]


def collate_fn(batch):
    src_batch, target_batch = [], []
    for src_tokens, target_tokens in batch:
        # Transform input sequence
        src_indices = [vocab[token] for token in src_tokens]
        # Add <sos> and <eos>
        target_indices = [vocab['<sos>']] + [vocab[token] for token in tokens]
        src_batch.append(torch.tensor(src_indices, dtype=torch.long))
        target_batch.append(torch.tensor(target_indices, dtype=torch.long))
    src_batch = pad_sequence(src_batch, padding_value=vocab['<pad>'])
    target_batch = pad_sequence(target_batch, padding_value=vocab['<pad>'])
    return src_batch, target_batch


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src.shape: (src_len, batch_size)
        embedded = self.dropout(self.embedding(src))  # (src_len, batch_size, emb_dim)
        outputs, (hidden, cell) = self.rnn(embedded)  # outputs: [src_len, batch_size, hid_dim]
        return outputs, hidden, cell


# Using MultiheadAttention mechanism for decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, num_layers, dropout, num_heads):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)

        # The embed_dim should be same as hidden_dim
        self.multi_head_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.rnn = nn.LSTM(emb_dim + hidden_dim, hidden_dim, num_layers, dropout=dropout)
        self.fc_out = nn.Linear(emb_dim + hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        #
        input_tokens = input.unsqueeze(0)  # (1, batch_size)
        embedded = self.dropout(self.embedding(input_tokens))  # (1, batch_size, emb_dim)
        # Use the last hidden states.
        query = hidden[-1].unsqueeze(0)  # (1, batch_size, hidden_dim)
        # encoder_outputs.shape(src_len, batch_size, hidden_dim).
        # Use the encoder_outputs as the key and value
        attn_output, attn_weights = self.multi_head_attn(query, encoder_outputs, encoder_outputs)
        # Concat embedding and multihead attentions as the context
        rnn_input = torch.cat((embedded, attn_output), dim=2)  # [1, batch_size, emb_dim + hid_dim]
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        # Use output of the LSTM, attention context and embeddings for prediction
        predictions = self.fc_out(torch.cat((output.squeeze(0),
                                             attn_output.squeeze(0),
                                             embedded.squeeze(0)), dim=1))
        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src: torch.Tensor, target: torch.Tensor, teacher_forcing_ratio=0.5):
        # src.shape: (src_len, batch_size)
        # target.shape: (target_len, batch_size)
        target_len, batch_size = target.shape[0], target.shape[1]
        target_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src)

        # The first token is <sos>
        input_tokens = target[0, :]
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(input_tokens, hidden, cell, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_tokens = target[t] if teacher_force else top1
        return outputs


def train(model: nn.Module,
          train_iter,
          optimizer,
          loss_fn,
          num_epochs,
          device):

    for epoch in range(num_epochs):
        model.train()

        epoch_loss = 0

        for src, target in train_iter:
            src, target = src.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(src, target, teacher_forcing_ratio=0.5)

            # output.shape: (target_len, batch_size, output_dim).
            # Skipping the first token <sos>
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            target = target[1:].view(-1)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}, ', f'Loss: {epoch_loss/len(train_iter):.4f}')


def inference(model: Seq2Seq, sequence: list[str], device: torch.device, max_tokens=10):
    model.eval()

    with torch.no_grad():
        src_indices = [vocab[token] for token in sequence]
        src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(1).to(device=device)
        encoder_outputs, hidden, cell = model.encoder(src_tensor)
        input_token = torch.tensor([vocab['<sos>']], dtype=torch.long).to(device)

        outputs = []

        # The maximum tokens is 10
        for _ in range(max_tokens):
            output, hidden, cell = model.decoder(input_token, hidden, cell, encoder_outputs)
            pred_token = output.argmax(1).item()
            if pred_token == vocab['<eos>']:
                break

            outputs.append(inv_vocab[pred_token])
            input_token = torch.tensor([pred_token], dtype=torch.long).to(device)

    return outputs


class IntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model_path = os.path.join(str(Path(__file__).resolve().parent), 'multi_head_attention_model.pth')

    def test_multi_head_attention_training(self):
        # hyperparameters
        input_dim = vocab_size
        output_dim = vocab_size
        emb_dim = 16
        hidden_dim = 32
        num_layers = 1
        dropout = 0.1
        num_heads = 4  # The hidden_dim should be divided by num_heads.
        assert (hidden_dim % num_heads) == 0

        num_epochs = 50

        device = dlf.devices()[0]

        train_iter = DataLoader(toy_data, batch_size=2, shuffle=True, collate_fn=collate_fn)

        encoder = Encoder(input_dim, emb_dim, hidden_dim, num_layers, dropout)
        decoder = Decoder(output_dim, emb_dim, hidden_dim, num_layers, dropout, num_heads)
        model = Seq2Seq(encoder, decoder, device).to(device)

        loss_fn = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
        optimizer = torch.optim.Adam(model.parameters())

        train(model, train_iter, optimizer, loss_fn, num_epochs, device)

        torch.save(model, self.model_path)

    def test_inference(self):
        if not os.path.exists(self.model_path):
            self.test_multi_head_attention_training()

        device = dlf.devices()[0]
        model = torch.load(self.model_path, weights_only=False).to(device)

        test_seq = ['1', '2', '3', '4']
        translated = inference(model, test_seq, device)
        print('The input sequence: ', test_seq)
        print('The predicted inverse sequence: ', translated)


if __name__ == "__main__":
    unittest.main(verbosity=True)
