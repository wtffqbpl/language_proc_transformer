#! coding: utf-8

import unittest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import re
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import utils.dlf as dlf
from recurrent_neural_network.rnn_utils import load_data_time_machine, RNNModelScratch, RNNModelWithTorch, train


def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    w_xi, w_hi, b_i = three()  # Input gate parameters
    w_xf, w_hf, b_f = three()  # Forget gate parameters
    w_xo, w_ho, b_o = three()  # Output gate parameters
    w_xc, w_hc, b_c = three()  # Candidate hidden state parameters

    # Output layer parameters
    w_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    # Attach gradients
    params = [w_xi, w_hi, b_i, w_xf, w_hf, b_f, w_xo, w_ho, b_o, w_xc, w_hc, b_c, w_hq, b_q]
    for param in params:
        param.requires_grad_(True)

    return params


def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))


def lstm(inputs, state, params):
    [w_xi, w_hi, b_i, w_xf, w_hf, b_f, w_xo, w_ho, b_o, w_xc, w_hc, b_c, w_hq, b_q] = params
    (h, c) = state
    outputs = []

    for x in inputs:
        i = torch.sigmoid(torch.mm(x, w_xi) + torch.mm(h, w_hi) + b_i)
        f = torch.sigmoid(torch.mm(x, w_xf) + torch.mm(h, w_hf) + b_f)
        o = torch.sigmoid(torch.mm(x, w_xo) + torch.mm(h, w_ho) + b_o)
        c_tilda = torch.tanh(torch.mm(x, w_xc) + torch.mm(h, w_hc) + b_c)
        c = f * c + i * c_tilda
        h = o * torch.tanh(c)
        y = torch.mm(h, w_hq) + b_q
        outputs.append(y)
    return torch.cat(outputs, dim=0), (h, c)


# The case for predicting the sine function

def generate_data(total_points=1000):
    x = np.linspace(0, 100, total_points)
    data = np.sin(x)
    return data


# Create a function to generate a dataset for the sine function
def create_dataset(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        _x = data[i:i+seq_length]
        _y = data[i+seq_length]

        xs.append(_x)
        ys.append(_y)
    return np.array(xs), np.array(ys)


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define the LSTM model
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Define the output layer, and map the hidden layer to the output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM.
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def tokenize(text):
    # Convert text to lowercase
    text = text.lower()
    tokens = re.findall(r'\w+', text)
    return tokens


def encode(text, vocab):
    tokens = tokenize(text)
    return [vocab.get(token, 0) for token in tokens]


# Set the max length of the text, and truncate the text if it is longer than the max length.
# And add padding to the text if it is shorter than the max length.
def pad_sequences(sequences, max_len=256):
    if len(sequences) < max_len:
        return sequences + [0] * (max_len - len(sequences))
    else:
        return sequences[:max_len]


class IMDBDataset(Dataset):
    def __init__(self, dataset, vocab, max_len=256):
        self.vocab = vocab
        self.max_len = max_len
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        label = self.data[idx]['label']
        encoded = encode(text, self.vocab)
        padded = pad_sequences(encoded, self.max_len)
        return torch.tensor(padded, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def load_imdb_dataset(batch_size=32):
    dataset = load_dataset('imdb')
    train_data = dataset['train']
    test_data = dataset['test']

    all_tokens = []
    for sample in train_data:
        tokens = tokenize(sample['text'])
        all_tokens.extend(tokens)

    vocab_counter = Counter(all_tokens)
    # Sort the words according to frequency, and reserve the 5 times most frequent words
    vocab_list = [word for word, freq in vocab_counter.items() if freq >= 5]
    vocab = {word: i + 1 for i, word in enumerate(vocab_list)}
    vocab_size = len(vocab) + 1

    max_len = 256
    train_dataset = IMDBDataset(train_data, vocab, max_len)
    test_dataset = IMDBDataset(test_data, vocab, max_len)

    return (vocab_size,
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4))


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        out, (h_n, c_n) = self.lstm(x)  # (batch_size, seq_len, hidden_dim)
        last_hidden = h_n[-1]  # (batch_size, hidden_dim)
        logits = self.fc(last_hidden)  # (batch_size, num_classes)
        return logits


class IntegrationTest(unittest.TestCase):

    def test_lstm(self):
        batch_size, num_steps = 32, 35
        train_iter, vocab = load_data_time_machine(batch_size, num_steps, use_random_iter=True)

        vocab_size, num_hiddens = len(vocab), 256
        device = dlf.devices()[0]
        num_epochs, learning_rate = 500, 1

        model = RNNModelScratch(len(vocab), num_hiddens, device, get_lstm_params, init_lstm_state, lstm)

        loss_fn = torch.nn.CrossEntropyLoss()
        train(model, train_iter, vocab, loss_fn, lr=learning_rate, num_epochs=num_epochs, device=device)

        self.assertTrue(True)

    def test_lstm_pytorch_api(self):
        batch_size, num_steps = 32, 35
        train_iter, vocab = load_data_time_machine(batch_size, num_steps, use_random_iter=True)

        vocab_size, num_hiddens = len(vocab), 256
        device = dlf.devices()[0]
        num_epochs, learning_rate = 600, 1
        num_inputs = vocab_size

        lstm_layer = nn.LSTM(num_inputs, num_hiddens)

        model = RNNModelWithTorch(lstm_layer, vocab_size).to(device=device)
        loss_fn = torch.nn.CrossEntropyLoss()
        train(model, train_iter, vocab, loss_fn, lr=learning_rate, num_epochs=num_epochs, device=device)

        self.assertTrue(True)

    def test_bidirectional_RNN_model(self):
        batch_size, num_steps = 32, 35
        device = dlf.devices()[0]

        train_iter, vocab = load_data_time_machine(batch_size, num_steps)

        vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
        num_inputs = vocab_size
        # Set `bidirectional=True` to enable bidirectional-LSTM RNN model.
        lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers, bidirectional=True)
        model = RNNModelWithTorch(lstm_layer, vocab_size).to(device)

        loss_fn = torch.nn.CrossEntropyLoss()

        # Training
        num_epochs, learning_rate = 500, 1
        # FIXME: This is not the correct way to use bidirectional RNN model.
        train(model, train_iter, vocab, loss_fn, learning_rate, num_epochs, device)

        # To avoid warnings
        self.assertTrue(True)

    def test_predict_sin_function(self):
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)

        # hyperparameters
        seq_length = 50  # Input sequence length
        hidden_size = 64  # LSTM number of hidden units
        num_layers = 2  # Number of layers in LSTM
        learning_rate = 0.002
        num_epochs = 100  # Number of training epochs

        device = dlf.devices()[0]

        input_size = 1
        data = generate_data()
        x, y = create_dataset(data, seq_length)

        # x.shape: (num_samples, seq_length, 1)
        x = torch.from_numpy(x).to(dtype=torch.float32).to(device=device).unsqueeze(-1)
        # y.shape: (num_samples, 1)
        y = torch.from_numpy(y).to(dtype=torch.float32).to(device=device).unsqueeze(-1)

        model = LSTMModel(input_size, hidden_size, num_layers).to(device=device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()

            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Inference
        model.eval()
        predicated = model(x).cpu().detach().numpy()

        # Plot the results
        plt.plot(data[seq_length:], label='Actual')
        plt.plot(predicated, label='Predicted')
        plt.legend()
        plt.xlabel('Time steps')
        plt.ylabel('Value')
        plt.title('Predicting the sine function using LSTM')
        plt.show()

        self.assertTrue(True)

    def test_imdb_predictions(self):
        embed_dim = 128
        hidden_dim = 256
        num_layers = 2
        num_classes = 2  # 0 -- negative, 1 -- positive
        batch_size = 32
        lr = 0.001

        vocab_size, data_iter, test_iter = load_imdb_dataset(batch_size)

        device = dlf.devices()[0]

        model = LSTMClassifier(vocab_size, embed_dim, hidden_dim, num_layers, num_classes)
        model.to(device=device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        num_epochs = 10
        for epoch in range(num_epochs):
            model.train()
            for i, (x, y) in enumerate(data_iter):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                optimizer.step()

                if i % 100 == 0:
                    print(f'Epoch {epoch + 1}, Iteration {i}, Loss: {loss.item()}')

        # To avoid warnings
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main(verbosity=True)
