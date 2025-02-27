#! coding: utf-8

import unittest
import torch
import torch.nn as nn
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import utils.dlf as dlf


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hide_dim, n_layers, dropout):
        """
        @param input_dim: The size of the one-hot vectors that will be input to the encoder.
        @param emb_dim: The dimensionality of the embedding layer.
        @param hide_dim: The dimensionality of the hidden and cell states.
        @param n_layers: The number of layers in the RNN.
        @param dropout: The dropout probability.
        """
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hide_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """
        @param src: The input to the encoder. It is a tensor of shape (src_len, batch_size).
        @return: The output of the encoder. It is a tensor of shape (src_len, batch_size, hid_dim).
        """
        embedded = self.dropout(self.embedding(src))  # [src_len, batch_size, emb_dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # output.shape: [src_len, batch_size, emb_dim]
        # hidden, cell.shape: [n_layers, batch_size, hid_dim]
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hide_dim, n_layers, dropout):
        """
        :param output_dim: The output dimension
        :param emb_dim: The dimensionality of the embedding layer
        :param hide_dim: The dimensionality of the hidden and cell states
        :param n_layers: The number of layers in the RNN
        :param dropout: The dropout probability
        """
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hide_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hide_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input: [batch_size], The input for the current time step.
        # Usually it is the output of the previous time step.
        input = input.unsqueeze(0)  # input shape: [1, batch_size]
        embedded = self.dropout(self.embedding(input))  # embedded shape: [1, batch_size, emb_dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        # output: [1, batch_size, hide_dim]
        prediction = self.fc_out(output.squeeze(0))  # output shape: [batch_size, output_dim]
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, target, teacher_forcing_ratio=0.5):
        """
        :param src: The source tensor. It is a tensor of shape (src_len, batch_size).
        :param target: The target tensor. It is a tensor of shape (target_len, batch_size)
        :param teacher_forcing_ratio: The probability of using teacher forcing.
        """
        target_len, batch_size = target.shape[0], target.shape[1]
        target_vocab_size = self.decoder.embedding.num_embeddings

        # Save the outputs of the decoder
        outputs = torch.zeros(size=(target_len, batch_size, target_vocab_size),
                              device=self.device)

        # The encoder hidden state is the initial hidden state of the decoder
        hidden, cell = self.encoder(src)

        # The first input to the decoder is the <sos> token
        input = target[0, :]  # [batch_size]

        for t in range(1, target_len):
            # The decoder takes the input, the hidden state, and the cell state
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output

            # Decide whether the output should be used as the next input
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)  # The max probability of the prediction
            input = target[t] if teacher_force else top1

        return outputs


def train(model, optimizer, loss_fn, src, target, clip):
    model.train()
    optimizer.zero_grad()
    output = model(src, target)

    # Ignore the <sos> token
    output_dim = output.shape[-1]
    output = output[1:].view(-1, output_dim)  # [(target_len - 1) * batch_size, output_dim]
    target = target[1:].reshape(-1)  # [(target_len - 1) * batch_size]
    loss = loss_fn(output, target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()

    return loss.item()


def evaluate(model, loss_fn, src, target):
    model.eval()

    with torch.no_grad():
        output = model(src, target, 0)  # Turn off teacher forcing

    output_dim = output.shape[-1]
    output = output[1:].view(-1, output_dim)
    target = target[1:].reshape(-1)

    loss = loss_fn(output, target)

    return loss.item()


def inference(model, src, max_len=10, sos_idx=0, eos_idx=2):
    """
    Specify the input src.shape=[src_len, 1], generating the output sequence(which
    does not include the <sos> and stop generating when the <eos> token is generated).
    """
    model.eval()

    with torch.no_grad():
        hidden, cell = model.encoder(src)
    input_token = torch.tensor([sos_idx], device=src.device)  # The initial input is the <sos> token
    # add the <sos> token
    outputs = [torch.tensor([sos_idx]).to(src.device)]

    for _ in range(max_len):
        with torch.no_grad():
            output, hidden, cell = model.decoder(input_token, hidden, cell)
        pred_token = output.argmax(1)
        if pred_token.item() == eos_idx:
            break
        outputs.append(pred_token)
        input_token = pred_token.clone().detach().to(device=src.device)

    # Add the <eos> token
    outputs.append(torch.tensor([eos_idx]).to(src.device))

    return torch.cat(outputs, dim=0)


class Seq2SeqTest(unittest.TestCase):
    def test_seq2seq_model(self):
        device = dlf.devices()[0]

        # hyperparameters
        input_dim, output_dim = 1000, 1000
        emb_dim = 256
        hidden_dim, num_layers, dropout = 512, 2, 0.5

        encoder = Encoder(input_dim, emb_dim, hidden_dim, num_layers, dropout).to(device)
        decoder = Decoder(output_dim, emb_dim, hidden_dim, num_layers, dropout).to(device)
        model = Seq2Seq(encoder, decoder, device).to(device)

        target_len, batch_size = 20, 32
        src = torch.randint(0, input_dim, (target_len, batch_size), device=device)
        target = torch.randint(0, output_dim, (target_len, batch_size), device=device)

        output = model(src, target)

        print(f'output.shape: {output.shape}')
        self.assertEqual(torch.Size([target_len, batch_size, output_dim]), output.shape)

    def test_reverse_string(self):
        device = dlf.devices()[0]

        # hyperparameters
        input_dim, output_dim = 11, 11
        emb_dim = 10
        hidden_dim, num_layers, dropout = 16, 2, 0.1

        encoder = Encoder(input_dim, emb_dim, hidden_dim, num_layers, dropout).to(device)
        decoder = Decoder(output_dim, emb_dim, hidden_dim, num_layers, dropout).to(device)
        model = Seq2Seq(encoder, decoder, device).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = nn.CrossEntropyLoss()

        # The toy dataset
        # The sequence format: [<sos>, token1, token2, token3, <eos>]
        toy_src, toy_target = [], []

        num_samples = 1000
        for _ in range(num_samples):
            training_seq = torch.randint(3, 10, (3,))
            a, b, c = training_seq[0].item(), training_seq[1].item(), training_seq[2].item()
            toy_src.append([1, a, b, c, 2])
            toy_target.append([1, c, b, a, 2])

        # Convert the sequence to tensor
        src_tensor = torch.tensor(toy_src, device=device).T
        target_tensor = torch.tensor(toy_target, device=device).T

        # Training
        clip = 1
        num_epochs = 2000
        for epoch in range(num_epochs):
            loss = train(model, optimizer, loss_fn, src_tensor, target_tensor, clip)

            if (epoch + 1) % 50 == 0:
                eval_loss = evaluate(model, loss_fn, src_tensor, target_tensor)
                print(f'Epoch {epoch}, Loss: {loss:.4f}, eval_loss: {eval_loss}')

        # Inference
        # The input sequence is [1, 7, 5, 4, 2]
        test_src = torch.tensor([1, 7, 5, 4, 2], device=device).unsqueeze(1)
        predicted_tokens = inference(model, test_src, max_len=10, sos_idx=1, eos_idx=2)
        print('The input sequence: ', test_src.squeeze(1).tolist())
        print('The inverted sequence: ', predicted_tokens)


if __name__ == "__main__":
    unittest.main(verbosity=True)
