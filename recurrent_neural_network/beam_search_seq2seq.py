#! coding: utf-8

import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import math
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import utils.dlf as dlf


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers, dropout):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src.shape = [src_len, batch_size]
        embedded = self.dropout(self.embedding(src))  # [src_len, batch_size, emb_dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # output.shape = (src_len, batch_size, hidden_dim)
        # hidden, cell = (num_layers, batch_size, hidden_dim)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, num_layers, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, cell):
        # input_token = (batch_size)
        input_token = input_token.unsqueeze(0)  # transfer to (1, batch_size)
        embedded = self.dropout(self.embedding(input_token))
        # embedded.shape = (1, batch_size, emb_dim)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output.shape = (1, batch_size, hidden_dim)
        prediction = self.fc_out(output.squeeze(0))
        # prediction.shape = (batch_size, output_dim)
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, label, teacher_forcing_ratio=0.5):
        batch_size = label.shape[1]
        label_len = label.shape[0]
        label_vocab_size = self.decoder.embedding.num_embeddings

        # 1. Encoder
        hidden, cell = self.encoder(src)

        # 2. The first input for the decoder is the <sos>
        input_token = label[0, :]  # [batch_size]

        outputs = [input_token.unsqueeze(1).unsqueeze(0).repeat(1, 1, label_vocab_size)]

        for t in range(1, label_len):
            # 3. The decoder generates the current output
            output, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs.append(output.unsqueeze(0))
            
            # 4. select the token which has the highest probability
            top1 = output.argmax(1)

            # 5. decide whether you should be teacher forcing.
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input_token = label[t] if teacher_force else top1
        return torch.cat(outputs, dim=0)


def beam_search_decoding(model, src, beam_size=3, max_len=10, sos_idx=1, eos_idx=2):
    """

    :param model:
    :param src:  [src_len, 1], the batch_size is 1
    :param beam_size: the beam size
    :param max_len: the maximum size of the sequence which includes <sos> and <eos>
    :param sos_idx: the index of the <sos>
    :param eos_idx: the index of the <eos>
    :return: the best sequence (does not include <sos>, and <eos>)
    """

    device = next(model.parameters()).device

    # 1. Acquire the hidden state and cell of the encoder
    with torch.no_grad():
        hidden, cell = model.encoder(src)  # [num_layers, 1, hidden_dim]

    # The information stored in each beam
    # tokens: the current tokens from the decoder
    # hidden_state, cell: the current states from the decoder
    # log_prob: the currumation of the current sequence.
    Hypothesis = lambda tokens, hidden, cell, log_prob: {
        "tokens": tokens,
        "hidden": hidden,
        "cell": cell,
        "log_prob": log_prob
    }

    # 2. Initialize the beam, starting from the <sos>
    init_hyp = Hypothesis(tokens=[sos_idx], hidden=hidden, cell=cell, log_prob=0.0)
    active_hypotheses = [init_hyp]
    completed_hypotheses = []

    # 3. Start to decode
    for _ in range(max_len):
        new_hypotheses = []

        # Stop if there's no more active states
        if len(active_hypotheses) == 0:
            break

        # Expand each hypothesis in the beam
        for hyp in active_hypotheses:
            last_token = hyp['tokens'][-1]

            # If <eos> has been generated, we should take each sentence generation
            # process has been done.
            if last_token == eos_idx:
                completed_hypotheses.append(hyp)
                continue

            # Either way, we should step into the next state
            input_token = torch.tensor([last_token], device=device)
            with torch.no_grad():
                output, hidden_next, cell_next = model.decoder(
                    input_token, hyp['hidden'], hyp['cell']
                )

                # output.shape = (batch_size=1, vocab_size)
                # Convert to log-probability
                lob_probs = torch.log_softmax(output, dim=1)

            # Select the top beam_size expanding vocabs.
            topk_log_probs, topk_ids = torch.topk(lob_probs, beam_size, dim=1)

            for i in range(beam_size):
                token_id = topk_ids[0, i].item()
                token_log_prob = topk_log_probs[0, i].item()
                new_tokens = hyp['tokens'] + [token_id]
                new_log_prob = hyp['log_prob'] + token_log_prob
                new_hyp = Hypothesis(
                    tokens=new_tokens,
                    hidden=hidden_next,
                    cell=cell_next,
                    log_prob=new_log_prob
                )
                new_hypotheses.append(new_hyp)
        # Select the top beam_sizes log_prob
        new_hypotheses.sort(key=lambda x: x['log_prob'], reverse=True)
        active_hypotheses = new_hypotheses[:beam_size]

    # 4. Mark all the sequences as completed
    completed_hypotheses += active_hypotheses

    # Sort all sequences by log_probs, and select the best one.
    completed_hypotheses.sort(key=lambda x: x['log_prob'], reverse=True)
    best_hyp = completed_hypotheses[0]

    # Remove the <sos>, and stop if we meet the <eos>
    best_tokens = best_hyp['tokens'][1:]  # remove the <sos>
    if eos_idx in best_tokens:
        eos_pos = best_tokens.index(eos_idx)
        best_tokens = best_tokens[:eos_pos]

    return best_tokens


def train(model, input_tensor, label_tensor, optimizer, loss_fn, num_epochs):
    for epoch in range(num_epochs):
        model.train()

        optimizer.zero_grad()

        output = model(input_tensor, label_tensor, teacher_forcing_ratio=0.5)
        # output.shape (label_len, batch_size, output_dim)
        # label_tensor: (label_len, batch_size)
        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)  # remove <sos> and expanding
        label = label_tensor[1:].contiguous().view(-1)
        loss = loss_fn(output, label)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}, loss: {loss.item():.4f}')


def inference(model, src, num_steps, beam_size=3, sos_idx=1, eos_idx=2):
    # Using greedy method for inference
    model.eval()
    with torch.no_grad():
        hidden, cell = model.encoder(src)
        input_token = torch.tensor([1], device=src.device)  # <sos>
        greedy_tokens = []

        for _ in range(num_steps):
            output, hidden, cell = model.decoder(input_token, hidden, cell)
            top1 = output.argmax(1)

            if top1.item() == eos_idx:  # <eos>
                break
            greedy_tokens.append(top1.item())
            input_token = top1
    print('Greedy decoding result: ', greedy_tokens)

    # 2. Using beam search for inference
    beam_tokens = beam_search_decoding(model, src, beam_size=beam_size,
                                       max_len=num_steps, sos_idx=sos_idx,
                                       eos_idx=eos_idx)
    print('Beam search result: ', beam_tokens)


class IntegrationTest(unittest.TestCase):
    def test_beam_training(self):
        input_dim, output_dim = 11, 11
        emb_dim = 8
        hidden_dim, num_layers, dropout = 16, 1, 0.1
        device = dlf.devices()[0]

        # Define the model
        encoder = Encoder(input_dim, emb_dim, hidden_dim, num_layers, dropout)
        decoder = Decoder(input_dim, emb_dim, hidden_dim, num_layers, dropout)
        model = Seq2Seq(encoder, decoder, device).to(device=device)

        batch_size, learning_rate = 1, 0.01
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # Assume 0 is PAD

        # Define the dataset
        # format: [<sos>, x1, x2, x3, <eos>]
        # label:  [<sos>, x3, x2, x1, <eos>]
        toy_src = [
            [1, 4, 5, 6, 2],
            [1, 7, 8, 9, 2],
            [1, 3, 5, 10, 2],
            [1, 6, 7, 8, 2]
        ]

        toy_label = [
            [1, 6, 5, 4, 2],
            [1, 9, 8, 7, 2],
            [1, 10,5, 3, 2],
            [1, 8, 7, 6, 2]
        ]

        src_tensor = torch.tensor(toy_src).transpose(0, 1).to(device)
        label_tensor = torch.tensor(toy_label).transpose(0, 1).to(device)

        num_epochs = 500

        train(model, src_tensor, label_tensor, optimizer, loss_fn, num_epochs)

        sos_idx, eos_idx = 1, 2
        beam_size = 3
        # def inference(model, src, num_steps, beam_size=3, sos_idx=1, eos_idx=2):
        test_src = torch.tensor([1, 7, 5, 4, 2]).unsqueeze(1).to(device)
        inference(
            model,
            test_src,
            num_steps=10,
            beam_size=beam_size,
            sos_idx=sos_idx,
            eos_idx=eos_idx)


if __name__ == "__main":
    unittest.main(verbosity=True)
