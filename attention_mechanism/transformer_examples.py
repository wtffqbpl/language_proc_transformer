#! coding: utf-8

import unittest
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import utils.dlf as dlf


# Set random seed for reproducibility
torch.manual_seed(42)

# Special token definitions
pad_token = 0  # Padding token
bos_token = 1  # beginning-of-sequence token
eos_token = 2  # end-of-sequence token


# Define the positional encoding module
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Initialize the PositionalEncoding module.
        :param d_model: the dimension of the embeddings.
        :param dropout: dropout rate.
        :param max_len: maximum length of the sequences.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a long enough 'pe' matrix with shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Compute the positional encodings once in long space.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(1)  # shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to the input embeddings.
        :param x: input embeddings with shape (seq_len, batch_size, d_model)
        :return:  the embeddings with positional encodings added.
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# Define the Transformer-based model
class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, target_vocab_size, d_model=32, num_head=4,
                 num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=64,
                 dropout=0.1):
        """
        Initialize the TransformerModel.
        :param src_vocab_size:  size of the source vocabulary.
        :param target_vocab_size: size of the target vocabulary.
        :param d_model: dimension of embeddings.
        :param num_head: number of attention heads.
        :param num_encoder_layers: number of encoder layers.
        :param num_decoder_layers: number of decoder layers.
        :param dim_feedforward: dimension of the feedforward network.
        :param dropout: dropout rate.
        """
        super(TransformerModel, self).__init__()
        self.d_model = d_model

        # Create embedding layers for source and target sequences
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.target_embedding = nn.Embedding(target_vocab_size, d_model)

        # Positional encoding for both source and target
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)

        # Create the Transformer module
        self.transformer = nn.Transformer(d_model, num_head, num_encoder_layers,
                                          num_decoder_layers, dim_feedforward, dropout)

        # Final output layer to map the transformer output to vocabulary logits
        self.fc_out = nn.Linear(d_model, target_vocab_size)

    def forward(self, src, target, src_mask=None, target_mask=None,
                src_padding_mask=None, target_padding_mask=None, memory_key_padding_mask=None):
        """
        Forward pass for the Transformer model.
        :param src: source sequence (seq_len, batch_size)
        :param target: target_sequence (seq_len, batch_size)
        :param src_mask: source mask
        :param target_mask: target mask
        :param src_padding_mask: source padding mask
        :param target_padding_mask: target padding mask
        :param memory_key_padding_mask: memory key padding mask
        :return: Output logits with shape (target_seq_len, batch_size, target_vocab_size)
        """
        # Embed and add positional encoding for source
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)

        # Embed and add position encoding for target
        target_emb = self.target_embedding(target) * math.sqrt(self.d_model)
        target_emb = self.pos_decoder(target_emb)

        # Pass through Transformer
        output = self.transformer(src_emb, target_emb, src_mask, target_mask,
                                  src_padding_mask, target_padding_mask, memory_key_padding_mask)

        # Project output to vocabulary dimension
        output = self.fc_out(output)
        return output


def generate_square_subsequent_mask(sz):
    """
    Generate a square mask for the sequence. The masked positions are filled with -inf
    This mask is used to prevent the decoder from "seeing" future tokens.
    :param sz: size of the mask (sequence length)
    :return: A mask tensor of shape (sz, sz)
    """
    mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
    mask = mask.float().masked_fill(mask, float('-inf'))
    return mask


def greedy_decode(model: TransformerModel, src: torch.Tensor, max_len, device):
    """
    Greedy decoding for inference.
    :param model: the trained Transformer model.
    :param src: source sequence tensor with shape (src_seq_len, 1)
    :param mask_len: maximum length of the generated sequence.
    :param device: device (cpu or cuda or mps)
    :return: A list of token indices representing the generated sequence.
    """
    model.eval()
    src = src.to(device)
    src_mask = torch.zeros((src.size(0), src.size(0)), device=device).type(torch.float32)
    memory = model.transformer.encoder(model.pos_encoder(model.src_embedding(src) * math.sqrt(model.d_model)), src_mask)

    ys = torch.tensor([[bos_token]], device=device)

    for i in range(max_len):
        target_mask = generate_square_subsequent_mask(ys.size(0)).to(device)
        out = model.transformer.decoder(
            model.pos_decoder(model.target_embedding(ys) * math.sqrt(model.d_model)),
            memory,
            tgt_mask=target_mask
        )
        out = model.fc_out(out)

        # Get the token probabilities from the last time step
        prob = out[-1, 0, :]
        next_token = torch.argmax(prob).item()
        ys = torch.cat([ys, torch.tensor([[next_token]], device=device)], dim=0)

        if next_token == eos_token:
            break
    return ys.flatten().tolist()


class IntegrationTest(unittest.TestCase):
    def test_transformer_model(self):
        # hyperparameters
        src_vocab_size = 20  # including PAD, BOS, EOS and other tokens
        target_vocab_size = 20
        d_model = 32
        num_head = 4
        num_encoder_layers = 3
        num_decoder_layers = 3
        dim_feedforward = 64
        dropout = 0.1

        # Training parameters
        num_epochs = 20
        batch_size = 32
        seq_length = 10  # length of the random sequence (without BOS/EOS tokens)
        learning_rate = 0.001

        device = dlf.devices()[0]

        # Initialize the Transformer model
        model = TransformerModel(src_vocab_size, target_vocab_size, d_model,num_head,
                                 num_encoder_layers, num_decoder_layers, dim_feedforward,
                                 dropout).to(device)

        # Define loss function and optimizer
        loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop on a simple copy task:
        # Given a random sequence (source), the target is the same sequence
        # The target input sequence is BOS_TOKEN followed by the source sequence.
        # The target output sequence is the source sequence followed by EOS_TOKEN.
        for epoch in range(1, num_epochs + 1):
            model.train()

            epoch_loss = 0
            # Generate a batch of random sequences
            src_batch = torch.randint(3, src_vocab_size, (seq_length, batch_size))
            # Create target sequence with BOS and EOS tokens
            target_input = torch.cat([torch.full((1, batch_size), bos_token), src_batch], dim=0)
            target_output = torch.cat([src_batch, torch.full((1, batch_size), eos_token)], dim=0)

            # Create target mask to prevent the decoder from looking ahead
            target_mask = generate_square_subsequent_mask(target_input.size(0)).to(device)

            src_batch, target_input, target_output = src_batch.to(device), target_input.to(device), target_output.to(device)

            optimizer.zero_grad()
            # Forward pass: output shape (target_seq_len, batch_size, target_vocab_size)
            output = model(src_batch, target_input, target_mask=target_mask)

            # Reshape output and target for loss computation
            output_dim = output.shape[-1]
            output = output.view(-1, output_dim)  # shape: ((tgt_seq_len * batch), tgt_vocab_size)
            target_output = target_output.view(-1)  # shape: ((tgt_seq_len * batch))

            loss = loss_fn(output, target_output)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if epoch % 2 == 0:
                print(f'Epoch {epoch}, Loss: {epoch_loss:.4f}')

        # Inference on a new random sequence
        model.eval()
        with torch.no_grad():
            # Generate a single random source sequence
            src_example = torch.randint(3, src_vocab_size, (seq_length, 1))  # shape: (seq_length, 1)
            print('\nSource sequence: ', src_example.flatten().tolist())

            # Create the target sequence for comparison (ground truth copy)
            target_example = src_example.flatten().tolist() + [eos_token]
            print('Target sequence: ', target_example)

            # Perform greedy decoding
            generated_sequence = greedy_decode(model, src_example, seq_length + 5, device)
            print('Predicted sequence: ', generated_sequence)

        self.assertTrue(True)


# PyTorch Transformer tutorials
# https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html


if __name__ == '__main__':
    unittest.main(verbosity=True)
