#! coding: utf-8

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from pathlib import Path


# Multi-Head Self-Attention Module
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        """
        Initializes the multi-head self-attention mechanism.
        :param hidden_size: The total hidden dimension of the model
        :param num_heads: Number of attention heads
        :param dropout: Dropout probability
        """
        super(MultiHeadSelfAttention, self).__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divided by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Linear layers to project the input to queries, keys, and values
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        # Final output linear layer
        self.out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for multi-head self-attention
        :param x: (Tensor) Input tensor of shape (batch_size, seq_length, hidden_size)
        :return: Output tensor of the same shape.
        """
        batch_size, seq_length, hidden_size = x.size()

        # Linear projections
        q = self.query(x)  # (batch_size, seq_length, hidden_size)
        k = self.key(x)
        v = self.value(x)

        # Split into multiple heads and transpose
        # New shape: (batch_size, num_heads, seq_length, head_dim)
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention scores and scale
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Compute weighted sum of values
        context = torch.matmul(attn_probs, v)  # (batch_size, num_heads, seq_length, head_dim)

        # Concatenate heads and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, hidden_size)
        out = self.out(context)
        return out


# Feed Forward Network (FFN)
class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout):
        """
        The FFN applies two linear transformations with an activation in between.
        :param hidden_size: The size of the hidden state.
        :param intermediate_size: The size of the intermediate (expansion) layer.
        :param dropout: Dropout probability
        """
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        Forward pass for the feed forward network
        :param x: Input tensor
        :return: Output Tensor
        """
        x = self.linear1(x)
        x = F.gelu(x)  # GELU activation is standard in BERT
        x = self.dropout(x)
        x = self.linear2(x)
        return x


# Transformer Block (BERT Layer)
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, dropout):
        """
        A single transformer block consisting of multi-head attention and FFN.
        It includes residual connections and layer normalization.
        """
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(hidden_size, num_attention_heads, dropout)
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.feed_forward = FeedForward(hidden_size, intermediate_size, dropout)
        self.ff_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor):
        """
        Applies the transformer block on input x
        :param x: shape (batch_size, seq_length, hidden_size)
        """
        # Multi-head attention sublayer with residual connection and normalization
        attn_output = self.attention(x)
        x = self.attention_norm(x + attn_output)

        # Feed forward sublayer with residual connection and normalization
        ff_output = self.feed_forward(x)
        x = self.ff_norm(x + ff_output)
        return x


# Mini BERT Model
class MiniBert(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_hidden_layers, num_attention_heads,
                 intermediate_size, max_position_embeddings, dropout):
        """
        A simplified BERT model that includes token embeddings, positional embeddings,
        and multiple transformer blocks.
        :param vocab_size: Size of the vocabulary.
        :param hidden_size: Dimensionality of the token embeddings.
        :param num_hidden_layers: Number of transformer blocks.
        :param num_attention_heads: Number of attention heads in each block.
        :param intermediate_size: Hidden size in the feed forward network.
        :param max_position_embeddings: Maximum sequence length.
        :param dropout: Dropout probability
        """
        super(MiniBert, self).__init__()

        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Stack transformer blocks (BERT layers)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_attention_heads, intermediate_size, dropout)
            for _ in range(num_hidden_layers)
        ])

    def forward(self, input_ids: torch.Tensor):
        """ Forward pass of the MiniBERT model """
        batch_size, seq_length = input_ids.size()
        # Create position ids for each token in the sequence
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)

        # Sum token and positional embeddings
        token_embed = self.token_embeddings(input_ids)
        position_embed = self.position_embeddings(position_ids)
        x = token_embed + position_embed

        # Apply layer normalization and dropout
        x = self.layer_norm(x)
        x = self.dropout(x)

        # Pass through each transformer block
        for layer in self.layers:
            x = layer(x)

        return x


# BERT for sequence classification
class BertForSequenceClassification(nn.Module):
    def __init__(self, bert_model, hidden_size, num_classes, dropout):
        """
        Adds a classification head on top of the MiniBERT model.
        It uses the hidden state of the first token([CLS]) for classification.
        :param bert_model: An instance of MiniBert.
        :param hidden_size: The hidden dimension (same as BERT's output dimension)
        :param num_classes: Number of output classes.
        :param dropout: Dropout probability
        """
        super(BertForSequenceClassification, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids):
        # Forward pass for sequence classification
        # Get BERT outputs (hidden states)
        bert_output = self.bert(input_ids)  # shape: (batch_size, seq_length, hidden_size)
        # Use the hidden state corresponding to the first token([CLS])
        cls_token = bert_output[:, 0, :]  # Shape: (batch_size, hidden_size)
        cls_token = self.dropout(cls_token)
        logits = self.classifier(cls_token)
        return logits


class IntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        # Hyperparameters for the mini BERT model
        self.vocab_size = 30522  # Typical BERT vocab size (can be adjusted)
        self.hidden_size = 64  # Reduced hidden size for faster computation
        self.num_hidden_layers = 2  # Fewer transformer blocks
        self.num_attention_heads = 2  # Fewer attention heads
        self.intermediate_size = 128  # Size of the intermediate (feed forward) layer
        self.max_position_embeddings = 32  # maximum sequence length (adjustable)
        self.dropout = 0.1
        self.num_classes = 2  # For binary classification
        self.batch_size = 8
        self.bert_model_path = os.path.join(str(Path(__file__).resolve().parent), 'bert_model.pth')

    def test_bert_training(self):
        # instantiate the MiniBERT model
        mini_bert = MiniBert(self.vocab_size,
                             self.hidden_size,
                             self.num_hidden_layers,
                             self.num_attention_heads,
                             self.intermediate_size,
                             self.max_position_embeddings,
                             self.dropout)

        # Create the full model with a classification head
        model = BertForSequenceClassification(mini_bert,
                                              self.hidden_size,
                                              self.num_classes,
                                              self.dropout)

        # Define optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        # Dummy Training Example
        # Create dummy input data: random token ids of shape (batch_size, max_position_size)
        inputs = torch.randint(0, self.vocab_size, (self.batch_size, self.max_position_embeddings))
        # Create dummy labels for binary classification (0 or 1)
        labels = torch.randint(0, self.num_classes, (self.batch_size, ))

        model.train()
        print('Starting training...')
        for epoch in range(10):
            optimizer.zero_grad()
            # Forward pass: compute logits
            logits = model(inputs)
            # Compute loss
            loss = loss_fn(logits, labels)
            # Backward pass: compute gradients and update parameters
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

        # Inference example
        model.eval()
        with torch.no_grad():
            logits = model(inputs)
            # use argmax to get predicted class indices
            predictions = torch.argmax(logits, dim=1)
            print('Predictions: ', predictions.tolist())


if __name__ == '__main__':
    unittest.main(verbosity=True)
