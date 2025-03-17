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


# MoE Layer replacing FFN
class MoE(nn.Module):
    """ MoE layer contains multiple FFN expert and one Gated network """
    def __init__(self, hidden_size, intermediate_size, dropout, num_experts=4):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        # Define multiple experts, each expert is a FFN layer.
        self.experts = nn.ModuleList([
            FeedForward(hidden_size, intermediate_size, dropout) for _ in range(num_experts)
        ])

        # Gated network: Output the expert score
        self.gate = nn.Linear(hidden_size, num_experts)

    def forward(self, x: torch.Tensor):
        # Compute each expert score for the input token, and calculate the probability using softmax.
        gate_logits = self.gate(x)
        gate_weights = F.softmax(gate_logits, dim=-1)  # (batch_size, seq_length, num_experts)

        # Compute each expert output, each tensor shape in the list is (batch_size, seq_length, hidden_size)
        expert_outputs = [expert(x) for expert in self.experts]
        # Concat expert outputs. Shape: (batch_size, seq_length, num_experts, hidden_size)
        expert_outputs = torch.stack(expert_outputs, dim=2)

        # Weighted output for each expert scores
        gate_weights = gate_weights.unsqueeze(-1)  # shape: (batch_size, seq_length, num_experts, 1)
        output = torch.sum(gate_weights * expert_outputs, dim=2)  # shape: (batch_size, req_length, hidden_size)
        return output


class TransformerBlockMoE(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, dropout, num_experts=4):
        super(TransformerBlockMoE, self).__init__()
        self.attention = MultiHeadSelfAttention(hidden_size, num_attention_heads, dropout)
        self.attention_norm = nn.LayerNorm(hidden_size)

        # Using MoE layer instead of FeedForward layer
        self.moe = MoE(hidden_size, intermediate_size, dropout, num_experts)
        self.ff_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor):
        # self-attention + residuals + layer normalization
        attn_output = self.attention(x)
        x = self.attention_norm(x + attn_output)
        # MoE + residuals + layer normalization
        moe_output = self.moe(x)
        x = self.ff_norm(x + moe_output)
        return x


class MiniBertMoE(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_hidden_layers, num_attention_heads,
                 intermediate_size, max_position_embeddings, num_experts, dropout):
        super(MiniBertMoE, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TransformerBlockMoE(hidden_size, num_attention_heads, intermediate_size, dropout, num_experts)
            for _ in range(num_hidden_layers)
        ])

    def forward(self, input_ids: torch.Tensor):
        batch_size, seq_length = input_ids.size()
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_length)
        x = self.token_embeddings(input_ids) + self.position_embeddings(position_ids)
        x = self.layer_norm(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)
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


class BertForSequenceClassificationMoE(nn.Module):
    def __init__(self, bert_model, hidden_size, num_classes, dropout):
        super(BertForSequenceClassificationMoE, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor):
        bert_output = self.bert(input_ids)
        cls_token = bert_output[:, 0, :]
        cls_token = self.dropout(cls_token)
        logits = self.classifier(cls_token)
        return logits


def bert_training(model, optimizer, loss_fn, inputs, labels):
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


class DistributeMoE(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout, num_experts=4, devices=None):
        """
        MoE layer that distributes experts across multiple devices and adds a load balancing loss.
        """
        super(DistributeMoE, self).__init__()
        self.num_experts = num_experts
        if devices is None:
            devices = [torch.device('cpu')] * num_experts
        self.devices = devices

        # Create experts and place each expert on a device (round-robin assignment)
        self.experts = nn.ModuleList([
            FeedForward(hidden_size, intermediate_size, dropout).to(self.devices[i % len(self.devices)])
            for i in range(num_experts)
        ])

        # Gating networkd (kept on main device) outputs scores for each expert.
        self.gate = nn.Linear(hidden_size, num_experts)

    def forward(self, x: torch.Tensor):
        batch_size, seq_length, hidden_size = x.size()

        # Compute gating scores and convert to probability distribution.
        gate_logits = self.gate(x)  # shape: (batch_size, seq_length, num_experts)
        gate_probs = F.softmax(gate_logits, dim=-1)

        # Compute load balance loss:
        importance = gate_probs.sum(dim=(0, 1))  # shape: (num_experts, )
        mean_importance = importance.mean()
        std_importance = importance.std()
        load_balancing_loss = std_importance / (mean_importance + 1e-8)

        # For each expert, move data and model to related devices, and compute output and weighted sum
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            # The i-th expert gate weight. Shape: (batch_size, seq_length, 1)
            weight = gate_probs[:, :, i].unsqueeze(-1)
            # Move inputs to the related device
            x_expert = x.to(self.devices[i % len(self.devices)])
            expert_out = expert(x_expert)  # shape: (batch_size, seq_length, hidden_size)
            # Move expert output to the main device (to ensure the following computation on the same device)
            expert_out = expert_out.to(x.device)
            expert_outputs.append(weight * expert_out)

        # Weighted sum all expert outputs
        output = sum(expert_outputs)
        return output, load_balancing_loss


class TransformerBlockDistributedMoE(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout, num_experts=4, devices=None):
        """
        A single Transformer block that includes a self-attention sub-layer and a MoE layer (replacing the FFN).
        """
        super(TransformerBlockDistributedMoE, self).__init__()
        self.attention = MultiHeadSelfAttention(hidden_size, num_heads, dropout)
        self.attention_norm = nn.LayerNorm(hidden_size)

        # Distributed MoE
        self.moe = DistributeMoE(hidden_size, intermediate_size, dropout, num_experts, devices)
        self.ff_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor):
        # Self-attention sub-layer with residual connections
        attn_output = self.attention(x)
        x = self.attention_norm(x + attn_output)
        # MoE (FFN) sub-layer with residual connection.
        moe_output, load_loss = self.moe(x)
        x = self.ff_norm(x + moe_output)
        return x, load_loss


# Mini Transformer Model with MoE Layers
class MiniTransformerMoE(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, intermediate_size,
                 max_seq_length, dropout, num_experts=4, devices=None):
        """
        A small Transformer model with token and positional embeddings.
        Some Transformer blocks use MoE layers (distributed on different devices).
        """
        super(MiniTransformerMoE, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_seq_length, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TransformerBlockDistributedMoE(hidden_size, num_heads, intermediate_size, dropout, num_experts, devices)
            for _ in range(num_layers)
        ])

    def forward(self, input_ids: torch.Tensor):
        batch_size, seq_length = input_ids.size()

        # Create positional ids.
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_length)
        x = self.token_embeddings(input_ids) + self.position_embeddings(position_ids)
        x = self.layer_norm(x)
        x = self.dropout(x)

        total_load_loss = 0.0
        for layer in self.layers:
            x, load_loss = layer(x)
            total_load_loss += load_loss
        return x, total_load_loss


# Transformer Model for Sequence Classification with MoE
class TransformerForSeqenceClassificationMoE(nn.Module):
    def __init__(self, transformer, hidden_size, num_classes, dropout):
        """
        Sequence classification model using a Transformer backbone with MoE layers.
        The [CLS] token representation is used for classification.
        """
        super(TransformerForSeqenceClassificationMoE, self).__init__()
        self.transformer = transformer
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor):
        transformer_output, load_loss = self.transformer(input_ids)
        # use the [CLS] token (first token) for classification
        cls_token = transformer_output[:, 0, :]
        cls_token = self.dropout(cls_token)
        logits = self.classifier(cls_token)
        return logits, load_loss


class IntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        # Hyperparameters for the mini BERT model
        self.vocab_size = 30522  # Typical BERT vocab size (can be adjusted)
        self.hidden_size = 64  # Reduced hidden size for faster computation
        self.num_hidden_layers = 2  # Fewer transformer blocks
        self.num_attention_heads = 2  # Fewer attention heads
        self.intermediate_size = 128  # Size of the intermediate (feed forward) layer
        self.max_position_embeddings = 32  # maximum sequence length (adjustable)
        self.max_seq_length = 32
        self.dropout = 0.1
        self.num_experts = 4  # The expert number
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

        bert_training(model, optimizer, loss_fn, inputs, labels)

    def test_bert_with_moe(self):
        # instantiate the MiniBERT model
        mini_bert_moe = MiniBertMoE(self.vocab_size,
                                    self.hidden_size,
                                    self.num_hidden_layers,
                                    self.num_attention_heads,
                                    self.intermediate_size,
                                    self.max_position_embeddings,
                                    self.num_experts,
                                    self.dropout)

        # Create the full model with a classification head
        model = BertForSequenceClassificationMoE(mini_bert_moe,
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

        bert_training(model, optimizer, loss_fn, inputs, labels)

    def test_distributed_moe(self):
        if torch.cuda.is_available() and torch.cuda.device_count() > 2:
            devices = [torch.device('cuda:0'), torch.device('cuda:1')]
        else:
            devices = [torch.device('cpu'), torch.device('cpu')]

        mini_transformer = MiniTransformerMoE(
            self.vocab_size,
            self.hidden_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.intermediate_size,
            self.max_seq_length,
            self.dropout,
            self.num_experts,
            devices)

        # Build the classification model.
        model = TransformerForSeqenceClassificationMoE(mini_transformer,
                                                       self.hidden_size,
                                                       self.num_classes,
                                                       self.dropout)

        # Move model to a main device (e.g. devices[0])
        main_device = devices[0]
        model.to(main_device)

        # Define optimizer and loss function.
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        # Create dummy input data.
        inputs = torch.randint(0, self.vocab_size, (self.batch_size, self.max_seq_length)).to(main_device)
        labels = torch.randint(0, self.num_classes, (self.batch_size,)).to(main_device)

        # Training loop
        model.train()
        for epoch in range(10):
            optimizer.zero_grad()

            logits, load_loss = model(inputs)
            classification_loss = loss_fn(logits, labels)

            total_loss = classification_loss + 0.01 * load_loss
            total_loss.backward()
            optimizer.step()

            print(f'Epoch {epoch+1}, ',
                  f'Total loss: {total_loss.item():.4f}, ',
                  f'Classification Loss: {classification_loss.item():.4f}, ',
                  f'Load Loss: {load_loss.item():.4f}')

        # Inference demonstration.
        model.eval()
        with torch.no_grad():
            logits, _ = model(inputs)
            predictions = torch.argmax(logits, dim=1)
            print('Predictions: ', predictions.tolist())


if __name__ == '__main__':
    unittest.main(verbosity=True)
