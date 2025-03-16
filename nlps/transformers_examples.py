#! coding: utf-8
import math
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import pipeline
from transformers import AutoTokenizer, AutoConfig
from bertviz.transformers_neuron_view import BertModel
from bertviz.neuron_view import show
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import utils.dlf as dlf


def scaled_dot_product_attention(query, key, value, mask=None):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(dim_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    weights = F.softmax(scores, dim=-1)
    return weights.bmm(value)


class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super(AttentionHead, self).__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, hidden_state):
        attn_outputs = scaled_dot_product_attention(
            self.q(hidden_state), self.k(hidden_state), self.v(hidden_state))
        return attn_outputs


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_state):
        x = torch.cat([h(hidden_state) for h in self.heads], dim=-1)
        x = self.output_linear(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):
    # We use pre-layer-normalization. This is the most common arrangement
    # found in the literature; it places layer normalization within the
    # span of the skip connections. This tends to be much more stable
    # during training, and it does not usually require any learning rate
    # warm-up.
    def __init__(self, config):
        super(TransformerEncoderLayer, self).__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        # Apply layer normalization and then copy input into query, key, value
        hidden_state = self.layer_norm_1(x)
        # Apply attention with a skip connection
        x = x + self.attention(hidden_state)
        # Apply feed-forward layer with a skp connection
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


# Positional embeddings are based on a simple, yet very effective idea: augment
# the token embeddings with a position-dependent pattern of values arranged in a
# vector. If the pattern is characteristic for each position, the attention heads
# and feed-forward layers in each stack can learn to incorporate positional
# information into their transformations.
# there are several ways to achieve this, and one of the most popular approaches is
# to use a learnable pattern, especially when the pretraining dataset is sufficiently
# large. This works exactly the same way as the token embeddings, but using the
# position index instead of the token ID as input. With this approach, an efficient
# way of encoding the positions of tokens is learned during pretraining.
#
# While learnable position embeddings are easy to implement and widely used, there
# are some alternatives:
# *Absolute positional representations*
# Transformer models can use static patterns consisting of modulated sine and cosine
# signals to encode the positions of the tokens. This works especially well when
# there are no large volumes of data available.
# *Relative positional representations*
# Although absolute positions are important, one can argue that when computing an
# embedding, the surrounding tokens are most important. Relative positional
# representations follow that intuition and encode the relative positions between
# tokens. This cannot be set up by just introducing a new relative embeddings layer
# at the beginning, since the relative embedding changes for each token depending
# on where from the sequence we are attending to it. Instead, the attention mechanism
# itself is modified with additional terms that take the relative position between
# itself is modified with additional terms that take the relative position between
# tokens into account. Models such as DeBERTa use such representations.
class Embeddings(nn.Module):
    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout()

    def forward(self, input_ids):
        # Create position IDs for input sequence
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)
        # Create token and position embeddings
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        # Combine token and position embeddings
        embeddings = token_embeddings + position_embeddings
        # Layer normalization here mainly serves to stabilize training and balance numerical
        # scales. Specifically:
        #  * Stabilize Training: The token embeddings and positional embeddings are
        #    learned separately and might end up having different numerical distributions.
        #    Directly adding them together could lead to significant differences in their
        #    distributions, which may adversely affect gradient flow and model convergence.
        #    Layer normalization helps mitigate this internal covariate shift, ensure that the
        #    data passed to subsequent layers is more stable.
        #  * Balance Numerical Scales: The normalization operation ensures that all features
        #    have a consistent scale. This consistency allows the model to better capture and
        #    integrate information in the following non-linear transformations, ultimately
        #    enhancing overall performance.
        # In summary, applying layer normalization to the combined embeddings helps make the
        # training process smoother, accelerates convergence, and improves the model's
        # robustness and performance.
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList([TransformerEncoderLayer(config)
                                     for _ in range(config.num_hidden_layers)])

    def forward(self, x):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerForSequenceClassification(nn.Module):
    def __init__(self, config):
        super(TransformerForSequenceClassification, self).__init__()
        self.encoder = TransformerEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.encoder(x)[:, 0, :]  # Select hidden state of [CLS] token
        x = self.dropout(x)
        x = self.classifier(x)
        return x
    pass


# The decoder has two attention sublayers:
# Masked multi-head self-attention layer
# Ensures that the tokens we generate at each timestep are only based on the past
# outputs and the current token being predicted. Without this, the decoder could
# cheat during training by simply copying the target translations; masking the
# inputs ensures the task is not trivial.
# Encoder-decoder attention layer
# Performs multi-head attention over the output key and value vectors of the encoder
# stack, with the intermediate representations of the decoder acting as the queries.
# This way the encoder-decoder attention layer learns how to relate tokens from two
# different sequences, such as two different languages. The decoder has access to
# the encoder keys and values in each block.

class IntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model_ckpt = 'bert-base-uncased'

        self.device = dlf.devices()[0]
        self.text = """
        Dear Amazon, last week I ordered an Optimus Prime action figure
        from your online store in Germany. Unfortunately, when I opened the package,
        I discovered to my horror that I had been sent an action figure of Megatron
        instead! As a lifelong enemy of the Decepticons, I hope you can understand my
        dilemma. To resolve the issue, I demand an exchange of megatron for the
        Optimus Prime figure I ordered. Enclosed are copies of my records concerning
        this purchase. I expect to hear from you soon. Sincerely, Bumblebee.
        """
        pass

    def test_text_generation(self):
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        generator = pipeline('text-generation', model='gpt2', tokenizer=tokenizer, device=self.device)
        generator.pad_token = tokenizer.eos_token
        response = "Dear Bumblebee, I am sorry to hear that your order was mixed up."
        prompt = self.text + '\n\nCustomer service response:\n' + response
        output = generator(prompt, max_length=512, truncation=True, pad_token_id=tokenizer.eos_token_id)
        print(output[0]['generated_text'])

    def test_show_qkv(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt)
        model = BertModel.from_pretrained(self.model_ckpt)
        text = 'time flies like an arrow'
        html_content = show(model, 'bert', tokenizer, text, display_mode='light', layer=0, head=8)

    def test_auto_config(self):
        model_ckpt = 'bert-base-uncased'
        config = AutoConfig.from_pretrained(model_ckpt)
        print(config)
        token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        print(token_emb)

        self.assertTrue(True)

    def test_multi_head_attention(self):
        text = 'time flies like an arrow'
        # Create tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt)

        # Tokenize input text, and return PyTorch tensor
        inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False)

        # Load model configs
        config = AutoConfig.from_pretrained(self.model_ckpt)

        # Create Embedding layer
        token_emb = nn.Embedding(config.vocab_size, config.hidden_size)

        # Current hyperparameters
        batch_size = inputs.input_ids.shape[0]
        seq_len = inputs.input_ids.shape[1]
        hidden_size = config.hidden_size

        # Embedding inputs
        inputs_embeds = token_emb(inputs.input_ids)
        self.assertEqual(torch.Size([batch_size, seq_len, hidden_size]), inputs_embeds.shape)

        # Create multi-head attention module
        multi_head_attn = MultiHeadAttention(config)

        # Compute attention results
        attn_output = multi_head_attn(inputs_embeds)

        self.assertEqual(torch.Size([batch_size, seq_len, hidden_size]), attn_output.shape)

        # Feedforward layer
        feed_forward = FeedForward(config)
        ff_outputs = feed_forward(attn_output)
        self.assertEqual(torch.Size([batch_size, seq_len, hidden_size]), ff_outputs.shape)

        # Create transformer encoder layer
        encoder_layer = TransformerEncoderLayer(config)
        encoder_outputs = encoder_layer(inputs_embeds)
        self.assertEqual(torch.Size([batch_size, seq_len, hidden_size]), encoder_outputs.shape)

        # Create embedding layer
        embedding_layer = Embeddings(config)
        embedding_output = embedding_layer(inputs.input_ids)
        self.assertEqual(torch.Size([batch_size, seq_len, hidden_size]), embedding_output.shape)

        # Create Encoder module
        encoder = TransformerEncoder(config)
        encoder_outputs = encoder(inputs.input_ids)
        self.assertEqual(torch.Size([batch_size, seq_len, hidden_size]), encoder_outputs.shape)

        # Before initializing the model, we need to define how many classes we would like
        # to predict:
        config.num_labels = 3
        encoder_classifier = TransformerForSequenceClassification(config)
        encoder_classifier_output = encoder_classifier(inputs.input_ids)

        self.assertEqual(torch.Size([batch_size, config.num_labels]), encoder_classifier_output.shape)

        print(encoder)


if __name__ == '__main__':
    unittest.main(verbosity=True)