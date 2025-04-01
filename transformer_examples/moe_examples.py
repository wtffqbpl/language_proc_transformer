#! coding: utf-8

import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import utils.dlf as dlf


# 1. Transformer MoE components
class ExpertFFN(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0.1):
        super(ExpertFFN, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.GELU()  # GELU is common in Transformers
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class MoELayer(nn.Module):
    """
    Mixture of Experts Layer
    Using a dense gating network to determine the mixture of experts
    Notes: The gating network is a dense layer with softmax activation
    """
    def __init__(self, d_model, num_experts, d_ffn, dropout=0.1):
        super(MoELayer, self).__init__()
        self.d_model = d_model
        self.num_experts = num_experts

        # Experts list
        self.experts = nn.ModuleList(
            [ExpertFFN(d_model, d_ffn, dropout) for _ in range(num_experts)])

        # Gating network
        self.gating_network = nn.Linear(d_model, num_experts)

        # Add a simple balance parameter
        self.load_balancing_loss_coef = 0.01  # The weight could be changed.

    def forward(self, x):
        # x.shape: (batch_size, seq_len, d_model)

        # 1. Compute the gate scores
        # gating_logits.shape: (batch_size, seq_len, num_experts)
        gating_logits = self.gating_network(x)
        # gating_weights.shape: (batch_size, seq_len, num_experts)
        gating_weights = F.softmax(gating_logits, dim=-1)  # Softmax along the expert dim

        # 2. Compute the expert outputs
        expert_outputs = []
        for i in range(self.num_experts):
            expert_outputs.append(self.experts[i](x))

        # expert_outputs.shape: (num_experts, batch_size, seq_len, d_model)
        expert_outputs_tensor = torch.stack(expert_outputs, dim=0)

        # 3. weighted sum of expert outputs
        # gating_weights_expanded.shape: (batch_size, seq_len, num_experts, 1)
        gating_weights_expanded = gating_weights.unsqueeze(-1)
        # Adjust expert_output_tensor dim to (batch_size, seq_len, num_experts, d_model)
        expert_outputs_tensor_permuted = expert_outputs_tensor.permute(1, 2, 0, 3)

        # weighted_expert_outputs.shape: (batch_size, seq_len, num_experts, d_model
        weighted_expert_outputs = expert_outputs_tensor_permuted * gating_weights_expanded

        # final_output.shape: (batch_size, seq_len, d_model)
        final_output = torch.sum(weighted_expert_outputs, dim=2)  # Sum along the expert dim

        # Compute the load balancing loss
        # mean_expert_weights_per_token = gating_weights.mean(dim=0).mean(dim=0)
        # variance_expert_load = torch.var(mean_expert_weights_per_token)
        # load_balancing_loss = variance_expert_load

        # The common load balancing loss (From Switch Transformer paper)
        # f_i = fraction of tokens routed to expert i
        # P_i = fraction of total router probability assigned to expert i
        # loss = num_experts * sum(f_i * P_i)
        tokens_per_expert = gating_weights.sum(dim=0).sum(dim=0)  # Sum over batch and seq_len -> (num_experts)
        total_tokens = x.size(0) * x.size(1)
        fraction_tokens_per_expert = tokens_per_expert / total_tokens  # f_i

        # Average over batch and seq_len -> (num_experts) # P_i
        router_probs_per_expert = gating_weights.mean(dim=0).mean(dim=0)

        load_balancing_loss = self.num_experts * torch.sum(fraction_tokens_per_expert * router_probs_per_expert)

        # Return the final output and the load balancing loss
        return final_output, self.load_balancing_loss_coef * load_balancing_loss, gating_weights


class TransformerEncoderLayerWithMoE(nn.Module):
    """ The Transformer Encoder Layer with MoE """
    def __init__(self, d_model, nhead, num_experts, d_ffn, dropout=0.1):
        super(TransformerEncoderLayerWithMoE, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.moe_ffn = MoELayer(d_model, num_experts, d_ffn, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 1. Multi-head self-attention
        src2 = self.norm1(src)
        attn_output, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(attn_output)  # Residual connection

        # 2. MoE feed-forward network
        src3 = self.norm2(src)
        ffn_output, aux_loss, gating_weights = self.moe_ffn(src3)
        src = src + self.dropout2(ffn_output)  # Residual connection

        return src, aux_loss, gating_weights


class PatchEmbedding(nn.Module):
    """ Patch Embedding """
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size

        # Using Conv2d to simulate the patch embedding
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x.shape: (batch_size, in_channels, img_size, img_size)
        x = self.proj(x)  # (batch_size, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)  # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)

        return x


class ViTMoEClassifier(nn.Module):
    """
    Vision Transformer with MoE Classifier
    """
    def __init__(self, img_size=28, patch_size=7, in_channels=1, num_classes=10,
                 d_model=128, nhead=4, num_encoder_layers=4, num_experts=4,
                 d_ffn=256, dropout=0.1):
        super(ViTMoEClassifier, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, d_model)
        num_patches = self.patch_embed.n_patches

        # Adding CLS token and positional encoding
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d_model))  # +1 for the CLS token
        nn.init.trunc_normal_(self.pos_embed, std=0.2)  # initialization
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.dropout = nn.Dropout(dropout)

        # Transformer Encoder list
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayerWithMoE(d_model, nhead, num_experts, d_ffn, dropout)
            for _ in range(num_encoder_layers)
        ])

        # Classifier
        self.norm = nn.LayerNorm(d_model)  # Layer norm before the final output
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x.shape: (batch_size, in_channels, img_size, img_size)
        batch_size = x.shape[0]

        # 1. Patch Embedding
        x = self.patch_embed(x)  # shape: (batch_size, num_patches, d_model)

        # 2. Add CLS Token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # shape: (batch_size, 1, d_model)
        x = torch.cat((cls_tokens, x), dim=1)  # shape: (batch_size, num_patches + 1, d_model)

        # 3. Add positional encoding
        x = x + self.pos_embed
        x = self.dropout(x)

        # 4. Transformer Encoder layer
        total_aux_loss = 0.0
        all_gating_weights = []
        for layer in self.encoder_layers:
            x, aux_loss, gating_weights = layer(x)
            total_aux_loss += aux_loss  # Accumulate the auxiliary loss
            all_gating_weights.append(gating_weights)

        # 5. Classification
        # Use the CLS token (first token) output for classification
        cls_output = x[:, 0]
        cls_output = self.norm(cls_output)
        logits = self.classifier(cls_output)  # shape: (batch_size, num_classes)

        # Return logits, total auxiliary loss, and all gating weights
        return logits, total_aux_loss / len(self.encoder_layers), all_gating_weights


# ----- 2. data loader and pre-processing -----
def get_dataloaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))  # Normalization to [-1, 1]
    ])
    train_dataset = datasets.FashionMNIST(
        root='../data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(
        root='../data', train=False, download=True, transform=transform)

    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True))


def train(model: nn.Module,
          data_iter: DataLoader,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          device,
          num_epochs: int = 10) -> None:
    model.train()
    print(f'Starting training on {device}...')
    num_layers = len(model.encoder_layers)  # Used for calculating the auxiliary loss

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_main_loss = 0.0
        running_aux_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for i, (inputs, labels) in enumerate(data_iter):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            logits, avg_aux_loss, _ = model(inputs)  # Acquire logits and the average auxiliary loss

            # Compute the main loss
            main_loss = criterion(logits, labels)

            # Total loss = main loss + auxiliary loss
            total_loss = main_loss + avg_aux_loss

            # Backward forward
            total_loss.backward()
            optimizer.step()

            # Statistics
            running_loss += total_loss.item() * inputs.size(0)
            running_main_loss += main_loss.item() * inputs.size(0)
            running_aux_loss += avg_aux_loss.item() * inputs.size(0)

            _, predicted = torch.max(logits.data, 1)
            total_samples += labels.size(0)

            correct_predictions += (predicted == labels).sum().item()

            if (i + 1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_iter)}], ',
                      f'Loss: {running_loss / total_samples:.4f}, ',
                      f'(Main: {running_main_loss / total_samples:.4f}, ',
                      f'Aux: {running_aux_loss / total_samples:.4f})')

        epoch_loss = running_loss / total_samples
        epoch_main_loss = running_main_loss / total_samples
        epoch_aux_loss = running_aux_loss / total_samples
        epoch_acc = correct_predictions / total_samples
        print(f'Epoch {epoch+1} finished. ', f'Avg Total Loss: {epoch_loss:.4f}',
              f'(Main: {epoch_main_loss:.4f}, Aux: {epoch_aux_loss:.4f}), Accuracy: {epoch_acc:.4f}')

    print('Training finished.')


@torch.no_grad()  # Using decorator to avoid gradient tracking
def evaluate_model(model: nn.Module, data_iter: DataLoader, device) -> float:
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    print('Starting evaluation...')
    for inputs, labels in data_iter:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        logits, _, _ = model(inputs)  # Ignore the auxiliary loss and gating weights during evaluation

        _, predicted = torch.max(logits.data, 1)
        total_samples += labels.size(0)

        correct_predictions += (predicted == labels).sum().item()

    avg_loss = running_loss / total_samples
    accuracy = correct_predictions / total_samples
    print(f'Test Set: Average Loss: {avg_loss:.4f}',
          f'Accuracy: {accuracy:.4f}',
          f'({correct_predictions}/{total_samples})')
    return accuracy


class IntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.image_size = 28
        self.patch_size = 7  # 28x28 -> 4x4=16 patches
        self.in_channels = 1
        self.num_classes = 10
        self.d_model = 128  # Transformer dimension
        self.nhead = 4  # Number of heads in multi-head attention
        self.num_encoder_layers = 4  # Transformer encoder layers
        self.num_experts = 4  # Number of experts in MoE
        self.d_ffn = self.d_model * 2  # FFN/Expert middle layer dimension (usually 2*d_model)
        self.dropout = 0.1
        self.learning_rate = 0.001
        self.batch_size = 128
        self.num_epochs = 10  # Transformer may need more epochs to converge
        self.model_save_path = 'vit_moe_fashionmnist.pth'

        self.fashion_mnist_classes = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        self.device = dlf.devices()[0]
        print(f'Using device: {self.device}')

    def test_training(self):
        print('\n--- Training Phase ---')
        train_iter, test_iter = get_dataloaders(batch_size=self.batch_size)
        model = ViTMoEClassifier(
            img_size=self.image_size, patch_size=self.patch_size,
            in_channels=self.in_channels, num_classes=self.num_classes,
            d_model=self.d_model, nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers, num_experts=self.num_experts,
            d_ffn=self.d_ffn, dropout=self.dropout).to(self.device)

        criterion = nn.CrossEntropyLoss()
        # AdamW is also a good choice for Transformer models
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        train(model, train_iter, optimizer, criterion, self.device, num_epochs=self.num_epochs)

        # Save model
        torch.save(model.state_dict(), self.model_save_path)
        print(f'Model saved to {self.model_save_path}')

    def test_inference(self):
        if not os.path.exists(self.model_save_path):
            self.test_training()

        inference_model = ViTMoEClassifier(
            img_size=self.image_size, patch_size=self.patch_size,
            in_channels=self.in_channels, num_classes=self.num_classes,
            d_model=self.d_model, nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers, num_experts=self.num_experts,
            d_ffn=self.d_ffn, dropout=self.dropout).to(self.device)

        inference_model.load_state_dict(torch.load(self.model_save_path, map_location=self.device))

        inference_model.eval()
        print(f'Model loaded from {self.model_save_path}')

        _, inference_test_iter = get_dataloaders(batch_size=5)
        data_iter = iter(inference_test_iter)
        images, labels = next(data_iter)
        images, labels = images.to(self.device), labels.to(self.device)

        with torch.no_grad():
            logits, _, gating_weights = inference_model(images)
            _, predicted_classes = torch.max(logits, 1)

        images_np = images.cpu().numpy()
        labels_np = labels.cpu().numpy()
        predicted_classes_np = predicted_classes.cpu().numpy()
        gating_weights_np = gating_weights.cpu().numpy()

        # all_gating_weights is a list of tensors, one per layer
        # shape of each tensor: (batch_size, seq_len, num_experts)

        print('\nInference Results (Sample Batch):')
        fig, axes = plt.subplots(1, images.size(0), figsize=(15, 5))
        if images.size(0) == 1:
            axes = [axes]

        for i in range(images.size(0)):
            ax = axes[i]
            img = images_np[i].squeeze() * 0.5 + 0.5  # De-normalize
            ax.imshow(img, cmap='gray')
            ax.set_title(
                f'Pred: {self.fashion_mnist_classes[predicted_classes_np[i]]}\nTrue:',
                f'{self.fashion_mnist_classes[labels_np[i]]}')
            ax.axis('off')

            # Print the gating weights
            print(f'\n--- Sample {i+1} ---')
            print(f'True Label: {self.fashion_mnist_classes[labels_np[i]]}')
            print(f'Predicted Label: {self.fashion_mnist_classes[predicted_classes_np[i]]}')
            print('Gating Weights:')
            for expert_idx in range(self.num_experts):
                print(f'  Expert {expert_idx+1}: {gating_weights_np[i, expert_idx]:.4f}')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    unittest.main()
