#! coding: utf-8

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights
import numpy as np
import os
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import utils.dlf as dlf


# Model Components
class ResNet50Encoder(nn.Module):
    def __init__(self, embed_dim=512, use_pretrained=False):
        super(ResNet50Encoder, self).__init__()
        weights = ResNet50_Weights.DEFAULT if use_pretrained else None
        resnet = models.resnet50(weights=weights)
        modules = list(resnet.children())[:-1]  # Remove the last fully connected layer
        self.backbone = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, embed_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding to provide sequence models with position information.
    """

    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input tensor.
        x.shape: (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1)].to(x.device)


class SimpleTransformerTextEncoder(nn.Module):
    """
    Text encoder using a simple transformer architecture.
    """

    def __init__(self, vocab_size, embed_dim=512, num_heads=8, ff_dim=2048, num_layers=6, max_len=77):
        super(SimpleTransformerTextEncoder, self).__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)  # Token embedding
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)  # Positional encoding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, embed_dim)  # Final linear layer to embed_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer encoder.
        tokens.shape: (batch_size, seq_len)
        """
        x = self.token_embed(tokens)  # Shape: (batch_size, seq_len, embed_dim)
        x = self.pos_encoder(x)  # Add positional encoding
        x = x.permute(1, 0, 2)  # Change shape to (seq_len, batch_size, embed_dim) for transformer
        x = self.transformer_encoder(x)  # Transformer encoding
        x = x.permute(1, 0, 2)  # Change back to (batch_size, seq_len, embed_dim)
        x = x[:, -1, :]  # Take the last token's output
        x = self.fc(x)
        return x


class CLIP(nn.Module):
    """
    CLIP model combining image and text encoders for contrastive learning.
    """

    def __init__(self,
                 class_names,
                 templates,
                 vocab,
                 max_len=40,
                 embed_dim=512,
                 pretrained_image=False):
        super(CLIP, self).__init__()
        self.image_encoder = ResNet50Encoder(embed_dim, use_pretrained=pretrained_image)
        self.text_encoder = SimpleTransformerTextEncoder(vocab_size=len(vocab),
                                                         embed_dim=embed_dim,
                                                         max_len=max_len)

        self.max_len = max_len
        self.templates = templates
        self.class_names = class_names

        # Precompute prompt token IDs: shape [num_classes, num_templates, max_len]
        prompt_ids = []
        for cls in class_names:
            ids_per_cls = []
            for tmpl in templates:
                tokens = tokenizer(tmpl.format(cls), vocab, max_len)
                ids_per_cls.append(tokens)
            prompt_ids.append(ids_per_cls)
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long)  # [C, T, L]
        self.register_buffer('prompt_ids', prompt_tensor)

        # Log temperature parameter for scaling the logits
        self.logit_scale = nn.Parameter(torch.ones([])) * np.log(1 / 0.07)  # Initial value for logit scale

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        :param images: [B, 3, H, W]
        :param labels: [B]
        """
        device = images.device

        # Encode images and text
        image_features = self.image_encoder(images)  # Shape: (batch_size, embed_dim)
        # Text features: gather per-sample prompts
        B = labels.size(0)
        # prompt_ids: [C, T, L] -> select by labels -> [B, T, L]
        selected = self.prompt_ids[labels]  # [B, T, L]
        flat = selected.view(-1, self.max_len).to(device)  # [B * T, L]
        text_features_flat = self.text_encoder(flat)  # [B * T, D]
        text_features = text_features_flat.view(B, len(self.templates), -1).mean(1)  # [B, D]

        # Normalize features
        image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

        # Scale logits by temperature
        logit_scale = self.logit_scale.exp()
        # Similarity matrix
        logits = logit_scale * image_norm @ text_norm.t()  # Shape: (batch_size, batch_size)

        return logits


def clip_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    Contrastive loss function for CLIP model.
    """
    batch_size = logits.size(0)
    labels = torch.arange(batch_size, device=logits.device)

    # Image-to-text loss
    loss_i = F.cross_entropy(logits, labels)

    # Text-to-image loss
    loss_t = F.cross_entropy(logits.t(), labels)

    return (loss_i + loss_t) / 2


def build_vocab(class_names, templates, max_len):
    """ Build vocabulary from templates and class names """
    words = set()
    for tmpl in templates:
        for cls in class_names:
            text = tmpl.format(cls)

            # Lowercase, remove punctuation
            for w in text.lower().replace('.', '').split():
                words.add(w)

    vocab = {w: i + 1 for i, w in enumerate(sorted(words))}
    vocab['<pad>'] = 0
    return vocab


def tokenizer(text, vocab, max_len):
    """ Tokenizer """
    tokens = [vocab.get(w, 0) for w in text.lower().replace('.', '').split()]
    # pad or truncate
    if len(tokens) < max_len:
        tokens = tokens + [vocab['<pad>']] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return tokens


class CIFARTextImageDataset(Dataset):
    def __init__(self, root: str = './data', train: bool = True, transform=None):
        self.cifar10 = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
        self.classes = [
            'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
        ]

        self.templates = [
            "a photo of a {}.",
            "a rendering of a {}.",
            "a sketch of a {}."
        ]

        self.max_len = 40

        self.texts = ['a photo of a ' + class_name for class_name in self.classes]
        self.vocab = build_vocab(self.classes, self.templates, self.max_len)
        self.vocab_size = len(self.vocab)

    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, item):
        image, label = self.cifar10[item]
        return image, label


def load_cifar10_dataset(batch_size: int = 64, transform=None, train: bool = True):
    """
    Load CIFAR-10 dataset with specified batch size and training mode.
    """
    dataset = CIFARTextImageDataset(transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=4), dataset.vocab_size


def train_clip(model: CLIP,
               data_iter: DataLoader,
               optimizer: torch.optim.Optimizer,
               num_epochs: int = 5,
               device: torch.device = torch.device('cpu')):
    for epoch in range(num_epochs):
        total_loss = 0

        model.train()
        for images, label in data_iter:
            images, labels = images.to(device), label.to(device)

            logits = model(images, labels)
            loss = clip_loss(logits)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(data_iter):.4f}')


def inference(model: CLIP, images: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """
    Inference function to get the logits from the CLIP model.
    """
    model.eval()
    with torch.no_grad():
        logits = model(images, tokens)
        return logits


class IntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 64
        self.learning_rate = 2e-4
        self.num_epochs = 5
        self.embed_dim = 512
        self.pretrained_image = True
        self.vocab_size = -1

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.clip_model_path = 'clip_model.pth'

        self.device = dlf.devices()[0]

    def test_clip_train(self):
        data_iter, self.vocab_size = load_cifar10_dataset(self.batch_size, self.transform)
        dataset = CIFARTextImageDataset(transform=self.transform)
        data_iter = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

        # Initialize model and optimizer
        model = CLIP(class_names=dataset.classes,
                     templates=dataset.templates,
                     vocab=dataset.vocab,
                     max_len=dataset.max_len,
                     embed_dim=self.embed_dim,
                     pretrained_image=self.pretrained_image).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        train_clip(model, data_iter, optimizer, num_epochs=self.num_epochs, device=self.device)

        # Save model state
        torch.save(model.state_dict(), self.clip_model_path)

    def test_clip_inference(self):
        if not os.path.exists(self.clip_model_path):
            self.test_clip_train()

        # Load clip model
        model = torch.load(self.clip_model_path).to(self.device)

        # Simple inference demo: predict class for one sample image

        dataset = CIFARTextImageDataset(transform=self.transform)

        model.eval()
        sample_img, _ = dataset[0]
        sample_img = sample_img.unsqueeze(0).to(self.device)
        all_tokens = torch.tensor(
            [[dataset.vocab[name]] for name in dataset.classes], dtype=torch.long).to(self.device)

        sims = inference(model, sample_img.repeat(len(dataset.classes), 1, 1, 1), all_tokens)
        best = sims.argmax().item()
        print(f'Predicted: {dataset.classes[best]}')


if __name__ == "__main__":
    unittest.main(verbosity=True)
