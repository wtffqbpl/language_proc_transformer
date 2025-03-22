#! coding: utf-8

import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import timm
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import utils.dlf as dlf


# Custom dataset
class MultiLabelDataset(Dataset):
    def __init__(self, img_dir, annotations, transform=None):
        """
        :param img_dir:
        :param annotations: A list each element (image_filename, label_vector)
        :param transform: The image preprocessing methods.
        """
        self.img_dir = img_dir
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name, labels = self.annotations[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        labels = torch.tensor(labels, dtype=torch.float32)
        return image, labels


def train(model: nn.Module, data_iter: DataLoader, optimizer, loss_fn, num_epochs, device):

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in data_iter:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)  # output logits, the shape is (batch_size, num_classes)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(data_iter)

        print(f'Epoch: [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')


def inference(model, image_path, transform, device, threshold=0.5):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        logits = model(image)
        probs = torch.sigmoid(logits)
        pred = (probs > threshold).int()
    return pred.cpu().numpy()


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Use convolutions for patch embedding, the conv kernel size and stride are patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x.shape: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H / patch_size, W / patch_size)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class MLP(nn.Module):
    """ The MLP for the Transformer model """
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.0):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    """ The Multi-Head attention module """
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super(Attention, self).__init__()
        assert dim % num_heads == 0, "The dim must be divided by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        # Calculate q, k, v
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Calculate attention weights
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Calculate output
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """ The Encoder module, including MLP and multi-head attention, for the Transformer model """
    def __init__(self, dim, num_heads, mlp_ratio=4., dropout=0., attn_dropout=0.):
        super(TransformerEncoderBlock, self).__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, dropout=attn_dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., dropout=0., attn_dropout=0.):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # class tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional encodings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer Encoder block
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout, attn_dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # classifier
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        B = x.shape[0]

        # Generate patch embeddings
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        # Extend cls tokens and concat to sequence
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer Encoder
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # Use cls token for classification
        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)
        return logits


class IntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        current_path = Path(__file__).resolve().parent
        self.training_img_dir = os.path.join(str(current_path.parent), 'data', 'vit', 'training')
        self.inference_img_dir = os.path.join(str(current_path.parent), 'data', 'vit', 'inference')
        self.model_path = os.path.join(str(current_path), 'vit_multilabel.pth')
        self.device = dlf.devices()[0]

        # Define data preprocessor
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # Normalization
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def test_vit_training(self):
        # Assume annotations has been prepared, each item is (image_name, tag_vector)
        # e.g.
        # annotations = [('img1.jpg', [1, 0, 1, 0]), ('img2.jpg', [0, 1, 0, 1]), ...]
        annotations = []

        # Define model. This can be changed.
        num_classes = 4
        learning_rate = 1e-4
        num_epochs = 10
        batch_size = 32
        num_workers = 4

        dataset = MultiLabelDataset(self.training_img_dir, annotations, transform=self.transform)
        data_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        # Load pre-trained Vision Transformer using timm
        model_name = 'vit_base_patch16_224'  # You can change to other models
        model = timm.create_model(model_name, pretrained=True)

        # Change model features. Note: The ViT model is a single classification in default.
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)  # The output dim is num_classes

        # Define loss function and optimizer
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train(model, data_iter, optimizer, loss_fn, num_epochs, self.device)

        torch.save(model.state_dict(), self.model_path)

        self.assertTrue(True)

    def test_vit_inference(self):
        if not os.path.exists(self.model_path):
            self.test_vit_training()

        test_image = os.path.join(self.inference_img_dir, 'test.jpg')

        model = torch.load(self.model_path).to(self.device)
        preds = inference(model, test_image, self.transform, self.device)
        print('Predicted labels: ', preds)

    def test_vit_scratch(self):
        x = torch.randn(2, 3, 224, 224)

        model = VisionTransformer(
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=10,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            dropout=0.1,
            attn_dropout=0.1,
        )

        logits = model(x)
        print("Logits shape: ", logits.shape)

        # Depress warnings
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main(verbosity=True)
