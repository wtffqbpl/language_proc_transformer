#! coding: utf-8
import os.path
import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import utils.dlf as dlf


# =============================================================
#           Utils
# =============================================================
def patchify(images, patch_size):
    """
    Split images to patches
    :param images: [B, C, H, W]
    :param patch_size: int
    :return: patches: [B, num_patches, patch_dim]
    """
    B, C, H, W = images.shape
    assert H % patch_size == 0 and W % patch_size == 0, "The image width and height must be divided by patch_size"
    grid_h, grid_w = H // patch_size, W // patch_size
    patches = images.reshape(B, C, grid_h, patch_size, grid_w, patch_size)
    patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(B, grid_h * grid_w, C * patch_size * patch_size)
    return patches


def unpatchify(patches, patch_size, image_size, channels=3):
    """
    Transform patches to images
    :param patches: [B, num_patches, patch_dim]
    :param patch_size:
    :param image_size:
    :param channels:
    :return: images: [B, C, H, W]
    """
    B, N, patch_dim = patches.shape
    grid_size = image_size // patch_size
    images = patches.reshape(B, grid_size, grid_size, channels, patch_size, patch_size)
    images = images.permute(0, 3, 1, 4, 2, 5).reshape(B, channels, image_size, image_size)
    return images


def random_masking(x: torch.Tensor, mask_ratio: float):
    """
    Randomize mask patch sequence
    :param x: [B, N, D]
    :param mask_ratio:
    :return:
        x_masked: un-masked patches
        mask: shape = [B, N], bool, (True --- indicates this patch is masked)
    """
    B, N, D = x.shape
    len_keep = int(N * (1 - mask_ratio))
    noise = torch.rand(B, N, device=x.device)  # Generate random ratio for each patch
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]

    # Collect un-masked patches according indices
    batch_idx = torch.arange(B, device=x.device)[:, None]
    x_masked = x[batch_idx, ids_keep]

    # Construct mask: True indicates this patch is masked, False indicates resolved.
    mask = torch.ones(B, N, device=x.device, dtype=torch.bool)
    mask[batch_idx, ids_keep] = False
    return x_masked, mask, ids_keep


class MAE(nn.Module):
    def __init__(self, image_size=32, patch_size=4, in_channels=3, embedding_dim=128,
                 decoder_embedding_dim=128, mask_ratio=0.5,
                 num_encoder_layers=3, num_decoder_layers=3, num_heads=4):
        super(MAE, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.mask_ratio = mask_ratio
        patch_dim = in_channels * patch_size * patch_size

        # 1. Patch embedding: Using linear mapping
        self.patch_embed = nn.Linear(patch_dim, embedding_dim)

        # 2. Add positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embedding_dim))

        # 3. Encoder: Using TransformerEncoderLayer.
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 4. Decoder: Adjust dim to decoder_embed_dim, and add mask token
        self.decoder_embed = nn.Linear(embedding_dim, decoder_embedding_dim, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embedding_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_embedding_dim))
        decoder_layer = nn.TransformerEncoderLayer(d_model=decoder_embedding_dim, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_decoder_layers)

        # 5. Prediction layer
        self.decoder_pred = nn.Linear(decoder_embedding_dim, patch_dim, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.xavier_uniform_(self.decoder_pred.weight)

    def forward(self, images):
        """ images: [B, C, H, W] """
        # 1. Transform images to patch sequence
        patches = patchify(images, self.patch_size)  # [B, N, patch_dim]

        # 2. Linear mapping
        x = self.patch_embed(patches)  # [B, N, embed_dim]
        x = x + self.pos_embed  # Add positional encodings

        # 3. Random mask some patches
        x_masked, mask, ids_keep = random_masking(x, self.mask_ratio)

        # 4. Encoder: Encoding un-masked patches
        encoding_output = self.encoder(x_masked)  # [B, N_keep, embed_dim]

        # 5. decoder: Mapping encoder results to decoder
        dec_input = self.decoder_embed(encoding_output)  # [B, N_keep, decoder_embed_dim]

        B = images.shape[0]

        # Construct decoder inputs, copy mask token firstly.
        # Construct a mask token sequence
        dec_tokens = self.mask_token.expand(B, self.num_patches, -1).clone()

        # Put the encoder output to the corresponding positions (un-masked positions)
        batch_idx = torch.arange(B, device=images.device)[:, None]
        dec_tokens[batch_idx, ids_keep] = dec_input

        # Add positional encodings for the decoder
        dec_tokens = dec_tokens + self.decoder_pos_embed

        # 6. decoder
        dec_output = self.decoder(dec_tokens)  # [B, N, decoder_embed_dim]

        # 7. Prediction for each patch
        pred = self.decoder_pred(dec_output)  # [B, N, patch_dim]
        return pred, mask

    def loss(self, pred, images):
        """
        Compute reconstruction loss
        :param pred: [B, C, H, W]
        :param images: [B, N, patch_dim]
        """
        patches = patchify(images, self.patch_size)

        # mse loss for the masked positions
        loss = ((pred - patches) ** 2).mean(dim=-1)  # [B, N]

        return loss.mean()


def load_cifar10_data(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(
        root='../data', train=False, download=True, transform=transform)
    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False))


def train(model, data_iter, optimizer, num_epochs, device):
    model.train()

    total_loss = 0
    for epoch in range(num_epochs):
        for images, _ in data_iter:
            images = images.to(device)
            optimizer.zero_grad()
            pred, _ = model(images)
            loss = model.loss(pred, images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        print(f'[{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}')


def inference(model, data_iter, device, patch_size, image_size):
    model.eval()

    images, _ = next(iter(data_iter))
    images = images.to(device)

    with torch.no_grad():
        pred, mask = model(images)

    # Re-construct images: using predicted patches to construct original images
    images_rec = unpatchify(pred, patch_size, image_size, channels=images.shape[1])
    return images.cpu(), images_rec.cpu(), mask.cpu()


def plot_results(original, reconstructed, num_images=8):
    """
    使用 matplotlib 显示原始图像与重构图像对比
    """
    # 将 tensor 转为 numpy，并调整通道顺序 [B, C, H, W] -> [B, H, W, C]
    original = original.permute(0, 2, 3, 1).numpy()
    reconstructed = reconstructed.permute(0, 2, 3, 1).numpy()

    num_images = min(num_images, original.shape[0])
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))

    for i in range(num_images):
        axes[0, i].imshow(np.clip(original[i], 0, 1))
        axes[0, i].axis('off')
        axes[0, i].set_title("Original")

        axes[1, i].imshow(np.clip(reconstructed[i], 0, 1))
        axes[1, i].axis('off')
        axes[1, i].set_title("Reconstructed")
    plt.tight_layout()
    plt.show()


class IntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mae_model_path = os.path.join(str(Path(__file__).resolve().parent), 'mae_model.pth')
        self.device = dlf.devices()[0]
        self.batch_size = 128
        self.num_epochs = 10
        self.learning_rate = 1e-3

        # Image and patch parameters
        self.image_size = 32  # The image size of the CIFAR10 is 32x32
        self.patch_size = 4  # The patch size is 4x4, so there are (32/4)^2 = 64 patches for each image
        self.embedding_dim = 12  # The embedding size for each patch
        self.mask_ratio = 0.5  # The random mask ratio

        # The Transformer parameters
        self.num_encoder_layers = 3
        self.num_decoder_layers = 3
        self.num_heads = 4
        self.decoder_embedding_dim = 128

    def test_mae_training(self):
        model = MAE(image_size=self.image_size, patch_size=self.patch_size,
                    in_channels=3, embedding_dim=self.embedding_dim,
                    decoder_embedding_dim=self.decoder_embedding_dim,
                    mask_ratio=self.mask_ratio,
                    num_encoder_layers=self.num_encoder_layers,
                    num_decoder_layers=self.num_decoder_layers,
                    num_heads=self.num_heads).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        train_iter, _ = load_cifar10_data(self.batch_size)

        train(model, train_iter, optimizer, self.num_epochs, self.device)

        torch.save(model, self.mae_model_path)

    def test_mae_inference(self):
        if not os.path.exists(self.mae_model_path):
            self.test_mae_training()

        _, test_iter = load_cifar10_data(self.batch_size)

        model = torch.load(self.mae_model_path, weights_only=False).to(self.device)

        images, images_rec, mask = inference(model, test_iter, self.device, self.patch_size, self.image_size)
        plot_results(images, images_rec)


if __name__ == '__main__':
    unittest.main(verbosity=True)
