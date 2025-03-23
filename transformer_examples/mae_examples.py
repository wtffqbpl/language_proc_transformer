#! coding: utf-8

import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
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
        pass


class IntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
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


if __name__ == '__main__':
    unittest.main(verbosity=True)
