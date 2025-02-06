#! coding: utf-8


import unittest
import os
import torch
import torchvision
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.plot import ImageUtils as Image


class IntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.img_path = os.path.join(str(Path(__file__).resolve().parent), 'cat.png')
        pass

    def test_show_image(self):
        img = Image.open(self.img_path)
        Image.imshow(img)

        self.assertTrue(True)

    def test_augmation_horizontal_flip(self):
        img = Image.open(self.img_path)
        Image.apply(img, torchvision.transforms.RandomHorizontalFlip())
        
        self.assertTrue(True)

    def test_augmation_vertical_flip(self):
        img = Image.open(self.img_path)
        Image.apply(img, torchvision.transforms.RandomVerticalFlip())

        self.assertTrue(True)

    def test_augmation_resized_crop(self):
        Image.apply(
            Image.open(self.img_path),
            torchvision.transforms.RandomResizedCrop((200, 200), scale=(0.1, 1), ratio=(0.5, 2)))

        self.assertTrue(True)

    def test_augmation_color_jitter(self):
        img_aug_list = [
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0),
            torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.5),
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        ]

        for img_aug in img_aug_list:
            Image.apply(Image.open(self.img_path), img_aug)

        self.assertTrue(True)

    def test_multiple_augmation(self):
        augs = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            torchvision.transforms.RandomResizedCrop((200, 200), scale=(0.1, 1), ratio=(0.5, 2)),
        ])

        Image.apply(Image.open(self.img_path), augs)

        self.assertTrue(True)

    def test_cifar10_dataset(self):
        all_images = torchvision.datasets.CIFAR10(root='../data', train=True, download=True)
        Image.show_images([all_images[i][0] for i in range(32)],
                          num_rows=4,
                          num_cols=8,
                          scale=0.8)

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main(verbosity=True)
