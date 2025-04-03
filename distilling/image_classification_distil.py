#! coding: utf-8

import unittest
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import utils.dlf as dlf


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class SimpleConv5(nn.Module):
    def __init__(self, nclass=2, inplanes=32, kernel=3):
        super(SimpleConv5, self).__init__()
        self.inplanes = inplanes
        self.kernel = kernel
        self.pad = self.kernel // 2

        # Convolution module
        self.conv_net = nn.Sequential(
            ConvBlock(3, self.inplanes, kernel_size=self.kernel, stride=2, padding=self.pad),
            ConvBlock(self.inplanes, self.inplanes * 2, kernel_size=self.kernel, stride=2, padding=self.pad),
            ConvBlock(self.inplanes * 2, self.inplanes * 4, kernel_size=self.kernel, stride=2, padding=self.pad),
            ConvBlock(self.inplanes * 4, self.inplanes * 8, kernel_size=self.kernel, stride=2, padding=self.pad),
            ConvBlock(self.inplanes * 8, self.inplanes * 16, kernel_size=self.kernel, stride=2, padding=self.pad),
        )

        # MLP
        self.classifier = nn.Sequential(
            nn.AvgPool2d(7),
            nn.Flatten(),
            nn.Linear(self.inplanes * 16, nclass)
        )

    def forward(self, x):
        out = self.conv_net(x)
        out = self.classifier(out)
        return out


class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.vgg16_bn = torchvision.models.vgg16_bn(pretrained=True)

        # Freeze all layers except the last one
        for param in self.vgg16_bn.parameters():
            param.requires_grad = False

        # Replace the last layer with a new one
        self.vgg16_bn.classifier[6] = nn.Sequential(torch.nn.Linear(4096, 20))
        for param in self.vgg16_bn.classifier[6].parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.vgg16_bn(x)


def load_datasets(batch_size, data_transforms):
    train_dataset = torchvision.datasets.CIFAR10(
        root='../data', train=True, download=True, transform=data_transforms)
    test_dataset = torchvision.datasets.CIFAR10(
        root='../data', train=False, download=True, transform=data_transforms)

    return (
        torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
        torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
    )


class IntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.device = dlf.devices()[0]

    def test_distil(self):
        image_size = 256  # Image scaling size
        crop_size = 224  # Image cropping size

        batch_size = 128
        num_epochs_teacher = 10
        num_epochs_student = 20
        learning_rate = 0.001
        temperature = 4.0

        # Data preprocessing and augmentation methods
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(crop_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
            'val': transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
        }

        train_iter, test_iter = load_datasets(batch_size=batch_size, data_transforms=data_transforms)

        print(next(train_iter))

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main(verbosity=True)
