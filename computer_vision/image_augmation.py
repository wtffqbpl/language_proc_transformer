#! coding: utf-8


import unittest
import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.models as models
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.plot import ImageUtils as Image
from utils.accumulator import Accumulator
from utils.timer import Timer
import utils.dlf as dlf


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Residual, self).__init__()

        self.main_model = nn.Sequential(
            nn.LazyConv2d(out_channels=out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels), nn.ReLU(),
            nn.LazyConv2d(out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels))

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.LazyConv2d(out_channels=out_channels, kernel_size=1, stride=stride))

    def forward(self, x):
        y = self.main_model(x)
        y += self.shortcut(x)
        return y


class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()

        block1 = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        block2 = nn.Sequential(*self.resnet_block(64, 64, 2, first_block=True))
        block3 = nn.Sequential(*self.resnet_block(64, 128, 2))
        block4 = nn.Sequential(*self.resnet_block(128, 256, 2))
        block5 = nn.Sequential(*self.resnet_block(256, 512, 2))

        self.net = nn.Sequential(
            block1, block2, block3, block4, block5,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LazyLinear(num_classes))

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def resnet_block(in_channels, num_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(in_channels, num_channels, stride=2))
            else:
                blk.append(Residual(num_channels, num_channels))

        return blk


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.net = models.resnet18(pretrained=True)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def forward(self, x):
        return self.net(x)


def load_cifar10_data(batch_size, train_augs, test_augs) -> tuple[data.DataLoader, data.DataLoader]:
    def data_load(augs, is_train=False):
        dataset = torchvision.datasets.CIFAR10(
            root='../data', train=is_train, transform=augs, download=True)
        dataloader = data.DataLoader(
            dataset, batch_size=batch_size, shuffle=is_train, num_workers=4)
        return dataloader

    return data_load(train_augs, True), data_load(test_augs, False)


def init_model(model: nn.Module):
    # initialize weight
    if type(model) in [torch.nn.Conv2d, torch.nn.LazyConv2d, torch.nn.Linear, torch.nn.LazyLinear]:
        torch.nn.init.xavier_uniform_(model.weight)

    # initialize bias
    if hasattr(model, 'bias') and model.bias is not None:
        torch.nn.init.zeros_(model.bias)


def accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    if y_hat.ndim > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)

    cmp = y_hat.type(y.dtype) == y
    return float(cmp.to(y.dtype).sum().item())


def evaluate(model: nn.Module, data_iter: data.DataLoader, device=None) -> float:
    assert device is not None

    model.eval()

    metric = Accumulator(2)

    with torch.no_grad():
        for x, y in data_iter:
            if x.device != device or y.device != device:
                x, y = x.to(device), y.to(device)
            metric.add(accuracy(model(x), y), torch.numel(y))

    return metric[0] / metric[1]


# Multi-devices training
def training_batch(model: nn.Module, x, y, loss_fn, optimizer, devices=None):
    assert devices is not None

    if isinstance(x, list):
        # Needed for BERT training
        x = [x_.to(device=devices[0]) for x_ in x]
    else:
        x = x.to(device=devices[0])
    y = y.to(device=devices[0])

    model.train()

    optimizer.zero_grad()
    y_hat = model(x)
    loss = loss_fn(y_hat, y)
    loss.sum().backward()
    optimizer.step()

    training_loss_sum = loss.sum()
    training_acc_sum = accuracy(y_hat, y)
    return training_loss_sum, training_acc_sum


def train(model: nn.Module,
          data_iter: data.DataLoader,
          test_iter: data.DataLoader,
          loss_fn, optimizer,
          num_epochs: int,
          devices=None):

    if devices is None:
        devices = dlf.devices()

    timer, num_batches = Timer(), len(data_iter)
    model = nn.DataParallel(model, device_ids=devices).to(devices[0])

    metric = Accumulator(4)
    for epoch in range(num_epochs):
        for i, (features, labels) in enumerate(data_iter):

            timer.start()
            loss, acc = training_batch(model, features, labels, loss_fn, optimizer, devices)
            metric.add(loss, acc, labels.shape[0], labels.numel())
            timer.stop()

            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                print(f'epoch {epoch+1}, ',
                      f'iter [{i+1}/{num_batches}], ',
                      f'loss {(metric[0]/metric[2]):.3f}, ',
                      f'train acc {(metric[1]/metric[3]):.3f}')
        print(f'epoch {epoch+1}, ',
              f'loss {(metric[0]/metric[2]):.3f}, ',
              f'train acc {(metric[1]/metric[3]):.3f}, ',
              f'test acc {(evaluate(model, test_iter, devices[0])):.3f}')

    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(devices)}')


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
        aug = torchvision.transforms.ToTensor()
        tmp = aug(all_images[0][0])
        print(f'Shape: {tmp.shape}')

        self.assertTrue(True)

    def test_cifar10_training(self):
        # hyperparameters
        batch_size, learning_rate, num_epochs = 128, 0.001, 10

        devices = dlf.devices()
        model = ResNet()
        model.to(device=devices[0])

        train_augs = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=96),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor()
        ])

        test_augs = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=96),
            torchvision.transforms.ToTensor()
        ])

        train_iter, test_iter = load_cifar10_data(batch_size, train_augs, test_augs)

        # dummy initialization
        _ = model(torch.randn(size=(1, 3, 96, 96)).to(devices[0]))
        model.apply(init_model)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        train(model, train_iter, test_iter, loss_fn, optimizer, num_epochs, devices)

        self.assertTrue(True)

    def test_cifar10_with_resnet18(self):
        # hyperparameters
        batch_size, learning_rate, num_epochs, num_classes = 128, 0.001, 10, 10

        devices = dlf.devices()
        model = ResNet18(num_classes=num_classes)
        model.to(device=devices[0])

        train_augs = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=96),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor()])
        test_augs = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=96),
            torchvision.transforms.ToTensor()])

        train_iter, test_iter = load_cifar10_data(batch_size, train_augs, test_augs)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        train(model, train_iter, test_iter, loss_fn, optimizer, num_epochs, devices)


if __name__ == "__main__":
    unittest.main(verbosity=True)
