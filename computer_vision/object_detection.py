#! coding: utf-8


import os
import unittest

import matplotlib.pyplot as plt
import torch
import pandas as pd
import torchvision
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.plot import ImageUtils
import utils.dlf as dlf


# Bounding box
def box_corner_to_center(boxes):
    # (left_upper_x, left_upper_y, right_lower_x, right_lower_y) -> (middle_x, middle_y, width, height)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack((cx, cy, w, h), dim=-1)


def box_center_to_corner(boxes):
    # (center_x, center_y, width, height) -> (left_upper_x, left_lower_y, right_lower_x, right_lower_y)
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    return torch.stack((x1, y1, x2, y2), dim=-1)


def bbox_to_rect(bbox, color):
    # bbox: The abbreviation for bounding box
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)


dlf.DATA_HUB['banana-detection'] = (
    dlf.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')


def read_data_bananas(is_train=True):
    # Read images and labels from banana dataset
    data_dir = dlf.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir,
                             'bananas_train' if is_train else 'bananas_val',
                             'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')

    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(
            torchvision.io.read_image(
                os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val',
                             'images', f'{img_name}')))
        # Here `target` contains (class, upper-left x, upper-left y,
        # lower-right x, lower-right y), where all the images have the same
        # banana class (index 0)
        targets.append(list(target))

    return images, torch.tensor(targets).unsqueeze(1) / 256


class BananasDataset(torch.utils.data.Dataset):
    # Define a custom bananas dataset. We should override __getitem__ and __len__ methods.
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read', str(len(self.features)),
              (f' training examples' if is_train else f' validation examples'))

    def __getitem__(self, idx):
        return self.features[idx].float(), self.labels[idx]

    def __len__(self):
        return len(self.features)


def load_data_bananas(batch_size):
    # 对于 物体检测 来说，batch_size 与实际检测得到的物体数不同，有可能一张图片中
    # 会检测出多个物体，一般来说设置一个上限，即最多检测多少个物体，如果超出该上限，
    # 则多出来的物体就直接扔掉， 如果少了，则填一些0在里面，这样就可以让一个batch
    # 中的数据构成一个规整的tensor.
    # 这样的话 [batch_size, num_objects, feature, x1, y1, x2, y2]
    # Load banana dataset
    train_iter = torch.utils.data.DataLoader(
        BananasDataset(is_train=True),
        batch_size=batch_size,shuffle=True, num_workers=4)
    val_iter = torch.utils.data.DataLoader(
        BananasDataset(is_train=False),
        batch_size=batch_size,shuffle=False, num_workers=4)

    return train_iter, val_iter


class IntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.img_path = os.path.join(str(Path(__file__).resolve().parent), 'catdog.png')
        pass

    def test_simple(self):
        img = ImageUtils.open(self.img_path)
        fig = ImageUtils.imshow(img)

        dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]

        boxes = torch.tensor((dog_bbox, cat_bbox))

        self.assertTrue(torch.all(box_center_to_corner(box_corner_to_center(boxes)) == boxes).item())

        fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
        fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))

        # Execute the plt.show() manually after the function.
        plt.show()

    def test_dataset_shapes(self):
        batch_size, edge_size = 32, 256
        train_iter, val_iter = load_data_bananas(batch_size)
        batch = next(iter(train_iter))
        print(batch[0].shape, batch[1].shape)

        self.assertEqual(torch.Size([batch_size, 3, edge_size, edge_size]), batch[0].shape)
        self.assertEqual(torch.Size([batch_size, 1, 5]), batch[1].shape)

        imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255
        axes = ImageUtils.show_images(imgs, 2, 5, scale=2)
        for ax, label in zip(axes, batch[1][0:10]):
            ImageUtils.show_boxes(ax, [label[0][1:5] * edge_size], colors=['w'])
        plt.show()


if __name__ == "__main__":
    unittest.main(verbosity=True)
