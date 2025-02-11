#! coding: utf-8


import os
import unittest

import matplotlib.pyplot as plt
import torch
import pandas as pd
import torchvision
import sys
from pathlib import Path
from typing import Any

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
        xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0], height=bbox[3] - bbox[1],
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
        batch_size=batch_size, shuffle=True, num_workers=4)
    val_iter = torch.utils.data.DataLoader(
        BananasDataset(is_train=False),
        batch_size=batch_size, shuffle=False, num_workers=4)

    return train_iter, val_iter


def multibox_prior(data: torch.Tensor, sizes: list[float], ratios: list[float]):
    """
    The multibox_prior function generates prior (anchor) boxes through the following steps:
        1. Determine the normalized center coordinates for each pixel based on
           the dimensions of the input feature map.
        2. Compute the widths and heights of the prior boxes using the provided
           scales and aspect ratios.
        3. Construct the offsets for multiple prior boxes at each pixel and add
           them to the pixel centers to obtain the complete prior box coordinates.
        4. Finally, add a batch dimension before returning the result.
    :param data: (batch_size, channels, width, height).
    :param sizes: list of the scale (width / height).
    :param ratios: list of the aspect ratio.
    :return: anchor_boxes.shape = (batch_size, num_anchors, 4)
    """
    # Generate a list of anchor boxes for each pixel.
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height
    steps_w = 1.0 / in_width

    # Generate all center pts
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:]))) \
        * in_height / in_width
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))

    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
        in_height * in_width, 1) / 2

    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                           dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)


def box_iou(boxes1, boxes2):
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    # The shapes of the boxes1, boxes2, areas1 and areas2.
    # (the number of boxes1, 4)
    # (the number of boxes2, 4)
    # (the number of boxes1, )
    # (the number of boxes2, )
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)

    # The shapes of the inter_upperlefts, inter_lowerrights, inters
    # (the number of boxes1, the number of boxes2, 2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # The shape of the inter_areasandunion_areas is (the number of boxes1, the number of boxes2)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas


def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    # Assign closest ground-truth bounding boxes to anchor boxes.
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # Element x_ij in the i-th row and j-th column is the IoU of the anchor
    # box i and the ground-truth bounding box j.
    jaccard = box_iou(anchors, ground_truth)

    # initialize the tensor to hold the assigned ground-truth bounding box for
    # each anchor.
    anchors_bbox_map = torch.full((num_anchors, ), -1, dtype=torch.long, device=device)

    # Assign ground-truth bounding boxes according to the threshold.
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors, ), -1)
    row_discard = torch.full((num_gt_boxes, ), -1)

    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)  # Find the largest IoU
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard

    return anchors_bbox_map


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

        # Add plt.show() at the end of the function, because I don't know how
        # to execute plt.show() before this function exiting automatically.
        plt.show()

    def test_anchor_boxes(self):
        img = ImageUtils.imread(self.img_path)
        h, w = img.shape[:2]
        print(h, w)
        self.assertEqual((728, 561), (w, h))

        x = torch.rand(size=(1, 3, h, w))
        y = multibox_prior(x, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
        print('y.shape: ', y.shape)
        self.assertEqual(y.shape, torch.Size([1, 2042040, 4]))

        boxes = y.reshape(h, w, 5, 4)
        print(boxes[250, 250, 0, :])

        bbox_scale = torch.tensor((w, h, w, h))
        fig = ImageUtils.imshow(img)
        ImageUtils.show_boxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
                              ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1',
                               's=0.75, r=2', 's=0.75, r=0.5'])

        # plt.show() manually
        plt.show()


if __name__ == "__main__":
    unittest.main(verbosity=True)
