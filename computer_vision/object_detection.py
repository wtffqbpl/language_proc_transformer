#! coding: utf-8


import os
import unittest

import matplotlib.pyplot as plt
import torch
import torchvision
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.plot import ImageUtils


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


if __name__ == "__main__":
    unittest.main(verbosity=True)
