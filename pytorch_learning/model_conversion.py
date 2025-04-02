#! coding: utf-8

import unittest
import torch
from torchvision.models import resnet18


class IntegrationTest(unittest.TestCase):
    def test_pt_to_onnx(self):
        model = resnet18(pretrained=True)
        model.eval()

        dummy_input = torch.randn((1, 3, 224, 224))
        torch.onnx.export(model, dummy_input, "resnet18.onnx")

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main(verbosity=True)
