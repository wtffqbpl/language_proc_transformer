#! coding: utf-8


import os
import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.ops import roi_pool
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import utils.dlf as dlf


class FastRCNN(nn.Module):
    def __init__(self, num_classes):
        super(FastRCNN, self).__init__()

        # Use reset18 pretrained model as features extraction model,
        # and remove the avgpool and fc layers.
        backbone = torchvision.models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        # ROI Pooling params: resnet18 uses 32-times down-sampling, so
        # the spatial_scale = 1 / 32
        self.roi_pool = roi_pool

        # Define two dense layers as the fc layers of the Fast R-CNN.
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5))

        # Classification branch: Calculate ROI for each class.
        self.classifier = nn.Linear(4096, num_classes)
        # Bounding box regression: Calculate ROI for the four positions.
        self.bbox_regressor = nn.Linear(4096, num_classes * 4)

    def forward(self, images, rois):
        """

        :param images: Tensor, shape(N, 3, H, W)
        :param rois: Tensor, shape (num_rois, 5) [batch_idx, x1, y1, x2, y2] based on the original images.
        :return:
            cls_scores: Tensor, (num_rois, num_classes)
            bbox_preds: Tensor, (num_rois, num_classes * 4)
        """
        # Extract the features using backbone. The output shape: (N, 512, H/32, W/32)
        features = self.backbone(images)

        # ROI Pooling: Mapping each ROI region to the 7x7 feature images.
        pooled_features = self.roi_pool(features, rois, output_size=(7, 7), spatial_scale=1/32)
        # Flatten for the fc layer
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        fc_features = self.fc(pooled_features)
        cls_scores = self.classifier(fc_features)
        bbox_preds = self.bbox_regressor(fc_features)
        return cls_scores, bbox_preds


# Dummy datasets
class DummyDataset(Dataset):
    def __init__(self, num_samples, num_rois_per_image, num_classes, image_size=(3, 224, 224)):
        super(DummyDataset, self).__init__()
        self.num_samples = num_samples
        self.num_rois_per_image = num_rois_per_image
        self.num_classes = num_classes
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a random image
        image = torch.rand(self.image_size)

        # Generate a random ROI. The ROI format: [batch_idx, x1, y1, x2, y2], for now, the batch_idx is 0.
        rois = []
        C, H, W = self.image_size

        for _ in range(self.num_rois_per_image):
            x1 = torch.randint(0, W // 2, (1,)).item()
            y1 = torch.randint(0, H // 2, (1,)).item()
            x2 = torch.randint(W // 2, (1,)).item()
            y2 = torch.randint(H // 2, (1,)).item()
            rois.append((0, x1, y1, x2, y2))
        rois = torch.tensor(rois, dtype=torch.float32)

        # Generate the classes for each ROI. The 0 implies the background.
        labels = torch.randint(0, self.num_classes, (self.num_rois_per_image,))
        # Generate the random target bbox.
        bbox_targets = torch.rand(self.num_rois_per_image, self.num_classes * 4)

        return image, rois, labels, bbox_targets


def collate_fn(batch):
    images = []
    all_rois = []
    label_list = []
    bbox_targets_list = []
    for i, (img, rois, labels, bbox_targets) in enumerate(batch):
        images.append(img)

        # Change batch_idx to the current batch idx
        rois[:, 0] = i
        all_rois.append(rois)
        label_list.append(labels)
        bbox_targets_list.append(bbox_targets)
    images = torch.stack(images, dim=0)
    all_rois = torch.cat(all_rois, dim=0)
    labels = torch.cat(label_list, dim=0)
    bbox_targets = torch.cat(bbox_targets_list, dim=0)

    return images, all_rois, labels, bbox_targets


def train(model, dataloader, optimizer, num_epochs, device):
    model.to(device=device)
    model.train()
    criterion_cls = nn.CrossEntropyLoss()
    criterion_bbox = nn.SmoothL1Loss()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, rois, labels, bbox_targets) in enumerate(dataloader):
            images = images.to(device=device)
            rois = rois.to(device=device)
            labels = labels.to(device)
            bbox_targets = bbox_targets.to(device=device)

            optimizer.zero_grad()
            cls_scores, bbox_preds = model(images, rois)

            # Classification loss
            loss_cls = criterion_cls(cls_scores, labels)
            # Regression loss (We only calculate the bbox loss for the positive samples).
            loss_bbox = criterion_bbox(bbox_preds, bbox_targets)
            loss = loss_cls + loss_bbox

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, ',
              f'Loss: {epoch_loss:.4f}')


# Inference function
def inference(model, image, rois, device):
    """
    :param image: Tensor, shape (3, H, W)
    :param rois:  ROI region, Tensor, shape (num_rois, 5), [0, x1, y1, x2, y2]
    :param device: device
    :return:
        probs: The probability for each ROI classification (num_rois, num_classes)
        bbox_preds: The regression for each ROI bbox. (num_rois, num_classes * 4)
    """
    model = model.to(device=device)
    model.eval()

    with torch.no_grad():
        image, rois = image.to(device=device), rois.to(device=device)
        # Add batch dimension since the batch_size=1 for the inference stage.
        cls_scores, bbox_preds = model(image.unsqueeze(0), rois)
        probs = nn.functional.softmax(cls_scores, dim=1)
    return probs, bbox_preds


class IntegrationTest(unittest.TestCase):
    def test_fast_rcnn_training(self):
        # Hyperparameters
        num_classes, learning_rate, num_epochs = 21, 0.001, 5

        model = FastRCNN(num_classes=num_classes)
        device = dlf.devices()[0]

        # Load datasets
        dataset = DummyDataset(num_samples=100, num_rois_per_image=10, num_classes=num_classes)
        data_iter = DataLoader(dataset,
                               batch_size=4,
                               shuffle=True,
                               collate_fn=collate_fn,
                               num_workers=4)

        # optimizer
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        # Training 5 epochs
        train(model, data_iter, optimizer, num_epochs=num_epochs, device=device)

        model_path = os.path.join(str(Path(__file__).resolve().parent), 'fast_rcnn.pth')

        # save model
        torch.save(model.state_dict(), model_path)

    def test_inference(self):

        model_path = os.path.join(str(Path(__file__).resolve().parent), 'fast_rcnn.pth')

        if not os.path.exists(model_path):
            self.test_fast_rcnn_training()

        device = dlf.devices()[0]

        # Load model before inference.
        model = torch.load(model_path)

        # Generate a random image
        h, w = 224, 224
        image = torch.rand(3, h, w)

        # Generate 5 random ROIs
        rois_ = []
        for _ in range(5):
            x1 = torch.randint(0, w // 2, (1,)).item()
            y1 = torch.randint(0, h // 2, (1,)).item()
            x2 = torch.randint(w // 2, w, (1,)).item()
            y2 = torch.randint(h // 2, h, (1,)).item()
            rois_.append([0, x1, y1, x2, y2])
        rois = torch.tensor(rois_, dtype=torch.float32)

        probs, bbox_preds = inference(model, image, rois, device)

        print("The inference result:")
        print("The probability of the classification: ", probs)
        print("The probability of the bbox: ", bbox_preds)

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main(verbosity=True)
