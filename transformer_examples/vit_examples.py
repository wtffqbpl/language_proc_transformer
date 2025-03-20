#! coding: utf-8

import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import timm
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import utils.dlf as dlf


# Custom dataset
class MultiLabelDataset(Dataset):
    def __init__(self, img_dir, annotations, transform=None):
        """
        :param img_dir:
        :param annotations: A list each element (image_filename, label_vector)
        :param transform: The image preprocessing methods.
        """
        self.img_dir = img_dir
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name, labels = self.annotations[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        labels = torch.tensor(labels, dtype=torch.float32)
        return image, labels


def train(model: nn.Module, data_iter: DataLoader, optimizer, loss_fn, num_epochs, device):

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in data_iter:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)  # output logits, the shape is (batch_size, num_classes)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(data_iter)

        print(f'Epoch: [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')


def inference(model, image_path, transform, device, threshold=0.5):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        logits = model(image)
        probs = torch.sigmoid(logits)
        pred = (probs > threshold).int()
    return pred.cpu().numpy()


class IntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        current_path = Path(__file__).resolve().parent
        self.training_img_dir = os.path.join(str(current_path.parent), 'data', 'vit', 'training')
        self.inference_img_dir = os.path.join(str(current_path.parent), 'data', 'vit', 'inference')
        self.model_path = os.path.join(str(current_path), 'vit_multilabel.pth')
        self.device = dlf.devices()[0]

        # Define data preprocessor
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # Normalization
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def test_vit_training(self):
        # Assume annotations has been prepared, each item is (image_name, tag_vector)
        # e.g.
        # annotations = [('img1.jpg', [1, 0, 1, 0]), ('img2.jpg', [0, 1, 0, 1]), ...]
        annotations = []

        # Define model. This can be changed.
        num_classes = 4
        learning_rate = 1e-4
        num_epochs = 10
        batch_size = 32
        num_workers = 4

        dataset = MultiLabelDataset(self.training_img_dir, annotations, transform=self.transform)
        data_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        # Load pre-trained Vision Transformer using timm
        model_name = 'vit_base_patch16_224'  # You can change to other models
        model = timm.create_model(model_name, pretrained=True)

        # Change model features. Note: The ViT model is a single classification in default.
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)  # The output dim is num_classes

        # Define loss function and optimizer
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train(model, data_iter, optimizer, loss_fn, num_epochs, self.device)

        torch.save(model.state_dict(), self.model_path)

        self.assertTrue(True)

    def test_vit_inference(self):
        if not os.path.exists(self.model_path):
            self.test_vit_training()

        test_image = os.path.join(self.inference_img_dir, 'test.jpg')

        model = torch.load(self.model_path).to(self.device)
        preds = inference(model, test_image, self.transform, self.device)
        print('Predicted labels: ', preds)


if __name__ == '__main__':
    unittest.main(verbosity=True)
