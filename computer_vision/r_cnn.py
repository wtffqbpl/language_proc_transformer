#! coding: utf-8

import os

import matplotlib.pyplot as plt
import torch
import torchvision
import cv2
import numpy as np
from torchvision import models, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path


model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()


def preprocess_image(image):
    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return trans(image).unsqueeze(0)


def selective_search(image):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    return ss.process()


# Load image
image_path = os.path.join(str(Path(__file__).resolve().parent.parent),
                          'data', 'banana-detection', 'bananas_val', 'images')
# TODO: complete the image path, for now, the image_path is a directory.
print(os.path.join(image_path, '0.png'))


image = cv2.imread(os.path.join(image_path, '0.png'))
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
rects = selective_search(image)


# Select the 100 regions
N = 100
regions = []
for (x, y, w, h) in rects[:N]:
    roi = image_rgb[y:y+h, x:x+w]  # Select the RoI
    if roi.shape[0] > 0 and roi.shape[1] > 0:
        regions.append((x, y, w, h, roi))


# Extract the features with CNN
features, labels = [], []
for idx, (x, y, w, h, roi) in enumerate(regions):
    img_tensor = preprocess_image(roi)
    print(type(img_tensor))
    roi_resized = Variable(preprocess_image(roi))
    feature = model(roi_resized).detach().numpy().flatten()
    features.append(feature)
    if idx < 50:
        labels.append(1)
    else:
        labels.append(0)


# Transform to NumPy ndarray
features = np.array(features)
labels = np.array(labels)


# training and validation datasets
x_train, x_test, y_train, y_test =\
    train_test_split(features, labels, test_size=0.2, random_state=42)


# training SVM classifier
classifier = SVC(kernel='linear')
classifier.fit(x_train, y_train)


# prediction and calculate accuracy
y_pred = classifier.predict(x_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%')


# Display final result
plt.figure(figsize=(10, 6))
plt.imshow(image_rgb)
ax = plt.gca()

for (x, y, w, h, _) in regions[:10]:
    rect = plt.Rectangle((x, y), w, h, edgecolor='red', facecolor='none', linewidth=2)
    ax.add_patch(rect)
plt.show()
