#! coding: utf-8

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def distillation_loss(y_student, y_teacher, labels, T, alpha):
    """ Compute distillation loss using KL divergence """
    # The ground truth cross entropy
    loss_ce = F.cross_entropy(y_student, labels)

    # soft label cross entropy loss
    p_student = F.log_softmax(y_student / T, dim=1)
    p_teacher = F.softmax(y_teacher / T, dim=1)

    loss_kd = F.kl_div(p_student, p_teacher, reduction='batchmean') * (T * T)

    return alpha * loss_kd + (1 - alpha) * loss_ce


def train_teacher(model: nn.Module,
                  data_iter: torch.utils.data.DataLoader,
                  optimizer, scheduler, criterion, num_epochs, device):

    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in data_iter:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(inputs)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(data_iter)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')


def train_distil_student(model_student: nn.Module,
                         model_teacher: nn.Module,
                         data_iter: torch.utils.data.DataLoader,
                         optimizer, scheduler, num_epochs,
                         temperature, alpha, device):

    model_student.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in data_iter:
            inputs, labels = inputs.to(device), labels.to(device)

            # Compute student model
            optimizer.zero_grad()
            output_student = model_student(inputs)

            # Compute teacher model (Notes: should not update teacher model parameters)
            with torch.no_grad():
                output_teacher = model_teacher(inputs)

            loss = distillation_loss(output_student, output_teacher, labels, temperature, alpha)

            loss.backward()  # Compute backward weights
            optimizer.step()  # Update weights
            scheduler.step()  # Update learning rate

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(data_iter)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')


def load_datasets(batch_size, data_transforms):
    train_dataset = torchvision.datasets.CIFAR10(
        root='../data', train=True, download=True, transform=data_transforms['train'])
    test_dataset = torchvision.datasets.CIFAR10(
        root='../data', train=False, download=True, transform=data_transforms['infer'])

    return (
        torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
        torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
    )


def inference(model, data_iter, device):
    model.eval()

    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in data_iter:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


class IntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.device = dlf.devices()[0]

    def test_distil(self):
        image_size = 256  # Image scaling size
        crop_size = 224  # Image cropping size

        batch_size = 128
        num_epochs_teacher = 20
        num_epochs_student = 140
        learning_rate_student_initial = 0.1
        update_student_lr_steps = 40
        update_student_lr_ratio = 0.2

        learning_rate_teacher_initial = 0.1
        update_teacher_lr_steps = 10
        update_teacher_lr_ratio = 0.1
        temperature = 4.0
        alpha = 0.7  # soft label (for teacher model) loss weight

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
            'infer': transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
        }

        train_iter, test_iter = load_datasets(batch_size=batch_size, data_transforms=data_transforms)

        teacher = TeacherModel().to(device=self.device)
        student = SimpleConv5().to(device=self.device)

        optimizer_teacher = torch.optim.Adam(teacher.parameters(), lr=learning_rate_teacher_initial)
        scheduler_teacher = torch.optim.lr_scheduler.StepLR(
            optimizer_teacher, step_size=update_teacher_lr_steps, gamma=update_teacher_lr_ratio)

        criterion_teacher = nn.CrossEntropyLoss()

        # Teacher model training
        print('Training teacher model...')
        train_teacher(teacher, train_iter, optimizer_teacher, scheduler_teacher,
                      criterion_teacher, num_epochs_teacher, self.device)

        # Fix teacher model parameters
        teacher.eval()

        # Student model optimizer and scheduler
        optimizer_student = torch.optim.SGD(student.parameters(), lr=learning_rate_student_initial)
        scheduler_student = torch.optim.lr_scheduler.StepLR(
            optimizer_student, step_size=update_student_lr_steps, gamma=update_student_lr_ratio)

        print('Training distilling student model...')
        train_distil_student(student, teacher, train_iter, optimizer_student, scheduler_student,
                             num_epochs_student, temperature, alpha, self.device)

        acc_student = inference(student, test_iter, self.device)

        print(f'The accuracy of the student model: {acc_student*100:.2f}%')

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main(verbosity=True)
