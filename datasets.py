from torchvision.datasets import CIFAR10, CIFAR100, Caltech101
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np


def load_cifar10(root='./data', download=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    train_dataset = CIFAR10(root=root, train=True,
                            download=download, transform=transform)
    test_dataset = CIFAR10(root=root, train=False,
                           download=download, transform=transform)
    train_dataloader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=16, shuffle=False, pin_memory=True)
    data_min = np.min((0 - np.array([0.5, 0.5, 0.5])) /
                      np.array([0.5, 0.5, 0.5]))
    data_max = np.max((1 - np.array([0.5, 0.5, 0.5])) /
                      np.array([0.5, 0.5, 0.5]))
    return train_dataloader, test_dataloader, data_min, data_max


def load_cifar100(root='./data', download=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    train_dataset = CIFAR100(root=root, train=True,
                             download=download, transform=transform)
    test_dataset = CIFAR100(root=root, train=False,
                            download=download, transform=transform)
    train_dataloader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=16, shuffle=False, pin_memory=True)
    data_min = np.min((0 - np.array([0.5, 0.5, 0.5])) /
                      np.array([0.5, 0.5, 0.5]))
    data_max = np.max((1 - np.array([0.5, 0.5, 0.5])) /
                      np.array([0.5, 0.5, 0.5]))
    return train_dataloader, test_dataloader, data_min, data_max


def load_caltech_101(root='./data', download=True):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    torch.manual_seed(0)
    dataset = Caltech101(root=root, download=download, transform=transform)
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
    train_targets = list()
    for i in range(len(train_dataset)):
        train_targets.append(train_dataset[i][1])
    train_targets = np.array(train_targets)
    test_targets = list()
    for i in range(len(test_dataset)):
        test_targets.append(test_dataset[i][1])
    test_targets = np.array(test_targets)
    train_dataloader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=16, shuffle=False, pin_memory=True)
    data_min = np.min((0 - np.array([0.485, 0.456, 0.406])) /
                      np.array([0.229, 0.224, 0.225]))
    data_max = np.max((1 - np.array([0.485, 0.456, 0.406])) /
                      np.array([0.229, 0.224, 0.225]))
    return train_dataloader, test_dataloader, data_min, data_max
