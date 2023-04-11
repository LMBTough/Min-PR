import os
import pickle as pkl
from tqdm import tqdm
import numpy as np
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10, CIFAR100


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageNetDataset(Dataset):
    def __init__(self, train, transform) -> None:
        super().__init__()
        if train != False:
            raise NotImplementedError
        self.data = pd.read_csv('data/mini-imagenet/test.csv')
        self.transform = transform
        self.class_name2label = pkl.load(
            open('data/class_name2label.pkl', 'rb'))
        self.classes = list(self.class_name2label.keys())
        if not os.path.exists(f'data/mini-imagenet/test_imgs.pt') or not os.path.exists(f'data/mini-imagenet/test_labels.pt'):
            self.imgs = list()
            self.labels = list()
            for i, row in tqdm(self.data.iterrows(), total=len(self.data)):
                img = pil_loader(
                    "data/mini-imagenet/images/" + row['filename'])
                img = self.transform(img)
                self.imgs.append(img)
                self.labels.append(int(self.class_name2label[row['label']]))
            self.imgs = torch.stack(self.imgs)
            self.labels = torch.LongTensor(self.labels)
            torch.save(
                self.imgs, f'data/mini-imagenet/test_imgs.pt')
            torch.save(
                self.labels, f'data/mini-imagenet/test_labels.pt')
        else:
            self.imgs = torch.load(
                f'data/mini-imagenet/test_imgs.pt')
            self.labels = torch.load(
                f'data/mini-imagenet/test_labels.pt')

    def __getitem__(self, index):
        return self.imgs[index], self.labels[index]

    def __len__(self):
        return len(self.data)


def load_imagenet(batch_size=32):
    _normalizer = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        _normalizer
    ])
    dataset = ImageNetDataset(train=False, transform=transform)
    data_min = np.min((0 - np.array(_normalizer.mean)) /
                      np.array(_normalizer.std))
    data_max = np.max((1 - np.array(_normalizer.mean)) /
                      np.array(_normalizer.std))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return None, dataloader, data_min, data_max


def load_cifar10(root='./data', download=True, batch_size=128):
    _normalizer = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        _normalizer
    ])
    train_dataset = CIFAR10(root=root, train=True,
                            download=download, transform=transform)
    test_dataset = CIFAR10(root=root, train=False,
                           download=download, transform=transform)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    data_min = np.min((0 - np.array(_normalizer.mean)) /
                      np.array(_normalizer.std))
    data_max = np.max((1 - np.array(_normalizer.mean)) /
                      np.array(_normalizer.std))
    return train_dataloader, test_dataloader, data_min, data_max


def load_cifar100(root='./data', download=True, batch_size=128):
    _normalizer = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
    transform = transforms.Compose([
        transforms.ToTensor(),

    ])
    train_dataset = CIFAR100(root=root, train=True,
                             download=download, transform=transform)
    test_dataset = CIFAR100(root=root, train=False,
                            download=download, transform=transform)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    data_min = np.min((0 - np.array(_normalizer.mean)) /
                      np.array(_normalizer.std))
    data_max = np.max((1 - np.array(_normalizer.mean)) /
                      np.array(_normalizer.std))
    return train_dataloader, test_dataloader, data_min, data_max
