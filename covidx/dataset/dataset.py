import random

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import ImageFolder


def xray_augmentation():
    """ Augmentation for X-ray images

    Reference: Section 3.5 in https://arxiv.org/pdf/2003.09871.pdf

    Including: Translation, Rotation, Horizontal Flip, Zoom and Intensity Shift
    """

    xray_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomAffine(degrees=10,
                                translate=(0.1, 0.1),
                                scale=(0.85, 1.15)),
        transforms.ColorJitter(brightness=0.1)
    ])

    return xray_aug


def create_balance_dl(dataset, batch_size, num_workers):

    unique, counts = torch.unique(torch.tensor(dataset.targets),
                                  return_counts=True)

    counts = counts.to(torch.float)
    class_weights = 1.0 / counts

    sample_weights = torch.tensor([class_weights[x] for x in dataset.targets])

    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        sample_weights, 30000) # Fix later

    dl = torch.utils.data.DataLoader(dataset,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     sampler=sampler,
                                     pin_memory=True)

    return dl


def croptop(img: Image, percent: float) -> Image:
    """Crop (percent*height) pixels from the top of the image.


    Args:
        img: Image to crop
        percent: float value from 0 to 1

    Returns:
        Image: cropped image
    """

    w, h = img.size
    offset = int(h * percent)

    cropped = img.crop((0, offset, w, h))

    return cropped


def central_crop(img: Image) -> Image:
    w, h = img.size
    size = min(w, h)
    offset_h = int((h - size) / 2)
    offset_w = int((w - size) / 2)

    cropped = img.crop((offset_w, offset_h, offset_w + size, offset_h + size))

    return cropped


class CovidxDataset(ImageFolder):
    def __init__(self, root, transform, state):
        super().__init__(root)
        self.covid_sample = [s for s in self.samples if s[1] == 0]
        self.normal_sample = [s for s in self.samples if s[1] == 1]
        self.pneumonia_sample = [s for s in self.samples if s[1] == 2]
        self.transform = transform
        self.turn = 0
        self.state = state

    def shuffleSamples(self):
        random.shuffle(self.covid_sample)
        random.shuffle(self.normal_sample)
        random.shuffle(self.pneumonia_sample)

    def __len__(self):
        if self.state == 'train':
            return max(len(self.covid_sample), len(self.normal_sample),
                       len(self.pneumonia_sample)) * 3
        return len(self.samples)

    def __getitem__(self, index):
        if self.state == 'train':
            pos = index // 3
            self.turn %= 3
            if self.turn == 0:
                pos %= len(self.pneumonia_sample)
                path, target = self.pneumonia_sample[pos]
            elif self.turn == 1:
                pos %= len(self.normal_sample)
                path, target = self.normal_sample[pos]
            else:
                pos %= len(self.covid_sample)
                path, target = self.covid_sample[pos]
            self.turn += 1
        else:
            path, target = self.samples[index]
        sample = self.loader(path)
        sample = croptop(sample, 0.15)
        sample = central_crop(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
