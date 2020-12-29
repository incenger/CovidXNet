import random
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder

def xray_augmentation():
    """ Augmentation for X-ray images

    Reference: Section 3.5 in https://arxiv.org/pdf/2003.09871.pdf

    Including: Translation, Rotation, Horizontal Flip, Zoom and Intensity Shift
    """

    xray_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomAffine(degrees=10, translate=(0.2, 0.2), scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2)
    ])

    return xray_aug

def create_balance_dl(dataset, batch_size, num_workers):

    unique, counts = torch.unique(torch.tensor(dataset.targets),
                                  return_counts=True)

    counts = counts.to(torch.float)
    class_weights = 1.0 / counts

    sample_weights = torch.tensor([class_weights[x] for x in dataset.targets])

    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        sample_weights, len(dataset))

    dl = torch.utils.data.DataLoader(dataset,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     sampler=sampler,
                                     pin_memory=True)

    return dl
def croptop(img,percent):
    img = np.asarray(img)
    offset = int(img.shape[0] * percent)
    img = img[offset:]
    return Image.fromarray(img)
def central_crop(img):
    img = np.asarray(img)
    size = min(img.shape[0], img.shape[1])
    offset_h = int((img.shape[0] - size) / 2)
    offset_w = int((img.shape[1] - size) / 2)
    img = img[offset_h:offset_h + size, offset_w:offset_w + size]
    return Image.fromarray(img)
class CovidxDataset(ImageFolder):
    def __init__(self,root,transform,state):
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

    def __getitem__(self, index):

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
    def __len__(self):
        if self.state=='train':
            return max(len(self.covid_sample),len(self.normal_sample),len(self.pneumonia_sample))*3
        return len(self.samples)
    def __getitem__(self, index):
        if self.state == 'train':
            pos = index // 3
            self.turn%=3
            if self.turn==0:
                pos %= len(self.pneumonia_sample)
                path,target = self.pneumonia_sample[pos]
            elif self.turn ==1:
                pos %= len(self.normal_sample)
                path,target = self.normal_sample[pos]
            else:
                pos %= len(self.covid_sample)
                path,target = self.covid_sample[pos]
            self.turn +=1
        else:
            path,target = self.samples[index]
        sample = self.loader(path)
        sample = croptop(sample,0.15)
        sample = central_crop(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
