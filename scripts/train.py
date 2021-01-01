import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
from covidx.dataset.dataset import CovidxDataset
from argparse import ArgumentParser
from covidx.models.baseline_lightning import XRayClassification
from covidx.dataset import create_balance_dl, xray_augmentation
import csv
import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from covidx.dataset.dataset import CovidxDataset
from covidx.models.baseline_lightning import XRayClassification

SEED = 1411

TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((260, 260)),  # change the input image size if needed
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, )),
])

CLS_MAPPING = ['COVID-19', 'normal', 'pneumonia']


def main(args):
    pl.seed_everything(SEED)

    transform = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.Grayscale(num_output_channels=1),
        xray_augmentation(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )),
    ])

    train_set = CovidxDataset('../data/covidx_image_folder/train',
                            transform=transform, state='train')
    valid_set = CovidxDataset('../data/covidx_image_folder/validation/',
                            transform=transform)

    # train_loader_folder = create_balance_dl(train_set,
    #                                         batch_size=32,
    #                                         num_workers=12)

    train_loader_folder = DataLoader(train_set,
                                     batch_size=32,
                                     shuffle=True,
                                     num_workers=12,
                                     pin_memory=True)

    valid_loader_folder = DataLoader(valid_set,
                                     batch_size=32,
                                     shuffle=False,
                                     num_workers=12,
                                     pin_memory=True)

    resnetx = XRayClassification(num_class=3)

    trainer = pl.Trainer.from_argparse_args(
        args, default_root_dir='/content/drive/My Drive/Models/CovidNetCheckpoints/')
    trainer.fit(resnetx, train_loader_folder, valid_loader_folder)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
