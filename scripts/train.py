from argparse import ArgumentParser

import pytorch_lightning as pl
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder

from covidx.dataset import create_balance_dl, xray_augmentation
from covidx.dataset.dataset import CovidxDataset
from covidx.models import (ConvNetXray, DenseNetCovidX, EfficientNetCovidXray,
                           ResnetCovidX, XRayClassification)

SEED = 1411
LOSS_WEIGHT=None


def main(args):
    pl.seed_everything(SEED)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        xray_augmentation(),
        transforms.ToTensor(),
        # Normalize for rgb image
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
        # Normalize for grayscale
        transforms.Normalize((0.5, ), (0.5, )),
    ])

    train_set = ImageFolder('../data/covidx_image_folder/train',
                            transform=transform)
    valid_set = ImageFolder('../data/covidx_image_folder/validation',
                            transform=transform)

    # train_loader_folder = create_balance_dl(train_set,
    #                                         batch_size=16,
    #                                         num_workers=12)

    train_loader_folder = DataLoader(train_set,
                                     batch_size=16,
                                     shuffle=True,
                                     num_workers=12,
                                     pin_memory=True)

    valid_loader_folder = DataLoader(valid_set,
                                     batch_size=32,
                                     shuffle=False,
                                     num_workers=12,
                                     pin_memory=True)

    backbone = EfficientNetCovidXray()
    model = XRayClassification(backbone, class_weight=LOSS_WEIGHT)

    checkpoint_callback = ModelCheckpoint(monitor='val_acc')
    trainer = pl.Trainer.from_argparse_args(
        args, checkpoint_callback=checkpoint_callback)
    trainer.fit(model, train_loader_folder, valid_loader_folder)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
