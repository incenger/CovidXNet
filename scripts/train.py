import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
from argparse import ArgumentParser
from covidx.models.baseline_lightning import XRayClassification
from covidx.dataset import create_balance_dl, xray_augmentation

SEED = 1411


def main(args):
    pl.seed_everything(SEED)

    transform = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.Grayscale(num_output_channels=1),
        xray_augmentation(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )),
    ])

    train_set = ImageFolder('../data/covidx_image_folder/train',
                            transform=transform)
    valid_set = ImageFolder('../data/covidx_image_folder/validation/',
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

    resnetx = XRayClassification()

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(resnetx, train_loader_folder, valid_loader_folder)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
