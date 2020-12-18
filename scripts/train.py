import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
from argparse import ArgumentParser
from covidx.models.baseline_lightning import XRayClassification

def main(args):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    train_set = ImageFolder('../data/covidx_image_folder/train',
                            transform=transform)
    valid_set = ImageFolder('../data/covidx_image_folder/validation/',
                            transform=transform)

    train_loader_folder = DataLoader(train_set,
                                     batch_size=24,
                                     shuffle=True,
                                     num_workers=12,
                                     pin_memory=True)
    valid_loader_folder = DataLoader(valid_set,
                                     batch_size=24,
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
