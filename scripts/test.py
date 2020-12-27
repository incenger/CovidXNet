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

from covidx.models.baseline_lightning import XRayClassification

TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((260, 260)),  # change the input image size if needed
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, )),
])

CLS_MAPPING = ['COVID-19', 'normal', 'pneumonia']


def test():

    test_set = ImageFolder('../data/covidx_image_folder/test/',
                           transform=TEST_TRANSFORM)

    test_loader = DataLoader(test_set,
                             batch_size=32,
                             shuffle=False,
                             num_workers=12,
                             pin_memory=True)

    resnetx = XRayClassification.load_from_checkpoint(
        './lightning_logs/version_40/checkpoints/epoch=28-step=11396.ckpt',
        map_location=lambda storage, loc: storage)

    resnetx.to(torch.device('cuda'))

    trainer = pl.Trainer(gpus=1, checkpoint_callback=False, logger=False)

    result_dict = trainer.test(resnetx, test_loader, verbose=False)[0]

    for k, v in result_dict.items():
        print(k)
        print(v)


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def test_debug(ckpt_path, test_folder, failure_path):
    """ Evaluate and write failure cases to a file

    The failure csv file has 3 fields: File name, Prediction, Label

    Args:
        ckpt_path: path to model checkpoint
        test_folder: test folder - ImageFolder-like structure
        failure_path: path to failure csv file
    """

    # Load model checkpoint
    model = XRayClassification.load_from_checkpoint(
        ckpt_path, map_location=lambda storage, loc: storage)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Evaluation
    failure = []
    for cls in os.listdir(test_folder):
        print(cls)
        for sample in tqdm(os.listdir(os.path.join(test_folder, cls))):
            sample_path = os.path.join(test_folder, cls, sample)
            sample_img = pil_loader(sample_path)
            sample_tensor = TEST_TRANSFORM(sample_img)
            sample_tensor = sample_tensor.to(device).unsqueeze(0)
            logit = model(sample_tensor)
            pred = torch.argmax(F.log_softmax(logit, dim=1), dim=1).item()
            pred_str = CLS_MAPPING[pred]

            if pred_str != cls:
                failure.append((os.path.join(cls, sample), pred_str, cls))

    # Write failure
    with open(failure_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["File", "Prediction", "Label"])
        writer.writerows(failure)


if __name__ == "__main__":
    test_debug(
        "./lightning_logs/version_40/checkpoints/epoch=28-step=11396.ckpt",
        "../data/covidx_image_folder/test/", "failure.csv")
