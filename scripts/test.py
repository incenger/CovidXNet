import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from covidx.models.baseline_lightning import XRayClassification

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

test_set = ImageFolder('../data/covidx_image_folder/test/',
                       transform=transform)

test_loader = DataLoader(test_set,
                         batch_size=24,
                         shuffle=False,
                         num_workers=12,
                         pin_memory=True)

resnetx = XRayClassification.load_from_checkpoint(
    './lightning_logs/version_1/checkpoints/epoch=6-step=3667.ckpt',
    map_location=lambda storage, loc: storage)

resnetx.to(torch.device('cuda'))

trainer = pl.Trainer(gpus=1)

result_dict = trainer.test(resnetx, test_loader, verbose=False)[0]

for k, v in result_dict.items():
    print(k)
    print(v)
