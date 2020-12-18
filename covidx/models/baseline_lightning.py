import os

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models, transforms
from torchvision.datasets import MNIST

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from .baseline import ResnetCovidX


class XRayClassification(pl.LightningModule):
    """
    Args:
        num_class: number of output classes
    """
    def __init__(self, num_class=3):
        super().__init__()
        self.num_class = num_class
        self.model = ResnetCovidX(num_class=num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input image, shape (B, C, H, W)

        Returns:
            torch.Tensor: output logits tensor, shape (B, num_class)
        """
        out: torch.Tensor = self.model(x)

    def training_step(self, batch, batch_idx):
        """
        """
        x, y = batch
        out = self.model(x)
        loss = F.cross_entropy(out, y)

        self.log('train_loss', loss, logger=True)
        return loss


    def validation_step(self, batch, batch_idx):
        """
        """
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)

        self.log('val_loss', loss)

        return {'val_loss': loss, 'labels': y, 'preds': preds}

    def validation_epoch_end(self, outputs):
        """
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        labels = torch.cat([x['labels'] for x in outputs])
        preds = torch.cat([x['preds'] for x in outputs])

        acc = accuracy(preds, labels)

        tensorboard_log = {'val_loss': avg_loss}

        self.log('val_acc', acc, prog_bar=True, logger=True)
        self.log('val_loss', avg_loss, prog_bar=False, logger=True)

    def configure_optimizers(self):
        """
        """
        return torch.optim.Adam(self.parameters(), lr=1e-3)
