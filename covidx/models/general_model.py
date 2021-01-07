import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torchvision import models, transforms
from torchvision.datasets import MNIST

from covidx.metrics import covid_xray_metrics

from .backbone import (ConvNetXray, DenseNetCovidX, EfficientNetCovidXray,
                       ResnetCovidX)


class XRayClassification(pl.LightningModule):
    """
    Args:
        num_class: number of output classes
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input image, shape (B, C, H, W)

        Returns:
            torch.Tensor: output logits tensor, shape (B, num_class)
        """
        out: torch.Tensor = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        """
        """
        x, y = batch
        out = self.model(x)
        # weight = torch.tensor([0.8, 0.1, 0.1]).to(x.device)

        loss = F.cross_entropy(out, y)

        self.log('train_loss', loss, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        """
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)

        # weight = torch.tensor([0.8, 0.1, 0.1]).to(x.device)
        # loss = F.cross_entropy(logits, y, weight=weight)
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
        self.log('val_loss', avg_loss, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)

        return {'labels': y, 'preds': preds}

    def test_epoch_end(self, outputs):
        labels = torch.cat([x['labels'] for x in outputs])
        preds = torch.cat([x['preds'] for x in outputs])

        acc = accuracy(preds, labels)
        xray_metrics = covid_xray_metrics(labels, preds)

        return {
            'test_acc': acc,
            'sensitivity': xray_metrics['sensitivity'],
            'ppv': xray_metrics['ppv'],
            'cm': xray_metrics['cm']
        }

    def configure_optimizers(self):
        """
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return optimizer
