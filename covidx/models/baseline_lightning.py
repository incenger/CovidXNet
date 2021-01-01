import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torch.optim.lr_scheduler import ReduceLROnPlateau

from covidx.metrics import covid_xray_metrics

from .baseline import ResnetCovidX, EfficientNetCovidXray, ConvNetXray


import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/gokulprasadthekkel/pytorch-multi-class-focal-loss/blob/master/focal_loss.py
# https://github.com/AdeelH/pytorch-multi-class-focal-loss
class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        # weight parameter will act as the alpha parameter to balance class weights
        self.weight = weight
        cuda0 = torch.device('cuda:0')
        self.class_weights = torch.tensor([2, 1.25, 1], device=cuda0)

    def forward(self, input, target):
        ce_loss = F.cross_entropy(
            input, target, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class XRayClassification(pl.LightningModule):
    """
    Args:
        num_class: number of output classes
    """
    def __init__(self, num_class=3, class_weights=None):
        super().__init__()
        self.num_class = num_class
        # self.model = ResnetCovidX(num_class=num_class)
        # self.model = ConvNetXray(num_class=num_class)
        self.model = EfficientNetCovidXray(num_class=num_class)
        self.focal_loss = FocalLoss()


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
        loss = self.focal_loss(out, y)

        self.log('train_loss', loss, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        """
        x, y = batch
        logits = self.model(x)
        loss = self.focal_loss(logits, y)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        lr_dict = {
            'scheduler': ReduceLROnPlateau(optimizer, patience=5),
            'interval': 'epoch', # The unit of the scheduler's step size
            'frequency': 1, # The frequency of the scheduler
            'reduce_on_plateau': True, # For ReduceLROnPlateau scheduler
            'monitor': 'val_loss', # Metric for ReduceLROnPlateau to monitor
            'strict': True, # Whether to crash the training if `monitor` is not found
        }

        return [optimizer], [lr_dict]
