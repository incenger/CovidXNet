import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
import numpy as np

from covidx.metrics import covid_xray_metrics

from .baseline import ResnetCovidX, EfficientNetCovidXray, ConvNetXray

class XRayClassification(pl.LightningModule):
    """
    Args:
        num_class: number of output classes
    """

    def __init__(self, num_class=3):
        super().__init__()
        self.num_class = num_class
        # self.model = ResnetCovidX(num_class=num_class)
        # self.model = ConvNetXray(num_class=num_class)
        self.model = EfficientNetCovidXray(num_class=num_class)

        # https://github.com/AdeelH/pytorch-multi-class-focal-loss
        self.focal_loss = torch.hub.load(
            'adeelh/pytorch-multi-class-focal-loss',
            model='FocalLoss',
            alpha=torch.tensor([2.0, 1, 1.25]),
            gamma=2,
            reduction='mean',
            force_reload=False
        )


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
        log_logits = F.log_softmax(out, dim=1)
        probs = F.softmax(out, dim=1)
        loss = self.focal_loss(probs, y)

        self.log('train_loss', loss, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        """
        x, y = batch
        logits = self.model(x)
        log_logits = F.log_softmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)
        loss = self.focal_loss(probs, y)
        preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)

        self.log('val_loss', loss)

        return {'val_loss': loss, 'labels': y, 'preds': preds, 'probs': probs}

    def validation_epoch_end(self, outputs):
        """
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        labels = torch.cat([x['labels'] for x in outputs])
        preds = torch.cat([x['preds'] for x in outputs])
        probs = torch.cat([x['probs'] for x in outputs])

        acc = accuracy(preds, labels)
        auc_score = None
        try:
            auc_score = roc_auc_score(
                labels.detach().cpu().numpy(), probs.detach().cpu().numpy(),  multi_class='ovr')
        except:
            auc_score = 0
            pass

        # tensorboard_log = {'val_loss': avg_loss}

        self.log('val_acc', acc, prog_bar=True, logger=True)
        self.log('val_loss', avg_loss, prog_bar=True, logger=True)
        self.log('auc_score', auc_score, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)

        return {'labels': y, 'preds': preds, 'probs': probs}

    def test_epoch_end(self, outputs):
        labels = torch.cat([x['labels'] for x in outputs])
        preds = torch.cat([x['preds'] for x in outputs])
        probs = torch.cat([x['probs'] for x in outputs])


        acc = accuracy(preds, labels)
        xray_metrics = covid_xray_metrics(labels, preds)

        auc_score = None
        try:
            auc_score = roc_auc_score(
                labels.detach().cpu().numpy(), probs.detach().cpu().numpy(),  multi_class='ovr')
        except:
            auc_score = 0
            pass

        return {
            'test_acc': acc,
            'sensitivity': xray_metrics['sensitivity'],
            'ppv': xray_metrics['ppv'],
            'cm': xray_metrics['cm'],
            # 'auc_score': auc_score
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
