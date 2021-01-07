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

class InceptionV3(nn.Module):
    """
    InceptionV3 Backbone

    Be careful, the model expects (299, 299) input size and 3 color channels
    """

    def __init__(self, out_features=3):
        super().__init__()
        self.out_features = out_features
        self.backbone = models.inception_v3(pretrained=True)

        # Auxilary net
        num_ftrs = self.backbone.AuxLogits.fc.in_features
        self.backbone.AuxLogits.fc = nn.Linear(num_ftrs, self.out_features)

        # Primary net
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, self.out_features)


    def forward(self, x):
        return self.backbone(x)


class InceptionV3Lightning(pl.LightningModule):
    """
    Args:
        num_class: number of output classes
    """
    def __init__(self, num_class=3):
        super().__init__()
        self.num_class = num_class
        self.model = InceptionV3(num_class)

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
        prim_out, aux_out  = self.model(x)
        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958

        weight = torch.tensor([0.6, 0.2, 0.2]).to(x.device)

        aux_loss = F.cross_entropy(aux_out, y, weight=None)
        prim_loss = F.cross_entropy(prim_out, y, weight=None)

        loss = prim_loss + 0.4*aux_loss

        self.log('train_loss', loss, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        """
        x, y = batch
        logits = self.model(x)

        weight = torch.tensor([0.6, 0.2, 0.2]).to(x.device)
        loss = F.cross_entropy(logits, y, weight=None)

        # loss = F.cross_entropy(logits, y)
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

#         lr_dict = {
#             'scheduler': ExponentialLR(optimizer, gamma=0.97, verbose=True),
#             # 'scheduler': ReduceLROnPlateau(optimizer, patience=5, verbose=True),
#             'interval': 'epoch',  # The unit of the scheduler's step size
#             'frequency': 2,  # The frequency of the scheduler
#             # 'reduce_on_plateau': Fj, # For ReduceLROnPlateau scheduler
#             # 'monitor': 'val_loss', # Metric for ReduceLROnPlateau to monitor
#             'strict':
#             True,  # Whether to crash the training if `monitor` is not found
#         }

        return optimizer
