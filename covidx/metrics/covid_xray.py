# Reference: https://deeplizard.com/learn/video/0LhiS6yu2qQ

import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix


def covid_xray_metrics(gts, preds):
    """ Evaluation metrics used for Covid X-ray Image classification

    Reference: https://arxiv.org/pdf/2003.09871.pdf

    Args:
        gts: ground truth tensor
        preds: prediction tensor

    Returns:
        (dict): A dictionary consiting of:
            - sensitivity
            - ppv
            - confusion matrix
    """
    if torch.is_tensor(gts):
        gts = gts.detach().cpu().numpy().reshape(-1)

    if torch.is_tensor(preds):
        preds = preds.detach().cpu().numpy().reshape(-1)

    nclasses = len(np.unique(gts))

    cm = confusion_matrix(gts, preds)

    sensitivity = np.zeros(nclasses)
    ppv = np.zeros(nclasses)

    for c in range(nclasses):
        sensitivity[c] = cm[c][c] / np.sum(cm, axis=1)[c]
        ppv[c] = cm[c][c] / np.sum(cm, axis=0)[c]

    return {"cm": cm, "sensitivity": sensitivity, "ppv": ppv}
