# Reference: https://deeplizard.com/learn/video/0LhiS6yu2qQ

import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

@torch.no_grad()
def get_all_preds(model, loader):
    model.eval()
    all_preds = torch.tensor([]).cuda()
    for batch in loader:
        images, labels = batch
      
        images = images.cuda()
        labels = labels.cuda()

        preds = model(images)
        all_preds = torch.cat(
            (all_preds, preds)
            ,dim=0
        )
    return all_preds

def evaluation_metrics(dataset, predictions):
    """
      return
        dict consiting of:
        - sensitivity (3)
        - ppv (3)
        - confusion_matrix (3x3)
    """
    nclasses = len(dataset.classes)

    predicted_labels =  predictions.argmax(dim=1).cpu().numpy()
    cmt = confusion_matrix(dataset.targets, predicted_labels)

    sensitivity = np.zeros(nclasses)
    ppv = np.zeros(nclasses)

    for c in range(nclasses):
      sensitivity[c] = cmt[c][c] / np.sum(cmt, axis=0)[c]
      ppv[c] = cmt[c][c] / np.sum(cmt, axis=1)[c]

    return {
        "cmt": cmt,
        "sensitivity": sensitivity,
        "ppv": ppv
    }

def plot_confusion_matrix(cmt, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
  if normalize:
    cmt = cmt.astype('float') / cmt.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
  else:
    print("Confusion matrix, without normalization")

  plt.imshow(cmt, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cmt.max() / 2.
  for i, j in itertools.product(range(cmt.shape[0]), range(cmt.shape[1])):
    plt.text(j, i, format(cmt[i, j], fmt), horizontalalignment="center", color="white" if cmt[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel("True label")

  plt.xlabel("Predicted label")


def get_all_evaluation_metrics(model, dataloader):
  plt.figure(figsize=(10,10))
  with torch.no_grad():
    predictions = get_all_preds(model, dataloader)

  evaluation_dict = evaluation_metrics(train_set, predictions)

  print("Sensitivity")
  print(evaluation_dict["sensitivity"])
  print("PPV")
  print(evaluation_dict["ppv"])
  print("Confusion matrix")
  print(evaluation_dict["cmt"])

  plot_confusion_matrix(cmt, train_set.classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues)
