import itertools

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cmt,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cmt = cmt.astype('float') / cmt.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    fig = plt.gcf()
    plt.figure(figsize=(10, 10))
    plt.imshow(cmt, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cmt.max() / 2.
    for i, j in itertools.product(range(cmt.shape[0]), range(cmt.shape[1])):
        plt.text(j,
                 i,
                 format(cmt[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cmt[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
