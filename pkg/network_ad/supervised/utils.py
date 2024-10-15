import itertools

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def compute_confusion_matrix(y_true, y_pred, class_names, normalize=False):
    """
    Compute confusion matrix and return matplotlib figure.

    :param y_true: true labels
    :param y_pred: predicted labels
    :param class_names: list of class names
    :param normalize: if True, normalize the confusion matrix
    :return: confusion matrix
    """

    cm = confusion_matrix(y_true, y_pred, labels=sorted(np.unique(y_true).tolist()))

    if normalize:
        # Normalize the confusion matrix
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm = cm.astype("float") / cm_sum

    # Addapt figure size to number of classes
    n_classes = len(class_names)
    if n_classes < 10:
        figsize = (10, 10)
    elif n_classes < 30:
        figsize = (20, 20)
    else:
        figsize = (n_classes, n_classes)

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)

    if normalize:
        fmt = ".2f"
    else:
        fmt = "d"

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    return plt.gcf()