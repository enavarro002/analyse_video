import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from random import randint


def plot_loss_acc(history):
    """Plot training and (optionally) validation loss and accuracy"""

    loss = history.history["loss"]
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(10, 10))

    plt.subplot(2, 1, 1)
    plt.plot(epochs, loss, ".--", label="Training loss")
    final_loss = loss[-1]
    title = "Training loss: {:.4f}".format(final_loss)
    plt.ylabel("Loss")
    if "val_loss" in history.history:
        val_loss = history.history["val_loss"]
        plt.plot(epochs, val_loss, "o-", label="Validation loss")
        final_val_loss = val_loss[-1]
        title += ", Validation loss: {:.4f}".format(final_val_loss)
    plt.title(title)
    plt.legend()

    acc = history.history["accuracy"]

    plt.subplot(2, 1, 2)
    plt.plot(epochs, acc, ".--", label="Training acc")
    final_acc = acc[-1]
    title = "Training accuracy: {:.2f}%".format(final_acc * 100)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    if "val_accuracy" in history.history:
        val_acc = history.history["val_accuracy"]
        plt.plot(epochs, val_acc, "o-", label="Validation acc")
        final_val_acc = val_acc[-1]
        title += ", Validation accuracy: {:.2f}%".format(final_val_acc * 100)
    plt.title(title)
    plt.legend()


def plot_multiclass_heatmap(y_test, y_predict, labels, figsize=(16, 14)):
    """Affichage de la matrice de confusion"""

    y_pred = np.argmax(y_predict, axis=-1)
    y_true = np.argmax(y_test, axis=-1)

    cm = confusion_matrix(y_true, y_pred, normalize="true")

    ## Get Class Labels
    # labels = le.classes_
    class_names = labels

    class_names = list(class_names)

    # Plot confusion matrix in a beautiful manner
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot()
    cm = cm * 100
    sns.heatmap(cm, annot=True, ax=ax, fmt=".1f")
    # annot=True to annotate cells
    # labels, title and ticks
    ax.set_xlabel("Predicted", fontsize=20)
    ax.xaxis.set_label_position("bottom")
    plt.xticks(rotation=90)
    ax.xaxis.set_ticklabels(class_names, fontsize=15)
    ax.xaxis.tick_bottom()

    ax.set_ylabel("True", fontsize=20)
    ax.yaxis.set_ticklabels(class_names, fontsize=15)
    plt.yticks(rotation=0)

    plt.title("Confusion Matrix", fontsize=20)

    plt.show()


def plot_box(plt, coords=None, rect=None ,label="", color="yellow", linewidth=2):
    """
    == Input ==

    plt   : matplotlib.pyplot object
    label : string containing the object class name
    coords: [
        top left corner x coordinate
        top left corner y coordinate
        bottom right corner x coordinate
        bottom right corner y coordinate
    ]
    rect: [
        top left corner x coordinate
        top left corner y coordinate
        width
        height
    ]
    """
    assert coords is not None or rect is not None, "Need rectangle position and dimension"

    if coords is not None:
        x1, y1, x2, y2 = coords
    else:
        assert rect[2] != 0
        assert rect[3] != 0
        x1, y1, x2, y2 = rect
        x2 += x1
        y2 += y1

    if label != "":
        plt.text(x1, y1, label, fontsize=20, backgroundcolor="magenta")

    plt.plot([x1, x1], [y1, y2], linewidth=linewidth, color=color)
    plt.plot([x2, x2], [y1, y2], linewidth=linewidth, color=color)
    plt.plot([x1, x2], [y1, y1], linewidth=linewidth, color=color)
    plt.plot([x1, x2], [y2, y2], linewidth=linewidth, color=color)


def draw_box_in_matrix(img, box, color=(255, 191, 0), linewidth=4):
    """
    Dessine un rectangle au coordonner de box
    Modifie l'objet en paramètre 
    Utiliser deepcopy avant si on veut garder l'objet sans modification
    """

    l = max(int(linewidth/2),1)

    x1, y1, x2, y2 = box
    x2 += x1
    y2 += y1

    img[y1 - l: y1 + l, x1 - l: x2 +l] = color
    img[ y2 - l: y2 + l, x1 - l: x2 +l] = color

    img[y1 - l: y2 + l, x1 - l: x1 + l] = color
    img[y1 - l: y2 + l, x2 - l: x2 + l] = color


def generate_rdm_boxes_from_box(
    rect, delta_move, delta_taille, w_frame, h_frame, nb_box=30
):
    """
    Génère de nouveau rectangle aléatoirement autour du rectangle rect

    Args:
    - rect         : [ x coin haut gauche,
                       y coint haut gauche,
                       largeur,
                       hauteur  ]
    - delta_move   : déplacement maximun en translation normalisé par rapport à la taille du rect (max = 1)
    - delta_taille : redimensionnement maximal normalisé par rapport à la taille du rect (max = 1)
    - w_lim        : image width
    - h_lim        : image height
    - nb_box       : nombre de nouveau rectangle
    """

    x1, y1, w, h = rect

    delta_taille = min(delta_taille, 1)
    delta_move = min(delta_move, 1)

    w_delta = int(w * delta_taille)
    h_delta = int(h * delta_taille)
    x_delta = int(w * delta_move) + int(w_delta/2)
    y_delta = int(h * delta_move) + int(h_delta/2)

    rdm_bboxes = []

    while len(rdm_bboxes) < nb_box:
        n_x1 = x1 + randint(-x_delta, x_delta)
        n_y1 = y1 + randint(-y_delta, y_delta)
        n_w = min(w + randint(-w_delta, w_delta), w_frame-n_x1)
        n_h = min(h + randint(-h_delta, h_delta), h_frame-n_y1)

        if (
            n_w >= w * (1 - delta_taille)
            and n_h >= h * (1 - delta_taille)
            and n_x1 > 0
            and n_y1 > 0
        ):
            rdm_bboxes.append((n_x1, n_y1, n_w, n_h))

    return rdm_bboxes