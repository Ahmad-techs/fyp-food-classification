# src/confusion_matrix_utils.py
# Generates publication-quality confusion matrices
# Professor specifically asked for these — include in paper

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os


def plot_confusion_matrix(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           class_names: list,
                           title: str,
                           save_path: str,
                           figsize_per_class: float = 0.5,
                           max_display: int = 20):
    """
    Generates and saves a normalised confusion matrix (percentages).

    If there are more than max_display classes, displays only the
    top classes by frequency to keep the figure readable.

    Args:
        y_true:     ground truth labels (numpy array)
        y_pred:     predicted labels (numpy array)
        class_names: list of class name strings
        title:      chart title (e.g. 'CoAtNet-0 on Food-11')
        save_path:  full file path to save PNG (300 dpi)
        figsize_per_class: inches per class for auto-sizing
        max_display: maximum classes to show (trims if more)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n = len(class_names)

    # If too many classes, keep only the most frequent ones
    if n > max_display:
        counts    = np.bincount(y_true)
        top_idxs  = np.argsort(counts)[::-1][:max_display]
        top_idxs  = sorted(top_idxs)
        mask      = np.isin(y_true, top_idxs)
        y_true    = y_true[mask]
        y_pred    = y_pred[mask]
        # Remap labels to 0..max_display-1
        remap     = {old: new for new, old in enumerate(top_idxs)}
        y_true    = np.array([remap[v] for v in y_true])
        y_pred    = np.array([remap.get(v, max_display) for v in y_pred])
        class_names = [class_names[i] for i in top_idxs]
        n = max_display
        title = title + f' (top {max_display} classes)'

    cm = confusion_matrix(y_true, y_pred)

    # Normalise to percentages (row = true class, col = predicted)
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums * 100

    # Auto-size figure based on number of classes
    size = max(8, n * figsize_per_class)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))

    # Heatmap — show percentage values, format depends on count
    fmt = '.0f' if n <= 15 else '.0f'
    annot = n <= 20   # only annotate small matrices (readable)
    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Recall (%)'},
        linewidths=0.3 if n <= 20 else 0.0
    )

    ax.set_xlabel('Predicted Class', fontsize=11, labelpad=10)
    ax.set_ylabel('True Class',      fontsize=11, labelpad=10)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)

    fontsize = max(6, 10 - n // 10)
    plt.xticks(rotation=35, ha='right', fontsize=fontsize)
    plt.yticks(rotation=0,              fontsize=fontsize)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Confusion matrix saved: {save_path}')
    return cm, cm_norm


def print_per_class_accuracy(y_true, y_pred, class_names):
    """Prints per-class recall — shows exactly where the model fails."""
    cm = confusion_matrix(y_true, y_pred)
    print(f'\n  Per-class accuracy (recall):')
    for i, name in enumerate(class_names):
        if cm[i].sum() > 0:
            acc = cm[i, i] / cm[i].sum() * 100
            bar = '█' * int(acc / 5)
            print(f'    {name:25s}: {acc:6.1f}%  {bar}')