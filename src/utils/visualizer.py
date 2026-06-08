"""
Visualization utilities.

Functions for plotting training curves, confusion matrices,
and MRI image grids. Used by both notebooks and the Streamlit app.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_training_rewards(rewards: list, window: int = 10):
    """
    Plot episode rewards over time with a rolling average.

    Args:
        rewards: List of episode rewards.
        window: Window size for rolling average.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rewards, alpha=0.3, label="Episode Reward")

    # Rolling average
    if len(rewards) >= window:
        rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), rolling, label=f"Rolling Avg ({window})")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Training Rewards Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_confusion_matrix(cm: np.ndarray, class_names: list):
    """
    Plot a confusion matrix heatmap.

    Args:
        cm: Confusion matrix (numpy array).
        class_names: List of class names for axis labels.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    return fig


def plot_image_grid(images: list, labels: list, predictions: list = None, n: int = 8):
    """
    Display a grid of MRI images with labels.

    Args:
        images: List of images (numpy arrays or tensors).
        labels: List of ground truth labels.
        predictions: Optional list of predicted labels.
        n: Number of images to display.
    """
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).flatten()

    for i in range(n):
        img = images[i]
        if hasattr(img, "numpy"):
            img = img.numpy()
        if img.ndim == 3 and img.shape[0] == 3:
            img = img.transpose(1, 2, 0)

        axes[i].imshow(img, cmap="gray" if img.ndim == 2 else None)
        title = f"True: {labels[i]}"
        if predictions is not None:
            title += f"\nPred: {predictions[i]}"
        axes[i].set_title(title, fontsize=9)
        axes[i].axis("off")

    # Hide unused subplots
    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    return fig
