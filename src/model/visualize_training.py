"""
Training Visualization - Graphs and Charts.

Generates publication-quality plots for:
1. Training & validation loss curves
2. Training & validation accuracy curves
3. Learning rate schedule
4. Confusion matrix heatmap
5. Per-class accuracy bar chart
6. Combined dashboard

Usage:
    from src.model.visualize_training import TrainingVisualizer
    viz = TrainingVisualizer(history, config)
    viz.plot_all()
"""

import json
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix

from src.utils.config import Config


class TrainingVisualizer:
    """
    Creates training visualization plots.

    Args:
        history: Dictionary with training history (loss, accuracy, lr, etc.).
        config: Config object.
    """

    def __init__(self, history: dict, config: Config):
        self.history = history
        self.config = config
        self.class_names = config.CLASS_NAMES

    # =========================================================================
    # LOSS CURVES
    # =========================================================================

    def plot_loss_curves(self, save_path: str = None):
        """
        Plot training and validation loss over epochs.

        What to look for:
        - Both losses should decrease over time
        - If train loss << val loss: overfitting (model memorizing training data)
        - If both are high: underfitting (model not learning enough)
        - Gap between curves should be small
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        epochs = range(1, len(self.history["train_loss"]) + 1)

        ax.plot(epochs, self.history["train_loss"], "b-o",
                label="Training Loss", markersize=4, linewidth=2)
        ax.plot(epochs, self.history["val_loss"], "r-o",
                label="Validation Loss", markersize=4, linewidth=2)

        # Mark the best validation loss
        best_epoch = np.argmin(self.history["val_loss"]) + 1
        best_loss = min(self.history["val_loss"])
        ax.axvline(x=best_epoch, color="green", linestyle="--", alpha=0.5)
        ax.annotate(f"Best: {best_loss:.4f}\n(Epoch {best_epoch})",
                    xy=(best_epoch, best_loss),
                    xytext=(best_epoch + 1, best_loss + 0.1),
                    arrowprops=dict(arrowstyle="->", color="green"),
                    fontsize=10, color="green")

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {save_path}")
        return fig

    # =========================================================================
    # ACCURACY CURVES
    # =========================================================================

    def plot_accuracy_curves(self, save_path: str = None):
        """
        Plot training and validation accuracy over epochs.

        What to look for:
        - Both should increase over time
        - Validation accuracy should stabilize near training accuracy
        - Large gap = overfitting
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        epochs = range(1, len(self.history["train_acc"]) + 1)

        ax.plot(epochs, self.history["train_acc"], "b-o",
                label="Training Accuracy", markersize=4, linewidth=2)
        ax.plot(epochs, self.history["val_acc"], "r-o",
                label="Validation Accuracy", markersize=4, linewidth=2)

        # Mark the best validation accuracy
        best_epoch = np.argmax(self.history["val_acc"]) + 1
        best_acc = max(self.history["val_acc"])
        ax.axvline(x=best_epoch, color="green", linestyle="--", alpha=0.5)
        ax.annotate(f"Best: {best_acc:.4f}\n(Epoch {best_epoch})",
                    xy=(best_epoch, best_acc),
                    xytext=(best_epoch + 1, best_acc - 0.05),
                    arrowprops=dict(arrowstyle="->", color="green"),
                    fontsize=10, color="green")

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("Training and Validation Accuracy", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {save_path}")
        return fig

    # =========================================================================
    # LEARNING RATE SCHEDULE
    # =========================================================================

    def plot_learning_rate(self, save_path: str = None):
        """
        Plot the learning rate over epochs.

        Shows how the LR scheduler reduces learning rate when
        validation loss plateaus.
        """
        fig, ax = plt.subplots(figsize=(10, 4))

        epochs = range(1, len(self.history["learning_rate"]) + 1)

        ax.plot(epochs, self.history["learning_rate"], "g-o",
                markersize=4, linewidth=2)

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Learning Rate", fontsize=12)
        ax.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {save_path}")
        return fig

    # =========================================================================
    # CONFUSION MATRIX
    # =========================================================================

    def plot_confusion_matrix(self, labels: list, predictions: list,
                               save_path: str = None):
        """
        Plot a confusion matrix heatmap.

        Reading the matrix:
        - Rows = true labels (what the image actually is)
        - Columns = predicted labels (what the model thinks it is)
        - Diagonal (top-left to bottom-right) = correct predictions
        - Off-diagonal = mistakes
        - Darker diagonal = better model
        """
        cm = confusion_matrix(labels, predictions,
                              labels=list(range(self.config.NUM_CLASSES)))

        fig, ax = plt.subplots(figsize=(8, 7))

        # Plot heatmap
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Add labels
        ax.set_xticks(range(self.config.NUM_CLASSES))
        ax.set_yticks(range(self.config.NUM_CLASSES))
        ax.set_xticklabels(self.class_names, rotation=45, ha="right", fontsize=11)
        ax.set_yticklabels(self.class_names, fontsize=11)

        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

        # Add numbers on each cell
        thresh = cm.max() / 2.0
        for i in range(self.config.NUM_CLASSES):
            for j in range(self.config.NUM_CLASSES):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=14, fontweight="bold")

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {save_path}")
        return fig

    # =========================================================================
    # PER-CLASS ACCURACY
    # =========================================================================

    def plot_per_class_accuracy(self, labels: list, predictions: list,
                                 save_path: str = None):
        """
        Bar chart showing accuracy for each tumor class.

        Helps identify which classes the model struggles with.
        """
        cm = confusion_matrix(labels, predictions,
                              labels=list(range(self.config.NUM_CLASSES)))

        fig, ax = plt.subplots(figsize=(8, 5))

        accuracies = []
        for i in range(self.config.NUM_CLASSES):
            total = cm[i].sum()
            correct = cm[i][i]
            acc = correct / total if total > 0 else 0
            accuracies.append(acc)

        colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
        bars = ax.bar(self.class_names, accuracies, color=colors,
                      edgecolor="black", linewidth=0.8)

        # Add percentage labels
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{acc:.1%}",
                    ha="center", va="bottom",
                    fontweight="bold", fontsize=11)

        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("Per-Class Accuracy", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {save_path}")
        return fig

    # =========================================================================
    # COMBINED DASHBOARD
    # =========================================================================

    def plot_dashboard(self, labels: list = None, predictions: list = None,
                        save_path: str = None):
        """
        Create a comprehensive dashboard with all training plots.

        Includes: loss curves, accuracy curves, learning rate, and
        (if labels/predictions provided) confusion matrix.
        """
        has_eval = labels is not None and predictions is not None
        n_plots = 3 + (1 if has_eval else 0)

        fig = plt.figure(figsize=(16, 5 * ((n_plots + 1) // 2)))
        gs = gridspec.GridSpec((n_plots + 1) // 2, 2, hspace=0.4, wspace=0.3)

        epochs = range(1, len(self.history["train_loss"]) + 1)

        # --- Loss Curves ---
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(epochs, self.history["train_loss"], "b-o", label="Train", markersize=3)
        ax1.plot(epochs, self.history["val_loss"], "r-o", label="Val", markersize=3)
        ax1.set_title("Loss", fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # --- Accuracy Curves ---
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(epochs, self.history["train_acc"], "b-o", label="Train", markersize=3)
        ax2.plot(epochs, self.history["val_acc"], "r-o", label="Val", markersize=3)
        ax2.set_title("Accuracy", fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)

        # --- Learning Rate ---
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(epochs, self.history["learning_rate"], "g-o", markersize=3)
        ax3.set_title("Learning Rate", fontweight="bold")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("LR")
        ax3.grid(True, alpha=0.3)
        ax3.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

        # --- Confusion Matrix (if available) ---
        if has_eval:
            ax4 = fig.add_subplot(gs[1, 1])
            cm = confusion_matrix(labels, predictions,
                                  labels=list(range(self.config.NUM_CLASSES)))
            im = ax4.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
            ax4.set_xticks(range(self.config.NUM_CLASSES))
            ax4.set_yticks(range(self.config.NUM_CLASSES))
            ax4.set_xticklabels(self.class_names, rotation=45, ha="right", fontsize=9)
            ax4.set_yticklabels(self.class_names, fontsize=9)
            ax4.set_xlabel("Predicted")
            ax4.set_ylabel("True")
            ax4.set_title("Confusion Matrix", fontweight="bold")
            thresh = cm.max() / 2.0
            for i in range(self.config.NUM_CLASSES):
                for j in range(self.config.NUM_CLASSES):
                    ax4.text(j, i, str(cm[i, j]),
                             ha="center", va="center",
                             color="white" if cm[i, j] > thresh else "black",
                             fontsize=11, fontweight="bold")

        fig.suptitle("Training Dashboard", fontsize=16, fontweight="bold", y=1.02)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {save_path}")
        return fig

    # =========================================================================
    # GENERATE ALL PLOTS
    # =========================================================================

    def plot_all(self, labels: list = None, predictions: list = None,
                  output_dir: str = "assets/screenshots"):
        """
        Generate and save all training visualization plots.

        Args:
            labels: True labels (for confusion matrix).
            predictions: Model predictions (for confusion matrix).
            output_dir: Directory to save plots.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\nGenerating training visualizations...")

        self.plot_loss_curves(save_path=str(output_dir / "loss_curves.png"))
        self.plot_accuracy_curves(save_path=str(output_dir / "accuracy_curves.png"))
        self.plot_learning_rate(save_path=str(output_dir / "learning_rate.png"))
        self.plot_dashboard(labels, predictions,
                           save_path=str(output_dir / "training_dashboard.png"))

        if labels is not None and predictions is not None:
            self.plot_confusion_matrix(labels, predictions,
                                       save_path=str(output_dir / "confusion_matrix.png"))
            self.plot_per_class_accuracy(labels, predictions,
                                          save_path=str(output_dir / "per_class_accuracy.png"))

        print("All visualizations saved!")


# ==============================================================================
# LOAD HISTORY FROM FILE
# ==============================================================================

def load_training_history(history_path: str) -> dict:
    """
    Load training history from a JSON file.

    Args:
        history_path: Path to training_history.json.

    Returns:
        History dictionary.
    """
    with open(history_path, "r") as f:
        return json.load(f)
