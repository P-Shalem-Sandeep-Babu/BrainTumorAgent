"""
Model Trainer - Training and Validation Loops.

This file contains the complete training pipeline for the ViT brain tumor
classifier. It handles:
- Training loop (forward pass, loss, backward pass, weight update)
- Validation loop (evaluate on validation set each epoch)
- Metrics computation (accuracy, precision, recall, F1)
- Confusion matrix generation
- Model checkpoint saving (best model, periodic saves)
- Learning rate scheduling (reduce LR when stuck)
- Early stopping (stop if not improving)
- Training history tracking (for plotting graphs)
- Mixed precision training (faster on modern GPUs)

Usage:
    from src.model.trainer import Trainer
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()
"""

import time
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from src.utils.config import Config
from src.utils.logger import setup_logger


# ==============================================================================
# TRAINER CLASS
# ==============================================================================

class Trainer:
    """
    Handles the complete training and validation pipeline.

    Args:
        model: The neural network model (BrainTumorViT).
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        config: Config object with hyperparameters.
    """

    def __init__(self, model, train_loader, val_loader, config: Config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.DEVICE
        self.logger = setup_logger("trainer")

        # ----------------------------------------------------------
        # Loss function
        # ----------------------------------------------------------
        # CrossEntropyLoss is standard for multi-class classification
        # It measures how far predicted probabilities are from true labels
        self.criterion = nn.CrossEntropyLoss()

        # ----------------------------------------------------------
        # Optimizer
        # ----------------------------------------------------------
        # AdamW is Adam with proper weight decay (regularization)
        # It adapts the learning rate for each parameter individually
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # ----------------------------------------------------------
        # Learning rate scheduler
        # ----------------------------------------------------------
        # Reduces learning rate when validation loss stops improving
        # This helps the model make finer adjustments as it converges
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",              # Reduce when metric stops decreasing
            factor=config.LR_SCHEDULER_FACTOR,
            patience=config.LR_SCHEDULER_PATIENCE,
        )

        # ----------------------------------------------------------
        # Mixed precision training (AMP)
        # ----------------------------------------------------------
        # Uses float16 on supported GPUs for ~2x speedup with
        # minimal accuracy loss. Falls back to float32 on CPU.
        self.use_amp = config.DEVICE == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # ----------------------------------------------------------
        # Training history (for plotting graphs)
        # ----------------------------------------------------------
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "learning_rate": [],
            "epoch_time": [],
        }

        # Best validation metrics (for saving best model)
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0

        # Early stopping counter
        self.early_stop_counter = 0

        # Ensure checkpoint directory exists
        config.MODEL_CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================

    def train_one_epoch(self, epoch: int) -> dict:
        """
        Train the model for one epoch (one pass through all training data).

        What happens each epoch:
        1. Loop through batches of images
        2. For each batch:
           a. Forward pass: feed images through model -> get predictions
           b. Compute loss: compare predictions to true labels
           c. Backward pass: compute gradients (how to adjust weights)
           d. Update weights: optimizer adjusts model parameters
        3. Track average loss and accuracy

        Args:
            epoch: Current epoch number (for logging).

        Returns:
            Dictionary with 'loss' and 'accuracy' for this epoch.
        """
        self.model.train()  # Set model to training mode (enables dropout)

        running_loss = 0.0
        all_preds = []
        all_labels = []
        num_batches = 0

        # tqdm shows a progress bar with ETA, speed, and loss
        pbar = tqdm(
            self.train_loader,
            desc=f"  Train Epoch {epoch:>3d}",
            leave=False,           # Clear bar when done (epoch summary prints instead)
            ncols=100,             # Width of the progress bar
            unit="batch",          # Label for the counter
        )

        for images, labels in pbar:
            # Move data to GPU (if available)
            images = images.to(self.device)
            labels = labels.to(self.device)

            # ----------------------------------------------------------
            # Step 1: Forward pass (with optional mixed precision)
            # ----------------------------------------------------------
            # Mixed precision uses float16 for speed, float32 for stability
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            # ----------------------------------------------------------
            # Step 2: Backward pass
            # ----------------------------------------------------------
            # Compute gradients (directions to adjust each weight)
            self.optimizer.zero_grad()  # Clear old gradients

            if self.use_amp:
                # Mixed precision: scale loss to prevent underflow
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            # ----------------------------------------------------------
            # Track metrics
            # ----------------------------------------------------------
            running_loss += loss.item()
            preds = logits.argmax(dim=1)  # Convert logits to predicted class
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            num_batches += 1

            # Update progress bar with current loss
            pbar.set_postfix(loss=f"{running_loss / num_batches:.4f}")

        # Calculate epoch metrics
        epoch_loss = running_loss / max(num_batches, 1)
        epoch_acc = accuracy_score(all_labels, all_preds)

        return {"loss": epoch_loss, "accuracy": epoch_acc}

    # =========================================================================
    # VALIDATION LOOP
    # =========================================================================

    def validate(self, epoch: int) -> dict:
        """
        Evaluate the model on the validation set.

        This is like training but WITHOUT updating weights.
        We want to see how well the model generalizes to unseen data.

        Args:
            epoch: Current epoch number (for logging).

        Returns:
            Dictionary with 'loss', 'accuracy', 'predictions', 'labels'.
        """
        self.model.eval()  # Set model to evaluation mode (disables dropout)

        running_loss = 0.0
        all_preds = []
        all_labels = []
        num_batches = 0

        # Progress bar for validation
        pbar = tqdm(
            self.val_loader,
            desc=f"  Val   Epoch {epoch:>3d}",
            leave=False,
            ncols=100,
            unit="batch",
        )

        # No gradient computation needed for validation (saves memory)
        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        logits = self.model(images)
                        loss = self.criterion(logits, labels)
                else:
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

                # Track metrics
                running_loss += loss.item()
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                num_batches += 1

                pbar.set_postfix(loss=f"{running_loss / num_batches:.4f}")

        epoch_loss = running_loss / max(num_batches, 1)
        epoch_acc = accuracy_score(all_labels, all_preds)

        return {
            "loss": epoch_loss,
            "accuracy": epoch_acc,
            "predictions": all_preds,
            "labels": all_labels,
        }

    # =========================================================================
    # FULL TRAINING PIPELINE
    # =========================================================================

    def train(self) -> dict:
        """
        Run the complete training pipeline.

        This is the main function that orchestrates training:
        1. For each epoch:
           a. Train on training set
           b. Validate on validation set
           c. Update learning rate scheduler
           d. Save checkpoint if model improved
           e. Check early stopping
        2. Save final model and training history

        Returns:
            Dictionary with training history and best metrics.
        """
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)
        print(f"Epochs: {self.config.NUM_EPOCHS}")
        print(f"Batch size: {self.config.BATCH_SIZE_DATA}")
        print(f"Learning rate: {self.config.LEARNING_RATE}")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("=" * 60)

        start_time = time.time()

        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            epoch_start = time.time()
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(f"\nEpoch {epoch}/{self.config.NUM_EPOCHS} "
                  f"(LR: {current_lr:.2e})")
            print("-" * 40)

            # ----------------------------------------------------------
            # Train for one epoch
            # ----------------------------------------------------------
            train_metrics = self.train_one_epoch(epoch)
            print(f"  Train | Loss: {train_metrics['loss']:.4f} "
                  f"| Acc: {train_metrics['accuracy']:.4f}")

            # ----------------------------------------------------------
            # Validate
            # ----------------------------------------------------------
            val_metrics = self.validate(epoch)
            print(f"  Val   | Loss: {val_metrics['loss']:.4f} "
                  f"| Acc: {val_metrics['accuracy']:.4f}")

            # ----------------------------------------------------------
            # Update learning rate scheduler
            # ----------------------------------------------------------
            # If validation loss stopped improving, reduce learning rate
            self.scheduler.step(val_metrics["loss"])

            # ----------------------------------------------------------
            # Save checkpoint if model improved
            # ----------------------------------------------------------
            self._save_checkpoint(epoch, val_metrics)

            # ----------------------------------------------------------
            # Update training history
            # ----------------------------------------------------------
            epoch_time = time.time() - epoch_start
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["learning_rate"].append(current_lr)
            self.history["epoch_time"].append(epoch_time)

            # ----------------------------------------------------------
            # Early stopping check
            # ----------------------------------------------------------
            if self._check_early_stopping(val_metrics["loss"]):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

        # ----------------------------------------------------------
        # Training complete
        # ----------------------------------------------------------
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        # Print final classification report
        if val_metrics.get("predictions") is not None:
            self._print_classification_report(
                val_metrics["labels"],
                val_metrics["predictions"],
            )
            self._print_confusion_matrix(
                val_metrics["labels"],
                val_metrics["predictions"],
            )

        # Save training history to JSON
        self._save_history()

        return {
            "history": self.history,
            "best_val_acc": self.best_val_acc,
            "best_val_loss": self.best_val_loss,
            "total_time": total_time,
        }

    # =========================================================================
    # CHECKPOINT MANAGEMENT
    # =========================================================================

    def _save_checkpoint(self, epoch: int, val_metrics: dict):
        """
        Save model checkpoint.

        Saves:
        - Best model: when validation loss improves
        - Periodic saves: every 5 epochs

        A checkpoint contains everything needed to resume training:
        - Model weights (state_dict)
        - Optimizer state (so learning rate continues correctly)
        - Scheduler state
        - Epoch number
        - Best metrics
        """
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]

        # Save best model (when validation loss improves)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_acc = max(self.best_val_acc, val_acc)
            self.early_stop_counter = 0  # Reset counter

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "config": {
                    "model_name": self.config.VIT_MODEL_NAME,
                    "num_classes": self.config.NUM_CLASSES,
                    "image_size": self.config.IMAGE_SIZE,
                },
            }

            # Save best model
            best_path = self.config.MODEL_CHECKPOINTS / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"  >> New best model saved (val_loss: {val_loss:.4f})")

        # Save periodic checkpoint every 5 epochs
        if epoch % 5 == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            path = self.config.MODEL_CHECKPOINTS / f"checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, path)

    def _check_early_stopping(self, val_loss: float) -> bool:
        """
        Check if training should stop early.

        If validation loss hasn't improved for N consecutive epochs,
        we stop training to prevent overfitting.

        Args:
            val_loss: Current validation loss.

        Returns:
            True if training should stop.
        """
        if val_loss >= self.best_val_loss:
            self.early_stop_counter += 1
            if self.early_stop_counter >= self.config.EARLY_STOPPING_PATIENCE:
                return True
        return False

    # =========================================================================
    # METRICS AND REPORTING
    # =========================================================================

    def compute_metrics(self, labels: list, predictions: list) -> dict:
        """
        Compute classification metrics.

        Args:
            labels: True labels.
            predictions: Model predictions.

        Returns:
            Dictionary with accuracy, precision, recall, F1, confusion matrix.
        """
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="weighted", zero_division=0
        )
        cm = confusion_matrix(labels, predictions, labels=list(range(self.config.NUM_CLASSES)))

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm.tolist(),
        }

    def _print_classification_report(self, labels: list, predictions: list):
        """
        Print a detailed classification report with per-class metrics.

        Shows precision, recall, F1-score, and support for each class.
        """
        report = classification_report(
            labels, predictions,
            target_names=self.config.CLASS_NAMES,
            digits=4,
        )
        print("\nClassification Report:")
        print("-" * 60)
        print(report)

    def _print_confusion_matrix(self, labels: list, predictions: list):
        """
        Print a formatted confusion matrix.

        The confusion matrix shows:
        - Rows: true labels
        - Columns: predicted labels
        - Diagonal: correct predictions (higher is better)
        - Off-diagonal: mistakes
        """
        cm = confusion_matrix(labels, predictions, labels=list(range(self.config.NUM_CLASSES)))

        print("\nConfusion Matrix:")
        print("-" * 50)

        # Header
        header = f"{'True \\ Pred':<12}"
        for name in self.config.CLASS_NAMES:
            header += f"{name:>10}"
        print(header)
        print("-" * 50)

        # Rows
        for i, name in enumerate(self.config.CLASS_NAMES):
            row = f"{name:<12}"
            for j in range(self.config.NUM_CLASSES):
                row += f"{cm[i][j]:>10d}"
            print(row)

        print("-" * 50)

        # Per-class accuracy
        print("\nPer-class Accuracy:")
        for i, name in enumerate(self.config.CLASS_NAMES):
            total = cm[i].sum()
            correct = cm[i][i]
            acc = correct / total if total > 0 else 0
            print(f"  {name:<12}: {correct}/{total} ({acc:.2%})")

    def _save_history(self):
        """Save training history to a JSON file for later visualization."""
        history_path = self.config.MODEL_CHECKPOINTS / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"\nTraining history saved to: {history_path}")


# ==============================================================================
# EVALUATION FUNCTION
# ==============================================================================

def evaluate_model(model, test_loader, config: Config) -> dict:
    """
    Evaluate a trained model on the test set.

    Call this AFTER training is complete to get final metrics.

    Args:
        model: Trained model.
        test_loader: DataLoader for test data.
        config: Config object.

    Returns:
        Dictionary with test metrics and predictions.
    """
    model.eval()
    device = config.DEVICE

    all_preds = []
    all_labels = []
    all_probs = []

    print("\n" + "=" * 50)
    print("Evaluating on Test Set")
    print("=" * 50)

    use_amp = device == "cuda"

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="  Testing", ncols=100, unit="batch"):
            images = images.to(device)
            labels = labels.to(device)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(images)
            else:
                logits = model(images)

            probs = torch.softmax(logits, dim=1)

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(config.NUM_CLASSES)))

    # Print classification report
    print(f"\n{classification_report(all_labels, all_preds, target_names=config.CLASS_NAMES)}")

    # Print confusion matrix
    print("Confusion Matrix:")
    print("-" * 50)
    header = f"{'True \\ Pred':<12}"
    for name in config.CLASS_NAMES:
        header += f"{name:>10}"
    print(header)
    print("-" * 50)
    for i, name in enumerate(config.CLASS_NAMES):
        row = f"{name:<12}"
        for j in range(config.NUM_CLASSES):
            row += f"{cm[i][j]:>10d}"
        print(row)

    # Per-class accuracy
    print("\nPer-class Accuracy:")
    for i, name in enumerate(config.CLASS_NAMES):
        total = cm[i].sum()
        correct = cm[i][i]
        acc = correct / total if total > 0 else 0
        print(f"  {name:<12}: {correct}/{total} ({acc:.2%})")

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": np.array(all_probs).tolist(),
    }

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")

    return results


# ==============================================================================
# LOAD TRAINED MODEL
# ==============================================================================

def load_trained_model(checkpoint_path: str, config: Config):
    """
    Load a trained model from a checkpoint file.

    Args:
        checkpoint_path: Path to the .pth checkpoint file.
        config: Config object.

    Returns:
        Loaded model in evaluation mode.
    """
    from src.model.vit_model import create_vit_model

    model = create_vit_model(config)

    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model from: {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Val Loss: {checkpoint.get('val_loss', 'unknown')}")
    print(f"  Val Acc: {checkpoint.get('val_acc', 'unknown')}")

    return model
