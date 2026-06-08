"""
Brain Tumor Classifier - Main Training Script.

This script runs the complete training pipeline:
1. Load and prepare the dataset (train/val/test DataLoaders)
2. Create the ViT model (pre-trained on ImageNet)
3. Train the model (training loop + validation loop)
4. Evaluate on test set
5. Generate training visualizations

Usage:
    python train.py

Prerequisites:
    - Dataset in data/processed/ (run prepare_data.py first)
    - Dependencies installed (pip install -r requirements.txt)
"""

import sys
import torch
import numpy as np
from pathlib import Path

from src.utils.config import Config
from src.preprocessing.dataset import create_dataloaders, check_dataset_exists
from src.model.vit_model import create_vit_model
from src.model.trainer import Trainer, evaluate_model
from src.model.visualize_training import TrainingVisualizer


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.

    This ensures that running the script twice with the same data
    produces the same results (as much as possible).
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # Note: DataLoader workers use separate seeds, but this covers the main process


def main():
    """Main training pipeline."""
    config = Config()

    print("=" * 60)
    print("Brain Tumor Classifier - Training Pipeline")
    print("=" * 60)
    print(f"Device: {config.DEVICE}")
    print(f"Model: {config.VIT_MODEL_NAME}")
    print(f"Classes: {config.NUM_CLASSES} ({', '.join(config.CLASS_NAMES)})")
    print(f"Image size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")

    # Print GPU info if available
    if config.DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Mixed precision: Enabled (float16)")
    else:
        print(f"Mixed precision: Disabled (CPU mode)")

    # Set random seeds
    set_seed(config.SEED)

    # ----------------------------------------------------------
    # Step 1: Check dataset
    # ----------------------------------------------------------
    print("\n" + "-" * 40)
    print("Step 1: Checking Dataset")
    print("-" * 40)

    if not check_dataset_exists(config.DATA_PROCESSED / "train"):
        print(f"\nProcessed dataset not found at: {config.DATA_PROCESSED}")
        print("Please run the data preparation script first:")
        print("  python prepare_data.py")
        sys.exit(1)

    print("Dataset found!")

    # ----------------------------------------------------------
    # Step 2: Create DataLoaders
    # ----------------------------------------------------------
    print("\n" + "-" * 40)
    print("Step 2: Creating DataLoaders")
    print("-" * 40)

    train_loader, val_loader, test_loader, dataset_stats = create_dataloaders(
        config,
        use_weighted_sampling=True,
    )

    # Print dataset sizes
    print(f"\nDataset sizes:")
    for split_name, stats in dataset_stats.items():
        print(f"  {split_name}: {stats['total_images']} images")

    # ----------------------------------------------------------
    # Step 3: Create Model
    # ----------------------------------------------------------
    print("\n" + "-" * 40)
    print("Step 3: Creating ViT Model")
    print("-" * 40)

    model = create_vit_model(config)

    # ----------------------------------------------------------
    # Step 4: Train
    # ----------------------------------------------------------
    print("\n" + "-" * 40)
    print("Step 4: Training")
    print("-" * 40)

    trainer = Trainer(model, train_loader, val_loader, config)
    training_results = trainer.train()

    # ----------------------------------------------------------
    # Step 5: Evaluate on Test Set
    # ----------------------------------------------------------
    print("\n" + "-" * 40)
    print("Step 5: Test Set Evaluation")
    print("-" * 40)

    # Load the best model checkpoint for evaluation
    best_model_path = config.MODEL_CHECKPOINTS / "best_model.pth"
    if best_model_path.exists():
        from src.model.trainer import load_trained_model
        best_model = load_trained_model(str(best_model_path), config)
    else:
        best_model = model
        print("Warning: No best model checkpoint found, using current model")

    test_results = evaluate_model(best_model, test_loader, config)

    # ----------------------------------------------------------
    # Step 6: Generate Visualizations
    # ----------------------------------------------------------
    print("\n" + "-" * 40)
    print("Step 6: Generating Visualizations")
    print("-" * 40)

    viz = TrainingVisualizer(trainer.history, config)
    viz.plot_all(
        labels=test_results["labels"],
        predictions=test_results["predictions"],
        output_dir="assets/screenshots",
    )

    # ----------------------------------------------------------
    # Done!
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Training Pipeline Complete!")
    print("=" * 60)
    print(f"\nResults:")
    print(f"  Best Validation Accuracy: {training_results['best_val_acc']:.4f}")
    print(f"  Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"  Test F1 Score: {test_results['f1']:.4f}")
    print(f"\nSaved files:")
    print(f"  Best model: {config.MODEL_CHECKPOINTS / 'best_model.pth'}")
    print(f"  Training history: {config.MODEL_CHECKPOINTS / 'training_history.json'}")
    print(f"  Visualizations: assets/screenshots/")
    print(f"\nNext steps:")
    print(f"  1. View training graphs in assets/screenshots/")
    print(f"  2. Launch the app: streamlit run app/main.py")
    print(f"  3. Use the model for predictions in the app")


if __name__ == "__main__":
    main()
