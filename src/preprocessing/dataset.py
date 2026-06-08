"""
Brain Tumor MRI Dataset - Loading and Preprocessing.

This file contains:
1. BrainTumorDataset - PyTorch Dataset class for loading MRI images
2. Dataset splitting logic (train/validation/test)
3. DataLoader creation functions
4. Dataset statistics computation

Expected folder structure:
    data/raw/
        glioma/       - Images of glioma tumors
        meningioma/   - Images of meningioma tumors
        notumor/      - Images of healthy brains
        pituitary/    - Images of pituitary tumors

Each subfolder contains .jpg/.jpeg/.png MRI scan images.
"""

import os
import shutil
import random
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from src.utils.config import Config
from src.preprocessing.transforms import get_train_augmentations, get_val_augmentations


# ==============================================================================
# DATASET CLASS
# ==============================================================================

class BrainTumorDataset(Dataset):
    """
    PyTorch Dataset for brain tumor MRI images.

    This class handles:
    - Scanning directories for images
    - Loading images from disk
    - Applying transforms (augmentations for training, normalization for testing)
    - Returning (image, label) pairs for the model

    Args:
        data_dir: Path to the image directory containing class subfolders.
        transform: Albumentations transform pipeline to apply to each image.
        image_size: Size to resize images to (default: 224 for ViT).

    Usage:
        dataset = BrainTumorDataset("data/raw", transform=get_train_augmentations())
        image, label = dataset[0]  # Get first image and its label
    """

    # Maps folder names to numeric labels
    # The model learns to output these numbers, which we map back to names
    CLASS_MAP = {
        "glioma": 0,
        "meningioma": 1,
        "notumor": 2,
        "pituitary": 3,
    }

    # Reverse mapping: number back to name (for displaying predictions)
    LABEL_MAP = {v: k for k, v in CLASS_MAP.items()}

    def __init__(self, data_dir: str, transform=None, image_size: int = 224):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_size = image_size

        # List of (image_path, label_index) tuples
        self.samples = []

        # Scan the directory to find all images
        self._load_samples()

        # Count images per class for statistics
        self.class_counts = Counter(label for _, label in self.samples)

    def _load_samples(self):
        """
        Walk through the data directory and collect all image paths with their labels.

        Expected structure:
            data_dir/
                glioma/image1.jpg
                glioma/image2.jpg
                meningioma/image1.jpg
                ...
        """
        # Supported image file extensions
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

        for class_name, class_idx in self.CLASS_MAP.items():
            class_dir = self.data_dir / class_name

            # Skip if the class directory doesn't exist
            if not class_dir.exists():
                print(f"  Warning: Directory not found - {class_dir}")
                continue

            # Find all image files in this class directory
            image_count = 0
            for img_file in sorted(class_dir.iterdir()):
                if img_file.suffix.lower() in valid_extensions:
                    self.samples.append((img_file, class_idx))
                    image_count += 1

            print(f"  Found {image_count:>5d} images in '{class_name}'")

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load and return one image with its label.

        This method is called by PyTorch's DataLoader when it needs a sample.

        Args:
            idx: Index of the image to load.

        Returns:
            image: PyTorch tensor of shape (3, 224, 224) - the MRI scan.
            label: Integer (0-3) - the tumor class.
        """
        # Get the file path and label for this index
        img_path, label = self.samples[idx]

        # Load the image and convert to RGB (some MRI scans are grayscale)
        image = Image.open(img_path).convert("RGB")

        # Convert PIL Image to numpy array (required by albumentations)
        image = np.array(image)

        # Apply transforms (augmentation + normalization + to tensor)
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label

    def get_class_name(self, label_idx: int) -> str:
        """Convert a numeric label back to the class name."""
        return self.LABEL_MAP.get(label_idx, "unknown")

    def get_statistics(self) -> dict:
        """
        Compute dataset statistics.

        Returns:
            Dictionary with dataset statistics (total count, per-class counts, etc.)
        """
        stats = {
            "total_images": len(self.samples),
            "num_classes": len(self.CLASS_MAP),
            "class_names": list(self.CLASS_MAP.keys()),
            "class_counts": dict(self.class_counts),
            "class_distribution": {},
        }

        # Calculate percentage for each class
        total = stats["total_images"]
        for class_name, class_idx in self.CLASS_MAP.items():
            count = self.class_counts.get(class_idx, 0)
            percentage = (count / total * 100) if total > 0 else 0
            stats["class_distribution"][class_name] = {
                "count": count,
                "percentage": round(percentage, 1),
            }

        return stats


# ==============================================================================
# DATASET SPLITTING
# ==============================================================================

def split_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """
    Split the raw dataset into train, validation, and test sets.

    Why split?
    - Training set (70%): The model learns from these images
    - Validation set (15%): Used during training to check if model is overfitting
    - Test set (15%): Never seen during training - used for final evaluation

    The split is done per-class to maintain the same class distribution
    across all splits (stratified split).

    Args:
        source_dir: Path to the raw dataset (data/raw/).
        output_dir: Path to save the split dataset (data/processed/).
        train_ratio: Fraction of data for training (default: 70%).
        val_ratio: Fraction of data for validation (default: 15%).
        test_ratio: Fraction of data for testing (default: 15%).
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with split statistics.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    # Validate ratios add up to 1
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    # Set random seed for reproducibility
    random.seed(seed)

    # Valid image extensions
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    split_stats = {}

    print("=" * 60)
    print("Splitting Dataset")
    print("=" * 60)
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Split ratios: Train={train_ratio:.0%}, Val={val_ratio:.0%}, Test={test_ratio:.0%}")
    print(f"Random seed: {seed}")
    print("-" * 60)

    # Process each class folder
    for class_name in BrainTumorDataset.CLASS_MAP.keys():
        class_source = source_dir / class_name

        if not class_source.exists():
            print(f"  Skipping '{class_name}' - directory not found")
            continue

        # Collect all image files
        images = [
            f for f in sorted(class_source.iterdir())
            if f.suffix.lower() in valid_extensions
        ]

        # Shuffle randomly
        random.shuffle(images)

        # Calculate split indices
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        # Split the list
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]

        # Copy files to split directories
        for split_name, split_images in [
            ("train", train_images),
            ("val", val_images),
            ("test", test_images),
        ]:
            split_dir = output_dir / split_name / class_name
            split_dir.mkdir(parents=True, exist_ok=True)

            for img_path in split_images:
                dest = split_dir / img_path.name
                if not dest.exists():
                    shutil.copy2(img_path, dest)

        # Store statistics
        split_stats[class_name] = {
            "total": n_total,
            "train": len(train_images),
            "val": len(val_images),
            "test": len(test_images),
        }

        print(f"  {class_name:>12s}: {n_total:>5d} total -> "
              f"{len(train_images):>5d} train, "
              f"{len(val_images):>5d} val, "
              f"{len(test_images):>5d} test")

    print("-" * 60)
    print("Dataset split complete!")
    print("=" * 60)

    return split_stats


# ==============================================================================
# DATALOADER CREATION
# ==============================================================================

def create_dataloaders(
    config: Config,
    use_weighted_sampling: bool = True,
) -> tuple:
    """
    Create PyTorch DataLoaders for train, validation, and test sets.

    DataLoaders handle:
    - Batching: Group multiple images together for efficient GPU processing
    - Shuffling: Randomize order each epoch (training only)
    - Sampling: Optionally balance classes using weighted sampling
    - Parallel loading: Load images in parallel workers

    Args:
        config: Config object with paths and hyperparameters.
        use_weighted_sampling: If True, oversample underrepresented classes
                               during training to balance the dataset.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, dataset_stats).
    """
    # Get transform pipelines
    train_transform = get_train_augmentations(config.IMAGE_SIZE)
    val_transform = get_val_augmentations(config.IMAGE_SIZE)

    # Create datasets
    print("\nLoading training set:")
    train_dataset = BrainTumorDataset(
        config.DATA_PROCESSED / "train",
        transform=train_transform,
        image_size=config.IMAGE_SIZE,
    )

    print("\nLoading validation set:")
    val_dataset = BrainTumorDataset(
        config.DATA_PROCESSED / "val",
        transform=val_transform,
        image_size=config.IMAGE_SIZE,
    )

    print("\nLoading test set:")
    test_dataset = BrainTumorDataset(
        config.DATA_PROCESSED / "test",
        transform=val_transform,
        image_size=config.IMAGE_SIZE,
    )

    # Collect statistics
    dataset_stats = {
        "train": train_dataset.get_statistics(),
        "val": val_dataset.get_statistics(),
        "test": test_dataset.get_statistics(),
    }

    # Optionally use weighted sampling for imbalanced classes
    # This makes the model see equal numbers of each class during training
    sampler = None
    shuffle = True
    if use_weighted_sampling and len(train_dataset) > 0:
        # Calculate weight for each sample based on class frequency
        class_counts = train_dataset.class_counts
        total = sum(class_counts.values())
        class_weights = {
            cls: total / count for cls, count in class_counts.items()
        }
        # Assign a weight to each sample in the dataset
        sample_weights = [class_weights[label] for _, label in train_dataset.samples]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False  # Can't use shuffle with sampler
        print("\nUsing weighted sampling to balance class distribution")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE_DATA,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,            # Speeds up CPU-to-GPU transfer
        drop_last=True,             # Drop last incomplete batch for consistent training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE_DATA,
        shuffle=False,              # No shuffling for validation
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE_DATA,
        shuffle=False,              # No shuffling for testing
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, dataset_stats


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def print_dataset_statistics(stats: dict):
    """
    Pretty-print dataset statistics.

    Args:
        stats: Dictionary returned by dataset.get_statistics().
    """
    print("\n" + "=" * 50)
    print("Dataset Statistics")
    print("=" * 50)
    print(f"Total images: {stats['total_images']}")
    print(f"Number of classes: {stats['num_classes']}")
    print("-" * 50)
    print(f"{'Class':<15} {'Count':>8} {'Percentage':>12}")
    print("-" * 50)
    for class_name in stats["class_names"]:
        dist = stats["class_distribution"].get(class_name, {"count": 0, "percentage": 0})
        print(f"{class_name:<15} {dist['count']:>8d} {dist['percentage']:>11.1f}%")
    print("=" * 50)


def check_dataset_exists(data_dir: str) -> bool:
    """
    Check if the dataset directory exists and contains images.

    Args:
        data_dir: Path to the dataset directory.

    Returns:
        True if at least one class folder with images exists.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return False

    for class_name in BrainTumorDataset.CLASS_MAP.keys():
        class_dir = data_dir / class_name
        if class_dir.exists() and any(class_dir.iterdir()):
            return True

    return False


# ==============================================================================
# MAIN - Run this file directly to test dataset loading
# ==============================================================================

if __name__ == "__main__":
    config = Config()

    print("Brain Tumor MRI Dataset Loader")
    print("=" * 50)

    # Check if raw dataset exists
    if not check_dataset_exists(config.DATA_RAW):
        print(f"\nDataset not found at: {config.DATA_RAW}")
        print("\nTo use this script:")
        print("1. Download a brain tumor MRI dataset (e.g., from Kaggle)")
        print("2. Place the class folders in data/raw/:")
        print("   data/raw/glioma/")
        print("   data/raw/meningioma/")
        print("   data/raw/notumor/")
        print("   data/raw/pituitary/")
        print("\nThen run: python -m src.preprocessing.dataset")
    else:
        # Load and display statistics
        print("\nLoading raw dataset...")
        transform = get_val_augmentations(config.IMAGE_SIZE)
        dataset = BrainTumorDataset(config.DATA_RAW, transform=transform)
        print_dataset_statistics(dataset.get_statistics())

        # Show a sample
        if len(dataset) > 0:
            image, label = dataset[0]
            print(f"\nSample image shape: {image.shape}")
            print(f"Sample label: {label} ({dataset.get_class_name(label)})")
            print(f"Image dtype: {image.dtype}")
            print(f"Pixel range: [{image.min():.3f}, {image.max():.3f}]")
