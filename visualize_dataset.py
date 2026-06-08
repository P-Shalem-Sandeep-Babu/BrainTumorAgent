"""
Dataset Visualization Script.

Displays sample MRI images from each class, class distribution charts,
and dataset statistics. Useful for understanding the data before training.

Usage:
    python visualize_dataset.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter

from src.utils.config import Config
from src.preprocessing.dataset import BrainTumorDataset, check_dataset_exists
from src.preprocessing.transforms import get_val_augmentations, get_train_augmentations


def plot_class_distribution(dataset: BrainTumorDataset, title: str = "Class Distribution"):
    """
    Create a bar chart showing how many images are in each class.

    Args:
        dataset: BrainTumorDataset instance.
        title: Chart title.

    Returns:
        matplotlib Figure object.
    """
    stats = dataset.get_statistics()
    class_names = stats["class_names"]
    counts = [stats["class_distribution"][c]["count"] for c in class_names]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Color palette for each class
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    bars = ax.bar(class_names, counts, color=colors, edgecolor="black", linewidth=0.8)

    # Add count labels on top of each bar
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,   # x position (center of bar)
            bar.get_height() + max(counts) * 0.02, # y position (slightly above bar)
            str(count),
            ha="center", va="bottom",
            fontweight="bold", fontsize=11,
        )

    ax.set_xlabel("Tumor Class", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


def plot_sample_images(
    dataset: BrainTumorDataset,
    n_per_class: int = 4,
    title: str = "Sample MRI Images",
):
    """
    Display a grid of sample MRI images, organized by class.

    Each row shows images from one class:
    - Row 1: Glioma
    - Row 2: Meningioma
    - Row 3: No Tumor
    - Row 4: Pituitary

    Args:
        dataset: BrainTumorDataset instance.
        n_per_class: Number of sample images per class.
        title: Overall figure title.

    Returns:
        matplotlib Figure object.
    """
    class_names = list(BrainTumorDataset.CLASS_MAP.keys())
    n_classes = len(class_names)

    # Create figure with subplots
    fig, axes = plt.subplots(
        n_classes, n_per_class,
        figsize=(n_per_class * 3, n_classes * 3),
    )

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)

    for row_idx, class_name in enumerate(class_names):
        # Find all indices for this class
        class_label = BrainTumorDataset.CLASS_MAP[class_name]
        class_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == class_label]

        # Pick random samples
        if len(class_indices) >= n_per_class:
            selected = np.random.choice(class_indices, n_per_class, replace=False)
        else:
            selected = class_indices + [class_indices[0]] * (n_per_class - len(class_indices))

        for col_idx, sample_idx in enumerate(selected):
            ax = axes[row_idx, col_idx]

            # Load image without transforms (for display)
            img_path, _ = dataset.samples[sample_idx]
            from PIL import Image
            img = Image.open(img_path).convert("RGB")
            img = img.resize((224, 224))

            ax.imshow(img)
            ax.axis("off")

            # Add class name label on the first column
            if col_idx == 0:
                ax.set_ylabel(
                    class_name.upper(),
                    fontsize=11, fontweight="bold",
                    rotation=90, labelpad=10,
                )

    plt.tight_layout()
    return fig


def plot_image_dimensions(dataset: BrainTumorDataset, n_samples: int = 200):
    """
    Plot the distribution of image widths and heights in the dataset.

    Useful to check if images have consistent sizes or vary a lot.

    Args:
        dataset: BrainTumorDataset instance.
        n_samples: Number of images to sample for dimension check.

    Returns:
        matplotlib Figure object.
    """
    widths = []
    heights = []

    # Sample a subset of images
    indices = np.random.choice(
        len(dataset),
        size=min(n_samples, len(dataset)),
        replace=False,
    )

    for idx in indices:
        img_path, _ = dataset.samples[idx]
        from PIL import Image
        img = Image.open(img_path)
        widths.append(img.width)
        heights.append(img.height)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.hist(widths, bins=30, color="#3498db", edgecolor="black", alpha=0.7)
    ax1.set_xlabel("Width (pixels)")
    ax1.set_ylabel("Count")
    ax1.set_title("Image Width Distribution")
    ax1.axvline(np.mean(widths), color="red", linestyle="--", label=f"Mean: {np.mean(widths):.0f}")
    ax1.legend()

    ax2.hist(heights, bins=30, color="#e74c3c", edgecolor="black", alpha=0.7)
    ax2.set_xlabel("Height (pixels)")
    ax2.set_ylabel("Count")
    ax2.set_title("Image Height Distribution")
    ax2.axvline(np.mean(heights), color="red", linestyle="--", label=f"Mean: {np.mean(heights):.0f}")
    ax2.legend()

    plt.tight_layout()
    return fig


def plot_augmentation_comparison(
    dataset: BrainTumorDataset,
    n_samples: int = 3,
):
    """
    Show the same images before and after augmentation.

    This helps visualize what the augmentation transforms do.

    Args:
        dataset: BrainTumorDataset instance.
        n_samples: Number of images to compare.

    Returns:
        matplotlib Figure object.
    """
    train_transform = get_train_augmentations(224)
    val_transform = get_val_augmentations(224)

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(
        "Original vs. Augmented Images",
        fontsize=14, fontweight="bold",
    )

    for i in range(n_samples):
        # Pick a random image
        idx = np.random.randint(0, len(dataset))
        img_path, label = dataset.samples[idx]
        class_name = dataset.get_class_name(label)

        # Load original image
        from PIL import Image
        original = Image.open(img_path).convert("RGB")
        original_resized = original.resize((224, 224))
        img_np = np.array(original)

        # Apply transforms
        val_result = val_transform(image=img_np)["image"]
        train_result = train_transform(image=img_np)["image"]

        # Convert tensors back to displayable format
        # Undo normalization for display
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        def unnormalize(tensor_img):
            img = tensor_img.numpy().transpose(1, 2, 0)  # CHW -> HWC
            img = img * std + mean  # Undo normalization
            img = np.clip(img, 0, 1)  # Clip to valid range
            return img

        # Plot
        axes[i, 0].imshow(original_resized)
        axes[i, 0].set_title(f"Original\n({class_name})")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(unnormalize(val_result))
        axes[i, 1].set_title("Normalized\n(no augmentation)")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(unnormalize(train_result))
        axes[i, 2].set_title("Augmented\n(training transform)")
        axes[i, 2].axis("off")

    plt.tight_layout()
    return fig


def print_dataset_summary(dataset: BrainTumorDataset, split_name: str):
    """Print a text summary of dataset statistics."""
    stats = dataset.get_statistics()

    print(f"\n{'=' * 50}")
    print(f"{split_name.upper()} SET")
    print(f"{'=' * 50}")
    print(f"Total images: {stats['total_images']}")
    print(f"Number of classes: {stats['num_classes']}")
    print(f"{'-' * 50}")
    print(f"{'Class':<15} {'Count':>8} {'Percentage':>12}")
    print(f"{'-' * 50}")

    for class_name in stats["class_names"]:
        dist = stats["class_distribution"].get(class_name, {"count": 0, "percentage": 0})
        bar = "#" * int(dist["percentage"] / 2)
        print(f"{class_name:<15} {dist['count']:>8d} {dist['percentage']:>10.1f}%  {bar}")

    print(f"{'=' * 50}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Run all visualizations."""
    config = Config()

    print("=" * 60)
    print("Brain Tumor Agent - Dataset Visualization")
    print("=" * 60)

    # Check which dataset is available
    data_dir = config.DATA_RAW
    split_name = "raw"

    if check_dataset_exists(config.DATA_PROCESSED):
        data_dir = config.DATA_PROCESSED
        split_name = "processed"

        # If processed exists, visualize each split
        for split in ["train", "val", "test"]:
            split_dir = config.DATA_PROCESSED / split
            if split_dir.exists():
                print(f"\nLoading {split} set...")
                transform = get_val_augmentations(config.IMAGE_SIZE)
                ds = BrainTumorDataset(split_dir, transform=transform, image_size=config.IMAGE_SIZE)
                print_dataset_summary(ds, split)

        # Use training set for visualizations
        train_dir = config.DATA_PROCESSED / "train"
        if train_dir.exists():
            data_dir = train_dir
            split_name = "train"

    elif not check_dataset_exists(config.DATA_RAW):
        print(f"\nNo dataset found!")
        print(f"  Checked: {config.DATA_RAW}")
        print(f"  Checked: {config.DATA_PROCESSED}")
        print("\nPlease download a brain tumor MRI dataset and place it in data/raw/")
        sys.exit(1)

    # Load dataset
    print(f"\nLoading {split_name} dataset from: {data_dir}")
    transform = get_val_augmentations(config.IMAGE_SIZE)
    dataset = BrainTumorDataset(data_dir, transform=transform, image_size=config.IMAGE_SIZE)

    if len(dataset) == 0:
        print("No images found in the dataset!")
        sys.exit(1)

    # Print statistics
    print_dataset_summary(dataset, split_name)

    # Create visualizations
    print("\nGenerating visualizations...")

    # 1. Class distribution bar chart
    fig1 = plot_class_distribution(dataset, f"Class Distribution ({split_name} set)")
    fig1.savefig("assets/screenshots/class_distribution.png", dpi=150, bbox_inches="tight")
    print("  Saved: assets/screenshots/class_distribution.png")

    # 2. Sample images grid
    fig2 = plot_sample_images(dataset, n_per_class=4, title=f"Sample MRI Images ({split_name} set)")
    fig2.savefig("assets/screenshots/sample_images.png", dpi=150, bbox_inches="tight")
    print("  Saved: assets/screenshots/sample_images.png")

    # 3. Image dimension distribution
    fig3 = plot_image_dimensions(dataset, n_samples=200)
    fig3.savefig("assets/screenshots/image_dimensions.png", dpi=150, bbox_inches="tight")
    print("  Saved: assets/screenshots/image_dimensions.png")

    # 4. Augmentation comparison
    fig4 = plot_augmentation_comparison(dataset, n_samples=3)
    fig4.savefig("assets/screenshots/augmentation_comparison.png", dpi=150, bbox_inches="tight")
    print("  Saved: assets/screenshots/augmentation_comparison.png")

    # Show all plots
    print("\nDisplaying plots...")
    plt.show()

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
