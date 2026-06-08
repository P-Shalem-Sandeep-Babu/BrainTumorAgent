"""
Data Preparation Script.

This script prepares the brain tumor MRI dataset for training:
1. Checks that raw images exist in data/raw/
2. Splits them into train/validation/test sets
3. Copies the split images to data/processed/
4. Prints dataset statistics

Usage:
    python prepare_data.py           # Interactive (asks before re-splitting)
    python prepare_data.py --yes     # Non-interactive (auto re-splits)

After running this script, the data/processed/ folder will contain:
    data/processed/
        train/
            glioma/       (70% of glioma images)
            meningioma/   (70% of meningioma images)
            notumor/      (70% of notumor images)
            pituitary/    (70% of pituitary images)
        val/
            glioma/       (15% of glioma images)
            ...
        test/
            glioma/       (15% of glioma images)
            ...
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import Config
from src.preprocessing.dataset import (
    split_dataset,
    BrainTumorDataset,
    print_dataset_statistics,
    check_dataset_exists,
)
from src.preprocessing.transforms import get_val_augmentations


def main():
    """Main data preparation pipeline."""
    parser = argparse.ArgumentParser(description="Prepare brain tumor MRI dataset")
    parser.add_argument(
        "-y", "--yes", action="store_true",
        help="Skip confirmation prompts (non-interactive mode)",
    )
    args = parser.parse_args()

    config = Config()

    print("=" * 60)
    print("Brain Tumor Agent - Data Preparation")
    print("=" * 60)

    # ----------------------------------------------------------
    # Step 1: Check if raw dataset exists
    # ----------------------------------------------------------
    print(f"\nStep 1: Checking for raw dataset at {config.DATA_RAW}")

    if not check_dataset_exists(config.DATA_RAW):
        print("\n" + "!" * 60)
        print("ERROR: Raw dataset not found!")
        print("!" * 60)
        print("\nPlease download a brain tumor MRI dataset and place it in:")
        print(f"  {config.DATA_RAW}/")
        print("\nExpected folder structure:")
        print("  data/raw/")
        print("    glioma/       <- glioma tumor MRI images")
        print("    meningioma/   <- meningioma tumor MRI images")
        print("    notumor/      <- healthy brain MRI images")
        print("    pituitary/    <- pituitary tumor MRI images")
        print("\nRecommended datasets:")
        print("  - Kaggle: 'Brain Tumor MRI Dataset' by Masoud Nickparvar")
        print("  - Kaggle: 'Brain MRI Images for Brain Tumor Detection'")
        print("\nAfter downloading, run this script again.")
        sys.exit(1)

    print("  Raw dataset found!")

    # Show raw dataset statistics
    print("\nRaw dataset contents:")
    raw_transform = get_val_augmentations(config.IMAGE_SIZE)
    raw_dataset = BrainTumorDataset(config.DATA_RAW, transform=raw_transform)
    print_dataset_statistics(raw_dataset.get_statistics())

    # ----------------------------------------------------------
    # Step 2: Split into train/val/test
    # ----------------------------------------------------------
    print("\nStep 2: Splitting dataset into train/val/test sets")

    # Check if processed data already exists
    if config.DATA_PROCESSED.exists() and any(config.DATA_PROCESSED.iterdir()):
        if args.yes:
            print("\nProcessed data already exists. Re-splitting (--yes flag set).")
        else:
            response = input("\nProcessed data already exists. Re-split? (y/n): ").strip().lower()
            if response != "y":
                print("Skipping split. Using existing processed data.")
                print("\nDone!")
                return

    split_stats = split_dataset(
        source_dir=config.DATA_RAW,
        output_dir=config.DATA_PROCESSED,
        train_ratio=0.70,       # 70% for training
        val_ratio=0.15,         # 15% for validation
        test_ratio=0.15,        # 15% for testing
        seed=config.SEED,       # For reproducible splits
    )

    # ----------------------------------------------------------
    # Step 3: Verify the split
    # ----------------------------------------------------------
    print("\nStep 3: Verifying split datasets")

    for split_name in ["train", "val", "test"]:
        split_dir = config.DATA_PROCESSED / split_name
        print(f"\n  {split_name.upper()} set:")
        split_dataset_obj = BrainTumorDataset(
            split_dir, transform=raw_transform, image_size=config.IMAGE_SIZE
        )
        stats = split_dataset_obj.get_statistics()
        total = stats["total_images"]
        for class_name in stats["class_names"]:
            dist = stats["class_distribution"].get(class_name, {"count": 0, "percentage": 0})
            print(f"    {class_name:>12s}: {dist['count']:>5d} images ({dist['percentage']:.1f}%)")
        print(f"    {'TOTAL':>12s}: {total:>5d} images")

    # ----------------------------------------------------------
    # Done!
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print(f"\nProcessed data saved to: {config.DATA_PROCESSED}")
    print("\nNext steps:")
    print("  1. Train the model:  python train.py")
    print("  2. Launch the app:   streamlit run app/main.py")


if __name__ == "__main__":
    main()
