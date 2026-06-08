"""
Image Transforms and Augmentations for Brain Tumor MRI Images.

This file defines how images are preprocessed before being fed to the model.
- Training: images get augmented (flipped, rotated, brightness changes) to
  increase variety and help the model generalize better.
- Validation/Test: images only get resized and normalized (no random changes).

ImageNet normalization values are used because we'll use a ViT pretrained on ImageNet.
"""

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet mean and standard deviation (used for normalization)
# These values help the model by scaling pixel values to a range it expects
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_augmentations(image_size: int = 224):
    """
    Create augmentation pipeline for TRAINING images.

    Augmentations artificially increase dataset size by creating modified
    versions of existing images. This helps the model learn to recognize
    tumors regardless of their position, angle, or brightness.

    What each augmentation does:
    - Resize: Makes all images the same size (224x224) for the ViT model
    - HorizontalFlip: Flips image left-to-right (50% chance)
    - VerticalFlip: Flips image top-to-bottom (20% chance)
    - ShiftScaleRotate: Moves, zooms, and rotates the image slightly
    - GaussNoise/GaussianBlur: Adds noise or blur to simulate real-world conditions
    - RandomBrightnessContrast: Changes brightness and contrast
    - CLAHE: Improves contrast (useful for MRI scans)
    - CoarseDropout: Randomly blacks out small regions (forces model to not rely on one area)
    - Normalize: Scales pixel values to match ImageNet statistics
    - ToTensorV2: Converts numpy array to PyTorch tensor

    Args:
        image_size: Target image size (images will be resized to image_size x image_size).

    Returns:
        Albumentations Compose pipeline.
    """
    return A.Compose([
        # Step 1: Resize all images to the same dimensions
        A.Resize(image_size, image_size),

        # Step 2: Random geometric transformations
        A.HorizontalFlip(p=0.5),                    # Flip left-right, 50% chance
        A.VerticalFlip(p=0.2),                       # Flip top-bottom, 20% chance
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},  # Shift up to 5%
            scale=(0.9, 1.1),                         # Zoom in/out up to 10%
            rotate=(-15, 15),                         # Rotate up to 15 degrees
            border_mode=0,                            # Fill borders with black
            p=0.5                                     # Apply 50% of the time
        ),

        # Step 3: Random noise/blur (pick one of these, 30% chance)
        A.OneOf([
            A.GaussNoise(std_range=(0.1, 0.5)),       # Add random noise
            A.GaussianBlur(blur_limit=(3, 7)),         # Blur the image
        ], p=0.3),

        # Step 4: Random brightness/contrast changes (pick one, 30% chance)
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,                  # Change brightness by up to 20%
                contrast_limit=0.2                     # Change contrast by up to 20%
            ),
            A.CLAHE(clip_limit=2.0),                   # Adaptive histogram equalization
            A.HueSaturationValue(
                hue_shift_limit=10,                    # Shift color hue
                sat_shift_limit=20                     # Shift color saturation
            ),
        ], p=0.3),

        # Step 5: Randomly black out rectangular regions (forces model to look at whole image)
        A.CoarseDropout(
            num_holes_range=(1, 8),                    # 1 to 8 holes
            hole_height_range=(8, 16),                 # Each hole 8-16px tall
            hole_width_range=(8, 16),                  # Each hole 8-16px wide
            fill=0,                                    # Fill holes with black
            p=0.2                                      # Apply 20% of the time
        ),

        # Step 6: Normalize using ImageNet statistics
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),

        # Step 7: Convert to PyTorch tensor
        ToTensorV2(),
    ])


def get_val_augmentations(image_size: int = 224):
    """
    Create preprocessing pipeline for VALIDATION/TEST images.

    No random augmentations here - we want consistent evaluation.
    Only resize and normalize.

    Args:
        image_size: Target image size.

    Returns:
        Albumentations Compose pipeline.
    """
    return A.Compose([
        A.Resize(image_size, image_size),              # Resize to model input size
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),  # Normalize
        ToTensorV2(),                                  # Convert to tensor
    ])


def get_inference_transform(image_size: int = 224):
    """
    Transform pipeline for single image inference (used in the Streamlit app).

    Same as validation transform - just resize, normalize, and convert to tensor.

    Args:
        image_size: Target image size.

    Returns:
        Albumentations Compose pipeline.
    """
    return get_val_augmentations(image_size)
