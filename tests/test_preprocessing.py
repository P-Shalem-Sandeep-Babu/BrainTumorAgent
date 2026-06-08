"""
Unit tests for the preprocessing module.

Tests verify that:
- Transforms produce correct tensor shapes
- Dataset class correctly scans directories
- Augmentations don't produce NaN values
- Config values are consistent

Run with:
    python -m pytest tests/test_preprocessing.py -v
"""

import pytest
import numpy as np
from PIL import Image

import torch
from src.preprocessing.dataset import BrainTumorDataset
from src.preprocessing.transforms import (
    get_train_augmentations,
    get_val_augmentations,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from src.utils.config import Config


# ==============================================================================
# TRANSFORM TESTS
# ==============================================================================

class TestTransforms:
    """Tests for image transform and augmentation functions."""

    def _make_dummy_image(self, width=300, height=250):
        """Create a dummy RGB image for testing."""
        return Image.fromarray(
            np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        )

    def test_train_transform_output_shape(self):
        """Training transforms should produce a (3, 224, 224) tensor."""
        transform = get_train_augmentations(image_size=224)
        dummy_img = self._make_dummy_image()
        img_np = np.array(dummy_img)
        result = transform(image=img_np)["image"]

        assert result.shape == (3, 224, 224), f"Expected (3, 224, 224), got {result.shape}"

    def test_val_transform_output_shape(self):
        """Validation transforms should produce a (3, 224, 224) tensor."""
        transform = get_val_augmentations(image_size=224)
        dummy_img = self._make_dummy_image()
        img_np = np.array(dummy_img)
        result = transform(image=img_np)["image"]

        assert result.shape == (3, 224, 224), f"Expected (3, 224, 224), got {result.shape}"

    def test_train_transform_dtype(self):
        """Training transforms should return a float32 tensor."""
        transform = get_train_augmentations(image_size=224)
        dummy_img = self._make_dummy_image()
        img_np = np.array(dummy_img)
        result = transform(image=img_np)["image"]

        assert result.dtype == torch.float32

    def test_val_transform_dtype(self):
        """Validation transforms should return a float32 tensor."""
        transform = get_val_augmentations(image_size=224)
        dummy_img = self._make_dummy_image()
        img_np = np.array(dummy_img)
        result = transform(image=img_np)["image"]

        assert result.dtype == torch.float32

    def test_no_nan_in_output(self):
        """Transforms should never produce NaN values."""
        transform = get_train_augmentations(image_size=224)
        dummy_img = self._make_dummy_image()
        img_np = np.array(dummy_img)
        result = transform(image=img_np)["image"]

        assert not torch.isnan(result).any(), "Transform produced NaN values"

    def test_custom_image_size(self):
        """Transforms should work with different image sizes."""
        for size in [128, 196, 224, 256]:
            transform = get_val_augmentations(image_size=size)
            dummy_img = self._make_dummy_image()
            img_np = np.array(dummy_img)
            result = transform(image=img_np)["image"]

            assert result.shape == (3, size, size), \
                f"Size {size}: expected (3, {size}, {size}), got {result.shape}"

    def test_imagenet_normalization_values(self):
        """Verify ImageNet normalization constants are correct."""
        assert len(IMAGENET_MEAN) == 3
        assert len(IMAGENET_STD) == 3
        assert abs(sum(IMAGENET_MEAN) - 1.347) < 0.01  # 0.485 + 0.456 + 0.406
        assert all(s > 0 for s in IMAGENET_STD)


# ==============================================================================
# DATASET CLASS TESTS
# ==============================================================================

class TestBrainTumorDataset:
    """Tests for the BrainTumorDataset class."""

    def test_class_map_has_four_classes(self):
        """Dataset should define exactly 4 tumor classes."""
        assert len(BrainTumorDataset.CLASS_MAP) == 4

    def test_class_map_values(self):
        """Class map should map to indices 0-3."""
        expected = {"glioma": 0, "meningioma": 1, "notumor": 2, "pituitary": 3}
        assert BrainTumorDataset.CLASS_MAP == expected

    def test_label_map_reverse(self):
        """LABEL_MAP should be the reverse of CLASS_MAP."""
        for name, idx in BrainTumorDataset.CLASS_MAP.items():
            assert BrainTumorDataset.LABEL_MAP[idx] == name

    def test_get_class_name(self):
        """get_class_name should return correct names."""
        from src.preprocessing.transforms import get_val_augmentations
        import tempfile, os

        # Create a minimal temp directory structure
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "glioma"))
            dataset = BrainTumorDataset(tmpdir, transform=None)

            assert dataset.get_class_name(0) == "glioma"
            assert dataset.get_class_name(1) == "meningioma"
            assert dataset.get_class_name(2) == "notumor"
            assert dataset.get_class_name(3) == "pituitary"
            assert dataset.get_class_name(99) == "unknown"


# ==============================================================================
# CONFIG TESTS
# ==============================================================================

class TestConfig:
    """Tests for configuration values related to preprocessing."""

    def test_class_count(self):
        config = Config()
        assert config.NUM_CLASSES == 4

    def test_class_names_match_count(self):
        config = Config()
        assert len(config.CLASS_NAMES) == config.NUM_CLASSES

    def test_image_size_is_224(self):
        """ViT models expect 224x224 input."""
        config = Config()
        assert config.IMAGE_SIZE == 224

    def test_num_channels_is_3(self):
        """Images should be RGB (3 channels)."""
        config = Config()
        assert config.NUM_CHANNELS == 3

    def test_batch_size_positive(self):
        config = Config()
        assert config.BATCH_SIZE_DATA > 0
