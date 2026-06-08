"""
Unit tests for the ViT model module.

Tests verify that:
- Model produces correct output shapes
- Feature extraction works
- No NaN values in outputs
- Parameter freezing works correctly
- Model can be saved and loaded

Run with:
    python -m pytest tests/test_model.py -v
"""

import pytest
import torch
import tempfile
from pathlib import Path

from src.model.vit_model import BrainTumorViT, create_vit_model
from src.utils.config import Config


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def config():
    """Create a Config object for testing."""
    return Config()


@pytest.fixture
def model(config):
    """Create a model for testing."""
    return create_vit_model(config)


@pytest.fixture
def dummy_batch(config):
    """Create a dummy batch of images."""
    return torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(config.DEVICE)


# ==============================================================================
# MODEL SHAPE TESTS
# ==============================================================================

class TestModelOutputShapes:
    """Verify model produces tensors with expected shapes."""

    def test_forward_output_shape(self, model, dummy_batch, config):
        """Forward pass should produce (batch_size, num_classes) logits."""
        output = model(dummy_batch)
        assert output.shape == (2, config.NUM_CLASSES)

    def test_single_image_output(self, model, config):
        """Model should work with a single image (batch_size=1)."""
        single = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(config.DEVICE)
        output = model(single)
        assert output.shape == (1, config.NUM_CLASSES)

    def test_feature_extraction_shape(self, model, dummy_batch, config):
        """get_features should return (batch_size, embed_dim) embeddings."""
        features = model.get_features(dummy_batch)
        assert features.shape[0] == 2
        assert features.shape[1] == config.VIT_EMBED_DIM

    def test_logits_are_finite(self, model, dummy_batch):
        """Model output should not contain NaN or Inf values."""
        output = model(dummy_batch)
        assert torch.isfinite(output).all(), "Model produced NaN or Inf values"


# ==============================================================================
# SOFTMAX / PROBABILITY TESTS
# ==============================================================================

class TestProbabilities:
    """Verify logits can be converted to valid probabilities."""

    def test_softmax_sums_to_one(self, model, dummy_batch):
        """After softmax, probabilities should sum to 1.0."""
        logits = model(dummy_batch)
        probs = torch.softmax(logits, dim=1)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_probabilities_are_positive(self, model, dummy_batch):
        """All probabilities should be >= 0 after softmax."""
        logits = model(dummy_batch)
        probs = torch.softmax(logits, dim=1)
        assert (probs >= 0).all()


# ==============================================================================
# PARAMETER TESTS
# ==============================================================================

class TestParameters:
    """Verify model parameters are set up correctly."""

    def test_has_trainable_parameters(self, model):
        """Model should have some trainable parameters."""
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable > 0

    def test_count_parameters_returns_dict(self, model):
        """count_parameters should return a dict with 'total' and 'trainable'."""
        params = model.count_parameters()
        assert "total" in params
        assert "trainable" in params
        assert params["total"] > 0
        assert params["trainable"] > 0

    def test_frozen_layers_reduce_trainable_params(self, config):
        """Freezing layers should reduce trainable parameter count."""
        # Create model with no frozen layers
        config_no_freeze = Config()
        config_no_freeze.VIT_FREEZE_LAYERS = 0
        model_full = create_vit_model(config_no_freeze)

        # Create model with frozen layers
        config_frozen = Config()
        config_frozen.VIT_FREEZE_LAYERS = 8
        model_frozen = create_vit_model(config_frozen)

        full_params = model_full.count_parameters()["trainable"]
        frozen_params = model_frozen.count_parameters()["trainable"]

        assert frozen_params < full_params, \
            f"Frozen model should have fewer trainable params: {frozen_params} vs {full_params}"


# ==============================================================================
# SAVE / LOAD TESTS
# ==============================================================================

class TestSaveLoad:
    """Verify model can be saved and loaded correctly."""

    def test_save_and_load_checkpoint(self, model, dummy_batch, config):
        """Saved model should produce identical outputs when loaded."""
        # Get original output
        model.eval()
        with torch.no_grad():
            original_output = model(dummy_batch)

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_checkpoint.pth"
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": {
                    "model_name": config.VIT_MODEL_NAME,
                    "num_classes": config.NUM_CLASSES,
                },
            }
            torch.save(checkpoint, checkpoint_path)

            # Load into a new model
            new_model = create_vit_model(config)
            loaded = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=True)
            new_model.load_state_dict(loaded["model_state_dict"])
            new_model.eval()

            with torch.no_grad():
                loaded_output = new_model(dummy_batch)

            # Outputs should be identical
            assert torch.allclose(original_output, loaded_output, atol=1e-6), \
                "Loaded model produces different outputs than original"


# ==============================================================================
# GRADIENT TESTS
# ==============================================================================

class TestGradients:
    """Verify gradient flow during training."""

    def test_gradients_computed_for_trainable_params(self, model, dummy_batch):
        """Backward pass should compute gradients for trainable parameters."""
        model.train()
        output = model(dummy_batch)

        # Create dummy loss and backward
        target = torch.randint(0, 4, (2,)).to(dummy_batch.device)
        loss = torch.nn.CrossEntropyLoss()(output, target)
        loss.backward()

        # Check that most trainable params have gradients
        # Note: ViT pooler and some encoder layers may not receive gradients
        # because they aren't in the forward computation path
        params_with_grad = 0
        params_without_grad = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    params_with_grad += 1
                else:
                    params_without_grad += 1

        # Most trainable parameters should have gradients
        assert params_with_grad > 0, "No trainable parameters received gradients"
        # Allow some params (like pooler) to not have gradients
        assert params_with_grad > params_without_grad, \
            f"Too many params without gradients: {params_without_grad} vs {params_with_grad}"

    def test_frozen_params_have_no_gradients(self, model, dummy_batch):
        """Frozen parameters should NOT have gradients after backward."""
        model.train()
        output = model(dummy_batch)
        target = torch.randint(0, 4, (2,)).to(dummy_batch.device)
        loss = torch.nn.CrossEntropyLoss()(output, target)
        loss.backward()

        for name, param in model.named_parameters():
            if not param.requires_grad:
                assert param.grad is None, f"Frozen param {name} has gradient"
