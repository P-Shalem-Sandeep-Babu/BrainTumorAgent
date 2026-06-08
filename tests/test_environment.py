"""
Unit tests for the RL environment.

Run with: python -m pytest tests/test_environment.py -v
"""

import pytest
import numpy as np
from unittest.mock import MagicMock
from src.environment.brain_tumor_env import BrainTumorEnv
from src.utils.config import Config


class TestBrainTumorEnv:
    """Tests for the Brain Tumor RL environment."""

    @pytest.fixture
    def mock_env(self):
        """Create env with mocked dataset and ViT model."""
        config = Config()

        # Mock dataset with 10 dummy samples
        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=10)
        dataset.__getitem__ = MagicMock(
            return_value=(MagicMock(shape=(3, 224, 224)), 0)
        )

        # Mock ViT model that returns dummy features
        vit_model = MagicMock()
        dummy_features = MagicMock()
        dummy_features.squeeze = MagicMock(
            return_value=MagicMock(cpu=MagicMock(
                numpy=MagicMock(return_value=np.random.randn(config.VIT_EMBED_DIM))
            ))
        )
        vit_model.get_features = MagicMock(return_value=dummy_features)

        env = BrainTumorEnv(dataset, vit_model, config)
        return env

    def test_action_space(self, mock_env):
        """Verify action space is Discrete(4)."""
        assert mock_env.action_space.n == 4

    def test_observation_space_shape(self, mock_env):
        """Verify observation space matches ViT embedding dim."""
        assert mock_env.observation_space.shape == (768,)

    def test_reset_returns_observation(self, mock_env):
        """Verify reset returns valid observation."""
        obs, info = mock_env.reset()
        assert obs.shape == (768,)
        assert isinstance(info, dict)

    def test_step_returns_correct_tuple(self, mock_env):
        """Verify step returns (obs, reward, terminated, truncated, info)."""
        mock_env.reset()
        result = mock_env.step(0)
        assert len(result) == 5

        obs, reward, terminated, truncated, info = result
        assert obs.shape == (768,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(info, dict)
