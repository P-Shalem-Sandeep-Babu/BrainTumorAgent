"""
Brain Tumor RL Environment.

A custom Gymnasium environment where the RL agent observes MRI image
features (from ViT) and selects an action (tumor class prediction).
The environment provides rewards based on correctness.

How it works:
    1. Environment samples an MRI image from the dataset
    2. ViT extracts features from the image
    3. RL agent sees the features and picks a class (action)
    4. Environment gives +1 reward for correct, -1 for wrong
    5. Episode ends after one classification attempt per image

This is a simple single-step environment. For a more advanced version,
the agent could "look around" the image by cropping regions (multi-step).
"""

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from src.utils.config import Config


class BrainTumorEnv(gym.Env):
    """
    Custom Gymnasium environment for brain tumor classification.

    The agent receives ViT features as observation and outputs
    a discrete action (0-3) corresponding to the tumor class.

    Args:
        dataset: BrainTumorDataset instance.
        vit_model: Trained ViT model for feature extraction.
        config: Config object.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, dataset, vit_model, config: Config):
        super().__init__()

        self.dataset = dataset
        self.vit_model = vit_model
        self.config = config
        self.device = config.DEVICE

        # --- Action Space ---
        # 4 discrete actions = 4 tumor classes
        self.action_space = spaces.Discrete(config.NUM_CLASSES)

        # --- Observation Space ---
        # ViT embedding dimension (768 for base ViT)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(config.VIT_EMBED_DIM,),
            dtype=np.float32,
        )

        # Internal state
        self.current_idx = 0
        self.current_label = None
        self.current_features = None
        self.steps = 0
        self.max_steps = config.MAX_STEPS_PER_EPISODE

    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode.

        Returns:
            observation: ViT feature vector.
            info: Additional information dict.
        """
        super().reset(seed=seed)

        # Pick a random image from the dataset
        self.current_idx = self.np_random.integers(0, len(self.dataset))
        image, label = self.dataset[self.current_idx]
        self.current_label = label

        # Extract features using ViT (no gradient needed)
        with torch.no_grad():
            image_batch = image.unsqueeze(0).to(self.device)
            self.current_features = self.vit_model.get_features(image_batch)
            self.current_features = self.current_features.squeeze(0).cpu().numpy()

        self.steps = 0

        return self.current_features.astype(np.float32), {}

    def step(self, action: int):
        """
        Take an action (classify the image).

        Args:
            action: Integer 0-3, the predicted class.

        Returns:
            observation: Same features (single-step episode).
            reward: +1 correct, -1 wrong, small step penalty.
            terminated: True (single-step).
            truncated: False.
            info: Dict with true label, prediction, correct flag.
        """
        self.steps += 1

        # Check if prediction is correct
        correct = int(action) == int(self.current_label)

        # Calculate reward
        if correct:
            reward = self.config.REWARD_CORRECT
        else:
            reward = self.config.REWARD_WRONG

        # Episode always terminates after one classification
        terminated = True
        truncated = False

        info = {
            "true_label": self.current_label,
            "predicted_label": action,
            "correct": correct,
            "class_name": self.config.CLASS_NAMES[self.current_label],
            "predicted_class_name": self.config.CLASS_NAMES[action],
        }

        return self.current_features.astype(np.float32), reward, terminated, truncated, info

    def render(self):
        """Print the current state to console."""
        if self.current_label is not None:
            print(f"Image #{self.current_idx} | True label: {self.config.CLASS_NAMES[self.current_label]}")


# --- Quick test (requires dataset and model) ---
if __name__ == "__main__":
    print("BrainTumorEnv defined. To test, run with a loaded dataset and ViT model.")
    print("Example:")
    print("  env = BrainTumorEnv(dataset, vit_model, config)")
    print("  obs, info = env.reset()")
    print("  obs, reward, done, trunc, info = env.step(2)")
