"""
ViT Inference Wrapper for DQN Treatment Environment.

Bridges the trained ViT classifier with the TreatmentRecommendationEnv
so the DQN agent observes REAL predictions (class + softmax confidence)
instead of random simulations.

Pipeline at inference:
    PIL Image
      -> preprocess (resize/normalize)
      -> ViT forward pass
      -> softmax -> probabilities
      -> argmax -> predicted tumor index
      -> max prob -> confidence
      -> env observation: [tumor_idx, confidence]

Training the DQN on this pipeline (vs the random simulation) makes the
RL agent learn to handle the noisy, sometimes-uncertain outputs of a
real classifier -- not a perfectly clean oracle.
"""

import numpy as np
import torch
from PIL import Image
from typing import Tuple

from src.utils.config import Config
from src.preprocessing.transforms import get_inference_transform


class ViTInferenceWrapper:
    """
    Light wrapper that runs a single image through the trained ViT
    and returns (tumor_type_index, confidence) ready to feed the DQN env.
    """

    def __init__(self, model, config: Config = None, device: str = None):
        self.config = config or Config()
        self.device = device or self.config.DEVICE
        self.model = model.to(self.device).eval()
        self.transform = get_inference_transform(self.config.IMAGE_SIZE)

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """PIL Image -> (1, 3, 224, 224) tensor on device."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        arr = np.array(image)
        tensor = self.transform(image=arr)["image"]
        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Tuple[int, float, np.ndarray]:
        """
        Run ViT on a single image.

        Returns:
            tumor_index: int in [0, 3]  -- argmax over class probabilities
            confidence:  float in [0, 1] -- max softmax probability
            probs:       np.ndarray shape (4,) -- full probability vector
        """
        x = self.preprocess(image)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        return idx, conf, probs


class TreatmentEnvWithViT:
    """
    Drop-in replacement for TreatmentRecommendationEnv that sources its
    observations from the trained ViT instead of random sampling.

    Use this in the DQN training pipeline when you want the agent to
    learn from realistic (sometimes misclassified) ViT outputs.

    Example:
        env = TreatmentEnvWithViT(vit_wrapper, config)
        obs, info = env.reset(image=pil_image)
    """

    def __init__(self, vit_wrapper: ViTInferenceWrapper, config: Config = None):
        from src.environment.treatment_env import TreatmentRecommendationEnv

        self.config = config or Config()
        self.vit = vit_wrapper
        # Reuse reward logic from the base env
        self._base = TreatmentRecommendationEnv(self.config)

    def reset(self, image: Image.Image = None, seed: int = None):
        """
        Reset the env using a ViT prediction on the given image.

        If `image` is None, falls back to a random simulation
        (for DQN training when you don't have a ViT online yet).
        """
        if image is None:
            return self._base.reset(seed=seed)

        idx, conf, probs = self.vit.predict(image)
        self._base.tumor_type = idx
        self._base.confidence = conf
        self._base.severity = self.config.TUMOR_SEVERITY[
            self.config.TUMOR_TYPES[idx]
        ]
        self._base.steps = 0

        obs = np.array([float(idx), conf], dtype=np.float32)
        info = {
            "tumor_type": self.config.TUMOR_TYPES[idx],
            "tumor_type_idx": idx,
            "confidence": conf,
            "severity": self._base.severity,
            "probs": probs,
            "source": "ViT",
        }
        return obs, info

    def step(self, action: int):
        return self._base.step(action)

    @property
    def action_space(self):
        return self._base.action_space

    @property
    def observation_space(self):
        return self._base.observation_space
