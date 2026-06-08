"""
Prediction Engine for the Brain Tumor Agent Streamlit App.

Handles:
- Loading the trained ViT classifier from checkpoint
- Loading the DQN treatment recommendation agent
- Preprocessing uploaded MRI images
- Running ViT inference to get tumor type + confidence
- Running DQN agent to get treatment recommendation

Usage:
    from app.utils.predictor import Predictor
    predictor = Predictor()
    result = predictor.predict(image_array)
"""

import numpy as np
import torch
from pathlib import Path
from PIL import Image

from src.utils.config import Config
from src.model.vit_model import BrainTumorViT
from src.preprocessing.transforms import get_inference_transform


class Predictor:
    """
    Main prediction engine that ties together:
    1. ViT classifier (image -> tumor type + confidence)
    2. DQN agent (tumor type + confidence -> treatment recommendation)

    Models are loaded lazily (on first prediction) to avoid
    slowing down app startup.
    """

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.device = self.config.DEVICE

        # Models (loaded lazily)
        self.vit_model = None
        self.dqn_model = None

        # Image transform for inference
        self.transform = get_inference_transform(self.config.IMAGE_SIZE)

        # Paths to trained model checkpoints
        self.vit_checkpoint_path = self.config.MODEL_CHECKPOINTS / "best_model.pth"
        self.dqn_checkpoint_path = self.config.MODEL_CHECKPOINTS / "dqn_treatment" / "best" / "best_model.zip"

    # =========================================================================
    # MODEL LOADING
    # =========================================================================

    def load_vit_model(self) -> bool:
        """
        Load the trained ViT classifier from checkpoint.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        if self.vit_model is not None:
            return True  # Already loaded

        if not self.vit_checkpoint_path.exists():
            return False

        try:
            # Create model architecture
            self.vit_model = BrainTumorViT(self.config)
            self.vit_model = self.vit_model.to(self.device)

            # Load trained weights
            checkpoint = torch.load(
                self.vit_checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
            self.vit_model.load_state_dict(checkpoint["model_state_dict"])
            self.vit_model.eval()

            return True
        except Exception as e:
            print(f"Error loading ViT model: {e}")
            self.vit_model = None
            return False

    def load_dqn_model(self) -> bool:
        """
        Load the trained DQN treatment recommendation agent.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        if self.dqn_model is not None:
            return True  # Already loaded

        if not self.dqn_checkpoint_path.exists():
            return False

        try:
            from stable_baselines3 import DQN

            self.dqn_model = DQN.load(
                str(self.dqn_checkpoint_path),
                device=self.device,
            )
            return True
        except Exception as e:
            print(f"Error loading DQN model: {e}")
            self.dqn_model = None
            return False

    # =========================================================================
    # AVAILABILITY CHECKS
    # =========================================================================

    def is_vit_available(self) -> bool:
        """Check if the trained ViT checkpoint exists on disk."""
        return self.vit_checkpoint_path.exists()

    def is_dqn_available(self) -> bool:
        """Check if the trained DQN checkpoint exists on disk."""
        return self.dqn_checkpoint_path.exists()

    def get_model_info(self) -> dict:
        """Get information about available models."""
        info = {
            "vit_available": self.is_vit_available(),
            "dqn_available": self.is_dqn_available(),
            "device": self.device,
            "vit_path": str(self.vit_checkpoint_path),
            "dqn_path": str(self.dqn_checkpoint_path),
        }

        # Add checkpoint metadata if available
        if self.is_vit_available():
            try:
                ckpt = torch.load(
                    self.vit_checkpoint_path,
                    map_location="cpu",
                    weights_only=False,
                )
                info["vit_epoch"] = ckpt.get("epoch", "N/A")
                info["vit_val_acc"] = ckpt.get("val_acc", "N/A")
                info["vit_val_loss"] = ckpt.get("val_loss", "N/A")
            except Exception:
                pass

        return info

    # =========================================================================
    # IMAGE PREPROCESSING
    # =========================================================================

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess a PIL image for the ViT model.

        Steps:
        1. Convert PIL to numpy array (RGB)
        2. Apply inference transform (resize + normalize + to tensor)
        3. Add batch dimension

        Args:
            image: PIL Image (any size, any mode).

        Returns:
            Tensor of shape (1, 3, 224, 224) ready for the model.
        """
        # Ensure RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Convert to numpy array
        image_np = np.array(image)

        # Apply albumentations transform
        transformed = self.transform(image=image_np)
        tensor = transformed["image"]  # Shape: (3, 224, 224)

        # Add batch dimension: (3, 224, 224) -> (1, 3, 224, 224)
        return tensor.unsqueeze(0).to(self.device)

    # =========================================================================
    # INFERENCE
    # =========================================================================

    def predict(self, image: Image.Image) -> dict:
        """
        Run full prediction pipeline on an MRI image.

        Steps:
        1. Preprocess the image
        2. Run ViT inference to get logits
        3. Convert logits to probabilities (softmax)
        4. Get predicted class and confidence
        5. Run DQN agent for treatment recommendation

        Args:
            image: PIL Image of a brain MRI scan.

        Returns:
            Dictionary with prediction results.

        Raises:
            RuntimeError: If the ViT model checkpoint is not found.
        """
        # Step 1: Load ViT model — this is required
        if not self.load_vit_model():
            raise RuntimeError(
                f"ViT model not found at:\n"
                f"  {self.vit_checkpoint_path}\n\n"
                f"Run 'python train.py' locally to train the model, "
                f"then upload the checkpoint to HuggingFace Hub."
            )

        # Step 2: Preprocess image
        pixel_values = self.preprocess_image(image)

        # Step 3: Run ViT inference
        with torch.no_grad():
            logits = self.vit_model(pixel_values)
            probabilities = torch.softmax(logits, dim=1)

        # Step 4: Extract results
        probs = probabilities.cpu().numpy()[0]  # Shape: (4,)
        predicted_index = int(np.argmax(probs))
        predicted_class = self.config.CLASS_NAMES[predicted_index]
        confidence = float(probs[predicted_index])

        # Build probability dict for all classes
        prob_dict = {
            name: float(prob)
            for name, prob in zip(self.config.CLASS_NAMES, probs)
        }

        # Build sorted class list (highest confidence first)
        all_classes = sorted(
            [
                {
                    "name": name,
                    "probability": float(prob),
                    "severity": self.config.TUMOR_SEVERITY.get(name, 0),
                }
                for name, prob in zip(self.config.CLASS_NAMES, probs)
            ],
            key=lambda x: x["probability"],
            reverse=True,
        )

        # Step 5: Get severity
        severity = self.config.TUMOR_SEVERITY.get(predicted_class, 0)

        # Step 6: Get DQN treatment recommendation
        recommendation, rec_index, rec_source = self._get_recommendation(
            predicted_index, confidence
        )

        return {
            "tumor_type": predicted_class,
            "tumor_index": predicted_index,
            "confidence": confidence,
            "probabilities": prob_dict,
            "recommendation": recommendation,
            "recommendation_index": rec_index,
            "severity": severity,
            "all_classes": all_classes,
            "vit_source": "Trained ViT",
            "rec_source": rec_source,
        }

    def _get_recommendation(
        self, tumor_index: int, confidence: float
    ) -> tuple[str, int, str]:
        """
        Get treatment recommendation from the DQN agent.

        If the DQN model is available, use it.
        Otherwise, fall back to rule-based recommendation.

        Args:
            tumor_index: Predicted tumor class index (0-3).
            confidence: Prediction confidence (0.0-1.0).

        Returns:
            Tuple of (recommendation_text, action_index, source_label).
        """
        # Try DQN agent first
        if self.load_dqn_model():
            try:
                obs = np.array(
                    [float(tumor_index), confidence], dtype=np.float32
                )
                action, _ = self.dqn_model.predict(obs, deterministic=True)
                action = int(action)
                recommendation = self.config.TREATMENT_ACTIONS[action]
                return recommendation, action, "DQN Agent"
            except Exception:
                pass  # Fall back to rule-based

        # Rule-based fallback (same logic as the environment)
        severity = self.config.TUMOR_SEVERITY[
            self.config.TUMOR_TYPES[tumor_index]
        ]
        # severity 0 -> Monitor, 1 -> Specialist, 2 -> Biopsy, 3 -> Emergency
        action = severity
        recommendation = self.config.TREATMENT_ACTIONS[action]
        return recommendation, action, "Rule-based"


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_predictor_instance = None


def get_predictor() -> Predictor:
    """
    Get or create the singleton Predictor instance.

    This avoids reloading models on every page navigation
    in the Streamlit multi-page app.
    """
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = Predictor()
    return _predictor_instance
