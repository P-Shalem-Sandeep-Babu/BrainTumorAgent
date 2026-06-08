"""
ViT Attention Rollout Visualization.

Generates a heatmap showing which image regions the Vision Transformer
attended to when making a prediction. This is the "explainability"
piece -- useful for a panel to demonstrate that the model is looking
at clinically relevant areas, not random pixels.

Method: Attention Rollout (Abnar & Zuidema, 2020)
    1. Hook the attention weights from the last transformer block.
    2. Average attention across heads.
    3. Recursively multiply attention through layers.
    4. Discard the [CLS] row, reshape to (14, 14) for a 224x224 / patch16 ViT.
    5. Upsample to 224x224 and overlay on the original image.

We don't need gradients -- this is a forward-only analysis.
"""

import numpy as np
import torch
from PIL import Image
from typing import Tuple

from src.utils.config import Config
from src.preprocessing.transforms import get_inference_transform

import cv2


class AttentionVisualizer:
    """
    Produces an attention heatmap for a single image using the
    last-block attention rollout of the underlying ViT.
    """

    def __init__(self, model, config: Config = None, device: str = None):
        self.config = config or Config()
        self.device = device or self.config.DEVICE
        self.model = model.to(self.device).eval()
        self.transform = get_inference_transform(self.config.IMAGE_SIZE)

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")
        arr = np.array(image)
        return self.transform(image=arr)["image"].unsqueeze(0).to(self.device)

    @torch.no_grad()
    def compute_attention_map(self, image: Image.Image) -> np.ndarray:
        """
        Compute the attention rollout heatmap.

        Returns:
            heatmap: np.ndarray (H, W) in [0, 1], same size as config.IMAGE_SIZE
        """
        x = self._preprocess(image)  # (1, 3, 224, 224)

        # We need attention weights -- enable output_attentions
        outputs = self.model.vit(
            pixel_values=x,
            output_attentions=True,
            return_dict=True,
        )
        # outputs.attentions is a tuple of (1, num_heads, 197, 197) per layer
        attentions = outputs.attentions
        num_layers = len(attentions)

        # Average across heads, then rollout
        # Resulting shape: (1, 197, 197)
        attn = attentions[0].mean(dim=1)
        for i in range(1, num_layers):
            attn = torch.bmm(attentions[i].mean(dim=1), attn)
        # attn[0, 0, :] is the [CLS] token's attention to all patches
        cls_attn = attn[0, 0, 1:]  # discard CLS->CLS, keep patch attentions

        # 196 patches for 224/16 -> 14x14 grid
        num_patches = int(np.sqrt(cls_attn.shape[0]))
        attn_map = cls_attn.reshape(num_patches, num_patches).cpu().numpy()
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

        # Upsample to image size
        img_size = self.config.IMAGE_SIZE
        attn_resized = cv2.resize(attn_map, (img_size, img_size))
        return attn_resized

    def overlay_on_image(
        self,
        image: Image.Image,
        alpha: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Overlay the heatmap on the original image.

        Args:
            image: PIL image
            alpha: blend factor (0 = pure image, 1 = pure heatmap)

        Returns:
            overlay_rgb: np.ndarray (H, W, 3) -- heatmap blended onto image
            heatmap_rgb: np.ndarray (H, W, 3) -- colormap-only heatmap
        """
        heatmap = self.compute_attention_map(image)

        # Apply JET colormap for visualization
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # Resize original image to match
        img_resized = image.convert("RGB").resize(
            (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)
        )
        img_arr = np.array(img_resized)

        overlay = cv2.addWeighted(img_arr, 1 - alpha, heatmap_color, alpha, 0)
        return overlay, heatmap_color
