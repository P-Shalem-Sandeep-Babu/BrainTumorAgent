"""
Vision Transformer (ViT) Model for Brain Tumor Classification.

Uses Hugging Face's pre-trained ViT model:
    google/vit-base-patch16-224

This model was pre-trained on ImageNet (1.2 million images, 1000 classes).
We fine-tune it for our 4-class brain tumor classification task.

Architecture (how it works):
    MRI Image (224x224)
        --> Split into 16x16 patches (196 patches total)
        --> Each patch becomes a 768-dimensional vector
        --> Pass through 12 Transformer encoder blocks
        --> Take the [CLS] token output (768-dim)
        --> Pass through classification head (768 -> 4 classes)
        --> Output: probability for each tumor class

Why use a pre-trained model?
    - It already knows how to "see" edges, textures, and shapes from ImageNet
    - We only need to teach it about brain tumors (much less data needed)
    - Fine-tuning is faster and more accurate than training from scratch
"""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from src.utils.config import Config


# ==============================================================================
# MODEL DEFINITION
# ==============================================================================

class BrainTumorViT(nn.Module):
    """
    Vision Transformer for brain tumor MRI classification.

    This wraps the Hugging Face ViT model and adds:
    - A dropout layer (prevents overfitting)
    - A classification head (maps features to 4 tumor classes)

    Args:
        config: Config object with model hyperparameters.
    """

    def __init__(self, config: Config):
        super().__init__()

        self.num_classes = config.NUM_CLASSES
        self.freeze_layers = config.VIT_FREEZE_LAYERS

        # ----------------------------------------------------------
        # Load pre-trained ViT from Hugging Face
        # ----------------------------------------------------------
        # This downloads the model weights on first run (~330 MB)
        # and caches them in ~/.cache/huggingface/
        print(f"Loading pre-trained ViT: {config.VIT_MODEL_NAME}")
        self.vit = ViTModel.from_pretrained(
            config.VIT_MODEL_NAME,
        )

        # ----------------------------------------------------------
        # Enable gradient checkpointing (saves GPU memory)
        # ----------------------------------------------------------
        # Trades ~30% slower training for ~40% less memory usage
        # Useful when GPU memory is limited
        if getattr(config, "VIT_GRADIENT_CHECKPOINTING", False):
            self.vit.gradient_checkpointing_enable()
            print("  Gradient checkpointing enabled (saves memory)")

        # ----------------------------------------------------------
        # Classification head
        # ----------------------------------------------------------
        # The ViT's [CLS] token output is 768-dimensional
        # We map it to 4 classes (glioma, meningioma, notumor, pituitary)
        self.dropout = nn.Dropout(config.VIT_DROPOUT)
        self.classifier = nn.Linear(self.vit.config.hidden_size, config.NUM_CLASSES)

        # ----------------------------------------------------------
        # Optionally freeze early layers
        # ----------------------------------------------------------
        # Freezing means those layers won't update during training
        # This preserves the pre-trained features and speeds up training
        if self.freeze_layers > 0:
            self._freeze_early_layers()

    def _freeze_early_layers(self):
        """
        Freeze the embedding layer and first N transformer blocks.

        The ViT has:
        - embeddings: converts image patches to vectors
        - encoder.layer[0..11]: 12 transformer blocks

        We freeze the embeddings + first N blocks, and only train
        the remaining blocks + our classification head.
        """
        # Freeze the patch embedding layer
        for param in self.vit.embeddings.parameters():
            param.requires_grad = False

        # Freeze the first N transformer blocks
        total_layers = len(self.vit.encoder.layer)
        for i in range(min(self.freeze_layers, total_layers)):
            for param in self.vit.encoder.layer[i].parameters():
                param.requires_grad = False

        print(f"  Froze: embeddings + encoder layers 0-{self.freeze_layers - 1}")
        print(f"  Trainable: encoder layers {self.freeze_layers}-{total_layers - 1} + classifier")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - convert image pixels to class predictions.

        Args:
            pixel_values: Tensor of shape (batch_size, 3, 224, 224)
                          Already normalized with ImageNet statistics.

        Returns:
            logits: Tensor of shape (batch_size, 4)
                    Raw scores for each class (higher = more confident).
                    Apply softmax to get probabilities.
        """
        # Pass through ViT encoder
        # outputs.last_hidden_state shape: (batch, 197, 768)
        #   - 197 = 196 patches + 1 [CLS] token
        #   - 768 = embedding dimension
        outputs = self.vit(pixel_values=pixel_values)

        # Extract the [CLS] token (first token, index 0)
        # This token summarizes the entire image
        cls_token = outputs.last_hidden_state[:, 0, :]  # Shape: (batch, 768)

        # Apply dropout for regularization, then classify
        x = self.dropout(cls_token)
        logits = self.classifier(x)  # Shape: (batch, 4)

        return logits

    def get_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract the [CLS] token embedding (before the classification head).

        This is useful for:
        - The RL agent (uses features as state representation)
        - Visualization (t-SNE plots of feature space)

        Args:
            pixel_values: Tensor of shape (batch_size, 3, 224, 224).

        Returns:
            features: Tensor of shape (batch_size, 768).
        """
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.last_hidden_state[:, 0, :]

    def count_parameters(self) -> dict:
        """
        Count total and trainable parameters.

        Returns:
            Dictionary with 'total' and 'trainable' parameter counts.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_vit_model(config: Config) -> BrainTumorViT:
    """
    Create and initialize the ViT model.

    This is the main function to call when you need the model.
    It handles:
    - Creating the model
    - Moving it to GPU (if available)
    - Printing parameter counts

    Args:
        config: Config object with all hyperparameters.

    Returns:
        BrainTumorViT model, ready for training.
    """
    model = BrainTumorViT(config)
    model = model.to(config.DEVICE)

    # Print model info
    params = model.count_parameters()
    print(f"\nModel: {config.VIT_MODEL_NAME}")
    print(f"Device: {config.DEVICE}")
    print(f"Parameters: {params['trainable']:,} trainable / {params['total']:,} total")
    print(f"Classes: {config.NUM_CLASSES} ({', '.join(config.CLASS_NAMES)})")

    # Print layer summary
    trainable_layers = sum(1 for p in model.parameters() if p.requires_grad)
    frozen_layers = sum(1 for p in model.parameters() if not p.requires_grad)
    print(f"Layer groups: {trainable_layers} trainable, {frozen_layers} frozen")

    return model


# ==============================================================================
# QUICK TEST
# ==============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("ViT Model - Quick Test")
    print("=" * 50)

    config = Config()
    model = create_vit_model(config)

    # Create a dummy batch of 2 images
    dummy_batch = torch.randn(2, 3, 224, 224).to(config.DEVICE)

    # Test forward pass
    logits = model(dummy_batch)
    print(f"\nForward pass:")
    print(f"  Input shape:  {dummy_batch.shape}")
    print(f"  Output shape: {logits.shape}")  # Should be (2, 4)
    print(f"  Logits: {logits[0].tolist()}")

    # Test feature extraction
    features = model.get_features(dummy_batch)
    print(f"\nFeature extraction:")
    print(f"  Features shape: {features.shape}")  # Should be (2, 768)

    # Test softmax probabilities
    probs = torch.softmax(logits, dim=1)
    print(f"\nProbabilities (after softmax):")
    for i, name in enumerate(config.CLASS_NAMES):
        print(f"  {name}: {probs[0][i]:.4f}")

    print(f"\nTotal: {probs[0].sum():.4f} (should be 1.0)")
