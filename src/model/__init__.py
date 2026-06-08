# src/model/ - Vision Transformer model, training, and evaluation
#
# This module contains:
# - vit_model.py: ViT model definition (Hugging Face pre-trained)
# - trainer.py: Training loop, validation loop, metrics, checkpointing
# - visualize_training.py: Training graphs and confusion matrix plots
#
# Main classes and functions:
#   BrainTumorViT    - ViT model for 4-class classification
#   create_vit_model - Factory function to create the model
#   Trainer          - Handles training and validation
#   evaluate_model   - Evaluate on test set
#   TrainingVisualizer - Generate training plots

from src.model.vit_model import BrainTumorViT, create_vit_model
from src.model.trainer import Trainer, evaluate_model, load_trained_model
from src.model.visualize_training import TrainingVisualizer, load_training_history
