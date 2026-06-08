"""
Central Configuration File.

All hyperparameters, paths, and settings are defined here.
Change values in one place instead of searching through multiple files.

Usage:
    from src.utils.config import Config
    config = Config()
    print(config.IMAGE_SIZE)  # 224
    print(config.DEVICE)      # 'cuda' or 'cpu'
"""

import torch
from pathlib import Path


class Config:
    # ==================================================================
    # PATHS
    # ==================================================================
    # Root directory is the project root (Brain Tumor Agent/)
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent

    # Data directories
    DATA_RAW = ROOT_DIR / "data" / "raw"            # Original unprocessed images
    DATA_PROCESSED = ROOT_DIR / "data" / "processed" # After train/val/test split
    DATA_AUGMENTED = ROOT_DIR / "data" / "augmented"  # Augmented training images

    # Model directories
    MODEL_PRETRAINED = ROOT_DIR / "models" / "pretrained"   # Pre-trained weights
    MODEL_CHECKPOINTS = ROOT_DIR / "models" / "checkpoints" # Training checkpoints

    # Logging
    LOG_DIR = ROOT_DIR / "logs"

    # ==================================================================
    # IMAGE SETTINGS
    # ==================================================================
    IMAGE_SIZE = 224             # ViT expects 224x224 input images
    NUM_CHANNELS = 3             # RGB (even if MRI is grayscale, ViT expects 3 channels)
    NUM_CLASSES = 4              # glioma, meningioma, notumor, pituitary
    CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

    # ==================================================================
    # VISION TRANSFORMER (ViT) SETTINGS
    # ==================================================================
    # Hugging Face model identifier
    # This downloads google/vit-base-patch16-224 (~330 MB) on first use
    VIT_MODEL_NAME = "google/vit-base-patch16-224"

    # Whether to use pre-trained weights (recommended: True)
    VIT_PRETRAINED = True

    # Embedding dimension of the [CLS] token (768 for base ViT)
    VIT_EMBED_DIM = 768

    # Number of early transformer blocks to freeze
    # Freezing preserves pre-trained features and speeds up training
    # Set to 0 to fine-tune all layers, 8-10 for partial fine-tuning
    VIT_FREEZE_LAYERS = 8

    # Dropout rate for the classification head (prevents overfitting)
    VIT_DROPOUT = 0.1

    # Gradient checkpointing: trade ~30% speed for ~40% less GPU memory
    # Enable if you run out of GPU memory during training
    VIT_GRADIENT_CHECKPOINTING = False

    # ==================================================================
    # TRAINING HYPERPARAMETERS
    # ==================================================================
    # These control how the model learns. Tune these for best results.

    # Learning rate: how fast the model updates its weights
    # Too high = unstable training, too low = slow convergence
    LEARNING_RATE = 2e-5          # Good starting point for fine-tuning ViT

    # Number of complete passes through the training data
    NUM_EPOCHS = 20

    # Batch size: number of images processed together
    # Higher = faster training but needs more GPU memory
    BATCH_SIZE_DATA = 16

    # Weight decay: L2 regularization to prevent overfitting
    WEIGHT_DECAY = 0.01

    # Learning rate scheduler: reduces LR when validation loss plateaus
    LR_SCHEDULER_PATIENCE = 3     # Wait this many epochs before reducing LR
    LR_SCHEDULER_FACTOR = 0.5     # Multiply LR by this factor on plateau

    # Early stopping: stop training if validation loss doesn't improve
    EARLY_STOPPING_PATIENCE = 5   # Stop after this many epochs without improvement

    # ==================================================================
    # RL AGENT (PPO) - for later use
    # ==================================================================
    TOTAL_TIMESTEPS = 100_000
    LEARNING_RATE_RL = 3e-4
    N_STEPS = 2048
    BATCH_SIZE_RL = 64
    N_EPOCHS_RL = 10
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.2

    # ==================================================================
    # RL ENVIRONMENT - for later use
    # ==================================================================
    MAX_STEPS_PER_EPISODE = 20
    REWARD_CORRECT = 1.0
    REWARD_WRONG = -1.0
    REWARD_STEP = -0.01

    # ==================================================================
    # DQN RECOMMENDATION AGENT
    # ==================================================================
    # Tumor types recognized by the system
    TUMOR_TYPES = ["glioma", "meningioma", "notumor", "pituitary"]

    # Treatment actions the agent can recommend
    TREATMENT_ACTIONS = [
        "Monitor patient",
        "Recommend specialist",
        "Recommend biopsy",
        "Emergency attention",
    ]

    # Severity of each tumor type (higher = more dangerous)
    # Used by the environment to give rewards
    TUMOR_SEVERITY = {
        "glioma": 3,        # Most aggressive brain tumor
        "meningioma": 2,    # Usually benign but can be serious
        "pituitary": 1,     # Usually benign
        "notumor": 0,       # No tumor found
    }

    # DQN hyperparameters
    DQN_TOTAL_TIMESTEPS = 50_000
    DQN_LEARNING_RATE = 1e-3
    DQN_BUFFER_SIZE = 10_000      # Replay buffer size
    DQN_LEARNING_STARTS = 1_000   # Steps before training begins
    DQN_BATCH_SIZE = 64
    DQN_GAMMA = 0.99              # Discount factor
    DQN_TAU = 1.0                 # Target network update rate
    DQN_TARGET_UPDATE_FREQ = 500  # How often to update target network
    DQN_EXPLORATION_FRACTION = 0.3  # Fraction of training for exploration
    DQN_EXPLORATION_FINAL_EPS = 0.05  # Final exploration rate

    # ==================================================================
    # GENERAL SETTINGS
    # ==================================================================
    NUM_WORKERS = 0              # DataLoader workers (0 = main process, safe on Windows)
    SEED = 42                    # Random seed for reproducibility
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==================================================================
    # STREAMLIT APP
    # ==================================================================
    APP_TITLE = "Brain Tumor Agent"
    APP_ICON = "brain"
