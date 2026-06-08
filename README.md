---
title: Brain Tumor Agent
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.28.0
app_file: app/main.py
pinned: false
---

# Reinforcement Learning Based Brain Tumor Agent using Vision Transformers

A minor project that combines **Reinforcement Learning** with **Vision Transformers (ViT)** to build an intelligent agent capable of detecting and localizing brain tumors from MRI scans.

## Project Overview

The agent uses a Vision Transformer as the backbone for feature extraction from brain MRI images. A reinforcement learning agent (PPO via Stable-Baselines3) learns to navigate the image, focusing on regions of interest to classify and localize tumors.

```
MRI Image --> ViT Feature Extractor --> RL Agent (PPO) --> Tumor Detection / Localization
```

## Project Structure

```
Brain Tumor Agent/
|
|-- data/                       # All dataset-related files
|   |-- raw/                    # Original unprocessed MRI scans
|   |-- processed/              # Cleaned, resized, normalized images
|   |-- augmented/              # Augmented training images
|
|-- models/                     # Saved model weights
|   |-- pretrained/             # Pre-trained ViT weights (e.g., from ImageNet)
|   |-- checkpoints/            # Training checkpoints from RL agent
|
|-- src/                        # All source code
|   |-- preprocessing/          # Data loading and image preprocessing
|   |   |-- __init__.py
|   |   |-- dataset.py          # Dataset class for loading MRI images
|   |   |-- transforms.py       # Image transformations and augmentations
|   |
|   |-- model/                  # Vision Transformer model definition
|   |   |-- __init__.py
|   |   |-- vit_model.py        # ViT architecture for feature extraction
|   |
|   |-- environment/            # RL environment (Gymnasium-based)
|   |   |-- __init__.py
|   |   |-- brain_tumor_env.py  # Custom Gym env wrapping the MRI dataset
|   |
|   |-- agent/                  # RL agent logic
|   |   |-- __init__.py
|   |   |-- train_agent.py      # Training loop using Stable-Baselines3
|   |   |-- evaluate_agent.py   # Evaluation and metrics
|   |
|   |-- utils/                  # Shared helper functions
|       |-- __init__.py
|       |-- config.py           # Hyperparameters and paths
|       |-- logger.py           # Training logger
|       |-- visualizer.py       # Plotting and visualization utilities
|
|-- app/                        # Streamlit frontend
|   |-- __init__.py
|   |-- main.py                 # Main Streamlit app entry point
|   |-- pages/                  # Multi-page Streamlit app
|       |-- 1_Dataset.py        # Dataset viewer page
|       |-- 2_Training.py       # Training dashboard page
|       |-- 3_Prediction.py     # Single image prediction page
|
|-- tests/                      # Unit tests
|   |-- test_preprocessing.py
|   |-- test_model.py
|   |-- test_environment.py
|
|-- notebooks/                  # Jupyter notebooks for exploration
|   |-- 01_data_exploration.ipynb
|   |-- 02_model_testing.ipynb
|
|-- assets/                     # Static assets
|   |-- screenshots/            # App screenshots for documentation
|
|-- logs/                       # Training logs (TensorBoard, etc.)
|
|-- requirements.txt            # Python dependencies
|-- setup.py                    # Package setup (optional)
|-- .gitignore                  # Git ignore rules
|-- train.py                    # Top-level training script
|-- run_app.py                  # Top-level Streamlit launcher
```

## Setup Instructions

### 1. Clone the repository
```bash
git clone <repo-url>
cd "Brain Tumor Agent"
```

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Download a brain tumor MRI dataset (e.g., from Kaggle: "Brain Tumor MRI Dataset") and place it in `data/raw/`.

Expected structure:
```
data/raw/
  |-- glioma/
  |-- meningioma/
  |-- notumor/
  |-- pituitary/
```

### 5. Preprocess the data
```bash
python -m src.preprocessing.dataset
```

### 6. Train the model
```bash
python train.py
```

### 7. Launch the web app
```bash
streamlit run app/main.py
```

## Tech Stack

| Component            | Technology              |
|----------------------|-------------------------|
| Language             | Python 3.10+            |
| Deep Learning        | PyTorch                 |
| Vision Model         | Vision Transformer (ViT)|
| Reinforcement Learning| Stable-Baselines3      |
| RL Environment       | Gymnasium               |
| Frontend             | Streamlit               |
| Logging              | TensorBoard             |

## License

This is a minor project submission.
