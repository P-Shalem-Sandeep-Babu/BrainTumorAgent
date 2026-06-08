"""
Model Stats Utility.

Measures:
    - Parameter count (total + trainable)
    - Approximate model size on disk (MB)
    - Inference latency (ms per image, median of N runs)

Used by the Streamlit sidebar to give users a quick "this is the
model that's running" overview.
"""

import time
import numpy as np
import torch
from pathlib import Path
from PIL import Image

from src.utils.config import Config
from src.preprocessing.transforms import get_inference_transform


def count_parameters(model) -> dict:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def model_disk_size_mb(path: str) -> float:
    """Get the file size in MB (0.0 if not found)."""
    p = Path(path)
    if not p.exists():
        return 0.0
    return p.stat().st_size / (1024 * 1024)


def measure_latency_ms(
    model,
    config: Config = None,
    n_runs: int = 10,
    warmup: int = 2,
) -> dict:
    """
    Measure median inference latency in milliseconds.

    Args:
        model: PyTorch model
        config: Config object
        n_runs: number of timed runs
        warmup: number of un-timed warm-up runs

    Returns:
        dict with median_ms, mean_ms, min_ms, max_ms, device
    """
    config = config or Config()
    device = config.DEVICE
    model = model.to(device).eval()

    transform = get_inference_transform(config.IMAGE_SIZE)
    # dummy 224x224 RGB image
    dummy = np.random.randint(0, 255, (config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8)
    tensor = transform(image=dummy)["image"].unsqueeze(0).to(device)

    # Warm-up (CUDA async, etc.)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(tensor)

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(tensor)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    return {
        "median_ms": float(np.median(times)),
        "mean_ms": float(np.mean(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "device": device,
        "n_runs": n_runs,
    }


def get_full_model_card(model, checkpoint_path: str, config: Config = None) -> dict:
    """
    Build a complete model info card for the Streamlit sidebar.

    Returns:
        dict with params, size_mb, latency
    """
    config = config or Config()
    params = count_parameters(model)
    return {
        "params": params,
        "size_mb": model_disk_size_mb(checkpoint_path),
        "latency": measure_latency_ms(model, config),
        "config": {
            "model_name": config.VIT_MODEL_NAME,
            "num_classes": config.NUM_CLASSES,
            "image_size": config.IMAGE_SIZE,
        },
    }
