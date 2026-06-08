"""
Agent Evaluation.

Loads a trained PPO agent and evaluates it on the test set,
computing accuracy, precision, recall, F1-score, and
generating a confusion matrix.
"""

import numpy as np
import torch
from pathlib import Path
from stable_baselines3 import PPO
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger("evaluate_agent")


def evaluate(model_path: str, env, config: Config, n_episodes: int = 100):
    """
    Evaluate a trained agent.

    Args:
        model_path: Path to the saved PPO model (.zip).
        env: Gymnasium environment.
        config: Config object.
        n_episodes: Number of evaluation episodes.

    Returns:
        dict with metrics (accuracy, precision, recall, f1, confusion_matrix).
    """
    # Load trained model
    model = PPO.load(model_path, device=config.DEVICE)
    logger.info(f"Loaded model from {model_path}")

    all_true = []
    all_pred = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False

        while not done:
            # Agent predicts action (no exploration noise during evaluation)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        all_true.append(info["true_label"])
        all_pred.append(info["predicted_label"])

    # Compute metrics
    accuracy = accuracy_score(all_true, all_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true, all_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(all_true, all_pred, labels=list(range(config.NUM_CLASSES)))

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "n_episodes": n_episodes,
    }

    # Print report
    logger.info(f"\nEvaluation Results ({n_episodes} episodes):")
    logger.info(f"  Accuracy:  {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1 Score:  {f1:.4f}")
    logger.info(f"\n{classification_report(all_true, all_pred, target_names=config.CLASS_NAMES)}")

    return results


# --- Run directly ---
if __name__ == "__main__":
    print("evaluate_agent.py - To evaluate, call evaluate(model_path, env, config)")
