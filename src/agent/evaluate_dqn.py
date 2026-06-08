"""
DQN Agent Evaluation for Treatment Recommendation.

Loads a trained DQN model and evaluates it over many episodes.
Reports accuracy, reward, and per-tumor-type breakdown.

Usage:
    # Evaluate best model
    python -m src.agent.evaluate_dqn

    # Evaluate specific model
    python -m src.agent.evaluate_dqn --model models/checkpoints/dqn_treatment/best/best_model.zip

    # Run more episodes
    python -m src.agent.evaluate_dqn --episodes 500
"""

import argparse
from collections import defaultdict

import numpy as np
from stable_baselines3 import DQN

from src.environment.treatment_env import TreatmentRecommendationEnv
from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger("evaluate_dqn")


def evaluate(model_path: str = None, config: Config = None, num_episodes: int = 200):
    """
    Evaluate a trained DQN agent.

    Args:
        model_path: Path to the trained model file (.zip).
                    If None, uses default best model path.
        config: Config object (uses defaults if None).
        num_episodes: Number of episodes to run.

    Returns:
        dict with evaluation metrics.
    """
    if config is None:
        config = Config()

    # Default model path
    if model_path is None:
        model_path = str(
            config.MODEL_CHECKPOINTS / "dqn_treatment" / "best" / "best_model.zip"
        )

    # Load trained model
    logger.info(f"Loading model from {model_path}")
    model = DQN.load(model_path, device=config.DEVICE)

    # Create environment
    env = TreatmentRecommendationEnv(config)

    # --- Run evaluation episodes ---
    results = {
        "rewards": [],
        "correct": 0,
        "total": 0,
        "per_tumor": defaultdict(lambda: {"correct": 0, "total": 0, "rewards": []}),
        "per_action": defaultdict(lambda: {"count": 0}),
        "confusion": np.zeros((4, 4), dtype=int),  # [true_severity][predicted_action]
    }

    logger.info(f"Running {num_episodes} evaluation episodes...")

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False

        while not done:
            # Get action from model (deterministic - no random exploration)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

        # Record results
        results["rewards"].append(reward)
        results["total"] += 1

        tumor_type = step_info["tumor_type"]
        results["per_tumor"][tumor_type]["total"] += 1
        results["per_tumor"][tumor_type]["rewards"].append(reward)

        results["per_action"][step_info["action_name"]]["count"] += 1

        if step_info["is_correct"]:
            results["correct"] += 1
            results["per_tumor"][tumor_type]["correct"] += 1

        # Update confusion matrix (severity vs action)
        results["confusion"][step_info["severity"]][action] += 1

    # --- Calculate metrics ---
    accuracy = results["correct"] / results["total"]
    avg_reward = np.mean(results["rewards"])

    # Print results
    print("\n" + "=" * 60)
    print("  DQN TREATMENT RECOMMENDATION AGENT - EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nOverall Accuracy: {accuracy:.1%} ({results['correct']}/{results['total']})")
    print(f"Average Reward:   {avg_reward:+.3f}")

    # Per-tumor-type breakdown
    print("\n--- Per Tumor Type ---")
    print(f"{'Tumor Type':<15} {'Accuracy':>10} {'Avg Reward':>12} {'Count':>8}")
    print("-" * 50)
    for tumor in sorted(results["per_tumor"].keys()):
        data = results["per_tumor"][tumor]
        tumor_acc = data["correct"] / data["total"] if data["total"] > 0 else 0
        tumor_reward = np.mean(data["rewards"]) if data["rewards"] else 0
        print(f"{tumor:<15} {tumor_acc:>9.1%} {tumor_reward:>+11.3f} {data['total']:>8}")

    # Action distribution
    print("\n--- Action Distribution ---")
    print(f"{'Action':<25} {'Count':>8} {'Percentage':>12}")
    print("-" * 50)
    for action_name in config.TREATMENT_ACTIONS:
        count = results["per_action"][action_name]["count"]
        pct = count / results["total"] * 100
        print(f"{action_name:<25} {count:>8} {pct:>10.1f}%")

    # Confusion matrix (severity vs action)
    print("\n--- Confusion Matrix (Severity -> Action) ---")
    print("Rows = True Severity, Columns = Recommended Action")
    print(f"{'':>12}", end="")
    for action_name in config.TREATMENT_ACTIONS:
        print(f"{action_name[:8]:>10}", end="")
    print()

    severity_names = ["None(0)", "Pitui(1)", "Mening(2)", "Glioma(3)"]
    for i, sev_name in enumerate(severity_names):
        print(f"{sev_name:>12}", end="")
        for j in range(4):
            print(f"{results['confusion'][i][j]:>10}", end="")
        print()

    print("\n" + "=" * 60)

    # Log results
    logger.info(f"Evaluation: accuracy={accuracy:.1%}, avg_reward={avg_reward:+.3f}")

    return {
        "accuracy": accuracy,
        "avg_reward": avg_reward,
        "total_episodes": num_episodes,
        "per_tumor": dict(results["per_tumor"]),
    }


# --- Run directly ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DQN treatment agent")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to trained model (.zip). Default: best model checkpoint",
    )
    parser.add_argument(
        "--episodes", type=int, default=200,
        help="Number of evaluation episodes (default: 200)",
    )
    args = parser.parse_args()

    config = Config()
    results = evaluate(
        model_path=args.model,
        config=config,
        num_episodes=args.episodes,
    )
