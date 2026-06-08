"""
DQN Agent Training for Treatment Recommendation.

Trains a DQN (Deep Q-Network) agent to recommend treatments
based on tumor type and confidence score.

Why DQN?
- Works well for discrete action spaces (4 treatment actions)
- Simple to understand and implement
- Uses experience replay for stable learning
- Good for single-step decision problems

Usage:
    # Train from scratch
    python -m src.agent.train_dqn

    # Resume from checkpoint
    python -m src.agent.train_dqn --resume models/checkpoints/dqn_treatment/best/best_model.zip
"""

import argparse
import os
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

from src.environment.treatment_env import TreatmentRecommendationEnv
from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger("train_dqn")


def create_dqn_agent(env, config: Config) -> DQN:
    """
    Create a DQN agent with hyperparameters from config.

    Args:
        env: Gymnasium environment.
        config: Config object.

    Returns:
        DQN model (not yet trained).
    """
    model = DQN(
        policy="MlpPolicy",          # Simple MLP: observation -> Q-values
        env=env,
        learning_rate=config.DQN_LEARNING_RATE,
        buffer_size=config.DQN_BUFFER_SIZE,
        learning_starts=config.DQN_LEARNING_STARTS,
        batch_size=config.DQN_BATCH_SIZE,
        gamma=config.DQN_GAMMA,
        tau=config.DQN_TAU,
        target_update_interval=config.DQN_TARGET_UPDATE_FREQ,
        exploration_fraction=config.DQN_EXPLORATION_FRACTION,
        exploration_final_eps=config.DQN_EXPLORATION_FINAL_EPS,
        verbose=1,
        tensorboard_log=str(config.LOG_DIR),
        seed=config.SEED,
        device=config.DEVICE,
    )

    logger.info(f"DQN agent created on {config.DEVICE}")
    return model


def train(config: Config = None, resume_path: str = None):
    """
    Train the DQN recommendation agent.

    Args:
        config: Config object (uses defaults if None).
        resume_path: Path to checkpoint to resume from (optional).

    Returns:
        Trained DQN model.
    """
    if config is None:
        config = Config()

    # Ensure output directories exist
    save_dir = config.MODEL_CHECKPOINTS / "dqn_treatment"
    save_dir.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Create environment
    env = TreatmentRecommendationEnv(config)
    env = Monitor(env)  # Monitor wraps env for logging

    # Create evaluation environment (separate from training)
    eval_env = TreatmentRecommendationEnv(config)
    eval_env = Monitor(eval_env)

    # Create or load agent
    if resume_path and os.path.exists(resume_path):
        logger.info(f"Resuming training from {resume_path}")
        model = DQN.load(resume_path, env=env, device=config.DEVICE)
    else:
        model = create_dqn_agent(env, config)

    # --- Callbacks ---
    # Save checkpoint every 10,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=str(save_dir),
        name_prefix="dqn_treatment",
    )

    # Evaluate every 5,000 steps and save best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir / "best"),
        log_path=str(config.LOG_DIR / "eval_dqn"),
        eval_freq=5_000,
        n_eval_episodes=50,
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback])

    # --- Train ---
    logger.info(f"Starting DQN training for {config.DQN_TOTAL_TIMESTEPS} timesteps")
    logger.info(f"Hyperparameters: lr={config.DQN_LEARNING_RATE}, "
                f"gamma={config.DQN_GAMMA}, buffer={config.DQN_BUFFER_SIZE}")

    model.learn(
        total_timesteps=config.DQN_TOTAL_TIMESTEPS,
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model
    final_path = save_dir / "dqn_treatment_final"
    model.save(str(final_path))
    logger.info(f"Training complete! Final model saved to {final_path}")

    return model


# --- Run directly ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN treatment recommendation agent")
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from",
    )
    args = parser.parse_args()

    config = Config()
    model = train(config=config, resume_path=args.resume)
