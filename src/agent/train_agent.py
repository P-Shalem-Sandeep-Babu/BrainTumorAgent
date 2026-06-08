"""
RL Agent Training using Stable-Baselines3 (PPO).

Trains a PPO agent in the BrainTumorEnv to learn correct
MRI classification through reward-based learning.

PPO (Proximal Policy Optimization) is chosen because:
- Stable and reliable for discrete action spaces
- Works well with relatively small datasets
- Easy to tune hyperparameters
"""

import os
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger("train_agent")


def create_ppo_agent(env, config: Config) -> PPO:
    """
    Create a PPO agent with hyperparameters from config.

    Args:
        env: Gymnasium environment.
        config: Config object.

    Returns:
        PPO model (not yet trained).
    """
    model = PPO(
        policy="MlpPolicy",          # Simple MLP policy (features -> action)
        env=env,
        learning_rate=config.LEARNING_RATE_RL,
        n_steps=config.N_STEPS,
        batch_size=config.BATCH_SIZE_RL,
        n_epochs=config.N_EPOCHS_RL,
        gamma=config.GAMMA,
        gae_lambda=config.GAE_LAMBDA,
        clip_range=config.CLIP_RANGE,
        verbose=1,
        tensorboard_log=str(config.LOG_DIR),
        seed=config.SEED,
        device=config.DEVICE,
    )

    logger.info(f"PPO agent created on {config.DEVICE}")
    return model


def train(env, config: Config, resume_path: str = None):
    """
    Train the PPO agent.

    Args:
        env: Gymnasium environment.
        config: Config object.
        resume_path: Path to a checkpoint to resume training from (optional).
    """
    # Ensure output directories exist
    config.MODEL_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Wrap env with Monitor for logging
    env = Monitor(env)

    # Create or load agent
    if resume_path and os.path.exists(resume_path):
        logger.info(f"Resuming training from {resume_path}")
        model = PPO.load(resume_path, env=env, device=config.DEVICE)
    else:
        model = create_ppo_agent(env, config)

    # --- Callbacks ---
    # Save a checkpoint every 10,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=str(config.MODEL_CHECKPOINTS),
        name_prefix="ppo_brain_tumor",
    )

    # Evaluate periodically and save best model
    eval_callback = EvalCallback(
        env,
        best_model_save_path=str(config.MODEL_CHECKPOINTS / "best"),
        log_path=str(config.LOG_DIR / "eval"),
        eval_freq=5_000,
        n_eval_episodes=20,
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback])

    # --- Train ---
    logger.info(f"Starting training for {config.TOTAL_TIMESTEPS} timesteps")
    model.learn(
        total_timesteps=config.TOTAL_TIMESTEPS,
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model
    final_path = config.MODEL_CHECKPOINTS / "ppo_brain_tumor_final"
    model.save(str(final_path))
    logger.info(f"Training complete. Final model saved to {final_path}")

    return model


# --- Run directly ---
if __name__ == "__main__":
    print("train_agent.py - To train, run train.py or call train(env, config)")
