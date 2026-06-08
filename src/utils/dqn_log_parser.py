"""
DQN Training Log Parser.

Stable-Baselines3 writes evaluation results to disk via EvalCallback.
The .npz files contain arrays like `timesteps`, `results` (episode rewards),
and `ep_lengths`. This module loads and reshapes them for plotting.
"""

import numpy as np
from pathlib import Path
from typing import Optional


def parse_dqn_eval_log(eval_dir: Path) -> Optional[dict]:
    """
    Parse the DQN evaluation log produced by EvalCallback.

    Looks for `evaluations.npz` inside the eval log directory.

    Returns:
        dict with 'timesteps', 'mean_rewards', 'std_rewards' (each a list),
        or None if no log found.
    """
    eval_dir = Path(eval_dir)
    if not eval_dir.exists():
        return None

    npz_files = sorted(eval_dir.glob("**/evaluations.npz"))
    if not npz_files:
        return None

    # Use the most recent one
    npz_path = npz_files[-1]
    try:
        data = np.load(str(npz_path))
        timesteps = data["timesteps"].tolist()
        # 'results' is shape (n_evals, n_eval_episodes)
        results = data["results"]
        mean_rewards = results.mean(axis=1).tolist()
        std_rewards = results.std(axis=1).tolist()
        return {
            "timesteps": timesteps,
            "mean_rewards": mean_rewards,
            "std_rewards": std_rewards,
            "source_path": str(npz_path),
        }
    except Exception:
        return None


def get_dqn_eval_path(log_dir: Path) -> Path:
    """Return the conventional path for DQN eval logs."""
    return Path(log_dir) / "eval_dqn"
