"""
Treatment Recommendation RL Environment.

A custom Gymnasium environment where the RL agent observes a patient's
tumor type and confidence score, then recommends a treatment action.

How it works:
    1. Environment generates a patient case (tumor type + confidence)
    2. Agent sees [tumor_type, confidence] as observation
    3. Agent picks a treatment action (0-3)
    4. Environment gives reward based on severity match
    5. Episode ends after one recommendation

Observation space: [tumor_type (0-3), confidence (0.0-1.0)]
Action space: 0=Monitor, 1=Specialist, 2=Biopsy, 3=Emergency
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.utils.config import Config


class TreatmentRecommendationEnv(gym.Env):
    """
    Custom environment for treatment recommendation.

    The agent learns to recommend appropriate treatments based on
    tumor type and confidence score from a classifier (e.g., ViT).

    Reward logic:
        - Correct action for the severity level: +1.0
        - Close but not ideal: +0.5
        - Wrong but not dangerous: -0.5
        - Dangerous mismatch: -1.0

    Example:
        Glioma (severity 3) -> Emergency attention = +1.0 (correct)
        Glioma (severity 3) -> Recommend biopsy   = +0.5 (close)
        Glioma (severity 3) -> Monitor patient     = -1.0 (dangerous!)
        Notumor (severity 0) -> Monitor patient    = +1.0 (correct)
    """

    metadata = {"render_modes": ["human"]}

    # The "correct" action for each severity level
    # severity 0 (notumor)    -> Monitor (0)
    # severity 1 (pituitary)  -> Specialist (1)
    # severity 2 (meningioma) -> Biopsy (2)
    # severity 3 (glioma)     -> Emergency (3)
    SEVERITY_TO_ACTION = {0: 0, 1: 1, 2: 2, 3: 3}

    def __init__(self, config: Config = None):
        super().__init__()

        if config is None:
            config = Config()
        self.config = config

        # --- Action Space ---
        # 4 treatment actions
        self.action_space = spaces.Discrete(4)

        # --- Observation Space ---
        # [tumor_type (0-3), confidence (0.0-1.0)]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([3.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Internal state
        self.tumor_type = 0       # Current patient's tumor type (0-3)
        self.confidence = 0.0     # Classifier's confidence (0-1)
        self.severity = 0         # Severity level (derived from tumor type)
        self.steps = 0

    def _generate_patient_case(self):
        """
        Generate a random patient case.

        In a real system, this would come from the ViT classifier.
        Here we simulate it for training the RL agent.

        Returns:
            tumor_type: int (0-3)
            confidence: float (0.0-1.0)
        """
        # Pick a random tumor type (0=notumor, 1=pituitary, 2=meningiary, 3=glioma)
        tumor_type = self.np_random.integers(0, 4)

        # Generate confidence score
        # Higher severity tumors tend to have more variable confidence
        # This makes training more realistic
        base_confidence = self.np_random.uniform(0.5, 1.0)
        noise = self.np_random.uniform(-0.1, 0.1)
        confidence = np.clip(base_confidence + noise, 0.0, 1.0)

        return int(tumor_type), float(confidence)

    def reset(self, seed=None, options=None):
        """
        Reset environment for a new patient case.

        Returns:
            observation: numpy array [tumor_type, confidence]
            info: dict with patient details
        """
        super().reset(seed=seed)

        # Generate a new patient case
        self.tumor_type, self.confidence = self._generate_patient_case()
        self.severity = self.config.TUMOR_SEVERITY[
            self.config.TUMOR_TYPES[self.tumor_type]
        ]
        self.steps = 0

        # Build observation
        obs = np.array(
            [float(self.tumor_type), self.confidence],
            dtype=np.float32,
        )

        info = {
            "tumor_type": self.config.TUMOR_TYPES[self.tumor_type],
            "tumor_type_idx": self.tumor_type,
            "confidence": self.confidence,
            "severity": self.severity,
        }

        return obs, info

    def step(self, action: int):
        """
        Take an action (recommend treatment) and get reward.

        Reward logic based on severity vs action match:

        Severity 3 (glioma - most dangerous):
            Emergency (3) -> +1.0  (correct!)
            Biopsy (2)    -> +0.5  (close, but should be emergency)
            Specialist (1)-> -0.5  (too slow for aggressive tumor)
            Monitor (0)   -> -1.0  (dangerous! could delay treatment)

        Severity 2 (meningioma):
            Biopsy (2)    -> +1.0  (correct!)
            Emergency (3) -> +0.5  (overcautious but safe)
            Specialist (1)-> +0.5  (acceptable)
            Monitor (0)   -> -0.5  (risky)

        Severity 1 (pituitary):
            Specialist (1)-> +1.0  (correct!)
            Monitor (0)   -> +0.5  (could work but not ideal)
            Biopsy (2)    -> -0.5  (unnecessary procedure)
            Emergency (3) -> -1.0  (wastes emergency resources)

        Severity 0 (no tumor):
            Monitor (0)   -> +1.0  (correct!)
            Specialist (1)-> +0.5  (cautious but fine)
            Biopsy (2)    -> -1.0  (unnecessary invasive procedure)
            Emergency (3) -> -1.0  (wastes emergency resources)

        Args:
            action: int (0-3), the treatment recommendation

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.steps += 1

        # Calculate reward based on severity-action match
        reward = self._calculate_reward(action, self.severity)

        # Episode always ends after one recommendation (single-step)
        terminated = True
        truncated = False

        # Check if the action matches the "ideal" action for this severity
        ideal_action = self.SEVERITY_TO_ACTION[self.severity]
        is_correct = (action == ideal_action)

        info = {
            "tumor_type": self.config.TUMOR_TYPES[self.tumor_type],
            "tumor_type_idx": self.tumor_type,
            "confidence": self.confidence,
            "severity": self.severity,
            "action": action,
            "action_name": self.config.TREATMENT_ACTIONS[action],
            "ideal_action": ideal_action,
            "ideal_action_name": self.config.TREATMENT_ACTIONS[ideal_action],
            "is_correct": is_correct,
            "reward": reward,
        }

        obs = np.array(
            [float(self.tumor_type), self.confidence],
            dtype=np.float32,
        )

        return obs, reward, terminated, truncated, info

    def _calculate_reward(self, action: int, severity: int) -> float:
        """
        Calculate reward based on how well the action matches the severity.

        Args:
            action: The agent's chosen action (0-3)
            severity: The tumor's severity level (0-3)

        Returns:
            reward: float
        """
        ideal_action = self.SEVERITY_TO_ACTION[severity]
        diff = abs(action - ideal_action)

        if diff == 0:
            # Perfect match
            return 1.0
        elif diff == 1:
            # Close enough (e.g., biopsy instead of emergency)
            return 0.5
        elif diff == 2:
            # Not great (e.g., specialist for glioma)
            return -0.5
        else:
            # Way off (e.g., monitor for glioma or emergency for notumor)
            return -1.0

    def render(self):
        """Print the current patient case to console."""
        tumor_name = self.config.TUMOR_TYPES[self.tumor_type]
        print(
            f"Patient: {tumor_name} | "
            f"Confidence: {self.confidence:.2f} | "
            f"Severity: {self.severity}"
        )


# --- Quick test ---
if __name__ == "__main__":
    print("=== TreatmentRecommendationEnv Quick Test ===\n")

    env = TreatmentRecommendationEnv()

    # Run 5 random episodes
    for i in range(5):
        obs, info = env.reset()
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, step_info = env.step(action)

        print(f"Episode {i+1}:")
        print(f"  Patient: {info['tumor_type']} (severity {info['severity']})")
        print(f"  Confidence: {info['confidence']:.2f}")
        print(f"  Action: {step_info['action_name']}")
        print(f"  Ideal:  {step_info['ideal_action_name']}")
        print(f"  Reward: {reward:+.1f} {'CORRECT' if step_info['is_correct'] else 'WRONG'}")
        print()

    print("Environment works! Ready for training.")
