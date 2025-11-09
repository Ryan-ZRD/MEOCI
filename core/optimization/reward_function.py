import numpy as np
from typing import Dict


class RewardFunction:
    """
    RewardFunction
    ==========================================================
    Computes the scalar reward for ADP-D3QN agent based on
    system performance metrics (delay, energy, accuracy, completion).

    Reference:
        Section 3.6 - Reward Function Design
        Eq. (21)–(22) in the MEOCI paper
    """

    def __init__(self,
                 w_accuracy: float = 0.4,
                 w_delay: float = 0.3,
                 w_energy: float = 0.2,
                 w_completion: float = 0.1,
                 delay_ref_ms: float = 100.0,
                 energy_ref_mJ: float = 50.0,
                 reward_clip: bool = True):
        """
        Args:
            w_*: Weight coefficients for different objectives.
            delay_ref_ms: Normalization reference for delay.
            energy_ref_mJ: Normalization reference for energy.
            reward_clip: Whether to clip reward to [-1, 1].
        """
        w_sum = w_accuracy + w_delay + w_energy + w_completion
        self.w_accuracy = w_accuracy / w_sum
        self.w_delay = w_delay / w_sum
        self.w_energy = w_energy / w_sum
        self.w_completion = w_completion / w_sum

        self.delay_ref_ms = delay_ref_ms
        self.energy_ref_mJ = energy_ref_mJ
        self.reward_clip = reward_clip

    # ------------------------------------------------------------
    # Compute normalized reward
    # ------------------------------------------------------------
    def compute_reward(self, metrics: Dict[str, float]) -> float:
        """
        Compute reward according to Eq. (21)–(22).

        Args:
            metrics: dict containing
                {
                    "accuracy": float,        # [0,1]
                    "latency_ms": float,
                    "energy_mJ": float,
                    "completion_rate": float  # [0,1]
                }

        Returns:
            reward (float)
        """
        acc = np.clip(metrics.get("accuracy", 0.0), 0.0, 1.0)
        delay = max(1e-3, metrics.get("latency_ms", self.delay_ref_ms))
        energy = max(1e-3, metrics.get("energy_mJ", self.energy_ref_mJ))
        completion = np.clip(metrics.get("completion_rate", 1.0), 0.0, 1.0)

        # Normalized terms (smaller is better for delay/energy)
        norm_delay = delay / self.delay_ref_ms
        norm_energy = energy / self.energy_ref_mJ

        # Eq. (21) composite reward
        reward = (self.w_accuracy * acc
                  - self.w_delay * norm_delay
                  - self.w_energy * norm_energy
                  + self.w_completion * completion)

        # Optional reward clipping
        if self.reward_clip:
            reward = np.clip(reward, -1.0, 1.0)

        return float(reward)

    # ------------------------------------------------------------
    # Compute dense reward for learning stability
    # ------------------------------------------------------------
    def dense_reward(self,
                     prev_metrics: Dict[str, float],
                     curr_metrics: Dict[str, float]) -> float:
        """
        Compute difference-based dense reward (Eq. 22)
        ΔR_t = R_t - R_{t-1} for smooth training signal.
        """
        prev_r = self.compute_reward(prev_metrics)
        curr_r = self.compute_reward(curr_metrics)
        delta_r = curr_r - prev_r
        return float(np.clip(delta_r, -1.0, 1.0))

    # ------------------------------------------------------------
    # Diagnostic utility
    # ------------------------------------------------------------
    def debug_summary(self, metrics: Dict[str, float]) -> Dict:
        """Return per-term contribution breakdown."""
        acc = np.clip(metrics.get("accuracy", 0.0), 0.0, 1.0)
        delay = metrics.get("latency_ms", self.delay_ref_ms)
        energy = metrics.get("energy_mJ", self.energy_ref_mJ)
        completion = metrics.get("completion_rate", 1.0)

        norm_delay = delay / self.delay_ref_ms
        norm_energy = energy / self.energy_ref_mJ

        terms = {
            "accuracy_term": round(self.w_accuracy * acc, 4),
            "delay_penalty": round(-self.w_delay * norm_delay, 4),
            "energy_penalty": round(-self.w_energy * norm_energy, 4),
            "completion_bonus": round(self.w_completion * completion, 4),
        }
        terms["reward_total"] = round(sum(terms.values()), 4)
        return terms


# ✅ Example quick test
if __name__ == "__main__":
    rf = RewardFunction()
    curr = {"accuracy": 0.85, "latency_ms": 80, "energy_mJ": 30, "completion_rate": 0.95}
    prev = {"accuracy": 0.80, "latency_ms": 100, "energy_mJ": 50, "completion_rate": 0.90}
    print("Reward:", rf.compute_reward(curr))
    print("ΔReward:", rf.dense_reward(prev, curr))
    print("Breakdown:", rf.debug_summary(curr))
