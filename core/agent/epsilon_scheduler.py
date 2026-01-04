import numpy as np
from collections import deque


class AdaptiveEpsilonScheduler:

    def __init__(self,
                 epsilon_max: float = 1.0,
                 epsilon_min: float = 0.05,
                 decay_rate: float = 0.995,
                 reward_window: int = 50,
                 beta: float = 0.3,
                 adaptive_env: bool = True):
        """
        Parameters
        ----------
        epsilon_max : float
            Initial maximum exploration rate.
        epsilon_min : float
            Minimum exploration rate (lower bound).
        decay_rate : float
            Exponential decay factor for time-based decay.
        reward_window : int
            Moving window size for reward variance tracking.
        beta : float
            Scaling factor for reward variance adjustment.
        adaptive_env : bool
            Whether to include environment volatility modulation (η(t)).
        """
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.decay_rate = decay_rate
        self.beta = beta
        self.reward_window = reward_window
        self.adaptive_env = adaptive_env

        # Internal states
        self.step_count = 0
        self.reward_history = deque(maxlen=reward_window)
        self.epsilon = epsilon_max

        # Environment dynamic variables (bandwidth / load)
        self.env_bandwidth = 1.0
        self.env_load = 1.0

    def update(self, reward: float, bandwidth_var: float = 0.0, load_factor: float = 1.0):
        """
        Update ε based on reward and environment feedback.
        """
        self.step_count += 1
        self.reward_history.append(reward)

        # (1) Base exponential decay
        base_eps = self.epsilon_max * (self.decay_rate ** self.step_count)

        # (2) Reward variance modulation (stability-sensitive)
        if len(self.reward_history) > 1:
            reward_std = np.std(self.reward_history)
        else:
            reward_std = 0.0

        adaptive_eps = base_eps + self.beta * reward_std

        # (3) Environment volatility modulation
        if self.adaptive_env:
            eta_t = np.clip(bandwidth_var * 0.5 + (1.0 - load_factor) * 0.3, 0, 1)
            adaptive_eps *= (1 + 0.5 * eta_t)

        # Clamp to [ε_min, ε_max]
        self.epsilon = float(np.clip(adaptive_eps, self.epsilon_min, self.epsilon_max))
        return self.epsilon

    def get_epsilon(self) -> float:
        """Return current ε value"""
        return self.epsilon

    def reset(self):
        """Reset scheduler"""
        self.step_count = 0
        self.reward_history.clear()
        self.epsilon = self.epsilon_max

    def summary(self) -> dict:
        """Return current scheduler stats for logging"""
        return {
            "epsilon": round(self.epsilon, 4),
            "step": self.step_count,
            "reward_std": round(np.std(self.reward_history), 4) if len(self.reward_history) > 1 else 0.0,
        }


if __name__ == "__main__":
    scheduler = AdaptiveEpsilonScheduler(decay_rate=0.995, beta=0.25)
    rewards = np.sin(np.linspace(0, 10, 100)) * 5  # simulate oscillating reward
    eps_values = []

    for t, r in enumerate(rewards):
        eps = scheduler.update(reward=r, bandwidth_var=np.random.rand() * 0.2, load_factor=np.random.rand())
        eps_values.append(eps)

    print(f"Final ε after 100 steps: {scheduler.get_epsilon():.4f}")
    print("Example stats:", scheduler.summary())
