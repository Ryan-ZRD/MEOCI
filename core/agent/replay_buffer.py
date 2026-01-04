import random
import numpy as np
import torch
from collections import deque, namedtuple


class ReplayBuffer:

    def __init__(self, capacity: int, device: str = "cpu"):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.device = device
        self.experience = namedtuple("Experience",
                                     ["state", "action", "reward", "next_state", "done", "td_error"])

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done, td_error=0.0):
        exp = self.experience(state, action, reward, next_state, done, td_error)
        self.buffer.append(exp)

    def sample(self, batch_size: int):
        """Uniform sampling from replay buffer"""
        batch = random.sample(self.buffer, batch_size)
        return self._batch_to_tensors(batch)

    def _batch_to_tensors(self, batch):
        """Convert batch list to torch tensors"""
        states = torch.tensor(np.array([b.state for b in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array([b.action for b in batch]), dtype=torch.long, device=self.device)
        rewards = torch.tensor(np.array([b.reward for b in batch]), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array([b.next_state for b in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array([b.done for b in batch]), dtype=torch.float32, device=self.device)
        return states, actions, rewards, next_states, dones


class DualPoolReplayBuffer:

    def __init__(self,
                 capacity_e1: int = 5000,
                 capacity_e2: int = 5000,
                 device: str = "cpu",
                 initial_lambda: float = 0.7,
                 min_lambda: float = 0.2,
                 decay_rate: float = 0.9995):
        """
        :param capacity_e1: Capacity of exploration pool
        :param capacity_e2: Capacity of exploitation pool
        :param device: torch device
        :param initial_lambda: Initial sampling weight for exploration pool
        :param min_lambda: Minimum sampling ratio for E1
        :param decay_rate: Decay factor for lambda(t)
        """
        self.device = device
        self.pool_e1 = ReplayBuffer(capacity_e1, device)
        self.pool_e2 = ReplayBuffer(capacity_e2, device)

        self.lambda_t = initial_lambda
        self.min_lambda = min_lambda
        self.decay_rate = decay_rate
        self.update_step = 0

    def __len__(self):
        return len(self.pool_e1) + len(self.pool_e2)

    def push(self, state, action, reward, next_state, done, td_error):
        """
        Insert new transition into E1 or E2.
        Routing logic:
            - High |TD-error| → Exploration pool (E1)
            - Low |TD-error| → Exploitation pool (E2)
        """
        if abs(td_error) > 0.5:  # threshold can be tuned per environment
            self.pool_e1.push(state, action, reward, next_state, done, td_error)
        else:
            self.pool_e2.push(state, action, reward, next_state, done, td_error)

    def sample(self, batch_size: int):
        """
        Sample a batch with adaptive ratio λ(t):
            batch = λ * E1 + (1-λ) * E2
        """
        self._decay_lambda()

        if len(self.pool_e1) < 1 or len(self.pool_e2) < 1:
            # fallback to uniform sampling if insufficient samples
            combined = list(self.pool_e1.buffer) + list(self.pool_e2.buffer)
            batch = random.sample(combined, min(batch_size, len(combined)))
            return self.pool_e1._batch_to_tensors(batch)

        n_e1 = int(batch_size * self.lambda_t)
        n_e2 = batch_size - n_e1

        batch_e1 = random.sample(self.pool_e1.buffer, min(n_e1, len(self.pool_e1)))
        batch_e2 = random.sample(self.pool_e2.buffer, min(n_e2, len(self.pool_e2)))
        batch = batch_e1 + batch_e2

        return self.pool_e1._batch_to_tensors(batch)

    def _decay_lambda(self):
        """Exponential decay of λ(t) per training step"""
        self.update_step += 1
        self.lambda_t = max(self.min_lambda, self.lambda_t * self.decay_rate)

    def stats(self):
        """Return statistics for visualization/logging"""
        total = len(self)
        return {
            "λ(t)": round(self.lambda_t, 4),
            "E1 size": len(self.pool_e1),
            "E2 size": len(self.pool_e2),
            "E1 ratio": len(self.pool_e1) / total if total > 0 else 0
        }

    def clear(self):
        """Empty both buffers"""
        self.pool_e1.buffer.clear()
        self.pool_e2.buffer.clear()
        self.lambda_t = 0.7
        self.update_step = 0


if __name__ == "__main__":
    buffer = DualPoolReplayBuffer(device="cpu")

    # Fake samples
    for i in range(200):
        s, a, r, ns, d = np.random.rand(4), np.random.randint(0, 5), np.random.randn(), np.random.rand(4), False
        td = np.random.randn()
        buffer.push(s, a, r, ns, d, td)

    print("Buffer Stats:", buffer.stats())

    batch = buffer.sample(16)
    print("Sample shapes:", [b.shape for b in batch])
