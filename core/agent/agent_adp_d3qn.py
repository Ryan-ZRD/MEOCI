import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import Dict, Tuple

from core.agent.network import DoubleDuelingDQN
from core.agent.replay_buffer import DualPoolReplayBuffer
from core.agent.epsilon_scheduler import AdaptiveEpsilonScheduler
from utils.logger import ExperimentLogger


class ADP_D3QNAgent:

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 device: str = "cpu",
                 gamma: float = 0.98,
                 lr: float = 1e-4,
                 batch_size: int = 64,
                 target_update_freq: int = 50,
                 buffer_capacity: int = 8000,
                 logger: ExperimentLogger = None):
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.global_step = 0
        self.logger = logger or Logger()

        # ---- Core components ----
        self.q_network = DoubleDuelingDQN(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.q_network.online_net.parameters(), lr=lr)

        self.replay_buffer = DualPoolReplayBuffer(device=device)
        self.epsilon_scheduler = AdaptiveEpsilonScheduler()

        # Statistics
        self.loss_fn = nn.SmoothL1Loss()
        self.training_loss = []
        self.rewards_record = []

    # ------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------
    def select_action(self, state: np.ndarray, env_stats: Dict) -> int:
        """
        Select action according to adaptive ε-greedy policy.

        :param state: current environment state
        :param env_stats: {'bandwidth_var': float, 'load_factor': float}
        :return: chosen action index
        """
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        epsilon = self.epsilon_scheduler.get_epsilon()

        # Exploration or exploitation
        if random.random() < epsilon:
            action = random.randint(0, self.q_network.online_net.action_dim - 1)
        else:
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action = int(torch.argmax(q_values, dim=1).item())

        if self.logger:
            self.logger.log_scalar("epsilon", epsilon, self.global_step)
        return action

    # ------------------------------------------------------------
    # Transition storage
    # ------------------------------------------------------------
    def store_transition(self, state, action, reward, next_state, done, td_error):
        """Store new experience into dual replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done, td_error)

    # ------------------------------------------------------------
    # TD target computation
    # ------------------------------------------------------------
    def compute_td_target(self, rewards, next_states, dones):
        """
        Compute TD target using Double DQN:
            y_i = r_i + γ * Q_target(s', argmax_a' Q_online(s', a'))
        """
        with torch.no_grad():
            q_next_online = self.q_network.online_net(next_states)
            best_actions = torch.argmax(q_next_online, dim=1, keepdim=True)

            q_next_target = self.q_network.target_net(next_states)
            target_q = q_next_target.gather(1, best_actions).squeeze(1)

            td_target = rewards + self.gamma * (1 - dones) * target_q
        return td_target

    # ------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------
    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Compute current Q estimates
        q_values = self.q_network.online_net(states)
        q_pred = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute TD targets
        td_target = self.compute_td_target(rewards, next_states, dones)

        # TD error
        td_error = td_target - q_pred

        # Compute loss
        loss = self.loss_fn(q_pred, td_target)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.online_net.parameters(), 1.0)
        self.optimizer.step()

        # Update target network periodically
        if self.global_step % self.target_update_freq == 0:
            self.q_network.soft_update()

        # Adaptive exploration update (ε)
        self.epsilon_scheduler.update(
            reward=float(rewards.mean().item()),
            bandwidth_var=np.random.rand() * 0.1,  # 可替换为真实环境波动
            load_factor=np.random.uniform(0.6, 1.0)
        )

        # Record & log
        self.training_loss.append(loss.item())
        self.rewards_record.append(float(rewards.mean().item()))
        if self.logger:
            self.logger.log_scalar("loss", loss.item(), self.global_step)
            self.logger.log_scalar("reward_mean", float(rewards.mean().item()), self.global_step)
            self.logger.log_scalar("lambda_t", self.replay_buffer.lambda_t, self.global_step)

        self.global_step += 1
        return loss.item(), float(rewards.mean().item())

    # ------------------------------------------------------------
    # Save & load
    # ------------------------------------------------------------
    def save(self, path: str):
        checkpoint = {
            "model_state_dict": self.q_network.online_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon_scheduler.get_epsilon(),
            "step": self.global_step
        }
        torch.save(checkpoint, path)
        if self.logger:
            self.logger.info(f"Agent saved at {path}")

    def load(self, path: str, map_location=None):
        checkpoint = torch.load(path, map_location=map_location)
        self.q_network.online_net.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("step", 0)
        if self.logger:
            self.logger.info(f"Loaded ADP-D3QN agent from {path}")

    # ------------------------------------------------------------
    # Evaluation (inference only)
    # ------------------------------------------------------------
    def evaluate(self, state: np.ndarray) -> Tuple[int, float]:
        """Evaluate deterministic policy (ε = 0)"""
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = int(torch.argmax(q_values, dim=1).item())
            q_val = float(torch.max(q_values).item())
        return action, q_val

    # ------------------------------------------------------------
    # Logging utilities
    # ------------------------------------------------------------
    def get_training_stats(self):
        return {
            "mean_loss": np.mean(self.training_loss[-50:]) if len(self.training_loss) > 0 else 0.0,
            "mean_reward": np.mean(self.rewards_record[-50:]) if len(self.rewards_record) > 0 else 0.0,
            "epsilon": self.epsilon_scheduler.get_epsilon(),
            "lambda_t": self.replay_buffer.lambda_t
        }


if __name__ == "__main__":
    import numpy as np
    from utils.logger import Logger

    agent = ADP_D3QNAgent(state_dim=4, action_dim=8, device="cpu", logger=Logger())
    for i in range(10):
        state = np.random.rand(4)
        action = agent.select_action(state, {"bandwidth_var": 0.1, "load_factor": 0.8})
        next_state = np.random.rand(4)
        reward = np.random.randn()
        done = False
        td_error = np.random.randn()
        agent.store_transition(state, action, reward, next_state, done, td_error)
        result = agent.train_step()
        print(f"Step {i}: Loss={result[0] if result else None:.4f}")
    print("Final Stats:", agent.get_training_stats())
