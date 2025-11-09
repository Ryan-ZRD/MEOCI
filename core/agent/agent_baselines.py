import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from core.agent.network import DoubleDuelingDQN
from core.agent.replay_buffer import DualPoolReplayBuffer
from core.agent.epsilon_scheduler import AdaptiveEpsilonScheduler
from utils.logger import ExperimentLogger


class BaseAgent:
    """
    Base class for DQN-style agents.
    Provides shared utilities for action selection and model management.
    """
    def __init__(self, state_dim, action_dim, device="cpu", gamma=0.98, lr=1e-4, batch_size=64, target_update_freq=50):
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.global_step = 0

        self.q_network = DoubleDuelingDQN(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.q_network.online_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    def compute_td_target(self, rewards, next_states, dones):
        """Double DQN TD target"""
        with torch.no_grad():
            q_next_online = self.q_network.online_net(next_states)
            best_actions = torch.argmax(q_next_online, dim=1, keepdim=True)
            q_next_target = self.q_network.target_net(next_states)
            target_q = q_next_target.gather(1, best_actions).squeeze(1)
            td_target = rewards + self.gamma * (1 - dones) * target_q
        return td_target


# ============================================================
# 1️⃣ D3QN Baseline (Vanilla)
# ============================================================
class D3QNAgent(BaseAgent):
    """
    Vanilla Dueling Double DQN
    ----------------------------------------------------------
    - No adaptive ε
    - Single replay buffer
    - Standard fixed ε decay
    """

    def __init__(self, state_dim, action_dim, device="cpu", logger=None):
        super().__init__(state_dim, action_dim, device)
        self.logger = logger or ExperimentLogger()
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.memory = []
        self.max_memory = 5000

    def select_action(self, state):
        """ε-greedy with fixed decay"""
        if random.random() < self.epsilon:
            return random.randint(0, self.q_network.online_net.action_dim - 1)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(torch.argmax(self.q_network(state_t)).item())

    def push(self, transition):
        if len(self.memory) >= self.max_memory:
            self.memory.pop(0)
        self.memory.append(transition)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32, device=self.device),
            torch.tensor(actions, dtype=torch.long, device=self.device),
            torch.tensor(rewards, dtype=torch.float32, device=self.device),
            torch.tensor(next_states, dtype=torch.float32, device=self.device),
            torch.tensor(dones, dtype=torch.float32, device=self.device),
        )

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.sample(self.batch_size)
        q_pred = self.q_network.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        td_target = self.compute_td_target(rewards, next_states, dones)

        loss = self.loss_fn(q_pred, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if self.global_step % self.target_update_freq == 0:
            self.q_network.soft_update()

        self.global_step += 1
        return loss.item(), float(rewards.mean().item())


# ============================================================
# 2️⃣ A-D3QN Baseline (with Adaptive ε)
# ============================================================
class AD3QNAgent(BaseAgent):
    """
    Adaptive ε–greedy D3QN
    ----------------------------------------------------------
    - Single replay buffer
    - Adaptive exploration using reward variance (ε_scheduler)
    """

    def __init__(self, state_dim, action_dim, device="cpu", logger=None):
        super().__init__(state_dim, action_dim, device)
        self.logger = logger or ExperimentLogger()
        self.memory = []
        self.max_memory = 5000
        self.epsilon_scheduler = AdaptiveEpsilonScheduler(decay_rate=0.995, beta=0.3)

    def select_action(self, state):
        epsilon = self.epsilon_scheduler.get_epsilon()
        if random.random() < epsilon:
            return random.randint(0, self.q_network.online_net.action_dim - 1)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(torch.argmax(self.q_network(state_t)).item())

    def push(self, transition):
        if len(self.memory) >= self.max_memory:
            self.memory.pop(0)
        self.memory.append(transition)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32, device=self.device),
            torch.tensor(actions, dtype=torch.long, device=self.device),
            torch.tensor(rewards, dtype=torch.float32, device=self.device),
            torch.tensor(next_states, dtype=torch.float32, device=self.device),
            torch.tensor(dones, dtype=torch.float32, device=self.device),
        )

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.sample(self.batch_size)
        q_pred = self.q_network.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        td_target = self.compute_td_target(rewards, next_states, dones)

        loss = self.loss_fn(q_pred, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Adaptive epsilon update
        self.epsilon_scheduler.update(float(np.mean(rewards)), bandwidth_var=np.random.rand() * 0.1)

        if self.global_step % self.target_update_freq == 0:
            self.q_network.soft_update()

        self.global_step += 1
        return loss.item(), float(rewards.mean().item())


# ============================================================
# 3️⃣ DP-D3QN Baseline (Dual Replay Buffer)
# ============================================================
class DP_D3QNAgent(BaseAgent):
    """
    Dual-Pool D3QN (without Adaptive ε)
    ----------------------------------------------------------
    - Dual replay buffers (E1 high-TD, E2 low-TD)
    - Fixed ε-greedy
    """

    def __init__(self, state_dim, action_dim, device="cpu", logger=None):
        super().__init__(state_dim, action_dim, device)
        self.logger = logger or ExperimentLogger()
        self.replay_buffer = DualPoolReplayBuffer(device=device)
        self.epsilon = 0.9
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.q_network.online_net.action_dim - 1)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(torch.argmax(self.q_network(state_t)).item())

    def store_transition(self, state, action, reward, next_state, done, td_error):
        self.replay_buffer.push(state, action, reward, next_state, done, td_error)

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        q_pred = self.q_network.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        td_target = self.compute_td_target(rewards, next_states, dones)
        loss = self.loss_fn(q_pred, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.global_step % self.target_update_freq == 0:
            self.q_network.soft_update()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.global_step += 1
        return loss.item(), float(rewards.mean().item())


# ✅ Quick test
if __name__ == "__main__":
    state_dim, action_dim = 4, 8
    agents = {
        "D3QN": D3QNAgent(state_dim, action_dim),
        "A-D3QN": AD3QNAgent(state_dim, action_dim),
        "DP-D3QN": DP_D3QNAgent(state_dim, action_dim),
    }
    for name, agent in agents.items():
        print(f"\nTesting {name}...")
        for _ in range(5):
            s, ns = np.random.rand(4), np.random.rand(4)
            a, r, d = np.random.randint(0, 8), np.random.randn(), False
            if name == "DP-D3QN":
                agent.store_transition(s, a, r, ns, d, td_error=np.random.randn())
            else:
                agent.push((s, a, r, ns, d))
            agent.train_step()
        print(f"{name} OK | ε = {getattr(agent, 'epsilon', None)}")
