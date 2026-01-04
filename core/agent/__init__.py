"""
core.agent
=============================

ADP-D3QN Agent Package
------------------------------------------------
This package implements the Adaptive Dual-Pool Dueling Double DQN (ADP-D3QN)
algorithm used in MEOCI for joint optimization of model partitioning
and early-exit point selection in Vehicular Edge Computing (VEC).

Main Components:
----------------
Modules included:
    - network.py             : Dueling Double Q-Network architecture
    - replay_buffer.py       : Dual experience replay buffers (E1 / E2)
    - epsilon_scheduler.py   : Adaptive ε-greedy exploration control
    - agent_adp_d3qn.py      : Core ADP-D3QN training and inference logic
    - agent_baselines.py     : Baseline agents for comparison (D3QN, A-D3QN, DP-D3QN)

Key Features:
--------------
    • Dual-pool replay mechanism for balanced exploration/exploitation.
    • Adaptive ε-decay driven by reward variance.
    • Shared network backbone with dueling state–advantage architecture.
    • Modular interfaces compatible with OpenAI-Gym–style environments.

Example:
--------
>>> from core.agent import ADP_D3QNAgent
>>> from core.environment import VehicularEdgeEnv
>>> env = VehicularEdgeEnv()
>>> agent = ADP_D3QNAgent(state_dim=8, action_dim=12)
>>> state = env.reset()
>>> action = agent.select_action(state)
>>> next_state, reward, done, info = env.step(action)
>>> agent.store_transition(state, action, reward, next_state, done)
>>> agent.learn()
"""

from .network import DuelingQNetwork
from .replay_buffer import ReplayBuffer
from .epsilon_scheduler import AdaptiveEpsilonScheduler
from .agent_adp_d3qn import ADP_D3QNAgent
from .agent_baselines import D3QNAgent, AD3QNAgent, DP_D3QNAgent

__all__ = [
    "DuelingQNetwork",
    "ReplayBuffer",
    "AdaptiveEpsilonScheduler",
    "ADP_D3QNAgent",
    "D3QNAgent",
    "AD3QNAgent",
    "DP_D3QNAgent",
]
