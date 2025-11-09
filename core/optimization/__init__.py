"""
core.optimization
=============================================================
Joint Optimization Package for MEOCI Framework
-------------------------------------------------------------
Implements the optimization logic for:
  • Model partitioning
  • Early-exit selection
  • Resource allocation
  • Reward function computation

This package defines the joint decision-making backbone of
the MEOCI algorithm, combining adaptive partition strategies
and dynamic early-exit decisions under real-time vehicular
edge computing constraints.

Modules Summary:
----------------
- partition_optimizer.py  : DNN layer partitioning optimization
- early_exit_selector.py  : Multi-exit early-exit point decision
- resource_allocator.py   : Bandwidth & computation allocation
- reward_function.py      : Scalar reward design for ADP-D3QN

References:
-----------
  Section 3.3–3.6 of MEOCI Paper
  Equations (13)–(22)

Example:
--------
>>> from core.optimization import (
...     PartitionOptimizer,
...     EarlyExitSelector,
...     ResourceAllocator,
...     RewardFunction
... )
>>> optimizer = PartitionOptimizer(num_layers=20)
>>> selector = EarlyExitSelector(acc_threshold=0.8)
>>> allocator = ResourceAllocator(total_bandwidth_Mbps=50)
>>> reward_fn = RewardFunction()
"""

from .partition_optimizer import PartitionOptimizer
from .early_exit_selector import EarlyExitSelector
from .resource_allocator import ResourceAllocator
from .reward_function import RewardFunction

__all__ = [
    "PartitionOptimizer",
    "EarlyExitSelector",
    "ResourceAllocator",
    "RewardFunction",
]
