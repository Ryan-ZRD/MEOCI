"""
core.environment
=============================

Vehicular Edge Computing Environment Components
------------------------------------------------
This package defines the simulation environment for the MEOCI framework,
including vehicle-side nodes, edge servers, network models, mobility,
and workload generation.

Each submodule provides a concrete component used in the collaborative
inference system simulation and DRL training process.
Modules included:
    - vec_env.py              : High-level VEC environment orchestrator
    - vehicle_node.py         : Vehicle node modeling (compute, queue, energy)
    - edge_server.py          : RSU/edge node with task queue and processing logic
    - network_channel.py      : Wireless link model (bandwidth, SNR, noise)
    - mobility_model.py       : Optional Gaussian–Markov mobility model
    - workload_generator.py   : Task generation and workload sampling

Usage Example:
--------------
>>> from core.environment import (
...     VehicleNode, EdgeServer, NetworkChannel,
...     MobilityModel, WorkloadGenerator, VehicularEdgeEnv
... )
>>> mobility = MobilityModel()
>>> vehicle = VehicleNode(vehicle_id=1)
>>> edge = EdgeServer()
>>> channel = NetworkChannel()
>>> workload = WorkloadGenerator()
>>> env = VehicularEdgeEnv(vehicle, edge, channel, workload, mobility)
>>> obs, reward, done, info = env.step()
"""

from .vehicle_node import VehicleNode
from .edge_server import EdgeServer
from .network_channel import NetworkChannel
from .mobility_model import MobilityModel
from .workload_generator import WorkloadGenerator
from .vec_env import VehicularEdgeEnv

__all__ = [
    "VehicleNode",
    "EdgeServer",
    "NetworkChannel",
    "MobilityModel",
    "WorkloadGenerator",
    "VehicularEdgeEnv",
]
