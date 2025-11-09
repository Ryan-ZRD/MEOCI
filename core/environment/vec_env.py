import numpy as np
import random
from typing import Dict, Tuple

from core.environment.vehicle_node import VehicleNode
from core.environment.edge_server import EdgeServer
from core.environment.network_channel import NetworkChannel
from core.environment.mobility_model import MobilityModel
from core.environment.workload_generator import WorkloadGenerator
from core.optimization.reward_function import RewardFunction
from utils.logger import ExperimentLogger


class VehicularEdgeEnv:
    """
    Vehicular Edge Computing (VEC) Environment
    ----------------------------------------------------------
    Core environment used to train and evaluate the ADP-D3QN agent.

    State space:
        s_t = [accuracy_t, queue_length_t, edge_resource_t, task_rate_t]

    Action space:
        a_t = (partition_point ∈ [0, L], exit_point ∈ [1, E])

    Reward:
        r_t = - (w_d * latency + w_e * energy - w_a * accuracy)
        (Implemented in reward_function.py)
    """

    def __init__(self,
                 model_layers: int = 10,
                 exit_points: int = 4,
                 num_vehicles: int = 10,
                 logger: ExperimentLogger = None,
                 bandwidth_range: Tuple[float, float] = (5, 25),
                 seed: int = 42):
        self.logger = logger or Logger()
        self.model_layers = model_layers
        self.exit_points = exit_points
        self.num_vehicles = num_vehicles
        self.bandwidth_range = bandwidth_range

        # Initialize subsystems
        self.channel = NetworkChannel(bandwidth_range=bandwidth_range)
        self.edge_server = EdgeServer(cpu_capacity=200, queue_capacity=100)
        self.vehicles = [VehicleNode(id=i) for i in range(num_vehicles)]
        self.mobility = MobilityModel(seed=seed)
        self.workload = WorkloadGenerator(seed=seed)

        self.current_step = 0
        self.state_dim = 4
        self.action_dim = model_layers * exit_points

        self.random_state = np.random.RandomState(seed)
        self.logger.info(f"[VEC-ENV] Initialized with {num_vehicles} vehicles.")

    # ----------------------------------------------------------
    # Environment state
    # ----------------------------------------------------------
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        for v in self.vehicles:
            v.reset()
        self.edge_server.reset()
        state = self._get_state()
        return state

    def _get_state(self) -> np.ndarray:
        """System-level state vector"""
        avg_acc = np.mean([v.last_accuracy for v in self.vehicles])
        avg_queue = self.edge_server.queue_length / self.edge_server.queue_capacity
        resource_util = self.edge_server.cpu_utilization
        task_rate = self.workload.get_current_rate()

        state = np.array([avg_acc, avg_queue, resource_util, task_rate], dtype=np.float32)
        return state

    # ----------------------------------------------------------
    # Action & Transition
    # ----------------------------------------------------------
    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one environment step.
        :param action: (partition_layer, exit_point)
        :return: next_state, reward, done, info
        """
        partition_layer, exit_point = action
        self.current_step += 1

        # Update network dynamics
        current_bw = self.channel.sample_bandwidth()
        self.mobility.update_positions(self.vehicles)

        # Generate workloads
        new_tasks = self.workload.generate_tasks(self.num_vehicles)

        total_latency, total_energy, total_acc = 0.0, 0.0, 0.0

        for v, task in zip(self.vehicles, new_tasks):
            # 1. Vehicle performs shallow layer inference
            local_latency, local_energy = v.run_local_inference(partition_layer, task)

            # 2. Upload intermediate features
            comm_delay = self.channel.compute_delay(size_mb=task.size_mb, bandwidth=current_bw)

            # 3. Edge server executes remaining layers
            edge_latency, edge_energy, acc = self.edge_server.process_task(
                task, partition_layer, exit_point)

            total_latency += local_latency + comm_delay + edge_latency
            total_energy += local_energy + edge_energy
            total_acc += acc

        avg_latency = total_latency / self.num_vehicles
        avg_energy = total_energy / self.num_vehicles
        avg_accuracy = total_acc / self.num_vehicles

        # Compute reward (negative cost)
        reward = RewardFunction.compute_reward(latency=avg_latency,
                                energy=avg_energy,
                                accuracy=avg_accuracy)

        # Update queue and system load
        self.edge_server.update_queue_load(len(new_tasks))

        # Construct next state
        next_state = self._get_state()

        # Done flag (episode termination)
        done = self.current_step >= 200

        info = {
            "avg_latency": avg_latency,
            "avg_energy": avg_energy,
            "avg_accuracy": avg_accuracy,
            "bandwidth": current_bw,
            "edge_queue": self.edge_server.queue_length
        }

        # Logging
        if self.logger:
            self.logger.log_scalar("avg_latency", avg_latency, self.current_step)
            self.logger.log_scalar("avg_energy", avg_energy, self.current_step)
            self.logger.log_scalar("avg_accuracy", avg_accuracy, self.current_step)

        return next_state, reward, done, info

    # ----------------------------------------------------------
    # Utility
    # ----------------------------------------------------------
    def sample_action(self) -> Tuple[int, int]:
        """Sample a random (partition, exit) pair"""
        partition_layer = random.randint(0, self.model_layers - 1)
        exit_point = random.randint(1, self.exit_points)
        return partition_layer, exit_point

    def render(self):
        """Optional visualization hook"""
        print(f"[Step {self.current_step}] Queue={self.edge_server.queue_length:.2f}, "
              f"Util={self.edge_server.cpu_utilization:.2f}")

    def get_env_stats(self) -> Dict[str, float]:
        """Return environment-level volatility indicators"""
        return {
            "bandwidth_var": np.var(self.channel.history_bandwidth[-10:]),
            "load_factor": self.edge_server.cpu_utilization
        }


# ✅ Quick test
if __name__ == "__main__":
    from utils.logger import Logger
    env = VehicularEdgeEnv(logger=Logger(), num_vehicles=5)
    state = env.reset()
    for t in range(5):
        a = env.sample_action()
        next_state, r, done, info = env.step(a)
        print(f"Step {t} | Reward={r:.3f} | Lat={info['avg_latency']:.2f} | Acc={info['avg_accuracy']:.2f}")
