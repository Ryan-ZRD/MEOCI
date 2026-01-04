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


    def __init__(self,
                 model_layers: int = 10,
                 exit_points: int = 4,
                 num_vehicles: int = 10,
                 logger: ExperimentLogger = None,
                 bandwidth_range: Tuple[float, float] = (5, 25),
                 seed: int = 42):

        # Logger is managed externally
        self.logger = logger

        # Core parameters
        self.model_layers = int(model_layers)
        self.exit_points = int(exit_points)
        self.num_vehicles = int(num_vehicles)
        self.bandwidth_range = bandwidth_range
        self.seed = int(seed)

        # Subsystems
        self.channel = NetworkChannel(bandwidth_range=bandwidth_range)
        self.edge_server = EdgeServer(cpu_capacity=200, queue_capacity=100)
        self.vehicles = [VehicleNode(id=i) for i in range(num_vehicles)]

        # Mobility model (global single state)
        self.mobility = MobilityModel(seed=seed)


        self.workload = WorkloadGenerator(random_seed=seed)

        # Reward function must be instantiated (not static)
        self.reward_fn = RewardFunction(
            delay_ref_ms=4000.0,
            energy_ref_mJ=1200.0,
            reward_clip=True
        )

        # Bookkeeping
        self.current_step = 0
        self.state_dim = 4
        self.action_dim = self.model_layers * self.exit_points
        self.random_state = np.random.RandomState(seed)

        if self.logger:
            self.logger.info(f"[VEC-ENV] Initialized with {num_vehicles} vehicles.")


    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        for v in self.vehicles:
            v.reset()
        self.edge_server.reset()
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """System-level state vector: [avg_acc, avg_queue_ratio, cpu_util, task_rate]"""
        avg_acc = np.mean([v.last_accuracy for v in self.vehicles])
        avg_queue = self.edge_server.queue_length / max(self.edge_server.queue_capacity, 1)
        resource_util = self.edge_server.cpu_utilization
        task_rate = self.workload.get_current_rate()

        state = np.array([avg_acc, avg_queue, resource_util, task_rate], dtype=np.float32)
        return state


    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, float, bool, Dict]:

        partition_layer, exit_point = action
        self.current_step += 1


        current_bw = self.channel.sample_bandwidth()


        self.mobility.step()


        new_tasks = self.workload.generate_tasks(
            self.num_vehicles,
            total_layers=self.model_layers,
            exit_points=self.exit_points
        )

        total_latency, total_energy, total_acc = 0.0, 0.0, 0.0


        for v, task in zip(self.vehicles, new_tasks):

            # Vehicle shallow inference
            local_latency, local_energy = v.run_local_inference(partition_layer, task)

            # Communication delay: upload intermediate features
            comm_delay = self.channel.compute_delay(size_mb=task.size_mb, bandwidth=current_bw)

            # Edge compute remaining layers
            edge_latency, edge_energy, acc = self.edge_server.process_task(task, partition_layer, exit_point)

            # accumulate
            total_latency += float(local_latency + comm_delay + edge_latency)
            total_energy += float(local_energy + edge_energy)
            total_acc += float(acc)

        # avg metrics across vehicles
        avg_latency = total_latency / max(self.num_vehicles, 1)
        avg_energy = total_energy / max(self.num_vehicles, 1)
        avg_accuracy = total_acc / max(self.num_vehicles, 1)


        metrics_dict = {
            "accuracy": float(avg_accuracy),
            "latency_ms": float(avg_latency),
            "energy_mJ": float(avg_energy * 1000.0),
            "completion_rate": 1.0
        }
        reward = self.reward_fn.compute_reward(metrics_dict)


        self.edge_server.update_queue_load(len(new_tasks))


        next_state = self._get_state()


        done = self.current_step >= 200

        info = {
            "avg_latency": float(avg_latency),
            "avg_energy": float(avg_energy),
            "avg_accuracy": float(avg_accuracy),
            "bandwidth": float(current_bw),
            "edge_queue": float(self.edge_server.queue_length)
        }


        if self.logger:
            self.logger.log_scalar("avg_latency", avg_latency, self.current_step)
            self.logger.log_scalar("avg_energy", avg_energy, self.current_step)
            self.logger.log_scalar("avg_accuracy", avg_accuracy, self.current_step)
            self.logger.log_scalar("reward", reward, self.current_step)

        return next_state, float(reward), bool(done), info


    def sample_action(self) -> Tuple[int, int]:
        """Sample random (partition_layer, exit_point)"""
        partition_layer = random.randint(0, self.model_layers - 1)
        exit_point = random.randint(1, self.exit_points)
        return partition_layer, exit_point

    def render(self):
        print(f"[Step {self.current_step}] Queue={self.edge_server.queue_length:.2f}, "
              f"Util={self.edge_server.cpu_utilization:.2f}")

    def get_env_stats(self) -> Dict[str, float]:

        hist = getattr(self.channel, "history_bandwidth", [])[-10:]
        bw_var = float(np.var(hist)) if len(hist) > 1 else 0.0
        return {
            "bandwidth_var": bw_var,
            "load_factor": float(self.edge_server.cpu_utilization)
        }


if __name__ == "__main__":
    logger = ExperimentLogger(enable_tensorboard=False)
    env = VehicularEdgeEnv(logger=logger, num_vehicles=5, model_layers=10, exit_points=4)
    state = env.reset()
    for t in range(5):
        action = env.sample_action()
        next_state, r, done, info = env.step(action)
        print(f"Step {t} | Reward={r:.3f} | Lat={info['avg_latency']:.2f} | Acc={info['avg_accuracy']:.2f}")
    logger.close()
