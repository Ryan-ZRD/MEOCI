import numpy as np
from typing import Dict, List, Optional
from core.environment.network_channel import NetworkChannel
from core.environment.vehicle_node import VehicleNode
from core.environment.edge_server import EdgeServer


class LatencyEstimator:


    def __init__(
        self,
        base_latency_ms: float = 0.5,
        queue_factor: float = 0.2,
        noise_std: float = 0.05,
        seed: int = 42
    ):
        np.random.seed(seed)
        self.base_latency_ms = base_latency_ms
        self.queue_factor = queue_factor
        self.noise_std = noise_std
        self.history: List[Dict] = []

    def estimate_total_delay(
        self,
        vehicle: VehicleNode,
        edge: EdgeServer,
        channel: NetworkChannel,
        task_size_mb: float,
        comp_split_ratio: float = 0.5,
    ) -> float:

        local_delay = self._compute_local_delay(vehicle, task_size_mb, comp_split_ratio)

        tx_delay = self._compute_transmission_delay(channel, task_size_mb, comp_split_ratio)

        edge_delay = self._compute_edge_delay(edge, task_size_mb, comp_split_ratio)

        total_delay = local_delay + tx_delay + edge_delay + self.base_latency_ms
        total_delay *= np.random.normal(1.0, self.noise_std)

        # Logging
        self.history.append({
            "vehicle": vehicle.vehicle_id,
            "edge": edge.server_id,
            "local_delay": round(local_delay, 3),
            "tx_delay": round(tx_delay, 3),
            "edge_delay": round(edge_delay, 3),
            "total_delay": round(total_delay, 3)
        })
        return total_delay

    def _compute_local_delay(self, vehicle: VehicleNode, task_size_mb: float, ratio: float) -> float:
        local_flops = task_size_mb * ratio * 1e6  # scale
        delay = local_flops / (vehicle.compute_power * 1e9) * 1e3  # convert to ms
        return delay

    def _compute_transmission_delay(self, channel: NetworkChannel, task_size_mb: float, ratio: float) -> float:
        channel.update_fading_state()
        tx_data = task_size_mb * (1 - ratio)
        return channel.compute_transmission_delay(tx_data)

    def _compute_edge_delay(self, edge: EdgeServer, task_size_mb: float, ratio: float) -> float:
        pending = edge.current_load
        compute_flops = task_size_mb * (1 - ratio) * 1e6
        processing_delay = compute_flops / (edge.capacity * 1e9) * 1e3
        queue_delay = self.queue_factor * pending
        total = processing_delay + queue_delay
        edge.current_load += task_size_mb * 0.01  # simulate task enqueue
        return total

    def collect_statistics(self) -> Dict[str, float]:
        if not self.history:
            return {"avg_delay": 0.0, "samples": 0}

        delays = [h["total_delay"] for h in self.history]
        stats = {
            "avg_delay": round(np.mean(delays), 3),
            "std_delay": round(np.std(delays), 3),
            "samples": len(delays),
            "avg_local": round(np.mean([h["local_delay"] for h in self.history]), 3),
            "avg_tx": round(np.mean([h["tx_delay"] for h in self.history]), 3),
            "avg_edge": round(np.mean([h["edge_delay"] for h in self.history]), 3),
        }
        return stats

    def export_log(self, filepath: str):
        import csv
        keys = ["vehicle", "edge", "local_delay", "tx_delay", "edge_delay", "total_delay"]
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for h in self.history:
                writer.writerow(h)

    def reset(self):
        self.history.clear()



if __name__ == "__main__":
    from core.environment.vehicle_node import VehicleNode
    from core.environment.edge_server import EdgeServer
    from core.environment.network_channel import NetworkChannel

    veh = VehicleNode(vehicle_id=0, compute_power=1.5)
    edge = EdgeServer(server_id=0, capacity=12.0)
    ch = NetworkChannel(bandwidth=10.0, noise_power=-90.0)

    estimator = LatencyEstimator()
    for _ in range(10):
        delay = estimator.estimate_total_delay(veh, edge, ch, task_size_mb=3.0, comp_split_ratio=0.4)
        print(f"Total delay = {delay:.3f} ms")

    print("Stats:", estimator.collect_statistics())
