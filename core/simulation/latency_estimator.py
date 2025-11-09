import numpy as np
from typing import Dict, List, Optional
from core.environment.network_channel import NetworkChannel
from core.environment.vehicle_node import VehicleNode
from core.environment.edge_server import EdgeServer


class LatencyEstimator:
    """
    LatencyEstimator
    ==========================================================
    Core latency modeling component for MEOCI framework.
    ----------------------------------------------------------
    Provides analytical and empirical estimation of total
    end-to-end inference delay between vehicles and edge nodes.

    Components:
        - Local Computation Delay
        - Transmission Delay
        - Edge Queue Delay
        - Processing Delay at Edge Server

    Used in:
        * ADP-D3QN agent environment simulation
        * VehicularNetworkSim
        * Scalability and Ablation experiments (Fig. 9–13)

    Paper Reference:
        Section 4.4: Latency and Energy Model
        Equation (16)–(20): Delay decomposition
        Fig. 9–11: Latency-related experiments
    """

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

    # ------------------------------------------------------------
    # ⚙️ Total End-to-End Latency Estimation
    # ------------------------------------------------------------
    def estimate_total_delay(
        self,
        vehicle: VehicleNode,
        edge: EdgeServer,
        channel: NetworkChannel,
        task_size_mb: float,
        comp_split_ratio: float = 0.5,
    ) -> float:
        """
        Estimate total end-to-end inference delay.

        Args:
            vehicle (VehicleNode): vehicle node generating the task
            edge (EdgeServer): target RSU/edge server
            channel (NetworkChannel): link between them
            task_size_mb (float): data size to transmit
            comp_split_ratio (float): proportion of computation done locally (0~1)

        Returns:
            float: total latency (ms)
        """
        # 1️⃣ Local computation delay
        local_delay = self._compute_local_delay(vehicle, task_size_mb, comp_split_ratio)

        # 2️⃣ Transmission delay
        tx_delay = self._compute_transmission_delay(channel, task_size_mb, comp_split_ratio)

        # 3️⃣ Edge-side queue and compute delay
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

    # ------------------------------------------------------------
    # 🧮 Local Computation Delay
    # ------------------------------------------------------------
    def _compute_local_delay(self, vehicle: VehicleNode, task_size_mb: float, ratio: float) -> float:
        """
        Delay for the portion of computation done locally.
        """
        local_flops = task_size_mb * ratio * 1e6  # scale
        delay = local_flops / (vehicle.compute_power * 1e9) * 1e3  # convert to ms
        return delay

    # ------------------------------------------------------------
    # 📡 Transmission Delay
    # ------------------------------------------------------------
    def _compute_transmission_delay(self, channel: NetworkChannel, task_size_mb: float, ratio: float) -> float:
        """
        Transmission time for data sent to edge (remaining portion).
        """
        channel.update_fading_state()
        tx_data = task_size_mb * (1 - ratio)
        return channel.compute_transmission_delay(tx_data)

    # ------------------------------------------------------------
    # 🏢 Edge Processing Delay
    # ------------------------------------------------------------
    def _compute_edge_delay(self, edge: EdgeServer, task_size_mb: float, ratio: float) -> float:
        """
        Edge-side queueing and computation delay.
        """
        pending = edge.current_load
        compute_flops = task_size_mb * (1 - ratio) * 1e6
        processing_delay = compute_flops / (edge.capacity * 1e9) * 1e3
        queue_delay = self.queue_factor * pending
        total = processing_delay + queue_delay
        edge.current_load += task_size_mb * 0.01  # simulate task enqueue
        return total

    # ------------------------------------------------------------
    # 📈 Summary statistics
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # 💾 Export CSV log
    # ------------------------------------------------------------
    def export_log(self, filepath: str):
        import csv
        keys = ["vehicle", "edge", "local_delay", "tx_delay", "edge_delay", "total_delay"]
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for h in self.history:
                writer.writerow(h)

    # ------------------------------------------------------------
    # 🔄 Reset
    # ------------------------------------------------------------
    def reset(self):
        self.history.clear()


# ✅ Example test
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
