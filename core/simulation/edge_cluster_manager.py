import numpy as np
from typing import Dict, List, Optional, Tuple
from core.environment.edge_server import EdgeServer
from core.environment.network_channel import NetworkChannel


class EdgeClusterManager:
    """
    EdgeClusterManager
    ==========================================================
    Simulation manager for edge RSU clusters in MEOCI framework.

    Responsibilities:
        - Manage a set of EdgeServers (RSUs)
        - Handle task migration / balancing among RSUs
        - Model inter-RSU communication latency
        - Support adaptive resource allocation and bandwidth sharing

    Key Features:
        ✅ Dynamic load balancing among RSUs
        ✅ Adaptive inter-edge transmission latency
        ✅ Distributed decision support for MEOCI algorithm
        ✅ Compatible with ADP-D3QN environment simulation

    Paper Reference:
        - Section 4.3 "Collaborative Edge Architecture"
        - Fig. 16 "System Scalability Analysis"
        - Table IV "Cluster Parameters"
    """

    def __init__(
        self,
        num_edges: int = 3,
        inter_rsu_bandwidth: float = 20.0,   # Mbps
        inter_rsu_latency_base: float = 1.5, # ms baseline
        migration_cost: float = 0.05,        # fraction of extra delay
        seed: int = 42
    ):
        np.random.seed(seed)
        self.num_edges = num_edges
        self.edges: List[EdgeServer] = [
            EdgeServer(server_id=i, capacity=np.random.uniform(10.0, 15.0))
            for i in range(num_edges)
        ]

        # Create inter-RSU channel links (fully connected)
        self.channels: Dict[Tuple[int, int], NetworkChannel] = {
            (i, j): NetworkChannel(bandwidth=inter_rsu_bandwidth)
            for i in range(num_edges) for j in range(num_edges) if i != j
        }

        self.inter_rsu_latency_base = inter_rsu_latency_base
        self.migration_cost = migration_cost
        self.history = []

    # ------------------------------------------------------------
    # ✅ Task distribution logic (load-aware balancing)
    # ------------------------------------------------------------
    def distribute_task(self, source_edge: int, task_size_mb: float) -> int:
        """
        Assign or migrate a task from a source edge to the most suitable target.
        Returns the target edge index.
        """
        source = self.edges[source_edge]
        target = self._select_target_edge(source_edge)
        if target == source_edge:
            # Stay local
            compute_delay = source.process_task_size(task_size_mb)
            self._log_event(source_edge, target, task_size_mb, compute_delay, "local")
            return source_edge
        else:
            # Migrate task
            ch = self.channels.get((source_edge, target))
            tx_delay = self._compute_inter_rsu_delay(ch, task_size_mb)
            compute_delay = self.edges[target].process_task_size(task_size_mb)
            total_delay = tx_delay + compute_delay * (1 + self.migration_cost)
            self._log_event(source_edge, target, task_size_mb, total_delay, "migrated")
            return target

    # ------------------------------------------------------------
    # 🔍 Target selection based on queue load
    # ------------------------------------------------------------
    def _select_target_edge(self, src: int) -> int:
        """
        Choose target edge server with minimal current load.
        Returns index of best edge (could be same as source).
        """
        loads = [e.current_load for e in self.edges]
        best_edge = int(np.argmin(loads))
        return best_edge if best_edge != src else src

    # ------------------------------------------------------------
    # 🕓 Inter-RSU latency computation
    # ------------------------------------------------------------
    def _compute_inter_rsu_delay(self, channel: NetworkChannel, task_size_mb: float) -> float:
        """
        Compute transmission delay between RSUs over inter-RSU channel.
        """
        channel.update_fading_state()
        base_delay = self.inter_rsu_latency_base
        tx_delay = channel.compute_transmission_delay(task_size_mb)
        noise_factor = np.random.uniform(0.9, 1.1)
        return (base_delay + tx_delay) * noise_factor

    # ------------------------------------------------------------
    # 📈 Collect statistics for cluster performance
    # ------------------------------------------------------------
    def collect_statistics(self) -> Dict[str, float]:
        total_loads = [e.current_load for e in self.edges]
        avg_load = np.mean(total_loads)
        load_var = np.var(total_loads)
        migration_ratio = np.mean([1 if h["type"] == "migrated" else 0 for h in self.history]) if self.history else 0

        return {
            "edges": self.num_edges,
            "avg_load": round(avg_load, 3),
            "load_var": round(load_var, 3),
            "migration_ratio": round(migration_ratio, 3),
        }

    # ------------------------------------------------------------
    # 🧹 Reset state (used for multiple episodes)
    # ------------------------------------------------------------
    def reset(self):
        for e in self.edges:
            e.reset()
        for ch in self.channels.values():
            ch.reset()
        self.history.clear()

    # ------------------------------------------------------------
    # 🧾 Logging utility
    # ------------------------------------------------------------
    def _log_event(self, src: int, tgt: int, size: float, delay: float, event_type: str):
        self.history.append({
            "src": src,
            "tgt": tgt,
            "size(MB)": round(size, 3),
            "delay(ms)": round(delay, 3),
            "type": event_type,
        })

    # ------------------------------------------------------------
    # 📤 Export event trace (for plotting Fig. 16)
    # ------------------------------------------------------------
    def export_log(self, filepath: str):
        import csv
        keys = ["src", "tgt", "size(MB)", "delay(ms)", "type"]
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for entry in self.history:
                writer.writerow(entry)


# ✅ Example test
if __name__ == "__main__":
    cluster = EdgeClusterManager(num_edges=3)
    for _ in range(20):
        src_edge = np.random.randint(0, cluster.num_edges)
        size = np.random.uniform(2.0, 5.0)  # MB
        cluster.distribute_task(src_edge, size)

    stats = cluster.collect_statistics()
    print("Cluster Summary:", stats)
    cluster.export_log("edge_cluster_log.csv")
