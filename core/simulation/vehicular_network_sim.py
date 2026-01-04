import random
import numpy as np
from typing import Dict, List, Tuple
from core.environment.vehicle_node import VehicleNode
from core.environment.edge_server import EdgeServer
from core.environment.network_channel import NetworkChannel
from core.environment.workload_generator import WorkloadGenerator


class VehicularNetworkSim:

    def __init__(
        self,
        num_vehicles: int = 10,
        num_edges: int = 2,
        bandwidth: float = 10.0,
        noise_power: float = -90.0,
        sim_time: float = 100.0,
        seed: int = 42
    ):
        random.seed(seed)
        np.random.seed(seed)

        self.time = 0.0
        self.sim_time = sim_time
        self.num_vehicles = num_vehicles
        self.num_edges = num_edges
        self.bandwidth = bandwidth
        self.noise_power = noise_power

        # Initialize entities
        self.vehicles: List[VehicleNode] = [
            VehicleNode(vehicle_id=i, compute_power=random.uniform(1.0, 2.5)) for i in range(num_vehicles)
        ]
        self.edges: List[EdgeServer] = [
            EdgeServer(server_id=j, capacity=random.uniform(10.0, 15.0)) for j in range(num_edges)
        ]
        self.channels: Dict[Tuple[int, int], NetworkChannel] = {
            (v.vehicle_id, e.server_id): NetworkChannel(bandwidth=self.bandwidth, noise_power=self.noise_power)
            for v in self.vehicles for e in self.edges
        }

        self.workload_gen = WorkloadGenerator(arrival_rate=0.3)
        self.history = []  # record per-step metrics

    def step(self, delta_t: float = 1.0):
        self.time += delta_t
        step_log = {"time": self.time, "vehicles": []}

        for v in self.vehicles:
            # 1. Channel update
            edge = self._select_best_edge(v)
            ch = self.channels[(v.vehicle_id, edge.server_id)]
            ch.update_fading_state()

            # 2. Task generation
            new_task = self.workload_gen.generate_task()
            if new_task:
                v.enqueue_task(new_task)

            # 3. Task offloading
            if v.has_pending_task():
                task = v.dequeue_task()
                tx_delay = ch.compute_transmission_delay(task.size)
                edge_delay = edge.process_task(task)
                total_delay = tx_delay + edge_delay
            else:
                total_delay = 0.0

            # 4. Log metrics
            step_log["vehicles"].append({
                "vehicle_id": v.vehicle_id,
                "edge_id": edge.server_id,
                "delay": total_delay,
                "queue": len(v.task_queue),
                "bandwidth": ch.bandwidth,
                "snr": ch.current_snr,
            })

        self.history.append(step_log)
        return step_log

    def _select_best_edge(self, vehicle: VehicleNode) -> EdgeServer:
        return min(self.edges, key=lambda e: e.current_load)

    def collect_statistics(self) -> Dict[str, float]:
        delays = [v["delay"] for step in self.history for v in step["vehicles"] if v["delay"] > 0]
        avg_delay = np.mean(delays) if delays else 0.0
        avg_bw = np.mean([v["bandwidth"] for step in self.history for v in step["vehicles"]])
        avg_snr = np.mean([v["snr"] for step in self.history for v in step["vehicles"]])

        return {
            "avg_delay": round(avg_delay, 3),
            "avg_bandwidth": round(avg_bw, 2),
            "avg_snr": round(avg_snr, 2),
            "steps": len(self.history),
            "vehicles": self.num_vehicles,
        }

    def reset(self):
        self.time = 0.0
        self.history.clear()
        for v in self.vehicles:
            v.reset()
        for e in self.edges:
            e.reset()
        for ch in self.channels.values():
            ch.reset()

    def export_log(self, filepath: str):

        import csv
        keys = ["time", "vehicle_id", "edge_id", "delay", "queue", "bandwidth", "snr"]
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for step in self.history:
                for v in step["vehicles"]:
                    writer.writerow({
                        "time": step["time"],
                        **{k: v[k] for k in keys if k != "time"}
                    })



if __name__ == "__main__":
    sim = VehicularNetworkSim(num_vehicles=5, num_edges=2)
    for _ in range(10):
        step_data = sim.step()
        print(f"t={step_data['time']:.1f}s | avg delay={[v['delay'] for v in step_data['vehicles']]}")
    stats = sim.collect_statistics()
    print("Simulation Summary:", stats)
