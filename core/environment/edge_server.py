import numpy as np
from typing import Tuple


class EdgeServer:
    """
    Edge Server (RSU) Model
    ----------------------------------------------------------
    Represents a roadside edge node that collaborates with vehicles
    for split DNN inference tasks.

    Each incoming task:
      - Queued in M/D/1 system
      - Processes remaining layers after partition point
      - Consumes edge-side computation energy
      - Produces final inference result or feeds back to vehicle
    """

    def __init__(self,
                 cpu_capacity: float = 200.0,        # GFLOPs/s
                 power_per_flop: float = 5e-10,      # J/FLOP
                 queue_capacity: int = 100,
                 base_latency: float = 3.0,          # ms
                 service_rate: float = 50.0,         # tasks/s (nominal)
                 energy_idle: float = 0.2):          # W idle power
        self.cpu_capacity = cpu_capacity
        self.power_per_flop = power_per_flop
        self.queue_capacity = queue_capacity
        self.base_latency = base_latency
        self.service_rate = service_rate
        self.energy_idle = energy_idle

        # Runtime stats
        self.queue_length = 0
        self.cpu_utilization = 0.0
        self.last_latency = 0.0
        self.last_energy = 0.0
        self.last_accuracy = 0.0

    # ----------------------------------------------------------
    # Reset
    # ----------------------------------------------------------
    def reset(self):
        self.queue_length = 0
        self.cpu_utilization = 0.0
        self.last_latency = 0.0
        self.last_energy = 0.0
        self.last_accuracy = 0.0

    # ----------------------------------------------------------
    # Process one task
    # ----------------------------------------------------------
    def process_task(self,
                     task: "Task",
                     partition_layer: int,
                     exit_point: int) -> Tuple[float, float, float]:
        """
        Simulate RSU-side inference process.
        :param task: task descriptor
        :param partition_layer: split index (layers already computed by vehicle)
        :param exit_point: early-exit index (1–E)
        :return: (latency_ms, energy_joule, accuracy)
        """

        # 1. Compute remaining FLOPs
        remaining_layers = task.total_layers - (partition_layer + 1)
        remaining_flops = max(remaining_layers, 0) * (task.flops / task.total_layers)

        # 2. Compute service time (ms)
        service_time_ms = self.base_latency + (remaining_flops / (self.cpu_capacity * 1e9)) * 1000

        # 3. Estimate queuing delay (M/D/1 queue)
        arrival_rate = task.arrival_rate
        rho = min(arrival_rate / self.service_rate, 0.95)  # utilization factor
        queue_delay_ms = (rho / (2 * (1 - rho))) * (1000 / self.service_rate)

        # 4. Compute total latency
        latency_ms = service_time_ms + queue_delay_ms

        # 5. Compute energy consumption
        energy_joule = remaining_flops * self.power_per_flop + self.energy_idle * (latency_ms / 1000.0)

        # 6. Approximate output accuracy (based on early-exit index)
        acc_curve = np.linspace(0.7, 0.95, task.exit_points)
        accuracy = acc_curve[min(exit_point - 1, len(acc_curve) - 1)]

        # 7. Update system state
        self.cpu_utilization = rho
        self.last_latency = latency_ms
        self.last_energy = energy_joule
        self.last_accuracy = accuracy

        return latency_ms, energy_joule, accuracy

    # ----------------------------------------------------------
    # Queue management
    # ----------------------------------------------------------
    def update_queue_load(self, incoming_tasks: int):
        """
        Update the queue length after processing or new arrivals.
        """
        self.queue_length = min(
            max(self.queue_length + incoming_tasks - int(self.service_rate / 10), 0),
            self.queue_capacity
        )

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    def summary(self) -> dict:
        return {
            "queue_len": round(self.queue_length, 2),
            "cpu_util": round(self.cpu_utilization, 3),
            "latency_ms": round(self.last_latency, 3),
            "energy_joule": round(self.last_energy, 6),
            "accuracy": round(self.last_accuracy, 3),
        }


# ✅ Example quick test
if __name__ == "__main__":
    class MockTask:
        def __init__(self):
            self.flops = 2e9
            self.total_layers = 10
            self.exit_points = 5
            self.arrival_rate = 30.0

    task = MockTask()
    edge = EdgeServer()
    latency, energy, acc = edge.process_task(task, partition_layer=4, exit_point=3)
    print(f"Latency={latency:.2f} ms | Energy={energy:.6f} J | Acc={acc:.3f}")
    print("Summary:", edge.summary())
