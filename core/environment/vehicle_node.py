import numpy as np
from typing import Tuple


class VehicleNode:
    """
    Vehicle Node Model
    ----------------------------------------------------------
    Represents a single vehicle participating in VEC collaborative inference.

    Each vehicle:
      - Executes the shallow part of the DNN up to partition_layer.
      - Transmits intermediate features to RSU if not early-exiting.
      - Consumes local compute energy and time.
      - Can early-exit locally based on model exit confidence.

    Corresponds to §4.1.1 of the MEOCI paper.
    """

    def __init__(self,
                 id: int,
                 compute_power: float = 3.0,          # GFLOPs/s
                 energy_per_flop: float = 1e-9,        # J per FLOP
                 memory_capacity: float = 512.0,       # MB
                 upload_power: float = 1.5,            # W
                 base_latency: float = 2.0):            # ms (base delay)
        self.id = id
        self.compute_power = compute_power
        self.energy_per_flop = energy_per_flop
        self.memory_capacity = memory_capacity
        self.upload_power = upload_power
        self.base_latency = base_latency

        # Runtime state
        self.last_latency = 0.0
        self.last_energy = 0.0
        self.last_accuracy = 0.0
        self.cached_feature_size = 0.0  # MB

    # ----------------------------------------------------------
    # Reset
    # ----------------------------------------------------------
    def reset(self):
        self.last_latency = 0.0
        self.last_energy = 0.0
        self.last_accuracy = 0.0
        self.cached_feature_size = 0.0

    # ----------------------------------------------------------
    # Local inference simulation
    # ----------------------------------------------------------
    def run_local_inference(self,
                            partition_layer: int,
                            task: "Task") -> Tuple[float, float]:
        """
        Simulate local inference until the partition layer.
        :param partition_layer: index of split layer
        :param task: input data and model metadata
        :return: (latency_ms, energy_joule)
        """
        # 1. Compute required FLOPs
        flops_per_layer = task.flops / task.total_layers
        local_flops = flops_per_layer * (partition_layer + 1)

        # 2. Compute latency (ms)
        latency_ms = self.base_latency + (local_flops / (self.compute_power * 1e9)) * 1000

        # 3. Compute energy (J)
        energy_joule = local_flops * self.energy_per_flop

        # 4. Cache intermediate features
        feature_size = task.feature_size_mb_per_layer * (partition_layer + 1)
        self.cached_feature_size = min(feature_size, self.memory_capacity)

        # 5. Simulate possible early-exit
        if task.enable_early_exit and partition_layer >= task.exit_layer:
            # Early exit locally: use exit model accuracy
            self.last_accuracy = task.exit_accuracy
            latency_ms *= 0.8  # slightly less computation
            energy_joule *= 0.7
        else:
            # Otherwise, rely on RSU inference later
            self.last_accuracy = 0.0

        self.last_latency = latency_ms
        self.last_energy = energy_joule
        return latency_ms, energy_joule

    # ----------------------------------------------------------
    # Upload intermediate data
    # ----------------------------------------------------------
    def upload_features(self, bandwidth_mbps: float) -> float:
        """
        Compute uplink transmission delay (ms)
        :param bandwidth_mbps: current available bandwidth
        """
        data_bits = self.cached_feature_size * 8 * 1024 * 1024
        delay_ms = (data_bits / (bandwidth_mbps * 1e6)) * 1000
        return delay_ms

    # ----------------------------------------------------------
    # Energy consumption of transmission
    # ----------------------------------------------------------
    def transmission_energy(self, transmission_delay_ms: float) -> float:
        """
        Transmission energy = Power × Time
        """
        return self.upload_power * (transmission_delay_ms / 1000.0)

    # ----------------------------------------------------------
    # Status summary
    # ----------------------------------------------------------
    def summary(self) -> dict:
        return {
            "id": self.id,
            "latency_ms": round(self.last_latency, 3),
            "energy_joule": round(self.last_energy, 6),
            "cached_feature_size_MB": round(self.cached_feature_size, 3),
            "accuracy": round(self.last_accuracy, 3)
        }


# ✅ Example quick test
if __name__ == "__main__":
    class MockTask:
        def __init__(self):
            self.flops = 2e9
            self.total_layers = 10
            self.feature_size_mb_per_layer = 1.2
            self.enable_early_exit = True
            self.exit_layer = 4
            self.exit_accuracy = 0.83
            self.size_mb = 5.0

    v = VehicleNode(id=0)
    t = MockTask()
    lat, energy = v.run_local_inference(partition_layer=3, task=t)
    uplink = v.upload_features(bandwidth_mbps=10)
    print(f"Latency={lat:.2f} ms, Energy={energy:.6f} J, Upload={uplink:.2f} ms, Summary={v.summary()}")
