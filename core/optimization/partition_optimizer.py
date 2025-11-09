import numpy as np
from typing import List, Dict, Tuple


class PartitionOptimizer:
    """
    PartitionOptimizer
    ==========================================================
    Implements the dynamic model partitioning mechanism for
    collaborative inference between vehicle and edge (RSU).

    It estimates the total latency for each possible partition point
    (layer index) and selects the one that minimizes expected delay
    under resource constraints.

    References:
        Section 3.3 - Model Partitioning Optimization
        Eq. (13)–(15) in the MEOCI paper
    """

    def __init__(self,
                 layer_profile: List[Dict],
                 uplink_bandwidth_mbps: float = 15.0,
                 vehicle_flops: float = 2.0,   # GFLOPS
                 edge_flops: float = 20.0,     # GFLOPS
                 alpha_weight: float = 0.6,    # latency weight
                 beta_weight: float = 0.4,     # energy/transfer weight
                 energy_per_flop_vehicle: float = 1e-9,  # J/FLOP
                 energy_per_flop_edge: float = 1e-10,
                 seed: int = 42):
        """
        Args:
            layer_profile: List of dicts describing each layer's
                           {'id', 'flops', 'feature_size_MB'}
            uplink_bandwidth_mbps: Current wireless uplink bandwidth
            vehicle_flops: Local computing capacity (GFLOPS)
            edge_flops: Edge computing capacity (GFLOPS)
            alpha_weight, beta_weight: Tradeoff weights for cost metric
            energy_per_flop_vehicle, energy_per_flop_edge: Power models
        """
        self.layer_profile = layer_profile
        self.uplink_bandwidth_mbps = uplink_bandwidth_mbps
        self.vehicle_flops = vehicle_flops
        self.edge_flops = edge_flops
        self.alpha = alpha_weight
        self.beta = beta_weight
        self.e_v = energy_per_flop_vehicle
        self.e_e = energy_per_flop_edge
        self.random_state = np.random.RandomState(seed)

    # ------------------------------------------------------------
    # Compute latency for a given partition index
    # ------------------------------------------------------------
    def compute_latency(self, partition_idx: int) -> Dict[str, float]:
        """
        Compute latency for a specific partition layer.

        Total delay = T_vehicle + T_transmission + T_edge
        """
        total_layers = len(self.layer_profile)
        local_layers = self.layer_profile[:partition_idx]
        edge_layers = self.layer_profile[partition_idx:]

        # Compute latency (seconds)
        flops_vehicle = np.sum([l["flops"] for l in local_layers])
        flops_edge = np.sum([l["flops"] for l in edge_layers])

        t_vehicle = flops_vehicle / (self.vehicle_flops * 1e9)
        t_edge = flops_edge / (self.edge_flops * 1e9)

        # Transmission latency (MB → bits / bandwidth)
        feature_MB = self.layer_profile[partition_idx - 1]["feature_size_MB"] if partition_idx > 0 else 0.1
        t_trans = (feature_MB * 8) / (self.uplink_bandwidth_mbps * 1e6)

        total_delay = t_vehicle + t_trans + t_edge
        return {
            "T_vehicle": t_vehicle,
            "T_trans": t_trans,
            "T_edge": t_edge,
            "T_total": total_delay,
        }

    # ------------------------------------------------------------
    # Compute energy consumption for given partition
    # ------------------------------------------------------------
    def compute_energy(self, partition_idx: int) -> Dict[str, float]:
        local_layers = self.layer_profile[:partition_idx]
        edge_layers = self.layer_profile[partition_idx:]

        e_vehicle = np.sum([l["flops"] * self.e_v for l in local_layers])
        e_edge = np.sum([l["flops"] * self.e_e for l in edge_layers])

        return {
            "E_vehicle": e_vehicle,
            "E_edge": e_edge,
            "E_total": e_vehicle + e_edge,
        }

    # ------------------------------------------------------------
    # Evaluate all possible partitions
    # ------------------------------------------------------------
    def evaluate_all_partitions(self) -> List[Dict]:
        results = []
        for idx in range(1, len(self.layer_profile)):
            latency = self.compute_latency(idx)
            energy = self.compute_energy(idx)
            cost = self.alpha * latency["T_total"] + self.beta * energy["E_total"]
            results.append({
                "partition_idx": idx,
                "total_latency": latency["T_total"],
                "total_energy": energy["E_total"],
                "cost": cost,
                "T_vehicle": latency["T_vehicle"],
                "T_trans": latency["T_trans"],
                "T_edge": latency["T_edge"],
            })
        return results

    # ------------------------------------------------------------
    # Select optimal partition
    # ------------------------------------------------------------
    def select_optimal_partition(self) -> Dict:
        results = self.evaluate_all_partitions()
        best = min(results, key=lambda x: x["cost"])
        return best

    # ------------------------------------------------------------
    # Simulate small noise for dynamic bandwidth
    # ------------------------------------------------------------
    def update_bandwidth(self, variation_factor: float = 0.1):
        """Simulate network fluctuation."""
        noise = self.random_state.uniform(-variation_factor, variation_factor)
        self.uplink_bandwidth_mbps = max(1.0, self.uplink_bandwidth_mbps * (1 + noise))

    # ------------------------------------------------------------
    # Pretty print
    # ------------------------------------------------------------
    def summary(self) -> Dict:
        best = self.select_optimal_partition()
        return {
            "optimal_partition": best["partition_idx"],
            "expected_latency_ms": round(best["total_latency"] * 1e3, 2),
            "expected_energy_mJ": round(best["total_energy"] * 1e3, 2),
            "bandwidth_Mbps": round(self.uplink_bandwidth_mbps, 2),
        }


# ✅ Example quick test
if __name__ == "__main__":
    # Example model profile (inspired by VGG16)
    dummy_layers = [
        {"id": 1, "flops": 1.2e8, "feature_size_MB": 0.5},
        {"id": 2, "flops": 2.1e8, "feature_size_MB": 1.0},
        {"id": 3, "flops": 3.4e8, "feature_size_MB": 1.8},
        {"id": 4, "flops": 5.0e8, "feature_size_MB": 2.2},
        {"id": 5, "flops": 8.0e8, "feature_size_MB": 3.0},
    ]
    opt = PartitionOptimizer(layer_profile=dummy_layers, uplink_bandwidth_mbps=10)
    print(opt.summary())
