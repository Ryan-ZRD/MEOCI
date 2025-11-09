import numpy as np
from typing import Dict, List, Tuple


class ResourceAllocator:
    """
    ResourceAllocator
    ==========================================================
    Dynamic resource allocation for collaborative inference.

    Optimizes the assignment of:
        - Bandwidth between vehicles
        - CPU frequency scaling between vehicle and RSU
        - Task load balancing factor

    Objective:
        Minimize system latency while maintaining fairness
        and total resource constraints.

    References:
        Section 3.5 - Resource Allocation Optimization
        Eq. (19)–(20) in MEOCI paper.
    """

    def __init__(self,
                 total_bandwidth_Mbps: float = 50.0,
                 max_vehicle_freq_GHz: float = 2.0,
                 max_edge_freq_GHz: float = 20.0,
                 fairness_weight: float = 0.3,
                 delay_weight: float = 0.6,
                 energy_weight: float = 0.1,
                 seed: int = 42):
        self.total_bandwidth_Mbps = total_bandwidth_Mbps
        self.max_vehicle_freq_GHz = max_vehicle_freq_GHz
        self.max_edge_freq_GHz = max_edge_freq_GHz
        self.fairness_weight = fairness_weight
        self.delay_weight = delay_weight
        self.energy_weight = energy_weight
        self.random_state = np.random.RandomState(seed)

    # ------------------------------------------------------------
    # Allocate bandwidth among vehicles
    # ------------------------------------------------------------
    def allocate_bandwidth(self,
                           num_vehicles: int,
                           demand_factors: List[float]) -> np.ndarray:
        """
        Allocate bandwidth proportionally to demand factors.
        Normalized proportional fair allocation.

        Eq. (19): B_i = (η_i / Ση_j) * B_total
        """
        demand_factors = np.array(demand_factors)
        demand_factors = np.clip(demand_factors, 1e-3, None)
        normalized = demand_factors / np.sum(demand_factors)
        allocated = normalized * self.total_bandwidth_Mbps
        return allocated

    # ------------------------------------------------------------
    # Allocate CPU frequency scaling between vehicle and edge
    # ------------------------------------------------------------
    def allocate_computation(self,
                             load_vehicle: float,
                             load_edge: float) -> Dict[str, float]:
        """
        Adaptive CPU frequency scaling.
        Eq. (20): f_v + f_e = f_total
        Goal: Minimize α*T_total + β*E_total + γ*Fairness
        """
        total_load = load_vehicle + load_edge + 1e-6
        f_vehicle = (load_vehicle / total_load) * self.max_vehicle_freq_GHz
        f_edge = (load_edge / total_load) * self.max_edge_freq_GHz

        return {
            "f_vehicle_GHz": round(f_vehicle, 3),
            "f_edge_GHz": round(f_edge, 3),
        }

    # ------------------------------------------------------------
    # Evaluate fairness metric (Jain’s index)
    # ------------------------------------------------------------
    def fairness_index(self, bandwidth_alloc: np.ndarray) -> float:
        """
        Jain's Fairness Index:
        J = (ΣB_i)^2 / (N * ΣB_i^2)
        """
        n = len(bandwidth_alloc)
        numerator = np.sum(bandwidth_alloc) ** 2
        denominator = n * np.sum(bandwidth_alloc ** 2)
        return float(numerator / denominator)

    # ------------------------------------------------------------
    # Compute composite cost
    # ------------------------------------------------------------
    def compute_system_cost(self,
                            total_delay: float,
                            total_energy: float,
                            fairness: float) -> float:
        """
        Weighted cost combining delay, energy, and fairness.
        J_total = α*Delay + β*Energy - γ*Fairness
        """
        return (self.delay_weight * total_delay +
                self.energy_weight * total_energy -
                self.fairness_weight * fairness)

    # ------------------------------------------------------------
    # Adaptive bandwidth fluctuation simulation
    # ------------------------------------------------------------
    def fluctuate_bandwidth(self, variation: float = 0.15):
        """
        Simulate stochastic fluctuation in total bandwidth.
        """
        noise = self.random_state.uniform(-variation, variation)
        self.total_bandwidth_Mbps = max(5.0, self.total_bandwidth_Mbps * (1 + noise))

    # ------------------------------------------------------------
    # Summary utility
    # ------------------------------------------------------------
    def summary(self, bw_alloc: np.ndarray, freq_alloc: Dict[str, float]) -> Dict:
        fairness = self.fairness_index(bw_alloc)
        return {
            "bandwidth_alloc": np.round(bw_alloc, 2).tolist(),
            "fairness_index": round(fairness, 4),
            "vehicle_freq_GHz": freq_alloc["f_vehicle_GHz"],
            "edge_freq_GHz": freq_alloc["f_edge_GHz"],
            "total_bandwidth_Mbps": round(self.total_bandwidth_Mbps, 2),
        }


# ✅ Example quick test
if __name__ == "__main__":
    allocator = ResourceAllocator(total_bandwidth_Mbps=40)
    demand = [0.3, 0.5, 1.0, 0.8, 0.4]
    bw = allocator.allocate_bandwidth(len(demand), demand)
    freq = allocator.allocate_computation(load_vehicle=0.7, load_edge=1.8)
    summary = allocator.summary(bw, freq)
    print("Allocation Summary:", summary)
