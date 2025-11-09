"""
core.simulation
==========================================================
Simulation subsystem for the MEOCI framework.
----------------------------------------------------------
This package integrates:
    1. VehicularNetworkSim — vehicle-edge interaction environment
    2. EdgeClusterManager  — multi-RSU cluster scheduling
    3. LatencyEstimator    — analytical latency modeling

Used by:
    * ADP-D3QN training loop
    * Ablation and scalability experiments
    * Visualization (Fig.9–Fig.16)
"""

from core.simulation.vehicular_network_sim import VehicularNetworkSim
from core.simulation.edge_cluster_manager import EdgeClusterManager
from core.simulation.latency_estimator import LatencyEstimator

import yaml
from typing import Optional, Dict, Any


def create_simulation_env(
    config_path: Optional[str] = None,
    num_vehicles: int = 10,
    num_edges: int = 2,
    bandwidth: float = 10.0,
    inter_rsu_bw: float = 20.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    create_simulation_env()
    ======================================================
    Unified entry point for creating the MEOCI simulation
    environment. Returns an integrated environment dict
    containing all three simulation components.

    Args:
        config_path (str, optional): Path to YAML configuration.
        num_vehicles (int): number of vehicles in simulation.
        num_edges (int): number of edge RSUs.
        bandwidth (float): wireless bandwidth (Mbps).
        inter_rsu_bw (float): inter-RSU bandwidth (Mbps).
        seed (int): random seed for reproducibility.

    Returns:
        dict: {
            "vehicular_env": VehicularNetworkSim,
            "cluster_manager": EdgeClusterManager,
            "latency_estimator": LatencyEstimator
        }

    Example:
    --------
    >>> envs = create_simulation_env(config_path="configs/env_cluster.yaml")
    >>> veh_env = envs["vehicular_env"]
    >>> stats = veh_env.collect_statistics()
    >>> print(stats)
    """

    # ------------------------------------------------------------
    # Load config file if provided
    # ------------------------------------------------------------
    cfg = {}
    if config_path:
        try:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"[Warning] Config file not found: {config_path}. Using defaults.")

    num_vehicles = cfg.get("num_vehicles", num_vehicles)
    num_edges = cfg.get("num_edges", num_edges)
    bandwidth = cfg.get("bandwidth", bandwidth)
    inter_rsu_bw = cfg.get("inter_rsu_bw", inter_rsu_bw)
    noise_power = cfg.get("noise_power", -90.0)
    sim_time = cfg.get("sim_time", 100.0)

    # ------------------------------------------------------------
    # Instantiate components
    # ------------------------------------------------------------
    vehicular_env = VehicularNetworkSim(
        num_vehicles=num_vehicles,
        num_edges=num_edges,
        bandwidth=bandwidth,
        noise_power=noise_power,
        sim_time=sim_time,
        seed=seed
    )

    cluster_manager = EdgeClusterManager(
        num_edges=num_edges,
        inter_rsu_bandwidth=inter_rsu_bw,
        seed=seed
    )

    latency_estimator = LatencyEstimator(seed=seed)

    # ------------------------------------------------------------
    # Combine environment components
    # ------------------------------------------------------------
    return {
        "vehicular_env": vehicular_env,
        "cluster_manager": cluster_manager,
        "latency_estimator": latency_estimator
    }


def reset_all(envs: Dict[str, Any]):
    """
    Reset all simulation modules simultaneously.
    """
    if "vehicular_env" in envs:
        envs["vehicular_env"].reset()
    if "cluster_manager" in envs:
        envs["cluster_manager"].reset()
    if "latency_estimator" in envs:
        envs["latency_estimator"].reset()
    print("[Simulation] All modules reset.")


# ------------------------------------------------------------
# ✅ Example usage (manual test)
# ------------------------------------------------------------
if __name__ == "__main__":
    envs = create_simulation_env(num_vehicles=5, num_edges=2)
    veh_env = envs["vehicular_env"]

    for _ in range(5):
        veh_env.step()
    print("Vehicular Environment Summary:", veh_env.collect_statistics())

    cluster_stats = envs["cluster_manager"].collect_statistics()
    print("Edge Cluster Summary:", cluster_stats)

    reset_all(envs)
