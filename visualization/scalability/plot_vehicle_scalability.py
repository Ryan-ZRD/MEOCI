"""
visualization.scalability.plot_vehicle_scalability
----------------------------------------------------------
Reproduces Fig.16 of the MEOCI paper.

Plots the scalability of different inference frameworks
under increasing vehicle counts (system load).

Metrics:
    - Average Inference Latency (ms)
    - Task Completion Rate (%)

Expected CSV format (results/csv/scalability.csv):
    VehicleCount,Latency_MEOCI,Latency_Edgent,Latency_FedAdapt,Completion_MEOCI,Completion_Edgent,Completion_FedAdapt
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from visualization.shared_styles.plot_style import set_global_plot_style
from visualization.shared_styles.smoothing import smooth_curve


def plot_vehicle_scalability(data_path: str, save_path: str = None):
    """Plot scalability comparison (latency & completion rate vs vehicle count)."""
    set_global_plot_style()

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"CSV file not found: {data_path}")

    df = pd.read_csv(data_path)
    vehicle_count = df["VehicleCount"].values

    latency_meoci = smooth_curve(df["Latency_MEOCI"].values, window_size=3)
    latency_edgent = smooth_curve(df["Latency_Edgent"].values, window_size=3)
    latency_fedadapt = smooth_curve(df["Latency_FedAdapt"].values, window_size=3)

    completion_meoci = smooth_curve(df["Completion_MEOCI"].values, window_size=3)
    completion_edgent = smooth_curve(df["Completion_Edgent"].values, window_size=3)
    completion_fedadapt = smooth_curve(df["Completion_FedAdapt"].values, window_size=3)

    fig, ax1 = plt.subplots(figsize=(7, 4))

    # Left axis: latency
    ax1.plot(vehicle_count, latency_meoci, color="tab:blue", marker="o", linewidth=2.0, label="Latency - MEOCI")
    ax1.plot(vehicle_count, latency_edgent, color="tab:orange", marker="^", linewidth=2.0, label="Latency - Edgent")
    ax1.plot(vehicle_count, latency_fedadapt, color="tab:green", marker="s", linewidth=2.0, label="Latency - FedAdapt")
    ax1.set_xlabel("Number of Vehicles", fontsize=12)
    ax1.set_ylabel("Average Inference Latency (ms)", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Right axis: completion rate
    ax2 = ax1.twinx()
    ax2.plot(vehicle_count, completion_meoci, color="tab:blue", linestyle="--", linewidth=2.0, label="Completion - MEOCI")
    ax2.plot(vehicle_count, completion_edgent, color="tab:orange", linestyle="--", linewidth=2.0, label="Completion - Edgent")
    ax2.plot(vehicle_count, completion_fedadapt, color="tab:green", linestyle="--", linewidth=2.0, label="Completion - FedAdapt")
    ax2.set_ylabel("Task Completion Rate (%)", fontsize=12)

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best", fontsize=9)

    plt.title("System Scalability with Increasing Vehicle Count", fontsize=13)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot scalability results vs vehicle count (Fig.16)")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV file (e.g., results/csv/scalability.csv)")
    parser.add_argument("--save", type=str, default="results/plots/fig16_vehicle_scalability.png", help="Path to save figure")
    args = parser.parse_args()

    plot_vehicle_scalability(data_path=args.data, save_path=args.save)
