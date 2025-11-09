"""
visualization.scalability.plot_dynamic_density
----------------------------------------------------------
Reproduces Fig.14 of the MEOCI paper.

Plots the effect of dynamic vehicle density on:
    - Average Inference Latency (ms)
    - Average Reward

Input:
    results/csv/scalability.csv

Columns Example:
    VehicleDensity,Latency_MEOCI,Latency_Edgent,Latency_FedAdapt,Reward_MEOCI,Reward_Edgent,Reward_FedAdapt
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from visualization.shared_styles.plot_style import set_global_plot_style
from visualization.shared_styles.smoothing import smooth_curve


def plot_dynamic_density(data_path: str, save_path: str = None):
    """Plot the impact of dynamic vehicle density on latency and reward."""
    set_global_plot_style()

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)

    densities = df["VehicleDensity"].values
    latency_meoci = smooth_curve(df["Latency_MEOCI"].values, window_size=3)
    latency_edgent = smooth_curve(df["Latency_Edgent"].values, window_size=3)
    latency_fedadapt = smooth_curve(df["Latency_FedAdapt"].values, window_size=3)

    reward_meoci = smooth_curve(df["Reward_MEOCI"].values, window_size=3)
    reward_edgent = smooth_curve(df["Reward_Edgent"].values, window_size=3)
    reward_fedadapt = smooth_curve(df["Reward_FedAdapt"].values, window_size=3)

    fig, ax1 = plt.subplots(figsize=(7, 4))

    # Left y-axis: Latency
    ax1.plot(densities, latency_meoci, color="tab:blue", marker="o", label="Latency - MEOCI")
    ax1.plot(densities, latency_edgent, color="tab:orange", marker="^", label="Latency - Edgent")
    ax1.plot(densities, latency_fedadapt, color="tab:green", marker="s", label="Latency - FedAdapt")
    ax1.set_xlabel("Vehicle Density (vehicles/km²)", fontsize=12)
    ax1.set_ylabel("Average Inference Latency (ms)", fontsize=12)
    ax1.tick_params(axis="y")

    # Right y-axis: Reward
    ax2 = ax1.twinx()
    ax2.plot(densities, reward_meoci, color="tab:blue", linestyle="--", label="Reward - MEOCI")
    ax2.plot(densities, reward_edgent, color="tab:orange", linestyle="--", label="Reward - Edgent")
    ax2.plot(densities, reward_fedadapt, color="tab:green", linestyle="--", label="Reward - FedAdapt")
    ax2.set_ylabel("Average Reward", fontsize=12)
    ax2.tick_params(axis="y")

    fig.tight_layout()
    fig.legend(loc="upper right", fontsize=9)
    plt.title("Effect of Dynamic Vehicle Density on Performance", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.4)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot effect of vehicle density on MEOCI performance (Fig.14)")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV file (e.g., results/csv/scalability.csv)")
    parser.add_argument("--save", type=str, default="results/plots/fig14_dynamic_density.png", help="Output image path")
    args = parser.parse_args()

    plot_dynamic_density(data_path=args.data, save_path=args.save)
