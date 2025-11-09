"""
visualization.transmission_effect.plot_latency_vs_rate
----------------------------------------------------------
Reproduces Fig.12(a): Average Inference Latency vs Transmission Rate

Description:
    Compares average inference latency (ms) of collaborative inference
    strategies under different wireless transmission rates (1–10 Mbps).

Input CSV Format (visualization/data_csv/transmission_effect.csv):
-----------------------------------------------------------------
Rate (Mbps),Vehicle-Only,Edge-Only,Edgent,FedAdapt,LBO,ADP-D3QN (Ours)
1,210.5,190.3,155.4,132.1,120.2,102.7
2,200.4,175.8,143.2,121.5,112.4,97.8
4,185.2,160.6,133.4,112.2,104.6,92.5
6,172.9,148.1,125.1,105.8,99.2,88.4
8,165.6,140.5,120.0,101.4,96.1,86.0
10,160.1,136.2,117.6,99.3,94.5,84.8
Units: Latency (milliseconds)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from visualization.shared_styles import set_global_plot_style


def plot_latency_vs_rate(
    csv_path: str = "visualization/data_csv/transmission_effect.csv",
    save_path: str = "results/plots/fig12a_latency_vs_rate.png"
):
    """
    Plot average inference latency vs wireless transmission rate.

    Args:
        csv_path (str): Path to the CSV file.
        save_path (str): Path to save the output figure.
    """
    # ------------------------------------------------------------
    # 1. Style Setup
    # ------------------------------------------------------------
    set_global_plot_style()

    # ------------------------------------------------------------
    # 2. Load Dataset
    # ------------------------------------------------------------
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing input file: {csv_path}")

    df = pd.read_csv(csv_path)
    if "Rate (Mbps)" not in df.columns:
        raise ValueError("CSV must contain column: 'Rate (Mbps)'")

    x = df["Rate (Mbps)"].values
    methods = [c for c in df.columns if c != "Rate (Mbps)"]

    # ------------------------------------------------------------
    # 3. Define Color Scheme
    # ------------------------------------------------------------
    color_map = {
        "Vehicle-Only": "#8c564b",
        "Edge-Only": "#7f7f7f",
        "Edgent": "#1f77b4",
        "FedAdapt": "#9467bd",
        "LBO": "#ff7f0e",
        "ADP-D3QN (Ours)": "#d62728"
    }

    # ------------------------------------------------------------
    # 4. Plotting
    # ------------------------------------------------------------
    plt.figure(figsize=(7.5, 5))
    for method in methods:
        y = df[method].values
        plt.plot(
            x, y,
            label=method,
            color=color_map.get(method, "#444444"),
            linewidth=2.2 if method == "ADP-D3QN (Ours)" else 1.6,
            linestyle="-" if method == "ADP-D3QN (Ours)" else "--",
            marker="o"
        )

    # ------------------------------------------------------------
    # 5. Formatting
    # ------------------------------------------------------------
    plt.xlabel("Transmission Rate (Mbps)")
    plt.ylabel("Average Inference Latency (ms)")
    plt.title("Latency vs Transmission Rate")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(title="Method", loc="upper right", frameon=True)
    plt.tight_layout()

    # ------------------------------------------------------------
    # 6. Save Output
    # ------------------------------------------------------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    plot_latency_vs_rate()
