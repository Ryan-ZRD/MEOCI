"""
visualization.heterogeneous.plot_latency_alexnet
----------------------------------------------------------
Reproduces Fig.9(a): Inference Latency Comparison on Heterogeneous Devices (AlexNet-ME)

Description:
    Compares average inference latency (in ms) of various collaborative inference
    strategies on two heterogeneous platforms:
        - NVIDIA Jetson Nano
        - Raspberry Pi 4B

Input CSV Format (visualization/data_csv/heterogeneous_latency.csv):
-------------------------------------------------------------------
Method,Device,Latency(ms)
Vehicle-Only,Nano,85.2
Vehicle-Only,Pi4B,142.8
Edge-Only,Nano,65.4
Edge-Only,Pi4B,90.1
Edgent,Nano,58.3
Edgent,Pi4B,72.4
FedAdapt,Nano,52.1
FedAdapt,Pi4B,66.8
DINA(Fog-Based),Nano,55.7
DINA(Fog-Based),Pi4B,69.3
LBO,Nano,50.8
LBO,Pi4B,63.7
ADP-D3QN (Ours),Nano,47.5
ADP-D3QN (Ours),Pi4B,58.9
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from visualization.shared_styles import set_global_plot_style


def plot_latency_alexnet(
    csv_path: str = "visualization/data_csv/heterogeneous_latency.csv",
    save_path: str = "results/plots/fig9a_heterogeneous_latency_alexnet.png"
):
    """
    Plot the heterogeneous inference latency comparison for AlexNet-ME.

    Args:
        csv_path (str): Path to the input CSV file.
        save_path (str): Output path for saving the resulting plot.
    """
    set_global_plot_style()

    # ------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing data file: {csv_path}")

    df = pd.read_csv(csv_path)

    if not {"Method", "Device", "Latency(ms)"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: 'Method', 'Device', 'Latency(ms)'")

    methods = df["Method"].unique()
    devices = ["Nano", "Pi4B"]

    # ------------------------------------------------------------
    # 2. Prepare data
    # ------------------------------------------------------------
    latency_nano = []
    latency_pi = []
    for method in methods:
        latency_nano.append(df[(df["Method"] == method) & (df["Device"] == "Nano")]["Latency(ms)"].values[0])
        latency_pi.append(df[(df["Method"] == method) & (df["Device"] == "Pi4B")]["Latency(ms)"].values[0])

    x = np.arange(len(methods))
    bar_width = 0.35

    # ------------------------------------------------------------
    # 3. Plot bars
    # ------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    plt.bar(
        x - bar_width / 2, latency_nano, width=bar_width,
        color="#1f77b4", label="Jetson Nano", edgecolor="black", linewidth=0.8
    )
    plt.bar(
        x + bar_width / 2, latency_pi, width=bar_width,
        color="#ff7f0e", label="Raspberry Pi 4B", edgecolor="black", linewidth=0.8
    )

    # ------------------------------------------------------------
    # 4. Formatting
    # ------------------------------------------------------------
    plt.xticks(x, methods, rotation=30, ha="right")
    plt.ylabel("Average Inference Latency (ms)")
    plt.title("Heterogeneous Device Comparison (AlexNet-ME)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    # Highlight the proposed method
    idx = list(methods).index("ADP-D3QN (Ours)")
    plt.bar(
        x[idx] - bar_width / 2, latency_nano[idx], width=bar_width,
        color="#d62728", edgecolor="black", linewidth=0.9
    )
    plt.bar(
        x[idx] + bar_width / 2, latency_pi[idx], width=bar_width,
        color="#d62728", edgecolor="black", linewidth=0.9,
        label="ADP-D3QN (Ours)"
    )

    plt.tight_layout()

    # ------------------------------------------------------------
    # 5. Save Figure
    # ------------------------------------------------------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    plot_latency_alexnet()
