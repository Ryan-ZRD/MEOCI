"""
visualization.heterogeneous.plot_latency_resnet50
----------------------------------------------------------
Reproduces Fig.9(c): Inference Latency Comparison on Heterogeneous Devices (ResNet50-ME)

Description:
    Compares average inference latency (ms) of collaborative inference strategies
    on heterogeneous edge devices:
        - NVIDIA Jetson Nano
        - Raspberry Pi 4B

Input CSV Format (visualization/data_csv/heterogeneous_latency.csv):
-------------------------------------------------------------------
Method,Device,Latency(ms)
Vehicle-Only,Nano,178.5
Vehicle-Only,Pi4B,245.8
Edge-Only,Nano,133.4
Edge-Only,Pi4B,188.7
Edgent,Nano,120.6
Edgent,Pi4B,155.3
FedAdapt,Nano,111.7
FedAdapt,Pi4B,148.2
DINA(Fog-Based),Nano,113.9
DINA(Fog-Based),Pi4B,150.6
LBO,Nano,108.2
LBO,Pi4B,141.9
ADP-D3QN (Ours),Nano,97.3
ADP-D3QN (Ours),Pi4B,128.6
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from visualization.shared_styles import set_global_plot_style


def plot_latency_resnet50(
    csv_path: str = "visualization/data_csv/heterogeneous_latency.csv",
    save_path: str = "results/plots/fig9c_heterogeneous_latency_resnet50.png"
):
    """
    Plot heterogeneous inference latency comparison for ResNet50-ME.

    Args:
        csv_path (str): Path to input CSV file.
        save_path (str): Output file path.
    """
    set_global_plot_style()

    # ------------------------------------------------------------
    # 1. Load Data
    # ------------------------------------------------------------
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing data file: {csv_path}")

    df = pd.read_csv(csv_path)
    if not {"Method", "Device", "Latency(ms)"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: 'Method', 'Device', 'Latency(ms)'")

    methods = df["Method"].unique()
    devices = ["Nano", "Pi4B"]

    # ------------------------------------------------------------
    # 2. Extract Latency Values
    # ------------------------------------------------------------
    latency_nano, latency_pi = [], []
    for method in methods:
        latency_nano.append(df[(df["Method"] == method) & (df["Device"] == "Nano")]["Latency(ms)"].values[0])
        latency_pi.append(df[(df["Method"] == method) & (df["Device"] == "Pi4B")]["Latency(ms)"].values[0])

    x = np.arange(len(methods))
    bar_width = 0.35

    # ------------------------------------------------------------
    # 3. Plot Bars
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
    # 4. Highlight ADP-D3QN (Ours)
    # ------------------------------------------------------------
    idx = list(methods).index("ADP-D3QN (Ours)")
    plt.bar(
        x[idx] - bar_width / 2, latency_nano[idx], width=bar_width,
        color="#d62728", edgecolor="black", linewidth=1.0
    )
    plt.bar(
        x[idx] + bar_width / 2, latency_pi[idx], width=bar_width,
        color="#d62728", edgecolor="black", linewidth=1.0,
        label="ADP-D3QN (Ours)"
    )

    # ------------------------------------------------------------
    # 5. Format and Labels
    # ------------------------------------------------------------
    plt.xticks(x, methods, rotation=30, ha="right")
    plt.ylabel("Average Inference Latency (ms)")
    plt.title("Heterogeneous Device Comparison (ResNet50-ME)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()

    # ------------------------------------------------------------
    # 6. Save Figure
    # ------------------------------------------------------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    plot_latency_resnet50()
