"""
visualization.heterogeneous.plot_latency_vgg16
----------------------------------------------------------
Reproduces Fig.9(b): Inference Latency Comparison on Heterogeneous Devices (VGG16-ME)

Description:
    This figure compares the average inference latency (ms)
    of multiple collaborative inference frameworks on two edge platforms:
        - NVIDIA Jetson Nano
        - Raspberry Pi 4B

Input CSV Format (visualization/data_csv/heterogeneous_latency.csv):
-------------------------------------------------------------------
Method,Device,Latency(ms)
Vehicle-Only,Nano,135.7
Vehicle-Only,Pi4B,210.3
Edge-Only,Nano,105.2
Edge-Only,Pi4B,158.7
Edgent,Nano,97.6
Edgent,Pi4B,128.4
FedAdapt,Nano,91.3
FedAdapt,Pi4B,122.5
DINA(Fog-Based),Nano,93.8
DINA(Fog-Based),Pi4B,126.2
LBO,Nano,88.9
LBO,Pi4B,117.3
ADP-D3QN (Ours),Nano,82.4
ADP-D3QN (Ours),Pi4B,109.6
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from visualization.shared_styles import set_global_plot_style


def plot_latency_vgg16(
    csv_path: str = "visualization/data_csv/heterogeneous_latency.csv",
    save_path: str = "results/plots/fig9b_heterogeneous_latency_vgg16.png"
):
    """
    Plot heterogeneous inference latency comparison for VGG16-ME.

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
    # 2. Extract Latency Data
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
    # 4. Highlight Proposed Method (ADP-D3QN)
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
    # 5. Formatting
    # ------------------------------------------------------------
    plt.xticks(x, methods, rotation=30, ha="right")
    plt.ylabel("Average Inference Latency (ms)")
    plt.title("Heterogeneous Device Comparison (VGG16-ME)")
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
    plot_latency_vgg16()
