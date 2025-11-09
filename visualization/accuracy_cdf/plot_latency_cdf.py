"""
visualization.accuracy_cdf.plot_latency_cdf
----------------------------------------------------------
Reproduces Fig.10(b): Latency CDF Comparison of Edge Inference Frameworks

Description:
    Plots the cumulative distribution function (CDF) of inference latency (ms)
    for different edge collaboration algorithms:
        Vehicle-Only, Edge-Only, Edgent, FedAdapt, LBO, ADP-D3QN (Ours)

Input CSV Format (visualization/data_csv/latency_cdf.csv):
-------------------------------------------------------------------
Method,Latency(ms)
Vehicle-Only,215.4
Vehicle-Only,221.7
Edge-Only,160.3
Edgent,145.9
FedAdapt,138.2
...
ADP-D3QN (Ours),98.3
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from visualization.shared_styles import set_global_plot_style


def plot_latency_cdf(
    csv_path: str = "visualization/data_csv/latency_cdf.csv",
    save_path: str = "results/plots/fig10b_latency_cdf.png"
):
    """
    Plot cumulative distribution function (CDF) of inference latency (ms)
    across multiple collaborative inference methods.

    Args:
        csv_path (str): Path to input CSV file.
        save_path (str): Output path for the saved figure.
    """
    set_global_plot_style()

    # ------------------------------------------------------------
    # 1. Load Data
    # ------------------------------------------------------------
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing CSV file: {csv_path}")

    df = pd.read_csv(csv_path)
    if not {"Method", "Latency(ms)"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: 'Method', 'Latency(ms)'")

    # ------------------------------------------------------------
    # 2. Define Method Order & Colors
    # ------------------------------------------------------------
    method_order = [
        "Vehicle-Only", "Edge-Only", "Edgent",
        "FedAdapt", "DINA(Fog-Based)", "LBO", "ADP-D3QN (Ours)"
    ]

    color_map = {
        "Vehicle-Only": "#8c564b",
        "Edge-Only": "#7f7f7f",
        "Edgent": "#1f77b4",
        "FedAdapt": "#9467bd",
        "DINA(Fog-Based)": "#2ca02c",
        "LBO": "#ff7f0e",
        "ADP-D3QN (Ours)": "#d62728",
    }

    # ------------------------------------------------------------
    # 3. Compute CDF for Each Method
    # ------------------------------------------------------------
    plt.figure(figsize=(7.5, 5))
    for method in method_order:
        lat_values = df[df["Method"] == method]["Latency(ms)"].dropna().values
        if len(lat_values) == 0:
            continue
        sorted_lat = np.sort(lat_values)
        cdf = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat)

        plt.plot(
            sorted_lat,
            cdf,
            label=method,
            linewidth=2.0 if method == "ADP-D3QN (Ours)" else 1.4,
            linestyle="-" if method == "ADP-D3QN (Ours)" else "--",
            color=color_map.get(method, "#333333")
        )

    # ------------------------------------------------------------
    # 4. Formatting
    # ------------------------------------------------------------
    plt.xlabel("Inference Latency (ms)")
    plt.ylabel("Cumulative Probability")
    plt.title("CDF of Inference Latency Across Frameworks")
    plt.grid(linestyle="--", alpha=0.35)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()

    # ------------------------------------------------------------
    # 5. Save Figure
    # ------------------------------------------------------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    plot_latency_cdf()
