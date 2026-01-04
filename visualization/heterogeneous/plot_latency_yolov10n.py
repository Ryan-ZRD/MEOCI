import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from visualization.shared_styles import set_global_plot_style


def plot_latency_yolov10n(
    csv_path: str = "visualization/data_csv/heterogeneous_latency.csv",
    save_path: str = "results/plots/fig9d_heterogeneous_latency_yolov10n.png"
):

    set_global_plot_style()

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing data file: {csv_path}")

    df = pd.read_csv(csv_path)
    if not {"Method", "Device", "Latency(ms)"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: 'Method', 'Device', 'Latency(ms)'")

    methods = df["Method"].unique()
    devices = ["Nano", "Pi4B"]

    latency_nano, latency_pi = [], []
    for method in methods:
        latency_nano.append(df[(df["Method"] == method) & (df["Device"] == "Nano")]["Latency(ms)"].values[0])
        latency_pi.append(df[(df["Method"] == method) & (df["Device"] == "Pi4B")]["Latency(ms)"].values[0])

    x = np.arange(len(methods))
    bar_width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(
        x - bar_width / 2, latency_nano, width=bar_width,
        color="#1f77b4", label="Jetson Nano", edgecolor="black", linewidth=0.8
    )
    plt.bar(
        x + bar_width / 2, latency_pi, width=bar_width,
        color="#ff7f0e", label="Raspberry Pi 4B", edgecolor="black", linewidth=0.8
    )

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

    plt.xticks(x, methods, rotation=30, ha="right")
    plt.ylabel("Average Inference Latency (ms)")
    plt.title("Heterogeneous Device Comparison (YOLOv10n-ME)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    plot_latency_yolov10n()
