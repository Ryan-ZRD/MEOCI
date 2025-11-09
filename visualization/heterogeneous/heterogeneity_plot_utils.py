"""
visualization.heterogeneous.heterogeneity_plot_utils
----------------------------------------------------------
Shared plotting utilities for heterogeneous inference latency visualization (Fig.9 series)

Description:
    Provides reusable visualization utilities used by:
        - plot_latency_alexnet.py
        - plot_latency_vgg16.py
        - plot_latency_resnet50.py
        - plot_latency_yolov10n.py

Core Features:
    • Unified color and style scheme across all models
    • Highlighting for "ADP-D3QN (Ours)"
    • Dynamic axis formatting and value labeling
    • Grid, font, and legend styling
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# 1. Global Style Configuration
# ------------------------------------------------------------
COLORS = {
    "Nano": "#1f77b4",          # Blue
    "Pi4B": "#ff7f0e",          # Orange
    "Ours": "#d62728",          # Red
}

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
})


# ------------------------------------------------------------
# 2. Utility Functions
# ------------------------------------------------------------
def create_output_dir(save_path: str):
    """Create directories for the save path if not exist."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)


def add_value_labels(ax, rects, spacing=3, fmt="{:.1f}"):
    """Add data labels above bars."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, spacing),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=9
        )


def highlight_bars(ax, x, nano_vals, pi_vals, bar_width, methods):
    """Highlight ADP-D3QN bars in red."""
    if "ADP-D3QN (Ours)" not in methods:
        return
    idx = list(methods).index("ADP-D3QN (Ours)")
    ax.bar(
        x[idx] - bar_width / 2, nano_vals[idx], width=bar_width,
        color=COLORS["Ours"], edgecolor="black", linewidth=1.0
    )
    ax.bar(
        x[idx] + bar_width / 2, pi_vals[idx], width=bar_width,
        color=COLORS["Ours"], edgecolor="black", linewidth=1.0,
        label="ADP-D3QN (Ours)"
    )


def plot_heterogeneous_latency(df: pd.DataFrame, title: str, save_path: str):
    """
    Unified grouped-bar plotting for heterogeneous device latency comparisons.

    Args:
        df (pd.DataFrame): Must contain columns ["Method", "Device", "Latency(ms)"].
        title (str): Figure title (e.g., "Heterogeneous Device Comparison (ResNet50-ME)").
        save_path (str): File path to save the figure.
    """
    if not {"Method", "Device", "Latency(ms)"}.issubset(df.columns):
        raise ValueError("DataFrame must contain ['Method', 'Device', 'Latency(ms)'].")

    methods = df["Method"].unique()
    devices = ["Nano", "Pi4B"]

    latency_nano = [df[(df["Method"] == m) & (df["Device"] == "Nano")]["Latency(ms)"].values[0] for m in methods]
    latency_pi = [df[(df["Method"] == m) & (df["Device"] == "Pi4B")]["Latency(ms)"].values[0] for m in methods]

    x = np.arange(len(methods))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    bars_nano = ax.bar(
        x - bar_width / 2, latency_nano, width=bar_width,
        color=COLORS["Nano"], label="Jetson Nano", edgecolor="black", linewidth=0.8
    )
    bars_pi = ax.bar(
        x + bar_width / 2, latency_pi, width=bar_width,
        color=COLORS["Pi4B"], label="Raspberry Pi 4B", edgecolor="black", linewidth=0.8
    )

    # Highlight our method
    highlight_bars(ax, x, latency_nano, latency_pi, bar_width, methods)

    # Add value labels
    add_value_labels(ax, bars_nano, spacing=3)
    add_value_labels(ax, bars_pi, spacing=3)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Average Inference Latency (ms)")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()

    create_output_dir(save_path)
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


def generate_demo_data() -> pd.DataFrame:
    """Generate synthetic data for quick demo or unit test."""
    data = {
        "Method": [
            "Vehicle-Only", "Edge-Only", "Edgent", "FedAdapt",
            "DINA(Fog-Based)", "LBO", "ADP-D3QN (Ours)"
        ] * 2,
        "Device": ["Nano"] * 7 + ["Pi4B"] * 7,
        "Latency(ms)": [135, 104, 95, 88, 90, 85, 77, 205, 160, 142, 129, 132, 120, 108]
    }
    return pd.DataFrame(data)


# ------------------------------------------------------------
# 3. Standalone Demo
# ------------------------------------------------------------
if __name__ == "__main__":
    df_demo = generate_demo_data()
    plot_heterogeneous_latency(
        df_demo,
        title="Heterogeneous Device Comparison (Demo)",
        save_path="results/plots/demo_heterogeneous_latency.png"
    )
