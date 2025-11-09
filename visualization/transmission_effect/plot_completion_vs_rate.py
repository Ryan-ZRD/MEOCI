"""
visualization.transmission_effect.plot_completion_vs_rate
----------------------------------------------------------
Reproduces Fig.12(b): Task Completion Rate vs Transmission Rate

Description:
    Compares task completion rate (%) across different collaborative
    inference methods under varying wireless transmission rates.

Input CSV Format (visualization/data_csv/transmission_effect.csv):
-----------------------------------------------------------------
Rate (Mbps),Vehicle-Only,Edge-Only,Edgent,FedAdapt,LBO,ADP-D3QN (Ours)
1,82.1,85.4,90.6,93.2,95.0,98.1
2,86.3,88.7,92.9,94.5,96.2,98.5
4,89.5,91.8,94.8,96.1,97.2,98.8
6,91.8,93.6,96.0,97.0,97.8,99.0
8,93.2,94.5,96.8,97.6,98.3,99.2
10,94.0,95.2,97.3,98.0,98.5,99.3
Units: Completion Rate (%)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from visualization.shared_styles import set_global_plot_style


def plot_completion_vs_rate(
    csv_path: str = "visualization/data_csv/transmission_effect.csv",
    save_path: str = "results/plots/fig12b_completion_vs_rate.png"
):
    """
    Plot task completion rate (%) vs wireless transmission rate.

    Args:
        csv_path (str): Path to the CSV file.
        save_path (str): Path to save the figure.
    """
    # ------------------------------------------------------------
    # 1. Setup Plot Style
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
    # 4. Plot Curves
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
    plt.ylabel("Task Completion Rate (%)")
    plt.title("Task Completion Rate vs Transmission Rate")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(title="Method", loc="lower right", frameon=True)
    plt.ylim(80, 100)
    plt.tight_layout()

    # ------------------------------------------------------------
    # 6. Save Figure
    # ------------------------------------------------------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    plot_completion_vs_rate()
