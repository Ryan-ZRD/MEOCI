"""
visualization.exit_analysis.plot_exit_probability_vgg16
----------------------------------------------------------
Reproduces Fig.8(b): Early-Exit Probability Distribution (VGG16-ME).

Description:
    Illustrates how exit selection probability varies across 5 exits
    under different vehicular load scenarios.

Input CSV Format (data_csv/exit_vgg16.csv):
-------------------------------------------
Exit, Low Load, Medium Load, High Load
1, 0.30, 0.20, 0.10
2, 0.25, 0.28, 0.18
3, 0.20, 0.27, 0.30
4, 0.15, 0.18, 0.25
5, 0.10, 0.07, 0.17
(Probabilities per column ≈ 1.0)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from visualization.shared_styles import set_global_plot_style


def plot_exit_probability_vgg16(
    csv_path: str = "visualization/data_csv/exit_vgg16.csv",
    save_path: str = "results/plots/fig8b_exit_probability_vgg16.png"
):
    """
    Plot early-exit probability distribution for VGG16-ME.

    Args:
        csv_path (str): Path to CSV file (Exit probability data)
        save_path (str): Path to save the output figure
    """
    set_global_plot_style()

    # -----------------------------
    # 1. Load CSV Data
    # -----------------------------
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing CSV file: {csv_path}")

    df = pd.read_csv(csv_path)

    if "Exit" not in df.columns:
        raise ValueError("CSV must contain 'Exit' column as index.")

    exits = df["Exit"].astype(str)
    scenarios = [col for col in df.columns if col != "Exit"]

    # -----------------------------
    # 2. Configure Figure Layout
    # -----------------------------
    x = np.arange(len(exits))
    bar_width = 0.25
    plt.figure(figsize=(7, 4.5))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Low, Medium, High
    offsets = np.linspace(-bar_width, bar_width, len(scenarios))

    # -----------------------------
    # 3. Plot Grouped Bars
    # -----------------------------
    for i, scenario in enumerate(scenarios):
        plt.bar(
            x + offsets[i],
            df[scenario],
            width=bar_width,
            label=scenario,
            color=colors[i % len(colors)],
            edgecolor="black",
            linewidth=0.8,
        )

    # -----------------------------
    # 4. Add Labels & Grid
    # -----------------------------
    plt.xlabel("Exit Branch Index")
    plt.ylabel("Selection Probability")
    plt.title("VGG16-ME: Exit Probability Distribution")
    plt.xticks(x, exits)
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend(title="Load Scenario", loc="upper right")
    plt.tight_layout()

    # -----------------------------
    # 5. Save Figure
    # -----------------------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    plot_exit_probability_vgg16()
