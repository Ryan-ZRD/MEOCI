"""
visualization.accuracy_cdf.plot_accuracy_comparison
----------------------------------------------------------
Reproduces Fig.10(a): Accuracy CDF Comparison of Edge Inference Frameworks

Description:
    Plots the cumulative distribution function (CDF) of prediction accuracy
    across multiple inference frameworks (Vehicle-Only, Edge-Only, Edgent, FedAdapt, LBO, ADP-D3QN).

Input CSV Format (visualization/data_csv/accuracy_comparison.csv):
-------------------------------------------------------------------
Method,Accuracy
Vehicle-Only,0.74
Vehicle-Only,0.76
...
ADP-D3QN (Ours),0.90
ADP-D3QN (Ours),0.92
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from visualization.shared_styles import set_global_plot_style


def plot_accuracy_cdf(
    csv_path: str = "visualization/data_csv/accuracy_comparison.csv",
    save_path: str = "results/plots/fig10a_accuracy_cdf.png"
):
    """
    Plot CDF comparison of model inference accuracy across different frameworks.

    Args:
        csv_path (str): Path to input CSV containing accuracy samples.
        save_path (str): Output path for saving the figure.
    """
    set_global_plot_style()

    # ------------------------------------------------------------
    # 1. Load Data
    # ------------------------------------------------------------
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing CSV file: {csv_path}")

    df = pd.read_csv(csv_path)
    if not {"Method", "Accuracy"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: 'Method', 'Accuracy'")

    # Sort methods in consistent order for comparison
    method_order = [
        "Vehicle-Only", "Edge-Only", "Edgent", "FedAdapt",
        "DINA(Fog-Based)", "LBO", "ADP-D3QN (Ours)"
    ]

    # Define color map (consistent with previous figures)
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
    # 2. Compute CDF for each method
    # ------------------------------------------------------------
    plt.figure(figsize=(7.5, 5))
    for method in method_order:
        acc_values = df[df["Method"] == method]["Accuracy"].dropna().values
        if len(acc_values) == 0:
            continue
        sorted_acc = np.sort(acc_values)
        cdf = np.arange(1, len(sorted_acc) + 1) / len(sorted_acc)
        plt.plot(
            sorted_acc,
            cdf,
            label=method,
            linewidth=2.0 if method == "ADP-D3QN (Ours)" else 1.4,
            color=color_map.get(method, "#333333"),
            linestyle="-" if method == "ADP-D3QN (Ours)" else "--"
        )

    # ------------------------------------------------------------
    # 3. Plot Formatting
    # ------------------------------------------------------------
    plt.xlabel("Accuracy")
    plt.ylabel("Cumulative Probability")
    plt.title("CDF of Inference Accuracy Across Frameworks")
    plt.grid(linestyle="--", alpha=0.35)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()

    # ------------------------------------------------------------
    # 4. Save
    # ------------------------------------------------------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    plot_accuracy_cdf()
