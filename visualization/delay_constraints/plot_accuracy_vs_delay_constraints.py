"""
visualization.delay_constraints.plot_accuracy_vs_delay_constraints
-----------------------------------------------------------------
Reproduces Fig.13(a): Accuracy vs Delay Constraint.

Description:
    Visualizes how varying latency constraints affect the
    accuracy of collaborative inference strategies.

Expected CSV Format (visualization/data_csv/delay_constraints.csv):
-------------------------------------------------------------------
Delay Constraint (ms),Vehicle-Only,Edge-Only,Edgent,FedAdapt,LBO,ADP-D3QN (Ours)
50,78.5,80.2,84.7,88.4,90.1,92.8
100,81.9,84.6,87.5,90.9,92.3,94.6
150,83.3,86.1,89.0,91.8,93.1,95.1
200,84.5,87.0,90.1,92.5,93.6,95.5
250,85.0,87.3,90.4,92.8,93.9,95.7
300,85.3,87.5,90.7,93.0,94.1,95.9
Units: Accuracy (%)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from visualization.shared_styles import set_global_plot_style


def plot_accuracy_vs_delay_constraints(
    csv_path: str = "visualization/data_csv/delay_constraints.csv",
    save_path: str = "results/plots/fig13a_accuracy_vs_delay_constraints.png"
):
    """
    Plot model accuracy vs delay constraint.

    Args:
        csv_path (str): Path to input CSV data.
        save_path (str): Path to save the output plot.
    """
    # ------------------------------------------------------------
    # 1. Style setup
    # ------------------------------------------------------------
    set_global_plot_style()

    # ------------------------------------------------------------
    # 2. Load dataset
    # ------------------------------------------------------------
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing data file: {csv_path}")

    df = pd.read_csv(csv_path)
    if "Delay Constraint (ms)" not in df.columns:
        raise ValueError("CSV must contain 'Delay Constraint (ms)' column.")

    x = df["Delay Constraint (ms)"].values
    methods = [c for c in df.columns if c != "Delay Constraint (ms)"]

    # ------------------------------------------------------------
    # 3. Define visual properties
    # ------------------------------------------------------------
    color_map = {
        "Vehicle-Only": "#8c564b",
        "Edge-Only": "#7f7f7f",
        "Edgent": "#1f77b4",
        "FedAdapt": "#9467bd",
        "LBO": "#ff7f0e",
        "ADP-D3QN (Ours)": "#d62728"
    }

    line_style = {
        "Vehicle-Only": "--",
        "Edge-Only": "--",
        "Edgent": "-.",
        "FedAdapt": "-.",
        "LBO": "--",
        "ADP-D3QN (Ours)": "-"
    }

    # ------------------------------------------------------------
    # 4. Plotting
    # ------------------------------------------------------------
    plt.figure(figsize=(7.5, 5))
    for method in methods:
        plt.plot(
            x, df[method],
            label=method,
            color=color_map.get(method, "#444444"),
            linestyle=line_style.get(method, "--"),
            linewidth=2.2 if method == "ADP-D3QN (Ours)" else 1.6,
            marker="o"
        )

    # ------------------------------------------------------------
    # 5. Formatting
    # ------------------------------------------------------------
    plt.xlabel("Delay Constraint (ms)")
    plt.ylabel("Model Accuracy (%)")
    plt.title("Accuracy vs Delay Constraint")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(title="Method", loc="lower right", frameon=True)
    plt.ylim(75, 100)
    plt.tight_layout()

    # ------------------------------------------------------------
    # 6. Save output
    # ------------------------------------------------------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    plot_accuracy_vs_delay_constraints()
