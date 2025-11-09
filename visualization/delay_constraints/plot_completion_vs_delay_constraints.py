"""
visualization.delay_constraints.plot_completion_vs_delay_constraints
--------------------------------------------------------------------
Reproduces Fig.13(b): Task Completion Rate vs Delay Constraint

Description:
    Visualizes how different latency constraints affect task
    completion rate (%) across collaborative inference strategies.

Expected CSV Format (visualization/data_csv/delay_constraints.csv):
--------------------------------------------------------------------
Delay Constraint (ms),Vehicle-Only,Edge-Only,Edgent,FedAdapt,LBO,ADP-D3QN (Ours)
50,78.4,80.1,83.5,88.7,90.9,94.1
100,81.0,83.2,86.8,91.5,93.0,95.6
150,83.2,85.1,88.7,92.7,94.1,96.3
200,84.5,86.0,89.8,93.4,94.7,96.7
250,85.0,86.4,90.3,93.8,95.1,96.9
300,85.3,86.6,90.6,94.0,95.3,97.0
Units: Task Completion Rate (%)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from visualization.shared_styles import set_global_plot_style


def plot_completion_vs_delay_constraints(
    csv_path: str = "visualization/data_csv/delay_constraints.csv",
    save_path: str = "results/plots/fig13b_completion_vs_delay_constraints.png"
):
    """
    Plot task completion rate vs delay constraint.

    Args:
        csv_path (str): Path to CSV data file.
        save_path (str): Path to save the output figure.
    """
    # ------------------------------------------------------------
    # 1. Initialize style
    # ------------------------------------------------------------
    set_global_plot_style()

    # ------------------------------------------------------------
    # 2. Load dataset
    # ------------------------------------------------------------
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing dataset: {csv_path}")

    df = pd.read_csv(csv_path)
    if "Delay Constraint (ms)" not in df.columns:
        raise ValueError("CSV must contain column 'Delay Constraint (ms)'")

    x = df["Delay Constraint (ms)"].values
    methods = [c for c in df.columns if c != "Delay Constraint (ms)"]

    # ------------------------------------------------------------
    # 3. Define color and style maps
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
    # 4. Plot curves
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
    plt.ylabel("Task Completion Rate (%)")
    plt.title("Task Completion Rate vs Delay Constraint")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(title="Method", loc="lower right", frameon=True)
    plt.ylim(75, 100)
    plt.tight_layout()

    # ------------------------------------------------------------
    # 6. Save figure
    # ------------------------------------------------------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    plot_completion_vs_delay_constraints()
