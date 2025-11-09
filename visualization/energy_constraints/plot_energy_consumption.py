"""
visualization.energy_constraints.plot_energy_consumption
--------------------------------------------------------
Reproduces Fig.14(b): Energy Consumption vs Energy Constraint

Description:
    This script visualizes the average energy consumption (mJ)
    under different maximum energy constraint limits across
    various collaborative inference strategies.

Expected CSV Format (visualization/data_csv/energy_constraints.csv):
--------------------------------------------------------------------
Energy Constraint (mJ),Vehicle-Only,Edge-Only,Edgent,FedAdapt,LBO,ADP-D3QN (Ours)
50,49.2,47.8,45.1,42.7,40.9,38.3
75,70.4,68.5,63.1,59.2,56.8,52.5
100,89.5,86.2,78.9,73.3,70.5,65.8
125,107.8,102.5,94.3,86.1,81.4,76.0
150,126.3,120.1,108.8,98.2,92.6,85.7
175,140.2,132.5,120.3,108.4,101.2,94.3
Units: Energy (mJ)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from visualization.shared_styles import set_global_plot_style
from visualization.shared_styles.smoothing import exponential_moving_average


def plot_energy_consumption(
    csv_path: str = "visualization/data_csv/energy_constraints.csv",
    save_path: str = "results/plots/fig14b_energy_consumption.png",
    smooth_alpha: float = 0.25
):
    """
    Plot average energy consumption (mJ) vs energy constraint (mJ).

    Args:
        csv_path (str): Path to the CSV data file.
        save_path (str): Path to save the output figure.
        smooth_alpha (float): Smoothing factor for exponential smoothing.
    """
    # ------------------------------------------------------------
    # 1. Apply global plot style
    # ------------------------------------------------------------
    set_global_plot_style()

    # ------------------------------------------------------------
    # 2. Load CSV data
    # ------------------------------------------------------------
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing dataset: {csv_path}")

    df = pd.read_csv(csv_path)
    if "Energy Constraint (mJ)" not in df.columns:
        raise ValueError("CSV must contain column 'Energy Constraint (mJ)'")

    x = df["Energy Constraint (mJ)"].values
    methods = [c for c in df.columns if c != "Energy Constraint (mJ)"]

    # ------------------------------------------------------------
    # 3. Define color/style mappings
    # ------------------------------------------------------------
    color_map = {
        "Vehicle-Only": "#8c564b",
        "Edge-Only": "#7f7f7f",
        "Edgent": "#1f77b4",
        "FedAdapt": "#9467bd",
        "LBO": "#ff7f0e",
        "ADP-D3QN (Ours)": "#d62728",
    }

    linestyle_map = {
        "Vehicle-Only": "--",
        "Edge-Only": "--",
        "Edgent": "-.",
        "FedAdapt": "-.",
        "LBO": "--",
        "ADP-D3QN (Ours)": "-",
    }

    # ------------------------------------------------------------
    # 4. Plot curves
    # ------------------------------------------------------------
    plt.figure(figsize=(7.2, 5))
    for method in methods:
        y_raw = df[method].values
        y_smooth = exponential_moving_average(y_raw, smooth_alpha)

        plt.plot(
            x, y_smooth,
            label=method,
            color=color_map.get(method, "#444444"),
            linestyle=linestyle_map.get(method, "--"),
            linewidth=2.2 if "Ours" in method else 1.6,
            marker="o",
            markersize=5
        )

    # ------------------------------------------------------------
    # 5. Plot formatting
    # ------------------------------------------------------------
    plt.xlabel("Maximum Energy Constraint (mJ)")
    plt.ylabel("Average Energy Consumption (mJ)")
    plt.title("Energy Consumption vs Energy Constraint")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(title="Method", loc="upper left", frameon=True)
    plt.tight_layout()

    # ------------------------------------------------------------
    # 6. Save figure
    # ------------------------------------------------------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    plot_energy_consumption()
