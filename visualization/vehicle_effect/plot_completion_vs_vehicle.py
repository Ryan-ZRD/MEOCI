"""
visualization.vehicle_effect.plot_completion_vs_vehicle
----------------------------------------------------------
Reproduces Fig.11(b): Task Completion Rate vs Number of Vehicles

Description:
    Plots the relationship between vehicular load (number of vehicles)
    and task completion rate (%) across various inference strategies:
        Vehicle-Only, Edge-Only, Edgent, FedAdapt, LBO, ADP-D3QN (Ours)

Input CSV Format (visualization/data_csv/vehicle_effect.csv):
-------------------------------------------------------------
Vehicles,Vehicle-Only,Edge-Only,Edgent,FedAdapt,LBO,ADP-D3QN (Ours)
5,97.5,98.6,99.1,99.3,99.5,99.8
10,95.2,96.8,97.3,98.5,98.8,99.6
15,90.7,92.9,94.3,96.8,97.1,99.3
20,85.1,88.3,90.8,94.1,95.6,98.8
25,77.4,81.6,84.9,91.2,93.3,98.4
30,70.3,75.4,80.2,88.5,91.1,97.9
Units: Completion Rate (%)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from visualization.shared_styles import set_global_plot_style


def plot_completion_vs_vehicle(
    csv_path: str = "visualization/data_csv/vehicle_effect.csv",
    save_path: str = "results/plots/fig11b_completion_vs_vehicle.png"
):
    """
    Plot task completion rate (%) vs number of vehicles.

    Args:
        csv_path (str): Path to the input CSV.
        save_path (str): Path to save the resulting figure.
    """
    set_global_plot_style()

    # ------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing input file: {csv_path}")

    df = pd.read_csv(csv_path)
    if "Vehicles" not in df.columns:
        raise ValueError("CSV must contain 'Vehicles' column.")

    x = df["Vehicles"].values
    methods = [c for c in df.columns if c != "Vehicles"]

    # ------------------------------------------------------------
    # 2. Define colors and line styles
    # ------------------------------------------------------------
    color_map = {
        "Vehicle-Only": "#8c564b",
        "Edge-Only": "#7f7f7f",
        "Edgent": "#1f77b4",
        "FedAdapt": "#9467bd",
        "LBO": "#ff7f0e",
        "ADP-D3QN (Ours)": "#d62728",
    }

    # ------------------------------------------------------------
    # 3. Plot curves
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
    # 4. Formatting
    # ------------------------------------------------------------
    plt.xlabel("Number of Vehicles")
    plt.ylabel("Task Completion Rate (%)")
    plt.title("Task Completion Rate vs Number of Vehicles")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(title="Method", loc="lower left", frameon=True)
    plt.ylim(60, 100)
    plt.tight_layout()

    # ------------------------------------------------------------
    # 5. Save figure
    # ------------------------------------------------------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    plot_completion_vs_vehicle()
