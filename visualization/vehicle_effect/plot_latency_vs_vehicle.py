"""
visualization.vehicle_effect.plot_latency_vs_vehicle
----------------------------------------------------------
Reproduces Fig.11(a): Average Inference Latency vs Number of Vehicles

Description:
    Compares average inference latency (ms) of different collaborative
    inference methods (Vehicle-Only, Edge-Only, Edgent, FedAdapt, LBO, ADP-D3QN)
    under varying vehicular load (5–30 vehicles).

Input CSV Format (visualization/data_csv/vehicle_effect.csv):
-------------------------------------------------------------
Vehicles,Vehicle-Only,Edge-Only,Edgent,FedAdapt,LBO,ADP-D3QN (Ours)
5,120.4,95.1,85.6,78.2,74.5,68.9
10,135.8,110.6,96.7,84.3,80.2,70.1
15,150.3,120.2,101.3,89.6,85.5,73.2
20,165.5,130.8,110.9,97.1,90.3,75.6
25,190.1,145.3,120.8,108.5,96.7,78.4
30,210.2,160.1,130.5,120.8,105.9,81.3
Units: Latency (ms)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from visualization.shared_styles import set_global_plot_style, apply_plot_style


def plot_latency_vs_vehicle(
    csv_path: str = "visualization/data_csv/vehicle_effect.csv",
    save_path: str = "results/plots/fig11a_latency_vs_vehicle.png",
    smooth: bool = False
):
    """
    Plot average inference latency vs number of vehicles.

    Args:
        csv_path (str): Path to input CSV file.
        save_path (str): Path to save the figure.
        smooth (bool): Whether to apply smoothing for visual clarity.
    """
    # 统一绘图风格
    set_global_plot_style()

    # ------------------------------------------------------------
    # 1. Load Data
    # ------------------------------------------------------------
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing input file: {csv_path}")

    df = pd.read_csv(csv_path)
    if "Vehicles" not in df.columns:
        raise ValueError("CSV must contain a 'Vehicles' column.")

    x = df["Vehicles"].values
    methods = [c for c in df.columns if c != "Vehicles"]

    # 定义颜色与线型
    color_map = {
        "Vehicle-Only": "#8c564b",
        "Edge-Only": "#7f7f7f",
        "Edgent": "#1f77b4",
        "FedAdapt": "#9467bd",
        "LBO": "#ff7f0e",
        "ADP-D3QN (Ours)": "#d62728"
    }

    # ------------------------------------------------------------
    # 2. Plotting
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
    # 3. Formatting
    # ------------------------------------------------------------
    plt.xlabel("Number of Vehicles")
    plt.ylabel("Average Inference Latency (ms)")
    plt.title("Latency vs Number of Vehicles")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(title="Method", loc="upper left", frameon=True)
    plt.tight_layout()

    # ------------------------------------------------------------
    # 4. Save
    # ------------------------------------------------------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    plot_latency_vs_vehicle()
