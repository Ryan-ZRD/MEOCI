import os
import pandas as pd
import matplotlib.pyplot as plt
from visualization.shared_styles import set_global_plot_style
from visualization.shared_styles.smoothing import exponential_moving_average


def plot_latency_vs_energy(
    csv_path: str = "visualization/data_csv/energy_constraints.csv",
    save_path: str = "results/plots/fig14a_latency_vs_energy.png",
    smooth_alpha: float = 0.3
):

    set_global_plot_style()


    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "Energy Constraint (mJ)" not in df.columns:
        raise ValueError("CSV must include column 'Energy Constraint (mJ)'")

    x = df["Energy Constraint (mJ)"].values
    methods = [c for c in df.columns if c != "Energy Constraint (mJ)"]


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


    plt.xlabel("Maximum Energy Constraint (mJ)")
    plt.ylabel("Average Inference Latency (ms)")
    plt.title("Average Latency vs Energy Constraint")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(title="Method", loc="upper right", frameon=True)
    plt.ylim(50, 100)
    plt.tight_layout()


    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    plot_latency_vs_energy()
