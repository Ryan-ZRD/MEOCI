import os
import pandas as pd
import matplotlib.pyplot as plt
from visualization.shared_styles import set_global_plot_style


def plot_latency_vs_rate(
    csv_path: str = "visualization/data_csv/transmission_effect.csv",
    save_path: str = "results/plots/fig12a_latency_vs_rate.png"
):

    set_global_plot_style()

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing input file: {csv_path}")

    df = pd.read_csv(csv_path)
    if "Rate (Mbps)" not in df.columns:
        raise ValueError("CSV must contain column: 'Rate (Mbps)'")

    x = df["Rate (Mbps)"].values
    methods = [c for c in df.columns if c != "Rate (Mbps)"]


    color_map = {
        "Vehicle-Only": "#8c564b",
        "Edge-Only": "#7f7f7f",
        "Edgent": "#1f77b4",
        "FedAdapt": "#9467bd",
        "LBO": "#ff7f0e",
        "ADP-D3QN (Ours)": "#d62728"
    }


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


    plt.xlabel("Transmission Rate (Mbps)")
    plt.ylabel("Average Inference Latency (ms)")
    plt.title("Latency vs Transmission Rate")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(title="Method", loc="upper right", frameon=True)
    plt.tight_layout()


    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    plot_latency_vs_rate()
