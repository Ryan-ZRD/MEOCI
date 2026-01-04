
import os
import pandas as pd
import matplotlib.pyplot as plt
from visualization.shared_styles.plot_style import set_global_plot_style
from visualization.shared_styles.smoothing import smooth_curve


def plot_computational_stress(data_path: str, save_path: str = None):
    """Plot inference latency vs. RSU computational stress (CPU load)."""
    set_global_plot_style()

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"CSV file not found: {data_path}")

    df = pd.read_csv(data_path)

    cpu_load = df["CPU_Load"].values
    meoci = smooth_curve(df["MEOCI"].values, window_size=3)
    edgent = smooth_curve(df["Edgent"].values, window_size=3)
    fedadapt = smooth_curve(df["FedAdapt"].values, window_size=3)
    lbo = smooth_curve(df["LBO"].values, window_size=3)

    plt.figure(figsize=(7, 4))
    plt.plot(cpu_load, meoci, color="tab:blue", marker="o", linewidth=2.0, label="MEOCI (ADP-D3QN)")
    plt.plot(cpu_load, edgent, color="tab:orange", marker="^", linewidth=2.0, label="Edgent")
    plt.plot(cpu_load, fedadapt, color="tab:green", marker="s", linewidth=2.0, label="FedAdapt")
    plt.plot(cpu_load, lbo, color="tab:red", marker="D", linewidth=2.0, label="LBO")

    plt.xlabel("RSU CPU Load (%)", fontsize=12)
    plt.ylabel("Average Inference Latency (ms)", fontsize=12)
    plt.title("Effect of Computational Stress on Inference Latency", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot effect of computational stress on latency (Fig.17)")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV file (e.g., results/csv/scalability.csv)")
    parser.add_argument("--save", type=str, default="results/plots/fig17_computational_stress.png", help="Path to save output figure")
    args = parser.parse_args()

    plot_computational_stress(data_path=args.data, save_path=args.save)
