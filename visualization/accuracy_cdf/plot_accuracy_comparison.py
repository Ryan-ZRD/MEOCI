

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from visualization.shared_styles import set_global_plot_style


def plot_accuracy_cdf(
    csv_path: str = "visualization/data_csv/accuracy_comparison.csv",
    save_path: str = "results/plots/fig10a_accuracy_cdf.png"
):

    set_global_plot_style()


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


    plt.xlabel("Accuracy")
    plt.ylabel("Cumulative Probability")
    plt.title("CDF of Inference Accuracy Across Frameworks")
    plt.grid(linestyle="--", alpha=0.35)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()


    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    plot_accuracy_cdf()
