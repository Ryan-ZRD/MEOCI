import os
import pandas as pd
import matplotlib.pyplot as plt
from visualization.shared_styles import set_global_plot_style


def plot_accuracy_vs_delay_constraints(
    csv_path: str = "visualization/data_csv/delay_constraints.csv",
    save_path: str = "results/plots/fig13a_accuracy_vs_delay_constraints.png"
):

    set_global_plot_style()

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing data file: {csv_path}")

    df = pd.read_csv(csv_path)
    if "Delay Constraint (ms)" not in df.columns:
        raise ValueError("CSV must contain 'Delay Constraint (ms)' column.")

    x = df["Delay Constraint (ms)"].values
    methods = [c for c in df.columns if c != "Delay Constraint (ms)"]


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


    plt.xlabel("Delay Constraint (ms)")
    plt.ylabel("Model Accuracy (%)")
    plt.title("Accuracy vs Delay Constraint")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(title="Method", loc="lower right", frameon=True)
    plt.ylim(75, 100)
    plt.tight_layout()


    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    plot_accuracy_vs_delay_constraints()
