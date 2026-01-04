import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from visualization.shared_styles import set_global_plot_style


def plot_exit_probability_vgg16(
    csv_path: str = "visualization/data_csv/exit_vgg16.csv",
    save_path: str = "results/plots/fig8b_exit_probability_vgg16.png"
):

    set_global_plot_style()


    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing CSV file: {csv_path}")

    df = pd.read_csv(csv_path)

    if "Exit" not in df.columns:
        raise ValueError("CSV must contain 'Exit' column as index.")

    exits = df["Exit"].astype(str)
    scenarios = [col for col in df.columns if col != "Exit"]


    x = np.arange(len(exits))
    bar_width = 0.25
    plt.figure(figsize=(7, 4.5))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Low, Medium, High
    offsets = np.linspace(-bar_width, bar_width, len(scenarios))


    for i, scenario in enumerate(scenarios):
        plt.bar(
            x + offsets[i],
            df[scenario],
            width=bar_width,
            label=scenario,
            color=colors[i % len(colors)],
            edgecolor="black",
            linewidth=0.8,
        )


    plt.xlabel("Exit Branch Index")
    plt.ylabel("Selection Probability")
    plt.title("VGG16-ME: Exit Probability Distribution")
    plt.xticks(x, exits)
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend(title="Load Scenario", loc="upper right")
    plt.tight_layout()


    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    plot_exit_probability_vgg16()
