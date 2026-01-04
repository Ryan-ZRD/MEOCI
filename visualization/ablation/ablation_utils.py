import os
import pandas as pd
import matplotlib.pyplot as plt
from visualization.shared_styles import (
    set_global_plot_style,
    smooth_curve,
)



COLOR_MAP = {
    "D3QN": "#1f77b4",         # Blue
    "A-D3QN": "#ff7f0e",       # Orange
    "DP-D3QN": "#2ca02c",      # Green
    "ADP-D3QN": "#d62728",     # Red (Proposed)
}



def load_ablation_csv(csv_path: str):
    """
    Load ablation data CSV (Reward or Delay).

    Args:
        csv_path (str): Path to CSV file (must contain 'Episode' column)

    Returns:
        (pd.DataFrame, list[str]): DataFrame and algorithm column names
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find CSV file: {csv_path}")

    df = pd.read_csv(csv_path)
    if "Episode" not in df.columns:
        raise ValueError("CSV file must contain an 'Episode' column.")

    algorithms = [col for col in df.columns if col != "Episode"]
    return df, algorithms



def setup_plot(title: str, xlabel: str, ylabel: str):
    """
    Apply consistent style and labeling to ablation plots.
    """
    set_global_plot_style()
    plt.figure(figsize=(7, 4.2))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.35)



def plot_curves(df, algorithms, smooth=True, method="ema", ylabel=None):
    """
    Plot ablation comparison curves with consistent color and smoothing.

    Args:
        df (pd.DataFrame): Input data (with 'Episode' column)
        algorithms (list[str]): List of algorithm names
        smooth (bool): Whether to apply smoothing
        method (str): Smoothing method ('ema', 'ma', 'sg', 'gaussian')
        ylabel (str): Optional Y-axis label
    """
    x = df["Episode"]

    for algo in algorithms:
        y = df[algo].values
        if smooth:
            y = smooth_curve(y, method=method, alpha=0.25)
        color = COLOR_MAP.get(algo, None)
        plt.plot(x, y, label=algo, linewidth=2, color=color)

    if ylabel:
        plt.ylabel(ylabel)
    plt.legend(title="Algorithm", loc="best")
    plt.tight_layout()


def save_plot(path: str):
    """
    Save current figure to file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")


if __name__ == "__main__":
    import numpy as np

    # Create demo dataset for testing
    demo_csv = "visualization/data_csv/ablation_reward.csv"
    if os.path.exists(demo_csv):
        df, algos = load_ablation_csv(demo_csv)
    else:
        print(f"Warning: {demo_csv} not found, generating synthetic data.")
        x = np.arange(1, 100)
        df = pd.DataFrame({
            "Episode": x,
            "D3QN": 0.3 * (1 - np.exp(-x / 40)),
            "A-D3QN": 0.4 * (1 - np.exp(-x / 35)),
            "DP-D3QN": 0.5 * (1 - np.exp(-x / 30)),
            "ADP-D3QN": 0.65 * (1 - np.exp(-x / 25)),
        })
        algos = ["D3QN", "A-D3QN", "DP-D3QN", "ADP-D3QN"]

    setup_plot(
        title="Ablation Study: Reward Convergence (Demo)",
        xlabel="Episode",
        ylabel="Average Reward"
    )
    plot_curves(df, algos, smooth=True, method="ema")
    save_plot("results/plots/demo_ablation_reward.png")
