"""
visualization.ablation.plot_ablation_convergence
----------------------------------------------------------
Reproduces Fig.7(a): Convergence comparison in ablation studies.

Compares reward convergence trends among:
    - D3QN
    - A-D3QN (Adaptive ε-Greedy)
    - DP-D3QN (Dual-Pool)
    - ADP-D3QN (Proposed, Adaptive Dual-Pool)

Input CSV Format (data_csv/ablation_reward.csv):
------------------------------------------------
Episode, D3QN, A-D3QN, DP-D3QN, ADP-D3QN
1, 0.12, 0.15, 0.18, 0.21
2, 0.20, 0.25, 0.28, 0.33
...
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from visualization.shared_styles import set_global_plot_style, smooth_curve


def plot_ablation_convergence(
    csv_path: str = "visualization/data_csv/ablation_reward.csv",
    save_path: str = "results/plots/fig7a_ablation_convergence.png",
    smooth: bool = True,
    method: str = "ema"
):
    """
    Plot reward convergence of different DRL algorithms.

    Args:
        csv_path (str): Path to CSV file with episode reward data.
        save_path (str): Path to save output figure.
        smooth (bool): Apply curve smoothing.
        method (str): Smoothing method ('ema', 'ma', 'sg', 'gaussian').
    """
    # Ensure consistent plot style
    set_global_plot_style()

    # Load data
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing input file: {csv_path}")
    df = pd.read_csv(csv_path)

    if "Episode" not in df.columns:
        raise ValueError("CSV must include 'Episode' column as X-axis.")

    x = df["Episode"]
    algorithms = [c for c in df.columns if c != "Episode"]

    plt.figure(figsize=(7, 4.2))
    for algo in algorithms:
        y = df[algo].values
        if smooth:
            y = smooth_curve(y, method=method, alpha=0.25)
        plt.plot(x, y, label=algo, linewidth=2)

    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Ablation Study: Convergence Comparison")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(title="Algorithm", loc="lower right")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    plot_ablation_convergence()
