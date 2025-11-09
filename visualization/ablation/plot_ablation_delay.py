"""
visualization.ablation.plot_ablation_delay
----------------------------------------------------------
Reproduces Fig.7(b): Delay comparison in ablation study.

Compares average inference latency trends of:
    - D3QN
    - A-D3QN
    - DP-D3QN
    - ADP-D3QN (Proposed)

Input CSV Format (data_csv/ablation_delay.csv):
------------------------------------------------
Episode, D3QN, A-D3QN, DP-D3QN, ADP-D3QN
1, 250, 230, 210, 195
2, 240, 225, 205, 190
...
Units: Latency (milliseconds)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from visualization.shared_styles import set_global_plot_style, smooth_curve


def plot_ablation_delay(
    csv_path: str = "visualization/data_csv/ablation_delay.csv",
    save_path: str = "results/plots/fig7b_ablation_delay.png",
    smooth: bool = True,
    method: str = "ema"
):
    """
    Plot average inference delay during ablation training.

    Args:
        csv_path (str): Path to CSV file containing delay data.
        save_path (str): Path to save output figure.
        smooth (bool): Apply smoothing to reduce noise.
        method (str): Smoothing method ('ema', 'ma', 'sg', 'gaussian').
    """
    # Apply unified style
    set_global_plot_style()

    # Load CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing input file: {csv_path}")
    df = pd.read_csv(csv_path)

    if "Episode" not in df.columns:
        raise ValueError("CSV must contain 'Episode' column as X-axis.")

    x = df["Episode"]
    algorithms = [c for c in df.columns if c != "Episode"]

    plt.figure(figsize=(7, 4.2))
    for algo in algorithms:
        y = df[algo].values
        if smooth:
            y = smooth_curve(y, method=method, alpha=0.25)
        plt.plot(x, y, label=algo, linewidth=2)

    plt.xlabel("Episode")
    plt.ylabel("Average Inference Latency (ms)")
    plt.title("Ablation Study: Delay Reduction Comparison")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(title="Algorithm", loc="upper right")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    plot_ablation_delay()
