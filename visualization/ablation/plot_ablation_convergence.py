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
