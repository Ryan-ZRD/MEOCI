"""
utils.visualization
==========================================================
Quick visualization utilities for MEOCI framework.
----------------------------------------------------------
Provides:
    - Training convergence plotting (reward curves)
    - Latency/Energy/Accuracy trend plots
    - Multi-agent or baseline comparison
Used for:
    - Experiment debugging
    - Sanity checks before official plotting
"""

import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional


# ------------------------------------------------------------
# 🎯 1. Basic Plot Settings
# ------------------------------------------------------------
plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.figsize": (7, 4),
    "figure.dpi": 120,
    "grid.alpha": 0.3,
    "lines.linewidth": 2.0
})


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


# ------------------------------------------------------------
# 📈 2. Plot Training Curve
# ------------------------------------------------------------
def plot_reward_curve(
    csv_file: str,
    out_path: Optional[str] = None,
    title: str = "Training Convergence (ADP-D3QN)",
    smooth: int = 10
):
    """
    Plot convergence curve of reward vs episodes.

    Args:
        csv_file: Path to metrics_log.csv
        out_path: Save path for figure
        title: Title of the plot
        smooth: Moving average window
    """
    steps, rewards = [], []
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "reward" in row:
                steps.append(int(row["step"]))
                rewards.append(float(row["reward"]))

    if len(rewards) == 0:
        print(f"[Visualization] No reward found in {csv_file}")
        return

    # Smooth curve
    def smooth_curve(y, box_pts):
        box = np.ones(box_pts) / box_pts
        y_smooth = np.convolve(y, box, mode="same")
        return y_smooth

    smoothed = smooth_curve(rewards, smooth)
    plt.figure()
    plt.plot(steps, smoothed, label="ADP-D3QN", color="tab:blue")
    plt.xlabel("Training Episodes")
    plt.ylabel("Reward")
    plt.title(title)
    plt.grid(True)
    plt.legend()

    if out_path:
        ensure_dir(os.path.dirname(out_path))
        plt.savefig(out_path, bbox_inches="tight")
        print(f"[Visualization] Saved plot to {out_path}")
    else:
        plt.show()


# ------------------------------------------------------------
# ⚡ 3. Plot Metric Comparison
# ------------------------------------------------------------
def plot_metric_comparison(
    data: Dict[str, List[float]],
    xlabel: str = "Scenario Index",
    ylabel: str = "Metric Value",
    title: str = "Performance Comparison",
    labels: Optional[List[str]] = None,
    out_path: Optional[str] = None
):
    """
    Plot line comparison of multiple methods (e.g., ablation or baselines).
    """
    plt.figure()
    n = len(data)
    cmap = plt.get_cmap("tab10")

    for i, (name, values) in enumerate(data.items()):
        x = np.arange(len(values))
        plt.plot(x, values, marker="o", label=labels[i] if labels else name, color=cmap(i))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if out_path:
        ensure_dir(os.path.dirname(out_path))
        plt.savefig(out_path, bbox_inches="tight")
        print(f"[Visualization] Saved plot to {out_path}")
    else:
        plt.show()


# ------------------------------------------------------------
# 🔋 4. Latency vs Energy Trade-off
# ------------------------------------------------------------
def plot_latency_energy_tradeoff(
    latencies: List[float],
    energies: List[float],
    method_labels: Optional[List[str]] = None,
    title: str = "Latency-Energy Trade-off",
    out_path: Optional[str] = None
):
    """
    Plot tradeoff curve for latency and energy (Pareto frontier visualization).
    """
    plt.figure()
    plt.scatter(latencies, energies, c="tab:blue", alpha=0.7, label="Samples")

    # Compute Pareto frontier
    sorted_points = sorted(zip(latencies, energies), key=lambda x: x[0])
    pareto = []
    best = float("inf")
    for l, e in sorted_points:
        if e < best:
            pareto.append((l, e))
            best = e
    pareto = np.array(pareto)

    plt.plot(pareto[:, 0], pareto[:, 1], "--", color="tab:red", label="Pareto Frontier")
    if method_labels:
        for i, label in enumerate(method_labels):
            plt.text(latencies[i] + 0.3, energies[i], label, fontsize=9)

    plt.xlabel("Latency (ms)")
    plt.ylabel("Energy (J)")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if out_path:
        ensure_dir(os.path.dirname(out_path))
        plt.savefig(out_path, bbox_inches="tight")
        print(f"[Visualization] Saved plot to {out_path}")
    else:
        plt.show()


# ------------------------------------------------------------
# 📊 5. Accuracy vs Delay Constraint
# ------------------------------------------------------------
def plot_accuracy_vs_delay(
    delays: List[float],
    accuracies: List[float],
    label: str = "ADP-D3QN",
    out_path: Optional[str] = None
):
    """
    Plot accuracy under different delay constraints (Fig.12 equivalent).
    """
    plt.figure()
    plt.plot(delays, accuracies, marker="s", color="tab:green", label=label)
    plt.xlabel("Delay Constraint (ms)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Delay Constraint")
    plt.grid(True)
    plt.legend()

    if out_path:
        ensure_dir(os.path.dirname(out_path))
        plt.savefig(out_path, bbox_inches="tight")
        print(f"[Visualization] Saved plot to {out_path}")
    else:
        plt.show()


# ------------------------------------------------------------
# ✅ Example Usage
# ------------------------------------------------------------
if __name__ == "__main__":
    # 1. Reward convergence example
    plot_reward_curve(
        csv_file="./results/logs/adp_d3qn_train/metrics_log.csv",
        out_path="./results/plots/convergence.png"
    )

    # 2. Metric comparison
    methods = {
        "ADP-D3QN": [92.3, 93.1, 93.6, 94.0],
        "D3QN": [88.2, 89.4, 90.5, 90.9],
        "A-D3QN": [90.1, 91.5, 92.0, 92.7]
    }
    plot_metric_comparison(
        data=methods,
        xlabel="Vehicle Density Scenario",
        ylabel="Accuracy (%)",
        title="Model Accuracy Comparison",
        out_path="./results/plots/accuracy_comparison.png"
    )

    # 3. Latency-Energy trade-off
    lat = [25.1, 26.5, 27.8, 29.2]
    en = [4.2, 3.9, 3.7, 3.5]
    plot_latency_energy_tradeoff(lat, en, out_path="./results/plots/tradeoff.png")
