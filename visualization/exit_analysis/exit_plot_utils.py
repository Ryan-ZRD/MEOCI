"""
visualization.exit_analysis.exit_plot_utils
----------------------------------------------------------
Utility module for Exit Probability visualization.

Used by:
    - plot_exit_probability_alexnet.py
    - plot_exit_probability_vgg16.py

Provides:
    - CSV data loader and validation
    - Consistent bar style and color mapping
    - Global style setup for exit probability plots
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from visualization.shared_styles import set_global_plot_style


# ------------------------------------------------------------
# 1. Standardized Color Map for Exit Probability Analysis
# ------------------------------------------------------------
COLOR_MAP = {
    "Low Load": "#1f77b4",     # Blue
    "Medium Load": "#ff7f0e",  # Orange
    "High Load": "#2ca02c",    # Green
}


# ------------------------------------------------------------
# 2. CSV Loader with Validation
# ------------------------------------------------------------
def load_exit_csv(csv_path: str) -> pd.DataFrame:
    """
    Load exit probability data from CSV file.

    Args:
        csv_path (str): Path to CSV file (should include columns: Exit, Low Load, Medium Load, High Load)

    Returns:
        df (pd.DataFrame): Loaded and validated DataFrame.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find CSV file: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {"Exit", "Low Load", "Medium Load", "High Load"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"CSV missing required columns. Expected: {required_cols}")

    # Normalize probabilities if they don't sum to 1 per scenario
    for scenario in ["Low Load", "Medium Load", "High Load"]:
        total = df[scenario].sum()
        if total > 0 and abs(total - 1.0) > 0.01:
            df[scenario] = df[scenario] / total
    return df


# ------------------------------------------------------------
# 3. Plot Style Helper
# ------------------------------------------------------------
def setup_exit_plot(title: str, xlabel: str = "Exit Branch Index", ylabel: str = "Selection Probability"):
    """
    Apply consistent plotting style for exit probability figures.
    """
    set_global_plot_style()
    plt.figure(figsize=(7, 4.5))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.ylim(0, 1.0)


# ------------------------------------------------------------
# 4. Bar Plot Drawer
# ------------------------------------------------------------
def draw_exit_bars(df: pd.DataFrame, scenarios=None, colors=None, bar_width: float = 0.25):
    """
    Draw grouped bar chart for exit probabilities.

    Args:
        df (pd.DataFrame): Exit probability data
        scenarios (list[str]): Scenarios (e.g., ["Low Load", "Medium Load", "High Load"])
        colors (list[str]): Color list for each scenario
        bar_width (float): Width of each bar
    """
    if scenarios is None:
        scenarios = ["Low Load", "Medium Load", "High Load"]
    if colors is None:
        colors = [COLOR_MAP[s] for s in scenarios]

    exits = df["Exit"].astype(str)
    x = range(len(exits))
    offsets = [(i - len(scenarios) / 2) * bar_width * 1.1 for i in range(len(scenarios))]

    for i, scenario in enumerate(scenarios):
        plt.bar(
            [pos + offsets[i] for pos in x],
            df[scenario],
            width=bar_width,
            label=scenario,
            color=colors[i],
            edgecolor="black",
            linewidth=0.8,
        )

    plt.xticks(x, exits)
    plt.legend(title="Load Scenario", loc="upper right")


# ------------------------------------------------------------
# 5. Figure Saving Utility
# ------------------------------------------------------------
def save_exit_plot(path: str):
    """
    Save exit probability plot to file with standard format.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")


# ------------------------------------------------------------
# 6. Example Demo (Optional)
# ------------------------------------------------------------
if __name__ == "__main__":
    example_csv = "visualization/data_csv/exit_alexnet.csv"

    if not os.path.exists(example_csv):
        print("Example CSV not found, generating mock data.")
        df = pd.DataFrame({
            "Exit": [1, 2, 3, 4],
            "Low Load": [0.45, 0.35, 0.15, 0.05],
            "Medium Load": [0.32, 0.40, 0.22, 0.06],
            "High Load": [0.18, 0.30, 0.35, 0.17],
        })
    else:
        df = load_exit_csv(example_csv)

    setup_exit_plot("Demo: Exit Probability (AlexNet-ME)")
    draw_exit_bars(df)
    save_exit_plot("results/plots/demo_exit_probability_alexnet.png")
