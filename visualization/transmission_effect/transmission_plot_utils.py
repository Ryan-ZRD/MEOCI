"""
visualization.transmission_effect.transmission_plot_utils
----------------------------------------------------------
Helper utilities for plotting transmission-rate experiments (Fig.12).

Provides:
    - Standardized color/style mapping across methods.
    - Smoothing of experimental curves (Savitzky-Golay).
    - Utility functions for consistent legend & axis formatting.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# ------------------------------------------------------------
# 1. Color and Style Dictionary
# ------------------------------------------------------------
METHOD_COLORS = {
    "Vehicle-Only": "#8c564b",
    "Edge-Only": "#7f7f7f",
    "Edgent": "#1f77b4",
    "FedAdapt": "#9467bd",
    "LBO": "#ff7f0e",
    "ADP-D3QN (Ours)": "#d62728"
}

METHOD_STYLES = {
    "Vehicle-Only": "--",
    "Edge-Only": "--",
    "Edgent": "-.",
    "FedAdapt": "-.",
    "LBO": "--",
    "ADP-D3QN (Ours)": "-"
}

METHOD_MARKERS = {
    "Vehicle-Only": "s",
    "Edge-Only": "D",
    "Edgent": "o",
    "FedAdapt": "^",
    "LBO": "v",
    "ADP-D3QN (Ours)": "o"
}


# ------------------------------------------------------------
# 2. Curve Smoothing (Savitzky-Golay)
# ------------------------------------------------------------
def smooth_curve(y, window=3, poly=1):
    """
    Smooth a curve using the Savitzky-Golay filter.

    Args:
        y (array-like): Input data series.
        window (int): Window length for smoothing (odd number).
        poly (int): Polynomial order for smoothing.

    Returns:
        np.ndarray: Smoothed curve.
    """
    if len(y) < window:
        return np.array(y)
    if window % 2 == 0:
        window += 1
    return savgol_filter(y, window_length=window, polyorder=poly)


# ------------------------------------------------------------
# 3. Plot Formatting Utilities
# ------------------------------------------------------------
def apply_plot_style(ax, xlabel, ylabel, title=None, grid=True):
    """
    Apply standard axis labels and optional title/grid.

    Args:
        ax (matplotlib.axes.Axes): Target axes object.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        title (str): Optional title.
        grid (bool): Whether to show grid lines.
    """
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=12, fontweight="semibold")
    if grid:
        ax.grid(True, linestyle="--", alpha=0.35)


def draw_transmission_curves(ax, x, df, ylabel, smooth=False):
    """
    Draw curves for multiple methods with consistent formatting.

    Args:
        ax (Axes): Target Matplotlib axes.
        x (array): Transmission rate array (Mbps).
        df (pd.DataFrame): DataFrame containing columns for each method.
        ylabel (str): y-axis label (e.g., 'Latency (ms)', 'Completion Rate (%)').
        smooth (bool): Whether to apply curve smoothing.
    """
    methods = [c for c in df.columns if "Rate" not in c]
    for method in methods:
        y = df[method].values
        if smooth:
            y = smooth_curve(y, window=3, poly=1)

        ax.plot(
            x, y,
            label=method,
            color=METHOD_COLORS.get(method, "#444444"),
            linestyle=METHOD_STYLES.get(method, "--"),
            marker=METHOD_MARKERS.get(method, "o"),
            linewidth=2.2 if method == "ADP-D3QN (Ours)" else 1.6,
            markersize=6
        )

    apply_plot_style(ax, xlabel="Transmission Rate (Mbps)", ylabel=ylabel)
    ax.legend(title="Method", loc="best", frameon=True)
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(bottom=0)


# ------------------------------------------------------------
# 4. Example Demo
# ------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd

    # Create mock data
    x = np.array([1, 2, 4, 6, 8, 10])
    df = pd.DataFrame({
        "Rate (Mbps)": x,
        "Vehicle-Only": [210, 200, 185, 173, 166, 160],
        "Edge-Only": [190, 176, 161, 148, 140, 136],
        "Edgent": [155, 143, 133, 125, 120, 117],
        "FedAdapt": [132, 122, 112, 106, 101, 99],
        "LBO": [120, 112, 105, 99, 96, 95],
        "ADP-D3QN (Ours)": [103, 98, 93, 89, 86, 85]
    })

    # Quick demo
    fig, ax = plt.subplots(figsize=(7, 5))
    draw_transmission_curves(ax, x, df, ylabel="Average Inference Latency (ms)", smooth=True)
    plt.tight_layout()
    plt.show()
