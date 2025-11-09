"""
visualization.energy_constraints.energy_plot_utils
--------------------------------------------------
Helper utilities for plotting energy constraint experiments (Fig.14a–b).

Includes:
    • METHOD_COLORS   → Unified color palette across methods
    • METHOD_STYLES   → Line and marker style presets
    • draw_energy_curve()  → Standardized curve plotting
    • annotate_figure()    → Auto-annotate key points on the figure
"""

import matplotlib.pyplot as plt
import numpy as np
from visualization.shared_styles.smoothing import exponential_moving_average


# ------------------------------------------------------------
# 1. Unified color and line style
# ------------------------------------------------------------
METHOD_COLORS = {
    "Vehicle-Only": "#8c564b",
    "Edge-Only": "#7f7f7f",
    "Edgent": "#1f77b4",
    "FedAdapt": "#9467bd",
    "LBO": "#ff7f0e",
    "ADP-D3QN (Ours)": "#d62728",
}

METHOD_STYLES = {
    "Vehicle-Only": {"linestyle": "--", "marker": "s"},
    "Edge-Only": {"linestyle": "--", "marker": "v"},
    "Edgent": {"linestyle": "-.", "marker": "o"},
    "FedAdapt": {"linestyle": "-.", "marker": "^"},
    "LBO": {"linestyle": "--", "marker": "D"},
    "ADP-D3QN (Ours)": {"linestyle": "-", "marker": "o"},
}


# ------------------------------------------------------------
# 2. Draw energy curves (shared between latency and energy plots)
# ------------------------------------------------------------
def draw_energy_curve(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
    ax: plt.Axes,
    smooth_alpha: float = 0.3,
    linewidth: float = 2.0
):
    """
    Draw a single smoothed curve for a given method on a provided axis.

    Args:
        x (np.ndarray): X-axis data (Energy Constraint, mJ).
        y (np.ndarray): Y-axis data (Latency / Energy Consumption).
        method (str): Method name (used for color/style mapping).
        ax (plt.Axes): Target matplotlib Axes to draw on.
        smooth_alpha (float): Exponential smoothing coefficient.
        linewidth (float): Line width for plotting.
    """
    y_smooth = exponential_moving_average(y, alpha=smooth_alpha)

    style = METHOD_STYLES.get(method, {"linestyle": "--", "marker": "o"})
    color = METHOD_COLORS.get(method, "#444444")

    ax.plot(
        x,
        y_smooth,
        label=method,
        color=color,
        linestyle=style["linestyle"],
        marker=style["marker"],
        linewidth=2.2 if "Ours" in method else linewidth,
        markersize=5,
        alpha=0.95,
    )


# ------------------------------------------------------------
# 3. Annotate figure
# ------------------------------------------------------------
def annotate_figure(ax: plt.Axes, x_pos: float, y_pos: float, text: str, color: str = "black"):
    """
    Add textual annotation to highlight specific results.

    Args:
        ax (plt.Axes): Target axis.
        x_pos (float): X coordinate for text.
        y_pos (float): Y coordinate for text.
        text (str): Annotation label.
        color (str): Text color.
    """
    ax.text(
        x_pos,
        y_pos,
        text,
        color=color,
        fontsize=10.5,
        fontweight="medium",
        ha="left",
        va="bottom",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.2"),
    )


# ------------------------------------------------------------
# 4. Shared grid & legend configuration
# ------------------------------------------------------------
def finalize_energy_plot(ax: plt.Axes, xlabel: str, ylabel: str, title: str):
    """
    Apply consistent final styling for all energy-related plots.

    Args:
        ax (plt.Axes): The matplotlib axis to format.
        xlabel (str): Label for the X-axis.
        ylabel (str): Label for the Y-axis.
        title (str): Plot title.
    """
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, weight="semibold", pad=8)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(
        title="Method",
        loc="best",
        frameon=True,
        fontsize=10,
        title_fontsize=10.5,
    )


# ------------------------------------------------------------
# 5. Example usage
# ------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Sample test data
    x = np.array([50, 75, 100, 125, 150, 175])
    y_methods = {
        "Vehicle-Only": np.array([90, 85, 81, 78, 76, 75]),
        "ADP-D3QN (Ours)": np.array([70, 66, 63, 61, 60, 59]),
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    for method, y in y_methods.items():
        draw_energy_curve(x, y, method, ax)

    annotate_figure(ax, 150, 60, "Ours (Lowest Latency)", color="#d62728")
    finalize_energy_plot(ax, "Energy Constraint (mJ)", "Latency (ms)", "Demo Energy Curve")
    plt.tight_layout()
    plt.show()
