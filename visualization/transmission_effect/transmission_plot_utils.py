

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter



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



def smooth_curve(y, window=3, poly=1):
    if len(y) < window:
        return np.array(y)
    if window % 2 == 0:
        window += 1
    return savgol_filter(y, window_length=window, polyorder=poly)



def apply_plot_style(ax, xlabel, ylabel, title=None, grid=True):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=12, fontweight="semibold")
    if grid:
        ax.grid(True, linestyle="--", alpha=0.35)


def draw_transmission_curves(ax, x, df, ylabel, smooth=False):
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
