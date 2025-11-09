"""
visualization.scalability.scalability_utils
----------------------------------------------------------
Utility functions shared by scalability visualization scripts.
Used in plotting Fig.14–Fig.17 for MEOCI paper.

Includes:
    - Data loading from CSV files
    - Moving average smoothing
    - Normalization & statistical helpers
    - Consistent color palette and line styles
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------
# 1. Data Loading
# ------------------------------------------------------
def load_scalability_data(csv_path: str) -> pd.DataFrame:
    """Load scalability experiment data from a CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Scalability CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Empty data file: {csv_path}")
    return df


# ------------------------------------------------------
# 2. Smoothing Utilities
# ------------------------------------------------------
def moving_average(data: np.ndarray, window_size: int = 3) -> np.ndarray:
    """Apply simple moving average smoothing."""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def exponential_smoothing(data: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Apply exponential moving average smoothing."""
    smoothed = [data[0]]
    for val in data[1:]:
        smoothed.append(alpha * val + (1 - alpha) * smoothed[-1])
    return np.array(smoothed)


# ------------------------------------------------------
# 3. Normalization Helpers
# ------------------------------------------------------
def normalize(data: np.ndarray) -> np.ndarray:
    """Normalize data to [0, 1] range."""
    dmin, dmax = np.min(data), np.max(data)
    return (data - dmin) / (dmax - dmin + 1e-8)


def standardize(data: np.ndarray) -> np.ndarray:
    """Standardize data to mean 0, std 1."""
    mean, std = np.mean(data), np.std(data)
    return (data - mean) / (std + 1e-8)


# ------------------------------------------------------
# 4. Statistical Utilities
# ------------------------------------------------------
def compute_stats(data: np.ndarray):
    """Return mean and standard deviation."""
    return np.mean(data), np.std(data)


# ------------------------------------------------------
# 5. Visualization Aesthetics
# ------------------------------------------------------
def get_color_palette():
    """Return a consistent color palette for scalability plots."""
    return {
        "MEOCI": "tab:blue",
        "Edgent": "tab:orange",
        "FedAdapt": "tab:green",
        "LBO": "tab:red",
        "EdgeOnly": "tab:purple",
    }


def set_plot_labels(ax, xlabel: str, ylabel: str, title: str):
    """Set plot labels and title with consistent formatting."""
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.4)


def add_legend(ax, loc="best", fontsize=10):
    """Add formatted legend."""
    ax.legend(loc=loc, fontsize=fontsize, frameon=False)


# ------------------------------------------------------
# 6. Example Use for Quick Debug
# ------------------------------------------------------
if __name__ == "__main__":
    # Example test to verify functions
    test_path = "results/csv/scalability.csv"
    if os.path.exists(test_path):
        df = load_scalability_data(test_path)
        print("[Loaded]", df.head())
        data = df["Latency_MEOCI"].values
        print("[Mean, Std]:", compute_stats(data))
        print("[Normalized Sample]:", normalize(data[:5]))
        plt.plot(data, label="Raw")
        plt.plot(moving_average(data), label="Moving Avg")
        plt.legend()
        plt.show()
    else:
        print("Test CSV not found — skipping demo.")
