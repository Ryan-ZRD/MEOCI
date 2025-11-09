"""
visualization.accuracy_cdf.cdf_utils
----------------------------------------------------------
Utility functions for computing and processing CDF data.

Description:
    Provides reusable CDF computation and smoothing utilities
    for accuracy and latency distribution visualization.

Used by:
    - plot_accuracy_comparison.py
    - plot_latency_cdf.py
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# ------------------------------------------------------------
# 1. Core CDF Computation
# ------------------------------------------------------------
def compute_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Cumulative Distribution Function (CDF) of a 1D data array.

    Args:
        values (np.ndarray): Input numeric array (e.g., accuracy or latency).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            x_sorted: sorted values
            cdf: cumulative probability [0, 1]
    """
    if len(values) == 0:
        raise ValueError("Input array for compute_cdf() is empty.")
    x_sorted = np.sort(values)
    cdf = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
    return x_sorted, cdf


# ------------------------------------------------------------
# 2. Group-wise CDF Calculation
# ------------------------------------------------------------
def compute_cdf_by_method(df: pd.DataFrame, value_col: str, group_col: str = "Method") -> dict:
    """
    Compute CDF for each method in a DataFrame.

    Args:
        df (pd.DataFrame): Input dataset.
        value_col (str): Column containing numeric values (e.g., 'Accuracy', 'Latency(ms)').
        group_col (str): Column containing categorical groups (e.g., 'Method').

    Returns:
        dict[str, tuple[np.ndarray, np.ndarray]]:
            Mapping from method -> (sorted_values, cdf)
    """
    if group_col not in df.columns or value_col not in df.columns:
        raise KeyError(f"Missing required columns: {group_col}, {value_col}")

    results = {}
    for method, group in df.groupby(group_col):
        values = group[value_col].dropna().values
        if len(values) == 0:
            continue
        results[method] = compute_cdf(values)
    return results


# ------------------------------------------------------------
# 3. Smoothing Utilities
# ------------------------------------------------------------
def smooth_curve(y: np.ndarray, window: int = 7, polyorder: int = 2) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing to reduce noise in CDF curves.

    Args:
        y (np.ndarray): Input CDF values.
        window (int): Smoothing window size (odd number).
        polyorder (int): Polynomial order for fitting.

    Returns:
        np.ndarray: Smoothed curve.
    """
    if len(y) < window:
        return y
    if window % 2 == 0:
        window += 1  # ensure odd
    return savgol_filter(y, window_length=window, polyorder=polyorder)


# ------------------------------------------------------------
# 4. Statistical Summaries
# ------------------------------------------------------------
def summarize_distribution(values: np.ndarray) -> dict:
    """
    Compute basic statistics (mean, std, percentiles) for distribution.

    Args:
        values (np.ndarray): Numeric array.

    Returns:
        dict: summary statistics
    """
    if len(values) == 0:
        return {"mean": None, "std": None, "p50": None, "p90": None, "p95": None}

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
    }


# ------------------------------------------------------------
# 5. Batch Summary by Method
# ------------------------------------------------------------
def summarize_by_method(df: pd.DataFrame, value_col: str, group_col: str = "Method") -> pd.DataFrame:
    """
    Compute per-method statistics for accuracy or latency.

    Args:
        df (pd.DataFrame): Input dataset.
        value_col (str): Numeric column name.
        group_col (str): Method/grouping column name.

    Returns:
        pd.DataFrame: Summary table.
    """
    summaries = []
    for method, group in df.groupby(group_col):
        stats = summarize_distribution(group[value_col].dropna().values)
        stats[group_col] = method
        summaries.append(stats)
    return pd.DataFrame(summaries).set_index(group_col)


# ------------------------------------------------------------
# 6. Demo
# ------------------------------------------------------------
if __name__ == "__main__":
    print("=== CDF Utility Demo ===")
    import matplotlib.pyplot as plt

    # Example synthetic dataset
    np.random.seed(42)
    data = pd.DataFrame({
        "Method": np.repeat(["A", "B", "C"], 100),
        "Latency(ms)": np.concatenate([
            np.random.normal(120, 15, 100),
            np.random.normal(100, 10, 100),
            np.random.normal(80, 8, 100)
        ])
    })

    results = compute_cdf_by_method(data, value_col="Latency(ms)")

    plt.figure(figsize=(7, 4))
    for m, (x, cdf) in results.items():
        plt.plot(x, cdf, label=m)
    plt.xlabel("Latency (ms)")
    plt.ylabel("CDF")
    plt.legend()
    plt.title("Demo Latency CDF")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
