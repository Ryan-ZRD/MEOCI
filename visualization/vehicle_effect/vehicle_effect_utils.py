"""
visualization.vehicle_effect.vehicle_effect_utils
----------------------------------------------------------
Utility functions for analyzing and preprocessing vehicular load
experiments (Fig.11). Used by:

    - plot_latency_vs_vehicle.py
    - plot_completion_vs_vehicle.py

Provides:
    • Latency curve smoothing
    • Completion rate normalization
    • Statistical summaries (mean, std, slope)
    • Trend detection for scalability evaluation
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# ------------------------------------------------------------
# 1. Curve Smoothing
# ------------------------------------------------------------
def smooth_curve(y: np.ndarray, window: int = 5, polyorder: int = 2) -> np.ndarray:
    """
    Apply Savitzky–Golay filter for curve smoothing.

    Args:
        y (np.ndarray): Input data array (latency or completion rate).
        window (int): Smoothing window (odd integer).
        polyorder (int): Polynomial order for filtering.

    Returns:
        np.ndarray: Smoothed curve.
    """
    if len(y) < window:
        return y
    if window % 2 == 0:
        window += 1
    return savgol_filter(y, window_length=window, polyorder=polyorder)


# ------------------------------------------------------------
# 2. Normalization
# ------------------------------------------------------------
def normalize_completion_rate(values: np.ndarray) -> np.ndarray:
    """
    Normalize completion rate values to 0–100 range (percentage).

    Args:
        values (np.ndarray): Input array of completion rates.

    Returns:
        np.ndarray: Normalized array (0–100).
    """
    v_min, v_max = np.min(values), np.max(values)
    if v_max == v_min:
        return np.full_like(values, 100.0)
    return (values - v_min) / (v_max - v_min) * 100


def normalize_latency(values: np.ndarray) -> np.ndarray:
    """
    Normalize latency values to relative performance scale (0–1).

    Args:
        values (np.ndarray): Latency array (ms).

    Returns:
        np.ndarray: Normalized latency values (lower is better).
    """
    v_min, v_max = np.min(values), np.max(values)
    return 1 - (values - v_min) / (v_max - v_min)


# ------------------------------------------------------------
# 3. Statistical Summaries
# ------------------------------------------------------------
def summarize_trends(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Compute trend statistics (mean, std, slope) for performance curves.

    Args:
        x (np.ndarray): Independent variable (e.g., number of vehicles).
        y (np.ndarray): Dependent variable (e.g., latency or completion rate).

    Returns:
        dict: Statistical summary containing slope and variance.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have equal length.")
    slope = np.polyfit(x, y, 1)[0]
    return {
        "mean": float(np.mean(y)),
        "std": float(np.std(y)),
        "slope": float(slope),
        "improvement": round((y[0] - y[-1]) / y[0] * 100, 2)
        if y[0] != 0 else 0.0
    }


# ------------------------------------------------------------
# 4. Comparison Across Methods
# ------------------------------------------------------------
def compare_methods(df: pd.DataFrame, x_col: str = "Vehicles") -> pd.DataFrame:
    """
    Generate a comparative summary table for all methods in a vehicular-load dataset.

    Args:
        df (pd.DataFrame): DataFrame containing columns ['Vehicles', method1, method2, ...].
        x_col (str): Independent variable (default='Vehicles').

    Returns:
        pd.DataFrame: Summary table with mean, std, and trend slope for each method.
    """
    methods = [c for c in df.columns if c != x_col]
    summary = []
    for m in methods:
        stats = summarize_trends(df[x_col].values, df[m].values)
        stats["Method"] = m
        summary.append(stats)
    return pd.DataFrame(summary).set_index("Method")


# ------------------------------------------------------------
# 5. Example Usage
# ------------------------------------------------------------
if __name__ == "__main__":
    print("=== Vehicle Effect Utility Demo ===")
    # Synthetic example data
    data = {
        "Vehicles": [5, 10, 15, 20, 25, 30],
        "Vehicle-Only": [120, 140, 160, 180, 205, 230],
        "ADP-D3QN (Ours)": [70, 73, 75, 78, 80, 82],
    }
    df = pd.DataFrame(data)

    # Compute summaries
    result = compare_methods(df)
    print(result)

    # Demonstrate smoothing
    y_smooth = smooth_curve(df["ADP-D3QN (Ours)"].values)
    print(f"Smoothed ADP-D3QN: {np.round(y_smooth, 2)}")
