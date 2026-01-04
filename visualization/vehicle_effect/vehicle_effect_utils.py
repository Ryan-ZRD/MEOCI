

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def smooth_curve(y: np.ndarray, window: int = 5, polyorder: int = 2) -> np.ndarray:

    if len(y) < window:
        return y
    if window % 2 == 0:
        window += 1
    return savgol_filter(y, window_length=window, polyorder=polyorder)


def normalize_completion_rate(values: np.ndarray) -> np.ndarray:

    v_min, v_max = np.min(values), np.max(values)
    if v_max == v_min:
        return np.full_like(values, 100.0)
    return (values - v_min) / (v_max - v_min) * 100


def normalize_latency(values: np.ndarray) -> np.ndarray:

    v_min, v_max = np.min(values), np.max(values)
    return 1 - (values - v_min) / (v_max - v_min)



def summarize_trends(x: np.ndarray, y: np.ndarray) -> dict:

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



def compare_methods(df: pd.DataFrame, x_col: str = "Vehicles") -> pd.DataFrame:

    methods = [c for c in df.columns if c != x_col]
    summary = []
    for m in methods:
        stats = summarize_trends(df[x_col].values, df[m].values)
        stats["Method"] = m
        summary.append(stats)
    return pd.DataFrame(summary).set_index("Method")


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
