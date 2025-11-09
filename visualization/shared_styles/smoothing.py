"""
visualization.shared_styles.smoothing
----------------------------------------------------------
Provides smoothing and filtering utilities for figure generation
in the MEOCI project.

These methods are mainly used to process noisy experiment logs
(e.g., reward, latency, loss) before visualization.

Supported Methods:
    - Moving Average
    - Exponential Moving Average (EMA)
    - Savitzky-Golay Polynomial Filter
    - Gaussian Smoothing
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d


def moving_average(data, window_size: int = 10):
    """
    Compute a simple moving average (SMA).

    Args:
        data (array-like): Input signal.
        window_size (int): Number of samples for averaging window.

    Returns:
        np.ndarray: Smoothed signal.
    """
    if len(data) < window_size:
        return np.array(data)
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode="same")


def exponential_moving_average(data, alpha: float = 0.3):
    """
    Exponential moving average (EMA), suitable for time-series smoothing.

    Args:
        data (array-like): Input signal.
        alpha (float): Smoothing factor [0, 1]. Smaller = smoother.

    Returns:
        np.ndarray: Smoothed signal.
    """
    data = np.asarray(data, dtype=float)
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for t in range(1, len(data)):
        smoothed[t] = alpha * data[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed


def savitzky_golay_smoothing(data, window_size: int = 15, poly_order: int = 3):
    """
    Apply Savitzky-Golay polynomial filter.

    Args:
        data (array-like): Input signal.
        window_size (int): Odd number of samples in the filter window.
        poly_order (int): Polynomial order (typically 2–5).

    Returns:
        np.ndarray: Smoothed signal.
    """
    if len(data) < window_size:
        return np.array(data)
    return savgol_filter(data, window_size, poly_order)


def gaussian_smoothing(data, sigma: float = 1.2):
    """
    Apply Gaussian smoothing to the input data.

    Args:
        data (array-like): Input signal.
        sigma (float): Gaussian kernel width (std. deviation).

    Returns:
        np.ndarray: Smoothed signal.
    """
    return gaussian_filter1d(np.asarray(data, dtype=float), sigma=sigma)


def smooth_curve(data, method: str = "ema", **kwargs):
    """
    Unified smoothing interface.

    Args:
        data (array-like): Input signal.
        method (str): One of ['ma', 'ema', 'sg', 'gaussian'].
        kwargs: Extra arguments passed to the selected smoothing method.

    Returns:
        np.ndarray: Smoothed output.
    """
    method = method.lower()
    if method in ["ma", "moving", "average"]:
        return moving_average(data, **kwargs)
    elif method in ["ema", "exp", "exponential"]:
        return exponential_moving_average(data, **kwargs)
    elif method in ["sg", "savitzky"]:
        return savitzky_golay_smoothing(data, **kwargs)
    elif method in ["gaussian", "g"]:
        return gaussian_smoothing(data, **kwargs)
    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def normalize(data):
    """
    Normalize data to [0, 1] range.
    """
    data = np.asarray(data, dtype=float)
    min_val, max_val = np.min(data), np.max(data)
    if max_val == min_val:
        return np.zeros_like(data)
    return (data - min_val) / (max_val - min_val)


def detrend(data, method="linear"):
    """
    Remove slow-changing trends from time-series data.

    Args:
        data (array-like): Input data.
        method (str): 'linear' or 'mean'.

    Returns:
        np.ndarray: Detrended signal.
    """
    data = np.asarray(data, dtype=float)
    if method == "mean":
        return data - np.mean(data)
    elif method == "linear":
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data, 1)
        trend = np.polyval(coeffs, x)
        return data - trend
    else:
        raise ValueError("method must be 'linear' or 'mean'.")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Generate synthetic noisy data
    x = np.linspace(0, 10, 200)
    y = np.sin(x) + np.random.normal(scale=0.2, size=len(x))

    # Test all smoothing methods
    plt.figure(figsize=(7, 4))
    plt.plot(x, y, color="lightgray", label="Raw Data", linewidth=1)
    plt.plot(x, moving_average(y, 10), label="Moving Avg")
    plt.plot(x, exponential_moving_average(y, 0.2), label="EMA")
    plt.plot(x, savitzky_golay_smoothing(y, 21, 3), label="Savitzky-Golay")
    plt.plot(x, gaussian_smoothing(y, 1.0), label="Gaussian")
    plt.legend()
    plt.title("Smoothing Techniques Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.show()
