import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d


def moving_average(data, window_size: int = 10):

    if len(data) < window_size:
        return np.array(data)
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode="same")


def exponential_moving_average(data, alpha: float = 0.3):

    data = np.asarray(data, dtype=float)
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for t in range(1, len(data)):
        smoothed[t] = alpha * data[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed


def savitzky_golay_smoothing(data, window_size: int = 15, poly_order: int = 3):

    if len(data) < window_size:
        return np.array(data)
    return savgol_filter(data, window_size, poly_order)


def gaussian_smoothing(data, sigma: float = 1.2):

    return gaussian_filter1d(np.asarray(data, dtype=float), sigma=sigma)


def smooth_curve(data, method: str = "ema", **kwargs):

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
