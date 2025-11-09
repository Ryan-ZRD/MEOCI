"""
visualization.delay_constraints.delay_constraints_utils
-------------------------------------------------------
Utility functions for delay constraint experiments (Fig.13a–13b).

Includes:
    • CSV data loader with validation
    • Accuracy/completion rate trend summarization
    • Optional synthetic data generator (for testing visualization)
"""

import os
import pandas as pd
import numpy as np
from visualization.shared_styles.smoothing import exponential_smooth


# ------------------------------------------------------------
# 1. Data loading
# ------------------------------------------------------------
def load_delay_constraints_data(
    csv_path: str = "visualization/data_csv/delay_constraints.csv"
) -> pd.DataFrame:
    """
    Load and validate delay constraint dataset.

    Args:
        csv_path (str): Path to the delay constraint CSV file.

    Returns:
        pd.DataFrame: Validated dataframe containing delay vs. metrics.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing dataset: {csv_path}")

    df = pd.read_csv(csv_path)
    required_col = "Delay Constraint (ms)"
    if required_col not in df.columns:
        raise ValueError(f"CSV must contain column: '{required_col}'")

    # Ensure numeric types
    df[required_col] = pd.to_numeric(df[required_col], errors="coerce")
    df = df.dropna(subset=[required_col])

    return df


# ------------------------------------------------------------
# 2. Trend summarization
# ------------------------------------------------------------
def summarize_delay_effects(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistical summary for delay constraint effects.

    Args:
        df (pd.DataFrame): Delay constraint dataset.

    Returns:
        pd.DataFrame: Summary statistics including growth rate and relative gain.
    """
    base_col = "Delay Constraint (ms)"
    methods = [c for c in df.columns if c != base_col]

    summary = []
    for method in methods:
        values = df[method].values
        rate = (values[-1] - values[0]) / (df[base_col].values[-1] - df[base_col].values[0])
        gain = (values[-1] - values[0]) / values[0] * 100
        summary.append({
            "Method": method,
            "Initial": values[0],
            "Final": values[-1],
            "GrowthRate(per_ms)": round(rate, 4),
            "RelativeGain(%)": round(gain, 2)
        })
    return pd.DataFrame(summary)


# ------------------------------------------------------------
# 3. Data smoothing
# ------------------------------------------------------------
def smooth_delay_curves(df: pd.DataFrame, alpha: float = 0.3) -> pd.DataFrame:
    """
    Apply exponential smoothing to each column except the delay axis.

    Args:
        df (pd.DataFrame): Raw dataframe.
        alpha (float): Smoothing coefficient (0–1).

    Returns:
        pd.DataFrame: Smoothed dataframe.
    """
    base_col = "Delay Constraint (ms)"
    smoothed_df = df.copy()
    for col in df.columns:
        if col != base_col:
            smoothed_df[col] = exponential_smooth(df[col].values, alpha)
    return smoothed_df


# ------------------------------------------------------------
# 4. Synthetic data generator (for testing)
# ------------------------------------------------------------
def generate_synthetic_delay_data(
    save_path: str = "visualization/data_csv/delay_constraints.csv",
    random_seed: int = 42
) -> None:
    """
    Generate synthetic delay constraint dataset for quick testing.

    Args:
        save_path (str): Destination to save the synthetic CSV.
        random_seed (int): Random seed for reproducibility.
    """
    np.random.seed(random_seed)

    delays = np.array([50, 100, 150, 200, 250, 300])
    base_accuracy = np.linspace(80, 90, len(delays))
    base_completion = np.linspace(82, 95, len(delays))

    noise = lambda scale=0.5: np.random.uniform(-scale, scale, len(delays))

    data = {
        "Delay Constraint (ms)": delays,
        "Vehicle-Only": base_accuracy - 3 + noise(),
        "Edge-Only": base_accuracy - 2 + noise(),
        "Edgent": base_accuracy + noise(),
        "FedAdapt": base_accuracy + 2 + noise(),
        "LBO": base_accuracy + 3 + noise(),
        "ADP-D3QN (Ours)": base_accuracy + 5 + noise(),
    }

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[Generated synthetic data] -> {save_path}")


# ------------------------------------------------------------
# 5. Example execution
# ------------------------------------------------------------
if __name__ == "__main__":
    print(">>> Generating synthetic dataset for delay constraint experiments...")
    generate_synthetic_delay_data()

    print(">>> Loading dataset...")
    df = load_delay_constraints_data()
    print(df.head())

    print(">>> Summary statistics:")
    print(summarize_delay_effects(df))

    print(">>> Applying smoothing...")
    smoothed = smooth_delay_curves(df)
    print(smoothed.head())
