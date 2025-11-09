"""
visualization.transmission_effect.transmission_data_loader
-----------------------------------------------------------
Utility for loading and validating transmission-rate experiment data.

Description:
    This module loads the experimental CSV file describing
    the effect of wireless transmission rate (Mbps)
    on inference latency (ms) and task completion rate (%).

    Used by:
        - plot_latency_vs_rate.py
        - plot_completion_vs_rate.py

Expected CSV Format:
-----------------------------------------------------------
Rate (Mbps),Vehicle-Only,Edge-Only,Edgent,FedAdapt,LBO,ADP-D3QN (Ours)
1,210.5,190.3,155.4,132.1,120.2,102.7
2,200.4,175.8,143.2,121.5,112.4,97.8
4,185.2,160.6,133.4,112.2,104.6,92.5
6,172.9,148.1,125.1,105.8,99.2,88.4
8,165.6,140.5,120.0,101.4,96.1,86.0
10,160.1,136.2,117.6,99.3,94.5,84.8
"""

import os
import pandas as pd


def load_transmission_data(csv_path: str = "visualization/data_csv/transmission_effect.csv") -> pd.DataFrame:
    """
    Load transmission rate experiment data.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing transmission rate data.

    Raises:
        FileNotFoundError: If CSV file does not exist.
        ValueError: If CSV is missing required columns.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing input CSV file: {csv_path}")

    df = pd.read_csv(csv_path)

    # Validate format
    required_column = "Rate (Mbps)"
    if required_column not in df.columns:
        raise ValueError(f"CSV must contain column '{required_column}'")

    # Basic type check
    if not pd.api.types.is_numeric_dtype(df[required_column]):
        raise ValueError(f"Column '{required_column}' must be numeric (Mbps values).")

    print(f"[INFO] Loaded transmission data from: {csv_path}")
    print(f"[INFO] Columns: {', '.join(df.columns)}")
    print(f"[INFO] Rows: {len(df)}")
    return df


def summarize_transmission_data(df: pd.DataFrame) -> None:
    """
    Print summary statistics for the dataset.

    Args:
        df (pd.DataFrame): Loaded transmission rate data.
    """
    rate_col = "Rate (Mbps)"
    methods = [c for c in df.columns if c != rate_col]

    print("\n[Dataset Summary]")
    print(f"Transmission Rate Range: {df[rate_col].min()} – {df[rate_col].max()} Mbps")
    print(f"Methods Evaluated: {', '.join(methods)}")

    for method in methods:
        mean_val = df[method].mean()
        min_val = df[method].min()
        max_val = df[method].max()
        print(f"  {method:20s} | Mean: {mean_val:7.2f} | Range: [{min_val:6.2f}, {max_val:6.2f}]")


def generate_synthetic_transmission_data(output_path: str = "visualization/data_csv/transmission_effect.csv") -> None:
    """
    Generate a synthetic transmission-rate dataset for testing.

    Args:
        output_path (str): Path to save generated CSV.
    """
    import numpy as np

    rates = np.array([1, 2, 4, 6, 8, 10])
    df = pd.DataFrame({
        "Rate (Mbps)": rates,
        "Vehicle-Only": 210 - 5 * np.log1p(rates),
        "Edge-Only": 190 - 6 * np.log1p(rates),
        "Edgent": 160 - 7 * np.log1p(rates),
        "FedAdapt": 140 - 8 * np.log1p(rates),
        "LBO": 125 - 9 * np.log1p(rates),
        "ADP-D3QN (Ours)": 110 - 10 * np.log1p(rates),
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"[Generated] Synthetic transmission data saved to: {output_path}")


if __name__ == "__main__":
    # Test loading or synthetic generation
    csv_path = "visualization/data_csv/transmission_effect.csv"

    if os.path.exists(csv_path):
        df_loaded = load_transmission_data(csv_path)
        summarize_transmission_data(df_loaded)
    else:
        print("[WARN] No dataset found, generating synthetic example...")
        generate_synthetic_transmission_data(csv_path)
        df_loaded = load_transmission_data(csv_path)
        summarize_transmission_data(df_loaded)
