"""
visualization.vehicle_effect.vehicle_data_generator
----------------------------------------------------------
Generate synthetic vehicular load experiment data (Fig.11).

Description:
    This script produces the CSV file used in:
        - plot_latency_vs_vehicle.py
        - plot_completion_vs_vehicle.py

Output File:
    visualization/data_csv/vehicle_effect.csv

Each row corresponds to a specific number of vehicles (system load),
and each column represents a collaborative inference strategy.

Columns:
    Vehicles, Vehicle-Only, Edge-Only, Edgent, FedAdapt, LBO, ADP-D3QN (Ours)
"""

import os
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# 1. Parameter Configuration
# ------------------------------------------------------------
def generate_vehicle_latency_data(vehicles: list[int]) -> pd.DataFrame:
    """
    Generate synthetic latency data (ms) for different inference strategies.

    Args:
        vehicles (list[int]): List of vehicle counts.

    Returns:
        pd.DataFrame: Simulated latency dataset.
    """
    data = {
        "Vehicles": vehicles,
        "Vehicle-Only": [120 + 3 * v for v in vehicles],
        "Edge-Only": [95 + 2.1 * v for v in vehicles],
        "Edgent": [85 + 1.6 * v for v in vehicles],
        "FedAdapt": [78 + 1.2 * v for v in vehicles],
        "LBO": [74 + 1.05 * v for v in vehicles],
        "ADP-D3QN (Ours)": [69 + 0.45 * v for v in vehicles],
    }
    df_latency = pd.DataFrame(data)
    return df_latency


def generate_vehicle_completion_data(vehicles: list[int]) -> pd.DataFrame:
    """
    Generate synthetic task completion rate data (%) for different strategies.

    Args:
        vehicles (list[int]): List of vehicle counts.

    Returns:
        pd.DataFrame: Simulated completion rate dataset.
    """
    data = {
        "Vehicles": vehicles,
        "Vehicle-Only": [97.5 - 0.9 * (v / 5) for v in vehicles],
        "Edge-Only": [98.6 - 0.75 * (v / 5) for v in vehicles],
        "Edgent": [99.1 - 0.6 * (v / 5) for v in vehicles],
        "FedAdapt": [99.3 - 0.35 * (v / 5) for v in vehicles],
        "LBO": [99.5 - 0.28 * (v / 5) for v in vehicles],
        "ADP-D3QN (Ours)": [99.8 - 0.1 * (v / 5) for v in vehicles],
    }
    df_completion = pd.DataFrame(data)
    return df_completion


# ------------------------------------------------------------
# 2. Merge Data (Latency + Completion Rate)
# ------------------------------------------------------------
def combine_vehicle_effect_data(latency_df: pd.DataFrame, completion_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge latency and completion rate datasets into a single CSV table.

    Args:
        latency_df (pd.DataFrame): Latency data.
        completion_df (pd.DataFrame): Completion rate data.

    Returns:
        pd.DataFrame: Combined dataframe for analysis.
    """
    df_latency = latency_df.copy()
    df_completion = completion_df.copy()

    # Rename completion rate columns for clarity
    rename_map = {col: f"{col} (Completion %)" for col in df_completion.columns if col != "Vehicles"}
    df_completion.rename(columns=rename_map, inplace=True)

    # Merge both datasets
    df_combined = pd.merge(df_latency, df_completion, on="Vehicles")
    return df_combined


# ------------------------------------------------------------
# 3. Save to CSV
# ------------------------------------------------------------
def save_vehicle_effect_data(output_path: str = "visualization/data_csv/vehicle_effect.csv"):
    """
    Generate and save vehicular effect dataset to CSV.

    Args:
        output_path (str): Output CSV path.
    """
    vehicles = [5, 10, 15, 20, 25, 30]

    df_latency = generate_vehicle_latency_data(vehicles)
    df_completion = generate_vehicle_completion_data(vehicles)
    df_combined = combine_vehicle_effect_data(df_latency, df_completion)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_latency.to_csv(output_path, index=False)

    print(f"[Generated] Latency dataset saved to: {output_path}")
    print(df_latency.head())

    # Optional: also save combined version for supplementary analysis
    combined_path = output_path.replace(".csv", "_combined.csv")
    df_combined.to_csv(combined_path, index=False)
    print(f"[Generated] Combined dataset saved to: {combined_path}")


# ------------------------------------------------------------
# 4. Main Execution
# ------------------------------------------------------------
if __name__ == "__main__":
    save_vehicle_effect_data()
