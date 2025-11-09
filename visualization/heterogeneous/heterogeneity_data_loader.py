"""
visualization.heterogeneous.heterogeneity_data_loader
----------------------------------------------------------
Unified data loader for heterogeneous inference latency visualization (Fig.9 Series)

Description:
    Loads latency data (ms) of multiple models and inference frameworks
    from CSV files under visualization/data_csv/.

Supported Models:
    - AlexNet-ME
    - VGG16-ME
    - ResNet50-ME
    - YOLOv10n-ME

Supported Devices:
    - NVIDIA Jetson Nano
    - Raspberry Pi 4B

Typical Use:
    >>> from visualization.heterogeneous.heterogeneity_data_loader import load_latency_data
    >>> df_vgg = load_latency_data("vgg16")
    >>> print(df_vgg.head())
"""

import os
import pandas as pd


DATA_CSV_PATH = "visualization/data_csv/heterogeneous_latency.csv"
REQUIRED_COLUMNS = ["Method", "Device", "Latency(ms)"]


def load_latency_data(model_name: str = "alexnet") -> pd.DataFrame:
    """
    Load heterogeneous latency data for the given model.

    Args:
        model_name (str): One of ["alexnet", "vgg16", "resnet50", "yolov10n"].

    Returns:
        pd.DataFrame: Filtered DataFrame with columns ['Method', 'Device', 'Latency(ms)'].
    """
    if not os.path.exists(DATA_CSV_PATH):
        raise FileNotFoundError(f"Missing CSV file: {DATA_CSV_PATH}")

    df = pd.read_csv(DATA_CSV_PATH)

    # Validation
    if not all(col in df.columns for col in REQUIRED_COLUMNS):
        raise ValueError(
            f"CSV file must contain columns: {REQUIRED_COLUMNS}. "
            f"Current columns: {list(df.columns)}"
        )

    # Add model column if not exist
    if "Model" not in df.columns:
        df["Model"] = None

    # Auto label based on naming pattern or external structure
    if df["Model"].isnull().all():
        # infer model tag from filename if available
        model_tag = model_name.lower()
        df["Model"] = model_tag

    # Filter for target model (if CSV contains multiple models)
    filtered_df = df[df["Model"].str.lower() == model_name.lower()].copy()

    if filtered_df.empty:
        # fallback: entire CSV for backward compatibility
        filtered_df = df.copy()
        print(f"[Warning] No model-specific data found. Returning all entries from {DATA_CSV_PATH}")

    # Sanitize
    filtered_df = filtered_df.dropna(subset=["Method", "Device", "Latency(ms)"])
    filtered_df["Latency(ms)"] = filtered_df["Latency(ms)"].astype(float)

    return filtered_df.reset_index(drop=True)


def summarize_latency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average latency per method across devices for summary table.

    Args:
        df (pd.DataFrame): Input DataFrame from load_latency_data().

    Returns:
        pd.DataFrame: Summary table with average latency values.
    """
    summary = (
        df.groupby("Method")["Latency(ms)"]
        .mean()
        .reset_index()
        .rename(columns={"Latency(ms)": "Avg_Latency(ms)"})
        .sort_values("Avg_Latency(ms)")
        .reset_index(drop=True)
    )
    return summary


def compare_devices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare Nano and Pi4B latencies for each method.

    Returns:
        pd.DataFrame: Table with per-device latency and delta.
    """
    pivot = df.pivot(index="Method", columns="Device", values="Latency(ms)").reset_index()
    pivot["Δ (Pi4B - Nano)"] = pivot["Pi4B"] - pivot["Nano"]
    return pivot


if __name__ == "__main__":
    print("=== Heterogeneous Latency Data Loader Demo ===")
    for model in ["alexnet", "vgg16", "resnet50", "yolov10n"]:
        try:
            df = load_latency_data(model)
            print(f"\n[{model.upper()}] {len(df)} records loaded.")
            print(summarize_latency(df).head())
        except Exception as e:
            print(f"[Error] {model}: {e}")
