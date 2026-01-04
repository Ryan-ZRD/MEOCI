

import os
import pandas as pd
import numpy as np
from visualization.shared_styles.smoothing import exponential_moving_average



def load_energy_data(
    csv_path: str = "visualization/data_csv/energy_constraints.csv"
) -> pd.DataFrame:

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_col = "Energy Constraint (mJ)"
    if required_col not in df.columns:
        raise ValueError(f"CSV must contain column: '{required_col}'")

    # Ensure numeric conversion
    df[required_col] = pd.to_numeric(df[required_col], errors="coerce")
    df = df.dropna(subset=[required_col])
    return df



def summarize_energy_trends(df: pd.DataFrame) -> pd.DataFrame:

    base_col = "Energy Constraint (mJ)"
    methods = [c for c in df.columns if c != base_col]
    summary = []

    for method in methods:
        values = df[method].values
        delta = values[-1] - values[0]
        slope = delta / (df[base_col].iloc[-1] - df[base_col].iloc[0])
        gain = (values[-1] - values[0]) / values[0] * 100
        summary.append({
            "Method": method,
            "Initial": round(values[0], 2),
            "Final": round(values[-1], 2),
            "Delta": round(delta, 2),
            "Slope(per_mJ)": round(slope, 4),
            "RelativeChange(%)": round(gain, 2)
        })

    return pd.DataFrame(summary)



def smooth_energy_curves(df: pd.DataFrame, alpha: float = 0.3) -> pd.DataFrame:

    base_col = "Energy Constraint (mJ)"
    smoothed_df = df.copy()

    for col in df.columns:
        if col != base_col:
            smoothed_df[col] = exponential_moving_average(df[col].values, alpha)

    return smoothed_df



def generate_synthetic_energy_data(
    save_path: str = "visualization/data_csv/energy_constraints.csv",
    random_seed: int = 42
) -> None:

    np.random.seed(random_seed)

    # Define X-axis (energy constraints)
    energy_constraints = np.array([50, 75, 100, 125, 150, 175])

    # Define base trends (approximate to MEOCI paper patterns)
    base_latency = np.array([92, 87, 83, 80, 78, 77])
    base_energy = np.array([49, 70, 89, 108, 126, 140])

    noise = lambda scale=1.0: np.random.uniform(-scale, scale, len(energy_constraints))

    latency_data = {
        "Energy Constraint (mJ)": energy_constraints,
        "Vehicle-Only": base_latency - 2 + noise(),
        "Edge-Only": base_latency - 3 + noise(),
        "Edgent": base_latency - 10 + noise(),
        "FedAdapt": base_latency - 13 + noise(),
        "LBO": base_latency - 15 + noise(),
        "ADP-D3QN (Ours)": base_latency - 18 + noise(),
    }

    energy_data = {
        "Energy Constraint (mJ)": energy_constraints,
        "Vehicle-Only": base_energy + 0 + noise(),
        "Edge-Only": base_energy - 2 + noise(),
        "Edgent": base_energy - 10 + noise(),
        "FedAdapt": base_energy - 14 + noise(),
        "LBO": base_energy - 17 + noise(),
        "ADP-D3QN (Ours)": base_energy - 22 + noise(),
    }

    latency_df = pd.DataFrame(latency_data)
    energy_df = pd.DataFrame(energy_data)

    # Merge for storage
    combined = energy_df.copy()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    combined.to_csv(save_path, index=False)

    print(f"[Synthetic energy dataset generated] → {save_path}")
    print(f"Columns: {', '.join(combined.columns)}")



if __name__ == "__main__":
    print(">>> Generating synthetic dataset for energy constraint experiments...")
    generate_synthetic_energy_data()

    print(">>> Loading dataset...")
    df = load_energy_data()
    print(df.head())

    print(">>> Statistical Summary:")
    print(summarize_energy_trends(df))

    print(">>> Applying smoothing...")
    df_smooth = smooth_energy_curves(df)
    print(df_smooth.head())
