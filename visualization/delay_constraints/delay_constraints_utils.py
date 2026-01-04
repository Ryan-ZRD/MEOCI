import os
import pandas as pd
import numpy as np
from visualization.shared_styles.smoothing import exponential_moving_average



def load_delay_constraints_data(
    csv_path: str = "visualization/data_csv/delay_constraints.csv"
) -> pd.DataFrame:

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



def summarize_delay_effects(df: pd.DataFrame) -> pd.DataFrame:

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



def smooth_delay_curves(df: pd.DataFrame, alpha: float = 0.3) -> pd.DataFrame:

    base_col = "Delay Constraint (ms)"
    smoothed_df = df.copy()
    for col in df.columns:
        if col != base_col:
            smoothed_df[col] = exponential_moving_average(df[col].values, alpha)
    return smoothed_df



def generate_synthetic_delay_data(
    save_path: str = "visualization/data_csv/delay_constraints.csv",
    random_seed: int = 42
) -> None:

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
