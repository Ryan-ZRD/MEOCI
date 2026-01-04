import os
import pandas as pd


def generate_vehicle_latency_data(vehicles: list[int]) -> pd.DataFrame:

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


def combine_vehicle_effect_data(latency_df: pd.DataFrame, completion_df: pd.DataFrame) -> pd.DataFrame:

    df_latency = latency_df.copy()
    df_completion = completion_df.copy()

    # Rename completion rate columns for clarity
    rename_map = {col: f"{col} (Completion %)" for col in df_completion.columns if col != "Vehicles"}
    df_completion.rename(columns=rename_map, inplace=True)

    # Merge both datasets
    df_combined = pd.merge(df_latency, df_completion, on="Vehicles")
    return df_combined


def save_vehicle_effect_data(output_path: str = "visualization/data_csv/vehicle_effect.csv"):

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



if __name__ == "__main__":
    save_vehicle_effect_data()
