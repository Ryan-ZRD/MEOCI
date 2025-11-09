"""
visualization.exit_analysis.exit_data_generator
----------------------------------------------------------
Generate synthetic early-exit probability data for multi-exit DNNs.

Generates CSV files:
    - visualization/data_csv/exit_alexnet.csv  (4 exits)
    - visualization/data_csv/exit_vgg16.csv    (5 exits)

Purpose:
    Reproduce Fig.8(a)-(b) in MEOCI paper.
    These files represent the exit selection probabilities
    under different vehicular load conditions.

Columns:
    Exit, Low Load, Medium Load, High Load
"""

import os
import pandas as pd
import numpy as np


def generate_exit_data_alexnet(save_path: str = "visualization/data_csv/exit_alexnet.csv"):
    """
    Generate synthetic exit probability data for AlexNet-ME (4 exits).

    Args:
        save_path (str): Path to save generated CSV file.
    """
    exits = np.arange(1, 5)
    data = {
        "Exit": exits,
        "Low Load": [0.45, 0.35, 0.15, 0.05],
        "Medium Load": [0.32, 0.40, 0.22, 0.06],
        "High Load": [0.18, 0.30, 0.35, 0.17],
    }
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[Generated] {save_path}")


def generate_exit_data_vgg16(save_path: str = "visualization/data_csv/exit_vgg16.csv"):
    """
    Generate synthetic exit probability data for VGG16-ME (5 exits).

    Args:
        save_path (str): Path to save generated CSV file.
    """
    exits = np.arange(1, 6)
    data = {
        "Exit": exits,
        "Low Load": [0.30, 0.25, 0.20, 0.15, 0.10],
        "Medium Load": [0.20, 0.28, 0.27, 0.18, 0.07],
        "High Load": [0.10, 0.18, 0.30, 0.25, 0.17],
    }
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[Generated] {save_path}")


def summarize_exit_data():
    """
    Print a quick summary table for generated datasets.
    """
    alex = pd.read_csv("visualization/data_csv/exit_alexnet.csv")
    vgg = pd.read_csv("visualization/data_csv/exit_vgg16.csv")

    print("\n=== Exit Probability Summary ===")
    print("AlexNet-ME (4 exits):")
    print(alex)
    print("\nVGG16-ME (5 exits):")
    print(vgg)


def generate_all_exit_data():
    """
    Generate both AlexNet and VGG16 exit probability CSV files.
    """
    generate_exit_data_alexnet()
    generate_exit_data_vgg16()
    summarize_exit_data()


if __name__ == "__main__":
    generate_all_exit_data()
