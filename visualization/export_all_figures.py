"""
visualization.export_all_figures
--------------------------------
Master script to reproduce all figures (Fig.7–Fig.16)
for the MEOCI framework.

Features:
    • Automatically generates all plots from submodules:
        - ablation (Fig.7)
        - exit_analysis (Fig.8)
        - heterogeneous (Fig.9)
        - accuracy_cdf (Fig.10)
        - vehicle_effect (Fig.11)
        - transmission_effect (Fig.12)
        - delay_constraints (Fig.13)
        - energy_constraints (Fig.14)
        - scalability (Fig.15–16)
    • Unified output path and format
    • Configurable CLI arguments

Usage:
    python visualization/export_all_figures.py --output results/plots --format png
"""

import os
import argparse
import sys
from pathlib import Path

# ------------------------------------------------------------
# Import all figure submodules
# ------------------------------------------------------------
from visualization.ablation.plot_ablation_convergence import plot_ablation_convergence
from visualization.ablation.plot_ablation_delay import plot_ablation_delay

from visualization.exit_analysis.plot_exit_probability_alexnet import plot_exit_probability_alexnet
from visualization.exit_analysis.plot_exit_probability_vgg16 import plot_exit_probability_vgg16

from visualization.heterogeneous.plot_latency_alexnet import plot_latency_alexnet
from visualization.heterogeneous.plot_latency_vgg16 import plot_latency_vgg16
from visualization.heterogeneous.plot_latency_resnet50 import plot_latency_resnet50
from visualization.heterogeneous.plot_latency_yolov10n import plot_latency_yolov10n

from visualization.accuracy_cdf.plot_accuracy_comparison import plot_accuracy_cdf
from visualization.accuracy_cdf.plot_latency_cdf import plot_latency_cdf

from visualization.vehicle_effect.plot_latency_vs_vehicle import plot_latency_vs_vehicle
from visualization.vehicle_effect.plot_completion_vs_vehicle import plot_completion_vs_vehicle

from visualization.transmission_effect.plot_latency_vs_rate import plot_latency_vs_rate
from visualization.transmission_effect.plot_completion_vs_rate import plot_completion_vs_rate

from visualization.delay_constraints.plot_accuracy_vs_delay_constraints import plot_accuracy_vs_delay_constraints
from visualization.delay_constraints.plot_completion_vs_delay_constraints import plot_completion_vs_delay_constraints

from visualization.energy_constraints.plot_latency_vs_energy import plot_latency_vs_energy
from visualization.energy_constraints.plot_energy_consumption import plot_energy_consumption

from visualization.scalability.plot_dynamic_density import plot_dynamic_density
from visualization.scalability.plot_latency_dynamic_density import plot_latency_dynamic_density
from visualization.scalability.plot_vehicle_scalability import plot_vehicle_scalability
from visualization.scalability.plot_computational_stress import plot_computational_stress


# ------------------------------------------------------------
# Helper: ensure output directories exist
# ------------------------------------------------------------
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Master figure export pipeline
# ------------------------------------------------------------
def export_all_figures(output_dir: str = "results/plots", fmt: str = "png", overwrite: bool = True):
    """
    Export all MEOCI paper figures (Fig.7–16) in batch.

    Args:
        output_dir (str): Destination folder for exported figures.
        fmt (str): Output image format ("png", "pdf", or "svg").
        overwrite (bool): Whether to overwrite existing files.
    """
    ensure_dir(output_dir)
    print(f"[INFO] Exporting figures to: {output_dir}")
    print(f"[INFO] Output format: .{fmt}")

    # List of (function, filename, description)
    tasks = [
        (plot_ablation_convergence, "fig7a_ablation_convergence", "Ablation Study: Convergence"),
        (plot_ablation_delay, "fig7b_ablation_delay", "Ablation Study: Delay Comparison"),
        (plot_exit_probability_alexnet, "fig8a_exit_alexnet", "Exit Probability (AlexNet)"),
        (plot_exit_probability_vgg16, "fig8b_exit_vgg16", "Exit Probability (VGG16)"),
        (plot_latency_alexnet, "fig9a_latency_alexnet", "Heterogeneous Latency (AlexNet)"),
        (plot_latency_vgg16, "fig9b_latency_vgg16", "Heterogeneous Latency (VGG16)"),
        (plot_latency_resnet50, "fig9c_latency_resnet50", "Heterogeneous Latency (ResNet50)"),
        (plot_latency_yolov10n, "fig9d_latency_yolov10n", "Heterogeneous Latency (YOLOv10n)"),
        (plot_accuracy_cdf, "fig10a_accuracy_comparison", "Accuracy Comparison (CDF)"),
        (plot_latency_cdf, "fig10b_latency_cdf", "Latency Distribution (CDF)"),
        (plot_latency_vs_vehicle, "fig11a_latency_vehicle", "Latency vs Vehicle Count"),
        (plot_completion_vs_vehicle, "fig11b_completion_vehicle", "Completion vs Vehicle Count"),
        (plot_latency_vs_rate, "fig12a_latency_rate", "Latency vs Transmission Rate"),
        (plot_completion_vs_rate, "fig12b_completion_rate", "Completion vs Transmission Rate"),
        (plot_accuracy_vs_delay_constraints, "fig13a_accuracy_delay", "Accuracy vs Delay Constraints"),
        (plot_completion_vs_delay_constraints, "fig13b_completion_delay", "Completion vs Delay Constraints"),
        (plot_latency_vs_energy, "fig14a_latency_energy", "Latency vs Energy Constraint"),
        (plot_energy_consumption, "fig14b_energy_consumption", "Energy Consumption vs Constraint"),
        (plot_dynamic_density, "fig15a_dynamic_density", "Dynamic Traffic Density"),
        (plot_latency_dynamic_density, "fig15b_latency_dynamic_density", "Latency under Dynamic Density"),
        (plot_vehicle_scalability, "fig16a_vehicle_scalability", "Scalability w.r.t Vehicle Count"),
        (plot_computational_stress, "fig16b_computational_stress", "Computational Stress Scaling"),
    ]

    for func, filename, desc in tasks:
        save_path = os.path.join(output_dir, f"{filename}.{fmt}")
        if not overwrite and os.path.exists(save_path):
            print(f"[SKIP] {desc} — file already exists.")
            continue
        try:
            print(f"[PLOT] Generating {desc}...")
            func(save_path=save_path, fmt=fmt)
        except Exception as e:
            print(f"[ERROR] Failed to generate {desc}: {e}")

    print(f"[DONE] All figures have been exported to: {output_dir}")


# ------------------------------------------------------------
# Command Line Interface
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Export all MEOCI paper figures (Fig.7–16).")
    parser.add_argument("--output", type=str, default="results/plots", help="Output directory for figures")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"], help="Figure format")
    parser.add_argument("--no-overwrite", action="store_true", help="Skip existing files")
    args = parser.parse_args()

    export_all_figures(output_dir=args.output, fmt=args.format, overwrite=not args.no_overwrite)


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    main()
